//! Conservative opaque-region occlusion test — CPU reference implementation,
//! also the oracle the `validate` mode diffs the compute path against.
//! See docs/gpu-native-architecture.md §5.2, R-N §8.
//!
//! [`coverage`] holds the pure geometry (R-N §8.3's five conditions and the
//! coverage sweep). This file holds the *instance tier* (R-N §8.1): the
//! per-layer decision about which primitives a dirty layer does not have to
//! emit, expressed over a flat item stream so the same computation can be a
//! CPU loop here and one compute invocation per item in
//! `shaders/occlusion.wgsl`.
//!
//! # Why the sweep is written per-item rather than incrementally
//!
//! `src/occlusion.rs` walks a layer backwards once, accumulating an occluder
//! set as it goes, and keeps a second *independent* implementation
//! (`compute_keep_mask_independently`) purely so validate mode has something to
//! disagree with. A compute shader has no backward walk to share — every
//! invocation redecides its own item from scratch — so the independent form is
//! the only form here, and it is the one both sides run. That collapses the
//! legacy pair into one implementation and moves the differential where §5.2
//! says it belongs: between the CPU reference and the compute path, not between
//! two CPU passes.
//!
//! # What this tier must never do (R-N §8.4)
//!
//! Culling suppresses `DISPLAY` work only. Nothing here touches hitboxes,
//! dispatch nodes, or layout — a culled primitive's record stays resident, its
//! `Hitbox`/`DispatchNode` entries are untouched, and this module has no way to
//! reach them: it is handed geometry and returns a mask.
//!
//! # What it is not: the layer tier
//!
//! R-N §8.1's coarse tier (a whole layer covered by opaque layers above it,
//! skipped at composite time) is not implemented here. It runs over layers, not
//! primitives — tens of items, not tens of thousands — so it is not a compute
//! problem, and §5.2 scopes the compute change to the instance tier
//! specifically. [`coverage::fully_covered`] is the routine it will reuse when
//! the compositor grows a per-layer opaque region to feed it.

pub mod coverage;

use crate::geometry::Rect;
use crate::patch::primitive::Quad;
use coverage::{MAX_OCCLUDERS, OccluderStyle, fully_covered, opaque_region};

/// How occlusion culling behaves this run — R-N §8.5's two required switches.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Mode {
    /// `WGPUI_OCCLUSION=0` — culling disabled entirely. Every primitive emits.
    Off,
    /// Normal operation.
    Normal,
    /// `WGPUI_OCCLUSION=validate` — run both the culled and the unculled path
    /// and diff them. Slow by design.
    Validate,
}

impl Mode {
    /// Read the mode from `WGPUI_OCCLUSION`, defaulting to [`Mode::Normal`].
    ///
    /// Unlike the legacy backend's `LazyLock` this re-reads on every call: the
    /// env var is process-global mutable state, and a `LazyLock` here would let
    /// whichever test ran first decide the mode for every later one. The read
    /// is not on any per-primitive path — callers resolve it once per frame.
    pub fn from_environment() -> Mode {
        match std::env::var("WGPUI_OCCLUSION") {
            Ok(value) if value == "0" || value == "off" => Mode::Off,
            Ok(value) if value == "validate" => Mode::Validate,
            _ => Mode::Normal,
        }
    }

    /// Whether culling runs at all.
    pub fn is_enabled(self) -> bool {
        self != Mode::Off
    }
}

/// One primitive as the occlusion tier sees it.
///
/// Deliberately not a `Quad`, a `RecordKey`, or anything else the scene knows
/// about: the compute shader reads exactly these fields and nothing more, so
/// keeping the CPU input identical is what makes the two paths comparable.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CoverageItem {
    /// What this primitive can actually paint: its bounds intersected with its
    /// content mask. Empty means it paints nothing.
    pub visible: Rect,
    /// Its conservative opaque region ([`coverage::opaque_region`]), or `None`
    /// when it does not qualify as an occluder.
    pub opaque: Option<Rect>,
    /// Whether this primitive may be dropped when covered.
    ///
    /// `false` for the kinds the legacy sweep keeps unconditionally: shadows
    /// (which bleed past their bounds by their blur radius, so no rectangle
    /// describes what they cover), surfaces, backdrop filters, and filter
    /// boundaries. Phase 1's two primitive kinds are both cullable; the flag
    /// exists so adding the other five does not need a second mechanism.
    pub cullable: bool,
    /// Whether this primitive is exempt from culling *and* from occluding,
    /// regardless of geometry.
    ///
    /// Two situations, both R-N §8.3's: it sits inside a filter group (whose
    /// blur samples its members' pixels), or it reaches into a boundary's
    /// overdraw margin (R-N §8.3's "overdraw regions are exempt" — content
    /// buffered precisely so a later `TRANSFORM` can reveal it). Both are
    /// properties of the emitting walk rather than of the primitive, so the
    /// caller supplies them.
    pub protected: bool,
}

impl CoverageItem {
    /// A primitive that can be culled and never occludes.
    pub fn cullee(visible: Rect) -> CoverageItem {
        CoverageItem {
            visible,
            opaque: None,
            cullable: true,
            protected: false,
        }
    }

    /// A primitive that can be culled and occludes over `opaque`.
    pub fn occluder(visible: Rect, opaque: Rect) -> CoverageItem {
        CoverageItem {
            visible,
            opaque: Some(opaque),
            cullable: true,
            protected: false,
        }
    }
}

/// One region that must never be culled *under*, and whose contents never
/// occlude.
///
/// R-N §8.3's last two conditions in one shape. A backdrop filter reads what is
/// behind it, and a filter group's blur samples neighbouring pixels, so both
/// poison everything painted below them within their own bounds *dilated by the
/// blur radius* — which is where the "blur margin" condition actually bites.
/// The caller dilates; this type carries the already-dilated rectangle so the
/// shader does not have to know what a blur radius is.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PoisonRegion {
    /// The filter's bounds, already dilated by its blur radius.
    pub region: Rect,
    /// Poisons every item whose index is strictly below this one — i.e. every
    /// primitive painted beneath the filter, which is exactly what a filter
    /// reads.
    pub above_index: u32,
}

/// What one occlusion pass decided, in aggregate. R-N §8.5's counters.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct OcclusionStats {
    /// Primitives that do not have to be emitted.
    pub culled: usize,
    /// Primitives that must be emitted.
    pub kept: usize,
    /// Primitives a filter reads through, and which are therefore exempt from
    /// culling whatever covers them. Counted whether or not anything actually
    /// covered them, because the useful question this answers is "how much of
    /// the layer is a filter holding hostage."
    pub poisoned: usize,
}

/// Which primitives of one layer must still be emitted, in paint order.
///
/// `items` is the layer's whole primitive stream in paint order — index order
/// *is* paint order, so `items[j]` for `j > i` is painted above `items[i]`.
/// Returns one flag per item: `true` to emit, `false` to cull.
///
/// This is the CPU reference `shaders/occlusion.wgsl` transcribes. Every
/// decision is a function of one item and the items above it, with no state
/// carried between items, so the compute path can run all of them at once and
/// get the same answer.
pub fn keep_mask(items: &[CoverageItem], poison: &[PoisonRegion]) -> Vec<bool> {
    let mut keep = vec![true; items.len()];
    for index in 0..items.len() {
        keep[index] = keep_item(items, poison, index);
    }
    keep
}

/// [`keep_mask`] plus R-N §8.5's counters.
pub fn keep_mask_with_stats(
    items: &[CoverageItem],
    poison: &[PoisonRegion],
) -> (Vec<bool>, OcclusionStats) {
    let keep = keep_mask(items, poison);
    let mut stats = OcclusionStats::default();
    for (index, kept) in keep.iter().enumerate() {
        if *kept {
            stats.kept += 1;
            if let Some(item) = items.get(index)
                && item.cullable
                && !item.protected
                && is_poisoned(poison, index, &item.visible)
            {
                stats.poisoned += 1;
            }
        } else {
            stats.culled += 1;
        }
    }
    (keep, stats)
}

/// The single-item decision, written exactly as the shader's one invocation.
fn keep_item(items: &[CoverageItem], poison: &[PoisonRegion], index: usize) -> bool {
    let Some(item) = items.get(index) else {
        return true;
    };
    if !item.cullable || item.protected || item.visible.is_empty() {
        return true;
    }
    if is_poisoned(poison, index, &item.visible) {
        return true;
    }

    let mut occluders = [Rect::EMPTY; MAX_OCCLUDERS];
    let occluder_count = gather_occluders(items, poison, index, &item.visible, &mut occluders);
    !fully_covered(item.visible, &occluders[..occluder_count])
}

/// Collect at most [`MAX_OCCLUDERS`] qualifying occluders painted above
/// `index`, in ascending paint order.
///
/// "Qualifying" is exactly the legacy sweep's rule: it has an opaque region, it
/// is not itself protected or poisoned (a quad inside a filter group never
/// collects), and its region actually overlaps the target. Note that an
/// occluder is collected whatever its *own* keep decision — an occluder that is
/// itself covered still hides what lies beneath it, and the legacy sweep says
/// so in as many words.
fn gather_occluders(
    items: &[CoverageItem],
    poison: &[PoisonRegion],
    index: usize,
    target: &Rect,
    occluders: &mut [Rect; MAX_OCCLUDERS],
) -> usize {
    let mut count = 0usize;
    let mut probe = index + 1;
    while probe < items.len() && count < MAX_OCCLUDERS {
        let candidate = probe;
        probe += 1;
        let Some(above) = items.get(candidate) else {
            continue;
        };
        let Some(region) = above.opaque else {
            continue;
        };
        if above.protected || is_poisoned(poison, candidate, &above.visible) {
            continue;
        }
        if region.intersect(target).is_empty() {
            continue;
        }
        occluders[count] = region;
        count += 1;
    }
    count
}

/// Whether a filter declared above `index` reads through `bounds`.
fn is_poisoned(poison: &[PoisonRegion], index: usize, bounds: &Rect) -> bool {
    let index = u32::try_from(index).unwrap_or(u32::MAX);
    poison
        .iter()
        .any(|zone| zone.above_index > index && !zone.region.intersect(bounds).is_empty())
}

/// One [`Quad`] as a [`CoverageItem`], applying R-N §8.3 to Phase 1's quad
/// field set.
///
/// **What Phase 1's `Quad` can and cannot express**, stated rather than
/// implied: it carries one uniform corner radius, one uniform border width, a
/// single solid background colour, and no content mask, border style, or
/// element-opacity field (`docs/phase-1-results.md` §2 is explicit that the
/// field set is a protocol exerciser, not the shipping GPU layout). So the
/// mapping is:
///
/// - Background is always solid — the type has no gradient variant — and its
///   alpha carries the element opacity already multiplied in, exactly as
///   `src/occlusion.rs`'s `quad_opaque_region` documents for the same reason.
/// - The border is opaque iff its colour is; there is no dashed style to reject
///   yet, and when one arrives it insets like a translucent border.
/// - `clip` is supplied by the caller because the quad has no content mask
///   field. Pass the layer's own clip, or an unbounded rectangle.
///
/// A phase that widens `Quad` widens this function and nothing else.
pub fn quad_coverage_item(quad: &Quad, clip: Rect, protected: bool) -> CoverageItem {
    let bounds = Rect::from_origin_size(quad.origin, quad.size);
    let style = OccluderStyle {
        background_is_solid: true,
        background_alpha: quad.background[3],
        element_opacity: 1.0,
        max_corner_radius: quad.corner_radius,
        border_is_opaque: quad.border_color[3] >= 1.0,
        max_border_width: quad.border_width,
        has_backdrop_filter: false,
    };
    CoverageItem {
        visible: bounds.intersect(&clip),
        opaque: opaque_region(bounds, clip, &style),
        cullable: true,
        protected,
    }
}

/// Bytes one [`CoverageItem`] occupies in the compute pass's input buffer.
///
/// `vec4<f32>` visible bounds, `vec4<f32>` opaque region, one flags word, three
/// words of padding — 16-byte aligned so a `std430` array indexes it without a
/// per-field fixup, the same reasoning `Quad::SLOT_STRIDE` gives.
pub const COVERAGE_ITEM_STRIDE: usize = 48;

/// Bytes one [`PoisonRegion`] occupies in the compute pass's input buffer.
pub const POISON_REGION_STRIDE: usize = 32;

/// [`CoverageItem::cullable`], as the shader's flags word sees it.
pub const FLAG_CULLABLE: u32 = 1;
/// [`CoverageItem::protected`], as the shader's flags word sees it.
pub const FLAG_PROTECTED: u32 = 2;
/// Whether [`CoverageItem::opaque`] is present.
pub const FLAG_HAS_OPAQUE: u32 = 4;

/// Encode a layer's items for `shaders/occlusion.wgsl`.
///
/// Byte-oriented rather than `bytemuck`-cast, for the reason
/// `patch/primitive.rs` gives: it keeps `wgpui-core` dependency-free and makes
/// the GPU layout an explicit decision rather than a consequence of Rust field
/// order.
pub fn encode_coverage_items(items: &[CoverageItem], destination: &mut Vec<u8>) {
    destination.clear();
    destination.reserve(items.len() * COVERAGE_ITEM_STRIDE);
    for item in items {
        push_rect(destination, item.visible);
        push_rect(destination, item.opaque.unwrap_or(Rect::EMPTY));
        let mut flags = 0u32;
        if item.cullable {
            flags |= FLAG_CULLABLE;
        }
        if item.protected {
            flags |= FLAG_PROTECTED;
        }
        if item.opaque.is_some() {
            flags |= FLAG_HAS_OPAQUE;
        }
        destination.extend_from_slice(&flags.to_le_bytes());
        destination.extend_from_slice(&[0u8; 12]);
    }
}

/// Encode a layer's poison regions for `shaders/occlusion.wgsl`.
pub fn encode_poison_regions(poison: &[PoisonRegion], destination: &mut Vec<u8>) {
    destination.clear();
    destination.reserve(poison.len() * POISON_REGION_STRIDE);
    for zone in poison {
        push_rect(destination, zone.region);
        destination.extend_from_slice(&zone.above_index.to_le_bytes());
        destination.extend_from_slice(&[0u8; 12]);
    }
}

fn push_rect(destination: &mut Vec<u8>, rect: Rect) {
    for value in rect.to_array() {
        destination.extend_from_slice(&value.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(x: f32, y: f32, width: f32, height: f32) -> Rect {
        Rect::from_origin_size([x, y], [width, height])
    }

    fn unclipped() -> Rect {
        rect(-100_000.0, -100_000.0, 200_000.0, 200_000.0)
    }

    fn opaque_at(bounds: Rect) -> CoverageItem {
        CoverageItem::occluder(bounds, bounds)
    }

    #[test]
    fn a_fully_covered_cullee_is_dropped() {
        let items = [
            CoverageItem::cullee(rect(10.0, 10.0, 40.0, 40.0)),
            CoverageItem::cullee(rect(200.0, 200.0, 40.0, 40.0)),
            opaque_at(rect(0.0, 0.0, 100.0, 100.0)),
        ];
        assert_eq!(keep_mask(&items, &[]), vec![false, true, true]);
    }

    #[test]
    fn paint_order_decides_which_way_coverage_runs() {
        let below = [
            opaque_at(rect(0.0, 0.0, 100.0, 100.0)),
            CoverageItem::cullee(rect(10.0, 10.0, 40.0, 40.0)),
        ];
        assert_eq!(
            keep_mask(&below, &[]),
            vec![true, true],
            "an occluder painted first is beneath the cullee and hides nothing"
        );

        let above = [
            CoverageItem::cullee(rect(10.0, 10.0, 40.0, 40.0)),
            opaque_at(rect(0.0, 0.0, 100.0, 100.0)),
        ];
        assert_eq!(keep_mask(&above, &[]), vec![false, true]);
    }

    #[test]
    fn a_chain_of_coverage_drops_every_link_but_the_top() {
        // Bottom is covered by middle; middle is covered by top. Both go — the
        // middle one's own keep decision never enters into the bottom one's,
        // which is the legacy sweep's "an occluder that is itself covered still
        // hides what lies beneath it" rule, restated per-item.
        let items = [
            CoverageItem::cullee(rect(10.0, 10.0, 10.0, 10.0)),
            opaque_at(rect(0.0, 0.0, 50.0, 50.0)),
            opaque_at(rect(0.0, 0.0, 100.0, 100.0)),
        ];
        assert_eq!(keep_mask(&items, &[]), vec![false, false, true]);
    }

    #[test]
    fn a_protected_item_neither_culls_nor_is_culled() {
        let mut occluder = opaque_at(rect(0.0, 0.0, 100.0, 100.0));
        occluder.protected = true;
        let items = [CoverageItem::cullee(rect(10.0, 10.0, 20.0, 20.0)), occluder];
        assert_eq!(
            keep_mask(&items, &[]),
            vec![true, true],
            "an occluder inside a filter group never joins the occluder set"
        );

        let mut cullee = CoverageItem::cullee(rect(10.0, 10.0, 20.0, 20.0));
        cullee.protected = true;
        let items = [cullee, opaque_at(rect(0.0, 0.0, 100.0, 100.0))];
        assert_eq!(keep_mask(&items, &[]), vec![true, true]);
    }

    #[test]
    fn a_non_cullable_item_survives_full_coverage() {
        let mut shadow = CoverageItem::cullee(rect(10.0, 10.0, 20.0, 20.0));
        shadow.cullable = false;
        let items = [shadow, opaque_at(rect(0.0, 0.0, 100.0, 100.0))];
        assert_eq!(keep_mask(&items, &[]), vec![true, true]);
    }

    #[test]
    fn a_backdrop_filter_poisons_everything_beneath_it() {
        let items = [
            CoverageItem::cullee(rect(30.0, 30.0, 20.0, 20.0)),
            opaque_at(rect(0.0, 0.0, 100.0, 100.0)),
        ];
        // Declared above both, dilated by its own blur radius of 8.
        let poison = [PoisonRegion {
            region: rect(10.0, 10.0, 60.0, 60.0).dilate(8.0),
            above_index: 2,
        }];
        assert_eq!(
            keep_mask(&items, &poison),
            vec![true, true],
            "the filter reads the pixels behind it, so the covered cullee must emit"
        );
        assert_eq!(
            keep_mask(&items, &[]),
            vec![false, true],
            "and without the filter the same cullee is dropped"
        );
    }

    #[test]
    fn the_blur_margin_is_what_extends_a_filters_reach() {
        let cullee = rect(0.0, 0.0, 6.0, 6.0);
        let items = [
            CoverageItem::cullee(cullee),
            opaque_at(rect(-10.0, -10.0, 40.0, 40.0)),
        ];
        // A filter group opening at stream position 1 — above the cullee,
        // below the occluder, so only the cullee is in reach. Its own bounds
        // start at x = 10, well clear of the cullee...
        let undilated = [PoisonRegion {
            region: rect(10.0, 0.0, 20.0, 20.0),
            above_index: 1,
        }];
        assert_eq!(keep_mask(&items, &undilated), vec![false, true]);
        // ...but a blur radius of 8 reaches back to x = 2 and protects it.
        let dilated = [PoisonRegion {
            region: rect(10.0, 0.0, 20.0, 20.0).dilate(8.0),
            above_index: 1,
        }];
        assert_eq!(keep_mask(&items, &dilated), vec![true, true]);
    }

    #[test]
    fn a_filter_does_not_poison_what_is_painted_above_it() {
        let items = [
            opaque_at(rect(0.0, 0.0, 100.0, 100.0)),
            CoverageItem::cullee(rect(10.0, 10.0, 20.0, 20.0)),
            opaque_at(rect(0.0, 0.0, 100.0, 100.0)),
        ];
        // Declared below the cullee: it reads item 0, not item 1.
        let poison = [PoisonRegion {
            region: rect(0.0, 0.0, 100.0, 100.0),
            above_index: 1,
        }];
        assert_eq!(keep_mask(&items, &poison), vec![true, false, true]);
    }

    #[test]
    fn a_poisoned_occluder_stops_occluding_even_when_the_cullee_is_clear() {
        // The occluder is wide; the cullee sits at its right-hand end. The
        // poison zone touches only the occluder's left-hand end, so the cullee
        // itself is *not* poisoned — the only thing that changed is that the
        // occluder may no longer collect.
        let items = [
            CoverageItem::cullee(rect(80.0, 0.0, 20.0, 50.0)),
            opaque_at(rect(0.0, 0.0, 100.0, 50.0)),
        ];
        assert_eq!(keep_mask(&items, &[]), vec![false, true]);
        let poison = [PoisonRegion {
            region: rect(0.0, 0.0, 20.0, 50.0),
            above_index: 2,
        }];
        assert_eq!(keep_mask(&items, &poison), vec![true, true]);
    }

    #[test]
    fn a_rounded_occluder_leaves_its_corners_visible() {
        let quad = Quad {
            origin: [0.0, 0.0],
            size: [100.0, 100.0],
            background: [0.0, 0.0, 0.0, 1.0],
            border_color: [0.0, 0.0, 0.0, 1.0],
            corner_radius: 15.0,
            border_width: 0.0,
        };
        let items = [
            CoverageItem::cullee(rect(0.0, 0.0, 100.0, 100.0)),
            quad_coverage_item(&quad, unclipped(), false),
        ];
        assert_eq!(
            keep_mask(&items, &[]),
            vec![true, true],
            "the corner-radius inset leaves a flush cullee's corners uncovered"
        );
    }

    #[test]
    fn a_square_opaque_quad_covers_a_flush_cullee() {
        let quad = Quad {
            origin: [0.0, 0.0],
            size: [100.0, 100.0],
            background: [0.2, 0.2, 0.2, 1.0],
            border_color: [0.0, 0.0, 0.0, 1.0],
            corner_radius: 0.0,
            border_width: 0.0,
        };
        let items = [
            CoverageItem::cullee(rect(0.0, 0.0, 100.0, 100.0)),
            quad_coverage_item(&quad, unclipped(), false),
        ];
        assert_eq!(keep_mask(&items, &[]), vec![false, true]);
    }

    #[test]
    fn a_translucent_quad_is_never_an_occluder() {
        let quad = Quad {
            origin: [0.0, 0.0],
            size: [100.0, 100.0],
            background: [0.2, 0.2, 0.2, 0.9],
            border_color: [0.0, 0.0, 0.0, 1.0],
            corner_radius: 0.0,
            border_width: 0.0,
        };
        let item = quad_coverage_item(&quad, unclipped(), false);
        assert_eq!(item.opaque, None);
    }

    #[test]
    fn a_quad_is_clipped_by_the_mask_the_caller_supplies() {
        let quad = Quad {
            origin: [0.0, 0.0],
            size: [100.0, 100.0],
            background: [0.2, 0.2, 0.2, 1.0],
            border_color: [0.0, 0.0, 0.0, 1.0],
            corner_radius: 0.0,
            border_width: 0.0,
        };
        let item = quad_coverage_item(&quad, rect(25.0, 25.0, 25.0, 25.0), false);
        assert_eq!(item.visible, rect(25.0, 25.0, 25.0, 25.0));
        assert_eq!(item.opaque, Some(rect(25.0, 25.0, 25.0, 25.0)));
    }

    #[test]
    fn two_side_by_side_occluders_jointly_cover_one_wide_cullee() {
        let items = [
            CoverageItem::cullee(rect(0.0, 0.0, 200.0, 50.0)),
            opaque_at(rect(0.0, 0.0, 100.0, 50.0)),
            opaque_at(rect(100.0, 0.0, 100.0, 50.0)),
        ];
        assert_eq!(keep_mask(&items, &[]), vec![false, true, true]);
    }

    #[test]
    fn stats_count_what_the_mask_decided() {
        let items = [
            CoverageItem::cullee(rect(10.0, 10.0, 20.0, 20.0)),
            CoverageItem::cullee(rect(500.0, 500.0, 20.0, 20.0)),
            opaque_at(rect(0.0, 0.0, 100.0, 100.0)),
        ];
        let (keep, stats) = keep_mask_with_stats(&items, &[]);
        assert_eq!(keep, vec![false, true, true]);
        assert_eq!(
            stats,
            OcclusionStats {
                culled: 1,
                kept: 2,
                poisoned: 0
            }
        );
    }

    #[test]
    fn stats_count_a_poisoned_survivor_separately() {
        let items = [
            CoverageItem::cullee(rect(10.0, 10.0, 20.0, 20.0)),
            opaque_at(rect(0.0, 0.0, 100.0, 100.0)),
        ];
        let poison = [PoisonRegion {
            region: rect(0.0, 0.0, 100.0, 100.0),
            above_index: 1,
        }];
        let (_, stats) = keep_mask_with_stats(&items, &poison);
        assert_eq!(stats.poisoned, 1);
    }

    #[test]
    fn an_empty_stream_decides_nothing() {
        assert!(keep_mask(&[], &[]).is_empty());
    }

    #[test]
    fn encoding_produces_one_fixed_stride_record_per_item() {
        let mut occluder = opaque_at(rect(1.0, 2.0, 3.0, 4.0));
        occluder.protected = true;
        let items = [CoverageItem::cullee(rect(0.0, 0.0, 1.0, 1.0)), occluder];
        let mut bytes = Vec::new();
        encode_coverage_items(&items, &mut bytes);
        assert_eq!(bytes.len(), 2 * COVERAGE_ITEM_STRIDE);

        let first_flags = u32::from_le_bytes([bytes[32], bytes[33], bytes[34], bytes[35]]);
        assert_eq!(first_flags, FLAG_CULLABLE);
        let second = COVERAGE_ITEM_STRIDE;
        let second_flags = u32::from_le_bytes([
            bytes[second + 32],
            bytes[second + 33],
            bytes[second + 34],
            bytes[second + 35],
        ]);
        assert_eq!(
            second_flags,
            FLAG_CULLABLE | FLAG_PROTECTED | FLAG_HAS_OPAQUE
        );
        assert_eq!(&bytes[second..second + 4], &1.0f32.to_le_bytes());
    }

    #[test]
    fn encoding_poison_regions_carries_the_index_they_apply_below() {
        let poison = [PoisonRegion {
            region: rect(5.0, 0.0, 10.0, 10.0),
            above_index: 7,
        }];
        let mut bytes = Vec::new();
        encode_poison_regions(&poison, &mut bytes);
        assert_eq!(bytes.len(), POISON_REGION_STRIDE);
        assert_eq!(&bytes[0..4], &5.0f32.to_le_bytes());
        assert_eq!(&bytes[16..20], &7u32.to_le_bytes());
    }

    #[test]
    fn the_mode_switch_reads_the_environment_r_n_8_5_specifies() {
        // Serialised implicitly: this is the only test that touches the var.
        // SAFETY: single-threaded within this test, and the value is restored
        // before returning so no other test observes it.
        let restore = std::env::var("WGPUI_OCCLUSION").ok();
        for (value, expected) in [
            ("0", Mode::Off),
            ("off", Mode::Off),
            ("validate", Mode::Validate),
            ("1", Mode::Normal),
        ] {
            unsafe { std::env::set_var("WGPUI_OCCLUSION", value) };
            assert_eq!(Mode::from_environment(), expected);
        }
        unsafe { std::env::remove_var("WGPUI_OCCLUSION") };
        assert_eq!(Mode::from_environment(), Mode::Normal);
        assert!(Mode::Normal.is_enabled());
        assert!(!Mode::Off.is_enabled());
        if let Some(value) = restore {
            unsafe { std::env::set_var("WGPUI_OCCLUSION", value) };
        }
    }
}
