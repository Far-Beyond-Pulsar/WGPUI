//! Scene-side packing substrate for per-layer persistent GPU slabs (retained
//! rendering Pillar III, epic #84, spec #94, `docs/retained-layers.md` §4.2).
//!
//! [`pack_layer_items`] turns one layer's recorded [`LayerItem`]s — paint
//! order, carrying layer-local draw orders — into a [`PackedLayer`]: per-kind
//! primitive arrays sorted exactly as [`Scene::finish`](crate::scene::Scene::finish)
//! sorts the frame's flat arrays, plus a [`PackedLayer::runs`] manifest
//! describing every contiguous same-kind stretch in final draw order, so the
//! renderer can issue one instanced draw per run without re-deriving any
//! ordering at draw time.
//!
//! The output is proven equivalent to today's legacy path by golden-model
//! tests (this module's `tests`), which replay synthetic scenes through
//! `Scene::finish` + `Scene::batches` and compare primitive-for-primitive,
//! byte-for-byte modulo the draw-order field (packed arrays carry layer-local
//! orders, the legacy arrays global ones — the same relative-order-only
//! equivalence the retained-layer ordering tests establish).
//!
//! # Kill switch
//!
//! Everything here sits behind [`slabs_enabled`] (`WGPUI_SLABS`, default on,
//! read once — the `WGPUI_LAYERS` precedent). Every frame-path consumer
//! checks the switch before doing work, so with it off the packing code is
//! simply never reached and the legacy per-frame path stands.
//!
//! # Fail-loud policy
//!
//! A layer whose content cannot be expressed as slab instances — `Surface`,
//! `BackdropFilter` or `FilterBoundary` primitives — must fall back to the
//! legacy per-frame path rather than pack partially: silently dropping one
//! would silently drop visible pixels. Rejections warn once, bump a `slab:`
//! render counter, and `debug_assert!`; the caller receives
//! [`PackOutcome::FellBack`] and composites that layer the old way. The same
//! applies to path items whose recorded ids are inconsistent (two paths
//! claiming one id — a producer that skipped id reassignment), which is the
//! only way "missing path data" can manifest, because items carry their path
//! geometry inline; see "Path ids" below.
//!
//! # Path ids
//!
//! A captured path's `id` addresses the *recording* frame's `Scene::paths`
//! array and means nothing afterwards. Packing renumbers ids densely per
//! layer (`packed.paths[i].id == i`) and counts path runs in **vertices**,
//! making the packed data fully self-contained: the renderer uploads
//! `paths[i].vertices` concatenated in array order and never resolves an id
//! against any global array. Splicing into legacy global draws later is pure
//! offset arithmetic (append the layer's vertex block at the current global
//! vertex count); no recorded global `PathId` survives the pack, which
//! removes the dangling-reference hazard by construction instead of trying to
//! detect it at use time.
//!
//! # What the consuming wave may rely on
//!
//! - Each non-path kind's array *is* the slab's content in order;
//!   `KindRun::start/count` are instance indices into it (and, offset by the
//!   layer's `SlabRange` base, into the global buffer).
//! - `SlabKind::Paths` runs address the layer's flattened vertex stream:
//!   prefix sums of `paths[i].vertices.len()` define each path's sub-range,
//!   and [`PackedLayer::total_path_vertices`] sizes the reservation handed to
//!   the slab allocator.
//! - Sprite runs never span two atlas textures (`texture_id` is `Some`
//!   exactly for sprite kinds); adjacent same-kind runs may share a texture
//!   only where the legacy batch iterator split a batch at an order threshold
//!   — merging such neighbours is left to the renderer if it ever wants
//!   fewer draws.
//! - `LayerItem::Nested` references are never packed here; a nested layer is
//!   its own retained record, packed independently under its own key.

// The renderer consumes `PackedLayer` through the scene's spans; the
// `dead_code` allowance stays because not every helper here has a non-test
// caller yet.
#![allow(dead_code)]

use std::sync::{Arc, LazyLock};
use std::sync::atomic::{AtomicBool, Ordering};

use collections::FxHashSet;

use crate::layer::{LayerItem, LayerKey};
use crate::platform::cross::slab::SlabKind;
use crate::scene::{
    DrawOrder, MonochromeSprite, Path, PathId, PolychromeSprite, Primitive, PrimitiveKind, Quad,
    Shadow, Underline,
};
use crate::{AtlasTextureId, ScaledPixels};

/// Whether per-layer persistent GPU slabs are live.
///
/// `WGPUI_SLABS=0` keeps every consumer on the legacy per-frame path. Read
/// once, at first use — same convention as `layers_enabled`.
pub(crate) fn slabs_enabled() -> bool {
    static ENABLED: LazyLock<bool> = LazyLock::new(|| {
        std::env::var("WGPUI_SLABS")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(true)
    });
    *ENABLED
}

/// Why a layer's content could not be packed and must composite through the
/// legacy per-frame path this slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FallbackReason {
    /// The layer holds a primitive with no slab kind. These primitives take
    /// part in render-target switching or external-surface composition rather
    /// than plain instanced draws, so there is no honest instance encoding
    /// for them yet.
    UnsupportedPrimitive(PrimitiveKind),
    /// Two path items claimed the same recorded `PathId`: a producer skipped
    /// the id-reassignment contract, so the item list cannot be trusted to be
    /// self-consistent.
    InconsistentPathIds,
}

/// The result of packing one layer's items.
pub(crate) enum PackOutcome {
    /// The layer's own primitives, ready for per-kind slab upload.
    Packed(Box<PackedLayer>),
    /// The layer must composite through the legacy per-frame path; the reason
    /// has already been reported (warn-once, counter, debug assert).
    FellBack(FallbackReason),
}

const COUNTER_LAYERS_PACKED: &str = "slab: layers packed";
const COUNTER_UNSUPPORTED_KIND: &str = "slab: pack fell back (unsupported kind)";
const COUNTER_INCONSISTENT_IDS: &str = "slab: pack failed (inconsistent path ids)";

/// One layer's own primitives, grouped per slab kind and sorted exactly as
/// `Scene::finish` sorts the frame's flat arrays, plus the interleaved
/// draw-order manifest.
///
/// Deliberately excludes nested layers ([`LayerItem::Nested`]): each is its
/// own retained record, packed separately under its own key.
pub(crate) struct PackedLayer {
    pub quads: Vec<Quad>,
    pub shadows: Vec<Shadow>,
    /// Sorted by draw order; `id` reassigned densely (`0..n`, matching the
    /// array position). See the module docs for the vertex-stream contract.
    pub paths: Vec<Path<ScaledPixels>>,
    pub underlines: Vec<Underline>,
    pub mono_sprites: Vec<MonochromeSprite>,
    pub poly_sprites: Vec<PolychromeSprite>,
    /// Every contiguous same-kind stretch of this layer's content, in exact
    /// final draw order — the per-layer analog of `Scene::batches`.
    pub runs: Vec<KindRun>,
}

impl PackedLayer {
    /// Total vertices across all packed paths: the count to reserve from the
    /// slab allocator for this layer's `Paths` slab.
    pub fn total_path_vertices(&self) -> u32 {
        self.paths.iter().fold(0u32, |total, path| {
            total.saturating_add(path.vertices.len() as u32)
        })
    }
}

/// One contiguous, single-kind stretch of a layer's packed content, drawn as
/// one instanced draw call.
///
/// `start`/`count` are element indices, never bytes: instance indices into
/// the kind's packed array for every kind except [`SlabKind::Paths`] — paths
/// draw as plain vertex ranges and each contributes a different number of
/// vertices, so there `start`/`count` address the layer's *flattened vertex
/// stream* (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct KindRun {
    pub kind: SlabKind,
    pub start: u32,
    pub count: u32,
    /// The atlas texture every sprite in this run binds. `Some` exactly when
    /// `kind` is a sprite kind; runs split on texture change because one draw
    /// call binds one texture.
    pub texture_id: Option<AtlasTextureId>,
}

/// A layer's own content, packed once at record time and spliced by every
/// subsequent composite until the next record or eviction.
///
/// Stretches of the layer's own primitives sit in paint order alongside the
/// nested-layer references that separated them — the cached analog of
/// `Window::build_slab_segments`' segmentation. Everything that does not vary
/// between frames is precomputed here: the per-kind totals, and each
/// stretch's runs already offset into the layer-wide streams. A composite
/// then costs one span emission per stretch, with no packing arithmetic —
/// which is the point, since the same content composites many times between
/// re-records.
///
/// Held behind an [`Arc`] on the layer so span emission clones references,
/// not bytes; see `Layer::packed` for the lifecycle.
pub(crate) struct RecordedSlabPack {
    /// Per-kind instance totals across ALL stretches: what whichever span
    /// arrives first reserves from the allocator. Paths count vertices —
    /// their slab holds the flattened stream, not path structs.
    pub totals: [u32; SlabKind::COUNT],
    /// The layer's content in paint order.
    pub pieces: Vec<SlabPackPiece>,
}

/// One piece of a [`RecordedSlabPack`].
pub(crate) enum SlabPackPiece {
    /// A maximal stretch of the layer's own primitives: its draws,
    /// `start`-offset into the layer-wide concatenated streams, over the
    /// packed bytes themselves (origin-relative; see `make_packed_relative`).
    Stretch {
        runs: Vec<crate::scene::SlabRun>,
        packed: Arc<PackedLayer>,
    },
    /// A layer painted inside this one. Never packed into the parent; it is
    /// composited recursively under its own key, choosing its own
    /// representation.
    Nested(LayerKey),
}

/// Check a layer's items against the packing contract without side effects.
///
/// Production entry points report rejections loudly (warn-once, counter,
/// debug assert) and fall back; tests call this directly to assert detection
/// without tripping the debug assertion.
pub(crate) fn validate_packable(items: &[LayerItem]) -> Result<(), FallbackReason> {
    let mut seen_path_ids: FxHashSet<PathId> = FxHashSet::default();
    for item in items {
        let LayerItem::Primitive(primitive) = item else {
            // A nested layer packs independently under its own key.
            continue;
        };
        let Some(kind) = slab_kind(primitive) else {
            return Err(FallbackReason::UnsupportedPrimitive(
                primitive_discriminant(primitive),
            ));
        };
        // `slab_kind` answers `Paths` only for `Primitive::Path`, and the
        // contract producers uphold is uniqueness of recorded ids within one
        // recording (`insert_primitive` assigns `PathId(self.paths.len())`).
        if let (SlabKind::Paths, Primitive::Path(path)) = (kind, primitive) {
            if !seen_path_ids.insert(path.id) {
                return Err(FallbackReason::InconsistentPathIds);
            }
        }
    }
    Ok(())
}

/// The cheapest possible packability answer: does any item hold a primitive
/// no slab encodes?
///
/// Unlike [`validate_packable`] this never allocates — it cannot check
/// path-id consistency, which needs the id set — so the rejection that is
/// both common and permanent for a given layer (a backdrop filter, a surface)
/// short-circuits before any packing work. Side-effect-free on purpose:
/// probes (tests, `build_slab_segments`) must stay quiet; production entry
/// points that decide to fall back report via [`report_rejection`].
pub(crate) fn first_unsupported_kind(items: &[LayerItem]) -> Option<FallbackReason> {
    items.iter().find_map(|item| {
        let LayerItem::Primitive(primitive) = item else {
            // A nested layer packs independently under its own key.
            return None;
        };
        slab_kind(primitive).is_none().then(|| {
            FallbackReason::UnsupportedPrimitive(primitive_discriminant(primitive))
        })
    })
}

/// Pack one layer's recorded items into per-kind slab arrays plus the
/// interleaved run manifest.
///
/// On any contract violation the outcome is [`PackOutcome::FellBack`] after a
/// loud report; pixels are never silently dropped.
pub(crate) fn pack_layer_items(items: &[LayerItem]) -> PackOutcome {
    match validate_packable(items) {
        Err(reason) => {
            report_rejection(reason);
            debug_assert!(
                false,
                "slab packing rejected a layer ({reason:?}); it must composite through the \
                 legacy path"
            );
            PackOutcome::FellBack(reason)
        }
        Ok(()) => {
            let packed = build_packed_layer(items);
            crate::render_stats::count(COUNTER_LAYERS_PACKED);
            PackOutcome::Packed(Box::new(packed))
        }
    }
}

/// Which slab kind hosts `primitive`, or `None` for the primitives no slab
/// encodes this slice.
fn slab_kind(primitive: &Primitive) -> Option<SlabKind> {
    match primitive {
        Primitive::Quad(_) => Some(SlabKind::Quads),
        Primitive::Shadow(_) => Some(SlabKind::Shadows),
        Primitive::Path(_) => Some(SlabKind::Paths),
        Primitive::Underline(_) => Some(SlabKind::Underlines),
        Primitive::MonochromeSprite(_) => Some(SlabKind::MonoSprites),
        Primitive::PolychromeSprite(_) => Some(SlabKind::PolySprites),
        // Surfaces compose external wgpu content through the surface
        // registry; backdrop filters and filter-group boundaries drive
        // render-target switches. None of that is an instanced slab yet.
        Primitive::Surface(_) | Primitive::BackdropFilter(_) | Primitive::FilterBoundary(_) => {
            None
        }
    }
}

/// Where `primitive` sorts relative to other kinds at an equal draw order —
/// the same discriminants `Scene::batches` merges the frame's batches by.
fn primitive_discriminant(primitive: &Primitive) -> PrimitiveKind {
    match primitive {
        Primitive::Shadow(_) => PrimitiveKind::Shadow,
        Primitive::Quad(_) => PrimitiveKind::Quad,
        Primitive::Path(_) => PrimitiveKind::Path,
        Primitive::Underline(_) => PrimitiveKind::Underline,
        Primitive::MonochromeSprite(_) => PrimitiveKind::MonochromeSprite,
        Primitive::PolychromeSprite(_) => PrimitiveKind::PolychromeSprite,
        Primitive::Surface(_) => PrimitiveKind::Surface,
        Primitive::BackdropFilter(_) => PrimitiveKind::BackdropFilter,
        Primitive::FilterBoundary(boundary) => {
            if boundary.is_start {
                PrimitiveKind::FilterBoundaryStart
            } else {
                PrimitiveKind::FilterBoundaryEnd
            }
        }
    }
}

static UNSUPPORTED_WARNED: LazyLock<AtomicBool> = LazyLock::new(|| AtomicBool::new(false));
static INCONSISTENT_IDS_WARNED: LazyLock<AtomicBool> = LazyLock::new(|| AtomicBool::new(false));

/// Warn once per reason and bump the corresponding counter.
///
/// Production entry points call this when *they* have decided to fall back —
/// typically after [`first_unsupported_kind`] rejected a layer before any
/// packing work. The debug assertion deliberately does not live here: a
/// record-time fallback is a supported outcome (the layer composites through
/// the legacy path), and `pack_layer_items` keeps its own assertion for the
/// producer-bug case where rejection is discovered mid-pack.
pub(crate) fn report_rejection(reason: FallbackReason) {
    match reason {
        FallbackReason::UnsupportedPrimitive(discriminant) => {
            if !UNSUPPORTED_WARNED.swap(true, Ordering::AcqRel) {
                log::warn!(
                    "slab packing: layer holds {discriminant:?}, which has no slab kind yet; \
                     compositing that layer through the legacy per-frame path"
                );
            }
            crate::render_stats::count(COUNTER_UNSUPPORTED_KIND);
        }
        FallbackReason::InconsistentPathIds => {
            if !INCONSISTENT_IDS_WARNED.swap(true, Ordering::AcqRel) {
                log::warn!(
                    "slab packing: two path items claim the same PathId; compositing that \
                     layer through the legacy per-frame path"
                );
            }
            crate::render_stats::count(COUNTER_INCONSISTENT_IDS);
        }
    }
}

/// Gather, sort, renumber, and manifest one validated layer.
fn build_packed_layer(items: &[LayerItem]) -> PackedLayer {
    let mut packed = PackedLayer {
        quads: Vec::new(),
        shadows: Vec::new(),
        paths: Vec::new(),
        underlines: Vec::new(),
        mono_sprites: Vec::new(),
        poly_sprites: Vec::new(),
        runs: Vec::new(),
    };
    for item in items {
        // Validation already rejected anything unpickable; a nested reference
        // simply contributes nothing here.
        let LayerItem::Primitive(primitive) = item else {
            continue;
        };
        let Some(kind) = slab_kind(primitive) else {
            continue;
        };
        match primitive {
            Primitive::Quad(quad) => packed.quads.push(*quad),
            Primitive::Shadow(shadow) => packed.shadows.push(*shadow),
            Primitive::Underline(underline) => packed.underlines.push(*underline),
            Primitive::MonochromeSprite(sprite) => packed.mono_sprites.push(*sprite),
            Primitive::PolychromeSprite(sprite) => packed.poly_sprites.push(*sprite),
            Primitive::Path(path) => packed.paths.push(path.clone()),
            _ => debug_assert!(false, "{kind:?} classified but not gathered"),
        }
    }

    // Mirror `Scene::finish`'s stable sorts exactly: same keys, so ties
    // (equal orders; sprites further tied on tile) resolve to insertion order
    // on both sides.
    packed.quads.sort_by_key(|quad| quad.order);
    packed.shadows.sort_by_key(|shadow| shadow.order);
    packed.paths.sort_by_key(|path| path.order);
    packed.underlines.sort_by_key(|underline| underline.order);
    packed
        .mono_sprites
        .sort_by_key(|sprite| (sprite.order, sprite.tile.tile_id));
    packed
        .poly_sprites
        .sort_by_key(|sprite| (sprite.order, sprite.tile.tile_id));

    // Renumber after sorting so a path's dense local id equals its position
    // in this array (and therefore its slot in the vertex-stream prefix
    // sums). The recorded ids addressed the recording frame's global paths
    // array and are discarded unconditionally.
    for (index, path) in packed.paths.iter_mut().enumerate() {
        path.id = PathId(index);
    }

    packed.runs = build_kind_runs(&packed);
    packed
}

/// Merge the six sorted per-kind arrays back into draw order, run-length
/// encoded by kind.
///
/// At an equal order kinds interleave by the same discriminants
/// `Scene::batches` merges by ([`PrimitiveKind`]'s declaration order), which
/// is what makes the expanded run sequence identical to the legacy batch
/// stream — asserted entry-for-entry by the golden-model tests.
fn build_kind_runs(packed: &PackedLayer) -> Vec<KindRun> {
    // Indexed via `SlabKind::index`, so this must stay in `SlabKind::ALL`
    // declaration order.
    let lengths = [
        packed.quads.len(),
        packed.shadows.len(),
        packed.paths.len(),
        packed.underlines.len(),
        packed.mono_sprites.len(),
        packed.poly_sprites.len(),
    ];
    let discriminant_of = |kind: SlabKind| match kind {
        SlabKind::Quads => PrimitiveKind::Quad,
        SlabKind::Shadows => PrimitiveKind::Shadow,
        SlabKind::Paths => PrimitiveKind::Path,
        SlabKind::Underlines => PrimitiveKind::Underline,
        SlabKind::MonoSprites => PrimitiveKind::MonochromeSprite,
        SlabKind::PolySprites => PrimitiveKind::PolychromeSprite,
    };
    let order_at = |kind: SlabKind, cursor: usize| -> DrawOrder {
        match kind {
            SlabKind::Quads => packed.quads[cursor].order,
            SlabKind::Shadows => packed.shadows[cursor].order,
            SlabKind::Paths => packed.paths[cursor].order,
            SlabKind::Underlines => packed.underlines[cursor].order,
            SlabKind::MonoSprites => packed.mono_sprites[cursor].order,
            SlabKind::PolySprites => packed.poly_sprites[cursor].order,
        }
    };
    let texture_at = |kind: SlabKind, cursor: usize| -> Option<AtlasTextureId> {
        match kind {
            SlabKind::MonoSprites => Some(packed.mono_sprites[cursor].tile.texture_id),
            SlabKind::PolySprites => Some(packed.poly_sprites[cursor].tile.texture_id),
            _ => None,
        }
    };

    let mut cursors = [0usize; SlabKind::COUNT];
    // Running total of vertices consumed by earlier path runs; path runs
    // count vertices because paths draw as vertex ranges, not instances.
    let mut path_vertex_cursor = 0u32;
    let mut runs: Vec<KindRun> = Vec::new();
    loop {
        let mut chosen: Option<(SlabKind, (DrawOrder, PrimitiveKind))> = None;
        for kind in SlabKind::ALL {
            let cursor = cursors[kind.index()];
            if cursor >= lengths[kind.index()] {
                continue;
            }
            let key = (order_at(kind, cursor), discriminant_of(kind));
            if chosen.is_none_or(|(_, chosen_key)| key < chosen_key) {
                chosen = Some((kind, key));
            }
        }
        let Some((kind, _)) = chosen else {
            break;
        };
        let cursor = cursors[kind.index()];
        if kind == SlabKind::Paths {
            let vertex_count = packed.paths[cursor].vertices.len() as u32;
            match runs.last_mut() {
                Some(last) if last.kind == SlabKind::Paths => last.count += vertex_count,
                _ => runs.push(KindRun {
                    kind,
                    start: path_vertex_cursor,
                    count: vertex_count,
                    texture_id: None,
                }),
            }
            path_vertex_cursor += vertex_count;
        } else {
            let texture_id = texture_at(kind, cursor);
            match runs.last_mut() {
                // A texture change splits the run even mid-order-bucket: one
                // draw call binds one texture.
                Some(last) if last.kind == kind && last.texture_id == texture_id => last.count += 1,
                _ => runs.push(KindRun {
                    kind,
                    start: cursor as u32,
                    count: 1,
                    texture_id,
                }),
            }
        }
        cursors[kind.index()] += 1;
    }
    runs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{
        BackdropFilter, FilterBoundary, PaintSurface, PathVertex, Scene, SurfaceContent,
        TransformationMatrix,
    };
    use crate::{
        AtlasTile, Bounds, ContentMask, Corners, DevicePixels, Hsla, LayerKey, Point, Size,
        TextColor, TextColorTag, TileId, point, px, size,
    };
    use std::ops::Range;

    /// Monotonic marker allocator for the harness.
    fn bump(counter: &mut u32) -> u32 {
        let value = *counter;
        *counter += 1;
        value
    }

    // ------------------------------------------------------------------
    // Golden-model harness.
    //
    // Scenes are built through begin_layer/end_layer/insert_primitive, like
    // the retained-layer ordering tests. Every primitive carries a unique
    // numeric marker in a field ordering never reads, which lets each layer's
    // expected output be identified inside the finished scene's batch stream
    // without relying on private bookkeeping. Packing is compared against
    // that stream entry-for-entry, byte-for-byte aside from the draw-order
    // field: packed arrays carry layer-local orders while the finished
    // scene's arrays carry global ones, and relative order — not equality of
    // the integers — is the invariant, exactly as the ordering-equivalence
    // tests establish for the retained-layer phase itself.
    // ------------------------------------------------------------------

    fn sp(value: f32) -> ScaledPixels {
        ScaledPixels(value)
    }

    fn rect(x: f32, y: f32, w: f32, h: f32) -> Bounds<ScaledPixels> {
        Bounds {
            origin: Point {
                x: sp(x),
                y: sp(y),
            },
            size: Size {
                width: sp(w),
                height: sp(h),
            },
        }
    }

    /// Wide enough that nothing inserted is clipped empty — `insert_primitive`
    /// drops empty-clipped primitives entirely, which would desync marker
    /// counts if it happened on only one side of the comparison.
    fn mask() -> ContentMask<ScaledPixels> {
        ContentMask {
            bounds: rect(-1000., -1000., 10_000., 10_000.),
        }
    }

    fn marked_background(marker: u32) -> crate::Background {
        let mut background = crate::Background::default();
        background.solid.h = marker as f32;
        background
    }

    fn sprite_tile(
        texture_index: u32,
        kind: crate::AtlasTextureKind,
        tile_number: u32,
    ) -> AtlasTile {
        AtlasTile {
            texture_id: AtlasTextureId {
                index: texture_index,
                kind,
            },
            tile_id: TileId(tile_number),
            padding: 0,
            bounds: Bounds {
                origin: point(DevicePixels(0), DevicePixels(0)),
                size: size(DevicePixels(8), DevicePixels(8)),
            },
        }
    }

    fn quad_marked(bounds: Bounds<ScaledPixels>, marker: u32) -> Quad {
        Quad {
            bounds,
            content_mask: mask(),
            corner_radii: Corners {
                top_left: sp(marker as f32),
                ..Corners::default()
            },
            ..Default::default()
        }
    }

    fn shadow_marked(bounds: Bounds<ScaledPixels>, marker: u32) -> Shadow {
        Shadow {
            order: 0,
            blur_radius: sp(4.),
            bounds,
            corner_radii: Corners {
                top_left: sp(marker as f32),
                ..Corners::default()
            },
            content_mask: mask(),
            color: Hsla::default(),
        }
    }

    fn underline_marked(bounds: Bounds<ScaledPixels>, marker: u32) -> Underline {
        Underline {
            order: 0,
            pad: 0,
            bounds,
            content_mask: mask(),
            color: Hsla::default(),
            thickness: sp(marker as f32),
            wavy: 0,
        }
    }

    fn mono_sprite_marked(
        bounds: Bounds<ScaledPixels>,
        texture_index: u32,
        tile_number: u32,
        marker: u32,
    ) -> MonochromeSprite {
        MonochromeSprite {
            order: 0,
            pad: 0,
            bounds,
            content_mask: mask(),
            text_color: TextColor {
                tag: TextColorTag::Solid,
                color_space: crate::ColorSpace::Srgb,
                solid: Hsla {
                    h: marker as f32,
                    s: 1.,
                    l: 0.5,
                    a: 1.,
                },
                gradient_angle_or_reserved: 0.,
                colors: Default::default(),
                pad: 0,
            },
            tile: sprite_tile(texture_index, crate::AtlasTextureKind::Monochrome, tile_number),
            transformation: TransformationMatrix::unit(),
        }
    }

    fn poly_sprite_marked(
        bounds: Bounds<ScaledPixels>,
        texture_index: u32,
        tile_number: u32,
        marker: u32,
    ) -> PolychromeSprite {
        PolychromeSprite {
            order: 0,
            pad: 0,
            grayscale: 0,
            opacity: marker as f32,
            bounds,
            content_mask: mask(),
            corner_radii: Corners::default(),
            tile: sprite_tile(texture_index, crate::AtlasTextureKind::Polychrome, tile_number),
        }
    }

    /// A path of `triangle_count` triangles carrying the marker twice: in its
    /// color (compared per path) and in its first vertex's `st_position.x`,
    /// which travels with the vertex stream so the flattened comparison
    /// proves vertex-level draw order. The zig-zag keeps every triangle
    /// two-dimensional — a flat path has empty bounds and would be dropped
    /// by `insert_primitive` on both sides of the comparison.
    fn path_marked(marker: u32, origin: (f32, f32), triangle_count: usize) -> Path<ScaledPixels> {
        let (x, y) = origin;
        let mut pixels_path = crate::scene::Path::new(point(px(x), px(y)));
        pixels_path.content_mask = ContentMask {
            bounds: Bounds {
                origin: point(px(-1000.), px(-1000.)),
                size: size(px(10_000.), px(10_000.)),
            },
        };
        pixels_path.move_to(point(px(x), px(y)));
        for step in 0..triangle_count {
            let offset = (step + 2) as f32 * 20.;
            let lift = if step % 2 == 0 { 15. } else { 0. };
            pixels_path.line_to(point(px(x + offset), px(y + lift)));
        }
        let mut scaled = pixels_path.scale(1.0);
        scaled.color = marked_background(marker);
        scaled.vertices[0].st_position.x = marker as f32;
        scaled
    }

    fn boundary(is_start: bool) -> FilterBoundary {
        FilterBoundary {
            order: 0,
            bounds: rect(0., 0., 100., 100.),
            content_mask: mask(),
            corner_radii: Corners::default(),
            blur_radius: sp(8.),
            opacity: 1.,
            is_start,
        }
    }

    fn backdrop() -> BackdropFilter {
        BackdropFilter {
            bounds: rect(0., 0., 100., 100.),
            content_mask: mask(),
            corner_radii: Corners::default(),
            blur_radius: sp(20.),
            opacity: 1.,
            ..Default::default()
        }
    }

    fn surface() -> PaintSurface {
        PaintSurface {
            order: 0,
            bounds: rect(0., 0., 100., 100.),
            content_mask: mask(),
            content: SurfaceContent::Wgpu(crate::platform::cross::surface_registry::SurfaceId(7)),
        }
    }

    /// One identified primitive: slab kind plus unique marker.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Entry {
        kind: SlabKind,
        marker: u32,
    }

    fn quad_entry(quad: &Quad) -> Entry {
        Entry {
            kind: SlabKind::Quads,
            marker: quad.corner_radii.top_left.0 as u32,
        }
    }

    fn shadow_entry(shadow: &Shadow) -> Entry {
        Entry {
            kind: SlabKind::Shadows,
            marker: shadow.corner_radii.top_left.0 as u32,
        }
    }

    fn underline_entry(underline: &Underline) -> Entry {
        Entry {
            kind: SlabKind::Underlines,
            marker: underline.thickness.0 as u32,
        }
    }

    fn mono_entry(sprite: &MonochromeSprite) -> Entry {
        Entry {
            kind: SlabKind::MonoSprites,
            marker: sprite.text_color.solid.h as u32,
        }
    }

    fn poly_entry(sprite: &PolychromeSprite) -> Entry {
        Entry {
            kind: SlabKind::PolySprites,
            marker: sprite.opacity as u32,
        }
    }

    fn path_entry(path: &Path<ScaledPixels>) -> Entry {
        Entry {
            kind: SlabKind::Paths,
            marker: path.color.solid.h as u32,
        }
    }

    /// A typed reference to one element of the finished scene's batch stream,
    /// so per-kind bytes can be compared directly against the packed arrays.
    #[derive(Clone, Copy)]
    enum OracleSlot<'a> {
        Quad(&'a Quad),
        Shadow(&'a Shadow),
        Path(&'a Path<ScaledPixels>),
        Underline(&'a Underline),
        Mono(&'a MonochromeSprite),
        Poly(&'a PolychromeSprite),
    }

    impl OracleSlot<'_> {
        fn entry(&self) -> Entry {
            match self {
                OracleSlot::Quad(quad) => quad_entry(quad),
                OracleSlot::Shadow(shadow) => shadow_entry(shadow),
                OracleSlot::Path(path) => path_entry(path),
                OracleSlot::Underline(underline) => underline_entry(underline),
                OracleSlot::Mono(sprite) => mono_entry(sprite),
                OracleSlot::Poly(sprite) => poly_entry(sprite),
            }
        }
    }

    /// The oracle: flatten the finished scene's batch stream — the exact draw
    /// sequence the legacy path issues today — keeping the elements whose
    /// marker falls in `marker_range` (one disjoint range per layer).
    fn oracle_stream(scene: &Scene, marker_range: Range<u32>) -> Vec<OracleSlot<'_>> {
        let mut slots = Vec::new();
        for batch in scene.batches() {
            match batch {
                crate::scene::PrimitiveBatch::Quads(quads) => {
                    slots.extend(quads.iter().map(OracleSlot::Quad));
                }
                crate::scene::PrimitiveBatch::Shadows(shadows) => {
                    slots.extend(shadows.iter().map(OracleSlot::Shadow));
                }
                crate::scene::PrimitiveBatch::Paths(paths) => {
                    slots.extend(paths.iter().map(OracleSlot::Path));
                }
                crate::scene::PrimitiveBatch::Underlines(underlines) => {
                    slots.extend(underlines.iter().map(OracleSlot::Underline));
                }
                crate::scene::PrimitiveBatch::MonochromeSprites { sprites, .. } => {
                    slots.extend(sprites.iter().map(OracleSlot::Mono));
                }
                crate::scene::PrimitiveBatch::PolychromeSprites { sprites, .. } => {
                    slots.extend(sprites.iter().map(OracleSlot::Poly));
                }
                // No slab kind yet — never part of a packed comparison.
                crate::scene::PrimitiveBatch::Surfaces(_)
                | crate::scene::PrimitiveBatch::BackdropFilters(_)
                | crate::scene::PrimitiveBatch::FilterBoundary(_) => {}
            }
        }
        slots.retain(|slot| marker_range.contains(&slot.entry().marker));
        slots
    }

    /// Expand a packed layer's run manifest back into its draw sequence.
    /// Path runs walk the flattened vertex stream's prefix sums to recover
    /// their constituent paths in order.
    fn expand_runs(packed: &PackedLayer) -> Vec<Entry> {
        let mut stream = Vec::new();
        for run in &packed.runs {
            let (start, count) = (run.start as usize, run.count as usize);
            match run.kind {
                SlabKind::Quads => {
                    stream.extend(packed.quads[start..start + count].iter().map(quad_entry));
                }
                SlabKind::Shadows => {
                    stream.extend(packed.shadows[start..start + count].iter().map(shadow_entry));
                }
                SlabKind::Paths => {
                    let run_end = run.start + run.count;
                    let mut covered = 0u32;
                    let mut offset = 0u32;
                    for path in &packed.paths {
                        let len = path.vertices.len() as u32;
                        if len > 0 && offset < run_end && offset + len > run.start {
                            stream.push(path_entry(path));
                            covered += len;
                        }
                        offset += len;
                    }
                    assert_eq!(covered, run.count, "path runs must cover whole paths");
                }
                SlabKind::Underlines => {
                    stream.extend(
                        packed.underlines[start..start + count]
                            .iter()
                            .map(underline_entry),
                    );
                }
                SlabKind::MonoSprites => {
                    stream.extend(
                        packed.mono_sprites[start..start + count]
                            .iter()
                            .map(mono_entry),
                    );
                }
                SlabKind::PolySprites => {
                    stream.extend(
                        packed.poly_sprites[start..start + count]
                            .iter()
                            .map(poly_entry),
                    );
                }
            }
        }
        stream
    }

    /// Raw per-vertex tuple covering everything `GpuPathVertex` carries
    /// (position, ST coords, content-mask bounds); the color rides on the
    /// owning path and is compared separately.
    type RawVertex = (f32, f32, f32, f32, f32, f32, f32, f32);

    fn raw_vertex(vertex: &PathVertex<ScaledPixels>) -> RawVertex {
        (
            vertex.xy_position.x.0,
            vertex.xy_position.y.0,
            vertex.st_position.x,
            vertex.st_position.y,
            vertex.content_mask.bounds.origin.x.0,
            vertex.content_mask.bounds.origin.y.0,
            vertex.content_mask.bounds.size.width.0,
            vertex.content_mask.bounds.size.height.0,
        )
    }

    fn raw_vertices(path: &Path<ScaledPixels>) -> Vec<RawVertex> {
        path.vertices.iter().map(raw_vertex).collect()
    }

    /// Byte image of a POD primitive with alignment-padding and draw-order
    /// fields zeroed: padding bytes are unspecified by `repr(C)` copies made
    /// through user code (both sides originate from the same insert, but the
    /// clone path differs), and the order field legitimately differs between
    /// the layer-local and global order spaces.
    fn pod_image<T, Zero>(zero: &Zero, element: &T) -> Vec<u8>
    where
        T: Copy + bytemuck::NoUninit,
        Zero: Fn(&mut T),
    {
        let mut copy = *element;
        zero(&mut copy);
        bytemuck::bytes_of(&copy).to_vec()
    }

    /// The full equivalence check between one layer's packed form and the
    /// oracle: identical expanded draw sequence, byte-identical per-kind
    /// arrays, and identical flattened path vertex streams.
    fn assert_packing_matches_oracle(items: &[LayerItem], expected: &[OracleSlot<'_>]) {
        let PackOutcome::Packed(packed) = pack_layer_items(items) else {
            panic!("packing unexpectedly fell back");
        };

        let expected_entries: Vec<Entry> = expected.iter().map(|slot| slot.entry()).collect();
        assert_eq!(
            expand_runs(&packed),
            expected_entries,
            "expanded run sequence must equal the legacy batch stream"
        );

        let oracle_quads: Vec<&Quad> = expected
            .iter()
            .filter_map(|slot| match slot {
                OracleSlot::Quad(quad) => Some(*quad),
                _ => None,
            })
            .collect();
        let oracle_shadows: Vec<&Shadow> = expected
            .iter()
            .filter_map(|slot| match slot {
                OracleSlot::Shadow(shadow) => Some(*shadow),
                _ => None,
            })
            .collect();
        let oracle_underlines: Vec<&Underline> = expected
            .iter()
            .filter_map(|slot| match slot {
                OracleSlot::Underline(underline) => Some(*underline),
                _ => None,
            })
            .collect();
        let oracle_mono: Vec<&MonochromeSprite> = expected
            .iter()
            .filter_map(|slot| match slot {
                OracleSlot::Mono(sprite) => Some(*sprite),
                _ => None,
            })
            .collect();
        let oracle_poly: Vec<&PolychromeSprite> = expected
            .iter()
            .filter_map(|slot| match slot {
                OracleSlot::Poly(sprite) => Some(*sprite),
                _ => None,
            })
            .collect();

        assert_pod_kind(
            "quads",
            &packed.quads,
            &oracle_quads,
            |quad: &mut Quad| quad.order = 0,
        );
        assert_pod_kind(
            "shadows",
            &packed.shadows,
            &oracle_shadows,
            |shadow: &mut Shadow| shadow.order = 0,
        );
        assert_pod_kind(
            "underlines",
            &packed.underlines,
            &oracle_underlines,
            |underline: &mut Underline| {
                underline.order = 0;
                underline.pad = 0;
            },
        );
        assert_pod_kind(
            "monochrome sprites",
            &packed.mono_sprites,
            &oracle_mono,
            |sprite: &mut MonochromeSprite| {
                sprite.order = 0;
                sprite.pad = 0;
            },
        );
        assert_pod_kind(
            "polychrome sprites",
            &packed.poly_sprites,
            &oracle_poly,
            |sprite: &mut PolychromeSprite| {
                sprite.order = 0;
                sprite.pad = 0;
            },
        );

        // Flattened path vertex stream in draw order, both sides; plus the
        // per-path identity (color) and the dense-id contract.
        let oracle_paths: Vec<&Path<ScaledPixels>> = expected
            .iter()
            .filter_map(|slot| match slot {
                OracleSlot::Path(path) => Some(*path),
                _ => None,
            })
            .collect();
        let packed_stream: Vec<RawVertex> = packed
            .paths
            .iter()
            .flat_map(|path| raw_vertices(path))
            .collect();
        let oracle_vertices: Vec<RawVertex> =
            oracle_paths.iter().flat_map(|path| raw_vertices(path)).collect();
        assert_eq!(
            packed_stream.len() as u32,
            packed.total_path_vertices(),
            "total_path_vertices must agree with the flattened stream"
        );
        assert_eq!(packed_stream, oracle_vertices, "flattened path vertices");
        let packed_colors: Vec<crate::Background> =
            packed.paths.iter().map(|path| path.color).collect();
        let oracle_colors: Vec<crate::Background> =
            oracle_paths.iter().map(|path| path.color).collect();
        assert_eq!(packed_colors, oracle_colors, "path colors in draw order");
        for (index, path) in packed.paths.iter().enumerate() {
            assert_eq!(path.id, PathId(index), "path ids must be dense");
        }
    }

    /// Byte-level comparison of one POD kind's packed array against the
    /// oracle's elements of the same kind in draw order.
    fn assert_pod_kind<T>(
        label: &str,
        packed: &[T],
        oracle: &[&T],
        zero_volatile_fields: impl Fn(&mut T),
    ) where
        T: Copy + bytemuck::NoUninit,
    {
        assert_eq!(
            packed.len(),
            oracle.len(),
            "{label}: packed length differs from the oracle"
        );
        for (index, (packed_element, oracle_element)) in packed.iter().zip(oracle).enumerate() {
            assert_eq!(
                pod_image(&zero_volatile_fields, packed_element),
                pod_image(&zero_volatile_fields, oracle_element),
                "{label}[{index}] bytes differ"
            );
        }
    }

    // Counter effects are deliberately not asserted with render-stats
    // snapshots: exercising them requires globally forcing instrumentation
    // on (`set_force_enabled`), which races `render_stats`' own tests for
    // the same process-global flag — the pre-existing flake class visible
    // when running this suite in parallel. Detection and fallback behaviour
    // are asserted directly instead; the counters fire alongside those code
    // paths in production.

    fn unpack(outcome: PackOutcome) -> Box<PackedLayer> {
        match outcome {
            PackOutcome::Packed(packed) => packed,
            PackOutcome::FellBack(reason) => panic!("packing unexpectedly fell back: {reason:?}"),
        }
    }

    #[test]
    fn kill_switch_reads_consistently() {
        // Read-once convention like `layers_enabled`: whatever the ambient
        // environment says, consecutive reads agree.
        assert_eq!(slabs_enabled(), slabs_enabled());
    }

    #[test]
    fn single_kind_layer_packs_to_one_run_matching_the_legacy_stream() {
        let mut scene = Scene::default();
        scene.begin_layer(LayerKey(1), rect(0., 0., 200., 60.), true);
        for index in 0..5u32 {
            scene.insert_primitive(quad_marked(
                rect(index as f32 * 30., 0., 50., 50.),
                10 + index,
            ));
        }
        let items = scene.end_layer().unwrap();
        scene.finish();

        let oracle = oracle_stream(&scene, 10..15);
        assert_eq!(oracle.len(), 5, "every layer quad must reach the oracle");
        assert_packing_matches_oracle(&items, &oracle);

        let packed = unpack(pack_layer_items(&items));
        assert_eq!(
            packed.runs,
            vec![KindRun {
                kind: SlabKind::Quads,
                start: 0,
                count: 5,
                texture_id: None,
            }]
        );

        // Packing is deterministic: repacking yields identical manifests.
        let repacked = unpack(pack_layer_items(&items));
        assert_eq!(repacked.runs, packed.runs);
        assert_eq!(repacked.quads.len(), packed.quads.len());
    }

    #[test]
    fn all_six_kinds_interleaved_match_the_legacy_draw_order() {
        let mut scene = Scene::default();
        scene.begin_layer(LayerKey(1), rect(0., 0., 120., 120.), true);
        // Every primitive overlaps the previous one, so local orders increase
        // strictly with paint order and no two neighbours share a kind — the
        // run manifest must reproduce eleven alternating stretches.
        let base = rect(0., 0., 100., 100.);
        let mut marker = 100u32;
        scene.insert_primitive(quad_marked(base, bump(&mut marker)));
        scene.insert_primitive(shadow_marked(base, bump(&mut marker)));
        scene.insert_primitive(path_marked(bump(&mut marker), (0., 0.), 1));
        scene.insert_primitive(underline_marked(base, bump(&mut marker)));
        scene.insert_primitive(mono_sprite_marked(base, 0, 1, bump(&mut marker)));
        scene.insert_primitive(poly_sprite_marked(base, 0, 1, bump(&mut marker)));
        scene.insert_primitive(quad_marked(base, bump(&mut marker)));
        scene.insert_primitive(shadow_marked(base, bump(&mut marker)));
        scene.insert_primitive(underline_marked(base, bump(&mut marker)));
        scene.insert_primitive(mono_sprite_marked(base, 0, 2, bump(&mut marker)));
        scene.insert_primitive(poly_sprite_marked(base, 0, 2, bump(&mut marker)));
        let items = scene.end_layer().unwrap();
        scene.finish();

        let oracle = oracle_stream(&scene, 100..111);
        assert_eq!(oracle.len(), 11);
        assert_packing_matches_oracle(&items, &oracle);

        let packed = unpack(pack_layer_items(&items));
        assert_eq!(
            packed.runs.len(),
            11,
            "no two adjacent primitives share a kind"
        );
        for run in &packed.runs {
            let expected_count = match run.kind {
                // The lone path contributes three vertices to its run.
                SlabKind::Paths => 3,
                _ => 1,
            };
            assert_eq!(run.count, expected_count);
        }
    }

    #[test]
    fn equal_order_ties_keep_insertion_order_within_a_kind() {
        // Disjoint bounds reuse one order in the bounds tree, so these two
        // quads tie and the stable sorts must keep insertion order on both
        // sides.
        let mut scene = Scene::default();
        scene.begin_layer(LayerKey(1), rect(0., 0., 700., 700.), true);
        scene.insert_primitive(quad_marked(rect(0., 0., 40., 40.), 21));
        scene.insert_primitive(quad_marked(rect(500., 500., 40., 40.), 22));
        let items = scene.end_layer().unwrap();
        scene.finish();

        let oracle = oracle_stream(&scene, 21..23);
        assert_eq!(
            oracle.iter().map(|slot| slot.entry()).collect::<Vec<_>>(),
            vec![
                Entry {
                    kind: SlabKind::Quads,
                    marker: 21,
                },
                Entry {
                    kind: SlabKind::Quads,
                    marker: 22,
                },
            ],
            "the oracle itself must show insertion-order stability for ties"
        );
        assert_packing_matches_oracle(&items, &oracle);
    }

    #[test]
    fn sprite_tile_id_tiebreak_matches_scene_finish() {
        // Three same-texture sprites tying on order, inserted with descending
        // tile ids: `finish` re-orders them by (order, tile.tile_id), so the
        // packed arrays and manifest must show tiles 1, 2, 3 — not 3, 1, 2.
        let mut scene = Scene::default();
        scene.begin_layer(LayerKey(1), rect(0., 0., 700., 100.), true);
        scene.insert_primitive(mono_sprite_marked(rect(0., 0., 20., 20.), 3, 3, 31));
        scene.insert_primitive(mono_sprite_marked(rect(100., 0., 20., 20.), 3, 1, 32));
        scene.insert_primitive(mono_sprite_marked(rect(200., 0., 20., 20.), 3, 2, 33));
        scene.insert_primitive(poly_sprite_marked(rect(300., 0., 20., 20.), 4, 9, 34));
        scene.insert_primitive(poly_sprite_marked(rect(400., 0., 20., 20.), 4, 7, 35));
        let items = scene.end_layer().unwrap();
        scene.finish();

        let oracle = oracle_stream(&scene, 31..36);
        let entries: Vec<Entry> = oracle.iter().map(|slot| slot.entry()).collect();
        assert_eq!(
            entries,
            vec![
                Entry {
                    kind: SlabKind::MonoSprites,
                    marker: 32,
                },
                Entry {
                    kind: SlabKind::MonoSprites,
                    marker: 33,
                },
                Entry {
                    kind: SlabKind::MonoSprites,
                    marker: 31,
                },
                Entry {
                    kind: SlabKind::PolySprites,
                    marker: 35,
                },
                Entry {
                    kind: SlabKind::PolySprites,
                    marker: 34,
                },
            ],
            "the oracle must be ordered by tile id within the tied order"
        );
        assert_packing_matches_oracle(&items, &oracle);
    }

    #[test]
    fn sprite_runs_split_on_texture_change_like_the_batch_iterator() {
        let mut scene = Scene::default();
        scene.begin_layer(LayerKey(1), rect(0., 0., 700., 100.), true);
        // Disjoint bounds tie the orders; ascending tiles keep the sort at
        // insertion order, isolating texture changes as the only split cause.
        scene.insert_primitive(mono_sprite_marked(rect(0., 0., 20., 20.), 0, 1, 41));
        scene.insert_primitive(mono_sprite_marked(rect(30., 0., 20., 20.), 1, 2, 42));
        scene.insert_primitive(mono_sprite_marked(rect(60., 0., 20., 20.), 0, 3, 43));
        scene.insert_primitive(poly_sprite_marked(rect(90., 0., 20., 20.), 5, 4, 44));
        scene.insert_primitive(poly_sprite_marked(rect(120., 0., 20., 20.), 6, 5, 45));
        let items = scene.end_layer().unwrap();
        scene.finish();

        let oracle = oracle_stream(&scene, 41..46);
        assert_eq!(oracle.len(), 5);
        assert_packing_matches_oracle(&items, &oracle);

        let texture = |index: u32, kind: crate::AtlasTextureKind| {
            Some(AtlasTextureId {
                index,
                kind,
            })
        };
        let packed = unpack(pack_layer_items(&items));
        assert_eq!(
            packed.runs,
            vec![
                KindRun {
                    kind: SlabKind::MonoSprites,
                    start: 0,
                    count: 1,
                    texture_id: texture(0, crate::AtlasTextureKind::Monochrome),
                },
                KindRun {
                    kind: SlabKind::MonoSprites,
                    start: 1,
                    count: 1,
                    texture_id: texture(1, crate::AtlasTextureKind::Monochrome),
                },
                KindRun {
                    kind: SlabKind::MonoSprites,
                    start: 2,
                    count: 1,
                    texture_id: texture(0, crate::AtlasTextureKind::Monochrome),
                },
                KindRun {
                    kind: SlabKind::PolySprites,
                    start: 0,
                    count: 1,
                    texture_id: texture(5, crate::AtlasTextureKind::Polychrome),
                },
                KindRun {
                    kind: SlabKind::PolySprites,
                    start: 1,
                    count: 1,
                    texture_id: texture(6, crate::AtlasTextureKind::Polychrome),
                },
            ]
        );
    }

    #[test]
    fn shadow_and_quad_overlap_orders_across_kinds() {
        // Painted order decides when orders differ: a shadow painted after an
        // overlapping quad draws above it, and vice versa.
        let mut scene = Scene::default();
        let panel = rect(0., 0., 100., 100.);
        scene.begin_layer(LayerKey(1), rect(0., 0., 250., 250.), true);
        scene.insert_primitive(quad_marked(panel, 51));
        scene.insert_primitive(shadow_marked(panel, 52));
        scene.insert_primitive(shadow_marked(panel, 53));
        scene.insert_primitive(quad_marked(panel, 54));
        let items = scene.end_layer().unwrap();
        scene.finish();

        let oracle = oracle_stream(&scene, 51..55);
        let entries: Vec<Entry> = oracle.iter().map(|slot| slot.entry()).collect();
        assert_eq!(
            entries,
            vec![
                Entry {
                    kind: SlabKind::Quads,
                    marker: 51,
                },
                Entry {
                    kind: SlabKind::Shadows,
                    marker: 52,
                },
                Entry {
                    kind: SlabKind::Shadows,
                    marker: 53,
                },
                Entry {
                    kind: SlabKind::Quads,
                    marker: 54,
                },
            ]
        );
        assert_packing_matches_oracle(&items, &oracle);
        let packed = unpack(pack_layer_items(&items));
        // Q S S Q: the two adjacent shadows merge into a single run.
        assert_eq!(
            packed.runs,
            vec![
                KindRun {
                    kind: SlabKind::Quads,
                    start: 0,
                    count: 1,
                    texture_id: None,
                },
                KindRun {
                    kind: SlabKind::Shadows,
                    start: 0,
                    count: 2,
                    texture_id: None,
                },
                KindRun {
                    kind: SlabKind::Quads,
                    start: 1,
                    count: 1,
                    texture_id: None,
                },
            ]
        );

        // At an EQUAL order (disjoint bounds), the legacy batch iterator's
        // kind discriminants decide: shadows (discriminant 1) draw before
        // quads (2) even though the quad was painted first.
        let mut tied = Scene::default();
        tied.begin_layer(LayerKey(2), rect(0., 0., 500., 100.), true);
        tied.insert_primitive(quad_marked(rect(0., 0., 40., 40.), 55));
        tied.insert_primitive(shadow_marked(rect(300., 0., 40., 40.), 56));
        let tied_items = tied.end_layer().unwrap();
        tied.finish();

        let tied_oracle = oracle_stream(&tied, 55..57);
        assert_eq!(
            tied_oracle.iter().map(|slot| slot.entry()).collect::<Vec<_>>(),
            vec![
                Entry {
                    kind: SlabKind::Shadows,
                    marker: 56,
                },
                Entry {
                    kind: SlabKind::Quads,
                    marker: 55,
                },
            ],
            "at an equal order the shadow's lower kind discriminant draws first"
        );
        assert_packing_matches_oracle(&tied_items, &tied_oracle);
    }

    #[test]
    fn path_runs_count_vertices_and_ids_renumber_dense() {
        let mut scene = Scene::default();
        let panel = rect(0., 0., 100., 100.);
        scene.begin_layer(LayerKey(1), rect(0., 0., 150., 150.), true);
        scene.insert_primitive(path_marked(61, (0., 0.), 1)); // 3 vertices
        scene.insert_primitive(quad_marked(panel, 62));
        scene.insert_primitive(path_marked(63, (0., 0.), 2)); // 6 vertices
        let items = scene.end_layer().unwrap();
        scene.finish();

        let oracle = oracle_stream(&scene, 61..64);
        assert_eq!(oracle.len(), 3);
        assert_packing_matches_oracle(&items, &oracle);

        let packed = unpack(pack_layer_items(&items));
        assert_eq!(
            packed.runs,
            vec![
                KindRun {
                    kind: SlabKind::Paths,
                    start: 0,
                    count: 3,
                    texture_id: None,
                },
                KindRun {
                    kind: SlabKind::Quads,
                    start: 0,
                    count: 1,
                    texture_id: None,
                },
                KindRun {
                    kind: SlabKind::Paths,
                    start: 3,
                    count: 6,
                    texture_id: None,
                },
            ],
            "path runs address the flattened vertex stream, other runs instances"
        );
        assert_eq!(packed.total_path_vertices(), 9);
        assert_eq!(packed.paths.len(), 2);
    }

    /// Like `quad_marked`, but with an alpha-one solid background so the #95
    /// instance sweep classifies it as an occluder.
    fn opaque_quad_marked(bounds: Bounds<ScaledPixels>, marker: u32) -> Quad {
        let mut quad = quad_marked(bounds, marker);
        quad.background.solid.a = 1.;
        quad
    }

    /// The #95 instance-tier sweep must leave packed arrays self-consistent:
    /// dropping a covered path shifts every later path's dense id down, and
    /// the vertex-stream runs must still address exactly what survives.
    #[test]
    fn instance_culling_keeps_path_ids_and_runs_consistent() {
        let mut scene = Scene::default();
        let cover = rect(0., 0., 100., 60.);
        scene.begin_layer(LayerKey(1), rect(0., 0., 150., 150.), true);
        scene.insert_primitive(path_marked(61, (0., 0.), 1)); // beneath the quad
        scene.insert_primitive(opaque_quad_marked(cover, 62));
        scene.insert_primitive(path_marked(63, (10., 10.), 2)); // above the quad
        let items = scene.end_layer().unwrap();
        scene.finish();

        let kept = crate::occlusion::cull_covered_instances(items);
        assert_eq!(kept.len(), 2, "the covered path must not survive the sweep");

        let packed = unpack(pack_layer_items(&kept));
        assert_eq!(packed.paths.len(), 1);
        assert_eq!(
            packed.paths[0].id,
            PathId(0),
            "the surviving path renumbers densely despite its recorded id"
        );
        assert_eq!(packed.paths[0].color.solid.h as u32, 63);
        assert_eq!(
            packed.total_path_vertices(),
            6,
            "only the surviving path's vertices remain"
        );
        assert_eq!(
            packed.runs,
            vec![
                KindRun {
                    kind: SlabKind::Quads,
                    start: 0,
                    count: 1,
                    texture_id: None,
                },
                KindRun {
                    kind: SlabKind::Paths,
                    start: 0,
                    count: 6,
                    texture_id: None,
                },
            ],
            "draw order (quad below, path above) survives the culling"
        );
    }

    /// Packing the post-cull stream must equal the finished scene built from
    /// exactly the kept primitives: the legacy oracle applied to the filtered
    /// stream, entry for entry — the golden-model guarantee extended over the
    /// cull (#95).
    #[test]
    fn packing_a_culled_stream_matches_the_oracle_for_its_kept_items() {
        let mut scene = Scene::default();
        let panel = rect(5., 5., 40., 40.);
        scene.begin_layer(LayerKey(1), rect(0., 0., 200., 200.), true);
        // Everything painted before the big opaque quad and fully inside it
        // gets culled; everything after stays.
        scene.insert_primitive(quad_marked(panel, 70));
        scene.insert_primitive(underline_marked(panel, 71));
        scene.insert_primitive(mono_sprite_marked(panel, 0, 1, 72));
        scene.insert_primitive(poly_sprite_marked(panel, 0, 2, 73));
        scene.insert_primitive(shadow_marked(panel, 74));
        scene.insert_primitive(opaque_quad_marked(rect(0., 0., 100., 100.), 75));
        scene.insert_primitive(underline_marked(rect(120., 120., 30., 10.), 76));
        let items = scene.end_layer().unwrap();
        scene.finish();

        let kept = crate::occlusion::cull_covered_instances(items);
        let kept_markers: Vec<u32> = kept
            .iter()
            .filter_map(|item| match item {
                LayerItem::Primitive(Primitive::Quad(quad)) => {
                    Some(vec![quad.corner_radii.top_left.0 as u32])
                }
                LayerItem::Primitive(Primitive::Underline(u)) => Some(vec![u.thickness.0 as u32]),
                LayerItem::Primitive(Primitive::Shadow(s)) => {
                    Some(vec![s.corner_radii.top_left.0 as u32])
                }
                LayerItem::Primitive(Primitive::MonochromeSprite(s)) => {
                    Some(vec![s.text_color.solid.h as u32])
                }
                LayerItem::Primitive(Primitive::PolychromeSprite(s)) => {
                    Some(vec![s.opacity as u32])
                }
                _ => None,
            })
            .flatten()
            .collect();
        assert_eq!(
            kept_markers,
            vec![74, 75, 76],
            "shadow survives (never culled), covered quad/underline/sprites go"
        );

        // Replay exactly the kept primitives into a fresh scene and use its
        // finished batch stream as the oracle.
        let mut reference = Scene::default();
        reference.begin_layer(LayerKey(9), rect(0., 0., 200., 200.), true);
        for item in &kept {
            if let LayerItem::Primitive(primitive) = item {
                reference.insert_primitive(primitive.clone());
            }
        }
        reference.end_layer().unwrap();
        reference.finish();

        let oracle = oracle_stream(&reference, 70..80);
        assert_eq!(oracle.len(), 3, "oracle sees exactly the kept primitives");
        assert_packing_matches_oracle(&kept, &oracle);
    }

    #[test]
    fn underline_runs_merge_adjacent_and_split_across_kinds() {
        let mut scene = Scene::default();
        let panel = rect(0., 0., 100., 30.);
        scene.begin_layer(LayerKey(1), rect(0., 0., 150., 150.), true);
        scene.insert_primitive(underline_marked(panel, 71));
        scene.insert_primitive(underline_marked(panel, 72));
        scene.insert_primitive(quad_marked(panel, 73));
        scene.insert_primitive(underline_marked(panel, 74));
        let items = scene.end_layer().unwrap();
        scene.finish();

        let oracle = oracle_stream(&scene, 71..75);
        assert_eq!(oracle.len(), 4);
        assert_packing_matches_oracle(&items, &oracle);

        let packed = unpack(pack_layer_items(&items));
        assert_eq!(
            packed.runs,
            vec![
                KindRun {
                    kind: SlabKind::Underlines,
                    start: 0,
                    count: 2,
                    texture_id: None,
                },
                KindRun {
                    kind: SlabKind::Quads,
                    start: 0,
                    count: 1,
                    texture_id: None,
                },
                KindRun {
                    kind: SlabKind::Underlines,
                    start: 2,
                    count: 1,
                    texture_id: None,
                },
            ]
        );
    }

    #[test]
    fn empty_layer_packs_to_an_empty_manifest() {
        let mut scene = Scene::default();
        scene.begin_layer(LayerKey(1), rect(0., 0., 50., 50.), true);
        let items = scene.end_layer().unwrap();
        scene.finish();
        assert!(items.is_empty());

        let packed = unpack(pack_layer_items(&items));
        assert!(packed.runs.is_empty());
        assert!(packed.quads.is_empty() && packed.shadows.is_empty());
        assert!(packed.paths.is_empty() && packed.underlines.is_empty());
        assert!(packed.mono_sprites.is_empty() && packed.poly_sprites.is_empty());
        assert_eq!(packed.total_path_vertices(), 0);
    }

    #[test]
    fn stale_global_path_ids_are_discarded_and_renumbered_dense() {
        // Pad the recording frame's global paths array with seven unrelated
        // paths, so the layer's captured paths carry ids 7 and 8 — valid only
        // in that frame, meaningless everywhere else.
        let mut scene = Scene::default();
        for index in 0..7u32 {
            scene.insert_primitive(path_marked(500 + index, (0., 0.), 1));
        }
        scene.begin_layer(LayerKey(1), rect(0., 0., 100., 100.), true);
        scene.insert_primitive(path_marked(91, (0., 0.), 1));
        scene.insert_primitive(path_marked(92, (0., 0.), 2));
        let items = scene.end_layer().unwrap();

        let recorded_ids: Vec<PathId> = items
            .iter()
            .filter_map(|item| match item {
                LayerItem::Primitive(Primitive::Path(path)) => Some(path.id),
                _ => None,
            })
            .collect();
        assert_eq!(
            recorded_ids,
            vec![PathId(7), PathId(8)],
            "precondition: the capture holds the recording frame's global ids"
        );

        // The recording scene dies; the ids now dangle. Packing must succeed
        // purely from the items' inline geometry and renumber densely.
        drop(scene);
        let packed = unpack(pack_layer_items(&items));
        assert_eq!(packed.paths.len(), 2);
        for (index, path) in packed.paths.iter().enumerate() {
            assert_eq!(path.id, PathId(index));
        }
        let packed_markers: Vec<u32> =
            packed.paths.iter().map(|path| path.color.solid.h as u32).collect();
        assert_eq!(packed_markers, vec![91, 92]);
        assert_eq!(
            packed.total_path_vertices(),
            3 + 6,
            "inline geometry survives verbatim"
        );
    }

    #[test]
    fn unsupported_kinds_are_detected_and_reported() {
        // Filter-group boundaries, backdrop filters and surfaces have no slab
        // kind: each must reject the whole layer with the offending kind
        // named, rather than pack the rest and drop pixels.
        let boundary_items = {
            let mut scene = Scene::default();
            scene.begin_layer(LayerKey(1), rect(0., 0., 100., 100.), true);
            scene.insert_primitive(quad_marked(rect(0., 0., 50., 50.), 81));
            scene.insert_primitive(boundary(true));
            scene.insert_primitive(boundary(false));
            scene.end_layer().unwrap()
        };
        assert_eq!(
            validate_packable(&boundary_items),
            Err(FallbackReason::UnsupportedPrimitive(
                crate::scene::PrimitiveKind::FilterBoundaryStart
            )),
        );

        let backdrop_items = {
            let mut scene = Scene::default();
            scene.begin_layer(LayerKey(2), rect(0., 0., 100., 100.), true);
            scene.insert_primitive(backdrop());
            scene.end_layer().unwrap()
        };
        assert_eq!(
            validate_packable(&backdrop_items),
            Err(FallbackReason::UnsupportedPrimitive(
                crate::scene::PrimitiveKind::BackdropFilter
            )),
        );

        let surface_items = {
            let mut scene = Scene::default();
            scene.begin_layer(LayerKey(3), rect(0., 0., 100., 100.), true);
            scene.insert_primitive(surface());
            scene.end_layer().unwrap()
        };
        assert_eq!(
            validate_packable(&surface_items),
            Err(FallbackReason::UnsupportedPrimitive(
                crate::scene::PrimitiveKind::Surface
            )),
        );
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "slab packing rejected a layer")]
    fn packing_unsupported_kinds_asserts_in_debug_builds() {
        let items = {
            let mut scene = Scene::default();
            scene.begin_layer(LayerKey(1), rect(0., 0., 100., 100.), true);
            scene.insert_primitive(backdrop());
            scene.end_layer().unwrap()
        };
        // In debug builds the production entry point fails loudly instead of
        // quietly returning a fallback the caller might ignore.
        pack_layer_items(&items);
    }

    #[test]
    fn inconsistent_path_ids_reject_packing() {
        // Two live paths claiming one recorded id cannot happen through the
        // scene APIs (`insert_primitive` always assigns fresh ids); it means
        // a producer skipped the reassignment contract. Detection is the
        // fail-loud stand-in for "missing path data": geometry travels
        // inline, so a dangling reference can only manifest as ids that no
        // longer describe consistent state.
        let mut first = path_marked(95, (0., 0.), 1);
        first.id = PathId(4);
        let mut second = path_marked(96, (40., 40.), 1);
        second.id = PathId(4);
        let items = vec![
            LayerItem::Primitive(Primitive::Path(first)),
            LayerItem::Primitive(Primitive::Path(second)),
        ];

        assert_eq!(
            validate_packable(&items),
            Err(FallbackReason::InconsistentPathIds),
        );
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "slab packing rejected a layer")]
    fn packing_inconsistent_path_ids_asserts_in_debug_builds() {
        let mut first = path_marked(95, (0., 0.), 1);
        first.id = PathId(4);
        let mut second = path_marked(96, (40., 40.), 1);
        second.id = PathId(4);
        let items = vec![
            LayerItem::Primitive(Primitive::Path(first)),
            LayerItem::Primitive(Primitive::Path(second)),
        ];
        pack_layer_items(&items);
    }

    #[test]
    fn nested_layers_are_never_packed_into_their_parent() {
        let outer_bounds = rect(0., 0., 300., 300.);
        let inner_bounds = rect(10., 10., 100., 100.);
        let mut scene = Scene::default();
        scene.begin_layer(LayerKey(1), outer_bounds, true);
        scene.insert_primitive(quad_marked(outer_bounds, 101));
        scene.begin_layer(LayerKey(2), inner_bounds, true);
        scene.insert_primitive(quad_marked(inner_bounds, 102));
        scene.insert_primitive(quad_marked(inner_bounds, 103));
        let inner_items = scene.end_layer().unwrap();
        let outer_items = scene.end_layer().unwrap();
        scene.finish();

        assert!(matches!(outer_items[1], LayerItem::Nested(LayerKey(2))));

        // The outer layer packs only its own quad; the nested reference
        // contributes nothing.
        let outer_oracle = oracle_stream(&scene, 101..102);
        assert_eq!(outer_oracle.len(), 1);
        assert_packing_matches_oracle(&outer_items, &outer_oracle);
        let outer_packed = unpack(pack_layer_items(&outer_items));
        assert_eq!(outer_packed.quads.len(), 1);

        // The inner layer packs independently under its own key.
        let inner_oracle = oracle_stream(&scene, 102..104);
        assert_eq!(inner_oracle.len(), 2);
        assert_packing_matches_oracle(&inner_items, &inner_oracle);
    }

    /// xorshift64*: deterministic, dependency-free, enough to drive a
    /// reproducible fuzz-style model check (same generator as slab.rs's).
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Rng(seed | 1)
        }

        fn next_u64(&mut self) -> u64 {
            let mut state = self.0;
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            self.0 = state;
            state
        }

        fn below(&mut self, bound: u32) -> u32 {
            (self.next_u64() % u64::from(bound)) as u32
        }
    }

    #[test]
    fn fuzz_five_layers_of_two_hundred_primitives_match_the_oracle() {
        // Seeded fuzz: ~220 primitives spread over five recording layers plus
        // root-scope noise, every kind, random overlaps and sprite
        // textures/tiles, compared layer-by-layer against the finished
        // scene's batch stream.
        const SEED: u64 = 0x00C0_FEE_5AB_5;
        let mut rng = Rng::new(SEED);

        let random_rect = |rng: &mut Rng| {
            let far = rng.below(4) == 0;
            let x = if far { 800 + rng.below(80) } else { rng.below(400) } as f32;
            let y = rng.below(300) as f32;
            rect(x, y, 20. + rng.below(120) as f32, 20. + rng.below(120) as f32)
        };

        let mut scene = Scene::default();
        let mut inserted_total = 0usize;
        let mut kinds_seen = [false; SlabKind::COUNT];

        // Each marker source owns a disjoint block of the marker space, so a
        // layer's oracle range identifies exactly its own primitives: layer i
        // draws from [i*1000, (i+1)*1000), root noise from [5000, 6000).
        let mut paint_random =
            |scene: &mut Scene, rng: &mut Rng, next_marker: &mut u32, block_base: u32| {
                let offset = *next_marker;
                *next_marker += 1;
                let marker = block_base + offset;
                assert!(offset < 1000, "marker block exhausted");
                let bounds = random_rect(rng);
                let kind_choice = rng.below(6);
                match kind_choice {
                    0 => scene.insert_primitive(quad_marked(bounds, marker)),
                    1 => scene.insert_primitive(shadow_marked(bounds, marker)),
                    2 => scene.insert_primitive(underline_marked(bounds, marker)),
                    3 => scene.insert_primitive(mono_sprite_marked(
                        bounds,
                        rng.below(3),
                        rng.below(8),
                        marker,
                    )),
                    4 => scene.insert_primitive(poly_sprite_marked(
                        bounds,
                        rng.below(3),
                        rng.below(8),
                        marker,
                    )),
                    _ => scene.insert_primitive(path_marked(
                        marker,
                        (rng.below(200) as f32, rng.below(200) as f32),
                        1 + rng.below(3) as usize,
                    )),
                }
                kinds_seen[kind_choice as usize] = true;
                inserted_total += 1;
            };

        let mut captures: Vec<(Vec<LayerItem>, Range<u32>)> = Vec::new();
        for layer_index in 0..5u32 {
            let block_base = layer_index * 1000;
            let mut layer_marker = 0u32;
            scene.begin_layer(
                LayerKey(1000 + u64::from(layer_index)),
                rect(0., 0., 600., 500.),
                true,
            );
            for _ in 0..40 {
                paint_random(&mut scene, &mut rng, &mut layer_marker, block_base);
            }
            let items = scene.end_layer().unwrap();
            captures.push((items, block_base..block_base + 1000));
            // Root-scope noise between layers, from the reserved root range.
            for _ in 0..4 {
                paint_random(&mut scene, &mut rng, &mut layer_marker, 5000);
            }
        }

        assert!(
            kinds_seen.iter().all(|seen| *seen),
            "the fuzz must exercise all six slab kinds"
        );

        scene.finish();

        // Sanity: every inserted primitive reaches the batch stream exactly
        // once (nothing was silently dropped on either side).
        let mut accounted = oracle_stream(&scene, 5000..6000).len();
        for (_, range) in &captures {
            accounted += oracle_stream(&scene, range.clone()).len();
        }
        assert_eq!(accounted, inserted_total, "marker accounting");

        for (items, range) in &captures {
            let oracle = oracle_stream(&scene, range.clone());
            assert_eq!(oracle.len(), 40, "each layer contributed 40 primitives");
            assert_packing_matches_oracle(items, &oracle);
        }
    }
}
