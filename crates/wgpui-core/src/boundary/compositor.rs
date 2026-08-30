//! The per-frame compositing decision a `.boundary()` makes, and the retained
//! state it makes it from. See docs/gpu-native-architecture.md §4.1 and §5.4.
//!
//! Not in §3.1's literal file map — a deliberate addition, recorded in
//! `docs/phase-2-results.md`. §3.1 gives `boundary/` a policy file (what an
//! author may tune) and an identity file (how a boundary finds itself), and no
//! home for the thing that consumes both plus
//! [`crate::invalidation::reason::Reason`] once per frame. In R-N/SFD that
//! decision had no separate home either, because it was interleaved into
//! `Interactivity`'s paint block inside `div.rs`; §3.4 lists breaking that
//! block apart as one of the four seams the widgets crate splits along, and
//! this is the half of it that is not any element type's business.
//!
//! # The decision, stated once
//!
//! A boundary composites transform-only when **both** of these hold:
//!
//! 1. Its content is clean — nothing inside it needed re-emitting this frame.
//! 2. The signal that woke the frame permits it — [`Reason::Scroll`], not
//!    [`Reason::DataChanged`].
//!
//! Under §4.0's ambient reconciliation, condition 1 is measured, not assumed:
//! the reconciler re-diffs every element inside the boundary every frame
//! whether or not the boundary exists, so "the content is clean" is a fact the
//! frame already established rather than something the boundary's key had to
//! promise. That is a genuine change from SFD §1.1, where the tagged
//! notification was the *only* evidence available and a wrong key meant
//! silently stale UI. Requiring condition 2 as well is therefore deliberately
//! conservative: it costs a `DataChanged`-signalled pure-scroll frame one
//! ordinary recomposite, and it buys that a bug in any element's `diff_key` can
//! only ever produce a slow frame, never a frame that slid stale content into
//! view. §4.1 asks for the signal "from day one — not retrofitted," and this is
//! what consuming it looks like once the diff is ambient underneath it.
//!
//! # What is decided here, and what is emphatically not
//!
//! Nothing in this file allocates, pools, or draws a texture.
//! [`Retention::Texture`] is a *decision* about a boundary, recorded and
//! observable; §3.1 puts every live `wgpu::Device` in `wgpui-wgpu`, and
//! Phase 4's `render/textures/layer_texture.rs` is what consumes the decision
//! into an actual texture.
//!
//! # What Phase 4 added: the composite entry, and the layer tier
//!
//! §5.5's Gap 2 asks that an externally-produced texture (`WgpuSurface`) and a
//! boundary's own baked texture reach the framebuffer through *one* mechanism
//! rather than two parallel ones. The half of that which belongs in a
//! device-free crate is the description: [`CompositeEntry`] says where a
//! texture lands, what clips it, how it is drawn, and — the part that makes
//! §5.5's promised win real — whether anything painted above it covers it
//! completely.
//!
//! That last question is R-N §8.1's **layer tier**, and it is deliberately CPU
//! work. `crate::occlusion`'s own module doc says why: the layer tier "runs
//! over layers, not primitives — tens of items, not tens of thousands — so it
//! is not a compute problem", and it names
//! [`crate::occlusion::coverage::fully_covered`] as "the routine it will reuse
//! when the compositor grows a per-layer opaque region to feed it."
//! [`visible_composites`] is that compositor growing it, and it calls exactly
//! that routine rather than a second copy of the rule.

use crate::boundary::policy::{BoundaryPolicy, Retention};
use crate::geometry::Rect;
use crate::invalidation::axes::Invalidation;
use crate::invalidation::reason::Reason;
use crate::occlusion::coverage::{MAX_OCCLUDERS, OccluderStyle, fully_covered, opaque_region};
use crate::scene::layer::{BoundaryId, LayerId, LayerKey, LayerTransform};
use crate::scene::tile::{EvictedTile, TileCoord, TileGrid, TileResidency};
use std::collections::HashMap;

/// What a boundary does with its content this frame.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Composite {
    /// Neither the content nor the transform moved. Zero work, zero upload.
    Clean,
    /// The content is untouched and only the composite transform changed —
    /// R-N §3.2's "a 1px scroll sets one flag on one layer. No render, no
    /// reconcile, no layout, no prepaint, no paint, no upload — one changed
    /// matrix." This is the fast path §8's Phase 2 gate names.
    TransformOnly,
    /// Something inside the boundary needed re-emitting, so its content is
    /// patched into residency as usual.
    Redisplay,
}

impl Composite {
    /// Whether this frame left the boundary's resident primitives untouched.
    pub const fn leaves_content_resident(self) -> bool {
        matches!(self, Composite::Clean | Composite::TransformOnly)
    }

    /// The invalidation axes this decision raises on the boundary's layer.
    pub const fn invalidation(self) -> Invalidation {
        match self {
            Composite::Clean => Invalidation::empty(),
            Composite::TransformOnly => Invalidation::TRANSFORM,
            Composite::Redisplay => Invalidation::DISPLAY,
        }
    }
}

/// One boundary's retained compositing state.
///
/// No longer `Copy` as of Phase 4.5: a tiled boundary carries its own
/// [`TileResidency`], which owns a map. Every accessor still hands out a copy of
/// the field it names, so nothing that read this type had to change.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundaryState {
    policy: BoundaryPolicy,
    layer: LayerId,
    transform: LayerTransform,
    retention: Retention,
    primitive_count: usize,
    last_visited_frame: u64,
    tiles: Option<TileResidency>,
}

impl BoundaryState {
    /// The policy this boundary was declared with.
    pub const fn policy(&self) -> BoundaryPolicy {
        self.policy
    }

    /// The layer this boundary's content lives in.
    pub const fn layer(&self) -> LayerId {
        self.layer
    }

    /// Where this boundary's content currently composites.
    pub const fn transform(&self) -> LayerTransform {
        self.transform
    }

    /// Whether this boundary is texture-retained or primitive-retained, as of
    /// the last frame its primitive count was resolved.
    pub const fn retention(&self) -> Retention {
        self.retention
    }

    /// How many primitives the boundary held at that point.
    pub const fn primitive_count(&self) -> usize {
        self.primitive_count
    }

    /// The last frame this boundary appeared in the tree.
    pub const fn last_visited_frame(&self) -> u64 {
        self.last_visited_frame
    }

    /// Which tiles of this boundary's content plane are resident, or `None` for
    /// a boundary that is not tiled.
    pub const fn tiles(&self) -> Option<&TileResidency> {
        self.tiles.as_ref()
    }
}

/// What one boundary did this frame, as inspectable data.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BoundaryComposite {
    /// The boundary.
    pub boundary: BoundaryId,
    /// Its layer.
    pub layer: LayerId,
    /// What it did.
    pub composite: Composite,
    /// Whether it is texture-retained or primitive-retained.
    pub retention: Retention,
    /// Where its content composites after this frame.
    pub transform: LayerTransform,
    /// The axes this decision raised on its layer.
    pub invalidation: Invalidation,
}

/// An externally-produced surface's identity, as the compositor sees it.
///
/// Opaque on purpose. The real handle is `WgpuSurfaceHandle`, which owns a
/// triple-buffered texture and a cross-thread producer protocol that §9's risk
/// table forbids this work from touching; all the compositor needs is something
/// equal to itself and unequal to a different surface. `wgpui-widgets`'
/// `SurfaceId` and `wgpui-wgpu`'s registry id both map onto this.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExternalSurfaceId(u64);

impl ExternalSurfaceId {
    /// Wrap a raw handle.
    pub const fn from_raw(raw: u64) -> Self {
        ExternalSurfaceId(raw)
    }

    /// The raw handle.
    pub const fn as_raw(self) -> u64 {
        self.0
    }
}

/// Which of the two producers a composite entry's pixels come from.
///
/// §5.5's Gap 2 in one type: "a `WgpuSurface` becomes the degenerate case of a
/// compositing boundary — here is a texture, produced externally instead of
/// baked by the rasterizer, composite it exactly like a boundary's baked
/// texture." Everything below this enum treats the two identically; the only
/// place the difference survives is where the texture is *fetched from*, which
/// is `wgpui-wgpu`'s business and not the compositor's.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum CompositeSource {
    /// A boundary that reached [`Retention::Texture`] and baked its own.
    BoundaryTexture(BoundaryId),
    /// A texture someone else's render loop produced.
    External(ExternalSurfaceId),
}

/// One already-rendered texture's place in the ordered scene.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CompositeEntry {
    /// Where the pixels come from.
    pub source: CompositeSource,
    /// Where the texture lands, in the window's coordinate space.
    pub bounds: Rect,
    /// What clips it.
    pub content_mask: Rect,
    /// Straight alpha the entry composites at.
    pub opacity: f32,
    /// Uniform corner radius it is clipped to.
    pub corner_radius: f32,
    /// Whether the source fills its own bounds with no transparency.
    ///
    /// Never inferred. A boundary's baked texture is transparent wherever its
    /// content is, and an external producer's contents are not the framework's
    /// to know at all (§5.5: "its pixel *content* is never part of the CPU
    /// description"), so an entry occludes only when its producer says it
    /// does. The conservative default ([`CompositeEntry::sampled`]) is `false`.
    pub source_is_opaque: bool,
    /// The producer's content generation, so a texture pool can tell a stale
    /// bake from a current one without comparing pixels.
    pub content_token: u64,
}

impl CompositeEntry {
    /// A translucent, square-cornered entry that never occludes — the safe
    /// shape for any source whose contents are unknown.
    pub fn sampled(source: CompositeSource, bounds: Rect, content_mask: Rect) -> CompositeEntry {
        CompositeEntry {
            source,
            bounds,
            content_mask,
            opacity: 1.0,
            corner_radius: 0.0,
            source_is_opaque: false,
            content_token: 0,
        }
    }

    /// What this entry can actually paint.
    pub fn visible(&self) -> Rect {
        self.bounds.intersect(&self.content_mask)
    }

    /// Its conservative opaque region, or `None` when it does not qualify.
    ///
    /// Goes through [`opaque_region`] rather than restating the rule: an entry
    /// is a rectangle with a fill alpha, a corner radius, and no border, which
    /// is a strict special case of the primitive test, and keeping one
    /// implementation is what stops the two tiers from disagreeing about what
    /// "opaque" means.
    pub fn opaque_region(&self) -> Option<Rect> {
        if !self.source_is_opaque {
            return None;
        }
        opaque_region(
            self.bounds,
            self.content_mask,
            &OccluderStyle {
                background_is_solid: true,
                background_alpha: 1.0,
                element_opacity: self.opacity,
                max_corner_radius: self.corner_radius,
                border_is_opaque: true,
                max_border_width: 0.0,
                has_backdrop_filter: false,
            },
        )
    }
}

/// R-N §8.1's **layer tier**: which composite entries still have to be drawn.
///
/// `entries` is in draw order, so `entries[j]` for `j > i` paints above
/// `entries[i]`. Returns one flag per entry, `true` to draw.
///
/// This is what makes §5.5's promise concrete — "a 3D viewport fully covered by
/// a modal stops being drawn at all, which it cannot today". An entry this
/// returns `false` for is dropped from the draw plan entirely: no bind group,
/// no texture fetch, no draw call, and for an external surface no interaction
/// with `SurfaceRegistry` at all.
///
/// Conservative in the same two ways the instance tier is: an entry with an
/// empty visible rectangle is kept (it paints nothing, but saying so is the
/// caller's business), and only the first [`MAX_OCCLUDERS`] qualifying
/// occluders are considered, which can only ever *miss* a cull.
pub fn visible_composites(entries: &[CompositeEntry]) -> Vec<bool> {
    (0..entries.len())
        .map(|index| {
            let Some(entry) = entries.get(index) else {
                return true;
            };
            let target = entry.visible();
            if target.is_empty() {
                return true;
            }
            let mut occluders = [Rect::EMPTY; MAX_OCCLUDERS];
            let mut count = 0usize;
            for above in entries.iter().skip(index + 1) {
                if count >= MAX_OCCLUDERS {
                    break;
                }
                let Some(region) = above.opaque_region() else {
                    continue;
                };
                if !region.intersects(&target) {
                    continue;
                }
                occluders[count] = region;
                count += 1;
            }
            !fully_covered(target, &occluders[..count])
        })
        .collect()
}

impl BoundaryComposite {
    /// The composite entry this boundary contributes, or `None` when it has
    /// none.
    ///
    /// **The one signal `wgpui-core` owes `wgpui-wgpu` about texture
    /// retention**, and deliberately the smallest one. `Retention::Primitives`
    /// returns `None`: such a boundary has no texture and its slab is drawn
    /// directly, which is R-N §3.3's "a layer holding twelve quads is cheaper
    /// to re-emit than to composite through a texture."
    /// [`Retention::Texture`] returns an entry naming this boundary as its
    /// source, and `wgpui-wgpu`'s texture pool is what turns that name into a
    /// `wgpu::Texture`.
    ///
    /// `content_token` is the generation the caller wants baked — in practice
    /// `crate::scene::Layer::generation`, which already changes on every
    /// reservation, resize, release, and content edit. Taking it as an argument
    /// rather than reading it here is what keeps this file free of the scene:
    /// the compositor decides *whether* a boundary composites through a
    /// texture, and the scene knows *what version* of its content that texture
    /// would hold.
    ///
    /// `source_is_opaque` is false and stays false. A baked boundary texture is
    /// transparent wherever its content is, and nothing in Phase 2 or Phase 4
    /// computes a boundary's own opaque region; a caller that has computed one
    /// may set the field, and until then the layer tier never culls under a
    /// boundary texture.
    pub fn composite_entry(
        &self,
        bounds: Rect,
        content_mask: Rect,
        content_token: u64,
    ) -> Option<CompositeEntry> {
        if self.retention != Retention::Texture {
            return None;
        }
        Some(CompositeEntry::sampled(
            CompositeSource::BoundaryTexture(self.boundary),
            bounds,
            content_mask,
        ))
        .map(|entry| CompositeEntry {
            content_token,
            ..entry
        })
    }
}

/// What one tiled boundary's frame resolved to (§4.3).
///
/// These are visibility, residency, and damage candidates for the one retained
/// boundary layer. They are not a render cache and do not create scene layers,
/// slabs, or invalidations. The native emitter uses this metadata to decide
/// which screen regions need presentation updates.
#[derive(Clone, Debug, PartialEq)]
pub struct TiledVisit {
    /// The boundary these tiles belong to.
    pub boundary: BoundaryId,
    /// Its grid.
    pub grid: TileGrid,
    /// The content-plane rectangle under the viewport at this pan offset.
    pub content_viewport: Rect,
    /// Every tile in range this frame, row-major and ascending.
    pub visible: Vec<TileCoord>,
    /// The tiles that were not resident before this frame — the `DISPLAY`
    /// targets, and the number §8's Phase 4.5 gate measures on a crossing.
    pub revealed: Vec<TileCoord>,
    /// The tiles that left residency, and under which rule.
    pub evicted: Vec<EvictedTile>,
    /// Resident tiles the budget could not account for, all of them in range.
    pub over_budget: usize,
    /// How many tiles are resident after this frame's sweep.
    pub resident: usize,
}

impl TiledVisit {
    /// The compatibility layer key for one tile of this boundary.
    ///
    /// Production emission does not create or retain this layer. The method
    /// remains available to callers that use the tile-key model in test
    /// support or custom renderers.
    pub fn tile_layer(&self, tile: TileCoord) -> LayerId {
        LayerId::from_key(LayerKey::tiled(self.boundary, tile))
    }

    /// The unbuffered overlay layer holding content that spans tiles — see
    /// [`crate::scene::TilePlacement`] for why that content is not clipped into
    /// each tile instead.
    ///
    /// It is the boundary's plain untiled layer, which is the point: the overlay
    /// is not a new kind of layer, it is the layer an untiled boundary would
    /// have had anyway.
    pub fn overlay_layer(&self) -> LayerId {
        LayerId::from_key(LayerKey::untiled(self.boundary))
    }

    /// Compatibility layer keys for visible tiles, in draw order.
    pub fn visible_layers(&self) -> Vec<LayerId> {
        self.visible
            .iter()
            .map(|tile| self.tile_layer(*tile))
            .collect()
    }

    /// The compatibility layer key for a primitive's tile placement.
    pub fn placement_layer(&self, bounds: Rect) -> LayerId {
        match self.grid.placement(bounds) {
            crate::scene::tile::TilePlacement::Tile(tile) => self.tile_layer(tile),
            crate::scene::tile::TilePlacement::Overlay => self.overlay_layer(),
        }
    }
}

/// Every live compositing boundary, across frames.
#[derive(Debug, Default)]
pub struct Compositor {
    boundaries: HashMap<BoundaryId, BoundaryState>,
}

impl Compositor {
    /// A compositor holding no boundaries.
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare that `boundary` exists this frame with `policy`, returning the
    /// layer its content lives in.
    ///
    /// Idempotent across frames: a boundary re-declared under the same
    /// identity keeps its transform and its residency, which is the entire
    /// point of deriving that identity positionally (SFD §1.0) rather than
    /// requiring a name.
    pub fn visit(&mut self, boundary: BoundaryId, policy: BoundaryPolicy, frame: u64) -> LayerId {
        let layer = LayerId::from_key(LayerKey::untiled(boundary));
        let state = self
            .boundaries
            .entry(boundary)
            .or_insert_with(|| BoundaryState {
                policy,
                layer,
                transform: LayerTransform::IDENTITY,
                retention: Retention::Primitives,
                primitive_count: 0,
                last_visited_frame: frame,
                tiles: None,
            });
        state.policy = policy;
        state.last_visited_frame = frame;
        if policy.buffering.tile_grid().is_none() {
            state.tiles = None;
        }
        state.layer
    }

    /// Declare a [`crate::boundary::policy::Buffering::Tiled`] boundary and
    /// resolve its live tile set for this frame.
    ///
    /// `viewport` is the boundary's visible rectangle in its own parent space;
    /// the content-plane rectangle under it is derived from the transform the
    /// boundary already has, so a caller pans by [`Compositor::set_transform`]
    /// and this reports what that pan revealed.
    ///
    /// Returns `None` when the policy is not tiled, or when its tile size cannot
    /// address the viewport at all — see [`crate::scene::TileSpan::MAX_TILES`].
    /// Both mean the same thing to a caller: buffer this boundary the untiled
    /// way, which [`Compositor::visit`] already did for it.
    ///
    /// # Visibility metadata, and nothing else new
    ///
    /// §4.3's whole claim is that tiling "needs almost no new machinery," and
    /// this method is where that is either true or not. It calls
    /// [`crate::scene::TileGrid::visible_span`], hands the result to
    /// [`TileResidency`], and reports which content-plane tiles are visible or
    /// newly revealed. It creates no layer, allocates no slab, and raises no
    /// invalidation. The caller uses the result as presentation damage
    /// metadata while the ordinary retained boundary layer remains the only
    /// scene ownership.
    ///
    /// - A **revealed** tile is newly exposed presentation damage.
    /// - A **still-visible** tile remains resident metadata; scrolling it does
    ///   not require shaping, layout, primitive upload, or atlas upload.
    /// - An **evicted** tile is no longer part of the resident-range metadata.
    ///
    /// # Two ordering and lifetime facts worth knowing before calling this
    ///
    /// **Position the boundary after declaring it.**
    /// [`Compositor::set_transform`] reports `false` rather than creating a
    /// boundary, so moving one the compositor has never seen is inert and the
    /// span below would resolve at the identity. On a boundary's first frame,
    /// call [`Compositor::visit`] (or this method) before setting the transform
    /// that positions it. Every later frame is unaffected.
    ///
    /// **A boundary switched away from `Tiled` drops its tile metadata on the
    /// next visit.** The ordinary untiled boundary layer remains live, and
    /// [`Compositor::sweep`] releases that layer when the boundary itself is
    /// evicted.
    pub fn visit_tiled(
        &mut self,
        boundary: BoundaryId,
        policy: BoundaryPolicy,
        frame: u64,
        viewport: Rect,
    ) -> Option<TiledVisit> {
        self.visit(boundary, policy, frame);
        let grid = policy.buffering.tile_grid()?;
        let state = self.boundaries.get_mut(&boundary)?;

        let content_viewport = TileGrid::content_viewport(viewport, state.transform);
        let span = grid.visible_span(content_viewport, policy.buffering.retain_radius())?;

        let residency = state
            .tiles
            .get_or_insert_with(|| TileResidency::new(policy.resident_tile_budget));
        // Re-applied every frame for the same reason `state.policy = policy`
        // above is: a re-declared boundary's policy is the one it was declared
        // with this frame, not the one it happened to be created with.
        residency.set_budget(policy.resident_tile_budget);
        let revealed = residency.mark(span, frame);
        let evicted = residency.sweep(frame, policy.evict_after_frames);

        Some(TiledVisit {
            boundary,
            grid,
            content_viewport,
            visible: span.tiles(),
            revealed,
            evicted,
            over_budget: residency.over_budget(),
            resident: residency.len(),
        })
    }

    /// Move a boundary's content to `transform`, reporting whether that is a
    /// change from where it already was.
    ///
    /// A boundary that is not live reports `false` rather than being created:
    /// a transform without a declaration has no content to apply to.
    pub fn set_transform(&mut self, boundary: BoundaryId, transform: LayerTransform) -> bool {
        match self.boundaries.get_mut(&boundary) {
            Some(state) if state.transform != transform => {
                state.transform = transform;
                true
            }
            _ => false,
        }
    }

    /// Decide what a boundary does this frame.
    ///
    /// `content_dirty` is the walk's own measurement — whether any element
    /// inside the boundary needed re-emitting — and `reason` is the signal that
    /// woke the frame for this boundary's layer. See this module's doc for why
    /// both are required rather than either alone.
    ///
    /// Returns `None` for a boundary that was never declared.
    pub fn resolve(
        &mut self,
        boundary: BoundaryId,
        reason: Reason,
        content_dirty: bool,
        primitive_count: usize,
        transform_moved: bool,
    ) -> Option<BoundaryComposite> {
        let state = self.boundaries.get_mut(&boundary)?;
        state.primitive_count = primitive_count;
        state.retention = state.policy.retention_for(primitive_count);

        let composite = if content_dirty {
            Composite::Redisplay
        } else if !transform_moved {
            Composite::Clean
        } else if reason.permits_transform_only() {
            Composite::TransformOnly
        } else {
            // The transform moved but nothing said this was a scroll. The
            // conservative answer is to treat the boundary as ordinary content
            // for this frame; the walk has already folded the displacement into
            // the emitted positions, so there is nothing left to slide.
            Composite::Redisplay
        };

        Some(BoundaryComposite {
            boundary,
            layer: state.layer,
            composite,
            retention: state.retention,
            transform: state.transform,
            invalidation: composite.invalidation(),
        })
    }

    /// A boundary's retained state.
    pub fn state(&self, boundary: BoundaryId) -> Option<&BoundaryState> {
        self.boundaries.get(&boundary)
    }

    /// Where a boundary's content currently composites, or the identity for
    /// one that does not exist.
    pub fn transform(&self, boundary: BoundaryId) -> LayerTransform {
        self.boundaries
            .get(&boundary)
            .map(|state| state.transform)
            .unwrap_or(LayerTransform::IDENTITY)
    }

    /// Drop the state of every boundary unvisited for longer than its own
    /// `evict_after_frames`, returning the retained boundary layers those
    /// boundaries owned.
    ///
    /// R-N §3.4's mark-and-sweep, with its deliberate delay: a panel scrolled
    /// out of the tree and back within the interval re-materialises at the
    /// scroll position it had, rather than snapping to the top. What Phase 2
    /// retains over that interval is this record only — a boundary's *records*
    /// leave residency as soon as it leaves the tree, because pooling their
    /// storage is the texture-pool work §8 puts in Phase 4.
    ///
    /// The evicted layers are returned rather than counted because a boundary's
    /// layer outlives its records: the records go when the elements leave the
    /// tree, but the `Layer` entry itself is the compositor's to release, and
    /// nothing else knows when the interval has elapsed. Tile metadata is
    /// discarded with the boundary state and is never returned as scene-layer
    /// ownership.
    pub fn sweep(&mut self, frame: u64) -> Vec<LayerId> {
        let mut evicted = Vec::new();
        self.boundaries.retain(|_boundary, state| {
            let elapsed = frame.saturating_sub(state.last_visited_frame);
            if elapsed <= u64::from(state.policy.evict_after_frames) {
                return true;
            }
            evicted.push(state.layer);
            // Tiles are visibility and damage metadata, not scene layers. The
            // boundary's untiled layer is the only retained GPU ownership that
            // this compositor can release here.
            false
        });
        evicted
    }

    /// How many boundaries are live.
    pub fn len(&self) -> usize {
        self.boundaries.len()
    }

    /// Whether no boundary is live.
    pub fn is_empty(&self) -> bool {
        self.boundaries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::policy::Buffering;

    const PANEL: BoundaryId = BoundaryId::from_raw(7);

    fn compositor_with_panel() -> Compositor {
        let mut compositor = Compositor::new();
        compositor.visit(PANEL, BoundaryPolicy::default(), 1);
        compositor
    }

    #[test]
    fn a_scroll_over_clean_content_resolves_to_transform_only() {
        let mut compositor = compositor_with_panel();
        assert!(compositor.set_transform(PANEL, LayerTransform::translated(0.0, -40.0)));
        let composite = compositor.resolve(PANEL, Reason::Scroll, false, 12, true);
        assert_eq!(
            composite.map(|composite| composite.composite),
            Some(Composite::TransformOnly)
        );
        assert_eq!(
            composite.map(|composite| composite.invalidation),
            Some(Invalidation::TRANSFORM)
        );
    }

    #[test]
    fn dirty_content_never_reaches_the_fast_path_however_it_was_signalled() {
        let mut compositor = compositor_with_panel();
        assert!(compositor.set_transform(PANEL, LayerTransform::translated(0.0, -40.0)));
        let composite = compositor.resolve(PANEL, Reason::Scroll, true, 12, true);
        assert_eq!(
            composite.map(|composite| composite.composite),
            Some(Composite::Redisplay)
        );
    }

    #[test]
    fn a_data_change_signal_is_refused_the_fast_path_even_over_clean_content() {
        let mut compositor = compositor_with_panel();
        assert!(compositor.set_transform(PANEL, LayerTransform::translated(0.0, -40.0)));
        let composite = compositor.resolve(PANEL, Reason::DataChanged, false, 12, true);
        assert_eq!(
            composite.map(|composite| composite.composite),
            Some(Composite::Redisplay),
            "the fast path requires the signal as well as the measurement"
        );
    }

    #[test]
    fn an_idle_boundary_is_clean_rather_than_transform_only() {
        let mut compositor = compositor_with_panel();
        let composite = compositor.resolve(PANEL, Reason::Scroll, false, 12, false);
        assert_eq!(
            composite.map(|composite| composite.composite),
            Some(Composite::Clean)
        );
        assert_eq!(
            composite.map(|composite| composite.invalidation),
            Some(Invalidation::empty())
        );
        assert!(
            composite
                .map(|composite| composite.composite.leaves_content_resident())
                .unwrap_or(false)
        );
    }

    #[test]
    fn retention_is_decided_per_boundary_from_its_own_primitive_count() {
        let mut compositor = compositor_with_panel();
        let small = compositor.resolve(PANEL, Reason::Scroll, false, 12, false);
        assert_eq!(
            small.map(|composite| composite.retention),
            Some(Retention::Primitives)
        );
        let large = compositor.resolve(PANEL, Reason::Scroll, false, 4_000, false);
        assert_eq!(
            large.map(|composite| composite.retention),
            Some(Retention::Texture)
        );
        assert_eq!(
            compositor.state(PANEL).map(BoundaryState::primitive_count),
            Some(4_000)
        );
    }

    #[test]
    fn a_boundary_keeps_its_transform_across_frames_under_positional_identity() {
        let mut compositor = compositor_with_panel();
        assert!(compositor.set_transform(PANEL, LayerTransform::translated(0.0, -40.0)));
        // The same identity, re-declared next frame: nothing is reset.
        compositor.visit(PANEL, BoundaryPolicy::default(), 2);
        assert_eq!(
            compositor.transform(PANEL),
            LayerTransform::translated(0.0, -40.0)
        );
        assert!(!compositor.set_transform(PANEL, LayerTransform::translated(0.0, -40.0)));
    }

    #[test]
    fn a_policy_change_does_not_disturb_residency_or_position() {
        let mut compositor = compositor_with_panel();
        assert!(compositor.set_transform(PANEL, LayerTransform::translated(3.0, 5.0)));
        let layer = compositor.state(PANEL).map(BoundaryState::layer);
        compositor.visit(
            PANEL,
            BoundaryPolicy {
                rasterize_above: 4,
                buffering: Buffering::None,
                ..BoundaryPolicy::default()
            },
            2,
        );
        assert_eq!(compositor.state(PANEL).map(BoundaryState::layer), layer);
        assert_eq!(
            compositor.transform(PANEL),
            LayerTransform::translated(3.0, 5.0)
        );
        assert_eq!(
            compositor
                .state(PANEL)
                .map(|state| state.policy().rasterize_above),
            Some(4)
        );
    }

    #[test]
    fn an_unvisited_boundary_survives_its_eviction_interval_and_then_does_not() {
        let mut compositor = compositor_with_panel();
        let interval = u64::from(BoundaryPolicy::DEFAULT_EVICT_AFTER_FRAMES);
        assert!(compositor.sweep(1 + interval).is_empty());
        assert_eq!(compositor.len(), 1);
        assert_eq!(
            compositor.sweep(2 + interval),
            vec![LayerId::from_key(LayerKey::untiled(PANEL))],
            "an evicted boundary must name the layer it owned, so its caller can release it"
        );
        assert!(compositor.is_empty());
    }

    /// The bridge Phase 4 consumes: only a texture-retained boundary
    /// contributes a composite entry, and it names itself as the source.
    #[test]
    fn only_a_texture_retained_boundary_contributes_a_composite_entry() {
        let mut compositor = compositor_with_panel();
        let bounds = Rect::from_origin_size([10.0, 10.0], [200.0, 150.0]);

        let small = compositor
            .resolve(PANEL, Reason::Scroll, false, 12, false)
            .expect("the panel is declared");
        assert_eq!(small.retention, Retention::Primitives);
        assert_eq!(
            small.composite_entry(bounds, bounds, 7),
            None,
            "R-N §3.3: a boundary holding twelve quads re-emits rather than \
             compositing through a texture"
        );

        let large = compositor
            .resolve(PANEL, Reason::Scroll, false, 4_000, false)
            .expect("the panel is declared");
        assert_eq!(large.retention, Retention::Texture);
        let entry = large
            .composite_entry(bounds, bounds, 7)
            .expect("a texture-retained boundary has an entry");
        assert_eq!(entry.source, CompositeSource::BoundaryTexture(PANEL));
        assert_eq!(entry.content_token, 7);
        assert_eq!(entry.bounds, bounds);
        assert!(
            !entry.source_is_opaque,
            "a baked boundary texture is transparent wherever its content is, \
             so it must not occlude by default"
        );
        assert_eq!(entry.opaque_region(), None);
    }

    fn rect(x: f32, y: f32, width: f32, height: f32) -> Rect {
        Rect::from_origin_size([x, y], [width, height])
    }

    fn window() -> Rect {
        rect(0.0, 0.0, 1000.0, 800.0)
    }

    /// An external viewport, then an opaque modal painted over it.
    fn viewport_under_modal(modal: Rect) -> [CompositeEntry; 2] {
        let viewport = CompositeEntry::sampled(
            CompositeSource::External(ExternalSurfaceId::from_raw(1)),
            rect(200.0, 150.0, 400.0, 300.0),
            window(),
        );
        let modal = CompositeEntry {
            source_is_opaque: true,
            ..CompositeEntry::sampled(
                CompositeSource::BoundaryTexture(BoundaryId::from_raw(2)),
                modal,
                window(),
            )
        };
        [viewport, modal]
    }

    #[test]
    fn a_composite_entry_fully_covered_by_an_opaque_one_above_it_is_dropped() {
        let entries = viewport_under_modal(rect(100.0, 100.0, 700.0, 500.0));
        assert_eq!(
            visible_composites(&entries),
            vec![false, true],
            "§5.5: a viewport fully covered by a modal must stop being drawn"
        );
    }

    #[test]
    fn a_partially_covered_entry_is_kept() {
        // The modal covers all but a strip of the viewport's left edge.
        let entries = viewport_under_modal(rect(250.0, 100.0, 700.0, 500.0));
        assert_eq!(visible_composites(&entries), vec![true, true]);
    }

    #[test]
    fn an_entry_whose_source_is_not_declared_opaque_never_occludes() {
        // The same geometry as the covering case, with the one flag flipped.
        // An external producer's pixels are not the framework's to know
        // (§5.5), so nothing may be culled under it by default.
        let mut entries = viewport_under_modal(rect(100.0, 100.0, 700.0, 500.0));
        entries[1].source_is_opaque = false;
        assert_eq!(visible_composites(&entries), vec![true, true]);
        assert_eq!(entries[1].opaque_region(), None);
    }

    #[test]
    fn a_translucent_or_rounded_cover_insets_or_disqualifies_itself() {
        let mut entries = viewport_under_modal(rect(100.0, 100.0, 700.0, 500.0));
        entries[1].opacity = 0.9;
        assert_eq!(
            visible_composites(&entries),
            vec![true, true],
            "a translucent cover lets its contents show through"
        );

        let mut rounded = viewport_under_modal(rect(100.0, 100.0, 700.0, 500.0));
        rounded[1].corner_radius = 400.0;
        assert_eq!(
            visible_composites(&rounded),
            vec![true, true],
            "a radius wide enough to eat the overlap must inset the region, \
             not be ignored"
        );
    }

    #[test]
    fn a_cover_painted_below_does_not_occlude() {
        // Draw order is paint order: index 0 paints first, so entry 0 covering
        // entry 1's geometry hides nothing.
        let [viewport, modal] = viewport_under_modal(rect(100.0, 100.0, 700.0, 500.0));
        assert_eq!(
            visible_composites(&[modal, viewport]),
            vec![true, true],
            "the tier must never look forward in paint order"
        );
    }

    #[test]
    fn an_entry_clipped_to_nothing_is_kept_rather_than_culled() {
        let mut entries = viewport_under_modal(rect(100.0, 100.0, 700.0, 500.0));
        entries[0].content_mask = rect(0.0, 0.0, 0.0, 0.0);
        assert!(entries[0].visible().is_empty());
        assert_eq!(
            visible_composites(&entries).first().copied(),
            Some(true),
            "conservative, exactly as the instance tier is for an empty item"
        );
    }

    #[test]
    fn both_producers_are_the_same_kind_of_entry() {
        // §5.5's Gap 2: the tier must not care which producer made the pixels.
        let external = CompositeEntry {
            source_is_opaque: true,
            ..CompositeEntry::sampled(
                CompositeSource::External(ExternalSurfaceId::from_raw(9)),
                rect(0.0, 0.0, 1000.0, 800.0),
                window(),
            )
        };
        let baked = CompositeEntry {
            source: CompositeSource::BoundaryTexture(BoundaryId::from_raw(9)),
            ..external
        };
        let target = CompositeEntry::sampled(
            CompositeSource::BoundaryTexture(BoundaryId::from_raw(1)),
            rect(10.0, 10.0, 100.0, 100.0),
            window(),
        );
        assert_eq!(
            visible_composites(&[target, external]),
            visible_composites(&[target, baked])
        );
        assert_eq!(
            visible_composites(&[target, external]).first().copied(),
            Some(false)
        );
    }

    fn tiled_policy() -> BoundaryPolicy {
        BoundaryPolicy {
            buffering: Buffering::Tiled {
                tile_size: crate::boundary::policy::Size::pixels(256.0, 256.0),
                retain_radius: 1,
            },
            ..BoundaryPolicy::default()
        }
    }

    fn canvas_viewport() -> Rect {
        Rect::from_origin_size([0.0, 0.0], [900.0, 600.0])
    }

    #[test]
    fn a_tiled_boundary_resolves_a_visible_span_and_reveals_it_once() {
        let mut compositor = Compositor::new();
        let first = compositor
            .visit_tiled(PANEL, tiled_policy(), 1, canvas_viewport())
            .expect("a tiled policy with a usable tile size resolves");
        assert!(
            first.visible.len() > 1,
            "a 900x600 viewport spans several tiles"
        );
        assert_eq!(
            first.revealed.len(),
            first.visible.len(),
            "every tile is new on the first frame"
        );
        assert_eq!(first.resident, first.visible.len());
        assert_eq!(first.over_budget, 0);

        // The same frame again reveals nothing: residency is idempotent, the
        // same way `visit` is.
        let second = compositor
            .visit_tiled(PANEL, tiled_policy(), 2, canvas_viewport())
            .expect("still tiled");
        assert!(second.revealed.is_empty());
        assert_eq!(second.visible, first.visible);
    }

    #[test]
    fn a_boundary_that_is_not_tiled_reports_no_tile_set_at_all() {
        let mut compositor = Compositor::new();
        assert!(
            compositor
                .visit_tiled(PANEL, BoundaryPolicy::default(), 1, canvas_viewport())
                .is_none(),
            "Margin(None) has no grid, so there is nothing to resolve"
        );
        assert!(
            compositor
                .visit_tiled(
                    PANEL,
                    BoundaryPolicy {
                        buffering: Buffering::Tiled {
                            tile_size: crate::boundary::policy::Size::pixels(0.0, 0.0),
                            retain_radius: 1,
                        },
                        ..BoundaryPolicy::default()
                    },
                    1,
                    canvas_viewport(),
                )
                .is_none(),
            "an unusable tile size must fall back rather than resolve a grid"
        );
        // Either way the boundary itself is still declared — falling back means
        // buffering it the untiled way, not dropping it.
        assert!(compositor.state(PANEL).is_some());
        assert!(
            compositor
                .state(PANEL)
                .and_then(BoundaryState::tiles)
                .is_none()
        );
    }

    /// Tiled visibility does not create a second retained scene ownership.
    #[test]
    fn sweeping_a_tiled_boundary_names_only_its_untiled_layer() {
        let mut compositor = Compositor::new();
        let visit = compositor
            .visit_tiled(PANEL, tiled_policy(), 1, canvas_viewport())
            .expect("a tiled boundary");
        let tiles = visit.visible.clone();
        assert!(tiles.len() > 4, "the premise: several tiles are resident");

        let interval = u64::from(BoundaryPolicy::DEFAULT_EVICT_AFTER_FRAMES);
        let evicted = compositor.sweep(2 + interval);
        assert!(compositor.is_empty());

        assert_eq!(
            evicted,
            vec![LayerId::from_key(LayerKey::untiled(PANEL))],
            "tile residency is not a second scene layer"
        );
    }

    #[test]
    fn panning_a_tiled_boundary_reveals_only_what_the_pan_uncovered() {
        let mut compositor = Compositor::new();
        // 8px in, so the viewport does not start on a tile boundary — see
        // `ui_walk`'s gate for why that distinction is load-bearing.
        compositor.set_transform(PANEL, LayerTransform::translated(-8.0, -8.0));
        let first = compositor
            .visit_tiled(PANEL, tiled_policy(), 1, canvas_viewport())
            .expect("a tiled boundary");
        assert!(
            compositor.set_transform(PANEL, LayerTransform::translated(-8.0, -8.0)),
            "the boundary now exists, so positioning it takes effect"
        );

        compositor.set_transform(PANEL, LayerTransform::translated(-264.0, -8.0));
        let panned = compositor
            .visit_tiled(PANEL, tiled_policy(), 2, canvas_viewport())
            .expect("still tiled");
        assert!(
            !panned.revealed.is_empty(),
            "one tile of pan reveals a column"
        );
        assert!(
            panned.revealed.len() < first.visible.len(),
            "a one-tile pan revealed {} of {} tiles, which is a refill",
            panned.revealed.len(),
            first.visible.len()
        );
        assert!(
            panned
                .revealed
                .iter()
                .all(|tile| !first.visible.contains(tile)),
            "a tile already resident must not be reported as revealed"
        );
    }

    #[test]
    fn resolving_or_moving_an_undeclared_boundary_is_inert() {
        let mut compositor = Compositor::new();
        assert!(!compositor.set_transform(PANEL, LayerTransform::translated(1.0, 1.0)));
        assert!(
            compositor
                .resolve(PANEL, Reason::Scroll, false, 0, false)
                .is_none()
        );
        assert_eq!(compositor.transform(PANEL), LayerTransform::IDENTITY);
        assert!(compositor.is_empty());
    }
}
