//! `FramePlan` → `ScenePatch`: the arrow §2's diagram draws and Phase 1 left
//! undone. See docs/gpu-native-architecture.md §2, §4.1, §5.0.
//!
//! Not in §3.1's literal file map — a deliberate addition, recorded in
//! `docs/phase-2-results.md`. `docs/phase-1-results.md` §6 names this seam
//! precisely: "nothing produces a `Description` or a `ScenePatch` from a real
//! element yet... the arrow from a `FramePlan` to a `ScenePatch` does not
//! exist." It does not exist because deciding *which primitives an element
//! emits* needs an element vocabulary §3.4 puts in `wgpui-widgets`. This module
//! is the smallest thing that closes the arrow without inventing that
//! vocabulary: a generic hook an element supplies, and a real walk that turns
//! the plan plus the hook plus computed layout into patches.
//!
//! # Why a trait and not a closure type
//!
//! §4.1's own framing offered either. [`Emit`] is a trait for three reasons,
//! and a blanket impl means call sites that want a closure still write one:
//!
//! 1. A real element emits from state it already holds — a resolved style, a
//!    glyph run, an image handle. A trait lets that be a method on a small
//!    value the element already has; a `Box<dyn Fn>` would force a
//!    capture-by-move closure to be constructed per element per frame, which is
//!    exactly the per-frame allocation §4.2 objects to elsewhere.
//! 2. The emitter writes into a **reused** [`Emission`] buffer rather than
//!    returning a `Vec`, so a thousand-element frame allocates once, not a
//!    thousand times. Expressing that as a closure type means naming
//!    `Fn(&EmitContext, &mut Emission)` at every call site that stores one; a
//!    trait names it once.
//! 3. `wgpui-widgets` will want emitters that are inspectable (an element
//!    inspector, §3.6's `inspector.rs`) and replayable (§3.6's capture/replay
//!    engine). A trait can grow a diagnostic method without breaking callers; a
//!    bare function type cannot grow anything.
//!
//! # What decides that an element re-emits
//!
//! Exactly one rule, and it is not "the plan said `Rebuilt`":
//!
//! > An element re-emits when the plan did not mark it reused, **or** when its
//! > resolved absolute bounds are not the ones it was last emitted with, **or**
//! > when it moved to a different layer.
//!
//! The second clause is the load-bearing one, and it is what makes `.boundary()`
//! observable rather than asserted. A reconciler diffs *descriptions*; it has
//! no opinion about where computed layout put an element. So a scroll container
//! that folds its offset into its children's positions moves those children,
//! and they re-emit — even though every one of them reconciled perfectly clean.
//! A scroll container that is a `.boundary()` puts the offset on its layer's
//! transform instead, its children's absolute bounds do not move, and none of
//! them re-emits. That is the entire difference between Phase 2's first and
//! second gates, and it falls out of one condition rather than being special
//! cased at either end.
//!
//! # What this deliberately does not do
//!
//! Paint order within a layer is append-order, not tree order: a record keeps
//! the slot it was first inserted at, and a newly-inserted record goes to the
//! end of its layer's list. Establishing z-order is §5.1's per-layer ordering
//! pass, which Phase 3 builds on top of these same slabs, and Phase 1's
//! `Scene::draw_ranges` is explicitly a CPU placeholder until then. Emitting a
//! correct *set* of primitives into a correct *layer* is what this phase needs
//! and all it claims.

use crate::boundary::compositor::{BoundaryComposite, Composite, Compositor, TiledVisit};
use crate::boundary::policy::BoundaryPolicy;
use crate::geometry::Rect;
use crate::invalidation::request::FrameSignals;
use crate::patch::apply::ScenePatch;
use crate::patch::primitive::{
    AtlasTileId, BackdropFilter, GlyphRun, Path, PolySprite, Primitive, Quad, Shadow, Underline,
};
use crate::patch::{PatchList, RecordKey};
use crate::reconcile::instance::InstanceKey;
use crate::reconcile::plan::FramePlan;
use crate::reconcile::walk::shared_walk;
use crate::scene::layer::{BoundaryId, LayerId, LayerKey, LayerTransform};
use crate::scene::{PrimitiveStore, Scene};
use std::collections::HashMap;
use wgpui_layout::taffy_tree::{LayoutError, LayoutRect, LayoutTree};

/// Where an element ended up, handed to its [`Emit`] so it can place what it
/// draws.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct EmitContext {
    /// The element's absolute rectangle, with every ancestor's position and
    /// every ancestor's folded-in scroll displacement already applied.
    pub bounds: LayoutRect,
    /// The layer this element's primitives land in.
    pub layer: LayerId,
    /// The compositing boundary owning that layer.
    pub boundary: BoundaryId,
    /// The accumulated rectangular clip from ancestors, if any.
    pub clip: Option<LayoutRect>,
}

/// What one element contributes to the scene this frame.
///
/// Written into rather than returned, so the walk reuses one buffer for the
/// whole frame. Order within each kind is the element's own and must be stable
/// across frames: a record's cross-frame address is its ordinal here, so an
/// element that emits its background quad and then its border quad keeps both
/// addresses, and one that reorders them swaps two records' values rather than
/// reordering the records.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Emission {
    shadows: Vec<Shadow>,
    quads: Vec<Quad>,
    underlines: Vec<Underline>,
    glyph_runs: Vec<GlyphRun>,
    poly_sprites: Vec<PolySprite>,
    paths: Vec<Path>,
    backdrop_filters: Vec<BackdropFilter>,
}

impl Emission {
    /// An emission holding nothing.
    pub fn new() -> Self {
        Self::default()
    }

    /// Contribute a drop shadow.
    pub fn shadow(&mut self, shadow: Shadow) -> &mut Self {
        self.shadows.push(shadow);
        self
    }

    /// The shadows contributed, in emission order.
    pub fn shadows(&self) -> &[Shadow] {
        &self.shadows
    }

    /// Contribute a quad.
    pub fn quad(&mut self, quad: Quad) -> &mut Self {
        self.quads.push(quad);
        self
    }

    /// Contribute a run of already-shaped glyphs.
    pub fn glyph_run(&mut self, run: GlyphRun) -> &mut Self {
        self.glyph_runs.push(run);
        self
    }

    /// The quads contributed, in emission order.
    pub fn quads(&self) -> &[Quad] {
        &self.quads
    }

    /// Contribute an underline or strikethrough rule.
    pub fn underline(&mut self, underline: Underline) -> &mut Self {
        self.underlines.push(underline);
        self
    }

    /// The underlines contributed, in emission order.
    pub fn underlines(&self) -> &[Underline] {
        &self.underlines
    }

    /// Contribute an image sprite.
    pub fn poly_sprite(&mut self, sprite: PolySprite) -> &mut Self {
        self.poly_sprites.push(sprite);
        self
    }

    /// The glyph runs contributed, in emission order.
    pub fn glyph_runs(&self) -> &[GlyphRun] {
        &self.glyph_runs
    }

    /// The image sprites contributed, in emission order.
    pub fn poly_sprites(&self) -> &[PolySprite] {
        &self.poly_sprites
    }

    /// Contribute a Lyon-tessellated vector path.
    pub fn path(&mut self, path: Path) -> &mut Self {
        self.paths.push(path);
        self
    }

    /// The paths contributed, in emission order.
    pub fn paths(&self) -> &[Path] {
        &self.paths
    }

    /// Contribute a framebuffer-sampling backdrop filter.
    pub fn backdrop_filter(&mut self, filter: BackdropFilter) -> &mut Self {
        self.backdrop_filters.push(filter);
        self
    }

    /// The backdrop filters contributed, in emission order.
    pub fn backdrop_filters(&self) -> &[BackdropFilter] {
        &self.backdrop_filters
    }

    /// Total primitives contributed.
    pub fn len(&self) -> usize {
        self.shadows.len()
            + self.quads.len()
            + self.underlines.len()
            + self.glyph_runs.len()
            + self.poly_sprites.len()
            + self.paths.len()
            + self.backdrop_filters.len()
    }

    /// Whether the element contributed nothing.
    pub fn is_empty(&self) -> bool {
        self.shadows.is_empty()
            && self.quads.is_empty()
            && self.underlines.is_empty()
            && self.glyph_runs.is_empty()
            && self.poly_sprites.is_empty()
            && self.paths.is_empty()
            && self.backdrop_filters.is_empty()
    }

    /// Drop everything, keeping the allocations for the next element.
    pub fn clear(&mut self) {
        self.shadows.clear();
        self.quads.clear();
        self.underlines.clear();
        self.glyph_runs.clear();
        self.poly_sprites.clear();
        self.paths.clear();
        self.backdrop_filters.clear();
    }

    /// Retain primitive slots while applying an inherited rectangular clip.
    /// Zeroing an outside quad rather than removing it keeps record ordinals
    /// stable, so scrolling across a clip boundary remains a value update.
    pub fn clip_quads_to(&mut self, clip: LayoutRect) {
        for quad in &mut self.quads {
            let left = quad.origin[0].max(clip.x);
            let top = quad.origin[1].max(clip.y);
            let right = (quad.origin[0] + quad.size[0]).min(clip.x + clip.width);
            let bottom = (quad.origin[1] + quad.size[1]).min(clip.y + clip.height);
            if right <= left || bottom <= top {
                quad.origin = [left, top];
                quad.size = [0.0, 0.0];
                quad.background[3] = 0.0;
                quad.border_color[3] = 0.0;
            } else {
                quad.origin = [left, top];
                quad.size = [right - left, bottom - top];
            }
        }
    }

    /// Clip every primitive emitted by a child of a rectangular scroll/clip
    /// container. Variable-size kinds keep their slots so retained addresses
    /// do not churn while scrolling; fully outside instances become inert.
    pub fn clip_to(&mut self, clip: LayoutRect) {
        self.clip_quads_to(clip);
        for run in &mut self.glyph_runs {
            for glyph in &mut run.glyphs {
                let bounds = LayoutRect {
                    x: glyph.position[0],
                    y: glyph.position[1],
                    width: glyph.atlas_size[0],
                    height: glyph.atlas_size[1],
                };
                if let Some(cropped) = intersect_rect(bounds, clip) {
                    let delta_x = cropped.x - bounds.x;
                    let delta_y = cropped.y - bounds.y;
                    glyph.atlas_origin[0] += delta_x;
                    glyph.atlas_origin[1] += delta_y;
                    glyph.position = [cropped.x, cropped.y];
                    glyph.atlas_size = [cropped.width, cropped.height];
                } else {
                    glyph.atlas_size = [0.0; 2];
                    glyph.atlas_tile = AtlasTileId::NONE;
                }
            }
        }
        for sprite in &mut self.poly_sprites {
            let bounds = LayoutRect {
                x: sprite.origin[0],
                y: sprite.origin[1],
                width: sprite.size[0],
                height: sprite.size[1],
            };
            if let Some(cropped) = intersect_rect(bounds, clip) {
                let scale_x = if sprite.size[0] > 0.0 {
                    sprite.atlas_size[0] / sprite.size[0]
                } else {
                    0.0
                };
                let scale_y = if sprite.size[1] > 0.0 {
                    sprite.atlas_size[1] / sprite.size[1]
                } else {
                    0.0
                };
                sprite.atlas_origin[0] += (cropped.x - bounds.x) * scale_x;
                sprite.atlas_origin[1] += (cropped.y - bounds.y) * scale_y;
                sprite.atlas_size = [cropped.width * scale_x, cropped.height * scale_y];
                sprite.origin = [cropped.x, cropped.y];
                sprite.size = [cropped.width, cropped.height];
            } else {
                sprite.size = [0.0; 2];
                sprite.opacity = 0.0;
                sprite.atlas_tile = AtlasTileId::NONE;
            }
        }
        for underline in &mut self.underlines {
            let bounds = LayoutRect {
                x: underline.origin[0],
                y: underline.origin[1],
                width: underline.size[0],
                height: underline.size[1],
            };
            if let Some(cropped) = intersect_rect(bounds, clip) {
                underline.origin = [cropped.x, cropped.y];
                underline.size = [cropped.width, cropped.height];
            } else {
                underline.size = [0.0; 2];
                underline.color[3] = 0.0;
            }
        }
        for shadow in &mut self.shadows {
            let (origin, size) = shadow.drawn_bounds();
            if !rects_intersect(
                LayoutRect {
                    x: origin[0],
                    y: origin[1],
                    width: size[0],
                    height: size[1],
                },
                clip,
            ) {
                shadow.size = [0.0; 2];
                shadow.color[3] = 0.0;
            }
        }
        for path in &mut self.paths {
            let path_clip = LayoutRect {
                x: path.clip_origin[0],
                y: path.clip_origin[1],
                width: path.clip_size[0],
                height: path.clip_size[1],
            };
            if let Some(cropped) = intersect_rect(path_clip, clip) {
                path.clip_origin = [cropped.x, cropped.y];
                path.clip_size = [cropped.width, cropped.height];
            } else {
                path.clip_size = [0.0; 2];
            }
        }
        for filter in &mut self.backdrop_filters {
            let filter_clip = LayoutRect {
                x: filter.clip_origin[0],
                y: filter.clip_origin[1],
                width: filter.clip_size[0],
                height: filter.clip_size[1],
            };
            if let Some(cropped) = intersect_rect(filter_clip, clip) {
                filter.clip_origin = [cropped.x, cropped.y];
                filter.clip_size = [cropped.width, cropped.height];
            } else {
                filter.clip_size = [0.0; 2];
            }
        }
    }
}

fn rects_intersect(first: LayoutRect, second: LayoutRect) -> bool {
    intersect_rect(first, second).is_some()
}

fn intersect_rect(first: LayoutRect, second: LayoutRect) -> Option<LayoutRect> {
    let x = first.x.max(second.x);
    let y = first.y.max(second.y);
    let right = (first.x + first.width).min(second.x + second.width);
    let bottom = (first.y + first.height).min(second.y + second.height);
    (right > x && bottom > y).then_some(LayoutRect {
        x,
        y,
        width: right - x,
        height: bottom - y,
    })
}

/// An element's contribution to the scene, given where layout put it.
///
/// See this module's doc for why this is a trait. The blanket impl below means
/// a closure is a valid emitter wherever one reads better.
pub trait Emit: 'static {
    /// Write this element's primitives into `emission`.
    ///
    /// Called only on frames where the element actually needs re-emitting — see
    /// this module's doc for the rule — so it must be a pure function of
    /// `context` and the element's own captured state, never a place to run a
    /// side effect that must happen every frame. R-N §2.4's `on_frame` is where
    /// that belongs.
    fn emit(&self, context: &EmitContext, emission: &mut Emission);
}

impl<F> Emit for F
where
    F: Fn(&EmitContext, &mut Emission) + 'static,
{
    fn emit(&self, context: &EmitContext, emission: &mut Emission) {
        self(context, emission)
    }
}

/// Emission could not complete.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmitError {
    /// A planned node's layout could not be read.
    Layout(LayoutError),
    /// The plan's depths do not describe a tree: a node claimed to be more than
    /// one level below the node before it.
    MalformedPlan {
        /// Index of the offending node.
        index: usize,
        /// The depth it claimed.
        depth: u32,
    },
}

impl std::fmt::Display for EmitError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmitError::Layout(error) => write!(formatter, "layout: {error}"),
            EmitError::MalformedPlan { index, depth } => write!(
                formatter,
                "planned node {index} claims depth {depth}, which is not reachable from its predecessor"
            ),
        }
    }
}

impl std::error::Error for EmitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            EmitError::Layout(error) => Some(error),
            EmitError::MalformedPlan { .. } => None,
        }
    }
}

impl From<LayoutError> for EmitError {
    fn from(error: LayoutError) -> Self {
        EmitError::Layout(error)
    }
}

/// How much emission work one frame actually did.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct EmissionStats {
    /// Elements the walk visited.
    pub nodes_visited: usize,
    /// Elements whose [`Emit`] was called.
    pub nodes_emitted: usize,
    /// Elements that had an emitter and did not need it called.
    pub nodes_skipped: usize,
    /// Records added.
    pub records_inserted: usize,
    /// Records whose value was replaced in place — §5.0's O(1) case.
    pub records_updated: usize,
    /// Records dropped, whether because an element shrank or left the tree.
    pub records_removed: usize,
    /// Boundaries live this frame, including the window root.
    pub boundaries: usize,
    /// Boundaries that reached the transform-only fast path.
    pub transform_only: usize,
}

/// One frame's emission: the patch to apply, and what each boundary decided.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FrameEmission {
    /// The patch. Pure data — §2's actual frontend/backend seam.
    pub patch: ScenePatch,
    /// What each live boundary did, in ascending layer order.
    pub composites: Vec<BoundaryComposite>,
    /// Tile visibility metadata for damage planning. Tiles are not scene
    /// layers and do not own primitive records; they identify portions of the
    /// retained presentation buffer that may need repainting.
    pub tiled_visits: Vec<TiledVisit>,
    /// Screen-space rectangles whose pixels no longer match the previous
    /// presentation. This is presentation damage, not a second copy of the
    /// scene: the GPU uses it to restrict rasterization while the retained
    /// scene remains the only source of primitive data.
    pub damage: Vec<Rect>,
    /// This frame's counters.
    pub stats: EmissionStats,
}

impl FrameEmission {
    /// What a given boundary did this frame.
    pub fn composite_for(&self, boundary: BoundaryId) -> Option<BoundaryComposite> {
        self.composites
            .iter()
            .find(|composite| composite.boundary == boundary)
            .copied()
    }

    /// Whether every live boundary left its resident primitives untouched.
    pub fn leaves_every_boundary_resident(&self) -> bool {
        !self.composites.is_empty()
            && self
                .composites
                .iter()
                .all(|composite| composite.composite.leaves_content_resident())
    }
}

/// What the emitter last put into the scene for one element.
#[derive(Copy, Clone, Debug, PartialEq)]
struct EmittedNode {
    layer: LayerId,
    bounds: LayoutRect,
    /// The inherited clip used when this node last emitted. A resize can
    /// change this without changing the node's own layout rectangle.
    clip: Option<LayoutRect>,
    visible_bounds: Rect,
    shadows: u32,
    quads: u32,
    underlines: u32,
    glyph_runs: u32,
    poly_sprites: u32,
    paths: u32,
    backdrop_filters: u32,
    last_visited_frame: u64,
}

impl EmittedNode {
    /// Records this element holds across every kind.
    fn record_count(&self) -> usize {
        self.shadows as usize
            + self.quads as usize
            + self.underlines as usize
            + self.glyph_runs as usize
            + self.poly_sprites as usize
            + self.paths as usize
            + self.backdrop_filters as usize
    }
}

/// One boundary's accumulated state for the frame being walked.
#[derive(Copy, Clone, Debug)]
struct BoundaryFrame {
    layer: LayerId,
    content_dirty: bool,
    primitive_count: usize,
    transform_moved: bool,
}

/// One kind's pending operations, kept apart so removals can be emitted before
/// insertions and every insertion index stays correct.
struct KindOperations<P> {
    removes: Vec<(LayerId, RecordKey)>,
    updates: Vec<(LayerId, RecordKey, P)>,
    inserts: Vec<(LayerId, RecordKey, P)>,
}

impl<P> Default for KindOperations<P> {
    fn default() -> Self {
        Self {
            removes: Vec::new(),
            updates: Vec::new(),
            inserts: Vec::new(),
        }
    }
}

/// Every kind's pending operations for one frame's walk.
///
/// Grouped into one value rather than passed as one parameter per kind: the
/// functions below take *all* of them or none, so a per-kind parameter list
/// makes the signature grow with the kind set and says nothing extra about what
/// the function reads. Phase 6.3 added the fourth and fifth kinds, at which
/// point [`Emitter::sweep_departed`]'s parameter list would have crossed
/// clippy's `too_many_arguments` threshold — the grouping is what that pressure
/// was pointing at, not a suppression of it.
#[derive(Default)]
struct PendingOperations {
    shadows: KindOperations<Shadow>,
    quads: KindOperations<Quad>,
    underlines: KindOperations<Underline>,
    glyph_runs: KindOperations<GlyphRun>,
    poly_sprites: KindOperations<PolySprite>,
    paths: KindOperations<Path>,
    backdrop_filters: KindOperations<BackdropFilter>,
}

impl<P: Primitive> KindOperations<P> {
    /// Fold this kind's operations into one ordered [`PatchList`].
    ///
    /// Removals first, then value updates, then insertions — so an insertion
    /// index computed against the layer's post-removal length is the index the
    /// scene will actually see. `store` supplies each layer's starting length;
    /// nothing here mutates the scene.
    fn into_patch_list(self, store: &PrimitiveStore<P>, stats: &mut EmissionStats) -> PatchList<P> {
        let mut list = PatchList::new();
        let mut lengths: HashMap<LayerId, u32> = HashMap::new();

        for (layer, key) in self.removes {
            let length = lengths.entry(layer).or_insert_with(|| store.len(layer));
            *length = length.saturating_sub(1);
            list.remove(layer, key);
            stats.records_removed += 1;
        }
        for (layer, key, value) in self.updates {
            list.update(layer, key, value);
            stats.records_updated += 1;
        }
        for (layer, key, value) in self.inserts {
            let length = lengths.entry(layer).or_insert_with(|| store.len(layer));
            list.insert(layer, key, *length, value);
            *length = length.saturating_add(1);
            stats.records_inserted += 1;
        }
        list
    }
}

/// The retained half of emission: what each element currently has resident, and
/// every live boundary's compositing state.
///
/// Held across frames by whoever drives the frame loop, alongside the
/// [`crate::reconcile::reconciler::Reconciler`] and the [`Scene`]. It owns the
/// [`Compositor`] rather than sitting beside it because every boundary decision
/// needs the walk's own content-dirty measurement, and splitting the two would
/// mean handing that measurement back and forth.
#[derive(Debug, Default)]
pub struct Emitter {
    compositor: Compositor,
    emitted: HashMap<InstanceKey, EmittedNode>,
    /// Layers whose boundary was evicted, released at the *start* of the next
    /// frame rather than at the end of the one that evicted them.
    ///
    /// The delay is what makes the release safe: [`Emitter::emit`] produces a
    /// patch and its caller applies it afterwards, so a layer dropped in the
    /// same call could still be named by ops in the patch just returned.
    /// Deferring one frame means the caller has applied that patch before the
    /// layer's reservations are handed back.
    pending_layer_removals: Vec<LayerId>,
    frame: u64,
}

impl Emitter {
    /// An emitter holding nothing.
    pub fn new() -> Self {
        Self::default()
    }

    /// The index of the most recently emitted frame.
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// Every live boundary's compositing state.
    pub fn compositor(&self) -> &Compositor {
        &self.compositor
    }

    /// How many elements currently have primitives resident.
    pub fn emitting_element_count(&self) -> usize {
        self.emitted.len()
    }

    /// Turn one reconciled, laid-out frame into a patch.
    ///
    /// `scene` is mutated only structurally — layers are declared and their
    /// composite transforms are set. The returned patch is applied separately
    /// via [`crate::patch::apply::apply`], which keeps "what changed" and
    /// "apply what changed" the two distinguishable steps §2 asks for and makes
    /// the emission inspectable before it lands.
    pub fn emit(
        &mut self,
        plan: &FramePlan,
        layout: &LayoutTree,
        signals: &FrameSignals,
        scene: &mut Scene,
    ) -> Result<FrameEmission, EmitError> {
        self.frame += 1;
        let frame = self.frame;
        for layer in std::mem::take(&mut self.pending_layer_removals) {
            scene.remove_layer(layer);
        }

        let mut stats = EmissionStats::default();
        let mut pending = PendingOperations::default();
        let mut boundaries: HashMap<BoundaryId, BoundaryFrame> = HashMap::new();
        let mut tiled_visits = Vec::new();
        let mut damage = Vec::new();
        let mut scroll_regions = HashMap::new();
        let mut emission = Emission::new();
        let effective_signals = self.infer_clean_scrolls(plan, signals);

        self.begin_boundary(
            BoundaryId::ROOT,
            BoundaryPolicy::default(),
            LayerTransform::IDENTITY,
            frame,
            scene,
            &mut boundaries,
        );
        let walked = shared_walk(plan, layout, &effective_signals, None)?;

        for (index, node) in plan.nodes().iter().enumerate() {
            let geometry = walked.get(index).ok_or(EmitError::MalformedPlan {
                index,
                depth: node.depth,
            })?;
            stats.nodes_visited += 1;

            let bounds = geometry.emission_bounds;

            // A boundary root's own paint belongs to the layer around it, not to
            // the layer it declares — see `PlannedNode::boundary`.
            let layer = geometry.layer;
            if let Some(declared) = node.declared_boundary {
                let policy = node.boundary_policy.unwrap_or_default();
                let declared_layer = LayerId::from_key(LayerKey::untiled(declared));
                // Decided from the signal alone, before the walk knows
                // whether the content is clean, because it changes where
                // the content is emitted: a boundary permitted the fast
                // path hands its displacement to its layer, and one that is
                // not folds it into its children exactly as an ordinary
                // element would.
                let slides = effective_signals
                    .reason_for_layer(declared_layer)
                    .permits_transform_only();
                if slides {
                    scroll_regions.insert(
                        declared,
                        geometry
                            .child_clip
                            .unwrap_or(geometry.absolute_bounds),
                    );
                }
                let transform = if slides {
                    LayerTransform {
                        translation: node.scroll_offset,
                    }
                } else {
                    LayerTransform::IDENTITY
                };
                let declared_layer = self.begin_boundary(
                    declared,
                    policy,
                    transform,
                    frame,
                    scene,
                    &mut boundaries,
                );
                scene.layers.set_clip(declared_layer, geometry.child_clip);
                if let Some(visit) = self.compositor.visit_tiled(
                    declared,
                    policy,
                    frame,
                    geometry.child_clip.unwrap_or(geometry.absolute_bounds),
                ) {
                    tiled_visits.push(visit);
                }
            }

            let previous = self.emitted.get(&node.address).copied();
            match plan.emitter(index) {
                Some(emitter) => {
                    let stale = previous.is_none_or(|record| {
                        record.bounds != bounds
                            || record.layer != layer
                            || record.clip != geometry.emission_clip
                    });
                    if node.skipped_prepaint_and_paint() && !stale {
                        stats.nodes_skipped += 1;
                        if let Some(record) = previous {
                            self.record_visited(node.address, record, frame);
                            Self::account(
                                &mut boundaries,
                                node.boundary,
                                false,
                                record.record_count(),
                            );
                        }
                    } else {
                        if let Some(previous) = previous {
                            damage.push(previous.visible_bounds);
                        }
                        damage.push(geometry.visible_bounds);
                        stats.nodes_emitted += 1;
                        emission.clear();
                        emitter.emit(
                            &EmitContext {
                                bounds,
                                layer,
                                boundary: node.boundary,
                                clip: geometry.emission_clip,
                            },
                            &mut emission,
                        );
                        if let Some(clip) = geometry.emission_clip {
                            emission.clip_to(clip);
                        }
                        Self::reconcile_records(
                            node.address,
                            layer,
                            previous.map(|record| (record.layer, record.shadows)),
                            emission.shadows(),
                            &mut pending.shadows,
                        );
                        Self::reconcile_records(
                            node.address,
                            layer,
                            previous.map(|record| (record.layer, record.quads)),
                            emission.quads(),
                            &mut pending.quads,
                        );
                        Self::reconcile_records(
                            node.address,
                            layer,
                            previous.map(|record| (record.layer, record.underlines)),
                            emission.underlines(),
                            &mut pending.underlines,
                        );
                        Self::reconcile_records(
                            node.address,
                            layer,
                            previous.map(|record| (record.layer, record.glyph_runs)),
                            emission.glyph_runs(),
                            &mut pending.glyph_runs,
                        );
                        Self::reconcile_records(
                            node.address,
                            layer,
                            previous.map(|record| (record.layer, record.poly_sprites)),
                            emission.poly_sprites(),
                            &mut pending.poly_sprites,
                        );
                        Self::reconcile_records(
                            node.address,
                            layer,
                            previous.map(|record| (record.layer, record.paths)),
                            emission.paths(),
                            &mut pending.paths,
                        );
                        Self::reconcile_records(
                            node.address,
                            layer,
                            previous.map(|record| (record.layer, record.backdrop_filters)),
                            emission.backdrop_filters(),
                            &mut pending.backdrop_filters,
                        );
                        let emitted = EmittedNode {
                            layer,
                            bounds,
                            clip: geometry.emission_clip,
                            visible_bounds: geometry.visible_bounds,
                            shadows: u32::try_from(emission.shadows().len()).unwrap_or(u32::MAX),
                            quads: u32::try_from(emission.quads().len()).unwrap_or(u32::MAX),
                            underlines: u32::try_from(emission.underlines().len())
                                .unwrap_or(u32::MAX),
                            glyph_runs: u32::try_from(emission.glyph_runs().len())
                                .unwrap_or(u32::MAX),
                            poly_sprites: u32::try_from(emission.poly_sprites().len())
                                .unwrap_or(u32::MAX),
                            paths: u32::try_from(emission.paths().len()).unwrap_or(u32::MAX),
                            backdrop_filters: u32::try_from(emission.backdrop_filters().len())
                                .unwrap_or(u32::MAX),
                            last_visited_frame: frame,
                        };
                        self.emitted.insert(node.address, emitted);
                        Self::account(&mut boundaries, node.boundary, true, emission.len());
                    }
                }
                None => {
                    // An element that stopped emitting must take its records
                    // with it, not leave them resident under an address nothing
                    // will address again.
                    if let Some(record) = previous {
                        damage.push(record.visible_bounds);
                        Self::retire_records(node.address, record, &mut pending);
                        self.emitted.remove(&node.address);
                        Self::account(&mut boundaries, node.boundary, true, 0);
                    }
                }
            }
        }

        self.sweep_departed(frame, &mut pending, &mut boundaries, &mut damage);

        let patch = ScenePatch {
            shadows: pending.shadows.into_patch_list(&scene.shadows, &mut stats),
            quads: pending.quads.into_patch_list(&scene.quads, &mut stats),
            underlines: pending
                .underlines
                .into_patch_list(&scene.underlines, &mut stats),
            glyph_runs: pending
                .glyph_runs
                .into_patch_list(&scene.glyph_runs, &mut stats),
            poly_sprites: pending
                .poly_sprites
                .into_patch_list(&scene.poly_sprites, &mut stats),
            paths: pending.paths.into_patch_list(&scene.paths, &mut stats),
            backdrop_filters: pending
                .backdrop_filters
                .into_patch_list(&scene.backdrop_filters, &mut stats),
            ..ScenePatch::new()
        };

        let mut composites = Vec::with_capacity(boundaries.len());
        for (boundary, state) in boundaries {
            let reason = effective_signals.reason_for_layer(state.layer);
            let Some(composite) = self.compositor.resolve(
                boundary,
                reason,
                state.content_dirty,
                state.primitive_count,
                state.transform_moved,
            ) else {
                continue;
            };
            scene.layers.set_transform(state.layer, composite.transform);
            if composite.composite == Composite::TransformOnly {
                stats.transform_only += 1;
                if let Some(region) = scroll_regions.get(&boundary) {
                    damage.push(*region);
                }
            }
            composites.push(composite);
        }
        composites.sort_by_key(|composite| composite.layer);
        stats.boundaries = composites.len();

        self.pending_layer_removals
            .extend(self.compositor.sweep(frame));
        Ok(FrameEmission {
            patch,
            composites,
            tiled_visits,
            damage,
            stats,
        })
    }

    fn begin_boundary(
        &mut self,
        boundary: BoundaryId,
        policy: BoundaryPolicy,
        transform: LayerTransform,
        frame: u64,
        scene: &mut Scene,
        boundaries: &mut HashMap<BoundaryId, BoundaryFrame>,
    ) -> LayerId {
        let layer = self.compositor.visit(boundary, policy, frame);
        let key = LayerKey::untiled(boundary);
        // A frame's invalidation is what *this* frame made stale, so a layer
        // that already existed starts the frame clean and accumulates only what
        // this frame raises: `LayerTable::set_transform` below, and
        // `patch::apply`'s own axis derivation once the patch lands. A layer
        // seen for the first time keeps the fully-invalid state it was created
        // in, because nothing about it is resident yet.
        let existing = scene.layers.contains(LayerId::from_key(key));
        scene.layer(key);
        if existing {
            scene.layers.mark_clean(layer);
        }
        let moved = self.compositor.set_transform(boundary, transform);
        let entry = boundaries.entry(boundary).or_insert(BoundaryFrame {
            layer,
            content_dirty: false,
            primitive_count: 0,
            transform_moved: false,
        });
        entry.transform_moved |= moved;
        layer
    }

    fn infer_clean_scrolls(&self, plan: &FramePlan, signals: &FrameSignals) -> FrameSignals {
        if !signals.is_empty() {
            return signals.clone();
        }
        let mut inferred = signals.clone();
        for (index, node) in plan.nodes().iter().enumerate() {
            let Some(boundary) = node.declared_boundary else {
                continue;
            };
            if !node.skipped_prepaint_and_paint()
                || !Self::subtree_is_clean(plan, index, node.depth)
            {
                continue;
            }
            let previous = self.compositor.transform(boundary).translation;
            if previous != node.scroll_offset {
                inferred.scrolled(LayerId::from_key(LayerKey::untiled(boundary)));
            }
        }
        inferred
    }

    fn subtree_is_clean(plan: &FramePlan, root_index: usize, root_depth: u32) -> bool {
        plan.nodes()
            .get(root_index..)
            .and_then(|nodes| nodes.first().map(|first| (nodes, first)))
            .is_some_and(|(nodes, first)| {
                nodes
                    .iter()
                    .take_while(|node| node.depth > root_depth || std::ptr::eq(*node, first))
                    .all(|node| node.skipped_prepaint_and_paint())
            })
    }

    fn account(
        boundaries: &mut HashMap<BoundaryId, BoundaryFrame>,
        boundary: BoundaryId,
        dirty: bool,
        primitive_count: usize,
    ) {
        if let Some(entry) = boundaries.get_mut(&boundary) {
            entry.content_dirty |= dirty;
            entry.primitive_count += primitive_count;
        }
    }

    fn record_visited(&mut self, address: InstanceKey, mut record: EmittedNode, frame: u64) {
        record.last_visited_frame = frame;
        self.emitted.insert(address, record);
    }

    /// Turn one element's emitted values for one kind into insert/update/remove
    /// operations against what it had resident.
    ///
    /// A record's cross-frame address is `(element, ordinal within this kind)`,
    /// so the `n`-th quad an element emits is the same record every frame and
    /// takes §5.0's O(1) in-place update. The two kinds' ordinals are
    /// independent, which is safe because a [`RecordKey`] is only ever unique
    /// within one kind's store.
    fn reconcile_records<P: Primitive>(
        address: InstanceKey,
        layer: LayerId,
        previous: Option<(LayerId, u32)>,
        values: &[P],
        operations: &mut KindOperations<P>,
    ) {
        let carried = match previous {
            Some((previous_layer, count)) if previous_layer == layer => count,
            Some((previous_layer, count)) => {
                for ordinal in 0..count {
                    operations
                        .removes
                        .push((previous_layer, RecordKey::new(address, ordinal)));
                }
                0
            }
            None => 0,
        };

        for (ordinal, value) in values.iter().enumerate() {
            let ordinal = u32::try_from(ordinal).unwrap_or(u32::MAX);
            let key = RecordKey::new(address, ordinal);
            if ordinal < carried {
                operations.updates.push((layer, key, value.clone()));
            } else {
                operations.inserts.push((layer, key, value.clone()));
            }
        }

        let emitted = u32::try_from(values.len()).unwrap_or(u32::MAX);
        for ordinal in emitted..carried {
            operations
                .removes
                .push((layer, RecordKey::new(address, ordinal)));
        }
    }

    fn retire_records(address: InstanceKey, record: EmittedNode, pending: &mut PendingOperations) {
        for ordinal in 0..record.shadows {
            pending
                .shadows
                .removes
                .push((record.layer, RecordKey::new(address, ordinal)));
        }
        for ordinal in 0..record.quads {
            pending
                .quads
                .removes
                .push((record.layer, RecordKey::new(address, ordinal)));
        }
        for ordinal in 0..record.underlines {
            pending
                .underlines
                .removes
                .push((record.layer, RecordKey::new(address, ordinal)));
        }
        for ordinal in 0..record.glyph_runs {
            pending
                .glyph_runs
                .removes
                .push((record.layer, RecordKey::new(address, ordinal)));
        }
        for ordinal in 0..record.poly_sprites {
            pending
                .poly_sprites
                .removes
                .push((record.layer, RecordKey::new(address, ordinal)));
        }
        for ordinal in 0..record.paths {
            pending
                .paths
                .removes
                .push((record.layer, RecordKey::new(address, ordinal)));
        }
        for ordinal in 0..record.backdrop_filters {
            pending
                .backdrop_filters
                .removes
                .push((record.layer, RecordKey::new(address, ordinal)));
        }
    }

    /// Drop the records of every element that left the tree.
    ///
    /// The counterpart of the reconciler's own instance sweep, and it has to be
    /// separate from it: an element inside an `.uncached()` subtree has no
    /// retained instance to sweep but does have resident primitives (§4.2 —
    /// "an `.uncached()` subtree still emits ordinary patches every frame"), so
    /// residency is bounded by this walk rather than by the instance table's.
    fn sweep_departed(
        &mut self,
        frame: u64,
        pending: &mut PendingOperations,
        boundaries: &mut HashMap<BoundaryId, BoundaryFrame>,
        damage: &mut Vec<Rect>,
    ) {
        let departed: Vec<(InstanceKey, EmittedNode)> = self
            .emitted
            .iter()
            .filter(|(_, record)| record.last_visited_frame != frame)
            .map(|(address, record)| (*address, *record))
            .collect();
        for (address, record) in departed {
            damage.push(record.visible_bounds);
            Self::retire_records(address, record, pending);
            self.emitted.remove(&address);
            for entry in boundaries.values_mut() {
                if entry.layer == record.layer {
                    entry.content_dirty = true;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::policy::{Buffering, Retention, Size};
    use crate::invalidation::axes::Invalidation;
    use crate::patch::apply::apply;
    use crate::patch::primitive::{Glyph, PathVertex};
    use crate::patch::{PatchError, PatchOp};
    use crate::reconcile::description::Description;
    use crate::reconcile::diff_key::{ReconcileKey, compare_by_equality};
    use crate::reconcile::plan::PlannedNode;
    use crate::reconcile::reconciler::{ReconcileError, Reconciler};
    use std::any::Any;
    use wgpui_layout::taffy_tree::{Dimension, FlexDirection, LayoutSize, LayoutStyle, definite};

    struct Panel;

    const VIEWPORT_WIDTH: f32 = 400.0;
    const VIEWPORT_HEIGHT: f32 = 300.0;
    const ROW_HEIGHT: f32 = 20.0;
    /// Enough rows that the boundary is over `rasterize_above` and therefore
    /// texture-retained, so gate #1 exercises the "independent GPU texture
    /// retention" half of §8's Phase 2 row rather than only the transform half.
    const ROW_COUNT: u32 = 300;

    #[derive(PartialEq, Debug)]
    struct Fingerprint(u32);

    impl ReconcileKey for Fingerprint {
        fn compare(&self, previous: &dyn ReconcileKey) -> Invalidation {
            compare_by_equality(self, previous, Invalidation::DISPLAY)
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    fn column(width: f32, height: f32) -> LayoutStyle {
        LayoutStyle {
            flex_direction: FlexDirection::Column,
            size: LayoutSize {
                width: Dimension::length(width),
                height: Dimension::length(height),
            },
            ..LayoutStyle::default()
        }
    }

    fn fixed(width: f32, height: f32) -> LayoutStyle {
        LayoutStyle {
            size: LayoutSize {
                width: Dimension::length(width),
                height: Dimension::length(height),
            },
            flex_shrink: 0.0,
            ..LayoutStyle::default()
        }
    }

    /// A quad the size of the element that emitted it, so a test can read an
    /// element's resolved position straight out of the scene.
    fn fill(tint: f32) -> impl Emit {
        move |context: &EmitContext, emission: &mut Emission| {
            emission.quad(Quad {
                origin: [context.bounds.x, context.bounds.y],
                size: [context.bounds.width, context.bounds.height],
                background: [tint, tint, tint, 1.0],
                ..Quad::ZERO
            });
        }
    }

    /// One scroll container holding [`ROW_COUNT`] rows, with `.boundary()`
    /// applied or not and nothing else different between the two.
    ///
    /// Nothing here names an element, keys a layer, tunes a policy, or declares
    /// an invalidation — SFD §0.1's five-call opt-in reduced to one call or
    /// zero, which is what §8's Phase 2 gate means by "no other API touched."
    fn scroller(boundaried: bool, offset: f32, revision: u32) -> Description {
        let rows = (0..ROW_COUNT).map(|row| {
            Description::new::<Panel>()
                .diff_key(Fingerprint(revision))
                .style(fixed(VIEWPORT_WIDTH, ROW_HEIGHT))
                .emit(fill(row as f32 / ROW_COUNT as f32))
        });
        let container = Description::new::<Panel>()
            .diff_key(Fingerprint(revision))
            .style(column(VIEWPORT_WIDTH, VIEWPORT_HEIGHT))
            .scroll_offset([0.0, offset])
            .emit(fill(0.0))
            .children(rows);
        let container = if boundaried {
            container.boundary()
        } else {
            container
        };
        Description::new::<Panel>()
            .diff_key(Fingerprint(revision))
            .style(column(VIEWPORT_WIDTH, VIEWPORT_HEIGHT))
            .child(container)
    }

    #[test]
    fn tiled_boundary_reports_damage_tiles_without_promoting_them_to_layers() -> Result<(), FrameError> {
        let policy = BoundaryPolicy {
            buffering: Buffering::Tiled {
                tile_size: Size::pixels(100.0, 100.0),
                retain_radius: 0,
            },
            ..BoundaryPolicy::default()
        };
        let content = Description::new::<Panel>()
            .diff_key(Fingerprint(1))
            .style(column(VIEWPORT_WIDTH, VIEWPORT_HEIGHT))
            .emit(fill(0.25))
            .children((0..4).map(|row| {
                Description::new::<Panel>()
                    .diff_key(Fingerprint(row))
                    .style(fixed(80.0, 80.0))
                    .emit(fill(0.5))
            }))
            .boundary_with_policy(policy);
        let description = Description::new::<Panel>()
            .diff_key(Fingerprint(1))
            .style(column(VIEWPORT_WIDTH, VIEWPORT_HEIGHT))
            .child(content);
        let mut window = Window::new();
        let frame = window.draw(description, &FrameSignals::new())?;
        let boundary = boundary_of(&window);
        assert_eq!(frame.emission.tiled_visits.len(), 1);
        assert_eq!(frame.emission.tiled_visits[0].boundary, boundary);
        assert!(!frame.emission.tiled_visits[0].visible.is_empty());
        assert!(frame.emission.stats.records_inserted > 0);
        assert!(!window.scene.layers.ids().iter().any(|layer| {
            window
                .scene
                .layers
                .get(*layer)
                .is_some_and(|layer| layer.key().tile.is_some())
        }));
        Ok(())
    }

    /// Everything one window holds across frames, so a test drives real frames
    /// rather than asserting on intermediate values.
    struct Window {
        reconciler: Reconciler,
        layout: LayoutTree,
        emitter: Emitter,
        scene: Scene,
    }

    #[derive(Debug)]
    enum FrameError {
        Reconcile(ReconcileError),
        Emit(EmitError),
        Patch(PatchError),
    }

    impl From<ReconcileError> for FrameError {
        fn from(error: ReconcileError) -> Self {
            FrameError::Reconcile(error)
        }
    }

    impl From<EmitError> for FrameError {
        fn from(error: EmitError) -> Self {
            FrameError::Emit(error)
        }
    }

    impl From<PatchError> for FrameError {
        fn from(error: PatchError) -> Self {
            FrameError::Patch(error)
        }
    }

    /// What one driven frame produced, kept together so a gate can assert on
    /// the reconciliation and the compositing halves side by side — which is
    /// the entire point of gate #2.
    struct Frame {
        reconciled: crate::reconcile::plan::FrameStats,
        layout_nodes: Vec<wgpui_layout::taffy_tree::LayoutNodeId>,
        fully_reused: bool,
        emission: FrameEmission,
        uploaded_bytes: u64,
        upload_calls: usize,
    }

    impl Window {
        fn new() -> Self {
            Self {
                reconciler: Reconciler::new(),
                layout: LayoutTree::new(),
                emitter: Emitter::new(),
                scene: Scene::new(),
            }
        }

        fn draw(
            &mut self,
            description: Description,
            signals: &FrameSignals,
        ) -> Result<Frame, FrameError> {
            let plan = self.reconciler.reconcile(description, &mut self.layout)?;
            let root = plan
                .root()
                .map(|node| node.layout_node)
                .ok_or(EmitError::MalformedPlan { index: 0, depth: 0 })?;
            self.layout
                .compute_layout(root, definite(VIEWPORT_WIDTH, VIEWPORT_HEIGHT))
                .map_err(EmitError::from)?;
            let emission = self
                .emitter
                .emit(&plan, &self.layout, signals, &mut self.scene)?;
            let uploads = apply(&mut self.scene, &emission.patch)?;
            Ok(Frame {
                reconciled: plan.stats(),
                layout_nodes: plan.nodes().iter().map(|node| node.layout_node).collect(),
                fully_reused: plan.fully_reused(),
                emission,
                uploaded_bytes: uploads.byte_count(),
                upload_calls: uploads.len(),
            })
        }
    }

    fn boundary_of(window: &Window) -> BoundaryId {
        // The scroll container is the root's first child, and neither names
        // itself, so its identity is `[Slot(0), Slot(0)]` — SFD §1.0's
        // positional fallback, resolved the same way a test outside the crate
        // would have to resolve it.
        use crate::boundary::identity::BoundaryIdentity;
        use crate::reconcile::description::ElementId;
        let _ = window;
        BoundaryIdentity::from_path(&[ElementId::Slot(0), ElementId::Slot(0)])
    }

    /// Recursively confirm nothing in the tree names itself, keys a layer, or
    /// opts out of anything — the "with no other API touched" clause of §8's
    /// Phase 2 gate, checked mechanically rather than by reading the helper.
    fn assert_only_boundary_is_touched(description: &Description, boundaried: bool) {
        assert!(
            description.element_id().is_none(),
            "the gate requires a tree with no explicit id anywhere"
        );
        assert!(!description.is_uncached());
        if let Some(policy) = description.boundary_policy() {
            assert!(
                boundaried,
                "no boundary may appear in the unboundaried tree"
            );
            assert_eq!(
                policy,
                BoundaryPolicy::default(),
                "`.boundary()` must be reachable with zero policy arguments"
            );
        }
        for child in description.child_descriptions() {
            assert_only_boundary_is_touched(child, boundaried);
        }
    }

    /// **Phase 2 gate #1** (§4.1, §5.4, §8): `.boundary()` with zero policy
    /// arguments reaches the fast path — a plain scroll container recomposites
    /// transform-only on a scroll-reason notification, with no other API
    /// touched.
    #[test]
    fn gate_1_a_bare_boundary_recomposites_a_scroll_transform_only() -> Result<(), FrameError> {
        let mut window = Window::new();
        let boundary = boundary_of(&window);
        let layer = LayerId::from_key(LayerKey::untiled(boundary));

        let first = scroller(true, 0.0, 0);
        assert_only_boundary_is_touched(&first, true);
        let built = window.draw(first, &FrameSignals::new())?;
        assert_eq!(
            built.emission.stats.records_inserted,
            ROW_COUNT as usize + 1,
            "one quad per row, plus the container's own background"
        );

        // A settled frame: nothing changed, nothing was signalled.
        let idle = window.draw(scroller(true, 0.0, 0), &FrameSignals::new())?;
        assert!(idle.fully_reused);
        assert!(idle.emission.patch.is_empty());
        assert_eq!(idle.uploaded_bytes, 0);
        assert_eq!(
            idle.emission.composite_for(boundary).map(|c| c.composite),
            Some(Composite::Clean)
        );
        assert_eq!(
            window
                .scene
                .layers
                .get(layer)
                .map(|layer| layer.invalidation()),
            Some(Invalidation::empty()),
            "a settled frame leaves nothing stale — R-N §3.2's clean layer"
        );

        // The scroll tick itself: one tagged notification, one changed offset.
        let mut signals = FrameSignals::new();
        signals.scrolled(layer);
        let scrolled = window.draw(scroller(true, -ROW_HEIGHT * 3.0, 0), &signals)?;

        assert!(
            scrolled.fully_reused,
            "ambient reconciliation must still find the whole tree clean"
        );
        let composite = scrolled
            .emission
            .composite_for(boundary)
            .ok_or(EmitError::MalformedPlan { index: 0, depth: 0 })?;
        assert_eq!(
            composite.composite,
            Composite::TransformOnly,
            "a bare `.boundary()` must reach R-N's fast path with no tuning"
        );
        assert_eq!(composite.invalidation, Invalidation::TRANSFORM);
        assert_eq!(
            composite.transform,
            LayerTransform::translated(0.0, -ROW_HEIGHT * 3.0),
            "the whole cost of the tick is one changed transform"
        );
        assert_eq!(
            composite.retention,
            Retention::Texture,
            "a 300-row boundary is over `rasterize_above`, so it retains independently"
        );

        // The observable consequence, which is what makes this a gate rather
        // than an assertion about the decision that produced it.
        assert!(
            scrolled.emission.patch.is_empty(),
            "transform-only means no content patch at all, not a small one"
        );
        assert_eq!(scrolled.uploaded_bytes, 0);
        assert_eq!(scrolled.upload_calls, 0);
        assert_eq!(scrolled.emission.stats.nodes_emitted, 0);
        assert_eq!(
            scrolled.emission.stats.nodes_skipped,
            ROW_COUNT as usize + 1,
            "every element that emits anything skipped emitting"
        );
        assert_eq!(scrolled.emission.stats.transform_only, 1);
        assert_eq!(
            scrolled.emission.damage,
            vec![Rect::from_origin_size([0.0, 0.0], [VIEWPORT_WIDTH, VIEWPORT_HEIGHT])],
            "a transform-only scroll damages only the scrolling boundary's viewport"
        );

        // And the scene agrees: the layer moved, its residency did not.
        assert_eq!(
            window
                .scene
                .layers
                .get(layer)
                .map(|layer| layer.transform()),
            Some(LayerTransform::translated(0.0, -ROW_HEIGHT * 3.0))
        );
        assert_eq!(
            window
                .scene
                .layers
                .get(layer)
                .map(|layer| layer.invalidation()),
            Some(Invalidation::TRANSFORM)
        );
        assert_eq!(window.scene.quads.len(layer), ROW_COUNT);
        Ok(())
    }

    #[test]
    fn a_clean_scroll_is_inferred_when_the_native_input_path_has_no_signal() -> Result<(), FrameError> {
        let mut window = Window::new();
        let boundary = boundary_of(&window);
        window.draw(scroller(true, 0.0, 0), &FrameSignals::new())?;
        let frame = window.draw(scroller(true, -ROW_HEIGHT, 0), &FrameSignals::new())?;
        assert!(frame.emission.patch.is_empty());
        assert_eq!(
            frame.emission.composite_for(boundary).map(|composite| composite.composite),
            Some(Composite::TransformOnly)
        );
        Ok(())
    }

    /// **Phase 2 gate #2** (§4.0, §4.1, §8): removing `.boundary()` from gate
    /// #1's case degrades the scroll to a per-tick recomposite but does **not**
    /// reintroduce a full rebuild.
    ///
    /// This is the load-bearing one. Phase 1's whole claim was that
    /// reconciliation is ambient and owes nothing to any boundary; this is
    /// where that gets proved *under* a boundary, by taking the boundary away
    /// and showing the reconciler's answer is bit-for-bit the same one.
    #[test]
    fn gate_2_removing_the_boundary_costs_a_recomposite_and_not_a_rebuild() -> Result<(), FrameError>
    {
        let mut boundaried = Window::new();
        let mut plain = Window::new();
        let boundary = boundary_of(&boundaried);
        let boundary_layer = LayerId::from_key(LayerKey::untiled(boundary));
        let root_layer = LayerId::from_key(LayerKey::untiled(BoundaryId::ROOT));

        let plain_description = scroller(false, 0.0, 0);
        assert_only_boundary_is_touched(&plain_description, false);
        boundaried.draw(scroller(true, 0.0, 0), &FrameSignals::new())?;
        plain.draw(plain_description, &FrameSignals::new())?;

        let mut boundaried_signals = FrameSignals::new();
        boundaried_signals.scrolled(boundary_layer);
        let mut plain_signals = FrameSignals::new();
        plain_signals.scrolled(root_layer);

        let offset = -ROW_HEIGHT * 3.0;
        let with = boundaried.draw(scroller(true, offset, 0), &boundaried_signals)?;
        let without = plain.draw(scroller(false, offset, 0), &plain_signals)?;

        // 1. Reconciliation is identical. Not "similar", not "also fast" —
        //    the same numbers and the same retained layout-node identities.
        assert_eq!(
            with.reconciled, without.reconciled,
            "reconciliation must not be able to tell whether a boundary exists"
        );
        assert_eq!(with.layout_nodes, without.layout_nodes);
        assert!(with.fully_reused && without.fully_reused);
        assert_eq!(without.reconciled.rebuilt, 0, "no element rebuilt");
        assert_eq!(
            without.reconciled.layout_nodes_created, 0,
            "no node recreated"
        );
        assert_eq!(without.reconciled.layout_nodes_swept, 0);
        assert_eq!(without.reconciled.instances_swept, 0);
        assert_eq!(
            plain.reconciler.instances().len(),
            boundaried.reconciler.instances().len()
        );

        // 2. Compositing is not. The unboundaried container folds its offset
        //    into its children, so every visible row moves and re-emits.
        assert!(
            with.emission.patch.is_empty(),
            "the boundaried case is the control: it emits nothing"
        );
        assert!(!without.emission.patch.is_empty());
        assert_eq!(without.emission.stats.nodes_emitted, ROW_COUNT as usize);
        assert!(without.uploaded_bytes > 0);
        assert_eq!(
            without
                .emission
                .composite_for(BoundaryId::ROOT)
                .map(|c| c.composite),
            Some(Composite::Redisplay)
        );

        // 3. It is a recomposite and *not* a rebuild: every operation is a
        //    value update in place. No record was inserted, removed, or
        //    relocated, so §5.0's O(1) path carried the whole tick.
        assert_eq!(without.emission.stats.records_updated, ROW_COUNT as usize);
        assert_eq!(without.emission.stats.records_inserted, 0);
        assert_eq!(without.emission.stats.records_removed, 0);
        assert!(
            without
                .emission
                .patch
                .quads
                .patches()
                .iter()
                .all(|patch| matches!(patch.op, PatchOp::Update { .. })),
            "a degraded scroll must stay a value update, never an insert/remove churn"
        );
        assert_eq!(
            without.uploaded_bytes,
            ROW_COUNT as u64 * Quad::SLOT_STRIDE as u64,
            "exactly the moved rows' bytes, and nothing wider"
        );

        // 4. The container's own background did not move in either case: a
        //    scroll container's own paint does not scroll with its contents.
        assert_eq!(plain.scene.quads.len(root_layer), ROW_COUNT + 1);
        assert_eq!(boundaried.scene.quads.len(root_layer), 1);
        assert_eq!(boundaried.scene.quads.len(boundary_layer), ROW_COUNT);
        Ok(())
    }

    #[test]
    fn a_boundary_that_was_not_told_this_was_a_scroll_folds_the_offset_in() -> Result<(), FrameError>
    {
        let mut window = Window::new();
        let boundary = boundary_of(&window);
        window.draw(scroller(true, 0.0, 0), &FrameSignals::new())?;

        let mut signals = FrameSignals::new();
        signals.data_changed(LayerId::from_key(LayerKey::untiled(boundary)));
        let frame = window.draw(scroller(true, -ROW_HEIGHT, 0), &signals)?;

        assert!(frame.fully_reused, "the content itself is still clean");
        assert_eq!(
            frame.emission.composite_for(boundary).map(|c| c.composite),
            Some(Composite::Redisplay),
            "the fast path needs the signal as well as the measurement"
        );
        assert_eq!(
            frame.emission.composite_for(boundary).map(|c| c.transform),
            Some(LayerTransform::IDENTITY),
            "a boundary refused the fast path must not leave a stale transform behind"
        );
        assert!(frame.uploaded_bytes > 0);
        Ok(())
    }

    #[test]
    fn a_content_change_inside_a_scrolling_boundary_redisplays_it() -> Result<(), FrameError> {
        let mut window = Window::new();
        let boundary = boundary_of(&window);
        let layer = LayerId::from_key(LayerKey::untiled(boundary));
        window.draw(scroller(true, 0.0, 0), &FrameSignals::new())?;

        let mut signals = FrameSignals::new();
        signals.scrolled(layer);
        let frame = window.draw(scroller(true, -ROW_HEIGHT, 1), &signals)?;

        assert!(!frame.fully_reused);
        assert_eq!(
            frame.emission.composite_for(boundary).map(|c| c.composite),
            Some(Composite::Redisplay),
            "a scroll signal must never override a measured-dirty subtree"
        );
        assert_eq!(
            frame
                .emission
                .composite_for(boundary)
                .map(|c| c.invalidation),
            Some(Invalidation::DISPLAY)
        );
        Ok(())
    }

    #[test]
    fn a_boundary_holding_little_stays_primitive_retained() -> Result<(), FrameError> {
        let mut window = Window::new();
        let leaf = || {
            Description::new::<Panel>()
                .diff_key(Fingerprint(0))
                .style(fixed(10.0, 10.0))
                .emit(fill(0.5))
        };
        let description = Description::new::<Panel>()
            .diff_key(Fingerprint(0))
            .style(column(VIEWPORT_WIDTH, VIEWPORT_HEIGHT))
            .child(
                Description::new::<Panel>()
                    .diff_key(Fingerprint(0))
                    .style(column(100.0, 100.0))
                    .boundary()
                    .child(leaf())
                    .child(leaf()),
            );
        let frame = window.draw(description, &FrameSignals::new())?;
        let boundary = boundary_of(&window);
        assert_eq!(
            frame.emission.composite_for(boundary).map(|c| c.retention),
            Some(Retention::Primitives),
            "R-N §3.3: a boundary holding two quads is cheaper to re-emit than to composite"
        );
        Ok(())
    }

    #[test]
    fn an_element_that_stops_emitting_takes_its_records_with_it() -> Result<(), FrameError> {
        let mut window = Window::new();
        let root_layer = LayerId::from_key(LayerKey::untiled(BoundaryId::ROOT));
        let describe = |emits: bool| {
            let leaf = Description::new::<Panel>()
                .diff_key(Fingerprint(emits as u32))
                .style(fixed(10.0, 10.0));
            let leaf = if emits { leaf.emit(fill(0.25)) } else { leaf };
            Description::new::<Panel>()
                .diff_key(Fingerprint(0))
                .style(column(VIEWPORT_WIDTH, VIEWPORT_HEIGHT))
                .child(leaf)
        };
        window.draw(describe(true), &FrameSignals::new())?;
        assert_eq!(window.scene.quads.len(root_layer), 1);
        assert_eq!(window.emitter.emitting_element_count(), 1);

        window.draw(describe(false), &FrameSignals::new())?;
        assert_eq!(window.scene.quads.len(root_layer), 0);
        assert_eq!(window.emitter.emitting_element_count(), 0);
        Ok(())
    }

    #[test]
    fn an_element_that_leaves_the_tree_takes_its_records_with_it() -> Result<(), FrameError> {
        let mut window = Window::new();
        let root_layer = LayerId::from_key(LayerKey::untiled(BoundaryId::ROOT));
        let describe = |leaves: u32| {
            Description::new::<Panel>()
                .diff_key(Fingerprint(leaves))
                .style(column(VIEWPORT_WIDTH, VIEWPORT_HEIGHT))
                .children((0..leaves).map(|index| {
                    Description::new::<Panel>()
                        .diff_key(Fingerprint(index))
                        .style(fixed(10.0, 10.0))
                        .emit(fill(0.5))
                }))
        };
        window.draw(describe(4), &FrameSignals::new())?;
        assert_eq!(window.scene.quads.len(root_layer), 4);

        let shrunk = window.draw(describe(1), &FrameSignals::new())?;
        assert_eq!(window.scene.quads.len(root_layer), 1);
        assert_eq!(shrunk.emission.stats.records_removed, 3);
        Ok(())
    }

    #[test]
    fn an_evicted_boundary_stops_costing_the_scene_a_layer() -> Result<(), FrameError> {
        let mut window = Window::new();
        let boundary = boundary_of(&window);
        let layer = LayerId::from_key(LayerKey::untiled(boundary));
        let root_layer = LayerId::from_key(LayerKey::untiled(BoundaryId::ROOT));
        let policy = BoundaryPolicy {
            evict_after_frames: 2,
            ..BoundaryPolicy::default()
        };
        let describe = move |present: bool| {
            let root = Description::new::<Panel>()
                .diff_key(Fingerprint(present as u32))
                .style(column(VIEWPORT_WIDTH, VIEWPORT_HEIGHT));
            if !present {
                return root;
            }
            root.child(
                Description::new::<Panel>()
                    .diff_key(Fingerprint(0))
                    .style(column(100.0, 100.0))
                    .boundary_with_policy(policy)
                    .child(
                        Description::new::<Panel>()
                            .diff_key(Fingerprint(0))
                            .style(fixed(10.0, 10.0))
                            .emit(fill(0.5)),
                    ),
            )
        };

        window.draw(describe(true), &FrameSignals::new())?;
        assert!(window.scene.layers.contains(layer));
        assert_eq!(window.scene.quads.len(layer), 1);

        // Residency goes with the elements, immediately.
        window.draw(describe(false), &FrameSignals::new())?;
        assert_eq!(window.scene.quads.len(layer), 0);
        assert!(
            window.scene.layers.contains(layer),
            "the layer record outlives the elements by the eviction interval, \
             so a panel that comes straight back keeps where it was"
        );

        // The layer record goes with the boundary, after it.
        for _ in 0..4 {
            window.draw(describe(false), &FrameSignals::new())?;
        }
        assert!(window.emitter.compositor().state(boundary).is_none());
        assert!(
            !window.scene.layers.contains(layer),
            "an evicted boundary must not leave a layer behind for the scene to carry forever"
        );
        assert!(window.scene.layers.contains(root_layer));
        Ok(())
    }

    #[test]
    fn an_uncached_subtree_still_emits_every_frame() -> Result<(), FrameError> {
        let mut window = Window::new();
        let root_layer = LayerId::from_key(LayerKey::untiled(BoundaryId::ROOT));
        let describe = || {
            Description::new::<Panel>()
                .diff_key(Fingerprint(0))
                .style(column(VIEWPORT_WIDTH, VIEWPORT_HEIGHT))
                .child(
                    Description::new::<Panel>()
                        .uncached()
                        .style(column(100.0, 100.0))
                        .child(
                            Description::new::<Panel>()
                                .style(fixed(10.0, 10.0))
                                .emit(fill(0.75)),
                        ),
                )
        };
        window.draw(describe(), &FrameSignals::new())?;
        assert_eq!(window.scene.quads.len(root_layer), 1);

        // §4.2: "an `.uncached()` subtree still emits ordinary patches every
        // frame (always a full replace, never a delta)" — and its residency is
        // bounded by the emitter's own sweep, since it has no instance record
        // for the reconciler's sweep to reach.
        let second = window.draw(describe(), &FrameSignals::new())?;
        assert_eq!(second.emission.stats.nodes_emitted, 1);
        assert_eq!(second.emission.stats.records_updated, 1);
        assert_eq!(window.scene.quads.len(root_layer), 1);
        assert_eq!(window.emitter.emitting_element_count(), 1);
        Ok(())
    }

    #[test]
    fn an_element_emitting_several_primitives_keeps_each_ones_address() -> Result<(), FrameError> {
        let mut window = Window::new();
        let root_layer = LayerId::from_key(LayerKey::untiled(BoundaryId::ROOT));
        let describe = |tint: f32, revision: u32| {
            Description::new::<Panel>()
                .diff_key(Fingerprint(revision))
                .style(column(VIEWPORT_WIDTH, VIEWPORT_HEIGHT))
                .child(
                    Description::new::<Panel>()
                        .diff_key(Fingerprint(revision))
                        .style(fixed(10.0, 10.0))
                        .emit(move |context: &EmitContext, emission: &mut Emission| {
                            emission
                                .quad(Quad {
                                    background: [tint, 0.0, 0.0, 1.0],
                                    size: [context.bounds.width, context.bounds.height],
                                    ..Quad::ZERO
                                })
                                .quad(Quad {
                                    border_color: [0.0, tint, 0.0, 1.0],
                                    ..Quad::ZERO
                                })
                                .glyph_run(GlyphRun::empty([tint, tint, tint, 1.0]));
                        }),
                )
        };
        window.draw(describe(0.25, 0), &FrameSignals::new())?;
        let keys = window.scene.quads.keys(root_layer);
        assert_eq!(keys.len(), 2);

        let second = window.draw(describe(0.5, 1), &FrameSignals::new())?;
        assert_eq!(
            second.emission.stats.records_updated, 3,
            "two quads and one glyph run, each addressed by its own ordinal"
        );
        assert_eq!(second.emission.stats.records_inserted, 0);
        assert_eq!(second.emission.patch.glyph_runs.len(), 1);
        assert_eq!(
            window.scene.quads.keys(root_layer),
            keys,
            "a stable emission order means stable per-primitive addresses (§5.0)"
        );
        Ok(())
    }

    #[test]
    fn an_emitter_is_never_run_for_an_element_that_did_not_move_or_change() -> Result<(), FrameError>
    {
        use std::cell::Cell;
        use std::rc::Rc;

        let mut window = Window::new();
        let calls = Rc::new(Cell::new(0u32));
        let describe = |revision: u32, calls: Rc<Cell<u32>>| {
            Description::new::<Panel>()
                .diff_key(Fingerprint(0))
                .style(column(VIEWPORT_WIDTH, VIEWPORT_HEIGHT))
                .child(
                    Description::new::<Panel>()
                        .diff_key(Fingerprint(revision))
                        .style(fixed(10.0, 10.0))
                        .emit(move |context: &EmitContext, emission: &mut Emission| {
                            calls.set(calls.get() + 1);
                            emission.quad(Quad {
                                size: [context.bounds.width, context.bounds.height],
                                ..Quad::ZERO
                            });
                        }),
                )
        };
        window.draw(describe(0, Rc::clone(&calls)), &FrameSignals::new())?;
        assert_eq!(calls.get(), 1);
        window.draw(describe(0, Rc::clone(&calls)), &FrameSignals::new())?;
        assert_eq!(
            calls.get(),
            1,
            "a clean, unmoved element must not be asked again"
        );
        window.draw(describe(1, Rc::clone(&calls)), &FrameSignals::new())?;
        assert_eq!(calls.get(), 2);
        Ok(())
    }

    #[test]
    fn clipping_crops_text_and_sprite_atlas_coordinates_instead_of_leaking() {
        let mut emission = Emission::default();
        emission.glyph_runs.push(GlyphRun {
            color: [1.0; 4],
            glyphs: vec![Glyph {
                position: [8.0, 8.0],
                atlas_origin: [40.0, 80.0],
                atlas_size: [16.0, 12.0],
                glyph_id: 7,
                atlas_tile: AtlasTileId::new(2, 3).expect("test tile id is representable"),
            }],
        });
        emission.poly_sprites.push(PolySprite {
            origin: [8.0, 8.0],
            size: [16.0, 12.0],
            atlas_origin: [100.0, 200.0],
            atlas_size: [32.0, 24.0],
            ..PolySprite::ZERO
        });

        emission.clip_to(LayoutRect {
            x: 12.0,
            y: 10.0,
            width: 8.0,
            height: 8.0,
        });

        let glyph = &emission.glyph_runs[0].glyphs[0];
        assert_eq!(glyph.position, [12.0, 10.0]);
        assert_eq!(glyph.atlas_origin, [44.0, 82.0]);
        assert_eq!(glyph.atlas_size, [8.0, 8.0]);
        assert!(!glyph.atlas_tile.is_none());

        let sprite = emission.poly_sprites[0];
        assert_eq!(sprite.origin, [12.0, 10.0]);
        assert_eq!(sprite.size, [8.0, 8.0]);
        assert_eq!(sprite.atlas_origin, [108.0, 204.0]);
        assert_eq!(sprite.atlas_size, [16.0, 16.0]);
    }

    #[test]
    fn clipping_intersects_existing_path_and_backdrop_masks() {
        let mut emission = Emission::default();
        emission.paths.push(Path::new(
            vec![
                PathVertex {
                    position: [0.0, 0.0],
                    st: [0.0, 0.0],
                },
                PathVertex {
                    position: [20.0, 0.0],
                    st: [1.0, 0.0],
                },
                PathVertex {
                    position: [0.0, 20.0],
                    st: [0.0, 1.0],
                },
            ],
            [1.0; 4],
        ));
        emission.backdrop_filters.push(BackdropFilter {
            origin: [0.0, 0.0],
            size: [20.0, 20.0],
            clip_origin: [2.0, 2.0],
            clip_size: [16.0, 16.0],
            ..BackdropFilter::ZERO
        });

        emission.clip_to(LayoutRect {
            x: 10.0,
            y: 10.0,
            width: 20.0,
            height: 20.0,
        });

        assert_eq!(emission.paths[0].clip_origin, [10.0, 10.0]);
        assert_eq!(emission.paths[0].clip_size, [10.0, 10.0]);
        assert_eq!(emission.backdrop_filters[0].clip_origin, [10.0, 10.0]);
        assert_eq!(emission.backdrop_filters[0].clip_size, [8.0, 8.0]);
    }

    #[test]
    fn a_malformed_plan_is_reported_rather_than_walked() {
        let mut plan = FramePlan::new();
        let mut node = PlannedNode {
            address: InstanceKey::from_raw(1),
            instance: None,
            boundary: BoundaryId::ROOT,
            declared_boundary: None,
            boundary_policy: None,
            scroll_offset: [0.0, 0.0],
            clip_children: false,
            state: crate::reconcile::state::StateScope::from_path(&[]),
            layout_node: wgpui_layout::taffy_tree::LayoutNodeId::from_raw(0),
            depth: 0,
            outcome: crate::reconcile::plan::NodeOutcome::Uncached,
            invalidation: Invalidation::all(),
        };
        plan.push(node);
        node.depth = 4;
        plan.push(node);

        let mut emitter = Emitter::new();
        let mut scene = Scene::new();
        let layout = LayoutTree::new();
        assert!(matches!(
            emitter.emit(&plan, &layout, &FrameSignals::new(), &mut scene),
            Err(EmitError::MalformedPlan { index: 0, .. }) | Err(EmitError::Layout(_))
        ));
    }
}
