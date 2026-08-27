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

use crate::boundary::compositor::{BoundaryComposite, Composite, Compositor};
use crate::boundary::policy::BoundaryPolicy;
use crate::invalidation::request::FrameSignals;
use crate::patch::apply::ScenePatch;
use crate::patch::primitive::{GlyphRun, Primitive, Quad};
use crate::patch::{PatchList, RecordKey};
use crate::reconcile::instance::InstanceKey;
use crate::reconcile::plan::FramePlan;
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
    quads: Vec<Quad>,
    glyph_runs: Vec<GlyphRun>,
}

impl Emission {
    /// An emission holding nothing.
    pub fn new() -> Self {
        Self::default()
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

    /// The glyph runs contributed, in emission order.
    pub fn glyph_runs(&self) -> &[GlyphRun] {
        &self.glyph_runs
    }

    /// Total primitives contributed.
    pub fn len(&self) -> usize {
        self.quads.len() + self.glyph_runs.len()
    }

    /// Whether the element contributed nothing.
    pub fn is_empty(&self) -> bool {
        self.quads.is_empty() && self.glyph_runs.is_empty()
    }

    /// Drop everything, keeping the allocations for the next element.
    pub fn clear(&mut self) {
        self.quads.clear();
        self.glyph_runs.clear();
    }
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
    quads: u32,
    glyph_runs: u32,
    last_visited_frame: u64,
}

/// One boundary's accumulated state for the frame being walked.
#[derive(Copy, Clone, Debug)]
struct BoundaryFrame {
    layer: LayerId,
    content_dirty: bool,
    primitive_count: usize,
    transform_moved: bool,
}

/// One ancestor on the walk stack.
#[derive(Copy, Clone, Debug)]
struct WalkFrame {
    depth: u32,
    layer: LayerId,
    boundary: BoundaryId,
    origin: [f32; 2],
    /// Displacement applied to this node's children's positions. Zero unless
    /// this node scrolls and could not hand that displacement to a layer.
    content_offset: [f32; 2],
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

impl<P: Primitive> KindOperations<P> {
    /// Fold this kind's operations into one ordered [`PatchList`].
    ///
    /// Removals first, then value updates, then insertions — so an insertion
    /// index computed against the layer's post-removal length is the index the
    /// scene will actually see. `store` supplies each layer's starting length;
    /// nothing here mutates the scene.
    fn into_patch_list(
        self,
        store: &PrimitiveStore<P>,
        stats: &mut EmissionStats,
    ) -> PatchList<P> {
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

        let mut stats = EmissionStats::default();
        let mut quads: KindOperations<Quad> = KindOperations::default();
        let mut glyph_runs: KindOperations<GlyphRun> = KindOperations::default();
        let mut boundaries: HashMap<BoundaryId, BoundaryFrame> = HashMap::new();
        let mut emission = Emission::new();

        let root_layer = self.begin_boundary(
            BoundaryId::ROOT,
            BoundaryPolicy::default(),
            LayerTransform::IDENTITY,
            frame,
            scene,
            &mut boundaries,
        );
        let mut stack: Vec<WalkFrame> = Vec::new();
        let root_frame = WalkFrame {
            depth: 0,
            layer: root_layer,
            boundary: BoundaryId::ROOT,
            origin: [0.0, 0.0],
            content_offset: [0.0, 0.0],
        };

        for (index, node) in plan.nodes().iter().enumerate() {
            while stack.last().is_some_and(|frame| frame.depth >= node.depth) {
                stack.pop();
            }
            let parent = *stack.last().unwrap_or(&root_frame);
            if node.depth != u32::try_from(stack.len()).unwrap_or(u32::MAX) {
                return Err(EmitError::MalformedPlan {
                    index,
                    depth: node.depth,
                });
            }
            stats.nodes_visited += 1;

            let rectangle = layout.layout_of(node.layout_node)?;
            let origin = [
                parent.origin[0] + rectangle.x + parent.content_offset[0],
                parent.origin[1] + rectangle.y + parent.content_offset[1],
            ];
            let bounds = LayoutRect {
                x: origin[0],
                y: origin[1],
                width: rectangle.width,
                height: rectangle.height,
            };

            // A boundary root's own paint belongs to the layer around it, not to
            // the layer it declares — see `PlannedNode::boundary`.
            let layer = parent.layer;
            let (child_layer, child_boundary, content_offset) = match node.declared_boundary {
                Some(declared) => {
                    let policy = node.boundary_policy.unwrap_or_default();
                    let declared_layer = LayerId::from_key(LayerKey::untiled(declared));
                    // Decided from the signal alone, before the walk knows
                    // whether the content is clean, because it changes where
                    // the content is emitted: a boundary permitted the fast
                    // path hands its displacement to its layer, and one that is
                    // not folds it into its children exactly as an ordinary
                    // element would.
                    let slides = signals
                        .reason_for_layer(declared_layer)
                        .permits_transform_only();
                    let transform = if slides {
                        LayerTransform {
                            translation: node.scroll_offset,
                        }
                    } else {
                        LayerTransform::IDENTITY
                    };
                    let boundary_layer = self.begin_boundary(
                        declared,
                        policy,
                        transform,
                        frame,
                        scene,
                        &mut boundaries,
                    );
                    let folded = if slides {
                        [0.0, 0.0]
                    } else {
                        node.scroll_offset
                    };
                    (boundary_layer, declared, folded)
                }
                None => (parent.layer, node.boundary, node.scroll_offset),
            };

            let previous = self.emitted.get(&node.address).copied();
            match plan.emitter(index) {
                Some(emitter) => {
                    let stale = previous
                        .is_none_or(|record| record.bounds != bounds || record.layer != layer);
                    if node.skipped_prepaint_and_paint() && !stale {
                        stats.nodes_skipped += 1;
                        if let Some(record) = previous {
                            self.record_visited(node.address, record, frame);
                            Self::account(
                                &mut boundaries,
                                node.boundary,
                                false,
                                record.quads as usize + record.glyph_runs as usize,
                            );
                        }
                    } else {
                        stats.nodes_emitted += 1;
                        emission.clear();
                        emitter.emit(
                            &EmitContext {
                                bounds,
                                layer,
                                boundary: node.boundary,
                            },
                            &mut emission,
                        );
                        Self::reconcile_records(
                            node.address,
                            layer,
                            previous.map(|record| (record.layer, record.quads)),
                            emission.quads(),
                            &mut quads,
                        );
                        Self::reconcile_records(
                            node.address,
                            layer,
                            previous.map(|record| (record.layer, record.glyph_runs)),
                            emission.glyph_runs(),
                            &mut glyph_runs,
                        );
                        let emitted = EmittedNode {
                            layer,
                            bounds,
                            quads: u32::try_from(emission.quads().len()).unwrap_or(u32::MAX),
                            glyph_runs: u32::try_from(emission.glyph_runs().len())
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
                        Self::retire_records(node.address, record, &mut quads, &mut glyph_runs);
                        self.emitted.remove(&node.address);
                        Self::account(&mut boundaries, node.boundary, true, 0);
                    }
                }
            }

            stack.push(WalkFrame {
                depth: node.depth,
                layer: child_layer,
                boundary: child_boundary,
                origin,
                content_offset,
            });
        }

        self.sweep_departed(frame, &mut quads, &mut glyph_runs, &mut boundaries);

        let patch = ScenePatch {
            quads: quads.into_patch_list(&scene.quads, &mut stats),
            glyph_runs: glyph_runs.into_patch_list(&scene.glyph_runs, &mut stats),
            ..ScenePatch::new()
        };

        let mut composites = Vec::with_capacity(boundaries.len());
        for (boundary, state) in boundaries {
            let reason = signals.reason_for_layer(state.layer);
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
            }
            composites.push(composite);
        }
        composites.sort_by_key(|composite| composite.layer);
        stats.boundaries = composites.len();

        self.compositor.sweep(frame);
        Ok(FrameEmission {
            patch,
            composites,
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
        scene.layer(LayerKey::untiled(boundary));
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

    fn retire_records(
        address: InstanceKey,
        record: EmittedNode,
        quads: &mut KindOperations<Quad>,
        glyph_runs: &mut KindOperations<GlyphRun>,
    ) {
        for ordinal in 0..record.quads {
            quads
                .removes
                .push((record.layer, RecordKey::new(address, ordinal)));
        }
        for ordinal in 0..record.glyph_runs {
            glyph_runs
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
        quads: &mut KindOperations<Quad>,
        glyph_runs: &mut KindOperations<GlyphRun>,
        boundaries: &mut HashMap<BoundaryId, BoundaryFrame>,
    ) {
        let departed: Vec<(InstanceKey, EmittedNode)> = self
            .emitted
            .iter()
            .filter(|(_, record)| record.last_visited_frame != frame)
            .map(|(address, record)| (*address, *record))
            .collect();
        for (address, record) in departed {
            Self::retire_records(address, record, quads, glyph_runs);
            self.emitted.remove(&address);
            for entry in boundaries.values_mut() {
                if entry.layer == record.layer {
                    entry.content_dirty = true;
                }
            }
        }
    }
}
