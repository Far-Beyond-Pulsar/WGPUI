//! `FramePlan` — what ambient reconciliation decided, as inspectable data.
//! See docs/gpu-native-architecture.md §4.0, §2.
//!
//! Not in §3.1's literal file map — a deliberate addition, recorded in
//! `docs/phase-1-results.md`. The legacy reconciler has nothing like this
//! because it *is* the draw walk: it calls `prepaint`/`paint` inline and the
//! only evidence a skip happened is a `render_stats` counter. §2's whole
//! premise is that the frontend/backend seam is pure data, so 2.0's reconciler
//! produces a plan and the caller executes it. Two things follow that matter
//! beyond tidiness: the "did this element skip `prepaint`/`paint`" question
//! Phase 1's second gate asks becomes a value to assert on rather than an
//! effect to detect, and Phase 3's compute passes get a consumable description
//! of the frame rather than a callback to hook.

use crate::boundary::policy::BoundaryPolicy;
use crate::invalidation::axes::Invalidation;
use crate::patch::emit::Emit;
use crate::reconcile::description::{
    DescriptionInteraction, DescriptionLayout, ExternalSurfaceProperties, ScrollInfo,
};
use crate::reconcile::instance::InstanceKey;
use crate::reconcile::state::StateScope;
use crate::scene::layer::BoundaryId;
use wgpui_layout::taffy_tree::LayoutNodeId;

/// What reconciliation decided for one element.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NodeOutcome {
    /// The element's fingerprint compared clean and its children kept their
    /// layout nodes, so it keeps its retained node and skips `prepaint` and
    /// `paint` entirely.
    Reused,
    /// The element has a retained record, but something about it changed.
    Rebuilt(RebuildReason),
    /// The element sits inside an `.uncached()` subtree (§4.2): no record was
    /// allocated, no fingerprint retained, no comparison run.
    Uncached,
}

/// Why an element was rebuilt rather than reused.
///
/// Carried so a diagnostic — or a test — can distinguish "this element
/// genuinely changed" from "this element can never be reused because it has no
/// fingerprint," which look identical from the outside and mean very different
/// things about whether the framework is working.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RebuildReason {
    /// First frame this element has been seen at this address.
    NewInstance,
    /// The address held a different element type last frame. R-N §2.2's rule:
    /// a mismatch causes a subtree rebuild — one slow frame, never incorrect
    /// output.
    TypeMismatch,
    /// The element supplied no fingerprint, so "assume changed" is the only
    /// correct answer (R-N §2.3's permissive default).
    NoDiffKey,
    /// The fingerprint compared different.
    KeyChanged,
    /// The fingerprint compared clean, but a child had to be rebuilt onto a
    /// fresh layout node, so this element's node must be relinked to it.
    ChildrenChanged,
    /// The element sits inside a subtree being rebuilt for one of the reasons
    /// above.
    AncestorRebuilt,
    /// Reconciliation is switched off window-wide (§9's kill switch).
    ReconciliationDisabled,
}

/// One element's entry in a [`FramePlan`].
///
/// Not [`Eq`], because a boundary policy and a scroll offset are both
/// float-valued and the crate does not invent a total order for floats — the
/// same reason `ScenePatch` stops at [`PartialEq`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PlannedNode {
    /// The element's path-derived address, present for **every** element the
    /// walk visited — including elements inside an `.uncached()` subtree, which
    /// have no retained record but still emit primitives every frame (§4.2).
    ///
    /// Distinct from [`PlannedNode::instance`] on purpose: that field answers
    /// "is a record retained here," this one answers "what is this element
    /// called," and §4.2's decoupling depends on the second having an answer
    /// wherever the first does not.
    pub address: InstanceKey,
    /// The element's retained instance, or `None` inside an `.uncached()`
    /// subtree where no record exists.
    pub instance: Option<InstanceKey>,
    /// The compositing boundary this element's own primitives belong to.
    ///
    /// For a boundary root this is its *parent's* boundary, not the one it
    /// declares: a scroll container's own background does not scroll with its
    /// contents, so its own paint stays in the layer around it. What it
    /// declares is [`PlannedNode::declared_boundary`], and that is what its
    /// children get.
    pub boundary: BoundaryId,
    /// The boundary this element declared, if it called `.boundary()`.
    pub declared_boundary: Option<BoundaryId>,
    /// The tuning that boundary was declared with.
    pub boundary_policy: Option<BoundaryPolicy>,
    /// The displacement this element applies to its children (§4.1's scroll
    /// signal, as a value rather than as an event).
    pub scroll_offset: [f32; 2],
    pub clip_children: bool,
    /// The element's state scope, which exists regardless of reconciliation
    /// (§4.2: state retention and reconciliation-suppression are decoupled).
    pub state: StateScope,
    /// The element's layout node this frame.
    pub layout_node: LayoutNodeId,
    /// Depth below the tree root, which is `0`.
    pub depth: u32,
    /// What reconciliation decided.
    pub outcome: NodeOutcome,
    /// Which respects of the element changed, as derived by the comparison.
    pub invalidation: Invalidation,
    /// The external texture sampled by this leaf, if any.
    pub external_surface: Option<ExternalSurfaceProperties>,
}

impl PlannedNode {
    /// Whether this element skipped `prepaint` and `paint` this frame.
    ///
    /// Phase 1's second gate, restated as a predicate.
    pub fn skipped_prepaint_and_paint(&self) -> bool {
        matches!(self.outcome, NodeOutcome::Reused)
    }

}

/// How much work a frame's reconciliation actually did.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct FrameStats {
    /// Elements visited.
    pub visited: usize,
    /// Elements reused, which is to say elements that skipped
    /// `prepaint`/`paint`.
    pub reused: usize,
    /// Elements rebuilt.
    pub rebuilt: usize,
    /// Elements inside an `.uncached()` subtree.
    pub uncached: usize,
    /// Layout nodes created this frame.
    pub layout_nodes_created: usize,
    /// Layout nodes reused from a previous frame.
    pub layout_nodes_reused: usize,
    /// Retained instance records dropped by the end-of-frame sweep.
    pub instances_swept: usize,
    /// Layout nodes dropped by the end-of-frame sweep.
    pub layout_nodes_swept: usize,
}

/// Everything reconciliation decided for one frame, in visit order.
///
/// Visit order is pre-order (a parent precedes its children), so index `0` is
/// always the tree root. Depth is recorded per node, so the tree's shape is
/// recoverable from the flat list with a stack and no back-pointers — which is
/// what `patch::emit` walks.
///
/// # The emitter table, and why it lives here
///
/// A plan carries one optional [`Emit`] per node, parallel to [`FramePlan::
/// nodes`]. It has to live somewhere: [`crate::reconcile::reconciler::
/// Reconciler::reconcile`] consumes the description (see [`crate::reconcile::
/// description::Description`]'s own doc for why), and the emit walk runs after
/// layout has been computed, so an element's emitter must survive the gap
/// between the two. Attaching it to the plan the emit walk already consumes is
/// the shortest such path.
///
/// The cost, stated plainly: a boxed trait object is neither [`Clone`] nor
/// [`PartialEq`], so `FramePlan` is neither. Nothing this loses is load-bearing
/// — §2's "the seam is pure data, never a callback" claim is about
/// [`crate::patch::apply::ScenePatch`], which is still exactly that, and is
/// what actually crosses into the backend.
#[derive(Default)]
pub struct FramePlan {
    nodes: Vec<PlannedNode>,
    emitters: Vec<Option<Box<dyn Emit>>>,
    interactions: Vec<Option<DescriptionInteraction>>,
    scroll_infos: Vec<Option<ScrollInfo>>,
    layout_callbacks: Vec<Option<DescriptionLayout>>,
    scroll_axes: Vec<[bool; 2]>,
    automatic_scroll: Vec<bool>,
    stats: FrameStats,
}

impl std::fmt::Debug for FramePlan {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FramePlan")
            .field("nodes", &self.nodes)
            .field(
                "emitters",
                &self.emitters.iter().filter(|slot| slot.is_some()).count(),
            )
            .field("stats", &self.stats)
            .finish()
    }
}

impl FramePlan {
    /// An empty plan.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a visited element, returning its index.
    pub fn push(&mut self, node: PlannedNode) -> usize {
        Self::count(&mut self.stats, node.outcome, 1);
        self.stats.visited += 1;
        self.nodes.push(node);
        self.emitters.push(None);
        self.interactions.push(None);
        self.scroll_infos.push(None);
        self.layout_callbacks.push(None);
        self.scroll_axes.push([false, false]);
        self.automatic_scroll.push(false);
        self.nodes.len() - 1
    }

    pub(crate) fn set_scroll_metadata(
        &mut self,
        index: usize,
        axes: [bool; 2],
        automatic: bool,
    ) {
        if let Some(value) = self.scroll_axes.get_mut(index) {
            *value = axes;
        }
        if let Some(value) = self.automatic_scroll.get_mut(index) {
            *value = automatic;
        }
    }

    pub fn scroll_axes(&self, index: usize) -> Option<[bool; 2]> {
        self.scroll_axes.get(index).copied()
    }

    pub fn has_automatic_scroll(&self, index: usize) -> bool {
        self.automatic_scroll.get(index).copied().unwrap_or(false)
    }

    pub fn set_scroll_offset(&mut self, index: usize, offset: [f32; 2]) -> bool {
        let Some(node) = self.nodes.get_mut(index) else {
            return false;
        };
        if node.scroll_offset == offset {
            return false;
        }
        node.scroll_offset = offset;
        true
    }

    /// Attach an element's emitter, if it had one.
    ///
    /// Separate from [`FramePlan::push`] for the same reason
    /// [`FramePlan::set_layout_node`] is: the reconciler records a node before
    /// visiting its children and settles the rest afterwards, and threading a
    /// second value through that split would put an `Option` in the hot
    /// constructor for the benefit of the minority of elements that emit
    /// anything.
    pub(crate) fn set_emitter(&mut self, index: usize, emitter: Option<Box<dyn Emit>>) {
        if let Some(slot) = self.emitters.get_mut(index) {
            *slot = emitter;
        }
    }

    pub(crate) fn set_interaction(
        &mut self,
        index: usize,
        interaction: Option<DescriptionInteraction>,
    ) {
        if let Some(slot) = self.interactions.get_mut(index) {
            *slot = interaction;
        }
    }

    pub(crate) fn set_scroll_info(&mut self, index: usize, scroll_info: Option<ScrollInfo>) {
        if let Some(slot) = self.scroll_infos.get_mut(index) {
            *slot = scroll_info;
        }
    }

    pub fn scroll_info(&self, index: usize) -> Option<ScrollInfo> {
        self.scroll_infos.get(index).copied().flatten()
    }

    pub fn take_interaction(&mut self, index: usize) -> Option<DescriptionInteraction> {
        self.interactions.get_mut(index)?.take()
    }

    pub(crate) fn set_layout_callback(
        &mut self,
        index: usize,
        callback: Option<DescriptionLayout>,
    ) {
        if let Some(slot) = self.layout_callbacks.get_mut(index) {
            *slot = callback;
        }
    }

    pub fn take_layout_callback(&mut self, index: usize) -> Option<DescriptionLayout> {
        self.layout_callbacks.get_mut(index)?.take()
    }

    /// The emitter for the element at `index`, if it has one.
    pub fn emitter(&self, index: usize) -> Option<&dyn Emit> {
        self.emitters.get(index)?.as_deref()
    }

    /// Revise a previously recorded element's outcome.
    ///
    /// The walk is pre-order — a parent is recorded before its children — but
    /// one of the parent's reuse conditions is only knowable *after* the
    /// children have been visited: a child rebuilt onto a fresh layout node
    /// forces its parent to relink. Rather than reorder the plan (which would
    /// stop index `0` being the root) or return descendant lists up the
    /// recursion (which would allocate per element per frame), the reconciler
    /// records its optimistic decision and revises it here.
    pub(crate) fn amend(&mut self, index: usize, outcome: NodeOutcome, invalidation: Invalidation) {
        let Some(node) = self.nodes.get_mut(index) else {
            return;
        };
        let previous = node.outcome;
        node.outcome = outcome;
        node.invalidation = invalidation;
        if previous != outcome {
            Self::count(&mut self.stats, previous, -1);
            Self::count(&mut self.stats, outcome, 1);
        }
    }

    /// Fill in a recorded element's layout node, which is only known after its
    /// children have been visited and its reuse decision settled.
    pub(crate) fn set_layout_node(&mut self, index: usize, layout_node: LayoutNodeId) {
        if let Some(node) = self.nodes.get_mut(index) {
            node.layout_node = layout_node;
        }
    }

    fn count(stats: &mut FrameStats, outcome: NodeOutcome, delta: isize) {
        let bucket = match outcome {
            NodeOutcome::Reused => &mut stats.reused,
            NodeOutcome::Rebuilt(_) => &mut stats.rebuilt,
            NodeOutcome::Uncached => &mut stats.uncached,
        };
        *bucket = bucket.saturating_add_signed(delta);
    }

    /// Overwrite the plan's counters. Called once by the reconciler after the
    /// walk, to fold in the layout tree's own create/reuse/sweep totals.
    pub fn set_frame_totals(
        &mut self,
        layout_nodes_created: usize,
        layout_nodes_reused: usize,
        instances_swept: usize,
        layout_nodes_swept: usize,
    ) {
        self.stats.layout_nodes_created = layout_nodes_created;
        self.stats.layout_nodes_reused = layout_nodes_reused;
        self.stats.instances_swept = instances_swept;
        self.stats.layout_nodes_swept = layout_nodes_swept;
    }

    /// Every visited element, in pre-order.
    pub fn nodes(&self) -> &[PlannedNode] {
        &self.nodes
    }

    /// The tree root, if the plan is not empty.
    pub fn root(&self) -> Option<&PlannedNode> {
        self.nodes.first()
    }

    /// This frame's counters.
    pub fn stats(&self) -> FrameStats {
        self.stats
    }

    /// Whether every visited element skipped `prepaint`/`paint`.
    ///
    /// An empty plan is vacuously *not* fully reused: a frame that visited
    /// nothing has not demonstrated anything about reconciliation, and letting
    /// it report `true` would make Phase 1's second gate passable by a test
    /// that accidentally built no tree.
    pub fn fully_reused(&self) -> bool {
        !self.nodes.is_empty() && self.stats.visited == self.stats.reused
    }

    /// The entry for a given instance, if it was visited.
    pub fn node_for_instance(&self, instance: InstanceKey) -> Option<&PlannedNode> {
        self.nodes
            .iter()
            .find(|node| node.instance == Some(instance))
    }

    /// The entry for a given state scope, if it was visited.
    pub fn node_for_state(&self, state: StateScope) -> Option<&PlannedNode> {
        self.nodes.iter().find(|node| node.state == state)
    }

    /// Every entry at `depth` below the root.
    pub fn nodes_at_depth(&self, depth: u32) -> Vec<&PlannedNode> {
        self.nodes
            .iter()
            .filter(|node| node.depth == depth)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::reconcile::description::ElementId;

    fn node(outcome: NodeOutcome, depth: u32, raw: u64) -> PlannedNode {
        PlannedNode {
            address: InstanceKey::from_raw(raw),
            instance: Some(InstanceKey::from_raw(raw)),
            boundary: BoundaryId::ROOT,
            declared_boundary: None,
            boundary_policy: None,
            scroll_offset: [0.0, 0.0],
            clip_children: false,
            state: StateScope::from_path(&[ElementId::Slot(raw as u32)]),
            layout_node: LayoutNodeId::from_raw(raw),
            depth,
            outcome,
            invalidation: Invalidation::empty(),
            external_surface: None,
        }
    }

    #[test]
    fn an_empty_plan_is_not_fully_reused() {
        let plan = FramePlan::new();
        assert!(!plan.fully_reused());
        assert!(plan.root().is_none());
        assert_eq!(plan.stats(), FrameStats::default());
    }

    #[test]
    fn counters_track_each_outcome_separately() {
        let mut plan = FramePlan::new();
        plan.push(node(NodeOutcome::Reused, 0, 1));
        plan.push(node(NodeOutcome::Rebuilt(RebuildReason::KeyChanged), 1, 2));
        plan.push(node(NodeOutcome::Uncached, 1, 3));
        let stats = plan.stats();
        assert_eq!(stats.visited, 3);
        assert_eq!(stats.reused, 1);
        assert_eq!(stats.rebuilt, 1);
        assert_eq!(stats.uncached, 1);
        assert!(!plan.fully_reused());
    }

    #[test]
    fn a_plan_of_only_reused_nodes_reports_fully_reused() {
        let mut plan = FramePlan::new();
        plan.push(node(NodeOutcome::Reused, 0, 1));
        plan.push(node(NodeOutcome::Reused, 1, 2));
        assert!(plan.fully_reused());
        assert!(
            plan.nodes()
                .iter()
                .all(PlannedNode::skipped_prepaint_and_paint)
        );
    }

    #[test]
    fn nodes_can_be_found_by_instance_and_by_depth() {
        let mut plan = FramePlan::new();
        plan.push(node(NodeOutcome::Reused, 0, 1));
        plan.push(node(NodeOutcome::Reused, 1, 2));
        plan.push(node(NodeOutcome::Reused, 1, 3));
        assert_eq!(plan.nodes_at_depth(1).len(), 2);
        assert!(plan.node_for_instance(InstanceKey::from_raw(3)).is_some());
        assert!(plan.node_for_instance(InstanceKey::from_raw(9)).is_none());
    }

    #[test]
    fn amending_an_outcome_moves_it_between_counters() {
        let mut plan = FramePlan::new();
        let index = plan.push(node(NodeOutcome::Reused, 0, 1));
        assert_eq!(plan.stats().reused, 1);
        plan.amend(
            index,
            NodeOutcome::Rebuilt(RebuildReason::ChildrenChanged),
            Invalidation::LAYOUT,
        );
        assert_eq!(plan.stats().reused, 0);
        assert_eq!(plan.stats().rebuilt, 1);
        assert_eq!(plan.stats().visited, 1);
        assert!(!plan.fully_reused());
    }

    #[test]
    fn amending_an_out_of_range_index_is_inert() {
        let mut plan = FramePlan::new();
        plan.amend(7, NodeOutcome::Uncached, Invalidation::all());
        assert_eq!(plan.stats(), FrameStats::default());
    }

    #[test]
    fn a_node_carries_no_emitter_until_one_is_attached() {
        let mut plan = FramePlan::new();
        let index = plan.push(node(NodeOutcome::Reused, 0, 1));
        assert!(plan.emitter(index).is_none());
        plan.set_emitter(
            index,
            Some(Box::new(
                |_: &crate::patch::emit::EmitContext, _: &mut crate::patch::emit::Emission| {},
            )),
        );
        assert!(plan.emitter(index).is_some());
        assert!(plan.emitter(index + 1).is_none());
    }

    #[test]
    fn attaching_an_emitter_out_of_range_is_inert() {
        let mut plan = FramePlan::new();
        plan.set_emitter(
            7,
            Some(Box::new(
                |_: &crate::patch::emit::EmitContext, _: &mut crate::patch::emit::Emission| {},
            )),
        );
        assert!(plan.emitter(7).is_none());
    }
}
