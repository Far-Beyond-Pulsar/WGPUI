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

use crate::invalidation::axes::Invalidation;
use crate::reconcile::instance::InstanceKey;
use crate::reconcile::state::StateScope;
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
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PlannedNode {
    /// The element's retained instance, or `None` inside an `.uncached()`
    /// subtree where no record exists.
    pub instance: Option<InstanceKey>,
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
/// always the tree root.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FramePlan {
    nodes: Vec<PlannedNode>,
    stats: FrameStats,
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
        self.nodes.len() - 1
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
    pub(crate) fn amend(
        &mut self,
        index: usize,
        outcome: NodeOutcome,
        invalidation: Invalidation,
    ) {
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
            instance: Some(InstanceKey::from_raw(raw)),
            state: StateScope::from_path(&[ElementId::Slot(raw as u32)]),
            layout_node: LayoutNodeId::from_raw(raw),
            depth,
            outcome,
            invalidation: Invalidation::empty(),
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
        plan.push(node(
            NodeOutcome::Rebuilt(RebuildReason::KeyChanged),
            1,
            2,
        ));
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
        assert!(plan.nodes().iter().all(PlannedNode::skipped_prepaint_and_paint));
    }

    #[test]
    fn nodes_can_be_found_by_instance_and_by_depth() {
        let mut plan = FramePlan::new();
        plan.push(node(NodeOutcome::Reused, 0, 1));
        plan.push(node(NodeOutcome::Reused, 1, 2));
        plan.push(node(NodeOutcome::Reused, 1, 3));
        assert_eq!(plan.nodes_at_depth(1).len(), 2);
        assert!(
            plan.node_for_instance(InstanceKey::from_raw(3))
                .is_some()
        );
        assert!(
            plan.node_for_instance(InstanceKey::from_raw(9))
                .is_none()
        );
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
}
