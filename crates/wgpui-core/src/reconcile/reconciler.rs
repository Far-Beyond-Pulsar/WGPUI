//! The ambient reconciliation walk itself.
//! See docs/gpu-native-architecture.md §4.0, constraint 5 (§0), §4.2.
//!
//! Not in §3.1's literal file map — a deliberate addition, recorded in
//! `docs/phase-1-results.md`. §3.1 lists `reconcile/instance.rs` (the retained
//! record), `reconcile/diff_key.rs` (the fingerprint trait), and
//! `reconcile/uncached.rs` (the scope flag) but no file for the walk that uses
//! all three, because in the legacy backend that walk *is* `Div::prepaint`'s
//! child loop inside `div.rs`. Ambient reconciliation is precisely the claim
//! that this is not one element type's business, so it gets its own file.
//!
//! # What makes it ambient
//!
//! Three properties, each an absence rather than a feature:
//!
//! 1. **[`Reconciler::reconcile`] takes a description tree and nothing else.**
//!    There is no layer, boundary, subtree, or scope parameter it could be
//!    fenced by.
//! 2. **Identity never requires a name.** An element that does not call `.id()`
//!    is addressed by its slot under its parent (SFD §1.0's positional
//!    identity), so a full path exists for every element by construction. A
//!    forgotten `.id()` costs identity stability under reordering; it never
//!    costs reconciliation.
//! 3. **Nothing consults a policy.** The only reasons an element rebuilds are
//!    listed in [`RebuildReason`], and none of them is "it was not inside a
//!    cached region."
//!
//! # The one cross-element dependency
//!
//! An element's own fingerprint comparing clean is necessary but not
//! sufficient: if a child had to be rebuilt onto a fresh layout node, or the
//! child list changed shape, this element's node must be relinked. So the walk
//! records an optimistic decision, visits the children, and revises the parent
//! through [`FramePlan::amend`] when that happens. This is the only place a
//! node's outcome depends on anything but its own fingerprint, and it is why
//! the revision reports [`RebuildReason::ChildrenChanged`] rather than
//! pretending the element's own key changed.

use crate::boundary::identity::BoundaryIdentity;
use crate::invalidation::axes::Invalidation;
use crate::reconcile::description::{Description, ElementId};
use crate::reconcile::instance::{InstanceKey, InstanceTable, RetainedElement};
use crate::reconcile::plan::{FramePlan, NodeOutcome, PlannedNode, RebuildReason};
use crate::reconcile::state::StateScope;
use crate::reconcile::uncached::UncachedScope;
use crate::scene::layer::BoundaryId;
use wgpui_layout::taffy_tree::{LayoutError, LayoutNodeId, LayoutStyle, LayoutTree};

/// Reconciliation could not complete.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReconcileError {
    /// The layout tree rejected an operation.
    Layout(LayoutError),
}

impl std::fmt::Display for ReconcileError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReconcileError::Layout(error) => write!(formatter, "layout: {error}"),
        }
    }
}

impl std::error::Error for ReconcileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ReconcileError::Layout(error) => Some(error),
        }
    }
}

impl From<LayoutError> for ReconcileError {
    fn from(error: LayoutError) -> Self {
        ReconcileError::Layout(error)
    }
}

/// Where in the walk an element sits, and under what suppression.
#[derive(Copy, Clone, Debug)]
struct WalkPosition {
    depth: u32,
    scope: UncachedScope,
    /// Set when an ancestor is being rebuilt in a way that invalidates
    /// descendants' retained records — today, only a type mismatch.
    force_rebuild: bool,
    /// The nearest enclosing `.boundary()`, or [`BoundaryId::ROOT`].
    ///
    /// Carried through the walk rather than looked up, because a boundary's
    /// identity is derived from the path and the path is exactly what the walk
    /// already has. Note what this is *not*: nothing below reads it before
    /// deciding whether to reuse an element. It is recorded on the plan and
    /// consumed downstream by `patch::emit`, which is what keeps §4.0's
    /// "reconciliation is not fenced by a boundary" true by construction rather
    /// than by discipline.
    boundary: BoundaryId,
}

/// The mutable state the walk threads through, kept together so the recursion
/// stays readable rather than growing a parameter per collaborator.
struct WalkContext<'a> {
    layout: &'a mut LayoutTree,
    plan: &'a mut FramePlan,
}

/// What one visited element hands back to its parent.
struct VisitResult {
    layout_node: LayoutNodeId,
    instance: Option<InstanceKey>,
    /// Whether this element's layout node is not the one it had last frame —
    /// the only thing about a child that a parent's own decision depends on.
    node_changed: bool,
}

/// Reconciles a per-frame description tree against the retained instance
/// tree, window-wide.
#[derive(Debug)]
pub struct Reconciler {
    instances: InstanceTable,
    frame: u64,
    enabled: bool,
    path: Vec<ElementId>,
}

impl Default for Reconciler {
    fn default() -> Self {
        Self::new()
    }
}

impl Reconciler {
    /// A reconciler with reconciliation on, which is the only default §4.0
    /// permits.
    pub fn new() -> Self {
        Self {
            instances: InstanceTable::new(),
            frame: 0,
            enabled: true,
            path: Vec::new(),
        }
    }

    /// A reconciler with reconciliation switched off window-wide.
    ///
    /// §9's risk table asks for this by name: making reconciliation ambient
    /// means a reconciliation bug's blast radius is the whole application from
    /// Phase 1 onward, so the pre-reconciliation path stays reachable —
    /// following the legacy backend's own `WGPUI_INSTANCES=0` precedent, but
    /// now gating the *default* path rather than an edge feature.
    ///
    /// With this set every element takes the unconditional rebuild path: no
    /// record retained, no fingerprint compared, a fresh layout node each
    /// frame.
    pub fn with_reconciliation_disabled() -> Self {
        Self {
            enabled: false,
            ..Self::new()
        }
    }

    /// Whether reconciliation is on.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// The index of the most recently reconciled frame.
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// The retained instance table. Phase 1's third gate reads its length.
    pub fn instances(&self) -> &InstanceTable {
        &self.instances
    }

    /// Reconcile one frame's description tree against the retained one.
    ///
    /// Consumes the description — see [`Description`]'s own doc for why — and
    /// returns the plan describing each element's fate and the layout node
    /// backing it.
    pub fn reconcile(
        &mut self,
        root: Description,
        layout: &mut LayoutTree,
    ) -> Result<FramePlan, ReconcileError> {
        self.frame += 1;
        layout.begin_frame();
        self.path.clear();

        let mut plan = FramePlan::new();
        let position = WalkPosition {
            depth: 0,
            scope: UncachedScope::new(),
            force_rebuild: false,
            boundary: BoundaryId::ROOT,
        };
        {
            let mut context = WalkContext {
                layout,
                plan: &mut plan,
            };
            self.visit(root, 0, position, &mut context)?;
        }

        let created = layout.stats().nodes_created;
        let reused = layout.stats().nodes_reused;
        let instances_swept = self.instances.sweep(self.frame);
        let layout_nodes_swept = layout.end_frame();
        plan.set_frame_totals(created, reused, instances_swept, layout_nodes_swept);
        Ok(plan)
    }

    fn visit(
        &mut self,
        mut description: Description,
        slot: u32,
        position: WalkPosition,
        context: &mut WalkContext<'_>,
    ) -> Result<VisitResult, ReconcileError> {
        let segment = description
            .element_id
            .take()
            .unwrap_or(ElementId::Slot(slot));
        self.path.push(segment);
        let result = self.visit_addressed(description, position, context);
        self.path.pop();
        result
    }

    fn visit_addressed(
        &mut self,
        description: Description,
        position: WalkPosition,
        context: &mut WalkContext<'_>,
    ) -> Result<VisitResult, ReconcileError> {
        let instance_key = InstanceKey::from_path(&self.path);
        let state = StateScope::from_path(&self.path);
        let position = WalkPosition {
            scope: position.scope.enter(description.uncached),
            ..position
        };

        if position.scope.is_active() || !self.enabled {
            return self.visit_suppressed(instance_key, state, description, position, context);
        }

        let Description {
            type_id,
            diff_key,
            boundary,
            scroll_offset,
            emitter,
            layout_style,
            children,
            ..
        } = description;
        let declared_boundary = boundary.map(|_| BoundaryIdentity::from_path(&self.path));

        let previous = self.instances.get(instance_key);
        let previous_node = previous.map(|instance| instance.layout_node());
        let previous_type_matches = previous.is_some_and(|instance| instance.type_id() == type_id);

        let (mut outcome, mut invalidation) = if position.force_rebuild {
            (
                NodeOutcome::Rebuilt(RebuildReason::AncestorRebuilt),
                Invalidation::all(),
            )
        } else if previous.is_none() {
            (
                NodeOutcome::Rebuilt(RebuildReason::NewInstance),
                Invalidation::all(),
            )
        } else if !previous_type_matches {
            (
                NodeOutcome::Rebuilt(RebuildReason::TypeMismatch),
                Invalidation::all(),
            )
        } else {
            match (
                diff_key.as_deref(),
                previous.and_then(|instance| instance.diff_key()),
            ) {
                (Some(current), Some(retained)) => {
                    let axes = current.compare(retained);
                    if axes.is_empty() {
                        (NodeOutcome::Reused, axes)
                    } else {
                        (NodeOutcome::Rebuilt(RebuildReason::KeyChanged), axes)
                    }
                }
                _ => (
                    NodeOutcome::Rebuilt(RebuildReason::NoDiffKey),
                    Invalidation::all(),
                ),
            }
        };

        // R-N §2.2: a type mismatch is a subtree rebuild, not a local one. The
        // records under this address describe a different element's children
        // and must not be matched against this one's.
        let type_mismatched = outcome == NodeOutcome::Rebuilt(RebuildReason::TypeMismatch);
        if type_mismatched {
            self.instances.remove_subtree(instance_key);
        }

        let plan_index = context.plan.push(PlannedNode {
            address: instance_key,
            instance: Some(instance_key),
            boundary: position.boundary,
            declared_boundary,
            boundary_policy: boundary,
            scroll_offset,
            state,
            // Provisional: filled in below, once the children have been
            // visited and the reuse decision has settled. Recorded now so the
            // plan stays pre-order — see this module's doc.
            layout_node: previous_node.unwrap_or(LayoutNodeId::from_raw(0)),
            depth: position.depth,
            outcome,
            invalidation,
        });
        context.plan.set_emitter(plan_index, emitter);

        let child_position = WalkPosition {
            depth: position.depth + 1,
            scope: position.scope,
            force_rebuild: position.force_rebuild || type_mismatched,
            boundary: declared_boundary.unwrap_or(position.boundary),
        };
        let mut child_nodes = Vec::with_capacity(children.len());
        let mut child_instances = Vec::with_capacity(children.len());
        let mut any_child_node_changed = false;
        for (index, child) in children.into_iter().enumerate() {
            let slot = u32::try_from(index).unwrap_or(u32::MAX);
            let visited = self.visit(child, slot, child_position, context)?;
            child_nodes.push(visited.layout_node);
            if let Some(key) = visited.instance {
                child_instances.push(key);
            }
            any_child_node_changed |= visited.node_changed;
        }

        let reusable_node = previous_node.filter(|_| !type_mismatched);
        let mut settled_node = None;

        if outcome == NodeOutcome::Reused {
            let children_unchanged = !any_child_node_changed
                && self.instances.child_nodes_match(instance_key, &child_nodes);
            match reusable_node.filter(|node| children_unchanged && context.layout.reuse(*node)) {
                Some(node) => settled_node = Some(node),
                None => {
                    outcome = NodeOutcome::Rebuilt(RebuildReason::ChildrenChanged);
                    invalidation = Invalidation::LAYOUT;
                    context.plan.amend(plan_index, outcome, invalidation);
                }
            }
        }

        let layout_node = match settled_node {
            Some(node) => node,
            None => Self::rebuild_layout_node(
                reusable_node,
                layout_style,
                &child_nodes,
                context.layout,
            )?,
        };

        if outcome == NodeOutcome::Reused {
            self.instances.touch(instance_key, self.frame);
        } else {
            self.instances.store(
                instance_key,
                RetainedElement {
                    type_id,
                    diff_key,
                    layout_node,
                    child_nodes,
                    children: child_instances,
                },
                self.frame,
            );
        }
        context.plan.set_layout_node(plan_index, layout_node);

        Ok(VisitResult {
            layout_node,
            instance: Some(instance_key),
            node_changed: previous_node != Some(layout_node),
        })
    }

    /// The unconditional-rebuild path, taken inside an `.uncached()` subtree
    /// (§4.2) and when reconciliation is switched off (§9).
    ///
    /// No record is allocated, no fingerprint retained, no comparison run —
    /// and the state scope is derived and reported exactly as on the
    /// reconciled path, which is what keeps state retention independent of
    /// reconciliation rather than a side effect of it.
    fn visit_suppressed(
        &mut self,
        instance_key: InstanceKey,
        state: StateScope,
        mut description: Description,
        position: WalkPosition,
        context: &mut WalkContext<'_>,
    ) -> Result<VisitResult, ReconcileError> {
        // An element reconciled last frame and uncached this frame must stop
        // costing a retained record immediately, not at the next sweep —
        // §4.2's whole point is that the bookkeeping goes away.
        self.instances.remove_subtree(instance_key);

        let outcome = if position.scope.is_active() {
            NodeOutcome::Uncached
        } else {
            NodeOutcome::Rebuilt(RebuildReason::ReconciliationDisabled)
        };
        // An uncached subtree still composites and still emits (§4.2), so its
        // boundary declaration and its emitter are recorded exactly as a
        // reconciled element's are — the flag suppresses diffing, not drawing.
        let declared_boundary = description
            .boundary
            .map(|_| BoundaryIdentity::from_path(&self.path));
        let plan_index = context.plan.push(PlannedNode {
            address: instance_key,
            instance: None,
            boundary: position.boundary,
            declared_boundary,
            boundary_policy: description.boundary,
            scroll_offset: description.scroll_offset,
            state,
            layout_node: LayoutNodeId::from_raw(0),
            depth: position.depth,
            outcome,
            invalidation: Invalidation::all(),
        });
        context
            .plan
            .set_emitter(plan_index, description.emitter.take());

        let child_position = WalkPosition {
            depth: position.depth + 1,
            scope: position.scope,
            force_rebuild: true,
            boundary: declared_boundary.unwrap_or(position.boundary),
        };
        let mut child_nodes = Vec::with_capacity(description.children.len());
        for (index, child) in description.children.into_iter().enumerate() {
            let slot = u32::try_from(index).unwrap_or(u32::MAX);
            let visited = self.visit(child, slot, child_position, context)?;
            child_nodes.push(visited.layout_node);
        }

        let layout_node = context
            .layout
            .request_layout(description.layout_style, &child_nodes)?;
        context.plan.set_layout_node(plan_index, layout_node);

        Ok(VisitResult {
            layout_node,
            instance: None,
            node_changed: true,
        })
    }

    fn rebuild_layout_node(
        reusable_node: Option<LayoutNodeId>,
        layout_style: LayoutStyle,
        child_nodes: &[LayoutNodeId],
        layout: &mut LayoutTree,
    ) -> Result<LayoutNodeId, ReconcileError> {
        // A rebuild is not a reason to throw the layout node away: R-N §2.5's
        // rule is that an instance keeps its node and Taffy's own dirty
        // propagation handles the rest. Only a type mismatch or a genuinely
        // new element needs a fresh node, and both arrive here with
        // `reusable_node` already `None`.
        if let Some(node) = reusable_node
            && layout.reuse(node)
        {
            layout.set_style(node, layout_style)?;
            layout.set_children(node, child_nodes)?;
            return Ok(node);
        }
        Ok(layout.request_layout(layout_style, child_nodes)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reconcile::diff_key::{AlwaysDirty, ReconcileKey, compare_by_equality};
    use crate::reconcile::state::{ElementStateStore, StateKey};
    use std::any::Any;

    struct Panel;
    struct Image;

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

    fn leaf(value: u32) -> Description {
        Description::new::<Panel>().diff_key(Fingerprint(value))
    }

    /// A three-level tree with no `.id()` and no boundary concept anywhere.
    fn tree(value: u32) -> Description {
        Description::new::<Panel>()
            .diff_key(Fingerprint(value))
            .child(
                Description::new::<Panel>()
                    .diff_key(Fingerprint(value))
                    .child(leaf(value))
                    .child(leaf(value)),
            )
            .child(
                Description::new::<Panel>()
                    .diff_key(Fingerprint(value))
                    .child(leaf(value)),
            )
    }

    /// Recursively confirm a description tree names no element and opts into
    /// nothing — the "zero API touched anywhere in the test" half of gate #2,
    /// checked mechanically rather than by reading the helper above.
    fn assert_names_nothing(description: &Description) {
        assert!(
            description.element_id().is_none(),
            "gate #2 requires an element tree with no explicit id anywhere"
        );
        assert!(
            !description.is_uncached(),
            "gate #2 requires an element tree that opts out of nothing"
        );
        for child in description.child_descriptions() {
            assert_names_nothing(child);
        }
    }

    /// **Phase 1 gate #2** (§4.0, §8): a plain, unboundaried, three-level-deep
    /// element tree that renders identically to the previous frame keeps the
    /// same layout-node identity and skips its equivalent of `prepaint` and
    /// `paint`.
    ///
    /// Zero boundary API, zero explicit id, zero opt-in of any kind is touched
    /// — [`assert_names_nothing`] checks the first two mechanically, and the
    /// third is an absence in [`Reconciler::reconcile`]'s own signature: it
    /// takes a description tree and a layout tree, and has no parameter a
    /// caller could fence it with.
    #[test]
    fn gate_2_an_unboundaried_three_level_tree_keeps_its_nodes_and_skips_prepaint_and_paint()
    -> Result<(), ReconcileError> {
        let mut reconciler = Reconciler::new();
        let mut layout = LayoutTree::new();

        let first_description = tree(0);
        assert_names_nothing(&first_description);
        let first = reconciler.reconcile(first_description, &mut layout)?;
        assert_eq!(first.stats().visited, 6);
        assert_eq!(
            first.nodes().iter().map(|node| node.depth).max(),
            Some(2),
            "the tree must actually be three levels deep"
        );

        let second_description = tree(0);
        assert_names_nothing(&second_description);
        let second = reconciler.reconcile(second_description, &mut layout)?;

        assert!(
            second
                .nodes()
                .iter()
                .all(PlannedNode::skipped_prepaint_and_paint),
            "every element in an unchanged unboundaried tree must skip prepaint/paint"
        );
        assert_eq!(second.stats().reused, 6);
        assert_eq!(second.stats().rebuilt, 0);
        assert_eq!(second.stats().layout_nodes_created, 0);
        assert_eq!(second.stats().layout_nodes_reused, 6);
        assert_eq!(second.stats().layout_nodes_swept, 0);
        assert_eq!(second.stats().instances_swept, 0);

        for node in second.nodes() {
            let instance = match node.instance {
                Some(instance) => instance,
                None => panic!("a reconciled element must have an instance record"),
            };
            let before = first
                .node_for_instance(instance)
                .map(|node| node.layout_node);
            assert_eq!(
                before,
                Some(node.layout_node),
                "a clean element must keep the exact layout node it had last frame"
            );
            assert!(layout.is_live(node.layout_node));
        }
        Ok(())
    }

    /// **Phase 1 gate #3** (§4.2, §8): a subtree marked `.uncached()` allocates
    /// no retained instance record, and its children's separately-keyed state
    /// survives across frames identically to a reconciled subtree's.
    ///
    /// The tree holds two structurally identical panels side by side — one
    /// reconciled, one `.uncached()` — so "identically" is checked against a
    /// live control rather than an asserted constant.
    #[test]
    fn gate_3_uncached_allocates_no_instance_while_state_survives_identically()
    -> Result<(), ReconcileError> {
        #[derive(Debug, PartialEq)]
        struct Visits(u32);

        fn panels() -> Description {
            Description::new::<Panel>()
                .diff_key(Fingerprint(0))
                .child(
                    Description::new::<Panel>()
                        .diff_key(Fingerprint(0))
                        .child(leaf(0)),
                )
                .child(
                    Description::new::<Panel>()
                        .uncached()
                        .diff_key(Fingerprint(0))
                        .child(leaf(0)),
                )
        }

        fn visit_state(store: &mut ElementStateStore, scope: StateScope, frame: u64) -> u32 {
            store
                .with_state(
                    StateKey::new::<Visits>(scope),
                    frame,
                    || Visits(0),
                    |visits| {
                        visits.0 += 1;
                        visits.0
                    },
                )
                .unwrap_or(0)
        }

        let reconciled_leaf =
            StateScope::from_path(&[ElementId::Slot(0), ElementId::Slot(0), ElementId::Slot(0)]);
        let uncached_leaf =
            StateScope::from_path(&[ElementId::Slot(0), ElementId::Slot(1), ElementId::Slot(0)]);

        let mut reconciler = Reconciler::new();
        let mut layout = LayoutTree::new();
        let mut state = ElementStateStore::new();
        let mut plan = reconciler.reconcile(panels(), &mut layout)?;

        for frame in 1..=3u64 {
            if frame > 1 {
                plan = reconciler.reconcile(panels(), &mut layout)?;
            }

            let reconciled = match plan.node_for_state(reconciled_leaf) {
                Some(node) => *node,
                None => panic!("the reconciled leaf must appear in the plan"),
            };
            let uncached = match plan.node_for_state(uncached_leaf) {
                Some(node) => *node,
                None => panic!("an uncached element is still visited, and still planned"),
            };

            assert!(reconciled.instance.is_some());
            assert_eq!(
                uncached.instance, None,
                "no retained record exists for anything inside an `.uncached()` subtree"
            );
            assert_eq!(uncached.outcome, NodeOutcome::Uncached);
            assert_eq!(plan.stats().uncached, 2, "the flag applies to the subtree");
            assert_eq!(
                reconciler.instances().len(),
                3,
                "only the root and the reconciled panel's two elements are retained"
            );

            let reconciled_visits = visit_state(&mut state, reconciled_leaf, frame);
            let uncached_visits = visit_state(&mut state, uncached_leaf, frame);
            assert_eq!(
                reconciled_visits, uncached_visits,
                "state must survive identically on both sides of the flag"
            );
            assert_eq!(reconciled_visits, frame as u32);
            assert_eq!(
                state.sweep(frame),
                0,
                "an uncached element visits its state like any other, so nothing is swept"
            );
        }

        // The control half: while state behaved identically, reconciliation did
        // not — which is what makes the two mechanisms decoupled rather than
        // merely both present.
        assert!(
            plan.node_for_state(reconciled_leaf)
                .is_some_and(PlannedNode::skipped_prepaint_and_paint)
        );
        assert!(
            !plan
                .node_for_state(uncached_leaf)
                .is_some_and(PlannedNode::skipped_prepaint_and_paint)
        );
        Ok(())
    }

    #[test]
    fn an_element_that_stops_being_uncached_is_reconciled_again() -> Result<(), ReconcileError> {
        let mut reconciler = Reconciler::new();
        let mut layout = LayoutTree::new();
        reconciler.reconcile(
            Description::new::<Panel>()
                .uncached()
                .diff_key(Fingerprint(0))
                .child(leaf(0)),
            &mut layout,
        )?;
        assert!(reconciler.instances().is_empty());

        reconciler.reconcile(
            Description::new::<Panel>()
                .diff_key(Fingerprint(0))
                .child(leaf(0)),
            &mut layout,
        )?;
        assert_eq!(reconciler.instances().len(), 2);
        let plan = reconciler.reconcile(
            Description::new::<Panel>()
                .diff_key(Fingerprint(0))
                .child(leaf(0)),
            &mut layout,
        )?;
        assert!(plan.fully_reused());
        Ok(())
    }

    #[test]
    fn an_element_that_becomes_uncached_drops_the_records_it_had() -> Result<(), ReconcileError> {
        let mut reconciler = Reconciler::new();
        let mut layout = LayoutTree::new();
        reconciler.reconcile(tree(0), &mut layout)?;
        assert_eq!(reconciler.instances().len(), 6);

        let mut uncached_root = tree(0);
        uncached_root = uncached_root.uncached();
        reconciler.reconcile(uncached_root, &mut layout)?;
        assert!(
            reconciler.instances().is_empty(),
            "the bookkeeping must go away immediately, not at the next sweep"
        );
        Ok(())
    }

    #[test]
    fn nested_uncached_subtrees_stay_suppressed_past_the_inner_one() -> Result<(), ReconcileError> {
        let mut reconciler = Reconciler::new();
        let mut layout = LayoutTree::new();
        let plan = reconciler.reconcile(
            Description::new::<Panel>().diff_key(Fingerprint(0)).child(
                Description::new::<Panel>()
                    .uncached()
                    .child(Description::new::<Panel>().uncached().child(leaf(0)))
                    .child(leaf(0)),
            ),
            &mut layout,
        )?;
        assert_eq!(plan.stats().uncached, 4);
        assert_eq!(reconciler.instances().len(), 1);
        Ok(())
    }

    #[test]
    fn a_first_frame_builds_everything() -> Result<(), ReconcileError> {
        let mut reconciler = Reconciler::new();
        let mut layout = LayoutTree::new();
        let plan = reconciler.reconcile(tree(0), &mut layout)?;
        assert_eq!(plan.stats().visited, 6);
        assert_eq!(plan.stats().rebuilt, 6);
        assert_eq!(plan.stats().reused, 0);
        assert_eq!(plan.stats().layout_nodes_created, 6);
        assert_eq!(reconciler.instances().len(), 6);
        Ok(())
    }

    #[test]
    fn an_identical_second_frame_reuses_everything() -> Result<(), ReconcileError> {
        let mut reconciler = Reconciler::new();
        let mut layout = LayoutTree::new();
        let first = reconciler.reconcile(tree(0), &mut layout)?;
        let second = reconciler.reconcile(tree(0), &mut layout)?;
        assert!(second.fully_reused());
        assert_eq!(second.stats().layout_nodes_created, 0);
        assert_eq!(second.stats().layout_nodes_swept, 0);
        let first_nodes: Vec<LayoutNodeId> =
            first.nodes().iter().map(|node| node.layout_node).collect();
        let second_nodes: Vec<LayoutNodeId> =
            second.nodes().iter().map(|node| node.layout_node).collect();
        assert_eq!(first_nodes, second_nodes);
        Ok(())
    }

    #[test]
    fn a_changed_leaf_rebuilds_only_itself() -> Result<(), ReconcileError> {
        let mut reconciler = Reconciler::new();
        let mut layout = LayoutTree::new();
        reconciler.reconcile(tree(0), &mut layout)?;

        let changed = Description::new::<Panel>()
            .diff_key(Fingerprint(0))
            .child(
                Description::new::<Panel>()
                    .diff_key(Fingerprint(0))
                    .child(leaf(0))
                    .child(leaf(99)),
            )
            .child(
                Description::new::<Panel>()
                    .diff_key(Fingerprint(0))
                    .child(leaf(0)),
            );
        let plan = reconciler.reconcile(changed, &mut layout)?;
        assert_eq!(plan.stats().rebuilt, 1);
        assert_eq!(plan.stats().reused, 5);
        // A `DISPLAY`-only change keeps the node, so no ancestor has to relink.
        assert_eq!(plan.stats().layout_nodes_created, 0);
        Ok(())
    }

    #[test]
    fn a_type_mismatch_rebuilds_the_subtree_and_takes_a_fresh_node() -> Result<(), ReconcileError> {
        let mut reconciler = Reconciler::new();
        let mut layout = LayoutTree::new();
        let before = reconciler.reconcile(
            Description::new::<Panel>().diff_key(Fingerprint(0)).child(
                Description::new::<Panel>()
                    .diff_key(Fingerprint(0))
                    .child(leaf(0)),
            ),
            &mut layout,
        )?;
        let child_node_before = before.nodes()[1].layout_node;

        let after = reconciler.reconcile(
            Description::new::<Panel>().diff_key(Fingerprint(0)).child(
                Description::new::<Image>()
                    .diff_key(Fingerprint(0))
                    .child(leaf(0)),
            ),
            &mut layout,
        )?;
        assert_eq!(
            after.nodes()[1].outcome,
            NodeOutcome::Rebuilt(RebuildReason::TypeMismatch)
        );
        assert_ne!(after.nodes()[1].layout_node, child_node_before);
        assert_eq!(
            after.nodes()[2].outcome,
            NodeOutcome::Rebuilt(RebuildReason::AncestorRebuilt),
            "a type mismatch rebuilds the whole subtree, never just the node"
        );
        // The root's own key was unchanged, but its child took a new node, so
        // it relinks rather than reusing.
        assert_eq!(
            after.nodes()[0].outcome,
            NodeOutcome::Rebuilt(RebuildReason::ChildrenChanged)
        );
        Ok(())
    }

    #[test]
    fn an_element_without_a_fingerprint_always_rebuilds_but_keeps_its_record()
    -> Result<(), ReconcileError> {
        let mut reconciler = Reconciler::new();
        let mut layout = LayoutTree::new();
        reconciler.reconcile(Description::new::<Panel>(), &mut layout)?;
        let plan = reconciler.reconcile(Description::new::<Panel>(), &mut layout)?;
        assert_eq!(
            plan.nodes()[0].outcome,
            NodeOutcome::Rebuilt(RebuildReason::NoDiffKey)
        );
        assert_eq!(
            reconciler.instances().len(),
            1,
            "no fingerprint is not the same mechanism as `.uncached()`"
        );
        Ok(())
    }

    #[test]
    fn an_always_dirty_key_rebuilds_every_frame() -> Result<(), ReconcileError> {
        let mut reconciler = Reconciler::new();
        let mut layout = LayoutTree::new();
        reconciler.reconcile(
            Description::new::<Panel>().diff_key(AlwaysDirty),
            &mut layout,
        )?;
        let plan = reconciler.reconcile(
            Description::new::<Panel>().diff_key(AlwaysDirty),
            &mut layout,
        )?;
        assert_eq!(
            plan.nodes()[0].outcome,
            NodeOutcome::Rebuilt(RebuildReason::KeyChanged)
        );
        Ok(())
    }

    #[test]
    fn the_kill_switch_reverts_to_unconditional_rebuild() -> Result<(), ReconcileError> {
        let mut reconciler = Reconciler::with_reconciliation_disabled();
        let mut layout = LayoutTree::new();
        reconciler.reconcile(tree(0), &mut layout)?;
        let plan = reconciler.reconcile(tree(0), &mut layout)?;
        assert_eq!(plan.stats().reused, 0);
        assert_eq!(plan.stats().rebuilt, 6);
        assert_eq!(plan.stats().layout_nodes_created, 6);
        assert!(reconciler.instances().is_empty());
        Ok(())
    }

    #[test]
    fn a_removed_subtree_is_swept() -> Result<(), ReconcileError> {
        let mut reconciler = Reconciler::new();
        let mut layout = LayoutTree::new();
        reconciler.reconcile(tree(0), &mut layout)?;
        assert_eq!(reconciler.instances().len(), 6);
        let plan = reconciler.reconcile(
            Description::new::<Panel>().diff_key(Fingerprint(0)),
            &mut layout,
        )?;
        assert_eq!(reconciler.instances().len(), 1);
        assert_eq!(plan.stats().instances_swept, 5);
        assert_eq!(plan.stats().layout_nodes_swept, 5);
        assert_eq!(layout.live_node_count(), 1);
        Ok(())
    }
}
