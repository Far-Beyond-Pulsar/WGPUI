//! `ElementInstance`, `InstanceKey` — the retained side of R-N §2.1's split,
//! ambient per §4.0. See docs/gpu-native-architecture.md §4.0.
//!
//! # Ambient means: no fence, anywhere in this file
//!
//! The legacy `ElementInstance` lives in `Layer::instances`, and its own
//! module doc states the consequence directly: "Content with no `.layer()`
//! ancestor gets no benefit from this phase; it rebuilds every frame exactly
//! as it does today." That fence is what constraint 5 (§0) rejects and §4.0
//! removes. Here instances live in one window-wide table
//! ([`InstanceTable`]) addressed by path, with no layer, boundary, or
//! subtree anywhere in the key — so there is no place a fence *could* be
//! reintroduced without changing this type.
//!
//! Memory is bounded by the same mark-and-sweep the legacy layer eviction
//! provided, applied window-wide instead: an instance not visited in a frame
//! is swept at the end of it.

use crate::reconcile::description::ElementId;
use crate::reconcile::diff_key::ReconcileKey;
use std::any::TypeId;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use wgpui_layout::taffy_tree::LayoutNodeId;

/// The stable, cross-frame address of one retained element.
///
/// Derived from the whole path of [`ElementId`]s down to the element, so two
/// elements at the same slot under different parents never collide. Hashing
/// the path rather than storing it is what keeps the key `Copy` and cheap to
/// compare, which matters because every element in the window is looked up by
/// one of these every frame.
///
/// Deliberately **not** typed by the element's Rust type: a position can hold
/// one type this frame and another the next, and the key alone cannot and
/// should not tell them apart. That is
/// [`crate::reconcile::diff_key::ReconcileKey::compare`]'s job, via a failed
/// downcast, so a type change is a rebuild rather than a key collision.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InstanceKey(u64);

impl InstanceKey {
    /// Derive the key for the element addressed by `path`.
    pub fn from_path(path: &[ElementId]) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        path.hash(&mut hasher);
        // Reserve 0 so a defaulted key is never mistaken for a live instance.
        InstanceKey(hasher.finish() | 1)
    }

    /// Wrap a raw value. Intended for tests.
    pub const fn from_raw(raw: u64) -> Self {
        InstanceKey(raw)
    }

    /// The raw value.
    pub const fn as_raw(self) -> u64 {
        self.0
    }
}

/// Retained state for one element, from the last frame it was visited.
pub struct ElementInstance {
    /// Last frame's fingerprint, compared against this frame's fresh one.
    diff_key: Option<Box<dyn ReconcileKey>>,
    /// The element's Rust type. A mismatch forces a subtree rebuild rather
    /// than reusing a `Div`'s record as an `Img`'s.
    type_id: TypeId,
    /// This element's retained layout node. Valid for as long as this record
    /// survives: a rebuild that replaces the record either keeps the node
    /// (style/children updated in place) or creates a fresh one and leaves the
    /// old to be swept.
    layout_node: LayoutNodeId,
    /// The layout nodes of this element's children, in order. A child whose
    /// node changed forces this element to relink even when its own
    /// fingerprint compared clean — the one cross-element dependency
    /// reconciliation genuinely has.
    child_nodes: Vec<LayoutNodeId>,
    /// The instances of this element's children, in order.
    children: Vec<InstanceKey>,
    /// The last frame this record was visited, for the sweep.
    last_visited_frame: u64,
}

impl std::fmt::Debug for ElementInstance {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ElementInstance")
            .field("has_diff_key", &self.diff_key.is_some())
            .field("layout_node", &self.layout_node)
            .field("children", &self.children.len())
            .field("last_visited_frame", &self.last_visited_frame)
            .finish()
    }
}

impl ElementInstance {
    /// This element's retained layout node.
    pub fn layout_node(&self) -> LayoutNodeId {
        self.layout_node
    }

    /// This element's children, in order.
    pub fn children(&self) -> &[InstanceKey] {
        &self.children
    }

    /// The element's Rust type as of the last frame it was visited.
    pub fn type_id(&self) -> TypeId {
        self.type_id
    }

    /// Last frame's fingerprint, if the element supplied one.
    pub fn diff_key(&self) -> Option<&dyn ReconcileKey> {
        self.diff_key.as_deref()
    }

    /// The last frame this record was visited.
    pub fn last_visited_frame(&self) -> u64 {
        self.last_visited_frame
    }
}

/// Every retained element instance in the window, addressed by
/// [`InstanceKey`].
#[derive(Debug, Default)]
pub struct InstanceTable {
    instances: HashMap<InstanceKey, ElementInstance>,
}

impl InstanceTable {
    /// An empty table.
    pub fn new() -> Self {
        Self::default()
    }

    /// The record for `key`, if one is retained.
    pub fn get(&self, key: InstanceKey) -> Option<&ElementInstance> {
        self.instances.get(&key)
    }

    /// Whether a record is retained for `key`.
    pub fn contains(&self, key: InstanceKey) -> bool {
        self.instances.contains_key(&key)
    }

    /// How many records are retained. Phase 1's third gate reads this: an
    /// `.uncached()` subtree must add nothing to it.
    pub fn len(&self) -> usize {
        self.instances.len()
    }

    /// Whether no records are retained.
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    /// Insert or replace the record for `key`.
    pub fn store(
        &mut self,
        key: InstanceKey,
        type_id: TypeId,
        diff_key: Option<Box<dyn ReconcileKey>>,
        layout_node: LayoutNodeId,
        child_nodes: Vec<LayoutNodeId>,
        children: Vec<InstanceKey>,
        frame: u64,
    ) {
        self.instances.insert(
            key,
            ElementInstance {
                diff_key,
                type_id,
                layout_node,
                child_nodes,
                children,
                last_visited_frame: frame,
            },
        );
    }

    /// Mark an existing record as visited this frame without otherwise
    /// changing it — what a fully reused element does.
    pub fn touch(&mut self, key: InstanceKey, frame: u64) -> bool {
        match self.instances.get_mut(&key) {
            Some(instance) => {
                instance.last_visited_frame = frame;
                true
            }
            None => false,
        }
    }

    /// Whether the retained child-node list for `key` still matches
    /// `child_nodes`.
    ///
    /// A clean fingerprint is not on its own enough to reuse an element: if a
    /// child had to be rebuilt onto a fresh layout node, this element's own
    /// node still has to be relinked to it.
    pub fn child_nodes_match(&self, key: InstanceKey, child_nodes: &[LayoutNodeId]) -> bool {
        match self.instances.get(&key) {
            Some(instance) => instance.child_nodes == child_nodes,
            None => false,
        }
    }

    /// Drop the record for `key` and every record beneath it.
    ///
    /// Used both by `.uncached()` (§4.2 — a subtree that stops being
    /// reconciled must stop holding records for anything inside it) and by a
    /// type mismatch (R-N §2.2 — a subtree rebuild starts from nothing).
    /// Returns how many records were dropped.
    pub fn remove_subtree(&mut self, key: InstanceKey) -> usize {
        let Some(instance) = self.instances.remove(&key) else {
            return 0;
        };
        let mut removed = 1;
        for child in instance.children {
            removed += self.remove_subtree(child);
        }
        removed
    }

    /// Drop every record not visited in `frame`, and every record beneath one.
    ///
    /// The window-wide equivalent of the legacy backend's per-layer eviction:
    /// same mark-and-sweep, same bounding argument, no layer required.
    /// Returns how many records were dropped.
    pub fn sweep(&mut self, frame: u64) -> usize {
        let stale: Vec<InstanceKey> = self
            .instances
            .iter()
            .filter(|(_, instance)| instance.last_visited_frame != frame)
            .map(|(key, _)| *key)
            .collect();
        let mut removed = 0;
        for key in stale {
            removed += self.remove_subtree(key);
        }
        removed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reconcile::diff_key::AlwaysDirty;

    struct Panel;

    fn node(raw: u64) -> LayoutNodeId {
        LayoutNodeId::from_raw(raw)
    }

    fn store_leaf(table: &mut InstanceTable, key: InstanceKey, frame: u64) {
        table.store(
            key,
            TypeId::of::<Panel>(),
            Some(Box::new(AlwaysDirty)),
            node(0),
            Vec::new(),
            Vec::new(),
            frame,
        );
    }

    #[test]
    fn instance_keys_are_stable_across_frames() {
        let path = [ElementId::from("root"), ElementId::Slot(0)];
        assert_eq!(InstanceKey::from_path(&path), InstanceKey::from_path(&path));
    }

    #[test]
    fn instance_keys_distinguish_positional_siblings() {
        let first = InstanceKey::from_path(&[ElementId::from("list"), ElementId::Slot(0)]);
        let second = InstanceKey::from_path(&[ElementId::from("list"), ElementId::Slot(1)]);
        assert_ne!(first, second);
    }

    #[test]
    fn the_same_slot_under_a_different_parent_is_a_different_instance() {
        let under_a = InstanceKey::from_path(&[ElementId::from("a"), ElementId::Slot(0)]);
        let under_b = InstanceKey::from_path(&[ElementId::from("b"), ElementId::Slot(0)]);
        assert_ne!(under_a, under_b);
    }

    #[test]
    fn instance_keys_are_never_zero() {
        for path in [
            vec![ElementId::Slot(0)],
            vec![ElementId::from("a")],
            Vec::new(),
        ] {
            assert_ne!(InstanceKey::from_path(&path).as_raw(), 0);
        }
    }

    #[test]
    fn an_empty_table_retains_nothing() {
        let table = InstanceTable::new();
        assert!(table.is_empty());
        assert_eq!(table.len(), 0);
        assert!(table.get(InstanceKey::from_raw(1)).is_none());
    }

    #[test]
    fn removing_a_subtree_removes_its_descendants() {
        let mut table = InstanceTable::new();
        let root = InstanceKey::from_raw(1);
        let child = InstanceKey::from_raw(3);
        let grandchild = InstanceKey::from_raw(5);
        store_leaf(&mut table, grandchild, 0);
        table.store(
            child,
            TypeId::of::<Panel>(),
            None,
            node(0),
            Vec::new(),
            vec![grandchild],
            0,
        );
        table.store(
            root,
            TypeId::of::<Panel>(),
            None,
            node(0),
            Vec::new(),
            vec![child],
            0,
        );
        assert_eq!(table.len(), 3);
        assert_eq!(table.remove_subtree(root), 3);
        assert!(table.is_empty());
    }

    #[test]
    fn the_sweep_drops_records_not_visited_this_frame() {
        let mut table = InstanceTable::new();
        let kept = InstanceKey::from_raw(1);
        let dropped = InstanceKey::from_raw(3);
        store_leaf(&mut table, kept, 1);
        store_leaf(&mut table, dropped, 0);
        assert_eq!(table.sweep(1), 1);
        assert!(table.contains(kept));
        assert!(!table.contains(dropped));
    }

    #[test]
    fn touching_a_record_keeps_it_through_the_sweep() {
        let mut table = InstanceTable::new();
        let key = InstanceKey::from_raw(1);
        store_leaf(&mut table, key, 0);
        assert!(table.touch(key, 1));
        assert_eq!(table.sweep(1), 0);
        assert!(table.contains(key));
        assert!(!table.touch(InstanceKey::from_raw(99), 1));
    }
}
