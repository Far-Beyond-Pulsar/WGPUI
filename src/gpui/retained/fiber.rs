use std::sync::Arc;

use collections::FxHashMap;

use super::dirty::DirtyFlags;
use crate::{ElementId, GlobalElementId};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct FiberId(u64);

#[derive(Default)]
pub(crate) struct FiberTree {
    next_id: u64,
    parents: FxHashMap<FiberId, FiberId>,
    dirty: FxHashMap<FiberId, DirtyFlags>,
    fibers_by_element: FxHashMap<GlobalElementId, FiberId>,
}

impl FiberTree {
    #[cfg(test)]
    pub(crate) fn create_root(&mut self) -> FiberId {
        self.next_id = self.next_id.saturating_add(1);
        let id = FiberId(self.next_id);
        self.dirty
            .insert(id, DirtyFlags::LAYOUT | DirtyFlags::PAINT);
        id
    }

    #[cfg(test)]
    pub(crate) fn create_child(&mut self, parent: FiberId) -> FiberId {
        self.next_id = self.next_id.saturating_add(1);
        let id = FiberId(self.next_id);
        self.parents.insert(id, parent);
        self.dirty
            .insert(id, DirtyFlags::LAYOUT | DirtyFlags::PAINT);
        self.mark_dirty(parent, DirtyFlags::DESCENDANT);
        id
    }

    pub(crate) fn dirty_flags(&self, id: FiberId) -> DirtyFlags {
        self.dirty.get(&id).copied().unwrap_or_default()
    }

    pub(crate) fn mark_dirty(&mut self, id: FiberId, flags: DirtyFlags) {
        self.dirty.entry(id).or_default().insert(flags);
        let mut current = id;
        // Bound the ancestor walk by the number of parent links. A valid tree path
        // to the root cannot traverse more edges than exist, so this never cuts a
        // legitimate walk short, while guaranteeing termination if a caller ever
        // introduces a self- or cyclic parent link.
        for _ in 0..self.parents.len() {
            let Some(parent) = self.parents.get(&current).copied() else {
                break;
            };
            if parent == current {
                break;
            }
            self.dirty
                .entry(parent)
                .or_default()
                .insert(DirtyFlags::DESCENDANT);
            current = parent;
        }
    }

    #[cfg(test)]
    pub(crate) fn clear_dirty(&mut self, id: FiberId) {
        self.dirty.insert(id, DirtyFlags::default());
    }

    pub(crate) fn fiber_for_element(&self, global_id: &GlobalElementId) -> Option<FiberId> {
        self.fibers_by_element.get(global_id).copied()
    }

    #[cfg(test)]
    pub(crate) fn fiber_ids(&self) -> Vec<FiberId> {
        let mut fibers = self.fibers_by_element.values().copied().collect::<Vec<_>>();
        fibers.sort_unstable();
        fibers
    }

    /// Number of id-bearing elements currently mirrored into the tree. Cheap and
    /// non-allocating (unlike [`Self::fiber_ids`]); read once per frame for
    /// diagnostics.
    pub(crate) fn fiber_count(&self) -> usize {
        self.fibers_by_element.len()
    }

    /// Number of mirrored fibers whose dirty flags are not clean after the last
    /// reconciliation. Non-allocating; read once per frame for diagnostics.
    pub(crate) fn dirty_fiber_count(&self) -> usize {
        self.fibers_by_element
            .values()
            .filter(|fiber_id| !self.dirty_flags(**fiber_id).is_clean())
            .count()
    }

    pub(crate) fn reconcile_element(
        &mut self,
        global_id: &GlobalElementId,
        previous: &FiberTree,
    ) -> FiberId {
        if let Some(fiber_id) = self.fiber_for_element(global_id) {
            return fiber_id;
        }

        // Seed the id high-water mark from the previous tree before allocating any
        // fresh id. Reused ids come from `previous` and can be any value up to
        // `previous.next_id`; without this, a freshly allocated id (which counts up
        // from this tree's own `next_id`) can collide with a not-yet-reconciled
        // reused id, mapping two distinct elements onto the same `FiberId`.
        self.next_id = self.next_id.max(previous.next_id);

        let reused_fiber = previous.fiber_for_element(global_id);
        let fiber_id = reused_fiber.unwrap_or_else(|| self.allocate_fiber());
        self.next_id = self.next_id.max(fiber_id.0);
        self.fibers_by_element.insert(global_id.clone(), fiber_id);

        if let Some(parent_id) = parent_global_id(global_id) {
            let parent_fiber = self.reconcile_element(&parent_id, previous);
            self.parents.insert(fiber_id, parent_fiber);
            if reused_fiber.is_none() {
                self.mark_dirty(parent_fiber, DirtyFlags::DESCENDANT);
            }
        }

        self.dirty.entry(fiber_id).or_insert_with(|| {
            if reused_fiber.is_some() {
                DirtyFlags::default()
            } else {
                DirtyFlags::LAYOUT | DirtyFlags::PAINT
            }
        });
        fiber_id
    }

    /// Cheaply carry a reused view's subtree forward from the previous frame.
    ///
    /// When a view's prepaint is replayed from cache its element tree is not
    /// re-walked, so its fibers would otherwise be dropped. Rather than
    /// re-reconcile each element (which re-derives parent id paths and heap
    /// allocates per element), copy the fiber id, parent link, and clean dirty
    /// state straight from the previous frame's tree by id lookup. This keeps the
    /// tree a complete, stable mirror at near-zero per-frame cost.
    pub(crate) fn carry_forward(&mut self, ids: &[GlobalElementId], previous: &FiberTree) {
        self.next_id = self.next_id.max(previous.next_id);
        for global_id in ids {
            if self.fibers_by_element.contains_key(global_id) {
                continue;
            }
            let Some(fiber_id) = previous.fiber_for_element(global_id) else {
                continue;
            };
            self.fibers_by_element.insert(global_id.clone(), fiber_id);
            if let Some(parent) = previous.parents.get(&fiber_id).copied() {
                self.parents.insert(fiber_id, parent);
            }
            // Reused unchanged from a clean previous frame -> clean.
            self.dirty.entry(fiber_id).or_default();
            self.next_id = self.next_id.max(fiber_id.0);
        }
    }

    fn allocate_fiber(&mut self) -> FiberId {
        self.next_id = self.next_id.saturating_add(1);
        FiberId(self.next_id)
    }
}

fn parent_global_id(global_id: &GlobalElementId) -> Option<GlobalElementId> {
    let parent_len = global_id.0.len().checked_sub(1)?;
    if parent_len == 0 {
        return None;
    }
    let parent_path: Arc<[ElementId]> = Arc::from(&global_id.0[..parent_len]);
    Some(GlobalElementId(parent_path))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fiber_tree_propagates_dirty_descendants_to_ancestors() {
        let mut tree = FiberTree::default();
        let root = tree.create_root();
        tree.clear_dirty(root);

        let child = tree.create_child(root);
        tree.clear_dirty(root);
        tree.clear_dirty(child);

        tree.mark_dirty(child, DirtyFlags::PAINT);

        // The child is dirty and the dirt propagates to the (previously clean)
        // root as a dirty descendant.
        assert!(!tree.dirty_flags(child).is_clean());
        assert!(!tree.dirty_flags(root).is_clean());
    }

    #[test]
    fn fiber_tree_reuses_element_fibers_across_reorder_and_drops_removed_paths() {
        let first = GlobalElementId(Arc::from([ElementId::Name("first".into())]));
        let second = GlobalElementId(Arc::from([ElementId::Name("second".into())]));
        let mut previous = FiberTree::default();
        let first_fiber = previous.reconcile_element(&first, &FiberTree::default());
        let second_fiber = previous.reconcile_element(&second, &FiberTree::default());

        let mut next = FiberTree::default();
        assert_eq!(next.reconcile_element(&second, &previous), second_fiber);
        assert_eq!(next.reconcile_element(&first, &previous), first_fiber);

        let mut removed_next = FiberTree::default();
        assert_eq!(
            removed_next.reconcile_element(&second, &previous),
            second_fiber
        );
        assert_eq!(removed_next.fiber_for_element(&first), None);
    }

    #[test]
    fn mark_dirty_terminates_even_with_a_cyclic_parent_link() {
        let mut tree = FiberTree::default();
        let a = tree.create_root();
        let b = tree.create_child(a);
        // Force a cycle that the normal API never produces, to prove the ancestor
        // walk is bounded and cannot hang.
        tree.parents.insert(a, b);

        tree.mark_dirty(b, DirtyFlags::PAINT);

        assert!(!tree.dirty_flags(b).is_clean());
        assert!(!tree.dirty_flags(a).is_clean());
    }

    #[test]
    fn fiber_tree_never_reuses_an_id_for_a_freshly_inserted_sibling() {
        let first = GlobalElementId(Arc::from([ElementId::Name("first".into())]));
        let second = GlobalElementId(Arc::from([ElementId::Name("second".into())]));
        let inserted = GlobalElementId(Arc::from([ElementId::Name("inserted".into())]));

        let mut previous = FiberTree::default();
        let first_fiber = previous.reconcile_element(&first, &FiberTree::default());
        let second_fiber = previous.reconcile_element(&second, &FiberTree::default());

        // In the next frame a brand-new element is reconciled BEFORE the reused
        // `first`. The freshly allocated id must not collide with `first`'s reused
        // id, otherwise two distinct elements would share a single `FiberId`.
        let mut next = FiberTree::default();
        let inserted_fiber = next.reconcile_element(&inserted, &previous);
        let reused_first = next.reconcile_element(&first, &previous);
        let reused_second = next.reconcile_element(&second, &previous);

        assert_eq!(reused_first, first_fiber);
        assert_eq!(reused_second, second_fiber);
        assert_ne!(inserted_fiber, first_fiber);
        assert_ne!(inserted_fiber, second_fiber);
        assert_ne!(inserted_fiber, reused_first);
        assert_ne!(inserted_fiber, reused_second);
    }

    #[test]
    fn reconcile_seeds_new_fibers_dirty_and_reused_fibers_clean() {
        let element = GlobalElementId(Arc::from([ElementId::Name("element".into())]));

        let mut first = FiberTree::default();
        let fiber = first.reconcile_element(&element, &FiberTree::default());
        assert!(
            !first.dirty_flags(fiber).is_clean(),
            "a freshly reconciled fiber must be dirty so its first frame paints"
        );
        assert_eq!(first.fiber_count(), 1);
        assert_eq!(first.dirty_fiber_count(), 1);

        let mut second = FiberTree::default();
        let reused = second.reconcile_element(&element, &first);
        assert_eq!(reused, fiber);
        assert!(
            second.dirty_flags(reused).is_clean(),
            "a fiber reused from a clean previous frame must stay clean"
        );
        assert_eq!(second.dirty_fiber_count(), 0);
    }
}
