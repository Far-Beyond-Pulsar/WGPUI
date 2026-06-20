use collections::FxHashSet;

/// Stable identity for a view's cached scene chunk, so reused chunks keep the same
/// segment id across frames.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct SceneSegmentId(u64);

#[derive(Default)]
pub(crate) struct SceneSegmentPool {
    next_id: u64,
    live_segments: FxHashSet<SceneSegmentId>,
}

impl SceneSegmentPool {
    pub(crate) fn allocate(&mut self) -> SceneSegmentId {
        self.next_id = self.next_id.saturating_add(1);
        let id = SceneSegmentId(self.next_id);
        self.live_segments.insert(id);
        id
    }

    #[cfg(test)]
    pub(crate) fn remove(&mut self, id: SceneSegmentId) -> bool {
        self.live_segments.remove(&id)
    }

    pub(crate) fn retain(&mut self, id: SceneSegmentId) {
        self.next_id = self.next_id.max(id.0);
        self.live_segments.insert(id);
    }

    pub(crate) fn clear_live(&mut self) {
        self.live_segments.clear();
    }

    #[cfg(test)]
    pub(crate) fn contains(&self, id: SceneSegmentId) -> bool {
        self.live_segments.contains(&id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_segment_pool_allocates_and_removes_stable_ids() {
        let mut pool = SceneSegmentPool::default();
        let first = pool.allocate();
        let second = pool.allocate();

        assert_ne!(first, second);
        assert!(pool.contains(first));
        assert!(pool.contains(second));
        assert!(pool.remove(first));
        assert!(!pool.contains(first));
        assert!(pool.contains(second));

        pool.clear_live();
        assert!(!pool.contains(second));

        pool.retain(first);
        assert!(pool.contains(first));
    }
}
