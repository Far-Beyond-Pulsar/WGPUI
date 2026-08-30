//! Element state, keyed by `(path, TypeId)` — the mechanism `.uncached()`
//! must not touch. See docs/gpu-native-architecture.md §4.2 and R-N §2.1.
//!
//! Not in §3.1's literal file map — a deliberate addition, recorded in
//! `docs/phase-1-results.md`. R-N §2.1's table lists State as "already
//! retained... unchanged" by any of Pillar I's mechanics, so neither R-N nor
//! §3.1 gives it a file: in the legacy backend it lives inside `window.rs`'s
//! frame bookkeeping. §4.2 makes it load-bearing anyway — "a slider or text
//! input living inside an `.uncached()` panel keeps its interactive state
//! exactly as it would anywhere else" is a claim that needs a mechanism to be
//! true *of*, and Phase 1's third gate is precisely a test that the claim
//! holds. Giving it its own file next to the flag it must be independent of
//! is what makes the independence legible.
//!
//! # The decoupling, stated as code
//!
//! State is addressed by [`StateKey`] — a hash of the element's path and the
//! state's type. Reconciliation is addressed by
//! [`crate::reconcile::instance::InstanceKey`] — a hash of the element's path.
//! Both derive from the same path, and *neither reads the other*. There is no
//! call from this module into `instance`, none from `instance` into this one,
//! and the reconciler stores state keys on its plan without ever consulting
//! them. That is why suppressing reconciliation cannot suppress state: the two
//! mechanisms share an input and nothing else.

use crate::reconcile::description::ElementId;
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// One element's state *scope*: everything its state addresses have in
/// common, which is its path and nothing else.
///
/// Separated from [`StateKey`] because the reconciler can derive a scope for
/// every element it visits without knowing what state types that element will
/// ask for, and because carrying a scope on a
/// [`crate::reconcile::plan::PlannedNode`] costs eight bytes where carrying
/// the path itself would cost an allocation per element per frame.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StateScope(u64);

impl StateScope {
    /// Derive the scope for the element at `path`.
    pub fn from_path(path: &[ElementId]) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        path.hash(&mut hasher);
        StateScope(hasher.finish() | 1)
    }

    /// The raw value.
    pub const fn as_raw(self) -> u64 {
        self.0
    }
}

/// The address of one piece of element state: which element, and which type of
/// state it is.
///
/// Typed as well as located, because one element legitimately holds several
/// unrelated pieces of state (a scroll offset and a hover flag, say) and they
/// must not alias.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StateKey(u64);

impl StateKey {
    /// Derive the key for state of type `T` within `scope`.
    pub fn new<T: 'static>(scope: StateScope) -> Self {
        Self::for_type(scope, TypeId::of::<T>())
    }

    /// Derive the key for state of a dynamically-known type within `scope`.
    pub fn for_type(scope: StateScope, type_id: TypeId) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        scope.as_raw().hash(&mut hasher);
        type_id.hash(&mut hasher);
        StateKey(hasher.finish() | 1)
    }

    /// The raw value.
    pub const fn as_raw(self) -> u64 {
        self.0
    }
}

struct StateEntry {
    value: Box<dyn Any>,
    last_visited_frame: u64,
}

/// Every element's retained state, addressed by [`StateKey`].
#[derive(Default)]
pub struct ElementStateStore {
    entries: HashMap<StateKey, StateEntry>,
}

impl std::fmt::Debug for ElementStateStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ElementStateStore")
            .field("entries", &self.entries.len())
            .finish()
    }
}

impl ElementStateStore {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Read and write the state at `key`, creating it from `initialise` on
    /// first use, and marking it visited in `frame`.
    ///
    /// This is the `use_state` shape: an element asks for its state, gets a
    /// mutable borrow of it, and the store records that the element is still
    /// alive. Returns `None` only when an entry exists under this key holding
    /// a different type, which a correctly-derived [`StateKey`] makes
    /// unreachable — the key hashes the type.
    pub fn with_state<T: 'static, R>(
        &mut self,
        key: StateKey,
        frame: u64,
        initialise: impl FnOnce() -> T,
        access: impl FnOnce(&mut T) -> R,
    ) -> Option<R> {
        let entry = self.entries.entry(key).or_insert_with(|| StateEntry {
            value: Box::new(initialise()),
            last_visited_frame: frame,
        });
        entry.last_visited_frame = frame;
        let value = entry.value.downcast_mut::<T>()?;
        Some(access(value))
    }

    /// Read the state at `key` without creating or touching it.
    pub fn peek<T: 'static>(&self, key: StateKey) -> Option<&T> {
        self.entries.get(&key)?.value.downcast_ref::<T>()
    }

    /// Whether any state is retained at `key`.
    pub fn contains(&self, key: StateKey) -> bool {
        self.entries.contains_key(&key)
    }

    /// How many entries are retained.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no state is retained.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Drop every entry not visited in `frame`.
    ///
    /// Deliberately driven by the *element's* visit, not by reconciliation's
    /// outcome: an element inside an `.uncached()` subtree visits its state
    /// exactly as any other element does, so it survives this sweep for the
    /// same reason a reconciled element's does.
    pub fn sweep(&mut self, frame: u64) -> usize {
        let before = self.entries.len();
        self.entries
            .retain(|_, entry| entry.last_visited_frame == frame);
        before - self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct ScrollOffset(f32);

    #[derive(Debug, PartialEq)]
    struct Focused(bool);

    fn scope(name: &str) -> StateScope {
        StateScope::from_path(&[ElementId::from(name), ElementId::Slot(0)])
    }

    #[test]
    fn state_survives_across_frames_when_the_element_keeps_visiting_it() {
        let mut store = ElementStateStore::new();
        let key = StateKey::new::<ScrollOffset>(scope("panel"));
        assert_eq!(
            store.with_state(
                key,
                0,
                || ScrollOffset(0.0),
                |offset| {
                    offset.0 = 12.0;
                    offset.0
                }
            ),
            Some(12.0)
        );
        assert_eq!(store.sweep(0), 0);
        assert_eq!(
            store.with_state(key, 1, || ScrollOffset(0.0), |offset| offset.0),
            Some(12.0)
        );
    }

    #[test]
    fn two_state_types_on_one_element_do_not_alias() {
        let mut store = ElementStateStore::new();
        let element = scope("input");
        let offset = StateKey::new::<ScrollOffset>(element);
        let focus = StateKey::new::<Focused>(element);
        assert_ne!(offset, focus);
        store.with_state(offset, 0, || ScrollOffset(3.0), |_| ());
        store.with_state(focus, 0, || Focused(true), |_| ());
        assert_eq!(store.peek::<ScrollOffset>(offset), Some(&ScrollOffset(3.0)));
        assert_eq!(store.peek::<Focused>(focus), Some(&Focused(true)));
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn the_same_state_type_on_two_elements_does_not_alias() {
        let first = StateKey::new::<Focused>(scope("a"));
        let second = StateKey::new::<Focused>(scope("b"));
        assert_ne!(first, second);
    }

    #[test]
    fn unvisited_state_is_swept() {
        let mut store = ElementStateStore::new();
        let key = StateKey::new::<Focused>(scope("gone"));
        store.with_state(key, 0, || Focused(true), |_| ());
        assert_eq!(store.sweep(1), 1);
        assert!(!store.contains(key));
        assert!(store.is_empty());
    }

    #[test]
    fn peeking_a_mismatched_type_reports_none_rather_than_panicking() {
        let mut store = ElementStateStore::new();
        let key = StateKey::new::<Focused>(scope("panel"));
        store.with_state(key, 0, || Focused(false), |_| ());
        assert!(store.peek::<ScrollOffset>(key).is_none());
    }
}
