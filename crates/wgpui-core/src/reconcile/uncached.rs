//! `.uncached()` — the scope flag threaded through the reconcile walk.
//! See docs/gpu-native-architecture.md §4.2.
//!
//! Ambient reconciliation (§4.0) is a bet that diffing pays for itself, and
//! for almost all UI content it does. The bet fails for one identifiable
//! shape: a subtree whose content is guaranteed to differ every single frame —
//! a live telemetry HUD, an audio waveform, a per-frame debug overlay. For
//! that content `diff_key` comparison is not "usually free, occasionally
//! expensive," it is unconditionally wasted, and the framework has been
//! holding a retained record and a fingerprint per element for a comparison
//! that will never once succeed.
//!
//! # What this is, mechanically
//!
//! A depth counter, pushed on entering an element that declared itself
//! uncached and popped on leaving it — the same shape as the content-mask and
//! text-style stacks the legacy `window.rs` already threads through its draw
//! walk. Nothing about it is new machinery; §4.2's own framing is that it
//! makes an existing code path (the unconditional rebuild every element took
//! before reconciliation existed) selectable on purpose.
//!
//! # What it does not touch, checked rather than asserted
//!
//! State (`use_state`, focus, tab stops) is keyed by `(path, TypeId)` and
//! lives in [`crate::reconcile::state::ElementStateStore`]. This module does
//! not import it, reference it, or know it exists. Occlusion culling and the
//! patch protocol are likewise untouched: an uncached subtree emits ordinary
//! patches every frame (always a full replace, never a delta), and nothing
//! downstream has a way to tell, or a reason to care, whether a patch arrived
//! because a diff proved change or because diffing was skipped.

/// Tracks whether the reconcile walk is currently inside an `.uncached()`
/// subtree.
///
/// A counter rather than a boolean because uncached subtrees nest: an
/// `.uncached()` panel inside an `.uncached()` overlay must not un-suppress
/// reconciliation when the inner one is popped.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct UncachedScope {
    depth: u32,
}

impl UncachedScope {
    /// A scope outside any uncached subtree.
    pub const fn new() -> Self {
        Self { depth: 0 }
    }

    /// Enter an element, suppressing reconciliation beneath it when
    /// `declared_uncached`.
    ///
    /// Returns the scope as it applies *inside* that element. Taking and
    /// returning by value rather than mutating in place is what makes the
    /// walk's restore-on-exit unmissable: a caller that forgets to pop simply
    /// never had a popped value to use.
    #[must_use]
    pub const fn enter(self, declared_uncached: bool) -> Self {
        if declared_uncached {
            Self {
                depth: self.depth.saturating_add(1),
            }
        } else {
            self
        }
    }

    /// Whether reconciliation is currently suppressed.
    pub const fn is_active(self) -> bool {
        self.depth > 0
    }

    /// How deep the current nesting of uncached subtrees is. Diagnostic only.
    pub const fn depth(self) -> u32 {
        self.depth
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_scope_suppresses_nothing() {
        assert!(!UncachedScope::new().is_active());
        assert_eq!(UncachedScope::new().depth(), 0);
    }

    #[test]
    fn entering_a_plain_element_leaves_the_scope_alone() {
        let outer = UncachedScope::new();
        assert_eq!(outer.enter(false), outer);
    }

    #[test]
    fn the_scope_applies_to_the_whole_subtree_not_just_the_declaring_element() {
        let outer = UncachedScope::new();
        let inside = outer.enter(true);
        assert!(inside.is_active());
        // Descending further without re-declaring keeps it active: this is the
        // property that makes `.uncached()` a subtree opt-out rather than a
        // per-element one.
        assert!(inside.enter(false).is_active());
    }

    #[test]
    fn nested_uncached_subtrees_do_not_un_suppress_on_the_inner_exit() {
        let outer = UncachedScope::new().enter(true);
        let inner = outer.enter(true);
        assert_eq!(inner.depth(), 2);
        // Leaving the inner scope means returning to `outer`, which is still
        // active.
        assert!(outer.is_active());
    }

    #[test]
    fn leaving_the_outermost_scope_restores_reconciliation() {
        let outside = UncachedScope::new();
        let inside = outside.enter(true);
        assert!(inside.is_active());
        assert!(!outside.is_active());
    }
}
