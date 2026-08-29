//! `ReconcileKey` — the cheap, owned fingerprint ambient reconciliation
//! compares. §6.2's invariant is enforced against this file's default.
//! See docs/gpu-native-architecture.md §4.0, §6.2, and R-N §2.3/§2.4.
//!
//! # What a key may hold, and what it must not
//!
//! A key is a *small plain value*, holding only what its element chose to
//! snapshot. It must never hold the description itself: a description's
//! children are arena-allocated and the arena is cleared every frame, so
//! retaining one verbatim across frames retains dangling state. R-N §2.4 sets
//! the rest of the rules and they are unchanged here:
//!
//! - **Listeners are never compared.** Closures are not comparable and do not
//!   need to be — a listener affects neither layout nor paint output, so it is
//!   swapped in unconditionally and contributes no invalidation.
//! - **Style is split by what it affects**, so a hover colour change reports
//!   `DISPLAY` and not `LAYOUT`.
//! - **The axes come from the comparison**, never from the call site that
//!   raised the change.
//!
//! # The permissive default stays permissive
//!
//! An element with no key at all reconciles to nothing: full rebuild, zero
//! savings, zero risk. That is the correct, unavoidable default for a
//! third-party element whose purity cannot be proven from outside, and §6.2
//! keeps it. What §6.2 raises is the bar for what ships *inside*
//! `wgpui-widgets`: every first-party element type implements a key. Phase 1
//! defines the trait; the widgets crate is where that standing rule is
//! checked.

use crate::invalidation::axes::Invalidation;
use std::any::Any;

/// A cheap, owned, arena-free fingerprint of one frame's description of an
/// element, compared against the previous frame's fingerprint for the same
/// [`crate::reconcile::instance::InstanceKey`].
pub trait ReconcileKey: Any {
    /// Compare against last frame's key for the same instance.
    ///
    /// [`Invalidation::empty`] means the element is fully reusable —
    /// `prepaint`, `paint`, and its retained layout node all stand.
    ///
    /// Implementations must downcast `previous` to `Self` first and treat a
    /// failed downcast as [`Invalidation::all`]: a position that held a
    /// different element type last frame is exactly the case that must never
    /// be reused, and a failed downcast is how that shows up here.
    fn compare(&self, previous: &dyn ReconcileKey) -> Invalidation;

    /// Downcasting helper. `dyn ReconcileKey: Any` alone is not enough to call
    /// `Any`'s own methods through a `&dyn ReconcileKey`.
    fn as_any(&self) -> &dyn Any;
}

/// The downcast-or-differ comparison every key implementation needs, written
/// once.
///
/// Compares `current` against `previous` by equality when the types match, and
/// reports [`Invalidation::all`] when they do not — which is the type-mismatch
/// rule R-N §2.2 describes, enforced in one place rather than re-derived by
/// each implementor.
pub fn compare_by_equality<T>(
    current: &T,
    previous: &dyn ReconcileKey,
    when_different: Invalidation,
) -> Invalidation
where
    T: PartialEq + 'static,
{
    match previous.as_any().downcast_ref::<T>() {
        Some(previous) if previous == current => Invalidation::empty(),
        Some(_) => when_different,
        None => Invalidation::all(),
    }
}

/// A key that compares equal to nothing, including another of itself.
///
/// This is what an element with no meaningful fingerprint uses to say "assume
/// changed" while still participating in the instance tree — distinct from
/// `.uncached()` (§4.2), which removes the instance record entirely. Useful on
/// its own for content whose description genuinely is opaque, and used by the
/// test suite to build always-dirty nodes without reaching for the scope flag.
#[derive(Copy, Clone, Debug, Default)]
pub struct AlwaysDirty;

impl ReconcileKey for AlwaysDirty {
    fn compare(&self, _previous: &dyn ReconcileKey) -> Invalidation {
        Invalidation::all()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(PartialEq, Debug)]
    struct PanelKey {
        width: u32,
        color: u32,
    }

    impl ReconcileKey for PanelKey {
        fn compare(&self, previous: &dyn ReconcileKey) -> Invalidation {
            match previous.as_any().downcast_ref::<PanelKey>() {
                Some(previous) => {
                    let mut axes = Invalidation::empty();
                    if previous.width != self.width {
                        axes |= Invalidation::LAYOUT;
                    }
                    if previous.color != self.color {
                        axes |= Invalidation::DISPLAY;
                    }
                    axes
                }
                None => Invalidation::all(),
            }
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[derive(PartialEq, Debug)]
    struct ImageKey(u64);

    impl ReconcileKey for ImageKey {
        fn compare(&self, previous: &dyn ReconcileKey) -> Invalidation {
            compare_by_equality(self, previous, Invalidation::DISPLAY)
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[test]
    fn an_unchanged_key_reports_no_invalidation() {
        let current = PanelKey {
            width: 10,
            color: 1,
        };
        let previous = PanelKey {
            width: 10,
            color: 1,
        };
        assert_eq!(current.compare(&previous), Invalidation::empty());
    }

    #[test]
    fn the_axes_reported_follow_which_field_changed() {
        let base = PanelKey {
            width: 10,
            color: 1,
        };
        let recoloured = PanelKey {
            width: 10,
            color: 2,
        };
        let resized = PanelKey {
            width: 11,
            color: 1,
        };
        assert_eq!(recoloured.compare(&base), Invalidation::DISPLAY);
        assert_eq!(resized.compare(&base), Invalidation::LAYOUT);
    }

    #[test]
    fn a_type_mismatch_is_always_a_full_invalidation() {
        let panel = PanelKey {
            width: 10,
            color: 1,
        };
        let image = ImageKey(7);
        assert_eq!(
            panel.compare(&image),
            Invalidation::all(),
            "a position that held a different element type must never be reused"
        );
        assert_eq!(image.compare(&panel), Invalidation::all());
    }

    #[test]
    fn compare_by_equality_matches_a_hand_written_downcast() {
        assert_eq!(ImageKey(1).compare(&ImageKey(1)), Invalidation::empty());
        assert_eq!(ImageKey(1).compare(&ImageKey(2)), Invalidation::DISPLAY);
    }

    #[test]
    fn always_dirty_never_compares_clean_even_against_itself() {
        assert_eq!(AlwaysDirty.compare(&AlwaysDirty), Invalidation::all());
    }
}
