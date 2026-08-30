//! `DivDiffKey` / `ReconcileKey` impl. See docs/gpu-native-architecture.md
//! §3.4, §6.2.
//!
//! # Why a `div`'s key is a whole style and `StyledText`'s is not
//!
//! `styled_text.rs`'s module doc quotes the legacy `TextDiffKey`'s own reasoning
//! for *not* splitting: almost every `TextStyle` field affects shaping, so a
//! finer split buys nothing. A `div`'s style is the opposite case and the legacy
//! `classify_style_change` (`src/elements/div.rs:2299`) already splits it, so
//! this key delegates the whole comparison to
//! [`crate::div::interactivity::style::classify_style_change`] rather than
//! comparing by equality: a hover recolour must report `DISPLAY` and not
//! `LAYOUT`, or every colour change in a real UI re-runs Taffy for its whole
//! subtree.
//!
//! # What the key does *not* hold
//!
//! Children. A `Description`'s children are reconciled as their own instances,
//! each against its own key, and folding a child's fingerprint into its parent's
//! would make a leaf's change rebuild the whole ancestry — the exact behaviour
//! ambient reconciliation exists to avoid. `diff_key.rs`'s module doc states
//! this as a rule ("a key must never hold the description itself"); this is the
//! first first-party element with children, so it is where the rule first has
//! something to bite on.

use crate::div::interactivity::style::{DivStyle, classify_style_change};
use std::any::Any;
use wgpui_core::invalidation::axes::Invalidation;
use wgpui_core::reconcile::diff_key::ReconcileKey;

/// The fingerprint a `div()` presents to ambient reconciliation.
#[derive(Clone, Debug, PartialEq)]
pub struct DivDiffKey {
    style: DivStyle,
    /// How many children this `div` described this frame.
    ///
    /// Not the children themselves — see this module's doc. The count alone is
    /// carried because a `div` whose child list changed length has a different
    /// Taffy child list, and the reconciler needs `LAYOUT` raised for that even
    /// when every surviving child is individually clean.
    child_count: usize,
    estimated_size: Option<[f32; 2]>,
}

impl DivDiffKey {
    /// The key for a `div` with `style` and `child_count` children.
    pub fn new(style: DivStyle, child_count: usize) -> DivDiffKey {
        Self::with_estimate(style, child_count, None)
    }

    pub fn with_estimate(
        style: DivStyle,
        child_count: usize,
        estimated_size: Option<[f32; 2]>,
    ) -> DivDiffKey {
        DivDiffKey {
            style,
            child_count,
            estimated_size,
        }
    }
}

impl ReconcileKey for DivDiffKey {
    fn compare(&self, previous: &dyn ReconcileKey) -> Invalidation {
        let Some(previous) = previous.as_any().downcast_ref::<DivDiffKey>() else {
            return Invalidation::all();
        };
        let mut axes = classify_style_change(&self.style, &previous.style);
        if self.child_count != previous.child_count {
            axes |= Invalidation::LAYOUT;
        }
        if self.estimated_size != previous.estimated_size {
            axes |= Invalidation::LAYOUT;
        }
        axes
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::div::interactivity::style::{Corners, Edges};

    fn style() -> DivStyle {
        DivStyle {
            background: Some([0.2, 0.4, 0.6, 1.0]),
            border_color: Some([1.0, 1.0, 1.0, 1.0]),
            border_widths: Edges::all(1.0),
            corner_radii: Corners::all(6.0),
            ..DivStyle::default()
        }
    }

    #[test]
    fn an_unchanged_div_reports_nothing_stale() {
        let key = DivDiffKey::new(style(), 3);
        assert_eq!(
            key.compare(&DivDiffKey::new(style(), 3)),
            Invalidation::empty()
        );
    }

    #[test]
    fn a_recolour_is_display_only() {
        let previous = DivDiffKey::new(style(), 2);
        let current = DivDiffKey::new(
            DivStyle {
                background: Some([1.0, 0.0, 0.0, 1.0]),
                ..style()
            },
            2,
        );
        assert_eq!(current.compare(&previous), Invalidation::DISPLAY);
    }

    #[test]
    fn a_changed_child_count_is_a_layout_change_even_with_an_identical_style() {
        let previous = DivDiffKey::new(style(), 2);
        assert_eq!(
            DivDiffKey::new(style(), 3).compare(&previous),
            Invalidation::LAYOUT,
            "a different child list is a different Taffy node list"
        );
    }

    #[test]
    fn changing_an_estimated_size_is_a_layout_change() {
        let previous = DivDiffKey::with_estimate(style(), 0, Some([40.0, 20.0]));
        let current = DivDiffKey::with_estimate(style(), 0, Some([80.0, 20.0]));
        assert_eq!(current.compare(&previous), Invalidation::LAYOUT);
        assert_eq!(previous.compare(&previous), Invalidation::empty());
    }

    #[test]
    fn a_key_compared_against_a_different_element_type_is_a_full_invalidation() {
        #[derive(PartialEq, Debug)]
        struct Other;
        impl ReconcileKey for Other {
            fn compare(&self, _: &dyn ReconcileKey) -> Invalidation {
                Invalidation::all()
            }
            fn as_any(&self) -> &dyn Any {
                self
            }
        }
        assert_eq!(
            DivDiffKey::new(style(), 0).compare(&Other),
            Invalidation::all()
        );
    }
}
