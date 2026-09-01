//! Optional intrinsic sizing for unresolved layout content.
//!
//! An estimate is a layout hint, not a replacement for Taffy's exact sizing
//! algorithm. The adapter below changes only `auto` dimensions and rejects
//! malformed hints, leaving callers with the exact path by default.

use crate::measure::IntrinsicSize;
use crate::taffy_tree::{Dimension, LayoutStyle};

/// Supplies a cheap intrinsic estimate for an element whose content may not
/// be available during the current layout pass.
///
/// Implementations should return `None` when an estimate is unavailable or
/// cannot be trusted. The layout adapter also validates returned values, so a
/// provider cannot accidentally turn NaN, infinity, or a negative dimension
/// into a Taffy constraint.
pub trait EstimatedSize {
    fn estimated_size(&self) -> Option<IntrinsicSize>;
}

impl EstimatedSize for IntrinsicSize {
    fn estimated_size(&self) -> Option<IntrinsicSize> {
        Some(*self)
    }
}

impl EstimatedSize for Option<IntrinsicSize> {
    fn estimated_size(&self) -> Option<IntrinsicSize> {
        *self
    }
}

/// Resolve a validated estimate into a style without changing authored
/// dimensions. An explicit width or height always wins over the estimate.
pub fn resolve_estimated_style(
    mut style: LayoutStyle,
    estimate: Option<IntrinsicSize>,
) -> LayoutStyle {
    let Some(estimate) = estimate.and_then(IntrinsicSize::validated) else {
        return style;
    };

    if style.size.width == Dimension::auto() {
        style.size.width = Dimension::length(estimate.width);
    }
    if style.size.height == Dimension::auto() {
        style.size.height = Dimension::length(estimate.height);
    }
    style
}

/// Resolve an element's optional estimate into a layout style.
pub fn resolve_element_style<E: EstimatedSize + ?Sized>(
    style: LayoutStyle,
    element: &E,
) -> LayoutStyle {
    resolve_estimated_style(style, element.estimated_size())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::taffy_tree::TaffySize;

    #[test]
    fn estimates_are_optional_and_strictly_validated() {
        assert!(IntrinsicSize::new(10.0, 20.0).is_valid());
        assert!(!IntrinsicSize::new(-1.0, 20.0).is_valid());
        assert!(!IntrinsicSize::new(f32::NAN, 20.0).is_valid());
        assert!(!IntrinsicSize::new(10.0, f32::INFINITY).is_valid());
        assert_eq!(
            IntrinsicSize::new(10.0, 20.0).validated(),
            Some(IntrinsicSize::new(10.0, 20.0))
        );
    }

    #[test]
    fn estimates_fill_only_auto_dimensions() {
        let style = LayoutStyle {
            size: TaffySize {
                width: Dimension::length(30.0),
                height: Dimension::auto(),
            },
            ..LayoutStyle::default()
        };

        let resolved = resolve_estimated_style(style, Some(IntrinsicSize::new(80.0, 40.0)));

        assert_eq!(resolved.size.width, Dimension::length(30.0));
        assert_eq!(resolved.size.height, Dimension::length(40.0));
    }

    #[test]
    fn missing_or_invalid_estimates_preserve_the_exact_style() {
        let style = LayoutStyle::default();

        for estimate in [
            None,
            Some(IntrinsicSize::new(-1.0, 20.0)),
            Some(IntrinsicSize::new(20.0, f32::NAN)),
        ] {
            assert_eq!(resolve_estimated_style(style.clone(), estimate), style);
        }
    }
}
