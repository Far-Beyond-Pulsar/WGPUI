//! Conservative layer-tier occlusion utilities.
//!
//! The coverage test deliberately accepts only axis-aligned opaque rectangles.
//! Rounded corners, filters, borders, and opacity are excluded by callers;
//! false negatives cost work, while false positives change pixels.

use crate::{Bounds, Pixels, Point};

/// Whether layer-tier occlusion is enabled.
pub(crate) fn enabled() -> bool {
    mode() != Mode::Off
}

/// The current occlusion mode.
#[derive(PartialEq)]
pub(crate) enum Mode {
    /// Fully disabled (`WGPUI_OCCLUSION=0`).
    Off,
    /// Normal operation.
    Normal,
    /// Render twice (culled and unculled) and diff the scenes.
    Validate,
}

pub(crate) fn mode() -> Mode {
    static MODE: std::sync::LazyLock<Mode> = std::sync::LazyLock::new(|| {
        match std::env::var("WGPUI_OCCLUSION") {
            Ok(v) if v == "0" || v == "off" => Mode::Off,
            Ok(v) if v == "validate" => Mode::Validate,
            _ => Mode::Normal,
        }
    });
    match *MODE {
        Mode::Off => Mode::Off,
        Mode::Normal => Mode::Normal,
        Mode::Validate => Mode::Validate,
    }
}

/// Whether validate mode is active.
pub(crate) fn validate_enabled() -> bool {
    matches!(mode(), Mode::Validate)
}

/// Returns whether a target rectangle is completely covered by opaque regions.
pub(crate) fn fully_covered(target: Bounds<Pixels>, occluders: &[Bounds<Pixels>]) -> bool {
    if target.size.width <= Pixels::ZERO || target.size.height <= Pixels::ZERO {
        return true;
    }

    let mut x_edges = Vec::with_capacity(occluders.len() * 2 + 2);
    x_edges.push(target.origin.x);
    x_edges.push(target.right());
    for region in occluders {
        let overlap = region.intersect(&target);
        if overlap.size.width > Pixels::ZERO && overlap.size.height > Pixels::ZERO {
            x_edges.push(overlap.origin.x);
            x_edges.push(overlap.right());
        }
    }
    x_edges.sort_unstable();
    x_edges.dedup();

    x_edges.windows(2).all(|x| {
        let left = x[0];
        let right = x[1];
        if right <= left {
            return true;
        }
        let midpoint = Point::new(left + (right - left) / 2., target.origin.y);
        let mut intervals = occluders
            .iter()
            .map(|region| region.intersect(&target))
            .filter(|region| region.size.width > Pixels::ZERO && region.size.height > Pixels::ZERO)
            .filter(|region| region.origin.x <= midpoint.x && region.right() >= midpoint.x)
            .map(|region| (region.origin.y, region.bottom()))
            .collect::<Vec<_>>();
        intervals.sort_unstable_by_key(|interval| interval.0);

        let mut covered_to = target.origin.y;
        for (top, bottom) in intervals {
            if top > covered_to {
                return false;
            }
            if bottom > covered_to {
                covered_to = bottom;
            }
            if covered_to >= target.bottom() {
                return true;
            }
        }
        covered_to >= target.bottom()
    })
}

/// Compute the conservative opaque region for an element, accounting for
/// corner radii and border insets.
///
/// Returns `None` if the element does not produce a fully opaque rectangle.
pub(crate) fn compute_opaque_region(
    bounds: Bounds<Pixels>,
    element_opacity: f32,
    has_solid_background: bool,
    max_corner_radius: Pixels,
    has_opaque_border: bool,
    border_inset: Pixels,
    has_backdrop_filter: bool,
) -> Option<Bounds<Pixels>> {
    if !has_solid_background || element_opacity < 1.0 || has_backdrop_filter {
        return None;
    }

    let mut inset_amount = max_corner_radius;
    if !has_opaque_border && border_inset > Pixels::ZERO {
        inset_amount = inset_amount.max(border_inset);
    }

    if inset_amount > Pixels::ZERO {
        let shrunk = bounds.inset(inset_amount);
        if shrunk.size.width <= Pixels::ZERO || shrunk.size.height <= Pixels::ZERO {
            return None;
        }
        Some(shrunk)
    } else {
        Some(bounds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{point, px, size};

    fn bounds(x: f32, y: f32, width: f32, height: f32) -> Bounds<Pixels> {
        Bounds::new(point(px(x), px(y)), size(px(width), px(height)))
    }

    #[test]
    fn coverage_requires_every_point() {
        let target = bounds(0., 0., 100., 100.);
        assert!(fully_covered(target, &[bounds(0., 0., 100., 100.)]));
        assert!(!fully_covered(target, &[bounds(0., 0., 50., 100.)]));
        assert!(fully_covered(
            target,
            &[bounds(0., 0., 50., 100.), bounds(50., 0., 50., 100.)]
        ));
    }

    #[test]
    fn coverage_with_gaps() {
        let target = bounds(0., 0., 100., 100.);
        assert!(!fully_covered(
            target,
            &[bounds(0., 0., 40., 100.), bounds(60., 0., 40., 100.)],
        ));
    }

    #[test]
    fn coverage_partial_y() {
        let target = bounds(0., 0., 100., 100.);
        assert!(!fully_covered(target, &[bounds(0., 0., 100., 50.)]));
    }

    #[test]
    fn zero_sized_target_is_covered() {
        let target = bounds(0., 0., 0., 100.);
        assert!(fully_covered(target, &[]));
    }

    #[test]
    fn coverage_with_multiple_x_slices() {
        let target = bounds(0., 0., 200., 100.);
        assert!(fully_covered(
            target,
            &[
                bounds(0., 0., 50., 100.),
                bounds(50., 0., 50., 50.),
                bounds(50., 50., 150., 50.),
                bounds(100., 0., 50., 50.),
            ],
        ));
    }

    #[test]
    fn opaque_region_rejects_transparent() {
        let bounds = bounds(0., 0., 100., 100.);
        assert_eq!(
            compute_opaque_region(bounds, 1.0, false, px(0.), false, px(0.), false),
            None,
        );
    }

    #[test]
    fn opaque_region_rejects_non_one_opacity() {
        let bounds = bounds(0., 0., 100., 100.);
        assert_eq!(
            compute_opaque_region(bounds, 0.5, true, px(0.), false, px(0.), false),
            None,
        );
    }

    #[test]
    fn opaque_region_insets_for_corner_radius() {
        let bounds = bounds(0., 0., 100., 100.);
        let region = compute_opaque_region(bounds, 1.0, true, px(10.), true, px(0.), false);
        assert_eq!(region, Some(bounds(10., 10., 80., 80.)));
    }

    #[test]
    fn opaque_region_insets_for_border() {
        let bounds = bounds(0., 0., 100., 100.);
        let region = compute_opaque_region(bounds, 1.0, true, px(0.), false, px(5.), false);
        assert_eq!(region, Some(bounds(5., 5., 90., 90.)));
    }

    #[test]
    fn opaque_region_rejects_backdrop_filter() {
        let bounds = bounds(0., 0., 100., 100.);
        assert_eq!(
            compute_opaque_region(bounds, 1.0, true, px(0.), true, px(0.), true),
            None,
        );
    }

    #[test]
    fn opaque_region_returns_none_when_inset_removes_all() {
        let bounds = bounds(0., 0., 5., 5.);
        assert_eq!(
            compute_opaque_region(bounds, 1.0, true, px(5.), true, px(0.), false),
            None,
        );
    }
}
