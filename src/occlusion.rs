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

    fn make_bounds(x: f32, y: f32, width: f32, height: f32) -> Bounds<Pixels> {
        Bounds::new(point(px(x), px(y)), size(px(width), px(height)))
    }

    #[test]
    fn coverage_requires_every_point() {
        let target = make_bounds(0., 0., 100., 100.);
        assert!(fully_covered(target, &[make_bounds(0., 0., 100., 100.)]));
        assert!(!fully_covered(target, &[make_bounds(0., 0., 50., 100.)]));
        assert!(fully_covered(
            target,
            &[make_bounds(0., 0., 50., 100.), make_bounds(50., 0., 50., 100.)]
        ));
    }

    #[test]
    fn coverage_with_gaps() {
        let target = make_bounds(0., 0., 100., 100.);
        assert!(!fully_covered(
            target,
            &[make_bounds(0., 0., 40., 100.), make_bounds(60., 0., 40., 100.)],
        ));
    }

    #[test]
    fn coverage_partial_y() {
        let target = make_bounds(0., 0., 100., 100.);
        assert!(!fully_covered(target, &[make_bounds(0., 0., 100., 50.)]));
    }

    #[test]
    fn zero_sized_target_is_covered() {
        let target = make_bounds(0., 0., 0., 100.);
        assert!(fully_covered(target, &[]));
    }

    #[test]
    fn coverage_with_two_occluders_side_by_side() {
        let target = make_bounds(0., 0., 200., 100.);
        assert!(fully_covered(
            target,
            &[
                make_bounds(0., 0., 100., 100.),
                make_bounds(100., 0., 100., 100.),
            ],
        ));
    }

    #[test]
    fn coverage_with_multi_x_slices_not_all_covered() {
        let target = make_bounds(0., 0., 200., 100.);
        // x-slice [150, 200] lacks an occluder for y 0-50.
        assert!(!fully_covered(
            target,
            &[
                make_bounds(0., 0., 50., 100.),
                make_bounds(50., 0., 50., 50.),
                make_bounds(50., 50., 150., 50.),
                make_bounds(100., 0., 50., 50.),
            ],
        ));
    }

    #[test]
    fn opaque_region_rejects_transparent() {
        let b = make_bounds(0., 0., 100., 100.);
        assert_eq!(
            compute_opaque_region(b, 1.0, false, px(0.), false, px(0.), false),
            None,
        );
    }

    #[test]
    fn opaque_region_rejects_non_one_opacity() {
        let b = make_bounds(0., 0., 100., 100.);
        assert_eq!(
            compute_opaque_region(b, 0.5, true, px(0.), false, px(0.), false),
            None,
        );
    }

    #[test]
    fn opaque_region_insets_for_corner_radius() {
        let b = make_bounds(0., 0., 100., 100.);
        let region = compute_opaque_region(b, 1.0, true, px(10.), true, px(0.), false);
        assert_eq!(region, Some(make_bounds(10., 10., 80., 80.)));
    }

    #[test]
    fn opaque_region_insets_for_border() {
        let b = make_bounds(0., 0., 100., 100.);
        let region = compute_opaque_region(b, 1.0, true, px(0.), false, px(5.), false);
        assert_eq!(region, Some(make_bounds(5., 5., 90., 90.)));
    }

    #[test]
    fn opaque_region_rejects_backdrop_filter() {
        let b = make_bounds(0., 0., 100., 100.);
        assert_eq!(
            compute_opaque_region(b, 1.0, true, px(0.), true, px(0.), true),
            None,
        );
    }

    #[test]
    fn opaque_region_returns_none_when_inset_removes_all() {
        let b = make_bounds(0., 0., 5., 5.);
        assert_eq!(
            compute_opaque_region(b, 1.0, true, px(5.), true, px(0.), false),
            None,
        );
    }

    // --- Adversarial / edge-case coverage tests ---

    #[test]
    fn coverage_empty_occluders_never_cover() {
        let target = make_bounds(10., 10., 50., 50.);
        assert!(!fully_covered(target, &[]));
    }

    #[test]
    fn coverage_single_pixel_target() {
        let target = make_bounds(100., 100., 1., 1.);
        assert!(fully_covered(target, &[make_bounds(100., 100., 1., 1.)]));
        assert!(!fully_covered(target, &[make_bounds(100., 100., 1., 0.)]));
    }

    #[test]
    fn coverage_occluder_smaller_than_target_everywhere() {
        // Occluder covers the center but leaves a margin on all four sides.
        let target = make_bounds(0., 0., 100., 100.);
        assert!(!fully_covered(target, &[make_bounds(10., 10., 80., 80.)]));
    }

    #[test]
    fn coverage_vertical_stack_of_horizontal_strips() {
        // Three occluders stacked vertically, each full-width.
        let target = make_bounds(0., 0., 100., 90.);
        assert!(fully_covered(
            target,
            &[
                make_bounds(0., 0., 100., 30.),
                make_bounds(0., 30., 100., 30.),
                make_bounds(0., 60., 100., 30.),
            ],
        ));
    }

    #[test]
    fn coverage_jagged_occluders_no_overlap() {
        // Occluders form an L shape — they cover different X-slices at different
        // Y-ranges but together cover every point.
        let target = make_bounds(0., 0., 60., 60.);
        assert!(fully_covered(
            target,
            &[
                make_bounds(0., 0., 60., 30.),  // full-width top half
                make_bounds(0., 30., 30., 30.), // left half of bottom
                make_bounds(30., 30., 30., 30.), // right half of bottom
            ],
        ));
    }

    #[test]
    fn coverage_two_occluders_missing_middle_x_slice() {
        // Gap in the middle X-slice — each occluder covers a different
        // X-region but neither covers x=40..60.
        let target = make_bounds(0., 0., 100., 50.);
        assert!(!fully_covered(
            target,
            &[
                make_bounds(0., 0., 40., 50.),
                make_bounds(60., 0., 40., 50.),
            ],
        ));
    }

    #[test]
    fn coverage_occluder_partially_outside_target() {
        // Occluder extends beyond the target — only the overlap matters.
        let target = make_bounds(20., 20., 60., 60.);
        assert!(fully_covered(
            target,
            &[make_bounds(0., 0., 100., 100.)],
        ));
    }

    #[test]
    fn coverage_occluder_entirely_outside_target() {
        let target = make_bounds(0., 0., 50., 50.);
        assert!(!fully_covered(target, &[make_bounds(100., 100., 50., 50.)]));
    }

    #[test]
    fn coverage_negative_coordinates() {
        let target = make_bounds(-50., -50., 100., 100.);
        assert!(fully_covered(target, &[make_bounds(-50., -50., 100., 100.)]));
        assert!(!fully_covered(target, &[make_bounds(-50., -50., 50., 100.)]));
    }

    #[test]
    fn coverage_two_occluders_same_region() {
        // Duplicate occluders should not break the algorithm.
        let target = make_bounds(0., 0., 100., 100.);
        assert!(fully_covered(
            target,
            &[
                make_bounds(0., 0., 100., 100.),
                make_bounds(0., 0., 100., 100.),
            ],
        ));
    }

    #[test]
    fn coverage_three_occluders_one_per_x_slice() {
        // Three occluders tile the target in the X direction.
        let target = make_bounds(0., 0., 150., 40.);
        assert!(fully_covered(
            target,
            &[
                make_bounds(0., 0., 50., 40.),
                make_bounds(50., 0., 50., 40.),
                make_bounds(100., 0., 50., 40.),
            ],
        ));
    }

    #[test]
    fn coverage_missing_horizontal_band() {
        // Two occluders stacked on top and bottom with a gap in the middle.
        let target = make_bounds(0., 0., 100., 100.);
        assert!(!fully_covered(
            target,
            &[
                make_bounds(0., 0., 100., 40.),
                make_bounds(0., 60., 100., 40.),
            ],
        ));
    }

    #[test]
    fn opaque_region_large_corner_radius_inset_to_zero() {
        // Corner radius large enough that the inset produces a degenerate
        // region.
        let b = make_bounds(0., 0., 20., 20.);
        assert_eq!(
            compute_opaque_region(b, 1.0, true, px(10.), true, px(0.), false),
            None,
        );
    }

    #[test]
    fn opaque_region_uneven_corner_radii_takes_max() {
        // Only the largest corner radius matters for the conservative inset.
        let b = make_bounds(0., 0., 100., 100.);
        let region = compute_opaque_region(b, 1.0, true, px(25.), true, px(0.), false);
        assert_eq!(region, Some(make_bounds(25., 25., 50., 50.)));
    }

    #[test]
    fn opaque_region_backdrop_filter_even_with_solid_bg() {
        // Backdrop filter takes precedence over solid background.
        let b = make_bounds(0., 0., 100., 100.);
        assert_eq!(
            compute_opaque_region(b, 1.0, true, px(0.), true, px(0.), true),
            None,
        );
    }

    #[test]
    fn opaque_region_border_inset_larger_than_corner() {
        // Non-opaque border causes a larger inset than corner radii.
        let b = make_bounds(0., 0., 100., 100.);
        let region = compute_opaque_region(b, 1.0, true, px(2.), false, px(20.), false);
        assert_eq!(region, Some(make_bounds(20., 20., 60., 60.)));
    }

    #[test]
    fn opaque_region_corner_larger_than_border_inset() {
        // Corner radius is the dominant inset.
        let b = make_bounds(0., 0., 100., 100.);
        let region = compute_opaque_region(b, 1.0, true, px(15.), false, px(3.), false);
        assert_eq!(region, Some(make_bounds(15., 15., 70., 70.)));
    }

    #[test]
    fn opaque_region_opaque_border_skips_border_inset() {
        // Opaque border means no border inset — only corner radii matter.
        let b = make_bounds(0., 0., 100., 100.);
        let region = compute_opaque_region(b, 1.0, true, px(10.), true, px(30.), false);
        assert_eq!(region, Some(make_bounds(10., 10., 80., 80.)));
    }

    #[test]
    fn coverage_sub_pixel_boundaries() {
        let target = make_bounds(0.5, 0.5, 99.5, 99.5);
        assert!(fully_covered(target, &[make_bounds(0., 0., 100., 100.)]));
        assert!(!fully_covered(
            target,
            &[make_bounds(0.5, 0.5, 49.5, 99.5)],
        ));
    }

    #[test]
    fn coverage_many_occluders() {
        // 100 occluders — each covers a 1px-wide vertical strip. Together
        // they cover the full target.
        let target = make_bounds(0., 0., 100., 50.);
        let occluders: Vec<_> = (0..100)
            .map(|i| make_bounds(i as f32, 0., 1., 50.))
            .collect();
        assert!(fully_covered(target, &occluders));
    }

    #[test]
    fn coverage_all_occluders_outside_target_bounds() {
        // No occluder touches the target.
        let target = make_bounds(50., 50., 10., 10.);
        assert!(!fully_covered(
            target,
            &[
                make_bounds(0., 0., 10., 10.),
                make_bounds(100., 100., 10., 10.),
            ],
        ));
    }

    #[test]
    fn coverage_target_has_zero_height() {
        let target = make_bounds(0., 0., 100., 0.);
        assert!(fully_covered(target, &[]));
    }

    #[test]
    fn coverage_target_has_zero_width() {
        let target = make_bounds(0., 0., 0., 100.);
        assert!(fully_covered(target, &[make_bounds(0., 0., 0., 50.)]));
    }
}
