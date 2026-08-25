//! Conservative occlusion utilities, shared by both culling tiers.
//!
//! The coverage test deliberately accepts only axis-aligned opaque rectangles.
//! Rounded corners, filters, borders, and opacity are excluded by callers;
//! false negatives cost work, while false positives change pixels.
//!
//! Two tiers consume this module:
//!
//! - **Layer tier** (#91) classifies whole layers in logical pixels and skips
//!   their draws at composite time.
//! - **Instance tier** (#95) filters a freshly-recorded layer's item stream,
//!   dropping primitives whose visible extent is fully covered by conservative
//!   opaque quads painted above them *within the same layer*. It runs in
//!   scaled pixels because that is what recorded primitives carry.
//!
//! The instance-tier sweep works on the layer's own ordered item stream — the
//! exact content its `BoundsTree` indexes. The transient occluder list it
//! builds is derived from that stream, consumed within one record, and never
//! stored, so there is no second spatial index to drift out of sync (#89).
//!
//! Coverage is only ever evaluated against content of the same layer: any
//! change to a within-layer occluder re-records the layer (it is part of the
//! same subtree), so a baked decision cannot go stale silently. Culling
//! against other layers' occluders would need cross-layer invalidation to stay
//! correct and is deliberately out of scope; #91 already handles whole-layer
//! coverage.

use crate::layer::LayerItem;
use crate::scene::{Primitive, Quad};
use crate::{BackgroundTag, BorderStyle, Bounds, Point, ScaledPixels};

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
///
/// Generic over the pixel unit: the layer tier queries logical pixels while
/// the instance tier queries scaled pixels, and both types expose the same
/// arithmetic surface.
pub(crate) fn fully_covered<T>(target: Bounds<T>, occluders: &[Bounds<T>]) -> bool
where
    T: Clone
        + Default
        + PartialEq
        + Ord
        + std::fmt::Debug
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Div<f32, Output = T>,
{
    let zero = T::default();
    if target.size.width <= zero || target.size.height <= zero {
        return true;
    }

    let mut x_edges = Vec::with_capacity(occluders.len() * 2 + 2);
    x_edges.push(target.origin.x.clone());
    x_edges.push(target.right());
    for region in occluders {
        let overlap = region.intersect(&target);
        if overlap.size.width > zero && overlap.size.height > zero {
            x_edges.push(overlap.origin.x.clone());
            x_edges.push(overlap.right());
        }
    }
    x_edges.sort_unstable();
    x_edges.dedup();

    let target_bottom = target.bottom();
    let target_origin_y = target.origin.y.clone();
    x_edges.windows(2).all(|x| {
        let left = &x[0];
        let right = &x[1];
        if right <= left {
            return true;
        }
        let midpoint = Point::new(
            left.clone() + (right.clone() - left.clone()) / 2.,
            target_origin_y.clone(),
        );
        let mut intervals = occluders
            .iter()
            .map(|region| region.intersect(&target))
            .filter(|region| region.size.width > zero && region.size.height > zero)
            .filter(|region| region.origin.x <= midpoint.x && region.right() >= midpoint.x)
            .map(|region| (region.origin.y.clone(), region.bottom()))
            .collect::<Vec<_>>();
        intervals.sort_unstable_by_key(|interval| interval.0.clone());

        let mut covered_to = target_origin_y.clone();
        for (top, bottom) in intervals {
            if top > covered_to {
                return false;
            }
            if bottom > covered_to {
                covered_to = bottom;
            }
            if covered_to >= target_bottom {
                return true;
            }
        }
        covered_to >= target_bottom
    })
}

/// Compute the conservative opaque region for an element, accounting for
/// corner radii and border insets.
///
/// Returns `None` if the element does not produce a fully opaque rectangle.
pub(crate) fn compute_opaque_region<T>(
    bounds: Bounds<T>,
    element_opacity: f32,
    has_solid_background: bool,
    max_corner_radius: T,
    has_opaque_border: bool,
    border_inset: T,
    has_backdrop_filter: bool,
) -> Option<Bounds<T>>
where
    T: Clone
        + Default
        + PartialOrd
        + std::fmt::Debug
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>,
{
    let zero = T::default();
    if !has_solid_background || element_opacity < 1.0 || has_backdrop_filter {
        return None;
    }

    let mut inset_amount = max_corner_radius;
    if !has_opaque_border && border_inset > inset_amount {
        inset_amount = border_inset;
    }

    if inset_amount > zero {
        // Spelled out rather than `inset`: `dilate(-amount)` would require
        // `Neg`, which `ScaledPixels` does not implement.
        let double = inset_amount.clone() + inset_amount.clone();
        let shrunk = Bounds {
            origin: Point::new(
                bounds.origin.x.clone() + inset_amount.clone(),
                bounds.origin.y.clone() + inset_amount,
            ),
            size: crate::Size::new(bounds.size.width - double.clone(), bounds.size.height - double),
        };
        if shrunk.size.width <= zero || shrunk.size.height <= zero {
            return None;
        }
        Some(shrunk)
    } else {
        Some(bounds)
    }
}

const COUNTER_INSTANCES_CULLED: &str = "occlusion: instances culled";
const COUNTER_INSTANCES_KEPT: &str = "occlusion: instances kept";
const COUNTER_VALIDATE_DIVERGENCES: &str = "occlusion: validate divergences";

/// The conservative opaque region of a recorded quad, or `None` when the quad
/// does not fully cover even its own bounds.
///
/// [`compute_opaque_region`] applied to a scene primitive, with the same rules
/// as `Style::opaque_region` (style.rs): solid background only, corner-radius
/// inset, translucent border inset by its width. Element opacity is not an
/// input here — it was already multiplied into the background color at paint
/// time (`Window::paint_quad`), so an element painted translucent carries an
/// alpha below one and is rejected by the alpha test alone. A backdrop filter
/// on the owning element is likewise not an input: it arrives as its own
/// primitive and poisons via [`cull_covered_instances`] instead.
fn quad_opaque_region(quad: &Quad) -> Option<Bounds<ScaledPixels>> {
    // Only a flat solid color qualifies; gradients and patterns are excluded
    // without further analysis, exactly as at the layer tier.
    let solid_background =
        matches!(quad.background.tag, BackgroundTag::Solid) && quad.background.solid.a >= 1.0;

    let max_corner_radius = quad
        .corner_radii
        .top_left
        .max(quad.corner_radii.top_right)
        .max(quad.corner_radii.bottom_right)
        .max(quad.corner_radii.bottom_left);

    // A dashed border leaves gaps even at alpha one, so it insets like a
    // translucent one. Strictly more conservative than the layer-tier rule,
    // which reads only the border color's alpha.
    let border_is_opaque = quad.border_color.a >= 1.0 && quad.border_style == BorderStyle::Solid;
    let border_inset = if border_is_opaque {
        ScaledPixels::default()
    } else {
        quad.border_widths
            .top
            .max(quad.border_widths.right)
            .max(quad.border_widths.bottom)
            .max(quad.border_widths.left)
    };

    let mut region = compute_opaque_region(
        quad.bounds,
        1.0,
        solid_background,
        max_corner_radius,
        border_is_opaque,
        border_inset,
        false,
    )?;
    // Only what survives the clip can actually hide anything.
    region = region.intersect(&quad.content_mask.bounds);
    if region.size.width <= ScaledPixels::default()
        || region.size.height <= ScaledPixels::default()
    {
        None
    } else {
        Some(region)
    }
}

/// The visible extent of a primitive that may be culled once fully covered,
/// or `None` for primitives this phase never drops.
///
/// Shadows bleed past their bounds by their blur radius, so no rectangle can
/// describe what they cover; surfaces, backdrop filters and filter boundaries
/// drive render-target state rather than plain draws. All are kept
/// unconditionally this phase.
fn cullee_visible_bounds(primitive: &Primitive) -> Option<Bounds<ScaledPixels>> {
    match primitive {
        Primitive::Quad(_)
        | Primitive::Underline(_)
        | Primitive::Path(_)
        | Primitive::MonochromeSprite(_)
        | Primitive::PolychromeSprite(_) => {}
        Primitive::Shadow(_)
        | Primitive::Surface(_)
        | Primitive::BackdropFilter(_)
        | Primitive::FilterBoundary(_) => return None,
    }
    let clipped = primitive
        .bounds()
        .intersect(&primitive.content_mask().bounds);
    if clipped.size.width <= ScaledPixels::default()
        || clipped.size.height <= ScaledPixels::default()
    {
        None
    } else {
        Some(clipped)
    }
}

fn intersects_any(bounds: &Bounds<ScaledPixels>, regions: &[Bounds<ScaledPixels>]) -> bool {
    regions.iter().any(|region| {
        let overlap = region.intersect(bounds);
        overlap.size.width > ScaledPixels::default()
            && overlap.size.height > ScaledPixels::default()
    })
}

/// Instance-tier culling (#95): drop primitives of a freshly-recorded layer
/// whose visible extent is fully covered by opaque quads painted above them in
/// the same layer. Returns the kept items in paint order.
///
/// Runs exactly once per record, upstream of the pack/legacy composite fork,
/// so the packed bytes and the legacy replay describe identical content by
/// construction. Clean layers never pass through here — they composite
/// whatever their last record baked — which is why an occluder animating over
/// a static layer can never churn that layer's slab (#94's headline invariant).
///
/// Backdrop filters poison everything beneath them within bounds dilated by
/// their blur radius, and a filter group poisons beneath its dilated start
/// bound; nothing inside a group is culled either, since the group's blur
/// samples its members' pixels. Nested-layer references are neither culled
/// nor consulted as occluders: their content changes never re-record this
/// parent layer, so relying on them could bake a hole that no later record
/// repairs.
pub(crate) fn cull_covered_instances(items: Vec<LayerItem>) -> Vec<LayerItem> {
    if !enabled() {
        return items;
    }

    let keep = compute_keep_mask(&items);

    if validate_enabled() {
        let independent = compute_keep_mask_independently(&items);
        for (index, (kept, independently_kept)) in
            keep.iter().zip(independent.iter()).enumerate()
        {
            if kept != independently_kept {
                crate::render_stats::count(COUNTER_VALIDATE_DIVERGENCES);
                log::error!(
                    "occlusion validate: divergent keep decision for item {index} \
                     (sweep={kept}, independent={independently_kept})"
                );
            }
        }
    }

    let kept_count = keep.iter().filter(|kept| **kept).count();
    let culled_count = keep.len() - kept_count;
    crate::render_stats::add(COUNTER_INSTANCES_CULLED, culled_count as u64);
    crate::render_stats::add(COUNTER_INSTANCES_KEPT, kept_count as u64);

    if culled_count == 0 {
        return items;
    }
    let mut kept_items = Vec::with_capacity(kept_count);
    for (item, kept) in items.into_iter().zip(keep) {
        if kept {
            kept_items.push(item);
        }
    }
    kept_items
}

/// One backward sweep over the layer's own primitives, from topmost paint
/// order down: every opaque quad met joins the running occluder set before
/// anything beneath it is tested against it.
fn compute_keep_mask(items: &[LayerItem]) -> Vec<bool> {
    let mut keep = vec![true; items.len()];
    let mut occluders: Vec<Bounds<ScaledPixels>> = Vec::new();
    // Dilated bounds of backdrop filters and filter groups seen so far.
    // Content intersecting one must be emitted whatever covers it visually:
    // the filter reads the pixels behind it.
    let mut poisoned: Vec<Bounds<ScaledPixels>> = Vec::new();
    // Greater than zero while the backward walk is inside a filter group.
    let mut group_depth: usize = 0;

    for (index, item) in items.iter().enumerate().rev() {
        let LayerItem::Primitive(primitive) = item else {
            continue;
        };
        match primitive {
            Primitive::FilterBoundary(boundary) => {
                if boundary.is_start {
                    group_depth = group_depth.saturating_sub(1);
                    poisoned.push(boundary.bounds.dilate(boundary.blur_radius));
                } else {
                    group_depth += 1;
                }
                continue;
            }
            Primitive::BackdropFilter(filter) => {
                poisoned.push(filter.bounds.dilate(filter.blur_radius));
                continue;
            }
            _ => {}
        }

        let Some(visible) = cullee_visible_bounds(primitive) else {
            continue;
        };
        let protected = group_depth > 0 || intersects_any(&visible, &poisoned);
        if !protected && fully_covered(visible, &occluders) {
            keep[index] = false;
        }

        // Collected after the coverage test, so the set holds exactly the
        // quads painted above this one. An occluder that is itself covered
        // still hides what lies beneath it, so its region joins regardless
        // of its own keep decision.
        if !protected {
            if let Primitive::Quad(quad) = primitive {
                if let Some(region) = quad_opaque_region(quad) {
                    occluders.push(region);
                }
            }
        }
    }
    keep
}

/// Validate-mode twin of [`compute_keep_mask`]: redecides every primitive
/// from scratch — group membership, poison zones and the occluder set above
/// each item are recomputed per item rather than carried through one
/// incremental sweep. Slow by design; validate mode only. Any disagreement
/// between the two passes means the sweep's bookkeeping drifted, and is
/// counted under `occlusion: validate divergences`.
fn compute_keep_mask_independently(items: &[LayerItem]) -> Vec<bool> {
    /// Filter-group nesting depth at `through`, counting boundaries up to and
    /// including that position. A non-boundary primitive sits inside a group
    /// exactly when this is greater than zero.
    fn group_depth_at(items: &[LayerItem], through: usize) -> usize {
        let mut depth = 0usize;
        for item in &items[..=through] {
            if let LayerItem::Primitive(Primitive::FilterBoundary(boundary)) = item {
                if boundary.is_start {
                    depth += 1;
                } else {
                    depth = depth.saturating_sub(1);
                }
            }
        }
        depth
    }

    /// Whether the primitive at `at` sits inside a filter group or intersects
    /// a backdrop/filter-group poison zone declared above it.
    fn protected_from_above(
        items: &[LayerItem],
        at: usize,
        visible: &Bounds<ScaledPixels>,
    ) -> bool {
        if group_depth_at(items, at) > 0 {
            return true;
        }
        items[at + 1..].iter().any(|item| match item {
            LayerItem::Primitive(Primitive::BackdropFilter(filter)) => {
                intersects_any(visible, &[filter.bounds.dilate(filter.blur_radius)])
            }
            LayerItem::Primitive(Primitive::FilterBoundary(boundary)) => {
                boundary.is_start
                    && intersects_any(visible, &[boundary.bounds.dilate(boundary.blur_radius)])
            }
            _ => false,
        })
    }

    let mut keep = vec![true; items.len()];
    for (index, item) in items.iter().enumerate() {
        let LayerItem::Primitive(primitive) = item else {
            continue;
        };
        let Some(visible) = cullee_visible_bounds(primitive) else {
            continue;
        };
        if protected_from_above(items, index, &visible) {
            // Poisoned content always emits; no occluder set can change that.
            continue;
        }

        let mut occluders: Vec<Bounds<ScaledPixels>> = Vec::new();
        for probe_index in index + 1..items.len() {
            let LayerItem::Primitive(Primitive::Quad(quad)) = &items[probe_index] else {
                continue;
            };
            let Some(region) = quad_opaque_region(quad) else {
                continue;
            };
            // The probed quad joins the set on exactly the sweep's terms:
            // a quad inside a group or beneath a poison zone never collects.
            let Some(probe_bounds) = cullee_visible_bounds(&Primitive::Quad(*quad)) else {
                continue;
            };
            if protected_from_above(items, probe_index, &probe_bounds) {
                continue;
            }
            occluders.push(region);
        }

        keep[index] = !fully_covered(visible, &occluders);
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{point, px, size, Pixels};

    fn make_bounds(x: f32, y: f32, width: f32, height: f32) -> Bounds<Pixels> {
        Bounds::new(point(px(x), px(y)), size(px(width), px(height)))
    }

    fn scaled_bounds(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
        Bounds::new(
            Point::new(ScaledPixels(x), ScaledPixels(y)),
            crate::Size::new(ScaledPixels(width), ScaledPixels(height)),
        )
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

    // --- Instance tier (#95) ---

    /// An opaque quad: solid alpha-one background, no border, square corners.
    fn opaque_quad(bounds: Bounds<ScaledPixels>) -> LayerItem {
        LayerItem::Primitive(Primitive::Quad(scene_quad(bounds, true)))
    }

    fn scene_quad(
        bounds: Bounds<ScaledPixels>,
        opaque_background: bool,
    ) -> crate::scene::Quad {
        let mut background = crate::Background::default();
        if opaque_background {
            background.solid.a = 1.0;
        }
        crate::scene::Quad {
            order: 0,
            border_style: BorderStyle::Solid,
            bounds,
            content_mask: unbounded_mask(),
            background,
            border_color: crate::Hsla::default(),
            corner_radii: Default::default(),
            border_widths: Default::default(),
        }
    }

    fn unbounded_mask() -> crate::ContentMask<ScaledPixels> {
        crate::ContentMask {
            bounds: scaled_bounds(-10_000., -10_000., 20_000., 20_000.),
        }
    }

    fn cullee(bounds: Bounds<ScaledPixels>) -> LayerItem {
        // A translucent quad is a pure cullee: it can be covered but never
        // covers.
        LayerItem::Primitive(Primitive::Quad(scene_quad(bounds, false)))
    }

    fn filter_boundary(bounds: Bounds<ScaledPixels>, blur: f32, is_start: bool) -> LayerItem {
        LayerItem::Primitive(Primitive::FilterBoundary(crate::scene::FilterBoundary {
            order: 0,
            bounds,
            content_mask: unbounded_mask(),
            corner_radii: Default::default(),
            blur_radius: ScaledPixels(blur),
            opacity: 1.0,
            is_start,
        }))
    }

    fn backdrop_filter(bounds: Bounds<ScaledPixels>, blur: f32) -> LayerItem {
        LayerItem::Primitive(Primitive::BackdropFilter(crate::scene::BackdropFilter {
            order: 0,
            blur_radius: ScaledPixels(blur),
            bounds,
            content_mask: unbounded_mask(),
            corner_radii: Default::default(),
            opacity: 1.0,
            _pad: 0,
        }))
    }

    fn kept_count(items: &[LayerItem]) -> usize {
        items
            .iter()
            .filter(|item| matches!(item, LayerItem::Primitive(_)))
            .count()
    }

    #[test]
    fn instance_sweep_culls_fully_covered_cullee() {
        let items = vec![
            cullee(scaled_bounds(10., 10., 40., 40.)),
            cullee(scaled_bounds(200., 200., 40., 40.)),
            opaque_quad(scaled_bounds(0., 0., 100., 100.)),
        ];
        let kept = cull_covered_instances(items);
        assert_eq!(kept.len(), 2, "the covered cullee must not emit");
    }

    #[test]
    fn instance_sweep_respects_paint_order() {
        // The occluder painted FIRST is beneath the cullee and must not cull
        // it; painted last, it must.
        let below = vec![
            opaque_quad(scaled_bounds(0., 0., 100., 100.)),
            cullee(scaled_bounds(10., 10., 40., 40.)),
        ];
        assert_eq!(cull_covered_instances(below).len(), 2);

        let above = vec![
            cullee(scaled_bounds(10., 10., 40., 40.)),
            opaque_quad(scaled_bounds(0., 0., 100., 100.)),
        ];
        assert_eq!(cull_covered_instances(above).len(), 1);
    }

    #[test]
    fn instance_sweep_translucent_occluder_never_culls() {
        let items = vec![
            cullee(scaled_bounds(10., 10., 40., 40.)),
            // Alpha one-half background over the same extent.
            LayerItem::Primitive(Primitive::Quad(scene_quad(
                scaled_bounds(0., 0., 100., 100.),
                false,
            ))),
        ];
        assert_eq!(kept_count(&cull_covered_instances(items)), 2);
    }

    #[test]
    fn instance_sweep_corner_radius_inset_leaves_a_visible_sliver() {
        let mut covering = scene_quad(scaled_bounds(0., 0., 100., 100.), true);
        covering.corner_radii.top_left = ScaledPixels(15.);
        let items = vec![
            cullee(scaled_bounds(0., 0., 100., 100.)),
            LayerItem::Primitive(Primitive::Quad(covering)),
        ];
        // The inset leaves the rounded corners uncovered, so the flush
        // cullee survives even though the occluder's rectangle overlaps it
        // almost everywhere.
        assert_eq!(kept_count(&cull_covered_instances(items)), 2);
    }

    #[test]
    fn instance_sweep_backdrop_filter_poisons_beneath_it() {
        let items = vec![
            cullee(scaled_bounds(30., 30., 20., 20.)),
            opaque_quad(scaled_bounds(0., 0., 100., 100.)),
            backdrop_filter(scaled_bounds(10., 10., 60., 60.), 8.),
        ];
        // The backdrop reads the pixels behind it within its dilated bounds,
        // so the covered cullee beneath it must still emit.
        assert_eq!(kept_count(&cull_covered_instances(items)), 3);
    }

    #[test]
    fn instance_sweep_backdrop_filter_outside_its_bounds_still_culls() {
        let items = vec![
            cullee(scaled_bounds(300., 300., 20., 20.)),
            opaque_quad(scaled_bounds(280., 280., 60., 60.)),
            backdrop_filter(scaled_bounds(10., 10., 60., 60.), 8.),
        ];
        assert_eq!(kept_count(&cull_covered_instances(items)), 2);
    }

    #[test]
    fn instance_sweep_filter_group_protects_interior_and_margin() {
        let group_bounds = scaled_bounds(50., 50., 80., 80.);
        let far_cullee = scaled_bounds(400., 400., 20., 20.);
        let items = vec![
            // Beneath the group, inside the dilated margin (blur 10): kept —
            // the group samples neighbouring pixels at its edge.
            cullee(scaled_bounds(55., 55., 10., 10.)),
            // Far outside the group: normal coverage rules apply.
            cullee(far_cullee),
            opaque_quad(scaled_bounds(380., 380., 60., 60.)),
            filter_boundary(group_bounds, 10., true),
            // Inside the group: never culled, never an occluder itself.
            cullee(scaled_bounds(60., 60., 20., 20.)),
            opaque_quad(scaled_bounds(55., 55., 70., 70.)),
            filter_boundary(group_bounds, 10., false),
        ];
        let kept = cull_covered_instances(items);
        // Six survivors: both markers, the margin cullee, both group members,
        // and the group-interior occluder. Only the far cullee goes.
        assert_eq!(kept.len(), 6, "group interior and margin survive");
        assert!(
            !kept.iter().any(|item| matches!(
                item,
                LayerItem::Primitive(Primitive::Quad(quad)) if quad.bounds == far_cullee
            )),
            "coverage outside the group still applies"
        );
    }

    #[test]
    fn instance_sweep_nested_references_are_transparent_to_coverage() {
        let occluder = opaque_quad(scaled_bounds(0., 0., 100., 100.));
        let items = vec![
            cullee(scaled_bounds(10., 10., 40., 40.)),
            LayerItem::Nested(crate::layer::LayerKey(7)),
            occluder.clone(),
        ];
        let kept = cull_covered_instances(items);
        // The nested reference survives untouched, and the cullee beneath it
        // is dropped exactly as if the reference were not there.
        assert_eq!(kept.len(), 2);
        assert!(matches!(&kept[0], LayerItem::Nested(crate::layer::LayerKey(7))));
        assert!(matches!(
            &kept[1],
            LayerItem::Primitive(Primitive::Quad(quad)) if quad.bounds == scaled_bounds(0., 0., 100., 100.)
        ));
    }

    #[test]
    fn instance_sweep_validate_passes_agree_on_mixed_streams() {
        let streams: Vec<Vec<LayerItem>> = vec![
            vec![],
            vec![cullee(scaled_bounds(0., 0., 10., 10.))],
            vec![
                cullee(scaled_bounds(10., 10., 40., 40.)),
                LayerItem::Nested(crate::layer::LayerKey(1)),
                opaque_quad(scaled_bounds(0., 0., 100., 100.)),
            ],
            vec![
                cullee(scaled_bounds(30., 30., 20., 20.)),
                opaque_quad(scaled_bounds(0., 0., 100., 100.)),
                backdrop_filter(scaled_bounds(5., 5., 70., 70.), 6.),
                cullee(scaled_bounds(500., 500., 20., 20.)),
            ],
            vec![
                cullee(scaled_bounds(52., 52., 8., 8.)),
                filter_boundary(scaled_bounds(50., 50., 40., 40.), 12., true),
                cullee(scaled_bounds(60., 60., 10., 10.)),
                opaque_quad(scaled_bounds(52., 52., 26., 26.)),
                filter_boundary(scaled_bounds(50., 50., 40., 40.), 12., false),
                cullee(scaled_bounds(900., 900., 30., 30.)),
                opaque_quad(scaled_bounds(890., 890., 50., 50.)),
            ],
            // Two side-by-side occluders jointly covering one wide cullee.
            vec![
                cullee(scaled_bounds(0., 0., 200., 50.)),
                opaque_quad(scaled_bounds(0., 0., 100., 50.)),
                opaque_quad(scaled_bounds(100., 0., 100., 50.)),
            ],
        ];
        for stream in &streams {
            assert_eq!(
                compute_keep_mask(stream),
                compute_keep_mask_independently(stream),
                "sweep and independent passes disagree on {} items",
                stream.len()
            );
        }
    }

    #[test]
    fn quad_opaque_region_rejects_alpha_below_one() {
        let mut quad = scene_quad(scaled_bounds(0., 0., 50., 50.), true);
        quad.background.solid.a = 0.9;
        assert_eq!(quad_opaque_region(&quad), None);
    }

    #[test]
    fn quad_opaque_region_rejects_gradients() {
        let mut quad = scene_quad(scaled_bounds(0., 0., 50., 50.), true);
        quad.background.tag = crate::BackgroundTag::LinearGradient;
        assert_eq!(quad_opaque_region(&quad), None);
    }

    #[test]
    fn quad_opaque_region_insets_for_corner_radius() {
        let mut quad = scene_quad(scaled_bounds(0., 0., 100., 100.), true);
        quad.corner_radii.bottom_right = ScaledPixels(10.);
        // The layer-tier rule insets uniformly by the largest radius.
        assert_eq!(
            quad_opaque_region(&quad),
            Some(scaled_bounds(10., 10., 80., 80.))
        );
    }

    #[test]
    fn quad_opaque_region_insets_for_translucent_border() {
        let mut quad = scene_quad(scaled_bounds(0., 0., 100., 100.), true);
        quad.border_color.a = 0.5;
        quad.border_widths.left = ScaledPixels(6.);
        quad.border_widths.top = ScaledPixels(2.);
        quad.border_widths.right = ScaledPixels(4.);
        quad.border_widths.bottom = ScaledPixels(8.);
        // ...and uniformly by the widest border edge.
        assert_eq!(
            quad_opaque_region(&quad),
            Some(scaled_bounds(8., 8., 84., 84.))
        );
    }

    #[test]
    fn quad_opaque_region_dashed_border_insets_like_a_translucent_one() {
        let mut quad = scene_quad(scaled_bounds(0., 0., 100., 100.), true);
        quad.border_color.a = 1.0;
        quad.border_style = BorderStyle::Dashed;
        quad.border_widths.left = ScaledPixels(5.);
        quad.border_widths.top = ScaledPixels(5.);
        quad.border_widths.right = ScaledPixels(5.);
        quad.border_widths.bottom = ScaledPixels(5.);
        assert_eq!(
            quad_opaque_region(&quad),
            Some(scaled_bounds(5., 5., 90., 90.))
        );
    }

    #[test]
    fn quad_opaque_region_clips_to_the_content_mask() {
        let mut quad = scene_quad(scaled_bounds(0., 0., 100., 100.), true);
        quad.content_mask = crate::ContentMask {
            bounds: scaled_bounds(25., 25., 25., 25.),
        };
        assert_eq!(
            quad_opaque_region(&quad),
            Some(scaled_bounds(25., 25., 25., 25.))
        );
    }

    #[test]
    fn quad_opaque_region_none_when_clipped_away() {
        let mut quad = scene_quad(scaled_bounds(0., 0., 100., 100.), true);
        quad.content_mask = crate::ContentMask {
            bounds: scaled_bounds(200., 200., 10., 10.),
        };
        assert_eq!(quad_opaque_region(&quad), None);
    }
}
