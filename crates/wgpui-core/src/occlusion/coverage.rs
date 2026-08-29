//! The conservative opaque-region test itself: solid background, opacity
//! 1.0, corner-radius inset, border-opacity inset, no backdrop filter above,
//! blur-margin exemption (R-N §8.3). See docs/gpu-native-architecture.md
//! §5.2.
//!
//! # Written once, run twice
//!
//! §5.2's claim about the compute path is that it is "the same computation,
//! restated as data-parallel," not a different one. This file is where "the
//! same computation" is actually written down: pure, allocation-free,
//! `f32`-only geometry with no reference to a `Scene`, a `Layer`, a device, or
//! a queue. It is consumed three ways, and the three must agree exactly:
//!
//! 1. As the CPU reference the instance-tier sweep in
//!    [`crate::occlusion`] runs.
//! 2. As the oracle `WGPUI_OCCLUSION=validate` (R-N §8.5) diffs the compute
//!    path against.
//! 3. As the body of `shaders/occlusion.wgsl`, ported statement for statement.
//!
//! Every routine below is therefore written in the intersection of Rust and
//! WGSL: fixed-capacity arrays instead of `Vec`, explicit comparisons instead
//! of `f32::min`/`max`, and no early `return` that a WGSL uniformity rule would
//! reject. Where the two languages could disagree — NaN ordering, `min`/`max`
//! operand choice — the expression is spelled out rather than delegated. See
//! [`crate::geometry`]'s module doc for the same reasoning applied to `Rect`.
//!
//! # The one deliberate difference from `src/occlusion.rs`
//!
//! The legacy sweep collects an *unbounded* occluder list as it walks a layer
//! backwards, and tests each cullee against all of it. A compute shader has no
//! unbounded per-invocation storage, so the test here takes at most
//! [`MAX_OCCLUDERS`] occluders, chosen as the first that many qualifying
//! candidates in ascending paint order above the target.
//!
//! Dropping occluders is *conservative in the safe direction*: it can only
//! decide "not covered" for something the unbounded test would have culled — a
//! kept primitive, never a dropped one, so the rendered result is unchanged and
//! only the saving shrinks. It is applied identically on both sides so the CPU
//! reference and the compute path still agree bit for bit, which is what makes
//! the differential harness a test of the port rather than of the cap.

use crate::geometry::Rect;

/// The most occluders one coverage query considers.
///
/// Sized so the WGSL port's function-scope working arrays stay small: the
/// x-edge array is `2 * MAX_OCCLUDERS + 2` floats and the y-interval array is
/// `MAX_OCCLUDERS` pairs. Raising it costs shader scratch memory per
/// invocation; lowering it costs culls, never correctness.
pub const MAX_OCCLUDERS: usize = 32;

/// Working capacity for the x-edge list: both target edges plus two per
/// occluder.
pub const MAX_EDGES: usize = 2 * MAX_OCCLUDERS + 2;

/// Everything R-N §8.3 needs to know about one primitive to decide whether it
/// contributes a conservative opaque region.
///
/// The five conditions §8.3 lists map onto these fields one-for-one:
/// solid background ([`Self::background_is_solid`] plus
/// [`Self::background_alpha`]), `element_opacity == 1.0`
/// ([`Self::element_opacity`]), corner-radius inset
/// ([`Self::max_corner_radius`]), border opacity ([`Self::border_is_opaque`]
/// plus [`Self::max_border_width`]), and no backdrop filter
/// ([`Self::has_backdrop_filter`]). The sixth — the blur margin — is not a
/// property of one primitive and lives in [`crate::occlusion`]'s poison list
/// instead, for the same reason the legacy sweep handles it there.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct OccluderStyle {
    /// Whether the fill is a flat colour. A gradient or pattern is excluded
    /// without further analysis, exactly as R-N §8.3 requires.
    pub background_is_solid: bool,
    /// The fill's own alpha.
    pub background_alpha: f32,
    /// The separate paint-time opacity multiplier (`Window::element_opacity`).
    ///
    /// Kept as its own field even though the legacy quad-level helper folds it
    /// into the background alpha before this point: R-N §8.3 names it as an
    /// independent condition, and a caller that has *not* pre-multiplied must
    /// be able to say so.
    pub element_opacity: f32,
    /// The largest of the four corner radii.
    pub max_corner_radius: f32,
    /// Whether the border is both alpha-one and solid-styled. A dashed border
    /// leaves gaps at alpha one, so it insets like a translucent one.
    pub border_is_opaque: bool,
    /// The widest of the four border edges.
    pub max_border_width: f32,
    /// Whether this primitive is itself a backdrop filter, which reads what is
    /// behind it and therefore never occludes.
    pub has_backdrop_filter: bool,
}

impl OccluderStyle {
    /// A fully opaque, square-cornered, borderless fill — the simplest thing
    /// that qualifies as an occluder.
    pub const OPAQUE: OccluderStyle = OccluderStyle {
        background_is_solid: true,
        background_alpha: 1.0,
        element_opacity: 1.0,
        max_corner_radius: 0.0,
        border_is_opaque: true,
        max_border_width: 0.0,
        has_backdrop_filter: false,
    };

    /// A fill that never qualifies, whatever its geometry.
    pub const TRANSLUCENT: OccluderStyle = OccluderStyle {
        background_alpha: 0.5,
        ..OccluderStyle::OPAQUE
    };
}

/// The conservative opaque region of one primitive, or `None` when it does not
/// fully cover even part of its own bounds.
///
/// `clip` is the primitive's content mask: only what survives the clip can hide
/// anything, which is the last step `src/occlusion.rs`'s `quad_opaque_region`
/// performs and the one a caller is most likely to forget.
pub fn opaque_region(bounds: Rect, clip: Rect, style: &OccluderStyle) -> Option<Rect> {
    if !style.background_is_solid
        || style.background_alpha < 1.0
        || style.element_opacity < 1.0
        || style.has_backdrop_filter
    {
        return None;
    }

    // The corner radius always insets; a non-opaque border insets further, but
    // only if it is wider than the radius already removed.
    let mut inset_amount = style.max_corner_radius;
    if !style.border_is_opaque && style.max_border_width > inset_amount {
        inset_amount = style.max_border_width;
    }

    let mut region = bounds;
    if inset_amount > 0.0 {
        region = region.inset(inset_amount);
        if region.is_empty() {
            return None;
        }
    }

    region = region.intersect(&clip);
    if region.is_empty() { None } else { Some(region) }
}

/// Whether `target` is completely covered by `occluders`.
///
/// A vertical-slab sweep: cut the target into x-slices at every occluder edge,
/// and for each slice walk the occluders spanning it in ascending top order,
/// requiring their y-intervals to reach from the target's top edge to its
/// bottom without a gap. This is the same algorithm `src/occlusion.rs`'s
/// `fully_covered` runs, restated over fixed-capacity arrays so the WGSL port
/// is a transcription rather than a redesign.
///
/// Only the first [`MAX_OCCLUDERS`] entries are considered; see this module's
/// doc for why dropping the rest is safe.
pub fn fully_covered(target: Rect, occluders: &[Rect]) -> bool {
    // A degenerate target covers no pixels, so nothing has to cover it. The
    // legacy sweep answers the same way, and the callers rely on it: a
    // fully-clipped-away primitive is culled rather than emitted.
    if target.is_empty() {
        return true;
    }

    // Clip every occluder to the target up front. Both the edge list and the
    // interval walk want the clipped form, and clipping once keeps the WGSL
    // port from re-deriving it inside two nested loops.
    let mut clipped = [Rect::EMPTY; MAX_OCCLUDERS];
    let mut clipped_count = 0usize;
    for region in occluders.iter().take(MAX_OCCLUDERS) {
        let overlap = region.intersect(&target);
        if !overlap.is_empty() {
            clipped[clipped_count] = overlap;
            clipped_count += 1;
        }
    }
    if clipped_count == 0 {
        return false;
    }

    let mut edges = [0.0f32; MAX_EDGES];
    let mut edge_count = 0usize;
    edge_count = push_edge(&mut edges, edge_count, target.min_x);
    edge_count = push_edge(&mut edges, edge_count, target.max_x);
    for region in clipped.iter().take(clipped_count) {
        edge_count = push_edge(&mut edges, edge_count, region.min_x);
        edge_count = push_edge(&mut edges, edge_count, region.max_x);
    }
    sort_ascending(&mut edges, edge_count);

    let mut slice = 0usize;
    while slice + 1 < edge_count {
        let left = edges[slice];
        let right = edges[slice + 1];
        slice += 1;
        // Duplicate edges survive the sort (dedup is the sweep's job, not the
        // sort's); an empty slice is covered vacuously.
        if right <= left {
            continue;
        }
        // Spelled exactly as the legacy sweep spells it — `left + (right -
        // left) / 2`, not `(left + right) / 2` — because the two differ in the
        // last bit for large coordinates and the differential harness compares
        // for equality.
        let midpoint = left + (right - left) / 2.0;

        let mut tops = [0.0f32; MAX_OCCLUDERS];
        let mut bottoms = [0.0f32; MAX_OCCLUDERS];
        let mut interval_count = 0usize;
        for region in clipped.iter().take(clipped_count) {
            if region.min_x <= midpoint && region.max_x >= midpoint {
                tops[interval_count] = region.min_y;
                bottoms[interval_count] = region.max_y;
                interval_count += 1;
            }
        }
        sort_intervals_by_top(&mut tops, &mut bottoms, interval_count);

        let mut covered_to = target.min_y;
        let mut index = 0usize;
        while index < interval_count {
            let top = tops[index];
            let bottom = bottoms[index];
            index += 1;
            if top > covered_to {
                return false;
            }
            if bottom > covered_to {
                covered_to = bottom;
            }
            if covered_to >= target.max_y {
                break;
            }
        }
        if covered_to < target.max_y {
            return false;
        }
    }
    true
}

/// Append one x-edge, refusing to overflow the fixed array.
///
/// The capacity is exact for [`MAX_OCCLUDERS`] occluders plus the two target
/// edges, so the guard can never fire for a caller that respects the cap; it
/// exists so a future caller that does not gets a dropped edge (a possible
/// missed cull) rather than an out-of-bounds write.
fn push_edge(edges: &mut [f32; MAX_EDGES], count: usize, value: f32) -> usize {
    if count >= MAX_EDGES {
        return count;
    }
    edges[count] = value;
    count + 1
}

/// Insertion sort over the first `count` entries.
///
/// Insertion sort rather than `slice::sort_unstable` because the WGSL port has
/// no sort to call and this is the shape it transcribes to; `count` never
/// exceeds [`MAX_EDGES`], so the quadratic term is bounded by a constant.
fn sort_ascending(values: &mut [f32; MAX_EDGES], count: usize) {
    let mut index = 1usize;
    while index < count {
        let value = values[index];
        let mut position = index;
        while position > 0 && values[position - 1] > value {
            values[position] = values[position - 1];
            position -= 1;
        }
        values[position] = value;
        index += 1;
    }
}

/// Insertion sort of the `(top, bottom)` pairs by `top`, keeping the two
/// arrays in step. Stable, so equal tops keep their collection order on both
/// sides of the differential.
fn sort_intervals_by_top(
    tops: &mut [f32; MAX_OCCLUDERS],
    bottoms: &mut [f32; MAX_OCCLUDERS],
    count: usize,
) {
    let mut index = 1usize;
    while index < count {
        let top = tops[index];
        let bottom = bottoms[index];
        let mut position = index;
        while position > 0 && tops[position - 1] > top {
            tops[position] = tops[position - 1];
            bottoms[position] = bottoms[position - 1];
            position -= 1;
        }
        tops[position] = top;
        bottoms[position] = bottom;
        index += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(x: f32, y: f32, width: f32, height: f32) -> Rect {
        Rect::from_origin_size([x, y], [width, height])
    }

    fn unclipped() -> Rect {
        rect(-100_000.0, -100_000.0, 200_000.0, 200_000.0)
    }

    // --- R-N §8.3's five conditions, one test each ---

    #[test]
    fn condition_one_a_non_solid_background_never_occludes() {
        let style = OccluderStyle {
            background_is_solid: false,
            ..OccluderStyle::OPAQUE
        };
        assert_eq!(
            opaque_region(rect(0.0, 0.0, 100.0, 100.0), unclipped(), &style),
            None
        );
    }

    #[test]
    fn condition_one_a_translucent_fill_never_occludes() {
        assert_eq!(
            opaque_region(
                rect(0.0, 0.0, 100.0, 100.0),
                unclipped(),
                &OccluderStyle::TRANSLUCENT
            ),
            None
        );
    }

    #[test]
    fn condition_two_element_opacity_below_one_never_occludes() {
        let style = OccluderStyle {
            element_opacity: 0.999,
            ..OccluderStyle::OPAQUE
        };
        assert_eq!(
            opaque_region(rect(0.0, 0.0, 100.0, 100.0), unclipped(), &style),
            None
        );
    }

    #[test]
    fn condition_three_corner_radius_insets_on_every_side() {
        let style = OccluderStyle {
            max_corner_radius: 10.0,
            ..OccluderStyle::OPAQUE
        };
        assert_eq!(
            opaque_region(rect(0.0, 0.0, 100.0, 100.0), unclipped(), &style),
            Some(rect(10.0, 10.0, 80.0, 80.0))
        );
    }

    #[test]
    fn condition_three_a_radius_that_eats_the_rectangle_yields_nothing() {
        let style = OccluderStyle {
            max_corner_radius: 10.0,
            ..OccluderStyle::OPAQUE
        };
        assert_eq!(
            opaque_region(rect(0.0, 0.0, 20.0, 20.0), unclipped(), &style),
            None
        );
    }

    #[test]
    fn condition_four_a_translucent_border_insets_by_its_width() {
        let style = OccluderStyle {
            border_is_opaque: false,
            max_border_width: 5.0,
            ..OccluderStyle::OPAQUE
        };
        assert_eq!(
            opaque_region(rect(0.0, 0.0, 100.0, 100.0), unclipped(), &style),
            Some(rect(5.0, 5.0, 90.0, 90.0))
        );
    }

    #[test]
    fn condition_four_an_opaque_border_costs_nothing() {
        let style = OccluderStyle {
            border_is_opaque: true,
            max_border_width: 30.0,
            max_corner_radius: 10.0,
            ..OccluderStyle::OPAQUE
        };
        assert_eq!(
            opaque_region(rect(0.0, 0.0, 100.0, 100.0), unclipped(), &style),
            Some(rect(10.0, 10.0, 80.0, 80.0)),
            "an opaque border does not inset; only the corner radius does"
        );
    }

    #[test]
    fn condition_four_the_larger_of_radius_and_border_wins() {
        let wide_border = OccluderStyle {
            border_is_opaque: false,
            max_border_width: 20.0,
            max_corner_radius: 2.0,
            ..OccluderStyle::OPAQUE
        };
        assert_eq!(
            opaque_region(rect(0.0, 0.0, 100.0, 100.0), unclipped(), &wide_border),
            Some(rect(20.0, 20.0, 60.0, 60.0))
        );
        let wide_radius = OccluderStyle {
            border_is_opaque: false,
            max_border_width: 3.0,
            max_corner_radius: 15.0,
            ..OccluderStyle::OPAQUE
        };
        assert_eq!(
            opaque_region(rect(0.0, 0.0, 100.0, 100.0), unclipped(), &wide_radius),
            Some(rect(15.0, 15.0, 70.0, 70.0))
        );
    }

    #[test]
    fn condition_five_a_backdrop_filter_never_occludes_however_opaque() {
        let style = OccluderStyle {
            has_backdrop_filter: true,
            ..OccluderStyle::OPAQUE
        };
        assert_eq!(
            opaque_region(rect(0.0, 0.0, 100.0, 100.0), unclipped(), &style),
            None
        );
    }

    #[test]
    fn the_content_mask_clips_the_region_and_can_erase_it() {
        assert_eq!(
            opaque_region(
                rect(0.0, 0.0, 100.0, 100.0),
                rect(25.0, 25.0, 25.0, 25.0),
                &OccluderStyle::OPAQUE
            ),
            Some(rect(25.0, 25.0, 25.0, 25.0))
        );
        assert_eq!(
            opaque_region(
                rect(0.0, 0.0, 100.0, 100.0),
                rect(500.0, 500.0, 10.0, 10.0),
                &OccluderStyle::OPAQUE
            ),
            None
        );
    }

    // --- The coverage sweep. These mirror `src/occlusion.rs`'s own tests case
    // for case, because agreeing with that implementation is the point.

    #[test]
    fn coverage_requires_every_point() {
        let target = rect(0.0, 0.0, 100.0, 100.0);
        assert!(fully_covered(target, &[rect(0.0, 0.0, 100.0, 100.0)]));
        assert!(!fully_covered(target, &[rect(0.0, 0.0, 50.0, 100.0)]));
        assert!(fully_covered(
            target,
            &[rect(0.0, 0.0, 50.0, 100.0), rect(50.0, 0.0, 50.0, 100.0)]
        ));
    }

    #[test]
    fn coverage_with_a_horizontal_gap() {
        let target = rect(0.0, 0.0, 100.0, 100.0);
        assert!(!fully_covered(
            target,
            &[rect(0.0, 0.0, 40.0, 100.0), rect(60.0, 0.0, 40.0, 100.0)]
        ));
    }

    #[test]
    fn coverage_with_a_vertical_gap() {
        let target = rect(0.0, 0.0, 100.0, 100.0);
        assert!(!fully_covered(
            target,
            &[rect(0.0, 0.0, 100.0, 40.0), rect(0.0, 60.0, 100.0, 40.0)]
        ));
    }

    #[test]
    fn coverage_with_multiple_x_slices_not_all_covered() {
        let target = rect(0.0, 0.0, 200.0, 100.0);
        assert!(!fully_covered(
            target,
            &[
                rect(0.0, 0.0, 50.0, 100.0),
                rect(50.0, 0.0, 50.0, 50.0),
                rect(50.0, 50.0, 150.0, 50.0),
                rect(100.0, 0.0, 50.0, 50.0),
            ]
        ));
    }

    #[test]
    fn coverage_of_an_l_shape_made_of_three_occluders() {
        let target = rect(0.0, 0.0, 60.0, 60.0);
        assert!(fully_covered(
            target,
            &[
                rect(0.0, 0.0, 60.0, 30.0),
                rect(0.0, 30.0, 30.0, 30.0),
                rect(30.0, 30.0, 30.0, 30.0),
            ]
        ));
    }

    #[test]
    fn an_empty_occluder_list_never_covers_a_real_target() {
        assert!(!fully_covered(rect(10.0, 10.0, 50.0, 50.0), &[]));
    }

    #[test]
    fn a_degenerate_target_is_covered_by_nothing_at_all() {
        assert!(fully_covered(rect(0.0, 0.0, 100.0, 0.0), &[]));
        assert!(fully_covered(rect(0.0, 0.0, 0.0, 100.0), &[]));
    }

    #[test]
    fn an_occluder_wholly_outside_the_target_does_not_cover_it() {
        assert!(!fully_covered(
            rect(0.0, 0.0, 50.0, 50.0),
            &[rect(100.0, 100.0, 50.0, 50.0)]
        ));
    }

    #[test]
    fn an_occluder_overhanging_the_target_still_covers_it() {
        assert!(fully_covered(
            rect(20.0, 20.0, 60.0, 60.0),
            &[rect(0.0, 0.0, 100.0, 100.0)]
        ));
    }

    #[test]
    fn duplicate_occluders_do_not_break_the_sweep() {
        let same = rect(0.0, 0.0, 100.0, 100.0);
        assert!(fully_covered(same, &[same, same, same]));
    }

    #[test]
    fn negative_coordinates_behave_the_same() {
        let target = rect(-50.0, -50.0, 100.0, 100.0);
        assert!(fully_covered(target, &[rect(-50.0, -50.0, 100.0, 100.0)]));
        assert!(!fully_covered(target, &[rect(-50.0, -50.0, 50.0, 100.0)]));
    }

    #[test]
    fn the_occluder_cap_drops_extras_rather_than_overflowing() {
        // Thirty-three one-pixel strips tile a 33px target exactly. Only the
        // first MAX_OCCLUDERS are considered, so the last strip's column is
        // left uncovered and the target survives — a missed cull, never a
        // wrong one.
        let target = rect(0.0, 0.0, (MAX_OCCLUDERS + 1) as f32, 50.0);
        let strips: Vec<Rect> = (0..=MAX_OCCLUDERS)
            .map(|index| rect(index as f32, 0.0, 1.0, 50.0))
            .collect();
        assert_eq!(strips.len(), MAX_OCCLUDERS + 1);
        assert!(!fully_covered(target, &strips));
        // The same strips minus the one past the cap do cover their target.
        let narrower = rect(0.0, 0.0, MAX_OCCLUDERS as f32, 50.0);
        assert!(fully_covered(narrower, &strips[..MAX_OCCLUDERS]));
    }

    #[test]
    fn a_full_row_of_occluders_at_the_cap_still_covers() {
        let target = rect(0.0, 0.0, MAX_OCCLUDERS as f32, 10.0);
        let strips: Vec<Rect> = (0..MAX_OCCLUDERS)
            .map(|index| rect(index as f32, 0.0, 1.0, 10.0))
            .collect();
        assert!(fully_covered(target, &strips));
    }

    #[test]
    fn sub_pixel_edges_are_handled_exactly() {
        let target = rect(0.5, 0.5, 99.5, 99.5);
        assert!(fully_covered(target, &[rect(0.0, 0.0, 100.0, 100.0)]));
        assert!(!fully_covered(target, &[rect(0.5, 0.5, 49.5, 99.5)]));
    }
}
