//! The one geometry type Phase 3's ordering and occlusion passes need, and
//! nothing else. See docs/gpu-native-architecture.md §5.1, §5.2.
//!
//! Not in §3's file map, and deliberately minimal, for exactly the reason
//! `boundary/policy.rs` gives for declaring its own `Pixels`/`Size`: the real
//! geometry surface (`Bounds<T>`, `Point<T>`, `Size<T>`, `Pixels`,
//! `ScaledPixels`) is part of the frontend contract §7 freezes, it still lives
//! in the legacy crate, and §3 gives `wgpui-core` no geometry module to move it
//! into yet. Pulling the legacy crate across the boundary §3 draws would be
//! worse than declaring the two hundred bytes of rectangle arithmetic both
//! compute passes actually need. Whichever phase moves geometry into the
//! workspace deletes this file and re-points its two users.
//!
//! **`f32`, not a unit-typed scalar, on purpose.** Both consumers exist to be
//! ported to WGSL, where the only float is `f32`. Every predicate here is
//! written so the Rust and the WGSL evaluate the *same* expression in the same
//! order, because §5.2's differential harness compares them for exact equality
//! rather than for approximate agreement.

use std::cmp::Ordering;

/// An axis-aligned rectangle in the owning layer's coordinate space.
///
/// Stored as min/max rather than origin/size because every predicate below
/// wants edges, and because that is the form the WGSL port reads out of a
/// `vec4<f32>`.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Rect {
    /// Left edge.
    pub min_x: f32,
    /// Top edge.
    pub min_y: f32,
    /// Right edge.
    pub max_x: f32,
    /// Bottom edge.
    pub max_y: f32,
}

impl Rect {
    /// A rectangle covering nothing, and covered by nothing.
    pub const EMPTY: Rect = Rect {
        min_x: 0.0,
        min_y: 0.0,
        max_x: 0.0,
        max_y: 0.0,
    };

    /// A rectangle from a top-left corner and a size, the form
    /// [`crate::patch::primitive::Quad`] carries.
    pub const fn from_origin_size(origin: [f32; 2], size: [f32; 2]) -> Rect {
        Rect {
            min_x: origin[0],
            min_y: origin[1],
            max_x: origin[0] + size[0],
            max_y: origin[1] + size[1],
        }
    }

    /// Width, which may be zero or negative for a degenerate rectangle.
    pub fn width(&self) -> f32 {
        self.max_x - self.min_x
    }

    /// Height, which may be zero or negative for a degenerate rectangle.
    pub fn height(&self) -> f32 {
        self.max_y - self.min_y
    }

    /// Whether this rectangle encloses no area at all.
    ///
    /// Matches the legacy sweep's own test (`src/occlusion.rs` treats a clipped
    /// region with non-positive width or height as absent), so a zero-height
    /// rectangle is empty rather than a hairline.
    ///
    /// Written against `partial_cmp` rather than as `max <= min` because the two
    /// disagree on NaN and this one has to answer "empty". The legacy sweep
    /// works in `ScaledPixels` and cannot produce a NaN edge; this type takes
    /// raw floats, and an unordered edge must never read as covering area.
    pub fn is_empty(&self) -> bool {
        !matches!(self.max_x.partial_cmp(&self.min_x), Some(Ordering::Greater))
            || !matches!(self.max_y.partial_cmp(&self.min_y), Some(Ordering::Greater))
    }

    /// The overlapping region, which may be empty.
    pub fn intersect(&self, other: &Rect) -> Rect {
        Rect {
            min_x: max_f32(self.min_x, other.min_x),
            min_y: max_f32(self.min_y, other.min_y),
            max_x: min_f32(self.max_x, other.max_x),
            max_y: min_f32(self.max_y, other.max_y),
        }
    }

    /// The smallest rectangle containing both.
    pub fn union(&self, other: &Rect) -> Rect {
        Rect {
            min_x: min_f32(self.min_x, other.min_x),
            min_y: min_f32(self.min_y, other.min_y),
            max_x: max_f32(self.max_x, other.max_x),
            max_y: max_f32(self.max_y, other.max_y),
        }
    }

    /// Whether the two rectangles share any area.
    ///
    /// Strict on every edge, exactly as `Bounds::intersects` is in the legacy
    /// backend (`src/geometry.rs`, consumed by `src/bounds_tree.rs`): two
    /// rectangles that merely touch along an edge do not intersect, and so do
    /// not step each other's painter order. The ordering pass's agreement with
    /// today's `BoundsTree` depends on this being the *same* predicate, not a
    /// similar one.
    pub fn intersects(&self, other: &Rect) -> bool {
        self.min_x < other.max_x
            && other.min_x < self.max_x
            && self.min_y < other.max_y
            && other.min_y < self.max_y
    }

    /// Half the perimeter — `BoundsTree`'s surface-area heuristic.
    pub fn half_perimeter(&self) -> f32 {
        self.width() + self.height()
    }

    /// This rectangle grown by `amount` on every side. Used for a filter's
    /// blur margin (R-N §8.3's last condition).
    pub fn dilate(&self, amount: f32) -> Rect {
        Rect {
            min_x: self.min_x - amount,
            min_y: self.min_y - amount,
            max_x: self.max_x + amount,
            max_y: self.max_y + amount,
        }
    }

    /// This rectangle shrunk by `amount` on every side, which may empty it.
    pub fn inset(&self, amount: f32) -> Rect {
        Rect {
            min_x: self.min_x + amount,
            min_y: self.min_y + amount,
            max_x: self.max_x - amount,
            max_y: self.max_y - amount,
        }
    }

    /// Whether every point of `other` lies within this rectangle.
    pub fn contains(&self, other: &Rect) -> bool {
        self.min_x <= other.min_x
            && self.min_y <= other.min_y
            && self.max_x >= other.max_x
            && self.max_y >= other.max_y
    }

    /// The four edges, in the order the WGSL port reads them out of a
    /// `vec4<f32>`.
    pub const fn to_array(self) -> [f32; 4] {
        [self.min_x, self.min_y, self.max_x, self.max_y]
    }
}

/// `f32::min`, spelled out so the Rust and the WGSL agree on NaN handling.
///
/// WGSL's `min` is unspecified for NaN operands and Rust's `f32::min` returns
/// the non-NaN operand; neither consumer ever produces a NaN coordinate (every
/// input is a finite layout result), so the difference is unreachable — but the
/// comparison form below is the one both languages compile to the same
/// instruction for finite inputs, which is the property the differential
/// harness relies on.
fn min_f32(left: f32, right: f32) -> f32 {
    if left < right { left } else { right }
}

fn max_f32(left: f32, right: f32) -> f32 {
    if left > right { left } else { right }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_touching_edge_is_not_an_intersection() {
        let left = Rect::from_origin_size([0.0, 0.0], [10.0, 10.0]);
        let right = Rect::from_origin_size([10.0, 0.0], [10.0, 10.0]);
        assert!(!left.intersects(&right));
        assert!(left.intersect(&right).is_empty());
    }

    #[test]
    fn a_zero_height_rectangle_is_empty() {
        assert!(Rect::from_origin_size([0.0, 0.0], [10.0, 0.0]).is_empty());
        assert!(Rect::from_origin_size([0.0, 0.0], [0.0, 10.0]).is_empty());
        assert!(!Rect::from_origin_size([0.0, 0.0], [1.0, 1.0]).is_empty());
    }

    #[test]
    fn inset_and_dilate_are_inverses_while_the_rectangle_survives() {
        let bounds = Rect::from_origin_size([5.0, 6.0], [40.0, 30.0]);
        assert_eq!(bounds.inset(4.0).dilate(4.0), bounds);
        assert!(bounds.inset(100.0).is_empty());
    }

    #[test]
    fn contains_is_inclusive_at_the_edges() {
        let outer = Rect::from_origin_size([0.0, 0.0], [10.0, 10.0]);
        assert!(outer.contains(&outer));
        assert!(!outer.contains(&Rect::from_origin_size([0.0, 0.0], [10.1, 10.0])));
    }

    #[test]
    fn union_covers_both_operands() {
        let left = Rect::from_origin_size([0.0, 0.0], [10.0, 4.0]);
        let right = Rect::from_origin_size([20.0, -3.0], [5.0, 5.0]);
        let union = left.union(&right);
        assert!(union.contains(&left) && union.contains(&right));
        assert_eq!(union.half_perimeter(), 25.0 + 7.0);
    }
}
