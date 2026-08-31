//! Native geometry primitives shared by layout, retained style, and rendering.
//!
//! The retained renderer continues to use its existing float arrays at the GPU
//! boundary. These types provide the typed frontend contract without pulling
//! the legacy crate into the native crates.

use std::cmp::Ordering;
use std::fmt::{self, Display};
use std::hash::{Hash, Hasher};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use serde::{Deserialize, Deserializer, Serialize, de};

/// An axis-aligned rectangle in the owning layer's coordinate space.
///
/// Stored as min/max rather than origin/size because every predicate below
/// wants edges, and because that is the form the WGSL port reads out of a
/// `vec4<f32>`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
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

/// A logical pixel distance.
#[derive(Copy, Clone, Default, PartialEq, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Pixels(pub f32);

impl fmt::Debug for Pixels {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(self, formatter)
    }
}

impl Eq for Pixels {}

impl PartialOrd for Pixels {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Pixels {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl Hash for Pixels {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl Pixels {
    pub const ZERO: Self = Self(0.0);
    pub const MIN: Self = Self(f32::MIN);
    pub const MAX: Self = Self(f32::MAX);

    pub const fn value(self) -> f32 {
        self.0
    }

    pub const fn to_f32(self) -> f32 {
        self.0
    }

    pub const fn to_f64(self) -> f64 {
        self.0 as f64
    }

    pub fn scaled(self, factor: f32) -> Self {
        Self(self.0 * factor)
    }

    pub fn floor(self) -> Self {
        Self(self.0.floor())
    }

    pub fn round(self) -> Self {
        Self(self.0.round())
    }

    pub fn ceil(self) -> Self {
        Self(self.0.ceil())
    }

    pub fn max(self, other: Self) -> Self {
        Self(self.0.max(other.0))
    }

    pub fn min(self, other: Self) -> Self {
        Self(self.0.min(other.0))
    }

    pub fn half(self) -> Self {
        Self(self.0 / 2.0)
    }

    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }

    pub fn pow(self, exponent: f32) -> Self {
        Self(self.0.powf(exponent))
    }

    pub fn signum(self) -> f32 {
        self.0.signum()
    }
}

impl Add for Pixels {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self(self.0 + other.0)
    }
}

impl AddAssign for Pixels {
    fn add_assign(&mut self, other: Self) {
        self.0 += other.0;
    }
}

impl Sub for Pixels {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self(self.0 - other.0)
    }
}

impl SubAssign for Pixels {
    fn sub_assign(&mut self, other: Self) {
        self.0 -= other.0;
    }
}

impl Neg for Pixels {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl Mul<f32> for Pixels {
    type Output = Self;

    fn mul(self, factor: f32) -> Self::Output {
        Self(self.0 * factor)
    }
}

impl MulAssign<f32> for Pixels {
    fn mul_assign(&mut self, factor: f32) {
        self.0 *= factor;
    }
}

impl Mul<Pixels> for f32 {
    type Output = Pixels;

    fn mul(self, pixels: Pixels) -> Self::Output {
        pixels * self
    }
}

impl Div for Pixels {
    type Output = f32;

    fn div(self, other: Self) -> Self::Output {
        self.0 / other.0
    }
}

impl DivAssign for Pixels {
    fn div_assign(&mut self, other: Self) {
        self.0 /= other.0;
    }
}

impl std::ops::Rem for Pixels {
    type Output = Self;

    fn rem(self, other: Self) -> Self::Output {
        Self(self.0 % other.0)
    }
}

impl std::ops::RemAssign for Pixels {
    fn rem_assign(&mut self, other: Self) {
        self.0 %= other.0;
    }
}

impl Mul<usize> for Pixels {
    type Output = Self;

    fn mul(self, factor: usize) -> Self::Output {
        self * factor as f32
    }
}

impl Mul<Pixels> for usize {
    type Output = Pixels;

    fn mul(self, pixels: Pixels) -> Self::Output {
        pixels * self
    }
}

impl From<f32> for Pixels {
    fn from(value: f32) -> Self {
        Self(value)
    }
}

impl From<f64> for Pixels {
    fn from(value: f64) -> Self {
        Self(value as f32)
    }
}

impl From<Pixels> for f32 {
    fn from(value: Pixels) -> Self {
        value.0
    }
}

impl Display for Pixels {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}px", self.0)
    }
}

impl TryFrom<&str> for Pixels {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        value
            .strip_suffix("px")
            .ok_or_else(|| format!("expected a pixel length such as 12px, got {value:?}"))?
            .parse::<f32>()
            .map(Self)
            .map_err(|error| format!("invalid pixel length {value:?}: {error}"))
    }
}

/// A width/height pair.
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub struct Size<T> {
    pub width: T,
    pub height: T,
}

impl<T> Size<T> {
    pub const fn new(width: T, height: T) -> Self {
        Self { width, height }
    }

    pub fn map<U>(&self, map: impl FnMut(T) -> U) -> Size<U>
    where
        T: Clone,
    {
        let mut map = map;
        Size::new(map(self.width.clone()), map(self.height.clone()))
    }
}

impl Size<Pixels> {
    pub const ZERO: Self = Self {
        width: Pixels::ZERO,
        height: Pixels::ZERO,
    };

    pub const fn pixels(width: f32, height: f32) -> Self {
        Self {
            width: Pixels(width),
            height: Pixels(height),
        }
    }

    pub fn scaled(self, factor: f32) -> Self {
        Self {
            width: self.width * factor,
            height: self.height * factor,
        }
    }
}

impl<T: Clone + Half> Size<T> {
    pub fn center(&self) -> Point<T> {
        point(self.width.clone().half(), self.height.clone().half())
    }
}

impl<T: Add<Output = T>> Add for Size<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self::new(self.width + other.width, self.height + other.height)
    }
}

impl<T: Sub<Output = T>> Sub for Size<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self::new(self.width - other.width, self.height - other.height)
    }
}

impl<T: Clone + PartialOrd> Size<T> {
    pub fn max(&self, other: &Self) -> Self {
        Self::new(
            if self.width >= other.width {
                self.width.clone()
            } else {
                other.width.clone()
            },
            if self.height >= other.height {
                self.height.clone()
            } else {
                other.height.clone()
            },
        )
    }

    pub fn min(&self, other: &Self) -> Self {
        Self::new(
            if self.width <= other.width {
                self.width.clone()
            } else {
                other.width.clone()
            },
            if self.height <= other.height {
                self.height.clone()
            } else {
                other.height.clone()
            },
        )
    }
}

impl<T> From<Point<T>> for Size<T> {
    fn from(value: Point<T>) -> Self {
        Self::new(value.x, value.y)
    }
}

impl From<Size<Pixels>> for Size<DefiniteLength> {
    fn from(value: Size<Pixels>) -> Self {
        Self::new(value.width.into(), value.height.into())
    }
}

/// Constructs a width/height pair.
pub const fn size<T>(width: T, height: T) -> Size<T> {
    Size { width, height }
}

/// A point in a two-dimensional coordinate space.
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub struct Point<T> {
    pub x: T,
    pub y: T,
}

impl<T> Point<T> {
    pub const fn new(x: T, y: T) -> Self {
        Self { x, y }
    }

    pub fn map<U>(&self, mut map: impl FnMut(T) -> U) -> Point<U>
    where
        T: Clone,
    {
        Point::new(map(self.x.clone()), map(self.y.clone()))
    }
}

/// Constructs a point.
pub const fn point<T>(x: T, y: T) -> Point<T> {
    Point { x, y }
}

impl<T: Add<Output = T>> Add for Point<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self::new(self.x + other.x, self.y + other.y)
    }
}

impl<T: AddAssign> AddAssign for Point<T> {
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
    }
}

impl<T: Sub<Output = T>> Sub for Point<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self::new(self.x - other.x, self.y - other.y)
    }
}

impl<T: SubAssign> SubAssign for Point<T> {
    fn sub_assign(&mut self, other: Self) {
        self.x -= other.x;
        self.y -= other.y;
    }
}

impl<T: Neg<Output = T>> Neg for Point<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y)
    }
}

impl<T, Rhs> Mul<Rhs> for Point<T>
where
    T: Mul<Rhs, Output = T> + Clone,
    Rhs: Clone,
{
    type Output = Self;

    fn mul(self, factor: Rhs) -> Self::Output {
        Self::new(self.x * factor.clone(), self.y * factor)
    }
}

impl<T> Point<T>
where
    T: Sub<Output = T> + Clone,
{
    pub fn relative_to(&self, origin: &Self) -> Self {
        Self::new(
            self.x.clone() - origin.x.clone(),
            self.y.clone() - origin.y.clone(),
        )
    }
}

impl Point<Pixels> {
    pub fn scaled(self, factor: f32) -> Self {
        self * factor
    }

    pub fn magnitude(self) -> f64 {
        f64::from(self.x.0).hypot(f64::from(self.y.0))
    }
}

impl From<Point<Pixels>> for [f32; 2] {
    fn from(value: Point<Pixels>) -> Self {
        [value.x.0, value.y.0]
    }
}

impl From<Size<Pixels>> for [f32; 2] {
    fn from(value: Size<Pixels>) -> Self {
        [value.width.0, value.height.0]
    }
}

pub const fn bounds<T>(origin: Point<T>, size: Size<T>) -> Bounds<T> {
    Bounds { origin, size }
}

/// An origin and extent in one coordinate space.
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub struct Bounds<T> {
    pub origin: Point<T>,
    pub size: Size<T>,
}

impl<T> Bounds<T> {
    pub const fn new(origin: Point<T>, size: Size<T>) -> Self {
        Self { origin, size }
    }
}

impl<T: Clone + Sub<Output = T>> Bounds<T> {
    pub fn from_corners(top_left: Point<T>, bottom_right: Point<T>) -> Self {
        let origin = top_left.clone();
        Self::new(
            origin,
            size(
                bottom_right.x - top_left.x.clone(),
                bottom_right.y - top_left.y.clone(),
            ),
        )
    }
}

impl<T: Clone + Sub<Output = T> + Half> Bounds<T> {
    pub fn centered_at(center: Point<T>, size: Size<T>) -> Self {
        Self::new(
            point(
                center.x - size.width.clone().half(),
                center.y - size.height.clone().half(),
            ),
            size,
        )
    }
}

impl<T: Clone + Add<Output = T>> Bounds<T> {
    pub fn bottom_right(&self) -> Point<T> {
        point(
            self.origin.x.clone() + self.size.width.clone(),
            self.origin.y.clone() + self.size.height.clone(),
        )
    }
}

impl<T: Clone + PartialOrd + Add<Output = T>> Bounds<T> {
    pub fn intersects(&self, other: &Self) -> bool {
        let self_bottom_right = self.bottom_right();
        let other_bottom_right = other.bottom_right();
        self.origin.x < other_bottom_right.x
            && other.origin.x < self_bottom_right.x
            && self.origin.y < other_bottom_right.y
            && other.origin.y < self_bottom_right.y
    }
}

impl<T: Clone + Add<Output = T> + Sub<Output = T>> Bounds<T> {
    pub fn center(&self) -> Point<T>
    where
        T: Half,
    {
        point(
            self.origin.x.clone() + self.size.width.clone().half(),
            self.origin.y.clone() + self.size.height.clone().half(),
        )
    }

    pub fn half_perimeter(&self) -> T {
        self.size.width.clone() + self.size.height.clone()
    }

    pub fn dilate(&self, amount: T) -> Self {
        let double_amount = amount.clone() + amount.clone();
        Self::new(
            self.origin.clone() - point(amount.clone(), amount),
            self.size.clone() + size(double_amount.clone(), double_amount),
        )
    }
}

impl<T: Clone + Add<Output = T> + Sub<Output = T> + Neg<Output = T>> Bounds<T> {
    pub fn inset(&self, amount: T) -> Self {
        self.dilate(-amount)
    }
}

impl<T: Clone + PartialOrd + Add<Output = T> + Sub<Output = T>> Bounds<T> {
    pub fn intersect(&self, other: &Self) -> Self {
        let top_left = point(
            if self.origin.x >= other.origin.x {
                self.origin.x.clone()
            } else {
                other.origin.x.clone()
            },
            if self.origin.y >= other.origin.y {
                self.origin.y.clone()
            } else {
                other.origin.y.clone()
            },
        );
        let self_bottom_right = self.bottom_right();
        let other_bottom_right = other.bottom_right();
        let bottom_right = point(
            if self_bottom_right.x <= other_bottom_right.x {
                self_bottom_right.x
            } else {
                other_bottom_right.x
            },
            if self_bottom_right.y <= other_bottom_right.y {
                self_bottom_right.y
            } else {
                other_bottom_right.y
            },
        );
        Self::from_corners(top_left, bottom_right)
    }

    pub fn union(&self, other: &Self) -> Self {
        let top_left = point(
            if self.origin.x <= other.origin.x {
                self.origin.x.clone()
            } else {
                other.origin.x.clone()
            },
            if self.origin.y <= other.origin.y {
                self.origin.y.clone()
            } else {
                other.origin.y.clone()
            },
        );
        let self_bottom_right = self.bottom_right();
        let other_bottom_right = other.bottom_right();
        let bottom_right = point(
            if self_bottom_right.x >= other_bottom_right.x {
                self_bottom_right.x
            } else {
                other_bottom_right.x
            },
            if self_bottom_right.y >= other_bottom_right.y {
                self_bottom_right.y
            } else {
                other_bottom_right.y
            },
        );
        Self::from_corners(top_left, bottom_right)
    }
}

impl<T: Clone + Add<Output = T> + Sub<Output = T>> Add<Point<T>> for Bounds<T> {
    type Output = Self;

    fn add(self, offset: Point<T>) -> Self::Output {
        Self::new(self.origin + offset, self.size)
    }
}

impl<T: Clone + Add<Output = T> + Sub<Output = T>> Sub<Point<T>> for Bounds<T> {
    type Output = Self;

    fn sub(self, offset: Point<T>) -> Self::Output {
        Self::new(self.origin - offset, self.size)
    }
}

impl Bounds<Pixels> {
    pub fn centered<C, D>(_display: Option<D>, size: Size<Pixels>, _cx: &C) -> Self {
        Self::new(point(Pixels::ZERO, Pixels::ZERO), size)
    }

    pub fn maximized<C, D>(_display: Option<D>, _cx: &C) -> Self {
        Self::new(
            point(Pixels::ZERO, Pixels::ZERO),
            size(px(1024.0), px(768.0)),
        )
    }
}

/// The initial placement mode for a native window.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum WindowBounds {
    Windowed(Bounds<Pixels>),
    Maximized(Bounds<Pixels>),
    Fullscreen(Bounds<Pixels>),
}

impl Default for WindowBounds {
    fn default() -> Self {
        Self::Windowed(Bounds::default())
    }
}

impl WindowBounds {
    pub fn get_bounds(&self) -> Bounds<Pixels> {
        match self {
            Self::Windowed(bounds) | Self::Maximized(bounds) | Self::Fullscreen(bounds) => *bounds,
        }
    }

    pub fn centered<C>(size: Size<Pixels>, cx: &C) -> Self {
        Self::Windowed(Bounds::centered(None::<()>, size, cx))
    }
}

/// A font-relative length in rem units.
#[derive(Copy, Clone, Debug, Default, PartialEq, PartialOrd, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Rems(pub f32);

impl Rems {
    pub fn to_pixels(self, rem_size: Pixels) -> Pixels {
        Pixels(self.0 * rem_size.0)
    }
}

impl Mul<Pixels> for Rems {
    type Output = Pixels;

    fn mul(self, rem_size: Pixels) -> Self::Output {
        self.to_pixels(rem_size)
    }
}

impl Display for Rems {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}rem", self.0)
    }
}

impl TryFrom<&str> for Rems {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        value
            .strip_suffix("rem")
            .ok_or_else(|| format!("expected a rem length such as 1rem, got {value:?}"))?
            .parse::<f32>()
            .map(Self)
            .map_err(|error| format!("invalid rem length {value:?}: {error}"))
    }
}

pub const fn rems(value: f32) -> Rems {
    Rems(value)
}

/// An absolute pixel or rem length.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum AbsoluteLength {
    Pixels(Pixels),
    Rems(Rems),
}

impl Neg for AbsoluteLength {
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            Self::Pixels(value) => Self::Pixels(-value),
            Self::Rems(value) => Self::Rems(Rems(-value.0)),
        }
    }
}

impl Default for AbsoluteLength {
    fn default() -> Self {
        Self::Pixels(Pixels::ZERO)
    }
}

impl AbsoluteLength {
    pub fn is_zero(self) -> bool {
        match self {
            Self::Pixels(value) => value.0 == 0.0,
            Self::Rems(value) => value.0 == 0.0,
        }
    }

    pub fn to_pixels(self, rem_size: Pixels) -> Pixels {
        match self {
            Self::Pixels(value) => value,
            Self::Rems(value) => value.to_pixels(rem_size),
        }
    }
}

impl From<Pixels> for AbsoluteLength {
    fn from(value: Pixels) -> Self {
        Self::Pixels(value)
    }
}

impl From<Rems> for AbsoluteLength {
    fn from(value: Rems) -> Self {
        Self::Rems(value)
    }
}

impl Display for AbsoluteLength {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pixels(value) => Display::fmt(value, formatter),
            Self::Rems(value) => Display::fmt(value, formatter),
        }
    }
}

impl TryFrom<&str> for AbsoluteLength {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        if let Ok(pixels) = Pixels::try_from(value) {
            Ok(Self::Pixels(pixels))
        } else if let Ok(rems) = Rems::try_from(value) {
            Ok(Self::Rems(rems))
        } else {
            Err(format!(
                "expected an absolute length ending in px or rem, got {value:?}"
            ))
        }
    }
}

/// A length resolved against a parent dimension or rem size.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum DefiniteLength {
    Absolute(AbsoluteLength),
    Fraction(f32),
}

impl Neg for DefiniteLength {
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            Self::Absolute(value) => Self::Absolute(-value),
            Self::Fraction(value) => Self::Fraction(-value),
        }
    }
}

impl Default for DefiniteLength {
    fn default() -> Self {
        Self::Absolute(AbsoluteLength::default())
    }
}

impl DefiniteLength {
    pub fn to_pixels(self, base_size: AbsoluteLength, rem_size: Pixels) -> Pixels {
        match self {
            Self::Absolute(value) => value.to_pixels(rem_size),
            Self::Fraction(fraction) => base_size.to_pixels(rem_size) * fraction,
        }
    }
}

impl From<Pixels> for DefiniteLength {
    fn from(value: Pixels) -> Self {
        Self::Absolute(value.into())
    }
}

impl From<Rems> for DefiniteLength {
    fn from(value: Rems) -> Self {
        Self::Absolute(value.into())
    }
}

impl From<AbsoluteLength> for DefiniteLength {
    fn from(value: AbsoluteLength) -> Self {
        Self::Absolute(value)
    }
}

impl TryFrom<&str> for DefiniteLength {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        if let Some(fraction) = value.strip_suffix('%') {
            return fraction
                .parse::<f32>()
                .map(|value| Self::Fraction(value / 100.0))
                .map_err(|error| format!("invalid relative length {value:?}: {error}"));
        }
        AbsoluteLength::try_from(value).map(Self::Absolute)
    }
}

impl Display for DefiniteLength {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Absolute(AbsoluteLength::Pixels(value)) => write!(formatter, "{value}"),
            Self::Absolute(AbsoluteLength::Rems(value)) => write!(formatter, "{value}"),
            Self::Fraction(value) => write!(formatter, "{}%", value * 100.0),
        }
    }
}

/// A definite or automatic layout length.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Length {
    Definite(DefiniteLength),
    Auto,
}

impl Neg for Length {
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            Self::Definite(value) => Self::Definite(-value),
            Self::Auto => Self::Auto,
        }
    }
}

impl Default for Length {
    fn default() -> Self {
        Self::Definite(DefiniteLength::default())
    }
}

impl From<DefiniteLength> for Length {
    fn from(value: DefiniteLength) -> Self {
        Self::Definite(value)
    }
}

impl From<Pixels> for Length {
    fn from(value: Pixels) -> Self {
        Self::Definite(value.into())
    }
}

impl From<Rems> for Length {
    fn from(value: Rems) -> Self {
        Self::Definite(value.into())
    }
}

impl From<AbsoluteLength> for Length {
    fn from(value: AbsoluteLength) -> Self {
        Self::Definite(value.into())
    }
}

impl TryFrom<&str> for Length {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        if value == "auto" {
            Ok(Self::Auto)
        } else {
            DefiniteLength::try_from(value).map(Self::Definite)
        }
    }
}

impl Size<Length> {
    pub fn full() -> Self {
        size(relative(1.0).into(), relative(1.0).into())
    }

    pub fn auto() -> Self {
        size(Length::Auto, Length::Auto)
    }
}

/// A relative half operation used by geometry centering helpers.
pub trait Half {
    fn half(self) -> Self;
}

impl Half for Pixels {
    fn half(self) -> Self {
        Self(self.0 / 2.0)
    }
}

pub const fn relative(fraction: f32) -> DefiniteLength {
    DefiniteLength::Fraction(fraction)
}

pub const fn phi() -> DefiniteLength {
    relative(1.618_034)
}

pub const fn px(value: f32) -> Pixels {
    Pixels(value)
}

impl<'de> Deserialize<'de> for AbsoluteLength {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = String::deserialize(deserializer)?;
        if let Ok(pixels) = Pixels::try_from(value.as_str()) {
            Ok(Self::Pixels(pixels))
        } else if let Ok(rems) = Rems::try_from(value.as_str()) {
            Ok(Self::Rems(rems))
        } else {
            Err(de::Error::custom("expected a length ending in px or rem"))
        }
    }
}

impl<'de> Deserialize<'de> for DefiniteLength {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = String::deserialize(deserializer)?;
        if let Some(fraction) = value.strip_suffix('%') {
            return fraction
                .parse::<f32>()
                .map(|value| Self::Fraction(value / 100.0))
                .map_err(de::Error::custom);
        }
        let absolute = if let Ok(pixels) = Pixels::try_from(value.as_str()) {
            AbsoluteLength::Pixels(pixels)
        } else if let Ok(rems) = Rems::try_from(value.as_str()) {
            AbsoluteLength::Rems(rems)
        } else {
            return Err(de::Error::custom("expected a length ending in px or rem"));
        };
        Ok(Self::Absolute(absolute))
    }
}

impl<'de> Deserialize<'de> for Length {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = String::deserialize(deserializer)?;
        if value == "auto" {
            Ok(Self::Auto)
        } else {
            let definite = if let Some(fraction) = value.strip_suffix('%') {
                let fraction = fraction.parse::<f32>().map_err(de::Error::custom)?;
                DefiniteLength::Fraction(fraction / 100.0)
            } else if let Ok(pixels) = Pixels::try_from(value.as_str()) {
                DefiniteLength::Absolute(pixels.into())
            } else if let Ok(rems) = Rems::try_from(value.as_str()) {
                DefiniteLength::Absolute(rems.into())
            } else {
                return Err(de::Error::custom(
                    "expected a length ending in px, rem, or %",
                ));
            };
            Ok(Self::Definite(definite))
        }
    }
}

impl Serialize for AbsoluteLength {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&match self {
            Self::Pixels(value) => format!("{}px", value.0),
            Self::Rems(value) => format!("{}rem", value.0),
        })
    }
}

impl Serialize for DefiniteLength {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl Serialize for Length {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Definite(value) => value.serialize(serializer),
            Self::Auto => serializer.serialize_str("auto"),
        }
    }
}

#[cfg(test)]
mod native_tests {
    use super::*;

    #[test]
    fn pixels_cover_arithmetic_scalar_conversion_and_edge_parsing() {
        assert_eq!(px(10.0) + px(2.5), px(12.5));
        assert_eq!(px(10.0) - px(2.5), px(7.5));
        assert_eq!(px(4.0) * 2.0, px(8.0));
        assert_eq!(2.0 * px(4.0), px(8.0));
        assert_eq!(px(10.0) / px(4.0), 2.5);
        assert_eq!(px(-1.5).abs(), px(1.5));
        assert_eq!(px(1.6).floor(), px(1.0));
        assert_eq!(px(1.4).ceil(), px(2.0));
        assert_eq!(Pixels::try_from("12.5px").expect("valid pixels"), px(12.5));
        assert!(Pixels::try_from("12").is_err());
        assert!(Pixels::try_from("12rem").is_err());
    }

    #[test]
    fn points_sizes_and_bounds_preserve_geometry_operations() {
        let original = point(px(10.0), px(20.0));
        let offset = point(px(2.0), px(-4.0));
        assert_eq!(original + offset, point(px(12.0), px(16.0)));
        assert_eq!(original.relative_to(&offset), point(px(8.0), px(24.0)));
        assert_eq!(point(px(3.0), px(4.0)).magnitude(), 5.0);

        let bounds = Bounds::new(point(px(10.0), px(20.0)), size(px(30.0), px(40.0)));
        assert_eq!(bounds.center(), point(px(25.0), px(40.0)));
        assert_eq!(bounds.bottom_right(), point(px(40.0), px(60.0)));
        assert_eq!(bounds.dilate(px(5.0)).origin, point(px(5.0), px(15.0)));
        assert_eq!(bounds.dilate(px(5.0)).size, size(px(40.0), px(50.0)));
        let overlapping = Bounds::new(point(px(39.0), px(59.0)), size(px(2.0), px(2.0)));
        assert!(bounds.intersects(&overlapping));
        let touching = Bounds::new(point(px(40.0), px(60.0)), size(px(2.0), px(2.0)));
        assert!(!bounds.intersects(&touching));
        let overlap = bounds.intersect(&Bounds::new(
            point(px(20.0), px(30.0)),
            size(px(30.0), px(40.0)),
        ));
        assert_eq!(overlap.size, size(px(20.0), px(30.0)));
    }

    #[test]
    fn lengths_resolve_and_round_trip_as_css_strings() {
        assert_eq!(rems(1.5).to_pixels(px(16.0)), px(24.0));
        assert_eq!(
            relative(0.25).to_pixels(AbsoluteLength::Pixels(px(200.0)), px(16.0)),
            px(50.0)
        );
        assert_eq!(
            DefiniteLength::try_from("25%").expect("valid percentage"),
            relative(0.25)
        );
        assert_eq!(
            AbsoluteLength::try_from("2rem").expect("valid rem"),
            rems(2.0).into()
        );
        assert_eq!(Length::try_from("auto").expect("valid auto"), Length::Auto);
        assert_eq!(
            serde_json::to_string(&relative(0.25)).expect("serialize length"),
            "\"25%\""
        );
        assert_eq!(
            serde_json::to_string(&Length::Auto).expect("serialize auto"),
            "\"auto\""
        );
        let decoded: DefiniteLength = serde_json::from_str("\"2rem\"").expect("deserialize length");
        assert_eq!(
            decoded.to_pixels(AbsoluteLength::Pixels(px(100.0)), px(16.0)),
            px(32.0)
        );
        assert!(serde_json::from_str::<Length>("\"12\"").is_err());
    }

    #[test]
    fn window_bounds_retain_the_requested_mode_and_extent() {
        let extent = size(px(640.0), px(480.0));
        let bounds = Bounds::centered(None::<()>, extent, &());
        assert_eq!(WindowBounds::centered(extent, &()).get_bounds(), bounds);
        assert_eq!(WindowBounds::Maximized(bounds).get_bounds(), bounds);
    }
}
