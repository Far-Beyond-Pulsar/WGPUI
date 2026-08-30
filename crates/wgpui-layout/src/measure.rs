//! Intrinsic sizing values shared by native elements.

/// A content measurement returned by a leaf or an inexpensive estimate.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct IntrinsicSize {
    pub width: f32,
    pub height: f32,
}

impl IntrinsicSize {
    pub const ZERO: Self = Self {
        width: 0.0,
        height: 0.0,
    };

    pub const fn new(width: f32, height: f32) -> Self {
        Self { width, height }
    }

    pub fn clamped(self) -> Self {
        Self {
            width: self.width.max(0.0),
            height: self.height.max(0.0),
        }
    }
}

/// A measurement callback is intentionally pure: it can run during Taffy's
/// CPU layout pass without touching the scene or GPU.
pub trait Measure {
    fn measure(&self, available: LayoutSize) -> IntrinsicSize;
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LayoutSize {
    pub width: Option<f32>,
    pub height: Option<f32>,
}

impl LayoutSize {
    pub const UNBOUNDED: Self = Self {
        width: None,
        height: None,
    };
}
