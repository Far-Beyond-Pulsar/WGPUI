//! `LAYOUT`/`DISPLAY`/`HIT`/`TRANSFORM` — R-N §3.2's four axes, finally
//! wired live. See docs/gpu-native-architecture.md §5.4.
//!
//! The axes are *derived by the framework* from what a reconcile comparison
//! found, never declared at the site that raised the invalidation: correctness
//! would then depend on every call site classifying its own change correctly,
//! and the failure mode of getting that wrong is silently stale UI (R-N §2.4).

/// Which respects of an element or layer stopped being valid.
///
/// Hand-rolled as a `u8` newtype rather than pulled from `bitflags`, matching
/// the legacy backend's own choice (`src/window.rs`): `wgpui-core` carries no
/// dependency this small a type would justify adding.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Invalidation(u8);

impl Invalidation {
    /// Sizes and positions must be recomputed.
    pub const LAYOUT: Self = Self(1 << 0);
    /// Painted output must be re-emitted.
    pub const DISPLAY: Self = Self(1 << 1);
    /// Hitboxes and dispatch nodes must be re-registered.
    pub const HIT: Self = Self(1 << 2);
    /// Only the composite transform changed, so nothing needs re-rendering.
    ///
    /// Unlike the legacy backend — where this bit exists and nothing ever sets
    /// it (§1's table) — 2.0 raises it for real once `.boundary()` makes
    /// independent compositing true (§5.4, Phase 2). Phase 1 defines it so the
    /// vocabulary is complete from the start rather than retrofitted.
    pub const TRANSFORM: Self = Self(1 << 3);

    /// No axis at all: the element is fully reusable.
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Every axis: what a type mismatch or a missing `diff_key` reports.
    pub const fn all() -> Self {
        Self(Self::LAYOUT.0 | Self::DISPLAY.0 | Self::HIT.0 | Self::TRANSFORM.0)
    }

    /// Whether no axis is set.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Whether every axis in `other` is set.
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Whether any axis in `other` is set.
    pub const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    /// The axes set in either operand. `BitOr` usable in a `const` context.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// The raw bit pattern, for debug rendering and test assertions.
    pub const fn bits(self) -> u8 {
        self.0
    }
}

impl std::ops::BitOr for Invalidation {
    type Output = Self;

    fn bitor(self, other: Self) -> Self {
        self.union(other)
    }
}

impl std::ops::BitOrAssign for Invalidation {
    fn bitor_assign(&mut self, other: Self) {
        self.0 |= other.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_contains_nothing_and_all_contains_everything() {
        assert!(Invalidation::empty().is_empty());
        assert!(!Invalidation::all().is_empty());
        for axis in [
            Invalidation::LAYOUT,
            Invalidation::DISPLAY,
            Invalidation::HIT,
            Invalidation::TRANSFORM,
        ] {
            assert!(Invalidation::all().contains(axis));
            assert!(!Invalidation::empty().intersects(axis));
        }
    }

    #[test]
    fn union_accumulates_without_losing_earlier_axes() {
        let mut axes = Invalidation::DISPLAY;
        axes |= Invalidation::HIT;
        assert!(axes.contains(Invalidation::DISPLAY));
        assert!(axes.contains(Invalidation::HIT));
        assert!(!axes.contains(Invalidation::LAYOUT));
    }

    #[test]
    fn transform_is_distinct_from_the_other_three() {
        assert!(!Invalidation::TRANSFORM.intersects(
            Invalidation::LAYOUT
                .union(Invalidation::DISPLAY)
                .union(Invalidation::HIT)
        ));
    }
}
