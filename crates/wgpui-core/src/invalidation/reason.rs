//! `Reason::Scroll` vs `Reason::DataChanged` — the fifth *signal* kind,
//! distinct from the four invalidation *axes*. See
//! docs/gpu-native-architecture.md §5.4 and §4.1.
//!
//! SFD's own retrospective ("What changed on implementation") records why this
//! has to exist at the point invalidation is *raised* rather than be inferred
//! afterwards: at the moment a boundary asks "can this notification resolve to
//! a transform-only recomposite," a scroll tick and a data change are
//! indistinguishable if both arrived as a generic notify. Phase 1 defines the
//! vocabulary so Phase 2's `.boundary()` consumes it from day one instead of
//! retrofitting it across every call site at once.

/// Why an [`crate::invalidation::request::InvalidationRequest`] was raised.
///
/// Consumed by `.boundary()` (§4.1, Phase 2) to decide whether a notification
/// can resolve to [`crate::invalidation::axes::Invalidation::TRANSFORM`]
/// alone. Every other consumer ignores it.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum Reason {
    /// The data an element renders from changed. The conservative default:
    /// a consumer that cannot tell what happened must assume this.
    #[default]
    DataChanged,
    /// The viewport moved over unchanged content — a scroll tick or a pan.
    /// Nothing an observer could see about the content itself changed.
    Scroll,
}

impl Reason {
    /// Whether this signal is, on its own, compatible with resolving to a
    /// transform-only recomposite.
    ///
    /// Compatible is not sufficient: a boundary still has to establish that
    /// its content is clean. This answers only "does the signal itself rule
    /// transform-only out," which [`Reason::DataChanged`] does and
    /// [`Reason::Scroll`] does not.
    pub const fn permits_transform_only(self) -> bool {
        matches!(self, Reason::Scroll)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_changed_is_the_conservative_default() {
        assert_eq!(Reason::default(), Reason::DataChanged);
        assert!(!Reason::default().permits_transform_only());
    }

    #[test]
    fn scroll_permits_transform_only() {
        assert!(Reason::Scroll.permits_transform_only());
    }
}
