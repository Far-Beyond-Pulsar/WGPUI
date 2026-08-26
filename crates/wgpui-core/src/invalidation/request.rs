//! `InvalidationRequest`/`InvalidationScope` — R-N §6's typed invalidation,
//! kept unchanged in shape. See docs/gpu-native-architecture.md §5.4.
//!
//! One operation every part of the framework invalidates through, replacing
//! three mechanisms with incompatible reach (a window-wide `refreshing`
//! boolean, an upward dispatch-tree walk, and a forward dependency-set check)
//! that the legacy backend still carries side by side.

use crate::invalidation::axes::Invalidation;
use crate::invalidation::reason::Reason;
use crate::reconcile::instance::InstanceKey;
use crate::scene::layer::LayerId;

/// What an [`InvalidationRequest`] applies to.
///
/// R-N §6's fourth variant, `Entity(EntityId)`, is deliberately absent in
/// Phase 1: entity identity belongs to `wgpui-core::app` (§3.1), which is
/// still a Phase 0 stub. Adding a placeholder `EntityId` here purely to fill
/// the variant would put a type in the public surface that nothing can
/// produce, so the variant lands with the module that owns its identity.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum InvalidationScope {
    /// One retained element instance, and nothing above or below it.
    Instance(InstanceKey),
    /// One retained layer, and nothing above or below it.
    Layer(LayerId),
    /// The window as a whole: device loss, scale factor change, focus moving.
    Window,
}

/// One typed invalidation: what stopped being valid, in what respect, and why.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct InvalidationRequest {
    scope: InvalidationScope,
    axes: Invalidation,
    reason: Reason,
}

impl InvalidationRequest {
    /// Raise an invalidation with an explicit reason (§5.4).
    pub const fn new(scope: InvalidationScope, axes: Invalidation, reason: Reason) -> Self {
        Self {
            scope,
            axes,
            reason,
        }
    }

    /// What a plain data-change notification means: painted output and hit
    /// geometry are stale for the named scope.
    pub const fn data_changed(scope: InvalidationScope) -> Self {
        Self::new(
            scope,
            Invalidation::DISPLAY.union(Invalidation::HIT),
            Reason::DataChanged,
        )
    }

    /// What a scroll tick means: the viewport moved over content that is not
    /// itself claimed to have changed. A `.boundary()` (Phase 2) is what turns
    /// this into a transform-only recomposite; until then it is recorded
    /// faithfully and treated conservatively by every consumer.
    pub const fn scrolled(scope: InvalidationScope) -> Self {
        Self::new(scope, Invalidation::TRANSFORM, Reason::Scroll)
    }

    /// What this request applies to.
    pub const fn scope(self) -> InvalidationScope {
        self.scope
    }

    /// Which respects stopped being valid.
    pub const fn axes(self) -> Invalidation {
        self.axes
    }

    /// Why the request was raised.
    pub const fn reason(self) -> Reason {
        self.reason
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_change_invalidates_display_and_hit_but_not_layout() {
        let request = InvalidationRequest::data_changed(InvalidationScope::Window);
        assert!(request.axes().contains(Invalidation::DISPLAY));
        assert!(request.axes().contains(Invalidation::HIT));
        assert!(!request.axes().contains(Invalidation::LAYOUT));
        assert_eq!(request.reason(), Reason::DataChanged);
    }

    #[test]
    fn scroll_carries_its_reason_so_a_boundary_can_recognise_it() {
        let request = InvalidationRequest::scrolled(InvalidationScope::Layer(LayerId::from_raw(7)));
        assert!(request.reason().permits_transform_only());
        assert_eq!(request.axes(), Invalidation::TRANSFORM);
    }
}
