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

/// Every invalidation raised since the last frame was drawn.
///
/// This is the queue side of §5.4's vocabulary: `Reason` says what a single
/// request meant, and this says what the *frame* means for a given layer, which
/// is the question `.boundary()` actually asks (§4.1). Kept here rather than in
/// `app/effects.rs` — §3.1's home for "deferred notifications,
/// flush_deferred_invalidations" — because that module needs an `App` to
/// deliver into and is still a Phase 0 stub; whoever builds it owns draining
/// into one of these rather than reimplementing it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FrameSignals {
    requests: Vec<InvalidationRequest>,
}

impl FrameSignals {
    /// A frame with nothing raised against it.
    pub const fn new() -> Self {
        Self {
            requests: Vec::new(),
        }
    }

    /// Record an invalidation.
    pub fn raise(&mut self, request: InvalidationRequest) -> &mut Self {
        self.requests.push(request);
        self
    }

    /// Record a scroll tick against a layer — SFD §1.1's `notify_scroll`, with
    /// the layer standing in for the view it names, since `wgpui-core` has no
    /// `EntityId` yet (see [`InvalidationScope`]'s own doc).
    pub fn scrolled(&mut self, layer: LayerId) -> &mut Self {
        self.raise(InvalidationRequest::scrolled(InvalidationScope::Layer(
            layer,
        )))
    }

    /// Record an ordinary data-change notification against a layer.
    pub fn data_changed(&mut self, layer: LayerId) -> &mut Self {
        self.raise(InvalidationRequest::data_changed(InvalidationScope::Layer(
            layer,
        )))
    }

    /// What this frame means for `layer`.
    ///
    /// [`Reason::Scroll`] only when at least one request applies to that layer
    /// and *every* applicable one permits transform-only. Both halves are
    /// deliberate: a frame carrying no signal at all has not claimed to be a
    /// scroll, and a frame carrying a scroll tick alongside a data change is a
    /// data change — §5.4 makes the scroll signal distinguishable so it can be
    /// trusted, which only works if a single unqualified request is enough to
    /// withdraw the claim.
    pub fn reason_for_layer(&self, layer: LayerId) -> Reason {
        let mut applicable = false;
        for request in &self.requests {
            let applies = match request.scope() {
                InvalidationScope::Layer(scoped) => scoped == layer,
                InvalidationScope::Window => true,
                InvalidationScope::Instance(_) => false,
            };
            if !applies {
                continue;
            }
            applicable = true;
            if !request.reason().permits_transform_only() {
                return Reason::DataChanged;
            }
        }
        if applicable {
            Reason::Scroll
        } else {
            Reason::DataChanged
        }
    }

    /// Every request raised, in the order they arrived.
    pub fn requests(&self) -> &[InvalidationRequest] {
        &self.requests
    }

    /// Whether nothing was raised.
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    /// How many requests were raised.
    pub fn len(&self) -> usize {
        self.requests.len()
    }

    /// Drop every request, keeping the allocation for the next frame.
    pub fn clear(&mut self) {
        self.requests.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PANEL: LayerId = LayerId::from_raw(11);
    const SIDEBAR: LayerId = LayerId::from_raw(13);

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

    #[test]
    fn a_frame_with_no_signal_has_not_claimed_to_be_a_scroll() {
        let signals = FrameSignals::new();
        assert!(signals.is_empty());
        assert_eq!(signals.reason_for_layer(PANEL), Reason::DataChanged);
    }

    #[test]
    fn a_scroll_tick_is_recognised_for_the_layer_it_names_and_no_other() {
        let mut signals = FrameSignals::new();
        signals.scrolled(PANEL);
        assert_eq!(signals.reason_for_layer(PANEL), Reason::Scroll);
        assert_eq!(
            signals.reason_for_layer(SIDEBAR),
            Reason::DataChanged,
            "an unrelated layer must not inherit another's scroll"
        );
    }

    #[test]
    fn one_data_change_withdraws_a_scroll_claim_for_the_same_layer() {
        let mut signals = FrameSignals::new();
        signals.scrolled(PANEL).data_changed(PANEL);
        assert_eq!(signals.reason_for_layer(PANEL), Reason::DataChanged);
        assert_eq!(signals.len(), 2);
    }

    #[test]
    fn a_window_scoped_data_change_withdraws_every_layers_scroll_claim() {
        let mut signals = FrameSignals::new();
        signals
            .scrolled(PANEL)
            .raise(InvalidationRequest::data_changed(InvalidationScope::Window));
        assert_eq!(signals.reason_for_layer(PANEL), Reason::DataChanged);
    }

    #[test]
    fn an_instance_scoped_request_does_not_speak_for_a_layer() {
        let mut signals = FrameSignals::new();
        signals
            .scrolled(PANEL)
            .raise(InvalidationRequest::data_changed(
                InvalidationScope::Instance(InstanceKey::from_raw(3)),
            ));
        assert_eq!(signals.reason_for_layer(PANEL), Reason::Scroll);
    }

    #[test]
    fn clearing_a_frame_keeps_nothing_from_it() {
        let mut signals = FrameSignals::new();
        signals.scrolled(PANEL);
        signals.clear();
        assert!(signals.is_empty());
        assert!(signals.requests().is_empty());
        assert_eq!(signals.reason_for_layer(PANEL), Reason::DataChanged);
    }
}
