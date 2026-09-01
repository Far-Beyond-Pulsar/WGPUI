//! `InvalidationRequest`/`InvalidationScope` — R-N §6's typed invalidation.
//! See docs/gpu-native-architecture.md §5.4.
//!
//! One operation every part of the framework invalidates through, replacing
//! three mechanisms with incompatible reach (a window-wide `refreshing`
//! boolean, an upward dispatch-tree walk, and a forward dependency-set check)
//! that the legacy backend still carries side by side.

use crate::app::EntityId;
use crate::invalidation::axes::Invalidation;
use crate::invalidation::reason::Reason;
use crate::reconcile::instance::InstanceKey;
use crate::scene::layer::LayerId;

/// What an [`InvalidationRequest`] applies to.
///
/// Entity identity is owned by [`crate::app`], while the request remains in
/// this backend-neutral module so an application can pass the signal to
/// retained reconciliation without involving a renderer.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum InvalidationScope {
    /// One retained element instance, and nothing above or below it.
    Instance(InstanceKey),
    /// One retained layer, and nothing above or below it.
    Layer(LayerId),
    /// One application entity. The runtime maps this identity to the
    /// retained element or elements that consume the entity's data.
    Entity(EntityId),
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

    /// What an entity data-change notification means.
    pub const fn entity_changed(entity: EntityId) -> Self {
        Self::data_changed(InvalidationScope::Entity(entity))
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

    /// Whether this request applies to `scope`.
    ///
    /// A window request is intentionally the only broad request. Entity,
    /// layer, and instance requests stay exact so a targeted notification
    /// cannot dirty an unrelated retained scope.
    pub fn applies_to(self, scope: InvalidationScope) -> bool {
        match (self.scope, scope) {
            (InvalidationScope::Window, _) => true,
            (InvalidationScope::Instance(requested), InvalidationScope::Instance(target)) => {
                requested == target
            }
            (InvalidationScope::Layer(requested), InvalidationScope::Layer(target)) => {
                requested == target
            }
            (InvalidationScope::Entity(requested), InvalidationScope::Entity(target)) => {
                requested == target
            }
            _ => false,
        }
    }

    /// Whether this request names `entity` directly.
    pub fn applies_to_entity(self, entity: EntityId) -> bool {
        matches!(self.scope, InvalidationScope::Entity(requested) if requested == entity)
    }

    /// Whether this request names `layer` directly.
    pub fn applies_to_layer(self, layer: LayerId) -> bool {
        matches!(self.scope, InvalidationScope::Layer(requested) if requested == layer)
    }

    /// Whether this request names `instance` directly.
    pub fn applies_to_instance(self, instance: InstanceKey) -> bool {
        matches!(self.scope, InvalidationScope::Instance(requested) if requested == instance)
    }
}

/// Every distinct invalidation raised since the last frame was drawn.
///
/// This is the queue side of §5.4's vocabulary: `Reason` says what a single
/// request meant, and this says what the *frame* means for a given layer, which
/// is the question `.boundary()` actually asks (§4.1). Kept here rather than in
/// `app/effects.rs` — §3.1's home for "deferred notifications,
/// flush_deferred_invalidations" — because that module needs an `App` to
/// deliver into; whoever builds it owns draining into one of these rather than
/// reimplementing it.
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

    /// Record an invalidation, retaining only the first copy of an identical
    /// request.
    pub fn raise(&mut self, request: InvalidationRequest) -> &mut Self {
        if !self.requests.contains(&request) {
            self.requests.push(request);
        }
        self
    }

    /// Record a scroll tick against a layer — SFD §1.1's `notify_scroll`.
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

    /// Record an ordinary data-change notification against one entity.
    ///
    /// Entity data changes are deliberately display and hit invalidations,
    /// never a transform-only signal. Layout invalidation is derived by the
    /// retained reconciliation that consumes this signal.
    pub fn entity_changed(&mut self, entity: EntityId) -> &mut Self {
        self.raise(InvalidationRequest::entity_changed(entity))
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
        self.reason_for_scope(InvalidationScope::Layer(layer))
    }

    /// What this frame means for `entity`.
    pub fn reason_for_entity(&self, entity: EntityId) -> Reason {
        self.reason_for_scope(InvalidationScope::Entity(entity))
    }

    /// What this frame means for `instance`.
    pub fn reason_for_instance(&self, instance: InstanceKey) -> Reason {
        self.reason_for_scope(InvalidationScope::Instance(instance))
    }

    fn reason_for_scope(&self, scope: InvalidationScope) -> Reason {
        let mut applicable = false;
        for request in &self.requests {
            if !request.applies_to(scope) {
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

    /// The axes currently invalidated for `entity`, including a window-wide
    /// request when one exists.
    pub fn invalidation_for_entity(&self, entity: EntityId) -> Invalidation {
        let mut axes = Invalidation::empty();
        for request in &self.requests {
            if request.applies_to(InvalidationScope::Entity(entity)) {
                axes |= request.axes();
            }
        }
        axes
    }

    /// Return the direct entity signal that retained reconciliation can
    /// consume, without consuming a window-wide request.
    pub fn entity_signal(&self, entity: EntityId) -> Option<InvalidationRequest> {
        self.aggregate_entity_signal(entity)
    }

    /// Consume all direct requests for `entity` as one deterministic signal.
    ///
    /// The first request establishes the signal's position in the queue;
    /// later requests only add axes or withdraw a transform-only claim. Other
    /// entity, layer, instance, and window requests remain queued.
    pub fn consume_entity_signal(&mut self, entity: EntityId) -> Option<InvalidationRequest> {
        let signal = self.aggregate_entity_signal(entity);
        if signal.is_some() {
            self.requests
                .retain(|request| !request.applies_to_entity(entity));
        }
        signal
    }

    fn aggregate_entity_signal(&self, entity: EntityId) -> Option<InvalidationRequest> {
        let mut aggregate: Option<InvalidationRequest> = None;
        for request in &self.requests {
            if !request.applies_to_entity(entity) {
                continue;
            }
            aggregate = Some(match aggregate {
                Some(current) => InvalidationRequest::new(
                    InvalidationScope::Entity(entity),
                    current.axes().union(request.axes()),
                    if current.reason().permits_transform_only()
                        && request.reason().permits_transform_only()
                    {
                        Reason::Scroll
                    } else {
                        Reason::DataChanged
                    },
                ),
                None => InvalidationRequest::new(
                    InvalidationScope::Entity(entity),
                    request.axes(),
                    request.reason(),
                ),
            });
        }
        aggregate
    }

    /// Every distinct request retained, in first-seen order.
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
    use crate::app::App;

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
    fn entity_requests_preserve_scope_axes_and_reason_accessors() {
        let app = App::create();
        let entity = app.new_entity(()).entity_id();
        let request = InvalidationRequest::entity_changed(entity);

        assert_eq!(request.scope(), InvalidationScope::Entity(entity));
        assert!(request.axes().contains(Invalidation::DISPLAY));
        assert!(request.axes().contains(Invalidation::HIT));
        assert!(!request.axes().contains(Invalidation::TRANSFORM));
        assert_eq!(request.reason(), Reason::DataChanged);
    }

    #[test]
    fn entity_requests_match_only_their_entity_not_layers_or_instances() {
        let app = App::create();
        let entity = app.new_entity(()).entity_id();
        let other_entity = app.new_entity(()).entity_id();
        let request = InvalidationRequest::entity_changed(entity);
        let instance = InstanceKey::from_raw(3);

        assert!(request.applies_to_entity(entity));
        assert!(!request.applies_to_entity(other_entity));
        assert!(!request.applies_to_layer(PANEL));
        assert!(!request.applies_to_instance(instance));
        assert!(!request.applies_to(InvalidationScope::Layer(PANEL)));
        assert!(!request.applies_to(InvalidationScope::Instance(instance)));

        let mut signals = FrameSignals::new();
        signals.scrolled(PANEL).entity_changed(entity);
        assert_eq!(signals.reason_for_entity(entity), Reason::DataChanged);
        assert_eq!(signals.reason_for_layer(PANEL), Reason::Scroll);
        assert_eq!(signals.reason_for_layer(SIDEBAR), Reason::DataChanged);
        assert_eq!(signals.reason_for_instance(instance), Reason::DataChanged);
    }

    #[test]
    fn identical_requests_coalesce_without_reordering_distinct_requests() {
        let app = App::create();
        let entity = app.new_entity(()).entity_id();
        let entity_request = InvalidationRequest::entity_changed(entity);
        let layer_request = InvalidationRequest::data_changed(InvalidationScope::Layer(PANEL));
        let mut signals = FrameSignals::new();

        signals
            .raise(entity_request)
            .raise(entity_request)
            .raise(layer_request)
            .raise(layer_request);

        assert_eq!(signals.requests(), &[entity_request, layer_request]);
    }

    #[test]
    fn consuming_an_entity_signal_leaves_unrelated_scopes_queued() {
        let app = App::create();
        let entity = app.new_entity(()).entity_id();
        let mut signals = FrameSignals::new();
        signals.entity_changed(entity).data_changed(PANEL);

        let consumed = signals.consume_entity_signal(entity);
        assert_eq!(
            consumed.map(|request| request.scope()),
            Some(InvalidationScope::Entity(entity))
        );
        assert!(signals.entity_signal(entity).is_none());
        assert_eq!(
            signals.requests(),
            &[InvalidationRequest::data_changed(InvalidationScope::Layer(
                PANEL
            ),)]
        );
    }

    #[test]
    fn entity_data_withdraws_an_entity_scroll_fast_path_claim() {
        let app = App::create();
        let entity = app.new_entity(()).entity_id();
        let mut signals = FrameSignals::new();
        signals
            .raise(InvalidationRequest::scrolled(InvalidationScope::Entity(
                entity,
            )))
            .entity_changed(entity);

        assert_eq!(signals.reason_for_entity(entity), Reason::DataChanged);
        let signal = signals.entity_signal(entity);
        assert_eq!(
            signal.map(|request| request.reason()),
            Some(Reason::DataChanged)
        );
        assert!(
            signals
                .invalidation_for_entity(entity)
                .contains(Invalidation::DISPLAY)
        );
        assert!(
            signals
                .invalidation_for_entity(entity)
                .contains(Invalidation::HIT)
        );
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
