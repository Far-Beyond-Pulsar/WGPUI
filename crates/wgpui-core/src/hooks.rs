//! Backend-neutral instrumentation hooks.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;

/// Version of the serialized trace event contract.
pub const TRACE_EVENT_VERSION: u16 = 1;
/// Descriptive alias for [`TRACE_EVENT_VERSION`].
pub const TRACE_EVENT_SCHEMA_VERSION: u16 = TRACE_EVENT_VERSION;

/// The versioned identifier types used by [`TraceEvent`].
pub type TraceFrameId = u64;
pub type TraceSpanId = u64;
pub type TraceThreadId = u64;
pub type TraceQueueId = u64;
pub type TraceElementAddress = u64;
pub type TraceBoundaryId = u64;
pub type TraceRootId = u64;
/// Monotonic nanoseconds from the clock selected by the trace producer.
pub type TraceTimestamp = u64;

/// A tile coordinate in a trace event's owning tile grid.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct TraceTileCoord {
    pub x: i32,
    pub y: i32,
}

/// Alias emphasizing that this coordinate is part of the serialized contract.
pub type TraceTileCoordinate = TraceTileCoord;

impl TraceTileCoord {
    /// Creates a tile coordinate.
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

/// The operation represented by a trace event.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum TraceEventKind {
    /// Starts a span identified by [`TraceEvent::span_id`].
    SpanBegin { name: String },
    /// Ends a span identified by [`TraceEvent::span_id`].
    SpanEnd,
    /// Adds a value to a named counter.
    Counter { name: String, amount: u64 },
    /// Marks the successful presentation of a frame.
    FramePresented,
    /// Carries a backend timestamp pair. The pair is in the event's start and
    /// end timestamp fields because GPU timestamp units are backend-defined.
    GpuTimestamp,
    /// Reports events that could not be retained by a bounded consumer.
    Dropped { count: u64, reason: TraceDropReason },
}

/// Why a trace consumer could not retain an event.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceDropReason {
    /// The event budget was full.
    EventLimit,
    /// The serialized byte budget was full.
    ByteLimit,
    /// The consumer rejected the event.
    Consumer,
    /// The producer could not classify the loss more precisely.
    Unknown,
}

/// One self-describing event in the core trace stream.
///
/// Attribution is optional because existing callers often know only a stage
/// name. New producers should fill every field that is available rather than
/// inventing identities for values they cannot prove.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TraceEvent {
    /// Schema version for this event.
    pub version: u16,
    /// The operation represented by this event.
    pub kind: TraceEventKind,
    /// The frame this event belongs to, when a frame is active.
    #[serde(default)]
    pub frame_id: Option<TraceFrameId>,
    /// The stable span identity, when this is a span event.
    #[serde(default)]
    pub span_id: Option<TraceSpanId>,
    /// The enclosing span identity, when this span has a parent.
    #[serde(default)]
    pub parent_span_id: Option<TraceSpanId>,
    /// Stable producer thread identity, when the producer can supply one.
    #[serde(default)]
    pub thread_id: Option<TraceThreadId>,
    /// Stable queue identity, such as a GPU queue or CPU work queue.
    #[serde(default)]
    pub queue_id: Option<TraceQueueId>,
    /// Retained element address, not an application pointer.
    #[serde(default)]
    pub element_address: Option<TraceElementAddress>,
    /// Compositing boundary owning the work.
    #[serde(default)]
    pub boundary_id: Option<TraceBoundaryId>,
    /// Scroll root owning the work.
    #[serde(default)]
    pub root_id: Option<TraceRootId>,
    /// Tile coordinate within the owning root or boundary.
    #[serde(default)]
    pub tile: Option<TraceTileCoord>,
    /// Monotonic event start (or point) timestamp.
    pub start_timestamp: TraceTimestamp,
    /// Optional end timestamp for completed spans and GPU timestamp pairs.
    #[serde(default)]
    pub end_timestamp: Option<TraceTimestamp>,
}

impl TraceEvent {
    /// Creates an event with the current contract version and no attribution.
    pub fn new(kind: TraceEventKind, start_timestamp: TraceTimestamp) -> Self {
        Self {
            version: TRACE_EVENT_VERSION,
            kind,
            frame_id: None,
            span_id: None,
            parent_span_id: None,
            thread_id: None,
            queue_id: None,
            element_address: None,
            boundary_id: None,
            root_id: None,
            tile: None,
            start_timestamp,
            end_timestamp: None,
        }
    }

    /// Creates a span-begin event.
    pub fn span_begin(name: impl Into<String>, start_timestamp: TraceTimestamp) -> Self {
        Self::new(
            TraceEventKind::SpanBegin { name: name.into() },
            start_timestamp,
        )
    }

    /// Creates a span-end event.
    pub fn span_end(span_id: TraceSpanId, end_timestamp: TraceTimestamp) -> Self {
        Self::new(TraceEventKind::SpanEnd, end_timestamp).with_span_ids(span_id, None)
    }

    /// Creates a counter event.
    pub fn counter(name: impl Into<String>, amount: u64, start_timestamp: TraceTimestamp) -> Self {
        Self::new(
            TraceEventKind::Counter {
                name: name.into(),
                amount,
            },
            start_timestamp,
        )
    }

    /// Creates a frame-presented event.
    pub fn frame_presented(frame_id: TraceFrameId, timestamp: TraceTimestamp) -> Self {
        Self::new(TraceEventKind::FramePresented, timestamp).with_frame_id(frame_id)
    }

    /// Creates a GPU timestamp event.
    pub fn gpu_timestamp(start_timestamp: TraceTimestamp, end_timestamp: TraceTimestamp) -> Self {
        Self::new(TraceEventKind::GpuTimestamp, start_timestamp).with_end_timestamp(end_timestamp)
    }

    /// Creates a dropped-event marker. `count` must be non-zero for the event
    /// to pass [`Self::validate`].
    pub fn dropped(count: u64, reason: TraceDropReason, start_timestamp: TraceTimestamp) -> Self {
        Self::new(TraceEventKind::Dropped { count, reason }, start_timestamp)
    }

    /// Sets the frame attribution.
    pub const fn with_frame_id(mut self, frame_id: TraceFrameId) -> Self {
        self.frame_id = Some(frame_id);
        self
    }

    /// Sets the span and parent-span attribution.
    pub const fn with_span_ids(
        mut self,
        span_id: TraceSpanId,
        parent_span_id: Option<TraceSpanId>,
    ) -> Self {
        self.span_id = Some(span_id);
        self.parent_span_id = parent_span_id;
        self
    }

    /// Sets the queue and thread attribution.
    pub const fn with_execution(
        mut self,
        thread_id: Option<TraceThreadId>,
        queue_id: Option<TraceQueueId>,
    ) -> Self {
        self.thread_id = thread_id;
        self.queue_id = queue_id;
        self
    }

    /// Sets element, boundary, root, and tile attribution.
    pub const fn with_ownership(
        mut self,
        element_address: Option<TraceElementAddress>,
        boundary_id: Option<TraceBoundaryId>,
        root_id: Option<TraceRootId>,
        tile: Option<TraceTileCoord>,
    ) -> Self {
        self.element_address = element_address;
        self.boundary_id = boundary_id;
        self.root_id = root_id;
        self.tile = tile;
        self
    }

    /// Sets an end timestamp for a completed event.
    pub const fn with_end_timestamp(mut self, end_timestamp: TraceTimestamp) -> Self {
        self.end_timestamp = Some(end_timestamp);
        self
    }

    /// Returns whether a serialized event version is understood by this core.
    pub const fn supports_version(version: u16) -> bool {
        version == TRACE_EVENT_VERSION
    }

    /// Checks the version and the timestamp/drop invariants of an event.
    pub fn validate(&self) -> Result<(), TraceEventError> {
        if !Self::supports_version(self.version) {
            return Err(TraceEventError::UnsupportedVersion(self.version));
        }
        if self
            .end_timestamp
            .is_some_and(|end_timestamp| end_timestamp < self.start_timestamp)
        {
            return Err(TraceEventError::EndBeforeStart);
        }
        if let TraceEventKind::Dropped { count, .. } = &self.kind
            && *count == 0
        {
            return Err(TraceEventError::ZeroDroppedCount);
        }
        Ok(())
    }
}

/// An invalid trace event contract value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TraceEventError {
    UnsupportedVersion(u16),
    EndBeforeStart,
    ZeroDroppedCount,
}

impl fmt::Display for TraceEventError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported trace event version {version}")
            }
            Self::EndBeforeStart => formatter.write_str("trace event ends before it starts"),
            Self::ZeroDroppedCount => formatter.write_str("dropped trace event has a zero count"),
        }
    }
}

impl std::error::Error for TraceEventError {}

/// The result of handing an event to an instrumentation consumer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TraceEventResult {
    /// The consumer retained the event.
    Recorded,
    /// The consumer was enabled but could not retain the event.
    Dropped { count: u64 },
    /// No consumer is active. This is the normal [`NoopHooks`] result.
    Disabled,
}

impl TraceEventResult {
    /// Returns the number of events lost by this submission.
    pub const fn dropped_count(self) -> u64 {
        match self {
            Self::Dropped { count } => count,
            Self::Recorded | Self::Disabled => 0,
        }
    }
}

/// A no-op instrumentation implementation.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopHooks;

/// Hooks used by core and backend frame assembly.
pub trait InstrumentationHooks: Send + Sync {
    /// Starts a named CPU span and returns an opaque token.
    fn begin_span(&self, name: &'static str) -> Option<u64>;
    /// Completes a CPU span.
    fn end_span(&self, token: u64);
    /// Adds to a named counter.
    fn counter(&self, name: &'static str, amount: u64);
    /// Notifies the implementation that a frame was presented.
    fn frame_presented(&self);
    /// Records a backend timestamp pair when supported.
    fn gpu_timestamp(&self, _name: &'static str, _start: u64, _end: u64) {}

    /// Delivers a versioned, attributed event to an optional trace consumer.
    ///
    /// This method has a default so existing hook implementations remain
    /// source-compatible. The event is borrowed to let disabled consumers
    /// return without taking ownership or retaining anything.
    fn record_trace_event(&self, _event: &TraceEvent) -> TraceEventResult {
        TraceEventResult::Disabled
    }

    /// Alias for consumers that prefer the shorter event-oriented spelling.
    fn trace_event(&self, event: &TraceEvent) -> TraceEventResult {
        self.record_trace_event(event)
    }

    /// Returns the cumulative number of events the consumer dropped.
    fn dropped_event_count(&self) -> u64 {
        0
    }

    /// Compatibility alias for dropped-event queries.
    fn dropped_trace_events(&self) -> u64 {
        self.dropped_event_count()
    }
}

impl InstrumentationHooks for NoopHooks {
    fn begin_span(&self, _name: &'static str) -> Option<u64> {
        None
    }
    fn end_span(&self, _token: u64) {}
    fn counter(&self, _name: &'static str, _amount: u64) {}
    fn frame_presented(&self) {}
}

/// A shared hook handle suitable for frame assembly.
pub type SharedHooks = Arc<dyn InstrumentationHooks>;

/// A CPU span that closes itself when dropped.
pub struct Span<'a> {
    hooks: &'a dyn InstrumentationHooks,
    token: Option<u64>,
}
impl<'a> Span<'a> {
    /// Begins a span, or creates an inert guard when instrumentation is off.
    pub fn new(hooks: &'a dyn InstrumentationHooks, name: &'static str) -> Self {
        Self {
            hooks,
            token: hooks.begin_span(name),
        }
    }
}
impl Drop for Span<'_> {
    fn drop(&mut self) {
        if let Some(token) = self.token.take() {
            self.hooks.end_span(token);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[derive(Default)]
    struct Hooks(std::sync::Mutex<Vec<&'static str>>);
    impl InstrumentationHooks for Hooks {
        fn begin_span(&self, name: &'static str) -> Option<u64> {
            self.0.lock().expect("test mutex poisoned").push(name);
            Some(1)
        }
        fn end_span(&self, _token: u64) {}
        fn counter(&self, _name: &'static str, _amount: u64) {}
        fn frame_presented(&self) {}
    }
    #[test]
    fn span_calls_begin() {
        let hooks = Hooks::default();
        let _span = Span::new(&hooks, "frame");
        assert_eq!(
            hooks.0.lock().expect("test mutex poisoned").as_slice(),
            &["frame"]
        );
    }

    #[test]
    fn existing_hook_implementations_use_the_disabled_rich_contract() {
        let hooks = Hooks::default();
        let event = TraceEvent::counter("layout:nodes", 3, 12);
        assert_eq!(hooks.record_trace_event(&event), TraceEventResult::Disabled);
        assert_eq!(hooks.trace_event(&event), TraceEventResult::Disabled);
        assert_eq!(hooks.dropped_event_count(), 0);
        assert_eq!(hooks.dropped_trace_events(), 0);
    }

    #[test]
    fn noop_hooks_do_not_retain_or_count_rich_events() {
        let hooks = NoopHooks;
        let event = TraceEvent::frame_presented(1, 2);
        assert_eq!(hooks.record_trace_event(&event), TraceEventResult::Disabled);
        assert_eq!(hooks.dropped_event_count(), 0);
    }

    #[test]
    fn trace_events_round_trip_all_attribution_fields() {
        let event = TraceEvent::span_begin("paint", 100)
            .with_frame_id(7)
            .with_span_ids(11, Some(5))
            .with_execution(Some(13), Some(17))
            .with_ownership(
                Some(19),
                Some(23),
                Some(29),
                Some(TraceTileCoord::new(-2, 4)),
            )
            .with_end_timestamp(101);

        let encoded = serde_json::to_string(&event).expect("trace event serializes");
        let decoded: TraceEvent = serde_json::from_str(&encoded).expect("trace event deserializes");

        assert_eq!(decoded, event);
        assert!(decoded.validate().is_ok());
        assert!(encoded.contains("\"version\":1"));
        assert!(encoded.contains("\"parent_span_id\":5"));
        assert!(encoded.contains("\"tile\":{"));
    }

    #[test]
    fn optional_attribution_can_be_absent_in_serialized_events() {
        let decoded: TraceEvent = serde_json::from_str(
            r#"{"version":1,"kind":{"type":"frame_presented"},"start_timestamp":9}"#,
        )
        .expect("minimal event deserializes");

        assert_eq!(decoded.frame_id, None);
        assert_eq!(decoded.start_timestamp, 9);
        assert!(decoded.validate().is_ok());
    }

    #[test]
    fn version_and_dropped_count_are_validated() {
        let mut event = TraceEvent::new(TraceEventKind::FramePresented, 20);
        event.version = TRACE_EVENT_VERSION + 1;
        assert_eq!(
            event.validate(),
            Err(TraceEventError::UnsupportedVersion(TRACE_EVENT_VERSION + 1))
        );

        let event = TraceEvent::dropped(0, TraceDropReason::EventLimit, 20);
        assert_eq!(event.validate(), Err(TraceEventError::ZeroDroppedCount));

        let event = TraceEvent::new(TraceEventKind::FramePresented, 20).with_end_timestamp(19);
        assert_eq!(event.validate(), Err(TraceEventError::EndBeforeStart));
    }

    #[derive(Default)]
    struct RecordingHooks {
        events: Mutex<Vec<TraceEvent>>,
    }

    impl InstrumentationHooks for RecordingHooks {
        fn begin_span(&self, _name: &'static str) -> Option<u64> {
            None
        }

        fn end_span(&self, _token: u64) {}

        fn counter(&self, _name: &'static str, _amount: u64) {}

        fn frame_presented(&self) {}

        fn record_trace_event(&self, event: &TraceEvent) -> TraceEventResult {
            self.events
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(event.clone());
            TraceEventResult::Recorded
        }

        fn dropped_event_count(&self) -> u64 {
            4
        }
    }

    #[test]
    fn rich_hooks_receive_events_and_report_drops_without_a_second_api() {
        let hooks = RecordingHooks::default();
        let event = TraceEvent::dropped(4, TraceDropReason::ByteLimit, 33);

        assert_eq!(hooks.trace_event(&event), TraceEventResult::Recorded);
        assert_eq!(hooks.dropped_trace_events(), 4);
        assert_eq!(
            hooks
                .events
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .as_slice(),
            &[event]
        );
    }
}
