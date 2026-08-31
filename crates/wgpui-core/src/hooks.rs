//! Backend-neutral instrumentation hooks.

use std::sync::{Arc, OnceLock};

static TRACE_CLOCK: OnceLock<std::time::Instant> = OnceLock::new();

/// Returns nanoseconds from a process-local monotonic clock.
pub fn monotonic_timestamp_ns() -> u64 {
    TRACE_CLOCK
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_nanos() as u64
}

/// The version of the transport-neutral trace event contract.
pub const TRACE_SCHEMA_VERSION: u16 = 1;

/// A queue on which a trace event was produced.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub enum TraceQueue {
    #[default]
    Cpu,
    Gpu,
}

/// A stable retained-element address used by diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub struct TraceElementAddress {
    pub id: u64,
    pub generation: u32,
}

/// A tile coordinate used by diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub struct TraceTileCoordinate {
    pub x: i32,
    pub y: i32,
}

/// Metadata supplied when a CPU or GPU span begins.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceSpan {
    pub name: &'static str,
    pub frame_id: u64,
    pub thread_id: u64,
    pub queue: TraceQueue,
    pub parent_span_id: Option<u64>,
    pub element: Option<TraceElementAddress>,
    pub boundary_id: Option<u64>,
    pub tile: Option<TraceTileCoordinate>,
}

impl TraceSpan {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            frame_id: 0,
            thread_id: 0,
            queue: TraceQueue::Cpu,
            parent_span_id: None,
            element: None,
            boundary_id: None,
            tile: None,
        }
    }
}

/// The payload of a versioned trace event.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub enum TraceEventKind {
    SpanBegin { name: String },
    SpanEnd,
    Counter { name: String, amount: u64 },
    FramePresented,
    GpuTimestamp { name: String, start: u64, end: u64 },
}

/// A backend-neutral trace event.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub struct TraceEvent {
    pub schema_version: u16,
    pub sequence: u64,
    pub frame_id: u64,
    pub thread_id: u64,
    pub queue: TraceQueue,
    pub timestamp_ns: u64,
    pub span_id: Option<u64>,
    pub parent_span_id: Option<u64>,
    pub element: Option<TraceElementAddress>,
    pub boundary_id: Option<u64>,
    pub tile: Option<TraceTileCoordinate>,
    pub kind: TraceEventKind,
}

impl TraceEvent {
    pub fn span_begin(timestamp_ns: u64, span_id: u64, span: &TraceSpan) -> Self {
        Self {
            schema_version: TRACE_SCHEMA_VERSION,
            sequence: 0,
            frame_id: span.frame_id,
            thread_id: span.thread_id,
            queue: span.queue,
            timestamp_ns,
            span_id: Some(span_id),
            parent_span_id: span.parent_span_id,
            element: span.element,
            boundary_id: span.boundary_id,
            tile: span.tile,
            kind: TraceEventKind::SpanBegin {
                name: span.name.to_owned(),
            },
        }
    }

    pub fn span_end(timestamp_ns: u64, span_id: u64, span: &TraceSpan) -> Self {
        Self {
            schema_version: TRACE_SCHEMA_VERSION,
            sequence: 0,
            frame_id: span.frame_id,
            thread_id: span.thread_id,
            queue: span.queue,
            timestamp_ns,
            span_id: Some(span_id),
            parent_span_id: span.parent_span_id,
            element: span.element,
            boundary_id: span.boundary_id,
            tile: span.tile,
            kind: TraceEventKind::SpanEnd,
        }
    }

    pub fn estimated_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + match &self.kind {
                TraceEventKind::SpanBegin { name }
                | TraceEventKind::Counter { name, .. }
                | TraceEventKind::GpuTimestamp { name, .. } => name.len(),
                TraceEventKind::SpanEnd | TraceEventKind::FramePresented => 0,
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

    /// Starts a span with stable trace metadata.
    fn begin_trace_span(&self, span: TraceSpan) -> Option<u64> {
        self.begin_span(span.name)
    }

    /// Completes a span with stable trace metadata.
    fn end_trace_span(&self, token: u64, _timestamp_ns: u64) {
        self.end_span(token);
    }

    /// Records an already assembled trace event.
    fn trace_event(&self, _event: TraceEvent) {}

    /// Announces the frame whose events are about to be collected.
    fn frame_started(&self, _frame_id: u64) {}

    /// Presentation notification carrying the stable frame ID.
    fn frame_presented_with(&self, _frame_id: u64) {
        self.frame_presented();
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
    uses_trace_contract: bool,
}
impl<'a> Span<'a> {
    /// Begins a span, or creates an inert guard when instrumentation is off.
    pub fn new(hooks: &'a dyn InstrumentationHooks, name: &'static str) -> Self {
        Self {
            hooks,
            token: hooks.begin_span(name),
            uses_trace_contract: false,
        }
    }

    /// Begins a span while preserving the stable trace metadata when supported.
    pub fn with_trace(hooks: &'a dyn InstrumentationHooks, span: TraceSpan) -> Self {
        Self {
            hooks,
            token: hooks.begin_trace_span(span),
            uses_trace_contract: true,
        }
    }
}
impl Drop for Span<'_> {
    fn drop(&mut self) {
        if let Some(token) = self.token.take() {
            if self.uses_trace_contract {
                self.hooks.end_trace_span(token, monotonic_timestamp_ns());
            } else {
                self.hooks.end_span(token);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn trace_span_defaults_to_legacy_hook() {
        let hooks = Hooks::default();
        let _span = Span::with_trace(&hooks, TraceSpan::new("frame"));
        assert_eq!(
            hooks.0.lock().expect("test mutex poisoned").as_slice(),
            &["frame"]
        );
    }
}
