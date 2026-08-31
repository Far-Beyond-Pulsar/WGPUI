//! Bounded, presentation-independent trace capture storage.

use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::Instant;

use wgpui_core::hooks::{
    InstrumentationHooks, TRACE_SCHEMA_VERSION, TraceEvent, TraceEventKind, TraceQueue, TraceSpan,
};

const DISABLED: u8 = 0;
const COLLECTING: u8 = 1;
const FROZEN: u8 = 2;

/// Limits for one recorder lifetime. Budgets are shared by all producer
/// threads and all frames until [`CaptureRecorder::reset`] is called.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub struct CaptureConfig {
    pub max_events: usize,
    pub max_bytes: usize,
}

impl CaptureConfig {
    pub const fn new(max_events: usize, max_bytes: usize) -> Self {
        Self {
            max_events,
            max_bytes,
        }
    }
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self::new(65_536, 16 * 1024 * 1024)
    }
}

/// Compatibility name for callers that refer to recorder limits directly.
pub type RecorderConfig = CaptureConfig;

#[derive(Debug)]
struct ActiveSpan {
    token: u64,
    span: TraceSpan,
}

#[derive(Debug, Default)]
struct ThreadBuffer {
    events: Mutex<Vec<TraceEvent>>,
    spans: Mutex<Vec<ActiveSpan>>,
}

#[derive(Debug)]
struct RecorderInner {
    config: CaptureConfig,
    state: AtomicU8,
    frame_id: AtomicU64,
    next_span_id: AtomicU64,
    next_sequence: AtomicU64,
    accepted_events: AtomicUsize,
    accepted_bytes: AtomicUsize,
    dropped_events: AtomicU64,
    dropped_bytes: AtomicU64,
    active_writers: AtomicUsize,
    buffers: Mutex<Vec<Arc<ThreadBuffer>>>,
    boundary: Mutex<()>,
    started_at: Instant,
}

thread_local! {
    static THREAD_BUFFERS: RefCell<HashMap<usize, Weak<ThreadBuffer>>> = RefCell::new(HashMap::new());
}

fn recover_lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn current_thread_id() -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    std::thread::current().id().hash(&mut hasher);
    hasher.finish()
}

/// A cloneable bounded recorder. It does not allocate or retain events until
/// [`CaptureRecorder::start`] is called.
#[derive(Clone, Debug)]
pub struct CaptureRecorder {
    inner: Arc<RecorderInner>,
}

impl CaptureRecorder {
    pub fn new(config: CaptureConfig) -> Self {
        Self {
            inner: Arc::new(RecorderInner {
                config,
                state: AtomicU8::new(DISABLED),
                frame_id: AtomicU64::new(0),
                next_span_id: AtomicU64::new(1),
                next_sequence: AtomicU64::new(1),
                accepted_events: AtomicUsize::new(0),
                accepted_bytes: AtomicUsize::new(0),
                dropped_events: AtomicU64::new(0),
                dropped_bytes: AtomicU64::new(0),
                active_writers: AtomicUsize::new(0),
                buffers: Mutex::new(Vec::new()),
                boundary: Mutex::new(()),
                started_at: Instant::now(),
            }),
        }
    }

    pub fn enabled(config: CaptureConfig) -> Self {
        let recorder = Self::new(config);
        recorder.start();
        recorder
    }

    pub fn config(&self) -> CaptureConfig {
        self.inner.config
    }

    pub fn start(&self) {
        self.inner.state.store(COLLECTING, Ordering::Release);
    }

    pub fn stop(&self) {
        self.inner.state.store(DISABLED, Ordering::Release);
        self.wait_for_writers();
    }

    pub fn is_collecting(&self) -> bool {
        self.inner.state.load(Ordering::Acquire) == COLLECTING
    }

    pub fn begin_frame(&self, frame_id: u64) {
        self.inner.frame_id.store(frame_id, Ordering::Release);
        self.start();
    }

    /// Freezes collection before reading any producer buffer. Producers that
    /// raced with the boundary either finish before the snapshot or are
    /// rejected; no event can be appended after this method returns.
    pub fn present_frame(&self, frame_id: u64) -> CaptureSnapshot {
        if !self.is_collecting() {
            return self.snapshot();
        }
        let _boundary = recover_lock(&self.inner.boundary);
        if !self.is_collecting() {
            return self.snapshot();
        }
        self.inner.frame_id.store(frame_id, Ordering::Release);
        self.record_frame_presented(frame_id);
        self.freeze()
    }

    pub fn freeze(&self) -> CaptureSnapshot {
        let state = self.inner.state.load(Ordering::Acquire);
        if state == COLLECTING {
            let _ = self.inner.state.compare_exchange(
                COLLECTING,
                FROZEN,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
        }
        self.wait_for_writers();
        self.snapshot()
    }

    pub fn reset(&self) {
        self.inner.state.store(DISABLED, Ordering::Release);
        self.wait_for_writers();
        let buffers = recover_lock(&self.inner.buffers);
        for buffer in buffers.iter() {
            recover_lock(&buffer.events).clear();
            recover_lock(&buffer.spans).clear();
        }
        self.inner.accepted_events.store(0, Ordering::Release);
        self.inner.accepted_bytes.store(0, Ordering::Release);
        self.inner.dropped_events.store(0, Ordering::Release);
        self.inner.dropped_bytes.store(0, Ordering::Release);
        self.inner.next_sequence.store(1, Ordering::Release);
        self.inner.frame_id.store(0, Ordering::Release);
    }

    pub fn snapshot(&self) -> CaptureSnapshot {
        let mut events = Vec::new();
        let buffers = recover_lock(&self.inner.buffers);
        for buffer in buffers.iter() {
            events.extend(recover_lock(&buffer.events).iter().cloned());
        }
        events.sort_unstable_by_key(|event| event.sequence);
        CaptureSnapshot {
            schema_version: TRACE_SCHEMA_VERSION,
            config: self.inner.config,
            events,
            dropped_events: self.inner.dropped_events.load(Ordering::Acquire),
            dropped_bytes: self.inner.dropped_bytes.load(Ordering::Acquire),
        }
    }

    pub fn export(&self) -> CaptureSnapshot {
        self.freeze()
    }

    pub fn export_json(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(&self.export())
    }

    pub fn record(&self, event: TraceEvent) -> bool {
        self.record_event(event)
    }

    pub fn dropped_events(&self) -> u64 {
        self.inner.dropped_events.load(Ordering::Acquire)
    }

    pub fn dropped_bytes(&self) -> u64 {
        self.inner.dropped_bytes.load(Ordering::Acquire)
    }

    fn buffer(&self) -> Arc<ThreadBuffer> {
        let key = Arc::as_ptr(&self.inner) as usize;
        THREAD_BUFFERS.with(|buffers| {
            let mut buffers = buffers.borrow_mut();
            if let Some(buffer) = buffers.get(&key).and_then(Weak::upgrade) {
                return buffer;
            }
            let buffer = Arc::new(ThreadBuffer::default());
            recover_lock(&self.inner.buffers).push(buffer.clone());
            buffers.insert(key, Arc::downgrade(&buffer));
            buffer
        })
    }

    fn record_event(&self, mut event: TraceEvent) -> bool {
        if self.inner.state.load(Ordering::Acquire) != COLLECTING {
            return false;
        }
        if event.schema_version != TRACE_SCHEMA_VERSION {
            self.inner.dropped_events.fetch_add(1, Ordering::Relaxed);
            self.inner
                .dropped_bytes
                .fetch_add(event.estimated_bytes() as u64, Ordering::Relaxed);
            return false;
        }
        self.inner.active_writers.fetch_add(1, Ordering::AcqRel);
        if self.inner.state.load(Ordering::Acquire) != COLLECTING {
            self.inner.active_writers.fetch_sub(1, Ordering::AcqRel);
            return false;
        }

        if event.frame_id == 0 {
            event.frame_id = self.inner.frame_id.load(Ordering::Acquire);
        }
        if event.thread_id == 0 {
            event.thread_id = current_thread_id();
        }
        event.sequence = self.inner.next_sequence.fetch_add(1, Ordering::Relaxed);
        let bytes = event.estimated_bytes();
        if !self.reserve(bytes) {
            self.inner.dropped_events.fetch_add(1, Ordering::Relaxed);
            self.inner
                .dropped_bytes
                .fetch_add(bytes as u64, Ordering::Relaxed);
            self.inner.active_writers.fetch_sub(1, Ordering::AcqRel);
            return false;
        }

        let buffer = self.buffer();
        recover_lock(&buffer.events).push(event);
        self.inner.active_writers.fetch_sub(1, Ordering::AcqRel);
        true
    }

    fn reserve(&self, bytes: usize) -> bool {
        let event_limit = self.inner.config.max_events;
        let byte_limit = self.inner.config.max_bytes;
        let mut events = self.inner.accepted_events.load(Ordering::Relaxed);
        loop {
            if events >= event_limit {
                return false;
            }
            match self.inner.accepted_events.compare_exchange_weak(
                events,
                events + 1,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(updated) => events = updated,
            }
        }

        let mut used_bytes = self.inner.accepted_bytes.load(Ordering::Relaxed);
        loop {
            let Some(next_bytes) = used_bytes.checked_add(bytes) else {
                self.inner.accepted_events.fetch_sub(1, Ordering::AcqRel);
                return false;
            };
            if next_bytes > byte_limit {
                self.inner.accepted_events.fetch_sub(1, Ordering::AcqRel);
                return false;
            }
            match self.inner.accepted_bytes.compare_exchange_weak(
                used_bytes,
                next_bytes,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(updated) => used_bytes = updated,
            }
        }
    }

    fn wait_for_writers(&self) {
        while self.inner.active_writers.load(Ordering::Acquire) != 0 {
            std::thread::yield_now();
        }
    }

    fn begin_trace_span(&self, mut span: TraceSpan) -> Option<u64> {
        if !self.is_collecting() {
            return None;
        }
        if span.frame_id == 0 {
            span.frame_id = self.inner.frame_id.load(Ordering::Acquire);
        }
        if span.thread_id == 0 {
            span.thread_id = current_thread_id();
        }
        let buffer = self.buffer();
        {
            let spans = recover_lock(&buffer.spans);
            if span.parent_span_id.is_none() {
                span.parent_span_id = spans.last().map(|active| active.token);
            }
        }
        let token = self.inner.next_span_id.fetch_add(1, Ordering::Relaxed);
        let event = TraceEvent::span_begin(now_ns(self.inner.started_at), token, &span);
        if !self.record_event(event) {
            return None;
        }
        recover_lock(&buffer.spans).push(ActiveSpan { token, span });
        Some(token)
    }

    fn end_trace_span(&self, token: u64, _timestamp_ns: u64) {
        let buffer = self.buffer();
        let active_span = {
            let mut spans = recover_lock(&buffer.spans);
            spans
                .iter()
                .rposition(|active| active.token == token)
                .map(|index| spans.remove(index))
        };
        if let Some(active_span) = active_span {
            let event =
                TraceEvent::span_end(now_ns(self.inner.started_at), token, &active_span.span);
            self.record_event(event);
        }
    }

    fn record_frame_presented(&self, frame_id: u64) {
        let event = TraceEvent {
            schema_version: TRACE_SCHEMA_VERSION,
            sequence: 0,
            frame_id,
            thread_id: current_thread_id(),
            queue: TraceQueue::Cpu,
            timestamp_ns: now_ns(self.inner.started_at),
            span_id: None,
            parent_span_id: None,
            element: None,
            boundary_id: None,
            tile: None,
            kind: TraceEventKind::FramePresented,
        };
        self.record_event(event);
    }
}

fn now_ns(started_at: Instant) -> u64 {
    started_at.elapsed().as_nanos() as u64
}

/// An immutable view of the recorder at a presentation boundary.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub struct CaptureSnapshot {
    pub schema_version: u16,
    pub config: CaptureConfig,
    pub events: Vec<TraceEvent>,
    pub dropped_events: u64,
    pub dropped_bytes: u64,
}

impl CaptureSnapshot {
    pub fn frame(&self, frame_id: u64) -> FrameSnapshot {
        let events = self
            .events
            .iter()
            .filter(|event| event.frame_id == frame_id)
            .cloned()
            .collect();
        FrameSnapshot {
            frame_id,
            events,
            dropped_events: self.dropped_events,
            dropped_bytes: self.dropped_bytes,
        }
    }

    pub fn frames(&self) -> Vec<FrameSnapshot> {
        let mut frame_ids = self
            .events
            .iter()
            .map(|event| event.frame_id)
            .collect::<Vec<_>>();
        frame_ids.sort_unstable();
        frame_ids.dedup();
        frame_ids
            .into_iter()
            .map(|frame_id| self.frame(frame_id))
            .collect()
    }

    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    pub fn byte_count(&self) -> usize {
        self.events.iter().map(TraceEvent::estimated_bytes).sum()
    }

    pub fn to_json(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }
}

pub type CaptureExport = CaptureSnapshot;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FrameSnapshot {
    pub frame_id: u64,
    pub events: Vec<TraceEvent>,
    pub dropped_events: u64,
    pub dropped_bytes: u64,
}

impl InstrumentationHooks for CaptureRecorder {
    fn begin_span(&self, name: &'static str) -> Option<u64> {
        self.begin_trace_span(TraceSpan::new(name))
    }

    fn end_span(&self, token: u64) {
        self.end_trace_span(token, now_ns(self.inner.started_at));
    }

    fn counter(&self, name: &'static str, amount: u64) {
        let event = TraceEvent {
            schema_version: TRACE_SCHEMA_VERSION,
            sequence: 0,
            frame_id: self.inner.frame_id.load(Ordering::Acquire),
            thread_id: current_thread_id(),
            queue: TraceQueue::Cpu,
            timestamp_ns: now_ns(self.inner.started_at),
            span_id: None,
            parent_span_id: None,
            element: None,
            boundary_id: None,
            tile: None,
            kind: TraceEventKind::Counter {
                name: name.to_owned(),
                amount,
            },
        };
        self.record_event(event);
    }

    fn frame_presented(&self) {
        let frame_id = self.inner.frame_id.load(Ordering::Acquire);
        self.present_frame(frame_id);
    }

    fn gpu_timestamp(&self, name: &'static str, start: u64, end: u64) {
        let event = TraceEvent {
            schema_version: TRACE_SCHEMA_VERSION,
            sequence: 0,
            frame_id: self.inner.frame_id.load(Ordering::Acquire),
            thread_id: current_thread_id(),
            queue: TraceQueue::Gpu,
            timestamp_ns: now_ns(self.inner.started_at),
            span_id: None,
            parent_span_id: None,
            element: None,
            boundary_id: None,
            tile: None,
            kind: TraceEventKind::GpuTimestamp {
                name: name.to_owned(),
                start,
                end,
            },
        };
        self.record_event(event);
    }

    fn begin_trace_span(&self, span: TraceSpan) -> Option<u64> {
        self.begin_trace_span(span)
    }

    fn end_trace_span(&self, token: u64, timestamp_ns: u64) {
        self.end_trace_span(token, timestamp_ns);
    }

    fn trace_event(&self, event: TraceEvent) {
        self.record_event(event);
    }

    fn frame_started(&self, frame_id: u64) {
        self.begin_frame(frame_id);
    }

    fn frame_presented_with(&self, frame_id: u64) {
        self.present_frame(frame_id);
    }
}

impl Default for CaptureRecorder {
    fn default() -> Self {
        Self::new(CaptureConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;
    use wgpui_core::hooks::{TraceElementAddress, TraceTileCoordinate};

    fn recorder() -> CaptureRecorder {
        CaptureRecorder::enabled(CaptureConfig::new(100, 100_000))
    }

    #[test]
    fn nested_spans_preserve_parent_relationships() {
        let recorder = recorder();
        recorder.begin_frame(7);
        let outer = recorder.begin_span("outer");
        let inner = recorder.begin_span("inner");
        if let Some(inner) = inner {
            recorder.end_span(inner);
        }
        if let Some(outer) = outer {
            recorder.end_span(outer);
        }
        let snapshot = recorder.snapshot();
        let begins = snapshot
            .events
            .iter()
            .filter_map(|event| match &event.kind {
                TraceEventKind::SpanBegin { name } => {
                    Some((name.as_str(), event.span_id, event.parent_span_id))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(begins.len(), 2);
        assert_eq!(begins[0].0, "outer");
        assert_eq!(begins[1].0, "inner");
        assert_eq!(begins[1].2, begins[0].1);
    }

    #[test]
    fn trace_metadata_survives_the_compatibility_hook() {
        let recorder = recorder();
        recorder.begin_frame(42);
        let mut trace_span = TraceSpan::new("paint");
        trace_span.queue = TraceQueue::Gpu;
        trace_span.element = Some(TraceElementAddress {
            id: 8,
            generation: 3,
        });
        trace_span.boundary_id = Some(13);
        trace_span.tile = Some(TraceTileCoordinate { x: -2, y: 4 });
        let _span = wgpui_core::hooks::Span::with_trace(&recorder, trace_span);
        let snapshot = recorder.snapshot();
        assert_eq!(
            snapshot.events.first().map(|event| {
                (
                    event.frame_id,
                    event.queue,
                    event.element,
                    event.boundary_id,
                    event.tile,
                )
            }),
            Some((
                42,
                TraceQueue::Gpu,
                Some(TraceElementAddress {
                    id: 8,
                    generation: 3,
                }),
                Some(13),
                Some(TraceTileCoordinate { x: -2, y: 4 }),
            ))
        );
    }

    #[test]
    fn event_and_byte_budgets_report_drops() {
        let sample = TraceEvent {
            schema_version: TRACE_SCHEMA_VERSION,
            sequence: 0,
            frame_id: 1,
            thread_id: 1,
            queue: TraceQueue::Cpu,
            timestamp_ns: 0,
            span_id: None,
            parent_span_id: None,
            element: None,
            boundary_id: None,
            tile: Some(TraceTileCoordinate { x: 1, y: 2 }),
            kind: TraceEventKind::FramePresented,
        };
        let bytes = sample.estimated_bytes();
        let recorder = CaptureRecorder::enabled(CaptureConfig::new(1, bytes));
        assert!(recorder.record(sample.clone()));
        assert!(!recorder.record(sample));
        assert_eq!(recorder.snapshot().event_count(), 1);
        assert_eq!(recorder.dropped_events(), 1);
    }

    #[test]
    fn concurrent_producers_share_bounded_storage() {
        let recorder = Arc::new(CaptureRecorder::enabled(CaptureConfig::new(1_000, 100_000)));
        let barrier = Arc::new(Barrier::new(8));
        let mut threads = Vec::new();
        for thread_index in 0..8 {
            let recorder = recorder.clone();
            let barrier = barrier.clone();
            threads.push(std::thread::spawn(move || {
                barrier.wait();
                for event_index in 0..50 {
                    recorder.counter("event", (thread_index * 50 + event_index) as u64);
                }
            }));
        }
        for thread in threads {
            assert!(thread.join().is_ok());
        }
        assert_eq!(recorder.snapshot().event_count(), 400);
    }

    #[test]
    fn freeze_is_atomic_at_frame_boundary() {
        let recorder = Arc::new(recorder());
        recorder.begin_frame(9);
        let producer = {
            let recorder = recorder.clone();
            std::thread::spawn(move || {
                for _ in 0..100 {
                    recorder.counter("race", 1);
                }
            })
        };
        let snapshot = recorder.present_frame(9);
        assert!(producer.join().is_ok());
        let after = recorder.snapshot();
        assert_eq!(snapshot, after);
        assert!(after.events.iter().all(|event| event.frame_id == 9));
    }

    #[test]
    fn disabled_mode_keeps_zero_storage() {
        let recorder = CaptureRecorder::new(CaptureConfig::default());
        recorder.counter("disabled", 1);
        let span = recorder.begin_span("disabled");
        assert!(span.is_none());
        assert_eq!(recorder.snapshot().event_count(), 0);
        assert_eq!(recorder.dropped_events(), 0);
        assert!(recorder.present_frame(1).events.is_empty());
    }

    #[test]
    fn frame_snapshots_reset_and_json_export_work() {
        let recorder = recorder();
        recorder.begin_frame(1);
        recorder.counter("one", 1);
        recorder.present_frame(1);
        let snapshot = recorder.snapshot();
        assert_eq!(snapshot.frame(1).events.len(), 2);
        assert!(snapshot.to_json().is_ok());
        recorder.reset();
        assert_eq!(recorder.snapshot().event_count(), 0);
        assert_eq!(recorder.dropped_events(), 0);
    }
}
