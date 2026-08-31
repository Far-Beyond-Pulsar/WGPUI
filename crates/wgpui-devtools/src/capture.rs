//! Bounded, presentation-independent trace capture storage.

use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::Instant;

use wgpui_core::hooks::{
    InstrumentationHooks, TRACE_SCHEMA_VERSION, TraceEvent, TraceSpan,
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
    events: Mutex<Vec<RecordedEvent>>,
    spans: Mutex<Vec<ActiveSpan>>,
}

#[derive(Clone, Debug)]
struct RecordedEvent {
    sequence: u64,
    event: TraceEvent,
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
            events: events.into_iter().map(|record| record.event).collect(),
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
        if event.version != TRACE_SCHEMA_VERSION {
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

        if event.frame_id.is_none() {
            event.frame_id = Some(self.inner.frame_id.load(Ordering::Acquire));
        }
        if event.thread_id.is_none() {
            event.thread_id = Some(current_thread_id());
        }
        let sequence = self.inner.next_sequence.fetch_add(1, Ordering::Relaxed);
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
        recover_lock(&buffer.events).push(RecordedEvent { sequence, event });
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
        let event = TraceEvent::span_begin(span.name, now_ns(self.inner.started_at))
            .with_frame_id(span.frame_id)
            .with_span_ids(token, span.parent_span_id)
            .with_execution(Some(span.thread_id), span.queue_id)
            .with_ownership(
                span.element_address,
                span.boundary_id,
                span.root_id,
                span.tile,
            );
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
            let event = TraceEvent::span_end(token, now_ns(self.inner.started_at))
                .with_frame_id(active_span.span.frame_id)
                .with_span_ids(token, active_span.span.parent_span_id)
                .with_execution(
                    Some(active_span.span.thread_id),
                    active_span.span.queue_id,
                )
                .with_ownership(
                    active_span.span.element_address,
                    active_span.span.boundary_id,
                    active_span.span.root_id,
                    active_span.span.tile,
                );
            self.record_event(event);
        }
    }

    fn record_frame_presented(&self, frame_id: u64) {
        let event = TraceEvent::frame_presented(frame_id, now_ns(self.inner.started_at))
            .with_execution(Some(current_thread_id()), None);
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
            .filter(|event| event.frame_id == Some(frame_id))
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
            .filter_map(|event| event.frame_id)
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
        let event = TraceEvent::counter(
            name,
            amount,
            now_ns(self.inner.started_at),
        )
        .with_frame_id(self.inner.frame_id.load(Ordering::Acquire))
        .with_execution(Some(current_thread_id()), None);
        self.record_event(event);
    }

    fn frame_presented(&self) {
        let frame_id = self.inner.frame_id.load(Ordering::Acquire);
        self.present_frame(frame_id);
    }

    fn gpu_timestamp(&self, name: &'static str, start: u64, end: u64) {
        let _ = name;
        let event = TraceEvent::gpu_timestamp(start, end)
            .with_frame_id(self.inner.frame_id.load(Ordering::Acquire))
            .with_execution(Some(current_thread_id()), Some(1));
        self.record_event(event);
    }

    fn begin_trace_span(&self, span: TraceSpan) -> Option<u64> {
        CaptureRecorder::begin_trace_span(self, span)
    }

    fn end_trace_span(&self, token: u64, timestamp_ns: u64) {
        CaptureRecorder::end_trace_span(self, token, timestamp_ns);
    }

    fn trace_event(&self, event: &TraceEvent) -> wgpui_core::hooks::TraceEventResult {
        if self.record_event(event.clone()) {
            wgpui_core::hooks::TraceEventResult::Recorded
        } else {
            wgpui_core::hooks::TraceEventResult::Dropped { count: 1 }
        }
    }

    fn frame_started(&self, frame_id: u64) {
        self.begin_frame(frame_id);
    }

    fn frame_presented_with(&self, frame_id: u64) {
        self.present_frame(frame_id);
    }

    fn dropped_event_count(&self) -> u64 {
        self.dropped_events()
    }
}

impl Default for CaptureRecorder {
    fn default() -> Self {
        Self::new(CaptureConfig::default())
    }
}

// Versioned, renderer-independent capture data for external consumers.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::Path;

pub const SCHEMA_VERSION: u16 = 1;
pub const FRAME_MAGIC: &[u8] = b"WGPUI-CAPTURE\0";
const MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptureError {
    Json(String),
    Io(String),
    InvalidFrame(&'static str),
    UnsupportedSchema(u16),
    FrameTooLarge(usize),
}

impl fmt::Display for CaptureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(message) => write!(formatter, "capture JSON is invalid: {message}"),
            Self::Io(message) => write!(formatter, "capture file could not be written: {message}"),
            Self::InvalidFrame(message) => write!(formatter, "capture frame is invalid: {message}"),
            Self::UnsupportedSchema(version) => {
                write!(formatter, "capture schema version {version} is unsupported")
            }
            Self::FrameTooLarge(size) => {
                write!(formatter, "capture frame is too large: {size} bytes")
            }
        }
    }
}

impl std::error::Error for CaptureError {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Availability<T> {
    Available { data: T },
    Unavailable { reason: String },
}

impl<T> Availability<T> {
    pub fn available(data: T) -> Self {
        Self::Available { data }
    }

    pub fn unavailable(reason: impl Into<String>) -> Self {
        Self::Unavailable {
            reason: reason.into(),
        }
    }

    pub fn as_ref(&self) -> Availability<&T> {
        match self {
            Self::Available { data } => Availability::Available { data },
            Self::Unavailable { reason } => Availability::Unavailable {
                reason: reason.clone(),
            },
        }
    }

    pub fn is_available(&self) -> bool {
        matches!(self, Self::Available { .. })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaptureBundle {
    pub schema_version: u16,
    pub producer: ProducerMetadata,
    pub capture: CaptureMetadata,
    pub capabilities: Capabilities,
    pub element_tree: Availability<ElementTree>,
    pub flamegraph: Availability<Flamegraph>,
    pub timeline: Availability<Timeline>,
    pub memory: Availability<MemoryMap>,
    pub listeners: Availability<ListenerTable>,
    pub damage: Availability<DamageMap>,
    pub tiles: Availability<TileMap>,
    pub resources: Availability<ResourceTable>,
    pub network: Availability<NetworkWaterfall>,
}

impl CaptureBundle {
    pub fn new(capture: CaptureMetadata) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            producer: ProducerMetadata::default(),
            capture,
            capabilities: Capabilities::default(),
            element_tree: Availability::unavailable("element traversal was not recorded"),
            flamegraph: Availability::unavailable("CPU spans were not recorded"),
            timeline: Availability::unavailable("timeline events were not recorded"),
            memory: Availability::unavailable("allocation registry was not enabled"),
            listeners: Availability::unavailable("listener metadata was not recorded"),
            damage: Availability::unavailable("damage records were not recorded"),
            tiles: Availability::unavailable("tile records were not recorded"),
            resources: Availability::unavailable("resource registry was not enabled"),
            network: Availability::unavailable("network capture was not armed"),
        }
    }

    pub fn to_json(&self) -> Result<String, CaptureError> {
        serde_json::to_string_pretty(self).map_err(|error| CaptureError::Json(error.to_string()))
    }

    pub fn write_json(&self, path: impl AsRef<Path>) -> Result<(), CaptureError> {
        let json = self.to_json()?;
        std::fs::write(path, json).map_err(|error| CaptureError::Io(error.to_string()))
    }

    pub fn from_json(json: &str) -> Result<Self, CaptureError> {
        let capture: Self =
            serde_json::from_str(json).map_err(|error| CaptureError::Json(error.to_string()))?;
        capture.validate()?;
        Ok(capture)
    }

    pub fn to_framed_json(&self) -> Result<Vec<u8>, CaptureError> {
        let payload =
            serde_json::to_vec(self).map_err(|error| CaptureError::Json(error.to_string()))?;
        if payload.len() > MAX_FRAME_BYTES || payload.len() > u32::MAX as usize {
            return Err(CaptureError::FrameTooLarge(payload.len()));
        }
        let mut frame = Vec::with_capacity(FRAME_MAGIC.len() + 4 + payload.len());
        frame.extend_from_slice(FRAME_MAGIC);
        frame.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        frame.extend_from_slice(&payload);
        Ok(frame)
    }

    pub fn write_framed_json(&self, path: impl AsRef<Path>) -> Result<(), CaptureError> {
        let frame = self.to_framed_json()?;
        std::fs::write(path, frame).map_err(|error| CaptureError::Io(error.to_string()))
    }

    pub fn from_framed_json(frame: &[u8]) -> Result<Self, CaptureError> {
        let header_length = FRAME_MAGIC.len() + std::mem::size_of::<u32>();
        if frame.len() < header_length {
            return Err(CaptureError::InvalidFrame("missing header"));
        }
        if frame.get(..FRAME_MAGIC.len()) != Some(FRAME_MAGIC) {
            return Err(CaptureError::InvalidFrame("bad magic"));
        }
        let length_start = FRAME_MAGIC.len();
        let length_end = length_start + std::mem::size_of::<u32>();
        let length_bytes = frame
            .get(length_start..length_end)
            .ok_or(CaptureError::InvalidFrame("missing payload length"))?;
        let payload_length = u32::from_le_bytes(
            length_bytes
                .try_into()
                .map_err(|_| CaptureError::InvalidFrame("invalid payload length"))?,
        ) as usize;
        if payload_length > MAX_FRAME_BYTES {
            return Err(CaptureError::FrameTooLarge(payload_length));
        }
        let expected_length = header_length
            .checked_add(payload_length)
            .ok_or(CaptureError::InvalidFrame("payload length overflow"))?;
        if frame.len() != expected_length {
            return Err(CaptureError::InvalidFrame(
                "payload length does not match frame",
            ));
        }
        let payload = frame
            .get(header_length..)
            .ok_or(CaptureError::InvalidFrame("missing payload"))?;
        let json = std::str::from_utf8(payload)
            .map_err(|_| CaptureError::InvalidFrame("payload is not UTF-8"))?;
        Self::from_json(json)
    }

    fn validate(&self) -> Result<(), CaptureError> {
        if self.schema_version == 0 || self.schema_version > SCHEMA_VERSION {
            return Err(CaptureError::UnsupportedSchema(self.schema_version));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProducerMetadata {
    pub framework: String,
    pub framework_version: String,
    pub backend: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureMetadata {
    pub capture_id: String,
    pub frame_id: u64,
    pub monotonic_start_ns: u64,
    pub monotonic_end_ns: u64,
    pub frozen_after_present: bool,
    pub dropped_events: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capabilities {
    pub gpu_timestamps: bool,
    pub gpu_readback: bool,
    pub render_pass_records: bool,
    pub network_phase_hooks: bool,
    pub network_body_previews: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceLocation {
    pub file: String,
    pub line: u32,
    pub column: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ElementTree {
    pub roots: Vec<ElementNode>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElementNode {
    pub address: String,
    pub generation: u64,
    pub type_name: String,
    pub source: Option<SourceLocation>,
    pub parent: Option<String>,
    pub children: Vec<String>,
    pub bounds: Rect,
    pub transform: [f32; 6],
    pub clip: Option<Rect>,
    pub scroll_root: Option<String>,
    pub boundary: Option<String>,
    pub tile: Option<TileCoordinate>,
    pub invalidation: Vec<String>,
    pub last_presented: bool,
    pub paint_records: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Flamegraph {
    pub roots: Vec<FlameNode>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FlameNode {
    pub name: String,
    pub start_ns: u64,
    pub duration_ns: u64,
    pub exclusive_ns: u64,
    pub element: Option<String>,
    pub parent: Option<String>,
    pub children: Vec<String>,
    pub phase: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Timeline {
    pub events: Vec<TimelineEvent>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimelineEvent {
    pub id: String,
    pub frame_id: u64,
    pub start_ns: u64,
    pub duration_ns: u64,
    pub thread_or_queue: String,
    pub kind: TimelineEventKind,
    pub element: Option<String>,
    pub boundary: Option<String>,
    pub tile: Option<TileCoordinate>,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimelineEventKind {
    CpuSpan,
    GpuSpan,
    Input,
    Invalidation,
    Damage,
    Upload,
    CommandEncoder,
    RenderPass,
    Submit,
    Present,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryMap {
    pub allocations: Vec<MemoryAllocation>,
    pub total_live_bytes: u64,
    pub total_capacity_bytes: u64,
    pub high_water_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryAllocation {
    pub id: String,
    pub owner: String,
    pub category: String,
    pub live_bytes: u64,
    pub capacity_bytes: u64,
    pub high_water_bytes: u64,
    pub allocation_count: u64,
    pub ranges: Vec<ByteRange>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ByteRange {
    pub offset: u64,
    pub length: u64,
    pub label: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ListenerTable {
    pub listeners: Vec<ListenerRecord>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ListenerRecord {
    pub id: String,
    pub owner: String,
    pub event: String,
    pub registration_order: u64,
    pub phase: ListenerPhase,
    pub handler_present: bool,
    pub hitbox: Option<Rect>,
    pub dispatch_ancestry: Vec<String>,
    pub clipped: bool,
    pub rejection_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ListenerPhase {
    Capture,
    Bubble,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DamageMap {
    pub records: Vec<DamageRecord>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DamageRecord {
    pub root: String,
    pub content_rect: Rect,
    pub reason: DamageReason,
    pub element: Option<String>,
    pub tile: Option<TileCoordinate>,
    pub primitive_slots: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DamageReason {
    Content,
    Hover,
    Clip,
    ScrollReveal,
    Resource,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TileMap {
    pub grids: Vec<TileGrid>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TileGrid {
    pub root: String,
    pub tile_size: u32,
    pub visible: Vec<TileRecord>,
    pub resident_budget: u32,
    pub parent_damage_subtracted: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TileRecord {
    pub coordinate: TileCoordinate,
    pub generation: u64,
    pub state: TileState,
    pub owner: String,
    pub content_slots: Vec<u64>,
    pub transform_only: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TileCoordinate {
    pub x: i32,
    pub y: i32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TileState {
    Visible,
    Resident,
    NewlyExposed,
    Evicted,
    UntiledFallback,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceTable {
    pub resources: Vec<ResourceRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceRecord {
    pub id: String,
    pub kind: ResourceKind,
    pub owner: String,
    pub generation: u64,
    pub last_use_frame: u64,
    pub live_bytes: u64,
    pub capacity_bytes: u64,
    pub format: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub readback: ReadbackAvailability,
    pub ranges: Vec<ByteRange>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceKind {
    PrimitiveSlab,
    IndirectBuffer,
    Atlas,
    LayerTexture,
    TileTexture,
    Surface,
    StagingBuffer,
    ReadbackBuffer,
    QueryBuffer,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadbackAvailability {
    Available,
    Unavailable { reason: String },
    Redacted { reason: String },
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NetworkWaterfall {
    pub requests: Vec<NetworkRequest>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NetworkRequest {
    pub id: String,
    pub method: String,
    pub url: String,
    pub resource_type: String,
    pub initiator: Option<String>,
    pub request_headers: Vec<HeaderMetadata>,
    pub response_headers: Vec<HeaderMetadata>,
    pub status: Option<u16>,
    pub transfer_bytes: Option<u64>,
    pub cache: Option<String>,
    pub observation: NetworkObservation,
    pub phases: Vec<NetworkPhase>,
    pub body_preview: Option<BodyPreview>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeaderMetadata {
    pub name: String,
    pub value: String,
    pub redacted: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NetworkPhase {
    pub name: String,
    pub start_ns: u64,
    pub duration_ns: u64,
    pub available: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BodyPreview {
    pub content_type: Option<String>,
    pub bytes_base64: String,
    pub truncated: bool,
    pub redacted: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NetworkObservation {
    Complete,
    PartiallyObserved,
    MetadataOnly,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;
    use wgpui_core::hooks::TraceTileCoordinate;

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
        trace_span.queue_id = Some(1);
        trace_span.element_address = Some(8);
        trace_span.boundary_id = Some(13);
        trace_span.tile = Some(TraceTileCoordinate { x: -2, y: 4 });
        let _span = wgpui_core::hooks::Span::with_trace(&recorder, trace_span);
        let snapshot = recorder.snapshot();
        assert_eq!(
            snapshot.events.first().map(|event| {
                (
                    event.frame_id,
                    event.queue_id,
                    event.element_address,
                    event.boundary_id,
                    event.tile,
                )
            }),
            Some((
                42,
                Some(1),
                Some(8),
                Some(13),
                Some(TraceTileCoordinate { x: -2, y: 4 }),
            ))
        );
    }

    fn metadata() -> CaptureMetadata {
        CaptureMetadata {
            capture_id: "test".into(),
            frame_id: 7,
            monotonic_start_ns: 10,
            monotonic_end_ns: 20,
            frozen_after_present: true,
            dropped_events: 0,
        }
    }

    #[test]
    fn framed_capture_round_trips_and_rejects_trailing_bytes() {
        let capture = CaptureBundle::new(metadata());
        let frame = capture.to_framed_json().expect("encode capture");
        let decoded = CaptureBundle::from_framed_json(&frame).expect("decode capture");
        assert_eq!(decoded, capture);

        let mut trailing = frame;
        trailing.push(0);
        assert_eq!(
            CaptureBundle::from_framed_json(&trailing),
            Err(CaptureError::InvalidFrame(
                "payload length does not match frame"
            ))
        );
    }

    #[test]
    fn event_and_byte_budgets_report_drops() {
        let sample = TraceEvent {
            version: TRACE_SCHEMA_VERSION,
            frame_id: Some(1),
            thread_id: Some(1),
            queue_id: Some(0),
            start_timestamp: 0,
            span_id: None,
            parent_span_id: None,
            element_address: None,
            boundary_id: None,
            root_id: None,
            tile: Some(TraceTileCoordinate { x: 1, y: 2 }),
            kind: TraceEventKind::FramePresented,
            end_timestamp: None,
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
        assert!(after.events.iter().all(|event| event.frame_id == Some(9)));
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

    #[test]
    fn newer_schema_is_not_silently_accepted() {
        let mut value = serde_json::to_value(CaptureBundle::new(metadata())).expect("value");
        value["schema_version"] = serde_json::json!(SCHEMA_VERSION + 1);
        let json = serde_json::to_string(&value).expect("json");
        assert_eq!(
            CaptureBundle::from_json(&json),
            Err(CaptureError::UnsupportedSchema(SCHEMA_VERSION + 1))
        );
    }

    #[test]
    fn unavailable_sections_keep_the_reason() {
        let section: Availability<MemoryMap> = Availability::unavailable("device lost");
        let json = serde_json::to_string(&section).expect("serialize unavailable");
        assert_eq!(json, r#"{"status":"unavailable","reason":"device lost"}"#);
        assert!(!section.is_available());
    }
}
