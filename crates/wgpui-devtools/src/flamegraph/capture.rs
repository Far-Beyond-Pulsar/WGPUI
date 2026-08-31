//! Capture-only frame recording.
//!
//! The controller deliberately keeps the normal state in an atomic. Recording
//! owns a mutex only after a caller has armed a capture, so an application can
//! keep a controller around without paying for a recorder on ordinary frames.

use std::fmt;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, OnceLock};
use std::time::Instant;

use super::CaptureRequest;

pub const CAPTURE_SCHEMA_NAME: &str = "wgpui.capture";
pub const CAPTURE_SCHEMA_VERSION: u32 = 1;
pub const DEFAULT_EVENT_BUDGET: usize = 16_384;
pub const DEFAULT_BYTE_BUDGET: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CaptureState {
    Disabled = 0,
    Armed = 1,
    Collecting = 2,
    Frozen = 3,
    Exported = 4,
}

impl CaptureState {
    fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Armed,
            2 => Self::Collecting,
            3 => Self::Frozen,
            4 => Self::Exported,
            _ => Self::Disabled,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureTarget {
    NextFrame,
    Frame(u64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CaptureLimits {
    pub max_events: usize,
    pub max_bytes: usize,
}

impl Default for CaptureLimits {
    fn default() -> Self {
        Self {
            max_events: DEFAULT_EVENT_BUDGET,
            max_bytes: DEFAULT_BYTE_BUDGET,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureError {
    AlreadyArmed,
    NotArmed,
    NotFrozen,
    InvalidLimits,
}

impl fmt::Display for CaptureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::AlreadyArmed => "a capture is already armed or collecting",
            Self::NotArmed => "the capture is not armed",
            Self::NotFrozen => "the capture has not reached the presentation boundary",
            Self::InvalidLimits => "capture limits must allow at least one event and byte",
        };
        formatter.write_str(message)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DroppedEventStatus {
    pub events: u64,
    pub bytes: u64,
    pub event_budget: usize,
    pub byte_budget: usize,
}

impl DroppedEventStatus {
    pub const fn none(limits: CaptureLimits) -> Self {
        Self {
            events: 0,
            bytes: 0,
            event_budget: limits.max_events,
            byte_budget: limits.max_bytes,
        }
    }

    pub const fn has_dropped_events(self) -> bool {
        self.events != 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordResult {
    Ignored,
    Recorded,
    Dropped(DroppedEventStatus),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureEventKind {
    SpanBegin,
    SpanEnd,
    Counter,
    Input,
    Invalidation,
    Damage,
    GpuTimestamp,
    Marker,
}

impl CaptureEventKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SpanBegin => "span_begin",
            Self::SpanEnd => "span_end",
            Self::Counter => "counter",
            Self::Input => "input",
            Self::Invalidation => "invalidation",
            Self::Damage => "damage",
            Self::GpuTimestamp => "gpu_timestamp",
            Self::Marker => "marker",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaptureEvent {
    pub sequence: u64,
    pub frame_id: u64,
    pub timestamp_ns: u64,
    pub thread_id: u64,
    pub kind: CaptureEventKind,
    pub name: String,
    pub payload: Vec<u8>,
}

impl CaptureEvent {
    pub fn new(frame_id: u64, timestamp_ns: u64, kind: CaptureEventKind) -> Self {
        Self {
            sequence: 0,
            frame_id,
            timestamp_ns,
            thread_id: 0,
            kind,
            name: String::new(),
            payload: Vec::new(),
        }
    }

    pub fn named(
        frame_id: u64,
        timestamp_ns: u64,
        kind: CaptureEventKind,
        name: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            ..Self::new(frame_id, timestamp_ns, kind)
        }
    }

    pub fn with_payload(mut self, payload: impl Into<Vec<u8>>) -> Self {
        self.payload = payload.into();
        self
    }

    fn estimated_size(&self) -> usize {
        40usize
            .saturating_add(self.name.len())
            .saturating_add(self.payload.len())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClockCalibration {
    pub cpu_origin_ns: u64,
    pub gpu_origin_ticks: u64,
    pub gpu_ticks_per_cpu_ns: f64,
}

impl ClockCalibration {
    pub fn new(
        cpu_origin_ns: u64,
        gpu_origin_ticks: u64,
        gpu_ticks_per_cpu_ns: f64,
    ) -> Option<Self> {
        if gpu_ticks_per_cpu_ns.is_finite() && gpu_ticks_per_cpu_ns > 0.0 {
            Some(Self {
                cpu_origin_ns,
                gpu_origin_ticks,
                gpu_ticks_per_cpu_ns,
            })
        } else {
            None
        }
    }

    pub fn from_samples(
        cpu_start_ns: u64,
        cpu_end_ns: u64,
        gpu_start_ticks: u64,
        gpu_end_ticks: u64,
    ) -> Option<Self> {
        let cpu_delta = cpu_end_ns.checked_sub(cpu_start_ns)?;
        let gpu_delta = gpu_end_ticks.checked_sub(gpu_start_ticks)?;
        if cpu_delta == 0 || gpu_delta == 0 {
            return None;
        }
        let cpu_midpoint = cpu_start_ns + cpu_delta / 2;
        let gpu_midpoint = gpu_start_ticks + gpu_delta / 2;
        Self::new(
            cpu_midpoint,
            gpu_midpoint,
            gpu_delta as f64 / cpu_delta as f64,
        )
    }

    pub fn cpu_to_gpu(self, cpu_timestamp_ns: u64) -> Option<u64> {
        let delta = cpu_timestamp_ns as f64 - self.cpu_origin_ns as f64;
        let value = self.gpu_origin_ticks as f64 + delta * self.gpu_ticks_per_cpu_ns;
        (value.is_finite() && value >= 0.0 && value <= u64::MAX as f64)
            .then(|| value.round() as u64)
    }

    pub fn gpu_to_cpu(self, gpu_timestamp_ticks: u64) -> Option<u64> {
        let delta = gpu_timestamp_ticks as f64 - self.gpu_origin_ticks as f64;
        let value = self.cpu_origin_ns as f64 + delta / self.gpu_ticks_per_cpu_ns;
        (value.is_finite() && value >= 0.0 && value <= u64::MAX as f64)
            .then(|| value.round() as u64)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CaptureMetadata {
    pub schema_name: &'static str,
    pub schema_version: u32,
}

impl Default for CaptureMetadata {
    fn default() -> Self {
        Self {
            schema_name: CAPTURE_SCHEMA_NAME,
            schema_version: CAPTURE_SCHEMA_VERSION,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CaptureBundle {
    metadata: CaptureMetadata,
    request: CaptureRequest,
    target: CaptureTarget,
    frame_id: u64,
    started_at_ns: u64,
    presented_at_ns: u64,
    calibration: Option<ClockCalibration>,
    dropped_events: DroppedEventStatus,
    events: Vec<CaptureEvent>,
}

impl CaptureBundle {
    pub fn metadata(&self) -> CaptureMetadata {
        self.metadata
    }

    pub fn schema_version(&self) -> u32 {
        self.metadata.schema_version
    }

    pub fn request(&self) -> CaptureRequest {
        self.request
    }

    pub fn target(&self) -> CaptureTarget {
        self.target
    }

    pub fn frame_id(&self) -> u64 {
        self.frame_id
    }

    pub fn started_at_ns(&self) -> u64 {
        self.started_at_ns
    }

    pub fn presented_at_ns(&self) -> u64 {
        self.presented_at_ns
    }

    pub fn clock_calibration(&self) -> Option<ClockCalibration> {
        self.calibration
    }

    pub fn dropped_events(&self) -> DroppedEventStatus {
        self.dropped_events
    }

    pub fn dropped_event_status(&self) -> DroppedEventStatus {
        self.dropped_events()
    }

    pub fn events(&self) -> &[CaptureEvent] {
        &self.events
    }

    pub fn to_json(&self) -> String {
        let calibration = self.calibration.map_or_else(
            || "null".to_owned(),
            |calibration| {
                format!(
                    "{{\"cpu_origin_ns\":{},\"gpu_origin_ticks\":{},\"gpu_ticks_per_cpu_ns\":{}}}",
                    calibration.cpu_origin_ns,
                    calibration.gpu_origin_ticks,
                    calibration.gpu_ticks_per_cpu_ns
                )
            },
        );
        let events = self
            .events
            .iter()
            .map(|event| {
                format!(
                    "{{\"sequence\":{},\"frame_id\":{},\"timestamp_ns\":{},\"thread_id\":{},\"kind\":\"{}\",\"name\":\"{}\",\"payload_bytes\":{},\"payload_hex\":\"{}\"}}",
                    event.sequence,
                    event.frame_id,
                    event.timestamp_ns,
                    event.thread_id,
                    event.kind.as_str(),
                    escape_json(&event.name),
                    event.payload.len(),
                    hex(&event.payload)
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"schema\":{{\"name\":\"{}\",\"version\":{} }},\"frame_id\":{},\"started_at_ns\":{},\"presented_at_ns\":{},\"clock\":{},\"dropped_events\":{},\"dropped_bytes\":{},\"events\":[{}]}}",
            escape_json(self.metadata.schema_name),
            self.metadata.schema_version,
            self.frame_id,
            self.started_at_ns,
            self.presented_at_ns,
            calibration,
            self.dropped_events.events,
            self.dropped_events.bytes,
            events
        )
    }
}

fn escape_json(value: &str) -> String {
    value
        .chars()
        .flat_map(|character| match character {
            '"' => "\\\"".chars().collect::<Vec<_>>(),
            '\\' => "\\\\".chars().collect::<Vec<_>>(),
            '\n' => "\\n".chars().collect::<Vec<_>>(),
            '\r' => "\\r".chars().collect::<Vec<_>>(),
            '\t' => "\\t".chars().collect::<Vec<_>>(),
            character if character.is_control() => {
                format!("\\u{:04x}", character as u32).chars().collect()
            }
            character => vec![character],
        })
        .collect()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

struct Session {
    target: CaptureTarget,
    request: CaptureRequest,
    limits: CaptureLimits,
    frame_id: Option<u64>,
    started_at_ns: u64,
    events: Vec<CaptureEvent>,
    used_bytes: usize,
    dropped_events: u64,
    dropped_bytes: u64,
    sequence: u64,
    calibration: Option<ClockCalibration>,
    bundle: Option<Arc<CaptureBundle>>,
}

impl Session {
    fn new(limits: CaptureLimits) -> Self {
        Self {
            target: CaptureTarget::NextFrame,
            request: CaptureRequest::default(),
            limits,
            frame_id: None,
            started_at_ns: 0,
            events: Vec::new(),
            used_bytes: 0,
            dropped_events: 0,
            dropped_bytes: 0,
            sequence: 0,
            calibration: None,
            bundle: None,
        }
    }

    fn clear_capture(&mut self) {
        self.frame_id = None;
        self.started_at_ns = 0;
        self.events.clear();
        self.used_bytes = 0;
        self.dropped_events = 0;
        self.dropped_bytes = 0;
        self.sequence = 0;
        self.calibration = None;
        self.bundle = None;
    }

    fn dropped_status(&self) -> DroppedEventStatus {
        DroppedEventStatus {
            events: self.dropped_events,
            bytes: self.dropped_bytes,
            event_budget: self.limits.max_events,
            byte_budget: self.limits.max_bytes,
        }
    }
}

pub struct CaptureController {
    state: Arc<AtomicU8>,
    current_frame_id: Arc<AtomicU64>,
    clock: Arc<Instant>,
    session: Arc<Mutex<Session>>,
}

impl fmt::Debug for CaptureController {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CaptureController")
            .field("state", &self.state())
            .field("current_frame_id", &self.current_frame_id())
            .finish_non_exhaustive()
    }
}

impl Default for CaptureController {
    fn default() -> Self {
        Self::from_valid_limits(CaptureLimits::default())
    }
}

impl Clone for CaptureController {
    fn clone(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
            current_frame_id: Arc::clone(&self.current_frame_id),
            clock: Arc::clone(&self.clock),
            session: Arc::clone(&self.session),
        }
    }
}

impl CaptureController {
    pub fn new() -> Self {
        Self::from_valid_limits(CaptureLimits::default())
    }

    pub fn with_limits(limits: CaptureLimits) -> Result<Self, CaptureError> {
        if limits.max_events == 0 || limits.max_bytes == 0 {
            return Err(CaptureError::InvalidLimits);
        }
        Ok(Self::from_valid_limits(limits))
    }

    fn from_valid_limits(limits: CaptureLimits) -> Self {
        Self {
            state: Arc::new(AtomicU8::new(CaptureState::Disabled as u8)),
            current_frame_id: Arc::new(AtomicU64::new(0)),
            clock: Arc::new(Instant::now()),
            session: Arc::new(Mutex::new(Session::new(limits))),
        }
    }

    pub fn state(&self) -> CaptureState {
        CaptureState::from_u8(self.state.load(Ordering::Acquire))
    }

    pub fn is_collecting(&self) -> bool {
        self.state() == CaptureState::Collecting
    }

    pub fn current_frame_id(&self) -> u64 {
        self.current_frame_id.load(Ordering::Acquire)
    }

    pub fn arm_next_frame(&self, request: CaptureRequest) -> Result<(), CaptureError> {
        self.arm(CaptureTarget::NextFrame, request)
    }

    pub fn request_next_frame(&self, request: CaptureRequest) -> Result<(), CaptureError> {
        self.arm_next_frame(request)
    }

    pub fn arm_selected_frame(
        &self,
        frame_id: u64,
        request: CaptureRequest,
    ) -> Result<(), CaptureError> {
        self.arm(CaptureTarget::Frame(frame_id), request)
    }

    pub fn request_selected_frame(
        &self,
        frame_id: u64,
        request: CaptureRequest,
    ) -> Result<(), CaptureError> {
        self.arm_selected_frame(frame_id, request)
    }

    fn arm(&self, target: CaptureTarget, request: CaptureRequest) -> Result<(), CaptureError> {
        let mut session = lock(&self.session);
        if self.state() != CaptureState::Disabled {
            return Err(CaptureError::AlreadyArmed);
        }
        session.clear_capture();
        session.target = target;
        session.request = request;
        self.state
            .store(CaptureState::Armed as u8, Ordering::Release);
        Ok(())
    }

    pub fn begin_frame(&self, frame_id: u64, started_at_ns: u64) -> bool {
        self.current_frame_id.store(frame_id, Ordering::Release);
        if self.state() != CaptureState::Armed {
            return false;
        }
        let mut session = lock(&self.session);
        let selected = match session.target {
            CaptureTarget::NextFrame => true,
            CaptureTarget::Frame(selected_frame_id) => selected_frame_id == frame_id,
        };
        if !selected || self.state() != CaptureState::Armed {
            return false;
        }
        session.frame_id = Some(frame_id);
        session.started_at_ns = started_at_ns;
        self.state
            .store(CaptureState::Collecting as u8, Ordering::Release);
        true
    }

    pub fn now_ns(&self) -> u64 {
        self.clock.elapsed().as_nanos().min(u64::MAX as u128) as u64
    }

    pub fn set_clock_calibration(&self, calibration: ClockCalibration) -> bool {
        if !self.is_collecting() && self.state() != CaptureState::Armed {
            return false;
        }
        lock(&self.session).calibration = Some(calibration);
        true
    }

    pub fn record_event(&self, mut event: CaptureEvent) -> RecordResult {
        if self.state() != CaptureState::Collecting {
            return RecordResult::Ignored;
        }
        let mut session = lock(&self.session);
        if self.state() != CaptureState::Collecting || session.frame_id != Some(event.frame_id) {
            return RecordResult::Ignored;
        }
        if event.kind == CaptureEventKind::GpuTimestamp && !session.request.include_gpu {
            return RecordResult::Ignored;
        }
        let event_size = event.estimated_size();
        if session.events.len() >= session.limits.max_events
            || event_size > session.limits.max_bytes.saturating_sub(session.used_bytes)
        {
            session.dropped_events = session.dropped_events.saturating_add(1);
            session.dropped_bytes = session.dropped_bytes.saturating_add(event_size as u64);
            return RecordResult::Dropped(session.dropped_status());
        }
        session.used_bytes = session.used_bytes.saturating_add(event_size);
        session.sequence = session.sequence.saturating_add(1);
        event.sequence = session.sequence;
        session.events.push(event);
        RecordResult::Recorded
    }

    pub fn record_current(&self, kind: CaptureEventKind, name: impl Into<String>) -> RecordResult {
        let frame_id = self.current_frame_id();
        self.record_event(CaptureEvent::named(frame_id, self.now_ns(), kind, name))
    }

    pub fn presentation_boundary(
        &self,
        frame_id: u64,
        presented_at_ns: u64,
        calibration: Option<ClockCalibration>,
    ) -> Option<Arc<CaptureBundle>> {
        if self.state() != CaptureState::Collecting {
            return None;
        }
        let mut session = lock(&self.session);
        if self.state() != CaptureState::Collecting || session.frame_id != Some(frame_id) {
            return None;
        }
        let bundle = Arc::new(CaptureBundle {
            metadata: CaptureMetadata::default(),
            request: session.request,
            target: session.target,
            frame_id,
            started_at_ns: session.started_at_ns,
            presented_at_ns,
            calibration: calibration.or(session.calibration),
            dropped_events: session.dropped_status(),
            events: std::mem::take(&mut session.events),
        });
        session.bundle = Some(Arc::clone(&bundle));
        self.state
            .store(CaptureState::Frozen as u8, Ordering::Release);
        Some(bundle)
    }

    pub fn freeze_at_presentation(
        &self,
        frame_id: u64,
        presented_at_ns: u64,
        calibration: Option<ClockCalibration>,
    ) -> Option<Arc<CaptureBundle>> {
        self.presentation_boundary(frame_id, presented_at_ns, calibration)
    }

    pub fn frozen_bundle(&self) -> Option<Arc<CaptureBundle>> {
        let state = self.state();
        if state != CaptureState::Frozen && state != CaptureState::Exported {
            return None;
        }
        lock(&self.session).bundle.as_ref().map(Arc::clone)
    }

    pub fn export(&self) -> Result<Arc<CaptureBundle>, CaptureError> {
        let session = lock(&self.session);
        match self.state() {
            CaptureState::Frozen => {
                let bundle = session
                    .bundle
                    .as_ref()
                    .map(Arc::clone)
                    .ok_or(CaptureError::NotFrozen)?;
                self.state
                    .store(CaptureState::Exported as u8, Ordering::Release);
                Ok(bundle)
            }
            CaptureState::Exported => session
                .bundle
                .as_ref()
                .map(Arc::clone)
                .ok_or(CaptureError::NotFrozen),
            _ => Err(CaptureError::NotFrozen),
        }
    }

    pub fn reset(&self) {
        let mut session = lock(&self.session);
        session.clear_capture();
        self.state
            .store(CaptureState::Disabled as u8, Ordering::Release);
    }

    pub fn discard(&self) {
        self.reset();
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

static GLOBAL_CONTROLLER: OnceLock<CaptureController> = OnceLock::new();
static GLOBAL_CLOCK: LazyLock<Instant> = LazyLock::new(Instant::now);
static NEXT_GLOBAL_FRAME_ID: AtomicU64 = AtomicU64::new(0);

/// Returns the process capture controller if capture support has been used.
///
/// Keeping this lookup non-initializing is important: instrumentation hooks
/// call it on every frame, and ordinary runtime must not allocate a recorder.
pub fn global() -> Option<&'static CaptureController> {
    GLOBAL_CONTROLLER.get()
}

/// Returns the process capture controller, initializing it on an explicit
/// capture-management call.
pub fn global_controller() -> &'static CaptureController {
    GLOBAL_CONTROLLER.get_or_init(CaptureController::default)
}

pub fn begin_global_frame(frame_id: u64, started_at_ns: Option<u64>) -> bool {
    let controller = global_controller();
    controller.begin_frame(frame_id, started_at_ns.unwrap_or_else(global_now_ns))
}

/// Starts the next backend frame when a process-wide capture is armed.
///
/// This is intentionally a no-op when the controller has not been explicitly
/// initialized by a capture request, which keeps the renderer's feature-gated
/// call out of the normal allocation and locking paths.
pub fn begin_global_backend_frame() -> Option<u64> {
    let controller = global()?;
    if controller.state() != CaptureState::Armed {
        return None;
    }
    let frame_id = NEXT_GLOBAL_FRAME_ID.fetch_add(1, Ordering::Relaxed) + 1;
    controller
        .begin_frame(frame_id, global_now_ns())
        .then_some(frame_id)
}

pub fn present_global_frame(
    frame_id: u64,
    presented_at_ns: Option<u64>,
    calibration: Option<ClockCalibration>,
) -> Option<Arc<CaptureBundle>> {
    let controller = global_controller();
    controller.presentation_boundary(
        frame_id,
        presented_at_ns.unwrap_or_else(global_now_ns),
        calibration,
    )
}

pub fn present_global_backend_frame() -> Option<Arc<CaptureBundle>> {
    let controller = global()?;
    let frame_id = controller.current_frame_id();
    controller.presentation_boundary(frame_id, global_now_ns(), None)
}

pub fn global_now_ns() -> u64 {
    GLOBAL_CLOCK.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;
    use std::thread;

    fn event(frame_id: u64, number: u64) -> CaptureEvent {
        CaptureEvent::named(
            frame_id,
            number,
            CaptureEventKind::Marker,
            format!("event-{number}"),
        )
    }

    #[test]
    fn lifecycle_freezes_only_at_presentation_and_exports_immutable_bundle() {
        let controller = CaptureController::default();
        assert_eq!(controller.state(), CaptureState::Disabled);
        controller
            .arm_next_frame(CaptureRequest { include_gpu: true })
            .expect("capture can be armed");
        assert_eq!(controller.state(), CaptureState::Armed);
        assert!(controller.begin_frame(7, 200));
        assert_eq!(controller.state(), CaptureState::Collecting);
        let calibration = ClockCalibration::new(200, 10_000, 2.0).expect("valid calibration");
        assert!(controller.set_clock_calibration(calibration));
        assert_eq!(
            controller.record_event(event(7, 210)),
            RecordResult::Recorded
        );
        assert!(controller.frozen_bundle().is_none());
        let frozen = controller
            .presentation_boundary(7, 300, None)
            .expect("the active frame freezes");
        assert_eq!(controller.state(), CaptureState::Frozen);
        assert_eq!(frozen.events().len(), 1);
        assert_eq!(frozen.request(), CaptureRequest { include_gpu: true });
        assert_eq!(frozen.schema_version(), CAPTURE_SCHEMA_VERSION);
        assert!(frozen.to_json().contains("wgpui.capture"));
        let exported = controller.export().expect("frozen capture exports");
        assert_eq!(controller.state(), CaptureState::Exported);
        assert_eq!(exported.events(), frozen.events());
        assert!(controller.record_event(event(7, 400)) == RecordResult::Ignored);
    }

    #[test]
    fn gpu_events_follow_the_capture_request() {
        let controller = CaptureController::default();
        controller
            .arm_next_frame(CaptureRequest::default())
            .expect("armed");
        assert!(controller.begin_frame(1, 0));
        assert_eq!(
            controller.record_event(CaptureEvent::new(1, 1, CaptureEventKind::GpuTimestamp)),
            RecordResult::Ignored
        );
        assert!(controller.presentation_boundary(1, 2, None).is_some());
        controller.reset();
        controller
            .arm_next_frame(CaptureRequest { include_gpu: true })
            .expect("armed");
        assert!(controller.begin_frame(2, 3));
        assert_eq!(
            controller.record_event(CaptureEvent::new(2, 4, CaptureEventKind::GpuTimestamp)),
            RecordResult::Recorded
        );
    }

    #[test]
    fn selected_frame_ignores_other_frames() {
        let controller = CaptureController::default();
        controller
            .arm_selected_frame(42, CaptureRequest::default())
            .expect("capture can be armed");
        assert!(!controller.begin_frame(41, 10));
        assert_eq!(controller.state(), CaptureState::Armed);
        assert!(controller.begin_frame(42, 20));
        assert!(controller.presentation_boundary(42, 30, None).is_some());
    }

    #[test]
    fn overflow_is_reported_in_the_frozen_bundle() {
        let controller = CaptureController::with_limits(CaptureLimits {
            max_events: 1,
            max_bytes: 256,
        })
        .expect("limits are valid");
        controller
            .arm_next_frame(CaptureRequest::default())
            .expect("armed");
        assert!(controller.begin_frame(1, 0));
        assert_eq!(controller.record_event(event(1, 1)), RecordResult::Recorded);
        let result = controller.record_event(event(1, 2));
        assert!(matches!(result, RecordResult::Dropped(status) if status.events == 1));
        let bundle = controller
            .presentation_boundary(1, 3, None)
            .expect("frozen");
        assert!(bundle.dropped_events().has_dropped_events());
        assert_eq!(bundle.dropped_events().events, 1);
    }

    #[test]
    fn concurrent_producers_are_bounded_and_frame_atomic() {
        let controller = Arc::new(
            CaptureController::with_limits(CaptureLimits {
                max_events: 32,
                max_bytes: 4096,
            })
            .expect("limits are valid"),
        );
        controller
            .arm_next_frame(CaptureRequest::default())
            .expect("armed");
        assert!(controller.begin_frame(9, 0));
        let barrier = Arc::new(Barrier::new(5));
        let mut workers = Vec::new();
        for worker in 0..4 {
            let controller = Arc::clone(&controller);
            let barrier = Arc::clone(&barrier);
            workers.push(thread::spawn(move || {
                barrier.wait();
                controller.record_event(event(9, worker));
            }));
        }
        barrier.wait();
        for worker in workers {
            worker.join().expect("producer completes");
        }
        let bundle = controller
            .presentation_boundary(9, 1, None)
            .expect("frozen");
        assert_eq!(bundle.events().len(), 4);
        assert_eq!(
            bundle
                .events()
                .iter()
                .filter(|event| event.frame_id != 9)
                .count(),
            0
        );
    }

    #[test]
    fn clock_calibration_round_trips_samples() {
        let calibration = ClockCalibration::from_samples(100, 300, 1_000, 1_400)
            .expect("non-zero samples calibrate");
        assert_eq!(calibration.cpu_to_gpu(200), Some(1_200));
        assert_eq!(calibration.gpu_to_cpu(1_200), Some(200));
        assert!(ClockCalibration::from_samples(1, 1, 1, 2).is_none());
    }

    #[test]
    fn idle_controller_does_not_collect_events() {
        let controller = CaptureController::default();
        assert_eq!(
            controller.record_current(CaptureEventKind::Marker, "idle"),
            RecordResult::Ignored
        );
        assert_eq!(controller.state(), CaptureState::Disabled);
    }
}
