//! Capture-only network tracing over an owned, backend-neutral HTTP seam.
//!
//! The native crates do not currently own an HTTP implementation. This module
//! therefore records a small streaming client contract that an application or
//! compatibility adapter can implement. A decorator is returned unchanged
//! while capture is disabled; a permanently installed decorator has a cheap
//! atomic fast path and can identify a request that becomes observable after
//! capture starts as partially observed.

use std::collections::BTreeSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll};
use std::time::Instant;

use futures::future::BoxFuture;
use futures::stream::{BoxStream, Stream, StreamExt};

pub type NetworkHeaders = Vec<(String, String)>;
pub type NetworkFuture = BoxFuture<'static, Result<NetworkResponse, NetworkError>>;

/// An owned error crossing the network seam.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NetworkError {
    message: String,
}

impl NetworkError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for NetworkError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for NetworkError {}

/// A streaming body. Chunks are preserved as delivered; the recorder never
/// coalesces them, because boundaries and timing are useful diagnostics.
pub struct NetworkBody {
    stream: BoxStream<'static, Result<Vec<u8>, NetworkError>>,
}

impl NetworkBody {
    pub fn empty() -> Self {
        Self::from_stream(futures::stream::empty())
    }

    pub fn once(chunk: impl Into<Vec<u8>> + Send + 'static) -> Self {
        Self::from_stream(futures::stream::once(async move { Ok(chunk.into()) }))
    }

    pub fn chunks<I, C>(chunks: I) -> Self
    where
        I: IntoIterator<Item = C>,
        I::IntoIter: Send + 'static,
        C: Into<Vec<u8>> + Send + 'static,
    {
        Self::from_stream(futures::stream::iter(
            chunks.into_iter().map(|chunk| Ok(chunk.into())),
        ))
    }

    pub fn from_stream<S>(stream: S) -> Self
    where
        S: Stream<Item = Result<Vec<u8>, NetworkError>> + Send + 'static,
    {
        Self {
            stream: stream.boxed(),
        }
    }
}

impl Stream for NetworkBody {
    type Item = Result<Vec<u8>, NetworkError>;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.stream.as_mut().poll_next(context)
    }
}

/// Redirect handling requested by a caller. The transport may still expose
/// redirect events even when it ultimately refuses to follow one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RedirectPolicy {
    #[default]
    Follow,
    Manual,
}

pub struct NetworkRequest {
    pub method: String,
    pub url: String,
    pub headers: NetworkHeaders,
    pub body: NetworkBody,
    pub redirect_policy: RedirectPolicy,
    transport_observer: Option<Arc<dyn NetworkTransportObserver>>,
}

impl NetworkRequest {
    pub fn new(method: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            method: method.into(),
            url: url.into(),
            headers: Vec::new(),
            body: NetworkBody::empty(),
            redirect_policy: RedirectPolicy::default(),
            transport_observer: None,
        }
    }

    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    pub fn with_body(mut self, body: NetworkBody) -> Self {
        self.body = body;
        self
    }

    pub fn with_redirect_policy(mut self, redirect_policy: RedirectPolicy) -> Self {
        self.redirect_policy = redirect_policy;
        self
    }

    pub fn with_transport_observer(mut self, observer: Arc<dyn NetworkTransportObserver>) -> Self {
        self.transport_observer = Some(observer);
        self
    }

    pub fn transport_observer(&self) -> Option<Arc<dyn NetworkTransportObserver>> {
        self.transport_observer.clone()
    }

    fn metadata(&self, redact: &Redaction) -> NetworkRequestMetadata {
        NetworkRequestMetadata {
            method: self.method.clone(),
            url: self.url.clone(),
            headers: redact.headers(&self.headers),
            redirect_policy: self.redirect_policy,
        }
    }
}

pub struct NetworkResponse {
    pub status: u16,
    pub headers: NetworkHeaders,
    pub body: NetworkBody,
}

impl NetworkResponse {
    pub fn new(status: u16, headers: NetworkHeaders, body: NetworkBody) -> Self {
        Self {
            status,
            headers,
            body,
        }
    }
}

/// A client implementation supplied by the application or a compatibility
/// adapter. The future may resolve to a response whose body remains live.
pub trait NetworkClient: Send + Sync {
    fn send(&self, request: NetworkRequest) -> NetworkFuture;
}

/// Optional transport callbacks. A client that cannot provide a phase simply
/// does not call it; its capture contains an explicit unavailable marker.
pub trait NetworkTransportObserver: Send + Sync {
    fn phase_started(&self, _phase: NetworkPhase) {}
    fn phase_finished(&self, _phase: NetworkPhase) {}
    fn redirect(&self, _status: u16, _from: &str, _to: &str) {}
    fn error(&self, _error: &NetworkError) {}
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum NetworkPhase {
    Dns,
    Connect,
    Tls,
    Protocol,
    Cache,
}

const NETWORK_PHASES: [NetworkPhase; 5] = [
    NetworkPhase::Dns,
    NetworkPhase::Connect,
    NetworkPhase::Tls,
    NetworkPhase::Protocol,
    NetworkPhase::Cache,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhaseAvailability {
    Unavailable,
    Available,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NetworkPhaseRecord {
    pub phase: NetworkPhase,
    pub availability: PhaseAvailability,
    pub started_at_ns: Option<u64>,
    pub ended_at_ns: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyDirection {
    Request,
    Response,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BodyPreview {
    pub bytes: Vec<u8>,
    pub truncated: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BodyChunkRecord {
    pub timestamp_ns: u64,
    pub size: usize,
    pub preview: BodyPreview,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObservationStatus {
    Complete,
    PartiallyObserved,
    Error,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NetworkRequestMetadata {
    pub method: String,
    pub url: String,
    pub headers: NetworkHeaders,
    pub redirect_policy: RedirectPolicy,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NetworkResponseRecord {
    pub status: u16,
    pub headers: NetworkHeaders,
    pub body_chunks: Vec<BodyChunkRecord>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NetworkRequestRecord {
    pub request_id: u64,
    pub started_at_ns: u64,
    pub status: ObservationStatus,
    pub request: NetworkRequestMetadata,
    pub request_body_chunks: Vec<BodyChunkRecord>,
    pub response: Option<NetworkResponseRecord>,
    pub phases: Vec<NetworkPhaseRecord>,
    pub redirects: Vec<NetworkRedirectRecord>,
    pub errors: Vec<String>,
    pub events: Vec<NetworkEvent>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NetworkRedirectRecord {
    pub timestamp_ns: u64,
    pub status: u16,
    pub from: String,
    pub to: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NetworkEvent {
    RequestStarted {
        timestamp_ns: u64,
    },
    RequestBodyChunk(BodyChunkRecord),
    ResponseStarted {
        timestamp_ns: u64,
        status: u16,
        headers: NetworkHeaders,
    },
    ResponseBodyChunk(BodyChunkRecord),
    Redirect(NetworkRedirectRecord),
    Phase(NetworkPhaseRecord),
    Error {
        timestamp_ns: u64,
        message: String,
    },
    Cancelled {
        timestamp_ns: u64,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NetworkCaptureSnapshot {
    pub schema_version: u16,
    pub records: Vec<NetworkRequestRecord>,
    pub dropped_events: u64,
    pub dropped_requests: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NetworkCaptureConfig {
    pub max_requests: usize,
    pub max_events: usize,
    pub max_preview_bytes_per_direction: usize,
    pub max_headers: usize,
    pub max_header_value_bytes: usize,
    pub max_error_bytes: usize,
    pub redact_headers: BTreeSet<String>,
}

impl Default for NetworkCaptureConfig {
    fn default() -> Self {
        let redact_headers = [
            "authorization",
            "proxy-authorization",
            "cookie",
            "set-cookie",
            "x-api-key",
            "api-key",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        Self {
            max_requests: 256,
            max_events: 16_384,
            max_preview_bytes_per_direction: 4 * 1024,
            max_headers: 128,
            max_header_value_bytes: 4 * 1024,
            max_error_bytes: 512,
            redact_headers,
        }
    }
}

struct Redaction {
    sensitive_headers: BTreeSet<String>,
    max_headers: usize,
    max_header_value_bytes: usize,
}

impl From<&NetworkCaptureConfig> for Redaction {
    fn from(config: &NetworkCaptureConfig) -> Self {
        Self {
            sensitive_headers: config
                .redact_headers
                .iter()
                .map(|header| header.to_ascii_lowercase())
                .collect(),
            max_headers: config.max_headers,
            max_header_value_bytes: config.max_header_value_bytes,
        }
    }
}

impl Redaction {
    fn headers(&self, headers: &NetworkHeaders) -> NetworkHeaders {
        headers
            .iter()
            .take(self.max_headers)
            .map(|(name, value)| {
                let normalized_name = name.to_ascii_lowercase();
                let value = if self.sensitive_headers.contains(&normalized_name) {
                    "[REDACTED]".to_owned()
                } else {
                    value.chars().take(self.max_header_value_bytes).collect()
                };
                (name.clone(), value)
            })
            .collect()
    }
}

struct Recorder {
    started_at: Instant,
    config: NetworkCaptureConfig,
    redaction: Redaction,
    next_request_id: AtomicU64,
    remaining_events: AtomicUsize,
    dropped_events: AtomicU64,
    dropped_requests: AtomicU64,
    records: Mutex<Vec<Arc<RequestRecord>>>,
}

struct RequestRecord {
    record: Mutex<NetworkRequestRecord>,
    preview_bytes: [AtomicUsize; 2],
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl Recorder {
    fn new(config: NetworkCaptureConfig) -> Self {
        let max_events = config.max_events;
        let redaction = Redaction::from(&config);
        Self {
            started_at: Instant::now(),
            config,
            redaction,
            next_request_id: AtomicU64::new(1),
            remaining_events: AtomicUsize::new(max_events),
            dropped_events: AtomicU64::new(0),
            dropped_requests: AtomicU64::new(0),
            records: Mutex::new(Vec::new()),
        }
    }

    fn timestamp(&self) -> u64 {
        self.started_at.elapsed().as_nanos().min(u64::MAX as u128) as u64
    }

    fn reserve_event(&self) -> bool {
        let mut remaining = self.remaining_events.load(Ordering::Relaxed);
        loop {
            if remaining == 0 {
                self.dropped_events.fetch_add(1, Ordering::Relaxed);
                return false;
            }
            match self.remaining_events.compare_exchange_weak(
                remaining,
                remaining - 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(updated) => remaining = updated,
            }
        }
    }

    fn start(
        self: &Arc<Self>,
        request: &NetworkRequest,
        partially_observed: bool,
    ) -> Option<Arc<RequestRecord>> {
        self.start_metadata(request.metadata(&self.redaction), partially_observed)
    }

    fn start_metadata(
        self: &Arc<Self>,
        metadata: NetworkRequestMetadata,
        partially_observed: bool,
    ) -> Option<Arc<RequestRecord>> {
        let mut records = lock(&self.records);
        if records.len() >= self.config.max_requests {
            self.dropped_requests.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let timestamp_ns = self.timestamp();
        let status = if partially_observed {
            ObservationStatus::PartiallyObserved
        } else {
            ObservationStatus::Complete
        };
        let record = Arc::new(RequestRecord {
            record: Mutex::new(NetworkRequestRecord {
                request_id: self.next_request_id.fetch_add(1, Ordering::Relaxed),
                started_at_ns: timestamp_ns,
                status,
                request: metadata,
                request_body_chunks: Vec::new(),
                response: None,
                phases: NETWORK_PHASES
                    .into_iter()
                    .map(|phase| NetworkPhaseRecord {
                        phase,
                        availability: PhaseAvailability::Unavailable,
                        started_at_ns: None,
                        ended_at_ns: None,
                    })
                    .collect(),
                redirects: Vec::new(),
                errors: Vec::new(),
                events: Vec::new(),
            }),
            preview_bytes: [AtomicUsize::new(0), AtomicUsize::new(0)],
        });
        records.push(Arc::clone(&record));
        drop(records);
        self.push_event(&record, NetworkEvent::RequestStarted { timestamp_ns });
        Some(record)
    }

    fn push_event(&self, record: &RequestRecord, event: NetworkEvent) {
        if self.reserve_event() {
            lock(&record.record).events.push(event);
        }
    }

    fn body_chunk(&self, record: &RequestRecord, direction: BodyDirection, bytes: &[u8]) {
        let index = match direction {
            BodyDirection::Request => 0,
            BodyDirection::Response => 1,
        };
        let limit = self.config.max_preview_bytes_per_direction;
        let previous = self.preview_bytes(record, index, bytes.len(), limit);
        let preview_length = limit.saturating_sub(previous);
        let preview = bytes[..bytes.len().min(preview_length)].to_vec();
        let chunk = BodyChunkRecord {
            timestamp_ns: self.timestamp(),
            size: bytes.len(),
            preview: BodyPreview {
                bytes: preview,
                truncated: previous.saturating_add(bytes.len()) > limit,
            },
        };
        if self.reserve_event() {
            let mut request_record = lock(&record.record);
            match direction {
                BodyDirection::Request => request_record.request_body_chunks.push(chunk.clone()),
                BodyDirection::Response => {
                    if let Some(response) = request_record.response.as_mut() {
                        response.body_chunks.push(chunk.clone());
                    }
                }
            }
            request_record.events.push(match direction {
                BodyDirection::Request => NetworkEvent::RequestBodyChunk(chunk),
                BodyDirection::Response => NetworkEvent::ResponseBodyChunk(chunk),
            });
        }
    }

    fn preview_bytes(
        &self,
        record: &RequestRecord,
        index: usize,
        chunk_length: usize,
        limit: usize,
    ) -> usize {
        let counter = &record.preview_bytes[index];
        let mut previous = counter.load(Ordering::Relaxed);
        loop {
            let remaining = limit.saturating_sub(previous);
            let added = remaining.min(chunk_length);
            match counter.compare_exchange_weak(
                previous,
                previous.saturating_add(added),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return previous,
                Err(updated) => previous = updated,
            }
        }
    }

    fn response(&self, record: &RequestRecord, response: &NetworkResponse) {
        let timestamp_ns = self.timestamp();
        let headers = self.redaction.headers(&response.headers);
        lock(&record.record).response = Some(NetworkResponseRecord {
            status: response.status,
            headers: headers.clone(),
            body_chunks: Vec::new(),
        });
        self.push_event(
            record,
            NetworkEvent::ResponseStarted {
                timestamp_ns,
                status: response.status,
                headers,
            },
        );
    }

    fn error(&self, record: &RequestRecord, error: &NetworkError) {
        let message: String = error
            .message()
            .chars()
            .take(self.config.max_error_bytes)
            .collect();
        let timestamp_ns = self.timestamp();
        let mut request_record = lock(&record.record);
        if request_record.status == ObservationStatus::Error
            && request_record.errors.last() == Some(&message)
        {
            return;
        }
        request_record.status = ObservationStatus::Error;
        drop(request_record);
        if self.reserve_event() {
            let mut request_record = lock(&record.record);
            request_record.errors.push(message.clone());
            request_record.events.push(NetworkEvent::Error {
                timestamp_ns,
                message,
            });
        }
    }

    fn complete(&self, record: &RequestRecord) {
        let mut request_record = lock(&record.record);
        if request_record.status == ObservationStatus::PartiallyObserved {
            request_record.status = ObservationStatus::Complete;
        }
    }

    fn cancel(&self, record: &RequestRecord) {
        let timestamp_ns = self.timestamp();
        let mut request_record = lock(&record.record);
        if matches!(
            request_record.status,
            ObservationStatus::Complete | ObservationStatus::PartiallyObserved
        ) {
            request_record.status = ObservationStatus::Cancelled;
            drop(request_record);
            self.push_event(record, NetworkEvent::Cancelled { timestamp_ns });
        }
    }

    fn phase_started(&self, record: &RequestRecord, phase: NetworkPhase) {
        let timestamp_ns = self.timestamp();
        let mut request_record = lock(&record.record);
        let phase_record = update_phase(&mut request_record.phases, phase);
        if let Some(phase_record) = phase_record {
            phase_record.availability = PhaseAvailability::Available;
            phase_record.started_at_ns = Some(timestamp_ns);
            let phase_event = phase_record.clone();
            drop(request_record);
            self.push_event(record, NetworkEvent::Phase(phase_event));
        }
    }

    fn phase_finished(&self, record: &RequestRecord, phase: NetworkPhase) {
        let timestamp_ns = self.timestamp();
        let mut request_record = lock(&record.record);
        let phase_record = update_phase(&mut request_record.phases, phase);
        if let Some(phase_record) = phase_record {
            phase_record.availability = PhaseAvailability::Available;
            if phase_record.started_at_ns.is_none() {
                phase_record.started_at_ns = Some(timestamp_ns);
            }
            phase_record.ended_at_ns = Some(timestamp_ns);
            let phase_event = phase_record.clone();
            drop(request_record);
            self.push_event(record, NetworkEvent::Phase(phase_event));
        }
    }

    fn redirect(&self, record: &RequestRecord, status: u16, from: &str, to: &str) {
        let redirect = NetworkRedirectRecord {
            timestamp_ns: self.timestamp(),
            status,
            from: from.to_owned(),
            to: to.to_owned(),
        };
        if self.reserve_event() {
            let mut request_record = lock(&record.record);
            request_record.redirects.push(redirect.clone());
            request_record.events.push(NetworkEvent::Redirect(redirect));
        }
    }

    fn snapshot(&self) -> NetworkCaptureSnapshot {
        let mut records: Vec<_> = lock(&self.records)
            .iter()
            .map(|record| lock(&record.record).clone())
            .collect();
        records.sort_by_key(|record| record.request_id);
        NetworkCaptureSnapshot {
            schema_version: 1,
            records,
            dropped_events: self.dropped_events.load(Ordering::Relaxed),
            dropped_requests: self.dropped_requests.load(Ordering::Relaxed),
        }
    }
}

fn update_phase(
    phases: &mut [NetworkPhaseRecord],
    phase: NetworkPhase,
) -> Option<&mut NetworkPhaseRecord> {
    phases.iter_mut().find(|record| record.phase == phase)
}

struct CaptureState {
    armed: AtomicBool,
    recorder: Mutex<Option<Arc<Recorder>>>,
}

/// Owns the lifecycle of one network capture. The normal path is a single
/// relaxed atomic load; recorder locks are only taken after that load succeeds.
#[derive(Clone)]
pub struct NetworkCapture {
    state: Arc<CaptureState>,
}

impl Default for NetworkCapture {
    fn default() -> Self {
        Self {
            state: Arc::new(CaptureState {
                armed: AtomicBool::new(false),
                recorder: Mutex::new(None),
            }),
        }
    }
}

impl NetworkCapture {
    pub fn arm(&self, config: NetworkCaptureConfig) -> bool {
        let recorder = Arc::new(Recorder::new(config));
        let mut active_recorder = lock(&self.state.recorder);
        if self.state.armed.load(Ordering::Acquire) || active_recorder.is_some() {
            return false;
        }
        *active_recorder = Some(recorder);
        self.state.armed.store(true, Ordering::Release);
        true
    }

    pub fn is_armed(&self) -> bool {
        self.state.armed.load(Ordering::Acquire)
    }

    pub fn finish(&self) -> Option<NetworkCaptureSnapshot> {
        self.state.armed.store(false, Ordering::Release);
        lock(&self.state.recorder)
            .take()
            .map(|recorder| recorder.snapshot())
    }

    pub fn snapshot(&self) -> Option<NetworkCaptureSnapshot> {
        if !self.is_armed() {
            return None;
        }
        lock(&self.state.recorder)
            .as_ref()
            .map(|recorder| recorder.snapshot())
    }

    /// Returns the original client when capture is disabled. This is the
    /// installation boundary used by application configuration code.
    pub fn decorate(&self, client: Arc<dyn NetworkClient>) -> Arc<dyn NetworkClient> {
        if self.is_armed() {
            Arc::new(CaptureNetworkClient::new(client, self.clone()))
        } else {
            client
        }
    }

    fn recorder(&self) -> Option<Arc<Recorder>> {
        if !self.is_armed() {
            return None;
        }
        lock(&self.state.recorder).clone()
    }
}

/// A decorator for callers that need a stable client handle across capture
/// arms. New requests are fully recorded while armed. A request already
/// submitted through this decorator is recorded as partially observed if it
/// is still pending when capture becomes armed.
pub struct CaptureNetworkClient {
    client: Arc<dyn NetworkClient>,
    capture: NetworkCapture,
}

pub type NetworkRecordingClient = CaptureNetworkClient;
pub type RecordingNetworkClient = CaptureNetworkClient;

impl CaptureNetworkClient {
    pub fn new(client: Arc<dyn NetworkClient>, capture: NetworkCapture) -> Self {
        Self { client, capture }
    }
}

impl NetworkClient for CaptureNetworkClient {
    fn send(&self, mut request: NetworkRequest) -> NetworkFuture {
        let metadata = NetworkRequestMetadata {
            method: request.method.clone(),
            url: request.url.clone(),
            headers: request.headers.clone(),
            redirect_policy: request.redirect_policy,
        };
        let recorder = self.capture.recorder().and_then(|recorder| {
            recorder
                .start(&request, false)
                .map(|record| (recorder, record))
        });
        if let Some((recorder, record)) = recorder.as_ref() {
            let recording_observer: Arc<dyn NetworkTransportObserver> =
                Arc::new(RecorderObserver {
                    recorder: Arc::clone(recorder),
                    record: Arc::clone(record),
                });
            request.transport_observer = Some(match request.transport_observer.take() {
                Some(existing) => Arc::new(CompositeObserver {
                    observers: vec![existing, recording_observer],
                }),
                None => recording_observer,
            });
        }
        let body = if let Some(recorder) = recorder.as_ref() {
            NetworkBody::from_stream(RecordingBody {
                inner: request.body,
                recorder: Arc::clone(&recorder.0),
                record: Arc::clone(&recorder.1),
                direction: BodyDirection::Request,
                completed: false,
            })
        } else {
            request.body
        };
        request.body = body;
        let future = self.client.send(request);
        Box::pin(RecordingSendFuture {
            inner: future,
            capture: self.capture.clone(),
            recorder,
            observability_attempted: self.capture.is_armed(),
            metadata,
            completed: false,
        })
    }
}

struct RecordingSendFuture {
    inner: NetworkFuture,
    capture: NetworkCapture,
    recorder: Option<(Arc<Recorder>, Arc<RequestRecord>)>,
    observability_attempted: bool,
    metadata: NetworkRequestMetadata,
    completed: bool,
}

struct RecorderObserver {
    recorder: Arc<Recorder>,
    record: Arc<RequestRecord>,
}

impl NetworkTransportObserver for RecorderObserver {
    fn phase_started(&self, phase: NetworkPhase) {
        self.recorder.phase_started(&self.record, phase);
    }

    fn phase_finished(&self, phase: NetworkPhase) {
        self.recorder.phase_finished(&self.record, phase);
    }

    fn redirect(&self, status: u16, from: &str, to: &str) {
        self.recorder.redirect(&self.record, status, from, to);
    }

    fn error(&self, error: &NetworkError) {
        self.recorder.error(&self.record, error);
    }
}

struct CompositeObserver {
    observers: Vec<Arc<dyn NetworkTransportObserver>>,
}

impl NetworkTransportObserver for CompositeObserver {
    fn phase_started(&self, phase: NetworkPhase) {
        for observer in &self.observers {
            observer.phase_started(phase);
        }
    }

    fn phase_finished(&self, phase: NetworkPhase) {
        for observer in &self.observers {
            observer.phase_finished(phase);
        }
    }

    fn redirect(&self, status: u16, from: &str, to: &str) {
        for observer in &self.observers {
            observer.redirect(status, from, to);
        }
    }

    fn error(&self, error: &NetworkError) {
        for observer in &self.observers {
            observer.error(error);
        }
    }
}

impl Future for RecordingSendFuture {
    type Output = Result<NetworkResponse, NetworkError>;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        if self.recorder.is_none()
            && !self.observability_attempted
            && let Some(recorder) = self.capture.recorder()
        {
            self.observability_attempted = true;
            let metadata = NetworkRequestMetadata {
                headers: recorder.redaction.headers(&self.metadata.headers),
                ..self.metadata.clone()
            };
            self.recorder = recorder
                .start_metadata(metadata, true)
                .map(|record| (recorder, record));
        }
        let recorder = self.recorder.clone();
        match self.inner.as_mut().poll(context) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Ok(mut response)) => {
                self.completed = true;
                if let Some((capture, record)) = recorder {
                    capture.response(&record, &response);
                    response.body = NetworkBody::from_stream(RecordingBody {
                        inner: response.body,
                        recorder: capture,
                        record,
                        direction: BodyDirection::Response,
                        completed: false,
                    });
                }
                Poll::Ready(Ok(response))
            }
            Poll::Ready(Err(error)) => {
                self.completed = true;
                if let Some((capture, record)) = recorder {
                    capture.error(&record, &error);
                }
                Poll::Ready(Err(error))
            }
        }
    }
}

impl Drop for RecordingSendFuture {
    fn drop(&mut self) {
        if !self.completed
            && let Some((capture, record)) = self.recorder.as_ref()
        {
            capture.cancel(record);
        }
    }
}

struct RecordingBody {
    inner: NetworkBody,
    recorder: Arc<Recorder>,
    record: Arc<RequestRecord>,
    direction: BodyDirection,
    completed: bool,
}

impl Stream for RecordingBody {
    type Item = Result<Vec<u8>, NetworkError>;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.poll_next_unpin(context) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Some(Ok(chunk))) => {
                self.recorder
                    .body_chunk(&self.record, self.direction, &chunk);
                Poll::Ready(Some(Ok(chunk)))
            }
            Poll::Ready(Some(Err(error))) => {
                self.completed = true;
                self.recorder.error(&self.record, &error);
                Poll::Ready(Some(Err(error)))
            }
            Poll::Ready(None) => {
                self.completed = true;
                self.recorder.complete(&self.record);
                Poll::Ready(None)
            }
        }
    }
}

impl Drop for RecordingBody {
    fn drop(&mut self) {
        if !self.completed {
            self.recorder.cancel(&self.record);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use std::sync::mpsc;

    struct FixedClient;

    impl NetworkClient for FixedClient {
        fn send(&self, mut request: NetworkRequest) -> NetworkFuture {
            Box::pin(async move {
                let mut body = Vec::new();
                while let Some(chunk) = request.body.next().await {
                    body.extend(chunk?);
                }
                Ok(NetworkResponse::new(
                    201,
                    vec![("content-type".into(), "application/octet-stream".into())],
                    NetworkBody::chunks([body, b"tail".to_vec()]),
                ))
            })
        }
    }

    #[test]
    fn disabled_installation_returns_the_original_client() {
        let capture = NetworkCapture::default();
        let client: Arc<dyn NetworkClient> = Arc::new(FixedClient);
        let decorated = capture.decorate(Arc::clone(&client));
        assert!(Arc::ptr_eq(&client, &decorated));
        let response = block_on(decorated.send(NetworkRequest::new("GET", "https://example.test")))
            .expect("fixed client should respond");
        assert_eq!(response.status, 201);
        assert!(capture.snapshot().is_none());
    }

    #[test]
    fn records_stream_boundaries_headers_and_bounded_previews() {
        let capture = NetworkCapture::default();
        assert!(capture.arm(NetworkCaptureConfig {
            max_preview_bytes_per_direction: 3,
            ..NetworkCaptureConfig::default()
        }));
        let client: Arc<dyn NetworkClient> = Arc::new(FixedClient);
        let decorated = capture.decorate(client);
        let request = NetworkRequest::new("POST", "https://example.test/upload")
            .header("Authorization", "secret")
            .header("X-Visible", "value")
            .with_body(NetworkBody::chunks([b"abc".to_vec(), b"def".to_vec()]));
        let mut response = block_on(decorated.send(request)).expect("fixed client should respond");
        let first = block_on(response.body.next());
        assert_eq!(first, Some(Ok(b"abcdef".to_vec())));
        let second = block_on(response.body.next());
        assert_eq!(second, Some(Ok(b"tail".to_vec())));
        assert_eq!(block_on(response.body.next()), None);
        let snapshot = capture.finish().expect("capture should finish");
        let record = snapshot.records.first().expect("one request");
        assert_eq!(record.status, ObservationStatus::Complete);
        assert_eq!(
            record.request.headers[0],
            ("Authorization".into(), "[REDACTED]".into())
        );
        assert_eq!(
            record.request.headers[1],
            ("X-Visible".into(), "value".into())
        );
        assert_eq!(
            record
                .request_body_chunks
                .iter()
                .map(|chunk| chunk.size)
                .collect::<Vec<_>>(),
            vec![3, 3]
        );
        assert_eq!(record.request_body_chunks[0].preview.bytes, b"abc");
        assert!(record.request_body_chunks[1].preview.truncated);
        assert_eq!(
            record
                .response
                .as_ref()
                .expect("response metadata")
                .body_chunks
                .iter()
                .map(|chunk| chunk.size)
                .collect::<Vec<_>>(),
            vec![6, 4]
        );
    }

    #[test]
    fn partial_request_is_observed_when_capture_arms_while_pending() {
        struct PendingClient {
            release: Mutex<Option<mpsc::Receiver<()>>>,
        }
        impl NetworkClient for PendingClient {
            fn send(&self, _request: NetworkRequest) -> NetworkFuture {
                let receiver = lock(&self.release).take().expect("receiver once");
                Box::pin(async move {
                    receiver.recv().expect("release sender should live");
                    Ok(NetworkResponse::new(204, Vec::new(), NetworkBody::empty()))
                })
            }
        }
        let (sender, receiver) = mpsc::channel();
        let capture = NetworkCapture::default();
        let client: Arc<dyn NetworkClient> = Arc::new(PendingClient {
            release: Mutex::new(Some(receiver)),
        });
        let decorated = CaptureNetworkClient::new(client, capture.clone());
        let future = decorated.send(NetworkRequest::new("GET", "https://example.test/wait"));
        assert!(capture.arm(NetworkCaptureConfig::default()));
        sender.send(()).expect("pending request should release");
        let response = block_on(future).expect("pending client should respond");
        assert_eq!(response.status, 204);
        let snapshot = capture.finish().expect("capture should finish");
        assert_eq!(
            snapshot.records[0].status,
            ObservationStatus::PartiallyObserved
        );
        assert!(snapshot.records[0].request_body_chunks.is_empty());
    }

    #[test]
    fn event_and_request_limits_are_reported_without_breaking_transport() {
        let capture = NetworkCapture::default();
        assert!(capture.arm(NetworkCaptureConfig {
            max_requests: 1,
            max_events: 1,
            ..NetworkCaptureConfig::default()
        }));
        let client: Arc<dyn NetworkClient> = Arc::new(FixedClient);
        let decorated = capture.decorate(client);
        let response = block_on(decorated.send(NetworkRequest::new("GET", "https://example.test")))
            .expect("transport should still respond");
        drop(response);
        let second = capture.decorate(Arc::new(FixedClient));
        let _second_response =
            block_on(second.send(NetworkRequest::new("GET", "https://example.test/2")))
                .expect("transport should still respond");
        let snapshot = capture.finish().expect("capture should finish");
        assert_eq!(snapshot.records.len(), 1);
        assert!(snapshot.dropped_events > 0);
        assert_eq!(snapshot.dropped_requests, 1);
    }

    #[test]
    fn optional_phase_callbacks_replace_unavailable_markers_and_record_redirects() {
        struct CallbackClient;
        impl NetworkClient for CallbackClient {
            fn send(&self, request: NetworkRequest) -> NetworkFuture {
                let observer = request.transport_observer();
                Box::pin(async move {
                    if let Some(observer) = observer {
                        observer.phase_started(NetworkPhase::Dns);
                        observer.phase_finished(NetworkPhase::Dns);
                        observer.redirect(302, "https://example.test", "https://example.test/next");
                    }
                    Ok(NetworkResponse::new(200, Vec::new(), NetworkBody::empty()))
                })
            }
        }
        let capture = NetworkCapture::default();
        assert!(capture.arm(NetworkCaptureConfig::default()));
        let decorated = capture.decorate(Arc::new(CallbackClient));
        let _response =
            block_on(decorated.send(NetworkRequest::new("GET", "https://example.test")))
                .expect("callback client should respond");
        let snapshot = capture.finish().expect("capture should finish");
        let record = &snapshot.records[0];
        assert_eq!(record.phases[0].availability, PhaseAvailability::Available);
        assert_eq!(
            record.phases[1].availability,
            PhaseAvailability::Unavailable
        );
        assert_eq!(record.redirects[0].status, 302);
    }

    #[test]
    fn cancellation_is_recorded_for_pending_requests_and_partial_bodies() {
        struct PendingClient;
        impl NetworkClient for PendingClient {
            fn send(&self, _request: NetworkRequest) -> NetworkFuture {
                Box::pin(futures::future::pending())
            }
        }

        let capture = NetworkCapture::default();
        assert!(capture.arm(NetworkCaptureConfig::default()));
        let decorated = capture.decorate(Arc::new(PendingClient));
        let future = decorated.send(NetworkRequest::new("GET", "https://example.test/pending"));
        drop(future);
        let snapshot = capture.finish().expect("capture should finish");
        assert_eq!(snapshot.records[0].status, ObservationStatus::Cancelled);
        assert!(
            snapshot.records[0]
                .events
                .iter()
                .any(|event| matches!(event, NetworkEvent::Cancelled { .. }))
        );

        let capture = NetworkCapture::default();
        assert!(capture.arm(NetworkCaptureConfig::default()));
        let decorated = capture.decorate(Arc::new(FixedClient));
        let mut response =
            block_on(decorated.send(NetworkRequest::new("GET", "https://example.test/body")))
                .expect("fixed client should respond");
        assert_eq!(block_on(response.body.next()), Some(Ok(b"".to_vec())));
        drop(response.body);
        let snapshot = capture.finish().expect("capture should finish");
        assert_eq!(snapshot.records[0].status, ObservationStatus::Cancelled);
    }

    #[test]
    fn transport_errors_are_bounded_and_recorded_once() {
        struct ErrorClient;
        impl NetworkClient for ErrorClient {
            fn send(&self, request: NetworkRequest) -> NetworkFuture {
                let observer = request.transport_observer();
                Box::pin(async move {
                    let error = NetworkError::new("secret transport failure");
                    if let Some(observer) = observer {
                        observer.error(&error);
                    }
                    Err(error)
                })
            }
        }
        let capture = NetworkCapture::default();
        assert!(capture.arm(NetworkCaptureConfig {
            max_error_bytes: 6,
            ..NetworkCaptureConfig::default()
        }));
        let decorated = capture.decorate(Arc::new(ErrorClient));
        let result =
            block_on(decorated.send(NetworkRequest::new("GET", "https://example.test/error")));
        match result {
            Ok(_) => panic!("error client should fail"),
            Err(error) => assert_eq!(error.message(), "secret transport failure"),
        }
        let snapshot = capture.finish().expect("capture should finish");
        assert_eq!(snapshot.records[0].status, ObservationStatus::Error);
        assert_eq!(snapshot.records[0].errors, vec!["secret".to_owned()]);
    }
}
