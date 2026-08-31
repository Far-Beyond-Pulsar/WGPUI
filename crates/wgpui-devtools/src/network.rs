//! Capture-only network diagnostics and deterministic fixture replay.
//!
//! The types in this module are transport-neutral. A runtime HTTP client can
//! adapt its owned request and response types to NetworkRecorder while a
//! capture is active. No client is installed, wrapped, or called by this
//! module when capture is disabled.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

pub const FROZEN_CAPTURE_SCHEMA_VERSION: u32 = 1;

const REDACTED_VALUE: &str = "<redacted>";

/// A header after capture-safe normalization.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Header {
    pub name: String,
    pub value: String,
}

impl Header {
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
        }
    }
}

/// The high-level kind of resource requested by an application.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NetworkResourceType {
    Document,
    Stylesheet,
    Script,
    Image,
    Font,
    Media,
    Fetch,
    Xhr,
    WebSocket,
    Other,
    #[default]
    Unknown,
}

/// Identifies the code or resource that caused a request.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct Initiator {
    pub kind: String,
    pub source: Option<String>,
    pub request_id: Option<u64>,
}

impl Initiator {
    pub fn new(kind: impl Into<String>, source: Option<String>, request_id: Option<u64>) -> Self {
        Self {
            kind: kind.into(),
            source,
            request_id,
        }
    }
}

/// Cache outcome visible to the transport adapter.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheStatus {
    #[default]
    None,
    Memory,
    Disk,
    Revalidated,
    ServiceWorker,
    Unknown,
}

/// Transfer and cache information for one response.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct TransferInfo {
    pub encoded_bytes: Option<u64>,
    pub decoded_bytes: Option<u64>,
    pub from_cache: bool,
    pub cache_status: CacheStatus,
}

/// Metadata for a bounded body preview. Body bytes are deliberately not part
/// of the frozen bundle because arbitrary response bodies are not safe to
/// export by default.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct BodyPreviewMetadata {
    pub available: bool,
    pub mime_type: Option<String>,
    pub captured_bytes: u64,
    pub total_bytes: Option<u64>,
    pub truncated: bool,
    pub redacted: bool,
}

/// The timing phases used by Chrome-style waterfall viewers.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NetworkPhase {
    Queueing,
    Proxy,
    Dns,
    Connect,
    Tls,
    Request,
    Upload,
    Response,
    Download,
    Decompression,
    Cache,
}

/// Whether a transport could observe one timing/capability value.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "state", content = "reason")]
pub enum CapabilityStatus {
    Available,
    Unavailable(String),
    NotObserved,
    NotApplicable,
}

/// A phase duration with an explicit unavailable marker.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PhaseTiming {
    pub duration_us: Option<u64>,
    pub capability: CapabilityStatus,
}

/// Monotonic request timing and phase breakdown.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct NetworkTiming {
    pub start_us: u64,
    pub end_us: Option<u64>,
    pub phases: BTreeMap<NetworkPhase, PhaseTiming>,
}

impl NetworkTiming {
    pub fn new(start_us: u64) -> Self {
        Self {
            start_us,
            ..Self::default()
        }
    }

    pub fn with_end(mut self, end_us: u64) -> Self {
        self.end_us = Some(end_us);
        self
    }

    pub fn set_phase(&mut self, phase: NetworkPhase, timing: PhaseTiming) {
        self.phases.insert(phase, timing);
    }

    pub fn unavailable_phase(&mut self, phase: NetworkPhase, reason: impl Into<String>) {
        self.set_phase(
            phase,
            PhaseTiming {
                duration_us: None,
                capability: CapabilityStatus::Unavailable(reason.into()),
            },
        );
    }
}

/// Whether a request was observed from start to finish.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationStatus {
    #[default]
    Complete,
    PartiallyObserved,
    Cancelled,
}

/// A safe, structured error captured from a transport.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct NetworkError {
    pub kind: NetworkErrorKind,
    pub message: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NetworkErrorKind {
    Cancelled,
    Dns,
    Connect,
    Tls,
    Timeout,
    Protocol,
    Transport,
    Unknown,
}

/// The result fields recorded for one request.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct NetworkRequest {
    pub id: u64,
    pub initiator: Option<Initiator>,
    pub resource_type: NetworkResourceType,
    pub method: String,
    pub url: String,
    pub request_headers: Vec<Header>,
    pub response_headers: Vec<Header>,
    pub status: Option<u16>,
    pub transfer: TransferInfo,
    pub timing: NetworkTiming,
    pub body_preview: BodyPreviewMetadata,
    pub error: Option<NetworkError>,
    pub capabilities: BTreeMap<NetworkPhase, CapabilityStatus>,
    pub observation: ObservationStatus,
}

impl NetworkRequest {
    pub fn new(id: u64, start: NetworkRequestStart) -> Self {
        Self {
            id,
            initiator: start.initiator,
            resource_type: start.resource_type,
            method: start.method,
            url: start.url,
            request_headers: start.headers,
            timing: NetworkTiming::new(start.started_at_us),
            ..Self::default()
        }
    }

    /// Returns the representation that is safe to place into a frozen bundle.
    pub fn redacted(mut self) -> Self {
        self.url = redact_url(&self.url);
        self.method = self.method.trim().to_ascii_uppercase();
        redact_headers(&mut self.request_headers);
        redact_headers(&mut self.response_headers);
        if let Some(initiator) = &mut self.initiator {
            initiator.kind = redact_text(&initiator.kind);
            initiator.source = initiator.source.take().map(|source| redact_url(&source));
        }
        if let Some(error) = &mut self.error {
            error.message = redact_text(&error.message);
        }
        for status in self.capabilities.values_mut() {
            if let CapabilityStatus::Unavailable(reason) = status {
                *reason = redact_text(reason);
            }
        }
        for timing in self.timing.phases.values_mut() {
            if let CapabilityStatus::Unavailable(reason) = &mut timing.capability {
                *reason = redact_text(reason);
            }
        }
        self
    }
}

/// Input supplied by a transport adapter when a request starts.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct NetworkRequestStart {
    pub initiator: Option<Initiator>,
    pub resource_type: NetworkResourceType,
    pub method: String,
    pub url: String,
    pub headers: Vec<Header>,
    pub started_at_us: u64,
}

impl NetworkRequestStart {
    pub fn new(method: impl Into<String>, url: impl Into<String>, started_at_us: u64) -> Self {
        Self {
            method: method.into(),
            url: url.into(),
            started_at_us,
            ..Self::default()
        }
    }
}

/// A handle identifying an in-flight recorder entry.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct NetworkRequestHandle(u64);

impl NetworkRequestHandle {
    pub fn id(self) -> u64 {
        self.0
    }
}

/// Indicates whether the capture was able to observe network activity.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NetworkCaptureStatus {
    #[default]
    Complete,
    PartiallyObserved,
}

/// Ordered network records belonging to one frozen capture.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct NetworkWaterfall {
    pub status: NetworkCaptureStatus,
    pub requests: Vec<NetworkRequest>,
}

impl NetworkWaterfall {
    pub fn new(requests: Vec<NetworkRequest>) -> Self {
        Self {
            status: NetworkCaptureStatus::Complete,
            requests,
        }
        .ordered_and_redacted()
    }

    pub fn ordered_and_redacted(mut self) -> Self {
        for request in &mut self.requests {
            let redacted = std::mem::take(request).redacted();
            *request = redacted;
        }
        self.requests
            .sort_by_key(|request| (request.timing.start_us, request.id));
        if self
            .requests
            .iter()
            .any(|request| request.observation != ObservationStatus::Complete)
        {
            self.status = NetworkCaptureStatus::PartiallyObserved;
        }
        self
    }

    pub fn requests(&self) -> &[NetworkRequest] {
        &self.requests
    }
}

/// Immutable frame data that can be written to disk or sent to an inspector.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct FrozenCaptureBundle {
    schema_version: u32,
    network: NetworkWaterfall,
}

impl FrozenCaptureBundle {
    pub fn new(network: NetworkWaterfall) -> Self {
        Self {
            schema_version: FROZEN_CAPTURE_SCHEMA_VERSION,
            network: network.ordered_and_redacted(),
        }
    }

    pub fn network(&self) -> &NetworkWaterfall {
        &self.network
    }

    pub fn schema_version(&self) -> u32 {
        self.schema_version
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    pub fn from_json(json: &str) -> Result<Self, FrozenCaptureBundleError> {
        let bundle: Self = serde_json::from_str(json).map_err(FrozenCaptureBundleError::Json)?;
        if bundle.schema_version != FROZEN_CAPTURE_SCHEMA_VERSION {
            return Err(FrozenCaptureBundleError::UnsupportedSchemaVersion(
                bundle.schema_version,
            ));
        }
        Ok(Self::new(bundle.network))
    }
}

#[derive(Debug)]
pub enum FrozenCaptureBundleError {
    Json(serde_json::Error),
    UnsupportedSchemaVersion(u32),
}

impl fmt::Display for FrozenCaptureBundleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(error) => write!(formatter, "invalid frozen capture bundle: {error}"),
            Self::UnsupportedSchemaVersion(version) => {
                write!(
                    formatter,
                    "unsupported frozen capture schema version {version}"
                )
            }
        }
    }
}

impl std::error::Error for FrozenCaptureBundleError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Json(error) => Some(error),
            Self::UnsupportedSchemaVersion(_) => None,
        }
    }
}

/// Capture-only recorder. Construct it with NetworkRecorder::disabled in the
/// normal path and with NetworkRecorder::enabled only at a safe capture
/// boundary.
#[derive(Debug)]
pub struct NetworkRecorder {
    enabled: bool,
    next_id: u64,
    requests: BTreeMap<u64, NetworkRequest>,
    partially_observed: bool,
}

impl NetworkRecorder {
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            next_id: 0,
            requests: BTreeMap::new(),
            partially_observed: false,
        }
    }

    pub fn enabled() -> Self {
        Self {
            enabled: true,
            next_id: 0,
            requests: BTreeMap::new(),
            partially_observed: false,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn begin(&mut self, start: NetworkRequestStart) -> Option<NetworkRequestHandle> {
        if !self.enabled {
            return None;
        }
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        self.requests.insert(id, NetworkRequest::new(id, start));
        Some(NetworkRequestHandle(id))
    }

    pub fn finish(&mut self, handle: NetworkRequestHandle, result: NetworkRequestResult) -> bool {
        let Some(request) = self.requests.get_mut(&handle.0) else {
            return false;
        };
        request.status = result.status;
        request.response_headers = result.response_headers;
        request.transfer = result.transfer;
        request.timing.end_us = result.finished_at_us;
        request.body_preview = result.body_preview;
        request.error = result.error;
        request.capabilities = result.capabilities;
        request.observation = result.observation;
        self.partially_observed |= request.observation != ObservationStatus::Complete;
        true
    }

    pub fn mark_partially_observed(&mut self, handle: NetworkRequestHandle) -> bool {
        let Some(request) = self.requests.get_mut(&handle.0) else {
            return false;
        };
        request.observation = ObservationStatus::PartiallyObserved;
        self.partially_observed = true;
        true
    }

    pub fn freeze(self) -> NetworkWaterfall {
        let waterfall = NetworkWaterfall {
            status: if self.partially_observed {
                NetworkCaptureStatus::PartiallyObserved
            } else {
                NetworkCaptureStatus::Complete
            },
            requests: self.requests.into_values().collect(),
        };
        waterfall.ordered_and_redacted()
    }
}

/// Result fields supplied when an in-flight request completes.
#[derive(Clone, Debug, Default)]
pub struct NetworkRequestResult {
    pub status: Option<u16>,
    pub response_headers: Vec<Header>,
    pub transfer: TransferInfo,
    pub finished_at_us: Option<u64>,
    pub body_preview: BodyPreviewMetadata,
    pub error: Option<NetworkError>,
    pub capabilities: BTreeMap<NetworkPhase, CapabilityStatus>,
    pub observation: ObservationStatus,
}

/// A sanitized request passed to an explicitly selected replay transport.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplayRequest {
    pub id: u64,
    pub method: String,
    pub url: String,
    pub headers: Vec<Header>,
}

/// Response metadata returned by a replay transport. The frozen capture does
/// not contain arbitrary body bytes; a transport may supply safe fixture bytes
/// from a separately controlled fixture store if the application needs them.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ReplayResponse {
    pub status: Option<u16>,
    pub headers: Vec<Header>,
    pub transfer: TransferInfo,
    pub body_preview: BodyPreviewMetadata,
}

/// The only operation a deterministic replay adapter needs from a transport.
pub trait ReplayTransport {
    type Error: fmt::Display;

    fn send(&self, request: &ReplayRequest) -> Result<ReplayResponse, Self::Error>;
}

/// A replay transport backed solely by the frozen network records.
#[derive(Clone, Debug)]
pub struct RecordedReplayTransport {
    records: BTreeMap<u64, NetworkRequest>,
}

impl RecordedReplayTransport {
    pub fn new(bundle: &FrozenCaptureBundle) -> Self {
        Self {
            records: bundle
                .network
                .requests
                .iter()
                .map(|request| (request.id, request.clone()))
                .collect(),
        }
    }
}

impl ReplayTransport for RecordedReplayTransport {
    type Error = ReplayError;

    fn send(&self, request: &ReplayRequest) -> Result<ReplayResponse, Self::Error> {
        let record = self
            .records
            .get(&request.id)
            .ok_or(ReplayError::UnknownRequest(request.id))?;
        if record.method != request.method
            || record.url != request.url
            || record.request_headers != request.headers
        {
            return Err(ReplayError::RequestMismatch(request.id));
        }
        Ok(ReplayResponse {
            status: record.status,
            headers: record.response_headers.clone(),
            transfer: record.transfer.clone(),
            body_preview: record.body_preview.clone(),
        })
    }
}

/// Errors returned by deterministic replay.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReplayError {
    UnknownRequest(u64),
    RequestMismatch(u64),
    Transport(String),
}

impl fmt::Display for ReplayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownRequest(id) => write!(formatter, "unknown replay request {id}"),
            Self::RequestMismatch(id) => write!(
                formatter,
                "replay request {id} does not match the frozen request"
            ),
            Self::Transport(message) => write!(formatter, "replay transport failed: {message}"),
        }
    }
}

/// Builds a replay request from a frozen record and delegates to a supplied
/// fake or application-owned transport. The default recorded transport is
/// side-effect free and never touches live application state.
pub fn replay_request<T: ReplayTransport>(
    bundle: &FrozenCaptureBundle,
    request_id: u64,
    transport: &T,
) -> Result<ReplayResponse, ReplayError> {
    let request = bundle
        .network
        .requests
        .iter()
        .find(|request| request.id == request_id)
        .ok_or(ReplayError::UnknownRequest(request_id))?;
    let replay_request = ReplayRequest {
        id: request.id,
        method: request.method.clone(),
        url: request.url.clone(),
        headers: request.request_headers.clone(),
    };
    transport
        .send(&replay_request)
        .map_err(|error| ReplayError::Transport(error.to_string()))
}

fn redact_headers(headers: &mut [Header]) {
    for header in headers.iter_mut() {
        let normalized_name = header.name.trim().to_ascii_lowercase();
        header.name = header.name.trim().to_string();
        if is_sensitive_name(&normalized_name) {
            header.value = REDACTED_VALUE.to_string();
        } else if matches!(normalized_name.as_str(), "referer" | "origin") {
            header.value = redact_url(&header.value);
        } else {
            header.value = redact_text(&header.value);
        }
    }
    headers.sort_by_key(|header| {
        (
            header.name.to_ascii_lowercase(),
            header.name.clone(),
            header.value.clone(),
        )
    });
}

fn is_sensitive_name(name: &str) -> bool {
    matches!(
        name,
        "authorization"
            | "proxy-authorization"
            | "cookie"
            | "set-cookie"
            | "x-api-key"
            | "api-key"
            | "www-authenticate"
    ) || [
        "token",
        "secret",
        "password",
        "credential",
        "auth",
        "api_key",
        "apikey",
        "access_key",
        "access_token",
        "refresh_token",
        "client_secret",
    ]
    .iter()
    .any(|part| name.contains(part))
}

fn redact_url(value: &str) -> String {
    let without_fragment = value.split('#').next().unwrap_or(value);
    let mut redacted = without_fragment.to_string();
    if let Some(scheme_end) = redacted.find("://") {
        let authority_start = scheme_end + 3;
        let authority_end = redacted[authority_start..]
            .find(['/', '?'])
            .map_or(redacted.len(), |offset| authority_start + offset);
        if let Some(user_info_end) = redacted[authority_start..authority_end].find('@') {
            redacted.replace_range(
                authority_start..authority_start + user_info_end + 1,
                "<redacted>@",
            );
        }
    }
    let Some(query_start) = redacted.find('?') else {
        return redacted;
    };
    let (prefix, query) = redacted.split_at(query_start + 1);
    let redacted_query = query
        .split('&')
        .map(|parameter| {
            let Some((key, _value)) = parameter.split_once('=') else {
                return parameter.to_string();
            };
            if is_sensitive_name(&key.to_ascii_lowercase()) {
                format!("{key}={REDACTED_VALUE}")
            } else {
                parameter.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("&");
    format!("{prefix}{redacted_query}")
}

fn redact_text(value: &str) -> String {
    let mut redacted = value.to_string();
    for prefix in ["Bearer ", "Basic "] {
        let mut search_start = 0;
        while let Some(relative_start) = redacted[search_start..].find(prefix) {
            let start = search_start + relative_start + prefix.len();
            let end = redacted[start..]
                .find(char::is_whitespace)
                .map_or(redacted.len(), |offset| start + offset);
            redacted.replace_range(start..end, REDACTED_VALUE);
            search_start = start + REDACTED_VALUE.len();
        }
    }
    redacted = redacted
        .split_whitespace()
        .map(|word| {
            let Some((key, _value)) = word.split_once('=') else {
                return word.to_string();
            };
            let key = key.trim_matches(|character: char| {
                !character.is_ascii_alphanumeric() && character != '_'
            });
            if is_sensitive_name(&key.to_ascii_lowercase()) {
                format!("{key}={REDACTED_VALUE}")
            } else {
                word.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ");
    redact_url(&redacted)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn start(id: u64, started_at_us: u64, url: &str) -> NetworkRequest {
        NetworkRequest::new(id, NetworkRequestStart::new("get", url, started_at_us))
    }

    #[test]
    fn frozen_bundle_redacts_credentials_and_sorts_headers() {
        let mut request = start(
            2,
            20,
            "https://alice:secret@example.test/data?z=1&token=abc#fragment",
        );
        request.initiator = Some(Initiator::new(
            "asset password=secret",
            Some("https://user:secret@example.test/source".to_string()),
            None,
        ));
        request.error = Some(NetworkError {
            kind: NetworkErrorKind::Transport,
            message: "request failed with token=secret".to_string(),
        });
        request.request_headers = vec![
            Header::new("X-Zeta", "last"),
            Header::new("Authorization", "Bearer very-secret"),
            Header::new("Cookie", "session=secret"),
            Header::new("Accept", "application/json"),
        ];
        let mut earlier = start(1, 10, "https://example.test/first");
        earlier.response_headers = vec![Header::new("Set-Cookie", "session=secret")];
        let bundle = FrozenCaptureBundle::new(NetworkWaterfall::new(vec![request, earlier]));
        let json = bundle.to_json().expect("bundle should serialize");
        assert!(!json.contains("secret"));
        assert!(!json.contains("very-secret"));
        assert!(json.contains("https://<redacted>@example.test/data?z=1&token=<redacted>"));
        assert_eq!(bundle.network().requests[0].id, 1);
        assert_eq!(
            bundle.network().requests[1].request_headers[0].name,
            "Accept"
        );
    }

    #[test]
    fn phase_capabilities_preserve_unavailable_markers() {
        let mut timing = NetworkTiming::new(10);
        timing.unavailable_phase(NetworkPhase::Tls, "client does not expose TLS timing");
        let mut request = start(0, 10, "https://example.test");
        request.timing = timing;
        request
            .capabilities
            .insert(NetworkPhase::Dns, CapabilityStatus::NotObserved);
        let bundle = FrozenCaptureBundle::new(NetworkWaterfall::new(vec![request]));
        let round_trip = FrozenCaptureBundle::from_json(&bundle.to_json().expect("serialize"))
            .expect("deserialize");
        assert_eq!(round_trip, bundle);
        assert_eq!(
            round_trip.network().requests[0].timing.phases[&NetworkPhase::Tls].capability,
            CapabilityStatus::Unavailable("client does not expose TLS timing".to_string())
        );
    }

    #[test]
    fn bundle_rejects_unknown_schema_versions() {
        let error = FrozenCaptureBundle::from_json(
            r#"{"schema_version":99,"network":{"status":"complete","requests":[]}}"#,
        )
        .expect_err("unknown schema versions must not be silently accepted");
        assert!(matches!(
            error,
            FrozenCaptureBundleError::UnsupportedSchemaVersion(99)
        ));
    }

    #[test]
    fn disabled_recorder_does_not_create_request_entries() {
        let mut recorder = NetworkRecorder::disabled();
        assert!(recorder
            .begin(NetworkRequestStart::new("GET", "https://example.test", 0))
            .is_none());
        assert!(recorder.freeze().requests.is_empty());
    }

    #[test]
    fn recorder_marks_unfinished_request_as_partially_observed() {
        let mut recorder = NetworkRecorder::enabled();
        let handle = recorder
            .begin(NetworkRequestStart::new("GET", "https://example.test", 42))
            .expect("enabled recorder should return a handle");
        assert!(recorder.mark_partially_observed(handle));
        let waterfall = recorder.freeze();
        assert_eq!(waterfall.status, NetworkCaptureStatus::PartiallyObserved);
        assert_eq!(
            waterfall.requests[0].observation,
            ObservationStatus::PartiallyObserved
        );
    }

    #[test]
    fn recorder_finishes_request_with_response_and_diagnostics() {
        let mut recorder = NetworkRecorder::enabled();
        let handle = recorder
            .begin(NetworkRequestStart::new(
                "POST",
                "https://example.test/upload",
                10,
            ))
            .expect("enabled recorder should return a handle");
        let mut capabilities = BTreeMap::new();
        capabilities.insert(NetworkPhase::Dns, CapabilityStatus::Available);
        assert!(recorder.finish(
            handle,
            NetworkRequestResult {
                status: Some(201),
                response_headers: vec![Header::new("Content-Type", "application/json")],
                transfer: TransferInfo {
                    encoded_bytes: Some(12),
                    decoded_bytes: Some(20),
                    from_cache: false,
                    cache_status: CacheStatus::None,
                },
                finished_at_us: Some(25),
                body_preview: BodyPreviewMetadata {
                    available: true,
                    mime_type: Some("application/json".to_string()),
                    captured_bytes: 20,
                    total_bytes: Some(40),
                    truncated: true,
                    redacted: true,
                },
                error: None,
                capabilities,
                observation: ObservationStatus::Complete,
            },
        ));
        let waterfall = recorder.freeze();
        let request = &waterfall.requests[0];
        assert_eq!(request.status, Some(201));
        assert_eq!(request.timing.end_us, Some(25));
        assert_eq!(request.transfer.encoded_bytes, Some(12));
        assert!(request.body_preview.truncated);
        assert_eq!(
            request.capabilities[&NetworkPhase::Dns],
            CapabilityStatus::Available
        );
    }

    #[test]
    fn recorded_replay_is_deterministic_and_uses_sanitized_request() {
        let mut request = start(7, 0, "https://example.test/data?password=one");
        request.status = Some(200);
        request.request_headers = vec![Header::new("Authorization", "Bearer one")];
        request.body_preview = BodyPreviewMetadata {
            available: true,
            mime_type: Some("application/json".to_string()),
            captured_bytes: 12,
            total_bytes: Some(12),
            truncated: false,
            redacted: true,
        };
        let bundle = FrozenCaptureBundle::new(NetworkWaterfall::new(vec![request]));
        let transport = RecordedReplayTransport::new(&bundle);
        let response = replay_request(&bundle, 7, &transport).expect("fixture replay should work");
        assert_eq!(response.status, Some(200));
        assert!(replay_request(&bundle, 8, &transport).is_err());
    }

    #[test]
    fn fake_transport_receives_only_the_frozen_safe_request() {
        #[derive(Debug)]
        struct FakeTransport {
            request: std::sync::Mutex<Option<ReplayRequest>>,
        }

        impl ReplayTransport for FakeTransport {
            type Error = std::convert::Infallible;

            fn send(&self, request: &ReplayRequest) -> Result<ReplayResponse, Self::Error> {
                *self
                    .request
                    .lock()
                    .expect("fake transport lock should not be poisoned") = Some(request.clone());
                Ok(ReplayResponse {
                    status: Some(204),
                    ..ReplayResponse::default()
                })
            }
        }

        let mut request = start(1, 0, "https://example.test/?api_key=secret");
        request.request_headers = vec![Header::new("X-Api-Key", "secret")];
        let bundle = FrozenCaptureBundle::new(NetworkWaterfall::new(vec![request]));
        let fake = FakeTransport {
            request: std::sync::Mutex::new(None),
        };
        let response = replay_request(&bundle, 1, &fake).expect("fake replay should work");
        assert_eq!(response.status, Some(204));
        let received = fake
            .request
            .into_inner()
            .expect("fake transport should see a request")
            .expect("fake transport should record its request");
        assert!(!received.url.contains("secret"));
        assert!(!received.headers[0].value.contains("secret"));
    }
}
