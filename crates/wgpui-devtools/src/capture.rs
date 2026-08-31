//! Versioned, renderer-independent capture data for external consumers.

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
