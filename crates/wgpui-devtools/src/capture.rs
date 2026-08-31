//! Capture state and the safe, immutable data boundary used by exporters.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

pub const CAPTURE_SCHEMA_VERSION: u16 = 1;
pub const DEFAULT_MAX_CAPTURE_BYTES: usize = 16 * 1024 * 1024;
pub const DEFAULT_MAX_RESOURCE_READBACK_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub struct ResourceId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceKind {
    Buffer,
    Texture,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceSnapshot {
    pub id: ResourceId,
    pub kind: ResourceKind,
    pub label: String,
    pub bytes: Vec<u8>,
}

impl ResourceSnapshot {
    pub fn new(
        id: ResourceId,
        kind: ResourceKind,
        label: impl Into<String>,
        bytes: Vec<u8>,
    ) -> Self {
        Self {
            id,
            kind,
            label: label.into(),
            bytes,
        }
    }

    pub fn byte_length(&self) -> usize {
        self.bytes.len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceReadback {
    pub id: ResourceId,
    pub offset: u64,
    pub total_length: u64,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrozenCapture {
    schema_version: u16,
    capture_id: u64,
    presented_frame: Option<u64>,
    snapshot: Vec<u8>,
    resources: Vec<ResourceSnapshot>,
}

impl FrozenCapture {
    pub fn new(
        capture_id: u64,
        presented_frame: Option<u64>,
        snapshot: Vec<u8>,
    ) -> Result<Self, CaptureError> {
        let capture = Self {
            schema_version: CAPTURE_SCHEMA_VERSION,
            capture_id,
            presented_frame,
            snapshot,
            resources: Vec::new(),
        };
        capture.validate(DEFAULT_MAX_CAPTURE_BYTES)
    }

    pub fn with_resources(
        capture_id: u64,
        presented_frame: Option<u64>,
        snapshot: Vec<u8>,
        resources: impl IntoIterator<Item = ResourceSnapshot>,
    ) -> Result<Self, CaptureError> {
        let mut resource_map = Vec::new();
        for resource in resources {
            if resource_map
                .iter()
                .any(|existing: &ResourceSnapshot| existing.id == resource.id)
            {
                return Err(CaptureError::DuplicateResource);
            }
            resource_map.push(resource);
        }
        resource_map.sort_unstable_by_key(|resource| resource.id);
        let capture = Self {
            schema_version: CAPTURE_SCHEMA_VERSION,
            capture_id,
            presented_frame,
            snapshot,
            resources: resource_map,
        };
        capture.validate(DEFAULT_MAX_CAPTURE_BYTES)
    }

    pub fn schema_version(&self) -> u16 {
        self.schema_version
    }
    pub fn capture_id(&self) -> u64 {
        self.capture_id
    }
    pub fn presented_frame(&self) -> Option<u64> {
        self.presented_frame
    }
    pub fn snapshot(&self) -> &[u8] {
        &self.snapshot
    }
    pub fn resources(&self) -> impl Iterator<Item = &ResourceSnapshot> {
        self.resources.iter()
    }

    pub fn resource(&self, id: ResourceId) -> Option<&ResourceSnapshot> {
        self.resources.iter().find(|resource| resource.id == id)
    }

    pub fn read_resource(
        &self,
        id: ResourceId,
        offset: u64,
        length: usize,
    ) -> Result<ResourceReadback, CaptureError> {
        if length > DEFAULT_MAX_RESOURCE_READBACK_BYTES {
            return Err(CaptureError::ReadbackTooLarge {
                requested: length,
                maximum: DEFAULT_MAX_RESOURCE_READBACK_BYTES,
            });
        }
        let resource = self.resource(id).ok_or(CaptureError::UnknownResource(id))?;
        let start = usize::try_from(offset).map_err(|_| CaptureError::InvalidReadbackRange)?;
        let end = start
            .checked_add(length)
            .ok_or(CaptureError::InvalidReadbackRange)?;
        if end > resource.bytes.len() {
            return Err(CaptureError::InvalidReadbackRange);
        }
        Ok(ResourceReadback {
            id,
            offset,
            total_length: resource.bytes.len() as u64,
            bytes: resource.bytes[start..end].to_vec(),
        })
    }

    pub fn validate(&self, maximum_bytes: usize) -> Result<Self, CaptureError> {
        if self.schema_version != CAPTURE_SCHEMA_VERSION {
            return Err(CaptureError::UnsupportedSchemaVersion(self.schema_version));
        }
        let mut resource_ids = HashSet::with_capacity(self.resources.len());
        for resource in &self.resources {
            if !resource_ids.insert(resource.id) {
                return Err(CaptureError::DuplicateResource);
            }
        }
        let encoded = serde_json::to_vec(self)
            .map_err(|error| CaptureError::Serialization(error.to_string()))?;
        if encoded.len() > maximum_bytes {
            return Err(CaptureError::CaptureTooLarge {
                actual: encoded.len(),
                maximum: maximum_bytes,
            });
        }
        Ok(self.clone())
    }

    pub fn export_to_writer(&self, writer: &mut impl std::io::Write) -> Result<(), CaptureError> {
        let body = serde_json::to_vec(self)
            .map_err(|error| CaptureError::Serialization(error.to_string()))?;
        let body_length = u64::try_from(body.len()).map_err(|_| CaptureError::CaptureTooLarge {
            actual: body.len(),
            maximum: DEFAULT_MAX_CAPTURE_BYTES,
        })?;
        let total_length = 8usize
            .checked_add(body.len())
            .ok_or(CaptureError::CaptureTooLarge {
                actual: usize::MAX,
                maximum: DEFAULT_MAX_CAPTURE_BYTES,
            })?;
        if total_length > DEFAULT_MAX_CAPTURE_BYTES {
            return Err(CaptureError::CaptureTooLarge {
                actual: total_length,
                maximum: DEFAULT_MAX_CAPTURE_BYTES,
            });
        }
        writer.write_all(b"WGPICAP1")?;
        writer.write_all(&body_length.to_le_bytes())?;
        writer.write_all(&body)?;
        Ok(())
    }

    pub fn export_to_file(&self, path: impl AsRef<std::path::Path>) -> Result<(), CaptureError> {
        let mut file = std::fs::File::create(path)?;
        self.export_to_writer(&mut file)
    }

    pub fn import_from_reader(reader: &mut impl std::io::Read) -> Result<Self, CaptureError> {
        let mut magic = [0; 8];
        reader.read_exact(&mut magic)?;
        if &magic != b"WGPICAP1" {
            return Err(CaptureError::InvalidFile);
        }
        let mut length = [0; 8];
        reader.read_exact(&mut length)?;
        let length =
            usize::try_from(u64::from_le_bytes(length)).map_err(|_| CaptureError::InvalidFile)?;
        if length > DEFAULT_MAX_CAPTURE_BYTES.saturating_sub(8) {
            return Err(CaptureError::CaptureTooLarge {
                actual: length + 8,
                maximum: DEFAULT_MAX_CAPTURE_BYTES,
            });
        }
        let mut body = vec![0; length];
        reader.read_exact(&mut body)?;
        let capture: Self = serde_json::from_slice(&body)
            .map_err(|error| CaptureError::Serialization(error.to_string()))?;
        capture.validate(DEFAULT_MAX_CAPTURE_BYTES)
    }

    pub fn import_from_file(path: impl AsRef<std::path::Path>) -> Result<Self, CaptureError> {
        let mut file = std::fs::File::open(path)?;
        Self::import_from_reader(&mut file)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CaptureError {
    #[error("capture serialization failed: {0}")]
    Serialization(String),
    #[error("capture is {actual} bytes, exceeding the {maximum}-byte limit")]
    CaptureTooLarge { actual: usize, maximum: usize },
    #[error("resource {0:?} is not present in the frozen capture")]
    UnknownResource(ResourceId),
    #[error("resource readback range is invalid")]
    InvalidReadbackRange,
    #[error("resource readback requested {requested} bytes, exceeding the {maximum}-byte limit")]
    ReadbackTooLarge { requested: usize, maximum: usize },
    #[error("a capture contains the same resource ID more than once")]
    DuplicateResource,
    #[error("capture schema version {0} is not supported")]
    UnsupportedSchemaVersion(u16),
    #[error("capture file has an invalid header")]
    InvalidFile,
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("capture is not in a state that can be changed")]
    InvalidState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureState {
    Disabled,
    Armed,
    Collecting,
    StopRequested,
    Frozen,
}

const DISABLED: u8 = 0;
const ARMED: u8 = 1;
const COLLECTING: u8 = 2;
const STOP_REQUESTED: u8 = 3;
const FROZEN: u8 = 4;

#[derive(Debug, Default)]
pub struct CaptureController {
    state: AtomicU8,
    frozen: Mutex<Option<FrozenCapture>>,
}

impl CaptureController {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn state(&self) -> CaptureState {
        match self.state.load(Ordering::Acquire) {
            ARMED => CaptureState::Armed,
            COLLECTING => CaptureState::Collecting,
            STOP_REQUESTED => CaptureState::StopRequested,
            FROZEN => CaptureState::Frozen,
            _ => CaptureState::Disabled,
        }
    }

    pub fn arm(&self) -> Result<(), CaptureError> {
        self.state
            .compare_exchange(DISABLED, ARMED, Ordering::AcqRel, Ordering::Acquire)
            .map(|_| ())
            .map_err(|_| CaptureError::InvalidState)
    }

    pub fn begin_collection(&self) -> bool {
        self.state
            .compare_exchange(ARMED, COLLECTING, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    pub fn request_stop(&self) -> Result<(), CaptureError> {
        loop {
            let state = self.state.load(Ordering::Acquire);
            let next = match state {
                ARMED | COLLECTING => STOP_REQUESTED,
                _ => return Err(CaptureError::InvalidState),
            };
            if self
                .state
                .compare_exchange(state, next, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Ok(());
            }
        }
    }

    pub fn publish_frozen(&self, capture: FrozenCapture) -> Result<(), CaptureError> {
        if !matches!(
            self.state(),
            CaptureState::Collecting | CaptureState::StopRequested
        ) {
            return Err(CaptureError::InvalidState);
        }
        let capture = capture.validate(DEFAULT_MAX_CAPTURE_BYTES)?;
        *self
            .frozen
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(capture);
        self.state.store(FROZEN, Ordering::Release);
        Ok(())
    }

    pub fn snapshot(&self) -> Option<FrozenCapture> {
        self.frozen
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    pub fn reset(&self) {
        *self
            .frozen
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
        self.state.store(DISABLED, Ordering::Release);
    }
}

pub trait CaptureService: Send + Sync {
    fn arm_capture(&self) -> Result<(), CaptureError>;
    fn stop_capture(&self) -> Result<(), CaptureError>;
    fn snapshot(&self) -> Result<Option<FrozenCapture>, CaptureError>;
    fn read_resource(
        &self,
        id: ResourceId,
        offset: u64,
        length: usize,
    ) -> Result<ResourceReadback, CaptureError>;
}

impl CaptureService for CaptureController {
    fn arm_capture(&self) -> Result<(), CaptureError> {
        self.arm()
    }
    fn stop_capture(&self) -> Result<(), CaptureError> {
        self.request_stop()
    }
    fn snapshot(&self) -> Result<Option<FrozenCapture>, CaptureError> {
        Ok(CaptureController::snapshot(self))
    }
    fn read_resource(
        &self,
        id: ResourceId,
        offset: u64,
        length: usize,
    ) -> Result<ResourceReadback, CaptureError> {
        CaptureController::snapshot(self)
            .ok_or(CaptureError::InvalidState)?
            .read_resource(id, offset, length)
    }
}

pub type SharedCaptureService = Arc<dyn CaptureService>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn controller_freezes_only_at_a_safe_boundary() {
        let controller = CaptureController::new();
        assert_eq!(controller.state(), CaptureState::Disabled);
        controller.arm().expect("arming from disabled should work");
        assert!(controller.begin_collection());
        controller
            .request_stop()
            .expect("stop should be accepted while collecting");
        let capture = FrozenCapture::with_resources(
            7,
            Some(3),
            vec![1, 2],
            [ResourceSnapshot::new(
                ResourceId(4),
                ResourceKind::Buffer,
                "arena",
                vec![9, 8, 7],
            )],
        )
        .expect("capture should be valid");
        controller
            .publish_frozen(capture.clone())
            .expect("publishing should work after stop");
        assert_eq!(controller.state(), CaptureState::Frozen);
        assert_eq!(controller.snapshot(), Some(capture));
    }

    #[test]
    fn readback_is_range_checked_and_never_reads_unknown_resources() {
        let capture = FrozenCapture::with_resources(
            1,
            None,
            Vec::new(),
            [ResourceSnapshot::new(
                ResourceId(2),
                ResourceKind::Buffer,
                "data",
                vec![1, 2, 3],
            )],
        )
        .expect("capture should be valid");
        assert_eq!(
            capture
                .read_resource(ResourceId(2), 1, 2)
                .expect("range should be valid")
                .bytes,
            vec![2, 3]
        );
        assert!(matches!(
            capture.read_resource(ResourceId(2), 2, 2),
            Err(CaptureError::InvalidReadbackRange)
        ));
        assert!(matches!(
            capture.read_resource(ResourceId(9), 0, 1),
            Err(CaptureError::UnknownResource(ResourceId(9)))
        ));
    }

    #[test]
    fn file_export_round_trips_and_rejects_bad_magic() {
        let capture = FrozenCapture::with_resources(
            5,
            Some(10),
            vec![4, 5],
            [ResourceSnapshot::new(
                ResourceId(3),
                ResourceKind::Texture,
                "atlas",
                vec![6, 7],
            )],
        )
        .expect("capture should be valid");
        let mut bytes = Vec::new();
        capture
            .export_to_writer(&mut bytes)
            .expect("export should work");
        assert_eq!(
            FrozenCapture::import_from_reader(&mut bytes.as_slice()).expect("export should import"),
            capture
        );
        let mut invalid = b"not-a-capture".as_slice();
        assert!(matches!(
            FrozenCapture::import_from_reader(&mut invalid),
            Err(CaptureError::InvalidFile)
        ));
    }
}
