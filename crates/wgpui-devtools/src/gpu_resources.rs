//! Capture-only descriptions of native GPU resources.
//!
//! The registry deliberately contains descriptions rather than `wgpu` handles.
//! This keeps the devtools crate backend-neutral and lets the native renderer
//! record resources without making the core crate depend on a graphics API.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};

/// Stable identifier for a resource within one capture.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResourceId(u64);

impl ResourceId {
    /// The value returned when GPU capture is disabled.
    pub const INVALID: Self = Self(0);

    /// Returns the capture-local numeric identifier.
    pub const fn raw(self) -> u64 {
        self.0
    }

    /// Reconstructs an id produced by a native backend adapter.
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
}

/// Native resource category.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ResourceKind {
    Buffer,
    Texture,
    QuerySet,
}

/// Renderer-owned role of a resource.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ResourceRole {
    PrimitiveBuffer,
    IndirectArguments,
    IndirectCount,
    Visibility,
    SlotTable,
    AtlasPage,
    LayerTexture,
    TileTexture,
    Surface,
    Staging,
    Readback,
    Query,
    Uniform,
    Other,
}

/// Format exposed in a capture without depending on a graphics API enum.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceFormat(pub String);

impl ResourceFormat {
    /// Creates a format name.
    pub fn new(format: impl Into<String>) -> Self {
        Self(format.into())
    }
}

/// Extent of a texture or query-addressable resource.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ResourceDimensions {
    pub width: u32,
    pub height: u32,
    pub depth_or_array_layers: u32,
}

/// Byte range touched by an upload or readback.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ByteRange {
    pub offset: u64,
    pub size: u64,
}

/// Texel region touched by a texture upload.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct TextureRegion {
    pub origin: [u32; 3],
    pub size: [u32; 3],
}

/// One upload or readback event.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TransferKind {
    Upload,
    Readback,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UploadRecord {
    pub frame: u64,
    pub kind: TransferKind,
    pub bytes: u64,
    pub byte_range: Option<ByteRange>,
    pub texture_region: Option<TextureRegion>,
}

/// One eviction event.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct EvictionRecord {
    pub frame: u64,
}

/// Description of a resource at creation time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceDescriptor {
    pub kind: ResourceKind,
    pub role: ResourceRole,
    pub label: String,
    pub format: Option<ResourceFormat>,
    pub dimensions: Option<ResourceDimensions>,
    pub byte_size: u64,
    pub usage: u64,
    pub generation: u64,
}

/// Complete resource history visible to a capture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceRecord {
    pub id: ResourceId,
    pub descriptor: ResourceDescriptor,
    pub created_frame: u64,
    pub last_use_frame: Option<u64>,
    pub resident: bool,
    pub uploads: Vec<UploadRecord>,
    pub evictions: Vec<EvictionRecord>,
}

/// Snapshot returned when a capture ends.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CaptureSnapshot {
    pub resources: Vec<ResourceRecord>,
}

#[derive(Default)]
struct State {
    frame: u64,
    next_id: u64,
    resources: BTreeMap<ResourceId, ResourceRecord>,
}

static STATE: LazyLock<Mutex<State>> = LazyLock::new(|| Mutex::new(State::default()));
static ENABLED: AtomicBool = AtomicBool::new(false);
static CURRENT_FRAME: AtomicU64 = AtomicU64::new(0);

fn with_state<T>(function: impl FnOnce(&mut State) -> T) -> T {
    let mut state = STATE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    function(&mut state)
}

pub(crate) fn begin(include_gpu: bool) {
    ENABLED.store(false, Ordering::Release);
    CURRENT_FRAME.store(0, Ordering::Release);
    with_state(|state| {
        state.frame = 0;
        state.next_id = 1;
        state.resources.clear();
    });
    ENABLED.store(include_gpu, Ordering::Release);
}

pub(crate) fn end() -> CaptureSnapshot {
    ENABLED.store(false, Ordering::Release);
    with_state(|state| CaptureSnapshot {
        resources: state.resources.values().cloned().collect(),
    })
}

/// Returns whether native GPU resource capture is enabled.
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Acquire)
}

/// Sets the current frame for subsequent events.
pub fn set_frame(frame: u64) {
    if !enabled() {
        return;
    }
    CURRENT_FRAME.store(frame, Ordering::Release);
    with_state(|state| state.frame = frame);
}

/// Returns the frame assigned to subsequent resource events.
pub fn current_frame() -> u64 {
    CURRENT_FRAME.load(Ordering::Acquire)
}

/// Registers a native resource, or [`ResourceId::INVALID`] outside GPU capture.
pub fn register(descriptor: ResourceDescriptor) -> ResourceId {
    if !enabled() {
        return ResourceId::INVALID;
    }
    with_state(|state| {
        let id = ResourceId(state.next_id);
        state.next_id = state.next_id.saturating_add(1);
        state.resources.insert(
            id,
            ResourceRecord {
                id,
                descriptor,
                created_frame: state.frame,
                last_use_frame: None,
                resident: true,
                uploads: Vec::new(),
                evictions: Vec::new(),
            },
        );
        id
    })
}

/// Marks a resource as used by a frame.
pub fn mark_used(id: ResourceId, frame: u64) {
    if !enabled() {
        return;
    }
    with_state(|state| {
        if let Some(resource) = state.resources.get_mut(&id) {
            resource.last_use_frame = Some(frame);
        }
    });
}

/// Records a buffer upload or readback.
pub fn record_buffer_upload(id: ResourceId, range: ByteRange, frame: u64) {
    if !enabled() {
        return;
    }
    with_state(|state| {
        if let Some(resource) = state.resources.get_mut(&id) {
            resource.uploads.push(UploadRecord {
                frame,
                kind: TransferKind::Upload,
                bytes: range.size,
                byte_range: Some(range),
                texture_region: None,
            });
        }
    });
}

/// Records bytes copied into a CPU-readable staging buffer.
pub fn record_buffer_readback(id: ResourceId, range: ByteRange, frame: u64) {
    if !enabled() {
        return;
    }
    with_state(|state| {
        if let Some(resource) = state.resources.get_mut(&id) {
            resource.uploads.push(UploadRecord {
                frame,
                kind: TransferKind::Readback,
                bytes: range.size,
                byte_range: Some(range),
                texture_region: None,
            });
        }
    });
}

/// Records a texture upload.
pub fn record_texture_upload(id: ResourceId, region: TextureRegion, bytes: u64, frame: u64) {
    if !enabled() {
        return;
    }
    with_state(|state| {
        if let Some(resource) = state.resources.get_mut(&id) {
            resource.uploads.push(UploadRecord {
                frame,
                kind: TransferKind::Upload,
                bytes,
                byte_range: None,
                texture_region: Some(region),
            });
        }
    });
}

/// Marks a resource non-resident while retaining its history.
pub fn evict(id: ResourceId, frame: u64) {
    if !enabled() {
        return;
    }
    with_state(|state| {
        if let Some(resource) = state.resources.get_mut(&id) {
            resource.resident = false;
            resource.evictions.push(EvictionRecord { frame });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::{self, CaptureRequest};
    use std::sync::Mutex;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn descriptor(role: ResourceRole) -> ResourceDescriptor {
        ResourceDescriptor {
            kind: ResourceKind::Buffer,
            role,
            label: "test".to_string(),
            format: None,
            dimensions: None,
            byte_size: 64,
            usage: 3,
            generation: 2,
        }
    }

    #[test]
    fn gpu_resources_are_not_recorded_outside_a_gpu_capture() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let id = register(descriptor(ResourceRole::PrimitiveBuffer));
        assert_eq!(id, ResourceId::INVALID);
        assert!(!enabled());
    }

    #[test]
    fn capture_exposes_lifetime_upload_and_use_metadata() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        assert!(capture::start(CaptureRequest { include_gpu: true }));
        set_frame(7);
        let id = register(descriptor(ResourceRole::IndirectArguments));
        record_buffer_upload(
            id,
            ByteRange {
                offset: 8,
                size: 16,
            },
            8,
        );
        mark_used(id, 9);
        evict(id, 10);
        let snapshot = capture::stop().expect("the capture was started");
        let resource = snapshot.resources.first().expect("one resource");
        assert_eq!(resource.created_frame, 7);
        assert_eq!(resource.descriptor.generation, 2);
        assert_eq!(resource.last_use_frame, Some(9));
        assert!(!resource.resident);
        assert_eq!(
            resource.uploads[0].byte_range,
            Some(ByteRange {
                offset: 8,
                size: 16
            })
        );
        assert_eq!(resource.evictions, vec![EvictionRecord { frame: 10 }]);
    }

    #[test]
    fn cpu_only_capture_does_not_enable_gpu_records() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        assert!(capture::start(CaptureRequest { include_gpu: false }));
        assert!(!enabled());
        assert_eq!(
            register(descriptor(ResourceRole::Query)),
            ResourceId::INVALID
        );
        let snapshot = capture::stop().expect("the capture was started");
        assert!(snapshot.resources.is_empty());
    }
}
