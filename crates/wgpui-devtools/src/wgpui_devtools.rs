//! `wgpui-devtools` — flamegraph/replay/inspector, moved wholesale behind a
//! small hook trait `wgpui-core`/`wgpui-wgpu` expose into. Pure
//! move-and-decouple, zero behavior change (Phase 7's gate: `wgpui-core`
//! builds and runs with this crate absent entirely).
//! See docs/gpu-native-architecture.md §3.6, §8 Phase 7.
pub mod capture;
#[cfg(any(feature = "flamegraph", feature = "render-stats"))]
pub mod flamegraph;
#[cfg(any(feature = "flamegraph", feature = "render-stats"))]
pub use flamegraph::capture::{begin_global_backend_frame, present_global_backend_frame};
#[cfg(any(feature = "flamegraph", feature = "render-stats", feature = "perf-ab"))]
pub mod hooks;
#[cfg(feature = "inspector")]
pub mod inspector;
pub mod memory;
#[cfg(feature = "perf-ab")]
pub mod perf_ab_tests;
pub mod reference_viewer;
pub mod resource_snapshot;
pub mod protocol;
#[cfg(any(feature = "flamegraph", feature = "render-stats", feature = "perf-ab"))]
pub mod render_stats;
pub mod transport;
#[cfg(any(feature = "flamegraph", feature = "render-stats", feature = "perf-ab"))]
pub use hooks::DevtoolsHooks;
#[cfg(feature = "inspector")]
pub use inspector::{ElementInfo, Inspector};
pub use memory::{
    AllocationCategory, AllocationCategorySnapshot, AllocationEntry, AllocationId,
    AllocationRegistry, AllocationSnapshot,
};
#[cfg(feature = "perf-ab")]
pub use perf_ab_tests::Sample;
pub use reference_viewer::{CaptureViewerError, ReferenceViewer};
#[cfg(any(feature = "flamegraph", feature = "render-stats", feature = "perf-ab"))]
pub use render_stats::{Scope, Snapshot, TimerSnapshot};
pub use capture::{
    Availability, CaptureBundle, CaptureConfig, CaptureController, CaptureError, CaptureExport,
    CaptureRecorder, CaptureService, CaptureSnapshot, CaptureState, DEFAULT_MAX_CAPTURE_BYTES,
    DEFAULT_MAX_RESOURCE_READBACK_BYTES, FrameSnapshot, FrozenCapture, RecorderConfig, ResourceId,
    ResourceKind, ResourceReadback, ResourceSnapshot,
};
pub use protocol::{
    BoundedMessageQueue, Capabilities, Capability, ClientMessage, DEFAULT_MAX_MESSAGE_BYTES,
    ErrorCode, ProtocolError, Request, Response, SUPPORTED_PROTOCOL_VERSION, ServerMessage,
    decode_message, encode_message, read_message, write_message,
};
pub use transport::{Endpoint, LocalIpcConfig, LocalIpcError, LocalIpcServer};
pub use resource_snapshot::{
    AtlasPackingSnapshot, AtlasPageRecord, AtlasPlacementRecord, BufferElementType,
    BufferViewSnapshot, ByteRange, IndirectDrawRecord, IndirectDrawSnapshot, RedactionPolicy,
    ResourceSnapshot as TypedResourceSnapshot, SlabAllocationRecord, SlabMapSnapshot, SnapshotError,
    SnapshotHeader, SnapshotLimits, TileOccupancyRecord, TileOccupancySnapshot, TruncationMetadata,
    TypedBufferView, TypedValue, SNAPSHOT_HEADER_BYTES, SNAPSHOT_MAGIC, SNAPSHOT_SCHEMA_VERSION,
};
