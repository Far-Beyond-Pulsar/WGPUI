//! `wgpui-devtools` — flamegraph/replay/inspector, moved wholesale behind a
//! small hook trait `wgpui-core`/`wgpui-wgpu` expose into. Pure
//! move-and-decouple, zero behavior change (Phase 7's gate: `wgpui-core`
//! builds and runs with this crate absent entirely).
//! See docs/gpu-native-architecture.md §3.6, §8 Phase 7.
pub mod capture;
#[cfg(feature = "flamegraph")]
pub mod flamegraph;
pub mod gpu_resources;
#[cfg(any(feature = "flamegraph", feature = "render-stats", feature = "perf-ab"))]
pub mod hooks;
#[cfg(feature = "inspector")]
pub mod inspector;
#[cfg(feature = "perf-ab")]
pub mod perf_ab_tests;
#[cfg(any(feature = "flamegraph", feature = "render-stats", feature = "perf-ab"))]
pub mod render_stats;

pub use capture::{
    CaptureRequest, active as capture_active, start as start_capture, stop as stop_capture,
};
pub use gpu_resources::{
    ByteRange, CaptureSnapshot, EvictionRecord, ResourceDescriptor, ResourceDimensions,
    ResourceFormat, ResourceId, ResourceKind, ResourceRecord, ResourceRole, TextureRegion,
    TransferKind, UploadRecord,
};
#[cfg(any(feature = "flamegraph", feature = "render-stats", feature = "perf-ab"))]
pub use hooks::DevtoolsHooks;
#[cfg(feature = "inspector")]
pub use inspector::{ElementInfo, Inspector};
#[cfg(feature = "perf-ab")]
pub use perf_ab_tests::Sample;
#[cfg(any(feature = "flamegraph", feature = "render-stats", feature = "perf-ab"))]
pub use render_stats::{Scope, Snapshot, TimerSnapshot};
