//! `wgpui-devtools` — flamegraph/replay/inspector, moved wholesale behind a
//! small hook trait `wgpui-core`/`wgpui-wgpu` expose into. Pure
//! move-and-decouple, zero behavior change (Phase 7's gate: `wgpui-core`
//! builds and runs with this crate absent entirely).
//! See docs/gpu-native-architecture.md §3.6, §8 Phase 7.
pub mod capture;
#[cfg(feature = "flamegraph")]
pub mod flamegraph;
#[cfg(any(feature = "flamegraph", feature = "render-stats", feature = "perf-ab"))]
pub mod hooks;
#[cfg(feature = "inspector")]
pub mod inspector;
#[cfg(feature = "perf-ab")]
pub mod perf_ab_tests;
pub mod protocol;
#[cfg(any(feature = "flamegraph", feature = "render-stats", feature = "perf-ab"))]
pub mod render_stats;
pub mod transport;

pub use capture::{
    CAPTURE_SCHEMA_VERSION, CaptureController, CaptureError, CaptureService, CaptureState,
    DEFAULT_MAX_CAPTURE_BYTES, DEFAULT_MAX_RESOURCE_READBACK_BYTES, FrozenCapture, ResourceId,
    ResourceKind, ResourceReadback, ResourceSnapshot,
};
#[cfg(any(feature = "flamegraph", feature = "render-stats", feature = "perf-ab"))]
pub use hooks::DevtoolsHooks;
#[cfg(feature = "inspector")]
pub use inspector::{ElementInfo, Inspector};
#[cfg(feature = "perf-ab")]
pub use perf_ab_tests::Sample;
pub use protocol::{
    BoundedMessageQueue, Capabilities, Capability, ClientMessage, DEFAULT_MAX_MESSAGE_BYTES,
    ErrorCode, ProtocolError, Request, Response, SUPPORTED_PROTOCOL_VERSION, ServerMessage,
    decode_message, encode_message, read_message, write_message,
};
#[cfg(any(feature = "flamegraph", feature = "render-stats", feature = "perf-ab"))]
pub use render_stats::{Scope, Snapshot, TimerSnapshot};
pub use transport::{Endpoint, LocalIpcConfig, LocalIpcError, LocalIpcServer};
