//! `wgpui-devtools` — flamegraph/replay/inspector, moved wholesale behind a
//! small hook trait `wgpui-core`/`wgpui-wgpu` expose into. Pure
//! move-and-decouple, zero behavior change (Phase 7's gate: `wgpui-core`
//! builds and runs with this crate absent entirely).
//! See docs/gpu-native-architecture.md §3.6, §8 Phase 7.
pub mod network;

pub use network::{
    replay_request, BodyPreviewMetadata, CacheStatus, CapabilityStatus, FrozenCaptureBundle,
    FrozenCaptureBundleError, Header, Initiator, NetworkCaptureStatus, NetworkError,
    NetworkErrorKind, NetworkPhase, NetworkRecorder, NetworkRequest, NetworkRequestHandle,
    NetworkRequestResult, NetworkRequestStart, NetworkResourceType, NetworkTiming,
    NetworkWaterfall, ObservationStatus, PhaseTiming, RecordedReplayTransport, ReplayError,
    ReplayRequest, ReplayResponse, ReplayTransport, TransferInfo, FROZEN_CAPTURE_SCHEMA_VERSION,
};

#[cfg(feature = "flamegraph")]
pub mod flamegraph;
#[cfg(any(feature = "flamegraph", feature = "render-stats", feature = "perf-ab"))]
pub mod hooks;
#[cfg(feature = "inspector")]
pub mod inspector;
#[cfg(feature = "perf-ab")]
pub mod perf_ab_tests;
#[cfg(any(feature = "flamegraph", feature = "render-stats", feature = "perf-ab"))]
pub mod render_stats;

#[cfg(any(feature = "flamegraph", feature = "render-stats", feature = "perf-ab"))]
pub use hooks::DevtoolsHooks;
#[cfg(feature = "inspector")]
pub use inspector::{ElementInfo, Inspector};
#[cfg(feature = "perf-ab")]
pub use perf_ab_tests::Sample;
#[cfg(any(feature = "flamegraph", feature = "render-stats", feature = "perf-ab"))]
pub use render_stats::{Scope, Snapshot, TimerSnapshot};
