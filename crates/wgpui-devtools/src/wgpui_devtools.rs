//! `wgpui-devtools` — flamegraph/replay/inspector, moved wholesale behind a
//! small hook trait `wgpui-core`/`wgpui-wgpu` expose into. Pure
//! move-and-decouple, zero behavior change (Phase 7's gate: `wgpui-core`
//! builds and runs with this crate absent entirely).
//! See docs/gpu-native-architecture.md §3.6, §8 Phase 7.
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
pub use inspector::{
    CaptureState, ElementInfo, ElementQuery, ElementRecord, ElementSelector, Inspector,
    InspectorQuery, InspectorSnapshot, OverlayStats, QueryError, SelectionError, SelectionOverlay,
    SelectionUpdate, SourceLocation, StableElementAddress,
};
#[cfg(feature = "perf-ab")]
pub use perf_ab_tests::Sample;
#[cfg(any(feature = "flamegraph", feature = "render-stats", feature = "perf-ab"))]
pub use render_stats::{Scope, Snapshot, TimerSnapshot};
#[cfg(feature = "inspector")]
pub use wgpui_core::boundary::ScrollRootId;
#[cfg(feature = "inspector")]
pub use wgpui_core::geometry::Rect;
#[cfg(feature = "inspector")]
pub use wgpui_core::reconcile::{ElementId, InstanceKey};
#[cfg(feature = "inspector")]
pub use wgpui_core::scene::{BoundaryId, TileCoord};
