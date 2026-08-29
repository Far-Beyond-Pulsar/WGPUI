//! `wgpui-devtools` — flamegraph/replay/inspector, moved wholesale behind a
//! small hook trait `wgpui-core`/`wgpui-wgpu` expose into. Pure
//! move-and-decouple, zero behavior change (Phase 7's gate: `wgpui-core`
//! builds and runs with this crate absent entirely).
//! See docs/gpu-native-architecture.md §3.6, §8 Phase 7.
#![allow(dead_code)]

pub mod flamegraph;
pub mod hooks;
pub mod inspector;
pub mod perf_ab_tests;
pub mod render_stats;
