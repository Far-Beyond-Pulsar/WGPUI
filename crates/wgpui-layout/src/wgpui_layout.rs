//! `wgpui-layout` — Taffy integration, isolated. See
//! docs/gpu-native-architecture.md §3.2. Depended on for heterogeneous
//! flexbox/grid layout, which stays on the CPU on purpose (§6); the regular-
//! content GPU layout kernel (§6.1) lives in `wgpui-core::shaders` /
//! `wgpui-wgpu::render::compute::layout_pass`, not here.
//!
//! Phase 1 fills in [`taffy_tree`] only — the persistent tree ambient
//! reconciliation needs in order to keep a clean element's layout node across
//! frames. [`measure`], [`containment`], and [`regular`] stay Phase 0 stubs:
//! measured leaves arrive with `wgpui-text` (Phase 5), containment is SFD
//! §0.-3's mechanism which nothing in Phase 1 consumes, and §6.1's
//! regular-content detection is gated on a phase Phase 0's Spike B already
//! measured as a loss in its original form.

pub mod containment;
pub mod measure;
pub mod regular;
pub mod taffy_tree;

pub use taffy_tree::{
    AvailableSpace, Dimension, Display, FlexDirection, LayoutError, LayoutFrameStats, LayoutNodeId,
    LayoutRect, LayoutSize, LayoutStyle, LayoutTree, definite,
};
