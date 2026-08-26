//! Boundary texture-retention pool and the unified `WgpuSurface` consumer
//! entry (§5.5, Gap 2 — same type as `layer_texture`).
//! See docs/gpu-native-architecture.md §3.5, §5.5.
#![allow(dead_code)]

pub mod external_surface;
pub mod layer_texture;
