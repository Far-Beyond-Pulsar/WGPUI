//! Producer-side triple-buffer for externally-rendered `WgpuSurface`
//! content — UNCHANGED (§5.5, §9's explicit "don't touch this"). Moved,
//! not rebuilt, from today's `src/surface_registry.rs` (772 lines).
//! See docs/gpu-native-architecture.md §5.5, §9.
#![allow(dead_code)]
