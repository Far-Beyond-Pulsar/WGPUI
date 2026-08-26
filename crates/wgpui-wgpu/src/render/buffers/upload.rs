//! Delta-upload adjacency coalescing (§5.0) — reuses the same coalescing
//! logic `OpenSlabRun` already does for draw calls, applied to writes
//! instead of draws. See docs/gpu-native-architecture.md §5.0.
#![allow(dead_code)]
