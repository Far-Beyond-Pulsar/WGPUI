//! Unified `WgpuSurface` consumer entry (§5.5, Gap 2) — the *consuming*
//! half of `SurfaceRegistry`'s composite path, folded into the same
//! indirect-draw entry mechanism `.boundary()`'s texture-retained layers
//! use. `SurfaceRegistry`'s producer side is untouched (§9's risk table).
//! See docs/gpu-native-architecture.md §5.5, §9.
#![allow(dead_code)]
