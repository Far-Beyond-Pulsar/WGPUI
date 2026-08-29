//! The small trait `wgpui-core`/`wgpui-wgpu` expose into: span push/pop, GPU
//! timestamp write, frame-capture trigger — the same shape `profiling`
//! (already a dependency of the legacy backend) uses for its own
//! backend-agnostic design. See docs/gpu-native-architecture.md §3.6.
#![allow(dead_code)]
