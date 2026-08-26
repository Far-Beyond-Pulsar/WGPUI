//! `wgpui-wgpu` — the only crate that touches a live `wgpu::Device`:
//! pipelines, compute dispatch, atlas, textures, winit windowing.
//! See docs/gpu-native-architecture.md §3.5.
//!
//! Phase 0's two spikes (§8) live in this crate's `benches/` directory and
//! `examples/adapter_probe.rs`, since they need a real device to compare
//! against the CPU reference paths in the legacy backend.
#![allow(dead_code)]

pub mod render;
pub mod window;
