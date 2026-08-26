//! `wgpui-layout` — Taffy integration, isolated. See
//! docs/gpu-native-architecture.md §3.2. Depended on for heterogeneous
//! flexbox/grid layout, which stays on the CPU on purpose (§6); the regular-
//! content GPU layout kernel (§6.1) lives in `wgpui-core::shaders` /
//! `wgpui-wgpu::render::compute::layout_pass`, not here.
#![allow(dead_code)]

pub mod containment;
pub mod measure;
pub mod regular;
pub mod taffy_tree;
