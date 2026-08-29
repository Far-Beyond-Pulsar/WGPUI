//! GPU-side buffer per slab kind, plus delta-upload adjacency coalescing.
//! See docs/gpu-native-architecture.md §3.5, §5.0.
#![allow(dead_code)]

pub mod slab_buffers;
pub mod upload;
