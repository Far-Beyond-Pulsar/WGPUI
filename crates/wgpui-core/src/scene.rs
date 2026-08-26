//! Persistent GPU-resident scene: layers, tiles, and the slab allocator
//! backing them. See docs/gpu-native-architecture.md §3.1, and R-N Pillar
//! III (the layer/slab concept this crate's mechanics replace, not discard).
#![allow(dead_code)]

pub mod layer;
pub mod slab;
pub mod slab_range;
pub mod tile;

/// Placeholder for the persistent per-layer-slab scene described in R-N
/// Pillar III and extended by §2's picture. Empty at Phase 0.
pub struct Scene;
