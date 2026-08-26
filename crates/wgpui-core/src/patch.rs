//! Patch-list protocol: `Patch`, `PatchList` — the one frontend/backend
//! boundary (docs/gpu-native-architecture.md §2, §5.0).
#![allow(dead_code)]

pub mod apply;
pub mod primitive;

/// Placeholder for the patch-list type described in §2 and §5.0. Left empty
/// at Phase 0 — the real shape (insert/update/remove per primitive kind,
/// with a stable GPU-side slot address per §5.0's O(1)-upload commitment)
/// lands in Phase 1.
pub struct PatchList;
