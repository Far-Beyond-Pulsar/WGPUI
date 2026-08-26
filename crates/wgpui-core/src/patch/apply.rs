//! Applies a `PatchList` to a `Scene`. Phase 1's round-trip gate lives here
//! (docs/gpu-native-architecture.md §8, Phase 1: "apply a patch sequence,
//! read back the resident buffer, matches an equivalent full-rebuild
//! reference exactly").
#![allow(dead_code)]
