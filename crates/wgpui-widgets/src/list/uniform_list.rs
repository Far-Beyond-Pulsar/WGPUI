//! `uniform_list()` — the CPU special case §6.1's GPU kernel generalizes.
//! Today's `src/elements/uniform_list.rs`; Phase 0's Spike B benchmarks
//! this file's per-item positioning loop directly. See
//! docs/gpu-native-architecture.md §3.4, §6.1, §8 Phase 0.
#![allow(dead_code)]
