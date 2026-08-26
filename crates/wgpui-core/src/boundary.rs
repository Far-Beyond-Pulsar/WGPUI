//! `.boundary()`: the single cache-boundary primitive, a pure compositing
//! and buffering policy layered on top of always-on reconciliation.
//! See docs/gpu-native-architecture.md §4.1.
#![allow(dead_code)]

pub mod identity;
pub mod policy;
