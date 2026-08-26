//! Ambient reconciliation: `ElementInstance`/`InstanceKey` diffing that
//! applies to every element in the window, not fenced to a `.boundary()`
//! subtree. See docs/gpu-native-architecture.md §4.0, constraint 5 (§0).
#![allow(dead_code)]

pub mod diff_key;
pub mod instance;
pub mod uncached;
