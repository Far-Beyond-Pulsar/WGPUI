//! Conservative opaque-region occlusion test — CPU reference implementation,
//! also the oracle the `validate` mode diffs the compute path against.
//! See docs/gpu-native-architecture.md §5.2, R-N §8.3.
#![allow(dead_code)]

pub mod coverage;
