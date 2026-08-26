//! Invalidation vocabulary: the four axes plus the `Reason::Scroll` signal.
//! See docs/gpu-native-architecture.md §5.4.
#![allow(dead_code)]

pub mod axes;
pub mod reason;
pub mod request;
