//! Smooth scroll animation and the overscroll buffer (R-N §7/SFD's
//! mechanism, generalized in name only by §4.1's `Buffering::Margin`).
//! See docs/gpu-native-architecture.md §3.4, §4.1.
#![allow(dead_code)]

pub mod scroll_buffer;
pub mod smooth_scroll;
