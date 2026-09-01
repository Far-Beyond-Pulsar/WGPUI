//! Overlay elements: anchored and deferred (SFD §2's unbuffered-overlay
//! pattern, reused by §4.3's tile-spanning content rule). See
//! docs/gpu-native-architecture.md §3.4, §4.3.
#![allow(dead_code)]

pub mod anchored;
pub mod deferred;

pub use anchored::{Anchor, AnchorHandle, AnchoredPosition, track_anchor};
pub use deferred::{DeferredOverlay, OverlayHandle, OverlayLayer};
