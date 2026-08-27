//! Invalidation vocabulary: the four axes plus the `Reason::Scroll` signal.
//! See docs/gpu-native-architecture.md §5.4.

pub mod axes;
pub mod reason;
pub mod request;

pub use axes::Invalidation;
pub use reason::Reason;
pub use request::{FrameSignals, InvalidationRequest, InvalidationScope};
