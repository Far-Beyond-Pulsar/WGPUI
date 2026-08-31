//! Backend-neutral capture primitives; GPU readback and UI traversal are adapters.
pub mod cpu;
pub mod gpu;
pub mod replay;
pub mod ui_capture;
pub use crate::capture::CaptureRequest;
