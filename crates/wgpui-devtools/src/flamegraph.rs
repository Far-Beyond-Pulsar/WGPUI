//! Backend-neutral capture primitives; GPU readback and UI traversal are adapters.
pub mod cpu;
pub mod gpu;
pub mod replay;
pub mod ui_capture;
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CaptureRequest {
    pub include_gpu: bool,
}
