//! Capture lifetime and the GPU-resource capture switch.

use crate::gpu_resources::CaptureSnapshot;
use std::sync::atomic::{AtomicBool, Ordering};

/// Options for a diagnostics capture.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CaptureRequest {
    /// Include native GPU resources and their lifetime events.
    pub include_gpu: bool,
}

static ACTIVE: AtomicBool = AtomicBool::new(false);

/// Starts a capture. A second start leaves the existing capture untouched.
pub fn start(request: CaptureRequest) -> bool {
    if ACTIVE
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return false;
    }
    crate::gpu_resources::begin(request.include_gpu);
    true
}

/// Stops the active capture and returns its GPU resource snapshot.
pub fn stop() -> Option<CaptureSnapshot> {
    if !ACTIVE.swap(false, Ordering::AcqRel) {
        return None;
    }
    Some(crate::gpu_resources::end())
}

/// Whether a capture is active.
pub fn active() -> bool {
    ACTIVE.load(Ordering::Acquire)
}
