//! CPU capture state.
use std::sync::atomic::{AtomicBool, Ordering};
static ACTIVE: AtomicBool = AtomicBool::new(false);
pub fn start() -> bool {
    ACTIVE
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
}
pub fn stop() {
    ACTIVE.store(false, Ordering::Release);
}
pub fn active() -> bool {
    ACTIVE.load(Ordering::Acquire)
}
