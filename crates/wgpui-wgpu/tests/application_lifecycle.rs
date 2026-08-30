//! Behavioral gate for the native W1 lifecycle.

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use wgpui_core::reconcile::description::Description;
use wgpui_wgpu::window::application::{Application, WindowOptions};

#[test]
fn native_run_creates_a_window_and_presents_multiple_retained_frames() {
    let frames = Arc::new(AtomicU64::new(0));
    let observed = Arc::clone(&frames);
    let result = Application::new(WindowOptions::default(), move |_window| {
        observed.fetch_add(1, Ordering::Relaxed);
        Description::new::<Root>()
    })
    .with_frame_limit(2)
    .run();

    assert!(
        result.is_ok(),
        "native lifecycle failed: {:?}",
        result.err()
    );
    assert!(
        frames.load(Ordering::Relaxed) >= 2,
        "run returned without presenting frames"
    );
}

struct Root;
