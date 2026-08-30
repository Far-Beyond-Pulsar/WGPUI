//! Behavioral gate for the native W1 lifecycle.

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use wgpui_core::{App, Render};
use wgpui_wgpu::window::application::{Application, WindowOptions};

#[test]
fn application_renders_two_independent_native_windows() {
    let frames = Arc::new([AtomicU64::new(0), AtomicU64::new(0)]);
    let result = Application::new().run({
        let frames = Arc::clone(&frames);
        move |app| {
            for index in 0..2 {
                let frames = Arc::clone(&frames);
                app.open_window(WindowOptions::default(), move |_, app| {
                    app.new_entity(TwoWindowRoot {
                        app: app.clone(),
                        frames,
                        index,
                    })
                })
                .expect("window request should be accepted");
            }
        }
    });

    assert!(
        result.is_ok(),
        "native lifecycle failed: {:?}",
        result.err()
    );
    assert!(
        frames
            .iter()
            .map(|frames| frames.load(Ordering::Relaxed))
            .sum::<u64>()
            >= 4
    );
}

struct TwoWindowRoot {
    app: App,
    frames: Arc<[AtomicU64; 2]>,
    index: usize,
}

impl Render for TwoWindowRoot {
    fn render(&mut self) -> impl wgpui_core::element::IntoElement + 'static {
        self.frames[self.index].fetch_add(1, Ordering::Relaxed);
        if self
            .frames
            .iter()
            .map(|frames| frames.load(Ordering::Relaxed))
            .sum::<u64>()
            >= 4
        {
            self.app.quit();
        }
        wgpui_core::reconcile::description::Description::new::<Self>()
    }
}
