//! Behavioral gate for the native W1 lifecycle.

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

use wgpui_core::{App, Render};
use wgpui_wgpu::window::application::{Application, WindowOptions};

#[test]
fn application_renders_two_independent_native_windows() {
    let frames = Arc::new([AtomicU64::new(0), AtomicU64::new(0)]);
    let closed_windows = Arc::new(Mutex::new(Vec::new()));
    let result = Application::new().run({
        let frames = Arc::clone(&frames);
        let closed_windows = Arc::clone(&closed_windows);
        move |app| {
            let closed_windows = Arc::clone(&closed_windows);
            app.on_window_closed(move |_, id| {
                if let Ok(mut closed_windows) = closed_windows.lock() {
                    closed_windows.push(id.as_raw());
                }
            })
            .detach();
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
    let closed_windows = closed_windows
        .lock()
        .expect("close callbacks should finish");
    assert_eq!(closed_windows.len(), 2);
    assert!(closed_windows[0] < closed_windows[1]);
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
