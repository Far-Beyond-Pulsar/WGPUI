//! Minimal repro for the window-teardown segfault: open a window, render a
//! frame, then quit cleanly via `cx.quit()`. If teardown is sound, it prints
//! "clean exit reached" and exits 0.
//!     cargo run --example teardown_repro

use std::time::Duration;

use gpui::{
    AppContext as _,
    Application,
    Bounds,
    Context,
    IntoElement,
    Render,
    Styled,
    Timer,
    Window,
    WindowBounds,
    WindowOptions,
    div,
    px,
    rgb,
    size,
};

struct ReproView;

impl Render for ReproView {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        div().size_full().bg(rgb(0x113355))
    }
}

fn main() {
    Application::new().run(|cx| {
        let bounds = Bounds::centered(None, size(px(400.), px(300.)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_window, cx| cx.new(|_cx| ReproView),
        )
        .expect("failed to open window");

        cx.spawn(async move |cx| {
            Timer::after(Duration::from_millis(400)).await;
            let _ = cx.update(|cx| cx.quit());
        })
        .detach();
    });
    println!("clean exit reached");
}
