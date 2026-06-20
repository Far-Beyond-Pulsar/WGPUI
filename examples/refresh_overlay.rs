//! UI refresh-heat overlay demo — a 2D analogue of Unreal's shader-complexity
//! view. Each view is tinted by how long its last refresh took (green = cheap or
//! cached, red = expensive), with a heat legend along the bottom.
//!
//! The overlay is enabled on start so running the example shows it immediately;
//! press F12 to toggle it. Run with `cargo run --example refresh_overlay`.

use gpui::*;

struct RefreshOverlayDemo {
    focus: FocusHandle,
    started: bool,
}

impl Render for RefreshOverlayDemo {
    fn render(&mut self, window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        // Turn the heat overlay on the first time we render so the example
        // demonstrates it without any input.
        if !self.started {
            window.set_refresh_overlay(true);
            self.started = true;
        }

        div()
            .key_context("RefreshOverlayDemo")
            .track_focus(&self.focus)
            .on_action(|_: &ToggleRefreshOverlay, window, _cx| window.toggle_refresh_overlay())
            .flex()
            .flex_col()
            .size_full()
            .bg(rgb(0x101418))
            .gap_3()
            .p_8()
            .child(
                div()
                    .text_color(rgb(0xf0f0f0))
                    .text_size(px(20.))
                    .child("UI refresh-heat overlay — press F12 to toggle"),
            )
            .children((0..8_usize).map(|row| {
                div()
                    .id(("row", row))
                    .w(px(420.))
                    .h(px(36.))
                    .bg(rgb(0x222b36))
                    .text_color(rgb(0xc8d2dc))
                    .child(format!("view row {row}"))
            }))
    }
}

fn main() {
    Application::new().run(|cx: &mut App| {
        cx.bind_keys([KeyBinding::new("f12", ToggleRefreshOverlay, None)]);

        let bounds = Bounds::centered(None, size(px(640.), px(520.)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_window, cx| {
                cx.new(|cx| RefreshOverlayDemo {
                    focus: cx.focus_handle(),
                    started: false,
                })
            },
        )
        .expect("failed to open window");
    });
}
