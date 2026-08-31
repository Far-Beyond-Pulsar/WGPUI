//! Native retained interaction example.

use std::cell::Cell;
use std::rc::Rc;

use wgpui::{
    ApplicationError, EventResult, FocusHandle, MouseButton, NativeApplication, Styled, WindowOptions,
    div,
    px, rgb,
};

fn main() -> Result<(), ApplicationError> {
    let clicks = Rc::new(Cell::new(0));
    let hovered = Rc::new(Cell::new(false));
    let focused = FocusHandle::new().with_tab_index(0).with_tab_stop(true);

    NativeApplication::with_window(WindowOptions::default(), {
        let clicks = clicks.clone();
        let hovered = hovered.clone();
        move |_| {
            let clicks_for_callback = clicks.clone();
            let hovered_for_callback = hovered.clone();
            let button_color = if hovered.get() { rgb(0x2563eb) } else { rgb(0x1d4ed8) };
            div()
                .size_full()
                .p_12()
                .flex()
                .flex_col()
                .gap_5()
                .bg(rgb(0x0f172a))
                .child(
                    div()
                        .text_color(rgb(0xffffff))
                        .text_xl()
                        .child("Interactive elements"),
                )
                .child(
                    div()
                        .w(px(260.0))
                        .h(px(64.0))
                        .rounded_md()
                        .bg(button_color)
                        .flex()
                        .items_center()
                        .justify_center()
                        .text_color(rgb(0xffffff))
                        .child(format!("Clicks: {}", clicks.get()))
                        .track_focus(&focused)
                        .focus_visible(|style| style.border_2().border_color(rgb(0xfacc15)))
                        .on_hover(move |inside, _, _| {
                            hovered_for_callback.set(inside);
                            EventResult {
                                handled: true,
                                propagate: true,
                            }
                        })
                        .on_mouse_up(MouseButton::Left, move |_, _, _| {
                            clicks_for_callback.set(clicks_for_callback.get() + 1);
                        }),
                )
                .child(
                    div()
                        .text_color(rgb(0x94a3b8))
                        .child("Hover, click, and press Tab to exercise retained routing."),
                )
        }
    })
    .run()
}
