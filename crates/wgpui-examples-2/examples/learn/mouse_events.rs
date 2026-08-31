//! Native mouse and drag/drop event routing.

use std::cell::Cell;
use std::rc::Rc;

use wgpui::{
    ApplicationError, EventResult, MouseButton, NativeApplication, Styled, WindowOptions, div, px,
    rgb,
};

fn main() -> Result<(), ApplicationError> {
    let hovered = Rc::new(Cell::new(false));
    let drag_hovered = Rc::new(Cell::new(false));
    let drop_count = Rc::new(Cell::new(0));

    NativeApplication::with_window(WindowOptions::default(), {
        let hovered = hovered.clone();
        let drag_hovered = drag_hovered.clone();
        let drop_count = drop_count.clone();
        move |_| {
            let hovered_for_style = hovered.get();
            let drag_hovered_for_style = drag_hovered.get();
            let hovered_for_callback = hovered.clone();
            let drag_hovered_for_callback = drag_hovered.clone();
            let drop_count_for_callback = drop_count.clone();
            div()
                .size_full()
                .p_12()
                .flex()
                .flex_col()
                .gap_6()
                .bg(rgb(0x111827))
                .child(
                    div()
                        .w(px(280.0))
                        .h(px(180.0))
                        .rounded_lg()
                        .bg(if drag_hovered_for_style {
                            rgb(0x16a34a)
                        } else if hovered_for_style {
                            rgb(0x2563eb)
                        } else {
                            rgb(0x1e3a5f)
                        })
                        .flex()
                        .items_center()
                        .justify_center()
                        .text_color(rgb(0xffffff))
                        .child("Move, click, or drop here")
                        .on_hover(move |inside, _, _| {
                            hovered_for_callback.set(inside);
                            EventResult {
                                handled: true,
                                propagate: true,
                            }
                        })
                        .on_drag_hover::<String, _>(move |inside, _, _| {
                            drag_hovered_for_callback.set(inside);
                        })
                        .on_drop::<String, _>(move |_, _, _| {
                            drop_count_for_callback.set(drop_count_for_callback.get() + 1);
                        }),
                )
                .child(
                    div()
                        .w(px(220.0))
                        .h(px(56.0))
                        .rounded_md()
                        .bg(rgb(0xea580c))
                        .flex()
                        .items_center()
                        .justify_center()
                        .text_color(rgb(0xffffff))
                        .child(format!("Drag source · drops {}", drop_count.get()))
                        .on_mouse_up(MouseButton::Left, move |_, _, _| {})
                        .on_drag(String::from("native payload"), |_, _, _, _| {}),
                )
        }
    })
    .run()
}
