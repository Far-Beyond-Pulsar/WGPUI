//! Native retained styling and regional interaction invalidation.

use std::cell::Cell;
use std::rc::Rc;

use wgpui::{ApplicationError, NativeApplication, Styled, WindowOptions, div, px, rgb};

fn main() -> Result<(), ApplicationError> {
    let hovered = Rc::new(Cell::new(false));
    NativeApplication::with_window(WindowOptions::default(), {
        let hovered = hovered.clone();
        move |_| {
            let hovered_for_callback = hovered.clone();
            div()
                .size_full()
                .p_12()
                .flex()
                .flex_col()
                .gap_6()
                .bg(rgb(0x111827))
                .child(
                    div()
                        .text_color(rgb(0xffffff))
                        .text_xl()
                        .child("Native styling"),
                )
                .child(
                    div()
                        .w(px(360.0))
                        .h(px(180.0))
                        .rounded_lg()
                        .border_2()
                        .border_color(if hovered.get() {
                            rgb(0x38bdf8)
                        } else {
                            rgb(0x334155)
                        })
                        .bg(if hovered.get() {
                            rgb(0x164e63)
                        } else {
                            rgb(0x1e293b)
                        })
                        .flex()
                        .items_center()
                        .justify_center()
                        .text_color(rgb(0xe0f2fe))
                        .child("Hover me")
                        .on_hover(move |inside, _, _| {
                            hovered_for_callback.set(inside);
                        }),
                )
                .child(
                    div()
                        .text_color(rgb(0x94a3b8))
                        .child("Styles are resolved in the retained description."),
                )
        }
    })
    .run()
}
