//! Native focus-visible styling example.

use wgpui::{ApplicationError, FocusHandle, NativeApplication, Styled, WindowOptions, div, px, rgb};

fn main() -> Result<(), ApplicationError> {
    let first = FocusHandle::new().with_tab_index(0).with_tab_stop(true);
    let second = FocusHandle::new().with_tab_index(1).with_tab_stop(true);
    NativeApplication::with_window(WindowOptions::default(), move |_| {
        div()
            .size_full()
            .p_12()
            .flex()
            .flex_col()
            .gap_4()
            .bg(rgb(0x111827))
            .child(div().text_color(rgb(0xffffff)).text_xl().child("Focus visible"))
            .child(focus_button("First", first))
            .child(focus_button("Second", second))
            .child(
                div()
                    .text_color(rgb(0x94a3b8))
                    .child("Use Tab and Shift-Tab to traverse the focus order."),
            )
    })
    .run()
}

fn focus_button(label: &'static str, focus: FocusHandle) -> impl wgpui::IntoElement {
    div()
        .w(px(260.0))
        .h(px(52.0))
        .rounded_md()
        .bg(rgb(0x1e3a5f))
        .flex()
        .items_center()
        .justify_center()
        .text_color(rgb(0xffffff))
        .child(label)
        .track_focus(&focus)
        .focus_visible(|style| style.border_2().border_color(rgb(0xfacc15)))
}
