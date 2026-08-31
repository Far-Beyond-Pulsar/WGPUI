use wgpui::{ApplicationError, NativeApplication, ScrollHandle, Styled, WindowOptions, div, px};

fn main() -> Result<(), ApplicationError> {
    let vertical = ScrollHandle::new();
    let horizontal = ScrollHandle::new();
    NativeApplication::new(WindowOptions::default(), move |_window| {
        div().w_full().h_full().p_4().track_scroll(&vertical).child(
            div()
                .h(px(5_000.0))
                .child(
                    div()
                        .w(px(2_000.0))
                        .h(150.0)
                        .track_scroll(&horizontal)
                        .child("Scroll Horizontal"),
                )
                .child("Scroll Vertical"),
        )
    })
    .run()
}
