use wgpui::{ApplicationError, NativeApplication, Styled, WindowOptions, div, rgb};

fn main() -> Result<(), ApplicationError> {
    NativeApplication::new(WindowOptions::default(), move |_window| {
        div()
            .id("root")
            .w(360.0)
            .h(160.0)
            .bg(rgb(0x2050a0))
            .child("Native text is GPU rendered")
    })
    .run()
}
