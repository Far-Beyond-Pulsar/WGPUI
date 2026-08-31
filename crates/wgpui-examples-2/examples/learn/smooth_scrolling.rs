use wgpui::{
    ApplicationError, NativeApplication, ScrollHandle, Size, Styled, WindowOptions, div,
    uniform_list,
};

fn main() -> Result<(), ApplicationError> {
    let scroll_handle = ScrollHandle::new();
    NativeApplication::new(WindowOptions::default(), move |_window| {
        uniform_list(200, Size::pixels(360.0, 30.0), |index| {
            div()
                .h(30.0)
                .px_2()
                .border_b_1()
                .child(format!("Item {}", index + 1))
        })
        .w_full()
        .h_full()
        .track_scroll(&scroll_handle)
    })
    .run()
}
