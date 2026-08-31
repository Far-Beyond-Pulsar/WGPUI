use wgpui::{
    ApplicationError, NativeApplication, Pixels, ScrollHandle, Styled, WindowOptions, div, px,
    virtual_list,
};

fn main() -> Result<(), ApplicationError> {
    let scroll_handle = ScrollHandle::new();
    let heights: Vec<Pixels> = (0..200)
        .map(|index| {
            if index % 5 == 0 {
                px(60.0)
            } else if index % 3 == 0 {
                px(45.0)
            } else {
                px(30.0)
            }
        })
        .collect();

    NativeApplication::new(WindowOptions::default(), move |_window| {
        virtual_list(heights.clone(), |index| {
            let height = if index % 5 == 0 {
                60.0
            } else if index % 3 == 0 {
                45.0
            } else {
                30.0
            };
            div()
                .h(height)
                .px_2()
                .border_b_1()
                .child(format!("Item {} • height {height:.0}px", index + 1))
        })
        .w_full()
        .h_full()
        .track_scroll(&scroll_handle)
    })
    .run()
}
