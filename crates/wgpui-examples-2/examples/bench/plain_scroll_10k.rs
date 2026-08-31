use wgpui::{ApplicationError, NativeApplication, ScrollHandle, Styled, WindowOptions, div};

const TOTAL_ITEMS: usize = 10_000;

fn main() -> Result<(), ApplicationError> {
    let scroll_handle = ScrollHandle::new();
    NativeApplication::new(WindowOptions::default(), move |_window| {
        div()
            .w_full()
            .h_full()
            .track_scroll(&scroll_handle)
            .children((0..TOTAL_ITEMS).map(|index| {
                div()
                    .h(24.0)
                    .px_2()
                    .border_b_1()
                    .child(format!("row {index}"))
            }))
    })
    .run()
}
