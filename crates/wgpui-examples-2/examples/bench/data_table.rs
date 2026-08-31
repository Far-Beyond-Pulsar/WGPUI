use wgpui::{
    ApplicationError, NativeApplication, ScrollHandle, Size, Styled, WindowOptions, div,
    uniform_list,
};

const TOTAL_ROWS: usize = 10_000;

fn main() -> Result<(), ApplicationError> {
    let scroll_handle = ScrollHandle::new();
    NativeApplication::new(WindowOptions::default(), move |_window| {
        uniform_list(TOTAL_ROWS, Size::pixels(720.0, 26.0), |index| {
            div().h(26.0).px_2().border_b_1().child(format!(
                "{index:05}  Quote {:04}  {:.2}",
                index % 1000,
                index as f32 * 1.25
            ))
        })
        .w_full()
        .h_full()
        .track_scroll(&scroll_handle)
    })
    .run()
}
