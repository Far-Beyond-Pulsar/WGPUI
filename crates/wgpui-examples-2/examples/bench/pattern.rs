use wgpui::{
    App, Application, ApplicationError, Bounds, WindowBounds, WindowOptions, div,
    gradient_color_stop, linear_gradient, pattern_slash, prelude::*, px, rgb, size,
};

struct PatternExample;

impl Render for PatternExample {
    fn render(&mut self) -> impl IntoElement + 'static {
        div()
            .flex()
            .flex_col()
            .gap_3()
            .bg(rgb(0xffffff))
            .size(px(600.0))
            .justify_center()
            .items_center()
            .shadow_lg()
            .text_xl()
            .text_color(rgb(0x000000))
            .child("Pattern Example")
            .child(
                div()
                    .flex()
                    .flex_col()
                    .border_1()
                    .border_color(wgpui::blue())
                    .child(div().w(px(54.0)).h(px(18.0)).bg(pattern_slash(
                        wgpui::red(),
                        18.0 / 4.0,
                        18.0 / 4.0,
                    )))
                    .child(div().w(px(54.0)).h(px(18.0)).bg(pattern_slash(
                        wgpui::red(),
                        18.0 / 4.0,
                        18.0 / 4.0,
                    )))
                    .child(div().w(px(54.0)).h(px(18.0)).bg(pattern_slash(
                        wgpui::red(),
                        18.0 / 4.0,
                        18.0 / 4.0,
                    )))
                    .child(div().w(px(54.0)).h(px(18.0)).bg(pattern_slash(
                        wgpui::red(),
                        18.0 / 4.0,
                        18.0 / 2.0,
                    ))),
            )
            .child(
                div()
                    .flex()
                    .flex_col()
                    .border_1()
                    .border_color(wgpui::blue())
                    .bg(wgpui::green().opacity(0.16))
                    .child("Elements the same height should align")
                    .child(div().w(px(256.0)).h(px(56.0)).bg(pattern_slash(
                        wgpui::red(),
                        56.0 / 6.0,
                        56.0 / 6.0,
                    )))
                    .child(div().w(px(256.0)).h(px(56.0)).bg(pattern_slash(
                        wgpui::green(),
                        56.0 / 6.0,
                        56.0 / 6.0,
                    )))
                    .child(div().w(px(256.0)).h(px(56.0)).bg(pattern_slash(
                        wgpui::blue(),
                        56.0 / 6.0,
                        56.0 / 6.0,
                    )))
                    .child(div().w(px(256.0)).h(px(26.0)).bg(pattern_slash(
                        wgpui::yellow(),
                        56.0 / 6.0,
                        56.0 / 6.0,
                    ))),
            )
            .child(
                div()
                    .border_1()
                    .border_color(wgpui::blue())
                    .w(px(240.0))
                    .h(px(40.0))
                    .bg(wgpui::red()),
            )
            .child(
                div()
                    .border_1()
                    .border_color(wgpui::blue())
                    .w(px(240.0))
                    .h(px(40.0))
                    .bg(linear_gradient(
                        45.,
                        gradient_color_stop(wgpui::red(), 0.),
                        gradient_color_stop(wgpui::blue(), 1.),
                    )),
            )
    }
}

fn main() -> Result<(), ApplicationError> {
    Application::new().run(|cx: &mut App| {
        let bounds = Bounds::centered(None::<()>, size(px(600.0), px(600.0)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_window, cx| cx.new(|_cx| PatternExample),
        )
        .unwrap();

        cx.activate(true);
    })
}
