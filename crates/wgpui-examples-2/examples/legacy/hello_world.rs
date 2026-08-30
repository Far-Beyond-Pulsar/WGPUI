use wgpui::{
    App, Application, Bounds, Context, SharedString, Window, WindowBounds, WindowOptions, div,
    prelude::*, px, rgb, size,
};

struct HelloWorld {
    text: SharedString,
}

impl Render for HelloWorld {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .flex()
            .flex_col()
            .gap_3()
            .bg(rgb(0x505050))
            .size(px(500.0))
            .justify_center()
            .items_center()
            .shadow_lg()
            .border_1()
            .border_color(rgb(0x0000ff))
            .text_xl()
            .text_color(rgb(0xffffff))
            .child(format!("Hello, {}!", &self.text))
            .child(
                div()
                    .flex()
                    .gap_2()
                    .child(
                        div()
                            .size_8()
                            .bg(wgpui::red())
                            .border_1()
                            .border_dashed()
                            .rounded_md()
                            .border_color(wgpui::white()),
                    )
                    .child(
                        div()
                            .size_8()
                            .bg(wgpui::green())
                            .border_1()
                            .border_dashed()
                            .rounded_md()
                            .border_color(wgpui::white()),
                    )
                    .child(
                        div()
                            .size_8()
                            .bg(wgpui::blue())
                            .border_1()
                            .border_dashed()
                            .rounded_md()
                            .border_color(wgpui::white()),
                    )
                    .child(
                        div()
                            .size_8()
                            .bg(wgpui::yellow())
                            .border_1()
                            .border_dashed()
                            .rounded_md()
                            .border_color(wgpui::white()),
                    )
                    .child(
                        div()
                            .size_8()
                            .bg(wgpui::black())
                            .border_1()
                            .border_dashed()
                            .rounded_md()
                            .rounded_md()
                            .border_color(wgpui::white()),
                    )
                    .child(
                        div()
                            .size_8()
                            .bg(wgpui::white())
                            .border_1()
                            .border_dashed()
                            .rounded_md()
                            .border_color(wgpui::black()),
                    ),
            )
    }
}

fn main() {
    Application::new().run(|cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(500.), px(500.0)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| {
                cx.new(|_| HelloWorld {
                    text: "World".into(),
                })
            },
        )
        .unwrap();
        cx.activate(true);
    });
}
