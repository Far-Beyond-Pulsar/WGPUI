use wgpui::{
    App, Application, Bounds, EntityFactory, KeyBinding, WindowBounds, WindowOptions, actions, div,
    prelude::*, px, rgb, size,
};

actions!(example, [CloseWindow]);

struct ExampleWindow {}

impl Render for ExampleWindow {
    fn render(&mut self) -> impl IntoElement + 'static {
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
            .child(
                "Closing this window with cmd-w or the traffic lights should quit the application!",
            )
    }
}

fn main() {
    if let Err(error) = Application::new().run(|cx: &mut App| {
        let mut bounds = Bounds::centered(None::<()>, size(px(500.), px(500.0)), cx);
        cx.on_action(|_: &CloseWindow, cx| cx.request_close());

        cx.bind_keys([KeyBinding::new("cmd-w", CloseWindow, None)]);
        cx.on_window_closed(|cx, _window_id| {
            if cx.windows().is_empty() {
                cx.quit();
            }
        })
        .detach();

        if let Err(error) = cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| cx.new(|_| ExampleWindow {}),
        ) {
            eprintln!("failed to open first window: {error}");
            cx.quit();
            return;
        }

        bounds.origin.x += bounds.size.width;

        if let Err(error) = cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| cx.new(|_| ExampleWindow {}),
        ) {
            eprintln!("failed to open second window: {error}");
            cx.quit();
        }
    }) {
        eprintln!("native application failed: {error}");
    }
}
