use wgpui::{
    App, Application, Bounds, Context, KeyBinding, PromptButton, PromptLevel, Timer, Window,
    WindowBounds, WindowKind, WindowOptions, actions, div, prelude::*, px, rgb, size,
};

struct SubWindow {
    custom_titlebar: bool,
}

fn button(text: &str, on_click: impl Fn(&mut Window, &mut App) + 'static) -> impl IntoElement {
    div()
        .id(text.to_string())
        .flex_none()
        .px_2()
        .bg(rgb(0xf7f7f7))
        .active(|this| this.opacity(0.85))
        .border_1()
        .border_color(rgb(0xe0e0e0))
        .rounded_sm()
        .cursor_pointer()
        .child(text.to_string())
        .on_click(wgpui::public_window_callback(move |_, window, cx| on_click(window, cx)))
}

impl Render for SubWindow {
    fn render(&mut self, _window: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .flex()
            .flex_col()
            .bg(rgb(0xffffff))
            .size_full()
            .gap_2()
            .when(self.custom_titlebar, |cx| {
                cx.child(
                    div()
                        .flex()
                        .h(px(32.))
                        .px_4()
                        .bg(wgpui::blue())
                        .text_color(wgpui::white())
                        .w_full()
                        .child(
                            div()
                                .flex()
                                .items_center()
                                .justify_center()
                                .size_full()
                                .child("Custom Titlebar"),
                        ),
                )
            })
            .child(
                div()
                    .p_8()
                    .gap_2()
                    .child("SubWindow")
                    .child(button("Close", |window, _| {
                        window.remove_window();
                    })),
            )
    }
}

struct WindowDemo {}

impl Render for WindowDemo {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let window_bounds =
            WindowBounds::Windowed(Bounds::centered(None, size(px(300.0), px(300.0)), cx));

        div()
            .p_4()
            .flex()
            .flex_wrap()
            .bg(rgb(0xffffff))
            .size_full()
            .justify_center()
            .content_center()
            .gap_2()
            .child(button("Normal", move |_, cx| {
                cx.open_window(
                    WindowOptions {
                        window_bounds: Some(window_bounds),
                        ..Default::default()
                    },
                    |_, cx| {
                        cx.new(|_| SubWindow {
                            custom_titlebar: false,
                        })
                    },
                )
                .unwrap();
            }))
            .child(button("Popup", move |_, cx| {
                cx.open_window(
                    WindowOptions {
                        window_bounds: Some(window_bounds),
                        kind: WindowKind::PopUp,
                        ..Default::default()
                    },
                    |_, cx| {
                        cx.new(|_| SubWindow {
                            custom_titlebar: false,
                        })
                    },
                )
                .unwrap();
            }))
            .child(button("Custom Titlebar", move |_, cx| {
                cx.open_window(
                    WindowOptions {
                        titlebar: None,
                        window_bounds: Some(window_bounds),
                        ..Default::default()
                    },
                    |_, cx| {
                        cx.new(|_| SubWindow {
                            custom_titlebar: true,
                        })
                    },
                )
                .unwrap();
            }))
            .child(button("Invisible", move |_, cx| {
                cx.open_window(
                    WindowOptions {
                        show: false,
                        window_bounds: Some(window_bounds),
                        ..Default::default()
                    },
                    |_, cx| {
                        cx.new(|_| SubWindow {
                            custom_titlebar: false,
                        })
                    },
                )
                .unwrap();
            }))
            .child(button("Unmovable", move |_, cx| {
                cx.open_window(
                    WindowOptions {
                        is_movable: false,
                        titlebar: None,
                        window_bounds: Some(window_bounds),
                        ..Default::default()
                    },
                    |_, cx| {
                        cx.new(|_| SubWindow {
                            custom_titlebar: false,
                        })
                    },
                )
                .unwrap();
            }))
            .child(button("Unresizable", move |_, cx| {
                cx.open_window(
                    WindowOptions {
                        is_resizable: false,
                        window_bounds: Some(window_bounds),
                        ..Default::default()
                    },
                    |_, cx| {
                        cx.new(|_| SubWindow {
                            custom_titlebar: false,
                        })
                    },
                )
                .unwrap();
            }))
            .child(button("Unminimizable", move |_, cx| {
                cx.open_window(
                    WindowOptions {
                        is_minimizable: false,
                        window_bounds: Some(window_bounds),
                        ..Default::default()
                    },
                    |_, cx| {
                        cx.new(|_| SubWindow {
                            custom_titlebar: false,
                        })
                    },
                )
                .unwrap();
            }))
            .child(button("Hide Application", |_window, cx| {
                cx.hide();
                let mut app = cx.clone();

                // Restore the application after 3 seconds
                cx.spawn(async move {
                    Timer::after(std::time::Duration::from_secs(3)).await;
                    app.activate(false);
                })
                .detach();
            }))
            .child(button("Resize", |window, _| {
                let content_size = window.bounds().size;
                window.resize(size(content_size.height, content_size.width));
            }))
            .child(button("Prompt", |window, cx| {
                let answer = window.prompt(
                    PromptLevel::Info,
                    "Are you sure?",
                    None,
                    &["Ok", "Cancel"],
                    &mut *cx,
                );

                cx.spawn(async move {
                    match answer.await {
                        Ok(Ok(0)) => println!("You have clicked Ok"),
                        Ok(Ok(_)) => println!("You have clicked Cancel"),
                        Ok(Err(error)) => eprintln!("Prompt failed: {error}"),
                        Err(error) => eprintln!("Prompt task failed: {error}"),
                    }
                })
                .detach();
            }))
            .child(button("Prompt (non-English)", |window, cx| {
                let answer = window.prompt(
                    PromptLevel::Info,
                    "Are you sure?",
                    None,
                    &[PromptButton::ok("确定"), PromptButton::cancel("取消")],
                    &mut *cx,
                );

                cx.spawn(async move {
                    match answer.await {
                        Ok(Ok(0)) => println!("You have clicked Ok"),
                        Ok(Ok(_)) => println!("You have clicked Cancel"),
                        Ok(Err(error)) => eprintln!("Prompt failed: {error}"),
                        Err(error) => eprintln!("Prompt task failed: {error}"),
                    }
                })
                .detach();
            }))
    }
}

actions!(window, [Quit]);

fn main() {
    Application::new().run(|cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(800.0), px(600.0)), cx);

        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |window, cx| {
                cx.new(|cx| {
                    window.observe_bounds(move |bounds| {
                        println!("Window bounds changed: {:?}", bounds);
                    })
                    .detach();

                    WindowDemo {}
                })
            },
        )
        .unwrap();

        cx.activate(true);
        cx.on_action(|_: &Quit, cx| cx.quit());
        cx.bind_keys([KeyBinding::new("cmd-q", Quit, None)]);
    });
}
