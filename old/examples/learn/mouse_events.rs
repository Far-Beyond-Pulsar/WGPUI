//! Mouse Events & Drag Hover Example — matches the style of interactive_elements.rs

#[path = "../prelude.rs"]
mod example_prelude;
use example_prelude::init_example;

use gpui::{
    App, Application, Bounds, Context, Hsla, IntoElement, Render, Styled, Window, WindowBounds,
    WindowOptions, div, prelude::*, px, size,
};

#[derive(Clone, Copy)]
struct DragPayload;

impl Render for DragPayload {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .w(px(120.))
            .h(px(40.))
            .bg(gpui::rgb(0xe85d04))
            .rounded_lg()
            .flex()
            .items_center()
            .justify_center()
            .text_color(gpui::white())
            .child("Dragging...")
    }
}

struct MouseEventsExample {
    hovered: bool,
    mouse_inside: bool,
    drag_hovered: bool,
}

impl Render for MouseEventsExample {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let target_bg: Hsla = if self.drag_hovered {
            gpui::rgb(0x22c55e).into()
        } else if self.mouse_inside {
            gpui::rgb(0x3b82f6).into()
        } else {
            gpui::rgb(0x1e3a5f).into()
        };

        let mi = cx.entity().downgrade();
        let mo = cx.entity().downgrade();
        let de = cx.entity().downgrade();
        let dl = cx.entity().downgrade();

        div()
            .size_full()
            .p_12()
            .flex()
            .flex_col()
            .gap_8()
            .bg(gpui::rgb(0x1a1a2e))
            .child(
                div()
                    .text_color(gpui::white())
                    .text_xl()
                    .child("Mouse Events Playground"),
            )
            .child(
                div()
                    .text_color(gpui::rgb(0x8888aa))
                    .text_sm()
                    .child("1. Hover target — on_hover AND on_mouse_enter fire")
                    .child("2. Drag the orange box over target — on_hover STOPS, on_mouse_enter continues")
                    .child("3. on_drag_hover fires ONLY when dragging over target"),
            )
            .child(
                div()
                    .flex()
                    .flex_row()
                    .gap_6()
                    .items_start()
                    .child(render_status_panel(
                        self.hovered,
                        self.mouse_inside,
                        self.drag_hovered,
                    ))
                    .child(
                        div()
                            .id("target")
                            .w(px(200.))
                            .h(px(200.))
                            .rounded_lg()
                            .bg(target_bg)
                            .flex()
                            .items_center()
                            .justify_center()
                            .text_color(gpui::white())
                            .text_lg()
                            .child(if self.drag_hovered {
                                "DROP HERE"
                            } else if self.mouse_inside {
                                "INSIDE"
                            } else {
                                "TARGET"
                            })
                            .on_hover(cx.listener(|this, &h, _, _| {
                                this.hovered = h;
                            }))
                            .on_mouse_enter(move |_, cx| {
                                let _ = mi.update(cx, |this, _| this.mouse_inside = true);
                            })
                            .on_mouse_leave(move |_, cx| {
                                let _ = mo.update(cx, |this, _| this.mouse_inside = false);
                            })
                            .on_drag_hover::<DragPayload>(move |&h, _, cx| {
                                if h {
                                    let _ = de.update(cx, |this, _| this.drag_hovered = true);
                                } else {
                                    let _ = dl.update(cx, |this, _| this.drag_hovered = false);
                                }
                            }),
                    )
                    .child(
                        div()
                            .id("source")
                            .w(px(120.))
                            .h(px(120.))
                            .rounded_lg()
                            .bg(gpui::rgb(0xe85d04))
                            .flex()
                            .items_center()
                            .justify_center()
                            .text_color(gpui::white())
                            .child("Drag Me")
                            .on_drag(DragPayload, |data: &DragPayload, position, _, cx| {
                                cx.new(|_| *data)
                            }),
                    ),
            )
    }
}

fn render_status_panel(hovered: bool, inside: bool, drag_hovered: bool) -> impl IntoElement {
    div()
        .flex()
        .flex_col()
        .gap_3()
        .p_6()
        .rounded_lg()
        .bg(gpui::rgb(0x16213e))
        .border_1()
        .border_color(gpui::rgb(0x0f3460))
        .child(div().text_color(gpui::white()).child("Status"))
        .child(status_row("on_hover", hovered))
        .child(status_row("on_mouse_enter", inside))
        .child(status_row("on_drag_hover", drag_hovered))
}

fn status_row(label: &str, active: bool) -> impl IntoElement {
    let color: Hsla = if active {
        gpui::green()
    } else {
        gpui::rgb(0x333333).into()
    };
    let text_color: Hsla = if active {
        gpui::white()
    } else {
        gpui::rgb(0x666666).into()
    };
    div()
        .flex()
        .items_center()
        .gap_2()
        .child(div().w(px(12.)).h(px(12.)).rounded_full().bg(color))
        .child(div().text_color(text_color).child(label.to_string()))
}

fn main() {
    Application::new().run(|cx: &mut App| {
        init_example(cx, "Mouse Events");
        let bounds = Bounds::centered(None, size(px(800.), px(600.)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| {
                cx.new(|_| MouseEventsExample {
                    hovered: false,
                    mouse_inside: false,
                    drag_hovered: false,
                })
            },
        )
        .unwrap();
    });
}
