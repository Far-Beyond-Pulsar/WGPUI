use wgpui::{
    App, Application, Bounds, Context, CursorStyle, Decorations, Hsla, MouseButton,
    Pixels, Point, ResizeEdge, Size, Window, WindowBackgroundAppearance, WindowBounds,
    WindowDecorations, WindowOptions, black, div, green, point, prelude::*, px, rgb, size,
    transparent_black, white,
};

struct WindowShadow {}

// Things to do:
// 1. We need a way of calculating which edge or corner the mouse is on,
//    and then dispatch on that
// 2. We need to improve the shadow rendering significantly
// 3. We need to implement the techniques in here in Zed

impl Render for WindowShadow {
    fn render(&mut self, window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        let decorations = window.window_decorations();
        let rounding = px(10.0);
        let shadow_size = px(10.0);
        let border_size = px(1.0);
        let grey = rgb(0x808080);

        let backdrop = div().id("window-backdrop").bg(transparent_black());
        let backdrop = match decorations {
            Decorations::Server => backdrop,
            Decorations::Client { tiling } => backdrop
                .bg(wgpui::transparent_black())
                .child(
                    div()
                        .size_full()
                        .absolute()
                        .on_mouse_move(wgpui::public_window_callback(|_, window, _| {
                            window.request_redraw();
                        }))
                        .on_mouse_down(
                            MouseButton::Left,
                            wgpui::public_window_callback(move |event: &wgpui::InputEvent, window, _| {
                                let wgpui::InputEvent::MouseDown(event) = event else {
                                    return;
                                };
                                let size = window.window_bounds().get_bounds().size;
                                let position = point(event.position[0], event.position[1]);
                                let result = match resize_edge(position, shadow_size, size) {
                                    Some(edge) => window.start_window_resize(edge),
                                    None => window.start_window_move(),
                                };
                                if let Err(error) = result {
                                    eprintln!("window interaction failed: {error}");
                                }
                            }),
                        ),
                )
                .when(!(tiling.top || tiling.right), |div| div.rounded_tr(rounding.into()))
                .when(!(tiling.top || tiling.left), |div| div.rounded_tl(rounding.into()))
                .when(!tiling.top, |div| div.pt(shadow_size))
                .when(!tiling.bottom, |div| div.pb(shadow_size))
                .when(!tiling.left, |div| div.pl(shadow_size))
                .when(!tiling.right, |div| div.pr(shadow_size)),
        };

        let content = div().cursor(CursorStyle::Arrow);
        let content = match decorations {
            Decorations::Server => content,
            Decorations::Client { tiling } => content
                .border_color(grey)
                .when(!(tiling.top || tiling.right), |div| div.rounded_tr(rounding.into()))
                .when(!(tiling.top || tiling.left), |div| div.rounded_tl(rounding.into()))
                .when(!tiling.top, |div| div.border_t(border_size))
                .when(!tiling.bottom, |div| div.border_b(border_size))
                .when(!tiling.left, |div| div.border_l(border_size))
                .when(!tiling.right, |div| div.border_r(border_size))
                .when(!tiling.is_tiled(), |div| {
                    div.shadow(vec![wgpui::BoxShadow {
                        color: Hsla {
                            h: 0.,
                            s: 0.,
                            l: 0.,
                            a: 0.4,
                        },
                        blur_radius: px(shadow_size.value() / 2.),
                        spread_radius: px(0.),
                        offset: point(px(0.0), px(0.0)),
                    }])
                }),
        };

        let titlebar = div()
            .flex()
            .bg(white())
            .size(px(300.0))
            .justify_center()
            .items_center()
            .shadow_lg()
            .border_1()
            .border_color(rgb(0x0000ff))
            .text_xl()
            .text_color(rgb(0xffffff))
            .child(
                div()
                    .id("hello")
                    .w(px(200.0))
                    .h(px(100.0))
                    .bg(green())
                    .shadow(vec![wgpui::BoxShadow {
                        color: Hsla {
                            h: 0.,
                            s: 0.,
                            l: 0.,
                            a: 1.0,
                        },
                        blur_radius: px(20.0),
                        spread_radius: px(0.0),
                        offset: point(px(0.0), px(0.0)),
                    }]),
            );
        let titlebar = match decorations {
            Decorations::Server => titlebar,
            Decorations::Client { .. } => titlebar
                .on_mouse_down(
                    MouseButton::Left,
                    wgpui::public_window_callback(|_e, window, _| {
                        if let Err(error) = window.start_window_move() {
                            eprintln!("window move failed: {error}");
                        }
                    }),
                )
                .on_click(wgpui::public_window_callback(|e: &wgpui::ClickEvent, window, _| {
                    if e.is_right_click() {
                        window.show_window_menu(e.position());
                    }
                }))
                .text_color(black())
                .child("this is the custom titlebar"),
        };

        backdrop
            .size_full()
            .child(content.bg(wgpui::rgb(0xCCCCFF)).size_full().flex().flex_col().justify_around().child(
                div().w_full().flex().flex_row().justify_around().child(titlebar),
            ))
    }
}

fn resize_edge(pos: Point<Pixels>, shadow_size: Pixels, size: Size<Pixels>) -> Option<ResizeEdge> {
    let edge = if pos.y < shadow_size && pos.x < shadow_size {
        ResizeEdge::TopLeft
    } else if pos.y < shadow_size && pos.x > size.width - shadow_size {
        ResizeEdge::TopRight
    } else if pos.y < shadow_size {
        ResizeEdge::Top
    } else if pos.y > size.height - shadow_size && pos.x < shadow_size {
        ResizeEdge::BottomLeft
    } else if pos.y > size.height - shadow_size && pos.x > size.width - shadow_size {
        ResizeEdge::BottomRight
    } else if pos.y > size.height - shadow_size {
        ResizeEdge::Bottom
    } else if pos.x < shadow_size {
        ResizeEdge::Left
    } else if pos.x > size.width - shadow_size {
        ResizeEdge::Right
    } else {
        return None;
    };
    Some(edge)
}

fn main() {
    Application::new().run(|cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(600.0), px(600.0)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                window_background: WindowBackgroundAppearance::Opaque,
                window_decorations: Some(WindowDecorations::Client),
                ..Default::default()
            },
            |window, cx| {
                cx.new(|cx| {
                    let handle = window.handle();
                    window.observe_appearance(move |_| handle.request_redraw())
                    .detach();
                    WindowShadow {}
                })
            },
        )
        .unwrap();
    });
}
