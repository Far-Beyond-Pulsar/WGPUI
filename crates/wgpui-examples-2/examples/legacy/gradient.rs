use wgpui::{
    App, Application, ColorSpace, Context, PathBuilder, Render, Window, WindowOptions,
    canvas, div, gradient_color_stop, linear_gradient, point, prelude::*, px, radial_gradient,
    size,
};

struct GradientViewer {
    color_space: ColorSpace,
}

impl GradientViewer {
    fn new() -> Self {
        Self {
            color_space: ColorSpace::default(),
        }
    }
}

impl Render for GradientViewer {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let color_space = self.color_space;

        div()
            .bg(wgpui::white())
            .size_full()
            .p_4()
            .flex()
            .flex_col()
            .gap_3()
            .child(
                div()
                    .flex()
                    .gap_2()
                    .justify_between()
                    .items_center()
                    .child("Gradient Examples")
                    .child(
                        div().flex().gap_2().items_center().child(
                            div()
                                .id("method")
                                .flex()
                                .px_3()
                                .py_1()
                                .text_sm()
                                .bg(wgpui::black())
                                .text_color(wgpui::white())
                                .child(format!("{:?}", color_space))
                                .active(|this| this.opacity(0.8))
                                .on_click(wgpui::public_listener(cx, move |this, _, _, cx| {
                                    this.color_space = match this.color_space {
                                        ColorSpace::Oklab => ColorSpace::Srgb,
                                        ColorSpace::Srgb => ColorSpace::Oklab,
                                    };
                                    cx.notify();
                                })),
                        ),
                    ),
            )
            .child(
                div()
                    .flex()
                    .flex_1()
                    .gap_3()
                    .child(
                        div()
                            .size_full()
                            .rounded_xl()
                            .flex()
                            .items_center()
                            .justify_center()
                            .bg(wgpui::red())
                            .text_color(wgpui::white())
                            .child("Solid Color"),
                    )
                    .child(
                        div()
                            .size_full()
                            .rounded_xl()
                            .flex()
                            .items_center()
                            .justify_center()
                            .bg(wgpui::blue())
                            .text_color(wgpui::white())
                            .child("Solid Color"),
                    ),
            )
            .child(
                div()
                    .flex()
                    .flex_1()
                    .gap_3()
                    .h_24()
                    .text_color(wgpui::white())
                    .child(
                        div().flex_1().rounded_xl().bg(linear_gradient(
                            45.,
                            gradient_color_stop(wgpui::red(), 0.),
                            gradient_color_stop(wgpui::blue(), 1.),
                        )
                        .color_space(color_space)),
                    )
                    .child(
                        div().flex_1().rounded_xl().bg(linear_gradient(
                            135.,
                            gradient_color_stop(wgpui::red(), 0.),
                            gradient_color_stop(wgpui::green(), 1.),
                        )
                        .color_space(color_space)),
                    )
                    .child(
                        div().flex_1().rounded_xl().bg(linear_gradient(
                            225.,
                            gradient_color_stop(wgpui::green(), 0.),
                            gradient_color_stop(wgpui::blue(), 1.),
                        )
                        .color_space(color_space)),
                    )
                    .child(
                        div().flex_1().rounded_xl().bg(linear_gradient(
                            315.,
                            gradient_color_stop(wgpui::green(), 0.),
                            gradient_color_stop(wgpui::yellow(), 1.),
                        )
                        .color_space(color_space)),
                    ),
            )
            .child(
                div()
                    .flex()
                    .flex_1()
                    .gap_3()
                    .h_24()
                    .text_color(wgpui::white())
                    .child(
                        div().flex_1().rounded_xl().bg(linear_gradient(
                            0.,
                            gradient_color_stop(wgpui::red(), 0.),
                            gradient_color_stop(wgpui::white(), 1.),
                        )
                        .color_space(color_space)),
                    )
                    .child(
                        div().flex_1().rounded_xl().bg(linear_gradient(
                            90.,
                            gradient_color_stop(wgpui::blue(), 0.),
                            gradient_color_stop(wgpui::white(), 1.),
                        )
                        .color_space(color_space)),
                    )
                    .child(
                        div().flex_1().rounded_xl().bg(linear_gradient(
                            180.,
                            gradient_color_stop(wgpui::green(), 0.),
                            gradient_color_stop(wgpui::white(), 1.),
                        )
                        .color_space(color_space)),
                    )
                    .child(
                        div().flex_1().rounded_xl().bg(linear_gradient(
                            360.,
                            gradient_color_stop(wgpui::yellow(), 0.),
                            gradient_color_stop(wgpui::white(), 1.),
                        )
                        .color_space(color_space)),
                    ),
            )
            .child(
                div().flex_1().rounded_xl().bg(linear_gradient(
                    0.,
                    gradient_color_stop(wgpui::green(), 0.05),
                    gradient_color_stop(wgpui::yellow(), 0.95),
                )
                .color_space(color_space)),
            )
            .child(
                div().flex_1().rounded_xl().bg(linear_gradient(
                    90.,
                    gradient_color_stop(wgpui::blue(), 0.05),
                    gradient_color_stop(wgpui::red(), 0.95),
                )
                .color_space(color_space)),
            )
            .child(
                div()
                    .flex()
                    .flex_1()
                    .gap_3()
                    .child(
                        div().flex().flex_1().gap_3().child(
                            div().flex_1().rounded_xl().bg(linear_gradient(
                                90.,
                                gradient_color_stop(wgpui::blue(), 0.5),
                                gradient_color_stop(wgpui::red(), 0.5),
                            )
                            .color_space(color_space)),
                        ),
                    )
                    .child(
                        div().flex_1().rounded_xl().bg(linear_gradient(
                            180.,
                            gradient_color_stop(wgpui::green(), 0.),
                            gradient_color_stop(wgpui::blue(), 0.5),
                        )
                        .color_space(color_space)),
                    ),
            )
            .child(div().h_16().child(canvas(move |context, emission| {
                let bounds = context.bounds();
                let width = bounds.width * 0.8;
                let height = 80.0_f32.min(bounds.height.max(0.0));
                let origin = point(
                    px(bounds.x + (bounds.width - width) * 0.5),
                    px(bounds.y + (bounds.height - height) * 0.5),
                );
                let path_size = size(px(width), px(height));
                let bottom_left = origin + point(px(0.0), path_size.height);
                let top_right = origin + point(path_size.width, px(0.0));
                let bottom_right = origin + point(path_size.width, path_size.height);
                let horizontal_offset = path_size.height;
                let mut builder = PathBuilder::fill();
                builder.move_to(bottom_left);
                builder.line_to(origin + point(horizontal_offset, px(0.0)));
                builder.line_to(top_right + point(-horizontal_offset, px(0.0)));
                builder.line_to(bottom_right);
                builder.line_to(bottom_left);
                if let Ok(path) = builder.build() {
                    emission.path(context.path(
                        path,
                        linear_gradient(
                            180.,
                            gradient_color_stop(wgpui::red(), 0.),
                            gradient_color_stop(wgpui::blue(), 1.),
                        )
                        .color_space(color_space),
                    ));
                }
            })))
            .child(
                div()
                    .flex()
                    .flex_1()
                    .gap_3()
                    .h_32()
                    .child(
                        div().flex_1().rounded_xl().bg(radial_gradient(
                            0.5,
                            0.5,
                            0.5,
                            0.5,
                            gradient_color_stop(wgpui::white(), 0.0),
                            gradient_color_stop(wgpui::blue(), 1.0),
                        )
                        .color_space(color_space)),
                    )
                    .child(
                        div().flex_1().rounded_xl().bg(radial_gradient(
                            0.3,
                            0.3,
                            0.7,
                            0.7,
                            gradient_color_stop(wgpui::yellow(), 0.0),
                            gradient_color_stop(wgpui::red(), 1.0),
                        )
                        .color_space(color_space)),
                    )
                    .child(
                        div().flex_1().rounded_xl().bg(radial_gradient(
                            0.8,
                            0.2,
                            0.8,
                            0.6,
                            gradient_color_stop(wgpui::green(), 0.0),
                            gradient_color_stop(wgpui::black(), 1.0),
                        )
                        .color_space(color_space)),
                    ),
            )
    }
}

fn main() {
    Application::new().run(|cx: &mut App| {
        cx.open_window(
            WindowOptions {
                focus: true,
                ..Default::default()
            },
            |_, cx| cx.new(|_| GradientViewer::new()),
        )
        .unwrap();
        cx.activate(true);
    });
}
