//! Native animation and retained visual-effects example.

use std::time::Duration;

use wgpui::{
    Animation, AnimationExt as _, ApplicationError, NativeApplication, Styled, WindowOptions, div,
    ease_in_out, linear_color_stop, rgb,
};

fn main() -> Result<(), ApplicationError> {
    NativeApplication::new(WindowOptions::default(), |_window| {
        div()
            .id("animation-root")
            .size_full()
            .p_8()
            .bg(rgb(0x101827))
            .child(
                div()
                    .id("animated-card")
                    .w(280.0)
                    .h(120.0)
                    .rounded_lg()
                    .shadow_lg()
                    .bg_gradient_horizontal(
                        linear_color_stop(rgb(0x2563eb), 0.0),
                        linear_color_stop(rgb(0x9333ea), 1.0),
                    )
                    .with_animation(
                        "card-opacity",
                        Animation::new(Duration::from_millis(1200))
                            .repeat()
                            .with_easing(ease_in_out),
                        |element, progress| element.opacity(0.55 + progress * 0.45),
                    ),
            )
    })
    .with_frame_limit(120)
    .run()
}
