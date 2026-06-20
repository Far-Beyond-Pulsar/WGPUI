//! End-to-end coverage for the multi-crate custom-primitive plugin flow:
//! a primitive defined in its own crate (`gpui-prim-solid`) is painted through
//! the public `Window::paint_custom_primitive` API and collected into the scene's
//! per-type batches, surfaced via `FrameMetrics::custom_primitive_count`.
//!
//! The GPU draw of collected primitives (`PrimitiveRegistry::draw_batch`) requires
//! a device and is validated on a display; this test covers the CPU collection
//! path headlessly.

use gpui_prim_solid::{SolidQuad, SolidQuads};

use crate::{
    Context,
    IntoElement,
    ParentElement,
    Render,
    Styled,
    TestAppContext,
    Window,
    canvas,
    div,
    px,
};

struct CustomPrimRoot {
    count: usize,
}

impl Render for CustomPrimRoot {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        let count = self.count;
        div().w(px(200.)).h(px(200.)).child(
            canvas(
                |_bounds, _window, _cx| (),
                move |bounds, _, window, _cx| {
                    for index in 0..count {
                        let quad = SolidQuad::new(
                            [bounds.origin.x.0 + index as f32, bounds.origin.y.0],
                            [10.0, 10.0],
                            [1.0, 0.0, 0.0, 1.0],
                        );
                        window.paint_custom_primitive(SolidQuads::TYPE, quad.as_bytes(), bounds);
                    }
                },
            )
            .size_full(),
        )
    }
}

#[gpui::test]
fn custom_primitives_are_collected_into_the_scene(cx: &mut TestAppContext) {
    let (_root, cx) = cx.add_window_view(|_window, _cx| CustomPrimRoot { count: 3 });
    // Custom primitives are collected during paint. Force a fresh repaint (rather
    // than a cached-paint replay) so the canvas's paint callback re-runs and
    // re-queues its instances this frame.
    cx.update(|window, cx| {
        window.refresh();
        window.draw(cx).clear();
    });

    let metrics = cx
        .update(|window, _cx| window.frame_metrics().copied())
        .expect("a draw must publish frame metrics");
    assert_eq!(
        metrics.custom_primitive_count, 3,
        "plugin SolidQuad instances must be collected into the scene's custom-primitive batches"
    );
}

#[gpui::test]
fn no_custom_primitives_when_none_painted(cx: &mut TestAppContext) {
    let (_root, cx) = cx.add_window_view(|_window, _cx| CustomPrimRoot { count: 0 });
    cx.update(|window, cx| window.draw(cx).clear());

    let metrics = cx
        .update(|window, _cx| window.frame_metrics().copied())
        .expect("a draw must publish frame metrics");
    assert_eq!(metrics.custom_primitive_count, 0);
}
