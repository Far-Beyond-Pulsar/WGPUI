//! Retained scrolling primitives used by list and overlay elements.

pub mod scroll_buffer;
pub mod smooth_scroll;

pub use scroll_buffer::{ScrollAnchor, ScrollClip, ScrollbarState, TiledScrollState};
pub use smooth_scroll::{ScrollPhysics, ScrollPhysicsMode};
pub use crate::div::scroll_state::ScrollHandle;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ScrollStrategy {
    Top,
    Bottom,
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpui_core::geometry::{Bounds, Pixels, Point, Size, point, size};
    use wgpui_core::window::{EventResult, Modifiers, ScrollWheelEvent};

    fn wheel(x: f32, y: f32) -> ScrollWheelEvent {
        ScrollWheelEvent {
            position: [Pixels::ZERO; 2],
            delta: [x, y],
            modifiers: Modifiers::none(),
        }
    }

    #[test]
    fn viewport_clamps_offset_and_preserves_revision_for_inert_updates() {
        let handle = ScrollHandle::new();
        assert!(handle.set_viewport(
            Bounds::new(
                point(Pixels::ZERO, Pixels::ZERO),
                size(Pixels(100.0), Pixels(80.0))
            ),
            size(Pixels(300.0), Pixels(240.0)),
        ));
        assert!(handle.set_offset(Point::new(Pixels(-500.0), Pixels(20.0))));
        assert_eq!(handle.offset(), Point::new(Pixels(-200.0), Pixels::ZERO));
        let revision = handle.revision();
        assert!(!handle.set_offset(Point::new(Pixels(-200.0), Pixels::ZERO)));
        assert_eq!(handle.revision(), revision);
    }

    #[test]
    fn wheel_consumes_available_axis_and_bubbles_remaining_delta() {
        let handle = ScrollHandle::new();
        handle.set_viewport(
            Bounds::new(
                point(Pixels::ZERO, Pixels::ZERO),
                size(Pixels(100.0), Pixels(100.0)),
            ),
            size(Pixels(100.0), Pixels(300.0)),
        );
        assert_eq!(handle.scroll_wheel(&wheel(0.0, 50.0)), EventResult::HANDLED);
        assert_eq!(handle.offset().y, Pixels(-50.0));
        let result = handle.scroll_wheel(&wheel(0.0, 300.0));
        assert_eq!(handle.offset().y, Pixels(-200.0));
        assert!(result.handled && result.propagate);
    }

    #[test]
    fn nested_scroll_handles_share_wheel_delta_at_inner_boundary() {
        let outer = ScrollHandle::new();
        let inner = ScrollHandle::new();
        let viewport = Bounds::new(
            point(Pixels::ZERO, Pixels::ZERO),
            size(Pixels(100.0), Pixels(100.0)),
        );
        outer.set_viewport(viewport, size(Pixels(100.0), Pixels(400.0)));
        inner.set_viewport(viewport, size(Pixels(100.0), Pixels(200.0)));
        inner.set_offset(Point::new(Pixels::ZERO, Pixels(-50.0)));

        let result = inner.scroll_wheel(&wheel(0.0, 75.0));
        assert_eq!(inner.offset().y, Pixels(-100.0));
        assert!(result.propagate);
        assert_eq!(outer.scroll_wheel(&wheel(0.0, 25.0)), EventResult::HANDLED);
        assert_eq!(outer.offset().y, Pixels(-25.0));
    }

    #[test]
    fn scroll_to_item_uses_the_realized_uniform_row_extent() {
        let handle = ScrollHandle::new();
        handle.set_viewport(
            Bounds::default(),
            size(Pixels(100.0), Pixels(2_000.0)),
        );
        handle.set_item_height(Pixels(20.0));

        assert!(handle.scroll_to_item(40, ScrollStrategy::Top));
        assert_eq!(handle.offset().y, Pixels(-800.0));
        assert!(handle.scroll_to_item(0, ScrollStrategy::Top));
        assert_eq!(handle.offset().y, Pixels::ZERO);
    }

    #[test]
    fn tracked_div_publishes_its_resolved_viewport_after_layout() {
        use crate::div::div;
        use crate::styled::Styled;
        use wgpui_core::reconcile::reconciler::Reconciler;
        use wgpui_layout::taffy_tree::{LayoutTree, definite};

        let handle = ScrollHandle::new();
        let description = div()
            .w(100.0)
            .h(80.0)
            .estimated_size([100.0, 240.0])
            .track_scroll(&handle)
            .describe();
        let mut reconciler = Reconciler::new();
        let mut layout = LayoutTree::new();
        let mut plan = reconciler
            .reconcile(description, &mut layout)
            .expect("reconcile");
        let root = plan.root().expect("root").layout_node;
        layout
            .compute_layout(root, definite(100.0, 80.0))
            .expect("layout");
        let mut callback = plan.take_layout_callback(0).expect("layout callback");
        let bounds = layout.layout_of(root).expect("bounds");
        callback.apply(bounds);

        assert_eq!(handle.viewport().size, Size::pixels(100.0, 80.0));
        assert_eq!(handle.content_size(), Size::pixels(100.0, 240.0));
        assert_eq!(handle.max_offset(), Size::pixels(0.0, 160.0));
    }

    #[test]
    fn tracked_div_uses_laid_out_children_as_content_extent() {
        use crate::div::div;
        use crate::styled::Styled;
        use wgpui_core::reconcile::reconciler::Reconciler;
        use wgpui_layout::taffy_tree::{LayoutTree, definite};

        let handle = ScrollHandle::new();
        let description = div()
            .w(100.0)
            .h(80.0)
            .child(div().w(100.0).h(240.0))
            .track_scroll(&handle)
            .describe();
        let mut reconciler = Reconciler::new();
        let mut layout = LayoutTree::new();
        let mut plan = reconciler
            .reconcile(description, &mut layout)
            .expect("reconcile");
        let root = plan.root().expect("root").layout_node;
        layout
            .compute_layout(root, definite(100.0, 80.0))
            .expect("layout");
        let child = layout
            .children(root)
            .expect("children")
            .first()
            .copied()
            .expect("child");
        let bounds = layout.layout_of(root).expect("bounds");
        let child_bounds = layout.layout_of(child).expect("child bounds");
        let mut callback = plan.take_layout_callback(0).expect("layout callback");
        callback.apply_with_content(
            bounds,
            wgpui_layout::taffy_tree::LayoutRect {
                x: 0.0,
                y: 0.0,
                width: child_bounds.x + child_bounds.width,
                height: child_bounds.y + child_bounds.height,
            },
        );

        assert_eq!(handle.content_size().height, Pixels(240.0));
        assert_eq!(handle.max_offset().height, Pixels(160.0));
    }
}
