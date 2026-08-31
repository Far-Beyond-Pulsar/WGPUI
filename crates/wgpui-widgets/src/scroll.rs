//! Retained scrolling primitives used by list and overlay elements.

pub mod scroll_buffer;
pub mod smooth_scroll;

pub use scroll_buffer::{ScrollAnchor, ScrollClip, ScrollbarState, TiledScrollState};
pub use smooth_scroll::{ScrollPhysics, ScrollPhysicsMode};

use std::cell::RefCell;
use std::rc::Rc;
use wgpui_core::geometry::{Bounds, Pixels, Point, Size};
use wgpui_core::window::{EventResult, ScrollWheelEvent};

#[derive(Clone, Debug, Default)]
struct State {
    offset: Point<Pixels>,
    max_offset: Size<Pixels>,
    viewport: Bounds<Pixels>,
    content: Size<Pixels>,
    revision: u64,
}

#[derive(Clone, Debug, Default)]
pub struct ScrollHandle(Rc<RefCell<State>>);

impl ScrollHandle {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn offset(&self) -> Point<Pixels> {
        self.0.borrow().offset
    }

    pub fn max_offset(&self) -> Size<Pixels> {
        self.0.borrow().max_offset
    }

    pub fn viewport(&self) -> Bounds<Pixels> {
        self.0.borrow().viewport
    }

    pub fn content_size(&self) -> Size<Pixels> {
        self.0.borrow().content
    }

    pub fn revision(&self) -> u64 {
        self.0.borrow().revision
    }

    /// Update the content extent while preserving the measured viewport.
    ///
    /// Lists know their content extent before layout runs. Keeping this
    /// operation separate from `set_viewport` lets them publish that extent
    /// without losing a viewport measured by the previous frame.
    pub fn set_content_size(&self, content: Size<Pixels>) -> bool {
        let viewport = self.viewport();
        self.set_viewport(viewport, content)
    }

    pub fn set_viewport(&self, viewport: Bounds<Pixels>, content: Size<Pixels>) -> bool {
        let mut state = self.0.borrow_mut();
        let max = Size::pixels(
            (content.width - viewport.size.width)
                .max(Pixels::ZERO)
                .value(),
            (content.height - viewport.size.height)
                .max(Pixels::ZERO)
                .value(),
        );
        let offset = Point {
            x: state.offset.x.clamp(-max.width, Pixels::ZERO),
            y: state.offset.y.clamp(-max.height, Pixels::ZERO),
        };
        let changed =
            state.viewport != viewport || state.content != content || state.offset != offset;
        state.viewport = viewport;
        state.content = content;
        state.max_offset = max;
        state.offset = offset;
        if changed {
            state.revision = state.revision.wrapping_add(1);
        }
        changed
    }

    pub fn set_offset(&self, offset: Point<Pixels>) -> bool {
        let mut state = self.0.borrow_mut();
        let next = Point {
            x: offset.x.clamp(-state.max_offset.width, Pixels::ZERO),
            y: offset.y.clamp(-state.max_offset.height, Pixels::ZERO),
        };
        if state.offset == next {
            return false;
        }
        state.offset = next;
        state.revision = state.revision.wrapping_add(1);
        true
    }

    /// Apply a platform wheel event and bubble axes that this container could
    /// not consume. Positive wheel deltas move content toward the pointer,
    /// matching the native event convention used by the application layer.
    pub fn scroll_wheel(&self, event: &ScrollWheelEvent) -> EventResult {
        let delta = [
            if event.delta[0].is_finite() {
                event.delta[0]
            } else {
                0.0
            },
            if event.delta[1].is_finite() {
                event.delta[1]
            } else {
                0.0
            },
        ];
        let before = self.offset();
        let requested = Point {
            x: Pixels(before.x.value() - delta[0]),
            y: Pixels(before.y.value() - delta[1]),
        };
        self.set_offset(requested);
        let after = self.offset();
        let consumed_x = before.x.value() - after.x.value();
        let consumed_y = before.y.value() - after.y.value();
        let remaining_x = delta[0] - consumed_x;
        let remaining_y = delta[1] - consumed_y;
        let consumed = consumed_x.abs() > f32::EPSILON || consumed_y.abs() > f32::EPSILON;
        let remaining = remaining_x.abs() > f32::EPSILON || remaining_y.abs() > f32::EPSILON;
        if consumed {
            EventResult {
                handled: true,
                propagate: remaining,
            }
        } else {
            EventResult::IGNORED
        }
    }

    pub fn scroll_by(&self, delta: Point<Pixels>) -> bool {
        self.set_offset(self.offset() + delta)
    }

    pub fn scroll_to_top(&self) -> bool {
        self.set_offset(Point::default())
    }

    pub fn scroll_to_bottom(&self) -> bool {
        let max = self.max_offset();
        self.set_offset(Point {
            x: -max.width,
            y: -max.height,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpui_core::geometry::{point, size};
    use wgpui_core::window::Modifiers;

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
