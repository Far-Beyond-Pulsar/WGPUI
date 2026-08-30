//! Retained two-axis scroll state shared by scroll containers and list widgets.

use std::cell::RefCell;
use std::rc::Rc;
use wgpui_core::geometry::{Bounds, Pixels, Point, Size, point, px, size};

#[derive(Clone, Debug, Default)]
struct ScrollState {
    offset: Point<Pixels>,
    max_offset: Size<Pixels>,
    viewport: Bounds<Pixels>,
    content_size: Size<Pixels>,
    pending: Option<Point<Pixels>>,
}

/// A cloneable handle for reading and changing a scroll container's retained
/// offset. Offsets are negative in the direction of scrolling, matching the
/// coordinates consumed by `Description::scroll_offset`.
#[derive(Clone, Debug, Default)]
pub struct ScrollHandle(Rc<RefCell<ScrollState>>);

impl ScrollHandle {
    pub fn new() -> Self { Self::default() }
    pub fn offset(&self) -> Point<Pixels> { self.0.borrow().offset }
    pub fn max_offset(&self) -> Size<Pixels> { self.0.borrow().max_offset }
    pub fn bounds(&self) -> Bounds<Pixels> { self.0.borrow().viewport }
    pub fn content_size(&self) -> Size<Pixels> { self.0.borrow().content_size }

    pub fn set_offset(&self, offset: Point<Pixels>) -> bool {
        let mut state = self.0.borrow_mut();
        let clamped = Point {
            x: offset.x.clamp(-state.max_offset.width, Pixels::ZERO),
            y: offset.y.clamp(-state.max_offset.height, Pixels::ZERO),
        };
        if state.offset == clamped { return false; }
        state.offset = clamped;
        state.pending = None;
        true
    }

    pub fn scroll_by(&self, delta: Point<Pixels>) -> bool {
        let offset = self.offset();
        self.set_offset(point(offset.x + delta.x, offset.y + delta.y))
    }
    pub fn scroll_to_top(&self) -> bool { self.set_offset(Point::default()) }
    pub fn scroll_to_bottom(&self) -> bool {
        let max = self.max_offset();
        self.set_offset(point(-max.width, -max.height))
    }

    pub fn set_viewport(&self, viewport: Bounds<Pixels>, content_size: Size<Pixels>) -> bool {
        let mut state = self.0.borrow_mut();
        let max_offset = size(
            (content_size.width - viewport.size.width).max(Pixels::ZERO),
            (content_size.height - viewport.size.height).max(Pixels::ZERO),
        );
        let changed = state.viewport != viewport
            || state.content_size != content_size
            || state.max_offset != max_offset;
        state.viewport = viewport;
        state.content_size = content_size;
        state.max_offset = max_offset;
        state.offset.x = state.offset.x.clamp(-max_offset.width, Pixels::ZERO);
        state.offset.y = state.offset.y.clamp(-max_offset.height, Pixels::ZERO);
        changed
    }

    pub fn request_scroll_to(&self, offset: Point<Pixels>) { self.0.borrow_mut().pending = Some(offset); }
    pub fn take_pending_scroll(&self) -> Option<Point<Pixels>> { self.0.borrow_mut().pending.take() }
    pub fn logical_scroll_top(&self) -> (usize, Pixels) { (0, -self.offset().y) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn two_axis_offset_is_clamped_to_content_extent() {
        let handle = ScrollHandle::new();
        assert!(handle.set_viewport(
            Bounds::new(point(px(0.0), px(0.0)), size(px(100.0), px(80.0))),
            size(px(300.0), px(240.0)),
        ));
        assert_eq!(handle.max_offset(), size(px(200.0), px(160.0)));
        assert!(handle.set_offset(point(px(-500.0), px(20.0))));
        assert_eq!(handle.offset(), point(px(-200.0), px(0.0)));
    }
    #[test]
    fn clones_share_state_and_noop_deltas_do_not_invalidate() {
        let first = ScrollHandle::new();
        let second = first.clone();
        first.set_viewport(Bounds::default(), size(px(100.0), px(100.0)));
        assert!(first.scroll_by(point(px(-10.0), px(-4.0))));
        assert!(!second.scroll_by(point(px(0.0), px(0.0))));
        assert_eq!(second.offset(), point(px(-10.0), px(-4.0)));
    }
}
