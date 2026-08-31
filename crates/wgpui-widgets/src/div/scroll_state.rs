//! Retained two-axis scroll state shared by scroll containers and list widgets.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use wgpui_core::geometry::{Bounds, Pixels, Point, Size, point, size};

static NEXT_SCROLL_HANDLE_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug, Default)]
struct ScrollState {
    offset: Point<Pixels>,
    max_offset: Size<Pixels>,
    viewport: Bounds<Pixels>,
    content_size: Size<Pixels>,
    pending: Option<Point<Pixels>>,
    revision: u64,
}

/// A cloneable handle for reading and changing a scroll container's retained
/// offset. Offsets are negative in the direction of scrolling, matching the
/// coordinates consumed by `Description::scroll_offset`.
#[derive(Clone, Debug)]
pub struct ScrollHandle {
    state: Rc<RefCell<ScrollState>>,
    id: u64,
}

impl Default for ScrollHandle {
    fn default() -> Self {
        Self::new()
    }
}

impl ScrollHandle {
    pub fn new() -> Self {
        Self {
            state: Rc::new(RefCell::new(ScrollState::default())),
            id: NEXT_SCROLL_HANDLE_ID.fetch_add(1, Ordering::Relaxed),
        }
    }
    pub fn offset(&self) -> Point<Pixels> {
        self.state.borrow().offset
    }
    pub fn max_offset(&self) -> Size<Pixels> {
        self.state.borrow().max_offset
    }
    pub fn bounds(&self) -> Bounds<Pixels> {
        self.state.borrow().viewport
    }
    pub fn content_size(&self) -> Size<Pixels> {
        self.state.borrow().content_size
    }
    pub fn revision(&self) -> u64 {
        self.state.borrow().revision
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    /// Copy-only state for the native inspector. The handle itself stays
    /// application-owned and is never exposed through a capture.
    pub fn inspector_info(&self) -> wgpui_core::reconcile::ScrollInfo {
        let state = self.state.borrow();
        wgpui_core::reconcile::ScrollInfo {
            handle_id: self.id,
            content_size: [
                state.content_size.width.value(),
                state.content_size.height.value(),
            ],
            max_offset: [
                state.max_offset.width.value(),
                state.max_offset.height.value(),
            ],
            offset: [state.offset.x.value(), state.offset.y.value()],
        }
    }

    pub fn set_offset(&self, offset: Point<Pixels>) -> bool {
        let mut state = self.state.borrow_mut();
        let clamped = Point {
            x: offset.x.clamp(-state.max_offset.width, Pixels::ZERO),
            y: offset.y.clamp(-state.max_offset.height, Pixels::ZERO),
        };
        if state.offset == clamped {
            return false;
        }
        state.offset = clamped;
        state.pending = None;
        state.revision = state.revision.wrapping_add(1);
        true
    }

    pub fn scroll_by(&self, delta: Point<Pixels>) -> bool {
        let offset = self.offset();
        self.set_offset(point(offset.x + delta.x, offset.y + delta.y))
    }
    pub fn scroll_to_top(&self) -> bool {
        self.set_offset(Point::default())
    }
    pub fn scroll_to_bottom(&self) -> bool {
        let max = self.max_offset();
        self.set_offset(point(-max.width, -max.height))
    }

    pub fn set_viewport(&self, viewport: Bounds<Pixels>, content_size: Size<Pixels>) -> bool {
        let mut state = self.state.borrow_mut();
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
        let clamped_x = state.offset.x.clamp(-max_offset.width, Pixels::ZERO);
        let clamped_y = state.offset.y.clamp(-max_offset.height, Pixels::ZERO);
        if state.offset.x != clamped_x || state.offset.y != clamped_y {
            state.offset.x = clamped_x;
            state.offset.y = clamped_y;
            state.revision = state.revision.wrapping_add(1);
        }
        if changed {
            state.revision = state.revision.wrapping_add(1);
        }
        changed
    }

    pub fn request_scroll_to(&self, offset: Point<Pixels>) {
        self.state.borrow_mut().pending = Some(offset);
    }
    pub fn take_pending_scroll(&self) -> Option<Point<Pixels>> {
        let pending = self.state.borrow_mut().pending.take();
        pending.map(|offset| {
            self.set_offset(offset);
            self.offset()
        })
    }
    pub fn logical_scroll_top(&self) -> (usize, Pixels) {
        (0, -self.offset().y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpui_core::geometry::px;
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

    #[test]
    fn inspector_info_is_copy_only_and_keeps_clone_identity() {
        let first = ScrollHandle::new();
        let second = first.clone();
        first.set_viewport(Bounds::default(), size(px(300.0), px(240.0)));
        first.set_offset(point(px(-20.0), px(-10.0)));
        let info = first.inspector_info();
        assert_eq!(info.handle_id, second.id());
        assert_eq!(info.content_size, [300.0, 240.0]);
        assert_eq!(info.max_offset, [300.0, 240.0]);
        assert_eq!(info.offset, [-20.0, -10.0]);
    }
}
