use crate::geometry::{Point, Pixels, Size};
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Clone, Debug, Default)]
struct State {
    offset: Point<Pixels>,
    max_offset: Size<Pixels>,
    viewport: Size<Pixels>,
    content_size: Size<Pixels>,
    axes: [bool; 2],
}

/// Retained offset and extent for a native scroll root.
///
/// The handle is intentionally independent of an element identity. The
/// renderer owns one per retained element address, while input registrations
/// clone it for the duration of a frame.
#[derive(Clone, Debug, Default)]
pub struct ScrollRootHandle(Rc<RefCell<State>>);

impl ScrollRootHandle {
    pub fn new(axes: [bool; 2]) -> Self {
        let handle = Self::default();
        handle.0.borrow_mut().axes = axes;
        handle
    }

    pub fn offset(&self) -> Point<Pixels> {
        self.0.borrow().offset
    }

    pub fn max_offset(&self) -> Size<Pixels> {
        self.0.borrow().max_offset
    }

    pub fn viewport(&self) -> Size<Pixels> {
        self.0.borrow().viewport
    }

    pub fn content_size(&self) -> Size<Pixels> {
        self.0.borrow().content_size
    }

    pub fn axes(&self) -> [bool; 2] {
        self.0.borrow().axes
    }

    pub fn set_viewport(
        &self,
        viewport: Size<Pixels>,
        content_size: Size<Pixels>,
        axes: [bool; 2],
    ) -> bool {
        let mut state = self.0.borrow_mut();
        let max_offset = Size::pixels(
            if axes[0] {
                (content_size.width - viewport.width).max(Pixels::ZERO).value()
            } else {
                0.0
            },
            if axes[1] {
                (content_size.height - viewport.height).max(Pixels::ZERO).value()
            } else {
                0.0
            },
        );
        let offset = Point {
            x: state.offset.x.clamp(-max_offset.width, Pixels::ZERO),
            y: state.offset.y.clamp(-max_offset.height, Pixels::ZERO),
        };
        let changed = state.viewport != viewport
            || state.content_size != content_size
            || state.max_offset != max_offset
            || state.axes != axes
            || state.offset != offset;
        state.viewport = viewport;
        state.content_size = content_size;
        state.max_offset = max_offset;
        state.axes = axes;
        state.offset = offset;
        changed
    }

    /// Scroll by `delta`, returning the portion accepted by this root.
    pub fn scroll_by(&self, delta: Point<Pixels>) -> Point<Pixels> {
        let before = self.offset();
        self.set_offset(before + delta);
        let after = self.offset();
        Point::new(after.x - before.x, after.y - before.y)
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
        true
    }

    pub fn scroll_to_top(&self) -> bool {
        self.set_offset(Point::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::size;

    #[test]
    fn partial_scroll_returns_the_delta_consumed_by_the_root() {
        let handle = ScrollRootHandle::new([false, true]);
        handle.set_viewport(size(Pixels(100.0), Pixels(80.0)), size(Pixels(100.0), Pixels(240.0)), [false, true]);
        assert_eq!(handle.scroll_by(Point::new(Pixels(0.0), Pixels(-200.0))), Point::new(Pixels(0.0), Pixels(-160.0)));
        assert_eq!(handle.scroll_by(Point::new(Pixels(0.0), Pixels(-20.0))), Point::new(Pixels(0.0), Pixels(0.0)));
        assert_eq!(handle.scroll_by(Point::new(Pixels(0.0), Pixels(40.0))), Point::new(Pixels(0.0), Pixels(40.0)));
    }

    #[test]
    fn resizing_clamps_a_retained_offset_and_returning_to_top_is_exact() {
        let handle = ScrollRootHandle::new([false, true]);
        handle.set_viewport(size(Pixels(100.0), Pixels(100.0)), size(Pixels(100.0), Pixels(400.0)), [false, true]);
        handle.set_offset(Point::new(Pixels(0.0), Pixels(-300.0)));
        handle.set_viewport(size(Pixels(100.0), Pixels(300.0)), size(Pixels(100.0), Pixels(400.0)), [false, true]);
        assert_eq!(handle.offset().y, Pixels(-100.0));
        assert!(handle.scroll_to_top());
        assert_eq!(handle.offset().y, Pixels(0.0));
    }

    #[test]
    fn an_inner_root_bubbles_only_its_unconsumed_delta_to_the_outer_root() {
        let inner = ScrollRootHandle::new([false, true]);
        let outer = ScrollRootHandle::new([false, true]);
        inner.set_viewport(
            size(Pixels(100.0), Pixels(60.0)),
            size(Pixels(100.0), Pixels(100.0)),
            [false, true],
        );
        outer.set_viewport(
            size(Pixels(100.0), Pixels(100.0)),
            size(Pixels(100.0), Pixels(300.0)),
            [false, true],
        );

        let requested = Point::new(Pixels(0.0), Pixels(-80.0));
        let consumed_by_inner = inner.scroll_by(requested);
        let remaining = Point::new(
            requested.x - consumed_by_inner.x,
            requested.y - consumed_by_inner.y,
        );
        let consumed_by_outer = outer.scroll_by(remaining);

        assert_eq!(consumed_by_inner.y, Pixels(-40.0));
        assert_eq!(consumed_by_outer.y, Pixels(-40.0));
        assert_eq!(inner.offset().y, Pixels(-40.0));
        assert_eq!(outer.offset().y, Pixels(-40.0));
    }
}
