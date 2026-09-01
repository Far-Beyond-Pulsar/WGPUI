use std::cell::RefCell;
use std::rc::Rc;

use wgpui_core::geometry::{Bounds, Pixels, Point, Rect, Size};

/// The corner of an anchor at which an overlay is placed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Anchor {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

/// The input to retained overlay placement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnchoredPosition {
    pub bounds: Bounds<Pixels>,
    pub anchor: Anchor,
    pub offset: Point<Pixels>,
}

impl AnchoredPosition {
    pub const fn new(bounds: Bounds<Pixels>, anchor: Anchor) -> Self {
        Self {
            bounds,
            anchor,
            offset: Point::new(Pixels::ZERO, Pixels::ZERO),
        }
    }

    pub const fn with_offset(mut self, offset: Point<Pixels>) -> Self {
        self.offset = offset;
        self
    }

    /// Resolve the overlay rectangle and keep it inside the supplied viewport.
    pub fn resolve(self, size: Size<Pixels>, viewport: Rect, margin: Pixels) -> Bounds<Pixels> {
        let anchor = match self.anchor {
            Anchor::TopLeft => self.bounds.origin,
            Anchor::TopRight => Point {
                x: self.bounds.origin.x + self.bounds.size.width,
                y: self.bounds.origin.y,
            },
            Anchor::BottomLeft => Point {
                x: self.bounds.origin.x,
                y: self.bounds.origin.y + self.bounds.size.height,
            },
            Anchor::BottomRight => self.bounds.bottom_right(),
        };
        let mut origin = Point {
            x: anchor.x + self.offset.x,
            y: anchor.y + self.offset.y,
        };
        if matches!(self.anchor, Anchor::TopRight | Anchor::BottomRight) {
            origin.x -= size.width;
        }
        if matches!(self.anchor, Anchor::BottomLeft | Anchor::BottomRight) {
            origin.y -= size.height;
        }

        let margin = margin.value().max(0.0);
        let min_x = Pixels(viewport.min_x + margin);
        let min_y = Pixels(viewport.min_y + margin);
        let max_x = Pixels((viewport.max_x - margin - size.width.value()).max(min_x.value()));
        let max_y = Pixels((viewport.max_y - margin - size.height.value()).max(min_y.value()));
        origin.x = origin.x.clamp(min_x, max_x);
        origin.y = origin.y.clamp(min_y, max_y);
        Bounds::new(origin, size)
    }
}

#[derive(Default)]
struct AnchorState {
    bounds: Bounds<Pixels>,
    revision: u64,
    invalidator: Option<Box<dyn Fn()>>,
}

/// A retained, layout-updated anchor shared by an anchor and its overlay.
#[derive(Clone, Default)]
pub struct AnchorHandle {
    state: Rc<RefCell<AnchorState>>,
}

impl AnchorHandle {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn bounds(&self) -> Bounds<Pixels> {
        self.state.borrow().bounds
    }

    pub fn revision(&self) -> u64 {
        self.state.borrow().revision
    }

    pub fn invalidate_with(&self, invalidator: impl Fn() + 'static) {
        self.state.borrow_mut().invalidator = Some(Box::new(invalidator));
    }

    /// Update the anchor and report whether its geometry changed.
    pub fn set_bounds(&self, bounds: Bounds<Pixels>) -> bool {
        let changed = {
            let mut state = self.state.borrow_mut();
            if state.bounds == bounds {
                false
            } else {
                state.bounds = bounds;
                state.revision = state.revision.wrapping_add(1);
                true
            }
        };
        if changed {
            self.notify_invalidator();
        }
        changed
    }

    pub fn position(&self, anchor: Anchor) -> AnchoredPosition {
        AnchoredPosition::new(self.bounds(), anchor)
    }

    fn notify_invalidator(&self) {
        let callback = self.state.borrow_mut().invalidator.take();
        if let Some(callback) = callback {
            callback();
            self.state.borrow_mut().invalidator = Some(callback);
        }
    }
}

/// Attach an anchor tracker to an existing retained description.
pub fn track_anchor(
    description: wgpui_core::reconcile::description::Description,
    handle: AnchorHandle,
) -> wgpui_core::reconcile::description::Description {
    description.on_layout_changed(move |bounds| {
        handle.set_bounds(Bounds::new(
            Point::new(Pixels(bounds.x), Pixels(bounds.y)),
            Size::pixels(bounds.width, bounds.height),
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds(x: f32, y: f32, width: f32, height: f32) -> Bounds<Pixels> {
        Bounds::new(
            Point::new(Pixels(x), Pixels(y)),
            Size::pixels(width, height),
        )
    }

    #[test]
    fn every_anchor_corner_resolves_from_the_matching_edge() {
        let anchor = bounds(100.0, 80.0, 40.0, 20.0);
        let viewport = Rect::from_origin_size([0.0, 0.0], [400.0, 300.0]);
        let size = Size::pixels(60.0, 30.0);
        assert_eq!(
            AnchoredPosition::new(anchor, Anchor::TopLeft).resolve(size, viewport, Pixels::ZERO),
            bounds(100.0, 80.0, 60.0, 30.0)
        );
        assert_eq!(
            AnchoredPosition::new(anchor, Anchor::TopRight).resolve(size, viewport, Pixels::ZERO),
            bounds(80.0, 80.0, 60.0, 30.0)
        );
        assert_eq!(
            AnchoredPosition::new(anchor, Anchor::BottomLeft).resolve(size, viewport, Pixels::ZERO),
            bounds(100.0, 70.0, 60.0, 30.0)
        );
        assert_eq!(
            AnchoredPosition::new(anchor, Anchor::BottomRight).resolve(
                size,
                viewport,
                Pixels::ZERO
            ),
            bounds(80.0, 70.0, 60.0, 30.0)
        );
    }

    #[test]
    fn movement_and_resize_are_observable_without_a_full_refresh() {
        let handle = AnchorHandle::new();
        let invalidations = Rc::new(RefCell::new(0));
        let count = invalidations.clone();
        handle.invalidate_with(move || *count.borrow_mut() += 1);
        assert_eq!(handle.revision(), 0);
        assert!(handle.set_bounds(bounds(20.0, 30.0, 40.0, 50.0)));
        assert_eq!(handle.revision(), 1);
        assert!(!handle.set_bounds(bounds(20.0, 30.0, 40.0, 50.0)));
        assert_eq!(*invalidations.borrow(), 1);
        let position = handle.position(Anchor::BottomRight);
        let first = position.resolve(
            Size::pixels(20.0, 20.0),
            Rect::from_origin_size([0.0, 0.0], [200.0, 200.0]),
            Pixels::ZERO,
        );
        let second = position.resolve(
            Size::pixels(20.0, 20.0),
            Rect::from_origin_size([0.0, 0.0], [50.0, 50.0]),
            Pixels::ZERO,
        );
        assert_ne!(first, second);
    }
}
