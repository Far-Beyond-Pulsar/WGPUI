//! Retained scrolling primitives used by list and overlay elements.

pub mod scroll_buffer;
pub mod smooth_scroll;

pub use scroll_buffer::{ScrollAnchor, ScrollClip, ScrollbarState, TiledScrollState};
pub use smooth_scroll::{ScrollPhysics, ScrollPhysicsMode};

use std::cell::RefCell;
use std::rc::Rc;
use wgpui_core::geometry::{Bounds, Pixels, Point, Size};

#[derive(Clone, Debug, Default)]
struct State { offset: Point<Pixels>, max_offset: Size<Pixels>, viewport: Bounds<Pixels>, content: Size<Pixels>, revision: u64 }

#[derive(Clone, Debug, Default)]
pub struct ScrollHandle(Rc<RefCell<State>>);

impl ScrollHandle {
    pub fn new() -> Self { Self::default() }
    pub fn offset(&self) -> Point<Pixels> { self.0.borrow().offset }
    pub fn max_offset(&self) -> Size<Pixels> { self.0.borrow().max_offset }
    pub fn viewport(&self) -> Bounds<Pixels> { self.0.borrow().viewport }
    pub fn content_size(&self) -> Size<Pixels> { self.0.borrow().content }
    pub fn revision(&self) -> u64 { self.0.borrow().revision }
    pub fn set_viewport(&self, viewport: Bounds<Pixels>, content: Size<Pixels>) -> bool {
        let mut state = self.0.borrow_mut();
        let max = Size::pixels((content.width - viewport.size.width).max(Pixels::ZERO).value(), (content.height - viewport.size.height).max(Pixels::ZERO).value());
        let offset = Point { x: state.offset.x.clamp(-max.width, Pixels::ZERO), y: state.offset.y.clamp(-max.height, Pixels::ZERO) };
        let changed = state.viewport != viewport || state.content != content || state.offset != offset;
        state.viewport = viewport; state.content = content; state.max_offset = max; state.offset = offset;
        if changed { state.revision = state.revision.wrapping_add(1); }
        changed
    }
    pub fn set_offset(&self, offset: Point<Pixels>) -> bool {
        let mut state = self.0.borrow_mut();
        let next = Point { x: offset.x.clamp(-state.max_offset.width, Pixels::ZERO), y: offset.y.clamp(-state.max_offset.height, Pixels::ZERO) };
        if state.offset == next { return false; }
        state.offset = next; state.revision = state.revision.wrapping_add(1); true
    }
    pub fn scroll_by(&self, delta: Point<Pixels>) -> bool { self.set_offset(self.offset() + delta) }
    pub fn scroll_to_top(&self) -> bool { self.set_offset(Point::default()) }
    pub fn scroll_to_bottom(&self) -> bool { let max = self.max_offset(); self.set_offset(Point { x: -max.width, y: -max.height }) }
}
