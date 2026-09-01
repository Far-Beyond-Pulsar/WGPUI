//! Retained scrolling primitives used by list and overlay elements.

pub mod scroll_buffer;
pub mod smooth_scroll;

pub use crate::div::scroll_state::ScrollHandle;
pub use scroll_buffer::{ScrollAnchor, ScrollClip, ScrollbarState, TiledScrollState};
pub use smooth_scroll::{ScrollPhysics, ScrollPhysicsMode};

use std::cell::RefCell;
use std::rc::Rc;
use wgpui_core::app::App;
use wgpui_core::geometry::{Bounds, Pixels, Point, Size, point, size};
use wgpui_core::window::{EventResult, InputEvent, MouseButton, ScrollWheelEvent, Window};

/// The axis controlled by a scrollbar.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum ScrollbarOrientation {
    #[default]
    Vertical,
    Horizontal,
}

impl ScrollbarOrientation {
    pub const fn axis(self) -> usize {
        match self {
            Self::Vertical => 1,
            Self::Horizontal => 0,
        }
    }

    pub const fn other_axis(self) -> usize {
        match self {
            Self::Vertical => 0,
            Self::Horizontal => 1,
        }
    }
}

/// Layout-independent information exposed to accessibility and inspection
/// consumers. The bounds are the thumb bounds in the scrollbar's coordinate
/// space; `progress` is normalized from the leading to the trailing edge.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ScrollbarAccessibilityState {
    pub orientation: ScrollbarOrientation,
    pub visible: bool,
    pub bounds: Bounds<Pixels>,
    pub progress: f32,
}

/// The resolved track and thumb geometry for one scrollbar.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ScrollbarGeometry {
    pub orientation: ScrollbarOrientation,
    pub track_bounds: Bounds<Pixels>,
    pub thumb_bounds: Bounds<Pixels>,
    pub visible: bool,
    pub progress: f32,
}

impl ScrollbarGeometry {
    pub fn accessibility(self) -> ScrollbarAccessibilityState {
        ScrollbarAccessibilityState {
            orientation: self.orientation,
            visible: self.visible,
            bounds: self.thumb_bounds,
            progress: self.progress,
        }
    }
}

#[derive(Debug)]
struct ScrollbarController {
    handle: ScrollHandle,
    orientation: ScrollbarOrientation,
    track_bounds: Bounds<Pixels>,
    minimum_thumb_length: Pixels,
    thickness: Pixels,
}

/// Retained pointer and geometry behavior for a scrollbar.
///
/// A controller is cloneable because the track and thumb descriptions may
/// each retain a handle to the same interaction state. Scrolling remains
/// owned by [`ScrollHandle`], including wheel propagation at an edge.
#[derive(Clone, Debug)]
pub struct Scrollbar(Rc<RefCell<ScrollbarController>>);

impl Scrollbar {
    pub fn new(handle: &ScrollHandle, orientation: ScrollbarOrientation) -> Self {
        let viewport = handle.viewport();
        let thickness = Pixels(8.0);
        let track_size = match orientation {
            ScrollbarOrientation::Vertical => size(thickness, viewport.size.height),
            ScrollbarOrientation::Horizontal => size(viewport.size.width, thickness),
        };
        Self(Rc::new(RefCell::new(ScrollbarController {
            handle: handle.clone(),
            orientation,
            track_bounds: Bounds::new(viewport.origin, track_size),
            minimum_thumb_length: Pixels(12.0),
            thickness: Pixels(8.0),
        })))
    }

    pub fn vertical(handle: &ScrollHandle) -> Self {
        Self::new(handle, ScrollbarOrientation::Vertical)
    }

    pub fn horizontal(handle: &ScrollHandle) -> Self {
        Self::new(handle, ScrollbarOrientation::Horizontal)
    }

    pub fn orientation(&self) -> ScrollbarOrientation {
        self.0.borrow().orientation
    }

    pub fn handle(&self) -> ScrollHandle {
        self.0.borrow().handle.clone()
    }

    pub fn set_track_bounds(&self, bounds: Bounds<Pixels>) -> bool {
        let mut state = self.0.borrow_mut();
        if state.track_bounds == bounds {
            return false;
        }
        state.track_bounds = bounds;
        true
    }

    pub fn track_bounds(&self) -> Bounds<Pixels> {
        self.0.borrow().track_bounds
    }

    pub fn set_minimum_thumb_length(&self, length: Pixels) -> bool {
        let mut state = self.0.borrow_mut();
        let length = finite_nonnegative(length.value());
        if state.minimum_thumb_length == length {
            return false;
        }
        state.minimum_thumb_length = length;
        true
    }

    pub fn minimum_thumb_length(&self) -> Pixels {
        self.0.borrow().minimum_thumb_length
    }

    pub fn set_thickness(&self, thickness: Pixels) -> bool {
        let mut state = self.0.borrow_mut();
        let thickness = finite_nonnegative(thickness.value());
        if state.thickness == thickness {
            return false;
        }
        state.thickness = thickness;
        true
    }

    pub fn thickness(&self) -> Pixels {
        self.0.borrow().thickness
    }

    pub fn state(&self) -> ScrollbarState {
        let state = self.0.borrow();
        let handle = &state.handle;
        let axis = state.orientation.axis();
        let viewport = axis_value(handle.viewport().size, axis);
        let content = axis_value(handle.content_size(), axis);
        let offset = axis_value(handle.offset(), axis);
        ScrollbarState::for_axis_with_minimum(
            viewport,
            content,
            offset,
            state.track_length(),
            state.minimum_thumb_length,
        )
    }

    pub fn geometry(&self) -> ScrollbarGeometry {
        let state = self.0.borrow();
        geometry_for(
            state.orientation,
            state.track_bounds,
            state.handle.viewport().size,
            state.handle.content_size(),
            state.handle.offset(),
            state.minimum_thumb_length,
        )
    }

    pub fn accessibility(&self) -> ScrollbarAccessibilityState {
        self.geometry().accessibility()
    }

    pub fn is_dragging(&self) -> bool {
        let state = self.0.borrow();
        state.handle.scrollbar_drag(state.orientation.axis()).is_some()
    }

    pub fn begin_drag(&self, position: [Pixels; 2]) -> bool {
        let state = self.0.borrow();
        let geometry = geometry_for(
            state.orientation,
            state.track_bounds,
            state.handle.viewport().size,
            state.handle.content_size(),
            state.handle.offset(),
            state.minimum_thumb_length,
        );
        let pointer = axis_position(position, state.orientation);
        if !geometry.visible || !contains_point(geometry.thumb_bounds, position) {
            return false;
        }
        state.handle.begin_scrollbar_drag(
            state.orientation.axis(),
            pointer,
            axis_value(state.handle.offset(), state.orientation.axis()),
        )
    }

    pub fn update_drag(&self, position: [Pixels; 2]) -> bool {
        let state = self.0.borrow();
        let Some((pointer_position, drag_offset)) =
            state.handle.scrollbar_drag(state.orientation.axis())
        else {
            return false;
        };
        let geometry = geometry_for(
            state.orientation,
            state.track_bounds,
            state.handle.viewport().size,
            state.handle.content_size(),
            state.handle.offset(),
            state.minimum_thumb_length,
        );
        let travel = axis_length(geometry.track_bounds, state.orientation)
            - axis_length(geometry.thumb_bounds, state.orientation);
        let maximum = axis_value(state.handle.max_offset(), state.orientation.axis()).value();
        if travel <= 0.0 || maximum <= 0.0 {
            return false;
        }
        let delta = axis_position(position, state.orientation).value() - pointer_position.value();
        let next = drag_offset.value() - delta * maximum / travel;
        let current = axis_value(state.handle.offset(), state.orientation.other_axis());
        let next = Pixels(next);
        let changed = match state.orientation {
            ScrollbarOrientation::Vertical => state.handle.set_offset(point(current, next)),
            ScrollbarOrientation::Horizontal => state.handle.set_offset(point(next, current)),
        };
        changed
    }

    pub fn end_drag(&self) -> bool {
        let state = self.0.borrow();
        state.handle.end_scrollbar_drag(state.orientation.axis())
    }

    pub fn cancel_drag(&self) -> bool {
        self.end_drag()
    }

    pub fn click_track(&self, position: [Pixels; 2]) -> bool {
        let state = self.0.borrow();
        let geometry = geometry_for(
            state.orientation,
            state.track_bounds,
            state.handle.viewport().size,
            state.handle.content_size(),
            state.handle.offset(),
            state.minimum_thumb_length,
        );
        if !geometry.visible || !contains_point(geometry.track_bounds, position) {
            return false;
        }
        if contains_point(geometry.thumb_bounds, position) {
            return false;
        }
        let pointer = axis_position(position, state.orientation).value();
        let thumb_start = axis_origin(geometry.thumb_bounds, state.orientation).value();
        let page = axis_value(state.handle.viewport().size, state.orientation.axis());
        let delta = if pointer < thumb_start { page } else { Pixels(-page.value()) };
        match state.orientation {
            ScrollbarOrientation::Vertical => state.handle.scroll_by(point(Pixels::ZERO, delta)),
            ScrollbarOrientation::Horizontal => state.handle.scroll_by(point(delta, Pixels::ZERO)),
        }
    }

    pub fn scroll_wheel(&self, event: &ScrollWheelEvent) -> EventResult {
        self.0.borrow().handle.scroll_wheel(event)
    }

    pub fn handle_input(
        &self,
        event: &InputEvent,
        _window: &mut Window,
        _app: &mut App,
    ) -> EventResult {
        match event {
            InputEvent::MouseDown(event) if event.button == MouseButton::Left => {
                if self.begin_drag(event.position) || self.click_track(event.position) {
                    EventResult::HANDLED
                } else {
                    EventResult::IGNORED
                }
            }
            InputEvent::MouseMove(event) if self.is_dragging() => {
                self.update_drag(event.position);
                EventResult::HANDLED
            }
            InputEvent::MouseUp(event) if event.button == MouseButton::Left => {
                if self.end_drag() {
                    EventResult::HANDLED
                } else {
                    EventResult::IGNORED
                }
            }
            InputEvent::Scroll(event) => self.scroll_wheel(event),
            _ => EventResult::IGNORED,
        }
    }

    /// Lower the visible track and thumb into retained `Div` descriptions.
    ///
    /// The controller is captured by the interaction handlers, so pointer
    /// state survives the per-frame description value being dropped. A track
    /// is emitted only while its associated content exceeds the viewport.
    pub fn description(&self) -> wgpui_core::reconcile::description::Description {
        use crate::div::div;
        use crate::styled::Styled;

        let geometry = self.geometry();
        let track_width = geometry.track_bounds.size.width;
        let track_height = geometry.track_bounds.size.height;
        let thumb_width = geometry.thumb_bounds.size.width;
        let thumb_height = geometry.thumb_bounds.size.height;
        let thumb_left = geometry.thumb_bounds.origin.x - geometry.track_bounds.origin.x;
        let thumb_top = geometry.thumb_bounds.origin.y - geometry.track_bounds.origin.y;
        let offset = self.handle().offset();
        let controller = self.clone();
        let track = match self.orientation() {
            ScrollbarOrientation::Vertical => div()
                .absolute()
                .top(-offset.y.value())
                .right(offset.x.value())
                .w(track_width)
                .h(track_height),
            ScrollbarOrientation::Horizontal => div()
                .absolute()
                .left(-offset.x.value())
                .bottom(offset.y.value())
                .w(track_width)
                .h(track_height),
        }
        .id(format!(
            "__wgpui_scrollbar_{}_{}",
            self.handle().id(),
            match self.orientation() {
                ScrollbarOrientation::Vertical => "vertical",
                ScrollbarOrientation::Horizontal => "horizontal",
            }
        ))
        .on_mouse_down(MouseButton::Left, {
            let controller = controller.clone();
            move |event, window, app| controller.handle_input(event, window, app)
        })
        .on_mouse_move({
            let controller = controller.clone();
            move |event, _, _| {
                if controller.is_dragging() {
                    controller.update_drag(event.position);
                    EventResult::HANDLED
                } else {
                    EventResult::IGNORED
                }
            }
        })
        .on_mouse_up(MouseButton::Left, {
            let controller = controller.clone();
            move |_, _, _| {
                if controller.end_drag() {
                    EventResult::HANDLED
                } else {
                    EventResult::IGNORED
                }
            }
        })
        .on_scroll({
            let controller = controller.clone();
            move |event, _, _| controller.scroll_wheel(event)
        })
        .child(
            div()
                .absolute()
                .left(thumb_left)
                .top(thumb_top)
                .w(thumb_width)
                .h(thumb_height)
                .bg([0.55, 0.55, 0.58, 0.8])
                .rounded_full(),
        );
        track.describe()
    }

    fn track_length(&self) -> Pixels {
        let state = self.0.borrow();
        state.track_length()
    }
}

impl ScrollbarController {
    fn track_length(&self) -> Pixels {
        match self.orientation {
            ScrollbarOrientation::Vertical => self.track_bounds.size.height,
            ScrollbarOrientation::Horizontal => self.track_bounds.size.width,
        }
    }
}

fn finite_nonnegative(value: f32) -> Pixels {
    Pixels(if value.is_finite() { value.max(0.0) } else { 0.0 })
}

fn axis_value<T>(value: T, axis: usize) -> Pixels
where
    T: AxisValue,
{
    value.axis_value(axis)
}

trait AxisValue {
    fn axis_value(self, axis: usize) -> Pixels;
}

impl AxisValue for Size<Pixels> {
    fn axis_value(self, axis: usize) -> Pixels {
        if axis == 0 { self.width } else { self.height }
    }
}

impl AxisValue for Point<Pixels> {
    fn axis_value(self, axis: usize) -> Pixels {
        if axis == 0 { self.x } else { self.y }
    }
}

fn axis_position(position: [Pixels; 2], orientation: ScrollbarOrientation) -> Pixels {
    position[orientation.axis()]
}

fn axis_origin(bounds: Bounds<Pixels>, orientation: ScrollbarOrientation) -> Pixels {
    if orientation.axis() == 0 { bounds.origin.x } else { bounds.origin.y }
}

fn axis_length(bounds: Bounds<Pixels>, orientation: ScrollbarOrientation) -> f32 {
    finite_nonnegative(if orientation.axis() == 0 {
        bounds.size.width.value()
    } else {
        bounds.size.height.value()
    }).value()
}

fn contains_point(bounds: Bounds<Pixels>, position: [Pixels; 2]) -> bool {
    let right = bounds.origin.x.value() + bounds.size.width.value();
    let bottom = bounds.origin.y.value() + bounds.size.height.value();
    position[0].value() >= bounds.origin.x.value()
        && position[0].value() < right
        && position[1].value() >= bounds.origin.y.value()
        && position[1].value() < bottom
}

fn geometry_for(
    orientation: ScrollbarOrientation,
    track_bounds: Bounds<Pixels>,
    viewport: Size<Pixels>,
    content: Size<Pixels>,
    offset: Point<Pixels>,
    minimum_thumb_length: Pixels,
) -> ScrollbarGeometry {
    let axis = orientation.axis();
    let track_length = axis_length(track_bounds, orientation);
    let state = ScrollbarState::for_axis_with_minimum(
        axis_value(viewport, axis),
        axis_value(content, axis),
        axis_value(offset, axis),
        Pixels(track_length),
        minimum_thumb_length,
    );
    let thumb_length = state.thumb_length.value();
    let thumb_offset = state.thumb_offset.value();
    let thumb_bounds = match orientation {
        ScrollbarOrientation::Vertical => Bounds::new(
            track_bounds.origin,
            size(track_bounds.size.width, Pixels(thumb_length)),
        ) + point(Pixels::ZERO, Pixels(thumb_offset)),
        ScrollbarOrientation::Horizontal => Bounds::new(
            track_bounds.origin,
            size(Pixels(thumb_length), track_bounds.size.height),
        ) + point(Pixels(thumb_offset), Pixels::ZERO),
    };
    ScrollbarGeometry {
        orientation,
        track_bounds,
        thumb_bounds,
        visible: state.visible,
        progress: state.progress,
    }
}

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

    #[test]
    fn scrollbar_geometry_is_proportional_in_both_orientations() {
        let vertical = ScrollbarState::vertical(Pixels(100.0), Pixels(400.0), Pixels(-150.0));
        assert!(vertical.visible);
        assert_eq!(vertical.track_length, Pixels(100.0));
        assert_eq!(vertical.thumb_length, Pixels(25.0));
        assert_eq!(vertical.thumb_offset, Pixels(37.5));
        assert_eq!(vertical.progress, 0.5);

        let horizontal = ScrollbarState::horizontal(Pixels(200.0), Pixels(400.0), Pixels(-100.0));
        assert!(horizontal.visible);
        assert_eq!(horizontal.thumb_length, Pixels(100.0));
        assert_eq!(horizontal.thumb_offset, Pixels(50.0));
        assert_eq!(horizontal.progress, 0.5);
    }

    #[test]
    fn scrollbar_minimum_thumb_never_exceeds_a_short_track() {
        let state = ScrollbarState::for_orientation(
            ScrollbarOrientation::Vertical,
            Pixels(100.0),
            Pixels(10_000.0),
            Pixels::ZERO,
        );
        assert!(state.visible);
        assert_eq!(state.track_length, Pixels(100.0));
        assert_eq!(state.thumb_length, Pixels(12.0));

        let handle = ScrollHandle::new();
        handle.set_viewport(
            Bounds::new(
                point(Pixels::ZERO, Pixels::ZERO),
                size(Pixels(10.0), Pixels(100.0)),
            ),
            size(Pixels(10.0), Pixels(1_000.0)),
        );
        let scrollbar = Scrollbar::new(&handle, ScrollbarOrientation::Vertical);
        scrollbar.set_track_bounds(Bounds::new(
            point(Pixels::ZERO, Pixels::ZERO),
            size(Pixels(4.0), Pixels(5.0)),
        ));
        assert_eq!(scrollbar.geometry().thumb_bounds.size.height, Pixels(5.0));
    }

    #[test]
    fn scrollbar_visibility_follows_content_extent() {
        let handle = ScrollHandle::new();
        handle.set_viewport(
            Bounds::new(
                point(Pixels::ZERO, Pixels::ZERO),
                size(Pixels(100.0), Pixels(100.0)),
            ),
            size(Pixels(100.0), Pixels(100.0)),
        );
        assert!(!Scrollbar::vertical(&handle).geometry().visible);

        handle.set_content_size(size(Pixels(300.0), Pixels(100.0)));
        let scrollbar = Scrollbar::horizontal(&handle);
        assert!(scrollbar.geometry().visible);
        assert_eq!(scrollbar.accessibility().orientation, ScrollbarOrientation::Horizontal);
        assert!(scrollbar.accessibility().bounds.size.width > Pixels::ZERO);
    }

    #[test]
    fn scrollbar_drag_clamps_and_clears_capture_state() {
        let handle = ScrollHandle::new();
        handle.set_viewport(
            Bounds::new(
                point(Pixels::ZERO, Pixels::ZERO),
                size(Pixels(100.0), Pixels(100.0)),
            ),
            size(Pixels(100.0), Pixels(300.0)),
        );
        let scrollbar = Scrollbar::vertical(&handle);
        scrollbar.set_track_bounds(Bounds::new(
            point(Pixels::ZERO, Pixels::ZERO),
            size(Pixels(10.0), Pixels(100.0)),
        ));
        let thumb = scrollbar.geometry().thumb_bounds;
        assert!(scrollbar.begin_drag([
            thumb.origin.x,
            thumb.origin.y + Pixels(thumb.size.height.value() / 2.0),
        ]));
        assert!(scrollbar.is_dragging());
        assert!(scrollbar.update_drag([Pixels(5.0), Pixels(10_000.0)]));
        assert_eq!(handle.offset().y, Pixels(-200.0));
        let replacement = Scrollbar::vertical(&handle);
        replacement.set_track_bounds(Bounds::new(
            point(Pixels::ZERO, Pixels::ZERO),
            size(Pixels(10.0), Pixels(100.0)),
        ));
        assert!(replacement.is_dragging());
        assert!(scrollbar.end_drag());
        assert!(!scrollbar.is_dragging());
        assert!(!replacement.is_dragging());
        assert!(!scrollbar.update_drag([Pixels(5.0), Pixels(50.0)]));
    }

    #[test]
    fn scrollbar_track_click_pages_toward_the_clicked_region() {
        let handle = ScrollHandle::new();
        handle.set_viewport(
            Bounds::new(
                point(Pixels::ZERO, Pixels::ZERO),
                size(Pixels(100.0), Pixels(100.0)),
            ),
            size(Pixels(100.0), Pixels(400.0)),
        );
        let scrollbar = Scrollbar::vertical(&handle);
        scrollbar.set_track_bounds(Bounds::new(
            point(Pixels::ZERO, Pixels::ZERO),
            size(Pixels(10.0), Pixels(100.0)),
        ));
        assert!(scrollbar.click_track([Pixels(5.0), Pixels(90.0)]));
        assert_eq!(handle.offset().y, Pixels(-100.0));
        assert!(scrollbar.click_track([Pixels(5.0), Pixels(0.0)]));
        assert_eq!(handle.offset().y, Pixels::ZERO);
    }

    #[test]
    fn scrollbar_input_uses_scroll_handle_wheel_propagation() {
        let handle = ScrollHandle::new();
        handle.set_viewport(
            Bounds::new(
                point(Pixels::ZERO, Pixels::ZERO),
                size(Pixels(100.0), Pixels(100.0)),
            ),
            size(Pixels(100.0), Pixels(300.0)),
        );
        let scrollbar = Scrollbar::vertical(&handle);
        let event = wheel(0.0, 50.0);
        assert_eq!(scrollbar.scroll_wheel(&event), EventResult::HANDLED);
        assert_eq!(handle.offset().y, Pixels(-50.0));
        let result = scrollbar.scroll_wheel(&wheel(0.0, 500.0));
        assert!(result.handled && result.propagate);
        assert_eq!(handle.offset().y, Pixels(-200.0));
    }

    #[test]
    fn tracked_div_retains_visible_scrollbar_descriptions() {
        use crate::div::div;
        use crate::styled::Styled;

        let handle = ScrollHandle::new();
        handle.set_viewport(
            Bounds::new(
                point(Pixels::ZERO, Pixels::ZERO),
                size(Pixels(100.0), Pixels(80.0)),
            ),
            size(Pixels(300.0), Pixels(240.0)),
        );
        let description = div()
            .w(100.0)
            .h(80.0)
            .track_scroll(&handle)
            .describe();
        assert_eq!(description.child_descriptions().len(), 2);
        assert_eq!(description.child_descriptions()[0].child_descriptions().len(), 1);
        assert_eq!(description.child_descriptions()[1].child_descriptions().len(), 1);
    }
}
