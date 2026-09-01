use std::any::Any;
use std::cell::RefCell;
use std::ops::Range;
use std::rc::Rc;

use crate::div::interactivity::style::{DivStyle, classify_style_change};
use crate::scroll::ScrollHandle;
use crate::styled::Styled;
use wgpui_core::element::{Element, IntoElement};
use wgpui_core::geometry::{Bounds, Pixels, Point, Size};
use wgpui_core::invalidation::axes::Invalidation;
use wgpui_core::patch::emit::{Emission, EmitContext};
use wgpui_core::reconcile::description::{Description, ElementId};
use wgpui_core::reconcile::diff_key::ReconcileKey;
use wgpui_layout::taffy_tree::{
    Dimension, LayoutSides, LayoutSize, LengthPercentageAuto, Position,
};

#[derive(Clone, Default)]
pub struct VirtualListScrollController {
    state: Rc<RefCell<Option<(ScrollHandle, Rc<Vec<Pixels>>)>>>,
}

impl VirtualListScrollController {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn scroll_to_item(&self, index: usize) -> bool {
        let state = self.state.borrow();
        let Some((handle, heights)) = state.as_ref() else {
            return false;
        };
        let Some(&height) = heights.get(index) else {
            return false;
        };
        let start = heights
            .iter()
            .take(index)
            .copied()
            .fold(Pixels::ZERO, |total, height| total + height);
        let end = start + height;
        let viewport = handle.viewport().size.height;
        let current_top = -handle.offset().y;
        let current_bottom = current_top + viewport;
        let target = if start < current_top {
            start
        } else if end > current_bottom {
            end - viewport
        } else {
            current_top
        };
        handle.set_offset(Point::new(Pixels::ZERO, -target))
    }

    fn attach(&self, handle: ScrollHandle, heights: Rc<Vec<Pixels>>) {
        *self.state.borrow_mut() = Some((handle, heights));
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VirtualItemTransform {
    pub index: usize,
    pub origin: Point<Pixels>,
    pub height: Pixels,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VirtualListState {
    heights: Vec<Pixels>,
    offsets: Vec<Pixels>,
    viewport: Size<Pixels>,
    offset: Point<Pixels>,
    overscan: Pixels,
    realized: std::collections::BTreeSet<usize>,
}

impl VirtualListState {
    pub fn new(heights: Vec<Pixels>) -> Self {
        let mut offsets = Vec::with_capacity(heights.len());
        let mut position = Pixels::ZERO;
        for height in &heights {
            offsets.push(position);
            position += *height;
        }
        Self {
            heights,
            offsets,
            viewport: Size::default(),
            offset: Point::default(),
            overscan: Pixels(24.0),
            realized: std::collections::BTreeSet::new(),
        }
    }
    pub fn item_count(&self) -> usize {
        self.heights.len()
    }
    pub fn item_height(&self, index: usize) -> Option<Pixels> {
        self.heights.get(index).copied()
    }
    pub fn item_offset(&self, index: usize) -> Option<Pixels> {
        self.offsets.get(index).copied()
    }
    pub fn content_height(&self) -> Pixels {
        self.heights
            .last()
            .zip(self.offsets.last())
            .map_or(Pixels::ZERO, |(height, offset)| *offset + *height)
    }
    pub fn offset(&self) -> Point<Pixels> {
        self.offset
    }
    pub fn viewport(&self) -> Size<Pixels> {
        self.viewport
    }
    pub fn realized(&self) -> &std::collections::BTreeSet<usize> {
        &self.realized
    }
    pub fn set_overscan(&mut self, overscan: Pixels) {
        self.overscan = overscan.max(Pixels::ZERO);
    }
    pub fn set_viewport(&mut self, viewport: Size<Pixels>) {
        self.viewport = viewport;
        self.set_offset(self.offset);
        self.realize();
    }
    pub fn set_offset(&mut self, offset: Point<Pixels>) {
        let max_y = (self.content_height() - self.viewport.height).max(Pixels::ZERO);
        self.offset = Point {
            x: Pixels::ZERO,
            y: offset.y.clamp(-max_y, Pixels::ZERO),
        };
    }
    pub fn scroll_by(&mut self, delta: Point<Pixels>) {
        self.set_offset(self.offset + delta);
    }
    pub fn visible_range(&self) -> Range<usize> {
        let start = (-self.offset.y - self.overscan).max(Pixels::ZERO);
        let end =
            (-self.offset.y + self.viewport.height + self.overscan).min(self.content_height());
        let first = self.offsets.partition_point(|position| *position < start);
        let last = self
            .offsets
            .partition_point(|position| *position < end)
            .min(self.heights.len());
        first.min(last)..last
    }
    pub fn transforms(&self) -> Vec<VirtualItemTransform> {
        self.visible_range()
            .map(|index| VirtualItemTransform {
                index,
                origin: Point {
                    x: Pixels::ZERO,
                    y: self.offset.y + self.offsets[index],
                },
                height: self.heights[index],
            })
            .collect()
    }
    pub fn realize(&mut self) -> Vec<usize> {
        let desired: std::collections::BTreeSet<_> = self.visible_range().collect();
        let added = desired.difference(&self.realized).copied().collect();
        self.realized.extend(desired);
        added
    }
    pub fn evict_outside_viewport(&mut self) -> Vec<usize> {
        let desired: std::collections::BTreeSet<_> = self.visible_range().collect();
        let evicted = self.realized.difference(&desired).copied().collect();
        self.realized.retain(|index| desired.contains(index));
        evicted
    }
    pub fn scroll_to_item(&mut self, index: usize) -> bool {
        let Some(&start) = self.offsets.get(index) else {
            return false;
        };
        let Some(&height) = self.heights.get(index) else {
            return false;
        };
        let end = start + height;
        let top = -self.offset.y;
        let bottom = top + self.viewport.height;
        let target = if start < top {
            start
        } else if end > bottom {
            end - self.viewport.height
        } else {
            top
        };
        self.set_offset(Point {
            x: Pixels::ZERO,
            y: -target,
        });
        self.realize();
        true
    }
}

#[derive(Clone, Debug, PartialEq)]
struct VirtualListKey {
    style: DivStyle,
    item_count: usize,
    content_height: Pixels,
    realized: Range<usize>,
}
impl ReconcileKey for VirtualListKey {
    fn compare(&self, previous: &dyn ReconcileKey) -> Invalidation {
        let Some(previous) = previous.as_any().downcast_ref::<Self>() else {
            return Invalidation::all();
        };
        let mut axes = classify_style_change(&self.style, &previous.style);
        if self.item_count != previous.item_count
            || self.content_height != previous.content_height
            || self.realized != previous.realized
        {
            axes |= Invalidation::LAYOUT;
        }
        axes
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Clone, Debug, PartialEq)]
struct VirtualItemKey {
    index: usize,
    origin: Pixels,
    height: Pixels,
}
impl ReconcileKey for VirtualItemKey {
    fn compare(&self, previous: &dyn ReconcileKey) -> Invalidation {
        let Some(previous) = previous.as_any().downcast_ref::<Self>() else {
            return Invalidation::all();
        };
        if self == previous {
            Invalidation::empty()
        } else {
            Invalidation::LAYOUT
        }
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A retained list for rows whose heights are known independently of layout.
pub struct VirtualList {
    element_id: Option<ElementId>,
    style: DivStyle,
    state: VirtualListState,
    scroll_handle: Option<ScrollHandle>,
    render_item: Box<dyn FnMut(usize) -> Description>,
}

pub fn virtual_list<F, I>(heights: Vec<Pixels>, render_item: F) -> VirtualList
where
    F: FnMut(usize) -> I + 'static,
    I: IntoElement + 'static,
{
    VirtualList::new(heights, render_item)
}

pub fn vlist<T, F, I>(
    entity: wgpui_core::app::Entity<T>,
    element_id: impl Into<ElementId>,
    heights: Rc<Vec<Pixels>>,
    scroll_handle: ScrollHandle,
    controller: VirtualListScrollController,
    mut processor: F,
) -> VirtualList
where
    T: 'static,
    F: FnMut(&mut T, Range<usize>, &mut wgpui_core::window::Window, &mut wgpui_core::app::Context<T>)
        -> Vec<I>
        + 'static,
    I: IntoElement + 'static,
{
    controller.attach(scroll_handle.clone(), heights.clone());
    let weak_entity = entity.downgrade();
    VirtualList::new(heights.as_ref().clone(), move |index| {
        let Some(entity) = weak_entity.upgrade() else {
            return Description::new::<VirtualListItem>();
        };
        let mut window = wgpui_core::window::Window::new();
        entity
            .update_in(&mut window, |value, window, context| {
                processor(value, index..index.saturating_add(1), window, context)
            })
            .into_iter()
            .next()
            .map(IntoElement::into_description)
            .unwrap_or_else(|| Description::new::<VirtualListItem>())
    })
    .id(element_id)
    .track_scroll(&scroll_handle)
}

impl VirtualList {
    pub fn new<F, I>(heights: Vec<Pixels>, mut render_item: F) -> Self
    where
        F: FnMut(usize) -> I + 'static,
        I: IntoElement + 'static,
    {
        Self {
            element_id: None,
            style: DivStyle::default(),
            state: VirtualListState::new(heights),
            scroll_handle: None,
            render_item: Box::new(move |index| render_item(index).into_description()),
        }
    }
    pub fn id(mut self, element_id: impl Into<ElementId>) -> Self {
        self.element_id = Some(element_id.into());
        self
    }
    pub fn overscan(mut self, pixels: Pixels) -> Self {
        self.state.set_overscan(pixels);
        self
    }
    pub fn viewport(mut self, viewport: Size<Pixels>) -> Self {
        self.state.set_viewport(viewport);
        self
    }
    pub fn track_scroll(mut self, handle: &ScrollHandle) -> Self {
        self.scroll_handle = Some(handle.clone());
        self
    }
    pub fn state(&self) -> &VirtualListState {
        &self.state
    }

    fn item_description(&mut self, transform: VirtualItemTransform) -> Description {
        let style = wgpui_layout::taffy_tree::LayoutStyle {
            position: Position::Absolute,
            size: LayoutSize {
                width: Dimension::percent(1.0),
                height: Dimension::length(transform.height.value()),
            },
            inset: LayoutSides {
                left: LengthPercentageAuto::length(0.0),
                top: LengthPercentageAuto::length(
                    transform.origin.y.value() - self.state.offset.y.value(),
                ),
                right: LengthPercentageAuto::length(0.0),
                bottom: LengthPercentageAuto::length(0.0),
            },
            ..Default::default()
        };
        Description::new::<VirtualListItem>()
            .id(transform.index as u64)
            .diff_key(VirtualItemKey {
                index: transform.index,
                origin: transform.origin.y - self.state.offset.y,
                height: transform.height,
            })
            .style(style)
            .child((self.render_item)(transform.index))
    }

    pub fn describe(mut self) -> Description {
        if let Some(handle) = &self.scroll_handle {
            let content_size = Size::pixels(0.0, self.state.content_height().value());
            if self.state.viewport != Size::default() && handle.viewport().size == Size::default() {
                handle.set_viewport(
                    Bounds::new(Point::default(), self.state.viewport),
                    content_size,
                );
            } else {
                handle.set_content_size(content_size);
            }
            let viewport = handle.viewport().size;
            if viewport != Size::default() {
                self.state.set_viewport(viewport);
            }
            self.state.set_offset(handle.offset());
        }
        self.state.realize();
        let realized = self.state.visible_range();
        let children = self
            .state
            .transforms()
            .into_iter()
            .map(|transform| self.item_description(transform))
            .collect::<Vec<_>>();
        let offset = self
            .scroll_handle
            .as_ref()
            .map_or(self.state.offset, ScrollHandle::offset);
        let key = VirtualListKey {
            style: self.style.clone(),
            item_count: self.state.item_count(),
            content_height: self.state.content_height(),
            realized,
        };
        let mut description = Description::new::<Self>()
            .diff_key(key)
            .style(self.style.layout.clone())
            .scroll_offset([offset.x.value(), offset.y.value()])
            .clip_children()
            .children(children);
        if let Some(handle) = self.scroll_handle.as_ref() {
            let handle = handle.clone();
            let content_size = Size::pixels(0.0, self.state.content_height().value());
            description = description.on_layout_changed(move |bounds| {
                handle.set_viewport(
                    Bounds::new(
                        Point::new(Pixels(bounds.x), Pixels(bounds.y)),
                        Size::pixels(bounds.width, bounds.height),
                    ),
                    content_size,
                )
            });
        }
        if let Some(element_id) = self.element_id {
            description = description.id(element_id);
        }
        if let Some(handle) = self.scroll_handle {
            description = description.interaction(
                wgpui_core::reconcile::description::DescriptionInteraction::new(
                    move |event, _, _| match event {
                        wgpui_core::window::InputEvent::Scroll(event) => handle.scroll_wheel(event),
                        _ => wgpui_core::window::EventResult::IGNORED,
                    },
                ),
            );
            description = description.boundary();
        }
        let paint = self.style;
        if paint.primitive_count() > 0 {
            description.emit(move |context: &EmitContext, emission: &mut Emission| {
                paint.paint(context.bounds, emission)
            })
        } else {
            description
        }
    }
}

struct VirtualListItem;
impl Element for VirtualList {
    fn into_description(self) -> Description {
        self.describe()
    }
}
impl Styled for VirtualList {
    fn style(&mut self) -> &mut DivStyle {
        &mut self.style
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::div::div;
    #[test]
    fn virtual_list_lowers_realized_rows_and_evicts_after_scroll() {
        let mut state = VirtualListState::new(vec![Pixels(20.0); 10_000]);
        state.set_viewport(Size::pixels(100.0, 80.0));
        assert!(state.realized().len() < 20);
        state.scroll_by(Point::new(Pixels::ZERO, Pixels(-200.0)));
        let evicted = state.evict_outside_viewport();
        assert!(!evicted.is_empty());
        let handle = ScrollHandle::new();
        let description = virtual_list(vec![Pixels(20.0); 100], |index| {
            div().h(20.0).child(index.to_string())
        })
        .viewport(Size::pixels(100.0, 80.0))
        .track_scroll(&handle)
        .describe();
        assert!(!description.child_descriptions().is_empty());
        assert!(description.is_boundary());
    }

    #[test]
    fn scroll_controller_uses_variable_row_offsets() {
        let handle = ScrollHandle::new();
        handle.set_viewport(
            Bounds::default(),
            Size::pixels(100.0, 30.0),
        );
        let controller = VirtualListScrollController::new();
        let heights = Rc::new(vec![Pixels(10.0), Pixels(30.0), Pixels(20.0)]);
        controller.attach(handle.clone(), heights);

        assert!(controller.scroll_to_item(2));
        assert_eq!(handle.offset().y, Pixels(-30.0));
        assert!(!controller.scroll_to_item(3));
    }
}
