use std::any::Any;
use std::ops::Range;

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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UniformItemTransform {
    pub index: usize,
    pub origin: Point<Pixels>,
    pub size: Size<Pixels>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct UniformListState {
    pub item_count: usize,
    pub item_size: Size<Pixels>,
    pub viewport: Size<Pixels>,
    pub offset: Point<Pixels>,
    pub overscan: usize,
}

impl UniformListState {
    pub fn new(item_count: usize, item_size: Size<Pixels>) -> Self {
        Self {
            item_count,
            item_size,
            viewport: Size::default(),
            offset: Point::default(),
            overscan: 1,
        }
    }

    pub fn content_size(&self) -> Size<Pixels> {
        Size::pixels(
            self.item_size.width.value(),
            self.item_size.height.value() * self.item_count as f32,
        )
    }

    pub fn set_viewport(&mut self, viewport: Size<Pixels>) {
        self.viewport = viewport;
        self.set_offset(self.offset);
    }

    pub fn set_offset(&mut self, offset: Point<Pixels>) {
        let content = self.content_size();
        self.offset = Point {
            x: offset.x.clamp(
                (self.viewport.width - content.width).min(Pixels::ZERO),
                Pixels::ZERO,
            ),
            y: offset.y.clamp(
                (self.viewport.height - content.height).min(Pixels::ZERO),
                Pixels::ZERO,
            ),
        };
    }

    pub fn realized_range(&self) -> Range<usize> {
        if self.item_count == 0 || self.item_size.height <= Pixels::ZERO {
            return 0..0;
        }
        let first = ((-self.offset.y.value() / self.item_size.height.value()).floor() as usize)
            .saturating_sub(self.overscan);
        let last = ((-self.offset.y.value() + self.viewport.height.value())
            / self.item_size.height.value())
        .ceil() as usize
            + self.overscan;
        first.min(self.item_count)..last.min(self.item_count)
    }

    pub fn transforms(&self) -> Vec<UniformItemTransform> {
        self.realized_range()
            .map(|index| UniformItemTransform {
                index,
                origin: Point {
                    x: self.offset.x,
                    y: self.offset.y + self.item_size.height.scaled(index as f32),
                },
                size: self.item_size,
            })
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq)]
struct UniformListKey {
    style: DivStyle,
    item_count: usize,
    item_size: Size<Pixels>,
    realized: Range<usize>,
}

impl ReconcileKey for UniformListKey {
    fn compare(&self, previous: &dyn ReconcileKey) -> Invalidation {
        let Some(previous) = previous.as_any().downcast_ref::<Self>() else {
            return Invalidation::all();
        };
        let mut axes = classify_style_change(&self.style, &previous.style);
        if self.item_count != previous.item_count
            || self.item_size != previous.item_size
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
struct UniformItemKey {
    index: usize,
    origin: Point<Pixels>,
    size: Size<Pixels>,
}

impl ReconcileKey for UniformItemKey {
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

/// A retained, uniformly-sized list. Only the current realized range becomes
/// part of the description; scrolling inside that range is a container offset.
pub struct UniformList {
    element_id: Option<ElementId>,
    style: DivStyle,
    state: UniformListState,
    scroll_handle: Option<ScrollHandle>,
    render_item: Box<dyn FnMut(usize) -> Description>,
}

pub trait UniformListArguments {
    fn build(self) -> UniformList;
}

impl<F, I> UniformListArguments for (usize, Size<Pixels>, F)
where
    F: FnMut(usize) -> I + 'static,
    I: IntoElement + 'static,
{
    fn build(self) -> UniformList {
        UniformList::new(self.0, self.1, self.2)
    }
}

impl<F, I> UniformListArguments for (ElementId, usize, F)
where
    F: FnMut(Range<usize>) -> Vec<I> + 'static,
    I: IntoElement + 'static,
{
    fn build(self) -> UniformList {
        let (element_id, item_count, mut processor) = self;
        UniformList::new(item_count, Size::pixels(0.0, 20.0), move |index| {
            processor(index..index.saturating_add(1))
                .into_iter()
                .next()
                .map(IntoElement::into_description)
                .unwrap_or_else(|| Description::new::<UniformListItem>())
        })
        .id(element_id)
    }
}

impl<F, I> UniformListArguments for (&str, usize, F)
where
    F: FnMut(Range<usize>) -> Vec<I> + 'static,
    I: IntoElement + 'static,
{
    fn build(self) -> UniformList {
        (ElementId::from(self.0), self.1, self.2).build()
    }
}

pub fn uniform_list<A, B, C>(first: A, second: B, render_item: C) -> UniformList
where
    (A, B, C): UniformListArguments,
{
    (first, second, render_item).build()
}

impl UniformList {
    pub fn new<F, I>(item_count: usize, item_size: Size<Pixels>, mut render_item: F) -> Self
    where
        F: FnMut(usize) -> I + 'static,
        I: IntoElement + 'static,
    {
        Self {
            element_id: None,
            style: DivStyle::default(),
            state: UniformListState::new(item_count, item_size),
            scroll_handle: None,
            render_item: Box::new(move |index| render_item(index).into_description()),
        }
    }
    pub fn id(mut self, element_id: impl Into<ElementId>) -> Self {
        self.element_id = Some(element_id.into());
        self
    }
    pub fn overscan(mut self, items: usize) -> Self {
        self.state.overscan = items;
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
    pub fn state(&self) -> &UniformListState {
        &self.state
    }

    fn item_description(&mut self, transform: UniformItemTransform) -> Description {
        let style = wgpui_layout::taffy_tree::LayoutStyle {
            position: Position::Absolute,
            size: LayoutSize {
                width: Dimension::percent(1.0),
                height: Dimension::length(transform.size.height.value()),
            },
            inset: LayoutSides {
                left: LengthPercentageAuto::length(
                    transform.origin.x.value() - self.state.offset.x.value(),
                ),
                top: LengthPercentageAuto::length(
                    transform.origin.y.value() - self.state.offset.y.value(),
                ),
                right: LengthPercentageAuto::length(0.0),
                bottom: LengthPercentageAuto::length(0.0),
            },
            ..Default::default()
        };
        Description::new::<UniformListItem>()
            .id(transform.index as u64)
            .diff_key(UniformItemKey {
                index: transform.index,
                origin: Point {
                    x: transform.origin.x - self.state.offset.x,
                    y: transform.origin.y - self.state.offset.y,
                },
                size: transform.size,
            })
            .style(style)
            .child((self.render_item)(transform.index))
    }

    pub fn describe(mut self) -> Description {
        let content_size = self.state.content_size();
        if let Some(handle) = &self.scroll_handle {
            handle.set_item_extent(self.state.item_size.height, self.state.item_count);
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
        let realized = self.state.realized_range();
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
        let key = UniformListKey {
            style: self.style.clone(),
            item_count: self.state.item_count,
            item_size: self.state.item_size,
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

struct UniformListItem;

impl Element for UniformList {
    fn into_description(self) -> Description {
        self.describe()
    }
}
impl Styled for UniformList {
    fn style(&mut self) -> &mut DivStyle {
        &mut self.style
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::div::div;
    use std::cell::RefCell;
    use std::rc::Rc;
    #[test]
    fn uniform_list_lowers_only_realized_items_and_retains_scroll_metadata() {
        let handle = ScrollHandle::new();
        let description = uniform_list(10_000, Size::pixels(100.0, 20.0), |index| {
            div().h(20.0).child(index.to_string())
        })
        .viewport(Size::pixels(100.0, 80.0))
        .track_scroll(&handle)
        .describe();
        assert_eq!(description.child_descriptions().len(), 5);
        assert_eq!(
            description.child_descriptions()[0].element_id(),
            Some(&ElementId::Integer(0))
        );
        assert!(description.is_boundary());
        assert!(description.clips_children());
    }

    #[test]
    fn processor_form_builds_rows_and_records_uniform_extent() {
        let requested_ranges = Rc::new(RefCell::new(Vec::new()));
        let observed_ranges = requested_ranges.clone();
        let handle = ScrollHandle::new();
        let description = uniform_list("messages", 10, move |range| {
            observed_ranges.borrow_mut().push(range.clone());
            vec![div().h(20.0).child(range.start.to_string())]
        })
        .viewport(Size::pixels(100.0, 40.0))
        .track_scroll(&handle)
        .describe();

        assert_eq!(description.element_id(), Some(&ElementId::from("messages")));
        assert_eq!(description.child_descriptions().len(), 3);
        assert_eq!(requested_ranges.borrow().as_slice(), [0..1, 1..2, 2..3]);
        assert!(handle.scroll_to_item(2, crate::scroll::ScrollStrategy::Top));
        assert_eq!(handle.offset().y, Pixels(-40.0));
    }
}
