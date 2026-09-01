//! Retained uniformly-sized horizontal lists.

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
pub struct HorizontalItemTransform {
    pub index: usize,
    pub origin: Point<Pixels>,
    pub size: Size<Pixels>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HListState {
    pub item_count: usize,
    pub item_size: Size<Pixels>,
    pub viewport: Size<Pixels>,
    pub offset: Point<Pixels>,
    pub overscan: usize,
}

impl HListState {
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
            self.item_size.width.value() * self.item_count as f32,
            self.item_size.height.value(),
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
        if self.item_count == 0 || self.item_size.width <= Pixels::ZERO {
            return 0..0;
        }
        let first = ((-self.offset.x.value() / self.item_size.width.value()).floor() as usize)
            .saturating_sub(self.overscan);
        let last = ((-self.offset.x.value() + self.viewport.width.value())
            / self.item_size.width.value())
        .ceil() as usize
            + self.overscan;
        first.min(self.item_count)..last.min(self.item_count)
    }

    pub fn transforms(&self) -> Vec<HorizontalItemTransform> {
        self.realized_range()
            .map(|index| HorizontalItemTransform {
                index,
                origin: Point {
                    x: self.offset.x + self.item_size.width.scaled(index as f32),
                    y: self.offset.y,
                },
                size: self.item_size,
            })
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq)]
struct HListKey {
    style: DivStyle,
    item_count: usize,
    item_size: Size<Pixels>,
    realized: Range<usize>,
}

impl ReconcileKey for HListKey {
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
struct HListItemKey {
    index: usize,
    origin: Point<Pixels>,
    size: Size<Pixels>,
}

impl ReconcileKey for HListItemKey {
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

/// A retained horizontal list with uniform item extents.
pub struct HList {
    element_id: Option<ElementId>,
    style: DivStyle,
    state: HListState,
    scroll_handle: Option<ScrollHandle>,
    render_item: Box<dyn FnMut(usize) -> Description>,
}

pub trait HListArguments {
    fn build(self) -> HList;
}

impl<F, I> HListArguments for (usize, Size<Pixels>, F)
where
    F: FnMut(usize) -> I + 'static,
    I: IntoElement + 'static,
{
    fn build(self) -> HList {
        HList::new(self.0, self.1, self.2)
    }
}

pub fn h_list<A, B, C>(first: A, second: B, render_item: C) -> HList
where
    (A, B, C): HListArguments,
{
    (first, second, render_item).build()
}

impl HList {
    pub fn new<F, I>(item_count: usize, item_size: Size<Pixels>, mut render_item: F) -> Self
    where
        F: FnMut(usize) -> I + 'static,
        I: IntoElement + 'static,
    {
        Self {
            element_id: None,
            style: DivStyle::default(),
            state: HListState::new(item_count, item_size),
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

    pub fn state(&self) -> &HListState {
        &self.state
    }

    fn item_description(&mut self, transform: HorizontalItemTransform) -> Description {
        let style = wgpui_layout::taffy_tree::LayoutStyle {
            position: Position::Absolute,
            size: LayoutSize {
                width: Dimension::length(transform.size.width.value()),
                height: Dimension::percent(1.0),
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
        Description::new::<HListItem>()
            .id(transform.index as u64)
            .diff_key(HListItemKey {
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
        let mut description = Description::new::<Self>()
            .diff_key(HListKey {
                style: self.style.clone(),
                item_count: self.state.item_count,
                item_size: self.state.item_size,
                realized,
            })
            .style(self.style.layout.clone())
            .scroll_offset([offset.x.value(), offset.y.value()])
            .clip_children()
            .children(children);
        if let Some(handle) = self.scroll_handle.as_ref() {
            let handle_for_layout = handle.clone();
            description = description.on_layout_changed(move |bounds| {
                handle_for_layout.set_viewport(
                    Bounds::new(
                        Point::new(Pixels(bounds.x), Pixels(bounds.y)),
                        Size::pixels(bounds.width, bounds.height),
                    ),
                    content_size,
                )
            });
            let handle_for_input = handle.clone();
            description = description
                .interaction(
                    wgpui_core::reconcile::description::DescriptionInteraction::new(
                        move |event, _, _| match event {
                            wgpui_core::window::InputEvent::Scroll(event) => {
                                handle_for_input.scroll_wheel(event)
                            }
                            _ => wgpui_core::window::EventResult::IGNORED,
                        },
                    ),
                )
                .boundary();
        }
        if let Some(element_id) = self.element_id {
            description = description.id(element_id);
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

struct HListItem;

impl Element for HList {
    fn into_description(self) -> Description {
        self.describe()
    }
}

impl Styled for HList {
    fn style(&mut self) -> &mut DivStyle {
        &mut self.style
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::div::div;

    #[test]
    fn horizontal_state_realizes_and_clamps_by_viewport() {
        let mut state = HListState::new(10, Size::pixels(20.0, 30.0));
        state.set_viewport(Size::pixels(60.0, 30.0));
        state.set_offset(Point::new(Pixels(-500.0), Pixels::ZERO));
        assert_eq!(state.offset.x, Pixels(-140.0));
        assert_eq!(state.realized_range(), 6..10);
    }

    #[test]
    fn horizontal_list_lowers_empty_and_realized_items() {
        let empty = h_list(0, Size::pixels(20.0, 30.0), |_| div()).describe();
        assert!(empty.child_descriptions().is_empty());

        let description = h_list(10, Size::pixels(20.0, 30.0), |index| {
            div().child(index.to_string())
        })
        .viewport(Size::pixels(60.0, 30.0))
        .describe();
        assert_eq!(description.child_descriptions().len(), 4);
        assert_eq!(
            description.child_descriptions()[0].element_id(),
            Some(&ElementId::Integer(0))
        );
    }
}
