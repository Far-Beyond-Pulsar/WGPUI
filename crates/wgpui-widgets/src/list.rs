//! Retained vertical list elements.

use std::any::Any;

use crate::div::interactivity::style::{DivStyle, classify_style_change};
use crate::scroll::ScrollHandle;
use crate::styled::Styled;
use wgpui_core::element::{Element, IntoElement};
use wgpui_core::geometry::{Bounds, Pixels, Point, Size};
use wgpui_core::invalidation::axes::Invalidation;
use wgpui_core::reconcile::description::{Description, ElementId};
use wgpui_core::reconcile::diff_key::ReconcileKey;
use wgpui_layout::taffy_tree::{Display, FlexDirection, LayoutStyle};

pub mod h_list;
pub mod uniform_list;
pub mod virtual_list;

/// State that is owned by a general list when it is not connected to a
/// [`ScrollHandle`]. A handle is the authoritative owner once one is tracked.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ListState {
    pub item_count: usize,
    pub viewport: Size<Pixels>,
    pub offset: Point<Pixels>,
}

impl ListState {
    pub fn set_viewport(&mut self, viewport: Size<Pixels>) {
        self.viewport = viewport;
    }

    pub fn set_offset(&mut self, offset: Point<Pixels>) {
        self.offset = offset;
    }
}

#[derive(Clone, Debug, PartialEq)]
struct ListKey {
    style: DivStyle,
    keys: Vec<ElementId>,
}

impl ReconcileKey for ListKey {
    fn compare(&self, previous: &dyn ReconcileKey) -> Invalidation {
        let Some(previous) = previous.as_any().downcast_ref::<Self>() else {
            return Invalidation::all();
        };
        let mut axes = classify_style_change(&self.style, &previous.style);
        if self.keys != previous.keys {
            axes |= Invalidation::LAYOUT;
        }
        axes
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Clone, Debug, PartialEq)]
struct ListItemKey {
    key: ElementId,
}

impl ReconcileKey for ListItemKey {
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

/// A general vertical list. All rows participate in normal Taffy layout; use
/// [`uniform_list`] or [`virtual_list`] when the row count requires
/// virtualization.
type ListKeyFunction<T> = Box<dyn FnMut(&T, usize) -> ElementId>;

pub struct List<T> {
    element_id: Option<ElementId>,
    style: DivStyle,
    state: ListState,
    scroll_handle: Option<ScrollHandle>,
    items: Vec<T>,
    render_item: Box<dyn FnMut(&T) -> Description>,
    key_item: ListKeyFunction<T>,
}

/// Construct a retained general vertical list.
pub fn list<T, I, F>(items: impl IntoIterator<Item = T>, mut render_item: F) -> List<T>
where
    T: 'static,
    I: IntoElement + 'static,
    F: FnMut(&T) -> I + 'static,
{
    let items = items.into_iter().collect::<Vec<_>>();
    let item_count = items.len();
    List {
        element_id: None,
        style: DivStyle::default(),
        state: ListState {
            item_count,
            ..ListState::default()
        },
        scroll_handle: None,
        items,
        render_item: Box::new(move |item| render_item(item).into_description()),
        key_item: Box::new(|_, index| ElementId::from(index)),
    }
}

impl<T: 'static> List<T> {
    pub fn id(mut self, element_id: impl Into<ElementId>) -> Self {
        self.element_id = Some(element_id.into());
        self
    }

    /// Supply a stable key for each row. Keys must be unique within this list.
    pub fn keyed_by<K, F>(mut self, mut key_item: F) -> Self
    where
        K: Into<ElementId>,
        F: FnMut(&T) -> K + 'static,
    {
        self.key_item = Box::new(move |item, _index| key_item(item).into());
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

    pub fn state(&self) -> &ListState {
        &self.state
    }

    fn row_description(&mut self, index: usize, key: ElementId) -> Description {
        Description::new::<ListItem>()
            .id(key.clone())
            .diff_key(ListItemKey { key })
            .style(LayoutStyle {
                display: Display::Flex,
                flex_direction: FlexDirection::Column,
                flex_shrink: 0.0,
                ..LayoutStyle::default()
            })
            .child((self.render_item)(&self.items[index]))
    }

    pub fn describe(mut self) -> Description {
        self.state.item_count = self.items.len();
        let keys = self
            .items
            .iter()
            .enumerate()
            .map(|(index, item)| (self.key_item)(item, index))
            .collect::<Vec<_>>();
        let children = keys
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, key)| self.row_description(index, key))
            .collect::<Vec<_>>();

        let mut layout_style = self.style.layout.clone();
        layout_style.display = Display::Flex;
        layout_style.flex_direction = FlexDirection::Column;
        layout_style.flex_shrink = 0.0;

        let offset = self
            .scroll_handle
            .as_ref()
            .map_or(self.state.offset, ScrollHandle::offset);
        self.state.offset = offset;
        let mut description = Description::new::<Self>()
            .diff_key(ListKey {
                style: self.style.clone(),
                keys,
            })
            .style(layout_style)
            .scroll_offset([offset.x.value(), offset.y.value()])
            .clip_children()
            .children(children);

        if let Some(handle) = self.scroll_handle.as_ref() {
            let handle_for_layout = handle.clone();
            let initial_viewport = self.state.viewport;
            description = description.on_layout_with_content_changed(move |bounds, content| {
                let viewport = Bounds::new(
                    Point::new(Pixels(bounds.x), Pixels(bounds.y)),
                    Size::pixels(bounds.width, bounds.height),
                );
                let content_size = Size::pixels(
                    content.width.max(viewport.size.width.value()),
                    content.height.max(viewport.size.height.value()),
                );
                let viewport =
                    if viewport.size == Size::default() && initial_viewport != Size::default() {
                        Bounds::new(Point::default(), initial_viewport)
                    } else {
                        viewport
                    };
                handle_for_layout.set_viewport(viewport, content_size)
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
        description
    }
}

struct ListItem;

impl<T: 'static> Element for List<T> {
    fn into_description(self) -> Description {
        self.describe()
    }
}

impl<T> Styled for List<T> {
    fn style(&mut self) -> &mut DivStyle {
        &mut self.style
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::div::div;
    use wgpui_core::reconcile::plan::NodeOutcome;
    use wgpui_core::reconcile::reconciler::Reconciler;
    use wgpui_layout::taffy_tree::{LayoutTree, definite};

    fn keyed_description(order: &[u64]) -> Description {
        list(order.iter().copied(), |item| {
            div().h(20.0).child(item.to_string())
        })
        .keyed_by(|item| *item)
        .describe()
    }

    #[test]
    fn empty_list_is_a_real_empty_retained_container() {
        let description = list(Vec::<u32>::new(), |_| div()).describe();
        assert_eq!(description.type_name(), std::any::type_name::<List<u32>>());
        assert!(description.child_descriptions().is_empty());
        assert!(description.key().is_some());
    }

    #[test]
    fn keyed_reorder_retains_row_instance_and_state_scope() {
        let mut reconciler = Reconciler::new();
        let mut layout = LayoutTree::new();
        let first = reconciler
            .reconcile(keyed_description(&[10, 20]), &mut layout)
            .expect("first list frame reconciles");
        let first_rows = first.nodes_at_depth(1);
        assert!(
            first_rows
                .iter()
                .all(|node| matches!(node.outcome, NodeOutcome::Rebuilt(_)))
        );

        let second = reconciler
            .reconcile(keyed_description(&[20, 10]), &mut layout)
            .expect("reordered list frame reconciles");
        let second_rows = second.nodes_at_depth(1);
        assert_eq!(first_rows[0].instance, second_rows[1].instance);
        assert_eq!(first_rows[0].state, second_rows[1].state);
        assert_eq!(first_rows[1].instance, second_rows[0].instance);
        assert_eq!(first_rows[1].state, second_rows[0].state);
    }

    #[test]
    fn tracked_list_updates_scroll_extent_from_resolved_content() {
        let handle = ScrollHandle::new();
        let description = list([1_u32, 2], |_| div().h(60.0))
            .w(100.0)
            .h(40.0)
            .track_scroll(&handle)
            .describe();
        let mut reconciler = Reconciler::new();
        let mut layout = LayoutTree::new();
        let mut plan = reconciler
            .reconcile(description, &mut layout)
            .expect("list reconciles");
        let root = plan.root().expect("list root").layout_node;
        layout
            .compute_layout(root, definite(100.0, 40.0))
            .expect("list lays out");
        let bounds = layout.layout_of(root).expect("list bounds");
        let mut callback = plan.take_layout_callback(0).expect("scroll callback");
        callback.apply_with_content(
            bounds,
            wgpui_layout::taffy_tree::LayoutRect {
                x: 0.0,
                y: 0.0,
                width: 100.0,
                height: 120.0,
            },
        );
        assert_eq!(handle.content_size(), Size::pixels(100.0, 120.0));
        assert_eq!(handle.max_offset(), Size::pixels(0.0, 80.0));
    }
}
