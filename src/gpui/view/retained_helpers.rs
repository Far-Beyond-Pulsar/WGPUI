use std::mem;

use crate::retained::{ViewCacheKey, ViewState};
use crate::{AnyElement, App, Bounds, EntityId, GlobalElementId, PaintIndex, Pixels, Window};

pub(super) fn prepaint_retained_view(
    view_id: EntityId,
    global_id: &GlobalElementId,
    bounds: Bounds<Pixels>,
    element: &mut Option<AnyElement>,
    window: &mut Window,
    cx: &mut App,
    render: impl FnOnce(&mut Window, &mut App) -> AnyElement,
    require_empty_dirty_views: bool,
) -> Option<AnyElement> {
    if let Some(mut element) = element.take() {
        return window.with_element_state::<ViewState, _>(global_id, |_, window| {
            let content_mask = window.content_mask();
            let text_style = window.text_style();
            let prepaint_start = window.prepaint_index();
            element.prepaint(window, cx);
            let prepaint_end = window.prepaint_index();
            let mut accessed_entities = cx.entities.accessed_entities.borrow().clone();
            accessed_entities.insert(view_id);
            cx.entities.accessed_entities.borrow_mut().insert(view_id);

            (
                Some(element),
                ViewState::new(
                    accessed_entities,
                    prepaint_start..prepaint_end,
                    PaintIndex::default()..PaintIndex::default(),
                    window.retained_node_ids_for_view(view_id),
                    ViewCacheKey::new(bounds, content_mask, text_style),
                ),
            )
        });
    }

    window.with_element_state::<ViewState, _>(global_id, |element_state, window| {
        let content_mask = window.content_mask();
        let text_style = window.text_style();

        if let Some(mut element_state) = element_state {
            let can_reuse_prepaint =
                element_state
                    .cache_key()
                    .matches(bounds, content_mask, text_style.clone())
                    && (!require_empty_dirty_views || window.dirty_views.is_empty())
                    && window.should_skip_view(view_id)
                    && !window.refreshing;
            if can_reuse_prepaint {
                // Carry the reused subtree's fibers forward, then replay the
                // cached prepaint without re-walking the element tree.
                window.update_retained_nodes(element_state.retained_node_ids());
                let prepaint_start = window.prepaint_index();
                window.reuse_prepaint(element_state.prepaint_range().clone());
                cx.entities
                    .extend_accessed(element_state.accessed_entities());
                let prepaint_end = window.prepaint_index();
                element_state.set_prepaint_range(prepaint_start..prepaint_end);

                return (None, element_state);
            }
        }

        let refreshing = mem::replace(&mut window.refreshing, true);
        let prepaint_start = window.prepaint_index();
        let (mut element, mut accessed_entities) = cx.detect_accessed_entities(|cx| {
            let mut element = render(window, cx);
            element.layout_as_root(bounds.size.into(), window, cx);
            element.prepaint_at(bounds.origin, window, cx);
            element
        });

        let prepaint_end = window.prepaint_index();
        window.refreshing = refreshing;
        accessed_entities.insert(view_id);
        cx.entities.accessed_entities.borrow_mut().insert(view_id);

        (
            Some(element),
            ViewState::new(
                accessed_entities,
                prepaint_start..prepaint_end,
                PaintIndex::default()..PaintIndex::default(),
                window.retained_node_ids_for_view(view_id),
                ViewCacheKey::new(bounds, content_mask, text_style),
            ),
        )
    })
}

pub(super) fn paint_retained_view(
    view_id: EntityId,
    global_id: Option<&GlobalElementId>,
    element: &mut Option<AnyElement>,
    window: &mut Window,
    cx: &mut App,
    honor_inspector_disable: bool,
) {
    let Some(global_id) = global_id else {
        if let Some(element) = element {
            window.with_rendered_view(view_id, |window| {
                window.begin_view_chunk(view_id);
                element.paint(window, cx);
                window.end_view_chunk();
            });
        }
        return;
    };
    window.with_rendered_view(view_id, |window| {
        let caching_disabled = honor_inspector_disable && window.is_inspector_picking(cx);
        if !caching_disabled {
            window.with_element_state::<ViewState, _>(global_id, |element_state, window| {
                let mut element_state = element_state.unwrap();

                window.begin_view_chunk(view_id);
                let paint_start = window.paint_index();

                if let Some(element) = element {
                    let refreshing = mem::replace(&mut window.refreshing, true);
                    element.paint(window, cx);
                    window.refreshing = refreshing;
                } else {
                    window.reuse_paint(view_id, element_state.paint_range().clone());
                }

                let paint_end = window.paint_index();
                window.end_view_chunk();
                element_state.set_paint_range(paint_start..paint_end);

                ((), element_state)
            })
        } else {
            window.begin_view_chunk(view_id);
            if let Some(element) = element {
                element.paint(window, cx);
            }
            window.end_view_chunk();
        }
    });
}
