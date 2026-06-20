use super::retained_helpers::{paint_retained_view, prepaint_retained_view};
use crate::{
    AnyElement,
    App,
    Bounds,
    Element,
    ElementId,
    Entity,
    GlobalElementId,
    InspectorElementId,
    IntoElement,
    LayoutId,
    Pixels,
    Render,
    Window,
};

impl<V: Render> Element for Entity<V> {
    type RequestLayoutState = Option<AnyElement>;
    type PrepaintState = Option<AnyElement>;

    fn id(&self) -> Option<ElementId> {
        Some(ElementId::View(self.entity_id()))
    }

    fn source_location(&self) -> Option<&'static std::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let caching_disabled = window.is_inspector_picking(cx);
        if !caching_disabled
            && let Some(layout_id) = window.cached_layout_for_view(self.entity_id())
        {
            return (layout_id, None);
        }

        let mut element = self.update(cx, |view, cx| view.render(window, cx).into_any_element());
        let layout_id = window.with_rendered_view(self.entity_id(), |window| {
            element.request_layout(window, cx)
        });
        if !caching_disabled {
            window.cache_layout_for_view(self.entity_id(), layout_id);
        }
        (layout_id, Some(element))
    }

    fn prepaint(
        &mut self,
        global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        element: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) -> Self::PrepaintState {
        window.set_view_id(self.entity_id());
        let Some(global_id) = global_id else {
            if let Some(mut element) = element.take() {
                element.prepaint(window, cx);
                return Some(element);
            }
            return None;
        };
        let view_id = self.entity_id();
        let refresh_started = std::time::Instant::now();
        let prepainted = window.with_rendered_view(view_id, |window| {
            prepaint_retained_view(
                view_id,
                global_id,
                bounds,
                element,
                window,
                cx,
                |window, cx| self.update(cx, |view, cx| view.render(window, cx).into_any_element()),
                false,
            )
        });
        window.record_view_refresh(view_id, bounds, refresh_started.elapsed());
        prepainted
    }

    fn paint(
        &mut self,
        global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _: Bounds<Pixels>,
        _: &mut Self::RequestLayoutState,
        element: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        paint_retained_view(self.entity_id(), global_id, element, window, cx, false);
    }
}

impl<V: 'static + Render> IntoElement for Entity<V> {
    type Element = Entity<V>;

    fn into_element(self) -> Self::Element {
        self
    }
}
