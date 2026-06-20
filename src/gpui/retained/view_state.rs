use std::ops::Range;

use collections::FxHashSet;

use crate::{
    Bounds,
    ContentMask,
    EntityId,
    GlobalElementId,
    PaintIndex,
    Pixels,
    PrepaintStateIndex,
    TextStyle,
};

#[derive(Clone)]
pub(crate) struct ViewState {
    accessed_entities: FxHashSet<EntityId>,
    prepaint_range: Range<PrepaintStateIndex>,
    paint_range: Range<PaintIndex>,
    retained_node_ids: Vec<GlobalElementId>,
    cache_key: ViewCacheKey,
}

impl ViewState {
    pub(crate) fn new(
        accessed_entities: FxHashSet<EntityId>,
        prepaint_range: Range<PrepaintStateIndex>,
        paint_range: Range<PaintIndex>,
        retained_node_ids: Vec<GlobalElementId>,
        cache_key: ViewCacheKey,
    ) -> Self {
        Self {
            accessed_entities,
            prepaint_range,
            paint_range,
            retained_node_ids,
            cache_key,
        }
    }

    pub(crate) fn accessed_entities(&self) -> &FxHashSet<EntityId> {
        &self.accessed_entities
    }

    pub(crate) fn prepaint_range(&self) -> &Range<PrepaintStateIndex> {
        &self.prepaint_range
    }

    pub(crate) fn set_prepaint_range(&mut self, prepaint_range: Range<PrepaintStateIndex>) {
        self.prepaint_range = prepaint_range;
    }

    pub(crate) fn paint_range(&self) -> &Range<PaintIndex> {
        &self.paint_range
    }

    pub(crate) fn set_paint_range(&mut self, paint_range: Range<PaintIndex>) {
        self.paint_range = paint_range;
    }

    pub(crate) fn retained_node_ids(&self) -> &[GlobalElementId] {
        &self.retained_node_ids
    }

    pub(crate) fn cache_key(&self) -> &ViewCacheKey {
        &self.cache_key
    }
}

#[derive(Clone, Default)]
pub(crate) struct ViewCacheKey {
    bounds: Bounds<Pixels>,
    content_mask: ContentMask<Pixels>,
    text_style: TextStyle,
}

impl ViewCacheKey {
    pub(crate) fn new(
        bounds: Bounds<Pixels>,
        content_mask: ContentMask<Pixels>,
        text_style: TextStyle,
    ) -> Self {
        Self {
            bounds,
            content_mask,
            text_style,
        }
    }

    pub(crate) fn matches(
        &self,
        bounds: Bounds<Pixels>,
        content_mask: ContentMask<Pixels>,
        text_style: TextStyle,
    ) -> bool {
        self.bounds == bounds && self.content_mask == content_mask && self.text_style == text_style
    }
}
