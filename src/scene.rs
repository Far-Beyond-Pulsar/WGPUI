#![expect(
    dead_code,
    reason = "scene keeps dormant damage and changed-range APIs for staged rendering performance work"
)]

use std::fmt::Debug;
use std::iter::Peekable;
use std::ops::{Add, Range, Sub};
use std::slice;

use collections::FxHashMap;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::bounds_tree::BoundsTree;
use crate::platform::cross::surface_registry::SurfaceId;
use crate::{
    AtlasTextureId,
    AtlasTile,
    Background,
    Bounds,
    ContentMask,
    Corners,
    Edges,
    EntityId,
    Hsla,
    Pixels,
    Point,
    Radians,
    ScaledPixels,
    Size,
    TextColor,
    point,
};

#[allow(non_camel_case_types, unused)]
pub(crate) type PathVertex_ScaledPixels = PathVertex<ScaledPixels>;

pub(crate) type DrawOrder = u32;

/// A per-view slice into each of the Scene's primitive vectors, plus a `dirty`
/// flag — the foundation for incremental sort / damage / GPU diff-uploads. A
/// view that didn't change keeps `dirty = false` and its primitives are reused.
#[derive(Debug, Clone)]
pub(crate) struct SceneChunk {
    #[allow(dead_code)]
    pub view_id: EntityId,
    pub shadows: Range<usize>,
    pub backdrop_blurs: Range<usize>,
    pub quads: Range<usize>,
    pub paths: Range<usize>,
    pub underlines: Range<usize>,
    pub monochrome_sprites: Range<usize>,
    pub polychrome_sprites: Range<usize>,
    pub surfaces: Range<usize>,
    pub paint_operations: Range<usize>,
    pub dirty: bool,
}

/// Per-primitive-type ranges that changed this frame (merged), for diff uploads.
#[derive(Debug, Default)]
pub(crate) struct ChangedRanges {
    pub shadows: Vec<Range<usize>>,
    pub backdrop_blurs: Vec<Range<usize>>,
    pub quads: Vec<Range<usize>>,
    pub paths: Vec<Range<usize>>,
    pub underlines: Vec<Range<usize>>,
    pub monochrome_sprites: Vec<Range<usize>>,
    pub polychrome_sprites: Vec<Range<usize>>,
    pub surfaces: Vec<Range<usize>>,
}

fn merge_ranges(ranges: &mut Vec<Range<usize>>) {
    ranges.retain(|range| range.start < range.end);
    ranges.sort_by_key(|range| range.start);

    let mut merged: Vec<Range<usize>> = Vec::with_capacity(ranges.len());
    for range in ranges.drain(..) {
        if let Some(last_range) = merged.last_mut()
            && range.start <= last_range.end
        {
            last_range.end = last_range.end.max(range.end);
            continue;
        }

        merged.push(range);
    }

    *ranges = merged;
}

impl SceneChunk {
    fn new(view_id: EntityId, scene: &Scene) -> Self {
        Self {
            view_id,
            shadows: scene.shadows.len()..scene.shadows.len(),
            backdrop_blurs: scene.backdrop_blurs.len()..scene.backdrop_blurs.len(),
            quads: scene.quads.len()..scene.quads.len(),
            paths: scene.paths.len()..scene.paths.len(),
            underlines: scene.underlines.len()..scene.underlines.len(),
            monochrome_sprites: scene.monochrome_sprites.len()..scene.monochrome_sprites.len(),
            polychrome_sprites: scene.polychrome_sprites.len()..scene.polychrome_sprites.len(),
            surfaces: scene.surfaces.len()..scene.surfaces.len(),
            paint_operations: scene.paint_operations.len()..scene.paint_operations.len(),
            dirty: true,
        }
    }

    fn end_at(&mut self, scene: &Scene) {
        self.shadows.end = scene.shadows.len();
        self.backdrop_blurs.end = scene.backdrop_blurs.len();
        self.quads.end = scene.quads.len();
        self.paths.end = scene.paths.len();
        self.underlines.end = scene.underlines.len();
        self.monochrome_sprites.end = scene.monochrome_sprites.len();
        self.polychrome_sprites.end = scene.polychrome_sprites.len();
        self.surfaces.end = scene.surfaces.len();
        self.paint_operations.end = scene.paint_operations.len();
    }
}

#[derive(Default)]
pub(crate) struct Scene {
    pub(crate) paint_operations: Vec<PaintOperation>,
    primitive_bounds: BoundsTree<ScaledPixels>,
    layer_stack: Vec<DrawOrder>,
    pub(crate) shadows: Vec<Shadow>,
    pub(crate) backdrop_blurs: Vec<BackdropBlur>,
    pub(crate) quads: Vec<Quad>,
    pub(crate) paths: Vec<Path<ScaledPixels>>,
    pub(crate) underlines: Vec<Underline>,
    pub(crate) monochrome_sprites: Vec<MonochromeSprite>,
    pub(crate) polychrome_sprites: Vec<PolychromeSprite>,
    pub(crate) surfaces: Vec<PaintSurface>,
    damage_rects: Vec<Bounds<ScaledPixels>>,
    pub(crate) chunks: Vec<SceneChunk>,
    pub(crate) active_chunk: Option<usize>,
    pub(crate) chunk_map: FxHashMap<EntityId, usize>,
    pub(crate) generation: u64,
    batch_count: u32,
}

impl Scene {
    pub fn clear(&mut self) {
        self.paint_operations.clear();
        self.primitive_bounds.clear();
        self.layer_stack.clear();
        self.paths.clear();
        self.shadows.clear();
        self.backdrop_blurs.clear();
        self.quads.clear();
        self.underlines.clear();
        self.monochrome_sprites.clear();
        self.polychrome_sprites.clear();
        self.surfaces.clear();
        self.damage_rects.clear();
        self.chunks.clear();
        self.active_chunk = None;
        self.chunk_map.clear();
        self.batch_count = 0;
    }

    /// Begin recording a view's primitives into a (possibly reused) chunk.
    pub fn begin_view_chunk(&mut self, view_id: EntityId) {
        let chunk = SceneChunk::new(view_id, self);
        let chunk_index = if let Some(&chunk_index) = self.chunk_map.get(&view_id) {
            if let Some(existing_chunk) = self.chunks.get_mut(chunk_index) {
                *existing_chunk = chunk;
            }
            chunk_index
        } else {
            let chunk_index = self.chunks.len();
            self.chunks.push(chunk);
            self.chunk_map.insert(view_id, chunk_index);
            chunk_index
        };

        self.active_chunk = Some(chunk_index);
    }

    /// Close the active chunk, recording where each primitive vector ended.
    pub fn end_view_chunk(&mut self) {
        if let Some(chunk_index) = self.active_chunk.take() {
            let mut chunk = if let Some(chunk) = self.chunks.get(chunk_index) {
                chunk.clone()
            } else {
                return;
            };
            chunk.end_at(self);
            if let Some(existing_chunk) = self.chunks.get_mut(chunk_index) {
                *existing_chunk = chunk;
            }
        }
    }

    /// Mark a reused (unchanged) view's chunk as clean.
    pub fn mark_chunk_clean(&mut self, view_id: EntityId) {
        if let Some(&chunk_index) = self.chunk_map.get(&view_id)
            && let Some(chunk) = self.chunks.get_mut(chunk_index)
        {
            chunk.dirty = false;
        }
    }

    /// Whether a view's chunk is dirty (defaults to dirty if unknown).
    pub fn is_chunk_dirty(&self, view_id: EntityId) -> bool {
        self.chunk_map
            .get(&view_id)
            .and_then(|&chunk_index| self.chunks.get(chunk_index))
            .is_none_or(|chunk| chunk.dirty)
    }

    /// A monotonically increasing counter bumped on scene mutations.
    pub fn scene_generation(&self) -> u64 {
        self.generation
    }

    /// How many chunks are currently dirty.
    pub fn dirty_chunk_count(&self) -> usize {
        self.chunks.iter().filter(|chunk| chunk.dirty).count()
    }

    /// The number of GPU draw batches in the most recently finished scene.
    pub fn batch_count(&self) -> u32 {
        self.batch_count
    }

    /// Count the GPU draw batches the current scene would produce.
    pub fn total_batch_count(&self) -> u32 {
        let mut counter = BatchCounter::default();
        let mut count = 0u32;
        while counter.advance(self) {
            count = count.saturating_add(1);
        }
        count
    }

    /// Whether any chunk changed this frame.
    pub fn has_changes(&self) -> bool {
        self.chunks.iter().any(|chunk| chunk.dirty)
    }

    /// The merged per-type ranges that changed this frame (for diff uploads).
    pub fn changed_ranges(&self) -> ChangedRanges {
        let mut ranges = ChangedRanges::default();

        for chunk in self.chunks.iter().filter(|chunk| chunk.dirty) {
            ranges.shadows.push(chunk.shadows.clone());
            ranges.backdrop_blurs.push(chunk.backdrop_blurs.clone());
            ranges.quads.push(chunk.quads.clone());
            ranges.paths.push(chunk.paths.clone());
            ranges.underlines.push(chunk.underlines.clone());
            ranges
                .monochrome_sprites
                .push(chunk.monochrome_sprites.clone());
            ranges
                .polychrome_sprites
                .push(chunk.polychrome_sprites.clone());
            ranges.surfaces.push(chunk.surfaces.clone());
        }

        merge_ranges(&mut ranges.shadows);
        merge_ranges(&mut ranges.backdrop_blurs);
        merge_ranges(&mut ranges.quads);
        merge_ranges(&mut ranges.paths);
        merge_ranges(&mut ranges.underlines);
        merge_ranges(&mut ranges.monochrome_sprites);
        merge_ranges(&mut ranges.polychrome_sprites);
        merge_ranges(&mut ranges.surfaces);

        ranges
    }

    pub fn len(&self) -> usize {
        self.paint_operations.len()
    }

    pub fn push_layer(&mut self, bounds: Bounds<ScaledPixels>) {
        let order = self.primitive_bounds.insert(bounds);
        self.layer_stack.push(order);
        self.paint_operations
            .push(PaintOperation::StartLayer(bounds));
    }

    pub fn pop_layer(&mut self) {
        self.layer_stack.pop();
        self.paint_operations.push(PaintOperation::EndLayer);
    }

    pub fn insert_primitive(&mut self, primitive: impl Into<Primitive>) {
        let mut primitive = primitive.into();
        let clipped_bounds = primitive
            .bounds()
            .intersect(&primitive.content_mask().bounds);

        if clipped_bounds.is_empty() {
            return;
        }

        let order = self
            .layer_stack
            .last()
            .copied()
            .unwrap_or_else(|| self.primitive_bounds.insert(clipped_bounds));
        match &mut primitive {
            Primitive::Shadow(shadow) => {
                shadow.order = order;
                self.shadows.push(shadow.clone());
            }
            Primitive::BackdropBlur(backdrop_blur) => {
                backdrop_blur.order = order;
                self.backdrop_blurs.push(backdrop_blur.clone());
            }
            Primitive::Quad(quad) => {
                quad.order = order;
                self.quads.push(quad.clone());
            }
            Primitive::Path(path) => {
                path.order = order;
                path.id = PathId(self.paths.len());
                self.paths.push(path.clone());
            }
            Primitive::Underline(underline) => {
                underline.order = order;
                self.underlines.push(underline.clone());
            }
            Primitive::MonochromeSprite(sprite) => {
                sprite.order = order;
                self.monochrome_sprites.push(sprite.clone());
            }
            Primitive::PolychromeSprite(sprite) => {
                sprite.order = order;
                self.polychrome_sprites.push(sprite.clone());
            }
            Primitive::Surface(surface) => {
                surface.order = order;
                self.surfaces.push(surface.clone());
            }
        }
        self.paint_operations
            .push(PaintOperation::Primitive(primitive));
    }

    pub fn replay(&mut self, range: Range<usize>, prev_scene: &Scene) {
        for operation in &prev_scene.paint_operations[range] {
            match operation {
                PaintOperation::Primitive(primitive) => self.insert_primitive(primitive.clone()),
                PaintOperation::StartLayer(bounds) => self.push_layer(*bounds),
                PaintOperation::EndLayer => self.pop_layer(),
            }
        }
    }

    pub fn finish(&mut self) {
        let has_dirty_chunks = self.chunks.iter().any(|chunk| chunk.dirty);
        self.finish_incremental();
        self.compute_damage_rects();
        for chunk in &mut self.chunks {
            chunk.dirty = false;
        }
        if has_dirty_chunks {
            self.generation += 1;
        }
        self.batch_count = self.total_batch_count();
    }

    /// Sort only the dirty chunks' primitive ranges when <30% of chunks changed,
    /// falling back to a full sort otherwise (or when there are no chunks).
    fn finish_incremental(&mut self) {
        let total_chunk_count = self.chunks.len();
        let dirty_chunk_count = self.dirty_chunk_count();

        if total_chunk_count == 0 || dirty_chunk_count * 10 >= total_chunk_count * 3 {
            self.sort_all_primitives();
        } else {
            for chunk in &mut self.chunks {
                if !chunk.dirty {
                    continue;
                }

                if let Some(shadows) = self.shadows.get_mut(chunk.shadows.clone()) {
                    shadows.sort_by_key(|shadow| shadow.order);
                }
                if let Some(backdrop_blurs) =
                    self.backdrop_blurs.get_mut(chunk.backdrop_blurs.clone())
                {
                    backdrop_blurs.sort_by_key(|backdrop_blur| backdrop_blur.order);
                }
                if let Some(quads) = self.quads.get_mut(chunk.quads.clone()) {
                    quads.sort_by_key(|quad| quad.order);
                }
                if let Some(paths) = self.paths.get_mut(chunk.paths.clone()) {
                    paths.sort_by_key(|path| path.order);
                }
                if let Some(underlines) = self.underlines.get_mut(chunk.underlines.clone()) {
                    underlines.sort_by_key(|underline| underline.order);
                }
                if let Some(monochrome_sprites) = self
                    .monochrome_sprites
                    .get_mut(chunk.monochrome_sprites.clone())
                {
                    monochrome_sprites.sort_by_key(|sprite| (sprite.order, sprite.tile.tile_id));
                }
                if let Some(polychrome_sprites) = self
                    .polychrome_sprites
                    .get_mut(chunk.polychrome_sprites.clone())
                {
                    polychrome_sprites.sort_by_key(|sprite| (sprite.order, sprite.tile.tile_id));
                }
                if let Some(surfaces) = self.surfaces.get_mut(chunk.surfaces.clone()) {
                    surfaces.sort_by_key(|surface| surface.order);
                }
            }
        }
    }

    /// Compute merged damage rectangles from the dirty chunks' primitive bounds.
    /// Dormant today (the renderer still does a full clear); consumed only when
    /// scissored/partial rendering is enabled.
    pub fn compute_damage_rects(&mut self) {
        self.damage_rects.clear();

        for chunk in &self.chunks {
            if !chunk.dirty {
                continue;
            }
            if let Some(bounds) = self.chunk_bounds(chunk) {
                self.damage_rects.push(bounds);
            }
        }

        self.merge_damage_rects();
    }

    pub fn damage_rects(&self) -> &[Bounds<ScaledPixels>] {
        &self.damage_rects
    }

    pub fn damage_area(&self) -> f32 {
        self.damage_rects
            .iter()
            .map(|bounds| bounds.size.width.0 * bounds.size.height.0)
            .sum()
    }

    pub fn full_redraw_needed(&self, viewport: Bounds<ScaledPixels>) -> bool {
        if self.chunks.is_empty() {
            return true;
        }
        let viewport_area = viewport.size.width.0 * viewport.size.height.0;
        self.damage_area() > viewport_area * 0.7
    }

    fn chunk_bounds(&self, chunk: &SceneChunk) -> Option<Bounds<ScaledPixels>> {
        let mut bounds = None;

        if let Some(shadows) = self.shadows.get(chunk.shadows.clone()) {
            for shadow in shadows {
                Self::include_bounds(&mut bounds, shadow.bounds);
            }
        }
        if let Some(backdrop_blurs) = self.backdrop_blurs.get(chunk.backdrop_blurs.clone()) {
            for backdrop_blur in backdrop_blurs {
                Self::include_bounds(&mut bounds, backdrop_blur.bounds);
            }
        }
        if let Some(quads) = self.quads.get(chunk.quads.clone()) {
            for quad in quads {
                Self::include_bounds(&mut bounds, quad.bounds);
            }
        }
        if let Some(paths) = self.paths.get(chunk.paths.clone()) {
            for path in paths {
                Self::include_bounds(&mut bounds, path.bounds);
            }
        }
        if let Some(underlines) = self.underlines.get(chunk.underlines.clone()) {
            for underline in underlines {
                Self::include_bounds(&mut bounds, underline.bounds);
            }
        }
        if let Some(monochrome_sprites) = self
            .monochrome_sprites
            .get(chunk.monochrome_sprites.clone())
        {
            for sprite in monochrome_sprites {
                Self::include_bounds(&mut bounds, sprite.bounds);
            }
        }
        if let Some(polychrome_sprites) = self
            .polychrome_sprites
            .get(chunk.polychrome_sprites.clone())
        {
            for sprite in polychrome_sprites {
                Self::include_bounds(&mut bounds, sprite.bounds);
            }
        }
        if let Some(surfaces) = self.surfaces.get(chunk.surfaces.clone()) {
            for surface in surfaces {
                Self::include_bounds(&mut bounds, surface.bounds);
            }
        }

        bounds
    }

    fn include_bounds(
        bounds: &mut Option<Bounds<ScaledPixels>>,
        primitive_bounds: Bounds<ScaledPixels>,
    ) {
        *bounds = Some(bounds.map_or(primitive_bounds, |bounds| bounds.union(&primitive_bounds)));
    }

    fn merge_damage_rects(&mut self) {
        let mut merged: Vec<Bounds<ScaledPixels>> = Vec::new();

        for mut bounds in self.damage_rects.drain(..) {
            let mut index = 0;
            while index < merged.len() {
                if bounds.intersects(&merged[index]) {
                    let existing_bounds = merged.remove(index);
                    bounds = bounds.union(&existing_bounds);
                    index = 0;
                } else {
                    index += 1;
                }
            }
            merged.push(bounds);
        }

        self.damage_rects = merged;
    }

    fn sort_all_primitives(&mut self) {
        self.shadows.sort_by_key(|shadow| shadow.order);
        self.backdrop_blurs
            .sort_by_key(|backdrop_blur| backdrop_blur.order);
        self.quads.sort_by_key(|quad| quad.order);
        self.paths.sort_by_key(|path| path.order);
        self.underlines.sort_by_key(|underline| underline.order);
        self.monochrome_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.tile_id));
        self.polychrome_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.tile_id));
        self.surfaces.sort_by_key(|surface| surface.order);
    }

    pub(crate) fn batches(&self) -> impl Iterator<Item = PrimitiveBatch<'_>> {
        BatchIterator {
            shadows: &self.shadows,
            shadows_start: 0,
            shadows_iter: self.shadows.iter().peekable(),
            backdrop_blurs: &self.backdrop_blurs,
            backdrop_blurs_start: 0,
            backdrop_blurs_iter: self.backdrop_blurs.iter().peekable(),
            quads: &self.quads,
            quads_start: 0,
            quads_iter: self.quads.iter().peekable(),
            paths: &self.paths,
            paths_start: 0,
            paths_iter: self.paths.iter().peekable(),
            underlines: &self.underlines,
            underlines_start: 0,
            underlines_iter: self.underlines.iter().peekable(),
            monochrome_sprites: &self.monochrome_sprites,
            monochrome_sprites_start: 0,
            monochrome_sprites_iter: self.monochrome_sprites.iter().peekable(),
            polychrome_sprites: &self.polychrome_sprites,
            polychrome_sprites_start: 0,
            polychrome_sprites_iter: self.polychrome_sprites.iter().peekable(),
            surfaces: &self.surfaces,
            surfaces_start: 0,
            surfaces_iter: self.surfaces.iter().peekable(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Default)]
pub(crate) enum PrimitiveKind {
    Shadow,
    BackdropBlur,
    #[default]
    Quad,
    Path,
    Underline,
    MonochromeSprite,
    PolychromeSprite,
    Surface,
}

pub(crate) enum PaintOperation {
    Primitive(Primitive),
    StartLayer(Bounds<ScaledPixels>),
    EndLayer,
}

#[derive(Clone)]
pub(crate) enum Primitive {
    Shadow(Shadow),
    BackdropBlur(BackdropBlur),
    Quad(Quad),
    Path(Path<ScaledPixels>),
    Underline(Underline),
    MonochromeSprite(MonochromeSprite),
    PolychromeSprite(PolychromeSprite),
    Surface(PaintSurface),
}

impl Primitive {
    pub fn bounds(&self) -> &Bounds<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.bounds,
            Primitive::BackdropBlur(backdrop_blur) => &backdrop_blur.bounds,
            Primitive::Quad(quad) => &quad.bounds,
            Primitive::Path(path) => &path.bounds,
            Primitive::Underline(underline) => &underline.bounds,
            Primitive::MonochromeSprite(sprite) => &sprite.bounds,
            Primitive::PolychromeSprite(sprite) => &sprite.bounds,
            Primitive::Surface(surface) => &surface.bounds,
        }
    }

    pub fn content_mask(&self) -> &ContentMask<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.content_mask,
            Primitive::BackdropBlur(backdrop_blur) => &backdrop_blur.content_mask,
            Primitive::Quad(quad) => &quad.content_mask,
            Primitive::Path(path) => &path.content_mask,
            Primitive::Underline(underline) => &underline.content_mask,
            Primitive::MonochromeSprite(sprite) => &sprite.content_mask,
            Primitive::PolychromeSprite(sprite) => &sprite.content_mask,
            Primitive::Surface(surface) => &surface.content_mask,
        }
    }
}

/// Counts the GPU draw batches a finished scene yields, by replaying the same
/// min-order-by-kind selection as `BatchIterator` (used for instrumentation).
#[derive(Default)]
struct BatchCounter {
    shadows_index: usize,
    backdrop_blurs_index: usize,
    quads_index: usize,
    paths_index: usize,
    underlines_index: usize,
    monochrome_sprites_index: usize,
    polychrome_sprites_index: usize,
    surfaces_index: usize,
}

impl BatchCounter {
    fn advance(&mut self, scene: &Scene) -> bool {
        let mut orders_and_kinds = [
            (
                scene.shadows.get(self.shadows_index).map(|s| s.order),
                PrimitiveKind::Shadow,
            ),
            (
                scene
                    .backdrop_blurs
                    .get(self.backdrop_blurs_index)
                    .map(|b| b.order),
                PrimitiveKind::BackdropBlur,
            ),
            (
                scene.quads.get(self.quads_index).map(|q| q.order),
                PrimitiveKind::Quad,
            ),
            (
                scene.paths.get(self.paths_index).map(|p| p.order),
                PrimitiveKind::Path,
            ),
            (
                scene.underlines.get(self.underlines_index).map(|u| u.order),
                PrimitiveKind::Underline,
            ),
            (
                scene
                    .monochrome_sprites
                    .get(self.monochrome_sprites_index)
                    .map(|s| s.order),
                PrimitiveKind::MonochromeSprite,
            ),
            (
                scene
                    .polychrome_sprites
                    .get(self.polychrome_sprites_index)
                    .map(|s| s.order),
                PrimitiveKind::PolychromeSprite,
            ),
            (
                scene.surfaces.get(self.surfaces_index).map(|s| s.order),
                PrimitiveKind::Surface,
            ),
        ];
        orders_and_kinds.sort_by_key(|(order, kind)| (order.unwrap_or(u32::MAX), *kind));

        let first = orders_and_kinds[0];
        let second = orders_and_kinds[1];
        if first.0.is_none() {
            return false;
        }

        let batch_kind = first.1;
        let max_order_and_kind = (second.0.unwrap_or(u32::MAX), second.1);

        match batch_kind {
            PrimitiveKind::Shadow => {
                self.shadows_index += 1;
                while let Some(shadow) = scene.shadows.get(self.shadows_index)
                    && (shadow.order, batch_kind) < max_order_and_kind
                {
                    self.shadows_index += 1;
                }
            }
            PrimitiveKind::BackdropBlur => {
                self.backdrop_blurs_index += 1;
                while let Some(blur) = scene.backdrop_blurs.get(self.backdrop_blurs_index)
                    && (blur.order, batch_kind) < max_order_and_kind
                {
                    self.backdrop_blurs_index += 1;
                }
            }
            PrimitiveKind::Quad => {
                self.quads_index += 1;
                while let Some(quad) = scene.quads.get(self.quads_index)
                    && (quad.order, batch_kind) < max_order_and_kind
                {
                    self.quads_index += 1;
                }
            }
            PrimitiveKind::Path => {
                self.paths_index += 1;
                while let Some(path) = scene.paths.get(self.paths_index)
                    && (path.order, batch_kind) < max_order_and_kind
                {
                    self.paths_index += 1;
                }
            }
            PrimitiveKind::Underline => {
                self.underlines_index += 1;
                while let Some(underline) = scene.underlines.get(self.underlines_index)
                    && (underline.order, batch_kind) < max_order_and_kind
                {
                    self.underlines_index += 1;
                }
            }
            PrimitiveKind::MonochromeSprite => {
                let Some(texture_id) = scene
                    .monochrome_sprites
                    .get(self.monochrome_sprites_index)
                    .map(|sprite| sprite.tile.texture_id)
                else {
                    return false;
                };
                self.monochrome_sprites_index += 1;
                while let Some(sprite) = scene.monochrome_sprites.get(self.monochrome_sprites_index)
                    && (sprite.order, batch_kind) < max_order_and_kind
                    && sprite.tile.texture_id == texture_id
                {
                    self.monochrome_sprites_index += 1;
                }
            }
            PrimitiveKind::PolychromeSprite => {
                let Some(texture_id) = scene
                    .polychrome_sprites
                    .get(self.polychrome_sprites_index)
                    .map(|sprite| sprite.tile.texture_id)
                else {
                    return false;
                };
                self.polychrome_sprites_index += 1;
                while let Some(sprite) = scene.polychrome_sprites.get(self.polychrome_sprites_index)
                    && (sprite.order, batch_kind) < max_order_and_kind
                    && sprite.tile.texture_id == texture_id
                {
                    self.polychrome_sprites_index += 1;
                }
            }
            PrimitiveKind::Surface => {
                self.surfaces_index += 1;
                while let Some(surface) = scene.surfaces.get(self.surfaces_index)
                    && (surface.order, batch_kind) < max_order_and_kind
                {
                    self.surfaces_index += 1;
                }
            }
        }

        true
    }
}

struct BatchIterator<'a> {
    shadows: &'a [Shadow],
    shadows_start: usize,
    shadows_iter: Peekable<slice::Iter<'a, Shadow>>,
    backdrop_blurs: &'a [BackdropBlur],
    backdrop_blurs_start: usize,
    backdrop_blurs_iter: Peekable<slice::Iter<'a, BackdropBlur>>,
    quads: &'a [Quad],
    quads_start: usize,
    quads_iter: Peekable<slice::Iter<'a, Quad>>,
    paths: &'a [Path<ScaledPixels>],
    paths_start: usize,
    paths_iter: Peekable<slice::Iter<'a, Path<ScaledPixels>>>,
    underlines: &'a [Underline],
    underlines_start: usize,
    underlines_iter: Peekable<slice::Iter<'a, Underline>>,
    monochrome_sprites: &'a [MonochromeSprite],
    monochrome_sprites_start: usize,
    monochrome_sprites_iter: Peekable<slice::Iter<'a, MonochromeSprite>>,
    polychrome_sprites: &'a [PolychromeSprite],
    polychrome_sprites_start: usize,
    polychrome_sprites_iter: Peekable<slice::Iter<'a, PolychromeSprite>>,
    surfaces: &'a [PaintSurface],
    surfaces_start: usize,
    surfaces_iter: Peekable<slice::Iter<'a, PaintSurface>>,
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = PrimitiveBatch<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut orders_and_kinds = [
            (
                self.shadows_iter.peek().map(|s| s.order),
                PrimitiveKind::Shadow,
            ),
            (
                self.backdrop_blurs_iter.peek().map(|b| b.order),
                PrimitiveKind::BackdropBlur,
            ),
            (self.quads_iter.peek().map(|q| q.order), PrimitiveKind::Quad),
            (self.paths_iter.peek().map(|q| q.order), PrimitiveKind::Path),
            (
                self.underlines_iter.peek().map(|u| u.order),
                PrimitiveKind::Underline,
            ),
            (
                self.monochrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::MonochromeSprite,
            ),
            (
                self.polychrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::PolychromeSprite,
            ),
            (
                self.surfaces_iter.peek().map(|s| s.order),
                PrimitiveKind::Surface,
            ),
        ];
        orders_and_kinds.sort_by_key(|(order, kind)| (order.unwrap_or(u32::MAX), *kind));

        let first = orders_and_kinds[0];
        let second = orders_and_kinds[1];
        let (batch_kind, max_order_and_kind) = if first.0.is_some() {
            (first.1, (second.0.unwrap_or(u32::MAX), second.1))
        } else {
            return None;
        };

        match batch_kind {
            PrimitiveKind::Shadow => {
                let shadows_start = self.shadows_start;
                let mut shadows_end = shadows_start + 1;
                self.shadows_iter.next();
                while self
                    .shadows_iter
                    .next_if(|shadow| (shadow.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    shadows_end += 1;
                }
                self.shadows_start = shadows_end;
                Some(PrimitiveBatch::Shadows(
                    &self.shadows[shadows_start..shadows_end],
                ))
            }
            PrimitiveKind::BackdropBlur => {
                let backdrop_blurs_start = self.backdrop_blurs_start;
                let mut backdrop_blurs_end = backdrop_blurs_start + 1;
                self.backdrop_blurs_iter.next();
                while self
                    .backdrop_blurs_iter
                    .next_if(|backdrop_blur| (backdrop_blur.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    backdrop_blurs_end += 1;
                }
                self.backdrop_blurs_start = backdrop_blurs_end;
                Some(PrimitiveBatch::BackdropBlurs(
                    &self.backdrop_blurs[backdrop_blurs_start..backdrop_blurs_end],
                ))
            }
            PrimitiveKind::Quad => {
                let quads_start = self.quads_start;
                let mut quads_end = quads_start + 1;
                self.quads_iter.next();
                while self
                    .quads_iter
                    .next_if(|quad| (quad.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    quads_end += 1;
                }
                self.quads_start = quads_end;
                Some(PrimitiveBatch::Quads(&self.quads[quads_start..quads_end]))
            }
            PrimitiveKind::Path => {
                let paths_start = self.paths_start;
                let mut paths_end = paths_start + 1;
                self.paths_iter.next();
                while self
                    .paths_iter
                    .next_if(|path| (path.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    paths_end += 1;
                }
                self.paths_start = paths_end;
                Some(PrimitiveBatch::Paths(&self.paths[paths_start..paths_end]))
            }
            PrimitiveKind::Underline => {
                let underlines_start = self.underlines_start;
                let mut underlines_end = underlines_start + 1;
                self.underlines_iter.next();
                while self
                    .underlines_iter
                    .next_if(|underline| (underline.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    underlines_end += 1;
                }
                self.underlines_start = underlines_end;
                Some(PrimitiveBatch::Underlines(
                    &self.underlines[underlines_start..underlines_end],
                ))
            }
            PrimitiveKind::MonochromeSprite => {
                let Some(first) = self.monochrome_sprites_iter.peek() else {
                    return None;
                };
                let texture_id = first.tile.texture_id;
                let sprites_start = self.monochrome_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.monochrome_sprites_iter.next();
                while self
                    .monochrome_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.monochrome_sprites_start = sprites_end;
                Some(PrimitiveBatch::MonochromeSprites {
                    texture_id,
                    sprites: &self.monochrome_sprites[sprites_start..sprites_end],
                })
            }
            PrimitiveKind::PolychromeSprite => {
                let Some(first) = self.polychrome_sprites_iter.peek() else {
                    return None;
                };
                let texture_id = first.tile.texture_id;
                let sprites_start = self.polychrome_sprites_start;
                let mut sprites_end = self.polychrome_sprites_start + 1;
                self.polychrome_sprites_iter.next();
                while self
                    .polychrome_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.polychrome_sprites_start = sprites_end;
                Some(PrimitiveBatch::PolychromeSprites {
                    texture_id,
                    sprites: &self.polychrome_sprites[sprites_start..sprites_end],
                })
            }
            PrimitiveKind::Surface => {
                let surfaces_start = self.surfaces_start;
                let mut surfaces_end = surfaces_start + 1;
                self.surfaces_iter.next();
                while self
                    .surfaces_iter
                    .next_if(|surface| (surface.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    surfaces_end += 1;
                }
                self.surfaces_start = surfaces_end;
                Some(PrimitiveBatch::Surfaces(
                    &self.surfaces[surfaces_start..surfaces_end],
                ))
            }
        }
    }
}

#[derive(Debug)]
pub(crate) enum PrimitiveBatch<'a> {
    Shadows(&'a [Shadow]),
    BackdropBlurs(&'a [BackdropBlur]),
    Quads(&'a [Quad]),
    Paths(&'a [Path<ScaledPixels>]),
    Underlines(&'a [Underline]),
    MonochromeSprites {
        texture_id: AtlasTextureId,
        sprites: &'a [MonochromeSprite],
    },
    PolychromeSprites {
        texture_id: AtlasTextureId,
        sprites: &'a [PolychromeSprite],
    },
    Surfaces(&'a [PaintSurface]),
}

#[derive(Default, Debug, Clone)]
#[repr(C)]
pub(crate) struct Quad {
    pub order: DrawOrder,
    pub border_style: BorderStyle,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub background: Background,
    pub border_color: Hsla,
    pub corner_radii: Corners<ScaledPixels>,
    pub border_widths: Edges<ScaledPixels>,
}

impl From<Quad> for Primitive {
    fn from(quad: Quad) -> Self {
        Primitive::Quad(quad)
    }
}

#[derive(Debug, Clone)]
#[repr(C)]
pub(crate) struct Underline {
    pub order: DrawOrder,
    pub pad: u32, // align to 8 bytes
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub thickness: ScaledPixels,
    pub wavy: u32,
}

impl From<Underline> for Primitive {
    fn from(underline: Underline) -> Self {
        Primitive::Underline(underline)
    }
}

#[derive(Debug, Clone)]
#[repr(C)]
pub(crate) struct Shadow {
    pub order: DrawOrder,
    pub blur_radius: ScaledPixels,
    pub bounds: Bounds<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
}

impl From<Shadow> for Primitive {
    fn from(shadow: Shadow) -> Self {
        Primitive::Shadow(shadow)
    }
}

#[derive(Debug, Clone)]
#[repr(C)]
pub(crate) struct BackdropBlur {
    pub order: DrawOrder,
    pub blur_radius: ScaledPixels,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub background: Background,
    pub border_color: Hsla,
    pub border_widths: Edges<ScaledPixels>,
}

impl From<BackdropBlur> for Primitive {
    fn from(backdrop_blur: BackdropBlur) -> Self {
        Primitive::BackdropBlur(backdrop_blur)
    }
}

/// The style of a border.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[repr(C)]
pub enum BorderStyle {
    /// A solid border.
    #[default]
    Solid = 0,
    /// A dashed border.
    Dashed = 1,
}

/// A data type representing a 2 dimensional transformation that can be applied to an element.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct TransformationMatrix {
    /// 2x2 matrix containing rotation and scale,
    /// stored row-major
    pub rotation_scale: [[f32; 2]; 2],
    /// translation vector
    pub translation: [f32; 2],
}

impl Eq for TransformationMatrix {}

impl TransformationMatrix {
    /// The unit matrix, has no effect.
    pub fn unit() -> Self {
        Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [0.0, 0.0],
        }
    }

    /// Move the origin by a given point
    pub fn translate(mut self, point: Point<ScaledPixels>) -> Self {
        self.compose(Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [point.x.0, point.y.0],
        })
    }

    /// Clockwise rotation in radians around the origin
    pub fn rotate(self, angle: Radians) -> Self {
        self.compose(Self {
            rotation_scale: [
                [angle.0.cos(), -angle.0.sin()],
                [angle.0.sin(), angle.0.cos()],
            ],
            translation: [0.0, 0.0],
        })
    }

    /// Scale around the origin
    pub fn scale(self, size: Size<f32>) -> Self {
        self.compose(Self {
            rotation_scale: [[size.width, 0.0], [0.0, size.height]],
            translation: [0.0, 0.0],
        })
    }

    /// Perform matrix multiplication with another transformation
    /// to produce a new transformation that is the result of
    /// applying both transformations: first, `other`, then `self`.
    #[inline]
    pub fn compose(self, other: TransformationMatrix) -> TransformationMatrix {
        if other == Self::unit() {
            return self;
        }
        // Perform matrix multiplication
        TransformationMatrix {
            rotation_scale: [
                [
                    self.rotation_scale[0][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][0],
                    self.rotation_scale[0][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][1],
                ],
                [
                    self.rotation_scale[1][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][0],
                    self.rotation_scale[1][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][1],
                ],
            ],
            translation: [
                self.translation[0]
                    + self.rotation_scale[0][0] * other.translation[0]
                    + self.rotation_scale[0][1] * other.translation[1],
                self.translation[1]
                    + self.rotation_scale[1][0] * other.translation[0]
                    + self.rotation_scale[1][1] * other.translation[1],
            ],
        }
    }

    /// Apply transformation to a point, mainly useful for debugging
    pub fn apply(&self, point: Point<Pixels>) -> Point<Pixels> {
        let input = [point.x.0, point.y.0];
        let mut output = self.translation;
        for (i, output_cell) in output.iter_mut().enumerate() {
            for (k, input_cell) in input.iter().enumerate() {
                *output_cell += self.rotation_scale[i][k] * *input_cell;
            }
        }
        Point::new(output[0].into(), output[1].into())
    }
}

impl Default for TransformationMatrix {
    fn default() -> Self {
        Self::unit()
    }
}

#[derive(Clone, Debug)]
#[repr(C)]
pub(crate) struct MonochromeSprite {
    pub order: DrawOrder,
    pub pad: u32, // align to 8 bytes
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub text_color: TextColor,
    pub tile: AtlasTile,
    pub transformation: TransformationMatrix,
}

impl From<MonochromeSprite> for Primitive {
    fn from(sprite: MonochromeSprite) -> Self {
        Primitive::MonochromeSprite(sprite)
    }
}

#[derive(Clone, Debug)]
#[repr(C)]
pub(crate) struct PolychromeSprite {
    pub order: DrawOrder,
    pub pad: u32, // align to 8 bytes
    pub grayscale: bool,
    pub opacity: f32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub tile: AtlasTile,
}

impl From<PolychromeSprite> for Primitive {
    fn from(sprite: PolychromeSprite) -> Self {
        Primitive::PolychromeSprite(sprite)
    }
}

/// The backing content for a painted surface.
#[derive(Clone, Debug)]
pub(crate) enum SurfaceContent {
    /// A WGPU surface managed by the SurfaceRegistry.
    Wgpu(SurfaceId),
}

#[derive(Clone, Debug)]
pub(crate) struct PaintSurface {
    pub order: DrawOrder,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub content: SurfaceContent,
}

impl From<PaintSurface> for Primitive {
    fn from(surface: PaintSurface) -> Self {
        Primitive::Surface(surface)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PathId(pub(crate) usize);

/// A line made up of a series of vertices and control points.
#[derive(Clone, Debug)]
pub struct Path<P: Clone + Debug + Default + PartialEq> {
    pub(crate) id: PathId,
    pub(crate) order: DrawOrder,
    pub(crate) bounds: Bounds<P>,
    pub(crate) content_mask: ContentMask<P>,
    pub(crate) vertices: Vec<PathVertex<P>>,
    pub(crate) color: Background,
    start: Point<P>,
    current: Point<P>,
    contour_count: usize,
}

impl Path<Pixels> {
    /// Create a new path with the given starting point.
    pub fn new(start: Point<Pixels>) -> Self {
        Self {
            id: PathId(0),
            order: DrawOrder::default(),
            vertices: Vec::new(),
            start,
            current: start,
            bounds: Bounds {
                origin: start,
                size: Default::default(),
            },
            content_mask: Default::default(),
            color: Default::default(),
            contour_count: 0,
        }
    }

    /// Scale this path by the given factor.
    pub fn scale(&self, factor: f32) -> Path<ScaledPixels> {
        Path {
            id: self.id,
            order: self.order,
            bounds: self.bounds.scale(factor),
            content_mask: self.content_mask.scale(factor),
            vertices: self
                .vertices
                .iter()
                .map(|vertex| vertex.scale(factor))
                .collect(),
            start: self.start.map(|start| start.scale(factor)),
            current: self.current.scale(factor),
            contour_count: self.contour_count,
            color: self.color,
        }
    }

    /// Move the start, current point to the given point.
    pub fn move_to(&mut self, to: Point<Pixels>) {
        self.contour_count += 1;
        self.start = to;
        self.current = to;
    }

    /// Draw a straight line from the current point to the given point.
    pub fn line_to(&mut self, to: Point<Pixels>) {
        self.contour_count += 1;
        if self.contour_count > 1 {
            self.push_triangle(
                (self.start, self.current, to),
                (point(0., 1.), point(0., 1.), point(0., 1.)),
            );
        }
        self.current = to;
    }

    /// Draw a curve from the current point to the given point, using the given control point.
    pub fn curve_to(&mut self, to: Point<Pixels>, ctrl: Point<Pixels>) {
        self.contour_count += 1;
        if self.contour_count > 1 {
            self.push_triangle(
                (self.start, self.current, to),
                (point(0., 1.), point(0., 1.), point(0., 1.)),
            );
        }

        self.push_triangle(
            (self.current, ctrl, to),
            (point(0., 0.), point(0.5, 0.), point(1., 1.)),
        );
        self.current = to;
    }

    /// Push a triangle to the Path.
    pub fn push_triangle(
        &mut self,
        xy: (Point<Pixels>, Point<Pixels>, Point<Pixels>),
        st: (Point<f32>, Point<f32>, Point<f32>),
    ) {
        self.bounds = self
            .bounds
            .union(&Bounds {
                origin: xy.0,
                size: Default::default(),
            })
            .union(&Bounds {
                origin: xy.1,
                size: Default::default(),
            })
            .union(&Bounds {
                origin: xy.2,
                size: Default::default(),
            });

        self.vertices.push(PathVertex {
            xy_position: xy.0,
            st_position: st.0,
            content_mask: Default::default(),
        });
        self.vertices.push(PathVertex {
            xy_position: xy.1,
            st_position: st.1,
            content_mask: Default::default(),
        });
        self.vertices.push(PathVertex {
            xy_position: xy.2,
            st_position: st.2,
            content_mask: Default::default(),
        });
    }
}

impl<T> Path<T>
where
    T: Clone + Debug + Default + PartialEq + PartialOrd + Add<T, Output = T> + Sub<Output = T>,
{
    #[allow(unused)]
    pub(crate) fn clipped_bounds(&self) -> Bounds<T> {
        self.bounds.intersect(&self.content_mask.bounds)
    }
}

impl From<Path<ScaledPixels>> for Primitive {
    fn from(path: Path<ScaledPixels>) -> Self {
        Primitive::Path(path)
    }
}

#[derive(Clone, Debug)]
#[repr(C)]
pub(crate) struct PathVertex<P: Clone + Debug + Default + PartialEq> {
    pub(crate) xy_position: Point<P>,
    pub(crate) st_position: Point<f32>,
    pub(crate) content_mask: ContentMask<P>,
}

impl PathVertex<Pixels> {
    pub fn scale(&self, factor: f32) -> PathVertex<ScaledPixels> {
        PathVertex {
            xy_position: self.xy_position.scale(factor),
            st_position: self.st_position,
            content_mask: self.content_mask.scale(factor),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_advances_for_dirty_chunks_not_clean_reuse() {
        let view_id = EntityId::from(1);
        let mut scene = Scene::default();

        scene.begin_view_chunk(view_id);
        scene.mark_chunk_clean(view_id);
        scene.end_view_chunk();
        scene.finish();

        assert_eq!(scene.scene_generation(), 0);

        scene.begin_view_chunk(view_id);
        scene.end_view_chunk();
        scene.finish();

        assert_eq!(scene.scene_generation(), 1);
    }
}
