use collections::FxHashMap;

use super::{
    BackdropBlur,
    ChangedRanges,
    DrawOrder,
    MonochromeSprite,
    PaintOperation,
    PaintSurface,
    Path,
    PolychromeSprite,
    Quad,
    SceneChunk,
    Shadow,
    Underline,
};
use crate::bounds_tree::BoundsTree;
#[cfg(test)]
use crate::retained::SceneSegmentId;
use crate::retained::SceneSegmentPool;
use crate::{Bounds, EntityId, ScaledPixels};

#[derive(Default)]
pub(crate) struct Scene {
    pub(crate) paint_operations: Vec<PaintOperation>,
    pub(crate) primitive_bounds: BoundsTree<ScaledPixels>,
    pub(crate) layer_stack: Vec<DrawOrder>,
    pub(crate) shadows: Vec<Shadow>,
    pub(crate) backdrop_blurs: Vec<BackdropBlur>,
    pub(crate) quads: Vec<Quad>,
    pub(crate) paths: Vec<Path<ScaledPixels>>,
    pub(crate) underlines: Vec<Underline>,
    pub(crate) monochrome_sprites: Vec<MonochromeSprite>,
    pub(crate) polychrome_sprites: Vec<PolychromeSprite>,
    pub(crate) surfaces: Vec<PaintSurface>,
    pub(crate) damage_rects: Vec<Bounds<ScaledPixels>>,
    pub(crate) chunks: Vec<SceneChunk>,
    pub(crate) active_chunk: Option<usize>,
    pub(crate) chunk_map: FxHashMap<EntityId, usize>,
    pub(crate) segment_pool: SceneSegmentPool,
    pub(crate) changed_ranges: ChangedRanges,
    pub(crate) generation: u64,
    pub(crate) batch_count: u32,
    /// Instances of custom (plugin) render primitives queued this frame, grouped
    /// by type for batched draw. Drawn after the built-in primitives.
    pub(crate) custom_primitives: gpui_render_primitive::PrimitiveBatches,
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
        self.segment_pool.clear_live();
        self.changed_ranges = ChangedRanges::default();
        self.batch_count = 0;
        self.custom_primitives.clear();
    }

    pub fn begin_view_chunk(&mut self, view_id: EntityId) {
        let segment_id = self
            .chunk_map
            .get(&view_id)
            .and_then(|chunk_index| self.chunks.get(*chunk_index))
            .map_or_else(|| self.segment_pool.allocate(), |chunk| chunk.segment_id);
        self.segment_pool.retain(segment_id);
        let chunk = SceneChunk::new(view_id, self, segment_id);
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

    #[cfg(test)]
    pub(crate) fn segment_id_for_view(&self, view_id: EntityId) -> Option<SceneSegmentId> {
        self.chunk_map
            .get(&view_id)
            .and_then(|chunk_index| self.chunks.get(*chunk_index))
            .map(|chunk| chunk.segment_id)
    }

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

    pub fn mark_chunk_clean(&mut self, view_id: EntityId) {
        if let Some(&chunk_index) = self.chunk_map.get(&view_id)
            && let Some(chunk) = self.chunks.get_mut(chunk_index)
        {
            chunk.dirty = false;
        }
    }

    pub fn scene_generation(&self) -> u64 {
        self.generation
    }

    pub fn dirty_chunk_count(&self) -> usize {
        self.chunks.iter().filter(|chunk| chunk.dirty).count()
    }

    pub fn batch_count(&self) -> u32 {
        self.batch_count
    }

    pub fn changed_ranges(&self) -> &ChangedRanges {
        &self.changed_ranges
    }

    pub fn len(&self) -> usize {
        self.paint_operations.len()
    }
}
