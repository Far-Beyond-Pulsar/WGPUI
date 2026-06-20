use super::chunk::merge_ranges;
use super::{ChangedRanges, Scene, SceneChunk};
use crate::{Bounds, ScaledPixels};

impl Scene {
    #[cfg(test)]
    pub fn finish(&mut self) {
        self.finish_with_previous(None);
    }

    pub fn finish_with_previous_scene(&mut self, previous_scene: &Scene) {
        self.finish_with_previous(Some(previous_scene));
    }

    fn finish_with_previous(&mut self, previous_scene: Option<&Scene>) {
        self.finish_incremental();
        self.reconcile_retained_chunks(previous_scene);
        let has_dirty_chunks = self.chunks.iter().any(|chunk| chunk.dirty);
        self.compute_damage_rects(previous_scene);
        self.changed_ranges = self.collect_changed_ranges();
        for chunk in &mut self.chunks {
            chunk.dirty = false;
        }
        if has_dirty_chunks {
            self.generation += 1;
        }
        self.batch_count = self.total_batch_count();
    }

    fn reconcile_retained_chunks(&mut self, previous_scene: Option<&Scene>) {
        let Some(previous_scene) = previous_scene else {
            return;
        };
        let segment_pool = &mut self.segment_pool;

        for chunk in &mut self.chunks {
            let Some(previous_chunk) = previous_scene
                .chunk_map
                .get(&chunk.view_id)
                .and_then(|chunk_index| previous_scene.chunks.get(*chunk_index))
            else {
                continue;
            };

            chunk.segment_id = previous_chunk.segment_id;
            segment_pool.retain(previous_chunk.segment_id);
        }
    }

    fn collect_changed_ranges(&self) -> ChangedRanges {
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

    pub fn compute_damage_rects(&mut self, previous_scene: Option<&Scene>) {
        self.damage_rects.clear();

        for chunk in &self.chunks {
            if !chunk.dirty {
                continue;
            }
            if let Some(previous_scene) = previous_scene
                && let Some(previous_chunk_index) = previous_scene.chunk_map.get(&chunk.view_id)
                && let Some(previous_chunk) = previous_scene.chunks.get(*previous_chunk_index)
                && let Some(bounds) = previous_scene.chunk_bounds(previous_chunk)
            {
                self.damage_rects.push(bounds);
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
}
