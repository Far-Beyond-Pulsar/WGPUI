use std::ops::Range;

use super::Scene;
use crate::EntityId;
use crate::retained::SceneSegmentId;

#[derive(Debug, Clone)]
pub(crate) struct SceneChunk {
    pub view_id: EntityId,
    pub(crate) segment_id: SceneSegmentId,
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

#[derive(Debug, Default, Clone, PartialEq, Eq)]
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

impl SceneChunk {
    pub(super) fn new(view_id: EntityId, scene: &Scene, segment_id: SceneSegmentId) -> Self {
        Self {
            view_id,
            segment_id,
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

    pub(super) fn end_at(&mut self, scene: &Scene) {
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

pub(super) fn merge_ranges(ranges: &mut Vec<Range<usize>>) {
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
