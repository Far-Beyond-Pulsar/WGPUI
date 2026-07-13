use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    AtlasTextureId, AtlasTile, Background, Bounds, ContentMask, Corners, Edges, Hsla, Pixels,
    Point, Radians, ScaledPixels, Size, TextColor, bounds_tree::BoundsTree,
    platform::cross::surface_registry::SurfaceId, point,
};
use std::{
    fmt::Debug,
    ops::{Add, Sub},
};

#[allow(non_camel_case_types, unused)]
pub(crate) type PathVertex_ScaledPixels = PathVertex<ScaledPixels>;

pub(crate) type DrawOrder = u32;

/// A persistent slot in a generational arena.
#[derive(Clone)]
pub(crate) struct GenSlot<T> {
    pub data: T,
    pub version: u64,
}

/// Persistent generational arena for GPU buffer data.
#[derive(Clone)]
pub(crate) struct GenerationalVec<T> {
    pub slots: Vec<Option<GenSlot<T>>>,
    pub free: Vec<usize>,
    pub generation: u64,
}

impl<T: Clone> GenerationalVec<T> {
    pub fn new() -> Self {
        Self { slots: Vec::new(), free: Vec::new(), generation: 0 }
    }

    pub fn allocate(&mut self) -> usize {
        if let Some(idx) = self.free.pop() {
            self.slots[idx] = Some(GenSlot {
                data: unsafe { std::mem::zeroed() },
                version: 0,
            });
            idx
        } else {
            let idx = self.slots.len();
            self.slots.push(Some(GenSlot {
                data: unsafe { std::mem::zeroed() },
                version: 0,
            }));
            idx
        }
    }

    pub fn free(&mut self, idx: usize) {
        if idx < self.slots.len() {
            self.slots[idx] = None;
            self.free.push(idx);
        }
    }

    pub fn write(&mut self, idx: usize, data: T) {
        if let Some(Some(slot)) = self.slots.get_mut(idx) {
            slot.data = data;
            self.generation += 1;
            slot.version = self.generation;
        }
    }

    pub fn get(&self, idx: usize) -> Option<&T> {
        self.slots.get(idx).and_then(|s| s.as_ref().map(|s| &s.data))
    }

    pub fn get_mut(&mut self, idx: usize) -> Option<&mut T> {
        self.slots.get_mut(idx).and_then(|s| s.as_mut().map(|s| &mut s.data))
    }

    pub fn drain_changes_since(&self, since_generation: u64) -> Vec<usize> {
        self.slots.iter().enumerate()
            .filter_map(|(idx, slot)| {
                slot.as_ref()
                    .filter(|s| s.version > since_generation)
                    .map(|_| idx)
            })
            .collect()
    }

    pub fn allocated_count(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    pub fn clear(&mut self) {
        self.slots.clear();
        self.free.clear();
    }
}

impl<T: Clone> Default for GenerationalVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Records one changed slot for the GPU upload path
#[derive(Clone, Debug)]
pub(crate) struct ChangedSlot {
    pub slot: usize,
    pub byte_offset: u64,
    pub byte_length: u64,
    pub bounds: Bounds<ScaledPixels>,
}

/// Accumulated changes during a single frame's paint phase.
#[derive(Clone, Debug)]
pub(crate) struct SceneDelta {
    pub changed_quads: Vec<ChangedSlot>,
    pub changed_shadows: Vec<ChangedSlot>,
    pub changed_backdrop_filters: Vec<ChangedSlot>,
    pub changed_underlines: Vec<ChangedSlot>,
    pub changed_monochrome_sprites: Vec<ChangedSlot>,
    pub changed_polychrome_sprites: Vec<ChangedSlot>,
    pub changed_surfaces: Vec<ChangedSlot>,
    pub paths_changed: bool,
    pub damage_rect: Option<Bounds<ScaledPixels>>,
    pub is_empty: bool,
}

impl SceneDelta {
    pub fn new() -> Self {
        Self {
            changed_quads: Vec::new(),
            changed_shadows: Vec::new(),
            changed_backdrop_filters: Vec::new(),
            changed_underlines: Vec::new(),
            changed_monochrome_sprites: Vec::new(),
            changed_polychrome_sprites: Vec::new(),
            changed_surfaces: Vec::new(),
            paths_changed: false,
            damage_rect: None,
            is_empty: true,
        }
    }

    pub fn clear(&mut self) {
        self.changed_quads.clear();
        self.changed_shadows.clear();
        self.changed_backdrop_filters.clear();
        self.changed_underlines.clear();
        self.changed_monochrome_sprites.clear();
        self.changed_polychrome_sprites.clear();
        self.changed_surfaces.clear();
        self.paths_changed = false;
        self.damage_rect = None;
        self.is_empty = true;
    }

    pub fn add_damage(&mut self, bounds: Bounds<ScaledPixels>) {
        self.damage_rect = Some(match self.damage_rect {
            Some(existing) => existing.union(&bounds),
            None => bounds,
        });
    }
}

impl Default for SceneDelta {
    fn default() -> Self {
        Self::new()
    }
}

/// Tracks which generational-vec slots each primitive type occupies for a paint operation.
#[derive(Clone, Debug)]
pub(crate) struct PaintOperationSlots {
    pub quad_slot: Option<usize>,
    pub shadow_slot: Option<usize>,
    pub backdrop_filter_slot: Option<usize>,
    pub underline_slot: Option<usize>,
    pub mono_sprite_slot: Option<usize>,
    pub poly_sprite_slot: Option<usize>,
    pub surface_slot: Option<usize>,
    pub path_slot: Option<usize>,
}

#[derive(Default, Clone)]
pub(crate) struct Scene {
    pub(crate) paint_operations: Vec<PaintOperation>,
    pub(crate) paint_slots: Vec<Option<PaintOperationSlots>>,
    primitive_bounds: BoundsTree<ScaledPixels>,
    layer_stack: Vec<DrawOrder>,

    // === PERSISTENT GENERATIONAL ARENAS (NOT cleared per frame) ===
    pub(crate) quads: GenerationalVec<Quad>,
    pub(crate) shadows: GenerationalVec<Shadow>,
    pub(crate) backdrop_filters: GenerationalVec<BackdropFilter>,
    pub(crate) underlines: GenerationalVec<Underline>,
    pub(crate) monochrome_sprites: GenerationalVec<MonochromeSprite>,
    pub(crate) polychrome_sprites: GenerationalVec<PolychromeSprite>,
    pub(crate) surfaces: GenerationalVec<PaintSurface>,
    pub(crate) paths: GenerationalVec<Path<ScaledPixels>>,

    // Filter boundaries (always rebuilt — NOT in a generational vec)
    pub(crate) filter_boundaries: Vec<FilterBoundary>,

    // === SORTED DRAW-ORDER INDICES (indirection buffers) ===
    pub(crate) sorted_quad_indices: Vec<u32>,
    pub(crate) sorted_shadow_indices: Vec<u32>,
    pub(crate) sorted_backdrop_filter_indices: Vec<u32>,
    pub(crate) sorted_underline_indices: Vec<u32>,
    pub(crate) sorted_mono_sprite_indices: Vec<u32>,
    pub(crate) sorted_poly_sprite_indices: Vec<u32>,
    pub(crate) sorted_surface_indices: Vec<u32>,
    pub(crate) sorted_path_indices: Vec<u32>,

    // === DELTA ACCUMULATION ===
    pub(crate) pending_delta: SceneDelta,
    pub(crate) last_uploaded_quad_gen: u64,
    pub(crate) last_uploaded_shadow_gen: u64,
    pub(crate) last_uploaded_backdrop_filter_gen: u64,
    pub(crate) last_uploaded_underline_gen: u64,
    pub(crate) last_uploaded_mono_sprite_gen: u64,
    pub(crate) last_uploaded_poly_sprite_gen: u64,
    pub(crate) last_uploaded_surface_gen: u64,
    pub(crate) last_uploaded_path_gen: u64,
}

impl Scene {
    pub fn clear(&mut self) {
        self.paint_operations.clear();
        self.paint_slots.clear();
        self.primitive_bounds.clear();
        self.layer_stack.clear();
        self.filter_boundaries.clear();
        self.sorted_quad_indices.clear();
        self.sorted_shadow_indices.clear();
        self.sorted_backdrop_filter_indices.clear();
        self.sorted_underline_indices.clear();
        self.sorted_mono_sprite_indices.clear();
        self.sorted_poly_sprite_indices.clear();
        self.sorted_surface_indices.clear();
        self.sorted_path_indices.clear();
        self.pending_delta.clear();
    }

    pub fn len(&self) -> usize {
        self.paint_operations.len()
    }

    pub fn push_layer(&mut self, bounds: Bounds<ScaledPixels>) {
        let order = self.primitive_bounds.insert(bounds);
        self.layer_stack.push(order);
        self.paint_slots.push(None);
        self.paint_operations
            .push(PaintOperation::StartLayer(bounds));
    }

    pub fn pop_layer(&mut self) {
        self.layer_stack.pop();
        self.paint_slots.push(None);
        self.paint_operations.push(PaintOperation::EndLayer);
    }

    /// Raise the draw-order floor so every primitive inserted afterwards sorts above everything
    /// inserted before. Called before painting deferred draws so overlays (tooltips, popovers,
    /// drag images) sort above the main scene — and a deferred backdrop's order can't fall inside
    /// a content-filter (`filter`) order range left behind by the main scene.
    pub fn raise_order_floor(&mut self) {
        let floor = self.primitive_bounds.max_order() + 1;
        self.primitive_bounds.set_order_floor(floor);
    }

    pub fn insert_primitive(&mut self, primitive: impl Into<Primitive>) {
        let mut primitive = primitive.into();
        let clipped_bounds = primitive
            .bounds()
            .intersect(&primitive.content_mask().bounds);

        let is_filter_boundary = matches!(primitive, Primitive::FilterBoundary(_));

        if clipped_bounds.is_empty() && !is_filter_boundary {
            return;
        }

        self.pending_delta.is_empty = false;

        let order = if is_filter_boundary {
            let order_bounds = if clipped_bounds.is_empty() {
                *primitive.bounds()
            } else {
                clipped_bounds
            };
            self.primitive_bounds.insert_above_all(order_bounds)
        } else {
            self.layer_stack
                .last()
                .copied()
                .unwrap_or_else(|| self.primitive_bounds.insert(clipped_bounds))
        };

        let mut slots = PaintOperationSlots {
            quad_slot: None, shadow_slot: None, backdrop_filter_slot: None,
            underline_slot: None, mono_sprite_slot: None, poly_sprite_slot: None,
            surface_slot: None, path_slot: None,
        };

        match &mut primitive {
            Primitive::Shadow(shadow) => {
                shadow.order = order;
                let slot = self.shadows.allocate();
                self.shadows.write(slot, shadow.clone());
                slots.shadow_slot = Some(slot);
                self.pending_delta.changed_shadows.push(ChangedSlot {
                    slot,
                    byte_offset: slot as u64 * std::mem::size_of::<Shadow>() as u64,
                    byte_length: std::mem::size_of::<Shadow>() as u64,
                    bounds: shadow.bounds,
                });
                self.pending_delta.add_damage(shadow.bounds);
            }
            Primitive::BackdropFilter(filter) => {
                filter.order = order;
                let slot = self.backdrop_filters.allocate();
                self.backdrop_filters.write(slot, *filter);
                slots.backdrop_filter_slot = Some(slot);
                self.pending_delta.changed_backdrop_filters.push(ChangedSlot {
                    slot,
                    byte_offset: slot as u64 * std::mem::size_of::<BackdropFilter>() as u64,
                    byte_length: std::mem::size_of::<BackdropFilter>() as u64,
                    bounds: filter.bounds,
                });
                self.pending_delta.add_damage(filter.bounds);
            }
            Primitive::FilterBoundary(boundary) => {
                boundary.order = order;
                self.filter_boundaries.push(*boundary);
            }
            Primitive::Quad(quad) => {
                quad.order = order;
                let slot = self.quads.allocate();
                self.quads.write(slot, quad.clone());
                slots.quad_slot = Some(slot);
                self.pending_delta.changed_quads.push(ChangedSlot {
                    slot,
                    byte_offset: slot as u64 * std::mem::size_of::<Quad>() as u64,
                    byte_length: std::mem::size_of::<Quad>() as u64,
                    bounds: quad.bounds,
                });
                self.pending_delta.add_damage(quad.bounds);
            }
            Primitive::Path(path) => {
                path.order = order;
                let slot = self.paths.allocate();
                self.paths.write(slot, path.clone());
                slots.path_slot = Some(slot);
                self.pending_delta.paths_changed = true;
                self.pending_delta.add_damage(path.bounds);
            }
            Primitive::Underline(underline) => {
                underline.order = order;
                let slot = self.underlines.allocate();
                self.underlines.write(slot, underline.clone());
                slots.underline_slot = Some(slot);
                self.pending_delta.changed_underlines.push(ChangedSlot {
                    slot,
                    byte_offset: slot as u64 * std::mem::size_of::<Underline>() as u64,
                    byte_length: std::mem::size_of::<Underline>() as u64,
                    bounds: underline.bounds,
                });
                self.pending_delta.add_damage(underline.bounds);
            }
            Primitive::MonochromeSprite(sprite) => {
                sprite.order = order;
                let slot = self.monochrome_sprites.allocate();
                self.monochrome_sprites.write(slot, sprite.clone());
                slots.mono_sprite_slot = Some(slot);
                self.pending_delta.changed_monochrome_sprites.push(ChangedSlot {
                    slot,
                    byte_offset: slot as u64 * std::mem::size_of::<MonochromeSprite>() as u64,
                    byte_length: std::mem::size_of::<MonochromeSprite>() as u64,
                    bounds: sprite.bounds,
                });
                self.pending_delta.add_damage(sprite.bounds);
            }
            Primitive::PolychromeSprite(sprite) => {
                sprite.order = order;
                let slot = self.polychrome_sprites.allocate();
                self.polychrome_sprites.write(slot, sprite.clone());
                slots.poly_sprite_slot = Some(slot);
                self.pending_delta.changed_polychrome_sprites.push(ChangedSlot {
                    slot,
                    byte_offset: slot as u64 * std::mem::size_of::<PolychromeSprite>() as u64,
                    byte_length: std::mem::size_of::<PolychromeSprite>() as u64,
                    bounds: sprite.bounds,
                });
                self.pending_delta.add_damage(sprite.bounds);
            }
            Primitive::Surface(surface) => {
                surface.order = order;
                let slot = self.surfaces.allocate();
                self.surfaces.write(slot, surface.clone());
                slots.surface_slot = Some(slot);
                self.pending_delta.changed_surfaces.push(ChangedSlot {
                    slot,
                    byte_offset: slot as u64 * std::mem::size_of::<PaintSurface>() as u64,
                    byte_length: std::mem::size_of::<PaintSurface>() as u64,
                    bounds: surface.bounds,
                });
                self.pending_delta.add_damage(surface.bounds);
            }
        }

        if is_filter_boundary {
            self.paint_slots.push(None);
        } else {
            self.paint_slots.push(Some(slots));
        }
        self.paint_operations.push(PaintOperation::Primitive(primitive));
    }

    pub fn replay(&mut self, operation: &PaintOperation, slot_info: &Option<PaintOperationSlots>) {
        self.paint_operations.push(operation.clone());
        self.paint_slots.push(slot_info.clone());
    }

    pub fn sort(&mut self) {
        self.filter_boundaries
            .sort_by_key(|boundary| (boundary.order, !boundary.is_start));
        self.build_indirection_buffers();
        self.free_orphaned_slots();
    }

    pub fn free_orphaned_slots(&mut self) {
        let used_quads: std::collections::HashSet<usize> = self.paint_slots.iter()
            .filter_map(|s| s.as_ref()?.quad_slot).collect();
        let used_shadows: std::collections::HashSet<usize> = self.paint_slots.iter()
            .filter_map(|s| s.as_ref()?.shadow_slot).collect();
        let used_backdrop_filters: std::collections::HashSet<usize> = self.paint_slots.iter()
            .filter_map(|s| s.as_ref()?.backdrop_filter_slot).collect();
        let used_underlines: std::collections::HashSet<usize> = self.paint_slots.iter()
            .filter_map(|s| s.as_ref()?.underline_slot).collect();
        let used_mono_sprites: std::collections::HashSet<usize> = self.paint_slots.iter()
            .filter_map(|s| s.as_ref()?.mono_sprite_slot).collect();
        let used_poly_sprites: std::collections::HashSet<usize> = self.paint_slots.iter()
            .filter_map(|s| s.as_ref()?.poly_sprite_slot).collect();
        let used_surfaces: std::collections::HashSet<usize> = self.paint_slots.iter()
            .filter_map(|s| s.as_ref()?.surface_slot).collect();

        for idx in 0..self.quads.slots.len() {
            if self.quads.slots[idx].is_some() && !used_quads.contains(&idx) {
                self.quads.free(idx);
            }
        }
        for idx in 0..self.shadows.slots.len() {
            if self.shadows.slots[idx].is_some() && !used_shadows.contains(&idx) {
                self.shadows.free(idx);
            }
        }
        for idx in 0..self.backdrop_filters.slots.len() {
            if self.backdrop_filters.slots[idx].is_some() && !used_backdrop_filters.contains(&idx) {
                self.backdrop_filters.free(idx);
            }
        }
        for idx in 0..self.underlines.slots.len() {
            if self.underlines.slots[idx].is_some() && !used_underlines.contains(&idx) {
                self.underlines.free(idx);
            }
        }
        for idx in 0..self.monochrome_sprites.slots.len() {
            if self.monochrome_sprites.slots[idx].is_some() && !used_mono_sprites.contains(&idx) {
                self.monochrome_sprites.free(idx);
            }
        }
        for idx in 0..self.polychrome_sprites.slots.len() {
            if self.polychrome_sprites.slots[idx].is_some() && !used_poly_sprites.contains(&idx) {
                self.polychrome_sprites.free(idx);
            }
        }
        for idx in 0..self.surfaces.slots.len() {
            if self.surfaces.slots[idx].is_some() && !used_surfaces.contains(&idx) {
                self.surfaces.free(idx);
            }
        }
    }

    fn build_indirection_buffers(&mut self) {
        let mut quad_pairs: Vec<(u32, u32)> = Vec::new();
        for (op, slots) in self.paint_operations.iter().zip(self.paint_slots.iter()) {
            if let PaintOperation::Primitive(Primitive::Quad(quad)) = op {
                if let Some(slots) = slots {
                    if let Some(slot) = slots.quad_slot {
                        quad_pairs.push((quad.order, slot as u32));
                    }
                }
            }
        }
        quad_pairs.sort_by_key(|(order, _)| *order);
        self.sorted_quad_indices = quad_pairs.into_iter().map(|(_, slot)| slot).collect();

        let mut shadow_pairs: Vec<(u32, u32)> = Vec::new();
        for (op, slots) in self.paint_operations.iter().zip(self.paint_slots.iter()) {
            if let PaintOperation::Primitive(Primitive::Shadow(shadow)) = op {
                if let Some(slots) = slots {
                    if let Some(slot) = slots.shadow_slot {
                        shadow_pairs.push((shadow.order, slot as u32));
                    }
                }
            }
        }
        shadow_pairs.sort_by_key(|(order, _)| *order);
        self.sorted_shadow_indices = shadow_pairs.into_iter().map(|(_, slot)| slot).collect();

        let mut bf_pairs: Vec<(u32, u32)> = Vec::new();
        for (op, slots) in self.paint_operations.iter().zip(self.paint_slots.iter()) {
            if let PaintOperation::Primitive(Primitive::BackdropFilter(bf)) = op {
                if let Some(slots) = slots {
                    if let Some(slot) = slots.backdrop_filter_slot {
                        bf_pairs.push((bf.order, slot as u32));
                    }
                }
            }
        }
        bf_pairs.sort_by_key(|(order, _)| *order);
        self.sorted_backdrop_filter_indices = bf_pairs.into_iter().map(|(_, slot)| slot).collect();

        let mut ul_pairs: Vec<(u32, u32)> = Vec::new();
        for (op, slots) in self.paint_operations.iter().zip(self.paint_slots.iter()) {
            if let PaintOperation::Primitive(Primitive::Underline(ul)) = op {
                if let Some(slots) = slots {
                    if let Some(slot) = slots.underline_slot {
                        ul_pairs.push((ul.order, slot as u32));
                    }
                }
            }
        }
        ul_pairs.sort_by_key(|(order, _)| *order);
        self.sorted_underline_indices = ul_pairs.into_iter().map(|(_, slot)| slot).collect();

        let mut mono_pairs: Vec<(u32, u32, u32)> = Vec::new();
        for (op, slots) in self.paint_operations.iter().zip(self.paint_slots.iter()) {
            if let PaintOperation::Primitive(Primitive::MonochromeSprite(sprite)) = op {
                if let Some(slots) = slots {
                    if let Some(slot) = slots.mono_sprite_slot {
                        mono_pairs.push((sprite.order, sprite.tile.texture_id.index, slot as u32));
                    }
                }
            }
        }
        mono_pairs.sort_by_key(|(order, _, _)| *order);
        self.sorted_mono_sprite_indices = mono_pairs.into_iter().map(|(_, _, slot)| slot).collect();

        let mut poly_pairs: Vec<(u32, u32, u32)> = Vec::new();
        for (op, slots) in self.paint_operations.iter().zip(self.paint_slots.iter()) {
            if let PaintOperation::Primitive(Primitive::PolychromeSprite(sprite)) = op {
                if let Some(slots) = slots {
                    if let Some(slot) = slots.poly_sprite_slot {
                        poly_pairs.push((sprite.order, sprite.tile.texture_id.index, slot as u32));
                    }
                }
            }
        }
        poly_pairs.sort_by_key(|(order, _, _)| *order);
        self.sorted_poly_sprite_indices = poly_pairs.into_iter().map(|(_, _, slot)| slot).collect();

        let mut surface_pairs: Vec<(u32, u32)> = Vec::new();
        for (op, slots) in self.paint_operations.iter().zip(self.paint_slots.iter()) {
            if let PaintOperation::Primitive(Primitive::Surface(surface)) = op {
                if let Some(slots) = slots {
                    if let Some(slot) = slots.surface_slot {
                        surface_pairs.push((surface.order, slot as u32));
                    }
                }
            }
        }
        surface_pairs.sort_by_key(|(order, _)| *order);
        self.sorted_surface_indices = surface_pairs.into_iter().map(|(_, slot)| slot).collect();

        let mut path_pairs: Vec<(u32, u32)> = Vec::new();
        for (op, slots) in self.paint_operations.iter().zip(self.paint_slots.iter()) {
            if let PaintOperation::Primitive(Primitive::Path(path)) = op {
                if let Some(slots) = slots {
                    if let Some(slot) = slots.path_slot {
                        path_pairs.push((path.order, slot as u32));
                    }
                }
            }
        }
        path_pairs.sort_by_key(|(order, _)| *order);
        self.sorted_path_indices = path_pairs.into_iter().map(|(_, slot)| slot).collect();
    }

    pub(crate) fn batches(&self) -> impl Iterator<Item = PrimitiveBatch<'_>> {
        BatchIterator {
            quad_indices: &self.sorted_quad_indices,
            quad_data: &self.quads,
            quad_cursor: 0,
            shadow_indices: &self.sorted_shadow_indices,
            shadow_data: &self.shadows,
            shadow_cursor: 0,
            path_indices: &self.sorted_path_indices,
            path_data: &self.paths,
            path_cursor: 0,
            underline_indices: &self.sorted_underline_indices,
            underline_data: &self.underlines,
            underline_cursor: 0,
            mono_sprite_indices: &self.sorted_mono_sprite_indices,
            mono_sprite_data: &self.monochrome_sprites,
            mono_sprite_cursor: 0,
            poly_sprite_indices: &self.sorted_poly_sprite_indices,
            poly_sprite_data: &self.polychrome_sprites,
            poly_sprite_cursor: 0,
            surface_indices: &self.sorted_surface_indices,
            surface_data: &self.surfaces,
            surface_cursor: 0,
            backdrop_filter_indices: &self.sorted_backdrop_filter_indices,
            backdrop_filter_data: &self.backdrop_filters,
            backdrop_filter_cursor: 0,
            filter_boundaries: &self.filter_boundaries,
            filter_boundaries_cursor: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Default)]
pub(crate) enum PrimitiveKind {
    // Lowest discriminant: at an equal order, a content-filter group-start is emitted before
    // the group's own content so the renderer redirects rendering before any child draws.
    FilterBoundaryStart,
    Shadow,
    #[default]
    Quad,
    Path,
    Underline,
    MonochromeSprite,
    PolychromeSprite,
    Surface,
    BackdropFilter,
    // Highest discriminant: at an equal order, a group-end is emitted after the group's content
    // so the renderer composites the filtered group only once every child has been drawn.
    FilterBoundaryEnd,
}

#[derive(Clone)]
pub(crate) enum PaintOperation {
    Primitive(Primitive),
    StartLayer(Bounds<ScaledPixels>),
    EndLayer,
}

#[derive(Clone)]
pub(crate) enum Primitive {
    Shadow(Shadow),
    Quad(Quad),
    Path(Path<ScaledPixels>),
    Underline(Underline),
    MonochromeSprite(MonochromeSprite),
    PolychromeSprite(PolychromeSprite),
    Surface(PaintSurface),
    BackdropFilter(BackdropFilter),
    FilterBoundary(FilterBoundary),
}

impl Primitive {
    pub fn bounds(&self) -> &Bounds<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.bounds,
            Primitive::Quad(quad) => &quad.bounds,
            Primitive::Path(path) => &path.bounds,
            Primitive::Underline(underline) => &underline.bounds,
            Primitive::MonochromeSprite(sprite) => &sprite.bounds,
            Primitive::PolychromeSprite(sprite) => &sprite.bounds,
            Primitive::Surface(surface) => &surface.bounds,
            Primitive::BackdropFilter(filter) => &filter.bounds,
            Primitive::FilterBoundary(boundary) => &boundary.bounds,
        }
    }

    pub fn content_mask(&self) -> &ContentMask<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.content_mask,
            Primitive::Quad(quad) => &quad.content_mask,
            Primitive::Path(path) => &path.content_mask,
            Primitive::Underline(underline) => &underline.content_mask,
            Primitive::MonochromeSprite(sprite) => &sprite.content_mask,
            Primitive::PolychromeSprite(sprite) => &sprite.content_mask,
            Primitive::Surface(surface) => &surface.content_mask,
            Primitive::BackdropFilter(filter) => &filter.content_mask,
            Primitive::FilterBoundary(boundary) => &boundary.content_mask,
        }
    }
}

pub(crate) unsafe fn as_bytes<T>(slice: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            slice.as_ptr() as *const u8,
            slice.len() * std::mem::size_of::<T>(),
        )
    }
}

/// Flatten a GenerationalVec into a byte buffer with zero-padding for free slots.
/// Each live slot's data occupies `sizeof<T>()` bytes at `slot_index * sizeof<T>()`.
pub(crate) fn flatten_generational_vec<T: Clone>(gv: &GenerationalVec<T>) -> Vec<u8> {
    let elem_size = std::mem::size_of::<T>();
    let mut bytes = Vec::with_capacity(gv.slots.len() * elem_size);
    for slot in &gv.slots {
        match slot {
            Some(s) => {
                let ptr = &s.data as *const T as *const u8;
                let slice = unsafe { std::slice::from_raw_parts(ptr, elem_size) };
                bytes.extend_from_slice(slice);
            }
            None => {
                bytes.extend(std::iter::repeat(0u8).take(elem_size));
            }
        }
    }
    bytes
}

struct BatchIterator<'a> {
    quad_indices: &'a [u32],
    quad_data: &'a GenerationalVec<Quad>,
    quad_cursor: usize,
    shadow_indices: &'a [u32],
    shadow_data: &'a GenerationalVec<Shadow>,
    shadow_cursor: usize,
    path_indices: &'a [u32],
    path_data: &'a GenerationalVec<Path<ScaledPixels>>,
    path_cursor: usize,
    underline_indices: &'a [u32],
    underline_data: &'a GenerationalVec<Underline>,
    underline_cursor: usize,
    mono_sprite_indices: &'a [u32],
    mono_sprite_data: &'a GenerationalVec<MonochromeSprite>,
    mono_sprite_cursor: usize,
    poly_sprite_indices: &'a [u32],
    poly_sprite_data: &'a GenerationalVec<PolychromeSprite>,
    poly_sprite_cursor: usize,
    surface_indices: &'a [u32],
    surface_data: &'a GenerationalVec<PaintSurface>,
    surface_cursor: usize,
    backdrop_filter_indices: &'a [u32],
    backdrop_filter_data: &'a GenerationalVec<BackdropFilter>,
    backdrop_filter_cursor: usize,
    filter_boundaries: &'a [FilterBoundary],
    filter_boundaries_cursor: usize,
}

impl<'a> BatchIterator<'a> {
    fn peek_shadow_order(&self) -> Option<u32> {
        let slot = *self.shadow_indices.get(self.shadow_cursor)?;
        self.shadow_data.get(slot as usize).map(|s| s.order)
    }
    fn peek_quad_order(&self) -> Option<u32> {
        let slot = *self.quad_indices.get(self.quad_cursor)?;
        self.quad_data.get(slot as usize).map(|q| q.order)
    }
    fn peek_path_order(&self) -> Option<u32> {
        let slot = *self.path_indices.get(self.path_cursor)?;
        self.path_data.get(slot as usize).map(|p| p.order)
    }
    fn peek_underline_order(&self) -> Option<u32> {
        let slot = *self.underline_indices.get(self.underline_cursor)?;
        self.underline_data.get(slot as usize).map(|u| u.order)
    }
    fn peek_mono_sprite_order(&self) -> Option<u32> {
        let slot = *self.mono_sprite_indices.get(self.mono_sprite_cursor)?;
        self.mono_sprite_data.get(slot as usize).map(|s| s.order)
    }
    fn peek_poly_sprite_order(&self) -> Option<u32> {
        let slot = *self.poly_sprite_indices.get(self.poly_sprite_cursor)?;
        self.poly_sprite_data.get(slot as usize).map(|s| s.order)
    }
    fn peek_surface_order(&self) -> Option<u32> {
        let slot = *self.surface_indices.get(self.surface_cursor)?;
        self.surface_data.get(slot as usize).map(|s| s.order)
    }
    fn peek_backdrop_filter_order(&self) -> Option<u32> {
        let slot = *self.backdrop_filter_indices.get(self.backdrop_filter_cursor)?;
        self.backdrop_filter_data.get(slot as usize).map(|f| f.order)
    }
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = PrimitiveBatch<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut orders_and_kinds = [
            (self.peek_shadow_order(), PrimitiveKind::Shadow),
            (self.peek_quad_order(), PrimitiveKind::Quad),
            (self.peek_path_order(), PrimitiveKind::Path),
            (self.peek_underline_order(), PrimitiveKind::Underline),
            (self.peek_mono_sprite_order(), PrimitiveKind::MonochromeSprite),
            (self.peek_poly_sprite_order(), PrimitiveKind::PolychromeSprite),
            (self.peek_surface_order(), PrimitiveKind::Surface),
            (self.peek_backdrop_filter_order(), PrimitiveKind::BackdropFilter),
            (
                self.filter_boundaries.get(self.filter_boundaries_cursor).map(|b| b.order),
                match self.filter_boundaries.get(self.filter_boundaries_cursor) {
                    Some(boundary) if boundary.is_start => PrimitiveKind::FilterBoundaryStart,
                    _ => PrimitiveKind::FilterBoundaryEnd,
                },
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
                let start = self.shadow_cursor;
                self.shadow_cursor += 1;
                while let Some(order) = self.peek_shadow_order() {
                    if (order, batch_kind) < max_order_and_kind {
                        self.shadow_cursor += 1;
                    } else {
                        break;
                    }
                }
                Some(PrimitiveBatch::Shadows(&self.shadow_indices[start..self.shadow_cursor]))
            }
            PrimitiveKind::Quad => {
                let start = self.quad_cursor;
                self.quad_cursor += 1;
                while let Some(order) = self.peek_quad_order() {
                    if (order, batch_kind) < max_order_and_kind {
                        self.quad_cursor += 1;
                    } else {
                        break;
                    }
                }
                Some(PrimitiveBatch::Quads(&self.quad_indices[start..self.quad_cursor]))
            }
            PrimitiveKind::Path => {
                let start = self.path_cursor;
                self.path_cursor += 1;
                while let Some(order) = self.peek_path_order() {
                    if (order, batch_kind) < max_order_and_kind {
                        self.path_cursor += 1;
                    } else {
                        break;
                    }
                }
                Some(PrimitiveBatch::Paths(&self.path_indices[start..self.path_cursor]))
            }
            PrimitiveKind::Underline => {
                let start = self.underline_cursor;
                self.underline_cursor += 1;
                while let Some(order) = self.peek_underline_order() {
                    if (order, batch_kind) < max_order_and_kind {
                        self.underline_cursor += 1;
                    } else {
                        break;
                    }
                }
                Some(PrimitiveBatch::Underlines(&self.underline_indices[start..self.underline_cursor]))
            }
            PrimitiveKind::MonochromeSprite => {
                let slot = self.mono_sprite_indices[self.mono_sprite_cursor] as usize;
                let texture_id = self.mono_sprite_data.get(slot).unwrap().tile.texture_id;
                let start = self.mono_sprite_cursor;
                self.mono_sprite_cursor += 1;
                while let Some(order) = self.peek_mono_sprite_order() {
                    let current_slot = self.mono_sprite_indices[self.mono_sprite_cursor] as usize;
                    let current_texture_id = self.mono_sprite_data.get(current_slot).unwrap().tile.texture_id;
                    if (order, batch_kind) < max_order_and_kind && current_texture_id == texture_id {
                        self.mono_sprite_cursor += 1;
                    } else {
                        break;
                    }
                }
                Some(PrimitiveBatch::MonochromeSprites {
                    texture_id,
                    indices: &self.mono_sprite_indices[start..self.mono_sprite_cursor],
                })
            }
            PrimitiveKind::PolychromeSprite => {
                let slot = self.poly_sprite_indices[self.poly_sprite_cursor] as usize;
                let texture_id = self.poly_sprite_data.get(slot).unwrap().tile.texture_id;
                let start = self.poly_sprite_cursor;
                self.poly_sprite_cursor += 1;
                while let Some(order) = self.peek_poly_sprite_order() {
                    let current_slot = self.poly_sprite_indices[self.poly_sprite_cursor] as usize;
                    let current_texture_id = self.poly_sprite_data.get(current_slot).unwrap().tile.texture_id;
                    if (order, batch_kind) < max_order_and_kind && current_texture_id == texture_id {
                        self.poly_sprite_cursor += 1;
                    } else {
                        break;
                    }
                }
                Some(PrimitiveBatch::PolychromeSprites {
                    texture_id,
                    indices: &self.poly_sprite_indices[start..self.poly_sprite_cursor],
                })
            }
            PrimitiveKind::Surface => {
                let start = self.surface_cursor;
                self.surface_cursor += 1;
                while let Some(order) = self.peek_surface_order() {
                    if (order, batch_kind) < max_order_and_kind {
                        self.surface_cursor += 1;
                    } else {
                        break;
                    }
                }
                Some(PrimitiveBatch::Surfaces(&self.surface_indices[start..self.surface_cursor]))
            }
            PrimitiveKind::BackdropFilter => {
                let start = self.backdrop_filter_cursor;
                self.backdrop_filter_cursor += 1;
                while let Some(order) = self.peek_backdrop_filter_order() {
                    if (order, batch_kind) < max_order_and_kind {
                        self.backdrop_filter_cursor += 1;
                    } else {
                        break;
                    }
                }
                Some(PrimitiveBatch::BackdropFilters(&self.backdrop_filter_indices[start..self.backdrop_filter_cursor]))
            }
            PrimitiveKind::FilterBoundaryStart | PrimitiveKind::FilterBoundaryEnd => {
                let index = self.filter_boundaries_cursor;
                self.filter_boundaries_cursor += 1;
                Some(PrimitiveBatch::FilterBoundary(index))
            }
        }
    }
}

#[derive(Debug)]
pub(crate) enum PrimitiveBatch<'a> {
    Shadows(&'a [u32]),
    Quads(&'a [u32]),
    Paths(&'a [u32]),
    Underlines(&'a [u32]),
    MonochromeSprites {
        texture_id: AtlasTextureId,
        indices: &'a [u32],
    },
    PolychromeSprites {
        texture_id: AtlasTextureId,
        indices: &'a [u32],
    },
    Surfaces(&'a [u32]),
    BackdropFilters(&'a [u32]),
    FilterBoundary(usize),
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

/// A backdrop filter blurs (and may otherwise filter) the content already rendered behind
/// `bounds`, compositing the result into a rounded rectangle — the frosted-glass effect.
/// Emitted by [`crate::Window::paint_backdrop_filter`]; produces the CSS `backdrop-filter` effect.
#[derive(Default, Debug, Copy, Clone)]
#[repr(C)]
pub(crate) struct BackdropFilter {
    pub order: DrawOrder,
    /// The largest blur radius among the element's backdrop filters, in scaled (device) pixels.
    ///
    /// Placed directly after `order` (rather than next to the other filter parameters) so the
    /// four bytes naturally pad the struct to an 8-byte boundary before `bounds`: WGSL aligns
    /// `Bounds` (built from `vec2<f32>`s) to 8 bytes in the storage buffer, and without an
    /// explicit 4-byte field here the compiler-inserted padding would desync this `#[repr(C)]`
    /// struct's layout from the shader's `BackdropFilter` struct, corrupting every field read on
    /// the GPU.
    pub blur_radius: ScaledPixels,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    /// Element opacity captured at paint time, multiplied into the composited result.
    pub opacity: f32,
    /// Pads the struct to 64 bytes (a multiple of 8) to match the storage-buffer stride WGSL
    /// derives for `array<BackdropFilter>` — see the note on `blur_radius` above.
    pub _pad: u32,
}

impl From<BackdropFilter> for Primitive {
    fn from(filter: BackdropFilter) -> Self {
        Primitive::BackdropFilter(filter)
    }
}

/// The start or end marker of a content-filter (`filter`) isolation group. The element's
/// subtree is painted between a matched start/end pair; the renderer redirects that span into
/// an offscreen target, filters it, and composites it back at `bounds`. Produces the CSS
/// `filter` effect (e.g. blurring the element and its children as a single group).
#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub(crate) struct FilterBoundary {
    pub order: DrawOrder,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub blur_radius: ScaledPixels,
    pub opacity: f32,
    /// `true` for the start marker (opens the group), `false` for the end marker (closes it).
    pub is_start: bool,
}

impl From<FilterBoundary> for Primitive {
    fn from(boundary: FilterBoundary) -> Self {
        Primitive::FilterBoundary(boundary)
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
    use crate::{Point, Size};

    fn sp(value: f32) -> ScaledPixels {
        ScaledPixels(value)
    }

    /// All test primitives cover the same region so the bounds tree assigns strictly
    /// increasing orders in insertion order — making the expected batch order deterministic.
    fn full_bounds() -> Bounds<ScaledPixels> {
        Bounds {
            origin: Point {
                x: sp(0.0),
                y: sp(0.0),
            },
            size: Size {
                width: sp(100.0),
                height: sp(100.0),
            },
        }
    }

    fn mask() -> ContentMask<ScaledPixels> {
        ContentMask {
            bounds: full_bounds(),
        }
    }

    fn quad() -> Quad {
        Quad {
            bounds: full_bounds(),
            content_mask: mask(),
            ..Default::default()
        }
    }

    fn boundary(is_start: bool) -> FilterBoundary {
        FilterBoundary {
            order: 0,
            bounds: full_bounds(),
            content_mask: mask(),
            corner_radii: Corners::default(),
            blur_radius: sp(8.0),
            opacity: 1.0,
            is_start,
        }
    }

    fn backdrop() -> BackdropFilter {
        BackdropFilter {
            bounds: full_bounds(),
            content_mask: mask(),
            corner_radii: Corners::default(),
            blur_radius: sp(20.0),
            opacity: 1.0,
            ..Default::default()
        }
    }

    fn batch_kinds(scene: &mut Scene) -> Vec<&'static str> {
        scene.sort();
        scene
            .batches()
            .map(|batch| match batch {
                PrimitiveBatch::Quads(_) => "quad",
                PrimitiveBatch::BackdropFilters(_) => "backdrop",
                PrimitiveBatch::FilterBoundary(ix) => {
                    if scene.filter_boundaries[ix].is_start {
                        "start"
                    } else {
                        "end"
                    }
                }
                _ => "other",
            })
            .collect()
    }

    #[test]
    fn content_filter_group_brackets_its_children() {
        let mut scene = Scene::default();
        // Background painted before the filtered element.
        scene.insert_primitive(quad());
        // A content-filtered element: start marker, its child, end marker.
        scene.insert_primitive(boundary(true));
        scene.insert_primitive(quad());
        scene.insert_primitive(boundary(false));

        // The start must precede the group's child and the end must follow it, so the
        // renderer can redirect rendering for exactly the group's span.
        assert_eq!(
            batch_kinds(&mut scene),
            vec!["quad", "start", "quad", "end"]
        );
    }

    // Note: this validates only the *scene ordering* of nested filter boundaries (start/child/
    // end interleaving), not that a renderer actually isolates both levels — that depends on the
    // backend's group-texture pool (see MAX_FILTER_DEPTH) and is exercised by the `blur` example.
    #[test]
    fn nested_content_filters_emit_well_nested_ordering() {
        let mut scene = Scene::default();
        scene.insert_primitive(boundary(true)); // outer start
        scene.insert_primitive(quad()); // outer child
        scene.insert_primitive(boundary(true)); // inner start
        scene.insert_primitive(quad()); // inner child
        scene.insert_primitive(boundary(false)); // inner end
        scene.insert_primitive(boundary(false)); // outer end

        assert_eq!(
            batch_kinds(&mut scene),
            vec!["start", "quad", "start", "quad", "end", "end"]
        );
    }

    #[test]
    fn backdrop_filter_sorts_before_a_later_overlapping_quad() {
        let mut scene = Scene::default();
        // Content behind the frosted panel.
        scene.insert_primitive(quad());
        // The panel: its backdrop snapshot, then its (translucent) background quad on top.
        scene.insert_primitive(backdrop());
        scene.insert_primitive(quad());

        assert_eq!(batch_kinds(&mut scene), vec!["quad", "backdrop", "quad"]);
    }
}
