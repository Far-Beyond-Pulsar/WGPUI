//! Persistent GPU-resident scene: layers, tiles, and the slab allocator
//! backing them. See docs/gpu-native-architecture.md §3.1, and R-N Pillar
//! III (the layer/slab concept this crate's mechanics replace, not discard).
//!
//! [`Scene`] is the assembly point only — the layer table, the one slab
//! allocator every kind's arena lives in, one [`PrimitiveStore`] per primitive
//! kind, and one [`RecordStore`] per CPU-side record category. Every mechanism
//! lives in the module that owns it; this file exists so `patch::apply` has a
//! single thing to apply a frame's patches *to*.

pub mod atlas;
pub mod layer;
pub mod primitive_store;
pub mod record;
pub mod slab;
pub mod slab_range;
pub mod tile;

pub use atlas::{
    AtlasEviction, AtlasKey, AtlasKind, GlyphRasterKey, GlyphTile, GlyphTileSource, ImageRasterKey,
    ImageTile, ImageTileSource, RasterizedGlyph, RasterizedImage,
};
pub use layer::{BoundaryId, Layer, LayerId, LayerKey, LayerTable};
pub use primitive_store::{DrawRange, PrimitiveStore};
pub use record::{DispatchNode, Hitbox, LayoutInput, RecordStore};
pub use slab::{Reallocation, SlabAllocator, SlabOverflow};
pub use slab_range::{
    PrimitiveSlotDiff, SlotChange, SlotSpan, SlabRange, UploadRange, coalesce_uploads,
    uploaded_byte_count,
};
pub use tile::{
    EvictedTile, TILE_DESCRIPTOR_STRIDE, TileCoord, TileDescriptor, TileEviction, TileGrid,
    TilePlacement, TileResidency, TileSpan, TileVisibility, encode_tiles, tile_visibility,
};

use crate::indirect::{DrawSlot, SlotTable};
use crate::patch::primitive::{
    BackdropFilter, GlyphRun, Path, PolySprite, PrimitiveKind, Quad, Shadow, Underline,
};

/// The persistent, patched-not-rebuilt scene (R-N Pillar III, §2's picture).
///
/// # Why the primitive stores are named fields rather than a map
///
/// `patch/primitive.rs`'s module doc sets out the reasoning in full: the kind
/// set is closed at compile time because each kind also needs its own render
/// pipeline in `wgpui-wgpu` (§3.5), so the protocol is generic and
/// monomorphised per kind rather than boxed. A `HashMap<PrimitiveKind, Box<dyn
/// AnyStore>>` would buy dynamic extensibility nothing wants and cost a
/// downcast per patch. Adding a kind is one field here and one
/// [`crate::patch::primitive::PrimitiveKind`] variant.
#[derive(Debug, Default)]
pub struct Scene {
    /// Every live layer's record.
    pub layers: LayerTable,
    /// Slot placement for every kind's arena.
    pub allocator: SlabAllocator,
    /// Blurred rounded rectangles, painted under everything else in their layer
    /// (Phase 6.3).
    pub shadows: PrimitiveStore<Shadow>,
    /// Fixed-size primitives (§2's "primitives").
    pub quads: PrimitiveStore<Quad>,
    /// Underline and strikethrough rules, painted under their layer's text
    /// (Phase 6.3).
    pub underlines: PrimitiveStore<Underline>,
    /// Variable-size primitives (§2's "primitives").
    pub glyph_runs: PrimitiveStore<GlyphRun>,
    /// Colour-atlas sprites — images and rasterised SVGs (Phase 6.2).
    pub poly_sprites: PrimitiveStore<PolySprite>,
    /// Lyon-tessellated vector paths (Phase 6.4).
    pub paths: PrimitiveStore<Path>,
    /// Framebuffer-sampling backdrop filters (Phase 6.4).
    pub backdrop_filters: PrimitiveStore<BackdropFilter>,
    /// §2's "layout inputs".
    pub layout_inputs: RecordStore<LayoutInput>,
    /// §2's "hitboxes".
    pub hitboxes: RecordStore<Hitbox>,
    /// §2's "dispatch nodes".
    pub dispatch_nodes: RecordStore<DispatchNode>,
}

impl Scene {
    /// An empty scene: no layers, no residency, no pending uploads.
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare the layer for `key`, returning its handle. Idempotent — see
    /// [`LayerTable::insert`].
    pub fn layer(&mut self, key: LayerKey) -> LayerId {
        self.layers.insert(key)
    }

    /// Every layer's CPU-computed instanced draw range, per kind, in layer
    /// order.
    ///
    /// Phase 1's scope is explicit that draw ranges stay CPU-computed —
    /// "same as today, just through the new protocol" (§8) — so this is
    /// deliberately the same first-instance/count pair the legacy renderer
    /// derives, read straight off the slab reservations the patches produced
    /// rather than off a per-frame scene walk. Phase 3/4 replace this with
    /// GPU-computed indirect draw args (§5.1–§5.3) over the identical slabs.
    ///
    /// Layers holding nothing of a kind are omitted, not reported as
    /// zero-length: an empty entry would become an empty draw call.
    pub fn draw_ranges(&self) -> Vec<(LayerId, PrimitiveKind, DrawRange)> {
        let mut ranges = Vec::new();
        for layer in self.layers.ids() {
            if let Some(range) = self.shadows.draw_range(layer) {
                ranges.push((layer, PrimitiveKind::Shadow, range));
            }
            if let Some(range) = self.quads.draw_range(layer) {
                ranges.push((layer, PrimitiveKind::Quad, range));
            }
            if let Some(range) = self.underlines.draw_range(layer) {
                ranges.push((layer, PrimitiveKind::Underline, range));
            }
            if let Some(range) = self.glyph_runs.draw_range(layer) {
                ranges.push((layer, PrimitiveKind::GlyphRun, range));
            }
            if let Some(range) = self.poly_sprites.draw_range(layer) {
                ranges.push((layer, PrimitiveKind::PolySprite, range));
            }
            if let Some(range) = self.paths.draw_range(layer) {
                ranges.push((layer, PrimitiveKind::Path, range));
            }
            if let Some(range) = self.backdrop_filters.draw_range(layer) {
                ranges.push((layer, PrimitiveKind::BackdropFilter, range));
            }
        }
        ranges
    }

    /// The fixed (layer, kind) slot sequence §5.3's indirect draw issues every
    /// frame, grouped by kind and ascending by layer within a kind.
    ///
    /// This is [`Self::draw_ranges`]'s successor and the difference between the
    /// two is the whole of §8's Phase 4 gate. `draw_ranges` reports how many
    /// instances each slot draws, which is a fact about the scene's *contents*;
    /// this reports only where each slot's reservation lives, which is a fact
    /// about its *residency*. The instance count is what the GPU decides
    /// ([`crate::indirect`]), so the CPU never asks for it and never learns it.
    ///
    /// Cost is `O(layers × kinds)` — one `SlabRange` read per slot, no
    /// primitive touched — which is exactly the claim §8's Phase 4 gate makes
    /// falsifiable.
    ///
    /// **Every live layer contributes a slot for every kind, including the
    /// kinds it holds nothing of.** §5.3's wording is deliberate — "one per
    /// (layer, kind) slot that *could* be populated … regardless of how many
    /// are actually zero" — and omitting the empty ones would make the sequence
    /// change shape whenever a layer gained or lost its first glyph run, which
    /// is precisely the per-frame CPU re-planning the phase exists to stop. An
    /// unreserved slot reports `base: 0, count: 0` and draws zero instances.
    pub fn draw_slots(&self) -> SlotTable {
        let ids = self.layers.ids();
        let mut slots = Vec::with_capacity(ids.len() * PrimitiveKind::COUNT);
        for kind in PrimitiveKind::ALL {
            for layer in &ids {
                let range = match kind {
                    PrimitiveKind::Shadow => self.shadows.slab(*layer),
                    PrimitiveKind::Quad => self.quads.slab(*layer),
                    PrimitiveKind::Path => self.paths.slab(*layer),
                    PrimitiveKind::Underline => self.underlines.slab(*layer),
                    PrimitiveKind::GlyphRun => self.glyph_runs.slab(*layer),
                    PrimitiveKind::PolySprite => self.poly_sprites.slab(*layer),
                    PrimitiveKind::BackdropFilter => self.backdrop_filters.slab(*layer),
                };
                slots.push(DrawSlot {
                    layer: *layer,
                    kind,
                    base: range.base,
                    count: range.count,
                });
            }
        }
        SlotTable::from_grouped(slots).unwrap_or_default()
    }

    /// How many slots one kind's arena currently holds — the length of the
    /// arena-shaped buffers [`crate::indirect`] addresses.
    /// Saturating rather than fallible: an arena wider than `u32::MAX` slots
    /// cannot exist, because every `SlabRange::base` is already a `u32` and the
    /// allocator rejects a reservation it cannot address (`SlabOverflow`).
    pub fn arena_slots(&self, kind: PrimitiveKind) -> u32 {
        u32::try_from(self.allocator.arena_slot_capacity(kind)).unwrap_or(u32::MAX)
    }

    /// Drop a layer and release every reservation and record it held.
    ///
    /// Returns whether the layer existed. Releasing the primitive stores'
    /// reservations *before* dropping the layer record is what keeps the
    /// allocator's accounting exact: the record is only the caller's view of
    /// the reservation, never its owner.
    pub fn remove_layer(&mut self, layer: LayerId) -> bool {
        self.shadows.remove_layer(layer, &mut self.allocator);
        self.quads.remove_layer(layer, &mut self.allocator);
        self.underlines.remove_layer(layer, &mut self.allocator);
        self.glyph_runs.remove_layer(layer, &mut self.allocator);
        self.poly_sprites.remove_layer(layer, &mut self.allocator);
        self.paths.remove_layer(layer, &mut self.allocator);
        self.backdrop_filters
            .remove_layer(layer, &mut self.allocator);
        self.layout_inputs.remove_layer(layer);
        self.hitboxes.remove_layer(layer);
        self.dispatch_nodes.remove_layer(layer);
        self.layers.remove(layer).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patch::PatchList;
    use crate::patch::primitive::Primitive;

    #[test]
    fn a_fresh_scene_holds_nothing() {
        let scene = Scene::new();
        assert!(scene.layers.is_empty());
        assert!(scene.shadows.resident_bytes().is_empty());
        assert!(scene.quads.resident_bytes().is_empty());
        assert!(scene.underlines.resident_bytes().is_empty());
        assert!(scene.glyph_runs.resident_bytes().is_empty());
        assert!(scene.poly_sprites.resident_bytes().is_empty());
        assert!(scene.paths.resident_bytes().is_empty());
        assert!(scene.backdrop_filters.resident_bytes().is_empty());
    }

    #[test]
    fn draw_ranges_come_straight_off_the_slabs_the_patches_produced()
    -> Result<(), crate::patch::PatchError> {
        use crate::patch::apply::{ScenePatch, apply};
        use crate::patch::primitive::{Glyph, GlyphRun};

        let mut scene = Scene::new();
        let empty = scene.layer(LayerKey::untiled(BoundaryId::from_raw(1)));
        let populated = scene.layer(LayerKey::untiled(BoundaryId::from_raw(2)));

        let mut patch = ScenePatch::new();
        for index in 0..3u32 {
            patch.quads.append(
                populated,
                crate::patch::RecordKey::from_raw(index as u64 + 1),
                index,
                Quad::ZERO,
            );
        }
        patch.glyph_runs.insert(
            populated,
            crate::patch::RecordKey::from_raw(100),
            0,
            GlyphRun {
                color: [1.0, 1.0, 1.0, 1.0],
                glyphs: vec![Glyph::ZERO; 5],
            },
        );
        apply(&mut scene, &patch)?;

        let ranges = scene.draw_ranges();
        assert_eq!(ranges.len(), 2, "the empty layer issues no draw at all");
        assert!(ranges.iter().all(|(layer, _, _)| *layer == populated));
        assert!(!scene.layers.contains(empty) || scene.quads.draw_range(empty).is_none());
        let quads = ranges
            .iter()
            .find(|(_, kind, _)| *kind == PrimitiveKind::Quad)
            .map(|(_, _, range)| *range);
        assert_eq!(
            quads,
            Some(DrawRange {
                first_instance: scene.quads.slab(populated).base,
                instance_count: 3,
            })
        );
        let runs = ranges
            .iter()
            .find(|(_, kind, _)| *kind == PrimitiveKind::GlyphRun)
            .map(|(_, _, range)| range.instance_count);
        assert_eq!(runs, Some(5), "a glyph run draws one instance per glyph");
        Ok(())
    }

    /// §5.3/§8's Phase 4 gate, at the level `wgpui-core` can state it: the
    /// slot table is a function of residency, not of contents. Two scenes with
    /// the same layers and wildly different primitive counts produce slot
    /// tables of the same length, and neither table names an instance count.
    #[test]
    fn the_slot_table_is_the_same_length_however_many_primitives_a_layer_holds()
    -> Result<(), crate::patch::PatchError> {
        use crate::patch::apply::{ScenePatch, apply};

        let build = |per_layer: u32| -> Result<SlotTable, crate::patch::PatchError> {
            let mut scene = Scene::new();
            let mut patch = ScenePatch::new();
            let mut key = 0u64;
            for boundary in 0..4u64 {
                let layer = scene.layer(LayerKey::untiled(BoundaryId::from_raw(boundary + 1)));
                for index in 0..per_layer {
                    key += 1;
                    patch.quads.append(
                        layer,
                        crate::patch::RecordKey::from_raw(key),
                        index,
                        Quad::ZERO,
                    );
                }
            }
            apply(&mut scene, &patch)?;
            Ok(scene.draw_slots())
        };

        let small = build(4)?;
        let large = build(40_000)?;
        assert_eq!(small.len(), 4 * PrimitiveKind::COUNT);
        assert_eq!(
            small.len(),
            large.len(),
            "the fixed draw sequence is one entry per (layer, kind) slot, \
             independent of resident primitive count"
        );
        assert_eq!(large.kind_slots(PrimitiveKind::Quad).len(), 4);
        assert_eq!(
            large.kind_slots(PrimitiveKind::GlyphRun).len(),
            4,
            "§5.3: a kind a layer holds nothing of is still a slot that could \
             be populated, so it keeps its place in the sequence"
        );
        assert!(
            large
                .kind_slots(PrimitiveKind::GlyphRun)
                .iter()
                .all(|slot| slot.count == 0)
        );
        Ok(())
    }

    #[test]
    fn slots_name_a_reservation_and_never_an_instance_count() -> Result<(), crate::patch::PatchError>
    {
        use crate::patch::apply::{ScenePatch, apply};

        let mut scene = Scene::new();
        let layer = scene.layer(LayerKey::untiled(BoundaryId::from_raw(9)));
        let mut patch = ScenePatch::new();
        for index in 0..7u32 {
            patch.quads.append(
                layer,
                crate::patch::RecordKey::from_raw(index as u64 + 1),
                index,
                Quad::ZERO,
            );
        }
        apply(&mut scene, &patch)?;

        let table = scene.draw_slots();
        // Addressed by kind rather than by `.first()`: the sequence is grouped
        // in `PrimitiveKind::ALL` order, so a position-indexed assertion here
        // silently becomes an assertion about *which kind sorts first* every
        // time a kind is added. Phase 6.3 added one below `Quad` and this test
        // failed for exactly that reason.
        let slot = table.kind_slots(PrimitiveKind::Quad).first().copied();
        assert_eq!(
            slot,
            Some(DrawSlot {
                layer,
                kind: PrimitiveKind::Quad,
                base: scene.quads.slab(layer).base,
                count: 7,
            })
        );
        assert!(scene.arena_slots(PrimitiveKind::Quad) >= 7);
        Ok(())
    }

    #[test]
    fn a_layer_with_no_reservation_of_a_kind_still_holds_its_place_in_the_sequence() {
        let mut scene = Scene::new();
        scene.layer(LayerKey::untiled(BoundaryId::ROOT));
        let table = scene.draw_slots();
        assert_eq!(table.len(), PrimitiveKind::COUNT);
        assert!(
            table.slots().iter().all(|slot| slot.count == 0),
            "an unreserved slot draws zero instances rather than being omitted"
        );
        assert!(Scene::new().draw_slots().is_empty(), "no layers, no slots");
    }

    #[test]
    fn dropping_a_layer_releases_every_store_it_touched() -> Result<(), crate::patch::PatchError> {
        let mut scene = Scene::new();
        let layer = scene.layer(LayerKey::untiled(BoundaryId::ROOT));
        let mut quads = PatchList::new();
        quads.insert(layer, crate::patch::RecordKey::from_raw(1), 0, Quad::ZERO);
        let mut uploads = Vec::new();
        scene
            .quads
            .apply(&quads, &mut scene.allocator, &mut uploads)?;
        assert!(scene.remove_layer(layer));
        assert_eq!(scene.allocator.arena_slot_capacity(Quad::KIND), 0);
        assert!(!scene.layers.contains(layer));
        assert!(!scene.remove_layer(layer));
        Ok(())
    }
}
