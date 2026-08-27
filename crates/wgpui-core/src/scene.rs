//! Persistent GPU-resident scene: layers, tiles, and the slab allocator
//! backing them. See docs/gpu-native-architecture.md §3.1, and R-N Pillar
//! III (the layer/slab concept this crate's mechanics replace, not discard).
//!
//! [`Scene`] is the assembly point only — the layer table, the one slab
//! allocator every kind's arena lives in, one [`PrimitiveStore`] per primitive
//! kind, and one [`RecordStore`] per CPU-side record category. Every mechanism
//! lives in the module that owns it; this file exists so `patch::apply` has a
//! single thing to apply a frame's patches *to*.

pub mod layer;
pub mod primitive_store;
pub mod record;
pub mod slab;
pub mod slab_range;
pub mod tile;

pub use layer::{BoundaryId, Layer, LayerId, LayerKey, LayerTable};
pub use primitive_store::{DrawRange, PrimitiveStore};
pub use record::{DispatchNode, Hitbox, LayoutInput, RecordStore};
pub use slab::{Reallocation, SlabAllocator, SlabOverflow};
pub use slab_range::{SlabRange, UploadRange, coalesce_uploads, uploaded_byte_count};
pub use tile::TileCoord;

use crate::patch::primitive::{GlyphRun, PrimitiveKind, Quad};

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
    /// Fixed-size primitives (§2's "primitives").
    pub quads: PrimitiveStore<Quad>,
    /// Variable-size primitives (§2's "primitives").
    pub glyph_runs: PrimitiveStore<GlyphRun>,
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
            if let Some(range) = self.quads.draw_range(layer) {
                ranges.push((layer, PrimitiveKind::Quad, range));
            }
            if let Some(range) = self.glyph_runs.draw_range(layer) {
                ranges.push((layer, PrimitiveKind::GlyphRun, range));
            }
        }
        ranges
    }

    /// Drop a layer and release every reservation and record it held.
    ///
    /// Returns whether the layer existed. Releasing the primitive stores'
    /// reservations *before* dropping the layer record is what keeps the
    /// allocator's accounting exact: the record is only the caller's view of
    /// the reservation, never its owner.
    pub fn remove_layer(&mut self, layer: LayerId) -> bool {
        self.quads.remove_layer(layer, &mut self.allocator);
        self.glyph_runs.remove_layer(layer, &mut self.allocator);
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
        assert!(scene.quads.resident_bytes().is_empty());
        assert!(scene.glyph_runs.resident_bytes().is_empty());
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
            patch
                .quads
                .append(populated, crate::patch::RecordKey::from_raw(index as u64 + 1), index, Quad::ZERO);
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

    #[test]
    fn dropping_a_layer_releases_every_store_it_touched() -> Result<(), crate::patch::PatchError> {
        let mut scene = Scene::new();
        let layer = scene.layer(LayerKey::untiled(BoundaryId::ROOT));
        let mut quads = PatchList::new();
        quads.insert(
            layer,
            crate::patch::RecordKey::from_raw(1),
            0,
            Quad::ZERO,
        );
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
