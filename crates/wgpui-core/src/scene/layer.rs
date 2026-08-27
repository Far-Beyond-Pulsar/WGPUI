//! Layer records: identity, per-kind slab ranges, invalidation state.
//! R-N's `Layer` concept with 2.0's mechanics.
//! See docs/gpu-native-architecture.md §3.1, §4.1, §4.3.
//!
//! A layer here is *only* the unit of GPU residency and invalidation — the
//! thing that owns slab ranges and can be clean or dirty. It is deliberately
//! not the unit of reconciliation: §4.0 makes reconciliation ambient, so
//! `wgpui-core` never asks "is this element inside a layer" before deciding
//! whether to diff it. That is the single largest substantive difference from
//! what shipped for R-N/SFD, and it shows up in this file as an absence: no
//! `Layer` field, method, or lifetime hook has anything to do with
//! `ElementInstance`.

use crate::invalidation::axes::Invalidation;
use crate::patch::primitive::PrimitiveKind;
use crate::scene::slab_range::SlabRange;
use crate::scene::tile::TileCoord;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Identity of the compositing boundary a layer belongs to.
///
/// Phase 1 has no `.boundary()` API at all (§8's Phase 1 row is explicit:
/// "with no `.boundary()` involved at all"), so a boundary id is simply an
/// opaque, caller-chosen identity — in practice the window root's. Phase 2 is
/// what derives these from positional identity (§4.1, SFD §1.0); nothing about
/// this type has to change when it does.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BoundaryId(u64);

impl BoundaryId {
    /// The window's own root boundary, which always exists.
    pub const ROOT: BoundaryId = BoundaryId(0);

    /// Wrap a raw identity.
    pub const fn from_raw(raw: u64) -> Self {
        BoundaryId(raw)
    }

    /// The raw identity.
    pub const fn as_raw(self) -> u64 {
        self.0
    }
}

/// A layer's cross-frame address: which boundary owns it, and — once §4.3's
/// tiling exists — which tile of that boundary's content plane it holds.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LayerKey {
    /// The owning compositing boundary.
    pub boundary: BoundaryId,
    /// The tile within that boundary, or `None` for an untiled boundary —
    /// which is every boundary in Phase 1.
    pub tile: Option<TileCoord>,
}

impl LayerKey {
    /// The untiled layer of `boundary`.
    pub const fn untiled(boundary: BoundaryId) -> Self {
        Self {
            boundary,
            tile: None,
        }
    }

    /// The layer holding `tile` of `boundary` (§4.3, Phase 4.5).
    pub const fn tiled(boundary: BoundaryId, tile: TileCoord) -> Self {
        Self {
            boundary,
            tile: Some(tile),
        }
    }
}

/// A layer's handle, derived deterministically from its [`LayerKey`].
///
/// Deriving rather than assigning is what keeps the patch protocol pure data
/// (§2): a producer can name a layer in a `PatchList` without first asking a
/// scene to allocate an id for it, so building a frame's patches never
/// requires a round trip into the backend.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LayerId(u64);

impl LayerId {
    /// The handle for `key`.
    pub fn from_key(key: LayerKey) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        // Reserve 0 so a defaulted handle can never name a live layer.
        LayerId(hasher.finish() | 1)
    }

    /// Wrap a raw handle. Intended for tests.
    pub const fn from_raw(raw: u64) -> Self {
        LayerId(raw)
    }

    /// The raw handle.
    pub const fn as_raw(self) -> u64 {
        self.0
    }
}

/// One layer's retained record.
#[derive(Clone, Debug)]
pub struct Layer {
    key: LayerKey,
    slabs: [SlabRange; PrimitiveKind::COUNT],
    invalidation: Invalidation,
    generation: u64,
}

impl Layer {
    /// The layer's cross-frame address.
    pub const fn key(&self) -> LayerKey {
        self.key
    }

    /// This layer's reservation in `kind`'s arena.
    pub fn slab(&self, kind: PrimitiveKind) -> SlabRange {
        self.slabs[kind.index() % PrimitiveKind::COUNT]
    }

    /// Which respects of this layer are currently stale.
    pub const fn invalidation(&self) -> Invalidation {
        self.invalidation
    }

    /// Whether this layer has nothing stale — the case §5.0 requires to upload
    /// zero bytes, not a small range.
    pub const fn is_clean(&self) -> bool {
        self.invalidation.is_empty()
    }

    /// Bumped on every change to this record: any reservation, resize,
    /// release, or content edit. Drawn from a table-global monotonic counter,
    /// so a layer destroyed and recreated under the same key can never alias a
    /// stale snapshot's generation.
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Whether this layer holds no slots of any kind.
    pub fn is_empty(&self) -> bool {
        self.slabs.iter().all(|range| range.is_empty())
    }
}

/// Every live layer, addressed by [`LayerId`].
#[derive(Debug, Default)]
pub struct LayerTable {
    layers: HashMap<LayerId, Layer>,
    next_generation: u64,
}

impl LayerTable {
    /// An empty table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create the layer for `key`, or return the existing handle unchanged.
    ///
    /// Idempotent on purpose: a frame that re-declares a layer it already has
    /// must not disturb that layer's residency, which is what makes "a clean
    /// layer uploads zero bytes" survive a producer that names its layers
    /// every frame.
    pub fn insert(&mut self, key: LayerKey) -> LayerId {
        let id = LayerId::from_key(key);
        if !self.layers.contains_key(&id) {
            let generation = self.next_generation();
            self.layers.insert(
                id,
                Layer {
                    key,
                    slabs: [SlabRange::EMPTY; PrimitiveKind::COUNT],
                    invalidation: Invalidation::all(),
                    generation,
                },
            );
        }
        id
    }

    /// Drop a layer, returning its record so the caller can release its slab
    /// reservations. `None` if no such layer exists.
    pub fn remove(&mut self, id: LayerId) -> Option<Layer> {
        self.layers.remove(&id)
    }

    /// The layer's record.
    pub fn get(&self, id: LayerId) -> Option<&Layer> {
        self.layers.get(&id)
    }

    /// Whether the table holds this layer.
    pub fn contains(&self, id: LayerId) -> bool {
        self.layers.contains_key(&id)
    }

    /// How many layers are live.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Whether no layers are live.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Every live layer handle, in ascending handle order.
    ///
    /// Sorted rather than in hash order so a caller that snapshots the whole
    /// scene gets a deterministic result — which is what makes Phase 1's
    /// round-trip gate an exact byte comparison rather than a set comparison.
    pub fn ids(&self) -> Vec<LayerId> {
        let mut ids: Vec<LayerId> = self.layers.keys().copied().collect();
        ids.sort_unstable();
        ids
    }

    /// Record a layer's new reservation for `kind` and bump its generation.
    pub fn set_slab(&mut self, id: LayerId, kind: PrimitiveKind, range: SlabRange) -> bool {
        let generation = self.next_generation();
        match self.layers.get_mut(&id) {
            Some(layer) => {
                layer.slabs[kind.index() % PrimitiveKind::COUNT] = range;
                layer.generation = generation;
                true
            }
            None => false,
        }
    }

    /// Add invalidation axes to a layer and bump its generation.
    pub fn invalidate(&mut self, id: LayerId, axes: Invalidation) -> bool {
        if axes.is_empty() {
            return self.layers.contains_key(&id);
        }
        let generation = self.next_generation();
        match self.layers.get_mut(&id) {
            Some(layer) => {
                layer.invalidation |= axes;
                layer.generation = generation;
                true
            }
            None => false,
        }
    }

    /// Clear a layer's invalidation, marking it clean for the next frame.
    /// Does *not* bump the generation: nothing about the layer's residency
    /// changed, only the framework's opinion of whether it needs work.
    pub fn mark_clean(&mut self, id: LayerId) -> bool {
        match self.layers.get_mut(&id) {
            Some(layer) => {
                layer.invalidation = Invalidation::empty();
                true
            }
            None => false,
        }
    }

    fn next_generation(&mut self) -> u64 {
        self.next_generation = self.next_generation.wrapping_add(1);
        self.next_generation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_handles_are_derived_deterministically_from_their_key() {
        let key = LayerKey::untiled(BoundaryId::ROOT);
        assert_eq!(LayerId::from_key(key), LayerId::from_key(key));
        assert_ne!(
            LayerId::from_key(key),
            LayerId::from_key(LayerKey::untiled(BoundaryId::from_raw(1)))
        );
    }

    #[test]
    fn tiles_of_one_boundary_are_distinct_layers() {
        let boundary = BoundaryId::from_raw(4);
        let first = LayerId::from_key(LayerKey::tiled(boundary, TileCoord::new(0, 0)));
        let second = LayerId::from_key(LayerKey::tiled(boundary, TileCoord::new(1, 0)));
        let untiled = LayerId::from_key(LayerKey::untiled(boundary));
        assert_ne!(first, second);
        assert_ne!(first, untiled);
    }

    #[test]
    fn inserting_an_existing_layer_leaves_it_untouched() {
        let mut table = LayerTable::new();
        let key = LayerKey::untiled(BoundaryId::ROOT);
        let id = table.insert(key);
        assert!(table.mark_clean(id));
        let generation = table.get(id).map(Layer::generation);
        assert_eq!(table.insert(key), id);
        assert_eq!(table.get(id).map(Layer::generation), generation);
        assert_eq!(table.get(id).map(Layer::is_clean), Some(true));
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn a_new_layer_starts_fully_invalidated() {
        let mut table = LayerTable::new();
        let id = table.insert(LayerKey::untiled(BoundaryId::ROOT));
        assert_eq!(
            table.get(id).map(Layer::invalidation),
            Some(Invalidation::all())
        );
    }

    #[test]
    fn generations_are_monotonic_across_layers() {
        let mut table = LayerTable::new();
        let first = table.insert(LayerKey::untiled(BoundaryId::from_raw(1)));
        let second = table.insert(LayerKey::untiled(BoundaryId::from_raw(2)));
        let before = table.get(first).map(Layer::generation);
        assert!(table.invalidate(first, Invalidation::DISPLAY));
        let after = table.get(first).map(Layer::generation);
        assert!(after > before);
        assert!(after > table.get(second).map(Layer::generation));
    }

    #[test]
    fn ids_are_sorted_so_a_scene_snapshot_is_deterministic() {
        let mut table = LayerTable::new();
        for raw in 0..16u64 {
            table.insert(LayerKey::untiled(BoundaryId::from_raw(raw)));
        }
        let ids = table.ids();
        assert_eq!(ids.len(), 16);
        assert!(ids.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn removing_an_unknown_layer_is_inert() {
        let mut table = LayerTable::new();
        assert!(table.remove(LayerId::from_raw(99)).is_none());
        assert!(!table.invalidate(LayerId::from_raw(99), Invalidation::HIT));
        assert!(!table.mark_clean(LayerId::from_raw(99)));
        assert!(table.is_empty());
    }
}
