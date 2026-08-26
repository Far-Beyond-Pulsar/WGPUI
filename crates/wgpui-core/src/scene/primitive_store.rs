//! The slab-backed half of the patch protocol: one primitive kind's resident
//! bytes, its per-layer slab ranges, and the upload instructions a frame's
//! patches produce. See docs/gpu-native-architecture.md §2, §5.0.
//!
//! Not in §3.1's literal file map — a deliberate addition, recorded in
//! `docs/phase-1-results.md`. §3.1 splits `scene/` into `layer.rs`, `tile.rs`,
//! `slab.rs`, and `slab_range.rs`; none of those is a home for "the resident
//! bytes themselves plus the per-record bookkeeping that turns a patch into an
//! upload," and putting it in `scene.rs` would have made that file the one
//! thing §3 exists to prevent.
//!
//! # The delta-upload contract, implemented
//!
//! §5.0 commits to three cases and this file is where each becomes true:
//!
//! - **Value update, slot count unchanged** — [`PrimitiveStore::apply`]'s
//!   `Update` arm re-encodes exactly that record's slots and emits exactly one
//!   [`UploadRange`] of `slot_count * SLOT_STRIDE` bytes, touching no other
//!   record and reading no other record, no matter how many layer-mates there
//!   are. This is the only path in the file that is O(1) in the layer's size,
//!   and it is the one §5.0's gate measures.
//! - **Insert/remove, or an update that changes a record's slot count** —
//!   every record after the edit shifts, so the upload covers the edited
//!   record and its successors, and a size-class change relocates the layer's
//!   whole range and rewrites it. Wider than O(1), bounded by the layer's own
//!   slab, disclosed rather than glossed.
//! - **A clean layer** — no patches, no upload instructions at all. Not a
//!   small range: zero.

use crate::patch::primitive::Primitive;
use crate::patch::{Patch, PatchError, PatchList, PatchOp, RecordKey};
use crate::scene::layer::LayerId;
use crate::scene::slab::SlabAllocator;
use crate::scene::slab_range::{SlabRange, UploadRange};
use std::collections::HashMap;

/// One record's value and its placement inside its layer's slab.
#[derive(Clone, Debug)]
struct StoredPrimitive<P> {
    value: P,
    /// Slots from the start of the layer's range to this record's first slot.
    slot_offset: u32,
    /// Slots this record occupies, cached so a removal or a reflow does not
    /// have to consult the value again after it has been replaced.
    slot_count: u32,
}

/// One layer's primitives of a single kind, in paint order.
#[derive(Clone, Debug)]
struct LayerPrimitives<P> {
    order: Vec<RecordKey>,
    records: HashMap<RecordKey, StoredPrimitive<P>>,
    range: SlabRange,
}

impl<P> Default for LayerPrimitives<P> {
    fn default() -> Self {
        Self {
            order: Vec::new(),
            records: HashMap::new(),
            range: SlabRange::EMPTY,
        }
    }
}

/// A CPU-computed instanced draw range for one (layer, kind) pair.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DrawRange {
    /// First instance index in the kind's arena.
    pub first_instance: u32,
    /// How many instances to draw.
    pub instance_count: u32,
}

/// Every layer's primitives of one kind, plus that kind's resident bytes.
///
/// Monomorphised per kind rather than boxed — see `patch/primitive.rs`'s
/// module doc for why the protocol is generic but not dynamic.
#[derive(Debug)]
pub struct PrimitiveStore<P: Primitive> {
    layers: HashMap<LayerId, LayerPrimitives<P>>,
    /// The kind's whole arena, byte for byte, exactly as a GPU buffer would
    /// hold it. Grown to the allocator's high-water mark and never shrunk: a
    /// GPU buffer is not reallocated mid-session either, and shrinking would
    /// invalidate offsets an already-produced upload instruction refers to.
    resident: Vec<u8>,
}

impl<P: Primitive> Default for PrimitiveStore<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: Primitive> PrimitiveStore<P> {
    /// An empty store.
    pub fn new() -> Self {
        Self {
            layers: HashMap::new(),
            resident: Vec::new(),
        }
    }

    /// Apply every patch in `list`, in order, appending upload instructions
    /// for exactly the bytes that changed.
    ///
    /// Stops at the first failure, leaving the store self-consistent; the
    /// caller's correct response is to rebuild the affected layer (R-N §2.2's
    /// "one slow frame, never incorrect output").
    pub fn apply(
        &mut self,
        list: &PatchList<P>,
        allocator: &mut SlabAllocator,
        uploads: &mut Vec<UploadRange>,
    ) -> Result<(), PatchError> {
        for Patch { layer, op } in list.patches() {
            match op {
                PatchOp::Insert { key, index, value } => {
                    self.insert(*layer, *key, *index, value.clone(), allocator, uploads)?;
                }
                PatchOp::Update { key, value } => {
                    self.update(*layer, *key, value.clone(), allocator, uploads)?;
                }
                PatchOp::Remove { key } => self.remove(*layer, *key, allocator, uploads)?,
            }
        }
        Ok(())
    }

    fn insert(
        &mut self,
        layer: LayerId,
        key: RecordKey,
        index: u32,
        value: P,
        allocator: &mut SlabAllocator,
        uploads: &mut Vec<UploadRange>,
    ) -> Result<(), PatchError> {
        let primitives = self.layers.entry(layer).or_default();
        if primitives.records.contains_key(&key) {
            return Err(PatchError::DuplicateKey { layer, key });
        }
        let len = primitives.order.len();
        let position = usize::try_from(index).unwrap_or(usize::MAX);
        if position > len {
            return Err(PatchError::IndexOutOfBounds {
                layer,
                index,
                len: u32::try_from(len).unwrap_or(u32::MAX),
            });
        }

        let slot_count = value.slot_count();
        primitives.order.insert(position, key);
        primitives.records.insert(
            key,
            StoredPrimitive {
                value,
                slot_offset: 0,
                slot_count,
            },
        );
        self.reflow(layer, position, allocator, uploads)
    }

    fn update(
        &mut self,
        layer: LayerId,
        key: RecordKey,
        value: P,
        allocator: &mut SlabAllocator,
        uploads: &mut Vec<UploadRange>,
    ) -> Result<(), PatchError> {
        let new_slot_count = value.slot_count();
        {
            let Self { layers, resident } = self;
            let primitives = layers
                .get_mut(&layer)
                .ok_or(PatchError::UnknownKey { layer, key })?;
            let range = primitives.range;
            let stored = primitives
                .records
                .get_mut(&key)
                .ok_or(PatchError::UnknownKey { layer, key })?;
            let keeps_its_slot = new_slot_count == stored.slot_count;
            stored.value = value;
            if keeps_its_slot {
                Self::write_record(resident, range, stored, layer)?;
                if let Some(span) =
                    range.slot_byte_range(stored.slot_offset, new_slot_count, P::SLOT_STRIDE)
                    && span.end > span.start
                {
                    uploads.push(UploadRange {
                        kind: P::KIND,
                        byte_offset: span.start,
                        byte_length: span.end - span.start,
                    });
                }
                return Ok(());
            }
            stored.slot_count = new_slot_count;
        }

        let position = self
            .layers
            .get(&layer)
            .and_then(|primitives| {
                primitives
                    .order
                    .iter()
                    .position(|existing| *existing == key)
            })
            .ok_or(PatchError::UnknownKey { layer, key })?;
        self.reflow(layer, position, allocator, uploads)
    }

    fn remove(
        &mut self,
        layer: LayerId,
        key: RecordKey,
        allocator: &mut SlabAllocator,
        uploads: &mut Vec<UploadRange>,
    ) -> Result<(), PatchError> {
        let primitives = self
            .layers
            .get_mut(&layer)
            .ok_or(PatchError::UnknownKey { layer, key })?;
        if primitives.records.remove(&key).is_none() {
            return Err(PatchError::UnknownKey { layer, key });
        }
        let position = primitives
            .order
            .iter()
            .position(|existing| *existing == key)
            .ok_or(PatchError::UnknownKey { layer, key })?;
        primitives.order.remove(position);
        self.reflow(layer, position, allocator, uploads)
    }

    /// Drop a layer's primitives and return its slab reservation.
    pub fn remove_layer(&mut self, layer: LayerId, allocator: &mut SlabAllocator) {
        if let Some(primitives) = self.layers.remove(&layer) {
            allocator.free(P::KIND, primitives.range);
        }
    }

    /// Recompute slot offsets, resize the layer's reservation to match, and
    /// re-encode every record from `first_dirty` onward.
    ///
    /// The one place placement changes, and therefore the one place that
    /// decides how wide an upload gets — which keeps §5.0's cases auditable in
    /// a single function rather than spread across three call sites that could
    /// drift apart.
    fn reflow(
        &mut self,
        layer: LayerId,
        first_dirty: usize,
        allocator: &mut SlabAllocator,
        uploads: &mut Vec<UploadRange>,
    ) -> Result<(), PatchError> {
        let Self { layers, resident } = self;
        let primitives = layers
            .get_mut(&layer)
            .ok_or(PatchError::UnknownLayer(layer))?;
        let LayerPrimitives {
            order,
            records,
            range,
        } = primitives;

        let overflow = |slots: u64| PatchError::SlabOverflow {
            kind: P::KIND,
            requested_slots: slots,
        };

        let mut total_slots: u64 = 0;
        for key in order.iter() {
            let stored = records
                .get_mut(key)
                .ok_or(PatchError::UnknownKey { layer, key: *key })?;
            stored.slot_offset =
                u32::try_from(total_slots).map_err(|_| overflow(total_slots))?;
            total_slots += stored.slot_count as u64;
        }
        let total_slots_u32 =
            u32::try_from(total_slots).map_err(|_| overflow(total_slots))?;

        let reallocation = allocator
            .reallocate(P::KIND, *range, total_slots_u32)
            .map_err(|error| overflow(error.requested_slots))?;
        *range = reallocation.range();
        let current = *range;

        let required = allocator.arena_slot_capacity(P::KIND) as usize * P::SLOT_STRIDE;
        if resident.len() < required {
            resident.resize(required, 0);
        }

        // A relocation moves every occupied slot's address, so the whole layer
        // is re-encoded regardless of where the edit happened — §5.0's second
        // case, stated there as "bounded by the layer's own slab, never the
        // whole scene."
        let rewrite_from = if reallocation.relocated() {
            0
        } else {
            first_dirty
        };

        let Some(tail) = order.get(rewrite_from..) else {
            return Ok(());
        };
        let Some(first_key) = tail.first() else {
            // Nothing at or after the edit point: a removal from the end. The
            // vacated bytes are outside the layer's occupied count, so they
            // are unreachable and need no upload to become correct.
            return Ok(());
        };
        let start_slot = records
            .get(first_key)
            .map(|stored| stored.slot_offset)
            .ok_or(PatchError::UnknownKey {
                layer,
                key: *first_key,
            })?;

        for key in tail {
            let stored = records
                .get(key)
                .ok_or(PatchError::UnknownKey { layer, key: *key })?;
            Self::write_record(resident, current, stored, layer)?;
        }

        let slot_length = current.count.saturating_sub(start_slot);
        if let Some(span) = current.slot_byte_range(start_slot, slot_length, P::SLOT_STRIDE)
            && span.end > span.start
        {
            uploads.push(UploadRange {
                kind: P::KIND,
                byte_offset: span.start,
                byte_length: span.end - span.start,
            });
        }
        Ok(())
    }

    /// Write one record's encoded bytes into the resident buffer at its
    /// current address.
    fn write_record(
        resident: &mut [u8],
        range: SlabRange,
        stored: &StoredPrimitive<P>,
        layer: LayerId,
    ) -> Result<(), PatchError> {
        if stored.slot_count == 0 {
            return Ok(());
        }
        let span = range
            .slot_byte_range(stored.slot_offset, stored.slot_count, P::SLOT_STRIDE)
            .ok_or(PatchError::IndexOutOfBounds {
                layer,
                index: stored.slot_offset,
                len: range.count,
            })?;
        let overflow = PatchError::SlabOverflow {
            kind: P::KIND,
            requested_slots: range.end() as u64,
        };
        let start = usize::try_from(span.start).map_err(|_| overflow.clone())?;
        let end = usize::try_from(span.end).map_err(|_| overflow.clone())?;
        let destination = resident.get_mut(start..end).ok_or(overflow)?;
        stored
            .value
            .encode(destination)
            .map_err(|error| PatchError::Encode {
                kind: P::KIND,
                error,
            })
    }

    /// The kind's whole resident buffer, exactly as a GPU buffer would hold
    /// it. Phase 1's round-trip gate reads this back.
    pub fn resident_bytes(&self) -> &[u8] {
        &self.resident
    }

    /// A layer's occupied bytes, or `None` if the store has never held
    /// primitives of this kind for that layer.
    pub fn layer_bytes(&self, layer: LayerId) -> Option<&[u8]> {
        let range = self.layers.get(&layer)?.range;
        let span = range.used_byte_range(P::SLOT_STRIDE);
        let start = usize::try_from(span.start).ok()?;
        let end = usize::try_from(span.end).ok()?;
        self.resident.get(start..end)
    }

    /// A layer's reservation.
    pub fn slab(&self, layer: LayerId) -> SlabRange {
        self.layers
            .get(&layer)
            .map(|primitives| primitives.range)
            .unwrap_or(SlabRange::EMPTY)
    }

    /// A layer's record keys, in paint order.
    pub fn keys(&self, layer: LayerId) -> Vec<RecordKey> {
        match self.layers.get(&layer) {
            Some(primitives) => primitives.order.clone(),
            None => Vec::new(),
        }
    }

    /// One record's value.
    pub fn get(&self, layer: LayerId, key: RecordKey) -> Option<&P> {
        Some(&self.layers.get(&layer)?.records.get(&key)?.value)
    }

    /// The byte span one record occupies in the resident buffer — §5.0's
    /// per-primitive address, exposed so a test can assert an upload
    /// instruction is scoped to exactly one primitive's slot.
    pub fn record_byte_range(
        &self,
        layer: LayerId,
        key: RecordKey,
    ) -> Option<std::ops::Range<u64>> {
        let primitives = self.layers.get(&layer)?;
        let stored = primitives.records.get(&key)?;
        primitives
            .range
            .slot_byte_range(stored.slot_offset, stored.slot_count, P::SLOT_STRIDE)
    }

    /// How many primitives of this kind a layer holds.
    pub fn len(&self, layer: LayerId) -> u32 {
        match self.layers.get(&layer) {
            Some(primitives) => u32::try_from(primitives.order.len()).unwrap_or(u32::MAX),
            None => 0,
        }
    }

    /// Whether a layer holds no primitives of this kind.
    pub fn is_empty(&self, layer: LayerId) -> bool {
        self.len(layer) == 0
    }

    /// The draw range for a layer, as CPU-computed first-instance/count — the
    /// same two numbers the legacy renderer computes today, produced through
    /// the new protocol instead of a per-frame scene walk.
    ///
    /// Phases 3 and 4 replace this with GPU-computed indirect draw args
    /// (§5.1–§5.3) consuming the same slab; Phase 1's scope is explicit that
    /// draw ranges stay CPU-computed for now.
    pub fn draw_range(&self, layer: LayerId) -> Option<DrawRange> {
        let range = self.layers.get(&layer)?.range;
        if range.is_empty() {
            return None;
        }
        Some(DrawRange {
            first_instance: range.base,
            instance_count: range.count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patch::primitive::{Glyph, GlyphRun, Quad};
    use crate::scene::slab_range::coalesce_uploads;

    const LAYER: LayerId = LayerId::from_raw(1);

    fn key(raw: u64) -> RecordKey {
        RecordKey::from_raw(raw)
    }

    fn quad(x: f32) -> Quad {
        Quad {
            origin: [x, 0.0],
            ..Quad::ZERO
        }
    }

    fn run(glyph_count: usize) -> GlyphRun {
        GlyphRun {
            color: [1.0, 1.0, 1.0, 1.0],
            glyphs: vec![Glyph::ZERO; glyph_count],
        }
    }

    struct Harness<P: Primitive> {
        store: PrimitiveStore<P>,
        allocator: SlabAllocator,
        uploads: Vec<UploadRange>,
    }

    impl<P: Primitive> Harness<P> {
        fn new() -> Self {
            Self {
                store: PrimitiveStore::new(),
                allocator: SlabAllocator::new(),
                uploads: Vec::new(),
            }
        }

        fn apply(&mut self, list: &PatchList<P>) -> Result<(), PatchError> {
            self.store.apply(list, &mut self.allocator, &mut self.uploads)
        }

        fn take_uploads(&mut self) -> Vec<UploadRange> {
            coalesce_uploads(&mut self.uploads);
            std::mem::take(&mut self.uploads)
        }
    }

    fn seed_quads(harness: &mut Harness<Quad>, count: u32) {
        let mut list = PatchList::new();
        for index in 0..count {
            list.insert(LAYER, key(index as u64 + 1), index, quad(index as f32));
        }
        assert_eq!(harness.apply(&list), Ok(()));
        let _ = harness.take_uploads();
    }

    #[test]
    fn a_fixed_size_value_update_uploads_exactly_one_slot() {
        let mut harness: Harness<Quad> = Harness::new();
        seed_quads(&mut harness, 1000);

        let mut edit = PatchList::new();
        edit.update(LAYER, key(500), quad(-1.0));
        assert_eq!(harness.apply(&edit), Ok(()));
        let uploads = harness.take_uploads();
        assert_eq!(uploads.len(), 1);
        assert_eq!(uploads[0].byte_length, Quad::SLOT_STRIDE as u64);
        assert_eq!(
            Some(uploads[0].byte_offset..uploads[0].byte_end()),
            harness.store.record_byte_range(LAYER, key(500))
        );
    }

    #[test]
    fn a_variable_size_value_update_that_keeps_its_slot_count_stays_o1() {
        let mut harness: Harness<GlyphRun> = Harness::new();
        let mut seed = PatchList::new();
        seed.insert(LAYER, key(1), 0, run(4))
            .insert(LAYER, key(2), 1, run(6))
            .insert(LAYER, key(3), 2, run(5));
        assert_eq!(harness.apply(&seed), Ok(()));
        let _ = harness.take_uploads();

        let mut recoloured = run(6);
        recoloured.color = [0.0, 1.0, 0.0, 1.0];
        let mut edit = PatchList::new();
        edit.update(LAYER, key(2), recoloured);
        assert_eq!(harness.apply(&edit), Ok(()));
        let uploads = harness.take_uploads();
        assert_eq!(uploads.len(), 1);
        assert_eq!(uploads[0].byte_length, 6 * GlyphRun::SLOT_STRIDE as u64);
    }

    #[test]
    fn a_variable_size_update_that_changes_slot_count_shifts_its_successors() {
        let mut harness: Harness<GlyphRun> = Harness::new();
        let mut seed = PatchList::new();
        seed.insert(LAYER, key(1), 0, run(4))
            .insert(LAYER, key(2), 1, run(6))
            .insert(LAYER, key(3), 2, run(5));
        assert_eq!(harness.apply(&seed), Ok(()));
        let _ = harness.take_uploads();

        let mut edit = PatchList::new();
        edit.update(LAYER, key(2), run(9));
        assert_eq!(harness.apply(&edit), Ok(()));
        let uploads = harness.take_uploads();
        assert_eq!(uploads.len(), 1);
        // The grown run plus the successor that shifted: 9 + 5 slots.
        assert_eq!(uploads[0].byte_length, 14 * GlyphRun::SLOT_STRIDE as u64);
        assert_eq!(harness.store.slab(LAYER).count, 18);
    }

    #[test]
    fn an_append_does_not_rewrite_earlier_primitives() {
        let mut harness: Harness<Quad> = Harness::new();
        seed_quads(&mut harness, 10);

        let mut append = PatchList::new();
        append.append(LAYER, key(11), 10, quad(10.0));
        assert_eq!(harness.apply(&append), Ok(()));
        let uploads = harness.take_uploads();
        assert_eq!(uploads.len(), 1);
        assert_eq!(uploads[0].byte_length, Quad::SLOT_STRIDE as u64);
        assert_eq!(uploads[0].byte_offset, 10 * Quad::SLOT_STRIDE as u64);
    }

    #[test]
    fn removing_the_last_primitive_uploads_nothing() {
        let mut harness: Harness<Quad> = Harness::new();
        seed_quads(&mut harness, 10);

        let mut remove = PatchList::new();
        remove.remove(LAYER, key(10));
        assert_eq!(harness.apply(&remove), Ok(()));
        assert!(
            harness.take_uploads().is_empty(),
            "shrinking a layer leaves no reachable stale bytes to rewrite"
        );
        assert_eq!(harness.store.len(LAYER), 9);
    }

    #[test]
    fn removing_an_interior_primitive_rewrites_only_its_successors() {
        let mut harness: Harness<Quad> = Harness::new();
        seed_quads(&mut harness, 10);

        let mut remove = PatchList::new();
        remove.remove(LAYER, key(4));
        assert_eq!(harness.apply(&remove), Ok(()));
        let uploads = harness.take_uploads();
        assert_eq!(uploads.len(), 1);
        assert_eq!(uploads[0].byte_offset, 3 * Quad::SLOT_STRIDE as u64);
        assert_eq!(uploads[0].byte_length, 6 * Quad::SLOT_STRIDE as u64);
    }

    #[test]
    fn crossing_a_size_class_relocates_and_rewrites_the_whole_layer() {
        let mut harness: Harness<Quad> = Harness::new();
        seed_quads(&mut harness, 64);
        let base_before = harness.store.slab(LAYER).base;
        // Hold a second layer immediately after this one so the grown range
        // cannot land back on the same base, making the relocation observable.
        let mut neighbour = PatchList::new();
        neighbour.insert(LayerId::from_raw(2), key(9001), 0, quad(0.0));
        assert_eq!(harness.apply(&neighbour), Ok(()));
        let _ = harness.take_uploads();

        let mut grow = PatchList::new();
        grow.append(LAYER, key(65), 64, quad(64.0));
        assert_eq!(harness.apply(&grow), Ok(()));
        let uploads = harness.take_uploads();
        assert_ne!(harness.store.slab(LAYER).base, base_before);
        assert_eq!(uploads.len(), 1);
        assert_eq!(uploads[0].byte_length, 65 * Quad::SLOT_STRIDE as u64);
    }

    #[test]
    fn an_empty_patch_list_produces_no_uploads() {
        let mut harness: Harness<Quad> = Harness::new();
        assert_eq!(harness.apply(&PatchList::new()), Ok(()));
        assert!(harness.take_uploads().is_empty());
    }

    #[test]
    fn an_empty_glyph_run_occupies_no_slots_and_uploads_nothing() {
        let mut harness: Harness<GlyphRun> = Harness::new();
        let mut seed = PatchList::new();
        seed.insert(LAYER, key(1), 0, GlyphRun::empty([1.0, 1.0, 1.0, 1.0]));
        assert_eq!(harness.apply(&seed), Ok(()));
        assert!(harness.take_uploads().is_empty());
        assert_eq!(harness.store.slab(LAYER).count, 0);
        assert_eq!(harness.store.len(LAYER), 1);
    }

    #[test]
    fn draw_ranges_are_cpu_computed_first_instance_and_count() {
        let mut harness: Harness<Quad> = Harness::new();
        assert_eq!(harness.store.draw_range(LAYER), None);
        seed_quads(&mut harness, 3);
        assert_eq!(
            harness.store.draw_range(LAYER),
            Some(DrawRange {
                first_instance: harness.store.slab(LAYER).base,
                instance_count: 3,
            })
        );
    }

    #[test]
    fn dropping_a_layer_returns_its_reservation() {
        let mut harness: Harness<Quad> = Harness::new();
        seed_quads(&mut harness, 1);
        harness.store.remove_layer(LAYER, &mut harness.allocator);
        assert_eq!(harness.allocator.arena_slot_capacity(Quad::KIND), 0);
        assert!(harness.store.layer_bytes(LAYER).is_none());
    }

    #[test]
    fn duplicate_and_out_of_range_inserts_are_rejected() {
        let mut harness: Harness<Quad> = Harness::new();
        seed_quads(&mut harness, 2);

        let mut duplicate = PatchList::new();
        duplicate.insert(LAYER, key(1), 0, quad(0.0));
        assert_eq!(
            harness.apply(&duplicate),
            Err(PatchError::DuplicateKey {
                layer: LAYER,
                key: key(1)
            })
        );

        let mut past_end = PatchList::new();
        past_end.insert(LAYER, key(99), 5, quad(0.0));
        assert_eq!(
            harness.apply(&past_end),
            Err(PatchError::IndexOutOfBounds {
                layer: LAYER,
                index: 5,
                len: 2
            })
        );
    }

    #[test]
    fn updating_an_unknown_record_is_rejected_rather_than_inserted() {
        let mut harness: Harness<Quad> = Harness::new();
        let mut edit = PatchList::new();
        edit.update(LAYER, key(1), quad(0.0));
        assert_eq!(
            harness.apply(&edit),
            Err(PatchError::UnknownKey {
                layer: LAYER,
                key: key(1)
            })
        );
        assert_eq!(harness.store.len(LAYER), 0);
    }
}
