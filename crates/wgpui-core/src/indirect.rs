//! Indirect draw-argument generation: the fixed (layer, kind) slot table, the
//! argument record a `draw_indirect` reads, and the CPU reference the compute
//! pass is checked against.
//! See docs/gpu-native-architecture.md §5.3, §8 Phase 4.
//!
//! Not in §3's file map, for the same reason [`crate::ordering`] and
//! [`crate::occlusion`] are not: §3.1 gives `shaders/indirect_args.wgsl` a home
//! and gives the computation it implements none, because in the legacy backend
//! that computation *is* `quads_first_instance` and its siblings sitting inline
//! in `renderer.rs`'s draw loop. Phase 4 needs the same three things Phase 3
//! needed in one place — the definition, the CPU reference, and the GPU-side
//! encoding — and putting them in `scene.rs` would make that file the thing §3
//! exists to prevent.
//!
//! # What §5.3 actually asks for
//!
//! > the CPU issues a **fixed** sequence of `draw_indirect`/`multi_draw_indirect`
//! > calls — one per (layer, kind) slot that *could* be populated — every frame,
//! > regardless of how many are actually zero.
//!
//! So the unit of work is a [`DrawSlot`]: one layer's reservation in one kind's
//! arena. [`crate::scene::Scene::draw_slots`] enumerates them, and that
//! enumeration is the *only* per-frame CPU work proportional to anything — it is
//! `O(layers × kinds)` and reads nothing but each layer's `SlabRange`. It never
//! looks at a primitive.
//!
//! # The indirection buffer, and why it is arena-shaped
//!
//! Per-instance data is storage-buffer vertex pulling (§1): the shader indexes
//! its arena with `@builtin(instance_index)`. Culling removes an arbitrary
//! subset and ordering permutes what is left, and neither is expressible as a
//! contiguous `first_instance..first_instance + count` range over the arena. So
//! the pass writes an **indirection buffer**: `visible[i]` holds the arena slot
//! the *i*-th drawn instance reads.
//!
//! That buffer mirrors the arena exactly — slot `(layer, kind)` owns
//! `[base, base + count)` in it, the same range the layer's `SlabRange` owns in
//! the arena. Three consequences, all of them the point:
//!
//! 1. **Every slot's run base is CPU-known and stable**, because it is the
//!    `SlabRange` the CPU already has. Nothing is read back to learn where a
//!    slot's instances start.
//! 2. Compaction is per-slot and local: survivors are packed from `base`
//!    upward, so a slot's write never touches another slot's range and the
//!    dispatch needs no global coordination.
//! 3. A slot's `instance_count` is bounded by its own reservation, so the
//!    indirection buffer is sized once, alongside the arena, and grows with it.
//!
//! # `firstInstance`, and the gotcha this crate already documented
//!
//! `README.md`'s "Custom Device Gotcha" records a real, already-hit failure:
//! many backends silently drop an indirect draw whose `firstInstance` is
//! nonzero unless `INDIRECT_FIRST_INSTANCE` is enabled. That is why
//! [`FirstInstance`] is an explicit choice rather than a constant:
//!
//! - [`FirstInstance::Zero`] — every argument record carries `first_instance: 0`
//!   and the shader is told its slot's base out of band (a per-slot uniform the
//!   CPU sets with a dynamic offset it already knows). This is the *default*
//!   path. It needs no device feature at all, it is what WebGPU permits, and it
//!   cannot be hit by the documented driver bug because it never produces the
//!   input that triggers it.
//! - [`FirstInstance::SlotBase`] — the base rides in the argument record, which
//!   is the only way a `multi_draw_indirect` can address per-entry ranges, since
//!   one call covers many entries and no bind group can change between them.
//!   Chosen only when `INDIRECT_FIRST_INSTANCE` was actually negotiated.
//!
//! The shader is the same either way: it computes `slot_base + instance_index`,
//! where `slot_base` is the uniform. Under [`FirstInstance::SlotBase`] the
//! uniform is zero and `first_instance` supplies the base; under
//! [`FirstInstance::Zero`] it is the reverse. One shader, two encodings.

use crate::patch::primitive::PrimitiveKind;
use crate::scene::layer::LayerId;

/// Bytes one [`DrawIndirectArgs`] record occupies. Fixed by the WebGPU
/// specification's `draw_indirect` layout, not by this crate.
pub const DRAW_INDIRECT_ARGS_STRIDE: usize = 16;

/// Bytes one slot's shader-side descriptor occupies: `[base, count, 0, 0]`,
/// padded to a `vec4<u32>` so an array of them is `std430`-addressable.
pub const DRAW_SLOT_STRIDE: usize = 16;

/// Vertices one quad-shaped instance draws: a four-vertex triangle strip, the
/// same topology every instanced pipeline in the legacy renderer uses
/// (`pass.draw(0..4, ..)`).
pub const QUAD_VERTEX_COUNT: u32 = 4;

/// Where a draw's per-slot base index is carried.
///
/// See this module's doc: the choice exists because of `README.md`'s "Custom
/// Device Gotcha", not for generality's sake.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FirstInstance {
    /// `first_instance` is always `0`; the base reaches the shader as a uniform.
    /// Requires no device feature and is the default.
    Zero,
    /// `first_instance` holds the slot's arena base. Requires
    /// `INDIRECT_FIRST_INSTANCE`, and is what a `multi_draw_indirect` needs.
    SlotBase,
}

impl FirstInstance {
    /// The shader parameter encoding this choice.
    pub const fn as_u32(self) -> u32 {
        match self {
            FirstInstance::Zero => 0,
            FirstInstance::SlotBase => 1,
        }
    }
}

/// One `draw_indirect` argument record, in WebGPU's own field order.
///
/// Deliberately mirrors `wgpu::util::DrawIndirectArgs` rather than wrapping it:
/// `wgpui-core` has no `wgpu` dependency (§3.1) and this layout is fixed by the
/// specification, so restating it here costs nothing and keeps the crate
/// device-free. `wgpui-wgpu` asserts the two agree.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct DrawIndirectArgs {
    /// Vertices per instance.
    pub vertex_count: u32,
    /// Instances to draw. **The one field the GPU decides.**
    pub instance_count: u32,
    /// First vertex. Always zero here.
    pub first_vertex: u32,
    /// First instance — see [`FirstInstance`].
    pub first_instance: u32,
}

impl DrawIndirectArgs {
    /// The four words, in the order a `draw_indirect` buffer holds them.
    pub const fn to_array(self) -> [u32; 4] {
        [
            self.vertex_count,
            self.instance_count,
            self.first_vertex,
            self.first_instance,
        ]
    }

    /// Rebuild a record from four words read back off the device.
    pub const fn from_array(words: [u32; 4]) -> DrawIndirectArgs {
        DrawIndirectArgs {
            vertex_count: words[0],
            instance_count: words[1],
            first_vertex: words[2],
            first_instance: words[3],
        }
    }

    /// Whether this record expands to no work at all.
    pub const fn is_empty(self) -> bool {
        self.instance_count == 0 || self.vertex_count == 0
    }
}

/// One (layer, kind) pair the CPU issues a draw for every frame, populated or
/// not.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DrawSlot {
    /// The layer whose reservation this slot draws.
    pub layer: LayerId,
    /// Which arena the reservation is in.
    pub kind: PrimitiveKind,
    /// First slot of the reservation in that arena — the slot's run base in the
    /// indirection buffer, and the number the CPU never has to read back.
    pub base: u32,
    /// Slots reserved, i.e. the upper bound on this slot's `instance_count`.
    pub count: u32,
}

/// Every slot of one kind, in the order they must be drawn.
///
/// Grouped by kind and ascending by layer within a kind, so a
/// `multi_draw_indirect` over one kind covers a contiguous run of argument
/// records in painter order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SlotTable {
    slots: Vec<DrawSlot>,
    /// Where each kind's run starts in `slots`, plus a terminating entry.
    kind_starts: [usize; PrimitiveKind::COUNT + 1],
}

impl SlotTable {
    /// Build a table from slots already grouped by kind in declaration order.
    ///
    /// Returns `None` if `slots` is not so grouped — the invariant every
    /// consumer relies on, checked once here rather than assumed at each use.
    pub fn from_grouped(slots: Vec<DrawSlot>) -> Option<SlotTable> {
        let mut kind_starts = [slots.len(); PrimitiveKind::COUNT + 1];
        kind_starts[0] = 0;
        let mut current = 0usize;
        for (index, slot) in slots.iter().enumerate() {
            let kind = slot.kind.index();
            if kind < current {
                return None;
            }
            while current < kind {
                current += 1;
                *kind_starts.get_mut(current)? = index;
            }
        }
        while current < PrimitiveKind::COUNT {
            current += 1;
            *kind_starts.get_mut(current)? = slots.len();
        }
        Some(SlotTable { slots, kind_starts })
    }

    /// Every slot, in draw order.
    pub fn slots(&self) -> &[DrawSlot] {
        &self.slots
    }

    /// One kind's contiguous run of slots.
    pub fn kind_range(&self, kind: PrimitiveKind) -> std::ops::Range<usize> {
        let index = kind.index();
        let start = self.kind_starts.get(index).copied().unwrap_or(0);
        let end = self.kind_starts.get(index + 1).copied().unwrap_or(start);
        start..end.max(start)
    }

    /// One kind's slots.
    pub fn kind_slots(&self, kind: PrimitiveKind) -> &[DrawSlot] {
        self.slots.get(self.kind_range(kind)).unwrap_or(&[])
    }

    /// How many slots the table holds — the length of the fixed draw sequence.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether the table names no slots at all.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

/// Encode one kind's slots for `shaders/indirect_args.wgsl`.
///
/// Byte-oriented for the reason `patch/primitive.rs` gives: it keeps
/// `wgpui-core` dependency-free and makes the GPU layout an explicit decision.
pub fn encode_slots(slots: &[DrawSlot], destination: &mut Vec<u8>) {
    destination.clear();
    destination.reserve(slots.len() * DRAW_SLOT_STRIDE);
    for slot in slots {
        for value in [slot.base, slot.count, 0, 0] {
            destination.extend_from_slice(&value.to_le_bytes());
        }
    }
}

/// Encode argument records for comparison against a readback.
pub fn encode_args(args: &[DrawIndirectArgs], destination: &mut Vec<u8>) {
    destination.clear();
    destination.reserve(args.len() * DRAW_INDIRECT_ARGS_STRIDE);
    for record in args {
        for value in record.to_array() {
            destination.extend_from_slice(&value.to_le_bytes());
        }
    }
}

/// Decode argument records read back off the device.
pub fn decode_args(words: &[u32]) -> Vec<DrawIndirectArgs> {
    words
        .chunks_exact(4)
        .map(|chunk| DrawIndirectArgs::from_array([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// What one indirect-arg generation produces, as plain data.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IndirectArgs {
    /// The indirection buffer, arena-shaped: `visible[i]` is the arena slot the
    /// *i*-th drawn instance reads. Entries outside a slot's live prefix are
    /// left at [`UNUSED_INSTANCE`].
    pub visible: Vec<u32>,
    /// One record per slot, in [`SlotTable`] order.
    pub args: Vec<DrawIndirectArgs>,
    /// The populated slots' records, order-preserved and packed — what a
    /// `multi_draw_indirect_count` reads.
    pub packed: Vec<DrawIndirectArgs>,
}

/// The value a never-written indirection entry holds.
///
/// Not zero: zero is a legitimate arena slot, and a shader that read past a
/// slot's `instance_count` because of an argument bug would then silently draw
/// primitive 0 many times over instead of producing something obviously wrong.
pub const UNUSED_INSTANCE: u32 = u32::MAX;

/// The CPU reference `shaders/indirect_args.wgsl` transcribes.
///
/// For each slot, walk that slot's primitives **in draw order** and pack the
/// ones the occlusion pass kept into the front of the slot's indirection range.
/// Order-preserving compaction, not an unordered append: painter order within a
/// layer is exactly what the ordering pass computed and exactly what a
/// scrambled compaction would destroy.
///
/// `draw_order` and `culled` are both arena-shaped, matching the buffers
/// Phase 3's passes write per layer scattered into their layer's own range:
/// `draw_order[base + position]` is a *layer-local* index in `[0, count)`, and
/// `culled[base + local]` is `1` when the occlusion pass dropped that primitive.
/// Keeping the local convention is what lets an `OrderingOutput` be copied into
/// place without a rewrite pass in between.
pub fn indirect_args(
    slots: &[DrawSlot],
    draw_order: &[u32],
    culled: &[u32],
    arena_slots: usize,
    vertex_count: u32,
    first_instance: FirstInstance,
) -> IndirectArgs {
    let mut visible = vec![UNUSED_INSTANCE; arena_slots];
    let mut args = Vec::with_capacity(slots.len());
    for slot in slots {
        let base = slot.base as usize;
        let mut written = 0u32;
        for position in 0..slot.count as usize {
            let Some(local) = draw_order.get(base + position).copied() else {
                continue;
            };
            if local >= slot.count {
                continue;
            }
            let arena_index = base + local as usize;
            if culled.get(arena_index).copied().unwrap_or(0) != 0 {
                continue;
            }
            let Some(entry) = visible.get_mut(base + written as usize) else {
                continue;
            };
            *entry = u32::try_from(arena_index).unwrap_or(UNUSED_INSTANCE);
            written += 1;
        }
        args.push(DrawIndirectArgs {
            vertex_count,
            instance_count: written,
            first_vertex: 0,
            first_instance: match first_instance {
                FirstInstance::Zero => 0,
                FirstInstance::SlotBase => slot.base,
            },
        });
    }
    let packed = args
        .iter()
        .copied()
        .filter(|record| !record.is_empty())
        .collect();
    IndirectArgs {
        visible,
        args,
        packed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::layer::LayerId;

    fn slot(layer: u64, kind: PrimitiveKind, base: u32, count: u32) -> DrawSlot {
        DrawSlot {
            layer: LayerId::from_raw(layer),
            kind,
            base,
            count,
        }
    }

    /// Identity draw order over an arena in which one slot sits at `base`.
    fn identity_order(arena: usize, slots: &[DrawSlot]) -> Vec<u32> {
        let mut order = vec![0u32; arena];
        for slot in slots {
            for local in 0..slot.count {
                if let Some(entry) = order.get_mut(slot.base as usize + local as usize) {
                    *entry = local;
                }
            }
        }
        order
    }

    #[test]
    fn an_uncalled_slot_still_produces_a_record_with_zero_instances() {
        let slots = [slot(1, PrimitiveKind::Quad, 0, 3)];
        let order = identity_order(3, &slots);
        let result = indirect_args(&slots, &order, &[1, 1, 1], 3, 4, FirstInstance::Zero);
        assert_eq!(
            result.args,
            vec![DrawIndirectArgs {
                vertex_count: 4,
                instance_count: 0,
                first_vertex: 0,
                first_instance: 0,
            }],
            "§5.3: the sequence is fixed; a fully culled slot is a zero-instance \
             record, not an omitted one"
        );
        assert!(result.packed.is_empty());
        assert!(result.visible.iter().all(|entry| *entry == UNUSED_INSTANCE));
    }

    #[test]
    fn compaction_preserves_draw_order_rather_than_arena_order() {
        // Draw order reverses the layer, and the middle primitive is culled.
        let slots = [slot(1, PrimitiveKind::Quad, 0, 4)];
        let order = vec![3, 2, 1, 0];
        let culled = vec![0, 0, 1, 0];
        let result = indirect_args(&slots, &order, &culled, 4, 4, FirstInstance::Zero);
        assert_eq!(
            result.visible,
            vec![3, 1, 0, UNUSED_INSTANCE],
            "survivors must keep the painter order the ordering pass computed"
        );
        assert_eq!(result.args[0].instance_count, 3);
    }

    #[test]
    fn each_slot_packs_into_its_own_arena_range_and_no_other() {
        let slots = [
            slot(1, PrimitiveKind::Quad, 0, 4),
            slot(2, PrimitiveKind::Quad, 4, 4),
        ];
        let order = identity_order(8, &slots);
        // Cull everything in the first layer; keep everything in the second.
        let culled = vec![1, 1, 1, 1, 0, 0, 0, 0];
        let result = indirect_args(&slots, &order, &culled, 8, 4, FirstInstance::Zero);
        assert_eq!(
            result.visible,
            vec![
                UNUSED_INSTANCE,
                UNUSED_INSTANCE,
                UNUSED_INSTANCE,
                UNUSED_INSTANCE,
                4,
                5,
                6,
                7
            ],
            "a fully culled layer must not shift its neighbour's instances down"
        );
        assert_eq!(result.args[0].instance_count, 0);
        assert_eq!(result.args[1].instance_count, 4);
    }

    #[test]
    fn the_first_instance_choice_is_the_only_difference_between_the_two_encodings() {
        let slots = [
            slot(1, PrimitiveKind::Quad, 0, 2),
            slot(2, PrimitiveKind::Quad, 2, 2),
        ];
        let order = identity_order(4, &slots);
        let culled = vec![0; 4];
        let zero = indirect_args(&slots, &order, &culled, 4, 4, FirstInstance::Zero);
        let based = indirect_args(&slots, &order, &culled, 4, 4, FirstInstance::SlotBase);
        assert_eq!(zero.visible, based.visible);
        assert_eq!(
            zero.args.iter().map(|a| a.first_instance).collect::<Vec<_>>(),
            vec![0, 0],
            "README's Custom Device Gotcha: the default path never emits a \
             nonzero firstInstance"
        );
        assert_eq!(
            based.args.iter().map(|a| a.first_instance).collect::<Vec<_>>(),
            vec![0, 2]
        );
        assert_eq!(
            zero.args.iter().map(|a| a.instance_count).collect::<Vec<_>>(),
            based
                .args
                .iter()
                .map(|a| a.instance_count)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn packing_drops_empty_records_without_reordering_the_rest() {
        let slots = [
            slot(1, PrimitiveKind::Quad, 0, 2),
            slot(2, PrimitiveKind::Quad, 2, 2),
            slot(3, PrimitiveKind::Quad, 4, 2),
        ];
        let order = identity_order(6, &slots);
        let culled = vec![0, 0, 1, 1, 0, 0];
        let result = indirect_args(&slots, &order, &culled, 6, 4, FirstInstance::SlotBase);
        assert_eq!(
            result.packed.len(),
            2,
            "the middle slot is empty and must not occupy a packed entry"
        );
        assert_eq!(
            result
                .packed
                .iter()
                .map(|record| record.first_instance)
                .collect::<Vec<_>>(),
            vec![0, 4],
            "packing must not reorder slots — painter order across layers is \
             layer order"
        );
    }

    #[test]
    fn a_slot_table_groups_kinds_into_contiguous_runs() {
        let table = SlotTable::from_grouped(vec![
            slot(1, PrimitiveKind::Quad, 0, 2),
            slot(2, PrimitiveKind::Quad, 2, 2),
            slot(1, PrimitiveKind::GlyphRun, 0, 5),
        ])
        .expect("slots are grouped by kind");
        assert_eq!(table.kind_slots(PrimitiveKind::Quad).len(), 2);
        assert_eq!(table.kind_slots(PrimitiveKind::GlyphRun).len(), 1);
        assert_eq!(table.len(), 3);
    }

    #[test]
    fn an_ungrouped_slot_list_is_rejected_rather_than_silently_mis_ranged() {
        assert!(
            SlotTable::from_grouped(vec![
                slot(1, PrimitiveKind::GlyphRun, 0, 1),
                slot(1, PrimitiveKind::Quad, 0, 1),
            ])
            .is_none()
        );
    }

    #[test]
    fn an_empty_table_reports_empty_ranges_for_every_kind() {
        let table = SlotTable::from_grouped(Vec::new()).expect("an empty list is grouped");
        assert!(table.is_empty());
        for kind in PrimitiveKind::ALL {
            assert!(table.kind_slots(kind).is_empty());
        }
    }

    #[test]
    fn args_round_trip_through_their_wire_encoding() {
        let args = [
            DrawIndirectArgs {
                vertex_count: 4,
                instance_count: 17,
                first_vertex: 0,
                first_instance: 64,
            },
            DrawIndirectArgs::default(),
        ];
        let mut bytes = Vec::new();
        encode_args(&args, &mut bytes);
        assert_eq!(bytes.len(), 2 * DRAW_INDIRECT_ARGS_STRIDE);
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        assert_eq!(decode_args(&words), args.to_vec());
    }

    #[test]
    fn slots_encode_as_one_padded_vec4_each() {
        let mut bytes = Vec::new();
        encode_slots(&[slot(1, PrimitiveKind::Quad, 7, 9)], &mut bytes);
        assert_eq!(bytes.len(), DRAW_SLOT_STRIDE);
        assert_eq!(&bytes[0..4], &7u32.to_le_bytes());
        assert_eq!(&bytes[4..8], &9u32.to_le_bytes());
    }

    #[test]
    fn an_out_of_range_permutation_entry_is_dropped_rather_than_read_past() {
        // The ordering pass pads its sort network beyond the primitive count and
        // the padding reads back as `u32::MAX`; a slot whose range is copied in
        // wholesale can therefore carry one. It must not become an instance.
        let slots = [slot(1, PrimitiveKind::Quad, 0, 3)];
        let order = vec![0, u32::MAX, 2];
        let result = indirect_args(&slots, &order, &[0, 0, 0], 3, 4, FirstInstance::Zero);
        assert_eq!(result.args[0].instance_count, 2);
        assert_eq!(result.visible, vec![0, 2, UNUSED_INSTANCE]);
    }
}
