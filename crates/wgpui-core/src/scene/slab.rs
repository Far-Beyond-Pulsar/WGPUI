//! Size-class slot allocator backing every layer's per-kind slab.
//! See docs/gpu-native-architecture.md §3.1, §5.0, and R-N §4.2.
//!
//! No device, no queue — the same commitment the legacy backend's `slab.rs`
//! makes and for the same reason: placement is pure arithmetic, so it is
//! exhaustively testable headlessly, and §3.1 requires this crate to stay that
//! way.
//!
//! # Why a buddy allocator rather than a port of the legacy free lists
//!
//! The legacy allocator (1,538 lines) keeps power-of-two size classes but
//! places blocks with a bump pointer plus per-class free lists, and therefore
//! needs explicit adjacency scanning to grow a block in place, an advisory
//! compaction pass to recover fragmentation, and a reserved-block index to
//! support both. Because every class here is already a power-of-two multiple
//! of [`MIN_CLASS`], keeping every block *aligned to its own size* makes the
//! buddy of a block a single XOR away, so coalescing on free is exact,
//! immediate, and about fifteen lines. That collapses three mechanisms
//! (adjacency scan, compaction plan, reserved index) into one and removes the
//! failure mode the legacy design discloses as its own residual risk (R-N
//! §4.3: fragmentation reclaimed only when someone remembers to ask).
//!
//! The cost, disclosed rather than glossed: alignment means a fresh arena can
//! leave a gap ahead of a large allocation. Those gaps are not lost — they are
//! split into aligned power-of-two blocks and pushed onto the free lists
//! immediately, so the next smaller request consumes them.

use crate::patch::primitive::PrimitiveKind;
use crate::scene::slab_range::{MIN_CLASS, SlabRange, size_class};
use std::collections::{BTreeMap, BTreeSet};

/// A slab request could not be addressed inside a single `u32`-indexed arena.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SlabOverflow {
    /// Kind whose arena could not fit the request.
    pub kind: PrimitiveKind,
    /// Slots the caller asked for.
    pub requested_slots: u64,
}

/// What a reallocation did to a range, which decides how much has to be
/// re-uploaded.
///
/// §5.0 states these as two of its three cases: a value update that keeps its
/// slot costs O(1) bytes, while "insert/remove that forces the allocator to
/// relocate a primitive ... costs a wider write for the primitives actually
/// moved — bounded by the layer's own slab, never the whole scene."
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Reallocation {
    /// The reservation's size class was unchanged, so the base did not move
    /// and nothing outside the edited slots needs re-uploading.
    InPlace(SlabRange),
    /// The size class changed, so the range moved. Every occupied slot has a
    /// new address and the layer's whole occupied span must be rewritten.
    Relocated {
        /// Where the range used to live.
        previous: SlabRange,
        /// Where it lives now.
        current: SlabRange,
    },
}

impl Reallocation {
    /// The range as it now stands.
    pub const fn range(self) -> SlabRange {
        match self {
            Reallocation::InPlace(range) => range,
            Reallocation::Relocated { current, .. } => current,
        }
    }

    /// Whether the range's base moved.
    pub const fn relocated(self) -> bool {
        matches!(self, Reallocation::Relocated { .. })
    }
}

/// One primitive kind's arena: a bump frontier plus per-class free lists of
/// self-aligned blocks.
///
/// Everything inside is counted in *units* of [`MIN_CLASS`] slots, which is
/// what makes every block a power of two and every buddy an XOR away. Slot
/// counts appear only at this type's boundary.
#[derive(Debug, Default)]
struct Arena {
    /// Frontier, in units. Slots beyond this have never been reserved.
    frontier_units: u32,
    /// Free blocks by class (in units), each set holding block bases in units.
    free_by_class: BTreeMap<u32, BTreeSet<u32>>,
}

impl Arena {
    fn allocate_units(&mut self, class_units: u32) -> Option<u32> {
        if let Some(base) = self.take_free(class_units) {
            return Some(base);
        }
        self.align_frontier_to(class_units)?;
        let base = self.frontier_units;
        self.frontier_units = self.frontier_units.checked_add(class_units)?;
        Some(base)
    }

    /// Take a free block of exactly `class_units`, splitting a larger one if
    /// that is all that is available. Splitting pushes the unused halves back
    /// as free blocks, so falling up never wastes the remainder.
    fn take_free(&mut self, class_units: u32) -> Option<u32> {
        let found = self
            .free_by_class
            .range(class_units..)
            .find(|(_, bases)| !bases.is_empty())
            .map(|(class, _)| *class)?;

        let base = {
            let bases = self.free_by_class.get_mut(&found)?;
            let base = *bases.iter().next()?;
            bases.remove(&base);
            base
        };

        let mut current_class = found;
        while current_class > class_units {
            current_class /= 2;
            self.push_free(base + current_class, current_class);
        }
        Some(base)
    }

    /// Advance the frontier to a multiple of `class_units`, banking the skipped
    /// span as aligned free blocks rather than losing it.
    fn align_frontier_to(&mut self, class_units: u32) -> Option<()> {
        while !self.frontier_units.is_multiple_of(class_units) {
            // The frontier's own alignment is its lowest set bit; a block of
            // that size starting there is self-aligned, which is what keeps
            // buddy coalescing exact.
            let chunk = self.frontier_units & self.frontier_units.wrapping_neg();
            self.push_free(self.frontier_units, chunk);
            self.frontier_units = self.frontier_units.checked_add(chunk)?;
        }
        Some(())
    }

    fn push_free(&mut self, base_units: u32, class_units: u32) {
        self.free_by_class
            .entry(class_units)
            .or_default()
            .insert(base_units);
    }

    /// Return a block, coalescing with its buddy for as long as the buddy is
    /// also free and of the same class.
    fn free_units(&mut self, base_units: u32, class_units: u32) {
        let mut base = base_units;
        let mut class = class_units;
        loop {
            let buddy = base ^ class;
            let buddy_is_free = self
                .free_by_class
                .get(&class)
                .is_some_and(|bases| bases.contains(&buddy));
            if !buddy_is_free {
                break;
            }
            if let Some(bases) = self.free_by_class.get_mut(&class) {
                bases.remove(&buddy);
            }
            base = base.min(buddy);
            match class.checked_mul(2) {
                Some(doubled) => class = doubled,
                None => break,
            }
        }

        // A merged block sitting exactly at the frontier is not free space,
        // it is space that was never used: retracting the frontier keeps the
        // arena's reported capacity honest for a scene that shrinks.
        if base + class == self.frontier_units {
            self.frontier_units = base;
            return;
        }
        self.push_free(base, class);
    }

    fn free_units_total(&self) -> u64 {
        self.free_by_class
            .iter()
            .map(|(class, bases)| *class as u64 * bases.len() as u64)
            .sum()
    }
}

/// Places every layer's per-kind slab inside one arena per primitive kind.
///
/// The allocator knows nothing about layers, patches, or bytes — a caller
/// hands it slot counts and receives [`SlabRange`]s. Ownership of which layer
/// holds which range lives in [`crate::scene::layer::LayerTable`], so this
/// type stays a pure placement engine.
#[derive(Debug, Default)]
pub struct SlabAllocator {
    arenas: [Arena; PrimitiveKind::COUNT],
}

impl SlabAllocator {
    /// An allocator with every arena empty.
    pub fn new() -> Self {
        Self::default()
    }

    /// Reserve `slot_count` slots of `kind`.
    ///
    /// A zero count is churn-free: it reserves nothing and returns
    /// [`SlabRange::EMPTY`], so a layer that holds no primitives of a kind
    /// costs no arena space and no bookkeeping.
    pub fn allocate(
        &mut self,
        kind: PrimitiveKind,
        slot_count: u32,
    ) -> Result<SlabRange, SlabOverflow> {
        let overflow = || SlabOverflow {
            kind,
            requested_slots: slot_count as u64,
        };
        let capacity = size_class(slot_count).ok_or_else(overflow)?;
        if capacity == 0 {
            return Ok(SlabRange::EMPTY);
        }
        let class_units = capacity / MIN_CLASS;
        let arena = self.arena_mut(kind);
        let base_units = arena.allocate_units(class_units).ok_or_else(overflow)?;
        let base = base_units.checked_mul(MIN_CLASS).ok_or_else(overflow)?;
        base.checked_add(capacity).ok_or_else(overflow)?;
        Ok(SlabRange {
            base,
            capacity,
            count: slot_count,
        })
    }

    /// Resize an existing reservation to hold `slot_count` slots.
    ///
    /// Stays in place whenever the size class is unchanged — which is what
    /// makes small count wobbles free, and what keeps §5.0's O(1) value-update
    /// case reachable in practice rather than only in principle.
    pub fn reallocate(
        &mut self,
        kind: PrimitiveKind,
        current: SlabRange,
        slot_count: u32,
    ) -> Result<Reallocation, SlabOverflow> {
        let capacity = size_class(slot_count).ok_or(SlabOverflow {
            kind,
            requested_slots: slot_count as u64,
        })?;
        if capacity == current.capacity {
            return Ok(Reallocation::InPlace(SlabRange {
                base: current.base,
                capacity,
                count: slot_count,
            }));
        }
        let replacement = self.allocate(kind, slot_count)?;
        self.free(kind, current);
        Ok(Reallocation::Relocated {
            previous: current,
            current: replacement,
        })
    }

    /// Release a reservation. An empty range is inert.
    pub fn free(&mut self, kind: PrimitiveKind, range: SlabRange) {
        if range.capacity == 0 {
            return;
        }
        let base_units = range.base / MIN_CLASS;
        let class_units = range.capacity / MIN_CLASS;
        self.arena_mut(kind).free_units(base_units, class_units);
    }

    /// Slots the kind's arena has ever reserved — the size a GPU buffer for
    /// this kind would have to be.
    pub fn arena_slot_capacity(&self, kind: PrimitiveKind) -> u64 {
        self.arena(kind).frontier_units as u64 * MIN_CLASS as u64
    }

    /// Slots inside the frontier that are currently unreserved.
    pub fn free_slots(&self, kind: PrimitiveKind) -> u64 {
        self.arena(kind).free_units_total() * MIN_CLASS as u64
    }

    // `PrimitiveKind::index` is the enum's own discriminant and `arenas` is
    // sized `PrimitiveKind::COUNT`, so the wrap below can never actually wrap.
    // It is written this way rather than as a bare index so that adding a kind
    // without widening the array is a wrong answer, not a panic on the render
    // path — and so no reader has to take "this index is in bounds" on faith.
    fn arena(&self, kind: PrimitiveKind) -> &Arena {
        &self.arenas[kind.index() % PrimitiveKind::COUNT]
    }

    fn arena_mut(&mut self, kind: PrimitiveKind) -> &mut Arena {
        &mut self.arenas[kind.index() % PrimitiveKind::COUNT]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const QUAD: PrimitiveKind = PrimitiveKind::Quad;

    fn allocate(allocator: &mut SlabAllocator, count: u32) -> SlabRange {
        match allocator.allocate(QUAD, count) {
            Ok(range) => range,
            Err(error) => panic!("allocation of {count} slots failed: {error:?}"),
        }
    }

    #[test]
    fn zero_slots_reserve_nothing() {
        let mut allocator = SlabAllocator::new();
        assert_eq!(allocate(&mut allocator, 0), SlabRange::EMPTY);
        assert_eq!(allocator.arena_slot_capacity(QUAD), 0);
    }

    #[test]
    fn fresh_arena_bumps_contiguously_at_the_minimum_class() {
        let mut allocator = SlabAllocator::new();
        let first = allocate(&mut allocator, 1);
        let second = allocate(&mut allocator, 10);
        assert_eq!(first.base, 0);
        assert_eq!(first.capacity, MIN_CLASS);
        assert_eq!(second.base, MIN_CLASS);
        assert_eq!(second.capacity, MIN_CLASS);
        assert_eq!(allocator.arena_slot_capacity(QUAD), 2 * MIN_CLASS as u64);
    }

    #[test]
    fn every_base_is_aligned_to_its_own_capacity() {
        let mut allocator = SlabAllocator::new();
        let mut ranges = Vec::new();
        for count in [1u32, 300, 5, 2000, 70, 100_000, 3] {
            let range = allocate(&mut allocator, count);
            assert_eq!(
                range.base % range.capacity,
                0,
                "a {count}-slot reservation landed unaligned at {}",
                range.base
            );
            ranges.push(range);
        }
        // Reservations must never overlap.
        ranges.sort_by_key(|range| range.base);
        for pair in ranges.windows(2) {
            assert!(pair[0].end() <= pair[1].base);
        }
    }

    #[test]
    fn a_freed_block_is_reused_without_growing_the_arena() {
        let mut allocator = SlabAllocator::new();
        let first = allocate(&mut allocator, 100);
        let _second = allocate(&mut allocator, 100);
        let capacity_before = allocator.arena_slot_capacity(QUAD);
        allocator.free(QUAD, first);
        let replacement = allocate(&mut allocator, 90);
        assert_eq!(replacement.base, first.base);
        assert_eq!(allocator.arena_slot_capacity(QUAD), capacity_before);
    }

    #[test]
    fn resize_within_a_class_stays_in_place() {
        let mut allocator = SlabAllocator::new();
        let range = allocate(&mut allocator, 100);
        let resized = match allocator.reallocate(QUAD, range, 120) {
            Ok(result) => result,
            Err(error) => panic!("reallocation failed: {error:?}"),
        };
        assert!(!resized.relocated());
        assert_eq!(resized.range().base, range.base);
        assert_eq!(resized.range().count, 120);
        assert_eq!(resized.range().capacity, range.capacity);
    }

    #[test]
    fn resize_across_a_class_boundary_relocates_and_recycles() {
        let mut allocator = SlabAllocator::new();
        let range = allocate(&mut allocator, 100);
        let neighbour = allocate(&mut allocator, 100);
        let resized = match allocator.reallocate(QUAD, range, 500) {
            Ok(result) => result,
            Err(error) => panic!("reallocation failed: {error:?}"),
        };
        assert!(resized.relocated());
        assert_ne!(resized.range().base, range.base);
        assert_eq!(resized.range().capacity, 512);
        // The vacated block must be reusable, not leaked.
        let recycled = allocate(&mut allocator, 100);
        assert_eq!(recycled.base, range.base);
        assert_ne!(recycled.base, neighbour.base);
    }

    #[test]
    fn buddies_coalesce_so_a_drained_arena_retracts_to_nothing() {
        let mut allocator = SlabAllocator::new();
        let ranges: Vec<SlabRange> = (0..8).map(|_| allocate(&mut allocator, 60)).collect();
        assert_eq!(allocator.arena_slot_capacity(QUAD), 8 * MIN_CLASS as u64);
        for range in ranges {
            allocator.free(QUAD, range);
        }
        assert_eq!(
            allocator.arena_slot_capacity(QUAD),
            0,
            "freeing everything must coalesce back to an empty arena"
        );
        assert_eq!(allocator.free_slots(QUAD), 0);
    }

    #[test]
    fn coalesced_space_satisfies_a_larger_later_request() {
        let mut allocator = SlabAllocator::new();
        let first = allocate(&mut allocator, 60);
        let second = allocate(&mut allocator, 60);
        let guard = allocate(&mut allocator, 60);
        allocator.free(QUAD, first);
        allocator.free(QUAD, second);
        let large = allocate(&mut allocator, 128);
        assert_eq!(
            large.base, first.base,
            "two freed buddies must merge into one block big enough for the pair"
        );
        assert_ne!(large.base, guard.base);
    }

    #[test]
    fn falling_up_to_a_larger_block_returns_the_remainder_to_the_free_lists() {
        let mut allocator = SlabAllocator::new();
        let large = allocate(&mut allocator, 500);
        let guard = allocate(&mut allocator, 60);
        allocator.free(QUAD, large);
        let small = allocate(&mut allocator, 60);
        assert_eq!(small.base, large.base);
        let another = allocate(&mut allocator, 60);
        assert!(
            another.base >= large.base && another.end() <= large.end(),
            "the split remainder of the freed 512-slot block must be reusable"
        );
        assert_ne!(another.base, guard.base);
    }

    #[test]
    fn arenas_are_independent_per_kind() {
        let mut allocator = SlabAllocator::new();
        let quad = allocate(&mut allocator, 100);
        let run = match allocator.allocate(PrimitiveKind::GlyphRun, 100) {
            Ok(range) => range,
            Err(error) => panic!("allocation failed: {error:?}"),
        };
        assert_eq!(quad.base, run.base, "each kind numbers its own slots from 0");
        assert_eq!(allocator.arena_slot_capacity(QUAD), MIN_CLASS as u64 * 2);
        assert_eq!(
            allocator.arena_slot_capacity(PrimitiveKind::GlyphRun),
            MIN_CLASS as u64 * 2
        );
    }

    #[test]
    fn overflow_is_reported_rather_than_wrapped() {
        let mut allocator = SlabAllocator::new();
        let result = allocator.allocate(QUAD, u32::MAX);
        assert_eq!(
            result,
            Err(SlabOverflow {
                kind: QUAD,
                requested_slots: u32::MAX as u64,
            })
        );
    }

    #[test]
    fn freeing_an_empty_range_is_inert() {
        let mut allocator = SlabAllocator::new();
        let range = allocate(&mut allocator, 100);
        allocator.free(QUAD, SlabRange::EMPTY);
        let after = allocate(&mut allocator, 100);
        assert_ne!(after.base, range.base);
    }

    /// Interleaved traffic against a brute-force oracle: no two live ranges
    /// may ever overlap, and every base stays self-aligned.
    #[test]
    fn interleaved_traffic_never_produces_overlapping_ranges() {
        let mut allocator = SlabAllocator::new();
        let mut live: Vec<SlabRange> = Vec::new();
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };

        for step in 0..500u32 {
            let action = next() % 3;
            if action == 2 && !live.is_empty() {
                let index = (next() % live.len() as u64) as usize;
                let range = live.swap_remove(index);
                allocator.free(QUAD, range);
            } else if action == 1 && !live.is_empty() {
                let index = (next() % live.len() as u64) as usize;
                let count = (next() % 900) as u32 + 1;
                match allocator.reallocate(QUAD, live[index], count) {
                    Ok(result) => live[index] = result.range(),
                    Err(error) => panic!("step {step} reallocation failed: {error:?}"),
                }
            } else {
                let count = (next() % 900) as u32 + 1;
                live.push(allocate(&mut allocator, count));
            }

            let mut sorted = live.clone();
            sorted.sort_by_key(|range| range.base);
            for range in &sorted {
                assert_eq!(range.base % range.capacity, 0, "step {step}: misaligned");
                assert!(range.count <= range.capacity, "step {step}: overfull");
            }
            for pair in sorted.windows(2) {
                assert!(
                    pair[0].end() <= pair[1].base,
                    "step {step}: {:?} overlaps {:?}",
                    pair[0],
                    pair[1]
                );
            }
        }
    }
}
