//! Per-layer persistent GPU-buffer suballocation (retained rendering Pillar III,
//! epic #84, spec #94, `docs/retained-layers.md` §4.2).
//!
//! [`SlabAllocator`] hands each layer a stable [`SlabRange`] per
//! primitive-kind buffer, so a clean layer uploads nothing and a dirty layer
//! re-uploads only its own slab. Ranges reallocate when a layer's primitive
//! count changes and are released when the layer dies.
//!
//! # Element units
//!
//! Every quantity this module stores or exchanges is counted in *instances*
//! (elements), never raw bytes. Byte offsets exist only at the use site, as
//! [`SlabRange::byte_offset`]. A byte-level allocator can express
//! `offset % stride != 0` (the failure mode of the previous Phase 9 attempt,
//! which shipped misaligned reads as silent GPU garbage); an element-level one
//! cannot, because the multiply happens once, after all placement decisions.
//!
//! # Size classes
//!
//! Reservations round up to powers of two floored at [`MIN_CLASS`] elements.
//! Small count wobbles therefore neither move nor resize a layer's range, and
//! freed space is reusable by any request of the same class. Requests whose
//! class no free block satisfies still succeed: allocation falls up to a
//! larger free block, carving off just the prefix it needs.
//!
//! # Compaction is advisory
//!
//! Fragmentation from resize traffic is reclaimed only when the caller asks:
//! [`SlabAllocator::compaction_plan`] computes the moves as pure data and
//! [`SlabAllocator::apply_compaction`] rewrites handles. Nothing calls these
//! automatically; [`SlabAllocator::should_compact`] merely reports whether it
//! looks worthwhile. Every other operation is correct whether or not
//! compaction has ever run, including against a stale plan.

// The renderer wires this in during the next phase; until then the lib target
// sees no references (tests do, under cfg(test), which is what keeps this
// honest). Scoped here rather than sprinkled per item so wiring it up later
// means deleting one line.
#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::Range;

/// The smallest size class a slab is ever rounded up to, in instances.
///
/// All reservations and free blocks are multiples of this, so a byte offset
/// derived via [`SlabRange::byte_offset`] is always a multiple of
/// `instance_stride * MIN_CLASS`.
pub const MIN_CLASS: u32 = 64;

/// Largest representable size class. Requests at or above this saturate to it
/// rather than overflowing `next_power_of_two`; a request larger than its
/// resulting class is rejected instead of silently truncated.
const MAX_CLASS: u32 = 1 << 31;

/// Round an instance count up to its size class. `0` maps to `0`: a layer with
/// no instances of a kind holds no reservation at all.
///
/// Counts above [`MAX_CLASS`] saturate to it; reserving them fails at
/// allocation time rather than truncating here.
pub fn size_class(count: u32) -> u32 {
    if count == 0 {
        return 0;
    }
    if count > MAX_CLASS / 2 {
        return MAX_CLASS;
    }
    count.max(MIN_CLASS).next_power_of_two()
}

/// A stable range of instance slots inside one primitive kind's global GPU
/// buffer.
///
/// `count` is how many instances the owner currently occupies; `capacity` is
/// the size-class-rounded reservation containing them (`capacity >= count`,
/// both zero for the canonical empty range). `base` is the index of the first
/// instance slot; the byte offset is `base * instance_stride`, computed once
/// at the use site.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SlabRange {
    /// Index of the first instance slot in the kind's global buffer.
    pub base: u32,
    /// Size-class-rounded reservation, in instances.
    pub capacity: u32,
    /// Instances actually occupied by the owner's current content.
    pub count: u32,
}

impl SlabRange {
    /// The canonical empty range: a layer with zero instances of a kind holds
    /// this rather than a null marker.
    pub const EMPTY: SlabRange = SlabRange {
        base: 0,
        capacity: 0,
        count: 0,
    };

    /// Whether this range holds no content.
    pub fn is_empty(self) -> bool {
        self.count == 0
    }

    /// One past the last reserved instance slot.
    pub fn end(self) -> u32 {
        self.base + self.capacity
    }

    /// Byte offset of the range start in a buffer whose instances are
    /// `instance_stride` bytes wide. Always a multiple of `instance_stride`,
    /// and of `instance_stride * MIN_CLASS` since every base is.
    pub fn byte_offset(self, instance_stride: u64) -> u64 {
        self.base as u64 * instance_stride
    }

    /// The occupied byte span. Buffer uploads should cover exactly this, not
    /// the unused capacity tail.
    pub fn used_byte_range(self, instance_stride: u64) -> Range<u64> {
        let offset = self.byte_offset(instance_stride);
        offset..offset + self.count as u64 * instance_stride
    }

    /// The full reserved byte span, including unused capacity slack.
    pub fn reserved_byte_range(self, instance_stride: u64) -> Range<u64> {
        let offset = self.byte_offset(instance_stride);
        offset..offset + self.capacity as u64 * instance_stride
    }
}

/// Which global instance buffer a slab range lives in, matching the renderer's
/// per-kind instanced draws (`QuadsData`, `ShadowsData`, `PathsData`,
/// `UnderlinesData`, `MonoSpritesData`, `PolySpritesData`).
///
/// Named distinctly from `scene::PrimitiveKind` (the sort-key enum) because
/// that one also covers non-instance primitives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum SlabKind {
    Quads,
    Shadows,
    Paths,
    Underlines,
    MonoSprites,
    PolySprites,
}

impl SlabKind {
    /// Every kind, in declaration order.
    pub const ALL: [SlabKind; SlabKind::COUNT] = [
        SlabKind::Quads,
        SlabKind::Shadows,
        SlabKind::Paths,
        SlabKind::Underlines,
        SlabKind::MonoSprites,
        SlabKind::PolySprites,
    ];

    /// Number of kinds; the width of [`LayerSlabs`]'s range array.
    pub const COUNT: usize = 6;

    /// Index into [`LayerSlabs::ranges`].
    pub fn index(self) -> usize {
        self as usize
    }
}

/// One layer's stable slab ranges across all kinds, plus the generation used
/// to detect stale references.
///
/// Callers snapshot this (`Copy`) alongside their uploaded GPU bytes;
/// comparing the snapshot's generation against [`SlabAllocator::generation`]
/// tells whether the resident copy can still be trusted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerSlabs {
    /// Per-kind ranges, indexed via [`SlabKind::index`]. Empty for kinds the
    /// layer doesn't use.
    pub ranges: [SlabRange; SlabKind::COUNT],
    /// Bumped on every change to this record: any allocation, resize,
    /// release, compaction move, or explicit
    /// [`SlabAllocator::mark_contents_changed`]. Drawn from an
    /// allocator-global monotonic counter, so a layer destroyed and recreated
    /// under the same key can never alias a stale snapshot's generation.
    pub generation: u64,
}

impl LayerSlabs {
    /// The layer's range for `kind`.
    pub fn slab(&self, kind: SlabKind) -> SlabRange {
        self.ranges[kind.index()]
    }

    /// Whether every kind's range is empty.
    pub fn is_empty(&self) -> bool {
        self.ranges.iter().all(|range| range.is_empty())
    }
}

/// One relocation computed by [`SlabAllocator::compaction_plan`]: the layer's
/// bytes must move from `src` to `dst` (same kind, capacity, and count; new
/// base). Pure data — performing the corresponding GPU copy before the old
/// region gets overwritten is the caller's job.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlabMove<K> {
    pub layer: K,
    pub kind: SlabKind,
    pub src: SlabRange,
    pub dst: SlabRange,
}

/// A defragmentation plan produced by [`SlabAllocator::compaction_plan`] and
/// consumed by [`SlabAllocator::apply_compaction`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactionPlan<K> {
    /// Repack moves, ordered by kind then ascending source base.
    pub moves: Vec<SlabMove<K>>,
}

impl<K: Copy + Eq + std::hash::Hash> CompactionPlan<K> {
    /// Distinct layers appearing in this plan, in first-appearance order.
    pub fn affected_layers(&self) -> Vec<K> {
        let mut seen = HashSet::new();
        let mut layers = Vec::new();
        for movement in &self.moves {
            if seen.insert(movement.layer) {
                layers.push(movement.layer);
            }
        }
        layers
    }
}

/// The one way slab reservation can fail: a request whose size class cannot
/// be addressed inside a single `u32`-indexed instance buffer. Requires a
/// single layer claiming over two billion instances of one kind, which no
/// viable GPU buffer supports; the caller should surface this as an error
/// rather than retrying.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlabOverflow {
    /// Kind whose arena could not fit the request.
    pub kind: SlabKind,
}

/// Allocates stable per-layer slab ranges across one buffer per [`SlabKind`],
/// tracking ownership, generations, and advisory compaction.
///
/// Generic over the layer-key type so it runs on CPU in tests without any GPU
/// machinery; the renderer instantiates it with the retained layer key.
pub struct SlabAllocator<K> {
    layers: HashMap<K, LayerSlabs>,
    arenas: [KindArena; SlabKind::COUNT],
    next_generation: u64,
}

impl<K> Default for SlabAllocator<K> {
    fn default() -> Self {
        SlabAllocator {
            layers: HashMap::new(),
            arenas: std::array::from_fn(|_| KindArena::default()),
            next_generation: 1,
        }
    }
}

impl<K: Copy + Eq + std::hash::Hash> SlabAllocator<K> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reserve a fresh range of `count` instances for `layer` in `kind`.
    ///
    /// The layer must not already hold a live range for this kind (use
    /// [`Self::realloc`]); violating that falls back to `realloc` semantics
    /// rather than corrupting state. `count == 0` returns
    /// [`SlabRange::EMPTY`] without creating a record or touching the arena.
    pub fn allocate(&mut self, layer: K, kind: SlabKind, count: u32) -> Result<SlabRange, SlabOverflow> {
        let existing = self.layers.get(&layer).map(|record| record.slab(kind));
        if existing.is_some_and(|range| !range.is_empty()) {
            debug_assert!(
                false,
                "allocate called for a layer that already holds a live slab range; use realloc"
            );
            return self.realloc(layer, kind, count);
        }
        if count == 0 {
            return Ok(SlabRange::EMPTY);
        }
        let range = self.arenas[kind.index()]
            .alloc(count)
            .ok_or(SlabOverflow { kind })?;
        let generation = self.next_generation;
        self.next_generation += 1;
        self.layers.entry(layer).or_insert(LayerSlabs {
            ranges: [SlabRange::EMPTY; SlabKind::COUNT],
            generation,
        }).ranges[kind.index()] = range;
        Ok(range)
    }

    /// Set `layer`'s range in `kind` to hold exactly `new_count` instances,
    /// allocating, resizing in place, or relocating as needed.
    ///
    /// Preference order:
    ///
    /// 1. Stay put when the current reservation's class still fits — count
    ///    wobbles within a class, and any shrink across classes (the oversized
    ///    tail goes back to the free lists rather than moving resident bytes).
    /// 2. Otherwise absorb adjacent free blocks to the right until the new
    ///    class is reached.
    /// 3. Otherwise relocate.
    ///
    /// Any change bumps the layer's generation. On relocation the caller must
    /// treat the result exactly like a fresh upload: bytes at the old location
    /// are not copied forward.
    ///
    /// Resizing to zero releases the range; resizing a range the layer
    /// doesn't hold behaves like [`Self::allocate`]. Fails with
    /// [`SlabOverflow`] only when the new size cannot be placed; the layer's
    /// previous range is left fully intact in that case.
    pub fn realloc(
        &mut self,
        layer: K,
        kind: SlabKind,
        new_count: u32,
    ) -> Result<SlabRange, SlabOverflow> {
        let Self {
            layers,
            arenas,
            next_generation,
        } = self;
        let kind_index = kind.index();
        let Some(record) = layers.get_mut(&layer) else {
            return Self::allocate_untracked(layers, arenas, next_generation, layer, kind, new_count);
        };
        let current = record.ranges[kind_index];
        let bump = |record: &mut LayerSlabs, next_generation: &mut u64| {
            record.generation = *next_generation;
            *next_generation += 1;
        };
        if current.is_empty() {
            if new_count == 0 {
                return Ok(SlabRange::EMPTY);
            }
            record.ranges[kind_index] = arenas[kind_index]
                .alloc(new_count)
                .ok_or(SlabOverflow { kind })?;
            bump(record, next_generation);
            return Ok(record.ranges[kind_index]);
        }
        if new_count == 0 {
            arenas[kind_index].free(current.base, current.capacity);
            record.ranges[kind_index] = SlabRange::EMPTY;
            bump(record, next_generation);
            return Ok(SlabRange::EMPTY);
        }
        let new_class = size_class(new_count);
        if new_class == current.capacity {
            record.ranges[kind_index].count = new_count;
        } else if new_class < current.capacity {
            arenas[kind_index].shrink(current.base, current.capacity, new_class);
            record.ranges[kind_index] = SlabRange {
                base: current.base,
                capacity: new_class,
                count: new_count,
            };
        } else if arenas[kind_index].grow_in_place(current.base, current.capacity, new_class) {
            record.ranges[kind_index].capacity = new_class;
            record.ranges[kind_index].count = new_count;
        } else {
            // Reserve before releasing: a failed placement must leave the
            // old range untouched so the caller can keep using it.
            let relocated = arenas[kind_index]
                .alloc(new_count)
                .ok_or(SlabOverflow { kind })?;
            arenas[kind_index].free(current.base, current.capacity);
            record.ranges[kind_index] = relocated;
        }
        bump(record, next_generation);
        Ok(record.ranges[kind_index])
    }

    /// Allocation tail shared by `allocate`/`realloc` when the layer holds no
    /// record yet; creates one so the returned handle has somewhere to live.
    fn allocate_untracked(
        layers: &mut HashMap<K, LayerSlabs>,
        arenas: &mut [KindArena; SlabKind::COUNT],
        next_generation: &mut u64,
        layer: K,
        kind: SlabKind,
        count: u32,
    ) -> Result<SlabRange, SlabOverflow> {
        if count == 0 {
            return Ok(SlabRange::EMPTY);
        }
        let range = arenas[kind.index()].alloc(count).ok_or(SlabOverflow { kind })?;
        let generation = *next_generation;
        *next_generation += 1;
        layers
            .entry(layer)
            .or_insert(LayerSlabs {
                ranges: [SlabRange::EMPTY; SlabKind::COUNT],
                generation,
            })
            .ranges[kind.index()] = range;
        Ok(range)
    }

    /// Release `layer`'s range in `kind`. Freeing a range the layer doesn't
    /// hold is a no-op.
    pub fn free(&mut self, layer: K, kind: SlabKind) {
        let Self {
            layers,
            arenas,
            next_generation,
        } = self;
        let Some(record) = layers.get_mut(&layer) else {
            return;
        };
        let kind_index = kind.index();
        let current = record.ranges[kind_index];
        if current.is_empty() {
            return;
        }
        arenas[kind_index].free(current.base, current.capacity);
        record.ranges[kind_index] = SlabRange::EMPTY;
        record.generation = *next_generation;
        *next_generation += 1;
    }

    /// Release every kind's range for `layer` and drop its record.
    ///
    /// The generation history disappears with the record, which is why
    /// generations come from an allocator-global counter: a future layer under
    /// the same key starts above every generation ever issued, so snapshots
    /// from the previous incarnation always read as stale.
    pub fn free_layer(&mut self, layer: K) {
        let Some(record) = self.layers.remove(&layer) else {
            return;
        };
        for kind in SlabKind::ALL {
            let range = record.slab(kind);
            if !range.is_empty() {
                self.arenas[kind.index()].free(range.base, range.capacity);
            }
        }
    }

    /// The layer's current slab record, if it holds one.
    pub fn slabs(&self, layer: K) -> Option<LayerSlabs> {
        self.layers.get(&layer).copied()
    }

    /// The layer's current generation, or `None` if it holds no record.
    pub fn generation(&self, layer: K) -> Option<u64> {
        self.layers.get(&layer).map(|record| record.generation)
    }

    /// Force-bump a layer's generation, marking previously uploaded content
    /// stale without touching the arena. Returns whether the layer exists.
    pub fn mark_contents_changed(&mut self, layer: K) -> bool {
        match self.layers.get_mut(&layer) {
            Some(record) => {
                record.generation = self.next_generation;
                self.next_generation += 1;
                true
            }
            None => false,
        }
    }

    /// Size of `kind`'s arena in instances: what the backing GPU buffer must
    /// provide.
    pub fn arena_element_capacity(&self, kind: SlabKind) -> u32 {
        self.arenas[kind.index()].high_water
    }

    /// Summed instance counts across all live ranges in `kind`.
    pub fn live_elements(&self, kind: SlabKind) -> u64 {
        self.live_totals()[kind.index()]
    }

    /// Summed capacities across all live ranges in `kind`.
    pub fn reserved_elements(&self, kind: SlabKind) -> u64 {
        self.reserved_totals()[kind.index()]
    }

    fn live_totals(&self) -> [u64; SlabKind::COUNT] {
        let mut totals = [0; SlabKind::COUNT];
        for record in self.layers.values() {
            for (total, range) in totals.iter_mut().zip(record.ranges) {
                *total += range.count as u64;
            }
        }
        totals
    }

    fn reserved_totals(&self) -> [u64; SlabKind::COUNT] {
        let mut totals = [0; SlabKind::COUNT];
        for record in self.layers.values() {
            for (total, range) in totals.iter_mut().zip(record.ranges) {
                *total += range.capacity as u64;
            }
        }
        totals
    }

    /// Whether repacking looks worthwhile: the arenas are large enough to be
    /// worth touching and aggregate utilization (live instances over total
    /// arena footprint, so free-but-unreused holes count against it) has
    /// fallen below `utilization_threshold`. Purely advisory — see the module
    /// docs.
    pub fn should_compact(&self, utilization_threshold: f32) -> bool {
        const MIN_ARENA_FOR_COMPACTION: u64 = 16 * 1024;
        let footprint: u64 = SlabKind::ALL
            .iter()
            .map(|kind| self.arena_element_capacity(*kind) as u64)
            .sum();
        if footprint < MIN_ARENA_FOR_COMPACTION || utilization_threshold <= 0.0 {
            return false;
        }
        let live: u64 = self.live_totals().iter().sum();
        (live as f32 / footprint as f32) < utilization_threshold
    }

    /// Compute the defragmentation moves that would repack every kind's live
    /// ranges contiguously from base zero. Pure: arenas and generations are
    /// untouched, repeated calls agree, and layers already at their packed
    /// position produce no move.
    pub fn compaction_plan(&self) -> CompactionPlan<K> {
        let mut plan = CompactionPlan { moves: Vec::new() };
        for kind in SlabKind::ALL {
            let mut placed: Vec<(u32, SlabRange, K)> = self
                .layers
                .iter()
                .filter_map(|(layer, record)| {
                    let range = record.slab(kind);
                    (!range.is_empty()).then_some((range.base, range, *layer))
                })
                .collect();
            placed.sort_by_key(|(base, _, _)| *base);
            debug_assert!(
                placed.windows(2).all(|pair| pair[0].1.end() <= pair[1].0),
                "live slab ranges overlap; allocator state is corrupt"
            );
            let mut cursor = 0u32;
            for (base, range, layer) in placed {
                if base != cursor {
                    plan.moves.push(SlabMove {
                        layer,
                        kind,
                        src: range,
                        dst: SlabRange {
                            base: cursor,
                            capacity: range.capacity,
                            count: range.count,
                        },
                    });
                }
                debug_assert!(
                    cursor.checked_add(range.capacity).is_some(),
                    "packed arena exceeds u32 address space"
                );
                cursor += range.capacity;
            }
        }
        plan
    }

    /// Rewrite handles according to `plan`, reclaiming the space the moves
    /// vacate and bumping each affected layer's generation once.
    ///
    /// Moves are validated at apply time: any whose `src` no longer matches
    /// the layer's current range (something reallocated or freed between
    /// planning and applying) is skipped, leaving both that layer and the
    /// arena internally consistent — just less compacted. Skipping is what
    /// makes compaction safe to run opportunistically, on top of correctness
    /// never depending on it running at all. Returns how many moves applied.
    pub fn apply_compaction(&mut self, plan: &CompactionPlan<K>) -> usize {
        let Self {
            layers,
            arenas,
            next_generation,
        } = self;
        let mut seen = HashSet::new();
        let mut touched: Vec<K> = Vec::new();
        let mut touched_kinds = [false; SlabKind::COUNT];
        let mut applied = 0usize;
        for movement in &plan.moves {
            let kind_index = movement.kind.index();
            if movement.src.is_empty() || movement.dst.is_empty() {
                continue;
            }
            let Some(record) = layers.get_mut(&movement.layer) else {
                continue;
            };
            if record.ranges[kind_index] != movement.src {
                continue;
            }
            record.ranges[kind_index] = movement.dst;
            applied += 1;
            touched_kinds[kind_index] = true;
            if seen.insert(movement.layer) {
                touched.push(movement.layer);
            }
        }
        if applied == 0 {
            return 0;
        }
        for layer in touched {
            if let Some(record) = layers.get_mut(&layer) {
                record.generation = *next_generation;
            }
            *next_generation += 1;
        }
        for (kind_index, arena) in arenas.iter_mut().enumerate() {
            if !touched_kinds[kind_index] {
                continue;
            }
            let blocks: Vec<ReservedBlock> = layers
                .values()
                .filter_map(|record| {
                    let range = record.ranges[kind_index];
                    (!range.is_empty()).then_some(ReservedBlock {
                        base: range.base,
                        capacity: range.capacity,
                    })
                })
                .collect();
            arena.rebuild(blocks);
        }
        applied
    }
}

/// One reserved interval inside a kind's arena. Owner-free on purpose:
/// ownership lives in `SlabAllocator::layers`, geometry lives here.
#[derive(Clone, Copy, Debug)]
struct ReservedBlock {
    base: u32,
    capacity: u32,
}

impl ReservedBlock {
    fn end(&self) -> u32 {
        self.base + self.capacity
    }
}

/// Spatial state for one kind's buffer: which intervals are reserved and
/// which are free. All bases and capacities are multiples of [`MIN_CLASS`];
/// free blocks are kept as distinct powers of two (times [`MIN_CLASS`]) so
/// they slot into class buckets directly.
#[derive(Debug, Default)]
struct KindArena {
    /// Live intervals, sorted by ascending base, pairwise disjoint.
    reserved: Vec<ReservedBlock>,
    /// Free blocks bucketed by size class; each holds candidate bases.
    free_by_class: BTreeMap<u32, Vec<u32>>,
    /// Nothing reserved or free sits at or beyond this base.
    high_water: u32,
}

impl KindArena {
    /// Reserve space for `count` instances and return the resulting range,
    /// or `None` when the request cannot be placed in a `u32` address space.
    fn alloc(&mut self, count: u32) -> Option<SlabRange> {
        let class = size_class(count);
        if class < count {
            return None;
        }
        let base = match self.take_free(class) {
            Some(base) => base,
            None => {
                let end = self.high_water.checked_add(class)?;
                let base = self.high_water;
                self.high_water = end;
                base
            }
        };
        let block = ReservedBlock { base, capacity: class };
        self.insert_reserved(block);
        Some(SlabRange {
            base,
            capacity: class,
            count,
        })
    }

    /// Take a whole free block of exactly `class`, else carve one off the
    /// front of the smallest larger block ("falls up"), else `None`.
    fn take_free(&mut self, class: u32) -> Option<u32> {
        if let Some(bucket) = self.free_by_class.get_mut(&class) {
            if let Some(base) = bucket.pop() {
                if bucket.is_empty() {
                    self.free_by_class.remove(&class);
                }
                return Some(base);
            }
            self.free_by_class.remove(&class);
        }
        let &bigger = self.free_by_class.range((class + 1)..).next()?.0;
        let bucket = self.free_by_class.get_mut(&bigger)?;
        let base = bucket.pop()?;
        if bucket.is_empty() {
            self.free_by_class.remove(&bigger);
        }
        // The remnant stays behind as fresh free blocks of smaller classes.
        self.push_free(base + class, bigger - class);
        Some(base)
    }

    /// Return `total` instances starting at `base` to the free lists, split
    /// into distinct power-of-two chunks so each lands in a class bucket.
    /// `total` must be a nonzero multiple of [`MIN_CLASS`], which makes every
    /// chunk at least [`MIN_CLASS`].
    fn push_free(&mut self, base: u32, total: u32) {
        debug_assert!(base.is_multiple_of(MIN_CLASS));
        debug_assert!(total.is_multiple_of(MIN_CLASS) && total >= MIN_CLASS);
        let mut remaining = total;
        let mut offset = base;
        while remaining > 0 {
            let chunk = 1 << (31 - remaining.leading_zeros());
            debug_assert!(chunk >= MIN_CLASS);
            self.free_by_class.entry(chunk).or_default().push(offset);
            offset += chunk;
            remaining -= chunk;
        }
    }

    /// Return a live reservation to the free lists, split into distinct
    /// power-of-two chunks so each lands in a class bucket.
    fn free(&mut self, base: u32, capacity: u32) {
        let index = self.reserved.partition_point(|block| block.base < base);
        debug_assert!(
            index < self.reserved.len()
                && self.reserved[index].base == base
                && self.reserved[index].capacity == capacity,
            "freeing a range that is not reserved"
        );
        if index < self.reserved.len() && self.reserved[index].base == base {
            self.reserved.remove(index);
        }
        self.push_free(base, capacity);
    }

    fn insert_reserved(&mut self, block: ReservedBlock) {
        debug_assert!(block.base.is_multiple_of(MIN_CLASS));
        let index = self.reserved.partition_point(|existing| existing.base < block.base);
        debug_assert!(
            self.reserved[..index].last().is_none_or(|prev| prev.end() <= block.base)
                && self.reserved[index..].first().is_none_or(|next| block.end() <= next.base),
            "reserved blocks must stay disjoint"
        );
        self.reserved.insert(index, block);
    }

    fn shrink(&mut self, base: u32, old_capacity: u32, new_capacity: u32) {
        debug_assert!(new_capacity < old_capacity);
        let index = self.reserved.partition_point(|block| block.base < base);
        debug_assert!(
            index < self.reserved.len() && self.reserved[index].base == base,
            "shrinking a range that is not reserved"
        );
        if index < self.reserved.len() && self.reserved[index].base == base {
            self.reserved[index].capacity = new_capacity;
        }
        self.push_free(base + new_capacity, old_capacity - new_capacity);
    }

    /// Grow a reservation in place by absorbing free capacity immediately to
    /// its right, however the adjacent space happens to be tiled. The free
    /// lists are first scanned read-only to confirm some combination of
    /// blocks covers `[old_end, new_end)` exactly; only then are blocks
    /// actually removed, so a failed probe leaves no partially grown state.
    /// Any tail left over beyond `new_end` returns to the free lists.
    fn grow_in_place(&mut self, base: u32, old_capacity: u32, new_capacity: u32) -> bool {
        debug_assert!(new_capacity > old_capacity);
        let index = self.reserved.partition_point(|block| block.base < base);
        debug_assert!(
            index < self.reserved.len() && self.reserved[index].base == base,
            "growing a range that is not reserved"
        );
        if index >= self.reserved.len() || self.reserved[index].base != base {
            return false;
        }
        let Some(run_start) = base.checked_add(old_capacity) else {
            return false;
        };
        let Some(run_end) = base.checked_add(new_capacity) else {
            return false;
        };
        let mut candidates: Vec<(u32, u32)> = self
            .free_by_class
            .iter()
            .flat_map(|(&class, bases)| {
                bases
                    .iter()
                    .map(move |&offset| (offset, class))
                    .filter(move |&(offset, class)| offset < run_end && offset + class > run_start)
            })
            .collect();
        candidates.sort_unstable();
        let mut consumed: Vec<(u32, u32)> = Vec::new();
        let mut remainder: Option<(u32, u32)> = None;
        let mut cursor = run_start;
        for &(offset, class) in &candidates {
            debug_assert!(offset >= cursor, "free blocks must not overlap");
            if offset != cursor {
                return false;
            }
            let block_end = offset + class;
            consumed.push((class, offset));
            if block_end >= run_end {
                if block_end > run_end {
                    remainder = Some((run_end, block_end - run_end));
                }
                cursor = run_end;
                break;
            }
            cursor = block_end;
        }
        if cursor != run_end {
            return false;
        }
        for (class, offset) in consumed {
            let Some(bucket) = self.free_by_class.get_mut(&class) else {
                continue;
            };
            if let Some(position) = bucket.iter().position(|&candidate| candidate == offset) {
                bucket.swap_remove(position);
            }
            if bucket.is_empty() {
                self.free_by_class.remove(&class);
            }
        }
        if let Some((remainder_base, remainder_count)) = remainder {
            self.push_free(remainder_base, remainder_count);
        }
        self.reserved[index].capacity = new_capacity;
        true
    }

    /// Replace all spatial state with exactly `blocks` (sorted, disjoint),
    /// turning every gap between them back into free blocks. Used after
    /// compaction applies moves, including partially-applied plans: gaps left
    /// by skipped stale moves are recovered rather than leaked.
    fn rebuild(&mut self, mut blocks: Vec<ReservedBlock>) {
        blocks.sort_by_key(|block| block.base);
        debug_assert!(
            blocks.windows(2).all(|pair| pair[0].end() <= pair[1].base),
            "rebuilt arena must be disjoint"
        );
        let mut gaps: Vec<(u32, u32)> = Vec::new();
        let mut cursor = 0u32;
        for block in &blocks {
            if block.base > cursor {
                gaps.push((cursor, block.base - cursor));
            }
            cursor = block.end();
        }
        self.reserved = blocks;
        self.high_water = cursor;
        self.free_by_class.clear();
        for (base, total) in gaps {
            self.push_free(base, total);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KIND: SlabKind = SlabKind::Quads;
    const OTHER: SlabKind = SlabKind::Shadows;

    fn range_of(allocator: &SlabAllocator<u32>, layer: u32, kind: SlabKind) -> SlabRange {
        allocator
            .slabs(layer)
            .map(|slabs| slabs.slab(kind))
            .unwrap_or(SlabRange::EMPTY)
    }

    /// xorshift64*: deterministic, dependency-free, enough to drive a
    /// reproducible fuzz-style model check.
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Rng(seed | 1)
        }

        fn next_u64(&mut self) -> u64 {
            let mut state = self.0;
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            self.0 = state;
            state
        }

        fn below(&mut self, bound: u32) -> u32 {
            (self.next_u64() % u64::from(bound)) as u32
        }
    }

    #[test]
    fn size_classes_round_to_powers_of_two_floored_at_min() {
        assert_eq!(size_class(0), 0);
        assert_eq!(size_class(1), MIN_CLASS);
        assert_eq!(size_class(MIN_CLASS), MIN_CLASS);
        assert_eq!(size_class(MIN_CLASS + 1), MIN_CLASS * 2);
        assert_eq!(size_class(100), 128);
        assert_eq!(size_class(1000), 1024);
        assert_eq!(size_class(1 << 20), 1 << 20);
        // Saturation must stay panic-free at every boundary.
        assert_eq!(size_class(MAX_CLASS / 2), MAX_CLASS / 2);
        assert_eq!(size_class(MAX_CLASS / 2 + 1), MAX_CLASS);
        assert_eq!(size_class(u32::MAX), MAX_CLASS);
    }

    #[test]
    fn fresh_allocator_bumps_contiguously_in_element_units() {
        let mut allocator = SlabAllocator::new();
        let first = allocator.allocate(1, KIND, 100).unwrap();
        assert_eq!(
            first,
            SlabRange {
                base: 0,
                capacity: 128,
                count: 100,
            }
        );
        assert_eq!(allocator.arena_element_capacity(KIND), 128);

        let second = allocator.allocate(2, KIND, 10).unwrap();
        assert_eq!(
            second,
            SlabRange {
                base: 128,
                capacity: MIN_CLASS,
                count: 10,
            },
            "second allocation must not overlap the first"
        );
        assert_eq!(allocator.arena_element_capacity(KIND), 128 + MIN_CLASS);
    }

    #[test]
    fn zero_count_allocate_is_churn_free() {
        let mut allocator = SlabAllocator::new();
        for _ in 0..3 {
            assert_eq!(allocator.allocate(7, KIND, 0).unwrap(), SlabRange::EMPTY);
        }
        assert_eq!(allocator.arena_element_capacity(KIND), 0);
        assert!(
            allocator.slabs(7).is_none(),
            "no record may exist for an all-empty layer"
        );
        assert_eq!(allocator.realloc(7, KIND, 0).unwrap(), SlabRange::EMPTY);
        assert!(allocator.slabs(7).is_none());
    }

    #[test]
    fn freed_block_is_reused_without_arena_growth() {
        let mut allocator = SlabAllocator::new();
        let original = allocator.allocate(1, KIND, 100).unwrap();
        allocator.free(1, KIND);
        assert_eq!(
            allocator.arena_element_capacity(KIND),
            128,
            "freeing must not shrink the high-water mark"
        );

        let reused = allocator.allocate(2, KIND, 90).unwrap();
        assert_eq!(
            reused.base, original.base,
            "a same-class request must reuse the freed block"
        );
        assert_eq!(allocator.arena_element_capacity(KIND), 128);

        allocator.free_layer(2);
        let cross_kind = allocator.allocate(3, OTHER, 90).unwrap();
        assert_eq!(
            cross_kind.base, 0,
            "kinds have independent arenas starting at their own zero"
        );
    }

    #[test]
    fn realloc_within_class_stays_in_place_and_bumps_generation() {
        let mut allocator = SlabAllocator::new();
        let initial = allocator.allocate(1, KIND, 100).unwrap();
        let generation_before = allocator.generation(1).unwrap();

        let resized = allocator.realloc(1, KIND, 120).unwrap();
        assert_eq!(resized.base, initial.base);
        assert_eq!(resized.capacity, 128);
        assert_eq!(resized.count, 120);
        assert_eq!(range_of(&allocator, 1, KIND), resized);
        assert!(
            allocator.generation(1).unwrap() > generation_before,
            "a count change is a content change and must invalidate snapshots"
        );

        // Shrinking across classes stays put byte-wise but tightens the
        // reservation: the oversized tail returns to the free lists without
        // moving any resident bytes.
        let shrunk = allocator.realloc(1, KIND, 5).unwrap();
        assert_eq!(shrunk.base, initial.base);
        assert_eq!(shrunk.capacity, MIN_CLASS);
        assert_eq!(shrunk.count, 5);
        // And regrowth one class up re-absorbs the adjacent tail in place.
        let regrown = allocator.realloc(1, KIND, 100).unwrap();
        assert_eq!(
            (regrown.base, regrown.capacity),
            (initial.base, 128)
        );
    }

    #[test]
    fn realloc_across_class_boundary_relocates_and_recycles_old_block() {
        let mut allocator = SlabAllocator::new();
        let initial = allocator.allocate(1, KIND, 100).unwrap(); // class 128

        let grown = allocator.realloc(1, KIND, 5000).unwrap(); // class 8192
        assert_ne!(
            grown.base, initial.base,
            "crossing a class boundary with no adjacent free space must relocate"
        );
        assert_eq!((grown.capacity, grown.count), (8192, 5000));
        assert_eq!(
            allocator.arena_element_capacity(KIND),
            initial.end().max(grown.end())
        );

        let recycled = allocator.allocate(2, KIND, 100).unwrap();
        assert_eq!(
            recycled.base, initial.base,
            "the vacated block must be back on the free list"
        );
    }

    #[test]
    fn grow_absorbs_adjacent_free_neighbor_in_place() {
        let mut allocator = SlabAllocator::new();
        let left = allocator.allocate(1, KIND, 100).unwrap(); // [0, 128)
        let right = allocator.allocate(2, KIND, 100).unwrap(); // [128, 256)
        allocator.free(2, KIND);

        let grown = allocator.realloc(1, KIND, 200).unwrap(); // needs class 256
        assert_eq!(
            (grown.base, grown.capacity),
            (left.base, 256),
            "an adjacent free buddy must be absorbed in place instead of relocating"
        );
        assert_eq!(grown.count, 200);
        assert_eq!(allocator.arena_element_capacity(KIND), right.end());

        // Without an adjacent free block, the same growth relocates.
        let mut dense = SlabAllocator::new();
        let a = dense.allocate(1, KIND, 100).unwrap();
        dense.allocate(2, KIND, 100).unwrap();
        let moved = dense.realloc(1, KIND, 200).unwrap();
        assert_ne!(moved.base, a.base);
        assert!(moved.base >= a.end());
    }

    #[test]
    fn grow_absorbs_a_chain_of_adjacent_free_blocks() {
        let mut allocator = SlabAllocator::new();
        allocator.allocate(1, KIND, 64).unwrap(); // [0, 64)
        for expected_base in [64u32, 128, 192] {
            let block = allocator.allocate(expected_base, KIND, 64).unwrap();
            assert_eq!(block.base, expected_base);
        }
        // Freeing all three 64s leaves the run [64, 256) tiled by small blocks.
        allocator.free_layer(64);
        allocator.free_layer(128);
        allocator.free_layer(192);
        let grown = allocator.realloc(1, KIND, 200).unwrap();
        assert_eq!(
            (grown.base, grown.capacity),
            (0, 256),
            "growth must absorb however the adjacent run happens to be tiled"
        );
    }

    #[test]
    fn shrink_splits_the_tail_into_reusable_chunks() {
        let mut allocator = SlabAllocator::new();
        let big = allocator.allocate(1, KIND, 200).unwrap(); // class 256
        let small = allocator.realloc(1, KIND, 40).unwrap(); // stays put at class 64
        assert_eq!(small.base, big.base);
        assert_eq!(small.capacity, MIN_CLASS);

        // The 192-element tail must return as independently reusable blocks:
        // a 128 directly after the shrunken range, then a 64.
        let taker_a = allocator.allocate(2, KIND, 100).unwrap();
        assert_eq!(taker_a.base, small.end());
        assert_eq!(taker_a.capacity, 128);
        let taker_b = allocator.allocate(3, KIND, 50).unwrap();
        assert_eq!(taker_b.base, taker_a.end());
        assert_eq!(taker_b.capacity, 64);
        assert_eq!(allocator.arena_element_capacity(KIND), big.end());
    }

    #[test]
    fn empty_and_unknown_ranges_are_inert_no_ops() {
        let mut allocator = SlabAllocator::new();
        allocator.free(9, KIND); // unknown layer
        allocator.allocate(9, KIND, 0).unwrap();
        allocator.free(9, KIND); // canonical-empty only
        assert_eq!(allocator.arena_element_capacity(KIND), 0);

        allocator.allocate(9, KIND, 70).unwrap();
        let generation_held = allocator.generation(9).unwrap();
        allocator.free(9, KIND);
        allocator.free(9, KIND); // double free is a no-op
        assert!(range_of(&allocator, 9, KIND).is_empty());
        assert_eq!(
            allocator.generation(9).unwrap(),
            generation_held + 1,
            "the real free bumps exactly once; no-op frees must not bump at all"
        );

        // realloc to zero behaves like free.
        let reallocated = allocator.allocate(9, KIND, 70).unwrap();
        assert_eq!(reallocated.capacity, 128);
        assert_eq!(allocator.realloc(9, KIND, 0).unwrap(), SlabRange::EMPTY);
        assert!(allocator.slabs(9).is_some_and(|slabs| slabs.is_empty()));

        // A record whose kinds are all empty survives until free_layer so
        // its generation history covers idle periods.
        allocator.free_layer(9);
        assert!(allocator.slabs(9).is_none());
    }

    #[test]
    fn exhausted_exact_class_falls_up_to_larger_bucket() {
        let mut allocator = SlabAllocator::new();
        let big = allocator.allocate(1, KIND, 128).unwrap(); // class 128 at [0, 128)
        allocator.free(1, KIND); // only a 128-class block is free

        let carved = allocator.allocate(2, KIND, 64).unwrap(); // class 64: falls up
        assert_eq!(
            carved,
            SlabRange {
                base: big.base,
                capacity: MIN_CLASS,
                count: 64,
            },
            "allocation must carve its prefix off the larger free block"
        );
        // The remnant must be independently usable by a same-class request.
        let remnant = allocator.allocate(3, KIND, 64).unwrap();
        assert_eq!((remnant.base, remnant.capacity), (big.base + 64, 64));
        assert_eq!(allocator.arena_element_capacity(KIND), 128);
    }

    #[test]
    fn overflow_is_reported_instead_of_wrapping_or_panicking() {
        let mut allocator = SlabAllocator::new();
        let huge = MAX_CLASS / 2; // three fit in a u32 address space, four do not
        for layer in [1u32, 2, 3] {
            allocator.allocate(layer, KIND, huge).unwrap();
        }
        let error = allocator.allocate(4, KIND, huge).unwrap_err();
        assert_eq!(error.kind, KIND);
        assert!(allocator.slabs(4).is_none(), "a failed allocate must not create a record");
        assert!(allocator.realloc(4, KIND, 10).is_ok(), "the arena stays usable afterwards");

        // A request larger than any class fails cleanly too.
        assert_eq!(allocator.allocate(5, KIND, u32::MAX).unwrap_err().kind, KIND);

        // A failed relocation must leave the old range fully intact: the
        // arena's remaining headroom is just under one more 2^30-class block,
        // so growing the small resident range cannot be placed anywhere.
        let resident = range_of(&allocator, 4, KIND);
        let error = allocator.realloc(4, KIND, MAX_CLASS / 2).unwrap_err();
        assert_eq!(error.kind, KIND);
        assert_eq!(range_of(&allocator, 4, KIND), resident);
    }

    #[test]
    fn generations_are_globally_monotonic_and_detect_stale_snapshots() {
        let mut allocator = SlabAllocator::new();
        allocator.allocate(1, OTHER, 30).unwrap();
        let snapshot = allocator.slabs(1).unwrap();

        // Traffic to unrelated layers leaves our generation untouched.
        allocator.allocate(2, KIND, 30).unwrap();
        assert_eq!(allocator.generation(1), Some(snapshot.generation));

        // Same-kind resize bumps even when the base does not move.
        allocator.realloc(1, OTHER, 31).unwrap();
        assert!(allocator.generation(1).unwrap() > snapshot.generation);

        // The explicit dirty-marking hook.
        let before_mark = allocator.generation(1).unwrap();
        assert!(allocator.mark_contents_changed(1));
        assert!(allocator.generation(1).unwrap() > before_mark);
        assert!(!allocator.mark_contents_changed(999));

        // Destroying and recreating under the same key can never alias old
        // snapshots because the counter is allocator-global.
        let last_seen = allocator.generation(1).unwrap();
        allocator.free_layer(1);
        allocator.allocate(1, OTHER, 30).unwrap();
        assert!(allocator.generation(1).unwrap() > last_seen);
        let reincarnated = allocator.slabs(1).unwrap();
        assert_ne!(
            reincarnated.generation, snapshot.generation,
            "a pre-destruction snapshot must read as stale after reincarnation"
        );
    }

    #[test]
    fn compaction_plan_is_pure_deterministic_and_correct() {
        let mut allocator = SlabAllocator::new();
        // Quads arena: [a][b][c], then free b → only c must move down.
        allocator.allocate(1, KIND, 60).unwrap(); // [0, 64)
        allocator.allocate(2, KIND, 60).unwrap(); // [64, 128)
        let c_quads = allocator.allocate(3, KIND, 60).unwrap(); // [128, 192)
        allocator.free(2, KIND);
        // Shadows arena stays packed: no moves may be planned there.
        allocator.allocate(1, OTHER, 60).unwrap();
        let shadows_3 = allocator.allocate(3, OTHER, 60).unwrap();

        let plan = allocator.compaction_plan();
        assert_eq!(plan, allocator.compaction_plan(), "planning twice must agree");
        assert_eq!(plan.moves.len(), 1);
        let movement = plan.moves[0];
        assert_eq!(
            (movement.layer, movement.kind),
            (3, KIND),
            "layer 1 already sits at base zero and must not appear"
        );
        assert_eq!(movement.src, c_quads);
        assert_eq!(
            movement.dst,
            SlabRange {
                base: 64,
                capacity: c_quads.capacity,
                count: c_quads.count,
            }
        );
        assert_eq!(plan.affected_layers(), vec![3]);

        // Purity: planning changed nothing observable.
        assert_eq!(allocator.slabs(3).unwrap().slab(OTHER), shadows_3);
        assert_eq!(range_of(&allocator, 3, KIND), c_quads);
    }

    #[test]
    fn apply_compaction_rewrites_handles_and_reclaims_space() {
        let mut allocator = SlabAllocator::new();
        allocator.allocate(1, KIND, 60).unwrap(); // [0, 64)
        allocator.allocate(2, KIND, 60).unwrap(); // [64, 128)
        allocator.allocate(3, KIND, 60).unwrap(); // [128, 192)
        allocator.free(2, KIND);
        let generation_3_before = allocator.generation(3).unwrap();
        let generation_1_before = allocator.generation(1).unwrap();

        let plan = allocator.compaction_plan();
        let applied = allocator.apply_compaction(&plan);
        assert_eq!(applied, 1);

        // Handles rewritten; unaffected layer's generation untouched.
        assert_eq!(
            range_of(&allocator, 3, KIND),
            SlabRange {
                base: 64,
                capacity: 64,
                count: 60,
            }
        );
        assert!(allocator.generation(3).unwrap() > generation_3_before);
        assert_eq!(allocator.generation(1).unwrap(), generation_1_before);

        // The hole is gone entirely: the next allocation bumps from 128.
        let fresh = allocator.allocate(4, KIND, 60).unwrap();
        assert_eq!(fresh.base, 128);

        // Applying the same plan again is a no-op (already packed).
        assert_eq!(allocator.apply_compaction(&plan), 0);
    }

    #[test]
    fn apply_compaction_tolerates_stale_plans_without_leaking() {
        let mut allocator = SlabAllocator::new();
        allocator.allocate(1, KIND, 60).unwrap(); // [0, 64)
        allocator.allocate(2, KIND, 60).unwrap(); // [64, 128)
        allocator.allocate(3, KIND, 60).unwrap(); // [128, 192)
        allocator.free(2, KIND);
        let plan = allocator.compaction_plan();

        // Invalidate layer 3's move between planning and applying: a count
        // change within the same class keeps the geometry identical but the
        // plan's `src` (which carries the count) no longer matches.
        allocator.realloc(3, KIND, 50).unwrap();
        assert_eq!(allocator.apply_compaction(&plan), 0);

        // Nothing leaked: a full repack still reaches the exact packed size.
        let full_plan = allocator.compaction_plan();
        assert_eq!(allocator.apply_compaction(&full_plan), 1);
        assert_eq!(range_of(&allocator, 3, KIND).base, 64);
        assert_eq!(
            allocator.arena_element_capacity(KIND),
            128,
            "live reservations are one 64-class block per surviving layer"
        );

        // And the allocator remains fully functional afterwards.
        let fresh = allocator.allocate(9, KIND, 200).unwrap();
        assert_eq!(fresh.base, 128);
        assert_eq!(fresh.capacity, 256);
    }

    #[test]
    fn should_compact_threshold_heuristic() {
        let mut allocator = SlabAllocator::new();
        allocator.allocate(1, KIND, 60).unwrap();
        allocator.free(1, KIND);
        // Tiny arenas never justify a pass, whatever the threshold says.
        assert!(!allocator.should_compact(0.99));

        // Build a large sparse arena without letting frees get reused:
        // escalating classes keep freed blocks from matching later requests.
        let mut sparse = SlabAllocator::new();
        for index in 0..64u32 {
            sparse.allocate(index, KIND, 1024).unwrap();
        }
        for index in 0..60u32 {
            sparse.free(index, KIND);
        }
        assert!(sparse.should_compact(0.5));
        assert!(!sparse.should_compact(0.01), "below a strict threshold nothing needs doing");

        // A dense arena of the same size stays above the threshold.
        let mut dense = SlabAllocator::new();
        for index in 0..64u32 {
            dense.allocate(index, KIND, 1024).unwrap();
        }
        assert!(!dense.should_compact(0.5));
        // Utilization equal to the threshold does not trigger (< not <=).
        let live = dense.live_elements(KIND) as f32;
        let footprint = [
            dense.arena_element_capacity(KIND),
            dense.arena_element_capacity(OTHER),
        ]
        .into_iter()
        .sum::<u32>() as f32;
        assert!(!dense.should_compact(live / footprint));
    }

    /// The regression class for the previous Phase 9 attempt's silent GPU
    /// garbage: byte offsets must stay stride-aligned no matter what sizes
    /// were requested, which element-unit math makes structural.
    #[test]
    fn byte_offsets_stay_stride_aligned_across_arbitrary_resizes() {
        let mut allocator = SlabAllocator::new();
        let mut rng = Rng::new(0x51AB_CAFE);
        for step in 0..200u32 {
            let layer = step % 8;
            let count = match rng.below(4) {
                0 => 0,
                1 => rng.below(MIN_CLASS + 16),
                _ => rng.below(700),
            };
            let kind = SlabKind::ALL[(rng.below(SlabKind::ALL.len() as u32)) as usize];
            allocator.realloc(layer, kind, count).unwrap();
        }

        for kind in SlabKind::ALL {
            for stride in [4u64, 16, 48, 64, 256] {
                for layer in 0..8u32 {
                    let range = range_of(&allocator, layer, kind);
                    if range.is_empty() {
                        continue;
                    }
                    let used = range.used_byte_range(stride);
                    assert_eq!(used.start % stride, 0);
                    assert_eq!(used.end % stride, 0);
                    let reserved = range.reserved_byte_range(stride);
                    assert_eq!(reserved.start % (stride * MIN_CLASS as u64), 0);
                    assert_eq!((reserved.end - reserved.start) % stride, 0);
                    // Common GPU binding alignments inherit from the
                    // MIN_CLASS-granular base for realistic strides.
                    if stride <= 256 {
                        assert_eq!(used.start % 256, 0, "stride {stride}");
                    }
                    assert_eq!(range.byte_offset(stride) % stride, 0);
                }
            }
        }
    }

    /// Interleaved multi-kind/multi-layer stress checked against a naive
    /// oracle: every op's effect on expected residency is mirrored in a plain
    /// map, and allocator geometry is validated independently of placement
    /// policy (disjointness, class invariant, sums, alignment).
    #[test]
    fn interleaved_multi_kind_multi_layer_matches_oracle() {
        const LAYERS: u32 = 12;
        let mut allocator = SlabAllocator::new();
        let mut model: HashMap<(u32, usize), u32> = HashMap::new();
        let mut touched: Vec<u32> = Vec::new();
        let mut seen_layers = HashSet::new();
        let mut rng = Rng::new(0xC0FFEE);

        let draw_count = |rng: &mut Rng| match rng.below(8) {
            0 => 0,
            1..=2 => MIN_CLASS / 2 + rng.below(MIN_CLASS + 17),
            _ => rng.below(600),
        };

        for _ in 0..1500 {
            let layer = rng.below(LAYERS);
            let kind_index = rng.below(SlabKind::ALL.len() as u32) as usize;
            let kind = SlabKind::ALL[kind_index];
            match rng.below(10) {
                0..=5 => {
                    let count = draw_count(&mut rng);
                    let range = allocator.realloc(layer, kind, count).unwrap();
                    if count == 0 {
                        model.remove(&(layer, kind_index));
                        assert!(range.is_empty());
                    } else {
                        model.insert((layer, kind_index), count);
                        assert_eq!((range.count, range.capacity), (count, size_class(count)));
                    }
                }
                6 | 7 => {
                    allocator.free(layer, kind);
                    model.remove(&(layer, kind_index));
                }
                8 => {
                    allocator.free_layer(layer);
                    for other_kind in 0..SlabKind::COUNT {
                        model.remove(&(layer, other_kind));
                    }
                }
                _ => {
                    // Directed adjacency churn: two same-class neighbors, the
                    // right one freed, the left grown one class up. Whenever
                    // the pair landed adjacently the growth must absorb the
                    // freed neighbor in place.
                    let left = LAYERS + rng.below(4096);
                    let right = left + 512;
                    let class_count = MIN_CLASS << rng.below(3); // 64, 128 or 256
                    let left_range = allocator.realloc(left, OTHER, class_count).unwrap();
                    let right_range = allocator.realloc(right, OTHER, class_count).unwrap();
                    let adjacent = right_range.base == left_range.end();
                    allocator.free(right, OTHER);
                    let grown = allocator.realloc(left, OTHER, class_count + 1).unwrap();
                    if adjacent {
                        assert_eq!(
                            (grown.base, grown.capacity),
                            (left_range.base, class_count * 2),
                            "adjacent free buddy must be absorbed without moving"
                        );
                    }
                    model.insert((left, OTHER.index()), class_count + 1);
                    allocator.free(left, OTHER);
                    model.remove(&(left, OTHER.index()));
                    allocator.free(right, OTHER);
                    if seen_layers.insert(left) {
                        touched.push(left);
                    }
                    continue;
                }
            }
            if seen_layers.insert(layer) {
                touched.push(layer);
            }
        }

        // Oracle comparison: residency, counts, and the class invariant.
        for &layer in &touched {
            let slabs = allocator.slabs(layer);
            for kind_index in 0..SlabKind::COUNT {
                let expected = model.get(&(layer, kind_index)).copied().unwrap_or(0);
                let actual = slabs.map_or(SlabRange::EMPTY, |s| s.ranges[kind_index]);
                assert_eq!(actual.count, expected, "layer {layer} kind {kind_index}");
                if expected == 0 {
                    assert!(actual.is_empty());
                } else {
                    assert_eq!(actual.capacity, size_class(expected));
                }
            }
        }

        // Geometry: disjointness via occupancy stamping, plus totals.
        for (kind_index, &kind) in SlabKind::ALL.iter().enumerate() {
            let capacity = allocator.arena_element_capacity(kind);
            let mut grid = vec![0u32; capacity as usize];
            for &layer in &touched {
                let range = allocator
                    .slabs(layer)
                    .map(|slabs| slabs.ranges[kind_index])
                    .unwrap_or_default();
                if range.is_empty() {
                    continue;
                }
                assert!(range.base % MIN_CLASS == 0);
                assert!(range.end() <= capacity);
                for cell in &mut grid[range.base as usize..range.end() as usize] {
                    *cell += 1;
                    assert!(*cell <= 1, "live ranges overlap in kind {kind:?}");
                }
            }
            let expected_live: u64 = model
                .iter()
                .filter(|&(key, _)| key.1 == kind_index)
                .map(|(_, count)| u64::from(*count))
                .sum();
            let expected_reserved: u64 = model
                .iter()
                .filter(|&(key, _)| key.1 == kind_index)
                .map(|(_, count)| u64::from(size_class(*count)))
                .sum();
            assert_eq!(allocator.live_elements(kind), expected_live);
            assert_eq!(allocator.reserved_elements(kind), expected_reserved);
        }

        // Compaction must remain optional: running it once now packs every
        // kind to exactly its summed capacities without losing anything.
        let plan = allocator.compaction_plan();
        allocator.apply_compaction(&plan);
        for (kind_index, &kind) in SlabKind::ALL.iter().enumerate() {
            let packed: u64 = allocator.reserved_elements(kind);
            assert_eq!(
                u64::from(allocator.arena_element_capacity(kind)),
                packed,
                "post-compaction footprint must equal summed capacities (kind {kind_index})"
            );
        }
    }
}
