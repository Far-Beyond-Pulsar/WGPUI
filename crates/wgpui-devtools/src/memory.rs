//! Capture-only accounting for framework-owned CPU allocations.
//!
//! The registry records typed ownership metadata and byte counts. It never
//! stores or exposes an address, and every mutating operation is a no-op until
//! a capture is active.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};

/// A framework-owned allocation category visible to a memory capture.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum AllocationCategory {
    /// Retained layer slab reservations and their allocator bookkeeping.
    RetainedSlab,
    /// Primitive-kind arena storage.
    PrimitiveArena,
    /// The per-frame description tree and its owned buffers.
    DescriptionBuffer,
    /// Retained layout-tree storage.
    LayoutStorage,
    /// Registered input/event dispatch records.
    EventRegistrations,
    /// CPU trace recorder buffers.
    TraceBuffer,
    /// Temporary storage used while assembling a frame.
    FrameScratch,
}

impl AllocationCategory {
    /// Categories in stable presentation order.
    pub const ALL: [Self; 7] = [
        Self::RetainedSlab,
        Self::PrimitiveArena,
        Self::DescriptionBuffer,
        Self::LayoutStorage,
        Self::EventRegistrations,
        Self::TraceBuffer,
        Self::FrameScratch,
    ];

    /// Stable name for presentation and serialization adapters.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RetainedSlab => "retained_slab",
            Self::PrimitiveArena => "primitive_arena",
            Self::DescriptionBuffer => "description_buffer",
            Self::LayoutStorage => "layout_storage",
            Self::EventRegistrations => "event_registrations",
            Self::TraceBuffer => "trace_buffer",
            Self::FrameScratch => "frame_scratch",
        }
    }
}

/// A stable, pointer-free identifier for one allocation record.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AllocationId(u64);

impl AllocationId {
    /// The identifier value, for capture formats that use integer IDs.
    pub const fn as_raw(self) -> u64 {
        self.0
    }
}

/// An immutable description of one live owned allocation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AllocationEntry {
    /// Stable registry-local identity. This is not a memory address.
    pub id: AllocationId,
    /// Typed ownership category.
    pub category: AllocationCategory,
    /// Stable owner label supplied by the framework adapter.
    pub owner: &'static str,
    /// Bytes currently occupied by live values.
    pub live_bytes: u64,
    /// Bytes reserved by the owning container.
    pub capacity_bytes: u64,
    /// Largest live byte count observed for this record during the capture.
    pub high_water_bytes: u64,
    /// Number of allocation events represented by this record.
    pub allocation_count: u64,
}

/// Aggregated accounting for one category.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AllocationCategorySnapshot {
    /// Category being summarized.
    pub category: AllocationCategory,
    /// Number of currently live records in this category.
    pub active_allocations: u64,
    /// Bytes currently occupied by live values.
    pub live_bytes: u64,
    /// Bytes currently reserved by live records.
    pub capacity_bytes: u64,
    /// Largest aggregate live byte count observed during the capture.
    pub high_water_bytes: u64,
    /// Number of allocation and reallocation events in this category.
    pub allocation_count: u64,
}

/// A coherent, immutable memory view captured at one point in time.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AllocationSnapshot {
    /// Per-category totals in [`AllocationCategory::ALL`] order.
    pub categories: Vec<AllocationCategorySnapshot>,
    /// Live owned allocations, sorted by category and then identifier.
    pub allocations: Vec<AllocationEntry>,
}

impl AllocationSnapshot {
    /// Summary for a category.
    pub fn category(&self, category: AllocationCategory) -> Option<&AllocationCategorySnapshot> {
        self.categories
            .iter()
            .find(|summary| summary.category == category)
    }

    /// Total live bytes across all categories.
    pub fn live_bytes(&self) -> u64 {
        self.categories
            .iter()
            .map(|summary| summary.live_bytes)
            .fold(0, u64::saturating_add)
    }

    /// Total reserved bytes across all categories.
    pub fn capacity_bytes(&self) -> u64 {
        self.categories
            .iter()
            .map(|summary| summary.capacity_bytes)
            .fold(0, u64::saturating_add)
    }

    /// Sum of the largest aggregate live byte count observed per category.
    pub fn high_water_bytes(&self) -> u64 {
        self.categories
            .iter()
            .map(|summary| summary.high_water_bytes)
            .fold(0, u64::saturating_add)
    }

    /// Total allocation and reallocation events across all categories.
    pub fn allocation_count(&self) -> u64 {
        self.categories
            .iter()
            .map(|summary| summary.allocation_count)
            .fold(0, u64::saturating_add)
    }
}

#[derive(Debug)]
struct AllocationRecord {
    category: AllocationCategory,
    owner: &'static str,
    live_bytes: u64,
    capacity_bytes: u64,
    high_water_bytes: u64,
    allocation_count: u64,
}

#[derive(Debug, Default)]
struct CategoryHistory {
    high_water_bytes: u64,
    allocation_count: u64,
}

#[derive(Debug, Default)]
struct RegistryState {
    records: BTreeMap<AllocationId, AllocationRecord>,
    history: BTreeMap<AllocationCategory, CategoryHistory>,
}

/// A capture-only, thread-safe registry for framework-owned allocations.
///
/// `new` and `default` create a disabled registry. `begin_capture` clears the
/// prior capture and enables recording; `end_capture` freezes and returns one
/// snapshot, then disables recording. Updates made while disabled do not
/// allocate, acquire the mutex, or leave state behind.
#[derive(Debug, Default)]
pub struct AllocationRegistry {
    enabled: AtomicBool,
    next_id: AtomicU64,
    state: Mutex<RegistryState>,
}

impl AllocationRegistry {
    /// Creates a disabled registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Starts a fresh capture and discards the previous capture state.
    pub fn begin_capture(&self) {
        let mut state = lock_state(&self.state);
        *state = RegistryState::default();
        self.enabled.store(true, Ordering::Release);
    }

    /// Stops recording and returns the immutable capture snapshot.
    pub fn end_capture(&self) -> AllocationSnapshot {
        let state = lock_state(&self.state);
        let snapshot = snapshot_state(&state);
        self.enabled.store(false, Ordering::Release);
        snapshot
    }

    /// Whether this registry is currently collecting allocation data.
    pub fn is_capturing(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }

    /// Registers one owned allocation and returns its pointer-free ID.
    ///
    /// `capacity_bytes` is normalized upward when a caller reports fewer
    /// reserved bytes than live bytes, keeping the snapshot invariant that
    /// live storage cannot exceed capacity.
    pub fn register(
        &self,
        category: AllocationCategory,
        owner: &'static str,
        live_bytes: u64,
        capacity_bytes: u64,
    ) -> Option<AllocationId> {
        if !self.enabled.load(Ordering::Acquire) {
            return None;
        }
        let mut state = lock_state(&self.state);
        if !self.enabled.load(Ordering::Acquire) {
            return None;
        }

        let id = AllocationId(self.next_id.fetch_add(1, Ordering::Relaxed));
        let capacity_bytes = capacity_bytes.max(live_bytes);
        state.records.insert(
            id,
            AllocationRecord {
                category,
                owner,
                live_bytes,
                capacity_bytes,
                high_water_bytes: live_bytes,
                allocation_count: 1,
            },
        );
        let history = state.history.entry(category).or_default();
        history.allocation_count = history.allocation_count.saturating_add(1);
        observe_category_high_water(&mut state, category);
        Some(id)
    }

    /// Updates byte usage without counting a new allocation event.
    pub fn update(&self, id: AllocationId, live_bytes: u64, capacity_bytes: u64) -> bool {
        self.with_record(id, |state, record| {
            let capacity_bytes = capacity_bytes.max(live_bytes);
            record.live_bytes = live_bytes;
            record.capacity_bytes = capacity_bytes;
            record.high_water_bytes = record.high_water_bytes.max(live_bytes);
            observe_category_high_water(state, record.category);
            true
        })
        .unwrap_or(false)
    }

    /// Records a reallocation and updates its byte usage.
    pub fn reallocate(&self, id: AllocationId, live_bytes: u64, capacity_bytes: u64) -> bool {
        self.with_record(id, |state, record| {
            let capacity_bytes = capacity_bytes.max(live_bytes);
            record.live_bytes = live_bytes;
            record.capacity_bytes = capacity_bytes;
            record.high_water_bytes = record.high_water_bytes.max(live_bytes);
            record.allocation_count = record.allocation_count.saturating_add(1);
            let category = record.category;
            let history = state.history.entry(category).or_default();
            history.allocation_count = history.allocation_count.saturating_add(1);
            observe_category_high_water(state, category);
            true
        })
        .unwrap_or(false)
    }

    /// Removes a live allocation from the capture.
    pub fn release(&self, id: AllocationId) -> bool {
        if !self.enabled.load(Ordering::Acquire) {
            return false;
        }
        let mut state = lock_state(&self.state);
        if !self.enabled.load(Ordering::Acquire) {
            return false;
        }
        let Some(record) = state.records.remove(&id) else {
            return false;
        };
        observe_category_high_water(&mut state, record.category);
        true
    }

    /// Returns a coherent snapshot while recording remains active.
    pub fn snapshot(&self) -> AllocationSnapshot {
        if !self.enabled.load(Ordering::Acquire) {
            return AllocationSnapshot::default();
        }
        let state = lock_state(&self.state);
        if !self.enabled.load(Ordering::Acquire) {
            return AllocationSnapshot::default();
        }
        snapshot_state(&state)
    }

    fn with_record<T>(
        &self,
        id: AllocationId,
        operation: impl FnOnce(&mut RegistryState, &mut AllocationRecord) -> T,
    ) -> Option<T> {
        if !self.enabled.load(Ordering::Acquire) {
            return None;
        }
        let mut state = lock_state(&self.state);
        if !self.enabled.load(Ordering::Acquire) {
            return None;
        }
        let mut record = state.records.remove(&id)?;
        let result = operation(&mut state, &mut record);
        let category = record.category;
        state.records.insert(id, record);
        observe_category_high_water(&mut state, category);
        Some(result)
    }
}

fn lock_state(state: &Mutex<RegistryState>) -> MutexGuard<'_, RegistryState> {
    state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn observe_category_high_water(state: &mut RegistryState, category: AllocationCategory) {
    let live_bytes = state
        .records
        .values()
        .filter(|record| record.category == category)
        .map(|record| record.live_bytes)
        .sum();
    let history = state.history.entry(category).or_default();
    history.high_water_bytes = history.high_water_bytes.max(live_bytes);
}

fn snapshot_state(state: &RegistryState) -> AllocationSnapshot {
    let mut categories = Vec::new();
    for category in AllocationCategory::ALL {
        let records = state
            .records
            .values()
            .filter(|record| record.category == category);
        let mut active_allocations: u64 = 0;
        let mut live_bytes: u64 = 0;
        let mut capacity_bytes: u64 = 0;
        for record in records {
            active_allocations += 1;
            live_bytes = live_bytes.saturating_add(record.live_bytes);
            capacity_bytes = capacity_bytes.saturating_add(record.capacity_bytes);
        }
        let history = state.history.get(&category);
        let allocation_count = history.map_or(0, |history| history.allocation_count);
        let high_water_bytes = history.map_or(0, |history| history.high_water_bytes);
        categories.push(AllocationCategorySnapshot {
            category,
            active_allocations,
            live_bytes,
            capacity_bytes,
            high_water_bytes,
            allocation_count,
        });
    }

    let mut allocations = state
        .records
        .iter()
        .map(|(id, record)| AllocationEntry {
            id: *id,
            category: record.category,
            owner: record.owner,
            live_bytes: record.live_bytes,
            capacity_bytes: record.capacity_bytes,
            high_water_bytes: record.high_water_bytes,
            allocation_count: record.allocation_count,
        })
        .collect::<Vec<_>>();
    allocations.sort_by_key(|entry| (entry.category, entry.id));
    AllocationSnapshot {
        categories,
        allocations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn disabled_registry_does_not_record_or_allocate_ids() {
        let registry = AllocationRegistry::new();
        assert!(!registry.is_capturing());
        assert_eq!(
            registry.register(AllocationCategory::TraceBuffer, "test", 4, 8),
            None
        );
        assert_eq!(registry.snapshot(), AllocationSnapshot::default());

        registry.begin_capture();
        let id = registry
            .register(AllocationCategory::TraceBuffer, "test", 4, 8)
            .expect("capture is enabled");
        assert_eq!(id.as_raw(), 0);
        let snapshot = registry.end_capture();
        assert_eq!(snapshot.live_bytes(), 4);
        assert!(!registry.is_capturing());
        assert_eq!(registry.snapshot(), AllocationSnapshot::default());
    }

    #[test]
    fn snapshot_reports_categories_owners_and_high_water_bytes() {
        let registry = AllocationRegistry::new();
        registry.begin_capture();
        assert_eq!(
            registry.snapshot().categories.len(),
            AllocationCategory::ALL.len()
        );
        let slab = registry
            .register(AllocationCategory::RetainedSlab, "scene/slabs", 10, 16)
            .expect("capture is enabled");
        let arena = registry
            .register(AllocationCategory::PrimitiveArena, "scene/quads", 20, 32)
            .expect("capture is enabled");

        assert!(registry.update(slab, 14, 24));
        assert!(registry.reallocate(arena, 30, 48));
        let snapshot = registry.snapshot();

        let slab_category = snapshot
            .category(AllocationCategory::RetainedSlab)
            .expect("slab category is present");
        assert_eq!(slab_category.live_bytes, 14);
        assert_eq!(slab_category.capacity_bytes, 24);
        assert_eq!(slab_category.high_water_bytes, 14);
        assert_eq!(slab_category.allocation_count, 1);

        let arena_category = snapshot
            .category(AllocationCategory::PrimitiveArena)
            .expect("arena category is present");
        assert_eq!(arena_category.live_bytes, 30);
        assert_eq!(arena_category.capacity_bytes, 48);
        assert_eq!(arena_category.high_water_bytes, 30);
        assert_eq!(arena_category.allocation_count, 2);
        assert_eq!(snapshot.categories.len(), AllocationCategory::ALL.len());
        assert_eq!(snapshot.allocations[0].owner, "scene/slabs");
        assert_eq!(snapshot.allocations[1].owner, "scene/quads");
    }

    #[test]
    fn snapshots_are_consistent_while_updates_are_concurrent() {
        let registry = Arc::new(AllocationRegistry::new());
        registry.begin_capture();
        let id = registry
            .register(AllocationCategory::FrameScratch, "frame", 1, 1)
            .expect("capture is enabled");
        let writer_registry = Arc::clone(&registry);
        let writer = thread::spawn(move || {
            for byte_count in 1..=512 {
                assert!(writer_registry.update(id, byte_count, byte_count));
            }
        });

        for _ in 0..128 {
            let snapshot = registry.snapshot();
            let entry = snapshot.allocations.first().expect("record is live");
            assert!(entry.live_bytes <= entry.capacity_bytes);
            assert!(entry.high_water_bytes >= entry.live_bytes);
            let category = snapshot
                .category(AllocationCategory::FrameScratch)
                .expect("category is present");
            assert_eq!(category.live_bytes, entry.live_bytes);
            assert_eq!(category.capacity_bytes, entry.capacity_bytes);
            assert!(category.high_water_bytes >= category.live_bytes);
        }
        writer.join().expect("writer thread completed");
        let snapshot = registry.end_capture();
        assert_eq!(snapshot.live_bytes(), 512);
        assert_eq!(snapshot.capacity_bytes(), 512);
    }

    #[test]
    fn releasing_a_record_keeps_category_history_but_not_live_bytes() {
        let registry = AllocationRegistry::new();
        registry.begin_capture();
        let id = registry
            .register(AllocationCategory::LayoutStorage, "layout", 8, 16)
            .expect("capture is enabled");
        assert!(registry.release(id));
        let snapshot = registry.snapshot();
        let category = snapshot
            .category(AllocationCategory::LayoutStorage)
            .expect("history is retained");
        assert_eq!(category.active_allocations, 0);
        assert_eq!(category.live_bytes, 0);
        assert_eq!(category.capacity_bytes, 0);
        assert_eq!(category.high_water_bytes, 8);
        assert_eq!(category.allocation_count, 1);
        assert!(snapshot.allocations.is_empty());
    }
}
