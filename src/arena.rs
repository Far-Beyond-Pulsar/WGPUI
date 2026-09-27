use std::{
    alloc::{Layout, handle_alloc_error},
    marker::PhantomData,
    mem,
    num::NonZeroUsize,
    ops::{Deref, DerefMut},
    panic::{self, AssertUnwindSafe},
    ptr::{self, NonNull},
    sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering},
};

use crate::shared_runtime::{self, ModuleId, SharedRuntime, os};

/// Runs a value's destructor and reports whether it panicked. The panic is
/// caught inside the copy of this crate that monomorphized the destructor,
/// because a panic unwinding into another copy's frames is a foreign
/// exception to that copy's `std` and aborts the process.
type DropFn = unsafe extern "C" fn(*mut u8) -> bool;

unsafe extern "C" fn drop_value<T>(value: *mut u8) -> bool {
    panic::catch_unwind(AssertUnwindSafe(|| unsafe {
        ptr::drop_in_place(value.cast::<T>())
    }))
    .is_err()
}

#[repr(C)]
pub(crate) struct DropRecord {
    value: *mut u8,
    drop: DropFn,
    module: ModuleId,
}

/// Prefix of every chunk; the chunk's bytes follow it directly. Two words so
/// chunk data keeps the OS allocator's 16-byte alignment.
#[repr(C)]
pub(crate) struct ChunkHeader {
    next: *mut ChunkHeader,
    _reserved: usize,
}

/// The arena state itself, shared by every copy of this crate that touches
/// the arena. `#[repr(C)]` and allocated from the OS heap by the shared
/// runtime; see [`crate::shared_runtime`].
///
/// Headers are never freed, only returned to the runtime's pool, so an
/// [`ArenaBox`] can always read its arena's `generation` to detect that the
/// arena was cleared or dropped, even after the header has been reused.
#[repr(C)]
pub(crate) struct RawArena {
    generation: AtomicU64,
    cursor: *mut u8,
    limit: *mut u8,
    current_chunk: *mut ChunkHeader,
    first_chunk: *mut ChunkHeader,
    chunk_size: usize,
    chunk_count: AtomicUsize,
    records: *mut DropRecord,
    record_len: usize,
    record_capacity: usize,
    allocation_count: AtomicUsize,
    alloc_depth: u32,
    clearing: u32,
    in_use: AtomicU32,
    next_free: *mut RawArena,
    next_all: *mut RawArena,
}

/// Sum of values currently allocated in every live arena in the process.
pub(crate) fn live_allocations(runtime: &SharedRuntime) -> u64 {
    let mut total = 0;
    let mut header = runtime.arena_all_head.load(Ordering::Acquire);
    while !header.is_null() {
        // SAFETY: headers are never freed, and `next_all` is written once,
        // before the header is published to `arena_all_head`.
        let arena = unsafe { &*header };
        if arena.in_use.load(Ordering::Relaxed) != 0 {
            total += arena.allocation_count.load(Ordering::Relaxed) as u64;
        }
        header = arena.next_all;
    }
    total
}

unsafe fn chunk_data(chunk: *mut ChunkHeader) -> *mut u8 {
    unsafe { chunk.add(1).cast() }
}

fn allocate_chunk(runtime: &SharedRuntime, chunk_size: usize) -> *mut ChunkHeader {
    let size = size_of::<ChunkHeader>()
        .checked_add(chunk_size)
        .expect("arena chunk size overflow");
    let chunk = unsafe { os::alloc(size) }.cast::<ChunkHeader>();
    if chunk.is_null() {
        handle_alloc_error(Layout::from_size_align(size, 16).unwrap_or(Layout::new::<ChunkHeader>()));
    }
    unsafe {
        chunk.write(ChunkHeader {
            next: ptr::null_mut(),
            _reserved: 0,
        })
    };
    runtime.counters.live_chunks.fetch_add(1, Ordering::Relaxed);
    runtime
        .counters
        .live_chunk_bytes
        .fetch_add(size as u64, Ordering::Relaxed);
    chunk
}

fn free_chunk(runtime: &SharedRuntime, chunk: *mut ChunkHeader, chunk_size: usize) {
    unsafe { os::free(chunk.cast()) };
    runtime.counters.live_chunks.fetch_sub(1, Ordering::Relaxed);
    runtime.counters.live_chunk_bytes.fetch_sub(
        (size_of::<ChunkHeader>() + chunk_size) as u64,
        Ordering::Relaxed,
    );
}

impl RawArena {
    fn acquire(chunk_size: usize) -> NonNull<RawArena> {
        let runtime = shared_runtime::runtime();
        let pooled = {
            let _lock = runtime.lock_arena_pool();
            let head = runtime.arena_free_head.load(Ordering::Relaxed);
            if !head.is_null() {
                runtime
                    .arena_free_head
                    .store(unsafe { (*head).next_free }, Ordering::Relaxed);
            }
            head
        };
        let header = if pooled.is_null() {
            let header = unsafe { os::alloc_zeroed(size_of::<RawArena>()) }.cast::<RawArena>();
            if header.is_null() {
                handle_alloc_error(Layout::new::<RawArena>());
            }
            let mut head = runtime.arena_all_head.load(Ordering::Relaxed);
            loop {
                unsafe { (*header).next_all = head };
                match runtime.arena_all_head.compare_exchange_weak(
                    head,
                    header,
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(current) => head = current,
                }
            }
            runtime.counters.arena_headers.fetch_add(1, Ordering::Relaxed);
            header
        } else {
            pooled
        };

        let first_chunk = allocate_chunk(runtime, chunk_size);
        unsafe {
            let arena = &mut *header;
            arena.cursor = chunk_data(first_chunk);
            arena.limit = arena.cursor.add(chunk_size);
            arena.current_chunk = first_chunk;
            arena.first_chunk = first_chunk;
            arena.chunk_size = chunk_size;
            arena.chunk_count.store(1, Ordering::Relaxed);
            arena.records = ptr::null_mut();
            arena.record_len = 0;
            arena.record_capacity = 0;
            arena.allocation_count.store(0, Ordering::Relaxed);
            arena.alloc_depth = 0;
            arena.clearing = 0;
            arena.next_free = ptr::null_mut();
            arena.in_use.store(1, Ordering::Relaxed);
        }
        runtime.counters.live_arenas.fetch_add(1, Ordering::Relaxed);
        // SAFETY: checked for null above.
        unsafe { NonNull::new_unchecked(header) }
    }

    /// Free the arena's memory and return its header to the pool. Values
    /// must already have been dropped.
    unsafe fn release(header: *mut RawArena) {
        let runtime = shared_runtime::runtime();
        unsafe {
            let chunk_size = (*header).chunk_size;
            let mut chunk = (*header).first_chunk;
            while !chunk.is_null() {
                let next = (*chunk).next;
                free_chunk(runtime, chunk, chunk_size);
                chunk = next;
            }
            if !(*header).records.is_null() {
                os::free((*header).records.cast());
                runtime.counters.live_record_bytes.fetch_sub(
                    ((*header).record_capacity * size_of::<DropRecord>()) as u64,
                    Ordering::Relaxed,
                );
            }
            let arena = &mut *header;
            arena.generation.fetch_add(1, Ordering::Release);
            arena.cursor = ptr::null_mut();
            arena.limit = ptr::null_mut();
            arena.current_chunk = ptr::null_mut();
            arena.first_chunk = ptr::null_mut();
            arena.chunk_count.store(0, Ordering::Relaxed);
            arena.records = ptr::null_mut();
            arena.record_len = 0;
            arena.record_capacity = 0;
            arena.allocation_count.store(0, Ordering::Relaxed);
            arena.in_use.store(0, Ordering::Relaxed);
        }
        runtime.counters.live_arenas.fetch_sub(1, Ordering::Relaxed);
        let _lock = runtime.lock_arena_pool();
        unsafe {
            (*header).next_free = runtime.arena_free_head.load(Ordering::Relaxed);
        }
        runtime.arena_free_head.store(header, Ordering::Relaxed);
    }

    #[inline(always)]
    unsafe fn bump(arena: *mut RawArena, layout: Layout) -> *mut u8 {
        unsafe {
            if let Some(pointer) = Self::fit((*arena).cursor, (*arena).limit, layout) {
                (*arena).cursor = pointer.add(layout.size());
                return pointer;
            }
            Self::bump_into_next_chunk(arena, layout)
        }
    }

    #[inline(always)]
    fn fit(cursor: *mut u8, limit: *mut u8, layout: Layout) -> Option<*mut u8> {
        let start = (cursor as usize).checked_next_multiple_of(layout.align())?;
        let end = start.checked_add(layout.size())?;
        (end <= limit as usize).then(|| cursor.wrapping_add(start - cursor as usize))
    }

    #[cold]
    unsafe fn bump_into_next_chunk(arena: *mut RawArena, layout: Layout) -> *mut u8 {
        unsafe {
            let current = (*arena).current_chunk;
            let chunk_size = (*arena).chunk_size;
            let mut next = (*current).next;
            if next.is_null() {
                next = allocate_chunk(shared_runtime::runtime(), chunk_size);
                (*current).next = next;
                let chunk_count = (*arena).chunk_count.load(Ordering::Relaxed) + 1;
                (*arena).chunk_count.store(chunk_count, Ordering::Relaxed);
                log::trace!(
                    "increased element arena capacity to {}kb",
                    chunk_count * chunk_size / 1024,
                );
            }
            (*arena).current_chunk = next;
            (*arena).cursor = chunk_data(next);
            (*arena).limit = (*arena).cursor.add(chunk_size);
            match Self::fit((*arena).cursor, (*arena).limit, layout) {
                Some(pointer) => {
                    (*arena).cursor = pointer.add(layout.size());
                    pointer
                }
                None => panic!(
                    "Arena chunk_size of {} is too small to allocate {} bytes",
                    chunk_size,
                    layout.size()
                ),
            }
        }
    }

    #[inline(always)]
    unsafe fn push_record(arena: *mut RawArena, record: DropRecord) {
        unsafe {
            if (*arena).record_len == (*arena).record_capacity {
                Self::grow_records(arena);
            }
            (*arena).records.add((*arena).record_len).write(record);
            (*arena).record_len += 1;
        }
    }

    #[cold]
    unsafe fn grow_records(arena: *mut RawArena) {
        unsafe {
            let old_capacity = (*arena).record_capacity;
            let new_capacity = (old_capacity * 2).max(64);
            let bytes = new_capacity
                .checked_mul(size_of::<DropRecord>())
                .expect("arena record array overflow");
            let records = os::realloc((*arena).records.cast(), bytes).cast::<DropRecord>();
            if records.is_null() {
                handle_alloc_error(Layout::array::<DropRecord>(new_capacity).unwrap_or(Layout::new::<DropRecord>()));
            }
            (*arena).records = records;
            (*arena).record_capacity = new_capacity;
            shared_runtime::runtime().counters.live_record_bytes.fetch_add(
                ((new_capacity - old_capacity) * size_of::<DropRecord>()) as u64,
                Ordering::Relaxed,
            );
        }
    }

    /// Drop every value and rewind to the first chunk, keeping all chunks for
    /// reuse.
    unsafe fn clear(arena: *mut RawArena) {
        unsafe {
            assert!(
                (*arena).alloc_depth == 0,
                "Arena::clear called while a value was still being constructed in the same arena"
            );
            assert!((*arena).clearing == 0, "Arena::clear called re-entrantly");
            // Invalidate outstanding `ArenaBox`es before running destructors,
            // so a destructor can't observe a sibling that's already dropped.
            (*arena).generation.fetch_add(1, Ordering::Release);
            (*arena).clearing = 1;

            let runtime = shared_runtime::runtime();
            let mut panicked = 0usize;
            let mut orphaned = 0usize;
            for index in 0..(*arena).record_len {
                let record = (*arena).records.add(index).read();
                if runtime.module_is_loaded(record.module) {
                    if (record.drop)(record.value) {
                        panicked += 1;
                    }
                } else {
                    orphaned += 1;
                }
            }

            (*arena).record_len = 0;
            (*arena).allocation_count.store(0, Ordering::Relaxed);
            (*arena).current_chunk = (*arena).first_chunk;
            (*arena).cursor = chunk_data((*arena).first_chunk);
            (*arena).limit = (*arena).cursor.add((*arena).chunk_size);
            (*arena).clearing = 0;

            if orphaned > 0 {
                runtime
                    .counters
                    .orphaned_records
                    .fetch_add(orphaned as u64, Ordering::Relaxed);
                log::error!(
                    "skipped {orphaned} arena value destructor(s) whose code was unloaded before \
                     the arena was cleared; their heap memory is leaked. Only unload a plugin \
                     between frames, after its arenas have been cleared."
                );
            }
            if panicked > 0 {
                panic!(
                    "{panicked} arena value destructor(s) panicked during Arena::clear \
                     (the remaining values were still dropped)"
                );
            }
        }
    }
}

/// A bump allocator used for per-frame element storage. Grows in fixed-size
/// chunks as needed and is reset (not deallocated) via [`Arena::clear`],
/// reusing already-grown chunks for the next frame's allocations instead of
/// freeing and reallocating them.
///
/// The arena's state and memory live in the process-wide shared runtime
/// rather than in whichever copy of this crate created it, so the host and
/// any plugin DLL can allocate into, clear, and drop the same arena. See
/// [`crate::shared_runtime`].
pub struct Arena {
    raw: NonNull<RawArena>,
    _not_send: PhantomData<*mut ()>,
}

impl Drop for Arena {
    fn drop(&mut self) {
        let raw = self.raw.as_ptr();
        let result = panic::catch_unwind(AssertUnwindSafe(|| unsafe { RawArena::clear(raw) }));
        unsafe { RawArena::release(raw) };
        if let Err(payload) = result {
            panic::resume_unwind(payload);
        }
    }
}

impl Arena {
    /// Create a new arena whose chunks grow in increments of `chunk_size`
    /// bytes.
    pub fn new(chunk_size: usize) -> Self {
        let chunk_size = NonZeroUsize::new(chunk_size).expect("arena chunk size must be non-zero");
        Self {
            raw: RawArena::acquire(chunk_size.get()),
            _not_send: PhantomData,
        }
    }

    /// A non-owning handle to an arena owned elsewhere. Must be wrapped in
    /// `ManuallyDrop`, since dropping an `Arena` releases it.
    pub(crate) unsafe fn borrow_raw(raw: NonNull<RawArena>) -> Self {
        Self {
            raw,
            _not_send: PhantomData,
        }
    }

    pub(crate) fn raw(&self) -> NonNull<RawArena> {
        self.raw
    }

    /// Total bytes currently reserved across all chunks this arena has
    /// grown to (its high-water mark), not the bytes currently in use.
    pub fn capacity(&self) -> usize {
        let arena = unsafe { self.raw.as_ref() };
        arena.chunk_count.load(Ordering::Relaxed) * arena.chunk_size
    }

    /// Number of values allocated since the arena was last cleared.
    pub fn allocation_count(&self) -> usize {
        unsafe { self.raw.as_ref() }
            .allocation_count
            .load(Ordering::Relaxed)
    }

    /// Drop every value allocated in this arena and reset the bump pointer
    /// back to the start of its first chunk, so already-grown chunks are
    /// reused by the next round of allocations instead of being freed.
    pub fn clear(&mut self) {
        unsafe { RawArena::clear(self.raw.as_ptr()) }
    }

    /// Allocate `f()`'s result in this arena, growing a new chunk if the
    /// current one is full.
    #[inline(always)]
    pub fn alloc<T>(&mut self, f: impl FnOnce() -> T) -> ArenaBox<T> {
        struct ConstructionGuard(*mut RawArena);

        impl Drop for ConstructionGuard {
            #[inline(always)]
            fn drop(&mut self) {
                unsafe { (*self.0).alloc_depth -= 1 };
            }
        }

        let raw = self.raw.as_ptr();
        unsafe {
            assert!(
                (*raw).clearing == 0,
                "attempted to allocate in an Arena while it is being cleared"
            );
            let value = RawArena::bump(raw, Layout::new::<T>()).cast::<T>();

            // `f` may itself allocate in this arena (the bump pointer has
            // already moved past `value`, so that's safe), but it must not
            // clear it; the depth counter lets `clear` catch that.
            (*raw).alloc_depth += 1;
            let guard = ConstructionGuard(raw);
            ptr::write(value, f());
            drop(guard);

            if mem::needs_drop::<T>() {
                RawArena::push_record(
                    raw,
                    DropRecord {
                        value: value.cast(),
                        drop: drop_value::<T>,
                        module: shared_runtime::current_module(),
                    },
                );
            }
            let count = (*raw).allocation_count.load(Ordering::Relaxed);
            (*raw).allocation_count.store(count + 1, Ordering::Relaxed);

            ArenaBox {
                ptr: value,
                arena: self.raw,
                generation: (*raw).generation.load(Ordering::Relaxed),
            }
        }
    }
}

/// An owning-by-arena pointer to a value in an [`Arena`]. Holds the arena's
/// generation at allocation time instead of a reference count, so it carries
/// no heap allocation of its own and is valid across copies of this crate.
pub struct ArenaBox<T: ?Sized> {
    ptr: *mut T,
    arena: NonNull<RawArena>,
    generation: u64,
}

impl<T: ?Sized> ArenaBox<T> {
    #[inline(always)]
    pub fn map<U: ?Sized>(mut self, f: impl FnOnce(&mut T) -> &mut U) -> ArenaBox<U> {
        ArenaBox {
            ptr: f(&mut self),
            arena: self.arena,
            generation: self.generation,
        }
    }

    #[track_caller]
    #[inline(always)]
    fn validate(&self) {
        // SAFETY: arena headers are never freed (see `RawArena`).
        let current = unsafe { self.arena.as_ref() }
            .generation
            .load(Ordering::Acquire);
        assert!(
            current == self.generation,
            "attempted to dereference an ArenaRef after its Arena was cleared"
        );
    }
}

impl<T: ?Sized> Deref for ArenaBox<T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.validate();
        unsafe { &*self.ptr }
    }
}

impl<T: ?Sized> DerefMut for ArenaBox<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.validate();
        unsafe { &mut *self.ptr }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        cell::{Cell, RefCell},
        rc::Rc,
        sync::{
            Arc, Barrier,
            atomic::{AtomicUsize, Ordering},
        },
    };

    use super::*;

    #[test]
    fn test_arena() {
        let mut arena = Arena::new(1024);
        let a = arena.alloc(|| 1u64);
        let b = arena.alloc(|| 2u32);
        let c = arena.alloc(|| 3u16);
        let d = arena.alloc(|| 4u8);
        assert_eq!(*a, 1);
        assert_eq!(*b, 2);
        assert_eq!(*c, 3);
        assert_eq!(*d, 4);

        arena.clear();
        let a = arena.alloc(|| 5u64);
        let b = arena.alloc(|| 6u32);
        let c = arena.alloc(|| 7u16);
        let d = arena.alloc(|| 8u8);
        assert_eq!(*a, 5);
        assert_eq!(*b, 6);
        assert_eq!(*c, 7);
        assert_eq!(*d, 8);

        // Ensure drop gets called.
        let dropped = Rc::new(Cell::new(false));
        struct DropGuard(Rc<Cell<bool>>);
        impl Drop for DropGuard {
            fn drop(&mut self) {
                self.0.set(true);
            }
        }
        arena.alloc(|| DropGuard(dropped.clone()));
        arena.clear();
        assert!(dropped.get());
    }

    #[test]
    fn test_arena_grow() {
        let mut arena = Arena::new(8);
        arena.alloc(|| 1u64);
        arena.alloc(|| 2u64);

        assert_eq!(arena.capacity(), 16);

        arena.alloc(|| 3u32);
        arena.alloc(|| 4u32);

        assert_eq!(arena.capacity(), 24);
    }

    #[test]
    fn test_arena_alignment() {
        let mut arena = Arena::new(256);
        let x1 = arena.alloc(|| 1u8);
        let x2 = arena.alloc(|| 2u16);
        let x3 = arena.alloc(|| 3u32);
        let x4 = arena.alloc(|| 4u64);
        let x5 = arena.alloc(|| 5u64);

        assert_eq!(*x1, 1);
        assert_eq!(*x2, 2);
        assert_eq!(*x3, 3);
        assert_eq!(*x4, 4);
        assert_eq!(*x5, 5);

        assert_eq!(x1.ptr.align_offset(std::mem::align_of_val(&*x1)), 0);
        assert_eq!(x2.ptr.align_offset(std::mem::align_of_val(&*x2)), 0);
    }

    #[test]
    #[should_panic(expected = "attempted to dereference an ArenaRef after its Arena was cleared")]
    fn test_arena_use_after_clear() {
        let mut arena = Arena::new(16);
        let value = arena.alloc(|| 1u64);

        arena.clear();
        let _read_value = *value;
    }

    #[test]
    #[should_panic(expected = "attempted to dereference an ArenaRef after its Arena was cleared")]
    fn arena_box_outliving_its_arena_is_detected() {
        let mut arena = Arena::new(16);
        let value = arena.alloc(|| 1u64);
        drop(arena);
        // The header was returned to the pool and may even be reused by
        // another arena already; the generation check still catches it.
        let _other = Arena::new(16);
        let _read_value = *value;
    }

    #[test]
    fn over_aligned_values_are_aligned() {
        #[repr(align(64))]
        struct CacheLine(u8);
        #[repr(align(4096))]
        struct Page(u8);

        let mut arena = Arena::new(16 * 1024);
        for index in 0..64u8 {
            let line = arena.alloc(|| CacheLine(index));
            assert_eq!(line.ptr as usize % 64, 0);
            assert_eq!(line.0, index);
            let byte = arena.alloc(|| index);
            assert_eq!(*byte, index);
        }
        let page = arena.alloc(|| Page(7));
        assert_eq!(page.ptr as usize % 4096, 0);
        assert_eq!(page.0, 7);
    }

    #[test]
    fn zero_sized_values_are_supported_and_dropped() {
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        struct Zst;
        impl Drop for Zst {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::Relaxed);
            }
        }
        let mut arena = Arena::new(8);
        for _ in 0..1000 {
            arena.alloc(|| Zst);
        }
        assert_eq!(arena.allocation_count(), 1000);
        arena.clear();
        assert_eq!(DROPS.load(Ordering::Relaxed), 1000);
        assert_eq!(arena.capacity(), 8);
    }

    #[test]
    #[should_panic(expected = "is too small to allocate")]
    fn oversized_values_panic() {
        let mut arena = Arena::new(8);
        arena.alloc(|| [0u8; 9]);
    }

    #[test]
    fn values_are_dropped_in_allocation_order() {
        let order = Rc::new(RefCell::new(Vec::new()));
        struct Tracked(usize, Rc<RefCell<Vec<usize>>>);
        impl Drop for Tracked {
            fn drop(&mut self) {
                self.1.borrow_mut().push(self.0);
            }
        }
        let mut arena = Arena::new(64);
        for index in 0..100 {
            arena.alloc(|| Tracked(index, order.clone()));
        }
        arena.clear();
        assert_eq!(*order.borrow(), (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn panicking_destructor_does_not_stop_the_others() {
        let dropped = Rc::new(Cell::new(0));
        struct Probe(bool, Rc<Cell<usize>>);
        impl Drop for Probe {
            fn drop(&mut self) {
                self.1.set(self.1.get() + 1);
                if self.0 {
                    panic!("probe destructor panicked on purpose");
                }
            }
        }
        let mut arena = Arena::new(256);
        for index in 0..10 {
            arena.alloc(|| Probe(index == 3 || index == 7, dropped.clone()));
        }
        let result = panic::catch_unwind(AssertUnwindSafe(|| arena.clear()));
        let message = result
            .expect_err("clear should report the destructor panics")
            .downcast::<String>()
            .expect("panic payload should be a String");
        assert!(message.contains("2 arena value destructor(s) panicked"), "{message}");
        assert_eq!(dropped.get(), 10);

        // The arena is fully reset and usable afterwards.
        assert_eq!(arena.allocation_count(), 0);
        let value = arena.alloc(|| 5u32);
        assert_eq!(*value, 5);
    }

    #[test]
    fn allocating_while_constructing_is_allowed() {
        let mut arena = Arena::new(1024);
        let raw = arena.raw();
        let outer = arena.alloc(|| {
            let mut inner_handle = mem::ManuallyDrop::new(unsafe { Arena::borrow_raw(raw) });
            let inner = inner_handle.alloc(|| String::from("inner"));
            (inner, 42u32)
        });
        assert_eq!(*outer.0, "inner");
        assert_eq!(outer.1, 42);
        assert_eq!(arena.allocation_count(), 2);
        arena.clear();
    }

    #[test]
    #[should_panic(expected = "while a value was still being constructed")]
    fn clearing_while_constructing_panics() {
        let mut arena = Arena::new(1024);
        let raw = arena.raw();
        arena.alloc(|| {
            let mut inner_handle = mem::ManuallyDrop::new(unsafe { Arena::borrow_raw(raw) });
            inner_handle.clear();
        });
    }

    #[test]
    fn values_from_an_unloaded_module_are_orphaned_not_dropped() {
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        struct Probe;
        impl Drop for Probe {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::Relaxed);
            }
        }
        let mut arena = Arena::new(1024);
        arena.alloc(|| Probe);

        // Simulate a record written by a module that has since been unloaded:
        // a generation that can never be current for this slot again.
        let current = shared_runtime::current_module();
        let unloaded_module = ModuleId {
            slot: current.slot,
            generation: current.generation.wrapping_sub(1),
        };
        unsafe {
            let raw = arena.raw().as_ptr();
            let value = RawArena::bump(raw, Layout::new::<Probe>()).cast::<Probe>();
            value.write(Probe);
            RawArena::push_record(
                raw,
                DropRecord {
                    value: value.cast(),
                    drop: drop_value::<Probe>,
                    module: unloaded_module,
                },
            );
        }

        let orphans_before = shared_runtime::stats().orphaned_records;
        arena.clear();
        assert_eq!(DROPS.load(Ordering::Relaxed), 1, "only the loaded module's value runs its destructor");
        assert!(shared_runtime::stats().orphaned_records > orphans_before);
    }

    #[test]
    fn clearing_many_frames_does_not_grow() {
        let mut arena = Arena::new(4096);
        for frame in 0..2_000 {
            for index in 0..300 {
                arena.alloc(|| (frame, index, String::from("owned heap data")));
            }
            arena.clear();
        }
        let capacity = arena.capacity();
        for frame in 0..2_000 {
            for index in 0..300 {
                arena.alloc(|| (frame, index, String::from("owned heap data")));
            }
            arena.clear();
        }
        assert_eq!(arena.capacity(), capacity, "steady-state frames must reuse chunks");
    }

    #[test]
    fn arena_headers_are_pooled() {
        // Other tests run concurrently, so only check that repeated
        // create/drop cycles on this thread don't each allocate a header.
        let before = shared_runtime::stats().arena_headers;
        for _ in 0..10_000 {
            let mut arena = Arena::new(64);
            arena.alloc(|| String::from("x"));
        }
        let after = shared_runtime::stats().arena_headers;
        assert!(
            after - before < 64,
            "10k arena lifetimes allocated {} headers",
            after - before
        );
    }

    #[test]
    fn concurrent_arenas_are_independent() {
        const THREADS: usize = 16;
        let barrier = Arc::new(Barrier::new(THREADS));
        let handles: Vec<_> = (0..THREADS)
            .map(|thread| {
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    let mut arena = Arena::new(512);
                    barrier.wait();
                    for frame in 0..500 {
                        let boxes: Vec<_> = (0..50)
                            .map(|index| arena.alloc(|| vec![thread, frame, index]))
                            .collect();
                        for (index, value) in boxes.iter().enumerate() {
                            assert_eq!(**value, vec![thread, frame, index]);
                        }
                        assert_eq!(arena.allocation_count(), 50);
                        arena.clear();
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
    }
}
