//! Process-wide state shared by every statically-linked copy of this crate.
//!
//! This crate is an rlib, so the host executable and every plugin DLL each
//! link their own copy of it. Anything stored in a `static` or a
//! `thread_local!` is therefore duplicated once per binary: a plugin reading
//! "the current element arena" from its own copy of a thread-local sees a
//! value the host never set (the `element arena not active` crash), and any
//! heap memory one copy parks in its own statics is leaked when that DLL is
//! unloaded.
//!
//! The fix is a single [`SharedRuntime`] per process that every copy finds at
//! runtime through an OS-level rendezvous (a named file mapping keyed by
//! process id on Windows, a pid-guarded environment variable elsewhere). The
//! runtime only contains `#[repr(C)]` data and is allocated from the OS heap,
//! never from any copy's `#[global_allocator]`, and it holds no pointer into
//! any copy's code or statics, so the copy that happened to create it can be
//! unloaded without invalidating it. Per-thread "ambient" state that used to
//! live in `thread_local!`s lives in its lock-free [`ThreadTable`], keyed by
//! the OS thread id, which is the same number in every copy (unlike
//! `std::thread::ThreadId`, which comes from a counter inside each copy's own
//! `std`).
//!
//! Every copy attaches itself to the runtime's module table the first time
//! it allocates arena memory and detaches from a destructor that runs when
//! its image is unloaded. Arena values remember which module's code has to
//! drop them, so values whose module has been unloaded are skipped (and
//! counted in [`SharedRuntimeStats::orphaned_records`]) instead of calling
//! into unmapped code.

use std::{
    ptr,
    sync::{
        OnceLock,
        atomic::{AtomicPtr, AtomicU32, AtomicU64, Ordering},
    },
};

use crate::arena::RawArena;

/// Version of the shared-runtime ABI. Copies of this crate built against
/// different versions refuse to share a runtime rather than misread it.
pub const ABI_VERSION: u32 = 1;

const MAGIC: u64 = u64::from_le_bytes(*b"GPUI-SRT");

pub(crate) const THREAD_SLOTS: usize = 512;
const MODULE_SLOTS: usize = 256;

const EMPTY_THREAD: u64 = 0;
const RELEASED_THREAD: u64 = u64::MAX;

/// Folded into the ABI check so that a layout change which forgot to bump
/// [`ABI_VERSION`] is still detected at rendezvous time.
const LAYOUT_FINGERPRINT: u64 = {
    let parts = [
        ABI_VERSION as usize,
        size_of::<SharedRuntime>(),
        align_of::<SharedRuntime>(),
        size_of::<RawArena>(),
        align_of::<RawArena>(),
        size_of::<crate::arena::DropRecord>(),
        size_of::<crate::arena::ChunkHeader>(),
        size_of::<ThreadSlot<ThreadAmbient>>(),
        size_of::<ModuleSlot>(),
        THREAD_SLOTS,
        MODULE_SLOTS,
    ];
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let mut index = 0;
    while index < parts.len() {
        hash ^= parts[index] as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        index += 1;
    }
    hash
};

/// A value stored per thread in a [`ThreadTable`]. `EMPTY` must be the
/// all-zero bit pattern, because the shared runtime's table is created by
/// zero-filling OS memory rather than by running constructors.
pub(crate) trait ThreadValue {
    const EMPTY: Self;
}

#[repr(C, align(64))]
pub(crate) struct ThreadSlot<V> {
    owner: AtomicU64,
    value: V,
}

/// A fixed-capacity, lock-free map from OS thread id to `V`, used in place of
/// `thread_local!`.
///
/// Each slot is only ever read or written by the thread that claimed it, so
/// the only cross-thread interaction is the compare-and-swap that claims a
/// slot. Slots are padded to a cache line so neighbouring threads never
/// false-share. Released slots become tombstones rather than empty so that a
/// probe chain passing through them stays intact.
#[repr(C)]
pub(crate) struct ThreadTable<V, const N: usize> {
    slots: [ThreadSlot<V>; N],
}

impl<V: ThreadValue, const N: usize> ThreadTable<V, N> {
    const MASK: usize = {
        assert!(N.is_power_of_two());
        N - 1
    };

    #[allow(clippy::new_without_default)]
    pub(crate) const fn new() -> Self {
        Self {
            slots: [const {
                ThreadSlot {
                    owner: AtomicU64::new(EMPTY_THREAD),
                    value: V::EMPTY,
                }
            }; N],
        }
    }

    fn probe_start(thread_key: u64) -> usize {
        // Fibonacci hashing: OS thread ids are small and often share low bits
        // (Windows ids are multiples of 4), so spread them before masking.
        (thread_key.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 32) as usize & Self::MASK
    }

    fn find_index(&self, thread_key: u64) -> Option<usize> {
        let mut index = Self::probe_start(thread_key);
        for _ in 0..N {
            let owner = self.slots[index].owner.load(Ordering::Acquire);
            if owner == thread_key {
                return Some(index);
            }
            if owner == EMPTY_THREAD {
                return None;
            }
            index = (index + 1) & Self::MASK;
        }
        None
    }

    /// The calling thread's value, if it has claimed a slot.
    pub(crate) fn find(&self, thread_key: u64) -> Option<&V> {
        self.find_index(thread_key).map(|index| &self.slots[index].value)
    }

    /// The calling thread's value, claiming a slot for it if it has none.
    pub(crate) fn find_or_claim(&self, thread_key: u64) -> &V {
        if let Some(value) = self.find(thread_key) {
            return value;
        }
        let mut index = Self::probe_start(thread_key);
        for _ in 0..N {
            let slot = &self.slots[index];
            let owner = slot.owner.load(Ordering::Relaxed);
            if (owner == EMPTY_THREAD || owner == RELEASED_THREAD)
                && slot
                    .owner
                    .compare_exchange(owner, thread_key, Ordering::AcqRel, Ordering::Relaxed)
                    .is_ok()
            {
                return &slot.value;
            }
            index = (index + 1) & Self::MASK;
        }
        panic!(
            "shared runtime thread table is full: more than {N} threads hold ambient UI state at once"
        );
    }

    /// Give the calling thread's slot back. The caller must have reset the
    /// value to its empty state first, since the next thread to claim the
    /// slot inherits it as-is.
    pub(crate) fn release(&self, thread_key: u64) {
        if let Some(index) = self.find_index(thread_key) {
            self.slots[index]
                .owner
                .store(RELEASED_THREAD, Ordering::Release);
        }
    }

    pub(crate) fn claimed_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| {
                let owner = slot.owner.load(Ordering::Relaxed);
                owner != EMPTY_THREAD && owner != RELEASED_THREAD
            })
            .count()
    }
}

/// Per-thread state that must be visible to every copy of this crate.
#[repr(C)]
pub(crate) struct ThreadAmbient {
    /// The arena `AnyElement::new` allocates into, set by `ElementArenaScope`.
    pub(crate) element_arena: AtomicPtr<RawArena>,
}

impl ThreadValue for ThreadAmbient {
    const EMPTY: Self = Self {
        element_arena: AtomicPtr::new(ptr::null_mut()),
    };
}

/// Identifies the copy of this crate whose code allocated an arena value, and
/// therefore whose code must run its destructor.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ModuleId {
    pub(crate) slot: u32,
    pub(crate) generation: u32,
}

#[repr(C)]
struct ModuleSlot {
    attached: AtomicU32,
    /// Bumped on detach, so `ModuleId`s handed out before an unload stop
    /// matching even after the slot is reused by a later load.
    generation: AtomicU32,
}

#[repr(C)]
pub(crate) struct SharedCounters {
    pub(crate) live_arenas: AtomicU64,
    pub(crate) arena_headers: AtomicU64,
    pub(crate) live_chunks: AtomicU64,
    pub(crate) live_chunk_bytes: AtomicU64,
    pub(crate) live_record_bytes: AtomicU64,
    pub(crate) orphaned_records: AtomicU64,
    pub(crate) attached_modules: AtomicU64,
    pub(crate) total_module_attaches: AtomicU64,
}

#[repr(C)]
pub(crate) struct SharedRuntime {
    magic: u64,
    abi_version: u64,
    layout_fingerprint: u64,
    pub(crate) counters: SharedCounters,
    pub(crate) arena_pool_lock: AtomicU32,
    pub(crate) arena_free_head: AtomicPtr<RawArena>,
    pub(crate) arena_all_head: AtomicPtr<RawArena>,
    modules: [ModuleSlot; MODULE_SLOTS],
    pub(crate) threads: ThreadTable<ThreadAmbient, THREAD_SLOTS>,
}

impl SharedRuntime {
    /// Allocate and initialize a runtime from OS memory. Never freed: other
    /// copies may still hold its address when the creating copy unloads.
    fn create() -> *mut SharedRuntime {
        let alignment = align_of::<SharedRuntime>();
        let size = size_of::<SharedRuntime>() + alignment;
        let base = unsafe { os::alloc_zeroed(size) };
        if base.is_null() {
            std::alloc::handle_alloc_error(std::alloc::Layout::new::<SharedRuntime>());
        }
        let runtime = unsafe { base.add(base.align_offset(alignment)) }.cast::<SharedRuntime>();
        // SAFETY: zero is a valid bit pattern for every field (atomics, raw
        // pointers, and `ThreadValue::EMPTY`), so only the header needs writing.
        unsafe {
            (*runtime).magic = MAGIC;
            (*runtime).abi_version = ABI_VERSION as u64;
            (*runtime).layout_fingerprint = LAYOUT_FINGERPRINT;
        }
        runtime
    }

    fn verify(&self) {
        assert!(
            self.magic == MAGIC,
            "gpui shared runtime rendezvous returned memory without the runtime header; \
             another component is using the same rendezvous name"
        );
        assert!(
            self.abi_version == ABI_VERSION as u64 && self.layout_fingerprint == LAYOUT_FINGERPRINT,
            "gpui shared runtime ABI mismatch: this binary was built with ABI {ABI_VERSION} \
             (layout {LAYOUT_FINGERPRINT:#x}) but the process already runs ABI {} (layout {:#x}). \
             Rebuild the plugin against the same gpui version as the host.",
            self.abi_version,
            self.layout_fingerprint,
        );
    }

    fn attach_module(&self) -> ModuleId {
        for (slot, module) in self.modules.iter().enumerate() {
            if module
                .attached
                .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                self.counters.attached_modules.fetch_add(1, Ordering::Relaxed);
                self.counters
                    .total_module_attaches
                    .fetch_add(1, Ordering::Relaxed);
                return ModuleId {
                    slot: slot as u32,
                    generation: module.generation.load(Ordering::Acquire),
                };
            }
        }
        panic!("gpui shared runtime module table is full: more than {MODULE_SLOTS} copies of gpui are loaded at once");
    }

    fn detach_module(&self, id: ModuleId) {
        let Some(module) = self.modules.get(id.slot as usize) else {
            return;
        };
        module.generation.fetch_add(1, Ordering::AcqRel);
        module.attached.store(0, Ordering::Release);
        self.counters.attached_modules.fetch_sub(1, Ordering::Relaxed);
    }

    /// Whether the code that allocated a value tagged `id` is still loaded.
    #[inline]
    pub(crate) fn module_is_loaded(&self, id: ModuleId) -> bool {
        self.modules
            .get(id.slot as usize)
            .is_some_and(|module| module.generation.load(Ordering::Acquire) == id.generation)
    }

    pub(crate) fn lock_arena_pool(&self) -> ArenaPoolGuard<'_> {
        while self
            .arena_pool_lock
            .compare_exchange_weak(0, 1, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            std::hint::spin_loop();
        }
        ArenaPoolGuard { runtime: self }
    }
}

pub(crate) struct ArenaPoolGuard<'a> {
    runtime: &'a SharedRuntime,
}

impl Drop for ArenaPoolGuard<'_> {
    fn drop(&mut self) {
        self.runtime.arena_pool_lock.store(0, Ordering::Release);
    }
}

/// This copy's cached pointer to the process runtime. Holds no heap memory,
/// so there is nothing for it to leak when this copy is unloaded.
static RUNTIME: OnceLock<&'static SharedRuntime> = OnceLock::new();

/// This copy's module id, packed as `(slot + 1) << 32 | generation`, or 0
/// while detached.
static MODULE: AtomicU64 = AtomicU64::new(0);
static MODULE_ATTACH_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The process-wide runtime, finding or creating it on first use.
#[inline]
pub(crate) fn runtime() -> &'static SharedRuntime {
    RUNTIME.get_or_init(|| {
        let runtime = unsafe { &*os::rendezvous(SharedRuntime::create) };
        runtime.verify();
        runtime
    })
}

/// The OS id of the calling thread: identical in every copy of this crate,
/// never 0 and never `u64::MAX`.
#[inline]
pub(crate) fn current_thread_key() -> u64 {
    os::thread_key()
}

/// The id of this copy of the crate, attaching it to the runtime first if
/// needed.
#[inline]
pub(crate) fn current_module() -> ModuleId {
    let packed = MODULE.load(Ordering::Acquire);
    if packed != 0 {
        return unpack_module(packed);
    }
    attach_current_module()
}

fn unpack_module(packed: u64) -> ModuleId {
    ModuleId {
        slot: ((packed >> 32) - 1) as u32,
        generation: packed as u32,
    }
}

#[cold]
fn attach_current_module() -> ModuleId {
    let _guard = MODULE_ATTACH_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let packed = MODULE.load(Ordering::Acquire);
    if packed != 0 {
        return unpack_module(packed);
    }
    let id = runtime().attach_module();
    MODULE.store(
        ((id.slot as u64 + 1) << 32) | id.generation as u64,
        Ordering::Release,
    );
    id
}

/// Runs when this copy's image is unloaded (and at process exit). Only
/// touches atomics: on Windows this runs under the loader lock.
fn detach_current_module() {
    let packed = MODULE.swap(0, Ordering::AcqRel);
    if packed == 0 {
        return;
    }
    if let Some(runtime) = RUNTIME.get() {
        runtime.detach_module(unpack_module(packed));
    }
}

#[cfg(not(target_family = "wasm"))]
#[ctor::dtor]
fn detach_current_module_on_unload() {
    detach_current_module();
}

/// A snapshot of the process-wide shared runtime. Every copy of this crate in
/// the process reports the same numbers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SharedRuntimeStats {
    /// Arenas that have been created and not yet dropped.
    pub live_arenas: u64,
    /// Arena headers ever allocated. Headers are pooled and reused, so this
    /// is the peak number of simultaneously live arenas, not a leak counter.
    pub arena_headers: u64,
    /// Arena chunks currently allocated across all live arenas.
    pub live_chunks: u64,
    /// Bytes held by those chunks.
    pub live_chunk_bytes: u64,
    /// Bytes held by the arenas' destructor-record arrays.
    pub live_record_bytes: u64,
    /// Values currently allocated in any live arena.
    pub live_allocations: u64,
    /// Values whose destructor was skipped because the module that
    /// allocated them was unloaded before the arena was cleared.
    pub orphaned_records: u64,
    /// Copies of this crate currently attached to the runtime.
    pub attached_modules: u64,
    /// Copies of this crate that have ever attached.
    pub total_module_attaches: u64,
    /// Threads currently inside an `ElementArenaScope`.
    pub active_threads: u64,
}

/// Read the shared runtime's counters.
pub fn stats() -> SharedRuntimeStats {
    let runtime = runtime();
    let counters = &runtime.counters;
    SharedRuntimeStats {
        live_arenas: counters.live_arenas.load(Ordering::Relaxed),
        arena_headers: counters.arena_headers.load(Ordering::Relaxed),
        live_chunks: counters.live_chunks.load(Ordering::Relaxed),
        live_chunk_bytes: counters.live_chunk_bytes.load(Ordering::Relaxed),
        live_record_bytes: counters.live_record_bytes.load(Ordering::Relaxed),
        live_allocations: crate::arena::live_allocations(runtime),
        orphaned_records: counters.orphaned_records.load(Ordering::Relaxed),
        attached_modules: counters.attached_modules.load(Ordering::Relaxed),
        total_module_attaches: counters.total_module_attaches.load(Ordering::Relaxed),
        active_threads: runtime.threads.claimed_count() as u64,
    }
}

/// The address of the process runtime. Identical in every copy of this
/// crate loaded into the process; useful to verify a plugin shares the host's
/// runtime.
pub fn runtime_address() -> usize {
    ptr::from_ref(runtime()) as usize
}

/// This copy's slot in the runtime's module table.
pub fn module_slot() -> u32 {
    current_module().slot
}

pub(crate) mod os {
    #[cfg(windows)]
    mod imp {
        use std::ffi::c_void;

        type Handle = isize;
        const INVALID_HANDLE_VALUE: Handle = -1;
        const PAGE_READWRITE: u32 = 0x04;
        const FILE_MAP_ALL_ACCESS: u32 = 0x000f_001f;
        const HEAP_ZERO_MEMORY: u32 = 0x08;

        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetCurrentProcessId() -> u32;
            fn GetCurrentThreadId() -> u32;
            fn GetLastError() -> u32;
            fn CreateFileMappingW(
                file: Handle,
                attributes: *const c_void,
                protect: u32,
                maximum_size_high: u32,
                maximum_size_low: u32,
                name: *const u16,
            ) -> Handle;
            fn MapViewOfFile(
                mapping: Handle,
                desired_access: u32,
                offset_high: u32,
                offset_low: u32,
                bytes: usize,
            ) -> *mut c_void;
            fn UnmapViewOfFile(base: *const c_void) -> i32;
            fn CloseHandle(handle: Handle) -> i32;
            fn GetProcessHeap() -> Handle;
            fn HeapAlloc(heap: Handle, flags: u32, bytes: usize) -> *mut c_void;
            fn HeapReAlloc(heap: Handle, flags: u32, memory: *mut c_void, bytes: usize) -> *mut c_void;
            fn HeapFree(heap: Handle, flags: u32, memory: *mut c_void) -> i32;
        }

        #[inline]
        pub(crate) fn thread_key() -> u64 {
            unsafe { GetCurrentThreadId() as u64 }
        }

        pub(crate) unsafe fn alloc(size: usize) -> *mut u8 {
            unsafe { HeapAlloc(GetProcessHeap(), 0, size).cast() }
        }

        pub(crate) unsafe fn alloc_zeroed(size: usize) -> *mut u8 {
            unsafe { HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, size).cast() }
        }

        pub(crate) unsafe fn realloc(memory: *mut u8, size: usize) -> *mut u8 {
            if memory.is_null() {
                return unsafe { alloc(size) };
            }
            unsafe { HeapReAlloc(GetProcessHeap(), 0, memory.cast(), size).cast() }
        }

        pub(crate) unsafe fn free(memory: *mut u8) {
            if !memory.is_null() && unsafe { HeapFree(GetProcessHeap(), 0, memory.cast()) } == 0 {
                panic!("HeapFree failed for shared runtime memory: error {}", unsafe {
                    GetLastError()
                });
            }
        }

        pub(crate) fn rendezvous(
            create: fn() -> *mut super::super::SharedRuntime,
        ) -> *mut super::super::SharedRuntime {
            let name: Vec<u16> = format!("Local\\gpui-ce-shared-runtime-{}", unsafe {
                GetCurrentProcessId()
            })
            .encode_utf16()
            .chain(Some(0))
            .collect();
            let mailbox_size = size_of::<super::Mailbox>();
            let mapping = unsafe {
                CreateFileMappingW(
                    INVALID_HANDLE_VALUE,
                    std::ptr::null(),
                    PAGE_READWRITE,
                    0,
                    mailbox_size as u32,
                    name.as_ptr(),
                )
            };
            if mapping == 0 {
                panic!("failed to create the gpui shared runtime rendezvous mapping: error {}", unsafe {
                    GetLastError()
                });
            }
            let view = unsafe { MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, mailbox_size) };
            if view.is_null() {
                let error = unsafe { GetLastError() };
                unsafe { CloseHandle(mapping) };
                panic!("failed to map the gpui shared runtime rendezvous mapping: error {error}");
            }
            // SAFETY: the view is at least `mailbox_size` bytes, page aligned,
            // and zero-filled by the OS when the mapping was first created.
            let mailbox = unsafe { &*view.cast::<super::Mailbox>() };
            let (runtime, created) = mailbox.publish_or_wait(create);
            unsafe { UnmapViewOfFile(view) };
            // The creator's handle is what keeps the mapping's name alive for
            // copies loaded later, so it stays open for the process lifetime.
            // Every other copy closes its handle so repeated plugin loads
            // don't accumulate handles.
            if !created {
                unsafe { CloseHandle(mapping) };
            }
            runtime
        }
    }

    #[cfg(all(unix, not(target_family = "wasm")))]
    mod imp {
        const RENDEZVOUS_VARIABLE: &std::ffi::CStr = c"GPUI_CE_SHARED_RUNTIME";

        #[inline]
        pub(crate) fn thread_key() -> u64 {
            unsafe { libc::pthread_self() as u64 }
        }

        pub(crate) unsafe fn alloc(size: usize) -> *mut u8 {
            unsafe { libc::malloc(size).cast() }
        }

        pub(crate) unsafe fn alloc_zeroed(size: usize) -> *mut u8 {
            unsafe { libc::calloc(1, size).cast() }
        }

        pub(crate) unsafe fn realloc(memory: *mut u8, size: usize) -> *mut u8 {
            unsafe { libc::realloc(memory.cast(), size).cast() }
        }

        pub(crate) unsafe fn free(memory: *mut u8) {
            unsafe { libc::free(memory.cast()) }
        }

        /// There is no process-scoped named object on unix that disappears
        /// with the process, so the runtime's address is published in the
        /// environment, tagged with the pid so a forked child that inherits
        /// the variable creates its own runtime instead of trusting the
        /// parent's address. Copies within one binary are serialized by
        /// `RUNTIME`'s `OnceLock`; the host creates the runtime from
        /// `App::new`, long before any plugin can be loaded, so two copies
        /// racing to create it is not a real configuration.
        pub(crate) fn rendezvous(
            create: fn() -> *mut super::super::SharedRuntime,
        ) -> *mut super::super::SharedRuntime {
            let pid = unsafe { libc::getpid() };
            let existing = unsafe { libc::getenv(RENDEZVOUS_VARIABLE.as_ptr()) };
            if !existing.is_null() {
                let value = unsafe { std::ffi::CStr::from_ptr(existing) }.to_string_lossy();
                if let Some((owner, address)) = value.split_once(':')
                    && owner.parse::<i32>().ok() == Some(pid)
                    && let Ok(address) = usize::from_str_radix(address, 16)
                    && address != 0
                {
                    return address as *mut super::super::SharedRuntime;
                }
            }
            let runtime = create();
            let value = std::ffi::CString::new(format!("{pid}:{:x}", runtime as usize))
                .expect("formatted integers contain no NUL bytes");
            if unsafe { libc::setenv(RENDEZVOUS_VARIABLE.as_ptr(), value.as_ptr(), 1) } != 0 {
                panic!(
                    "failed to publish the gpui shared runtime: {}",
                    std::io::Error::last_os_error()
                );
            }
            runtime
        }
    }

    #[cfg(target_family = "wasm")]
    mod imp {
        use std::alloc::{Layout, alloc as rust_alloc, alloc_zeroed as rust_alloc_zeroed};

        // wasm has no dynamic loading, so there is exactly one copy of this
        // crate and its own allocator is the process allocator. Memory here
        // is only ever freed by `free`/`realloc` from this same copy, which
        // need the size, so each block carries a small size prefix.
        const PREFIX: usize = 16;

        pub(crate) fn thread_key() -> u64 {
            1
        }

        fn layout(size: usize) -> Layout {
            Layout::from_size_align(size + PREFIX, PREFIX).expect("allocation size overflow")
        }

        pub(crate) unsafe fn alloc(size: usize) -> *mut u8 {
            unsafe {
                let base = rust_alloc(layout(size));
                if base.is_null() {
                    return base;
                }
                base.cast::<usize>().write(size);
                base.add(PREFIX)
            }
        }

        pub(crate) unsafe fn alloc_zeroed(size: usize) -> *mut u8 {
            unsafe {
                let base = rust_alloc_zeroed(layout(size));
                if base.is_null() {
                    return base;
                }
                base.cast::<usize>().write(size);
                base.add(PREFIX)
            }
        }

        pub(crate) unsafe fn realloc(memory: *mut u8, size: usize) -> *mut u8 {
            unsafe {
                if memory.is_null() {
                    return alloc(size);
                }
                let base = memory.sub(PREFIX);
                let old_size = base.cast::<usize>().read();
                let new_base = std::alloc::realloc(base, layout(old_size), size + PREFIX);
                if new_base.is_null() {
                    return new_base;
                }
                new_base.cast::<usize>().write(size);
                new_base.add(PREFIX)
            }
        }

        pub(crate) unsafe fn free(memory: *mut u8) {
            unsafe {
                if memory.is_null() {
                    return;
                }
                let base = memory.sub(PREFIX);
                let size = base.cast::<usize>().read();
                std::alloc::dealloc(base, layout(size));
            }
        }

        pub(crate) fn rendezvous(
            create: fn() -> *mut super::super::SharedRuntime,
        ) -> *mut super::super::SharedRuntime {
            create()
        }
    }

    pub(crate) use imp::*;

    /// The shared page every copy maps during rendezvous on Windows.
    #[cfg_attr(not(windows), allow(dead_code))]
    #[repr(C)]
    pub(super) struct Mailbox {
        state: std::sync::atomic::AtomicU32,
        runtime: std::sync::atomic::AtomicUsize,
    }

    #[cfg_attr(not(windows), allow(dead_code))]
    impl Mailbox {
        const UNINITIALIZED: u32 = 0;
        const INITIALIZING: u32 = 1;
        const READY: u32 = 2;

        /// Returns the runtime, and whether this call created it.
        pub(super) fn publish_or_wait(
            &self,
            create: fn() -> *mut super::SharedRuntime,
        ) -> (*mut super::SharedRuntime, bool) {
            use std::sync::atomic::Ordering;
            if self
                .state
                .compare_exchange(
                    Self::UNINITIALIZED,
                    Self::INITIALIZING,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                let runtime = create();
                self.runtime.store(runtime as usize, Ordering::Release);
                self.state.store(Self::READY, Ordering::Release);
                return (runtime, true);
            }
            while self.state.load(Ordering::Acquire) != Self::READY {
                std::thread::yield_now();
            }
            (
                self.runtime.load(Ordering::Acquire) as *mut super::SharedRuntime,
                false,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        collections::HashSet,
        sync::{Arc, Barrier},
    };

    #[derive(Debug)]
    struct Counter(AtomicU64);

    impl ThreadValue for Counter {
        const EMPTY: Self = Self(AtomicU64::new(0));
    }

    #[test]
    fn runtime_is_a_singleton_with_a_valid_header() {
        let first = runtime_address();
        let second = runtime_address();
        assert_eq!(first, second);
        runtime().verify();
        assert_eq!(runtime().abi_version, ABI_VERSION as u64);
    }

    #[test]
    fn current_module_is_stable_and_loaded() {
        let module = current_module();
        assert_eq!(module, current_module());
        assert!(runtime().module_is_loaded(module));
        assert!(stats().attached_modules >= 1);
    }

    #[test]
    fn detached_module_ids_never_match_again_even_after_slot_reuse() {
        let runtime = runtime();
        let first = runtime.attach_module();
        assert!(runtime.module_is_loaded(first));
        runtime.detach_module(first);
        assert!(!runtime.module_is_loaded(first));

        // The slot is free again; reusing it must hand out a new generation.
        let mut reused = None;
        let mut held = Vec::new();
        for _ in 0..MODULE_SLOTS {
            let id = runtime.attach_module();
            if id.slot == first.slot {
                reused = Some(id);
                held.push(id);
                break;
            }
            held.push(id);
        }
        let reused = reused.expect("the detached slot should be reusable");
        assert_ne!(reused.generation, first.generation);
        assert!(!runtime.module_is_loaded(first));
        assert!(runtime.module_is_loaded(reused));
        for id in held {
            runtime.detach_module(id);
        }
    }

    #[test]
    fn thread_table_claims_finds_and_releases() {
        let table = ThreadTable::<Counter, 8>::new();
        assert!(table.find(42).is_none());
        table.find_or_claim(42).0.store(7, Ordering::Relaxed);
        assert_eq!(table.find(42).unwrap().0.load(Ordering::Relaxed), 7);
        assert_eq!(table.claimed_count(), 1);

        table.find(42).unwrap().0.store(0, Ordering::Relaxed);
        table.release(42);
        assert!(table.find(42).is_none());
        assert_eq!(table.claimed_count(), 0);
    }

    #[test]
    fn thread_table_keeps_probe_chains_intact_across_releases() {
        // Every key hashes somewhere in an 8-slot table, so filling it forces
        // collisions; releasing entries in the middle of chains must not hide
        // the entries behind them.
        let table = ThreadTable::<Counter, 8>::new();
        let keys: Vec<u64> = (1..=8).collect();
        for &key in &keys {
            table.find_or_claim(key).0.store(key, Ordering::Relaxed);
        }
        for &key in keys.iter().step_by(2) {
            table.find(key).unwrap().0.store(0, Ordering::Relaxed);
            table.release(key);
        }
        for &key in keys.iter().skip(1).step_by(2) {
            assert_eq!(table.find(key).unwrap().0.load(Ordering::Relaxed), key);
        }
        // Released slots are reused.
        for key in 100..104 {
            table.find_or_claim(key);
        }
        assert_eq!(table.claimed_count(), 8);
    }

    #[test]
    #[should_panic(expected = "thread table is full")]
    fn thread_table_reports_exhaustion() {
        let table = ThreadTable::<Counter, 4>::new();
        for key in 1..=5 {
            table.find_or_claim(key);
        }
    }

    #[test]
    fn thread_table_is_race_free_under_contention() {
        const THREADS: usize = 64;
        const ROUNDS: usize = 2_000;
        let table = Arc::new(ThreadTable::<Counter, 128>::new());
        let barrier = Arc::new(Barrier::new(THREADS));
        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let table = table.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    let key = current_thread_key();
                    barrier.wait();
                    for round in 0..ROUNDS {
                        let slot = table.find_or_claim(key);
                        assert_eq!(slot.0.load(Ordering::Relaxed), 0, "slot shared by two threads");
                        slot.0.store(round as u64 + 1, Ordering::Relaxed);
                        assert_eq!(
                            table.find(key).unwrap().0.load(Ordering::Relaxed),
                            round as u64 + 1
                        );
                        slot.0.store(0, Ordering::Relaxed);
                        table.release(key);
                    }
                    key
                })
            })
            .collect();
        let keys: HashSet<u64> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();
        assert_eq!(keys.len(), THREADS, "OS thread keys must be distinct");
        assert_eq!(table.claimed_count(), 0);
    }

    #[test]
    fn thread_keys_are_never_sentinels() {
        let key = current_thread_key();
        assert_ne!(key, EMPTY_THREAD);
        assert_ne!(key, RELEASED_THREAD);
        let other = std::thread::spawn(current_thread_key).join().unwrap();
        assert_ne!(key, other);
    }
}
