//! Cross-DLL tests for the shared runtime and the element arena.
//!
//! These build `tests/dll_fixture`, a cdylib that statically links its own
//! copy of gpui exactly like a Pulsar plugin, load it into this process with
//! `libloading`, and drive element allocation across the boundary: plugin
//! code allocating into host arenas, host code clearing plugin values,
//! plugins being loaded and unloaded hundreds of times, several plugin
//! copies at once, many threads at once, and plugins unloaded at the worst
//! possible moment.
//!
//! This binary and the fixture each install a `TaggedAllocator` with a
//! different tag, so any cross-binary free aborts the process, and both
//! heaps are compared byte-for-byte against their baselines.

#[path = "support/fixture_build.rs"]
mod fixture_build;
#[path = "support/tagged_allocator.rs"]
mod tagged_allocator;

use std::{
    cell::RefCell,
    panic::{self, AssertUnwindSafe},
    path::{Path, PathBuf},
    sync::{
        Arc, Barrier, Mutex, MutexGuard,
        atomic::{AtomicU64, Ordering},
    },
};

#[cfg(windows)]
use fixture_build::process_handle_count;
use fixture_build::{fixture_copy_path, fixture_path, is_loaded};
use gpui::{Arena, ElementArenaScope, shared_runtime};
use libloading::Library;
use tagged_allocator::{HOST_TAG, TaggedAllocator};

#[global_allocator]
static ALLOCATOR: TaggedAllocator = TaggedAllocator::new(HOST_TAG);

const STATUS_OK: u32 = 0;
const STATUS_PANICKED: u32 = 1;
const NO_PANIC: u32 = u32::MAX;

/// Arena allocations `fixture_build_elements` makes per element: a `div`, a
/// drop probe, and a text element.
const ALLOCATIONS_PER_ELEMENT: usize = 3;

/// Every test here drives process-wide state (the runtime's counters, the
/// set of loaded modules, both heaps), so they run one at a time.
fn serial() -> MutexGuard<'static, ()> {
    static SERIAL: Mutex<()> = Mutex::new(());
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

struct Fixture {
    path: PathBuf,
    runtime_address: extern "C" fn() -> usize,
    abi_version: extern "C" fn() -> u32,
    module_slot: extern "C" fn() -> u32,
    live_heap_bytes: extern "C" fn() -> i64,
    live_heap_blocks: extern "C" fn() -> i64,
    live_allocations: extern "C" fn() -> u64,
    build_elements: unsafe extern "C" fn(u32, *const AtomicU64, u32) -> u32,
    run_own_frames: unsafe extern "C" fn(u32, u32, *const AtomicU64, *mut u64) -> u32,
    library: Library,
}

impl Fixture {
    fn load(path: &Path) -> Self {
        unsafe fn symbol<T: Copy>(library: &Library, name: &str) -> T {
            unsafe {
                *library
                    .get::<T>(name.as_bytes())
                    .unwrap_or_else(|error| panic!("fixture is missing `{name}`: {error}"))
            }
        }
        let library = unsafe { Library::new(path) }
            .unwrap_or_else(|error| panic!("failed to load {}: {error}", path.display()));
        unsafe {
            Self {
                path: path.to_path_buf(),
                runtime_address: symbol(&library, "fixture_runtime_address"),
                abi_version: symbol(&library, "fixture_abi_version"),
                module_slot: symbol(&library, "fixture_module_slot"),
                live_heap_bytes: symbol(&library, "fixture_live_heap_bytes"),
                live_heap_blocks: symbol(&library, "fixture_live_heap_blocks"),
                live_allocations: symbol(&library, "fixture_live_allocations"),
                build_elements: symbol(&library, "fixture_build_elements"),
                run_own_frames: symbol(&library, "fixture_run_own_frames"),
                library,
            }
        }
    }

    fn build(&self, count: u32, drops: &'static AtomicU64) -> u32 {
        unsafe { (self.build_elements)(count, drops, NO_PANIC) }
    }

    fn build_with_panicking_drop(&self, count: u32, drops: &'static AtomicU64, panic_index: u32) -> u32 {
        unsafe { (self.build_elements)(count, drops, panic_index) }
    }

    fn run_own_frames(&self, frames: u32, count: u32, drops: &'static AtomicU64) -> u64 {
        let mut last_frame_allocations = 0;
        let status = unsafe { (self.run_own_frames)(frames, count, drops, &mut last_frame_allocations) };
        assert_eq!(status, STATUS_OK, "fixture_run_own_frames panicked");
        last_frame_allocations
    }

    /// The plugin's heap usage, for exact before/after comparisons.
    fn heap(&self) -> (i64, i64) {
        ((self.live_heap_bytes)(), (self.live_heap_blocks)())
    }

    fn unload(self) {
        let path = self.path;
        self.library.close().expect("failed to unload the fixture");
        assert!(!is_loaded(&path), "the fixture stayed mapped after its last handle was closed");
    }
}

fn host_heap() -> (i64, i64) {
    (ALLOCATOR.live_bytes(), ALLOCATOR.live_blocks())
}

/// Runs one frame: `build` inside a scope on `arena`, then a clear.
fn frame(arena: &RefCell<Arena>, build: impl FnOnce()) {
    {
        let _scope = ElementArenaScope::enter(arena);
        build();
    }
    arena.borrow_mut().clear();
}

#[test]
fn plugin_attaches_to_the_host_runtime_and_detaches_on_unload() {
    let _serial = serial();
    let host_slot = shared_runtime::module_slot();
    let before = shared_runtime::stats();

    let fixture = Fixture::load(fixture_path());
    assert_eq!((fixture.runtime_address)(), shared_runtime::runtime_address());
    assert_eq!((fixture.abi_version)(), shared_runtime::ABI_VERSION);
    let plugin_slot = (fixture.module_slot)();
    assert_ne!(plugin_slot, host_slot);
    let attached = shared_runtime::stats();
    assert_eq!(attached.attached_modules, before.attached_modules + 1);
    assert_eq!(attached.total_module_attaches, before.total_module_attaches + 1);

    fixture.unload();
    let after = shared_runtime::stats();
    assert_eq!(after.attached_modules, before.attached_modules, "unloading must detach the module");
    assert_eq!(after.total_module_attaches, before.total_module_attaches + 1);
}

#[test]
fn plugin_elements_land_in_the_host_arena_without_a_plugin_scope() {
    static DROPS: AtomicU64 = AtomicU64::new(0);
    let _serial = serial();
    let fixture = Fixture::load(fixture_path());
    let arena = RefCell::new(Arena::new(64 * 1024));
    frame(&arena, || assert_eq!(fixture.build(1, &DROPS), STATUS_OK));
    let plugin_baseline = fixture.heap();
    let drops_before = DROPS.load(Ordering::SeqCst);

    {
        let _scope = ElementArenaScope::enter(&arena);
        assert_eq!(fixture.build(100, &DROPS), STATUS_OK);
    }
    assert_eq!(arena.borrow().allocation_count(), 100 * ALLOCATIONS_PER_ELEMENT);
    assert_eq!(
        (fixture.live_allocations)(),
        shared_runtime::stats().live_allocations,
        "the plugin and the host must see the same runtime"
    );
    assert!(fixture.heap().0 > plugin_baseline.0, "elements own plugin-heap memory");
    assert_eq!(DROPS.load(Ordering::SeqCst), drops_before, "nothing is dropped before the clear");

    arena.borrow_mut().clear();
    assert_eq!(DROPS.load(Ordering::SeqCst), drops_before + 100);
    assert_eq!(fixture.heap(), plugin_baseline, "clearing must free every plugin allocation");
    assert_eq!(arena.borrow().allocation_count(), 0);

    drop(arena);
    fixture.unload();
}

#[test]
fn thousands_of_frames_leave_every_heap_exactly_where_it_started() {
    static DROPS: AtomicU64 = AtomicU64::new(0);
    const FRAMES: u32 = 3_000;
    const ELEMENTS: u32 = 200;
    let _serial = serial();
    let fixture = Fixture::load(fixture_path());
    let arena = RefCell::new(Arena::new(32 * 1024));
    for _ in 0..5 {
        frame(&arena, || assert_eq!(fixture.build(ELEMENTS, &DROPS), STATUS_OK));
    }

    let plugin_baseline = fixture.heap();
    let host_baseline = host_heap();
    let runtime_baseline = shared_runtime::stats();
    let capacity = arena.borrow().capacity();
    let drops_before = DROPS.load(Ordering::SeqCst);

    for frame_index in 0..FRAMES {
        frame(&arena, || assert_eq!(fixture.build(ELEMENTS, &DROPS), STATUS_OK));
        assert_eq!(fixture.heap(), plugin_baseline, "plugin heap drifted on frame {frame_index}");
        assert_eq!(arena.borrow().capacity(), capacity, "arena grew on frame {frame_index}");
    }

    assert_eq!(host_heap(), host_baseline, "host heap drifted");
    let runtime_after = shared_runtime::stats();
    assert_eq!(runtime_after.live_chunks, runtime_baseline.live_chunks);
    assert_eq!(runtime_after.live_chunk_bytes, runtime_baseline.live_chunk_bytes);
    assert_eq!(runtime_after.live_record_bytes, runtime_baseline.live_record_bytes);
    assert_eq!(runtime_after.live_allocations, 0);
    assert_eq!(
        DROPS.load(Ordering::SeqCst),
        drops_before + u64::from(FRAMES * ELEMENTS)
    );

    drop(arena);
    fixture.unload();
}

#[test]
fn hundreds_of_load_unload_cycles_leave_nothing_behind() {
    static DROPS: AtomicU64 = AtomicU64::new(0);
    const CYCLES: u32 = 200;
    const ELEMENTS: u32 = 100;
    let _serial = serial();
    let arena = RefCell::new(Arena::new(64 * 1024));
    let cycle = || {
        let fixture = Fixture::load(fixture_path());
        frame(&arena, || assert_eq!(fixture.build(ELEMENTS, &DROPS), STATUS_OK));
        fixture.unload();
    };
    cycle();
    cycle();

    let host_baseline = host_heap();
    let runtime_baseline = shared_runtime::stats();
    #[cfg(windows)]
    let handles_baseline = process_handle_count();
    let drops_before = DROPS.load(Ordering::SeqCst);

    for _ in 0..CYCLES {
        cycle();
    }

    let runtime_after = shared_runtime::stats();
    assert_eq!(runtime_after.attached_modules, runtime_baseline.attached_modules);
    assert_eq!(
        runtime_after.total_module_attaches,
        runtime_baseline.total_module_attaches + u64::from(CYCLES)
    );
    assert_eq!(runtime_after.live_arenas, runtime_baseline.live_arenas);
    assert_eq!(runtime_after.arena_headers, runtime_baseline.arena_headers);
    assert_eq!(runtime_after.live_chunks, runtime_baseline.live_chunks);
    assert_eq!(runtime_after.live_chunk_bytes, runtime_baseline.live_chunk_bytes);
    assert_eq!(runtime_after.live_record_bytes, runtime_baseline.live_record_bytes);
    assert_eq!(runtime_after.orphaned_records, runtime_baseline.orphaned_records);
    assert_eq!(runtime_after.active_threads, 0);
    assert_eq!(host_heap(), host_baseline, "host heap drifted across load/unload cycles");
    #[cfg(windows)]
    {
        let handles = process_handle_count();
        assert!(
            handles <= handles_baseline + 2,
            "handle count grew from {handles_baseline} to {handles} over {CYCLES} load/unload cycles"
        );
    }
    assert_eq!(
        DROPS.load(Ordering::SeqCst),
        drops_before + u64::from(CYCLES * ELEMENTS)
    );
}

#[test]
fn two_plugin_copies_and_the_host_share_one_runtime() {
    static DROPS: AtomicU64 = AtomicU64::new(0);
    let _serial = serial();
    let first = Fixture::load(fixture_path());
    let second = Fixture::load(fixture_copy_path());
    let host_slot = shared_runtime::module_slot();
    let slots = [host_slot, (first.module_slot)(), (second.module_slot)()];
    assert!(
        slots[0] != slots[1] && slots[1] != slots[2] && slots[0] != slots[2],
        "each copy needs its own module slot: {slots:?}"
    );
    assert_eq!((first.runtime_address)(), shared_runtime::runtime_address());
    assert_eq!((second.runtime_address)(), shared_runtime::runtime_address());

    let arena = RefCell::new(Arena::new(64 * 1024));
    frame(&arena, || {
        assert_eq!(first.build(1, &DROPS), STATUS_OK);
        assert_eq!(second.build(1, &DROPS), STATUS_OK);
    });
    let first_baseline = first.heap();
    let second_baseline = second.heap();

    for _ in 0..500 {
        {
            let _scope = ElementArenaScope::enter(&arena);
            assert_eq!(first.build(40, &DROPS), STATUS_OK);
            assert_eq!(second.build(60, &DROPS), STATUS_OK);
            // Host-compiled allocations interleaved with both plugins'.
            let mut host_arena = arena.borrow_mut();
            host_arena.alloc(|| String::from("host value"));
        }
        assert_eq!(arena.borrow().allocation_count(), 100 * ALLOCATIONS_PER_ELEMENT + 1);
        arena.borrow_mut().clear();
        assert_eq!(first.heap(), first_baseline);
        assert_eq!(second.heap(), second_baseline);
    }

    // Unloading one copy must not disturb the other.
    first.unload();
    for _ in 0..100 {
        frame(&arena, || assert_eq!(second.build(50, &DROPS), STATUS_OK));
        assert_eq!(second.heap(), second_baseline);
    }
    drop(arena);
    second.unload();
}

#[test]
fn unloading_a_plugin_with_live_elements_orphans_them_instead_of_crashing() {
    static DROPS: AtomicU64 = AtomicU64::new(0);
    const ELEMENTS: u32 = 50;
    let _serial = serial();
    let arena = RefCell::new(Arena::new(64 * 1024));
    let fixture = Fixture::load(fixture_path());
    {
        let _scope = ElementArenaScope::enter(&arena);
        assert_eq!(fixture.build(ELEMENTS, &DROPS), STATUS_OK);
    }
    let live = arena.borrow().allocation_count() as u64;
    assert_eq!(live, u64::from(ELEMENTS) * ALLOCATIONS_PER_ELEMENT as u64);
    let orphans_before = shared_runtime::stats().orphaned_records;

    // The worst case: the plugin's code, including every destructor and
    // vtable those values point at, is unmapped while they're still live.
    fixture.unload();
    arena.borrow_mut().clear();

    assert_eq!(DROPS.load(Ordering::SeqCst), 0, "destructors in unloaded code must never run");
    assert_eq!(shared_runtime::stats().orphaned_records, orphans_before + live);

    // The arena, and the plugin once reloaded, keep working normally.
    let fixture = Fixture::load(fixture_path());
    frame(&arena, || assert_eq!(fixture.build(ELEMENTS, &DROPS), STATUS_OK));
    assert_eq!(DROPS.load(Ordering::SeqCst), u64::from(ELEMENTS));
    assert_eq!(shared_runtime::stats().orphaned_records, orphans_before + live);
    drop(arena);
    fixture.unload();
}

#[test]
fn plugin_destructor_panics_are_contained_and_reported_by_the_host() {
    static DROPS: AtomicU64 = AtomicU64::new(0);
    let _serial = serial();
    let fixture = Fixture::load(fixture_path());
    let arena = RefCell::new(Arena::new(64 * 1024));
    frame(&arena, || assert_eq!(fixture.build(1, &DROPS), STATUS_OK));
    let plugin_baseline = fixture.heap();
    let drops_before = DROPS.load(Ordering::SeqCst);

    {
        let _scope = ElementArenaScope::enter(&arena);
        assert_eq!(fixture.build_with_panicking_drop(10, &DROPS, 3), STATUS_OK);
    }
    // The panic starts in the plugin's `std`; if it unwound into this
    // binary's frames it would be a foreign exception and abort. It must
    // instead come back as an ordinary panic raised by the host's copy.
    let result = panic::catch_unwind(AssertUnwindSafe(|| arena.borrow_mut().clear()));
    let message = result
        .expect_err("the destructor panic must be reported")
        .downcast::<String>()
        .expect("the host's panic payload is a String");
    assert!(message.contains("1 arena value destructor(s) panicked"), "{message}");
    assert_eq!(DROPS.load(Ordering::SeqCst), drops_before + 10, "every destructor still ran");
    assert_eq!(fixture.heap(), plugin_baseline, "the panicking value's fields were still freed");

    frame(&arena, || assert_eq!(fixture.build(10, &DROPS), STATUS_OK));
    assert_eq!(fixture.heap(), plugin_baseline);
    drop(arena);
    fixture.unload();
}

#[test]
fn building_without_an_active_scope_is_rejected_rather_than_leaked() {
    static DROPS: AtomicU64 = AtomicU64::new(0);
    let _serial = serial();
    let fixture = Fixture::load(fixture_path());
    let before = shared_runtime::stats();

    // No scope on this thread: the old per-DLL fallback arena grew a fresh
    // chunk here and never cleared it (Pulsar-Native issue #261).
    assert_eq!(fixture.build(10, &DROPS), STATUS_PANICKED);

    let after = shared_runtime::stats();
    assert_eq!(after.live_arenas, before.live_arenas);
    assert_eq!(after.live_chunks, before.live_chunks);
    assert_eq!(after.live_allocations, before.live_allocations);
    fixture.unload();
}

#[test]
fn each_thread_routes_plugin_allocations_to_its_own_arena() {
    static DROPS: AtomicU64 = AtomicU64::new(0);
    const THREADS: usize = 8;
    const FRAMES: u32 = 200;
    const ELEMENTS: u32 = 25;
    let _serial = serial();
    let fixture = Arc::new(Fixture::load(fixture_path()));
    let barrier = Arc::new(Barrier::new(THREADS));
    let drops_before = DROPS.load(Ordering::SeqCst);

    let threads: Vec<_> = (0..THREADS)
        .map(|_| {
            let fixture = fixture.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                let arena = RefCell::new(Arena::new(16 * 1024));
                barrier.wait();
                for _ in 0..FRAMES {
                    {
                        let _scope = ElementArenaScope::enter(&arena);
                        assert_eq!(fixture.build(ELEMENTS, &DROPS), STATUS_OK);
                    }
                    assert_eq!(
                        arena.borrow().allocation_count(),
                        ELEMENTS as usize * ALLOCATIONS_PER_ELEMENT,
                        "another thread's plugin allocations leaked into this arena"
                    );
                    arena.borrow_mut().clear();
                }
            })
        })
        .collect();
    for thread in threads {
        thread.join().expect("worker thread panicked");
    }

    assert_eq!(
        DROPS.load(Ordering::SeqCst),
        drops_before + (THREADS as u64) * u64::from(FRAMES * ELEMENTS)
    );
    assert_eq!(shared_runtime::stats().active_threads, 0, "every thread released its slot");
    let fixture = Arc::into_inner(fixture).expect("all workers have exited");
    fixture.unload();
}

#[test]
fn plugin_owned_arenas_nest_inside_the_host_scope() {
    static DROPS: AtomicU64 = AtomicU64::new(0);
    let _serial = serial();
    let fixture = Fixture::load(fixture_path());
    let live_arenas_before = shared_runtime::stats().live_arenas;
    let arena = RefCell::new(Arena::new(64 * 1024));
    let scope = ElementArenaScope::enter(&arena);

    assert_eq!(fixture.build(10, &DROPS), STATUS_OK);
    assert_eq!(arena.borrow().allocation_count(), 10 * ALLOCATIONS_PER_ELEMENT);

    // The plugin creates, fills, clears and drops its own arena, entering
    // its own scope on top of ours.
    let plugin_frame = fixture.run_own_frames(50, 20, &DROPS);
    assert_eq!(plugin_frame, 20 * ALLOCATIONS_PER_ELEMENT as u64);
    assert_eq!(arena.borrow().allocation_count(), 10 * ALLOCATIONS_PER_ELEMENT);

    // Our scope is active again once the plugin's has ended.
    assert_eq!(fixture.build(10, &DROPS), STATUS_OK);
    assert_eq!(arena.borrow().allocation_count(), 20 * ALLOCATIONS_PER_ELEMENT);
    drop(scope);
    arena.borrow_mut().clear();
    drop(arena);

    assert_eq!(shared_runtime::stats().live_arenas, live_arenas_before);
    assert_eq!(shared_runtime::stats().active_threads, 0);
    fixture.unload();
}

#[test]
fn randomized_frames_with_reloads_and_chunk_growth() {
    static DROPS: AtomicU64 = AtomicU64::new(0);
    const FRAMES: u32 = 4_000;
    let _serial = serial();
    let arena = RefCell::new(Arena::new(4 * 1024));
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    let mut fixture = Fixture::load(fixture_path());
    frame(&arena, || assert_eq!(fixture.build(1, &DROPS), STATUS_OK));
    let mut plugin_baseline = fixture.heap();
    let mut peak_capacity = arena.borrow().capacity();
    let mut expected_drops = DROPS.load(Ordering::SeqCst);

    for frame_index in 0..FRAMES {
        if frame_index % 250 == 249 {
            fixture.unload();
            fixture = Fixture::load(fixture_path());
            frame(&arena, || assert_eq!(fixture.build(1, &DROPS), STATUS_OK));
            expected_drops += 1;
            plugin_baseline = fixture.heap();
        }
        let elements = (next() % 600) as u32;
        let host_values = (next() % 50) as usize;
        {
            let _scope = ElementArenaScope::enter(&arena);
            assert_eq!(fixture.build(elements, &DROPS), STATUS_OK);
            let mut host_arena = arena.borrow_mut();
            for index in 0..host_values {
                host_arena.alloc(|| vec![index; index % 17]);
            }
        }
        assert_eq!(
            arena.borrow().allocation_count(),
            elements as usize * ALLOCATIONS_PER_ELEMENT + host_values
        );
        arena.borrow_mut().clear();
        expected_drops += u64::from(elements);
        assert_eq!(DROPS.load(Ordering::SeqCst), expected_drops);
        assert_eq!(fixture.heap(), plugin_baseline, "plugin heap drifted on frame {frame_index}");
        let capacity = arena.borrow().capacity();
        assert!(capacity >= peak_capacity, "an arena never gives chunks back while alive");
        peak_capacity = capacity;
    }

    let chunks_with_arena = shared_runtime::stats().live_chunk_bytes;
    drop(arena);
    assert!(shared_runtime::stats().live_chunk_bytes < chunks_with_arena);
    fixture.unload();
}
