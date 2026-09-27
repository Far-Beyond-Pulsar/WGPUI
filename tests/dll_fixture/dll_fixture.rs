//! A plugin-shaped cdylib for `tests/dll_boundary.rs`.
//!
//! Like a Pulsar plugin, it statically links its own copy of gpui (and of
//! `std`, and its own global allocator), so everything it does exercises the
//! real cross-copy boundary: it allocates elements with *its* copy of
//! `AnyElement::new` into arenas the host created, and its destructors run
//! when the host clears them.
//!
//! Every export catches panics and reports them as a status code, since a
//! panic unwinding out of an `extern "C"` function into the host aborts.

#[path = "../support/tagged_allocator.rs"]
mod tagged_allocator;

use std::{
    cell::RefCell,
    panic::{self, AssertUnwindSafe},
    sync::atomic::{AtomicU64, Ordering},
};

use gpui::{
    AnyElement, App, Arena, Bounds, Element, ElementArenaScope, ElementId, GlobalElementId,
    InspectorElementId, IntoElement, LayoutId, ParentElement, Pixels, SharedString, Styled, Window,
    div,
};

#[global_allocator]
static ALLOCATOR: tagged_allocator::TaggedAllocator =
    tagged_allocator::TaggedAllocator::new(tagged_allocator::PLUGIN_TAG);

pub const STATUS_OK: u32 = 0;
pub const STATUS_PANICKED: u32 = 1;
pub const NO_PANIC: u32 = u32::MAX;

/// Owns plugin-heap memory and counts its own destruction.
struct DropProbe {
    drops: &'static AtomicU64,
    payload: Vec<u64>,
    panic_on_drop: bool,
}

impl Drop for DropProbe {
    fn drop(&mut self) {
        assert!(self.payload.iter().all(|&value| value == self.payload[0]));
        self.drops.fetch_add(1, Ordering::SeqCst);
        if self.panic_on_drop {
            panic!("fixture probe destructor panicked on purpose");
        }
    }
}

impl IntoElement for DropProbe {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Element for DropProbe {
    type RequestLayoutState = ();
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _window: &mut Window,
        _cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        unreachable!("fixture elements are never laid out")
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        _window: &mut Window,
        _cx: &mut App,
    ) -> Self::PrepaintState {
        unreachable!("fixture elements are never laid out")
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        _prepaint: &mut Self::PrepaintState,
        _window: &mut Window,
        _cx: &mut App,
    ) {
        unreachable!("fixture elements are never laid out")
    }
}

/// Builds `count` small element trees: each is a `div` (with a heap-owning
/// child list) holding a `DropProbe` and a text element, so every index
/// makes three arena allocations, all through this DLL's copy of gpui.
fn build_elements(count: u32, drops: &'static AtomicU64, panic_index: u32) {
    for index in 0..count {
        let probe = DropProbe {
            drops,
            payload: vec![u64::from(index); 1 + index as usize % 64],
            panic_on_drop: index == panic_index,
        };
        let element: AnyElement = div()
            .flex()
            .child(probe)
            .child(SharedString::from(format!("fixture element {index}")))
            .into_any_element();
        // The handle doesn't own the value; the arena drops it on clear.
        drop(element);
    }
}

fn guarded(body: impl FnOnce()) -> u32 {
    match panic::catch_unwind(AssertUnwindSafe(body)) {
        Ok(()) => STATUS_OK,
        Err(_) => STATUS_PANICKED,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn fixture_runtime_address() -> usize {
    gpui::shared_runtime::runtime_address()
}

#[unsafe(no_mangle)]
pub extern "C" fn fixture_abi_version() -> u32 {
    gpui::shared_runtime::ABI_VERSION
}

#[unsafe(no_mangle)]
pub extern "C" fn fixture_module_slot() -> u32 {
    gpui::shared_runtime::module_slot()
}

#[unsafe(no_mangle)]
pub extern "C" fn fixture_live_heap_bytes() -> i64 {
    ALLOCATOR.live_bytes()
}

#[unsafe(no_mangle)]
pub extern "C" fn fixture_live_heap_blocks() -> i64 {
    ALLOCATOR.live_blocks()
}

/// The shared runtime's live allocation count, as seen from this copy.
#[unsafe(no_mangle)]
pub extern "C" fn fixture_live_allocations() -> u64 {
    gpui::shared_runtime::stats().live_allocations
}

/// Build elements into whatever arena is active on the calling thread,
/// without entering a scope of its own.
///
/// # Safety
/// `drops` must point to an `AtomicU64` that outlives every element built.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fixture_build_elements(
    count: u32,
    drops: *const AtomicU64,
    panic_index: u32,
) -> u32 {
    let drops = unsafe { &*drops };
    guarded(|| build_elements(count, drops, panic_index))
}

/// Run whole frames on an arena this DLL creates and owns, entering its own
/// scope (nested inside any scope the host holds). Writes the number of
/// arena allocations the last frame made to `allocations_per_frame`.
///
/// # Safety
/// `drops` must point to an `AtomicU64` that outlives the call, and
/// `allocations_per_frame` must be valid for writes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fixture_run_own_frames(
    frames: u32,
    count: u32,
    drops: *const AtomicU64,
    allocations_per_frame: *mut u64,
) -> u32 {
    let drops = unsafe { &*drops };
    guarded(|| {
        let arena = RefCell::new(Arena::new(16 * 1024));
        let mut last = 0;
        for _ in 0..frames {
            {
                let _scope = ElementArenaScope::enter(&arena);
                build_elements(count, drops, NO_PANIC);
            }
            last = arena.borrow().allocation_count() as u64;
            arena.borrow_mut().clear();
        }
        unsafe { allocations_per_frame.write(last) };
    })
}
