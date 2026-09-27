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

#[path = "../support/content_rng.rs"]
mod content_rng;
#[path = "../support/tagged_allocator.rs"]
mod tagged_allocator;

use std::{
    cell::RefCell,
    panic::{self, AssertUnwindSafe},
    sync::atomic::{AtomicU64, Ordering},
};

use content_rng::ContentRng;
use gpui::{
    AnyElement, AnyView, App, AppContext, Arena, Bounds, Context, Element, ElementArenaScope,
    ElementId, GlobalElementId, InspectorElementId, IntoElement, LayoutId, ParentElement, Pixels,
    Render, SharedString, Styled, Window, div, prelude::*, px, rgb,
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

/// A plugin-owned view: its entity lives in the host `App`, and its
/// `render`, listeners and element tree are all compiled into this DLL.
///
/// Everything it renders is generated from the seed the host last set with
/// `fixture_set_panel_seed`, so the host decides when the panel shows novel
/// content and when it shows a fixed, canonical frame.
struct PluginPanel;

static PANEL_SEED: AtomicU64 = AtomicU64::new(0);
static PANEL_RENDERS: AtomicU64 = AtomicU64::new(0);
static PANEL_CLICKS: AtomicU64 = AtomicU64::new(0);

struct PanelRow {
    id: ElementId,
    selected: bool,
    label: SharedString,
    cells: Vec<SharedString>,
}

fn nested_chain(depth: usize, max_depth: usize, rng: &mut ContentRng) -> AnyElement {
    if depth == max_depth {
        return div()
            .child(SharedString::from(rng.text(24)))
            .into_any_element();
    }
    let id: ElementId = if rng.one_in(3) {
        ("plugin-nest-fresh", rng.next()).into()
    } else {
        ("plugin-nest", depth).into()
    };
    div()
        .id(id)
        .pl(px(1.))
        .border_1()
        .border_color(rgb(0x3a3a3a))
        .hover(|style| style.bg(rgb(0x2a2a2a)))
        .child(nested_chain(depth + 1, max_depth, rng))
        .into_any_element()
}

impl Render for PluginPanel {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        PANEL_RENDERS.fetch_add(1, Ordering::SeqCst);
        // Mixed with the entity id so several panels on screen differ.
        let seed = PANEL_SEED.load(Ordering::SeqCst) ^ cx.entity_id().as_u64().rotate_left(32);
        let mut rng = ContentRng::new(seed);
        let header = SharedString::from(rng.text(40));
        let nesting = rng.below(60) as usize;
        let chain = nested_chain(0, nesting, &mut rng);
        let rows: Vec<PanelRow> = (0..rng.below(250) as usize)
            .map(|row| PanelRow {
                // A quarter of rows get an id never seen before, so their
                // element state must be created and then collected.
                id: if rng.one_in(4) {
                    ("plugin-row-fresh", rng.next()).into()
                } else {
                    ("plugin-row", row).into()
                },
                selected: rng.one_in(7),
                label: SharedString::from(rng.text(16)),
                cells: (0..rng.below(9))
                    .map(|_| SharedString::from(rng.text(6)))
                    .collect(),
            })
            .collect();

        div()
            .id("plugin-panel")
            .flex()
            .flex_col()
            .flex_1()
            .overflow_y_scroll()
            .bg(rgb(0x202020))
            .text_color(rgb(0xe0e0e0))
            .child(
                div()
                    .id("plugin-header")
                    .h(px(20.))
                    .hover(|style| style.bg(rgb(0x404040)))
                    .child(header),
            )
            .child(chain)
            .children(rows.into_iter().map(|row| {
                div()
                    .id(row.id)
                    .flex()
                    .flex_row()
                    .gap(px(2.))
                    .h(px(18.))
                    .border_1()
                    .border_color(rgb(0x505050))
                    .when(row.selected, |row| row.bg(rgb(0x304060)))
                    .hover(|style| style.bg(rgb(0x383838)))
                    .on_click(cx.listener(|_, _, _, cx| {
                        PANEL_CLICKS.fetch_add(1, Ordering::SeqCst);
                        cx.notify();
                    }))
                    .child(row.label)
                    .children(
                        row.cells
                            .into_iter()
                            .map(|cell| div().w(px(28.)).rounded_sm().child(cell)),
                    )
            }))
    }
}

/// Create a plugin view inside the host's `App`.
///
/// # Safety
/// `cx` must be the host's live `App`, and `out` valid for a write. Only
/// sound when this DLL was built with the same gpui version, features and
/// profile as the host, which is the contract real plugins live under too.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fixture_create_panel(cx: *mut App, out: *mut AnyView) -> u32 {
    let cx = unsafe { &mut *cx };
    guarded(|| {
        let panel: AnyView = cx.new(|_| PluginPanel).into();
        unsafe { out.write(panel) };
    })
}

/// Sets the seed the panels generate their next render from.
#[unsafe(no_mangle)]
pub extern "C" fn fixture_set_panel_seed(seed: u64) {
    PANEL_SEED.store(seed, Ordering::SeqCst);
}

/// How many times panels have rendered since this DLL was loaded.
#[unsafe(no_mangle)]
pub extern "C" fn fixture_panel_renders() -> u64 {
    PANEL_RENDERS.load(Ordering::SeqCst)
}

/// How many clicks panel rows have handled since this DLL was loaded.
#[unsafe(no_mangle)]
pub extern "C" fn fixture_panel_clicks() -> u64 {
    PANEL_CLICKS.load(Ordering::SeqCst)
}

#[unsafe(no_mangle)]
pub extern "C" fn fixture_set_strict_allocator(strict: bool) {
    ALLOCATOR.set_strict(strict);
}

/// # Safety
/// `out` must be valid for a write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fixture_ledger(out: *mut tagged_allocator::Ledger) {
    unsafe { out.write(ALLOCATOR.ledger()) };
}

/// Copies the backtrace of this DLL's first cross free (if any) into
/// `buffer`, returning its full length.
///
/// # Safety
/// `buffer` must be valid for `capacity` bytes of writes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fixture_first_foreign_free(buffer: *mut u8, capacity: usize) -> usize {
    let Some(text) = ALLOCATOR.first_foreign_free() else {
        return 0;
    };
    let copied = text.len().min(capacity);
    unsafe { std::ptr::copy_nonoverlapping(text.as_ptr(), buffer, copied) };
    text.len()
}

/// Copies this DLL's live-blocks-by-size histogram into `out`.
///
/// # Safety
/// `out` must be valid for `len` writes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fixture_size_histogram(out: *mut i64, len: usize) {
    let mut histogram = [0i64; tagged_allocator::HISTOGRAM_LEN];
    ALLOCATOR.size_histogram(&mut histogram);
    let copied = len.min(histogram.len());
    unsafe { std::ptr::copy_nonoverlapping(histogram.as_ptr(), out, copied) };
}
