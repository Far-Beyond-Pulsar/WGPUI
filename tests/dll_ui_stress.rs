//! End-to-end leak stress test across the plugin boundary.
//!
//! A real `App` and window draw a large host UI (a grid of interactive cells
//! and a column of child views) next to a complex plugin UI: view entities
//! created by the plugin DLL inside the host's `App`, whose `render`,
//! listeners and element trees all live in the DLL's own copy of gpui.
//!
//! The run alternates two phases:
//!
//! - **Churn.** Every frame, both sides render freshly generated random
//!   content, so the UI keeps entering states it has never been in: random
//!   cell and row counts, nesting depths and text, and element ids that are
//!   new every frame (so element state is created and must be collected).
//!   Host card views are replaced, extra plugin views are created and dropped,
//!   and the pointer moves and clicks at random.
//! - **Canonical checkpoint.** Identical frames of one fixed state, with no
//!   input, for longer than retained layers live, then the heap is measured.
//!
//! At every checkpoint the shared runtime must be exactly at rest (no live
//! arena values, no orphans, no open scopes, only the two expected modules and
//! arenas). The heap is the combined heap of host and plugin, tracked by
//! tagged allocators in both binaries, which balances exactly even though
//! values allocated by one copy are routinely freed by the other.
//!
//! The canonical heap cannot be required to be byte-identical: gpui keeps
//! some bounded, history-dependent state by design (layout nodes reused with
//! their earlier measurements, retained layers), which moves it by a few KiB
//! within a plugin load and by megabytes when the plugin's views are removed
//! and recreated. So a leak is detected the way it shows up: within each
//! plugin load the canonical heap may rise above noise at most once (a
//! collection reaching a new capacity) where a leak rises at every
//! checkpoint, and across reloads the per-load floor must keep coming back
//! down, which it could not if unloaded copies left memory behind.
//!
//! Run it longer with, for example:
//! `GPUI_STRESS_SECONDS=1800 cargo test --release --features test-support --test dll_ui_stress -- --ignored --nocapture`
//! `GPUI_STRESS_TRACE_ONLY=1` prints every checkpoint and which allocation
//! sizes changed, without asserting.

#[path = "support/content_rng.rs"]
mod content_rng;
#[path = "support/fixture_build.rs"]
mod fixture_build;
#[path = "support/tagged_allocator.rs"]
mod tagged_allocator;

use std::{
    mem::MaybeUninit,
    path::Path,
    time::{Duration, Instant},
};

use content_rng::ContentRng;
use fixture_build::{fixture_path, is_loaded};
use gpui::{
    AnyView, App, AppContext, Context, ElementId, Entity, IntoElement, LayerPolicy, Modifiers, ParentElement,
    Render, SharedString, Styled, TestAppContext, VisualTestContext, Window, div, point,
    prelude::*, px, rgb, shared_runtime,
};
use libloading::Library;
use tagged_allocator::{
    HISTOGRAM_LEN, HOST_TAG, Ledger, TaggedAllocator, bucket_label, combined_live_blocks,
    combined_live_bytes,
};

#[global_allocator]
static ALLOCATOR: TaggedAllocator = TaggedAllocator::new(HOST_TAG);

const HOST_CARDS: usize = 24;
const CHURN_FRAMES: u64 = 300;
const CANONICAL_SEED: u64 = 0x5eed_cafe_f00d_d00d;
const MAX_EXTRA_PANELS: usize = 3;

struct HostCard {
    seed: u64,
}

impl Render for HostCard {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        let mut rng = ContentRng::new(self.seed);
        let title = SharedString::from(rng.text(20));
        let lines: Vec<(bool, SharedString)> = (0..rng.below(24))
            .map(|_| (rng.one_in(5), SharedString::from(rng.text(30))))
            .collect();
        div()
            .id("host-card")
            .flex()
            .flex_col()
            .p(px(2.))
            .border_1()
            .border_color(rgb(0x606060))
            .rounded_md()
            .hover(|style| style.bg(rgb(0x303030)))
            .child(title)
            .children(lines.into_iter().map(|(highlighted, text)| {
                div()
                    .h(px(12.))
                    .when(highlighted, |line| line.bg(rgb(0x404020)))
                    .child(text)
            }))
    }
}

struct HostRoot {
    seed: u64,
    clicks: u64,
    cards: Vec<Entity<HostCard>>,
    panel: Option<AnyView>,
    extra_panels: Vec<AnyView>,
}

impl Render for HostRoot {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let mut rng = ContentRng::new(self.seed);
        let cells: Vec<(ElementId, bool, SharedString)> = (0..600 + rng.below(900) as usize)
            .map(|cell| {
                // A quarter of cells get an id never seen before, so their
                // element state must be created and then collected.
                let id: ElementId = if rng.one_in(4) {
                    ("host-cell-fresh", rng.next()).into()
                } else {
                    ("host-cell", cell).into()
                };
                (id, rng.one_in(11), SharedString::from(rng.text(6)))
            })
            .collect();
        div()
            .id("host-root")
            .size_full()
            .flex()
            .flex_row()
            .bg(rgb(0x101010))
            .text_color(rgb(0xd0d0d0))
            .child(
                div()
                    .id("host-grid")
                    .w(px(1_100.))
                    .flex()
                    .flex_wrap()
                    .overflow_y_scroll()
                    .children(cells.into_iter().map(|(id, selected, text)| {
                        div()
                            .id(id)
                            .min_w(px(22.))
                            .h(px(16.))
                            .border_1()
                            .border_color(rgb(0x404040))
                            .when(selected, |cell| cell.bg(rgb(0x204020)))
                            .hover(|style| style.bg(rgb(0x505050)))
                            .on_click(cx.listener(|root, _, _, cx| {
                                root.clicks += 1;
                                cx.notify();
                            }))
                            .child(text)
                    })),
            )
            .child(
                div()
                    .id("host-cards")
                    .w(px(300.))
                    .flex()
                    .flex_col()
                    .overflow_y_scroll()
                    .children(self.cards.iter().cloned()),
            )
            .child(
                div()
                    .flex_1()
                    .h_full()
                    .flex()
                    .flex_col()
                    .children(self.panel.clone())
                    .children(self.extra_panels.iter().cloned()),
            )
    }
}

struct Plugin {
    create_panel: unsafe extern "C" fn(*mut App, *mut AnyView) -> u32,
    set_panel_seed: extern "C" fn(u64),
    panel_renders: extern "C" fn() -> u64,
    panel_clicks: extern "C" fn() -> u64,
    set_strict_allocator: extern "C" fn(bool),
    ledger: unsafe extern "C" fn(*mut Ledger),
    first_foreign_free: unsafe extern "C" fn(*mut u8, usize) -> usize,
    size_histogram: unsafe extern "C" fn(*mut i64, usize),
    runtime_address: extern "C" fn() -> usize,
    library: Library,
}

impl Plugin {
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
        let plugin = unsafe {
            Self {
                create_panel: symbol(&library, "fixture_create_panel"),
                set_panel_seed: symbol(&library, "fixture_set_panel_seed"),
                panel_renders: symbol(&library, "fixture_panel_renders"),
                panel_clicks: symbol(&library, "fixture_panel_clicks"),
                set_strict_allocator: symbol(&library, "fixture_set_strict_allocator"),
                ledger: symbol(&library, "fixture_ledger"),
                first_foreign_free: symbol(&library, "fixture_first_foreign_free"),
                size_histogram: symbol(&library, "fixture_size_histogram"),
                runtime_address: symbol(&library, "fixture_runtime_address"),
                library,
            }
        };
        // Plugin views put gpui's own Rust values (entities, listener
        // closures, element state, the host's Vecs and maps) on both sides of
        // the boundary, and Rust frees and reallocates with the allocator of
        // whichever copy of the code touches them. Both tagged allocators sit
        // on the same `System` heap, so that is physically sound here; ledger
        // mode records it so the combined heap still balances exactly.
        (plugin.set_strict_allocator)(false);
        assert_eq!((plugin.runtime_address)(), shared_runtime::runtime_address());
        plugin
    }

    fn ledger(&self) -> Ledger {
        let mut ledger = Ledger::default();
        unsafe { (self.ledger)(&mut ledger) };
        ledger
    }

    fn first_foreign_free(&self) -> Option<String> {
        let mut buffer = vec![0u8; 16 * 1024];
        let length = unsafe { (self.first_foreign_free)(buffer.as_mut_ptr(), buffer.len()) };
        (length > 0).then(|| {
            buffer.truncate(length.min(buffer.len()));
            String::from_utf8_lossy(&buffer).into_owned()
        })
    }

    fn create_panel(&self, cx: &mut App) -> AnyView {
        let mut panel = MaybeUninit::<AnyView>::uninit();
        let status = unsafe { (self.create_panel)(cx, panel.as_mut_ptr()) };
        assert_eq!(status, 0, "the plugin panicked while creating a panel");
        unsafe { panel.assume_init() }
    }
}

/// Everything that must be identical at every canonical checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Snapshot {
    combined_bytes: i64,
    combined_blocks: i64,
    live_arenas: u64,
    arena_headers: u64,
    live_chunks: u64,
    live_chunk_bytes: u64,
    live_record_bytes: u64,
    live_allocations: u64,
    orphaned_records: u64,
    attached_modules: u64,
    active_threads: u64,
}

/// Physical bytes/blocks left behind by plugin copies already unloaded, as
/// their ledgers read just before unloading.
#[derive(Default)]
struct Retired {
    bytes: i64,
    blocks: i64,
}

fn snapshot(plugin: &Plugin, retired: &Retired) -> Snapshot {
    let ledgers = [ALLOCATOR.ledger(), plugin.ledger()];
    let runtime = shared_runtime::stats();
    Snapshot {
        combined_bytes: combined_live_bytes(&ledgers) + retired.bytes,
        combined_blocks: combined_live_blocks(&ledgers) + retired.blocks,
        live_arenas: runtime.live_arenas,
        arena_headers: runtime.arena_headers,
        live_chunks: runtime.live_chunks,
        live_chunk_bytes: runtime.live_chunk_bytes,
        live_record_bytes: runtime.live_record_bytes,
        live_allocations: runtime.live_allocations,
        orphaned_records: runtime.orphaned_records,
        attached_modules: runtime.attached_modules,
        active_threads: runtime.active_threads,
    }
}

/// Live blocks by size across the host and the loaded plugin.
fn combined_histogram(plugin: &Plugin) -> Vec<i64> {
    let mut host = [0i64; HISTOGRAM_LEN];
    ALLOCATOR.size_histogram(&mut host);
    let mut guest = vec![0i64; HISTOGRAM_LEN];
    unsafe { (plugin.size_histogram)(guest.as_mut_ptr(), guest.len()) };
    host.iter().zip(&guest).map(|(host, guest)| host + guest).collect()
}

/// The sizes whose live block counts changed, largest bytes first.
fn histogram_diff(previous: &[i64], current: &[i64]) -> String {
    let mut changes: Vec<(usize, i64)> = previous
        .iter()
        .zip(current)
        .enumerate()
        .filter(|(_, (before, after))| before != after)
        .map(|(index, (before, after))| (index, after - before))
        .collect();
    changes.sort_by_key(|&(index, delta)| std::cmp::Reverse((delta * index.max(1) as i64).abs()));
    let summary: Vec<String> = changes
        .iter()
        .take(12)
        .map(|&(index, delta)| format!("{}: {delta:+}", bucket_label(index)))
        .collect();
    format!("{} sizes changed: {}", changes.len(), summary.join(", "))
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn draw(cx: &mut VisualTestContext, root: &Entity<HostRoot>, plugin: &Plugin, seed: u64) {
    (plugin.set_panel_seed)(seed);
    cx.update(|window, cx| {
        let (cards, panels) = root.update(cx, |root, cx| {
            root.seed = seed;
            cx.notify();
            let panels: Vec<AnyView> = root
                .panel
                .iter()
                .chain(&root.extra_panels)
                .cloned()
                .collect();
            (root.cards.clone(), panels)
        });
        for (index, card) in cards.iter().enumerate() {
            card.update(cx, |card, cx| {
                card.seed = seed ^ (index as u64).wrapping_mul(0x2545_f491_4f6c_dd1d);
                cx.notify();
            });
        }
        for panel in panels {
            cx.notify(panel.entity_id());
        }
        window.refresh();
        window.draw(cx).clear();
    });
}

/// One frame of novel content, entity churn, and random input.
fn churn_frame(
    cx: &mut VisualTestContext,
    root: &Entity<HostRoot>,
    plugin: &Plugin,
    rng: &mut ContentRng,
) {
    if rng.one_in(10) {
        let index = rng.below(HOST_CARDS as u64) as usize;
        cx.update(|_, cx| {
            let card = cx.new(|_| HostCard { seed: 0 });
            root.update(cx, |root, _| {
                if let Some(slot) = root.cards.get_mut(index) {
                    *slot = card;
                }
            });
        });
    }
    if rng.one_in(40) {
        cx.update(|_, cx| {
            let panel = plugin.create_panel(cx);
            root.update(cx, |root, _| {
                if root.extra_panels.len() == MAX_EXTRA_PANELS {
                    root.extra_panels.remove(0);
                }
                root.extra_panels.push(panel);
            });
        });
    }

    draw(cx, root, plugin, rng.next());

    cx.simulate_mouse_move(
        point(px(rng.below(1_900) as f32), px(rng.below(1_000) as f32)),
        None,
        Modifiers::none(),
    );
    if rng.one_in(6) {
        cx.simulate_click(
            point(px(rng.below(1_900) as f32), px(rng.below(1_000) as f32)),
            Modifiers::none(),
        );
    }
}

/// Retained layers keep their content for `evict_after_frames` unvisited
/// frames and their record for twice that (`Window::evict_stale_layers`), so
/// layers created by churn content only disappear after that many frames of
/// something else.
fn canonical_frame_count() -> u32 {
    LayerPolicy::default().evict_after_frames * 2 + 10
}

/// Enough identical frames of the canonical state, with no input, that every
/// frame-scoped structure and every retained layer holds only canonical
/// content.
fn canonical_frames(cx: &mut VisualTestContext, root: &Entity<HostRoot>, plugin: &Plugin) {
    cx.update(|_, cx| root.update(cx, |root, _| root.extra_panels.clear()));
    cx.simulate_mouse_move(point(px(2.), px(2.)), None, Modifiers::none());
    for _ in 0..canonical_frame_count() {
        draw(cx, root, plugin, CANONICAL_SEED);
    }
    cx.run_until_parked();
}

/// Take every plugin view out of the UI and let every trace of them (their
/// entities, element state, listeners, and the frames that referenced them)
/// age out, so the plugin's code can be unmapped.
fn remove_plugin_views(cx: &mut VisualTestContext, root: &Entity<HostRoot>, plugin: &Plugin) {
    cx.update(|_, cx| {
        root.update(cx, |root, _| {
            root.panel = None;
            root.extra_panels.clear();
        })
    });
    canonical_frames(cx, root, plugin);
}

/// Within one plugin load, bounded caches with history-dependent occupancy
/// (layout node reuse, retained layers) move the canonical heap by a few KiB
/// between checkpoints; a rise larger than this is real growth.
const NOISE_BYTES: i64 = 8 * 1024;
/// A collection growing to a new high-water capacity raises the heap once, by
/// up to a few hundred KiB (observed: one +274 KiB step in a 30 minute run).
/// A leak instead rises at every checkpoint: at this noise floor, anything
/// over ~27 bytes per frame (300 frames per checkpoint) rises at every one.
const MAX_RISES_PER_LOAD: usize = 1;
/// The most a load's canonical heap may move in total, capacity steps
/// included.
const MAX_BAND_BYTES: i64 = 1024 * 1024;
/// Removing and re-adding the plugin's views creates new layer and element
/// identities, which shifts the canonical heap by megabytes in either
/// direction per reload. A copy leaving memory behind would instead raise
/// every later load's floor, so later floors must come back down to earlier
/// ones.
const MAX_RELOAD_FLOOR_RISE_BYTES: i64 = 1024 * 1024;

#[test]
#[ignore = "runs for many minutes; run explicitly with --ignored (see the module docs)"]
fn giant_host_and_plugin_uis_churned_nonstop_leak_nothing() {
    let seconds = env_u64("GPUI_STRESS_SECONDS", 600);
    let settle = env_u64("GPUI_STRESS_SETTLE_CHECKPOINTS", 2) as usize;
    let reload_every = env_u64("GPUI_STRESS_RELOAD_CHECKPOINTS", 6);
    let trace_only = env_u64("GPUI_STRESS_TRACE_ONLY", 0) != 0;
    // Diagnostics: attribute blocks of this size allocated during churn that
    // survive the canonical frames (only host-allocated blocks are seen).
    let watch_size = env_u64("GPUI_STRESS_WATCH_SIZE", 0) as usize;
    let deadline = Instant::now() + Duration::from_secs(seconds);
    // See `Plugin::load` for why cross frees are recorded, not fatal, here.
    ALLOCATOR.set_strict(false);
    if watch_size != 0 {
        ALLOCATOR.watch(watch_size);
    }

    let mut app = TestAppContext::single();
    let (root, cx) = app.add_window_view(|_, cx| HostRoot {
        seed: 0,
        clicks: 0,
        cards: (0..HOST_CARDS)
            .map(|_| cx.new(|_| HostCard { seed: 0 }))
            .collect(),
        panel: None,
        extra_panels: Vec::with_capacity(MAX_EXTRA_PANELS),
    });

    let mut retired = Retired::default();
    let mut plugin = Plugin::load(fixture_path());
    let panel = cx.update(|_, cx| plugin.create_panel(cx));
    cx.update(|_, cx| root.update(cx, |root, _| root.panel = Some(panel)));

    let mut rng = ContentRng::new(0x0123_4567_89ab_cdef);
    // Preallocated: the test's own bookkeeping must not move the heap it
    // measures.
    let mut canonical_bytes: Vec<(u32, i64)> = Vec::with_capacity(16_384);
    let mut previous_histogram: Option<Vec<i64>> = None;
    let mut checkpoints_since_load = 0u64;
    let mut churn_frames = 0u64;
    let mut reloads = 0u32;
    let mut max_chunks = 0;
    let mut slowest_frame = Duration::ZERO;
    let started = Instant::now();

    loop {
        let churn_started = Instant::now();
        let renders_before = (plugin.panel_renders)();
        ALLOCATOR.set_watch_recording(watch_size != 0);
        for _ in 0..CHURN_FRAMES {
            let frame_started = Instant::now();
            churn_frame(cx, &root, &plugin, &mut rng);
            slowest_frame = slowest_frame.max(frame_started.elapsed());
        }
        ALLOCATOR.set_watch_recording(false);
        churn_frames += CHURN_FRAMES;
        let churn_fps = CHURN_FRAMES as f64 / churn_started.elapsed().as_secs_f64();
        let plugin_renders = (plugin.panel_renders)() - renders_before;

        canonical_frames(cx, &root, &plugin);
        checkpoints_since_load += 1;
        let snapshot = snapshot(&plugin, &retired);
        // Nothing runs between these two reads, so if they differ another
        // thread is allocating and the noise is in the measurement.
        assert_eq!(
            self::snapshot(&plugin, &retired),
            snapshot,
            "the heap changed between two back-to-back reads"
        );
        // Between frames the shared runtime must be exactly at rest, however
        // much churn came before.
        assert_eq!(snapshot.live_allocations, 0, "arena values outlived their frame");
        assert_eq!(snapshot.orphaned_records, 0, "arena values were orphaned");
        assert_eq!(snapshot.active_threads, 0, "an element arena scope was left open");
        assert_eq!(snapshot.attached_modules, 2, "host and plugin should be the only copies");
        assert_eq!(snapshot.live_arenas, 2, "only the app's element and event arenas should exist");
        max_chunks = max_chunks.max(snapshot.live_chunks);
        canonical_bytes.push((reloads, snapshot.combined_bytes));

        #[cfg(windows)]
        let private_bytes = fixture_build::process_private_bytes();
        #[cfg(not(windows))]
        let private_bytes = 0;
        eprintln!(
            "load {reloads} checkpoint {checkpoints_since_load:3} after {churn_frames} churn frames \
             ({churn_fps:.0} fps, {plugin_renders} plugin renders): canonical heap {} bytes / {} \
             blocks, arena chunks {}, process private bytes {private_bytes}",
            snapshot.combined_bytes, snapshot.combined_blocks, snapshot.live_chunks,
        );
        if watch_size != 0 {
            for (count, site) in ALLOCATOR.watch_report(14).into_iter().take(4) {
                eprintln!("    {count} surviving {watch_size}-byte blocks from churn, allocated at:\n{site}\n");
            }
        }
        if trace_only {
            let histogram = combined_histogram(&plugin);
            if let Some(previous_histogram) = &previous_histogram {
                eprintln!("    {}", histogram_diff(previous_histogram, &histogram));
            }
            previous_histogram = Some(histogram);
        }

        if Instant::now() >= deadline {
            break;
        }

        if checkpoints_since_load >= reload_every {
            remove_plugin_views(cx, &root, &plugin);
            let ledger = plugin.ledger();
            retired.bytes += ledger.live_bytes - ledger.foreign_freed_bytes;
            retired.blocks += ledger.live_blocks - ledger.foreign_freed_blocks;
            let path = fixture_path();
            plugin.library.close().expect("failed to unload the plugin");
            assert!(!is_loaded(path), "the plugin stayed mapped after unloading");

            plugin = Plugin::load(path);
            let panel = cx.update(|_, cx| plugin.create_panel(cx));
            cx.update(|_, cx| root.update(cx, |root, _| root.panel = Some(panel)));
            reloads += 1;
            checkpoints_since_load = 0;
        }
    }

    let elapsed = started.elapsed();
    let mut loads: Vec<Vec<i64>> = Vec::new();
    for &(load, bytes) in &canonical_bytes {
        let load = load as usize;
        if loads.len() <= load {
            loads.resize_with(load + 1, Vec::new);
        }
        if let Some(points) = loads.get_mut(load) {
            points.push(bytes);
        }
    }
    let settled: Vec<&[i64]> = loads
        .iter()
        .map(|points| points.get(settle..).unwrap_or(&[]))
        .collect();
    // (load, rises above noise between consecutive checkpoints, band) for
    // every load long enough to judge.
    let fits: Vec<(usize, usize, i64)> = settled
        .iter()
        .enumerate()
        .filter(|(_, points)| points.len() >= 3)
        .map(|(load, points)| {
            let rises = points
                .windows(2)
                .filter(|pair| pair[1] - pair[0] > NOISE_BYTES)
                .count();
            let band = points.iter().max().zip(points.iter().min()).map_or(0, |(max, min)| max - min);
            (load, rises, band)
        })
        .collect();
    let floors: Vec<i64> = settled.iter().filter_map(|points| points.iter().min().copied()).collect();
    let (early_floor, late_floor) = if floors.len() >= 4 {
        let (early, late) = floors.split_at(floors.len() / 2);
        (early.iter().min().copied(), late.iter().min().copied())
    } else {
        (None, None)
    };
    let host_clicks = cx.update(|_, cx| root.read(cx).clicks);
    let plugin_clicks = (plugin.panel_clicks)();
    let host_ledger = ALLOCATOR.ledger();
    let plugin_ledger = plugin.ledger();
    eprintln!(
        "\n=== dll_ui_stress ===\n\
         {churn_frames} churn frames in {:.1}s, slowest frame {:.1}ms, {reloads} plugin reloads\n\
         per-load canonical heap (load, rises over {NOISE_BYTES} bytes, band bytes): {fits:?}\n\
         per-load canonical heap floors: {floors:?}\n\
         arena chunks peaked at {max_chunks}\n\
         clicks handled: host {host_clicks}, plugin {plugin_clicks}\n\
         runtime: {:?}\n\
         cross-binary frees (sound here, both allocators share one heap): \
         host freed {} plugin blocks ({} bytes); plugin freed {} host blocks ({} bytes)",
        elapsed.as_secs_f64(),
        slowest_frame.as_secs_f64() * 1_000.,
        shared_runtime::stats(),
        host_ledger.foreign_freed_blocks,
        host_ledger.foreign_freed_bytes,
        plugin_ledger.foreign_freed_blocks,
        plugin_ledger.foreign_freed_bytes,
    );
    if let Some(trace) = ALLOCATOR.first_foreign_free() {
        eprintln!("first plugin block freed by the host:\n{trace}");
    }
    if let Some(trace) = plugin.first_foreign_free() {
        eprintln!("first host block freed by the plugin:\n{trace}");
    }
    #[cfg(windows)]
    eprintln!("process private bytes at the end: {}", fixture_build::process_private_bytes());

    if !trace_only {
        assert!(
            !fits.is_empty(),
            "no plugin load lasted {} checkpoints; run longer (GPUI_STRESS_SECONDS)",
            settle + 3
        );
        for &(load, rises, band) in &fits {
            assert!(
                rises <= MAX_RISES_PER_LOAD,
                "during plugin load {load} the canonical heap rose by more than {NOISE_BYTES} \
                 bytes at {rises} checkpoints: a leak"
            );
            assert!(
                band <= MAX_BAND_BYTES,
                "during plugin load {load} the canonical heap moved by {band} bytes"
            );
        }
        if let (Some(early), Some(late)) = (early_floor, late_floor) {
            assert!(
                late <= early + MAX_RELOAD_FLOOR_RISE_BYTES,
                "the canonical heap floor rose from {early} to {late} bytes across plugin \
                 reloads: unloaded copies are leaving memory behind"
            );
        }
        assert!(host_clicks > 0, "no click reached a host listener");
        assert!(plugin_clicks > 0, "no click reached a plugin listener");
    }

    remove_plugin_views(cx, &root, &plugin);
    plugin.library.close().expect("failed to unload the plugin");
    app.quit();
}
