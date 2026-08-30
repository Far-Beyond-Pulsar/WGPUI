//! CPU-side frame-pipeline A/B measurement for the Phase 9 slab work.
//!
//! Builds a deterministic editor-like UI (~880 elements: sidebar list, an
//! animated `.layer()` viewport with ~690 quads, a stable-keyed `.layer_keyed`
//! chrome overlay, shadowed cards, status bar) inside the headless
//! `#[gpui::test]` TestWindow pipeline — no wgpu — so every measured
//! millisecond is CPU pipeline work (layout / prepaint / paint / scene
//! finish), never GPU submission.
//!
//! Two profiles are timed separately over fixed frame counts:
//! - DIRTY: every frame notifies the root view, so the viewport layer
//!   re-records and the keyed chrome layer must still composite across a
//!   notify.
//! - IDLE: no invalidation between draws; both layers take their composite
//!   paths (slab splice when slabs are on, retained replay otherwise).
//!
//! Wall-clock comes from `std::time::Instant` wrapped tightly around
//! `Window::draw`; per-stage attribution comes from `render_stats`
//! force-enabled accumulators, drained via `reset()`/`snapshot()` around each
//! measured stretch so counts/totals/maxes describe exactly that stretch.
//!
//! # Configuration is per process
//!
//! `WGPUI_SLABS` / `WGPUI_LAYERS` / `WGPUI_INSTANCES` feed production
//! `LazyLock`s that resolve at first use, so one invocation measures exactly
//! one configuration. Reproduce (PowerShell):
//!
//! ```text
//! cargo test --locked --lib perf_ab -- --ignored --nocapture --test-threads=1
//! $env:WGPUI_SLABS='0'; cargo test --locked --lib perf_ab -- --ignored --nocapture --test-threads=1
//! $env:WGPUI_INSTANCES='0'; $env:WGPUI_LAYERS='0'; $env:WGPUI_SLABS='1'; cargo test --locked --lib perf_ab -- --ignored --nocapture --test-threads=1
//! Remove-Item Env:WGPUI_SLABS,Env:WGPUI_LAYERS,Env:WGPUI_INSTANCES -ErrorAction SilentlyContinue
//! ```
//!
//! `--test-threads=1` is required: force-enabling render stats makes this
//! test's stage timers visible to any concurrently-running sibling test that
//! draws, which would pollute the windows below. Frame counts can be shrunk
//! for smoke runs with `WGPUI_PERF_FRAMES` / `WGPUI_PERF_WARMUP`.
//!
//! Instrumentation overhead (an atomic load plus a mutex-guarded accumulate
//! per record) applies identically to both configurations, so A/B deltas stay
//! valid; absolute values carry a small inflation.

use std::time::{Duration, Instant};

use crate::{
    Entity, FontWeight, TestAppContext, VisualTestContext, Window, div, hsla, prelude::*, px,
    relative, render_stats, size,
};

const WINDOW_WIDTH: f32 = 1440.;
const WINDOW_HEIGHT: f32 = 900.;
const GRID_COLS: usize = 30;
const GRID_ROWS: usize = 23;
const SIDEBAR_ROWS: usize = 40;
const CARD_COUNT: usize = 4;
/// Content key for the chrome overlay: stable across frames so the layer
/// composites even while its view is notified every dirty frame.
const CHROME_KEY: u64 = 0x5AB09;

const STAGE_NAMES: &[&str] = &[
    "frame: layout",
    "frame: prepaint",
    "frame: paint",
    "frame: bounds tree",
    "frame: scene finish",
    "frame: render",
    "frame: text shaping",
    "layer: composite",
];

const COUNTER_NAMES: &[&str] = &[
    "layer: created",
    "layer: re-rendered",
    "layer: composited",
    "layer: composited (slab)",
    "layer: composited (transform-only)",
    "slab: layers packed",
    "slab: bytes uploaded",
    "frame: primitives emitted (quad)",
    "frame: taffy nodes created",
    "instance: reused",
    "instance: rebuilt",
];

/// The root view. `frame_counter` drives every animated value; it only moves
/// when the harness explicitly bumps it, so frames are fully deterministic.
struct EditorPerfView {
    frame_counter: u64,
}

impl crate::Render for EditorPerfView {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        let tick = self.frame_counter;
        div()
            .size_full()
            .flex()
            .flex_col()
            .bg(hsla(0.62, 0.15, 0.07, 1.))
            .text_color(hsla(0., 0., 0.85, 1.))
            .child(header_bar())
            .child(
                div()
                    .flex_1()
                    .min_h_0()
                    .flex()
                    .flex_row()
                    .child(sidebar(tick))
                    .child(central_region(tick)),
            )
            .child(cards_row())
            .child(status_bar(tick))
    }
}

fn header_bar() -> impl IntoElement {
    div()
        .h(px(44.))
        .flex()
        .flex_row()
        .items_center()
        .gap(px(16.))
        .px(px(14.))
        .border_b_1()
        .border_color(hsla(0., 0., 0.5, 0.25))
        .bg(hsla(0.62, 0.2, 0.11, 1.))
        .child(
            div()
                .size(px(18.))
                .rounded_sm()
                .bg(hsla(0.55, 0.7, 0.55, 1.)),
        )
        .child(
            div()
                .text_base()
                .font_weight(FontWeight::MEDIUM)
                .child("Pulsar — workspace"),
        )
        .children(["File", "Edit", "View", "Run"].map(|menu| {
            div().text_sm().text_color(hsla(0., 0., 0.65, 1.)).child(menu)
        }))
}

fn sidebar(tick: u64) -> impl IntoElement {
    let selected = (tick as usize) % SIDEBAR_ROWS;
    div()
        .w(px(232.))
        .h_full()
        .flex()
        .flex_col()
        .py(px(6.))
        .border_r_1()
        .border_color(hsla(0., 0., 0.5, 0.2))
        .bg(hsla(0.62, 0.18, 0.09, 1.))
        .overflow_hidden()
        .children((0..SIDEBAR_ROWS).map(move |row| {
            let selected = row == selected;
            let hue = (row as f32 * 0.13) % 1.;
            div()
                .h(px(22.))
                .mx(px(6.))
                .px(px(8.))
                .mb(px(1.))
                .flex()
                .flex_row()
                .items_center()
                .gap(px(8.))
                .rounded_sm()
                .when(selected, |row| row.bg(hsla(0.58, 0.55, 0.3, 0.9)))
                .child(
                    div()
                        .size(px(10.))
                        .rounded_sm()
                        .bg(hsla(hue, 0.5, 0.5, 0.85)),
                )
                .child(format!("module_{row:02}.rs"))
        }))
}

fn central_region(tick: u64) -> impl IntoElement {
    div()
        .flex_1()
        .h_full()
        .relative()
        .overflow_hidden()
        .bg(hsla(0.6, 0.12, 0.05, 1.))
        .child(viewport_layer(tick))
        .child(chrome_overlay())
}

fn viewport_layer(tick: u64) -> impl IntoElement {
    div()
        .id("viewport")
        .layer()
        .absolute()
        .inset_0()
        .flex()
        .flex_wrap()
        .content_start()
        .gap(px(2.))
        .p(px(6.))
        .children((0..GRID_COLS * GRID_ROWS).map(|index| grid_cell(index, tick)))
}

fn grid_cell(index: usize, tick: u64) -> impl IntoElement {
    // Integer hash arithmetic keeps every frame's description a pure function
    // of (index, tick): no clocks, no RNG, identical trees across configs.
    let wobble =
        ((index as u64).wrapping_mul(2654435761).wrapping_add(tick.wrapping_mul(97)) % 7) as f32
            - 3.;
    let hue = ((index as u64).wrapping_mul(37).wrapping_add(tick.wrapping_mul(11)) % 100) as f32
        / 100.;
    let side = if (index as u64 + tick).is_multiple_of(11) {
        22.
    } else {
        18.
    };
    div()
        .relative()
        .w(px(side))
        .h(px(side))
        .left(px(wobble))
        .top(px(wobble * 0.5))
        .rounded_sm()
        .bg(hsla(hue, 0.55, 0.45, 0.92))
        .when(index.is_multiple_of(5), |cell| {
            cell.border_1()
                .border_color(hsla(hue, 0.8, 0.75, 0.9))
        })
}

fn chrome_overlay() -> impl IntoElement {
    div()
        .id("chrome")
        .layer_keyed(CHROME_KEY)
        .absolute()
        .top(px(12.))
        .right(px(12.))
        .w(px(244.))
        .p(px(12.))
        .flex()
        .flex_col()
        .gap(px(6.))
        .rounded_md()
        .border_1()
        .border_color(hsla(0., 0., 0.6, 0.35))
        .bg(hsla(0., 0., 0.06, 0.88))
        .shadow_md()
        .child(
            div()
                .text_sm()
                .font_weight(FontWeight::MEDIUM)
                .child("VIEWPORT CHROME"),
        )
        .child(
            div()
                .text_xs()
                .text_color(hsla(0., 0., 0.6, 1.))
                .child("grid: 30 x 23 quads"),
        )
        .child(
            div()
                .text_xs()
                .text_color(hsla(0., 0., 0.6, 1.))
                .child("keyed layer: composites across notify"),
        )
        .child(
            div()
                .flex()
                .flex_row()
                .gap(px(4.))
                .children((0..4).map(|swatch| {
                    div().size(px(12.)).rounded_sm().bg(hsla(
                        swatch as f32 * 0.25,
                        0.7,
                        0.5,
                        1.,
                    ))
                })),
        )
        .child(
            div()
                .h(px(1.))
                .w_full()
                .bg(hsla(0., 0., 0.5, 0.3)),
        )
        .child(
            div()
                .text_xs()
                .text_color(hsla(0., 0., 0.6, 1.))
                .child("borders · labels · legend"),
        )
}

fn cards_row() -> impl IntoElement {
    div().flex().flex_row().gap(px(12.)).px(px(12.)).py(px(10.)).children(
        (0..CARD_COUNT).map(|card_index| card(card_index)),
    )
}

fn card(index: usize) -> impl IntoElement {
    let fill = relative(((index + 1) as f32) * 0.18);
    div()
        .flex_1()
        .rounded_lg()
        .border_1()
        .border_color(hsla(0., 0., 0.5, 0.25))
        .bg(hsla(0.62, 0.18, 0.1, 1.))
        .shadow_md()
        .p(px(10.))
        .flex()
        .flex_col()
        .gap(px(6.))
        .child(format!("pipeline stage {}", index + 1))
        .child(
            div()
                .text_xs()
                .text_color(hsla(0., 0., 0.55, 1.))
                .child("retained · packed · composited"),
        )
        .child(
            div()
                .h(px(6.))
                .w_full()
                .rounded_full()
                .bg(hsla(0., 0., 0.25, 1.))
                .child(
                    div()
                        .h_full()
                        .w(fill)
                        .rounded_full()
                        .bg(hsla(0.55, 0.65, 0.5, 1.)),
                ),
        )
        .child(stat_row("budget", "16.6 ms"))
        .child(stat_row("mode", "headless"))
}

fn stat_row(label: &'static str, value: &'static str) -> impl IntoElement {
    div()
        .flex()
        .flex_row()
        .justify_between()
        .text_xs()
        .child(
            div()
                .text_color(hsla(0., 0., 0.55, 1.))
                .child(label),
        )
        .child(value)
}

fn status_bar(tick: u64) -> impl IntoElement {
    div()
        .h(px(26.))
        .flex()
        .flex_row()
        .items_center()
        .gap(px(14.))
        .px(px(14.))
        .border_t_1()
        .border_color(hsla(0., 0., 0.5, 0.25))
        .bg(hsla(0.62, 0.2, 0.11, 1.))
        .text_xs()
        .child(format!("frame {tick:06}"))
        .children(["UTF-8", "Ln 42, Col 7", "Rust", "main*", "slab A/B"].map(|item| {
            div()
                .text_color(hsla(0., 0., 0.6, 1.))
                .child(item)
        }))
}

fn estimated_element_count() -> usize {
    // header(9) + main row(1) + sidebar(1 + rows*3) + central container(1)
    // + viewport layer(1 + cells) + chrome(1 + 7 children incl. swatch row
    // internals) + cards(1 + count*8) + status(1 + 6).
    9 + 1 + 1 + SIDEBAR_ROWS * 3 + 1 + 1 + GRID_COLS * GRID_ROWS + 9 + 1 + CARD_COUNT * 8 + 7
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_display(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| "<unset>".into())
}

fn counter(snapshot: &render_stats::Snapshot, name: &str) -> u64 {
    snapshot
        .counters
        .iter()
        .find(|(key, _)| **key == name)
        .map(|(_, value)| *value)
        .unwrap_or(0)
}

struct WallStats {
    mean_ms: f64,
    median_ms: f64,
    p95_ms: f64,
    max_ms: f64,
}

fn wall_stats(samples: &[Duration]) -> WallStats {
    let mut sorted: Vec<f64> = samples.iter().map(|d| d.as_secs_f64() * 1e3).collect();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let percentile = |fraction: f64| -> f64 {
        let index = ((sorted.len() - 1) as f64 * fraction).round() as usize;
        sorted[index]
    };
    WallStats {
        mean_ms: sorted.iter().sum::<f64>() / sorted.len() as f64,
        median_ms: percentile(0.5),
        p95_ms: percentile(0.95),
        max_ms: sorted[sorted.len() - 1],
    }
}

fn timed_draw(cx: &mut VisualTestContext) -> Duration {
    let mut elapsed = Duration::ZERO;
    cx.update(|window, cx| {
        let start = Instant::now();
        window.draw(cx).clear();
        elapsed = start.elapsed();
    });
    elapsed
}

/// One dirty frame, the way a live app produces it: the notify lands inside
/// an `App::update`, whose trailing `flush_effects` draws the invalidated
/// window before returning. Timing the update therefore captures exactly one
/// pipeline pass — issuing a separate `Window::draw` afterwards would measure
/// a second, already-idle frame.
fn timed_dirty_frame(view: &Entity<EditorPerfView>, cx: &mut VisualTestContext) -> Duration {
    let start = Instant::now();
    view.update(cx, |view, cx| {
        view.frame_counter += 1;
        cx.notify();
    });
    start.elapsed()
}

fn report_profile(
    label: &str,
    frames: usize,
    samples: &[Duration],
    snapshot: &render_stats::Snapshot,
) {
    let wall = wall_stats(samples);
    eprintln!(
        "---- {label}: {frames} frames ----"
    );
    eprintln!(
        "wall frame : n={}  mean={:.3}ms  median={:.3}ms  p95={:.3}ms  max={:.3}ms",
        samples.len(),
        wall.mean_ms,
        wall.median_ms,
        wall.p95_ms,
        wall.max_ms,
    );
    eprintln!("{:<34} {:>7} {:>10} {:>10} {:>7}", "stage", "n", "mean ms", "max ms", "share");

    let mut rows: Vec<(&'static str, u64, f64, f64)> = STAGE_NAMES
        .iter()
        .filter_map(|name| {
            let timer = snapshot.timers.get(*name)?;
            (timer.count > 0).then_some((
                *name,
                timer.count,
                timer.total.as_secs_f64() * 1e3 / timer.count as f64,
                timer.max.as_secs_f64() * 1e3,
            ))
        })
        .collect();
    rows.sort_by(|a, b| b.2.total_cmp(&a.2));
    for (name, count, mean_ms, max_ms) in rows {
        eprintln!(
            "{:<34} {:>7} {:>10.4} {:>10.4} {:>6.1}%",
            name,
            count,
            mean_ms,
            max_ms,
            mean_ms / wall.mean_ms * 100.,
        );
    }

    let counters: Vec<String> = COUNTER_NAMES
        .iter()
        .filter_map(|name| {
            let value = counter(snapshot, name);
            (value > 0).then_some(format!("{name}={value}"))
        })
        .collect();
    if !counters.is_empty() {
        eprintln!("counters      : {}", counters.join("  "));
    }
}

/// The measurement itself. One process runs exactly one configuration; see
/// the module docs for the required invocations.
#[gpui::test]
#[ignore = "manual Phase 9 A/B measurement harness; run via the invocations in perf_ab_tests.rs module docs"]
fn perf_ab_frame_pipeline_slabs_on_vs_off(cx: &mut TestAppContext) {
    let frames = env_usize("WGPUI_PERF_FRAMES", 200);
    let warmup_frames = env_usize("WGPUI_PERF_WARMUP", 30);
    let layers_on = crate::layer::layers_enabled();
    let slabs_on = crate::scene_pack::slabs_enabled();

    eprintln!(
        "=== wgpui perf_ab ===\nenv: WGPUI_SLABS={} WGPUI_LAYERS={} WGPUI_INSTANCES={} | resolved: slabs={slabs_on} layers={layers_on}\ntree: ~{} elements (grid {GRID_COLS}x{GRID_ROWS}, sidebar {SIDEBAR_ROWS} rows, {CARD_COUNT} cards)\nknobs: WGPUI_PERF_FRAMES={frames} WGPUI_PERF_WARMUP={warmup_frames}",
        env_display("WGPUI_SLABS"),
        env_display("WGPUI_LAYERS"),
        env_display("WGPUI_INSTANCES"),
        estimated_element_count(),
    );

    let window = cx.open_window(
        size(px(WINDOW_WIDTH), px(WINDOW_HEIGHT)),
        |_, _| EditorPerfView { frame_counter: 0 },
    );
    cx.run_until_parked();
    let view = window.root(cx).expect("root view should be accessible");
    let mut cx = VisualTestContext::from_window(window.into(), cx);

    render_stats::set_force_enabled(true);

    // Warmup exercises both paths (dirty then idle) so taffy's persistent
    // tree, shaping caches and slab tokens describe steady state before any
    // timing starts.
    for _ in 0..warmup_frames.div_ceil(2) {
        timed_dirty_frame(&view, &mut cx);
    }
    for _ in 0..warmup_frames / 2 {
        timed_draw(&mut cx);
    }

    // DIRTY profile: notify before every draw. The viewport layer re-records
    // each frame; the keyed chrome layer should keep compositing across the
    // notify.
    render_stats::reset();
    let mut dirty_samples = Vec::with_capacity(frames);
    for _ in 0..frames {
        dirty_samples.push(timed_dirty_frame(&view, &mut cx));
    }
    let dirty_snapshot = render_stats::snapshot();

    // IDLE profile: draws with nothing invalidated. Every layer takes its
    // composite path — slab splicing (packing every idle frame) when slabs
    // are on, retained primitive replay when not.
    render_stats::reset();
    let mut idle_samples = Vec::with_capacity(frames);
    for _ in 0..frames {
        idle_samples.push(timed_draw(&mut cx));
    }
    let idle_snapshot = render_stats::snapshot();

    render_stats::set_force_enabled(false);

    report_profile("DIRTY (notified frames)", frames, &dirty_samples, &dirty_snapshot);
    report_profile("IDLE (uninvalidated frames)", frames, &idle_samples, &idle_snapshot);

    // Guard against measuring a degenerate pipeline: without these the wall
    // times could silently describe a tree where layers never engage.
    let layout_passes = |snapshot: &render_stats::Snapshot| -> u64 {
        snapshot
            .timers
            .get("frame: layout")
            .map(|timer| timer.count)
            .unwrap_or(0)
    };
    assert_eq!(
        layout_passes(&dirty_snapshot),
        frames as u64,
        "dirty stretch must run exactly one pipeline pass per timed frame; \
         a mismatch means an extra draw is piggybacking on the update"
    );
    assert_eq!(
        layout_passes(&idle_snapshot),
        frames as u64,
        "idle stretch must run exactly one pipeline pass per timed draw"
    );
    if layers_on {
        let re_renders = counter(&dirty_snapshot, "layer: re-rendered");
        assert!(
            re_renders >= frames as u64,
            "expected the viewport layer to re-record on every dirty frame \
             (got {re_renders} re-renders over {frames} frames)"
        );
        let composited_idle = counter(&idle_snapshot, "layer: composited");
        assert!(
            composited_idle >= frames as u64,
            "expected at least one composite per idle frame \
             (got {composited_idle} over {frames} frames)"
        );
        let composited_dirty = counter(&dirty_snapshot, "layer: composited");
        assert!(
            composited_dirty >= frames as u64,
            "the keyed chrome layer must composite across notifies \
             (got {composited_dirty} composites over {frames} dirty frames)"
        );
    }
    if layers_on && slabs_on {
        // Packing happens once per content generation at record time; idle
        // composites splice the cached pack, so nothing may repack here.
        let packed_idle = counter(&idle_snapshot, "slab: layers packed");
        assert_eq!(
            packed_idle, 0,
            "idle composites must splice the record-time pack, not repack"
        );
        let composited_slab = counter(&idle_snapshot, "layer: composited (slab)");
        assert!(
            composited_slab >= frames as u64,
            "slabs are enabled but no idle-frame composite spliced a cached pack \
             (got {composited_slab} over {frames} frames)"
        );
    }
}
