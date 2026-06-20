//! Headless rendering-pipeline benchmark.
//!
//! Drives real frames through the test platform (no GPU/display required) and
//! measures the CPU-side rendering pipeline that the rendering-perf work targets:
//! layout, scene build, retained-fiber reconciliation, scene-damage computation,
//! the partial-redraw decision, and frame-metric collection. The GPU present is a
//! no-op under the test platform, so these numbers isolate the per-frame CPU cost.
//!
//! Run with:
//!     cargo bench --features test-support --bench render_pipeline
//!
//! This complements (does not replace) the on-display GPU/visual validation the
//! maintainer runs against the examples and the game-engine UI before merge.

use std::time::{Duration, Instant};

use gpui::{
    Context,
    InteractiveElement,
    IntoElement,
    ParentElement,
    Render,
    Styled,
    TestAppContext,
    VisualTestContext,
    Window,
    canvas,
    div,
    px,
    rgb,
};
use gpui_prim_solid::{SolidQuad, SolidQuads};

/// A representative app UI: a scrollable-looking column of styled, id-bearing
/// rows (each a small flex row with a background and a text child), which is the
/// shape that exercises layout, scene chunking, and retained-fiber reconcile.
struct BenchRoot {
    rows: usize,
    tick: usize,
}

impl Render for BenchRoot {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        // `tick` participates in a child's text so a dirty re-render actually
        // produces different content (not a no-op).
        let tick = self.tick;
        div()
            .id("bench-root")
            .flex()
            .flex_col()
            .w(px(900.))
            .h(px(700.))
            .bg(rgb(0x101418))
            .children((0..self.rows).map(move |row| {
                div()
                    .id(("row", row))
                    .flex()
                    .w(px(880.))
                    .h(px(22.))
                    .bg(if row % 2 == 0 {
                        rgb(0x1b2430)
                    } else {
                        rgb(0x222b36)
                    })
                    .text_color(rgb(0xc8d2dc))
                    .child(div().w(px(120.)).child(format!("row {row}")))
                    .child(div().child(format!("value {}", row.wrapping_add(tick))))
            }))
    }
}

struct Stats {
    mean: Duration,
    min: Duration,
    p50: Duration,
    max: Duration,
}

fn summarize(mut samples: Vec<Duration>) -> Stats {
    samples.sort_unstable();
    let count = samples.len().max(1);
    let total: Duration = samples.iter().sum();
    Stats {
        mean: total / count as u32,
        min: *samples.first().unwrap_or(&Duration::ZERO),
        p50: samples[samples.len() / 2],
        max: *samples.last().unwrap_or(&Duration::ZERO),
    }
}

fn draw_once(cx: &mut VisualTestContext) {
    cx.update(|window, cx| {
        window.draw(cx).clear();
    });
}

/// Time `iters` draws produced by `before_each` + a draw, after `warmup` warmup draws.
fn bench(
    cx: &mut VisualTestContext,
    warmup: usize,
    iters: usize,
    mut before_each: impl FnMut(&mut VisualTestContext),
) -> Stats {
    for _ in 0..warmup {
        before_each(cx);
        draw_once(cx);
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        before_each(cx);
        let started = Instant::now();
        draw_once(cx);
        samples.push(started.elapsed());
    }
    summarize(samples)
}

fn run_for_size(rows: usize) {
    let mut cx = TestAppContext::single();
    let (root, cx) = cx.add_window_view(|_window, _cx| BenchRoot { rows, tick: 0 });

    // First (cold) full draw — everything is fresh, nothing reused.
    let cold = {
        let started = Instant::now();
        draw_once(cx);
        started.elapsed()
    };

    // Steady-state clean redraw: no entity changes, so the view's prepaint/paint
    // is reused and retained fibers are carried forward. This is the redraw the
    // perf work optimizes.
    let clean = bench(cx, 8, 64, |_cx| {});

    // Dirty redraw: the root re-renders every frame (full walk), exercising
    // layout + scene rebuild + fiber reconcile from scratch.
    let dirty = bench(cx, 8, 64, |cx| {
        root.update(cx, |view, view_cx| {
            view.tick = view.tick.wrapping_add(1);
            view_cx.notify();
        });
        cx.run_until_parked();
    });

    // Refresh-heat overlay enabled (forces a full repaint + injects overlay quads).
    cx.update(|window, _cx| window.set_refresh_overlay(true));
    let overlay = bench(cx, 8, 64, |cx| {
        cx.run_until_parked();
    });
    cx.update(|window, _cx| window.set_refresh_overlay(false));

    let metrics = cx
        .update(|window, _cx| window.frame_metrics().copied())
        .expect("frame metrics");

    println!("\n=== {rows} rows ===");
    println!(
        "  scene: {} quads, {} chunks, {} batches | retained: {} fibers ({} dirty)",
        metrics.primitive_count.quads,
        metrics.total_view_count,
        metrics.batch_count,
        metrics.retained_fiber_count,
        metrics.retained_dirty_fiber_count,
    );
    // p50 (median) is the stable metric here; the test executor occasionally
    // spikes (allocator / async bookkeeping), which inflates mean and max.
    let us = |d: Duration| d.as_secs_f64() * 1_000_000.0;
    let row = |label: &str, s: &Stats| {
        println!(
            "  {label:<22}: p50 {:>8.1} us  (min {:.1}, mean {:.1}, max {:.1})",
            us(s.p50),
            us(s.min),
            us(s.mean),
            us(s.max),
        );
    };
    println!(
        "  {:<22}: {:>8.1} us  (single sample)",
        "cold full draw",
        us(cold)
    );
    row("clean redraw (reuse)", &clean);
    row("dirty redraw (rebuild)", &dirty);
    row("refresh-overlay draw", &overlay);
    if clean.p50.as_nanos() > 0 {
        println!(
            "  {:<22}: {:>8.2}x  (dirty p50 / clean p50)",
            "view-reuse speedup",
            dirty.p50.as_secs_f64() / clean.p50.as_secs_f64(),
        );
    }
}

/// A view that paints `count` custom (plugin) `SolidQuad` primitives each frame
/// via a canvas, exercising the multi-crate primitive collection path end to end.
struct CustomPrimBench {
    count: usize,
}

impl Render for CustomPrimBench {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        let count = self.count;
        div().w(px(900.)).h(px(700.)).child(
            canvas(
                |_bounds, _window, _cx| (),
                move |bounds, _, window, _cx| {
                    let base_x = f32::from(bounds.origin.x);
                    let base_y = f32::from(bounds.origin.y);
                    for index in 0..count {
                        let x = base_x + (index % 30) as f32 * 12.0;
                        let y = base_y + (index / 30) as f32 * 12.0;
                        let quad = SolidQuad::new([x, y], [10.0, 10.0], [0.2, 0.6, 1.0, 1.0]);
                        window.paint_custom_primitive(SolidQuads::TYPE, quad.as_bytes(), bounds);
                    }
                },
            )
            .size_full(),
        )
    }
}

fn run_custom_primitives(count: usize) {
    let mut cx = TestAppContext::single();
    let (_root, cx) = cx.add_window_view(|_window, _cx| CustomPrimBench { count });
    // Custom primitives are collected on (re)paint; refresh + draw in one update so
    // the measured draw is the fresh one that re-queues all instances.
    let mut samples = Vec::with_capacity(64);
    for iteration in 0..(8 + 64) {
        let started = Instant::now();
        cx.update(|window, app| {
            window.refresh();
            window.draw(app).clear();
        });
        if iteration >= 8 {
            samples.push(started.elapsed());
        }
    }
    let stats = summarize(samples);
    let metrics = cx
        .update(|window, _cx| window.frame_metrics().copied())
        .expect("frame metrics");
    let us = |d: Duration| d.as_secs_f64() * 1_000_000.0;
    println!("\n=== {count} custom plugin primitives (SolidQuad, own crate) ===");
    println!(
        "  collected into scene   : {} instances",
        metrics.custom_primitive_count
    );
    println!(
        "  fresh draw + collect   : p50 {:>8.1} us  (min {:.1}, mean {:.1}, max {:.1})",
        us(stats.p50),
        us(stats.min),
        us(stats.mean),
        us(stats.max),
    );
}

fn main() {
    println!("WGPUI headless rendering-pipeline benchmark (CPU pipeline; GPU present is a no-op)");
    for rows in [100usize, 500, 1500] {
        run_for_size(rows);
    }
    for count in [50usize, 500, 2000] {
        run_custom_primitives(count);
    }
    println!("\nNote: pixel-level/GPU validation still requires a display run.");
}
