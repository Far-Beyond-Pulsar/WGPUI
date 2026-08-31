//! 10,000-item scrollable list with NO virtualization — demonstrates what the
//! overscroll buffer does and does NOT make cheap.
//!
//! Every row is a real element in the tree. *Shift* frames are cheap: the
//! list sits under a keyed, texture-retained layer with an overscroll buffer,
//! so most wheel ticks shift a persistent texture instead of repainting.
//!
//! **Refill frames are not cheap, and at this scale that's a real cost, not
//! a rounding error.** A refill (first mount, every resize, every scroll past
//! the margin) still lays out every one of the 10,000 real rows — measured at
//! ~1.3s each (docs/scroll-free-by-default.md §0.-1). The buffer bounds
//! *shift* cost by the margin; it does not and structurally cannot bound
//! *refill* cost for a plain div, because the div's children are already a
//! fully materialized `Vec`, not a range the buffer can select into. Resize
//! this window and watch `frame: layout` in the HUD (or `WGPUI_RENDER_STATS`
//! on stderr) to see it. For a real list this large, use `uniform_list`,
//! `virtual_list`, or `h_list` instead — this demo is deliberately the wrong
//! tool for its own row count, to make that distinction visible.
//!
//! Watch the HUD while scrolling:
//! - `texture shifts` grows on every scrolled frame (the cheap path),
//! - `full repaints` stays flat between refills,
//! - `buffer refills` ticks up once per half-margin of accumulated scroll —
//!   and each one costs a full relayout, per the above.
//!
//! The scrollbar thumb deliberately lives OUTSIDE the buffered layer: it must
//! track the scroll position every frame, including frames the buffer only
//! shifts.

use std::{cell::Cell, rc::Rc, time::Instant};

use gpui::{
    App, Application, Bounds, Context, Hsla, LayerPolicy, Pixels, Render, ScrollHandle,
    SharedString, Window, WindowBounds, WindowOptions, div, hsla, prelude::*, px, render_stats,
    rgb, size,
};

const TOTAL_ITEMS: usize = 10_000;
const ITEM_HEIGHT: f32 = 24.0;
const OVERDRAW_MARGIN: f32 = 160.0;

/// Deterministic pseudo-random ticker names, so the demo needs no rng dep.
struct Ticker {
    state: u64,
}

impl Ticker {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
}

#[derive(Clone)]
struct Row {
    label: SharedString,
    price: SharedString,
    accent: Hsla,
}

impl Row {
    fn generate(ix: usize, rng: &mut Ticker) -> Self {
        let cents = rng.next() % 100_000;
        let label: SharedString = format!("{:06}  {}", ix, Self::symbol(rng)).into();
        let price: SharedString = format!("${}.{:02}", cents / 100, cents % 100).into();
        let hue = (ix as f32 * 0.37) % 1.0;
        Self {
            label,
            price,
            accent: hsla(hue, 0.62, 0.52, 1.0),
        }
    }

    fn symbol(rng: &mut Ticker) -> String {
        const LETTERS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let len = 3 + (rng.next() % 2) as usize;
        let mut s = String::new();
        for _ in 0..len {
            s.push(LETTERS[(rng.next() % 26) as usize] as char);
        }
        s
    }
}

/// Counters sampled at the previous rendered frame, for the delta readout.
#[derive(Default, Clone)]
struct StatsDelta {
    shifted: i64,
    repaints: i64,
    refills: i64,
}

impl StatsDelta {
    fn sample(previous: Option<(StatsTotals, Instant)>, totals: StatsTotals) -> Self {
        match previous {
            None => Self::default(),
            Some((prev, _)) => Self {
                shifted: totals.shifted as i64 - prev.shifted as i64,
                repaints: totals.repaints as i64 - prev.repaints as i64,
                refills: totals.refills as i64 - prev.refills as i64,
            },
        }
    }
}

#[derive(Clone, Copy)]
struct StatsTotals {
    shifted: u64,
    repaints: u64,
    refills: u64,
}

impl StatsTotals {
    // `render_stats::snapshot` is gated `#[cfg(any(test, feature =
    // "test-support"))]` — it's a test-assertion API, not a general one (the
    // library's own consumption model is `WGPUI_RENDER_STATS=1` dumping to
    // stderr). This demo wants live counters for its in-window HUD, which
    // only test-support builds provide; plain `cargo run` falls back to a
    // zeroed HUD rather than failing to build at all. Run with
    // `--features test-support` for the live numbers.
    #[cfg(any(test, feature = "test-support"))]
    fn read() -> Self {
        let snap = render_stats::snapshot();
        let get = |name: &str| -> u64 { find_counter(&snap.counters, name) };
        Self {
            shifted: get("layer: composited (texture)"),
            repaints: get("layer: re-rendered"),
            refills: get("scroll: buffer refills"),
        }
    }

    #[cfg(not(any(test, feature = "test-support")))]
    fn read() -> Self {
        Self {
            shifted: 0,
            repaints: 0,
            refills: 0,
        }
    }
}

fn find_counter(counters: &std::collections::BTreeMap<&'static str, u64>, name: &str) -> u64 {
    counters
        .iter()
        .find(|(key, _)| **key == name)
        .map(|(_, value)| *value)
        .unwrap_or(0)
}

struct OverscrollDemo {
    rows: Vec<Row>,
    handle: ScrollHandle,
    previous: Option<(StatsTotals, Instant)>,
    delta: StatsDelta,
    renders: Rc<Cell<u64>>,
}

impl OverscrollDemo {
    fn new(cx: &mut Context<Self>) -> Self {
        let mut rng = Ticker::new(0x9E3779B97F4A7C15);
        let rows = (0..TOTAL_ITEMS)
            .map(|ix| Row::generate(ix, &mut rng))
            .collect();

        // Sample render_stats on a background timer, never from inside
        // `render()`. Calling `render_stats::snapshot()` synchronously from
        // `render()` — even throttled to a fixed cadence — was found to
        // break this demo's own visual output at 10,000-row scale: it isn't
        // a matter of call frequency, since throttling the read still failed
        // once instrumentation was recording during the multi-pass startup
        // burst. Sampling off the draw path entirely and pushing the result
        // in via `cx.notify()` sidesteps whatever that interaction is. See
        // docs/scroll-free-by-default.md §0.-1.
        #[cfg(any(test, feature = "test-support"))]
        cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor()
                    .timer(std::time::Duration::from_millis(200))
                    .await;
                let totals = StatsTotals::read();
                let now = Instant::now();
                if this
                    .update(cx, |view, cx| {
                        let previous = view.previous.take();
                        view.delta = StatsDelta::sample(previous, totals);
                        view.previous = Some((totals, now));
                        cx.notify();
                    })
                    .is_err()
                {
                    return; // window closed
                }
            }
        })
        .detach();

        let handle = ScrollHandle::new();

        // Opt-in, reproducible scroll-rate benchmark: `WGPUI_AUTO_SCROLL=1`
        // drives the scroll offset on a timer instead of requiring a human
        // (or a flaky OS-level wheel-event simulation) to generate sustained
        // scroll input. Off by default — normal interactive use is
        // unaffected.
        if std::env::var("WGPUI_AUTO_SCROLL").is_ok_and(|v| v == "1") {
            let auto_scroll_handle = handle.clone();
            cx.spawn(async move |this, cx| {
                loop {
                    cx.background_executor()
                        .timer(std::time::Duration::from_millis(16))
                        .await;
                    let max_offset = auto_scroll_handle.max_offset().height;
                    let mut offset = auto_scroll_handle.offset();
                    // ~300px/s: a brisk but plausible human scroll speed at
                    // 60 ticks/s.
                    offset.y -= px(5.0);
                    if -offset.y > max_offset {
                        offset.y = px(0.);
                    }
                    auto_scroll_handle.set_offset(offset);
                    if this.update(cx, |_, cx| cx.notify()).is_err() {
                        return; // window closed
                    }
                }
            })
            .detach();
        }

        Self {
            rows,
            handle,
            previous: None,
            delta: StatsDelta::default(),
            renders: Rc::new(Cell::new(0)),
        }
    }

    fn scrollbar(&self, viewport_height: Pixels) -> impl IntoElement {
        let max_offset = self.handle.max_offset().height;
        let offset_y = -self.handle.offset().y;
        let content_height = max_offset + viewport_height;
        let progress = if content_height > px(0.) {
            (offset_y.to_f32() / content_height.to_f32()).clamp(0.0, 1.0)
        } else {
            0.0
        };
        const THUMB_HEIGHT: f32 = 72.0;
        let travel = (viewport_height - px(THUMB_HEIGHT)).max(px(0.));
        let top = px(progress * travel.to_f32());

        div()
            .absolute()
            .top(top)
            .right_1()
            .w(px(8.))
            .h(px(THUMB_HEIGHT))
            .rounded_full()
            .bg(rgb(0x8A8A98))
            .opacity(0.85)
    }
}

impl Render for OverscrollDemo {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        self.renders.set(self.renders.get() + 1);
        let now = Instant::now();
        // No live render_stats access here — see `OverscrollDemo::new`'s doc
        // comment. `self.previous`/`self.delta` are pushed in by the
        // background sampler; this just reads them.
        let totals = self
            .previous
            .map(|(totals, _)| totals)
            .unwrap_or(StatsTotals {
                shifted: 0,
                repaints: 0,
                refills: 0,
            });
        let frame_dt_ms = self
            .previous
            .as_ref()
            .map(|(_, at)| now.duration_since(*at).as_secs_f64() * 1000.0);

        let viewport_height = self.handle.bounds().size.height;

        // One static key: none of the content depends on anything mutable, so
        // the wheel handler's notify never invalidates the buffered layer's
        // content claim. It MUST NOT include the scroll offset.
        let content_key = "demo-list-content";

        let rows = self.rows.clone();

        div()
            .flex()
            .flex_col()
            .size_full()
            .bg(rgb(0x14141B))
            .text_color(rgb(0xD8D8E2))
            .text_size(px(12.))
            // ---- HUD (outside the buffered layer: updates every frame) ----
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .p_3()
                    .bg(rgb(0x1C1C26))
                    .border_b_1()
                    .border_color(rgb(0x2A2A38))
                    .child(
                        div()
                            .flex()
                            .justify_between()
                            .child(div().child(
                                "overscroll buffer demo — 10,000 plain (non-virtualized) rows",
                            ))
                            .child(
                                div()
                                    .text_color(rgb(0x8A8A98))
                                    .child("scroll with the mouse wheel"),
                            ),
                    )
                    .child(
                        div()
                            .flex()
                            .gap_4()
                            .text_size(px(11.))
                            .child(stat(
                                "texture shifts",
                                self.delta.shifted,
                                totals.shifted,
                                rgb(0x7FD48A),
                            ))
                            .child(stat(
                                "full repaints",
                                self.delta.repaints,
                                totals.repaints,
                                rgb(0xE0A05A),
                            ))
                            .child(stat(
                                "buffer refills",
                                self.delta.refills,
                                totals.refills,
                                rgb(0x6AAFE0),
                            ))
                            .child(
                                div()
                                    .flex()
                                    .flex_col()
                                    .child(div().text_color(rgb(0x6A6A78)).child(format!(
                                        "last frame gap {:.1} ms · render #{}",
                                        frame_dt_ms.unwrap_or(0.0),
                                        self.renders.get(),
                                    )))
                                    .child(div().text_color(rgb(0x6A6A78)).child(format!(
                                        "margin {} px · refill at ±{} px",
                                        OVERDRAW_MARGIN as u32,
                                        OVERDRAW_MARGIN as u32 / 2,
                                    ))),
                            ),
                    ),
            )
            // ---- the list itself ----
            .child(
                div()
                    .relative()
                    .flex_1()
                    .overflow_hidden()
                    .border_y_1()
                    .border_color(rgb(0x2A2A38))
                    .child(
                        div()
                            .id("scroller")
                            .layer_keyed(content_key)
                            .layer_with_policy(LayerPolicy {
                                overdraw_margin: size(px(0.), px(OVERDRAW_MARGIN)),
                                ..Default::default()
                            })
                            .overflow_y_scroll()
                            .size_full()
                            .track_scroll(&self.handle)
                            .children(rows.into_iter().enumerate().map(|(ix, row)| {
                                div()
                                    .flex()
                                    .items_center()
                                    .justify_between()
                                    .w_full()
                                    .h(px(ITEM_HEIGHT))
                                    .px_3()
                                    .border_l_3()
                                    .border_color(row.accent)
                                    .bg(if ix.is_multiple_of(2) {
                                        rgb(0x17171F)
                                    } else {
                                        rgb(0x191922)
                                    })
                                    .child(div().child(row.label))
                                    .child(div().text_color(rgb(0xB8B8C8)).child(row.price))
                            })),
                    )
                    // Thumb overlays the list but lives outside the buffered
                    // layer, so it tracks every shift frame.
                    .when(viewport_height > px(0.), |this| {
                        this.child(self.scrollbar(viewport_height))
                    }),
            )
    }
}

fn stat(name: &'static str, delta: i64, total: u64, color: gpui::Rgba) -> impl IntoElement {
    div().flex().flex_col().child(
        div()
            .flex()
            .gap_2()
            .child(div().w(px(110.)).text_color(rgb(0x6A6A78)).child(name))
            .child(
                div()
                    .w(px(56.))
                    .text_color(color)
                    .child(format!("{delta:+}")),
            )
            .child(
                div()
                    .text_color(rgb(0x50505E))
                    .child(format!("total {total}")),
            ),
    )
}

fn main() {
    Application::new().run(|cx: &mut App| {
        // Feed the HUD regardless of WGPUI_RENDER_STATS. Test-support only —
        // see `StatsTotals::read`'s doc comment.
        // Feed the HUD regardless of WGPUI_RENDER_STATS. Test-support only —
        // see `StatsTotals::read`'s doc comment. Confirmed innocent of the
        // §0.-1 rendering bug: disabling this alone did not fix it (only
        // dropping `--features test-support` entirely does) — see
        // docs/scroll-free-by-default.md for the full isolation trail.
        #[cfg(any(test, feature = "test-support"))]
        render_stats::set_force_enabled(true);

        cx.open_window(
            WindowOptions {
                focus: true,
                window_bounds: Some(WindowBounds::Windowed(Bounds::centered(
                    None,
                    size(px(1280.), px(800.)),
                    cx,
                ))),
                ..Default::default()
            },
            |_, cx| cx.new(|cx| OverscrollDemo::new(cx)),
        )
        .unwrap();

        cx.activate(true);
    });
}
