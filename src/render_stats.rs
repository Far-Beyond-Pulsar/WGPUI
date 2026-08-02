//! Frame-path instrumentation.
//!
//! Enable with `WGPUI_RENDER_STATS=1`. Once per second every accumulated timer
//! and counter is dumped to stderr, so a slow frame can be attributed to a
//! specific stage instead of guessed at.
//!
//! Everything here compiles to an atomic load of `ENABLED` when disabled, so it
//! is safe to leave the call sites in place.
//!
//! Timers report count, mean and max in milliseconds; the max column is the one
//! that matters for stalls, since a single multi-second frame vanishes into a
//! mean over 120 samples.
//!
//! # Reading the `frame:` timers
//!
//! Timers accumulate wall time, so a timer entered inside another one is counted
//! by both. The `frame:` family is deliberately not flat, because the frame path
//! itself is not flat — `render()` runs inside `request_layout`, text is shaped
//! from both layout measure callbacks and prepaint, and primitives are emitted
//! during paint. The containment is:
//!
//! - `frame: layout`, `frame: prepaint` and `frame: paint` are disjoint. They
//!   are taken at the window root, so summing them gives the frame's element
//!   cost once.
//! - `frame: render` is a subset of `frame: layout`, since a view's `render()`
//!   is invoked from its `request_layout`. The exception is a `.cached()` view
//!   rebuilding, which renders from prepaint — so for those it falls under
//!   `frame: prepaint` instead.
//! - `frame: text shaping` is a subset of `frame: layout` (text measure
//!   callbacks) plus `frame: prepaint` (line shaping for paint).
//! - `frame: bounds tree` is a subset of `frame: paint`.
//! - `frame: scene finish` and `frame: gpu upload` are outside all of the above.
//!
//! So the Phase 0 question — is building the element description cheap relative
//! to the work reconciliation would skip? — is `frame: render` over
//! `frame: layout + frame: prepaint + frame: paint`, with no double counting to
//! correct for, because `frame: render` is wholly contained in that sum.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};
use std::time::Duration;
use crate::time_ext::Instant;

static ENABLED: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("WGPUI_RENDER_STATS")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false)
});

/// Whether instrumentation is on. Check this before doing any work that only
/// exists to feed the stats.
#[inline]
pub fn enabled() -> bool {
    *ENABLED
}

#[derive(Default)]
struct Accum {
    count: u64,
    total_ns: u64,
    max_ns: u64,
}

struct Registry {
    timers: Mutex<BTreeMap<&'static str, Accum>>,
    counters: Mutex<BTreeMap<&'static str, u64>>,
    last_report: Mutex<Instant>,
    reporting: AtomicBool,
    frames: AtomicU64,
}

static REGISTRY: LazyLock<Registry> = LazyLock::new(|| Registry {
    timers: Mutex::new(BTreeMap::new()),
    counters: Mutex::new(BTreeMap::new()),
    last_report: Mutex::new(Instant::now()),
    reporting: AtomicBool::new(false),
    frames: AtomicU64::new(0),
});

/// Record a single timing sample.
pub fn record(name: &'static str, dur: Duration) {
    if !enabled() {
        return;
    }
    let ns = dur.as_nanos() as u64;
    let mut timers = REGISTRY.timers.lock().unwrap();
    let entry = timers.entry(name).or_default();
    entry.count += 1;
    entry.total_ns += ns;
    entry.max_ns = entry.max_ns.max(ns);
}

/// Bump a named counter (frames drawn, blits attempted, refreshes forced, ...).
pub fn count(name: &'static str) {
    if !enabled() {
        return;
    }
    *REGISTRY.counters.lock().unwrap().entry(name).or_insert(0) += 1;
}

/// RAII timer. Records on drop.
pub struct Scope {
    name: &'static str,
    start: Instant,
}

impl Drop for Scope {
    fn drop(&mut self) {
        record(self.name, self.start.elapsed());
    }
}

/// Start a scoped timer. Returns `None` when instrumentation is disabled, so the
/// whole thing costs one atomic load.
#[inline]
pub fn scope(name: &'static str) -> Option<Scope> {
    if !enabled() {
        return None;
    }
    Some(Scope {
        name,
        start: Instant::now(),
    })
}

/// Call once per presented frame. Dumps and clears the accumulators every second.
pub fn tick_frame() {
    if !enabled() {
        return;
    }

    REGISTRY.frames.fetch_add(1, Ordering::Relaxed);

    // Only one thread reports; the rest return immediately.
    if REGISTRY.reporting.swap(true, Ordering::AcqRel) {
        return;
    }

    let elapsed = {
        let last = REGISTRY.last_report.lock().unwrap();
        last.elapsed()
    };

    if elapsed < Duration::from_secs(1) {
        REGISTRY.reporting.store(false, Ordering::Release);
        return;
    }

    *REGISTRY.last_report.lock().unwrap() = Instant::now();

    let timers: Vec<(&'static str, Accum)> = {
        let mut guard = REGISTRY.timers.lock().unwrap();
        std::mem::take(&mut *guard).into_iter().collect()
    };
    let counters: Vec<(&'static str, u64)> = {
        let mut guard = REGISTRY.counters.lock().unwrap();
        std::mem::take(&mut *guard).into_iter().collect()
    };
    let frames = REGISTRY.frames.swap(0, Ordering::Relaxed);

    let secs = elapsed.as_secs_f64();
    let mut out = String::new();
    out.push_str(&format!(
        "\n=== WGPUI RENDER STATS ({:.2}s, {} frames, {:.1} fps) ===\n",
        secs,
        frames,
        frames as f64 / secs
    ));

    if !timers.is_empty() {
        out.push_str(&format!(
            "{:<38} {:>7} {:>10} {:>10} {:>10}\n",
            "stage", "n", "mean ms", "max ms", "total ms"
        ));
        // Worst max first: that is what a stall looks like.
        let mut rows = timers;
        rows.sort_by_key(|(_, a)| std::cmp::Reverse(a.max_ns));
        for (name, a) in rows {
            if a.count == 0 {
                continue;
            }
            out.push_str(&format!(
                "{:<38} {:>7} {:>10.3} {:>10.3} {:>10.2}\n",
                name,
                a.count,
                (a.total_ns as f64 / a.count as f64) / 1.0e6,
                a.max_ns as f64 / 1.0e6,
                a.total_ns as f64 / 1.0e6,
            ));
        }
    }

    if !counters.is_empty() {
        out.push_str("--- counters ---\n");
        for (name, v) in counters {
            out.push_str(&format!("{:<38} {:>7}  ({:.1}/s)\n", name, v, v as f64 / secs));
        }
    }

    eprint!("{}", out);

    REGISTRY.reporting.store(false, Ordering::Release);
}
