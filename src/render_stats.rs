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
use std::sync::LazyLock;
use std::time::Duration;
use parking_lot::Mutex;
use crate::time_ext::Instant;

static ENABLED: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("WGPUI_RENDER_STATS")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false)
});

#[cfg(any(test, feature = "test-support"))]
static FORCE_ENABLED: AtomicBool = AtomicBool::new(false);

/// Whether instrumentation is on. Check this before doing any work that only
/// exists to feed the stats.
#[inline]
pub fn enabled() -> bool {
    #[cfg(any(test, feature = "test-support"))]
    if FORCE_ENABLED.load(Ordering::Relaxed) {
        return true;
    }
    *ENABLED
}

/// Turn instrumentation on regardless of `WGPUI_RENDER_STATS`, so tests can
/// exercise the recording paths without depending on ambient environment
/// variables. Only available in test builds.
#[cfg(any(test, feature = "test-support"))]
pub fn set_force_enabled(force: bool) {
    FORCE_ENABLED.store(force, Ordering::Relaxed);
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

impl Registry {
    fn record_sample(&self, name: &'static str, duration_ns: u64) {
        let mut timers = self.timers.lock();
        let entry = timers.entry(name).or_default();
        entry.count += 1;
        entry.total_ns += duration_ns;
        entry.max_ns = entry.max_ns.max(duration_ns);
    }

    fn bump_counter(&self, name: &'static str, amount: u64) {
        *self.counters.lock().entry(name).or_insert(0) += amount;
    }
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
    REGISTRY.record_sample(name, dur.as_nanos() as u64);
}

/// Bump a named counter (frames drawn, blits attempted, refreshes forced, ...).
pub fn count(name: &'static str) {
    add(name, 1);
}

/// Add an arbitrary amount to a named counter, for quantities not counted in
/// units of one, such as bytes uploaded to a GPU slab.
pub fn add(name: &'static str, amount: u64) {
    if !enabled() {
        return;
    }
    REGISTRY.bump_counter(name, amount);
}

/// Bridge to an embedder's span profiler (the engine's flamegraph).
///
/// wgpui cannot depend on the engine's profiler crate, and the crates.io
/// `profiling` crate it does link is a separate instance with its own global
/// state, so nothing recorded through it can ever reach the engine flamegraph.
/// The embedder installs this once at startup; every [`scope`] (and
/// [`external_scope`]) then also opens an embedder span for the same interval.
pub struct ScopeHook {
    /// Cheap check (one relaxed atomic load) for whether the embedder wants spans.
    pub enabled: fn() -> bool,
    /// Open a span named `name`. The returned guard closes it on drop. It must
    /// be `Send` because a [`Scope`] may be moved across threads with its owner.
    pub begin: fn(&'static str) -> Box<dyn Send>,
    /// Like `begin` for a name built at runtime (a task's spawn location, a
    /// view's type). Only called while `enabled()`, so the allocation is free
    /// when the embedder is not recording.
    pub begin_owned: fn(String) -> Box<dyn Send>,
    /// Record a span that already finished, `elapsed` ago, at the current
    /// nesting depth. Lets hot per-item sites time themselves with one clock
    /// read and only emit a span when the item was slow.
    pub record_elapsed: fn(String, Duration),
}

static SCOPE_HOOK: std::sync::OnceLock<ScopeHook> = std::sync::OnceLock::new();

/// Install the embedder span bridge. Only the first call takes effect.
pub fn set_scope_hook(hook: ScopeHook) {
    let _ = SCOPE_HOOK.set(hook);
}

/// Open an embedder span only (no stats timer). `None` when no hook is
/// installed or the embedder is not recording.
#[inline]
pub fn external_scope(name: &'static str) -> Option<Box<dyn Send>> {
    let hook = SCOPE_HOOK.get()?;
    if !(hook.enabled)() {
        return None;
    }
    Some((hook.begin)(name))
}

/// Whether the embedder profiler is recording. One relaxed load; check before
/// doing any work (clock reads, name formatting) that only feeds spans.
#[inline]
pub fn external_enabled() -> bool {
    SCOPE_HOOK.get().is_some_and(|hook| (hook.enabled)())
}

/// Open an embedder span whose name is built lazily. The closure runs only while
/// the embedder is recording.
#[inline]
pub fn external_scope_with(name: impl FnOnce() -> String) -> Option<Box<dyn Send>> {
    let hook = SCOPE_HOOK.get()?;
    if !(hook.enabled)() {
        return None;
    }
    Some((hook.begin_owned)(name()))
}

/// Emit a finished span only if it took at least `min`. `start` should come from
/// `Instant::now()` taken only when [`external_enabled`] (pass `None` otherwise).
#[inline]
pub fn external_record_if_slow(start: Option<Instant>, min: Duration, name: impl FnOnce() -> String) {
    let Some(start) = start else { return };
    let elapsed = start.elapsed();
    if elapsed < min {
        return;
    }
    if let Some(hook) = SCOPE_HOOK.get() {
        (hook.record_elapsed)(name(), elapsed);
    }
}

/// Zero-duration marker naming a caller: `what @ file:line`. Pair with
/// `#[track_caller]` on the function being observed to learn who invoked it.
#[inline]
pub fn external_caller_marker(what: &'static str, location: &'static std::panic::Location<'static>) {
    if !external_enabled() {
        return;
    }
    external_record_if_slow(Some(Instant::now()), Duration::ZERO, || {
        format!("{what} @ {}:{}", location.file(), location.line())
    });
}

/// Clock read for [`external_record_if_slow`], skipped when not recording.
#[inline]
pub fn external_timer() -> Option<Instant> {
    external_enabled().then(Instant::now)
}

/// RAII timer. Records on drop.
pub struct Scope {
    name: &'static str,
    /// `Some` only when stats are enabled, so a flamegraph-only scope does not
    /// pollute the once-per-second stats dump.
    start: Option<Instant>,
    /// Closes the embedder span when the scope ends.
    _external: Option<Box<dyn Send>>,
}

impl Drop for Scope {
    fn drop(&mut self) {
        if let Some(start) = self.start {
            record(self.name, start.elapsed());
        }
    }
}

/// Start a scoped timer. Returns `None` when both the stats dump and the
/// embedder profiler are disabled, so the idle cost is two atomic loads.
#[inline]
pub fn scope(name: &'static str) -> Option<Scope> {
    let stats = enabled();
    let external = external_scope(name);
    if !stats && external.is_none() {
        return None;
    }
    Some(Scope {
        name,
        start: stats.then(Instant::now),
        _external: external,
    })
}

/// Like [`scope`] but never opens an embedder span. For sites that run once per
/// element, where a span each would flood the flamegraph.
#[inline]
pub fn scope_stats_only(name: &'static str) -> Option<Scope> {
    if !enabled() {
        return None;
    }
    Some(Scope {
        name,
        start: Some(Instant::now()),
        _external: None,
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
        let last = REGISTRY.last_report.lock();
        last.elapsed()
    };

    if elapsed < Duration::from_secs(1) {
        REGISTRY.reporting.store(false, Ordering::Release);
        return;
    }

    *REGISTRY.last_report.lock() = Instant::now();

    // Unit tests force-enable instrumentation process-globally while sibling
    // tests drive real draw paths concurrently. Letting this report fire in a
    // unit-test binary would print to stderr mid-test and drain the shared
    // accumulators out from under render_stats' own snapshot-based
    // assertions, so test builds only age out the reporting window here.
    #[cfg(not(test))]
    {
        let timers: Vec<(&'static str, Accum)> = {
            let mut guard = REGISTRY.timers.lock();
            std::mem::take(&mut *guard).into_iter().collect()
        };
        let counters: Vec<(&'static str, u64)> = {
            let mut guard = REGISTRY.counters.lock();
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
                out.push_str(&format!(
                    "{:<38} {:>7}  ({:.1}/s)\n",
                    name,
                    v,
                    v as f64 / secs
                ));
            }
        }

        eprint!("{}", out);
    }

    REGISTRY.reporting.store(false, Ordering::Release);
}

#[cfg(any(test, feature = "test-support"))]
mod test_support {
    use super::*;

    /// Point-in-time values of one timer.
    #[derive(Clone, Debug, Default, PartialEq, Eq)]
    pub struct TimerSnapshot {
        /// How many samples accumulated under this name.
        pub count: u64,
        /// Sum of all sample durations.
        pub total: Duration,
        /// Longest single sample seen since the last drain or report.
        pub max: Duration,
    }

    /// Current accumulator values, ordered by name. Reading one never disturbs
    /// the accumulators and never triggers or waits on the periodic report.
    #[derive(Clone, Debug, Default, PartialEq, Eq)]
    pub struct Snapshot {
        /// Counter totals by name.
        pub counters: BTreeMap<&'static str, u64>,
        /// Timer accumulations by name.
        pub timers: BTreeMap<&'static str, TimerSnapshot>,
    }

    /// Read every counter and timer without consuming them. Works whether or
    /// not instrumentation is enabled, bypassing the once-per-second reporting
    /// gate entirely.
    pub fn snapshot() -> Snapshot {
        let counters = REGISTRY.counters.lock().clone();
        let timers = REGISTRY
            .timers
            .lock()
            .iter()
            .map(|(name, accum)| {
                (
                    *name,
                    TimerSnapshot {
                        count: accum.count,
                        total: Duration::from_nanos(accum.total_ns),
                        max: Duration::from_nanos(accum.max_ns),
                    },
                )
            })
            .collect();
        Snapshot { counters, timers }
    }

    /// Drain all accumulators: timers, counters and per-frame counts drop back
    /// to zero. Like [`snapshot`] this ignores both the enablement flag and the
    /// reporting gate.
    pub fn reset() {
        REGISTRY.timers.lock().clear();
        REGISTRY.counters.lock().clear();
        REGISTRY.frames.store(0, Ordering::Relaxed);
        *REGISTRY.last_report.lock() = Instant::now();
    }
}

#[cfg(any(test, feature = "test-support"))]
pub use test_support::{reset, snapshot, Snapshot, TimerSnapshot};

#[cfg(test)]
mod tests {
    use super::*;

    // Registry state (`FORCE_ENABLED` and the timer/counter accumulators) is
    // process-global, and parallel tests drive real draw paths whose stats
    // call sites become live whenever force-enablement is on. Every test that
    // touches the registry therefore holds this lock for its whole body,
    // including helpers such as `ForceEnabled`.
    static SERIALIZATION: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

    // Only stat names owned by this module. Sibling tests exercising real
    // render paths can add entries under other names while force-enablement
    // is on, so assertions read through `owned_snapshot` instead of comparing
    // whole registries.
    const TEST_NAME_PREFIX: &str = "test:";

    fn owned_snapshot() -> Snapshot {
        let raw = snapshot();
        Snapshot {
            counters: raw
                .counters
                .into_iter()
                .filter(|(name, _)| name.starts_with(TEST_NAME_PREFIX))
                .collect(),
            timers: raw
                .timers
                .into_iter()
                .filter(|(name, _)| name.starts_with(TEST_NAME_PREFIX))
                .collect(),
        }
    }

    /// Must be created while holding [`SERIALIZATION`].
    struct ForceEnabled;

    impl ForceEnabled {
        fn new() -> Self {
            set_force_enabled(true);
            ForceEnabled
        }
    }

    impl Drop for ForceEnabled {
        fn drop(&mut self) {
            set_force_enabled(false);
        }
    }

    #[test]
    fn force_enabled_overrides_the_environment_flag() {
        let _serialization_guard = SERIALIZATION.lock();
        set_force_enabled(true);
        assert!(enabled());
        set_force_enabled(false);
        assert_eq!(enabled(), *ENABLED);
    }

    #[test]
    fn counters_accumulate_by_one_and_by_added_amount() {
        let _serialization_guard = SERIALIZATION.lock();
        let _force_enabled = ForceEnabled::new();
        reset();
        count("test: frames drawn");
        count("test: frames drawn");
        add("test: slab bytes uploaded", 512);
        add("test: slab bytes uploaded", 256);

        let values = owned_snapshot();
        assert_eq!(values.counters.get("test: frames drawn"), Some(&2));
        assert_eq!(values.counters.get("test: slab bytes uploaded"), Some(&768));
        reset();
    }

    #[test]
    fn snapshot_reads_without_consuming() {
        let _serialization_guard = SERIALIZATION.lock();
        let _force_enabled = ForceEnabled::new();
        reset();
        add("test: uniform bytes written", 96);

        let first = owned_snapshot();
        let second = owned_snapshot();
        assert_eq!(first, second);
        assert_eq!(
            first.counters.get("test: uniform bytes written"),
            Some(&96)
        );

        add("test: uniform bytes written", 32);
        let third = owned_snapshot();
        assert_eq!(
            third.counters.get("test: uniform bytes written"),
            Some(&128)
        );
        reset();
    }

    #[test]
    fn timers_accumulate_from_record_and_scope() {
        let _serialization_guard = SERIALIZATION.lock();
        let _force_enabled = ForceEnabled::new();
        reset();
        record("test: gpu upload", Duration::from_millis(2));
        record("test: gpu upload", Duration::from_millis(5));
        {
            let _timer =
                scope("test: gpu upload").expect("scope should exist while forced on");
        }

        let values = owned_snapshot();
        let timer = values
            .timers
            .get("test: gpu upload")
            .expect("recorded timer should appear in snapshot");
        assert_eq!(timer.count, 3);
        assert!(timer.total >= Duration::from_millis(7));
        assert!(timer.max >= Duration::from_millis(5));
        reset();
    }

    #[test]
    fn snapshot_and_reset_bypass_enablement_and_reporting_gates() {
        let _serialization_guard = SERIALIZATION.lock();
        set_force_enabled(false);
        reset();
        REGISTRY.record_sample("test: manual stage", 1_000);
        REGISTRY.bump_counter("test: manual counter", 3);

        let values = owned_snapshot();
        assert_eq!(values.counters.get("test: manual counter"), Some(&3));
        let timer = values
            .timers
            .get("test: manual stage")
            .expect("manually recorded timer should appear in snapshot");
        assert_eq!(timer.count, 1);
        assert_eq!(timer.total, Duration::from_nanos(1_000));

        reset();
        let drained = owned_snapshot();
        assert!(drained.counters.is_empty());
        assert!(drained.timers.is_empty());
    }
}

static UNCAPPED_PRESENTATION: AtomicBool = AtomicBool::new(false);

/// Lift (or restore) the presentation cap process-wide. While set, each window's
/// renderer switches its swapchain from vsync (`Fifo`) to the best non-vsync
/// present mode the surface supports, so presentation is no longer limited to the
/// display refresh rate. Used by the profiler's "uncap frame rate" recording option.
pub fn set_uncapped_presentation(uncapped: bool) {
    UNCAPPED_PRESENTATION.store(uncapped, Ordering::Relaxed);
}

/// Whether the presentation cap is currently lifted.
#[inline]
pub fn uncapped_presentation() -> bool {
    UNCAPPED_PRESENTATION.load(Ordering::Relaxed)
}
