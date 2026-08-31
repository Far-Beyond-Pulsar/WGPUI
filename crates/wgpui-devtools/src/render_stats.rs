//! Thread-safe render counters and CPU timing, independent of a UI backend.
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicI8, Ordering};
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TimerSnapshot {
    pub count: u64,
    pub total: Duration,
    pub max: Duration,
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Snapshot {
    pub counters: BTreeMap<&'static str, u64>,
    pub timers: BTreeMap<&'static str, TimerSnapshot>,
}
#[derive(Default)]
struct Registry {
    counters: BTreeMap<&'static str, u64>,
    timers: BTreeMap<&'static str, TimerSnapshot>,
}
static REGISTRY: LazyLock<Mutex<Registry>> = LazyLock::new(|| Mutex::new(Registry::default()));
// The environment remains the normal application-facing switch. The override
// exists for a benchmark that needs to compare both arms in one process.
static ENABLED_OVERRIDE: AtomicI8 = AtomicI8::new(0);
fn lock_registry() -> std::sync::MutexGuard<'static, Registry> {
    REGISTRY
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}
/// Whether instrumentation is enabled by `WGPUI_RENDER_STATS`.
pub fn enabled() -> bool {
    match ENABLED_OVERRIDE.load(Ordering::Relaxed) {
        1 => return true,
        2 => return false,
        _ => {}
    }
    std::env::var("WGPUI_RENDER_STATS")
        .map(|value| !value.is_empty() && value != "0")
        .unwrap_or(false)
}

/// Force instrumentation on or off until [`clear_enabled_override`] is called.
///
/// This is intended for in-process benchmark A/B runs. Applications should
/// continue to use `WGPUI_RENDER_STATS`, which avoids changing a process-wide
/// setting while a frame is running.
pub fn set_enabled(enabled: bool) {
    ENABLED_OVERRIDE.store(if enabled { 1 } else { 2 }, Ordering::Relaxed);
}

/// Return control of instrumentation to `WGPUI_RENDER_STATS`.
pub fn clear_enabled_override() {
    ENABLED_OVERRIDE.store(0, Ordering::Relaxed);
}
/// Adds to a named counter.
pub fn add(name: &'static str, amount: u64) {
    if enabled() {
        lock_registry()
            .counters
            .entry(name)
            .and_modify(|value| *value += amount)
            .or_insert(amount);
    }
}
/// Increments a named counter.
pub fn count(name: &'static str) {
    add(name, 1);
}
/// Records a timing sample.
pub fn record(name: &'static str, duration: Duration) {
    if enabled() {
        let mut registry = lock_registry();
        let timer = registry.timers.entry(name).or_default();
        timer.count += 1;
        timer.total += duration;
        timer.max = timer.max.max(duration);
    }
}
/// Starts a timing scope.
pub fn scope(name: &'static str) -> Option<Scope> {
    enabled().then(|| Scope {
        name,
        start: Instant::now(),
    })
}
/// Reads values without consuming them.
pub fn snapshot() -> Snapshot {
    let registry = lock_registry();
    Snapshot {
        counters: registry.counters.clone(),
        timers: registry.timers.clone(),
    }
}
/// Clears all values.
pub fn reset() {
    *lock_registry() = Registry::default();
}
pub struct Scope {
    name: &'static str,
    start: Instant,
}
impl Drop for Scope {
    fn drop(&mut self) {
        record(self.name, self.start.elapsed());
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn disabled_does_not_accumulate() {
        set_enabled(false);
        reset();
        count("test: disabled");
        assert!(snapshot().counters.is_empty());
        clear_enabled_override();
    }

    #[test]
    fn override_can_run_an_ab_pair_without_mutating_the_environment() {
        set_enabled(true);
        reset();
        count("test: enabled");
        assert_eq!(snapshot().counters.get("test: enabled"), Some(&1));

        set_enabled(false);
        reset();
        count("test: disabled");
        assert!(snapshot().counters.is_empty());
        clear_enabled_override();
    }

    #[test]
    fn poisoned_registry_is_recovered_without_panicking() {
        clear_enabled_override();
        assert!(
            std::panic::catch_unwind(|| {
                let _guard = REGISTRY.lock();
                panic!("fault injection");
            })
            .is_err()
        );
        reset();
        assert_eq!(snapshot(), Snapshot::default());
    }
}
