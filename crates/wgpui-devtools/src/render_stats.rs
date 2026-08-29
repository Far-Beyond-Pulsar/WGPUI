//! Thread-safe render counters and CPU timing, independent of a UI backend.
use std::collections::BTreeMap;
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
/// Whether instrumentation is enabled by `WGPUI_RENDER_STATS`.
pub fn enabled() -> bool {
    std::env::var("WGPUI_RENDER_STATS")
        .map(|value| !value.is_empty() && value != "0")
        .unwrap_or(false)
}
/// Adds to a named counter.
pub fn add(name: &'static str, amount: u64) {
    if enabled() {
        REGISTRY
            .lock()
            .expect("render stats mutex poisoned")
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
        let mut registry = REGISTRY.lock().expect("render stats mutex poisoned");
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
    let registry = REGISTRY.lock().expect("render stats mutex poisoned");
    Snapshot {
        counters: registry.counters.clone(),
        timers: registry.timers.clone(),
    }
}
/// Clears all values.
pub fn reset() {
    *REGISTRY.lock().expect("render stats mutex poisoned") = Registry::default();
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
        reset();
        count("test: disabled");
        assert!(snapshot().counters.is_empty());
    }
}
