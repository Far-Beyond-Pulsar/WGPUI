//! Devtools implementation of the core instrumentation contract.
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use wgpui_core::hooks::InstrumentationHooks;
#[derive(Default)]
pub struct DevtoolsHooks {
    next_token: AtomicU64,
    starts: Mutex<HashMap<u64, (&'static str, Instant)>>,
}
impl InstrumentationHooks for DevtoolsHooks {
    fn begin_span(&self, name: &'static str) -> Option<u64> {
        if !super::render_stats::enabled() {
            return None;
        }
        let token = self.next_token.fetch_add(1, Ordering::Relaxed);
        self.starts
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(token, (name, Instant::now()));
        Some(token)
    }
    fn end_span(&self, token: u64) {
        if let Some((name, start)) = self
            .starts
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&token)
        {
            super::render_stats::record(name, start.elapsed());
        }
    }
    fn counter(&self, name: &'static str, amount: u64) {
        super::render_stats::add(name, amount);
    }
    fn frame_presented(&self) {
        super::render_stats::count("frame: presented");
    }
}
