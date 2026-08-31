//! Devtools implementation of the core instrumentation contract.
use super::flamegraph::capture::{self, CaptureEvent, CaptureEventKind};
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
        let capture_active = capture::global().is_some_and(|controller| controller.is_collecting());
        if !super::render_stats::enabled() && !capture_active {
            return None;
        }
        let token = self.next_token.fetch_add(1, Ordering::Relaxed);
        if super::render_stats::enabled() || capture_active {
            self.starts
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .insert(token, (name, Instant::now()));
        }
        if capture_active && let Some(controller) = capture::global() {
            controller.record_current(CaptureEventKind::SpanBegin, name);
        }
        Some(token)
    }
    fn end_span(&self, token: u64) {
        let span = self
            .starts
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&token);
        if let Some((name, start)) = span {
            if super::render_stats::enabled() {
                super::render_stats::record(name, start.elapsed());
            }
            if let Some(controller) = capture::global()
                && controller.is_collecting()
            {
                controller.record_current(CaptureEventKind::SpanEnd, name);
            }
        }
    }
    fn counter(&self, name: &'static str, amount: u64) {
        super::render_stats::add(name, amount);
        if let Some(controller) = capture::global()
            && controller.is_collecting()
        {
            let event = CaptureEvent::named(
                controller.current_frame_id(),
                controller.now_ns(),
                CaptureEventKind::Counter,
                name,
            )
            .with_payload(amount.to_le_bytes().to_vec());
            controller.record_event(event);
        }
    }
    fn frame_presented(&self) {
        super::render_stats::count("frame: presented");
        if let Some(controller) = capture::global() {
            let frame_id = controller.current_frame_id();
            controller.presentation_boundary(frame_id, controller.now_ns(), None);
        }
    }

    fn gpu_timestamp(&self, name: &'static str, start: u64, end: u64) {
        if let Some(controller) = capture::global()
            && controller.is_collecting()
        {
            let payload = start
                .to_le_bytes()
                .into_iter()
                .chain(end.to_le_bytes())
                .collect::<Vec<_>>();
            let event = CaptureEvent::named(
                controller.current_frame_id(),
                controller.now_ns(),
                CaptureEventKind::GpuTimestamp,
                name,
            )
            .with_payload(payload);
            controller.record_event(event);
        }
    }
}

impl DevtoolsHooks {
    /// Marks the beginning of a backend frame at its safe capture point.
    pub fn frame_started(&self, frame_id: u64, timestamp_ns: u64) -> bool {
        capture::global_controller().begin_frame(frame_id, timestamp_ns)
    }

    /// Freezes the active capture after the backend has presented this frame.
    pub fn capture_presentation_boundary(
        &self,
        frame_id: u64,
        timestamp_ns: u64,
        calibration: Option<capture::ClockCalibration>,
    ) -> Option<std::sync::Arc<capture::CaptureBundle>> {
        capture::global_controller().presentation_boundary(frame_id, timestamp_ns, calibration)
    }
}
