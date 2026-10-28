use crate::time_ext::Instant;
#[cfg(not(target_family = "wasm"))]
use device_query::{DeviceQuery, DeviceState};
use std::cell::Cell;
use std::time::Duration;

/// Detects whether a window is actively being resized by watching the stream
/// of [`winit::event::WindowEvent::Resized`] events.
///
/// **Primary strategy:** poll global mouse button state via `device_query`.
/// While the left button is held after a resize event started, the user is
/// still dragging. The moment it releases the resize is done instantly.
///
/// **Fallback:** idle-threshold timer when global mouse state is unavailable
/// (e.g. macOS Accessibility permission is denied), or for keyboard/programmatic
/// resizes where no mouse button is held.
pub struct ResizeDetector {
    /// Set when a resize event has been seen and we haven't confirmed the end yet.
    active: Cell<bool>,
    /// Deadline for the timer-based fallback (programmatic / non-mouse resizes).
    deadline: Cell<Option<Instant>>,
    /// Cached device_query state — cheap to call, avoids re-initializing each poll.
    /// `None` on Linux without an X display (pure Wayland), where `DeviceState::new`
    /// would panic; resizes then end on the idle deadline alone.
    #[cfg(not(target_family = "wasm"))]
    device_state: Option<DeviceState>,
}

/// Fallback idle period when global mouse state is unavailable or no button is held.
const IDLE_THRESHOLD: Duration = Duration::from_millis(150);

impl ResizeDetector {
    pub fn new() -> Self {
        Self {
            active: Cell::new(false),
            deadline: Cell::new(None),
            #[cfg(not(target_family = "wasm"))]
            device_state: DeviceState::checked_new(),
        }
    }

    /// Call on every `WindowEvent::Resized` from the winit event loop.
    pub fn on_resize_event(&self) {
        self.active.set(true);
        self.deadline.set(Some(Instant::now() + IDLE_THRESHOLD));
    }

    /// Returns `true` while the resize is in progress.
    pub fn is_resizing(&self) -> bool {
        if !self.active.get() {
            return false;
        }

        #[cfg(not(target_family = "wasm"))]
        if let Some(device_state) = &self.device_state {
            // Poll the real global left-button state. If it's held the user is
            // still dragging the resize handle; keep deferring and push the
            // fallback deadline out so it doesn't fire during the drag.
            let mouse = device_state.get_mouse();
            // button_pressed is 1-based; index 1 = left button.
            let left_held = mouse.button_pressed.get(1).copied().unwrap_or(false);

            if left_held {
                self.deadline.set(Some(Instant::now() + IDLE_THRESHOLD));
                return true;
            }
        }

        // Without global mouse state, resize events keep extending this deadline
        // until the event stream stops.
        match self.deadline.get() {
            Some(deadline) if Instant::now() < deadline => {
                // Could be a split-second between event and button release detection;
                // keep active until the deadline just to be safe.
                true
            }
            _ => {
                self.active.set(false);
                self.deadline.set(None);
                false
            }
        }
    }
}

impl Default for ResizeDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::*;

    fn without_device_state() -> ResizeDetector {
        ResizeDetector {
            active: Cell::new(false),
            deadline: Cell::new(None),
            device_state: None,
        }
    }

    #[test]
    fn unavailable_mouse_state_is_idle_until_a_resize() {
        let detector = without_device_state();

        assert!(!detector.is_resizing());
        detector.on_resize_event();
        assert!(detector.is_resizing());
    }

    #[test]
    fn unavailable_mouse_state_finishes_after_idle_deadline() {
        let detector = without_device_state();
        detector.on_resize_event();
        detector.deadline.set(Some(Instant::now() - IDLE_THRESHOLD));

        assert!(!detector.is_resizing());
        assert!(!detector.active.get());
        assert!(detector.deadline.get().is_none());
        assert!(!detector.is_resizing());
    }

    #[test]
    fn further_resize_events_extend_the_idle_deadline() {
        let detector = without_device_state();
        detector.on_resize_event();
        detector.deadline.set(Some(Instant::now() - IDLE_THRESHOLD));

        detector.on_resize_event();

        assert!(detector.is_resizing());
    }
}
