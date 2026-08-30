//! Frame requests used by the 2.0 frontend animation driver.

use std::time::{Duration, Instant};

pub use super::timer::{TimerHandle, TimerState};
use super::timer::{TimerId, TimerScheduler};

/// A coalescing request queue for a window's next animation frame.
#[derive(Debug, Default)]
pub struct AnimationScheduler {
    requested: bool,
}

impl AnimationScheduler {
    /// Create an idle scheduler.
    pub const fn new() -> Self {
        Self { requested: false }
    }
    /// Request the next display frame.
    pub fn request_animation_frame(&mut self) {
        self.requested = true;
    }
    /// Whether a frame has been requested.
    pub const fn is_requested(&self) -> bool {
        self.requested
    }
    /// Consume the request when the platform begins drawing it.
    pub const fn take_request(&mut self) -> bool {
        let requested = self.requested;
        self.requested = false;
        requested
    }
}

/// Return a safe deadline for a frame paced at `frame_rate`.
pub fn next_frame_deadline(now: Instant, frame_rate: u32) -> Instant {
    now + Duration::from_secs_f64(1.0 / f64::from(frame_rate.max(1)))
}

#[derive(Debug, Default)]
pub struct WindowTimers(TimerScheduler);

impl WindowTimers {
    pub fn schedule(&mut self, now: Instant, delay: Duration) -> TimerHandle {
        self.0.schedule(now, delay)
    }
    pub fn cancel(&mut self, timer: TimerHandle) -> bool {
        self.0.cancel(timer)
    }
    pub fn due(&mut self, now: Instant) -> Vec<TimerId> {
        self.0.take_due(now)
    }
    pub fn next_deadline(&self) -> Option<Instant> {
        self.0.next_deadline()
    }
    pub fn state(&self, timer: TimerHandle) -> Option<TimerState> {
        self.0.state(timer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn requests_are_coalesced_until_consumed() {
        let mut scheduler = AnimationScheduler::new();
        scheduler.request_animation_frame();
        scheduler.request_animation_frame();
        assert!(scheduler.is_requested());
        assert!(scheduler.take_request());
        assert!(!scheduler.take_request());
    }
    #[test]
    fn zero_rate_still_has_a_deadline() {
        let now = Instant::now();
        assert!(next_frame_deadline(now, 0) > now);
    }
}
