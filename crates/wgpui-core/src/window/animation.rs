//! Frame requests used by the 2.0 frontend animation driver.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

pub use super::timer::{TimerHandle, TimerState};
use super::timer::{TimerId, TimerScheduler};
use crate::reconcile::description::ElementId;

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

/// Retained start times for declarative animations built during repeated
/// renders of the same window.
///
/// The widget API intentionally creates a fresh element value for every
/// render. This clock is the small piece of state that gives an animation with
/// a stable element id one continuous timeline without making the description
/// or the renderer own application state.
#[derive(Debug, Default)]
pub struct AnimationClock {
    starts: HashMap<ElementId, Instant>,
    touched: HashSet<ElementId>,
}

impl AnimationClock {
    /// Create an empty clock.
    pub fn new() -> Self {
        Self::default()
    }

    fn start_for(&mut self, id: &ElementId, fallback: Instant) -> Instant {
        self.touched.insert(id.clone());
        *self.starts.entry(id.clone()).or_insert(fallback)
    }

    fn retain_touched(&mut self) {
        self.starts.retain(|id, _| self.touched.contains(id));
        self.touched.clear();
    }
}

thread_local! {
    static ACTIVE_CLOCK: RefCell<Option<AnimationClock>> = const { RefCell::new(None) };
}

/// Run a frame builder with a window-owned animation clock active.
///
/// The clock is returned even when the builder returns a value, allowing the
/// caller to keep the state alongside the rest of its retained frame loop.
pub fn with_animation_clock<R>(
    mut clock: AnimationClock,
    builder: impl FnOnce() -> R,
) -> (AnimationClock, R) {
    clock.touched.clear();
    let previous = ACTIVE_CLOCK.with(|active| active.replace(Some(clock)));
    let result = builder();
    let Some(mut clock) = ACTIVE_CLOCK.with(|active| active.replace(previous)) else {
        return (AnimationClock::default(), result);
    };
    clock.retain_touched();
    (clock, result)
}

/// Resolve an element's retained animation start time.
pub fn animation_start(id: &ElementId, fallback: Instant) -> Instant {
    ACTIVE_CLOCK.with(|active| {
        active
            .borrow_mut()
            .as_mut()
            .map_or(fallback, |clock| clock.start_for(id, fallback))
    })
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

    #[test]
    fn a_stable_element_id_keeps_its_start_time_across_frame_builds() {
        let id = ElementId::from("fade");
        let first = Instant::now();
        let (clock, _) =
            with_animation_clock(AnimationClock::new(), || animation_start(&id, first));
        let later = first + Duration::from_secs(1);
        let (clock, second) = with_animation_clock(clock, || animation_start(&id, later));

        assert_eq!(second, first);
        let (clock, _) = with_animation_clock(clock, || ());
        assert!(
            clock.starts.is_empty(),
            "unseen animation state is reclaimed"
        );
    }

    #[test]
    fn independent_ids_do_not_share_animation_state() {
        let first_id = ElementId::from("first");
        let second_id = ElementId::from("second");
        let first = Instant::now();
        let second = first + Duration::from_secs(1);
        let (clock, starts) = with_animation_clock(AnimationClock::new(), || {
            (
                animation_start(&first_id, first),
                animation_start(&second_id, second),
            )
        });

        assert_eq!(starts, (first, second));
        assert_eq!(clock.starts.len(), 2);
    }
}
