//! Momentum ("flick") scrolling for trackpads (issue #42).
//!
//! A trackpad that is lifted while still moving should keep scrolling and
//! slow down smoothly. macOS sends that coast itself, as more scroll events
//! after the fingers lift. Elsewhere winit reports the lift (an `Ended` phase,
//! from Wayland's `axis_stop`) and nothing after it, so the window coasts on
//! its own: [`ScrollMomentum`] measures how fast the fingers were moving when
//! they lifted and hands back a [`Fling`], which says how far the content
//! moves between any two moments after that.
//!
//! The coast decays exponentially: velocity `v0 · e^(-t/τ)`, which moves the
//! content `v0 · τ · (1 - e^(-t/τ))` in total by time `t`. Wheel mice report
//! line deltas and Windows and X11 report no lift, so neither ever flings.

use std::collections::VecDeque;
use std::time::Duration;

use crate::time_ext::Instant;

use crate::{Pixels, Point, ScrollDelta, ScrollWheelEvent, TouchPhase, point, px};

/// Whether the platform sends its own momentum events, so the window must not
/// add a second coast on top.
pub(crate) const PLATFORM_SENDS_MOMENTUM: bool = cfg!(target_os = "macos");

/// How long the coast takes to lose about two thirds of its speed.
const TIME_CONSTANT: Duration = Duration::from_millis(325);
/// Only the movement in this window before the lift counts toward its speed.
const VELOCITY_WINDOW: Duration = Duration::from_millis(100);
/// Fingers that rested this long before lifting were not flicking.
const MAX_REST_BEFORE_LIFT: Duration = Duration::from_millis(50);
/// Slower lifts than this, in pixels per second, don't coast.
const MIN_FLING_SPEED: f32 = 250.;
/// The coast stops once it is slower than this, in pixels per second.
const STOP_SPEED: f32 = 15.;
/// Faster flicks are capped to this, in pixels per second.
const MAX_FLING_SPEED: f32 = 8000.;
/// How often a coast moves the content.
pub(crate) const FLING_STEP: Duration = Duration::from_millis(8);

/// Follows one trackpad gesture and decides whether its lift starts a coast.
#[derive(Default)]
pub(crate) struct ScrollMomentum {
    /// Recent pixel deltas of the gesture, with when they arrived.
    samples: VecDeque<(Instant, Point<Pixels>)>,
}

impl ScrollMomentum {
    /// Feed a scroll event from the platform. Returns the coast its lift
    /// starts, if it is a fast enough lift.
    pub(crate) fn observe(&mut self, event: &ScrollWheelEvent, now: Instant) -> Option<Fling> {
        if PLATFORM_SENDS_MOMENTUM {
            return None;
        }
        let ScrollDelta::Pixels(delta) = event.delta else {
            // A wheel: no gesture to follow.
            self.samples.clear();
            return None;
        };
        match event.touch_phase {
            TouchPhase::Started => self.samples.clear(),
            TouchPhase::Cancelled => {
                self.samples.clear();
                return None;
            }
            TouchPhase::Moved | TouchPhase::Ended => {}
        }
        if delta != Point::default() {
            self.samples.push_back((now, delta));
            while self
                .samples
                .front()
                .is_some_and(|(at, _)| now.duration_since(*at) > VELOCITY_WINDOW)
            {
                self.samples.pop_front();
            }
        }
        if event.touch_phase != TouchPhase::Ended {
            return None;
        }
        let velocity = self.lift_velocity(now);
        self.samples.clear();
        let velocity = velocity?;
        let speed = velocity.x.hypot(velocity.y);
        if speed < MIN_FLING_SPEED {
            return None;
        }
        let scale = (MAX_FLING_SPEED / speed).min(1.);
        Some(Fling {
            velocity: point(velocity.x * scale, velocity.y * scale),
            position: event.position,
            modifiers: event.modifiers,
            started: now,
            elapsed: Duration::ZERO,
            began: false,
        })
    }

    /// The fingers' speed when they lifted at `now`, in pixels per second:
    /// the movement since the first recent sample over the time it took.
    fn lift_velocity(&self, now: Instant) -> Option<Point<f32>> {
        let (last, _) = *self.samples.back()?;
        if now.duration_since(last) > MAX_REST_BEFORE_LIFT {
            return None;
        }
        let (first, _) = *self.samples.front()?;
        let span = last.duration_since(first).as_secs_f32();
        if self.samples.len() < 2 || span <= 0. {
            return None;
        }
        // The first sample's movement happened before the span began.
        let moved = self
            .samples
            .iter()
            .skip(1)
            .fold(Point::<f32>::default(), |sum, (_, delta)| {
                point(sum.x + f32::from(delta.x), sum.y + f32::from(delta.y))
            });
        Some(point(moved.x / span, moved.y / span))
    }
}

/// A coast after a flick: the content keeps moving at a decaying speed.
#[derive(Clone, Debug)]
pub(crate) struct Fling {
    /// The fingers' speed at the lift, in pixels per second.
    velocity: Point<f32>,
    /// Where the pointer was when the fingers lifted; the coast scrolls
    /// whatever is there.
    pub(crate) position: Point<Pixels>,
    pub(crate) modifiers: crate::Modifiers,
    started: Instant,
    /// How far into the coast the content has already moved.
    elapsed: Duration,
    /// Whether the coast has sent its first event, which opens it with a
    /// `Started` phase the way macOS opens its own momentum events.
    began: bool,
}

impl Fling {
    /// The scroll event that moves the content from where the last one left
    /// it to where the coast is at `now`, or `None` once it has stopped. The
    /// first opens the coast as a new gesture (`Started`), the last closes it
    /// (`Ended`).
    pub(crate) fn step(&mut self, now: Instant) -> Option<ScrollWheelEvent> {
        let before = self.elapsed;
        let after = now.duration_since(self.started);
        if after <= before {
            return Some(self.event(Point::default(), TouchPhase::Moved));
        }
        self.elapsed = after;
        if self.speed_at(before) < STOP_SPEED {
            return None;
        }
        let moved = |t: Duration| {
            let tau = TIME_CONSTANT.as_secs_f32();
            tau * (1. - (-t.as_secs_f32() / tau).exp())
        };
        let distance = moved(after) - moved(before);
        let delta = point(
            px(self.velocity.x * distance),
            px(self.velocity.y * distance),
        );
        let phase = if self.speed_at(after) < STOP_SPEED {
            TouchPhase::Ended
        } else if !self.began {
            TouchPhase::Started
        } else {
            TouchPhase::Moved
        };
        self.began = true;
        Some(self.event(delta, phase))
    }

    fn speed_at(&self, t: Duration) -> f32 {
        let decay = (-t.as_secs_f32() / TIME_CONSTANT.as_secs_f32()).exp();
        self.velocity.x.hypot(self.velocity.y) * decay
    }

    fn event(&self, delta: Point<Pixels>, touch_phase: TouchPhase) -> ScrollWheelEvent {
        ScrollWheelEvent {
            position: self.position,
            delta: ScrollDelta::Pixels(delta),
            modifiers: self.modifiers,
            touch_phase,
        }
    }

    /// How far the whole coast moves the content.
    #[cfg(test)]
    pub(crate) fn total_distance(&self) -> Point<f32> {
        let tau = TIME_CONSTANT.as_secs_f32();
        point(self.velocity.x * tau, self.velocity.y * tau)
    }
}

#[cfg(all(test, not(target_os = "macos")))]
mod tests {
    use super::*;

    fn event(dy: f32, phase: TouchPhase) -> ScrollWheelEvent {
        ScrollWheelEvent {
            position: point(px(10.), px(10.)),
            delta: ScrollDelta::Pixels(point(px(0.), px(dy))),
            modifiers: Default::default(),
            touch_phase: phase,
        }
    }

    /// Feed a gesture of `steps` events `every` apart, each `dy` pixels, then
    /// lift `rest` after the last one.
    fn gesture(dy: f32, steps: u32, every: Duration, rest: Duration) -> Option<Fling> {
        let mut momentum = ScrollMomentum::default();
        let start = Instant::now();
        let mut now = start;
        assert!(
            momentum
                .observe(&event(dy, TouchPhase::Started), now)
                .is_none()
        );
        for _ in 1..steps {
            now += every;
            assert!(
                momentum
                    .observe(&event(dy, TouchPhase::Moved), now)
                    .is_none()
            );
        }
        momentum.observe(&event(0., TouchPhase::Ended), now + rest)
    }

    #[test]
    fn a_fast_lift_coasts_at_the_fingers_speed_and_stops() {
        // 20 px every 10 ms: 2000 px/s.
        let mut fling = gesture(20., 10, Duration::from_millis(10), Duration::from_millis(5))
            .expect("a fast lift coasts");
        let expected = fling.total_distance().y;
        assert!((expected - 2000. * 0.325).abs() < 1., "{expected}");

        let mut now = fling.started;
        let mut moved = 0.;
        let mut previous = f32::MAX;
        let mut ended = false;
        for i in 0..1000 {
            now += FLING_STEP;
            let Some(step) = fling.step(now) else { break };
            if i == 0 {
                assert_eq!(
                    step.touch_phase,
                    TouchPhase::Started,
                    "the coast opens a gesture"
                );
            }
            let ScrollDelta::Pixels(delta) = step.delta else {
                unreachable!()
            };
            let dy = f32::from(delta.y);
            assert!(dy <= previous + 0.001, "the coast only slows down");
            previous = dy;
            moved += dy;
            ended = step.touch_phase == TouchPhase::Ended;
            if ended {
                break;
            }
        }
        assert!(ended, "the coast ends with an Ended event");
        assert!(
            moved > expected * 0.95 && moved <= expected,
            "moved {moved} of {expected}"
        );
    }

    #[test]
    fn slow_rested_or_wheel_scrolls_do_not_coast() {
        // 1 px every 10 ms: 100 px/s.
        assert!(gesture(1., 10, Duration::from_millis(10), Duration::ZERO).is_none());
        // Fast, but the fingers rested before lifting.
        assert!(
            gesture(
                20.,
                10,
                Duration::from_millis(10),
                Duration::from_millis(80)
            )
            .is_none()
        );
        // A single event has no speed.
        assert!(gesture(50., 1, Duration::from_millis(10), Duration::ZERO).is_none());

        let mut momentum = ScrollMomentum::default();
        let wheel = ScrollWheelEvent {
            delta: ScrollDelta::Lines(point(0., 3.)),
            touch_phase: TouchPhase::Ended,
            ..Default::default()
        };
        assert!(momentum.observe(&wheel, Instant::now()).is_none());
    }
}
