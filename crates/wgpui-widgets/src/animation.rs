//! Frontend animation definitions and the clock-driven 2.0 animation driver.
//!
//! Sampling is separate from rendering: the sampled value builds an ordinary
//! `Description`, so reconciliation and the scene pipeline need no special
//! animation path.

use std::rc::Rc;
use std::time::{Duration, Instant};
use wgpui_core::element::Element;
use wgpui_core::reconcile::description::{Description, ElementId};
use wgpui_core::window::animation::{AnimationScheduler, animation_start};

/// An animation definition, compatible with the legacy public names.
#[derive(Clone)]
pub struct Animation {
    /// Time spent in one pass.
    pub duration: Duration,
    /// Repeat forever when false; advance to the next definition when true.
    pub oneshot: bool,
    /// Maps linear progress to eased progress.
    pub easing: Rc<dyn Fn(f32) -> f32>,
}

impl Animation {
    /// Create a one-shot linear animation.
    pub fn new(duration: Duration) -> Self {
        Self {
            duration,
            oneshot: true,
            easing: Rc::new(linear),
        }
    }
    /// Repeat this animation forever.
    pub fn repeat(mut self) -> Self {
        self.oneshot = false;
        self
    }
    /// Set the easing function.
    pub fn with_easing(mut self, easing: impl Fn(f32) -> f32 + 'static) -> Self {
        self.easing = Rc::new(easing);
        self
    }
}

/// The result of sampling an animation sequence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnimationSample {
    pub animation_index: usize,
    pub progress: f32,
    pub finished: bool,
}

/// A clock-bound sequence of animation definitions.
#[derive(Clone)]
pub struct AnimationTimeline {
    animations: Vec<Animation>,
    started: Instant,
}

impl AnimationTimeline {
    /// Start a non-empty sequence at `started`.
    pub fn new(animations: Vec<Animation>, started: Instant) -> Option<Self> {
        (!animations.is_empty()).then_some(Self {
            animations,
            started,
        })
    }
    /// Sample at an absolute time.
    pub fn sample_at(&self, now: Instant) -> AnimationSample {
        let mut elapsed = now.saturating_duration_since(self.started);
        for (index, animation) in self.animations.iter().enumerate() {
            let duration = animation.duration;
            if duration.is_zero() || elapsed >= duration {
                if !animation.oneshot {
                    let progress = if duration.is_zero() {
                        1.0
                    } else {
                        (elapsed.as_secs_f64() / duration.as_secs_f64()).fract() as f32
                    };
                    return sample(animation, index, progress, false);
                }
                elapsed = elapsed.saturating_sub(duration);
                if index + 1 == self.animations.len() {
                    return sample(animation, index, 1.0, true);
                }
            } else {
                return sample(
                    animation,
                    index,
                    elapsed.as_secs_f64() as f32 / duration.as_secs_f64() as f32,
                    false,
                );
            }
        }
        let index = self.animations.len() - 1;
        sample(&self.animations[index], index, 1.0, true)
    }
    /// Request another frame while this sequence is active.
    pub fn request_next_frame(&self, now: Instant, scheduler: &mut AnimationScheduler) {
        if !self.sample_at(now).finished {
            scheduler.request_animation_frame();
        }
    }
}

fn sample(
    animation: &Animation,
    animation_index: usize,
    progress: f32,
    finished: bool,
) -> AnimationSample {
    AnimationSample {
        animation_index,
        progress: (animation.easing)(progress.clamp(0.0, 1.0)).clamp(0.0, 1.0),
        finished,
    }
}

/// Extension for turning an element into an animated description.
pub trait AnimationExt: Sized {
    /// Apply a one-definition animation.
    fn with_animation(
        self,
        id: impl Into<ElementId>,
        animation: Animation,
        animator: impl Fn(Self, f32) -> Self + 'static,
    ) -> AnimationElement<Self>;
    /// Apply a sequence of animations.
    fn with_animations(
        self,
        id: impl Into<ElementId>,
        animations: Vec<Animation>,
        animator: impl Fn(Self, usize, f32) -> Self + 'static,
    ) -> AnimationElement<Self>;
}

impl<E: crate::div::IntoDescription> AnimationExt for E {
    fn with_animation(
        self,
        id: impl Into<ElementId>,
        animation: Animation,
        animator: impl Fn(Self, f32) -> Self + 'static,
    ) -> AnimationElement<Self> {
        self.with_animations(id, vec![animation], move |element, _, progress| {
            animator(element, progress)
        })
    }

    fn with_animations(
        self,
        id: impl Into<ElementId>,
        animations: Vec<Animation>,
        animator: impl Fn(Self, usize, f32) -> Self + 'static,
    ) -> AnimationElement<Self> {
        let id = id.into();
        let started = animation_start(&id, Instant::now());
        let timeline = match AnimationTimeline::new(animations, started) {
            Some(timeline) => timeline,
            None => AnimationTimeline {
                animations: vec![Animation::new(Duration::ZERO)],
                started,
            },
        };
        AnimationElement {
            id,
            element: self,
            timeline,
            animator: Box::new(animator),
        }
    }
}

/// An animated element whose output remains a normal `Description`.
pub struct AnimationElement<E> {
    id: ElementId,
    element: E,
    timeline: AnimationTimeline,
    animator: Box<dyn Fn(E, usize, f32) -> E>,
}

impl<E: crate::div::IntoDescription> AnimationElement<E> {
    /// Produce a description at a deterministic elapsed offset.
    pub fn describe_after(self, elapsed: Duration) -> Description {
        let started = self.timeline.started;
        self.describe_at(started + elapsed)
    }
    /// Produce the description at a chosen time.
    pub fn describe_at(self, now: Instant) -> Description {
        let sample = self.timeline.sample_at(now);
        let mut description =
            (self.animator)(self.element, sample.animation_index, sample.progress)
                .into_description();
        if description.element_id().is_none() {
            description = description.id(self.id);
        }
        if !sample.finished {
            description = description.active_animation();
        }
        description
    }
    /// Request a future frame if the sequence is still active.
    pub fn request_next_frame(&self, now: Instant, scheduler: &mut AnimationScheduler) {
        self.timeline.request_next_frame(now, scheduler);
    }
}

impl<E: crate::div::IntoDescription + 'static> Element for AnimationElement<E> {
    fn into_description(self) -> Description {
        self.describe_at(Instant::now())
    }
}

/// Linear easing.
pub fn linear(value: f32) -> f32 {
    value
}
/// Quadratic ease-in.
pub fn quadratic(value: f32) -> f32 {
    value * value
}
/// Symmetric quadratic ease-in-out.
pub fn ease_in_out(value: f32) -> f32 {
    if value < 0.5 {
        2.0 * value * value
    } else {
        1.0 - (-2.0 * value + 2.0).powi(2) / 2.0
    }
}
/// Quintic ease-out.
pub fn ease_out_quint(value: f32) -> f32 {
    1.0 - (1.0 - value).powi(5)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::div::div;
    use crate::styled::Styled;
    #[test]
    fn timeline_advances_and_clamps_a_chain() {
        let start = Instant::now();
        let timeline = AnimationTimeline::new(
            vec![
                Animation::new(Duration::from_millis(100)),
                Animation::new(Duration::from_millis(100)).with_easing(|value| value * 2.0),
            ],
            start,
        )
        .expect("definitions");
        assert_eq!(
            timeline
                .sample_at(start + Duration::from_millis(50))
                .progress,
            0.5
        );
        assert_eq!(
            timeline
                .sample_at(start + Duration::from_millis(125))
                .animation_index,
            1
        );
        assert_eq!(
            timeline
                .sample_at(start + Duration::from_millis(125))
                .progress,
            0.5
        );
        assert!(
            timeline
                .sample_at(start + Duration::from_millis(250))
                .finished
        );
    }
    #[test]
    fn repeat_requests_another_frame() {
        let start = Instant::now();
        let timeline = AnimationTimeline::new(
            vec![Animation::new(Duration::from_millis(10)).repeat()],
            start,
        )
        .expect("definition");
        assert_eq!(
            timeline
                .sample_at(start + Duration::from_millis(25))
                .progress,
            0.5
        );
        let mut scheduler = AnimationScheduler::new();
        timeline.request_next_frame(start + Duration::from_millis(25), &mut scheduler);
        assert!(scheduler.take_request());
    }

    #[test]
    fn a_repeating_definition_stays_active_when_it_is_part_of_a_chain() {
        let start = Instant::now();
        let timeline = AnimationTimeline::new(
            vec![
                Animation::new(Duration::from_millis(10)).repeat(),
                Animation::new(Duration::from_millis(10)),
            ],
            start,
        )
        .expect("definitions");

        let sample = timeline.sample_at(start + Duration::from_millis(25));
        assert_eq!(sample.animation_index, 0);
        assert!(!sample.finished);
        assert_eq!(sample.progress, 0.5);
    }

    #[test]
    fn active_animation_metadata_survives_as_a_description_property() {
        let description = div()
            .bg([1.0, 0.0, 0.0, 1.0])
            .with_animation(
                "active",
                Animation::new(Duration::from_secs(1)).repeat(),
                |element, progress| element.opacity(progress),
            )
            .describe_after(Duration::from_millis(100));

        assert!(description.has_active_animation());
    }

    #[test]
    fn finished_animation_does_not_keep_the_frame_loop_alive() {
        let description = div()
            .bg([1.0, 0.0, 0.0, 1.0])
            .with_animation(
                "finished",
                Animation::new(Duration::from_millis(10)),
                |element, progress| element.opacity(progress),
            )
            .describe_after(Duration::from_millis(10));

        assert!(!description.has_active_animation());
    }

    #[test]
    fn sampled_widget_output_is_an_ordinary_description_diff() {
        let first = div().bg([0.0, 0.0, 0.0, 1.0]).with_animation(
            "fade",
            Animation::new(Duration::from_millis(100)),
            |element, progress| element.bg([progress, 0.0, 0.0, 1.0]),
        );
        let first = first.describe_after(Duration::from_millis(0));
        let second = div().bg([0.0, 0.0, 0.0, 1.0]).with_animation(
            "fade",
            Animation::new(Duration::from_millis(100)),
            |element, progress| element.bg([progress, 0.0, 0.0, 1.0]),
        );
        let second = second.describe_after(Duration::from_millis(50));
        assert!(
            !first
                .key()
                .expect("key")
                .compare(second.key().expect("key"))
                .is_empty()
        );
    }
}
