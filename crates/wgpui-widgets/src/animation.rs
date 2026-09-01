//! Frontend animation definitions and the clock-driven 2.0 animation driver.
//!
//! Sampling is separate from rendering: the sampled value builds an ordinary
//! `Description`, so reconciliation and the scene pipeline need no special
//! animation path.

use std::rc::Rc;
use std::time::{Duration, Instant};
use wgpui_core::element::Element;
use wgpui_core::geometry::{Pixels, Point, Size};
use wgpui_core::reconcile::description::{Description, ElementId};
use wgpui_core::window::animation::{AnimationScheduler, animation_start};

/// A transform that can be sampled by an animated element.
///
/// The retained image primitive currently applies scale and translation to
/// its axis-aligned bounds. Rotation is kept in the value and reconciliation
/// key so a renderer that supports affine sprites can consume it without
/// changing the animation API.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transformation {
    pub rotation: f32,
    pub scale: [f32; 2],
    pub translation: [f32; 2],
}

impl Default for Transformation {
    fn default() -> Self {
        Self {
            rotation: 0.0,
            scale: [1.0, 1.0],
            translation: [0.0, 0.0],
        }
    }
}

impl Transformation {
    pub fn rotate(rotation: f32) -> Self {
        Self {
            rotation,
            ..Self::default()
        }
    }

    pub fn scale(scale: Size<Pixels>) -> Self {
        Self {
            scale: [scale.width.value(), scale.height.value()],
            ..Self::default()
        }
    }

    pub fn translate(translation: Point<Pixels>) -> Self {
        Self {
            translation: [translation.x.value(), translation.y.value()],
            ..Self::default()
        }
    }

    pub fn with_scaling(mut self, scale: Size<Pixels>) -> Self {
        self.scale = [scale.width.value(), scale.height.value()];
        self
    }

    pub fn with_translation(mut self, translation: Point<Pixels>) -> Self {
        self.translation = [translation.x.value(), translation.y.value()];
        self
    }

    pub fn with_rotation(mut self, rotation: f32) -> Self {
        self.rotation = rotation;
        self
    }
}

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

/// Convert normalized progress to one complete turn in degrees.
pub fn percentage(value: f32) -> f32 {
    value * 360.0
}

/// Add a soft overshoot to an easing function.
pub fn bounce(easing: impl Fn(f32) -> f32 + 'static) -> impl Fn(f32) -> f32 + 'static {
    move |value| {
        let value = easing(value.clamp(0.0, 1.0));
        if value < 0.5 {
            2.0 * value * value
        } else {
            let remaining = 1.0 - value;
            1.0 - 2.0 * remaining * remaining
        }
    }
}

pub fn pulsating_between(minimum: f32, maximum: f32) -> impl Fn(f32) -> f32 + 'static {
    move |value| minimum + (maximum - minimum) * (0.5 - 0.5 * (std::f32::consts::PI * 2.0 * value).cos())
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

    #[test]
    fn transformation_builders_preserve_independent_components() {
        let transformed = Transformation::rotate(percentage(0.25))
            .with_scaling(Size::new(Pixels(2.0), Pixels(3.0)))
            .with_translation(Point::new(Pixels(4.0), Pixels(-5.0)));

        assert_eq!(transformed.rotation, 90.0);
        assert_eq!(transformed.scale, [2.0, 3.0]);
        assert_eq!(transformed.translation, [4.0, -5.0]);
    }

    #[test]
    fn bounce_stays_within_animation_progress() {
        let easing = bounce(linear);
        for value in [0.0, 0.25, 0.5, 0.75, 1.0] {
            assert!((0.0..=1.0).contains(&easing(value)));
        }
    }
}
