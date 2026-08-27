//! The per-frame compositing decision a `.boundary()` makes, and the retained
//! state it makes it from. See docs/gpu-native-architecture.md §4.1 and §5.4.
//!
//! Not in §3.1's literal file map — a deliberate addition, recorded in
//! `docs/phase-2-results.md`. §3.1 gives `boundary/` a policy file (what an
//! author may tune) and an identity file (how a boundary finds itself), and no
//! home for the thing that consumes both plus
//! [`crate::invalidation::reason::Reason`] once per frame. In R-N/SFD that
//! decision had no separate home either, because it was interleaved into
//! `Interactivity`'s paint block inside `div.rs`; §3.4 lists breaking that
//! block apart as one of the four seams the widgets crate splits along, and
//! this is the half of it that is not any element type's business.
//!
//! # The decision, stated once
//!
//! A boundary composites transform-only when **both** of these hold:
//!
//! 1. Its content is clean — nothing inside it needed re-emitting this frame.
//! 2. The signal that woke the frame permits it — [`Reason::Scroll`], not
//!    [`Reason::DataChanged`].
//!
//! Under §4.0's ambient reconciliation, condition 1 is measured, not assumed:
//! the reconciler re-diffs every element inside the boundary every frame
//! whether or not the boundary exists, so "the content is clean" is a fact the
//! frame already established rather than something the boundary's key had to
//! promise. That is a genuine change from SFD §1.1, where the tagged
//! notification was the *only* evidence available and a wrong key meant
//! silently stale UI. Requiring condition 2 as well is therefore deliberately
//! conservative: it costs a `DataChanged`-signalled pure-scroll frame one
//! ordinary recomposite, and it buys that a bug in any element's `diff_key` can
//! only ever produce a slow frame, never a frame that slid stale content into
//! view. §4.1 asks for the signal "from day one — not retrofitted," and this is
//! what consuming it looks like once the diff is ambient underneath it.
//!
//! # What is decided here, and what is emphatically not
//!
//! Nothing in this file allocates, pools, or draws a texture.
//! [`Retention::Texture`] is a *decision* about a boundary, recorded and
//! observable; §3.1 puts every live `wgpu::Device` in `wgpui-wgpu` and §8 puts
//! the compositing entry that would consume this decision in Phase 4.

use crate::boundary::policy::{BoundaryPolicy, Retention};
use crate::invalidation::axes::Invalidation;
use crate::invalidation::reason::Reason;
use crate::scene::layer::{BoundaryId, LayerId, LayerKey, LayerTransform};
use std::collections::HashMap;

/// What a boundary does with its content this frame.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Composite {
    /// Neither the content nor the transform moved. Zero work, zero upload.
    Clean,
    /// The content is untouched and only the composite transform changed —
    /// R-N §3.2's "a 1px scroll sets one flag on one layer. No render, no
    /// reconcile, no layout, no prepaint, no paint, no upload — one changed
    /// matrix." This is the fast path §8's Phase 2 gate names.
    TransformOnly,
    /// Something inside the boundary needed re-emitting, so its content is
    /// patched into residency as usual.
    Redisplay,
}

impl Composite {
    /// Whether this frame left the boundary's resident primitives untouched.
    pub const fn leaves_content_resident(self) -> bool {
        matches!(self, Composite::Clean | Composite::TransformOnly)
    }

    /// The invalidation axes this decision raises on the boundary's layer.
    pub const fn invalidation(self) -> Invalidation {
        match self {
            Composite::Clean => Invalidation::empty(),
            Composite::TransformOnly => Invalidation::TRANSFORM,
            Composite::Redisplay => Invalidation::DISPLAY,
        }
    }
}

/// One boundary's retained compositing state.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BoundaryState {
    policy: BoundaryPolicy,
    layer: LayerId,
    transform: LayerTransform,
    retention: Retention,
    primitive_count: usize,
    last_visited_frame: u64,
}

impl BoundaryState {
    /// The policy this boundary was declared with.
    pub const fn policy(&self) -> BoundaryPolicy {
        self.policy
    }

    /// The layer this boundary's content lives in.
    pub const fn layer(&self) -> LayerId {
        self.layer
    }

    /// Where this boundary's content currently composites.
    pub const fn transform(&self) -> LayerTransform {
        self.transform
    }

    /// Whether this boundary is texture-retained or primitive-retained, as of
    /// the last frame its primitive count was resolved.
    pub const fn retention(&self) -> Retention {
        self.retention
    }

    /// How many primitives the boundary held at that point.
    pub const fn primitive_count(&self) -> usize {
        self.primitive_count
    }

    /// The last frame this boundary appeared in the tree.
    pub const fn last_visited_frame(&self) -> u64 {
        self.last_visited_frame
    }
}

/// What one boundary did this frame, as inspectable data.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BoundaryComposite {
    /// The boundary.
    pub boundary: BoundaryId,
    /// Its layer.
    pub layer: LayerId,
    /// What it did.
    pub composite: Composite,
    /// Whether it is texture-retained or primitive-retained.
    pub retention: Retention,
    /// Where its content composites after this frame.
    pub transform: LayerTransform,
    /// The axes this decision raised on its layer.
    pub invalidation: Invalidation,
}

/// Every live compositing boundary, across frames.
#[derive(Debug, Default)]
pub struct Compositor {
    boundaries: HashMap<BoundaryId, BoundaryState>,
}

impl Compositor {
    /// A compositor holding no boundaries.
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare that `boundary` exists this frame with `policy`, returning the
    /// layer its content lives in.
    ///
    /// Idempotent across frames: a boundary re-declared under the same
    /// identity keeps its transform and its residency, which is the entire
    /// point of deriving that identity positionally (SFD §1.0) rather than
    /// requiring a name.
    pub fn visit(
        &mut self,
        boundary: BoundaryId,
        policy: BoundaryPolicy,
        frame: u64,
    ) -> LayerId {
        let layer = LayerId::from_key(LayerKey::untiled(boundary));
        let state = self
            .boundaries
            .entry(boundary)
            .or_insert_with(|| BoundaryState {
                policy,
                layer,
                transform: LayerTransform::IDENTITY,
                retention: Retention::Primitives,
                primitive_count: 0,
                last_visited_frame: frame,
            });
        state.policy = policy;
        state.last_visited_frame = frame;
        state.layer
    }

    /// Move a boundary's content to `transform`, reporting whether that is a
    /// change from where it already was.
    ///
    /// A boundary that is not live reports `false` rather than being created:
    /// a transform without a declaration has no content to apply to.
    pub fn set_transform(&mut self, boundary: BoundaryId, transform: LayerTransform) -> bool {
        match self.boundaries.get_mut(&boundary) {
            Some(state) if state.transform != transform => {
                state.transform = transform;
                true
            }
            _ => false,
        }
    }

    /// Decide what a boundary does this frame.
    ///
    /// `content_dirty` is the walk's own measurement — whether any element
    /// inside the boundary needed re-emitting — and `reason` is the signal that
    /// woke the frame for this boundary's layer. See this module's doc for why
    /// both are required rather than either alone.
    ///
    /// Returns `None` for a boundary that was never declared.
    pub fn resolve(
        &mut self,
        boundary: BoundaryId,
        reason: Reason,
        content_dirty: bool,
        primitive_count: usize,
        transform_moved: bool,
    ) -> Option<BoundaryComposite> {
        let state = self.boundaries.get_mut(&boundary)?;
        state.primitive_count = primitive_count;
        state.retention = state.policy.retention_for(primitive_count);

        let composite = if content_dirty {
            Composite::Redisplay
        } else if !transform_moved {
            Composite::Clean
        } else if reason.permits_transform_only() {
            Composite::TransformOnly
        } else {
            // The transform moved but nothing said this was a scroll. The
            // conservative answer is to treat the boundary as ordinary content
            // for this frame; the walk has already folded the displacement into
            // the emitted positions, so there is nothing left to slide.
            Composite::Redisplay
        };

        Some(BoundaryComposite {
            boundary,
            layer: state.layer,
            composite,
            retention: state.retention,
            transform: state.transform,
            invalidation: composite.invalidation(),
        })
    }

    /// A boundary's retained state.
    pub fn state(&self, boundary: BoundaryId) -> Option<&BoundaryState> {
        self.boundaries.get(&boundary)
    }

    /// Where a boundary's content currently composites, or the identity for
    /// one that does not exist.
    pub fn transform(&self, boundary: BoundaryId) -> LayerTransform {
        self.boundaries
            .get(&boundary)
            .map(|state| state.transform)
            .unwrap_or(LayerTransform::IDENTITY)
    }

    /// Drop the state of every boundary unvisited for longer than its own
    /// `evict_after_frames`, returning how many were dropped.
    ///
    /// R-N §3.4's mark-and-sweep, with its deliberate delay: a panel scrolled
    /// out of the tree and back within the interval re-materialises at the
    /// scroll position it had, rather than snapping to the top. What Phase 2
    /// retains over that interval is this record only — a boundary's *records*
    /// leave residency as soon as it leaves the tree, because pooling their
    /// storage is the texture-pool work §8 puts in Phase 4.
    pub fn sweep(&mut self, frame: u64) -> usize {
        let before = self.boundaries.len();
        self.boundaries.retain(|_, state| {
            let elapsed = frame.saturating_sub(state.last_visited_frame);
            elapsed <= u64::from(state.policy.evict_after_frames)
        });
        before - self.boundaries.len()
    }

    /// How many boundaries are live.
    pub fn len(&self) -> usize {
        self.boundaries.len()
    }

    /// Whether no boundary is live.
    pub fn is_empty(&self) -> bool {
        self.boundaries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::policy::Buffering;

    const PANEL: BoundaryId = BoundaryId::from_raw(7);

    fn compositor_with_panel() -> Compositor {
        let mut compositor = Compositor::new();
        compositor.visit(PANEL, BoundaryPolicy::default(), 1);
        compositor
    }

    #[test]
    fn a_scroll_over_clean_content_resolves_to_transform_only() {
        let mut compositor = compositor_with_panel();
        assert!(compositor.set_transform(PANEL, LayerTransform::translated(0.0, -40.0)));
        let composite = compositor.resolve(PANEL, Reason::Scroll, false, 12, true);
        assert_eq!(
            composite.map(|composite| composite.composite),
            Some(Composite::TransformOnly)
        );
        assert_eq!(
            composite.map(|composite| composite.invalidation),
            Some(Invalidation::TRANSFORM)
        );
    }

    #[test]
    fn dirty_content_never_reaches_the_fast_path_however_it_was_signalled() {
        let mut compositor = compositor_with_panel();
        assert!(compositor.set_transform(PANEL, LayerTransform::translated(0.0, -40.0)));
        let composite = compositor.resolve(PANEL, Reason::Scroll, true, 12, true);
        assert_eq!(
            composite.map(|composite| composite.composite),
            Some(Composite::Redisplay)
        );
    }

    #[test]
    fn a_data_change_signal_is_refused_the_fast_path_even_over_clean_content() {
        let mut compositor = compositor_with_panel();
        assert!(compositor.set_transform(PANEL, LayerTransform::translated(0.0, -40.0)));
        let composite = compositor.resolve(PANEL, Reason::DataChanged, false, 12, true);
        assert_eq!(
            composite.map(|composite| composite.composite),
            Some(Composite::Redisplay),
            "the fast path requires the signal as well as the measurement"
        );
    }

    #[test]
    fn an_idle_boundary_is_clean_rather_than_transform_only() {
        let mut compositor = compositor_with_panel();
        let composite = compositor.resolve(PANEL, Reason::Scroll, false, 12, false);
        assert_eq!(
            composite.map(|composite| composite.composite),
            Some(Composite::Clean)
        );
        assert_eq!(
            composite.map(|composite| composite.invalidation),
            Some(Invalidation::empty())
        );
        assert!(
            composite
                .map(|composite| composite.composite.leaves_content_resident())
                .unwrap_or(false)
        );
    }

    #[test]
    fn retention_is_decided_per_boundary_from_its_own_primitive_count() {
        let mut compositor = compositor_with_panel();
        let small = compositor.resolve(PANEL, Reason::Scroll, false, 12, false);
        assert_eq!(
            small.map(|composite| composite.retention),
            Some(Retention::Primitives)
        );
        let large = compositor.resolve(PANEL, Reason::Scroll, false, 4_000, false);
        assert_eq!(
            large.map(|composite| composite.retention),
            Some(Retention::Texture)
        );
        assert_eq!(
            compositor.state(PANEL).map(BoundaryState::primitive_count),
            Some(4_000)
        );
    }

    #[test]
    fn a_boundary_keeps_its_transform_across_frames_under_positional_identity() {
        let mut compositor = compositor_with_panel();
        assert!(compositor.set_transform(PANEL, LayerTransform::translated(0.0, -40.0)));
        // The same identity, re-declared next frame: nothing is reset.
        compositor.visit(PANEL, BoundaryPolicy::default(), 2);
        assert_eq!(
            compositor.transform(PANEL),
            LayerTransform::translated(0.0, -40.0)
        );
        assert!(!compositor.set_transform(PANEL, LayerTransform::translated(0.0, -40.0)));
    }

    #[test]
    fn a_policy_change_does_not_disturb_residency_or_position() {
        let mut compositor = compositor_with_panel();
        assert!(compositor.set_transform(PANEL, LayerTransform::translated(3.0, 5.0)));
        let layer = compositor.state(PANEL).map(BoundaryState::layer);
        compositor.visit(
            PANEL,
            BoundaryPolicy {
                rasterize_above: 4,
                buffering: Buffering::None,
                ..BoundaryPolicy::default()
            },
            2,
        );
        assert_eq!(compositor.state(PANEL).map(BoundaryState::layer), layer);
        assert_eq!(compositor.transform(PANEL), LayerTransform::translated(3.0, 5.0));
        assert_eq!(
            compositor
                .state(PANEL)
                .map(|state| state.policy().rasterize_above),
            Some(4)
        );
    }

    #[test]
    fn an_unvisited_boundary_survives_its_eviction_interval_and_then_does_not() {
        let mut compositor = compositor_with_panel();
        let interval = u64::from(BoundaryPolicy::DEFAULT_EVICT_AFTER_FRAMES);
        assert_eq!(compositor.sweep(1 + interval), 0);
        assert_eq!(compositor.len(), 1);
        assert_eq!(compositor.sweep(2 + interval), 1);
        assert!(compositor.is_empty());
    }

    #[test]
    fn resolving_or_moving_an_undeclared_boundary_is_inert() {
        let mut compositor = Compositor::new();
        assert!(!compositor.set_transform(PANEL, LayerTransform::translated(1.0, 1.0)));
        assert!(
            compositor
                .resolve(PANEL, Reason::Scroll, false, 0, false)
                .is_none()
        );
        assert_eq!(compositor.transform(PANEL), LayerTransform::IDENTITY);
        assert!(compositor.is_empty());
    }
}
