//! `Img` — gets `diff_key` here (§6.2 invariant, Phase 5), closing R-N
//! Phase 7's self-documented gap. See docs/gpu-native-architecture.md §3.4,
//! §6.2.
//!
//! # The legacy blocker, and why it does not apply here
//!
//! `Img` not having a `diff_key` was not an oversight anybody forgot about —
//! the legacy element says so in a twelve-line comment above its `Element`
//! impl (`src/elements/img.rs`, just before `impl Element for Img`), and the
//! reason it gives is a real one:
//!
//! > What `paint` shows for an `Img` depends on `ImgState` (per-element,
//! > `with_optional_element_state`-keyed: `frame_index`, `started_loading`,
//! > `last_frame_time`) and `ImgLayoutState.replacement` (a fallback/loading
//! > `AnyElement` substituted in when `request_layout` finds no data yet) —
//! > neither of which is reachable from `Img::diff_key(&self, _)`.
//!
//! That is a statement about *where the state lived*, not about images. In the
//! legacy element the animation frame and the load phase are discovered during
//! `request_layout`/`paint`, which run strictly after `diff_key` is asked for
//! its answer, so a key over `source`/`style` alone would report "unchanged"
//! across a GIF advancing a frame or a pending load resolving, and paint would
//! replay stale content. Opting out unconditionally was the correct call under
//! that ordering.
//!
//! 2.0 does not have that ordering. An element contributes a
//! [`Description`] — built from a value that already holds its resolved state,
//! the same way [`crate::wgpu_surface::WgpuSurface`] already holds its
//! resolved `surface_id` — and the fingerprint is taken from that value. So
//! [`ImgKey`] carries [`ImgKey::frame_index`] and [`ImgKey::load_state`]
//! directly, and the two transitions the legacy comment names are exactly the
//! two the key reports as changed. The fix is the state being *addressable*,
//! not a cleverer comparison.
//!
//! # What the key deliberately does not hold
//!
//! Not the decoded pixels, and not anything requiring a decode to compute.
//! §6.2's whole point is that the key is cheap enough to take every frame for
//! every element in the tree; hashing an image's texels would cost more than
//! the rebuild it exists to avoid. Source *identity* plus the resolved frame
//! index is a complete answer regardless: two different pixel buffers cannot
//! share one source identity at one frame index without the image cache having
//! substituted content behind the same handle, which it does not do — a
//! reloaded source gets a new [`ImageSourceId`].

use std::any::Any;
use wgpui_core::invalidation::axes::Invalidation;
use wgpui_core::patch::emit::{Emission, EmitContext};
use wgpui_core::patch::primitive::Quad;
use wgpui_core::reconcile::description::Description;
use wgpui_core::reconcile::diff_key::ReconcileKey;
use wgpui_layout::taffy_tree::{Dimension, LayoutSize, LayoutStyle};

/// Identity of the thing an image is loaded from — a path, a URI, an asset
/// handle, an in-memory buffer's registration.
///
/// Opaque on purpose. What reconciliation needs is that two `Img`s showing the
/// same resource compare equal and two showing different resources do not; how
/// a source is named is the image cache's business (`image_cache.rs`), not the
/// fingerprint's. A source that is reloaded — re-fetched, re-decoded, replaced
/// on disk — is issued a new id rather than mutating in place, which is what
/// makes comparing identity rather than content sound.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ImageSourceId(u64);

impl ImageSourceId {
    /// Wrap a raw source handle.
    pub const fn from_raw(raw: u64) -> Self {
        ImageSourceId(raw)
    }

    /// The raw source handle.
    pub const fn as_raw(self) -> u64 {
        self.0
    }
}

/// Which of an image's three possible renderings is actually on screen.
///
/// The legacy element expresses this as `ImgLayoutState.replacement`: an
/// `AnyElement` standing in for the image while it loads or after it fails.
/// Swapping that in or out changes both what is painted and what the layout
/// tree contains, so it is named here rather than left implicit.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum ImageLoadState {
    /// Decoded and available; the image itself is what paints.
    #[default]
    Ready,
    /// Still loading; a placeholder subtree paints instead.
    Loading,
    /// Loading failed; a fallback subtree paints instead.
    Failed,
}

/// How an image's own aspect ratio is reconciled with the box it was given.
///
/// Mirrors the legacy `ObjectFit` (`src/elements/img.rs`). It affects only
/// where inside an already-decided rectangle the content lands, so it is a
/// `DISPLAY`-axis property, never a `LAYOUT` one.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum ObjectFit {
    /// Scale to fill, ignoring aspect ratio.
    Fill,
    /// Scale to fit entirely inside, preserving aspect ratio.
    #[default]
    Contain,
    /// Scale to cover entirely, preserving aspect ratio, cropping the excess.
    Cover,
    /// Draw at natural size.
    None,
    /// `None`, unless that overflows, in which case `Contain`.
    ScaleDown,
}

/// The display-affecting styling of an image.
///
/// Deliberately not the legacy `ImageStyle` verbatim: that type also holds
/// `loading` and `fallback`, which are `Box<dyn Fn() -> AnyElement>` closures.
/// Closures are not comparable and, per R-N §2.4, are never compared —
/// [`ImageLoadState`] carries the part of them that is observable (which of the
/// three renderings is active), and the closures themselves are swapped in
/// unconditionally like any other listener.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct ImageStyle {
    /// Draw desaturated.
    pub grayscale: bool,
    /// How the content is fitted into its box.
    pub object_fit: ObjectFit,
    /// Straight alpha the image composites at.
    pub opacity: f32,
    /// Uniform corner radius the image is clipped to.
    pub corner_radius: f32,
}

/// The fingerprint an `Img` presents to ambient reconciliation.
///
/// Five fields, each of which changes what a viewer sees without any of the
/// others changing — which is the test for whether a field belongs in a key at
/// all. Everything else about an image is either derived from these (the
/// decoded texels, from the source and the frame) or invisible (the cache
/// handle it was fetched through).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ImgKey {
    /// Which resource is displayed.
    pub source: ImageSourceId,
    /// Which frame of an animated source is displayed. `0` for still images,
    /// which is why a still image's key is stable across frames for free.
    pub frame_index: u32,
    /// Whether the image, a loading placeholder, or a failure fallback is what
    /// actually paints.
    pub load_state: ImageLoadState,
    /// The box the image asked layout for.
    pub requested_size: [f32; 2],
    /// How that box is drawn.
    pub style: ImageStyle,
}

impl ReconcileKey for ImgKey {
    fn compare(&self, previous: &dyn ReconcileKey) -> Invalidation {
        let Some(previous) = previous.as_any().downcast_ref::<ImgKey>() else {
            return Invalidation::all();
        };
        let mut axes = Invalidation::empty();
        if previous.requested_size != self.requested_size {
            // Moves the Taffy leaf and repaints, exactly like a resized
            // `WgpuSurface`.
            axes |= Invalidation::LAYOUT;
            axes |= Invalidation::DISPLAY;
        }
        if previous.load_state != self.load_state {
            // A load transition swaps a whole subtree in or out (the legacy
            // `ImgLayoutState.replacement`), so it is a layout change and not
            // only a repaint — the one case where being conservative is not
            // merely defensible but required.
            axes |= Invalidation::LAYOUT;
            axes |= Invalidation::DISPLAY;
        }
        if previous.source != self.source
            || previous.frame_index != self.frame_index
            || previous.style != self.style
        {
            axes |= Invalidation::DISPLAY;
        }
        axes
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// An image element's description shape.
///
/// Like [`crate::wgpu_surface::WgpuSurface`], this is the shape an image
/// presents to reconciliation and emission, not the full element: there is no
/// decoder, no `ImageCache`, and no `AnyElement` replacement subtree here,
/// because those need `App`/`Window`, which §3.4 puts elsewhere. What is real
/// is the fingerprint and the fact that it is complete.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Img {
    source: ImageSourceId,
    frame_index: u32,
    load_state: ImageLoadState,
    requested_size: [f32; 2],
    style: ImageStyle,
}

/// An image showing `source`, at its first frame, ready, unsized, unstyled.
pub fn img(source: ImageSourceId) -> Img {
    Img::new(source)
}

impl Img {
    /// An image showing `source`, at its first frame, ready, unsized, unstyled.
    pub fn new(source: ImageSourceId) -> Self {
        Self {
            source,
            frame_index: 0,
            load_state: ImageLoadState::Ready,
            requested_size: [0.0, 0.0],
            style: ImageStyle {
                grayscale: false,
                object_fit: ObjectFit::Contain,
                opacity: 1.0,
                corner_radius: 0.0,
            },
        }
    }

    /// Select the frame of an animated source that is currently displayed.
    pub fn frame_index(mut self, frame_index: u32) -> Self {
        self.frame_index = frame_index;
        self
    }

    /// Record which of the three renderings is active this frame.
    pub fn load_state(mut self, load_state: ImageLoadState) -> Self {
        self.load_state = load_state;
        self
    }

    /// Request a size.
    ///
    /// Named *requested* for the same reason [`crate::wgpu_surface::WgpuSurface::bounds`]
    /// is: resolved bounds arrive from layout, after the description exists. An
    /// image resolved somewhere new without its own request changing is still
    /// handled, by `patch::emit`'s own "did this element move" rule.
    pub fn size(mut self, width: f32, height: f32) -> Self {
        self.requested_size = [width, height];
        self
    }

    /// Set how the image is drawn.
    pub fn style(mut self, style: ImageStyle) -> Self {
        self.style = style;
        self
    }

    /// This image's fingerprint.
    pub fn diff_key(&self) -> ImgKey {
        ImgKey {
            source: self.source,
            frame_index: self.frame_index,
            load_state: self.load_state,
            requested_size: self.requested_size,
            style: self.style,
        }
    }

    /// The per-frame description of this image.
    pub fn describe(&self) -> Description {
        let [width, height] = self.requested_size;
        let opacity = self.style.opacity;
        let corner_radius = self.style.corner_radius;
        Description::new::<Img>()
            .diff_key(self.diff_key())
            .style(LayoutStyle {
                size: LayoutSize {
                    width: Dimension::length(width),
                    height: Dimension::length(height),
                },
                flex_shrink: 0.0,
                ..LayoutStyle::default()
            })
            .emit(move |context: &EmitContext, emission: &mut Emission| {
                // One quad standing in for the polychrome sprite. 2.0 has two
                // primitive kinds (`Quad`, `GlyphRun`) and no sprite kind yet;
                // adding one is the three-line change `patch::primitive`'s own
                // doc describes, and is not what Phase 5 was scoped to do.
                emission.quad(Quad {
                    origin: [context.bounds.x, context.bounds.y],
                    size: [context.bounds.width, context.bounds.height],
                    background: [1.0, 1.0, 1.0, opacity],
                    corner_radius,
                    ..Quad::ZERO
                });
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpui_core::reconcile::description::ElementId;
    use wgpui_core::reconcile::diff_key::compare_by_equality;
    use wgpui_core::reconcile::instance::InstanceKey;
    use wgpui_core::reconcile::plan::{FramePlan, NodeOutcome, PlannedNode, RebuildReason};
    use wgpui_core::reconcile::reconciler::{ReconcileError, Reconciler};
    use wgpui_layout::taffy_tree::{FlexDirection, LayoutTree};

    struct Panel;

    #[derive(PartialEq, Debug)]
    struct PanelKey(u32);

    impl ReconcileKey for PanelKey {
        fn compare(&self, previous: &dyn ReconcileKey) -> Invalidation {
            compare_by_equality(self, previous, Invalidation::DISPLAY)
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    const IMG_SLOT: [ElementId; 2] = [ElementId::Slot(0), ElementId::Slot(0)];
    const PANEL_SLOT: [ElementId; 2] = [ElementId::Slot(0), ElementId::Slot(1)];

    fn base() -> Img {
        Img::new(ImageSourceId::from_raw(1)).size(48.0, 48.0)
    }

    /// An avatar beside a plain reconciled sibling — SFD §3's own list-row
    /// shape, minus the text half, which `styled_text.rs` covers.
    fn tree(image: Img) -> Description {
        Description::new::<Panel>()
            .diff_key(PanelKey(0))
            .style(LayoutStyle {
                flex_direction: FlexDirection::Column,
                ..LayoutStyle::default()
            })
            .child(image.describe())
            .child(
                Description::new::<Panel>()
                    .diff_key(PanelKey(0))
                    .style(LayoutStyle {
                        size: LayoutSize {
                            width: Dimension::length(120.0),
                            height: Dimension::length(40.0),
                        },
                        flex_shrink: 0.0,
                        ..LayoutStyle::default()
                    }),
            )
    }

    fn node_at<'plan>(plan: &'plan FramePlan, path: &[ElementId]) -> Option<&'plan PlannedNode> {
        plan.node_for_instance(InstanceKey::from_path(path))
    }

    #[test]
    fn an_unchanged_image_is_reused_like_any_other_element() -> Result<(), ReconcileError> {
        let mut reconciler = Reconciler::new();
        let mut layout = LayoutTree::new();

        let first = reconciler.reconcile(tree(base()), &mut layout)?;
        let before = node_at(&first, &IMG_SLOT).copied();
        assert_eq!(
            before.map(|node| node.outcome),
            Some(NodeOutcome::Rebuilt(RebuildReason::NewInstance))
        );

        let second = reconciler.reconcile(tree(base()), &mut layout)?;
        let after = node_at(&second, &IMG_SLOT).copied();
        assert_eq!(
            after.map(|node| node.outcome),
            Some(NodeOutcome::Reused),
            "an unchanged image must not rebuild — this is the gap §6.2 names"
        );
        assert_eq!(
            after.map(|node| node.skipped_prepaint_and_paint()),
            node_at(&second, &PANEL_SLOT)
                .copied()
                .map(|node| node.skipped_prepaint_and_paint()),
            "the image and the ordinary element must get the same treatment"
        );
        assert_eq!(
            after.map(|node| node.layout_node),
            before.map(|node| node.layout_node)
        );
        assert!(second.fully_reused());
        Ok(())
    }

    /// The two transitions the legacy comment names as the reason `Img` had no
    /// key at all — a GIF advancing a frame, and a pending load resolving —
    /// are the two this key exists to report.
    #[test]
    fn the_state_the_legacy_key_could_not_reach_is_exactly_what_this_key_reports() {
        let still = base();
        let next_frame = base().frame_index(1);
        let loading = base().load_state(ImageLoadState::Loading);

        assert_eq!(
            still.diff_key().compare(&still.diff_key()),
            Invalidation::empty()
        );
        assert_eq!(
            next_frame.diff_key().compare(&still.diff_key()),
            Invalidation::DISPLAY,
            "an animated source advancing a frame must repaint"
        );
        assert_eq!(
            loading.diff_key().compare(&still.diff_key()),
            Invalidation::LAYOUT.union(Invalidation::DISPLAY),
            "a load transition swaps a replacement subtree, so it is a layout change too"
        );
    }

    #[test]
    fn each_field_reports_exactly_the_axes_it_affects() -> Result<(), ReconcileError> {
        let cases: [(&str, Img, Invalidation); 5] = [
            (
                "a different source",
                Img::new(ImageSourceId::from_raw(2)).size(48.0, 48.0),
                Invalidation::DISPLAY,
            ),
            ("a new animation frame", base().frame_index(3), Invalidation::DISPLAY),
            (
                "a load transition",
                base().load_state(ImageLoadState::Failed),
                Invalidation::LAYOUT.union(Invalidation::DISPLAY),
            ),
            (
                "a resize",
                base().size(48.0, 49.0),
                Invalidation::LAYOUT.union(Invalidation::DISPLAY),
            ),
            (
                "a style change",
                base().style(ImageStyle {
                    grayscale: true,
                    object_fit: ObjectFit::Contain,
                    opacity: 1.0,
                    corner_radius: 0.0,
                }),
                Invalidation::DISPLAY,
            ),
        ];

        for (what, changed, expected) in cases {
            assert_ne!(
                changed.diff_key(),
                base().diff_key(),
                "{what} must actually differ from the control"
            );

            let mut reconciler = Reconciler::new();
            let mut layout = LayoutTree::new();
            reconciler.reconcile(tree(base()), &mut layout)?;
            let plan = reconciler.reconcile(tree(changed), &mut layout)?;

            let image = node_at(&plan, &IMG_SLOT).copied();
            assert_eq!(
                image.map(|node| node.outcome),
                Some(NodeOutcome::Rebuilt(RebuildReason::KeyChanged)),
                "{what} must rebuild the image"
            );
            assert_eq!(
                image.map(|node| node.invalidation),
                Some(expected),
                "{what} must report exactly the axes it affects"
            );
            assert_eq!(
                node_at(&plan, &PANEL_SLOT)
                    .copied()
                    .map(|node| node.skipped_prepaint_and_paint()),
                Some(true),
                "{what} must not disturb the sibling"
            );
        }
        Ok(())
    }

    #[test]
    fn object_fit_is_a_display_change_and_never_a_layout_one() {
        let contained = base();
        let covered = base().style(ImageStyle {
            object_fit: ObjectFit::Cover,
            ..contained.style
        });
        assert_eq!(
            covered.diff_key().compare(&contained.diff_key()),
            Invalidation::DISPLAY,
            "object-fit decides where content sits inside a box already decided by layout"
        );
    }

    #[test]
    fn a_key_compared_against_a_different_element_type_is_a_full_invalidation() {
        assert_eq!(base().diff_key().compare(&PanelKey(0)), Invalidation::all());
    }
}
