//! A retained leaf describing a texture produced by a WGPU surface renderer.
//! The producer handle lives in `wgpui-wgpu`; this crate only carries the
//! renderer-independent surface identity, style, and layout description.
//! Reconciliation never fingerprints the surface pixels. The GPU compositor
//! samples the registry's current display buffer and the frame loop damages
//! only the surface's resolved visible rectangle when a new buffer is ready.

use std::any::Any;
use wgpui_core::boundary::compositor::ExternalSurfaceId;
use wgpui_core::element::Element;
use wgpui_core::invalidation::axes::Invalidation;
use wgpui_core::reconcile::description::Description;
use wgpui_core::reconcile::diff_key::ReconcileKey;
use wgpui_layout::taffy_tree::{Dimension, LayoutRect, LayoutSize, LayoutStyle};

/// A handle to an externally-produced surface.
///
/// An opaque identity for a producer-owned surface.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SurfaceId(u64);

impl SurfaceId {
    /// Wrap a raw handle.
    pub const fn from_raw(raw: u64) -> Self {
        SurfaceId(raw)
    }

    /// The raw handle.
    pub const fn as_raw(self) -> u64 {
        self.0
    }
}

/// The visual properties of a surface's own composite entry.
///
/// The visual properties that affect this surface's composite entry.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct SurfaceStyle {
    /// Uniform corner radius the surface is clipped to.
    pub corner_radius: f32,
    /// Straight alpha the surface composites at.
    pub opacity: f32,
}

/// The fingerprint a `WgpuSurface` presents to ambient reconciliation.
///
/// §5.5: "A `diff_key` comparing only `(bounds, style, surface_id)` is
/// sufficient and correct by construction, because those are the only three
/// things that affect *its own* composite entry (Taffy leaf, order-tree
/// position, indirect-draw slot)." Nothing about the surface's pixels appears
/// here, deliberately and permanently: the compositor always samples whatever
/// the producer currently has ready, so the framework never has to ask whether
/// the texture changed the way it asks that of a `div`'s children.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct WgpuSurfaceKey {
    /// The rectangle the surface's own style asks for.
    pub bounds: LayoutRect,
    /// How its composite entry is drawn.
    pub style: SurfaceStyle,
    /// Which externally-produced surface it samples.
    pub surface_id: SurfaceId,
}

impl ReconcileKey for WgpuSurfaceKey {
    fn compare(&self, previous: &dyn ReconcileKey) -> Invalidation {
        let Some(previous) = previous.as_any().downcast_ref::<WgpuSurfaceKey>() else {
            return Invalidation::all();
        };
        let mut axes = Invalidation::empty();
        if previous.bounds != self.bounds {
            // A resize moves the Taffy leaf and the composite entry both, so it
            // is not the `DISPLAY`-only change a recolour is.
            axes |= Invalidation::LAYOUT;
            axes |= Invalidation::DISPLAY;
        }
        if previous.style != self.style || previous.surface_id != self.surface_id {
            axes |= Invalidation::DISPLAY;
        }
        axes
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// An externally-rendered surface composited into the scene.
///
/// The retained description of an externally-produced surface.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct WgpuSurface {
    surface_id: SurfaceId,
    requested_bounds: LayoutRect,
    style: SurfaceStyle,
}

impl WgpuSurface {
    /// A surface sampling `surface_id`, requesting no particular size.
    pub fn new(surface_id: SurfaceId) -> Self {
        Self {
            surface_id,
            requested_bounds: LayoutRect {
                x: 0.0,
                y: 0.0,
                width: 0.0,
                height: 0.0,
            },
            style: SurfaceStyle::default(),
        }
    }

    /// Request a size and position.
    ///
    /// Named *requested* rather than resolved because that is what a
    /// description can honestly know: resolved bounds arrive from layout, after
    /// the description is built. A surface that is resolved somewhere new
    /// without its own request changing is still handled — `patch::emit`'s own
    /// "did this element move" rule covers it, for this element exactly as for
    /// every other.
    pub fn bounds(mut self, bounds: LayoutRect) -> Self {
        self.requested_bounds = bounds;
        self
    }

    /// Set how the composite entry is drawn.
    pub fn style(mut self, style: SurfaceStyle) -> Self {
        self.style = style;
        self
    }

    /// This surface's fingerprint: exactly `(bounds, style, surface_id)`.
    pub fn diff_key(&self) -> WgpuSurfaceKey {
        WgpuSurfaceKey {
            bounds: self.requested_bounds,
            style: self.style,
            surface_id: self.surface_id,
        }
    }

    /// The per-frame description of this surface.
    ///
    /// Note the two absences, both load-bearing: no `.id()` — identity is
    /// positional (SFD §1.0), which is what closes §5.5's Gap 1 — and no
    /// children, because a surface's content is produced outside the framework
    /// entirely (§5.5).
    pub fn describe(&self) -> Description {
        let bounds = self.requested_bounds;
        Description::new::<WgpuSurface>()
            .diff_key(self.diff_key())
            .style(LayoutStyle {
                size: LayoutSize {
                    width: Dimension::length(bounds.width),
                    height: Dimension::length(bounds.height),
                },
                flex_shrink: 0.0,
                ..LayoutStyle::default()
            })
            .external_surface(
                ExternalSurfaceId::from_raw(self.surface_id.as_raw()),
                self.style.opacity,
                self.style.corner_radius,
            )
    }
}

impl Element for WgpuSurface {
    fn into_description(self) -> Description {
        self.describe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpui_core::boundary::compositor::{CompositeSource, ExternalSurfaceId};
    use wgpui_core::invalidation::request::FrameSignals;
    use wgpui_core::patch::apply::apply;
    use wgpui_core::patch::emit::Emitter;
    use wgpui_core::reconcile::description::ElementId;
    use wgpui_core::reconcile::diff_key::compare_by_equality;
    use wgpui_core::reconcile::instance::InstanceKey;
    use wgpui_core::reconcile::plan::{FramePlan, NodeOutcome, PlannedNode, RebuildReason};
    use wgpui_core::reconcile::reconciler::{ReconcileError, Reconciler};
    use wgpui_core::scene::Scene;
    use wgpui_layout::taffy_tree::{FlexDirection, LayoutTree, definite};

    /// The ordinary reconciled element the surface is measured against.
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

    const SURFACE_SLOT: [ElementId; 2] = [ElementId::Slot(0), ElementId::Slot(0)];
    const PANEL_SLOT: [ElementId; 2] = [ElementId::Slot(0), ElementId::Slot(1)];

    fn viewport(width: f32, height: f32) -> LayoutRect {
        LayoutRect {
            x: 0.0,
            y: 0.0,
            width,
            height,
        }
    }

    /// A surface and a plain reconciled element as siblings, neither named.
    fn tree(surface: WgpuSurface) -> Description {
        Description::new::<Panel>()
            .diff_key(PanelKey(0))
            .style(LayoutStyle {
                flex_direction: FlexDirection::Column,
                ..LayoutStyle::default()
            })
            .child(surface.describe())
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

    fn skipped(node: Option<PlannedNode>) -> Option<bool> {
        node.map(|node| node.skipped_prepaint_and_paint())
    }

    fn base() -> WgpuSurface {
        WgpuSurface::new(SurfaceId::from_raw(1))
            .bounds(viewport(640.0, 480.0))
            .style(SurfaceStyle {
                corner_radius: 0.0,
                opacity: 1.0,
            })
    }

    /// **Phase 2 gate #3** (§5.5 Gap 1, §8): a `WgpuSurface`-shaped element
    /// with real (positional) identity and a `diff_key` over exactly
    /// `(bounds, style, surface_id)` skips reconciliation work across frames
    /// when none of the three changes — exactly like a reconciled `div` would.
    ///
    /// The plain sibling is the control: every assertion about the surface is
    /// made about it too, so "exactly like" is measured rather than asserted.
    #[test]
    fn gate_3_an_unmoved_unresized_surface_skips_work_like_a_reconciled_element_would()
    -> Result<(), ReconcileError> {
        let mut reconciler = Reconciler::new();
        let mut layout = LayoutTree::new();

        let description = tree(base());
        // §5.5's Gap 1 is that `id()` returns `None`. It still does — and that
        // no longer costs the element its identity, which is the whole point.
        let surface_description = description
            .child_descriptions()
            .first()
            .map(Description::element_id);
        assert_eq!(surface_description, Some(None));

        let first = reconciler.reconcile(description, &mut layout)?;
        let surface_before = node_at(&first, &SURFACE_SLOT).copied();
        let panel_before = node_at(&first, &PANEL_SLOT).copied();
        assert_eq!(
            surface_before.map(|node| node.outcome),
            Some(NodeOutcome::Rebuilt(RebuildReason::NewInstance))
        );
        assert_eq!(
            reconciler.instances().len(),
            3,
            "the surface is a retained instance like any other element"
        );

        let second = reconciler.reconcile(tree(base()), &mut layout)?;
        let surface_after = node_at(&second, &SURFACE_SLOT).copied();
        let panel_after = node_at(&second, &PANEL_SLOT).copied();

        assert_eq!(
            surface_after.map(|node| node.outcome),
            Some(NodeOutcome::Reused),
            "an unmoved, unresized surface must not rebuild"
        );
        assert_eq!(
            skipped(surface_after),
            skipped(panel_after),
            "the surface and the ordinary element must get the same treatment"
        );
        assert_eq!(skipped(surface_after), Some(true));
        assert_eq!(
            surface_after.map(|node| node.layout_node),
            surface_before.map(|node| node.layout_node),
            "§5.5: it must stop re-registering a Taffy leaf unconditionally, forever"
        );
        assert_eq!(
            panel_after.map(|node| node.layout_node),
            panel_before.map(|node| node.layout_node)
        );
        assert!(second.fully_reused());
        assert_eq!(second.stats().layout_nodes_created, 0);
        assert_eq!(second.stats().layout_nodes_reused, 3);
        assert_eq!(second.stats().instances_swept, 0);
        Ok(())
    }

    #[test]
    fn each_of_the_three_fields_is_the_only_thing_that_rebuilds_the_surface()
    -> Result<(), ReconcileError> {
        // Each case changes exactly one of the fingerprint's three fields away
        // from `base()` and leaves the other two alone, so the reported axes
        // are attributable to that field and to nothing else.
        let cases: [(&str, WgpuSurface, Invalidation); 3] = [
            (
                "a different surface handle",
                WgpuSurface::new(SurfaceId::from_raw(2))
                    .bounds(viewport(640.0, 480.0))
                    .style(SurfaceStyle {
                        corner_radius: 0.0,
                        opacity: 1.0,
                    }),
                Invalidation::DISPLAY,
            ),
            (
                "a resize",
                base().bounds(viewport(640.0, 481.0)),
                Invalidation::LAYOUT.union(Invalidation::DISPLAY),
            ),
            (
                "a style change",
                base().style(SurfaceStyle {
                    corner_radius: 4.0,
                    opacity: 1.0,
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

            let surface = node_at(&plan, &SURFACE_SLOT).copied();
            assert_eq!(
                surface.map(|node| node.outcome),
                Some(NodeOutcome::Rebuilt(RebuildReason::KeyChanged)),
                "{what} must rebuild the surface"
            );
            assert_eq!(
                surface.map(|node| node.invalidation),
                Some(expected),
                "{what} must report exactly the axes it affects"
            );
            assert_eq!(
                skipped(node_at(&plan, &PANEL_SLOT).copied()),
                Some(true),
                "{what} must not disturb the sibling"
            );
        }
        Ok(())
    }

    /// The `paint` half of §8's Phase 2 wording: an unchanged surface does not
    /// emit scene primitives or upload anything. Its composite entry is
    /// retained separately for the GPU texture consumer.
    #[test]
    fn an_unchanged_surface_is_never_asked_to_emit_again() -> Result<(), Box<dyn std::error::Error>>
    {
        let mut reconciler = Reconciler::new();
        let mut layout = LayoutTree::new();
        let mut emitter = Emitter::new();
        let mut scene = Scene::new();
        let signals = FrameSignals::new();

        let draw = |description: Description,
                    reconciler: &mut Reconciler,
                    layout: &mut LayoutTree,
                    emitter: &mut Emitter,
                    scene: &mut Scene|
         -> Result<(usize, u64, usize, Option<wgpui_core::boundary::compositor::CompositeEntry>), Box<dyn std::error::Error>> {
            let plan = reconciler.reconcile(description, layout)?;
            let root = plan.nodes().first().map(|node| node.layout_node);
            if let Some(root) = root {
                layout.compute_layout(root, definite(640.0, 520.0))?;
            }
            let emission = emitter.emit(&plan, layout, &signals, scene)?;
            let uploads = apply(scene, &emission.patch)?;
            Ok((
                emission.stats.nodes_emitted,
                uploads.byte_count(),
                emission.external_surfaces.len(),
                emission.external_surfaces.first().copied(),
            ))
        };

        let (built, first_bytes, first_surfaces, first_surface) = draw(
            tree(base()),
            &mut reconciler,
            &mut layout,
            &mut emitter,
            &mut scene,
        )?;
        assert_eq!(built, 0, "the external surface does not emit scene primitives");
        assert_eq!(first_bytes, 0);
        assert_eq!(first_surfaces, 1);
        assert_eq!(
            first_surface.map(|entry| entry.source),
            Some(CompositeSource::External(ExternalSurfaceId::from_raw(1)))
        );

        let (again, bytes, again_surfaces, again_surface) = draw(
            tree(base()),
            &mut reconciler,
            &mut layout,
            &mut emitter,
            &mut scene,
        )?;
        assert_eq!(again, 0, "an unchanged surface costs nothing per frame");
        assert_eq!(bytes, 0);
        assert_eq!(again_surfaces, 1);
        assert_eq!(again_surface, first_surface);
        Ok(())
    }

    #[test]
    fn a_surface_keeps_its_identity_across_frames_without_ever_being_named() {
        let first = InstanceKey::from_path(&SURFACE_SLOT);
        let second = InstanceKey::from_path(&SURFACE_SLOT);
        assert_eq!(first, second);
        assert_ne!(first, InstanceKey::from_path(&PANEL_SLOT));
    }

    #[test]
    fn a_key_compared_against_a_different_element_type_is_a_full_invalidation() {
        assert_eq!(
            base().diff_key().compare(&PanelKey(0)),
            Invalidation::all(),
            "an address that held a different element type must never be reused"
        );
    }

    #[test]
    fn an_identical_key_reports_nothing_stale() {
        assert_eq!(
            base().diff_key().compare(&base().diff_key()),
            Invalidation::empty()
        );
    }
}
