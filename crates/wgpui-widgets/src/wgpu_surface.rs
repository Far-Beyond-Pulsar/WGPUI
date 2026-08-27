//! Real identity + trivial `diff_key` (§5.5, Gap 1) — today's
//! `src/elements/wgpu_surface.rs` (545 lines), whose `id()` is hardcoded to
//! `None`; this is where that gets fixed. See
//! docs/gpu-native-architecture.md §3.4, §5.5.
//!
//! # Scope: this is the description shape, not the element
//!
//! Stated plainly up front, because the distinction matters more here than
//! anywhere else in Phase 2. What this module contains is the *shape* a
//! `WgpuSurface` presents to reconciliation: a positional identity, no
//! children, and a fingerprint over exactly `(bounds, style, surface_id)`. What
//! it does **not** contain is the element: no `WgpuSurfaceHandle`, no
//! `SurfaceRegistry`, no triple buffer, no external render thread, no texture,
//! no `wgpu` dependency of any kind. Building those means wiring
//! `wgpui-widgets` to `wgpui-wgpu`, which §8 places in Phase 4 alongside Gap
//! 2's compositing unification.
//!
//! So [`SurfaceId`] here is a plain opaque handle standing in for the real
//! one, and [`SurfaceStyle`] carries two representative visual properties
//! standing in for the frozen `Style` (§7). Both are placeholders, marked as
//! such, and neither is a design decision a later phase has to live with.
//!
//! # What is genuinely proved, and why it is worth proving now
//!
//! §5.5's Gap 1 is not "this element lacks a fingerprint." It is that
//! `WgpuSurface::id()` returns `None`, so the element "can never be addressed
//! by `InstanceKey`/`GlobalElementId` and so never participates in
//! reconciliation, Taffy-node reuse, or `.boundary()` at all" — there is no
//! identity to hang a `diff_key` on. Under R-N/SFD's model that was a real
//! blocker, because identity came only from an explicit `.id()`.
//!
//! Under 2.0 it is not a blocker and this module is the demonstration:
//! [`crate::wgpu_surface::WgpuSurface::describe`] calls no `.id()` anywhere, and
//! the element is addressed anyway, positionally (SFD §1.0, implemented in
//! `wgpui-core::boundary::identity` and in the reconciler's own walk). The
//! fingerprint is then exactly the trivial one §5.5 argues is "sufficient and
//! correct by construction," because a surface's pixel content is produced by
//! someone else's render loop and is never part of the CPU description at all.
//!
//! The gate below drives a real reconciler over a tree holding one of these
//! beside an ordinary reconciled element, and asserts the two get the same
//! treatment — which is §8's Phase 2 wording ("skips
//! `request_layout`/`prepaint`/`paint` across frames exactly like a reconciled
//! `div` would") measured against a live control rather than an asserted
//! constant.

use wgpui_core::invalidation::axes::Invalidation;
use wgpui_core::patch::emit::{Emission, EmitContext};
use wgpui_core::patch::primitive::Quad;
use wgpui_core::reconcile::description::Description;
use wgpui_core::reconcile::diff_key::ReconcileKey;
use wgpui_layout::taffy_tree::{Dimension, LayoutRect, LayoutSize, LayoutStyle};
use std::any::Any;

/// A handle to an externally-produced surface.
///
/// **Placeholder.** The real handle is `WgpuSurfaceHandle`, which owns a
/// triple-buffered texture and a cross-thread producer protocol
/// (`surface_registry.rs`, 772 lines) that §9's risk table forbids this work
/// from touching. All reconciliation needs from it is an identity that is equal
/// to itself and unequal to a different surface, which is what this is.
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
/// **Placeholder.** §7 freezes the real `Style`, which lives in the legacy
/// crate; these two fields stand in for it because they are enough to exercise
/// the one thing the fingerprint has to get right — that a style change is a
/// `DISPLAY` change and not a `LAYOUT` one.
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
/// See this module's doc: this is the description shape, not the element.
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
        let opacity = self.style.opacity;
        let corner_radius = self.style.corner_radius;
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
            .emit(move |context: &EmitContext, emission: &mut Emission| {
                // One composite entry, standing in for the surface draw. §5.5's
                // Gap 2 replaces this with the unified indirect-draw entry that
                // `.boundary()`'s texture-retained layers use, in Phase 4.
                emission.quad(Quad {
                    origin: [context.bounds.x, context.bounds.y],
                    size: [context.bounds.width, context.bounds.height],
                    background: [0.0, 0.0, 0.0, opacity],
                    corner_radius,
                    ..Quad::ZERO
                });
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    /// The `paint` half of §8's Phase 2 wording, which in this workspace is
    /// emission: an unchanged surface is never asked to produce its composite
    /// entry again, and nothing is uploaded on its account.
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
         -> Result<(usize, u64), Box<dyn std::error::Error>> {
            let plan = reconciler.reconcile(description, layout)?;
            let root = plan.nodes().first().map(|node| node.layout_node);
            if let Some(root) = root {
                layout.compute_layout(root, definite(640.0, 520.0))?;
            }
            let emission = emitter.emit(&plan, layout, &signals, scene)?;
            let uploads = apply(scene, &emission.patch)?;
            Ok((emission.stats.nodes_emitted, uploads.byte_count()))
        };

        let (built, first_bytes) = draw(
            tree(base()),
            &mut reconciler,
            &mut layout,
            &mut emitter,
            &mut scene,
        )?;
        assert_eq!(built, 1, "only the surface emits anything in this tree");
        assert!(first_bytes > 0);

        let (again, bytes) = draw(
            tree(base()),
            &mut reconciler,
            &mut layout,
            &mut emitter,
            &mut scene,
        )?;
        assert_eq!(again, 0, "an unchanged surface costs nothing per frame");
        assert_eq!(bytes, 0);
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
