//! Explicit, capture-only retained element snapshots.
//!
//! The live frame path does not retain an inspector mirror. A caller that has
//! armed a capture can freeze the plan, shared walk, and damage plan into this
//! immutable value at a safe frame boundary.

use crate::damage::DamagePlan;
use crate::invalidation::Invalidation;
use crate::reconcile::instance::InstanceKey;
use crate::reconcile::plan::{FramePlan, NodeOutcome, RebuildReason};
use crate::reconcile::walk::{RetainedWalk, TileOwnership};
use crate::scene::BoundaryId;
use wgpui_layout::taffy_tree::{LayoutNodeId, LayoutRect};

/// Version of the in-memory retained element snapshot contract.
pub const RETAINED_FRAME_SNAPSHOT_VERSION: u16 = 1;

/// Capture-time metadata for one retained element.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RetainedElementSnapshot {
    /// Stable address used by reconciliation and primitive records.
    pub address: InstanceKey,
    /// Retained layout node backing this element.
    pub layout_node: LayoutNodeId,
    /// Pre-order depth.
    pub depth: u32,
    /// Resolved absolute bounds.
    pub bounds: LayoutRect,
    /// Inherited effective clip.
    pub effective_clip: Option<LayoutRect>,
    /// Boundary/root owning the element's paint.
    pub owning_root: BoundaryId,
    /// Tile or overlay owner in that root.
    pub tile_ownership: TileOwnership,
    /// Reconciliation outcome.
    pub outcome: NodeOutcome,
    /// Exact invalidation axes reported by reconciliation.
    pub invalidation: Invalidation,
    /// Whether layout was skipped by reuse.
    pub layout_skipped: bool,
    /// Whether paint was skipped by reuse.
    pub paint_skipped: bool,
    /// The precise rebuild reason, if any.
    pub rebuild_reason: Option<RebuildReason>,
}

/// An immutable retained-frame capture assembled explicitly by a caller.
#[derive(Clone, Debug, PartialEq)]
pub struct RetainedFrameSnapshot {
    /// Schema version for consumers that persist or transport captures.
    pub schema_version: u16,
    /// Reconciler frame number represented by this capture.
    pub frame: u64,
    /// Elements in the shared walk's pre-order.
    pub elements: Vec<RetainedElementSnapshot>,
    /// Damage attribution observed/planned for this frame.
    pub damage: DamagePlan,
}

impl RetainedFrameSnapshot {
    /// Freeze metadata from an existing plan and shared walk.
    ///
    /// Returning `None` for a mismatched walk keeps a capture from silently
    /// pairing metadata from different frames or from a malformed producer.
    pub fn capture(
        frame: u64,
        plan: &FramePlan,
        walk: &RetainedWalk,
        damage: DamagePlan,
    ) -> Option<Self> {
        if plan.nodes().len() != walk.nodes().len() {
            return None;
        }
        let elements = plan
            .nodes()
            .iter()
            .zip(walk.nodes())
            .map(|(planned, walked)| {
                if planned.address != walked.address || planned.depth != walked.depth {
                    return None;
                }
                Some(RetainedElementSnapshot {
                    address: planned.address,
                    layout_node: planned.layout_node,
                    depth: planned.depth,
                    bounds: walked.bounds,
                    effective_clip: walked.effective_clip,
                    owning_root: walked.owning_root,
                    tile_ownership: walked.tile_ownership,
                    outcome: walked.outcome,
                    invalidation: walked.invalidation,
                    layout_skipped: walked.layout_skipped,
                    paint_skipped: walked.paint_skipped,
                    rebuild_reason: walked.rebuild_reason(),
                })
            })
            .collect::<Option<Vec<_>>>()?;
        Some(Self {
            schema_version: RETAINED_FRAME_SNAPSHOT_VERSION,
            frame,
            elements,
            damage,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::damage::DamagePlan;
    use crate::invalidation::FrameSignals;
    use crate::reconcile::description::Description;
    use crate::reconcile::reconciler::Reconciler;
    use wgpui_layout::taffy_tree::LayoutTree;

    #[test]
    fn capture_is_explicit_and_preserves_reuse_metadata() {
        let description = || Description::raw_text("retained");
        let mut reconciler = Reconciler::new();
        let mut layout = LayoutTree::new();
        let first = reconciler
            .reconcile(description(), &mut layout)
            .expect("first");
        let second = reconciler
            .reconcile(description(), &mut layout)
            .expect("second");
        layout
            .compute_layout(
                second.nodes()[0].layout_node,
                wgpui_layout::taffy_tree::definite(40.0, 20.0),
            )
            .expect("layout");
        let walk = RetainedWalk::build(&second, &layout, &FrameSignals::new()).expect("walk");
        let snapshot = RetainedFrameSnapshot::capture(
            reconciler.frame(),
            &second,
            &walk,
            DamagePlan::from_walks(
                Some(&RetainedWalk::build(&first, &layout, &FrameSignals::new()).expect("walk")),
                &walk,
                &[],
            ),
        )
        .expect("matching plan and walk");
        assert_eq!(snapshot.schema_version, RETAINED_FRAME_SNAPSHOT_VERSION);
        assert_eq!(snapshot.elements.len(), 1);
        assert!(snapshot.elements[0].layout_skipped);
        assert!(snapshot.elements[0].paint_skipped);
    }
}
