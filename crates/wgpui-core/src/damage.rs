//! Capture-time regional damage attribution.
//!
//! The scene patch remains the authoritative rendering protocol. This module
//! provides the strongest damage seam available before a renderer has a
//! primitive-to-tile upload planner: it maps retained element changes and
//! interaction regions to the shared walk's root and tile ownership. It is
//! explicit and caller-owned, so normal frames do not allocate damage records.

use crate::invalidation::Invalidation;
use crate::reconcile::instance::InstanceKey;
use crate::reconcile::plan::{NodeOutcome, RebuildReason};
use crate::reconcile::walk::{RetainedWalk, RetainedWalkNode, TileOwnership};
use crate::scene::BoundaryId;
use wgpui_layout::taffy_tree::LayoutRect;

/// Why a region was included in a damage plan.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DamageReason {
    /// A primitive or its layout changed.
    Content,
    /// The effective clip changed, including a clip expanding to reveal old
    /// content.
    Clip,
    /// A hover transition affected this region.
    Hover,
    /// A non-compositor scroll moved content into or out of the visible area.
    ScrollReveal,
    /// A resource backing the element changed.
    Resource,
}

/// One element-attributed regional damage record.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DamageRecord {
    /// Stable retained address responsible for the region.
    pub address: InstanceKey,
    /// Root that owns the region's paint.
    pub owning_root: BoundaryId,
    /// The affected absolute region, already intersected with its effective
    /// clip where one exists.
    pub content_rect: LayoutRect,
    /// Why the region must be considered.
    pub reason: DamageReason,
    /// Tile or overlay attribution in `owning_root`.
    pub tile_ownership: TileOwnership,
}

/// Regional damage for one explicitly captured/planned frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DamagePlan {
    records: Vec<DamageRecord>,
}

impl DamagePlan {
    /// Build damage by comparing two shared walks. `previous` may be `None`
    /// for the first captured frame.
    pub fn from_walks(
        previous: Option<&RetainedWalk>,
        current: &RetainedWalk,
        hover_regions: &[LayoutRect],
    ) -> Self {
        let mut plan = Self::default();
        for current_node in current.nodes() {
            let previous_node =
                previous.and_then(|walk| walk.node_for_address(current_node.address));
            let content_changed = current_node.invalidation.intersects(Invalidation::DISPLAY)
                || matches!(
                    current_node.outcome,
                    NodeOutcome::Rebuilt(RebuildReason::NewInstance)
                        | NodeOutcome::Rebuilt(RebuildReason::TypeMismatch)
                        | NodeOutcome::Rebuilt(RebuildReason::NoDiffKey)
                        | NodeOutcome::Rebuilt(RebuildReason::KeyChanged)
                        | NodeOutcome::Rebuilt(RebuildReason::AncestorRebuilt)
                        | NodeOutcome::Rebuilt(RebuildReason::ReconciliationDisabled)
                );
            let clip_changed = previous_node.is_some_and(|previous_node| {
                previous_node.effective_clip != current_node.effective_clip
            });
            let bounds_changed = previous_node.is_some_and(|previous_node| {
                previous_node.bounds != current_node.bounds
                    || previous_node.owning_root != current_node.owning_root
                    || previous_node.tile_ownership != current_node.tile_ownership
            });

            if content_changed {
                plan.push(current_node, current_node.bounds, DamageReason::Content);
            } else if clip_changed {
                let rect = previous_node
                    .map(|previous_node| union(previous_node.bounds, current_node.bounds))
                    .unwrap_or(current_node.bounds);
                plan.push(current_node, rect, DamageReason::Clip);
            } else if bounds_changed {
                plan.push(
                    current_node,
                    current_node.bounds,
                    DamageReason::ScrollReveal,
                );
            }

            for region in hover_regions {
                if let Some(rect) = intersect(*region, current_node.bounds)
                    && current_node
                        .effective_clip
                        .is_none_or(|clip| intersect(clip, rect).is_some())
                {
                    plan.push(current_node, rect, DamageReason::Hover);
                }
            }
        }

        if let Some(previous) = previous {
            for previous_node in previous.nodes() {
                if current.node_for_address(previous_node.address).is_none() {
                    plan.push(previous_node, previous_node.bounds, DamageReason::Content);
                }
            }
        }
        plan
    }

    /// Every damage record, in shared-walk order.
    pub fn records(&self) -> &[DamageRecord] {
        &self.records
    }

    /// Whether no regional work was attributed.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    fn push(&mut self, node: &RetainedWalkNode, rect: LayoutRect, reason: DamageReason) {
        let content_rect = match node.effective_clip {
            Some(clip) => {
                let Some(content_rect) = intersect(clip, rect) else {
                    return;
                };
                content_rect
            }
            None => rect.with_non_negative_size(),
        };
        if content_rect.width <= 0.0 || content_rect.height <= 0.0 {
            return;
        }
        self.records.push(DamageRecord {
            address: node.address,
            owning_root: node.owning_root,
            content_rect,
            reason,
            tile_ownership: node.tile_ownership,
        });
    }
}

fn intersect(left: LayoutRect, right: LayoutRect) -> Option<LayoutRect> {
    let x = left.x.max(right.x);
    let y = left.y.max(right.y);
    let right_edge = (left.x + left.width).min(right.x + right.width);
    let bottom_edge = (left.y + left.height).min(right.y + right.height);
    (right_edge > x && bottom_edge > y).then_some(LayoutRect {
        x,
        y,
        width: right_edge - x,
        height: bottom_edge - y,
    })
}

fn union(left: LayoutRect, right: LayoutRect) -> LayoutRect {
    let x = left.x.min(right.x);
    let y = left.y.min(right.y);
    let right_edge = (left.x + left.width).max(right.x + right.width);
    let bottom_edge = (left.y + left.height).max(right.y + right.height);
    LayoutRect {
        x,
        y,
        width: (right_edge - x).max(0.0),
        height: (bottom_edge - y).max(0.0),
    }
}

trait NonNegativeLayoutRect {
    fn with_non_negative_size(self) -> LayoutRect;
}

impl NonNegativeLayoutRect for LayoutRect {
    fn with_non_negative_size(mut self) -> LayoutRect {
        self.width = self.width.max(0.0);
        self.height = self.height.max(0.0);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::invalidation::axes::Invalidation;
    use crate::reconcile::description::Description;
    use crate::reconcile::diff_key::ReconcileKey;
    use crate::reconcile::reconciler::Reconciler;
    use crate::reconcile::walk::RetainedWalk;
    use std::any::Any;
    use wgpui_layout::taffy_tree::{Dimension, LayoutSize, LayoutStyle};

    #[derive(Debug, PartialEq)]
    struct Key(u32);

    impl ReconcileKey for Key {
        fn compare(&self, previous: &dyn ReconcileKey) -> Invalidation {
            crate::reconcile::diff_key::compare_by_equality(self, previous, Invalidation::all())
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    fn scene(value: u32, child_value: u32) -> Description {
        Description::new::<u8>()
            .diff_key(Key(value))
            .style(LayoutStyle {
                size: LayoutSize {
                    width: Dimension::length(100.0),
                    height: Dimension::length(100.0),
                },
                ..LayoutStyle::default()
            })
            .children([Description::new::<u16>()
                .diff_key(Key(child_value))
                .style(LayoutStyle {
                    size: LayoutSize {
                        width: Dimension::length(20.0),
                        height: Dimension::length(20.0),
                    },
                    ..LayoutStyle::default()
                })])
    }

    #[test]
    fn child_display_damage_does_not_attribute_the_clean_parent() {
        let mut reconciler = Reconciler::new();
        let mut layout = wgpui_layout::taffy_tree::LayoutTree::new();
        let first = reconciler
            .reconcile(scene(0, 0), &mut layout)
            .expect("first");
        layout
            .compute_layout(
                first.nodes()[0].layout_node,
                wgpui_layout::taffy_tree::definite(100.0, 100.0),
            )
            .expect("first layout");
        let first_walk =
            RetainedWalk::build(&first, &layout, &crate::invalidation::FrameSignals::new())
                .expect("walk");
        let second = reconciler
            .reconcile(scene(0, 1), &mut layout)
            .expect("second");
        layout
            .compute_layout(
                second.nodes()[0].layout_node,
                wgpui_layout::taffy_tree::definite(100.0, 100.0),
            )
            .expect("second layout");
        let second_walk =
            RetainedWalk::build(&second, &layout, &crate::invalidation::FrameSignals::new())
                .expect("walk");
        let damage = DamagePlan::from_walks(Some(&first_walk), &second_walk, &[]);
        assert!(
            damage
                .records()
                .iter()
                .any(|record| record.reason == DamageReason::Content)
        );
        assert_eq!(
            damage
                .records()
                .iter()
                .filter(|record| record.address == second.nodes()[0].address)
                .count(),
            0,
            "a child display change must not dirty its clean parent"
        );
    }
}
