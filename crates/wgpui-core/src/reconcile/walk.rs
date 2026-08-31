//! The shared retained geometry walk.
//!
//! Reconciliation decides whether an element may reuse its retained records,
//! but it does not know the resolved geometry. This walk is the one place that
//! combines the plan, layout, scroll transforms, clips, and boundary policies.
//! Emission, interaction registration, damage planning, and capture all consume
//! its result so those consumers cannot drift into different coordinate chains.

use crate::invalidation::request::FrameSignals;
use crate::reconcile::instance::InstanceKey;
use crate::reconcile::plan::{FramePlan, NodeOutcome, RebuildReason};
use crate::scene::layer::{BoundaryId, LayerId, LayerKey};
use crate::scene::tile::{TileCoord, TileGrid, TilePlacement};
use wgpui_layout::taffy_tree::{LayoutError, LayoutRect, LayoutTree};

/// Which retained tile domain owns an element's paint.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TileOwnership {
    /// The owning root is not using a tile grid.
    Untiled,
    /// The element's paint is anchored to one tile in its owning root.
    Tile(TileCoord),
    /// The element's paint belongs to the owning root's unbuffered overlay.
    Overlay,
}

impl TileOwnership {
    fn for_bounds(grid: Option<TileGrid>, bounds: LayoutRect) -> Self {
        let Some(grid) = grid else {
            return Self::Untiled;
        };
        match grid.placement(crate::geometry::Rect {
            min_x: bounds.x,
            min_y: bounds.y,
            max_x: bounds.x + bounds.width,
            max_y: bounds.y + bounds.height,
        }) {
            TilePlacement::Tile(coord) => Self::Tile(coord),
            TilePlacement::Overlay => Self::Overlay,
        }
    }
}

/// Metadata for one element after the shared transform/clip walk.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RetainedWalkNode {
    /// Stable retained address from reconciliation.
    pub address: InstanceKey,
    /// Resolved absolute bounds after folded scroll offsets.
    pub bounds: LayoutRect,
    /// Scroll translation folded into this node's absolute position.
    pub accumulated_scroll_translation: [f32; 2],
    /// The clip inherited by this node from its ancestors.
    pub effective_clip: Option<LayoutRect>,
    /// The boundary/root that owns this node's paint.
    pub owning_root: BoundaryId,
    /// Layer that receives this node's own primitives.
    pub layer: LayerId,
    /// The root and layer children inherit after this node.
    pub child_root: BoundaryId,
    pub child_layer: LayerId,
    /// Whether this node folds its scroll offset into child positions.
    pub child_scroll_translation: [f32; 2],
    /// Whether the declared boundary carries the scroll as a layer transform.
    pub compositor_scroll: bool,
    /// Whether this node clips its children to its resolved bounds.
    pub clips_children: bool,
    /// The tile in which this node's paint is retained.
    pub tile_ownership: TileOwnership,
    /// Reconciliation outcome, including the precise rebuild reason.
    pub outcome: NodeOutcome,
    /// Axes invalidated by reconciliation.
    pub invalidation: crate::invalidation::Invalidation,
    /// Whether layout was reused for this node.
    pub layout_skipped: bool,
    /// Whether paint was skipped for this node.
    pub paint_skipped: bool,
    /// Pre-order depth in the retained tree.
    pub depth: u32,
}

impl RetainedWalkNode {
    /// The rebuild reason, when this node was rebuilt.
    pub fn rebuild_reason(&self) -> Option<RebuildReason> {
        match self.outcome {
            NodeOutcome::Rebuilt(reason) => Some(reason),
            NodeOutcome::Reused | NodeOutcome::Uncached => None,
        }
    }
}

/// The complete shared walk for one laid-out frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RetainedWalk {
    nodes: Vec<RetainedWalkNode>,
}

impl RetainedWalk {
    /// Walk a plan using the same boundary scroll decision as emission.
    pub fn build(
        plan: &FramePlan,
        layout: &LayoutTree,
        signals: &FrameSignals,
    ) -> Result<Self, LayoutError> {
        let root = WalkFrame {
            depth: 0,
            layer: LayerId::from_key(LayerKey::untiled(BoundaryId::ROOT)),
            root: BoundaryId::ROOT,
            origin: [0.0, 0.0],
            accumulated_scroll_translation: [0.0, 0.0],
            child_scroll_translation: [0.0, 0.0],
            clip: None,
            tile_grid: None,
        };
        let mut stack = Vec::new();
        let mut nodes = Vec::with_capacity(plan.nodes().len());

        for planned in plan.nodes() {
            while stack
                .last()
                .is_some_and(|frame: &WalkFrame| frame.depth >= planned.depth)
            {
                stack.pop();
            }
            let parent = *stack.last().unwrap_or(&root);
            if planned.depth != u32::try_from(stack.len()).unwrap_or(u32::MAX) {
                return Err(LayoutError::Taffy(
                    "planned node depth is not reachable from its predecessor".to_string(),
                ));
            }
            let rectangle = layout.layout_of(planned.layout_node)?;
            let origin = [
                parent.origin[0] + rectangle.x + parent.child_scroll_translation[0],
                parent.origin[1] + rectangle.y + parent.child_scroll_translation[1],
            ];
            let bounds = LayoutRect {
                x: origin[0],
                y: origin[1],
                width: rectangle.width,
                height: rectangle.height,
            };
            let child_root = planned.declared_boundary.unwrap_or(parent.root);
            let child_layer = LayerId::from_key(LayerKey::untiled(child_root));
            let compositor_scroll = planned.declared_boundary.is_some()
                && signals
                    .reason_for_layer(child_layer)
                    .permits_transform_only();
            let child_scroll_translation = if compositor_scroll {
                [0.0, 0.0]
            } else {
                planned.scroll_offset
            };
            let child_tile_grid = planned
                .boundary_policy
                .and_then(|policy| policy.buffering.tile_grid())
                .or_else(|| {
                    (planned.declared_boundary.is_none())
                        .then_some(parent.tile_grid)
                        .flatten()
                });
            let tile_ownership = TileOwnership::for_bounds(parent.tile_grid, bounds);
            let node = RetainedWalkNode {
                address: planned.address,
                bounds,
                accumulated_scroll_translation: [
                    parent.accumulated_scroll_translation[0] + parent.child_scroll_translation[0],
                    parent.accumulated_scroll_translation[1] + parent.child_scroll_translation[1],
                ],
                effective_clip: parent.clip,
                owning_root: planned.boundary,
                layer: parent.layer,
                child_root,
                child_layer,
                child_scroll_translation,
                compositor_scroll,
                clips_children: planned.clip_children,
                tile_ownership,
                outcome: planned.outcome,
                invalidation: planned.invalidation,
                layout_skipped: matches!(planned.outcome, NodeOutcome::Reused),
                paint_skipped: planned.skipped_prepaint_and_paint(),
                depth: planned.depth,
            };
            nodes.push(node);
            stack.push(WalkFrame {
                depth: planned.depth,
                layer: child_layer,
                root: child_root,
                origin,
                accumulated_scroll_translation: node.accumulated_scroll_translation,
                child_scroll_translation,
                clip: if planned.clip_children {
                    Some(intersect_layout_rect(parent.clip, bounds))
                } else {
                    parent.clip
                },
                tile_grid: child_tile_grid,
            });
        }
        Ok(Self { nodes })
    }

    /// Every walked node in pre-order.
    pub fn nodes(&self) -> &[RetainedWalkNode] {
        &self.nodes
    }

    /// The entry at the plan's pre-order index.
    pub fn node(&self, index: usize) -> Option<&RetainedWalkNode> {
        self.nodes.get(index)
    }

    /// Find the current geometry for a retained address.
    pub fn node_for_address(&self, address: InstanceKey) -> Option<&RetainedWalkNode> {
        self.nodes.iter().find(|node| node.address == address)
    }
}

#[derive(Copy, Clone, Debug)]
struct WalkFrame {
    depth: u32,
    layer: LayerId,
    root: BoundaryId,
    origin: [f32; 2],
    accumulated_scroll_translation: [f32; 2],
    child_scroll_translation: [f32; 2],
    clip: Option<LayoutRect>,
    tile_grid: Option<TileGrid>,
}

fn intersect_layout_rect(clip: Option<LayoutRect>, bounds: LayoutRect) -> LayoutRect {
    let Some(clip) = clip else {
        return bounds;
    };
    let left = clip.x.max(bounds.x);
    let top = clip.y.max(bounds.y);
    let right = (clip.x + clip.width).min(bounds.x + bounds.width);
    let bottom = (clip.y + clip.height).min(bounds.y + bounds.height);
    LayoutRect {
        x: left,
        y: top,
        width: (right - left).max(0.0),
        height: (bottom - top).max(0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::identity::BoundaryIdentity;
    use crate::invalidation::axes::Invalidation;
    use crate::reconcile::description::{Description, ElementId};
    use crate::reconcile::diff_key::ReconcileKey;
    use crate::reconcile::reconciler::Reconciler;
    use std::any::Any;
    use wgpui_layout::taffy_tree::{Dimension, FlexDirection, LayoutSize, LayoutStyle};

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

    fn style(width: f32, height: f32) -> LayoutStyle {
        LayoutStyle {
            flex_direction: FlexDirection::Column,
            size: LayoutSize {
                width: Dimension::length(width),
                height: Dimension::length(height),
            },
            ..LayoutStyle::default()
        }
    }

    fn frame(scroll: [f32; 2], order: &[u64]) -> Description {
        Description::new::<u8>()
            .diff_key(Key(0))
            .style(style(200.0, 100.0))
            .clip_children()
            .scroll_offset(scroll)
            .children(order.iter().map(|id| {
                Description::new::<u16>()
                    .id(*id)
                    .diff_key(Key(*id as u32))
                    .style(style(50.0, 20.0))
            }))
    }

    #[test]
    fn shared_walk_keeps_explicit_addresses_across_reorder() -> Result<(), LayoutError> {
        let mut reconciler = Reconciler::new();
        let mut layout = LayoutTree::new();
        let first = reconciler
            .reconcile(frame([0.0, 0.0], &[1, 2]), &mut layout)
            .map_err(|error| LayoutError::Taffy(error.to_string()))?;
        let first_walk = RetainedWalk::build(&first, &layout, &FrameSignals::new())?;
        let second = reconciler
            .reconcile(frame([0.0, 0.0], &[2, 1]), &mut layout)
            .map_err(|error| LayoutError::Taffy(error.to_string()))?;
        let second_walk = RetainedWalk::build(&second, &layout, &FrameSignals::new())?;
        for address in [
            InstanceKey::from_path(&[ElementId::Slot(0), ElementId::Integer(1)]),
            InstanceKey::from_path(&[ElementId::Slot(0), ElementId::Integer(2)]),
        ] {
            assert_eq!(
                first_walk
                    .node_for_address(address)
                    .map(|node| node.address),
                second_walk
                    .node_for_address(address)
                    .map(|node| node.address)
            );
        }
        Ok(())
    }

    #[test]
    fn shared_walk_folds_ordinary_scroll_but_not_boundary_transform_scroll() {
        let mut reconciler = Reconciler::new();
        let mut layout = LayoutTree::new();
        let description = Description::new::<u8>()
            .diff_key(Key(0))
            .style(style(200.0, 100.0))
            .scroll_offset([-10.0, -5.0])
            .children([Description::new::<u16>()
                .diff_key(Key(1))
                .style(style(40.0, 20.0))]);
        let plan = reconciler
            .reconcile(description, &mut layout)
            .expect("description reconciles");
        let ordinary =
            RetainedWalk::build(&plan, &layout, &FrameSignals::new()).expect("walk succeeds");
        assert_eq!(ordinary.nodes()[1].bounds.x, -10.0);

        let mut reconciler = Reconciler::new();
        let mut layout = LayoutTree::new();
        let description = Description::new::<u8>()
            .diff_key(Key(0))
            .style(style(200.0, 100.0))
            .boundary()
            .scroll_offset([-10.0, -5.0])
            .children([Description::new::<u16>()
                .diff_key(Key(1))
                .style(style(40.0, 20.0))]);
        let plan = reconciler
            .reconcile(description, &mut layout)
            .expect("description reconciles");
        let mut signals = FrameSignals::new();
        signals.scrolled(LayerId::from_key(LayerKey::untiled(
            BoundaryIdentity::from_path(&[ElementId::Slot(0)]),
        )));
        let boundary = RetainedWalk::build(&plan, &layout, &signals).expect("walk succeeds");
        assert_eq!(boundary.nodes()[1].bounds.x, 0.0);
    }

    #[test]
    fn retained_attribution_survives_reorder_and_boundary_scroll() {
        let policy = crate::boundary::policy::BoundaryPolicy {
            buffering: crate::boundary::policy::Buffering::Tiled {
                tile_size: crate::geometry::Size::pixels(64.0, 64.0),
                retain_radius: 1,
            },
            ..crate::boundary::policy::BoundaryPolicy::default()
        };
        let make_frame = |scroll, order: &[u64]| {
            Description::new::<u8>()
                .diff_key(Key(0))
                .style(style(160.0, 100.0))
                .boundary_with_policy(policy)
                .scroll_offset(scroll)
                .children(order.iter().map(|id| {
                    Description::new::<u16>()
                        .id(*id)
                        .diff_key(Key(*id as u32))
                        .style(style(40.0, 20.0))
                }))
        };

        let mut reconciler = Reconciler::new();
        let mut layout = LayoutTree::new();
        let first = reconciler
            .reconcile(make_frame([0.0, 0.0], &[1, 2]), &mut layout)
            .expect("first frame reconciles");
        layout
            .compute_layout(
                first.nodes()[0].layout_node,
                wgpui_layout::taffy_tree::definite(160.0, 100.0),
            )
            .expect("first layout");
        let first_walk =
            RetainedWalk::build(&first, &layout, &FrameSignals::new()).expect("first walk");

        let second = reconciler
            .reconcile(make_frame([-20.0, -12.0], &[2, 1]), &mut layout)
            .expect("second frame reconciles");
        layout
            .compute_layout(
                second.nodes()[0].layout_node,
                wgpui_layout::taffy_tree::definite(160.0, 100.0),
            )
            .expect("second layout");
        let boundary = BoundaryIdentity::from_path(&[ElementId::Slot(0)]);
        let mut signals = FrameSignals::new();
        signals.scrolled(LayerId::from_key(LayerKey::untiled(boundary)));
        let second_walk = RetainedWalk::build(&second, &layout, &signals).expect("second walk");

        for id in [1_u64, 2] {
            let address = InstanceKey::from_path(&[ElementId::Slot(0), ElementId::Integer(id)]);
            let first_node = first_walk.node_for_address(address).expect("first node");
            let second_node = second_walk.node_for_address(address).expect("second node");
            assert_eq!(first_node.owning_root, second_node.owning_root);
            assert_eq!(first_node.tile_ownership, second_node.tile_ownership);
        }
        assert!(second_walk.nodes()[0].compositor_scroll);
        let reordered_child = second_walk
            .node_for_address(InstanceKey::from_path(&[
                ElementId::Slot(0),
                ElementId::Integer(1),
            ]))
            .expect("reordered child");
        assert_eq!(reordered_child.bounds.x, 0.0);
        assert_eq!(reordered_child.bounds.y, 20.0);
    }
}
