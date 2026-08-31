//! The retained geometry walk shared by emission and interaction collection.
//!
//! Layout produces rectangles in a parent's content space, while scrolling can
//! either be folded into emitted primitive coordinates or applied by a retained
//! layer transform. Keeping those two decisions in separate walks makes the
//! renderer and hit testing disagree exactly when a nested scroll root moves.
//! This module computes both representations from the same stack.

use crate::geometry::Rect;
use crate::invalidation::request::FrameSignals;
use crate::reconcile::instance::InstanceKey;
use crate::reconcile::plan::{FramePlan, NodeOutcome, RebuildReason};
use crate::scene::layer::{BoundaryId, LayerId, LayerKey};
use crate::scene::tile::{TileCoord, TileGrid, TilePlacement};
use wgpui_layout::taffy_tree::{LayoutError, LayoutRect, LayoutTree};

/// One node's result from [`shared_walk`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct WalkNode {
    /// The stable retained address of the node.
    pub address: InstanceKey,
    /// The rectangle used when an emitter writes primitive coordinates.
    pub emission_bounds: LayoutRect,
    /// The same rectangle after every ancestor scroll has been applied.
    pub absolute_bounds: Rect,
    /// The effective clip in window coordinates for this node's own content.
    pub clip: Option<Rect>,
    /// The same clip expressed in the node's emission coordinate space.
    pub emission_clip: Option<LayoutRect>,
    /// The part of `absolute_bounds` that can actually receive input or paint.
    pub visible_bounds: Rect,
    /// The total scroll displacement from the emission space to window space.
    pub accumulated_scroll: [f32; 2],
    /// The boundary owning the node's own primitive records.
    pub owning_root: BoundaryId,
    /// The boundary children enter, which may be the node's declared boundary.
    pub child_root: BoundaryId,
    /// The layer owning the node's own primitive records.
    pub layer: LayerId,
    /// The layer children enter when this node declares a boundary.
    pub child_layer: LayerId,
    /// Scroll folded into child emission coordinates.
    pub child_emission_offset: [f32; 2],
    /// Scroll applied to child screen coordinates in all cases.
    pub child_screen_offset: [f32; 2],
    /// The node's effective child clip.
    pub child_clip: Option<Rect>,
    /// The node's depth in the retained plan.
    pub depth: u32,
}

#[derive(Copy, Clone)]
struct WalkContext {
    emission_origin: [f32; 2],
    screen_origin: [f32; 2],
    emission_offset: [f32; 2],
    screen_offset: [f32; 2],
    clip: Option<Rect>,
    emission_clip: Option<Rect>,
    layer: LayerId,
}

fn intersect(clip: Option<Rect>, bounds: Rect) -> Rect {
    clip.map_or(bounds, |clip| bounds.intersect(&clip))
}

/// Walk a frame plan with one transform and clip calculation.
///
/// `viewport` is included in the effective clip when supplied. The emitter
/// passes `None` to preserve its existing scene-space contract; the native
/// input path passes the window rectangle so fully clipped registrations never
/// become hit targets.
pub fn shared_walk(
    plan: &FramePlan,
    layout: &LayoutTree,
    signals: &FrameSignals,
    viewport: Option<Rect>,
) -> Result<Vec<WalkNode>, LayoutError> {
    let root_layer = LayerId::from_key(crate::scene::layer::LayerKey::untiled(BoundaryId::ROOT));
    let root = WalkContext {
        emission_origin: [0.0; 2],
        screen_origin: [0.0; 2],
        emission_offset: [0.0; 2],
        screen_offset: [0.0; 2],
        clip: viewport,
        emission_clip: viewport,
        layer: root_layer,
    };
    let mut stack = Vec::new();
    let mut result = Vec::with_capacity(plan.nodes().len());

    for node in plan.nodes() {
        let depth = usize::try_from(node.depth).unwrap_or(usize::MAX);
        while stack.len() > depth {
            stack.pop();
        }
        if stack.len() != depth {
            return Err(LayoutError::UnknownNode(node.layout_node));
        }
        let parent = stack.last().copied().unwrap_or(root);
        let rectangle = layout.layout_of(node.layout_node)?;
        let emission_origin = [
            parent.emission_origin[0] + rectangle.x + parent.emission_offset[0],
            parent.emission_origin[1] + rectangle.y + parent.emission_offset[1],
        ];
        let screen_origin = [
            parent.screen_origin[0] + rectangle.x + parent.screen_offset[0],
            parent.screen_origin[1] + rectangle.y + parent.screen_offset[1],
        ];
        let emission_bounds = LayoutRect {
            x: emission_origin[0],
            y: emission_origin[1],
            width: rectangle.width,
            height: rectangle.height,
        };
        let absolute_bounds =
            Rect::from_origin_size(screen_origin, [rectangle.width, rectangle.height]);
        let visible_bounds = intersect(parent.clip, absolute_bounds);

        let child_root = node.declared_boundary.unwrap_or(node.boundary);
        let slides = node.declared_boundary.is_some_and(|boundary| {
            let layer = LayerId::from_key(crate::scene::layer::LayerKey::untiled(boundary));
            signals.reason_for_layer(layer).permits_transform_only()
        });
        let child_emission_offset = if slides { [0.0; 2] } else { node.scroll_offset };
        let child_screen_offset = node.scroll_offset;
        let child_clip = if node.clip_children {
            Some(intersect(parent.clip, absolute_bounds))
        } else {
            parent.clip
        };
        let child_layer = if node.declared_boundary.is_some() {
            LayerId::from_key(crate::scene::layer::LayerKey::untiled(child_root))
        } else {
            parent.layer
        };
        let accumulated_scroll = [
            screen_origin[0] - emission_origin[0],
            screen_origin[1] - emission_origin[1],
        ];
        let emission_clip = parent.emission_clip.map(|clip| LayoutRect {
            x: clip.min_x - accumulated_scroll[0],
            y: clip.min_y - accumulated_scroll[1],
            width: clip.width(),
            height: clip.height(),
        });
        let child_emission_clip = if node.declared_boundary.is_some() {
            None
        } else if node.clip_children {
            child_clip
        } else {
            parent.emission_clip
        };
        result.push(WalkNode {
            address: node.address,
            emission_bounds,
            absolute_bounds,
            clip: parent.clip,
            emission_clip,
            visible_bounds,
            accumulated_scroll,
            owning_root: node.boundary,
            child_root,
            layer: parent.layer,
            child_layer,
            child_emission_offset,
            child_screen_offset,
            child_clip,
            depth: node.depth,
        });
        stack.push(WalkContext {
            emission_origin,
            screen_origin,
            emission_offset: child_emission_offset,
            screen_offset: child_screen_offset,
            clip: child_clip,
            emission_clip: child_emission_clip,
            layer: child_layer,
        });
    }
    Ok(result)
}
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
    use crate::boundary::policy::BoundaryPolicy;
    use crate::invalidation::request::FrameSignals;
    use crate::reconcile::description::{Description, ElementId};
    use crate::reconcile::diff_key::AlwaysDirty;
    use crate::reconcile::reconciler::Reconciler;
    use crate::boundary::identity::BoundaryIdentity;
    use crate::invalidation::axes::Invalidation;
    use crate::reconcile::diff_key::ReconcileKey;
    use crate::scene::layer::{BoundaryId, LayerId, LayerKey};
    use std::any::Any;
    use wgpui_layout::taffy_tree::{Dimension, FlexDirection, LayoutSize, LayoutStyle, definite};

    struct Root;
    struct Scroll;
    struct Leaf;

    fn plan() -> (FramePlan, LayoutTree) {
        let leaf = Description::new::<Leaf>()
            .id(ElementId::from("leaf"))
            .diff_key(AlwaysDirty)
            .style(LayoutStyle {
                size: LayoutSize {
                    width: Dimension::length(40.0),
                    height: Dimension::length(40.0),
                },
                ..LayoutStyle::default()
            });
        let scroll = Description::new::<Scroll>()
            .id(ElementId::from("scroll"))
            .diff_key(AlwaysDirty)
            .style(LayoutStyle {
                size: LayoutSize {
                    width: Dimension::length(100.0),
                    height: Dimension::length(100.0),
                },
                ..LayoutStyle::default()
            })
            .scroll_offset([-12.0, -18.0])
            .clip_children()
            .boundary_with_policy(BoundaryPolicy::default())
            .child(leaf);
        let description = Description::new::<Root>()
            .id(ElementId::from("root"))
            .diff_key(AlwaysDirty)
            .style(LayoutStyle {
                size: LayoutSize {
                    width: Dimension::length(200.0),
                    height: Dimension::length(200.0),
                },
                ..LayoutStyle::default()
            })
            .child(scroll);
        let mut layout = LayoutTree::new();
        let mut reconciler = Reconciler::new();
        let plan = reconciler
            .reconcile(description, &mut layout)
            .expect("valid plan");
        layout
            .compute_layout(
                plan.root().expect("root").layout_node,
                definite(200.0, 200.0),
            )
            .expect("valid layout");
        (plan, layout)
    }

    #[test]
    fn shared_walk_applies_nested_scroll_to_screen_bounds_and_clip() {
        let (plan, layout) = plan();
        let nodes = shared_walk(
            &plan,
            &layout,
            &FrameSignals::new(),
            Some(Rect::from_origin_size([0.0, 0.0], [200.0, 200.0])),
        )
        .expect("walk succeeds");
        let leaf = nodes.last().expect("leaf is present");
        assert_eq!(leaf.accumulated_scroll, [0.0, 0.0]);
        assert_eq!(leaf.absolute_bounds.min_x, -12.0);
        assert_eq!(leaf.absolute_bounds.min_y, -18.0);
        assert_eq!(
            leaf.visible_bounds,
            Rect::from_origin_size([0.0, 0.0], [28.0, 22.0])
        );
        assert_ne!(leaf.owning_root, BoundaryId::ROOT);
        assert_eq!(leaf.depth, 2);

        let boundary = nodes[1].child_layer;
        let mut signals = FrameSignals::new();
        signals.scrolled(boundary);
        let transformed = shared_walk(
            &plan,
            &layout,
            &signals,
            Some(Rect::from_origin_size([0.0, 0.0], [200.0, 200.0])),
        )
        .expect("transform-only walk succeeds");
        let leaf = transformed.last().expect("leaf is present");
        assert_eq!(leaf.emission_bounds.x, 0.0);
        assert_eq!(leaf.absolute_bounds.min_x, -12.0);
        assert_eq!(
            leaf.emission_clip, None,
            "a declared boundary owns its clip in the GPU layer even when its scroll is not transform-only"
        );
    }

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
