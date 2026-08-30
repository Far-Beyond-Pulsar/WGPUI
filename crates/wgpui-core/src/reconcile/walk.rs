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
use crate::reconcile::plan::FramePlan;
use crate::scene::layer::{BoundaryId, LayerId};
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
        result.push(WalkNode {
            address: node.address,
            emission_bounds,
            absolute_bounds,
            clip: parent.clip,
            emission_clip: parent.clip.map(|clip| LayoutRect {
                x: clip.min_x - accumulated_scroll[0],
                y: clip.min_y - accumulated_scroll[1],
                width: clip.width(),
                height: clip.height(),
            }),
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
            layer: child_layer,
        });
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::policy::BoundaryPolicy;
    use crate::invalidation::request::FrameSignals;
    use crate::reconcile::description::{Description, ElementId};
    use crate::reconcile::diff_key::AlwaysDirty;
    use crate::reconcile::reconciler::Reconciler;
    use crate::scene::layer::BoundaryId;
    use wgpui_layout::taffy_tree::{Dimension, LayoutSize, LayoutStyle, definite};

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
        assert_eq!(leaf.emission_clip.map(|clip| clip.x), Some(12.0));
    }
}
