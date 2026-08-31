//! Capture-time interaction and retained geometry records.

use super::Window;
use super::hitbox::HitboxId;
use crate::geometry::Rect;
use crate::invalidation::request::FrameSignals;
use crate::reconcile::{FramePlan, InstanceKey, ScrollInfo, shared_walk};
use crate::scene::layer::BoundaryId;
use wgpui_layout::taffy_tree::{LayoutError, LayoutTree};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum InputEventFamily {
    All,
    Keyboard,
    Pointer,
    Focus,
    Scroll,
    Click,
}

impl InputEventFamily {
    pub(crate) fn matches(self, event: &super::input::InputEvent) -> bool {
        match self {
            Self::All => true,
            Self::Keyboard => matches!(
                event,
                super::input::InputEvent::KeyDown(_) | super::input::InputEvent::KeyUp(_)
            ),
            Self::Pointer => matches!(
                event,
                super::input::InputEvent::MouseDown(_)
                    | super::input::InputEvent::MouseUp(_)
                    | super::input::InputEvent::MouseMove(_)
                    | super::input::InputEvent::MouseEnter(_)
                    | super::input::InputEvent::MouseLeave(_)
            ),
            Self::Focus => false,
            Self::Scroll => matches!(event, super::input::InputEvent::Scroll(_)),
            Self::Click => matches!(event, super::input::InputEvent::Click(_)),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DispatchPhase {
    Capture,
    Bubble,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ListenerInfo {
    pub family: InputEventFamily,
    pub phase: DispatchPhase,
    pub registration_order: u64,
    pub handler_present: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DispatchNodeInfo {
    pub id: super::dispatch::DispatchNodeId,
    pub parent: Option<super::dispatch::DispatchNodeId>,
    pub ancestry: Vec<super::dispatch::DispatchNodeId>,
    pub address: Option<InstanceKey>,
    pub listeners: Vec<ListenerInfo>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DispatchTreeSnapshot {
    pub nodes: Vec<DispatchNodeInfo>,
    pub hitbox_nodes: Vec<(HitboxId, super::dispatch::DispatchNodeId)>,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum InputRejectionReason {
    OutsideBounds,
    NotHitTestable,
    Clipped { clip: Rect },
    MissingDispatchNode,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct HitboxInfo {
    pub id: HitboxId,
    pub bounds: Rect,
    pub visible_bounds: Rect,
    pub clip: Option<Rect>,
    pub z_index: i32,
    pub order: u64,
    pub hit_testable: bool,
    pub dispatch_node: Option<super::dispatch::DispatchNodeId>,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct InputRejection {
    pub position: [f32; 2],
    pub reason: InputRejectionReason,
    pub hitbox: Option<HitboxId>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InteractionSnapshot {
    pub active: bool,
    pub hovered: Option<HitboxId>,
    pub pressed: Option<HitboxId>,
    pub focused: Option<super::focus::FocusId>,
    pub focus_visible: bool,
    pub hitboxes: Vec<HitboxInfo>,
    pub dispatch: DispatchTreeSnapshot,
    pub input_rejection: Option<InputRejection>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ElementInteractionInfo {
    pub address: InstanceKey,
    pub bounds: Rect,
    pub visible_bounds: Rect,
    pub clip: Option<Rect>,
    pub accumulated_scroll: [f32; 2],
    pub owning_root: BoundaryId,
    pub child_root: BoundaryId,
    pub scroll_offset: [f32; 2],
    pub scroll: Option<ScrollInfo>,
    pub dispatch_node: Option<super::dispatch::DispatchNodeId>,
    pub dispatch_ancestry: Vec<super::dispatch::DispatchNodeId>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FrameInteractionSnapshot {
    pub interaction: InteractionSnapshot,
    pub elements: Vec<ElementInteractionInfo>,
}

impl Window {
    /// Inspect interaction state and hit-test geometry without exposing any
    /// application callback or closure internals.
    pub fn inspect_interaction(&self, point: Option<[f32; 2]>) -> InteractionSnapshot {
        let hitboxes = self
            .hit_test
            .entries()
            .iter()
            .map(|hitbox| {
                let clip = self.hitbox_clips.get(&hitbox.id).copied();
                let visible_bounds =
                    clip.map_or(hitbox.bounds, |clip| hitbox.bounds.intersect(&clip));
                HitboxInfo {
                    id: hitbox.id,
                    bounds: hitbox.bounds,
                    visible_bounds,
                    clip,
                    z_index: hitbox.z_index,
                    order: hitbox.order,
                    hit_testable: hitbox.hit_testable,
                    dispatch_node: self.dispatch.node_for_hitbox(hitbox.id),
                }
            })
            .collect();
        let input_rejection = point.and_then(|point| self.explain_hit_test(point));
        InteractionSnapshot {
            active: self.pressed.is_some(),
            hovered: self.hovered,
            pressed: self.pressed.map(|(id, _)| id),
            focused: self.focus.focused(),
            focus_visible: self.focus.focus_visible(),
            hitboxes,
            dispatch: self.dispatch.inspection_snapshot(),
            input_rejection,
        }
    }

    fn explain_hit_test(&self, point: [f32; 2]) -> Option<InputRejection> {
        if self.hit_test_point(point).is_some() {
            return None;
        }
        let Some(hitbox) = self
            .hit_test
            .entries()
            .iter()
            .filter(|hitbox| hitbox.contains(point))
            .max_by_key(|hitbox| (hitbox.z_index, hitbox.order))
        else {
            return Some(InputRejection {
                position: point,
                reason: InputRejectionReason::OutsideBounds,
                hitbox: None,
            });
        };
        let reason = if !hitbox.hit_testable {
            InputRejectionReason::NotHitTestable
        } else if let Some(clip) = self.hitbox_clips.get(&hitbox.id)
            && !contains_point(*clip, point)
        {
            InputRejectionReason::Clipped { clip: *clip }
        } else if self.dispatch.node_for_hitbox(hitbox.id).is_none() {
            InputRejectionReason::MissingDispatchNode
        } else {
            return None;
        };
        Some(InputRejection {
            position: point,
            reason,
            hitbox: Some(hitbox.id),
        })
    }

    /// Inspect the element geometry produced by the same transform/clip walk
    /// used by emission and native interaction collection.
    pub fn inspect_frame(
        &self,
        plan: &FramePlan,
        layout: &LayoutTree,
        signals: &FrameSignals,
        viewport: Rect,
        point: Option<[f32; 2]>,
    ) -> Result<FrameInteractionSnapshot, LayoutError> {
        let walked = shared_walk(plan, layout, signals, Some(viewport))?;
        let elements = walked
            .iter()
            .enumerate()
            .filter_map(|(index, geometry)| {
                let address = geometry.address;
                let dispatch_node = self.dispatch.node_for_address(address);
                Some(ElementInteractionInfo {
                    address,
                    bounds: geometry.absolute_bounds,
                    visible_bounds: geometry.visible_bounds,
                    clip: geometry.clip,
                    accumulated_scroll: geometry.accumulated_scroll,
                    owning_root: geometry.owning_root,
                    child_root: geometry.child_root,
                    scroll_offset: plan.nodes().get(index)?.scroll_offset,
                    scroll: plan.scroll_info(index),
                    dispatch_ancestry: dispatch_node
                        .map(|node| self.dispatch.ancestry(node))
                        .unwrap_or_default(),
                    dispatch_node,
                })
            })
            .collect();
        Ok(FrameInteractionSnapshot {
            interaction: self.inspect_interaction(point),
            elements,
        })
    }
}

fn contains_point(rectangle: Rect, point: [f32; 2]) -> bool {
    !rectangle.is_empty()
        && point[0] >= rectangle.min_x
        && point[0] < rectangle.max_x
        && point[1] >= rectangle.min_y
        && point[1] < rectangle.max_y
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::policy::BoundaryPolicy;
    use crate::reconcile::{Description, ElementId, Reconciler};
    use crate::window::{EventResult, InputEvent, Modifiers, MouseButton, MouseDownEvent};
    use std::cell::Cell;
    use std::rc::Rc;
    use wgpui_layout::taffy_tree::{Dimension, LayoutSize, LayoutStyle, definite};

    fn hitbox(id: u64, bounds: Rect, z_index: i32) -> super::super::hitbox::Hitbox {
        super::super::hitbox::Hitbox {
            id: HitboxId::from_raw(id),
            bounds,
            z_index,
            order: 0,
            hit_testable: true,
        }
    }

    #[test]
    fn interaction_snapshot_reports_z_order_and_dispatch_listener_metadata() {
        let mut window = Window::new();
        let root = window.dispatch_tree().root();
        let child = window.dispatch_tree().new_node(Some(root));
        let target = hitbox(1, Rect::from_origin_size([0.0, 0.0], [20.0, 20.0]), 7);
        window.register_hitbox(target, child);
        window.bind_dispatch_address(child, InstanceKey::from_raw(4));
        assert!(
            window
                .dispatch_tree()
                .on_input_for(root, InputEventFamily::Pointer, |_| EventResult::IGNORED,)
        );
        assert!(window.dispatch_tree().on_input_capture_for(
            child,
            InputEventFamily::Click,
            |_| EventResult::IGNORED,
        ));

        let snapshot = window.inspect_interaction(None);
        assert_eq!(snapshot.hitboxes[0].z_index, 7);
        assert!(snapshot.hitboxes[0].order > 0);
        assert_eq!(snapshot.dispatch.nodes[1].ancestry, vec![root, child]);
        assert_eq!(
            snapshot.dispatch.nodes[0].listeners[0].family,
            InputEventFamily::Pointer
        );
        assert_eq!(
            snapshot.dispatch.nodes[1].listeners[0].phase,
            DispatchPhase::Capture
        );
        assert!(snapshot.dispatch.nodes[1].listeners[0].handler_present);
    }

    #[test]
    fn clipped_hitboxes_are_rejected_and_never_dispatch_input() {
        let mut window = Window::new();
        let root = window.dispatch_tree().root();
        let child = window.dispatch_tree().new_node(Some(root));
        let target = hitbox(2, Rect::from_origin_size([0.0, 0.0], [100.0, 100.0]), 0);
        window.register_hitbox_with_clip(
            target,
            Rect::from_origin_size([0.0, 0.0], [20.0, 20.0]),
            child,
        );
        let calls = Rc::new(Cell::new(0));
        let observed = calls.clone();
        assert!(window.dispatch_tree().on_input(child, move |_| {
            observed.set(observed.get() + 1);
            EventResult::HANDLED
        }));

        let snapshot = window.inspect_interaction(Some([50.0, 50.0]));
        assert_eq!(snapshot.hitboxes[0].visible_bounds.width(), 20.0);
        assert_eq!(snapshot.hitboxes[0].visible_bounds.height(), 20.0);
        assert_eq!(
            snapshot.input_rejection,
            Some(InputRejection {
                position: [50.0, 50.0],
                reason: InputRejectionReason::Clipped {
                    clip: Rect::from_origin_size([0.0, 0.0], [20.0, 20.0]),
                },
                hitbox: Some(target.id),
            })
        );
        assert!(!window.handle_input(InputEvent::MouseDown(MouseDownEvent {
            button: MouseButton::Left,
            position: [crate::boundary::Pixels(50.0), crate::boundary::Pixels(50.0)],
            modifiers: Modifiers::none(),
            click_count: 1,
        })));
        assert_eq!(calls.get(), 0);
    }

    #[test]
    fn nested_scroll_inspection_uses_the_shared_transform_and_clip_walk() {
        struct Root;
        struct Outer;
        struct Inner;
        struct Leaf;
        let leaf = Description::new::<Leaf>()
            .id(ElementId::from("leaf"))
            .style(LayoutStyle {
                size: LayoutSize {
                    width: Dimension::length(40.0),
                    height: Dimension::length(40.0),
                },
                ..LayoutStyle::default()
            });
        let inner = Description::new::<Inner>()
            .id(ElementId::from("inner"))
            .style(LayoutStyle {
                size: LayoutSize {
                    width: Dimension::length(60.0),
                    height: Dimension::length(60.0),
                },
                ..LayoutStyle::default()
            })
            .scroll_offset([-5.0, -7.0])
            .scroll_info(ScrollInfo {
                handle_id: 2,
                content_size: [160.0, 160.0],
                max_offset: [100.0, 100.0],
                offset: [-5.0, -7.0],
            })
            .clip_children()
            .boundary_with_policy(BoundaryPolicy::default())
            .child(leaf);
        let outer = Description::new::<Outer>()
            .id(ElementId::from("outer"))
            .style(LayoutStyle {
                size: LayoutSize {
                    width: Dimension::length(100.0),
                    height: Dimension::length(100.0),
                },
                ..LayoutStyle::default()
            })
            .scroll_offset([-11.0, -13.0])
            .scroll_info(ScrollInfo {
                handle_id: 1,
                content_size: [200.0, 200.0],
                max_offset: [100.0, 100.0],
                offset: [-11.0, -13.0],
            })
            .clip_children()
            .boundary_with_policy(BoundaryPolicy::default())
            .child(inner);
        let description = Description::new::<Root>()
            .style(LayoutStyle {
                size: LayoutSize {
                    width: Dimension::length(200.0),
                    height: Dimension::length(200.0),
                },
                ..LayoutStyle::default()
            })
            .child(outer);
        let mut layout = LayoutTree::new();
        let mut reconciler = Reconciler::new();
        let plan = reconciler
            .reconcile(description, &mut layout)
            .expect("nested plan is valid");
        layout
            .compute_layout(
                plan.root().expect("root exists").layout_node,
                definite(200.0, 200.0),
            )
            .expect("nested layout is valid");
        let snapshot = Window::new()
            .inspect_frame(
                &plan,
                &layout,
                &FrameSignals::new(),
                Rect::from_origin_size([0.0, 0.0], [200.0, 200.0]),
                None,
            )
            .expect("shared inspection walk succeeds");
        let leaf = snapshot.elements.last().expect("leaf is inspected");
        assert_eq!(leaf.bounds.min_x, -16.0);
        assert_eq!(leaf.bounds.min_y, -20.0);
        assert_eq!(
            leaf.visible_bounds,
            Rect::from_origin_size([0.0, 0.0], [24.0, 20.0])
        );
        assert_eq!(leaf.scroll, None);
        assert_eq!(
            snapshot.elements[1].scroll.map(|scroll| scroll.handle_id),
            Some(1)
        );
        assert_eq!(
            snapshot.elements[2].scroll.map(|scroll| scroll.handle_id),
            Some(2)
        );
    }
}
