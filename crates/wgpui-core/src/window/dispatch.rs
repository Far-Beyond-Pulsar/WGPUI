use super::hitbox::HitboxId;
use super::input::{EventResult, InputEvent};
use super::inspector::{DispatchNodeInfo, DispatchTreeSnapshot, ListenerInfo};
use crate::action::Action;
use crate::reconcile::InstanceKey;
use std::collections::HashMap;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DispatchNodeId(pub u64);
type ActionCallback = Box<dyn FnMut(&dyn Action) -> EventResult>;
type InputCallback = Box<dyn FnMut(&InputEvent) -> EventResult>;
struct ActionHandler {
    registration_order: u64,
    callback: ActionCallback,
}
struct InputHandler {
    family: super::inspector::InputEventFamily,
    phase: super::inspector::DispatchPhase,
    registration_order: u64,
    callback: InputCallback,
}
struct Node {
    parent: Option<DispatchNodeId>,
    address: Option<InstanceKey>,
    action_handlers: Vec<ActionHandler>,
    capture_handlers: Vec<InputHandler>,
    input_handlers: Vec<InputHandler>,
}
#[derive(Default)]
pub struct DispatchTree {
    nodes: HashMap<DispatchNodeId, Node>,
    hitbox_nodes: HashMap<HitboxId, DispatchNodeId>,
    next: u64,
    root: Option<DispatchNodeId>,
    next_listener_order: u64,
}
impl DispatchTree {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn root(&mut self) -> DispatchNodeId {
        if let Some(root) = self.root {
            return root;
        }
        let root = self.new_node(None);
        self.root = Some(root);
        root
    }
    pub fn root_id(&self) -> Option<DispatchNodeId> {
        self.root
    }
    pub fn new_node(&mut self, parent: Option<DispatchNodeId>) -> DispatchNodeId {
        self.next = self.next.wrapping_add(1);
        let id = DispatchNodeId(self.next);
        self.nodes.insert(
            id,
            Node {
                parent,
                address: None,
                action_handlers: Vec::new(),
                capture_handlers: Vec::new(),
                input_handlers: Vec::new(),
            },
        );
        id
    }
    pub fn bind_hitbox(&mut self, hitbox: HitboxId, node: DispatchNodeId) -> bool {
        if !self.nodes.contains_key(&node) {
            return false;
        }
        self.hitbox_nodes.insert(hitbox, node);
        true
    }
    pub fn unbind_hitbox(&mut self, hitbox: HitboxId) -> bool {
        self.hitbox_nodes.remove(&hitbox).is_some()
    }
    pub fn on_action<A: Action>(
        &mut self,
        node: DispatchNodeId,
        mut handler: impl FnMut(&A) -> EventResult + 'static,
    ) -> bool {
        if !self.nodes.contains_key(&node) {
            return false;
        }
        self.next_listener_order = self.next_listener_order.wrapping_add(1);
        let Some(node) = self.nodes.get_mut(&node) else {
            return false;
        };
        node.action_handlers.push(ActionHandler {
            registration_order: self.next_listener_order,
            callback: Box::new(move |action| {
                action
                    .as_any()
                    .downcast_ref::<A>()
                    .map_or(EventResult::IGNORED, &mut handler)
            }),
        });
        true
    }
    pub fn on_input(
        &mut self,
        node: DispatchNodeId,
        handler: impl FnMut(&InputEvent) -> EventResult + 'static,
    ) -> bool {
        self.on_input_for(node, super::inspector::InputEventFamily::All, handler)
    }
    pub fn on_input_capture(
        &mut self,
        node: DispatchNodeId,
        handler: impl FnMut(&InputEvent) -> EventResult + 'static,
    ) -> bool {
        self.on_input_capture_for(node, super::inspector::InputEventFamily::All, handler)
    }
    pub fn on_input_for(
        &mut self,
        node: DispatchNodeId,
        family: super::inspector::InputEventFamily,
        handler: impl FnMut(&InputEvent) -> EventResult + 'static,
    ) -> bool {
        if !self.nodes.contains_key(&node) {
            return false;
        }
        self.next_listener_order = self.next_listener_order.wrapping_add(1);
        let Some(node) = self.nodes.get_mut(&node) else {
            return false;
        };
        node.input_handlers.push(InputHandler {
            family,
            phase: super::inspector::DispatchPhase::Bubble,
            registration_order: self.next_listener_order,
            callback: Box::new(handler),
        });
        true
    }
    pub fn on_input_capture_for(
        &mut self,
        node: DispatchNodeId,
        family: super::inspector::InputEventFamily,
        handler: impl FnMut(&InputEvent) -> EventResult + 'static,
    ) -> bool {
        if !self.nodes.contains_key(&node) {
            return false;
        }
        self.next_listener_order = self.next_listener_order.wrapping_add(1);
        let Some(node) = self.nodes.get_mut(&node) else {
            return false;
        };
        node.capture_handlers.push(InputHandler {
            family,
            phase: super::inspector::DispatchPhase::Capture,
            registration_order: self.next_listener_order,
            callback: Box::new(handler),
        });
        true
    }
    pub(crate) fn bind_address(&mut self, node: DispatchNodeId, address: InstanceKey) -> bool {
        let Some(node) = self.nodes.get_mut(&node) else {
            return false;
        };
        node.address = Some(address);
        true
    }
    pub(crate) fn node_for_hitbox(&self, hitbox: HitboxId) -> Option<DispatchNodeId> {
        self.hitbox_nodes.get(&hitbox).copied()
    }
    pub fn dispatch_action(&mut self, target: DispatchNodeId, action: &dyn Action) -> bool {
        for node_id in self.path(target) {
            let Some(node) = self.nodes.get_mut(&node_id) else {
                continue;
            };
            for handler in node.action_handlers.iter_mut().rev() {
                let result = (handler.callback)(action);
                if result.handled && !result.propagate {
                    return true;
                }
            }
        }
        false
    }
    pub fn dispatch_input(&mut self, target: HitboxId, event: &InputEvent) -> bool {
        let Some(node) = self.hitbox_nodes.get(&target).copied() else {
            return false;
        };
        let path = self.path(node);
        for node_id in path.iter().rev() {
            let Some(node) = self.nodes.get_mut(node_id) else {
                continue;
            };
            for handler in node.capture_handlers.iter_mut().rev() {
                if !handler.family.matches(event) {
                    continue;
                }
                let result = (handler.callback)(event);
                if result.handled && !result.propagate {
                    return true;
                }
            }
        }
        for node_id in path {
            let Some(node) = self.nodes.get_mut(&node_id) else {
                continue;
            };
            for handler in node.input_handlers.iter_mut().rev() {
                if !handler.family.matches(event) {
                    continue;
                }
                let result = (handler.callback)(event);
                if result.handled && !result.propagate {
                    return true;
                }
            }
        }
        false
    }
    fn path(&self, target: DispatchNodeId) -> Vec<DispatchNodeId> {
        let mut path = Vec::new();
        let mut current = Some(target);
        while let Some(id) = current {
            path.push(id);
            current = self.nodes.get(&id).and_then(|node| node.parent);
        }
        path
    }

    pub(crate) fn inspection_snapshot(&self) -> DispatchTreeSnapshot {
        let mut nodes: Vec<_> = self
            .nodes
            .iter()
            .map(|(id, node)| DispatchNodeInfo {
                id: *id,
                parent: node.parent,
                ancestry: self.path_from_root(*id),
                address: node.address,
                listeners: node
                    .capture_handlers
                    .iter()
                    .chain(node.input_handlers.iter())
                    .map(|listener| ListenerInfo {
                        family: listener.family,
                        phase: listener.phase,
                        registration_order: listener.registration_order,
                        handler_present: true,
                    })
                    .chain(node.action_handlers.iter().map(|handler| ListenerInfo {
                        family: super::inspector::InputEventFamily::Keyboard,
                        phase: super::inspector::DispatchPhase::Bubble,
                        registration_order: handler.registration_order,
                        handler_present: true,
                    }))
                    .collect(),
            })
            .collect();
        nodes.sort_unstable_by_key(|node| node.id);
        let mut hitbox_nodes: Vec<_> = self
            .hitbox_nodes
            .iter()
            .map(|(hitbox, node)| (*hitbox, *node))
            .collect();
        hitbox_nodes.sort_unstable_by_key(|(hitbox, _)| *hitbox);
        DispatchTreeSnapshot {
            nodes,
            hitbox_nodes,
        }
    }

    fn path_from_root(&self, target: DispatchNodeId) -> Vec<DispatchNodeId> {
        let mut path = self.path(target);
        path.reverse();
        path
    }

    pub(crate) fn node_for_address(&self, address: InstanceKey) -> Option<DispatchNodeId> {
        self.nodes
            .iter()
            .find_map(|(id, node)| (node.address == Some(address)).then_some(*id))
    }

    pub(crate) fn ancestry(&self, node: DispatchNodeId) -> Vec<DispatchNodeId> {
        self.path_from_root(node)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    crate::actions!(dispatch_test, [Activate]);
    #[test]
    fn actions_bubble_from_target_to_root_until_handled() {
        let mut tree = DispatchTree::new();
        let root = tree.root();
        let child = tree.new_node(Some(root));
        let hitbox = HitboxId::from_raw(100);
        assert!(tree.bind_hitbox(hitbox, child));
        let calls = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let child_calls = calls.clone();
        tree.on_action(child, move |_: &Activate| {
            child_calls.borrow_mut().push("child");
            EventResult::IGNORED
        });
        let root_calls = calls.clone();
        tree.on_action(root, move |_: &Activate| {
            root_calls.borrow_mut().push("root");
            EventResult::HANDLED
        });
        assert!(tree.dispatch_action(child, &Activate));
        assert!(!tree.dispatch_input(
            hitbox,
            &InputEvent::KeyUp(crate::window::KeyUpEvent {
                key: "x".into(),
                modifiers: crate::window::Modifiers::none(),
            })
        ));
        assert_eq!(&*calls.borrow(), &["child", "root"]);
    }
}
