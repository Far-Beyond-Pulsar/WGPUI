use super::hitbox::HitboxId;
use super::input::{EventResult, InputEvent};
use crate::action::Action;
use std::collections::HashMap;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DispatchNodeId(pub u64);
type Handler = Box<dyn FnMut(&dyn Action) -> EventResult>;
type InputHandler = Box<dyn FnMut(&InputEvent) -> EventResult>;
struct Node {
    parent: Option<DispatchNodeId>,
    action_handlers: Vec<Handler>,
    input_handlers: Vec<InputHandler>,
}
#[derive(Default)]
pub struct DispatchTree {
    nodes: HashMap<DispatchNodeId, Node>,
    next: u64,
    root: Option<DispatchNodeId>,
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
                action_handlers: Vec::new(),
                input_handlers: Vec::new(),
            },
        );
        id
    }
    pub fn on_action<A: Action>(
        &mut self,
        node: DispatchNodeId,
        mut handler: impl FnMut(&A) -> EventResult + 'static,
    ) -> bool {
        let Some(node) = self.nodes.get_mut(&node) else {
            return false;
        };
        node.action_handlers.push(Box::new(move |action| {
            action
                .as_any()
                .downcast_ref::<A>()
                .map_or(EventResult::IGNORED, &mut handler)
        }));
        true
    }
    pub fn on_input(
        &mut self,
        node: DispatchNodeId,
        handler: impl FnMut(&InputEvent) -> EventResult + 'static,
    ) -> bool {
        let Some(node) = self.nodes.get_mut(&node) else {
            return false;
        };
        node.input_handlers.push(Box::new(handler));
        true
    }
    pub fn dispatch_action(&mut self, target: DispatchNodeId, action: &dyn Action) -> bool {
        for node_id in self.path(target) {
            let Some(node) = self.nodes.get_mut(&node_id) else {
                continue;
            };
            for handler in node.action_handlers.iter_mut().rev() {
                let result = handler(action);
                if result.handled && !result.propagate {
                    return true;
                }
            }
        }
        false
    }
    pub fn dispatch_input(&mut self, target: HitboxId, event: &InputEvent) -> bool {
        for node_id in self.path(DispatchNodeId(target.as_raw())) {
            let Some(node) = self.nodes.get_mut(&node_id) else {
                continue;
            };
            for handler in node.input_handlers.iter_mut().rev() {
                let result = handler(event);
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
        assert_eq!(&*calls.borrow(), &["child", "root"]);
    }
}
