//! Retained window state and native input/action coordination.

pub mod animation;
pub mod close;
pub mod dispatch;
pub mod focus;
pub mod hitbox;
pub mod input;
pub mod keymap;
pub mod menu;
pub mod timer;

use crate::action::Action;
use crate::geometry::{Bounds, Pixels, Size, point};
use crate::reconcile::{ElementStateStore, StateKey, StateScope};
use std::time::{Duration, Instant};

pub use animation::{AnimationClock, AnimationScheduler, WindowTimers};
pub use close::CloseState;
pub use dispatch::{DispatchNodeId, DispatchTree};
pub use focus::{FocusHandle, FocusId, FocusManager, FocusTransition, Focusable};
pub use hitbox::{HitTestIndex, Hitbox, HitboxId};
pub use input::{
    ClickEvent, DragData, DragHoverEvent, DropEvent, EventResult, FocusEvent, InputEvent, KeyDownEvent, KeyUpEvent, KeyboardButton,
    KeyboardClickEvent, Modifiers, MouseButton, MouseButtonState, MouseClickEvent, MouseDownEvent,
    MouseMoveEvent, MouseUpEvent, ScrollWheelEvent,
};
pub use keymap::{KeyBinding, KeyParseError, Keymap, Keystroke};
pub use menu::{Menu, MenuItem};
pub use timer::{TimerHandle, TimerId, TimerState};

#[derive(Clone, Debug)]
pub struct WindowOptions {
    pub title: String,
    pub width: u32,
    pub height: u32,
    pub resizable: bool,
    pub window_bounds: Option<crate::geometry::WindowBounds>,
    pub focus: bool,
    pub show: bool,
}

impl Default for WindowOptions {
    fn default() -> Self {
        Self {
            title: "WGPUI".to_string(),
            width: 800,
            height: 600,
            resizable: true,
            window_bounds: None,
            focus: true,
            show: true,
        }
    }
}

pub struct Window {
    bounds: Bounds<Pixels>,
    active: bool,
    state: ElementStateStore,
    frame: u64,
    focus: FocusManager,
    hit_test: HitTestIndex,
    dispatch: DispatchTree,
    keymap: Keymap,
    hovered: Option<HitboxId>,
    pressed: Option<(HitboxId, MouseDownEvent)>,
    focus_hitboxes: std::collections::HashMap<HitboxId, FocusId>,
    timers: WindowTimers,
    close: CloseState,
    interaction_revision: u64,
}
impl Default for Window {
    fn default() -> Self {
        Self::new()
    }
}
impl Window {
    pub fn new() -> Self {
        let mut dispatch = DispatchTree::new();
        dispatch.root();
        Self {
            bounds: Bounds::new(point(Pixels::ZERO, Pixels::ZERO), Size::default()),
            active: false,
            state: ElementStateStore::new(),
            frame: 0,
            focus: FocusManager::default(),
            hit_test: HitTestIndex::default(),
            dispatch,
            keymap: Keymap::default(),
            hovered: None,
            pressed: None,
            focus_hitboxes: std::collections::HashMap::new(),
            timers: WindowTimers::default(),
            close: CloseState::default(),
            interaction_revision: 0,
        }
    }
    pub fn next_frame(&mut self) {
        self.frame = self.frame.wrapping_add(1);
    }
    pub fn bounds(&self) -> Bounds<Pixels> {
        self.bounds
    }
    pub fn resize(&mut self, size: Size<Pixels>) {
        self.bounds.size = size;
    }
    pub fn set_bounds(&mut self, bounds: Bounds<Pixels>) {
        self.bounds = bounds;
    }
    pub fn activate(&mut self) {
        self.active = true;
    }
    pub fn deactivate(&mut self) {
        self.active = false;
    }
    pub fn is_active(&self) -> bool {
        self.active
    }
    pub fn use_state<T: 'static, R>(
        &mut self,
        scope: StateScope,
        initialise: impl FnOnce() -> T,
        access: impl FnOnce(&mut T) -> R,
    ) -> Option<R> {
        self.state
            .with_state(StateKey::new::<T>(scope), self.frame, initialise, access)
    }
    pub fn state_len(&self) -> usize {
        self.state.len()
    }
    pub fn focus_manager(&self) -> &FocusManager {
        &self.focus
    }
    pub fn register_focus_handle(&mut self, handle: FocusHandle) {
        self.focus
            .register_ordered(&handle, handle.tab_index_value(), None);
    }
    pub fn register_focus_handle_ordered(&mut self, handle: FocusHandle, order: u64) {
        self.focus
            .register_ordered(&handle, handle.tab_index_value(), Some(order));
    }
    pub fn unregister_focus_handle(&mut self, handle: FocusHandle) {
        self.focus.unregister(handle.id());
    }
    pub fn retain_focus_handles(&mut self, handles: impl IntoIterator<Item = FocusHandle>) {
        self.focus.retain(handles.into_iter().map(FocusHandle::id));
    }
    pub fn resolve_action(&self, event: &KeyDownEvent) -> Option<Box<dyn Action>> {
        self.keymap.resolve(event, None).map(Action::boxed_clone)
    }
    pub fn focus(&mut self, handle: &FocusHandle) -> bool {
        self.focus.register(handle, handle.tab_index_value());
        self.focus_id(handle.id(), true)
    }
    pub fn focus_id(&mut self, id: FocusId, visible: bool) -> bool {
        let changed = self.focus.focus(id, visible);
        if changed {
            self.interaction_revision = self.interaction_revision.wrapping_add(1);
        }
        changed
    }
    pub fn blur(&mut self) -> bool {
        let changed = self.focus.blur();
        if changed {
            self.interaction_revision = self.interaction_revision.wrapping_add(1);
        }
        changed
    }
    pub fn is_focused(&self, handle: &FocusHandle) -> bool {
        self.focus.focused() == Some(handle.id())
    }
    pub fn focused(&self) -> Option<FocusId> {
        self.focus.focused()
    }
    pub fn focus_next(&mut self) -> Option<FocusId> {
        let focused = self.focus.next(false);
        if focused.is_some() {
            self.interaction_revision = self.interaction_revision.wrapping_add(1);
        }
        focused
    }
    pub fn focus_previous(&mut self) -> Option<FocusId> {
        let focused = self.focus.next(true);
        if focused.is_some() {
            self.interaction_revision = self.interaction_revision.wrapping_add(1);
        }
        focused
    }

    pub fn take_focus_transition(&mut self) -> Option<FocusTransition> {
        self.focus.take_transition()
    }
    pub fn interaction_revision(&self) -> u64 {
        self.interaction_revision
    }
    pub fn hit_test(&mut self) -> &mut HitTestIndex {
        &mut self.hit_test
    }
    pub fn register_hitbox(&mut self, hitbox: Hitbox, node: DispatchNodeId) {
        self.hit_test
            .insert_with_id(hitbox.id, hitbox.bounds, hitbox.z_index);
        self.dispatch.bind_hitbox(hitbox.id, node);
        self.hit_test
            .set_hit_testable(hitbox.id, hitbox.hit_testable);
    }
    pub fn register_focus_hitbox(
        &mut self,
        hitbox: Hitbox,
        node: DispatchNodeId,
        focus: FocusHandle,
    ) {
        self.register_hitbox(hitbox, node);
        self.register_focus_handle(focus);
        self.focus_hitboxes.insert(hitbox.id, focus.id());
    }
    pub fn unregister_hitbox(&mut self, id: HitboxId) {
        self.hit_test.remove(id);
        self.dispatch.unbind_hitbox(id);
        self.focus_hitboxes.remove(&id);
        if self.hovered == Some(id) {
            self.hovered = None;
        }
        if self.pressed.is_some_and(|(pressed, _)| pressed == id) {
            self.pressed = None;
        }
    }
    pub fn dispatch_tree(&mut self) -> &mut DispatchTree {
        &mut self.dispatch
    }
    pub fn bind_key(&mut self, binding: KeyBinding) {
        self.keymap.add(binding);
    }
    pub fn bind_keys(&mut self, bindings: impl IntoIterator<Item = KeyBinding>) {
        self.keymap.add_all(bindings);
    }
    pub fn keymap(&self) -> &Keymap {
        &self.keymap
    }
    pub fn hovered_hitbox(&self) -> Option<HitboxId> {
        self.hovered
    }
    pub fn clear_hover(&mut self) -> Option<HitboxId> {
        self.hovered.take()
    }
    pub fn dispatch_action(&mut self, target: DispatchNodeId, action: &dyn Action) -> bool {
        self.dispatch.dispatch_action(target, action)
    }
    pub fn handle_input(&mut self, event: InputEvent) -> bool {
        match &event {
            InputEvent::KeyDown(key) => {
                if key.key.eq_ignore_ascii_case("tab") {
                    if key.modifiers.shift {
                        return self.focus_previous().is_some();
                    }
                    return self.focus_next().is_some();
                }
                let action = self.keymap.resolve(key, None).map(Action::boxed_clone);
                action.is_some_and(|action| {
                    self.dispatch
                        .root_id()
                        .is_some_and(|root| self.dispatch_action(root, &*action))
                })
            }
            InputEvent::MouseMove(mouse) => {
                let hit = self
                    .hit_test
                    .hit_test([mouse.position[0].value(), mouse.position[1].value()]);
                let previous = self.hovered;
                self.hovered = hit;
                let mut handled = false;
                if previous != hit {
                    self.interaction_revision = self.interaction_revision.wrapping_add(1);
                    if let Some(id) = previous {
                        handled |= self
                            .dispatch
                            .dispatch_input(id, &InputEvent::MouseLeave(*mouse));
                    }
                    if let Some(id) = hit {
                        handled |= self
                            .dispatch
                            .dispatch_input(id, &InputEvent::MouseEnter(*mouse));
                    }
                }
                handled |= hit.is_some_and(|id| self.dispatch.dispatch_input(id, &event));
                handled
            }
            InputEvent::MouseDown(mouse) => {
                let hit = self
                    .hit_test
                    .hit_test([mouse.position[0].value(), mouse.position[1].value()]);
                self.pressed = hit.map(|id| (id, *mouse));
                let mut handled = false;
                if let Some(id) = hit {
                    if mouse.is_focusing()
                        && let Some(focus_id) = self.focus_hitboxes.get(&id).copied()
                        && self.focus_id(focus_id, false)
                    {
                        self.interaction_revision = self.interaction_revision.wrapping_add(1);
                        handled = true;
                    }
                    handled |= self.dispatch.dispatch_input(id, &event);
                }
                handled
            }
            InputEvent::MouseUp(mouse) => {
                let hit = self
                    .hit_test
                    .hit_test([mouse.position[0].value(), mouse.position[1].value()]);
                let pressed = self.pressed.take();
                let Some((pressed_id, down)) = pressed else {
                    return false;
                };
                let mut handled = self.dispatch.dispatch_input(pressed_id, &event);
                if hit == Some(pressed_id) {
                    let click =
                        InputEvent::Click(ClickEvent::Mouse(MouseClickEvent { down, up: *mouse }));
                    handled |= self.dispatch.dispatch_input(pressed_id, &click);
                }
                handled
            }
            InputEvent::Scroll(scroll) => {
                let hit = self
                    .hit_test
                    .hit_test([scroll.position[0].value(), scroll.position[1].value()]);
                if hit.is_some() {
                    self.interaction_revision = self.interaction_revision.wrapping_add(1);
                }
                hit.is_some_and(|id| self.dispatch.dispatch_input(id, &event))
            }
            InputEvent::KeyUp(_)
            | InputEvent::MouseEnter(_)
            | InputEvent::MouseLeave(_)
            | InputEvent::Click(_)
            | InputEvent::Focus(_)
            | InputEvent::DragHover(_)
            | InputEvent::Drop(_) => false,
        }
    }
    pub fn schedule_timer(&mut self, delay: Duration) -> TimerHandle {
        self.timers.schedule(Instant::now(), delay)
    }
    pub fn schedule_timer_at(&mut self, now: Instant, delay: Duration) -> TimerHandle {
        self.timers.schedule(now, delay)
    }
    pub fn cancel_timer(&mut self, timer: TimerHandle) -> bool {
        self.timers.cancel(timer)
    }
    pub fn take_due_timers(&mut self, now: Instant) -> Vec<TimerId> {
        self.timers.due(now)
    }
    pub fn request_close(&mut self) {
        self.close.request();
    }
    pub fn prevent_close(&mut self) {
        self.close.prevent();
    }
    pub fn allow_close(&mut self) {
        self.close.allow();
    }
    pub fn should_close(&self) -> bool {
        self.close.should_close()
    }
    pub fn close_requested(&self) -> bool {
        self.close.requested()
    }
    pub fn clear_close_request(&mut self) {
        self.close.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{Rect, point, px, size};
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn window_creation_bounds_resize_activation_and_close_are_stateful() {
        let mut window = Window::new();
        assert_eq!(window.bounds(), Bounds::default());
        assert!(!window.is_active());
        assert!(!window.close_requested());

        let bounds = Bounds::new(point(px(12.0), px(24.0)), size(px(320.0), px(240.0)));
        window.set_bounds(bounds);
        window.resize(size(px(640.0), px(480.0)));
        window.activate();
        window.request_close();

        assert_eq!(
            window.bounds(),
            Bounds::new(bounds.origin, size(px(640.0), px(480.0)))
        );
        assert!(window.is_active());
        assert!(window.should_close());
    }

    fn hitbox(window: &mut Window, id: u64, node: DispatchNodeId) -> Hitbox {
        let value = Hitbox {
            id: HitboxId::from_raw(id),
            bounds: Rect::from_origin_size([0.0, 0.0], [100.0, 100.0]),
            z_index: 0,
            order: 0,
            hit_testable: true,
        };
        window.register_hitbox(value, node);
        value
    }

    #[test]
    fn mouse_lifecycle_emits_enter_leave_and_click_only_after_release_inside() {
        let mut window = Window::new();
        let root = window.dispatch_tree().root();
        let node = window.dispatch_tree().new_node(Some(root));
        let target = hitbox(&mut window, 1, node);
        let events = Rc::new(RefCell::new(Vec::new()));
        let observed = events.clone();
        assert!(window.dispatch_tree().on_input(node, move |event| {
            observed.borrow_mut().push(match event {
                InputEvent::MouseEnter(_) => "enter",
                InputEvent::MouseLeave(_) => "leave",
                InputEvent::MouseDown(_) => "down",
                InputEvent::MouseUp(_) => "up",
                InputEvent::Click(_) => "click",
                _ => "other",
            });
            EventResult::HANDLED
        }));
        let position = [Pixels(10.0), Pixels(10.0)];
        let movement = InputEvent::MouseMove(MouseMoveEvent {
            position,
            modifiers: Modifiers::none(),
            buttons: MouseButtonState::default(),
        });
        assert!(window.handle_input(movement));
        assert!(window.handle_input(InputEvent::MouseDown(MouseDownEvent {
            button: MouseButton::Left,
            position,
            modifiers: Modifiers::none(),
            click_count: 1,
        })));
        assert!(window.handle_input(InputEvent::MouseUp(MouseUpEvent {
            button: MouseButton::Left,
            position,
            modifiers: Modifiers::none(),
            click_count: 1,
        })));
        assert!(window.handle_input(InputEvent::MouseMove(MouseMoveEvent {
            position: [Pixels(200.0), Pixels(200.0)],
            modifiers: Modifiers::none(),
            buttons: MouseButtonState::default(),
        })));
        assert_eq!(
            &*events.borrow(),
            &["enter", "other", "down", "up", "click", "leave"]
        );
        assert_eq!(window.hovered_hitbox(), None);
        assert_eq!(target.id, HitboxId::from_raw(1));
    }

    #[test]
    fn capture_runs_from_root_and_can_cancel_bubbling() {
        let mut window = Window::new();
        let root = window.dispatch_tree().root();
        let child = window.dispatch_tree().new_node(Some(root));
        let _target = hitbox(&mut window, 2, child);
        let calls = Rc::new(RefCell::new(Vec::new()));
        let capture_calls = calls.clone();
        assert!(window.dispatch_tree().on_input_capture(root, move |_| {
            capture_calls.borrow_mut().push("capture-root");
            EventResult::HANDLED
        }));
        let child_calls = calls.clone();
        assert!(window.dispatch_tree().on_input(child, move |_| {
            child_calls.borrow_mut().push("bubble-child");
            EventResult::HANDLED
        }));
        assert!(window.dispatch_tree().dispatch_input(
            HitboxId::from_raw(2),
            &InputEvent::KeyUp(KeyUpEvent {
                key: "x".into(),
                modifiers: Modifiers::none()
            })
        ));
        assert_eq!(&*calls.borrow(), &["capture-root"]);
    }

    #[test]
    fn focused_hitboxes_focus_on_mouse_down_and_tab_wraps() {
        let mut window = Window::new();
        let root = window.dispatch_tree().root();
        let first = FocusHandle::new().with_tab_index(0);
        let second = FocusHandle::new().with_tab_index(1);
        let first_node = window.dispatch_tree().new_node(Some(root));
        let second_node = window.dispatch_tree().new_node(Some(root));
        window.register_focus_hitbox(
            Hitbox {
                id: HitboxId::from_raw(3),
                bounds: Rect::from_origin_size([0.0, 0.0], [20.0, 20.0]),
                z_index: 0,
                order: 0,
                hit_testable: true,
            },
            first_node,
            first,
        );
        window.register_focus_hitbox(
            Hitbox {
                id: HitboxId::from_raw(4),
                bounds: Rect::from_origin_size([30.0, 0.0], [20.0, 20.0]),
                z_index: 0,
                order: 0,
                hit_testable: true,
            },
            second_node,
            second,
        );
        assert!(window.handle_input(InputEvent::MouseDown(MouseDownEvent {
            button: MouseButton::Left,
            position: [Pixels(5.0), Pixels(5.0)],
            modifiers: Modifiers::none(),
            click_count: 1,
        })));
        assert_eq!(window.focused(), Some(first.id()));
        assert!(window.handle_input(InputEvent::KeyDown(KeyDownEvent::new(
            "Tab",
            Modifiers::none()
        ))));
        assert_eq!(window.focused(), Some(second.id()));
        assert!(window.handle_input(InputEvent::KeyDown(KeyDownEvent::new(
            "Tab",
            Modifiers::none()
        ))));
        assert_eq!(window.focused(), Some(first.id()));
    }

    #[test]
    fn scroll_events_are_dispatched_to_the_hit_target() {
        let mut window = Window::new();
        let root = window.dispatch_tree().root();
        let node = window.dispatch_tree().new_node(Some(root));
        let _target = hitbox(&mut window, 5, node);
        assert!(window.dispatch_tree().on_input(node, |event| {
            if matches!(event, InputEvent::Scroll(_)) {
                EventResult::HANDLED
            } else {
                EventResult::IGNORED
            }
        }));
        assert!(window.handle_input(InputEvent::Scroll(ScrollWheelEvent {
            position: [Pixels(4.0), Pixels(4.0)],
            delta: [0.0, -20.0],
            modifiers: Modifiers::none(),
        })));
    }
}
