//! Retained window state and native input/action coordination.

pub mod animation;
pub mod close;
pub mod dispatch;
pub mod focus;
pub mod hitbox;
pub mod input;
pub mod keymap;
pub mod menu;
pub mod prompts;
pub mod timer;

use crate::action::Action;
use crate::reconcile::{ElementStateStore, StateKey, StateScope};
use std::time::{Duration, Instant};

pub use animation::{AnimationScheduler, WindowTimers};
pub use close::CloseState;
pub use dispatch::{DispatchNodeId, DispatchTree};
pub use focus::{FocusHandle, FocusId, FocusManager, FocusTransition, Focusable};
pub use hitbox::{HitTestIndex, Hitbox, HitboxId};
pub use input::{
    ClickEvent, EventResult, InputEvent, KeyDownEvent, KeyUpEvent, KeyboardButton,
    KeyboardClickEvent, Modifiers, MouseButton, MouseButtonState, MouseClickEvent, MouseDownEvent,
    MouseMoveEvent, MouseUpEvent, ScrollWheelEvent,
};
pub use keymap::{KeyBinding, KeyParseError, Keymap, Keystroke};
pub use menu::{Menu, MenuItem};
pub use timer::{TimerHandle, TimerId, TimerState};

pub struct Window {
    state: ElementStateStore,
    frame: u64,
    focus: FocusManager,
    hit_test: HitTestIndex,
    dispatch: DispatchTree,
    keymap: Keymap,
    hovered: Option<HitboxId>,
    pressed: Option<(HitboxId, MouseDownEvent)>,
    timers: WindowTimers,
    close: CloseState,
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
            state: ElementStateStore::new(),
            frame: 0,
            focus: FocusManager::default(),
            hit_test: HitTestIndex::default(),
            dispatch,
            keymap: Keymap::default(),
            hovered: None,
            pressed: None,
            timers: WindowTimers::default(),
            close: CloseState::default(),
        }
    }
    pub fn next_frame(&mut self) {
        self.frame = self.frame.wrapping_add(1);
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
    pub fn focus(&mut self, handle: &FocusHandle) -> bool {
        self.focus.register(handle, handle.tab_index_value());
        self.focus.focus(handle.id(), true)
    }
    pub fn blur(&mut self) -> bool {
        self.focus.blur()
    }
    pub fn is_focused(&self, handle: &FocusHandle) -> bool {
        self.focus.focused() == Some(handle.id())
    }
    pub fn focused(&self) -> Option<FocusId> {
        self.focus.focused()
    }
    pub fn focus_next(&mut self) -> Option<FocusId> {
        self.focus.next(false)
    }
    pub fn focus_previous(&mut self) -> Option<FocusId> {
        self.focus.next(true)
    }
    pub fn take_focus_transition(&mut self) -> Option<FocusTransition> {
        self.focus.take_transition()
    }
    pub fn hit_test(&mut self) -> &mut HitTestIndex {
        &mut self.hit_test
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
                self.hovered = hit;
                hit.is_some_and(|id| self.dispatch.dispatch_input(id, &event))
            }
            InputEvent::MouseDown(mouse) => {
                let hit = self
                    .hit_test
                    .hit_test([mouse.position[0].value(), mouse.position[1].value()]);
                self.pressed = hit.map(|id| (id, *mouse));
                hit.is_some_and(|id| self.dispatch.dispatch_input(id, &event))
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
            InputEvent::KeyUp(_) | InputEvent::Scroll(_) | InputEvent::Click(_) => false,
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
