//! Retained interaction state and event callbacks for interactive elements.
//!
//! The state machine is intentionally independent of layout and rendering.
//! The frame builder registers its hitbox and dispatch node, then feeds input
//! into this object; hover/active transitions are therefore observable by the
//! style resolver and do not require repainting unchanged siblings.

use wgpui_core::app::App;
use std::sync::Arc;
use wgpui_core::action::Action;
use wgpui_core::window::{
    ClickEvent, DragData, EventResult, FocusEvent, ImeEvent, InputEvent, KeyDownEvent,
    KeyUpEvent, ModifiersChangedEvent, MouseButton, MouseMoveEvent, MouseUpEvent,
    ScrollWheelEvent, TextInputEvent, Window,
};
use wgpui_core::reconcile::description::DescriptionInteraction;

type ClickHandler = Box<dyn FnMut(&ClickEvent, &mut Window, &mut App) -> EventResult>;
type MouseDownHandler = Box<dyn FnMut(&InputEvent, &mut Window, &mut App) -> EventResult>;
type MouseUpHandler = Box<dyn FnMut(&MouseUpEvent, &mut Window, &mut App) -> EventResult>;
type MouseMoveHandler = Box<dyn FnMut(&MouseMoveEvent, &mut Window, &mut App) -> EventResult>;
type MouseBoundaryHandler = Box<dyn FnMut(&mut Window, &mut App) -> EventResult>;
type HoverHandler = Box<dyn FnMut(bool, &mut Window, &mut App) -> EventResult>;
type ScrollHandler = Box<dyn FnMut(&ScrollWheelEvent, &mut Window, &mut App) -> EventResult>;
type KeyDownHandler = Box<dyn FnMut(&KeyDownEvent, &mut Window, &mut App) -> EventResult>;
type KeyUpHandler = Box<dyn FnMut(&KeyUpEvent, &mut Window, &mut App) -> EventResult>;
type TextInputHandler = Box<dyn FnMut(&TextInputEvent, &mut Window, &mut App) -> EventResult>;
type ImeHandler = Box<dyn FnMut(&ImeEvent, &mut Window, &mut App) -> EventResult>;
type ModifiersHandler = Box<dyn FnMut(&ModifiersChangedEvent, &mut Window, &mut App) -> EventResult>;
type ActionHandler = Box<dyn FnMut(&dyn Action, &mut Window, &mut App) -> EventResult>;
type DragStartHandler = Box<dyn FnMut(&DragData, [wgpui_core::boundary::Pixels; 2], &mut Window, &mut App)>;
type DragHoverHandler = Box<dyn FnMut(bool, &DragData, &mut Window, &mut App) -> EventResult>;
type DropHandler = Box<dyn FnMut(&DragData, &mut Window, &mut App) -> EventResult>;

pub trait IntoEventResult {
    fn into_event_result(self) -> EventResult;
}

impl IntoEventResult for () {
    fn into_event_result(self) -> EventResult {
        EventResult::HANDLED
    }
}

impl IntoEventResult for EventResult {
    fn into_event_result(self) -> EventResult {
        self
    }
}

fn merge_result(result: &mut EventResult, current: EventResult) {
    if current.handled {
        result.handled = true;
        result.propagate = current.propagate;
    } else if !result.handled {
        result.propagate |= current.propagate;
    }
}

#[derive(Default)]
pub struct InteractionState {
    hovered: bool,
    active: bool,
    focused: bool,
    focus_visible: bool,
    click: Vec<ClickHandler>,
    mouse_down: Vec<(MouseButton, MouseDownHandler)>,
    mouse_up: Vec<(MouseButton, MouseUpHandler)>,
    mouse_move: Vec<MouseMoveHandler>,
    mouse_enter: Vec<MouseBoundaryHandler>,
    mouse_leave: Vec<MouseBoundaryHandler>,
    hover: Vec<HoverHandler>,
    scroll: Vec<ScrollHandler>,
    key_down: Vec<KeyDownHandler>,
    key_up: Vec<KeyUpHandler>,
    text_input: Vec<TextInputHandler>,
    ime: Vec<ImeHandler>,
    modifiers_changed: Vec<ModifiersHandler>,
    actions: Vec<ActionHandler>,
    drag: Option<(DragData, DragStartHandler)>,
    drag_hover: Vec<(std::any::TypeId, DragHoverHandler)>,
    drop: Vec<(std::any::TypeId, DropHandler)>,
}

impl InteractionState {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn is_hovered(&self) -> bool {
        self.hovered
    }
    pub fn is_active(&self) -> bool {
        self.active
    }
    pub fn is_focused(&self) -> bool {
        self.focused
    }
    pub fn is_focus_visible(&self) -> bool {
        self.focused && self.focus_visible
    }
    pub fn set_focused(&mut self, focused: bool) -> bool {
        let changed = self.focused != focused;
        self.focused = focused;
        changed
    }
    pub fn on_click<R: IntoEventResult + 'static>(
        &mut self,
        mut handler: impl FnMut(&ClickEvent, &mut Window, &mut App) -> R + 'static,
    ) {
        self.click.push(Box::new(move |event, window, app| {
            handler(event, window, app).into_event_result()
        }));
    }
    pub fn on_mouse_down<R: IntoEventResult + 'static>(
        &mut self,
        button: MouseButton,
        mut handler: impl FnMut(&InputEvent, &mut Window, &mut App) -> R + 'static,
    ) {
        self.mouse_down.push((
            button,
            Box::new(move |event, window, app| handler(event, window, app).into_event_result()),
        ));
    }
    pub fn on_hover<R: IntoEventResult + 'static>(
        &mut self,
        mut handler: impl FnMut(bool, &mut Window, &mut App) -> R + 'static,
    ) {
        self.hover.push(Box::new(move |hovered, window, app| {
            handler(hovered, window, app).into_event_result()
        }));
    }
    pub fn on_mouse_up<R: IntoEventResult + 'static>(
        &mut self,
        button: MouseButton,
        mut handler: impl FnMut(&MouseUpEvent, &mut Window, &mut App) -> R + 'static,
    ) {
        self.mouse_up.push((button, Box::new(move |event, window, app| {
            handler(event, window, app).into_event_result()
        })));
    }
    pub fn on_mouse_move<R: IntoEventResult + 'static>(
        &mut self,
        mut handler: impl FnMut(&MouseMoveEvent, &mut Window, &mut App) -> R + 'static,
    ) {
        self.mouse_move.push(Box::new(move |event, window, app| {
            handler(event, window, app).into_event_result()
        }));
    }
    pub fn on_mouse_enter<R: IntoEventResult + 'static>(
        &mut self,
        mut handler: impl FnMut(&mut Window, &mut App) -> R + 'static,
    ) {
        self.mouse_enter.push(Box::new(move |window, app| {
            handler(window, app).into_event_result()
        }));
    }
    pub fn on_mouse_leave<R: IntoEventResult + 'static>(
        &mut self,
        mut handler: impl FnMut(&mut Window, &mut App) -> R + 'static,
    ) {
        self.mouse_leave.push(Box::new(move |window, app| {
            handler(window, app).into_event_result()
        }));
    }
    pub fn on_action<A: Action, R: IntoEventResult + 'static>(
        &mut self,
        mut handler: impl FnMut(&A, &mut Window, &mut App) -> R + 'static,
    ) {
        self.actions.push(Box::new(move |action, window, app| {
            action.as_any().downcast_ref::<A>().map_or(EventResult::IGNORED, |action| {
                handler(action, window, app).into_event_result()
            })
        }));
    }
    pub fn on_drag<D: 'static, R: 'static>(
        &mut self,
        data: D,
        mut handler: impl FnMut(&D, [wgpui_core::boundary::Pixels; 2], &mut Window, &mut App) -> R
            + 'static,
    ) {
        let data = Arc::new(data);
        self.drag = Some((
            DragData::new_arc(data),
            Box::new(move |value, position, window, app| {
                if let Some(value) = value.downcast_ref::<D>() {
                    handler(value, position, window, app);
                }
            }),
        ));
    }
    pub fn on_drag_hover<D: 'static, R: IntoEventResult + 'static>(
        &mut self,
        mut handler: impl FnMut(bool, &mut Window, &mut App) -> R + 'static,
    ) {
        self.drag_hover.push((std::any::TypeId::of::<D>(), Box::new(move |hovered, data, window, app| {
            if data.downcast_ref::<D>().is_some() {
                handler(hovered, window, app).into_event_result()
            } else {
                EventResult::IGNORED
            }
        })));
    }
    pub fn on_drop<D: 'static, R: IntoEventResult + 'static>(
        &mut self,
        mut handler: impl FnMut(&D, &mut Window, &mut App) -> R + 'static,
    ) {
        self.drop.push((std::any::TypeId::of::<D>(), Box::new(move |data, window, app| {
            data.downcast_ref::<D>().map_or(EventResult::IGNORED, |data| {
                handler(data, window, app).into_event_result()
            })
        })));
    }
    pub fn on_scroll<R: IntoEventResult + 'static>(
        &mut self,
        mut handler: impl FnMut(&ScrollWheelEvent, &mut Window, &mut App) -> R + 'static,
    ) {
        self.scroll.push(Box::new(move |event, window, app| {
            handler(event, window, app).into_event_result()
        }));
    }
    pub fn on_key_down<R: IntoEventResult + 'static>(
        &mut self,
        mut handler: impl FnMut(&KeyDownEvent, &mut Window, &mut App) -> R + 'static,
    ) {
        self.key_down.push(Box::new(move |event, window, app| {
            handler(event, window, app).into_event_result()
        }));
    }
    pub fn on_key_up<R: IntoEventResult + 'static>(
        &mut self,
        mut handler: impl FnMut(&KeyUpEvent, &mut Window, &mut App) -> R + 'static,
    ) {
        self.key_up.push(Box::new(move |event, window, app| {
            handler(event, window, app).into_event_result()
        }));
    }
    pub fn on_text_input<R: IntoEventResult + 'static>(
        &mut self,
        mut handler: impl FnMut(&TextInputEvent, &mut Window, &mut App) -> R + 'static,
    ) {
        self.text_input.push(Box::new(move |event, window, app| {
            handler(event, window, app).into_event_result()
        }));
    }
    pub fn on_ime<R: IntoEventResult + 'static>(
        &mut self,
        mut handler: impl FnMut(&ImeEvent, &mut Window, &mut App) -> R + 'static,
    ) {
        self.ime.push(Box::new(move |event, window, app| {
            handler(event, window, app).into_event_result()
        }));
    }
    pub fn on_modifiers_changed<R: IntoEventResult + 'static>(
        &mut self,
        mut handler: impl FnMut(&ModifiersChangedEvent, &mut Window, &mut App) -> R + 'static,
    ) {
        self.modifiers_changed.push(Box::new(move |event, window, app| {
            handler(event, window, app).into_event_result()
        }));
    }
    pub fn update_hover(
        &mut self,
        hovered: bool,
        window: &mut Window,
        app: &mut App,
    ) -> EventResult {
        if self.hovered == hovered {
            return EventResult::IGNORED;
        }
        self.hovered = hovered;
        let mut result = EventResult::IGNORED;
        for handler in &mut self.hover {
            let current = handler(hovered, window, app);
            merge_result(&mut result, current);
            if !result.propagate {
                break;
            }
        }
        result
    }
    pub fn handle_input(
        &mut self,
        event: &InputEvent,
        window: &mut Window,
        app: &mut App,
    ) -> EventResult {
        match event {
            InputEvent::ModifiersChanged(event) => dispatch_handlers(
                &mut self.modifiers_changed,
                event,
                window,
                app,
            ),
            InputEvent::KeyDown(event) => dispatch_handlers(&mut self.key_down, event, window, app),
            InputEvent::KeyUp(event) => dispatch_handlers(&mut self.key_up, event, window, app),
            InputEvent::TextInput(event) => {
                dispatch_handlers(&mut self.text_input, event, window, app)
            }
            InputEvent::Ime(event) => dispatch_handlers(&mut self.ime, event, window, app),
            InputEvent::MouseLeave(_) => {
                let mut result = self.update_hover(false, window, app);
                for handler in &mut self.mouse_leave {
                    if !result.propagate {
                        break;
                    }
                    merge_result(&mut result, handler(window, app));
                    if !result.propagate {
                        break;
                    }
                }
                result
            }
            InputEvent::MouseDown(mouse) => {
                let changed = self.active != (mouse.button == MouseButton::Left);
                self.active = mouse.button == MouseButton::Left;
                let mut result = EventResult::IGNORED;
                for (button, handler) in &mut self.mouse_down {
                    if *button == mouse.button {
                        let current = handler(event, window, app);
                        merge_result(&mut result, current);
                        if !result.propagate {
                            break;
                        }
                    }
                }
                if changed {
                    result.handled = true;
                }
                result
            }
            InputEvent::MouseUp(mouse) => {
                let changed = self.active;
                self.active = false;
                let mut result = if changed {
                    EventResult {
                        handled: true,
                        propagate: true,
                    }
                } else {
                    EventResult::IGNORED
                };
                for (button, handler) in &mut self.mouse_up {
                    if *button == mouse.button {
                        merge_result(&mut result, handler(mouse, window, app));
                        if !result.propagate {
                            break;
                        }
                    }
                }
                result
            }
            InputEvent::MouseMove(mouse) => {
                let mut result = EventResult::IGNORED;
                for handler in &mut self.mouse_move {
                    merge_result(&mut result, handler(mouse, window, app));
                    if !result.propagate {
                        break;
                    }
                }
                result
            }
            InputEvent::MouseEnter(_) => {
                let mut result = self.update_hover(true, window, app);
                for handler in &mut self.mouse_enter {
                    if !result.propagate {
                        break;
                    }
                    merge_result(&mut result, handler(window, app));
                    if !result.propagate {
                        break;
                    }
                }
                result
            }
            InputEvent::Click(click) => {
                let mut result = EventResult::IGNORED;
                for handler in &mut self.click {
                    let current = handler(click, window, app);
                    merge_result(&mut result, current);
                    if !result.propagate {
                        break;
                    }
                }
                result
            }
            InputEvent::Scroll(scroll) => {
                let mut result = EventResult::IGNORED;
                for handler in &mut self.scroll {
                    let current = handler(scroll, window, app);
                    merge_result(&mut result, current);
                    if !result.propagate {
                        break;
                    }
                }
                result
            }
            InputEvent::Focus(FocusEvent { focused, visible }) => {
                self.focused = *focused;
                self.focus_visible = *visible;
                EventResult::HANDLED
            }
            InputEvent::DragHover(event) => {
                let mut result = EventResult::IGNORED;
                for (type_id, handler) in &mut self.drag_hover {
                    if *type_id == event.data.type_id() {
                        merge_result(&mut result, handler(event.hovered, &event.data, window, app));
                        if !result.propagate {
                            break;
                        }
                    }
                }
                result
            }
            InputEvent::Drop(event) => {
                let mut result = EventResult::IGNORED;
                for (type_id, handler) in &mut self.drop {
                    if *type_id == event.data.type_id() {
                        merge_result(&mut result, handler(&event.data, window, app));
                        if !result.propagate {
                            break;
                        }
                    }
                }
                result
            }
        }
    }

    pub fn into_description_interaction(
        self,
        focus: Option<wgpui_core::window::FocusHandle>,
    ) -> Option<DescriptionInteraction> {
        if self.click.is_empty()
            && self.mouse_down.is_empty()
            && self.mouse_up.is_empty()
            && self.mouse_move.is_empty()
            && self.mouse_enter.is_empty()
            && self.mouse_leave.is_empty()
            && self.hover.is_empty()
            && self.scroll.is_empty()
            && self.key_down.is_empty()
            && self.key_up.is_empty()
            && self.text_input.is_empty()
            && self.ime.is_empty()
            && self.modifiers_changed.is_empty()
            && self.actions.is_empty()
            && self.drag.is_none()
            && self.drag_hover.is_empty()
            && self.drop.is_empty()
            && focus.is_none()
        {
            return None;
        }
        let mut state = self;
        let actions = std::mem::take(&mut state.actions);
        let drag = state.drag.take();
        let drag_hover = std::mem::take(&mut state.drag_hover);
        let drop = std::mem::take(&mut state.drop);
        let mut interaction = DescriptionInteraction::new(move |event, window, app| {
            state.handle_input(event, window, app)
        });
        for action in actions {
            interaction = interaction.with_action_handler(action);
        }
        if let Some(focus) = focus {
            interaction = interaction.with_focus(focus);
        }
        if let Some((data, mut callback)) = drag {
            interaction = interaction.with_drag_source(data, move |data, window, app| {
                callback(data, data.position, window, app);
            });
        }
        for (_, mut callback) in drag_hover {
            interaction = interaction.with_drag_hover_handler(move |hovered, data, window, app| {
                callback(hovered, data, window, app)
            });
        }
        for (_, mut callback) in drop {
            interaction = interaction.with_drop_handler(move |data, window, app| {
                callback(data, window, app)
            });
        }
        Some(interaction)
    }
}

type EventHandler<T> = Box<dyn FnMut(&T, &mut Window, &mut App) -> EventResult>;

fn dispatch_handlers<T>(
    handlers: &mut [EventHandler<T>],
    event: &T,
    window: &mut Window,
    app: &mut App,
) -> EventResult {
    let mut result = EventResult::IGNORED;
    for handler in handlers {
        merge_result(&mut result, handler(event, window, app));
        if !result.propagate {
            break;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;
    use wgpui_core::boundary::Pixels;
    use wgpui_core::window::{Modifiers, MouseDownEvent};

    wgpui_core::actions!(events_test, [Activate]);

    #[test]
    fn hover_is_edge_triggered_and_mouse_down_is_button_specific() {
        let mut state = InteractionState::new();
        let enters = Rc::new(Cell::new(0));
        let observed_enters = enters.clone();
        state.on_hover(move |value, _, _| {
            if value {
                observed_enters.set(observed_enters.get() + 1);
            }
            EventResult::HANDLED
        });
        assert!(
            state
                .update_hover(true, &mut Window::new(), &mut App::create())
                .handled
        );
        assert!(
            !state
                .update_hover(true, &mut Window::new(), &mut App::create())
                .handled
        );
        assert_eq!(enters.get(), 1);
        let downs = Rc::new(Cell::new(0));
        let observed_downs = downs.clone();
        state.on_mouse_down(MouseButton::Left, move |_, _, _| {
            observed_downs.set(observed_downs.get() + 1);
            EventResult::HANDLED
        });
        let event = InputEvent::MouseDown(MouseDownEvent {
            button: MouseButton::Right,
            position: [Pixels(0.0); 2],
            modifiers: Modifiers::none(),
            click_count: 1,
        });
        assert!(
            !state
                .handle_input(&event, &mut Window::new(), &mut App::create())
                .handled
        );
        assert_eq!(downs.get(), 0);
    }

    #[test]
    fn mouse_boundary_move_and_release_callbacks_receive_their_event_family() {
        let mut state = InteractionState::new();
        let calls = Rc::new(Cell::new(0));
        let observed_calls = calls.clone();
        state.on_mouse_enter(move |_, _| {
            observed_calls.set(observed_calls.get() + 1);
        });
        let observed_calls = calls.clone();
        state.on_mouse_move(move |event, _, _| {
            if event.position == [Pixels(3.0), Pixels(4.0)] {
                observed_calls.set(observed_calls.get() + 1);
            }
        });
        let observed_calls = calls.clone();
        state.on_mouse_up(MouseButton::Left, move |event, _, _| {
            if event.click_count == 2 {
                observed_calls.set(observed_calls.get() + 1);
            }
        });
        let move_event = MouseMoveEvent {
            position: [Pixels(3.0), Pixels(4.0)],
            modifiers: Modifiers::none(),
            buttons: Default::default(),
        };
        state.handle_input(
            &InputEvent::MouseEnter(move_event),
            &mut Window::new(),
            &mut App::create(),
        );
        state.handle_input(
            &InputEvent::MouseMove(move_event),
            &mut Window::new(),
            &mut App::create(),
        );
        state.handle_input(
            &InputEvent::MouseDown(MouseDownEvent {
                button: MouseButton::Left,
                position: move_event.position,
                modifiers: Modifiers::none(),
                click_count: 2,
            }),
            &mut Window::new(),
            &mut App::create(),
        );
        state.handle_input(
            &InputEvent::MouseUp(MouseUpEvent {
                button: MouseButton::Left,
                position: move_event.position,
                modifiers: Modifiers::none(),
                click_count: 2,
            }),
            &mut Window::new(),
            &mut App::create(),
        );
        assert_eq!(calls.get(), 3);
        assert!(!state.is_active());
    }

    #[test]
    fn focus_visibility_and_typed_action_and_drag_callbacks_are_retained_in_description() {
        let mut state = InteractionState::new();
        let focus = wgpui_core::window::FocusHandle::new();
        state.handle_input(
            &InputEvent::Focus(FocusEvent {
                focused: true,
                visible: true,
            }),
            &mut Window::new(),
            &mut App::create(),
        );
        assert!(state.is_focus_visible());

        let actions = Rc::new(Cell::new(0));
        let observed_actions = actions.clone();
        state.on_action::<Activate, _>(move |_, _, _| {
            observed_actions.set(observed_actions.get() + 1);
        });
        let drags = Rc::new(Cell::new(0));
        let observed_drags = drags.clone();
        state.on_drag(String::from("payload"), move |value, position, _, _| {
            assert_eq!(value, "payload");
            assert_eq!(position, [Pixels(8.0), Pixels(9.0)]);
            observed_drags.set(observed_drags.get() + 1);
        });
        let hover_calls = Rc::new(Cell::new(0));
        let observed_hover_calls = hover_calls.clone();
        state.on_drag_hover::<String, _>(move |hovered, _, _| {
            assert!(hovered);
            observed_hover_calls.set(observed_hover_calls.get() + 1);
        });
        let drop_calls = Rc::new(Cell::new(0));
        let observed_drop_calls = drop_calls.clone();
        state.on_drop::<String, _>(move |value, _, _| {
            assert_eq!(value, "payload");
            observed_drop_calls.set(observed_drop_calls.get() + 1);
        });

        let mut interaction = state.into_description_interaction(Some(focus)).expect("handlers");
        let action = Activate;
        let mut window = Window::new();
        let mut app = App::create();
        assert!(interaction.dispatch_action(&action, &mut window, &mut app).handled);
        let data = interaction
            .drag_source()
            .expect("drag source")
            .with_position([Pixels(8.0), Pixels(9.0)]);
        interaction.start_drag(&data, &mut window, &mut app);
        assert!(interaction.dispatch_drag_hover(true, &data, &mut window, &mut app).handled);
        assert!(interaction.dispatch_drop(&data, &mut window, &mut app).handled);
        assert_eq!(actions.get(), 1);
        assert_eq!(drags.get(), 1);
        assert_eq!(hover_calls.get(), 1);
        assert_eq!(drop_calls.get(), 1);
        assert_eq!(interaction.focus_handle(), Some(focus));
    }

    #[test]
    fn keyboard_text_ime_and_modifier_callbacks_are_dispatched() {
        let mut state = InteractionState::new();
        let events = Rc::new(std::cell::RefCell::new(Vec::new()));
        let observed = events.clone();
        state.on_key_down(move |event, _, _| {
            observed.borrow_mut().push(format!("down:{}", event.key));
        });
        let observed = events.clone();
        state.on_text_input(move |event, _, _| {
            observed.borrow_mut().push(format!("text:{}", event.text));
        });
        let observed = events.clone();
        state.on_ime(move |event, _, _| {
            if matches!(event, ImeEvent::Commit(_)) {
                observed.borrow_mut().push("commit".to_string());
            }
        });
        let observed = events.clone();
        state.on_modifiers_changed(move |event, _, _| {
            if event.modifiers.shift {
                observed.borrow_mut().push("shift".to_string());
            }
        });
        let mut window = Window::new();
        let mut app = App::create();
        state.handle_input(
            &InputEvent::KeyDown(KeyDownEvent::new("a", Modifiers::none())),
            &mut window,
            &mut app,
        );
        state.handle_input(
            &InputEvent::TextInput(TextInputEvent { text: "é".into() }),
            &mut window,
            &mut app,
        );
        state.handle_input(
            &InputEvent::Ime(ImeEvent::Commit("語".into())),
            &mut window,
            &mut app,
        );
        state.handle_input(
            &InputEvent::ModifiersChanged(ModifiersChangedEvent {
                modifiers: Modifiers::shift(),
            }),
            &mut window,
            &mut app,
        );
        assert_eq!(&*events.borrow(), &["down:a", "text:é", "commit", "shift"]);
    }
}
