//! Retained interaction state and event callbacks for interactive elements.
//!
//! The state machine is intentionally independent of layout and rendering.
//! The frame builder registers its hitbox and dispatch node, then feeds input
//! into this object; hover/active transitions are therefore observable by the
//! style resolver and do not require repainting unchanged siblings.

use wgpui_core::app::App;
use wgpui_core::window::{ClickEvent, EventResult, InputEvent, MouseButton, ScrollWheelEvent, Window};
use wgpui_core::reconcile::description::DescriptionInteraction;

type ClickHandler = Box<dyn FnMut(&ClickEvent, &mut Window, &mut App) -> EventResult>;
type MouseDownHandler = Box<dyn FnMut(&InputEvent, &mut Window, &mut App) -> EventResult>;
type HoverHandler = Box<dyn FnMut(bool, &mut Window, &mut App) -> EventResult>;
type ScrollHandler = Box<dyn FnMut(&ScrollWheelEvent, &mut Window, &mut App) -> EventResult>;

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
    click: Vec<ClickHandler>,
    mouse_down: Vec<(MouseButton, MouseDownHandler)>,
    hover: Vec<HoverHandler>,
    scroll: Vec<ScrollHandler>,
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
    pub fn on_scroll<R: IntoEventResult + 'static>(
        &mut self,
        mut handler: impl FnMut(&ScrollWheelEvent, &mut Window, &mut App) -> R + 'static,
    ) {
        self.scroll.push(Box::new(move |event, window, app| {
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
            InputEvent::MouseEnter(_) => self.update_hover(true, window, app),
            InputEvent::MouseLeave(_) => self.update_hover(false, window, app),
            InputEvent::MouseDown(mouse) => {
                self.active = mouse.button == MouseButton::Left;
                let mut result = EventResult::IGNORED;
                for (button, handler) in &mut self.mouse_down {
                    if *button == mouse.button {
                        let current = handler(event, window, app);
                        merge_result(&mut result, current);
                    }
                }
                result
            }
            InputEvent::MouseUp(_) => {
                let changed = self.active;
                self.active = false;
                if changed {
                    EventResult::HANDLED
                } else {
                    EventResult::IGNORED
                }
            }
            InputEvent::Click(click) => {
                let mut result = EventResult::IGNORED;
                for handler in &mut self.click {
                    let current = handler(click, window, app);
                    merge_result(&mut result, current);
                }
                result
            }
            InputEvent::Scroll(scroll) => {
                let mut result = EventResult::IGNORED;
                for handler in &mut self.scroll {
                    let current = handler(scroll, window, app);
                    merge_result(&mut result, current);
                }
                result
            }
            _ => EventResult::IGNORED,
        }
    }

    pub fn into_description_interaction(self) -> Option<DescriptionInteraction> {
        if self.click.is_empty()
            && self.mouse_down.is_empty()
            && self.hover.is_empty()
            && self.scroll.is_empty()
        {
            return None;
        }
        let mut state = self;
        Some(DescriptionInteraction::new(move |event, window, app| {
            state.handle_input(event, window, app)
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;
    use wgpui_core::boundary::Pixels;
    use wgpui_core::window::{Modifiers, MouseDownEvent};

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
                .update_hover(true, &mut Window::new(), &mut App::new())
                .handled
        );
        assert!(
            !state
                .update_hover(true, &mut Window::new(), &mut App::new())
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
                .handle_input(&event, &mut Window::new(), &mut App::new())
                .handled
        );
        assert_eq!(downs.get(), 0);
    }
}
