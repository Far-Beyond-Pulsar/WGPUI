use crate::boundary::Pixels;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Modifiers {
    pub shift: bool,
    pub control: bool,
    pub alt: bool,
    pub command: bool,
}
impl Modifiers {
    pub const fn none() -> Self {
        Self {
            shift: false,
            control: false,
            alt: false,
            command: false,
        }
    }
    pub const fn shift() -> Self {
        Self {
            shift: true,
            ..Self::none()
        }
    }
    pub const fn control() -> Self {
        Self {
            control: true,
            ..Self::none()
        }
    }
    pub const fn alt() -> Self {
        Self {
            alt: true,
            ..Self::none()
        }
    }
    pub const fn command() -> Self {
        Self {
            command: true,
            ..Self::none()
        }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum MouseButton {
    #[default]
    Left,
    Right,
    Middle,
    Other(u16),
}
#[derive(Clone, Debug, PartialEq)]
pub struct KeyDownEvent {
    pub key: String,
    pub modifiers: Modifiers,
    pub repeat: bool,
}
impl KeyDownEvent {
    pub fn new(key: impl Into<String>, modifiers: Modifiers) -> Self {
        Self {
            key: key.into(),
            modifiers,
            repeat: false,
        }
    }
}
#[derive(Clone, Debug, PartialEq)]
pub struct KeyUpEvent {
    pub key: String,
    pub modifiers: Modifiers,
}
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MouseDownEvent {
    pub button: MouseButton,
    pub position: [Pixels; 2],
    pub modifiers: Modifiers,
    pub click_count: u32,
}
impl MouseDownEvent {
    pub fn is_focusing(&self) -> bool {
        self.button == MouseButton::Left
    }
}
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MouseUpEvent {
    pub button: MouseButton,
    pub position: [Pixels; 2],
    pub modifiers: Modifiers,
    pub click_count: u32,
}
impl MouseUpEvent {
    pub fn is_focusing(&self) -> bool {
        self.button == MouseButton::Left
    }
}
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct MouseButtonState {
    pub left: bool,
    pub right: bool,
    pub middle: bool,
}
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MouseMoveEvent {
    pub position: [Pixels; 2],
    pub modifiers: Modifiers,
    pub buttons: MouseButtonState,
}
impl MouseMoveEvent {
    pub fn dragging(&self) -> bool {
        self.buttons.left || self.buttons.right || self.buttons.middle
    }
}
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ScrollWheelEvent {
    pub position: [Pixels; 2],
    pub delta: [f32; 2],
    pub modifiers: Modifiers,
}
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum KeyboardButton {
    #[default]
    Enter,
    Space,
}
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct KeyboardClickEvent {
    pub button: KeyboardButton,
    pub bounds: crate::geometry::Rect,
}
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MouseClickEvent {
    pub down: MouseDownEvent,
    pub up: MouseUpEvent,
}
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ClickEvent {
    Mouse(MouseClickEvent),
    Keyboard(KeyboardClickEvent),
}
impl Default for ClickEvent {
    fn default() -> Self {
        Self::Keyboard(KeyboardClickEvent {
            button: KeyboardButton::Enter,
            bounds: crate::geometry::Rect::EMPTY,
        })
    }
}
impl ClickEvent {
    pub fn modifiers(&self) -> Modifiers {
        match self {
            Self::Mouse(click) => click.up.modifiers,
            Self::Keyboard(_) => Modifiers::none(),
        }
    }
    pub fn position(&self) -> [Pixels; 2] {
        match self {
            Self::Mouse(click) => click.up.position,
            Self::Keyboard(click) => [Pixels(click.bounds.max_x), Pixels(click.bounds.max_y)],
        }
    }
    pub fn mouse_position(&self) -> Option<[Pixels; 2]> {
        match self {
            Self::Mouse(click) => Some(click.up.position),
            Self::Keyboard(_) => None,
        }
    }
    pub fn is_right_click(&self) -> bool {
        matches!(self, Self::Mouse(click) if click.down.button == MouseButton::Right && click.up.button == MouseButton::Right)
    }
    pub fn standard_click(&self) -> bool {
        matches!(
            self,
            Self::Keyboard(_)
                | Self::Mouse(MouseClickEvent {
                    down: MouseDownEvent {
                        button: MouseButton::Left,
                        ..
                    },
                    up: MouseUpEvent {
                        button: MouseButton::Left,
                        ..
                    }
                })
        )
    }
    pub fn first_focus(&self) -> bool {
        matches!(self, Self::Mouse(click) if click.down.is_focusing())
    }
    pub fn click_count(&self) -> u32 {
        match self {
            Self::Mouse(click) => click.up.click_count,
            Self::Keyboard(_) => 1,
        }
    }
    pub fn is_keyboard(&self) -> bool {
        matches!(self, Self::Keyboard(_))
    }
}
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct EventResult {
    pub handled: bool,
    pub propagate: bool,
}
impl EventResult {
    pub const HANDLED: Self = Self {
        handled: true,
        propagate: false,
    };
    pub const IGNORED: Self = Self {
        handled: false,
        propagate: true,
    };
}
#[derive(Clone, Debug, PartialEq)]
pub enum InputEvent {
    KeyDown(KeyDownEvent),
    KeyUp(KeyUpEvent),
    MouseDown(MouseDownEvent),
    MouseUp(MouseUpEvent),
    MouseMove(MouseMoveEvent),
    Scroll(ScrollWheelEvent),
    Click(ClickEvent),
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn click_event_preserves_mouse_semantics() {
        let down = MouseDownEvent {
            button: MouseButton::Left,
            position: [Pixels(2.0), Pixels(3.0)],
            modifiers: Modifiers::shift(),
            click_count: 2,
        };
        let up = MouseUpEvent {
            button: MouseButton::Left,
            position: [Pixels(4.0), Pixels(5.0)],
            modifiers: Modifiers::shift(),
            click_count: 2,
        };
        let click = ClickEvent::Mouse(MouseClickEvent { down, up });
        assert!(click.standard_click() && click.first_focus());
        assert_eq!(click.click_count(), 2);
        assert_eq!(click.mouse_position(), Some([Pixels(4.0), Pixels(5.0)]));
    }
}
