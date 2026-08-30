use crate::action::Action;

use super::input::{KeyDownEvent, Modifiers};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Keystroke {
    pub key: String,
    pub modifiers: Modifiers,
}

impl Keystroke {
    pub fn parse(source: &str) -> Result<Self, KeyParseError> {
        let mut modifiers = Modifiers::default();
        let mut key = None;
        for part in source.split('-') {
            let part = part.trim().to_ascii_lowercase();
            match part.as_str() {
                "cmd" | "command" | "meta" | "super" => modifiers.command = true,
                "ctrl" | "control" => modifiers.control = true,
                "alt" | "option" => modifiers.alt = true,
                "shift" => modifiers.shift = true,
                "" => return Err(KeyParseError::EmptyComponent),
                _ if key.is_none() => key = Some(normalize_key(&part)),
                _ => return Err(KeyParseError::MultipleKeys),
            }
        }
        key.map(|key| Self { key, modifiers })
            .ok_or(KeyParseError::MissingKey)
    }

    pub fn matches(&self, event: &KeyDownEvent) -> bool {
        self.key == event.key.to_ascii_lowercase() && self.modifiers == event.modifiers
    }
}

fn normalize_key(key: &str) -> String {
    match key {
        "return" => "enter".to_string(),
        "esc" => "escape".to_string(),
        "spacebar" => "space".to_string(),
        _ => key.to_string(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyParseError {
    EmptyComponent,
    MissingKey,
    MultipleKeys,
}

impl std::fmt::Display for KeyParseError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::EmptyComponent => "keystroke contains an empty component",
            Self::MissingKey => "keystroke has no key",
            Self::MultipleKeys => "keystroke has more than one key",
        };
        formatter.write_str(message)
    }
}
impl std::error::Error for KeyParseError {}

pub struct KeyBinding {
    keystroke: Keystroke,
    action: Box<dyn Action>,
    context: Option<String>,
}

impl Clone for KeyBinding {
    fn clone(&self) -> Self {
        Self {
            keystroke: self.keystroke.clone(),
            action: self.action.boxed_clone(),
            context: self.context.clone(),
        }
    }
}

impl std::fmt::Debug for KeyBinding {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("KeyBinding")
            .field("keystroke", &self.keystroke)
            .field("action", &self.action)
            .field("context", &self.context)
            .finish()
    }
}

impl KeyBinding {
    pub fn new<A: Action + Clone>(keystroke: &str, action: A, context: Option<&str>) -> Self {
        match Self::try_new(keystroke, action.clone(), context) {
            Ok(binding) => binding,
            Err(_) => Self {
                keystroke: Keystroke {
                    key: "\0".to_string(),
                    modifiers: Modifiers::default(),
                },
                action: Box::new(action),
                context: context.map(str::to_string),
            },
        }
    }

    pub fn try_new<A: Action + Clone>(
        keystroke: &str,
        action: A,
        context: Option<&str>,
    ) -> Result<Self, KeyParseError> {
        Ok(Self {
            keystroke: Keystroke::parse(keystroke)?,
            action: Box::new(action),
            context: context.map(str::to_string),
        })
    }

    pub fn keystroke(&self) -> &Keystroke {
        &self.keystroke
    }
    pub fn action(&self) -> &dyn Action {
        &*self.action
    }
    pub fn context(&self) -> Option<&str> {
        self.context.as_deref()
    }
}

#[derive(Default)]
pub struct Keymap {
    bindings: Vec<KeyBinding>,
}

impl Keymap {
    pub fn add(&mut self, binding: KeyBinding) {
        self.bindings.push(binding);
    }
    pub fn add_all(&mut self, bindings: impl IntoIterator<Item = KeyBinding>) {
        self.bindings.extend(bindings);
    }
    pub fn clear(&mut self) {
        self.bindings.clear();
    }
    pub fn bindings(&self) -> &[KeyBinding] {
        &self.bindings
    }

    pub fn resolve(&self, event: &KeyDownEvent, context: Option<&str>) -> Option<&dyn Action> {
        self.bindings
            .iter()
            .rev()
            .find(|binding| {
                binding.keystroke.matches(event)
                    && binding
                        .context
                        .as_deref()
                        .is_none_or(|value| Some(value) == context)
            })
            .map(|binding| binding.action())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    crate::actions!(test, [Open, Close]);

    #[test]
    fn latest_matching_binding_wins() {
        let mut keymap = Keymap::default();
        keymap.add(KeyBinding::new("ctrl-k", Open, None));
        keymap.add(KeyBinding::new("ctrl-k", Close, None));
        let event = KeyDownEvent::new("k", Modifiers::control());
        assert_eq!(
            keymap.resolve(&event, None).map(Action::name),
            Some("test::Close")
        );
    }
}
