use crate::action::Action;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MenuItem {
    Action { label: String, action_name: String },
    Separator,
    Submenu { label: String, items: Vec<MenuItem> },
}

impl MenuItem {
    pub fn action<A: Action>(label: impl Into<String>, action: A) -> Self {
        Self::Action {
            label: label.into(),
            action_name: action.name().to_string(),
        }
    }
    pub fn separator() -> Self {
        Self::Separator
    }
    pub fn submenu(label: impl Into<String>, items: Vec<Self>) -> Self {
        Self::Submenu {
            label: label.into(),
            items,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Menu {
    pub name: String,
    pub items: Vec<MenuItem>,
}

impl Menu {
    pub fn new(name: impl Into<String>, items: Vec<MenuItem>) -> Self {
        Self {
            name: name.into(),
            items,
        }
    }
}
