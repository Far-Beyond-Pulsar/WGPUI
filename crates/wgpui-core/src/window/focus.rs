use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_FOCUS_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FocusId(u64);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FocusHandle {
    id: FocusId,
    tab_index: i32,
}

impl FocusHandle {
    pub fn new() -> Self {
        Self {
            id: FocusId(NEXT_FOCUS_ID.fetch_add(1, Ordering::Relaxed)),
            tab_index: 0,
        }
    }
    pub const fn id(self) -> FocusId {
        self.id
    }
    pub const fn tab_index_value(self) -> i32 {
        self.tab_index
    }
    pub const fn with_tab_index(mut self, tab_index: i32) -> Self {
        self.tab_index = tab_index;
        self
    }
    pub fn focus(self, window: &mut super::Window) -> bool {
        window.focus(&self)
    }
    pub fn is_focused(self, window: &super::Window) -> bool {
        window.is_focused(&self)
    }
    pub fn contains_focused(self, window: &super::Window) -> bool {
        self.is_focused(window)
    }
}
impl Default for FocusHandle {
    fn default() -> Self {
        Self::new()
    }
}

pub trait Focusable {
    fn focus_handle(&self) -> FocusHandle;
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FocusTransition {
    pub from: Option<FocusId>,
    pub to: Option<FocusId>,
    pub visible: bool,
}

#[derive(Debug, Default)]
pub struct FocusManager {
    focused: Option<FocusId>,
    focus_visible: bool,
    tab_stops: HashMap<FocusId, i32>,
    pending: Option<FocusTransition>,
}
impl FocusManager {
    pub fn register(&mut self, handle: &FocusHandle, tab_index: i32) {
        self.tab_stops.insert(handle.id, tab_index);
    }
    pub fn unregister(&mut self, id: FocusId) {
        self.tab_stops.remove(&id);
        if self.focused == Some(id) {
            self.blur();
        }
    }
    pub fn focused(&self) -> Option<FocusId> {
        self.focused
    }
    pub fn focus_visible(&self) -> bool {
        self.focus_visible
    }
    pub fn focus(&mut self, id: FocusId, visible: bool) -> bool {
        if self.focused == Some(id) {
            self.focus_visible = visible;
            return false;
        }
        let from = self.focused.replace(id);
        self.focus_visible = visible;
        self.pending = Some(FocusTransition {
            from,
            to: Some(id),
            visible,
        });
        true
    }
    pub fn blur(&mut self) -> bool {
        let Some(from) = self.focused.take() else {
            return false;
        };
        self.pending = Some(FocusTransition {
            from: Some(from),
            to: None,
            visible: false,
        });
        self.focus_visible = false;
        true
    }
    pub fn take_transition(&mut self) -> Option<FocusTransition> {
        self.pending.take()
    }
    pub fn contains(&self, id: FocusId) -> bool {
        self.tab_stops.contains_key(&id)
    }
    pub fn next(&mut self, reverse: bool) -> Option<FocusId> {
        let mut stops = self
            .tab_stops
            .iter()
            .map(|(id, index)| (*index, *id))
            .collect::<Vec<_>>();
        stops.sort_by_key(|(index, id)| (*index, *id));
        if reverse {
            stops.reverse();
        }
        if stops.is_empty() {
            return None;
        }
        let next = match self
            .focused
            .and_then(|id| stops.iter().position(|(_, stop)| *stop == id))
        {
            Some(position) => stops.get((position + 1) % stops.len()).map(|(_, id)| *id),
            None => stops.first().map(|(_, id)| *id),
        }?;
        self.focus(next, true);
        Some(next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn focus_transitions_are_deterministic() {
        let first = FocusHandle::new().with_tab_index(1);
        let second = FocusHandle::new().with_tab_index(2);
        let mut manager = FocusManager::default();
        manager.register(&first, first.tab_index_value());
        manager.register(&second, second.tab_index_value());
        assert_eq!(manager.next(false), Some(first.id()));
        assert_eq!(
            manager.take_transition(),
            Some(FocusTransition {
                from: None,
                to: Some(first.id()),
                visible: true
            })
        );
        assert_eq!(manager.next(false), Some(second.id()));
        assert_eq!(
            manager.take_transition().map(|change| change.to),
            Some(Some(second.id()))
        );
    }
}
