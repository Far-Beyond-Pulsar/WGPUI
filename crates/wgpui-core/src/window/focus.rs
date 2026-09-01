use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_FOCUS_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FocusId(u64);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FocusHandle {
    pub id: FocusId,
    pub tab_index: i32,
    pub tab_stop: bool,
}

impl FocusHandle {
    pub fn new() -> Self {
        Self {
            id: FocusId(NEXT_FOCUS_ID.fetch_add(1, Ordering::Relaxed)),
            tab_index: 0,
            tab_stop: true,
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
    pub const fn tab_index(self, tab_index: i32) -> Self {
        self.with_tab_index(tab_index)
    }
    pub const fn tab_stop(mut self, enabled: bool) -> Self {
        self.tab_stop = enabled;
        self
    }
    pub const fn with_tab_stop(self, enabled: bool) -> Self {
        self.tab_stop(enabled)
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
    tab_stops: HashMap<FocusId, TabStop>,
    next_order: u64,
    pending: Option<FocusTransition>,
}

#[derive(Copy, Clone, Debug)]
struct TabStop {
    index: i32,
    enabled: bool,
    order: u64,
}
impl FocusManager {
    pub fn register(&mut self, handle: &FocusHandle, tab_index: i32) {
        if let Some(entry) = self.tab_stops.get_mut(&handle.id) {
            entry.index = tab_index;
            entry.enabled = true;
            return;
        }
        self.next_order = self.next_order.wrapping_add(1);
        self.tab_stops.insert(
            handle.id,
            TabStop {
                index: tab_index,
                enabled: true,
                order: self.next_order,
            },
        );
    }
    pub fn register_ordered(&mut self, handle: &FocusHandle, tab_index: i32, order: Option<u64>) {
        if let Some(entry) = self.tab_stops.get_mut(&handle.id) {
            entry.index = tab_index;
            entry.enabled = handle.tab_stop;
            if let Some(order) = order {
                entry.order = order;
            }
            return;
        }
        self.next_order = self.next_order.wrapping_add(1);
        self.tab_stops.insert(
            handle.id,
            TabStop {
                index: tab_index,
                enabled: handle.tab_stop,
                order: order.unwrap_or(self.next_order),
            },
        );
    }
    pub fn unregister(&mut self, id: FocusId) {
        self.tab_stops.remove(&id);
        if self.focused == Some(id) {
            self.blur();
        }
    }
    pub fn retain(&mut self, ids: impl IntoIterator<Item = FocusId>) {
        let ids = ids.into_iter().collect::<std::collections::HashSet<_>>();
        let removed_focused = self.focused.is_some_and(|id| !ids.contains(&id));
        self.tab_stops.retain(|id, _| ids.contains(id));
        if removed_focused {
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
        if !self.tab_stops.contains_key(&id) {
            return false;
        }
        if self.focused == Some(id) {
            let visibility_changed = self.focus_visible != visible;
            if visibility_changed {
                self.pending = Some(FocusTransition {
                    from: Some(id),
                    to: Some(id),
                    visible,
                });
            }
            self.focus_visible = visible;
            return visibility_changed;
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
            .filter(|(_, stop)| stop.enabled && stop.index >= 0)
            .map(|(id, stop)| (stop.index, stop.order, *id))
            .collect::<Vec<_>>();
        stops.sort_by_key(|(index, order, id)| (*index, *order, *id));
        if reverse {
            stops.reverse();
        }
        if stops.is_empty() {
            return None;
        }
        let next = match self
            .focused
            .and_then(|id| stops.iter().position(|(_, _, stop)| *stop == id))
        {
            Some(position) => stops
                .get((position + 1) % stops.len())
                .map(|(_, _, id)| *id),
            None => stops.first().map(|(_, _, id)| *id),
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

    #[test]
    fn disabled_tab_stops_are_skipped_and_explicit_order_breaks_ties() {
        let first = FocusHandle::new().with_tab_index(1).with_tab_stop(true);
        let disabled = FocusHandle::new().with_tab_index(0).with_tab_stop(false);
        let second = FocusHandle::new().with_tab_index(1).with_tab_stop(true);
        let mut manager = FocusManager::default();
        manager.register_ordered(&first, first.tab_index_value(), Some(20));
        manager.register_ordered(&disabled, disabled.tab_index_value(), Some(0));
        manager.register_ordered(&second, second.tab_index_value(), Some(10));

        assert_eq!(manager.next(false), Some(second.id()));
        manager.take_transition();
        assert_eq!(manager.next(false), Some(first.id()));
        manager.take_transition();
        assert_eq!(manager.next(false), Some(second.id()));
        assert_eq!(manager.focused(), Some(second.id()));
    }

    #[test]
    fn changing_focus_visibility_invalidates_the_same_focused_handle() {
        let handle = FocusHandle::new();
        let mut manager = FocusManager::default();
        manager.register_ordered(&handle, 0, None);
        assert!(manager.focus(handle.id(), false));
        manager.take_transition();
        assert!(manager.focus(handle.id(), true));
        assert_eq!(
            manager.take_transition(),
            Some(FocusTransition {
                from: Some(handle.id()),
                to: Some(handle.id()),
                visible: true,
            })
        );
    }
}
