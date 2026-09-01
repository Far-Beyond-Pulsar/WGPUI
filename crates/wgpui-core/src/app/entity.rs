use super::{App, Context, Observer, Subscription};
use std::cell::{Ref, RefCell};
use std::rc::{Rc, Weak};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EntityId(pub(crate) u64);
impl EntityId {
    pub const fn as_raw(self) -> u64 {
        self.0
    }
}

pub struct Entity<T> {
    id: EntityId,
    value: Rc<RefCell<Option<T>>>,
    app: App,
}
impl<T> Clone for Entity<T> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            value: self.value.clone(),
            app: self.app.clone(),
        }
    }
}
impl<T> Entity<T> {
    pub(crate) fn new(id: EntityId, value: T, app: App) -> Self {
        Self {
            id,
            value: Rc::new(RefCell::new(Some(value))),
            app,
        }
    }
    pub(crate) fn new_uninitialized(id: EntityId, app: App) -> Self {
        Self {
            id,
            value: Rc::new(RefCell::new(None)),
            app,
        }
    }
    pub(crate) fn initialize(&self, value: T) {
        let mut stored = self.value.borrow_mut();
        *stored = Some(value);
    }
    pub fn entity_id(&self) -> EntityId {
        self.id
    }
    pub(crate) fn app(&self) -> App {
        self.app.clone()
    }
    pub(crate) fn app_ref(&self) -> &App {
        &self.app
    }
    pub(crate) fn notify(&self) {
        self.app.notify_entity(self.id);
    }
    pub fn downgrade(&self) -> WeakEntity<T> {
        WeakEntity {
            id: self.id,
            value: Rc::downgrade(&self.value),
            app: self.app.clone(),
        }
    }
    pub fn read(&self, _app: &App) -> Ref<'_, T> {
        Ref::map(self.value.borrow(), |value| {
            value
                .as_ref()
                .expect("an entity cannot be read before its constructor completes")
        })
    }
    pub fn read_with<R>(&self, _app: &App, access: impl FnOnce(&T, &App) -> R) -> R {
        access(
            self.value
                .borrow()
                .as_ref()
                .expect("an entity cannot be read before its constructor completes"),
            &self.app,
        )
    }
    pub fn update<A, R>(&self, _access_context: A, access: impl FnOnce(&mut T, &mut Context<T>) -> R) -> R {
        let result = {
            let mut value = self.value.borrow_mut();
            let mut context = Context::from_entity(self.clone());
            access(
                value
                    .as_mut()
                    .expect("an entity cannot be updated before its constructor completes"),
                &mut context,
            )
        };
        self.app.notify_entity(self.id);
        result
    }
    pub fn update_in<R>(
        &self,
        window: &mut crate::window::Window,
        access: impl FnOnce(&mut T, &mut crate::window::Window, &mut Context<T>) -> R,
    ) -> R {
        let result = {
            let mut value = self.value.borrow_mut();
            let mut context = Context::from_entity(self.clone());
            access(
                value
                    .as_mut()
                    .expect("an entity cannot be updated before its constructor completes"),
                window,
                &mut context,
            )
        };
        self.app.notify_entity(self.id);
        result
    }
    pub fn observe(&self, callback: impl Fn(EntityId) + 'static) -> Subscription {
        self.app
            .add_observer(self.id, Rc::new(callback) as Observer)
    }
}

pub struct WeakEntity<T> {
    id: EntityId,
    value: Weak<RefCell<Option<T>>>,
    app: App,
}
impl<T> Clone for WeakEntity<T> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            value: self.value.clone(),
            app: self.app.clone(),
        }
    }
}
impl<T> WeakEntity<T> {
    pub fn entity_id(&self) -> EntityId {
        self.id
    }
    pub fn upgrade(&self) -> Option<Entity<T>> {
        Some(Entity {
            id: self.id,
            value: self.value.upgrade()?,
            app: self.app.clone(),
        })
    }
    pub fn update<A, R>(
        &self,
        _access_context: A,
        access: impl FnOnce(&mut T, &mut Context<T>) -> R,
    ) -> Result<R, EntityError> {
        self.upgrade()
            .ok_or(EntityError::Dropped)
            .map(|entity| entity.update((), access))
    }
}
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum EntityError {
    Dropped,
}
impl std::fmt::Display for EntityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "entity no longer exists")
    }
}
impl std::error::Error for EntityError {}
