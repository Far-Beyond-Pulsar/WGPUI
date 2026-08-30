use super::{App, Entity, EntityId, Subscription, Task};
use crate::window::{FocusHandle, Window};

pub struct Context<T> {
    entity: Entity<T>,
    notified: bool,
}
impl<T> Context<T> {
    pub(crate) fn new(entity: Entity<T>) -> Self {
        Self {
            entity,
            notified: false,
        }
    }
    pub fn entity(&self) -> Entity<T> {
        self.entity.clone()
    }
    /// Mark this entity dirty. `update` delivers the notification after its
    /// mutable borrow ends, so observers may safely read the entity.
    pub fn notify(&mut self) {
        self.notified = true;
    }
    pub fn spawn<F, Fut, R>(&self, make: F) -> Task<R>
    where
        F: FnOnce(super::WeakEntity<T>, Context<T>) -> Fut,
        Fut: std::future::Future<Output = R> + 'static,
        R: 'static,
    {
        self.entity.app().spawn(make(
            self.entity.downgrade(),
            Context::new(self.entity.clone()),
        ))
    }
    pub fn observe(&self, callback: impl Fn(EntityId) + 'static) -> Subscription {
        self.entity.observe(callback)
    }
    pub fn app(&self) -> App {
        self.entity.app()
    }

    /// Activate the application owning this entity.
    pub fn activate(&self, ignoring_other_apps: bool) {
        let mut app = self.app();
        app.activate(ignoring_other_apps);
    }

    /// Request application shutdown from an entity callback.
    pub fn quit(&self) {
        let mut app = self.app();
        app.quit();
    }

    /// Create a focus handle for a control owned by this entity.
    pub fn focus_handle(&self) -> FocusHandle {
        FocusHandle::new()
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn listener_updates_the_live_entity_and_can_use_window_state() {
        let app = App::new();
        let entity = app.new_entity(0_u32);
        let context = Context::new(entity.clone());
        let listener = context.listener(|value, event: &u32, window, context| {
            *value += *event;
            window.activate();
            context.notify();
        });
        let mut window = Window::new();

        let mut callback_app = app.clone();
        listener(&3, &mut window, &mut callback_app);

        assert_eq!(*entity.read(&app), 3);
        assert!(window.is_active());
    }

    #[test]
    fn listener_does_not_keep_an_entity_alive() {
        let app = App::new();
        let entity = app.new_entity(0_u32);
        let context = Context::new(entity.clone());
        let listener = context.listener(|_: &mut u32, _: &(), _: &mut Window, _| {});
        drop(entity);
        let mut window = Window::new();
        let mut callback_app = app.clone();
        listener(&(), &mut window, &mut callback_app);
    }

    #[test]
    fn context_activation_and_quit_delegate_to_shared_app_state() {
        let app = App::new();
        let entity = app.new_entity(());
        let context = Context::new(entity);

        context.activate(true);
        context.quit();

        assert!(app.is_active());
        assert!(app.quit_requested());
    }
}

impl<T: 'static> Context<T> {
    /// Bind an entity method to an input callback without capturing a strong
    /// entity reference. This is the same lifetime behavior as `listener` in
    /// the legacy API: a callback silently stops being callable once its
    /// entity has been dropped.
    pub fn listener<E: ?Sized>(
        &self,
        callback: impl Fn(&mut T, &E, &mut Window, &mut Context<T>) + 'static,
    ) -> impl Fn(&E, &mut Window, &mut App) + 'static
    where
        T: 'static,
    {
        let entity = self.entity.downgrade();
        move |event: &E, window: &mut Window, _app: &mut App| {
            if let Some(entity) = entity.upgrade() {
                entity.update_in(window, |value, window, context| {
                    callback(value, event, window, context);
                });
            }
        }
    }
}
