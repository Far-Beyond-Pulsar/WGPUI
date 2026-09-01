use super::{App, Entity, EntityId, Subscription, Task};
use crate::window::{BackgroundExecutor, FocusHandle, Window};
use crate::element::IntoElement;
use std::ops::Range;
use std::ops::Deref;

pub struct Context<T> {
    entity: Entity<T>,
    notified: bool,
}
impl<T> Context<T> {
    pub(crate) fn from_entity(entity: Entity<T>) -> Self {
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
    pub fn spawn<F, R>(&self, make: F) -> Task<R>
    where
        T: 'static,
        F: AsyncFnOnce(super::WeakEntity<T>, &Context<T>) -> R + 'static,
        R: 'static,
    {
        let entity = self.entity.clone();
        self.entity.app().spawn(async move {
            let context = Context::from_entity(entity.clone());
            make(entity.downgrade(), &context).await
        })
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

    /// Hide the application's windows until the application is shown again.
    pub fn hide(&self) {
        let mut app = self.app();
        app.hide();
    }

    /// Register a window-bounds observer with the active window backend.
    ///
    /// Core does not own native window events, but it provides the same
    /// subscription lifetime used by backend adapters.
    pub fn observe_window_bounds<W, F>(&self, _window: &W, _callback: F) -> Subscription
    where
        F: 'static,
    {
        self.entity.observe(|_| {})
    }

    /// Register a window-appearance observer with the active window backend.
    pub fn observe_window_appearance<W, F>(&self, _window: &W, _callback: F) -> Subscription
    where
        F: 'static,
    {
        self.entity.observe(|_| {})
    }

    /// Create a focus handle for a control owned by this entity.
    pub fn focus_handle(&self) -> FocusHandle {
        FocusHandle::new()
    }

    /// Run sendable work away from the foreground executor.
    pub fn background_spawn<Fut, R>(&self, future: Fut) -> Task<R>
    where
        Fut: std::future::Future<Output = R> + Send + 'static,
        R: Send + 'static,
    {
        self.app().background_spawn(future)
    }

    pub fn background_executor(&self) -> BackgroundExecutor {
        BackgroundExecutor
    }

    /// Bind a retained entity to a range-based list renderer.
    pub fn processor<I, F>(
        &self,
        mut callback: F,
    ) -> impl FnMut(Range<usize>) -> Vec<I> + 'static
    where
        T: 'static,
        I: IntoElement + 'static,
        F: FnMut(&mut T, Range<usize>, &mut Window, &mut Context<T>) -> Vec<I> + 'static,
    {
        let entity = self.entity.downgrade();
        move |range| {
            let Some(entity) = entity.upgrade() else {
                return Vec::new();
            };
            let mut window = Window::new();
            entity.update_in(&mut window, |value, window, context| {
                callback(value, range, window, context)
            })
        }
    }
}

impl<T> Deref for Context<T> {
    type Target = App;

    fn deref(&self) -> &Self::Target {
        self.entity.app_ref()
    }
}

impl<T: 'static> Context<T> {
    /// Bind an entity method to an input callback without capturing a strong
    /// entity reference.
    pub fn listener<E: ?Sized>(
        &self,
        callback: impl Fn(&mut T, &E, &mut Window, &mut Context<T>) + 'static,
    ) -> impl Fn(&E, &mut Window, &mut App) + 'static {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn listener_updates_the_live_entity_and_can_use_window_state() {
        let app = App::create();
        let entity = app.new_entity(0_u32);
        let context = Context::from_entity(entity.clone());
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
        let app = App::create();
        let entity = app.new_entity(0_u32);
        let context = Context::from_entity(entity.clone());
        let listener = context.listener(|_: &mut u32, _: &(), _: &mut Window, _| {});
        drop(entity);
        let mut window = Window::new();
        let mut callback_app = app.clone();
        listener(&(), &mut window, &mut callback_app);
    }

    #[test]
    fn context_activation_and_quit_delegate_to_shared_app_state() {
        let app = App::create();
        let entity = app.new_entity(());
        let context = Context::from_entity(entity);

        context.activate(true);
        context.quit();

        assert!(app.is_active());
        assert!(app.quit_requested());
    }

    #[test]
    fn context_can_hide_and_show_the_shared_application() {
        let mut app = App::create();
        let entity = app.new_entity(());
        let context = Context::from_entity(entity);

        context.hide();
        assert!(app.is_hidden());

        app.show();
        assert!(!app.is_hidden());
    }

    #[test]
    fn detached_entity_observers_survive_handle_drop() {
        let app = App::create();
        let entity = app.new_entity(0_u32);
        let notifications = std::rc::Rc::new(std::cell::Cell::new(0));
        let notifications_for_callback = notifications.clone();
        let subscription = entity.observe(move |_| {
            notifications_for_callback.set(notifications_for_callback.get() + 1);
        });
        subscription.detach();

        entity.update((), |value, _| *value += 1);
        assert_eq!(notifications.get(), 1);
    }

    #[test]
    fn processor_updates_its_entity_for_each_requested_range() {
        let app = App::create();
        let entity = app.new_entity(0_u32);
        let context = Context::from_entity(entity.clone());
        let mut processor = context.processor(|value, range, _window, _context| {
            *value += range.len() as u32;
            range.map(|index| index.to_string()).collect()
        });

        let items = processor(2..5);

        assert_eq!(items, vec!["2", "3", "4"]);
        assert_eq!(*entity.read(&app), 3);
    }

    #[test]
    fn processor_passes_a_live_window_and_inner_context() {
        let app = App::create();
        let entity = app.new_entity(false);
        let entity_id = entity.entity_id();
        let context = Context::from_entity(entity.clone());
        let mut processor = context.processor(move |value, range, window, inner_context| {
            window.activate();
            *value = window.is_active()
                && inner_context.entity().entity_id() == entity_id
                && range.start == 4;
            Vec::<String>::new()
        });

        processor(4..5);

        assert!(*entity.read(&app));
    }

    #[test]
    fn context_spawn_uses_the_weak_handle_and_live_context() {
        let app = App::create();
        let entity = app.new_entity(0_u32);
        let context = Context::from_entity(entity.clone());
        let mut task = context.spawn(async move |handle, inner_context| {
            assert_eq!(handle.entity_id(), inner_context.entity().entity_id());
            handle
                .update((), |value, callback_context| {
                    *value += 1;
                    callback_context.notify();
                })
                .expect("the entity should still be alive");
            7_u32
        });

        app.run_pending_tasks();

        assert_eq!(futures::executor::block_on(&mut task), Ok(7));
        assert_eq!(*entity.read(&app), 1);
    }
}
