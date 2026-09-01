use super::{App, Entity, EntityId, Subscription, Task};
use crate::boundary::Pixels;
use crate::element::IntoElement;
use crate::geometry::Bounds;
use crate::window::{BackgroundExecutor, FocusHandle, Window, WindowAppearance};
use std::ops::Deref;
use std::ops::Range;

/// The backend hook used by [`Context`] to observe native window changes.
///
/// A backend returns an owned observer guard. The guard remains active until
/// the [`Subscription`] returned by `Context` is dropped, and the backend is
/// responsible for delivering callbacks from its native event stream.
pub trait WindowObserverSource {
    type Observer: 'static;

    fn observe_bounds(
        &mut self,
        callback: Box<dyn FnMut(Bounds<Pixels>) + 'static>,
    ) -> Self::Observer;

    fn observe_appearance(
        &mut self,
        callback: Box<dyn FnMut(WindowAppearance) + 'static>,
    ) -> Self::Observer;
}

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
    /// The backend owns the event source; core owns the subscription lifetime.
    pub fn observe_window_bounds<W, F>(&self, window: &mut W, callback: F) -> Subscription
    where
        W: WindowObserverSource,
        F: FnMut(Bounds<Pixels>) + 'static,
    {
        let observer = window.observe_bounds(Box::new(callback));
        self.entity.observe(move |_| {
            let _ = &observer;
        })
    }

    /// Register a window-appearance observer with the active window backend.
    /// The backend owns the event source; core owns the subscription lifetime.
    pub fn observe_window_appearance<W, F>(&self, window: &mut W, callback: F) -> Subscription
    where
        W: WindowObserverSource,
        F: FnMut(WindowAppearance) + 'static,
    {
        let observer = window.observe_appearance(Box::new(callback));
        self.entity.observe(move |_| {
            let _ = &observer;
        })
    }

    pub(crate) fn take_notified(&mut self) -> bool {
        std::mem::take(&mut self.notified)
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
    pub fn processor<I, F>(&self, mut callback: F) -> impl FnMut(Range<usize>) -> Vec<I> + 'static
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

        entity.update((), |value, context| {
            *value += 1;
            context.notify();
        });
        assert_eq!(notifications.get(), 1);
    }

    #[derive(Default)]
    struct TestWindow {
        bounds_observers: Vec<(
            std::rc::Rc<std::cell::Cell<bool>>,
            Box<dyn FnMut(crate::geometry::Bounds<crate::boundary::Pixels>)>,
        )>,
        appearance_observers: Vec<(
            std::rc::Rc<std::cell::Cell<bool>>,
            Box<dyn FnMut(crate::window::WindowAppearance)>,
        )>,
    }

    struct TestObserver {
        active: std::rc::Rc<std::cell::Cell<bool>>,
    }

    impl Drop for TestObserver {
        fn drop(&mut self) {
            self.active.set(false);
        }
    }

    impl WindowObserverSource for TestWindow {
        type Observer = TestObserver;

        fn observe_bounds(
            &mut self,
            callback: Box<dyn FnMut(crate::geometry::Bounds<crate::boundary::Pixels>)>,
        ) -> Self::Observer {
            let active = std::rc::Rc::new(std::cell::Cell::new(true));
            self.bounds_observers.push((active.clone(), callback));
            TestObserver { active }
        }

        fn observe_appearance(
            &mut self,
            callback: Box<dyn FnMut(crate::window::WindowAppearance)>,
        ) -> Self::Observer {
            let active = std::rc::Rc::new(std::cell::Cell::new(true));
            self.appearance_observers.push((active.clone(), callback));
            TestObserver { active }
        }
    }

    impl TestWindow {
        fn emit_bounds(&mut self, bounds: crate::geometry::Bounds<crate::boundary::Pixels>) {
            for (active, callback) in &mut self.bounds_observers {
                if active.get() {
                    callback(bounds);
                }
            }
        }

        fn emit_appearance(&mut self, appearance: crate::window::WindowAppearance) {
            for (active, callback) in &mut self.appearance_observers {
                if active.get() {
                    callback(appearance);
                }
            }
        }
    }

    #[test]
    fn window_observers_forward_backend_events_and_stop_after_subscription_drop() {
        let app = App::create();
        let entity = app.new_entity(());
        let context = Context::from_entity(entity);
        let mut window = TestWindow::default();
        let bounds = crate::geometry::Bounds::new(
            crate::geometry::point(crate::boundary::Pixels(2.0), crate::boundary::Pixels(3.0)),
            crate::geometry::size(crate::boundary::Pixels(40.0), crate::boundary::Pixels(50.0)),
        );
        let bounds_seen = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let bounds_seen_by_callback = bounds_seen.clone();
        let bounds_subscription = context.observe_window_bounds(&mut window, move |value| {
            bounds_seen_by_callback.borrow_mut().push(value);
        });
        let appearance_seen = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let appearance_seen_by_callback = appearance_seen.clone();
        let appearance_subscription =
            context.observe_window_appearance(&mut window, move |value| {
                appearance_seen_by_callback.borrow_mut().push(value);
            });

        window.emit_bounds(bounds);
        window.emit_appearance(crate::window::WindowAppearance::Dark);
        assert_eq!(&*bounds_seen.borrow(), &[bounds]);
        assert_eq!(
            &*appearance_seen.borrow(),
            &[crate::window::WindowAppearance::Dark]
        );

        drop(bounds_subscription);
        drop(appearance_subscription);
        window.emit_bounds(bounds);
        window.emit_appearance(crate::window::WindowAppearance::Light);
        assert_eq!(&*bounds_seen.borrow(), &[bounds]);
        assert_eq!(
            &*appearance_seen.borrow(),
            &[crate::window::WindowAppearance::Dark]
        );
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
