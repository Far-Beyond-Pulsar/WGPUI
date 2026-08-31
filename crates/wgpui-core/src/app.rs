//! `App`/`Context<T>` root context assembly. See
//! docs/gpu-native-architecture.md §1, §3.1.
use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::action::Action;
use crate::element::{IntoElement, Render};
use crate::reconcile::description::Description;
use crate::window::{KeyBinding, KeyDownEvent, Keymap, Menu, WindowOptions};
use futures::channel::oneshot;
use futures::future::{AbortHandle, Abortable};
use futures::task::LocalSpawnExt;

pub use context::Context;
pub use entity::{Entity, EntityError, EntityId, WeakEntity};

type Observer = Rc<dyn Fn(EntityId)>;
type ActionHandler = Rc<RefCell<dyn FnMut(&dyn Action, &mut App)>>;
type WindowBuild = Box<dyn FnOnce(&mut crate::window::Window, &mut App) -> WindowRenderer>;
type WindowRender = Box<dyn FnMut(&mut crate::window::Window, &mut App) -> Description>;

struct AppState {
    observers: HashMap<EntityId, Vec<(u64, Observer)>>,
    next_observer: u64,
    next_entity: u64,
    globals: HashMap<TypeId, Rc<dyn Any>>,
    keymap: Keymap,
    action_handlers: HashMap<TypeId, Vec<ActionHandler>>,
    propagate_actions: bool,
    active: bool,
    quit_requested: bool,
    menus: Vec<Menu>,
    pending_windows: Vec<WindowRequest>,
}

pub struct WindowRequest {
    pub options: WindowOptions,
    pub build: WindowBuild,
}

pub struct WindowRenderer {
    pub render: WindowRender,
}

/// The foreground application context. It owns entity identity and delivers
/// notifications after an update has released the entity borrow.
#[derive(Clone)]
pub struct App {
    state: Rc<RefCell<AppState>>,
    foreground: Rc<RefCell<futures::executor::LocalPool>>,
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl App {
    pub fn new() -> Self {
        Self {
            state: Rc::new(RefCell::new(AppState {
                observers: HashMap::new(),
                next_observer: 0,
                next_entity: 0,
                globals: HashMap::new(),
                keymap: Keymap::default(),
                action_handlers: HashMap::new(),
                propagate_actions: false,
                active: false,
                quit_requested: false,
                menus: Vec::new(),
                pending_windows: Vec::new(),
            })),
            foreground: Rc::new(RefCell::new(futures::executor::LocalPool::new())),
        }
    }

    pub fn new_entity<T: 'static>(&self, value: T) -> Entity<T> {
        let mut state = self.state.borrow_mut();
        state.next_entity += 1;
        Entity::new(EntityId(state.next_entity), value, self.clone())
    }

    pub(crate) fn add_observer(&self, entity: EntityId, callback: Observer) -> Subscription {
        let mut state = self.state.borrow_mut();
        state.next_observer += 1;
        let id = state.next_observer;
        state
            .observers
            .entry(entity)
            .or_default()
            .push((id, callback));
        Subscription {
            app: self.clone(),
            entity,
            id,
        }
    }

    pub(crate) fn notify_entity(&self, entity: EntityId) {
        let callbacks = self
            .state
            .borrow()
            .observers
            .get(&entity)
            .cloned()
            .unwrap_or_default();
        for (_, callback) in callbacks {
            callback(entity);
        }
    }

    pub fn run_pending_tasks(&self) {
        self.foreground.borrow_mut().run_until_stalled();
    }

    /// Install application-scoped state without making the core crate depend
    /// on the state owner's implementation crate.
    pub fn set_global<T: 'static>(&mut self, value: T) {
        self.state
            .borrow_mut()
            .globals
            .insert(TypeId::of::<T>(), Rc::new(value));
    }

    /// Read application-scoped state installed with [`Self::set_global`].
    pub fn global<T: 'static>(&self) -> Option<Rc<T>> {
        self.state
            .borrow()
            .globals
            .get(&TypeId::of::<T>())
            .and_then(|value| Rc::clone(value).downcast::<T>().ok())
    }

    /// Remove application-scoped state and return it to the caller.
    pub fn remove_global<T: 'static>(&mut self) -> Option<T> {
        self.state
            .borrow_mut()
            .globals
            .remove(&TypeId::of::<T>())
            .and_then(|value| Rc::downcast::<T>(value).ok())
            .and_then(|value| Rc::try_unwrap(value).ok())
    }

    pub fn bind_keys(&mut self, bindings: impl IntoIterator<Item = KeyBinding>) {
        self.state.borrow_mut().keymap.add_all(bindings);
    }

    pub fn clear_key_bindings(&mut self) {
        self.state.borrow_mut().keymap.clear();
    }

    pub fn keymap(&self) -> std::cell::Ref<'_, Keymap> {
        std::cell::Ref::map(self.state.borrow(), |state| &state.keymap)
    }

    pub fn on_action<A: Action>(
        &mut self,
        mut listener: impl FnMut(&A, &mut Self) + 'static,
    ) -> &mut Self {
        let handler: ActionHandler =
            Rc::new(RefCell::new(move |action: &dyn Action, app: &mut App| {
                if let Some(action) = action.as_any().downcast_ref::<A>() {
                    listener(action, app);
                }
            }));
        self.state
            .borrow_mut()
            .action_handlers
            .entry(TypeId::of::<A>())
            .or_default()
            .push(handler);
        self
    }

    pub fn dispatch_action(&mut self, action: &dyn Action) -> bool {
        self.state.borrow_mut().propagate_actions = false;
        let handlers = self
            .state
            .borrow()
            .action_handlers
            .get(&action.as_any().type_id())
            .cloned()
            .unwrap_or_default();
        let mut handled = false;
        for handler in handlers {
            handler.borrow_mut()(action, self);
            handled = true;
            if !self.state.borrow().propagate_actions {
                break;
            }
        }
        handled
    }

    pub fn dispatch_key(&mut self, event: &KeyDownEvent) -> bool {
        let action = self
            .state
            .borrow()
            .keymap
            .resolve(event, None)
            .map(Action::boxed_clone);
        action.is_some_and(|action| self.dispatch_action(&*action))
    }

    pub fn propagate(&mut self) {
        self.state.borrow_mut().propagate_actions = true;
    }

    pub fn activate(&mut self, _ignoring_other_apps: bool) {
        self.state.borrow_mut().active = true;
    }

    pub fn is_active(&self) -> bool {
        self.state.borrow().active
    }

    pub fn quit(&mut self) {
        self.state.borrow_mut().quit_requested = true;
    }

    pub fn quit_requested(&self) -> bool {
        self.state.borrow().quit_requested
    }

    pub fn open_window<V: Render>(
        &mut self,
        options: WindowOptions,
        build_root_view: impl FnOnce(&mut crate::window::Window, &mut App) -> Entity<V> + 'static,
    ) -> Result<(), &'static str> {
        if self.quit_requested() {
            return Err("application is quitting");
        }
        self.state.borrow_mut().pending_windows.push(WindowRequest {
            options,
            build: Box::new(move |window, app| {
                let entity = build_root_view(window, app);
                WindowRenderer {
                    render: Box::new(move |window, _app| {
                        entity.update_in(window, |value, _window, _context| {
                            value.render().into_description()
                        })
                    }),
                }
            }),
        });
        Ok(())
    }

    pub fn take_window_requests(&mut self) -> Vec<WindowRequest> {
        std::mem::take(&mut self.state.borrow_mut().pending_windows)
    }

    pub fn set_menus(&mut self, menus: impl IntoIterator<Item = Menu>) {
        self.state.borrow_mut().menus = menus.into_iter().collect();
    }

    pub fn menus(&self) -> std::cell::Ref<'_, [Menu]> {
        std::cell::Ref::map(self.state.borrow(), |state| state.menus.as_slice())
    }

    pub fn spawn<Fut, T>(&self, future: Fut) -> Task<T>
    where
        Fut: std::future::Future<Output = T> + 'static,
        T: 'static,
    {
        let (abort, registration) = AbortHandle::new_pair();
        let (sender, receiver) = oneshot::channel();
        let future = Abortable::new(future, registration);
        let spawn_result = self.foreground.borrow().spawner().spawn_local(async move {
            if let Ok(value) = future.await {
                let _ = sender.send(value);
            }
        });
        if spawn_result.is_err() {
            abort.abort();
        }
        Task {
            receiver,
            abort: Some(abort),
            completed: false,
        }
    }

    pub fn background_spawn<Fut, T>(&self, future: Fut) -> Task<T>
    where
        Fut: std::future::Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        let (abort, registration) = AbortHandle::new_pair();
        let (sender, receiver) = oneshot::channel();
        std::thread::spawn(move || {
            let result = futures::executor::block_on(Abortable::new(future, registration));
            if let Ok(value) = result {
                let _ = sender.send(value);
            }
        });
        Task {
            receiver,
            abort: Some(abort),
            completed: false,
        }
    }
}

/// A cancellable future returned by foreground and background work.
pub struct Task<T> {
    receiver: oneshot::Receiver<T>,
    abort: Option<AbortHandle>,
    completed: bool,
}

impl<T> Task<T> {
    pub fn ready(value: T) -> Self {
        let (sender, receiver) = oneshot::channel();
        let _ = sender.send(value);
        Self {
            receiver,
            abort: None,
            completed: false,
        }
    }
    pub fn detach(mut self) {
        self.abort = None;
    }
    pub fn cancel(&mut self) {
        if let Some(abort) = self.abort.take() {
            abort.abort();
        }
    }
    pub fn is_cancelled(&self) -> bool {
        self.abort.is_none() && !self.completed
    }
}

impl<T> std::future::Future for Task<T> {
    type Output = Result<T, TaskError>;
    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        match std::pin::Pin::new(&mut self.receiver).poll(cx) {
            std::task::Poll::Ready(Ok(value)) => {
                self.completed = true;
                std::task::Poll::Ready(Ok(value))
            }
            std::task::Poll::Ready(Err(_)) => {
                self.completed = true;
                std::task::Poll::Ready(Err(TaskError::Cancelled))
            }
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

impl<T> Drop for Task<T> {
    fn drop(&mut self) {
        if let Some(abort) = self.abort.take() {
            abort.abort();
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum TaskError {
    Cancelled,
}
impl std::fmt::Display for TaskError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "task cancelled")
    }
}
impl std::error::Error for TaskError {}

pub struct Subscription {
    app: App,
    entity: EntityId,
    id: u64,
}
impl Drop for Subscription {
    fn drop(&mut self) {
        if let Some(observers) = self.app.state.borrow_mut().observers.get_mut(&self.entity) {
            observers.retain(|(id, _)| *id != self.id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::window::MenuItem;
    use std::cell::Cell;
    use std::future::pending;
    use std::rc::Rc;

    crate::actions!(app_test, [Activate]);

    #[test]
    fn entity_identity_and_state_survive_high_frequency_updates() {
        let app = App::new();
        let entity = app.new_entity(0_u64);
        let identity = entity.entity_id();
        let notifications = Rc::new(Cell::new(0_u64));
        let observed = notifications.clone();
        let readable = entity.clone();
        let readable_app = app.clone();
        let _subscription = entity.observe(move |changed| {
            assert_eq!(changed, identity);
            assert_eq!(*readable.read(&readable_app), observed.get() + 1);
            observed.set(observed.get() + 1);
        });
        for _ in 0..10_000 {
            entity.update(|value, _context| *value += 1);
        }
        assert_eq!(entity.entity_id(), identity);
        assert_eq!(*entity.read(&app), 10_000);
        assert_eq!(notifications.get(), 10_000);
    }

    #[test]
    fn foreground_tasks_run_when_the_app_pumps_and_return_errors_as_values() {
        let app = App::new();
        let mut task = app.spawn(async { Ok::<_, &'static str>(42_u32) });
        app.run_pending_tasks();
        assert_eq!(futures::executor::block_on(&mut task), Ok(Ok(42)));
    }

    #[test]
    fn type_erased_globals_are_shared_by_app_clones_and_can_be_removed() {
        let mut app = App::new();
        let clone = app.clone();
        app.set_global(String::from("native service"));
        assert_eq!(
            clone.global::<String>().as_deref().map(String::as_str),
            Some("native service")
        );
        assert_eq!(
            app.remove_global::<String>().as_deref(),
            Some("native service")
        );
        assert!(clone.global::<String>().is_none());
    }

    #[test]
    fn dropping_a_task_cancels_the_underlying_work() {
        let app = App::new();
        let mut task = app.spawn(pending::<()>());
        task.cancel();
        assert!(task.is_cancelled());
        app.run_pending_tasks();
        assert_eq!(
            futures::executor::block_on(&mut task),
            Err(TaskError::Cancelled)
        );
    }

    #[test]
    fn weak_entities_fail_after_the_strong_identity_is_dropped() {
        let app = App::new();
        let entity = app.new_entity(1_u8);
        let weak = entity.downgrade();
        assert_eq!(weak.entity_id(), entity.entity_id());
        drop(entity);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn window_state_uses_the_same_retained_scope_across_frames() {
        let mut window = crate::window::Window::new();
        let scope = crate::reconcile::StateScope::from_path(&[]);
        assert_eq!(
            window.use_state(
                scope,
                || 0_u32,
                |value| {
                    *value = 9;
                    *value
                }
            ),
            Some(9)
        );
        window.next_frame();
        assert_eq!(window.use_state(scope, || 0_u32, |value| *value), Some(9));
        assert_eq!(window.state_len(), 1);
    }

    #[test]
    fn actions_route_in_registration_order_and_can_propagate() {
        let mut app = App::new();
        let calls = Rc::new(RefCell::new(Vec::new()));
        let first_calls = calls.clone();
        app.on_action(move |_: &Activate, app| {
            first_calls.borrow_mut().push("first");
            app.propagate();
        });
        let second_calls = calls.clone();
        app.on_action(move |_: &Activate, _| second_calls.borrow_mut().push("second"));
        assert!(app.dispatch_action(&Activate));
        assert_eq!(&*calls.borrow(), &["first", "second"]);
    }

    #[test]
    fn activation_quit_and_menus_update_shared_application_state() {
        let mut app = App::new();
        assert!(!app.is_active());
        assert!(!app.quit_requested());
        assert!(app.menus().is_empty());

        app.activate(true);
        app.set_menus([Menu::new("File", vec![MenuItem::separator()])]);
        app.quit();

        assert!(app.is_active());
        assert!(app.quit_requested());
        assert_eq!(app.menus()[0].name, "File");
    }

    struct TestRoot;

    impl Render for TestRoot {
        fn render(&mut self) -> impl IntoElement + 'static {
            Description::new::<Self>()
        }
    }

    #[test]
    fn open_window_queues_each_root_and_rejects_new_windows_after_quit() {
        let mut app = App::new();
        app.open_window(WindowOptions::default(), |_, app| app.new_entity(TestRoot))
            .expect("first window request should be accepted");
        app.open_window(WindowOptions::default(), |_, app| app.new_entity(TestRoot))
            .expect("second window request should be accepted");
        assert_eq!(app.take_window_requests().len(), 2);
        app.quit();
        assert_eq!(
            app.open_window(WindowOptions::default(), |_, app| app.new_entity(TestRoot)),
            Err("application is quitting")
        );
    }
}

pub mod async_context;
pub mod context;
pub mod effects;
pub mod entity;
pub mod globals;
