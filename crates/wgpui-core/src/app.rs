//! `App`/`Context<T>` root context assembly. See
//! docs/gpu-native-architecture.md §1, §3.1.
use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use crate::action::Action;
#[cfg(test)]
use crate::element::IntoElement;
use crate::reconcile::description::Description;
use crate::window::{KeyBinding, KeyDownEvent, Keymap, Menu, WindowOptions};
use futures::channel::oneshot;
use futures::future::{AbortHandle, Abortable};
use futures::task::LocalSpawnExt;

pub use context::Context;
pub use entity::{Entity, EntityError, EntityId, WeakEntity};

/// Constructs an entity while giving its constructor a context owned by the
/// same eventual entity. The entity is initialized immediately after the
/// constructor returns, so handles captured by the constructor remain valid.
#[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
pub trait EntityFactory {
    fn new<T: 'static>(&mut self, build: impl FnOnce(&mut Context<T>) -> T) -> Entity<T>;
}

type Observer = Rc<dyn Fn(EntityId)>;
type ActionHandler = Rc<RefCell<dyn FnMut(&dyn Action, &mut App)>>;
type WindowClosedHandler = Rc<RefCell<dyn FnMut(&mut App, WindowId)>>;

struct AppState {
    observers: HashMap<EntityId, Vec<(u64, Observer)>>,
    entity_invalidations: Vec<EntityId>,
    queued_entity_invalidations: HashSet<EntityId>,
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
    next_window: u64,
    windows: HashSet<WindowId>,
    close_requested: bool,
    hidden: bool,
    next_window_closed_handler: u64,
    window_closed_handlers: Vec<(u64, WindowClosedHandler)>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WindowId(u64);

impl WindowId {
    pub const fn as_raw(self) -> u64 {
        self.0
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct WindowList {
    count: usize,
}

impl WindowList {
    pub const fn len(self) -> usize {
        self.count
    }
    pub const fn is_empty(self) -> bool {
        self.count == 0
    }
}

pub struct WindowRequest {
    pub id: WindowId,
    pub options: WindowOptions,
    pub build: Box<dyn FnOnce(&mut App, &mut dyn Any) -> WindowRenderer>,
}

pub struct WindowRenderer {
    pub render: Box<dyn FnMut(&mut App, &mut dyn Any) -> Description>,
}

/// The foreground application context. It owns entity identity and delivers
/// notifications after an update has released the entity borrow.
#[derive(Clone)]
pub struct App {
    state: Rc<RefCell<AppState>>,
    foreground: Rc<RefCell<futures::executor::LocalPool>>,
    pending_tasks: Arc<AtomicUsize>,
}

impl Default for App {
    fn default() -> Self {
        Self::create()
    }
}

impl App {
    pub fn create() -> Self {
        Self {
            state: Rc::new(RefCell::new(AppState {
                observers: HashMap::new(),
                entity_invalidations: Vec::new(),
                queued_entity_invalidations: HashSet::new(),
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
                next_window: 0,
                windows: HashSet::new(),
                close_requested: false,
                hidden: false,
                next_window_closed_handler: 0,
                window_closed_handlers: Vec::new(),
            })),
            foreground: Rc::new(RefCell::new(futures::executor::LocalPool::new())),
            pending_tasks: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn new_entity<T: 'static>(&self, value: T) -> Entity<T> {
        let mut state = self.state.borrow_mut();
        state.next_entity += 1;
        Entity::new(EntityId(state.next_entity), value, self.clone())
    }

    pub fn new_entity_with<T: 'static>(
        &self,
        build: impl FnOnce(&mut Context<T>) -> T,
    ) -> Entity<T> {
        let mut state = self.state.borrow_mut();
        state.next_entity += 1;
        let entity = Entity::new_uninitialized(EntityId(state.next_entity), self.clone());
        drop(state);
        let mut context = Context::from_entity(entity.clone());
        let value = build(&mut context);
        entity.initialize(value);
        entity
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
            detached: false,
        }
    }

    pub(crate) fn notify_entity(&self, entity: EntityId) {
        {
            let mut state = self.state.borrow_mut();
            if state.queued_entity_invalidations.insert(entity) {
                state.entity_invalidations.push(entity);
            }
        }
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

    /// Notify observers of an entity from an application callback.
    pub fn notify(&self, entity: EntityId) {
        self.notify_entity(entity);
    }

    /// Drain entity changes raised since the last native frame consumed them.
    ///
    /// Entity IDs are coalesced without changing their first-seen order. The
    /// queue is application-owned so cloned app handles and every native
    /// window observe the same notification stream.
    pub fn drain_entity_invalidations(&self) -> Vec<EntityId> {
        let mut state = self.state.borrow_mut();
        state.queued_entity_invalidations.clear();
        std::mem::take(&mut state.entity_invalidations)
    }

    /// Whether `entity` has a queued change awaiting native consumption.
    pub fn has_pending_entity_invalidation(&self, entity: EntityId) -> bool {
        self.state
            .borrow()
            .queued_entity_invalidations
            .contains(&entity)
    }

    pub fn run_pending_tasks(&self) {
        self.foreground.borrow_mut().run_until_stalled();
    }

    pub fn has_pending_tasks(&self) -> bool {
        self.pending_tasks.load(Ordering::Acquire) != 0
    }

    pub fn background_executor(&self) -> crate::window::BackgroundExecutor {
        crate::window::BackgroundExecutor
    }

    /// Installs application-scoped state shared by cloned application handles.
    pub fn set_global<T: 'static>(&mut self, value: T) {
        self.state
            .borrow_mut()
            .globals
            .insert(TypeId::of::<T>(), Rc::new(value));
    }

    /// Reads application-scoped state installed with [`Self::set_global`].
    pub fn global<T: 'static>(&self) -> Option<Rc<T>> {
        self.state
            .borrow()
            .globals
            .get(&TypeId::of::<T>())
            .and_then(|value| Rc::clone(value).downcast::<T>().ok())
    }

    /// Removes application-scoped state and returns it when uniquely owned.
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
        let mut state = self.state.borrow_mut();
        state.quit_requested = true;
        state.close_requested = true;
    }

    pub fn quit_requested(&self) -> bool {
        self.state.borrow().quit_requested
    }

    pub fn request_close(&mut self) {
        self.state.borrow_mut().close_requested = true;
    }

    pub fn close_requested(&self) -> bool {
        self.state.borrow().close_requested
    }

    /// Hide the application's windows until [`Self::show`] is called.
    pub fn hide(&mut self) {
        self.state.borrow_mut().hidden = true;
    }

    /// Show the application's windows after a previous [`Self::hide`] call.
    pub fn show(&mut self) {
        self.state.borrow_mut().hidden = false;
    }

    /// Whether the application has requested that its windows be hidden.
    pub fn is_hidden(&self) -> bool {
        self.state.borrow().hidden
    }

    pub fn windows(&self) -> WindowList {
        WindowList {
            count: self.state.borrow().windows.len(),
        }
    }

    pub fn on_window_closed(
        &mut self,
        handler: impl FnMut(&mut App, WindowId) + 'static,
    ) -> WindowClosedSubscription {
        let mut state = self.state.borrow_mut();
        state.next_window_closed_handler += 1;
        let id = state.next_window_closed_handler;
        state
            .window_closed_handlers
            .push((id, Rc::new(RefCell::new(handler))));
        WindowClosedSubscription {
            app: self.clone(),
            id,
            detached: false,
        }
    }

    pub fn window_closed(&mut self, id: WindowId) {
        let handlers = {
            let mut state = self.state.borrow_mut();
            if !state.windows.remove(&id) {
                return;
            }
            state
                .window_closed_handlers
                .iter()
                .map(|(_, handler)| handler.clone())
                .collect::<Vec<_>>()
        };
        for handler in handlers {
            handler.borrow_mut()(self, id);
        }
    }

    pub fn window_creation_failed(&mut self, id: WindowId) {
        self.state.borrow_mut().windows.remove(&id);
    }

    /// Queue a window request for a renderer-owned application boundary.
    ///
    /// `Any` is intentional here: core records lifecycle and description
    /// ownership without naming Winit or any concrete renderer window. The
    /// renderer that owns the request supplies its concrete window when it
    /// invokes these callbacks.
    pub fn enqueue_window(
        &mut self,
        options: WindowOptions,
        build: Box<dyn FnOnce(&mut App, &mut dyn Any) -> WindowRenderer>,
    ) -> Result<(), &'static str> {
        if self.quit_requested() {
            return Err("application is quitting");
        }
        if self.close_requested() {
            return Err("application is closing");
        }
        let id = self.reserve_window();
        self.state.borrow_mut().pending_windows.push(WindowRequest {
            id,
            options,
            build,
        });
        Ok(())
    }

    #[cfg(test)]
    fn open_window<V: crate::element::Render>(
        &mut self,
        options: WindowOptions,
        build_root_view: impl FnOnce(&mut crate::window::Window, &mut App) -> Entity<V> + 'static,
    ) -> Result<(), &'static str> {
        self.enqueue_window(
            options,
            Box::new(move |app, window| {
                let Some(window) = window.downcast_mut::<crate::window::Window>() else {
                    return WindowRenderer {
                        render: Box::new(|_, _| Description::new::<()>()),
                    };
                };
                let entity = build_root_view(window, app);
                WindowRenderer {
                    render: Box::new(move |app, window| {
                        let Some(window) = window.downcast_mut::<crate::window::Window>() else {
                            return Description::new::<()>();
                        };
                        entity.update((), |value, context| {
                            value
                                .render(window, context)
                                .into_description_in(window, app)
                        })
                    }),
                }
            }),
        )
    }

    pub fn reserve_window(&mut self) -> WindowId {
        let mut state = self.state.borrow_mut();
        state.next_window += 1;
        let id = WindowId(state.next_window);
        state.windows.insert(id);
        id
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
        let pending_tasks = self.pending_tasks.clone();
        pending_tasks.fetch_add(1, Ordering::AcqRel);
        let future = Abortable::new(future, registration);
        let spawn_result = self.foreground.borrow().spawner().spawn_local(async move {
            let _guard = PendingTaskGuard(pending_tasks);
            if let Ok(value) = future.await {
                let _ = sender.send(value);
            }
        });
        if spawn_result.is_err() {
            abort.abort();
            self.pending_tasks.fetch_sub(1, Ordering::AcqRel);
        }
        Task {
            receiver,
            abort: Some(abort),
            completed: false,
            cancelled: false,
        }
    }

    pub fn background_spawn<Fut, T>(&self, future: Fut) -> Task<T>
    where
        Fut: std::future::Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        let (abort, registration) = AbortHandle::new_pair();
        let (sender, receiver) = oneshot::channel();
        let pending_tasks = self.pending_tasks.clone();
        pending_tasks.fetch_add(1, Ordering::AcqRel);
        std::thread::spawn(move || {
            let _guard = PendingTaskGuard(pending_tasks);
            let result = futures::executor::block_on(Abortable::new(future, registration));
            if let Ok(value) = result {
                let _ = sender.send(value);
            }
        });
        Task {
            receiver,
            abort: Some(abort),
            completed: false,
            cancelled: false,
        }
    }
}

struct PendingTaskGuard(Arc<AtomicUsize>);

impl Drop for PendingTaskGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// A cancellable future returned by foreground and background work.
pub struct Task<T> {
    receiver: oneshot::Receiver<T>,
    abort: Option<AbortHandle>,
    completed: bool,
    cancelled: bool,
}

impl<T> Task<T> {
    pub fn ready(value: T) -> Self {
        let (sender, receiver) = oneshot::channel();
        let _ = sender.send(value);
        Self {
            receiver,
            abort: None,
            completed: false,
            cancelled: false,
        }
    }
    pub fn detach(mut self) {
        self.abort = None;
    }
    pub fn cancel(&mut self) {
        if !self.completed
            && let Some(abort) = self.abort.take()
        {
            abort.abort();
            self.cancelled = true;
        }
    }
    pub fn is_cancelled(&self) -> bool {
        self.cancelled
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
                self.cancelled = true;
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
    detached: bool,
}

impl Subscription {
    /// Keep this observer registered after the handle is dropped.
    pub fn detach(mut self) {
        self.detached = true;
    }
}

pub struct WindowClosedSubscription {
    app: App,
    id: u64,
    detached: bool,
}

impl WindowClosedSubscription {
    pub fn detach(mut self) {
        self.detached = true;
    }
}

impl Drop for WindowClosedSubscription {
    fn drop(&mut self) {
        if !self.detached {
            self.app
                .state
                .borrow_mut()
                .window_closed_handlers
                .retain(|(id, _)| *id != self.id);
        }
    }
}

impl EntityFactory for App {
    fn new<T: 'static>(&mut self, build: impl FnOnce(&mut Context<T>) -> T) -> Entity<T> {
        self.new_entity_with(build)
    }
}

impl<T> EntityFactory for Context<T> {
    fn new<U: 'static>(&mut self, build: impl FnOnce(&mut Context<U>) -> U) -> Entity<U> {
        self.app().new_entity_with(build)
    }
}
impl Drop for Subscription {
    fn drop(&mut self) {
        if !self.detached
            && let Some(observers) = self.app.state.borrow_mut().observers.get_mut(&self.entity)
        {
            observers.retain(|(id, _)| *id != self.id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{IntoElement, Render};
    use crate::window::MenuItem;
    use std::cell::Cell;
    use std::future::pending;
    use std::rc::Rc;

    crate::actions!(app_test, [Activate]);

    #[test]
    fn entity_identity_and_state_survive_high_frequency_updates() {
        let app = App::create();
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
            entity.update((), |value, context| {
                *value += 1;
                context.notify();
            });
        }
        assert_eq!(entity.entity_id(), identity);
        assert_eq!(*entity.read(&app), 10_000);
        assert_eq!(notifications.get(), 10_000);
        assert_eq!(app.drain_entity_invalidations(), vec![identity]);
    }

    #[test]
    fn entity_invalidations_coalesce_in_first_seen_order_and_drain_once() {
        let app = App::create();
        let first = app.new_entity(()).entity_id();
        let second = app.new_entity(()).entity_id();

        app.notify(first);
        app.notify(second);
        app.notify(first);

        assert_eq!(app.drain_entity_invalidations(), vec![first, second]);
        assert!(app.drain_entity_invalidations().is_empty());

        app.notify(first);
        assert!(app.has_pending_entity_invalidation(first));
        assert!(!app.has_pending_entity_invalidation(second));
    }

    #[test]
    fn draining_entity_invalidations_does_not_change_observer_drop_semantics() {
        let app = App::create();
        let entity = app.new_entity(());
        let notifications = Rc::new(Cell::new(0));
        let observed_notifications = notifications.clone();
        let subscription = entity.observe(move |_| {
            observed_notifications.set(observed_notifications.get() + 1);
        });

        app.notify(entity.entity_id());
        assert_eq!(notifications.get(), 1);
        assert_eq!(app.drain_entity_invalidations(), vec![entity.entity_id()]);

        drop(subscription);
        app.notify(entity.entity_id());
        assert_eq!(notifications.get(), 1);
        assert_eq!(app.drain_entity_invalidations(), vec![entity.entity_id()]);
    }

    #[test]
    fn foreground_tasks_run_when_the_app_pumps_and_return_errors_as_values() {
        let app = App::create();
        let mut task = app.spawn(async { Ok::<_, &'static str>(42_u32) });
        assert!(app.has_pending_tasks());
        app.run_pending_tasks();
        assert_eq!(futures::executor::block_on(&mut task), Ok(Ok(42)));
        assert!(!app.has_pending_tasks());
    }

    #[test]
    fn dropping_a_task_cancels_the_underlying_work() {
        let app = App::create();
        let mut task = app.spawn(pending::<()>());
        task.cancel();
        assert!(task.is_cancelled());
        app.run_pending_tasks();
        assert_eq!(
            futures::executor::block_on(&mut task),
            Err(TaskError::Cancelled)
        );
        assert!(!app.has_pending_tasks());
    }

    #[test]
    fn background_tasks_complete_and_release_the_pending_task_count() {
        let app = App::create();
        let (start_sender, start_receiver) = std::sync::mpsc::channel();
        let mut task = app.background_spawn(async move {
            if start_receiver.recv().is_err() {
                return 0_u32;
            }
            42_u32
        });
        assert!(app.has_pending_tasks());
        assert!(start_sender.send(()).is_ok());
        assert_eq!(futures::executor::block_on(&mut task), Ok(42));
        for _ in 0..1000 {
            if !app.has_pending_tasks() {
                break;
            }
            std::thread::yield_now();
        }
        assert!(!app.has_pending_tasks());
    }

    #[test]
    fn entity_constructors_can_capture_weak_self_and_create_owned_children() {
        struct Parent {
            child: Entity<u32>,
            self_handle: WeakEntity<Parent>,
        }

        let app = App::create();
        let parent = app.new_entity_with(|cx| Parent {
            child: cx.new(|_| 7_u32),
            self_handle: cx.entity().downgrade(),
        });
        let self_handle = parent.read(&app).self_handle.clone();
        let child = parent.read(&app).child.clone();

        assert_eq!(
            self_handle.upgrade().map(|entity| entity.entity_id()),
            Some(parent.entity_id())
        );
        assert_eq!(*child.read(&app), 7);
        parent.update((), |value, _| value.child.update((), |child, _| *child += 1));
        assert_eq!(*child.read(&app), 8);
    }

    #[test]
    fn weak_entities_fail_after_the_strong_identity_is_dropped() {
        let app = App::create();
        let entity = app.new_entity(1_u8);
        let weak = entity.downgrade();
        assert_eq!(weak.entity_id(), entity.entity_id());
        drop(entity);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn weak_entity_accessors_report_dropped_entities_and_update_live_entities() {
        let app = App::create();
        let entity = app.new_entity(1_u32);
        let weak = entity.downgrade();

        assert_eq!(weak.read_with(&app, |value, _| *value), Ok(1));
        let mut window = crate::window::Window::new();
        assert_eq!(
            weak.update_in(&mut window, |value, window, _| {
                window.activate();
                *value += 1;
                *value
            }),
            Ok(2)
        );
        assert!(window.is_active());
        assert_eq!(*entity.read(&app), 2);

        drop(entity);
        assert_eq!(weak.read_with(&app, |value, _| *value), Err(EntityError::Dropped));
        assert_eq!(
            weak.update_in(&mut window, |_, _, _| ()),
            Err(EntityError::Dropped)
        );
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
        let mut app = App::create();
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
        let mut app = App::create();
        assert!(!app.is_active());
        assert!(!app.quit_requested());
        assert!(app.menus().is_empty());

        app.activate(true);
        app.set_menus([Menu::new("File", vec![MenuItem::separator()])]);
        app.quit();

        assert!(app.is_active());
        assert!(app.quit_requested());
        assert!(app.close_requested());
        assert_eq!(app.menus()[0].name, "File");
    }

    #[test]
    fn window_requests_have_owned_ids_and_close_requests_stop_new_windows() {
        let mut app = App::create();
        assert!(app.windows().is_empty());
        assert!(
            app.open_window(WindowOptions::default(), |_, app| app.new_entity(TestRoot))
                .is_ok()
        );
        assert!(
            app.open_window(WindowOptions::default(), |_, app| app.new_entity(TestRoot))
                .is_ok()
        );
        let requests = app.take_window_requests();
        assert_eq!(requests.len(), 2);
        assert_ne!(requests[0].id, requests[1].id);
        assert_eq!(app.windows().len(), 2);

        app.request_close();
        assert!(app.close_requested());
        assert_eq!(
            app.open_window(WindowOptions::default(), |_, app| app.new_entity(TestRoot)),
            Err("application is closing")
        );
        app.window_creation_failed(requests[0].id);
        assert_eq!(app.windows().len(), 1);
    }

    #[test]
    fn window_closed_callbacks_run_in_order_once_and_can_observe_remaining_windows() {
        let mut app = App::create();
        let first = app.reserve_window();
        let second = app.reserve_window();
        let callbacks = Rc::new(RefCell::new(Vec::new()));
        let first_callbacks = callbacks.clone();
        let first_subscription = app.on_window_closed(move |app, id| {
            first_callbacks
                .borrow_mut()
                .push((1_u8, id, app.windows().len()));
        });
        let second_callbacks = callbacks.clone();
        let second_subscription = app.on_window_closed(move |_, id| {
            second_callbacks.borrow_mut().push((2_u8, id, 99));
        });

        app.window_closed(first);
        app.window_closed(first);
        assert_eq!(app.windows().len(), 1);
        assert_eq!(&*callbacks.borrow(), &[(1, first, 1), (2, first, 99)]);

        drop(first_subscription);
        drop(second_subscription);
        app.window_closed(second);
        assert!(app.windows().is_empty());
        assert_eq!(&*callbacks.borrow(), &[(1, first, 1), (2, first, 99)]);
    }

    struct TestRoot;

    impl Render for TestRoot {
        fn render(
            &mut self,
            _window: &mut crate::window::Window,
            _context: &mut Context<Self>,
        ) -> impl IntoElement + 'static {
            Description::new::<Self>()
        }
    }

    struct ContextTestRoot {
        expected_entity: Option<EntityId>,
        received_window: bool,
        received_context: bool,
    }

    impl Render for ContextTestRoot {
        fn render(
            &mut self,
            window: &mut crate::window::Window,
            context: &mut Context<Self>,
        ) -> impl IntoElement + 'static {
            window.activate();
            self.received_window = window.is_active();
            self.received_context = self.expected_entity == Some(context.entity().entity_id());
            Description::new::<Self>()
        }
    }

    #[test]
    fn open_window_queues_each_root_and_rejects_new_windows_after_quit() {
        let mut app = App::create();
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

    #[test]
    fn window_entry_point_renders_with_live_window_and_context() {
        let mut app = App::create();
        let root = app.new_entity(ContextTestRoot {
            expected_entity: None,
            received_window: false,
            received_context: false,
        });
        let entity_id = root.entity_id();
        root.update((), |value, _| value.expected_entity = Some(entity_id));

        app.open_window(WindowOptions::default(), {
            let root = root.clone();
            move |_, _| root
        })
        .expect("legacy window request should be accepted");

        let mut requests = app.take_window_requests();
        let request = requests.pop().expect("legacy request should be queued");
        let mut build_window = crate::window::Window::new();
        let mut renderer = (request.build)(
            &mut app,
            &mut build_window as &mut dyn std::any::Any,
        );
        let mut frame_window = crate::window::Window::new();
        let description = (renderer.render)(
            &mut app,
            &mut frame_window as &mut dyn std::any::Any,
        );

        assert_eq!(description.type_id(), std::any::TypeId::of::<ContextTestRoot>());
        let root = root.read(&app);
        assert!(root.received_window);
        assert!(root.received_context);
        assert!(frame_window.is_active());
    }
}

pub mod async_context;
pub mod context;
pub mod effects;
pub mod entity;
pub mod globals;
