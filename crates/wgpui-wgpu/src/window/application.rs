//! Native application lifecycle for the retained WGPUI renderer.

use std::sync::Arc;

use wgpui_core::app::App;
use wgpui_core::boundary::Pixels;
use wgpui_core::boundary::compositor::CompositeEntry;
use wgpui_core::element::IntoElement;
use wgpui_core::geometry::{Bounds, Point, Rect, Size, WindowBounds, point, size};
use wgpui_core::invalidation::request::FrameSignals;
use wgpui_core::reconcile::description::Description;
use wgpui_core::reconcile::plan::FrameStats;
use wgpui_core::reconcile::{ElementStateStore, StateKey, StateScope};
pub use wgpui_core::window::WindowOptions;
use wgpui_core::window::{
    InputEvent, KeyDownEvent, KeyUpEvent, Modifiers, MouseButton as CoreMouseButton,
    MouseButtonState, MouseDownEvent, MouseMoveEvent, MouseUpEvent, ScrollWheelEvent,
};

use crate::render::draw::DrawMode;
use crate::debug::PerformanceDebug;
use crate::render::frame::RenderTarget;
use crate::window::frame_loop::{FrameLoop, InteractionRegistration, LoopInput};
use crate::window::resize_detector::ResizeDetector;
use crate::window::{Acquired, WindowError, WindowSurface};

fn initial_bounds(options: &WindowOptions) -> Bounds<Pixels> {
    let fallback = size(Pixels(options.width as f32), Pixels(options.height as f32));
    let Some(window_bounds) = options.window_bounds else {
        return Bounds::new(point(Pixels::ZERO, Pixels::ZERO), fallback);
    };
    let bounds = window_bounds.get_bounds();
    let size = if bounds.size.width.value() > 0.0 && bounds.size.height.value() > 0.0 {
        bounds.size
    } else {
        fallback
    };
    Bounds::new(bounds.origin, size)
}

/// A handle to the native window visible to the application callback.
type CloseHandler = Box<dyn FnMut(&mut Window) -> bool>;
type AppInitializer = Box<dyn FnOnce(&mut App)>;
type WindowBuildCallback = Box<dyn FnMut(&mut Window) -> Description>;

pub struct Window {
    native: Arc<winit::window::Window>,
    scale_factor: f64,
    close_requested: bool,
    last_frame: Option<FrameReport>,
    state: ElementStateStore,
    state_frame: u64,
    interaction: wgpui_core::window::Window,
    close_handler: Option<CloseHandler>,
    interaction_modifiers: Modifiers,
    mouse_buttons: MouseButtonState,
    cursor: [Pixels; 2],
    interactions: Vec<InteractionRegistration>,
    hovered_interaction: Option<usize>,
    pressed_interaction: Option<usize>,
    pressed_event: Option<MouseDownEvent>,
    hover_dirty_regions: Vec<Rect>,
    performance_debug: PerformanceDebug,
}

/// A clonable handle for scheduling work on a native window.
#[derive(Clone)]
pub struct WindowHandle(Arc<winit::window::Window>);

/// The monitor containing a window, when the platform reports one.
pub type DisplayId = winit::monitor::MonitorHandle;

impl Window {
    pub fn id(&self) -> winit::window::WindowId {
        self.native.id()
    }
    pub fn inner_size(&self) -> winit::dpi::PhysicalSize<u32> {
        self.native.inner_size()
    }
    pub fn bounds(&self) -> Bounds<Pixels> {
        let origin = self.native.outer_position().map_or(
            Point::new(Pixels::ZERO, Pixels::ZERO),
            |position| {
                point(
                    Pixels(position.x as f32 / self.scale_factor as f32),
                    Pixels(position.y as f32 / self.scale_factor as f32),
                )
            },
        );
        let size = self.inner_size();
        Bounds::new(
            origin,
            Size::new(
                Pixels(size.width as f32 / self.scale_factor as f32),
                Pixels(size.height as f32 / self.scale_factor as f32),
            ),
        )
    }
    pub fn resize(&self, size: Size<Pixels>) -> Option<winit::dpi::PhysicalSize<u32>> {
        let logical_size = winit::dpi::LogicalSize::new(size.width.value(), size.height.value());
        self.native.request_inner_size(logical_size)
    }
    pub fn scale_factor(&self) -> f64 {
        self.scale_factor
    }
    pub fn request_redraw(&self) {
        self.native.request_redraw();
    }

    pub fn take_hover_dirty_regions(&mut self) -> Vec<Rect> {
        std::mem::take(&mut self.hover_dirty_regions)
    }

    pub fn clear_hover_with_app(&mut self, app: &mut App) -> bool {
        let Some(index) = self.hovered_interaction.take() else {
            return false;
        };
        if let Some(interaction) = self.interactions.get(index) {
            self.hover_dirty_regions.push(interaction.bounds);
        }
        let event = InputEvent::MouseLeave(MouseMoveEvent {
            position: self.cursor,
            modifiers: self.modifiers(),
            buttons: self.mouse_buttons,
        });
        self.dispatch_interaction(index, &event, app)
    }
    /// Access opt-in visual performance diagnostics for this window.
    pub fn performance_debug(&mut self) -> &mut PerformanceDebug {
        &mut self.performance_debug
    }
    pub fn set_title(&self, title: &str) {
        self.native.set_title(title);
    }
    pub fn set_resizable(&self, resizable: bool) {
        self.native.set_resizable(resizable);
    }
    pub fn set_visible(&self, visible: bool) {
        self.native.set_visible(visible);
    }
    pub fn focus_window(&self) {
        self.native.focus_window();
    }
    pub fn set_minimized(&self, minimized: bool) {
        self.native.set_minimized(minimized);
    }
    pub fn set_maximized(&self, maximized: bool) {
        self.native.set_maximized(maximized);
    }
    pub fn is_maximized(&self) -> bool {
        self.native.is_maximized()
    }
    pub fn set_fullscreen(&self, fullscreen: bool) {
        self.native.set_fullscreen(
            fullscreen
                .then(|| winit::window::Fullscreen::Borderless(self.native.current_monitor())),
        );
    }
    pub fn outer_position(&self) -> Option<winit::dpi::PhysicalPosition<i32>> {
        self.native.outer_position().ok()
    }
    pub fn handle(&self) -> WindowHandle {
        WindowHandle(Arc::clone(&self.native))
    }
    pub fn current_monitor(&self) -> Option<DisplayId> {
        self.native.current_monitor()
    }
    pub fn close(&mut self) {
        self.close_requested = true;
    }
    pub fn on_close_requested(&mut self, handler: impl FnMut(&mut Self) -> bool + 'static) {
        self.close_handler = Some(Box::new(handler));
    }
    pub fn try_close(&mut self) -> bool {
        if self.close_requested {
            return true;
        }
        let allowed = self.close_handler.take().is_none_or(|mut handler| {
            let allowed = handler(self);
            self.close_handler = Some(handler);
            allowed
        });
        if allowed {
            self.close_requested = true;
        }
        allowed
    }
    pub fn interaction(&mut self) -> &mut wgpui_core::window::Window {
        &mut self.interaction
    }
    pub fn handle_input(&mut self, event: InputEvent) -> bool {
        self.interaction.handle_input(event)
    }
    pub fn handle_input_with_app(&mut self, event: InputEvent, app: &mut App) -> bool {
        if self.interactions.is_empty() {
            return self.interaction.handle_input(event);
        }
        match &event {
            InputEvent::MouseMove(mouse) => {
                let hit = self.hit_interaction(mouse.position);
                let mut handled = false;
                if self.hovered_interaction != hit {
                    if let Some(previous) = self.hovered_interaction
                        .and_then(|index| self.interactions.get(index))
                    {
                        self.hover_dirty_regions.push(previous.bounds);
                    }
                    if let Some(current) = hit.and_then(|index| self.interactions.get(index)) {
                        self.hover_dirty_regions.push(current.bounds);
                    }
                    if let Some(previous) = self.hovered_interaction {
                        handled |= self.dispatch_interaction(previous, &InputEvent::MouseLeave(*mouse), app);
                    }
                    if let Some(current) = hit {
                        handled |= self.dispatch_interaction(current, &InputEvent::MouseEnter(*mouse), app);
                    }
                    self.hovered_interaction = hit;
                }
                if let Some(current) = hit {
                    handled |= self.dispatch_interaction(current, &event, app);
                }
                handled
            }
            InputEvent::MouseDown(mouse) => {
                self.pressed_interaction = self.hit_interaction(mouse.position);
                self.pressed_event = Some(*mouse);
                self.pressed_interaction
                    .is_some_and(|index| self.dispatch_interaction(index, &event, app))
            }
            InputEvent::MouseUp(mouse) => {
                let pressed = self.pressed_interaction.take();
                let down = self.pressed_event.take().unwrap_or(MouseDownEvent {
                    button: mouse.button,
                    position: mouse.position,
                    modifiers: mouse.modifiers,
                    click_count: mouse.click_count,
                });
                let Some(index) = pressed else { return false };
                let mut handled = self.dispatch_interaction(index, &event, app);
                if self.hit_interaction(mouse.position) == Some(index) {
                    handled |= self.dispatch_interaction(index, &InputEvent::Click(
                        wgpui_core::window::ClickEvent::Mouse(
                            wgpui_core::window::MouseClickEvent {
                                down,
                                up: *mouse,
                            },
                        ),
                    ), app);
                }
                handled
            }
            _ => self.interaction.handle_input(event),
        }
    }
    fn hit_interaction(&self, position: [Pixels; 2]) -> Option<usize> {
        let point = [
            position[0].value() * self.scale_factor as f32,
            position[1].value() * self.scale_factor as f32,
        ];
        self.interactions
            .iter()
            .enumerate()
            .filter(|(_, registration)| {
                let bounds = registration.bounds;
                point[0] >= bounds.min_x
                    && point[0] < bounds.max_x
                    && point[1] >= bounds.min_y
                    && point[1] < bounds.max_y
            })
            .max_by_key(|(_, registration)| registration.order)
            .map(|(index, _)| index)
    }
    fn dispatch_interaction(&mut self, index: usize, event: &InputEvent, app: &mut App) -> bool {
        let mut interactions = std::mem::take(&mut self.interactions);
        let handled = interactions
            .get_mut(index)
            .is_some_and(|registration| registration.interaction.dispatch(event, &mut self.interaction, app).handled);
        self.interactions = interactions;
        handled
    }
    fn set_interactions(&mut self, interactions: Vec<InteractionRegistration>) {
        self.interactions = interactions;
        if self
            .hovered_interaction
            .is_some_and(|index| index >= self.interactions.len())
        {
            self.hovered_interaction = None;
        }
        self.pressed_interaction = None;
        self.pressed_event = None;
    }
    pub fn cursor_position(&self) -> [Pixels; 2] {
        self.cursor
    }
    fn modifiers(&self) -> Modifiers {
        self.interaction_modifiers
    }
    fn set_mouse_button(&mut self, button: CoreMouseButton, pressed: bool) {
        match button {
            CoreMouseButton::Left => self.mouse_buttons.left = pressed,
            CoreMouseButton::Right => self.mouse_buttons.right = pressed,
            CoreMouseButton::Middle => self.mouse_buttons.middle = pressed,
            CoreMouseButton::Other(_) => {}
        }
    }
    fn logical_point_for_physical(&mut self, x: f64, y: f64) -> [Pixels; 2] {
        let point = [
            Pixels((x / self.scale_factor) as f32),
            Pixels((y / self.scale_factor) as f32),
        ];
        self.cursor = point;
        point
    }
    pub fn last_frame(&self) -> Option<&FrameReport> {
        self.last_frame.as_ref()
    }

    /// Access state retained by an element scope. The store belongs to the
    /// logical window, not to a rendered layer, so cache boundaries cannot
    /// accidentally discard interactive state.
    pub fn use_state<T: 'static, R>(
        &mut self,
        scope: StateScope,
        initialise: impl FnOnce() -> T,
        access: impl FnOnce(&mut T) -> R,
    ) -> Option<R> {
        self.state.with_state(
            StateKey::new::<T>(scope),
            self.state_frame,
            initialise,
            access,
        )
    }

    pub fn begin_frame(&mut self) {
        self.state_frame = self.state_frame.wrapping_add(1);
    }

    pub fn end_frame(&mut self) -> usize {
        self.state.sweep(self.state_frame)
    }
}

impl WindowHandle {
    pub fn id(&self) -> winit::window::WindowId {
        self.0.id()
    }
    pub fn request_redraw(&self) {
        self.0.request_redraw();
    }
    pub fn set_title(&self, title: &str) {
        self.0.set_title(title);
    }
    pub fn focus_window(&self) {
        self.0.focus_window();
    }
    pub fn set_minimized(&self, minimized: bool) {
        self.0.set_minimized(minimized);
    }
    pub fn set_maximized(&self, maximized: bool) {
        self.0.set_maximized(maximized);
    }
}

/// Public state reported after a successfully presented frame.
#[derive(Clone, Debug, Default)]
pub struct FrameReport {
    pub frame_number: u64,
    pub retained: FrameStats,
    pub uploaded_bytes: u64,
    pub viewport_changed: bool,
    pub primitives_resident: u32,
}

/// Errors raised while creating or running an application.
#[derive(Debug)]
pub enum ApplicationError {
    EventLoop(winit::error::EventLoopError),
    Window(WindowError),
    Render(String),
    CreateWindow(String),
}

impl std::fmt::Display for ApplicationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EventLoop(error) => write!(formatter, "event loop: {error}"),
            Self::Window(error) => write!(formatter, "window: {error}"),
            Self::Render(error) => write!(formatter, "render: {error}"),
            Self::CreateWindow(error) => write!(formatter, "create window: {error}"),
        }
    }
}

impl std::error::Error for ApplicationError {}
impl From<winit::error::EventLoopError> for ApplicationError {
    fn from(error: winit::error::EventLoopError) -> Self {
        Self::EventLoop(error)
    }
}
impl From<WindowError> for ApplicationError {
    fn from(error: WindowError) -> Self {
        Self::Window(error)
    }
}

/// The native retained application.
pub struct NativeApplication<F, R> {
    options: WindowOptions,
    build: F,
    max_frames: Option<u64>,
    marker: std::marker::PhantomData<fn() -> R>,
}

impl<F, R> NativeApplication<F, R>
where
    F: FnMut(&mut Window) -> R + 'static,
    R: IntoElement,
{
    pub fn new(options: WindowOptions, build: F) -> Self {
        Self {
            options,
            build,
            max_frames: None,
            marker: std::marker::PhantomData,
        }
    }

    /// Explicit name for the direct retained-window entry point.
    ///
    /// `Application::new()` is the legacy GPUI constructor shape, while this
    /// crate's original direct constructor takes window options and a frame
    /// builder. Rust does not support overloaded associated functions, so the
    /// direct form remains available under this deliberate name.
    pub fn with_window(options: WindowOptions, build: F) -> Self {
        Self::new(options, build)
    }

    /// Run the direct window application after initializing a shared app.
    ///
    /// The initializer runs from Winit's `resumed` callback, before the first
    /// redraw, and the same `App` is retained for the lifetime of the window.
    /// Calling [`App::quit`] from the initializer or any retained callback
    /// exits the real event loop after the current frame.
    pub fn run_with_app(
        mut self,
        initialize: impl FnOnce(&mut App) + 'static,
    ) -> Result<(), ApplicationError> {
        let event_loop = event_loop()?;
        let mut handler = Handler {
            initial: Some((
                self.options,
                Box::new(move |window| (self.build)(window).into_description()),
            )),
            max_frames: self.max_frames,
            initialize: Some(Box::new(initialize)),
            live: Vec::new(),
            failure: None,
            app: App::new(),
        };
        event_loop
            .run_app(&mut handler)
            .map_err(ApplicationError::from)?;
        handler.failure.map_or(Ok(()), Err)
    }

    /// Stop automatically after a number of presented frames.
    ///
    /// This is useful for deterministic behavioral gates and command-line
    /// smoke tests. Without it, `run` owns the event loop until the caller
    /// closes the last window or the callback calls [`Window::close`].
    pub fn with_frame_limit(mut self, frames: u64) -> Self {
        self.max_frames = Some(frames);
        self
    }

    pub fn run(mut self) -> Result<(), ApplicationError> {
        let event_loop = event_loop()?;
        let mut handler = Handler {
            initial: Some((
                self.options,
                Box::new(move |window| (self.build)(window).into_description()),
            )),
            max_frames: self.max_frames,
            initialize: None,
            live: Vec::new(),
            failure: None,
            app: App::new(),
        };
        event_loop
            .run_app(&mut handler)
            .map_err(ApplicationError::from)?;
        handler.failure.map_or(Ok(()), Err)
    }
}

pub struct Application;

impl Application {
    pub fn new() -> Self {
        Self
    }

    pub fn run(self, initialize: impl FnOnce(&mut App) + 'static) -> Result<(), ApplicationError> {
        let event_loop = event_loop()?;
        let mut handler = Handler {
            initial: Some((
                WindowOptions::default(),
                Box::new(|_| Description::new::<()>()),
            )),
            max_frames: None,
            initialize: Some(Box::new(initialize)),
            app: App::new(),
            live: Vec::new(),
            failure: None,
        };
        event_loop
            .run_app(&mut handler)
            .map_err(ApplicationError::from)?;
        handler.failure.map_or(Ok(()), Err)
    }
}

impl Default for Application {
    fn default() -> Self {
        Self::new()
    }
}

fn event_loop() -> Result<winit::event_loop::EventLoop<()>, ApplicationError> {
    #[cfg(target_os = "windows")]
    {
        use winit::platform::windows::EventLoopBuilderExtWindows;
        winit::event_loop::EventLoop::builder()
            .with_any_thread(true)
            .build()
            .map_err(ApplicationError::from)
    }
    #[cfg(not(target_os = "windows"))]
    {
        winit::event_loop::EventLoop::new().map_err(ApplicationError::from)
    }
}

struct Live {
    surface: WindowSurface,
    context: crate::render::device::ComputeContext,
    window: Window,
    resizes: ResizeDetector,
    frame_loop: FrameLoop,
    mode: DrawMode,
    frames: u64,
    last_report: Option<FrameReport>,
    app: App,
    build: WindowBuildCallback,
}

struct Handler {
    initial: Option<(WindowOptions, WindowBuildCallback)>,
    max_frames: Option<u64>,
    initialize: Option<AppInitializer>,
    live: Vec<Live>,
    failure: Option<ApplicationError>,
    app: App,
}

impl Handler {
    fn fail(&mut self, event_loop: &winit::event_loop::ActiveEventLoop, error: ApplicationError) {
        self.failure = Some(error);
        event_loop.exit();
    }

    fn create_window(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        options: WindowOptions,
        build: WindowBuildCallback,
    ) -> Result<(), ApplicationError> {
        let attributes = winit::window::Window::default_attributes()
            .with_title(options.title.clone())
            .with_resizable(options.resizable)
            .with_visible(options.show);
        let bounds = initial_bounds(&options);
        let mut attributes = attributes.with_inner_size(winit::dpi::LogicalSize::new(
            bounds.size.width.value(),
            bounds.size.height.value(),
        ));
        if options.window_bounds.is_some() {
            attributes = attributes.with_position(winit::dpi::LogicalPosition::new(
                bounds.origin.x.value(),
                bounds.origin.y.value(),
            ));
        }
        let native = Arc::new(
            event_loop
                .create_window(attributes)
                .map_err(|error| ApplicationError::CreateWindow(error.to_string()))?,
        );
        let scale_factor = native.scale_factor();
        if let Some(window_bounds) = options.window_bounds {
            match window_bounds {
                WindowBounds::Windowed(_) => {}
                WindowBounds::Maximized(_) => native.set_maximized(true),
                WindowBounds::Fullscreen(_) => native.set_fullscreen(Some(
                    winit::window::Fullscreen::Borderless(native.current_monitor()),
                )),
            }
        }
        if options.focus {
            native.focus_window();
        }
        let (surface, context) = WindowSurface::new(Arc::clone(&native))?;
        let (width, height) = surface.size();
        let mut resizes = ResizeDetector::new();
        resizes.seed(width, height);
        let mode = DrawMode::best_available(context.indirect);
        let window = Window {
            native,
            scale_factor,
            close_requested: false,
            last_frame: None,
            state: ElementStateStore::new(),
            state_frame: 0,
            interaction: wgpui_core::window::Window::new(),
            close_handler: None,
            interaction_modifiers: Modifiers::default(),
            mouse_buttons: MouseButtonState::default(),
            cursor: [Pixels::ZERO, Pixels::ZERO],
            interactions: Vec::new(),
            hovered_interaction: None,
            pressed_interaction: None,
            pressed_event: None,
            hover_dirty_regions: Vec::new(),
            performance_debug: PerformanceDebug::default(),
        };
        self.live.push(Live {
            frame_loop: FrameLoop::new(&context.device),
            surface,
            context,
            window,
            resizes,
            mode,
            frames: 0,
            last_report: None,
            app: self.app.clone(),
            build,
        });
        if let Some(live) = self.live.last() {
            live.window.request_redraw();
        }
        Ok(())
    }

    fn create_pending_windows(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        loop {
            let requests = self.app.take_window_requests();
            if requests.is_empty() {
                break;
            }
            let mut first_error = None;
            for request in requests {
                let mut renderer =
                    (request.build)(&mut wgpui_core::window::Window::new(), &mut self.app);
                let mut app = self.app.clone();
                let build = Box::new(move |window: &mut Window| {
                    (renderer.render)(&mut window.interaction, &mut app)
                });
                if let Err(error) = self.create_window(event_loop, request.options, build)
                    && first_error.is_none()
                {
                    first_error = Some(error);
                }
            }
            if let Some(error) = first_error {
                self.fail(event_loop, error);
                return;
            }
        }
    }

    fn draw(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
    ) {
        let Some(index) = self
            .live
            .iter()
            .position(|live| live.window.id() == window_id)
        else {
            return;
        };
        let all_other_windows_reached_limit = self.max_frames.is_some_and(|limit| {
            self.live
                .iter()
                .enumerate()
                .all(|(other_index, live)| other_index == index || live.frames >= limit)
        });
        let live = &mut self.live[index];
        if let Some((width, height)) = live.resizes.take_pending() {
            live.surface.resize(&live.context.device, width, height);
        }
        let texture = match live.surface.acquire(&live.context.device) {
            Acquired::Frame(texture) => texture,
            Acquired::Skipped(_) => {
                live.window.request_redraw();
                return;
            }
            Acquired::Lost => {
                self.fail(
                    event_loop,
                    ApplicationError::Render("surface image was lost after recovery".to_string()),
                );
                return;
            }
        };
        let view = texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let (width, height) = live.surface.size();
        live.window.begin_frame();
        let description = (live.build)(&mut live.window);
        live.window.end_frame();
        live.frame_loop
            .set_performance_debug(*live.window.performance_debug());
        live.frame_loop.set_scale_factor(live.window.scale_factor);
        let target = RenderTarget {
            view: &view,
            width,
            height,
            clear: wgpu::Color::BLACK,
            source: Some(&texture.texture),
        };
        let signals = FrameSignals::new();
        let result = live.frame_loop.draw(
            &live.context.device,
            &live.context.queue,
            description,
            &LoopInput {
                atlas: None,
                target: &target,
                mode: live.mode,
                signals: &signals,
                composites: &[] as &[CompositeEntry],
            },
        );
        match result {
            Ok(frame) => {
                live.window.set_interactions(frame.interactions);
                live.frames += 1;
                let report = FrameReport {
                    frame_number: live.frames,
                    retained: frame.reconciled,
                    uploaded_bytes: frame.uploaded_bytes,
                    viewport_changed: frame.viewport_changed,
                    primitives_resident: frame.frame.primitives_resident,
                };
                live.window.last_frame = Some(report.clone());
                live.last_report = Some(report);
                live.surface.present(&live.context.queue, texture);
                if live.window.close_requested || live.app.quit_requested() {
                    self.live.remove(index);
                    if self.live.is_empty() {
                        event_loop.exit();
                    }
                } else if self.max_frames.is_some_and(|limit| live.frames >= limit)
                    && all_other_windows_reached_limit
                {
                    event_loop.exit();
                } else {
                    live.window.request_redraw();
                }
            }
            Err(error) => self.fail(event_loop, ApplicationError::Render(error.to_string())),
        }
    }
}

impl winit::application::ApplicationHandler for Handler {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if !self.live.is_empty() {
            return;
        }
        if let Some(initialize) = self.initialize.take() {
            initialize(&mut self.app);
        }
        self.create_pending_windows(event_loop);
        if let Some((options, build)) = self.initial.take()
            && let Err(error) = self.create_window(event_loop, options, build)
        {
            self.fail(event_loop, error);
        }
    }

    /*
        let attributes = winit::window::Window::default_attributes()
            .with_title(self.options.title.clone())
            .with_resizable(self.options.resizable)
            .with_visible(self.options.show);
        let initial_bounds = initial_bounds(&self.options);
        let mut attributes = attributes.with_inner_size(winit::dpi::LogicalSize::new(
            initial_bounds.size.width.value(),
            initial_bounds.size.height.value(),
        ));
        if self.options.window_bounds.is_some() {
            attributes = attributes.with_position(winit::dpi::LogicalPosition::new(
                initial_bounds.origin.x.value(),
                initial_bounds.origin.y.value(),
            ));
        }
        let native = match event_loop.create_window(attributes) {
            Ok(window) => Arc::new(window),
            Err(error) => {
                self.fail(
                    event_loop,
                    ApplicationError::CreateWindow(error.to_string()),
                );
                return;
            }
        };
        let scale_factor = native.scale_factor();
        if let Some(window_bounds) = self.options.window_bounds {
            match window_bounds {
                WindowBounds::Windowed(_) => {}
                WindowBounds::Maximized(_) => native.set_maximized(true),
                WindowBounds::Fullscreen(_) => native.set_fullscreen(Some(
                    winit::window::Fullscreen::Borderless(native.current_monitor()),
                )),
            }
        }
        if self.options.focus {
            native.focus_window();
        }
        let (surface, context) = match WindowSurface::new(Arc::clone(&native)) {
            Ok(pair) => pair,
            Err(error) => {
                self.fail(event_loop, error.into());
                return;
            }
        };
        let (width, height) = surface.size();
        let mut resizes = ResizeDetector::new();
        resizes.seed(width, height);
        let mode = DrawMode::best_available(context.indirect);
        let window = Window {
            native,
            scale_factor,
            close_requested: false,
            last_frame: None,
            state: ElementStateStore::new(),
            state_frame: 0,
            interaction: wgpui_core::window::Window::new(),
            close_handler: None,
            interaction_modifiers: Modifiers::default(),
            mouse_buttons: MouseButtonState::default(),
            cursor: [Pixels::ZERO, Pixels::ZERO],
            interactions: Vec::new(),
            hovered_interaction: None,
            pressed_interaction: None,
            pressed_event: None,
            hover_dirty_regions: Vec::new(),
            performance_debug: PerformanceDebug::default(),
        };
        self.live = Some(Live {
            frame_loop: FrameLoop::new(&context.device),
            surface,
            context,
            window,
            resizes,
            mode,
            frames: 0,
            last_report: None,
            app: self.app.clone(),
        });
        if let Some(initialize) = self.initialize.take()
            && let Some(live) = self.live.as_mut()
        {
            let mut app = live.app.clone();
            initialize(&mut app);
        }
        if let Some(live) = self.live.as_ref() {
            live.window.request_redraw();
        }
    }
    */

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let Some(index) = self
            .live
            .iter()
            .position(|live| live.window.id() == window_id)
        else {
            return;
        };
        let live = &mut self.live[index];
        match event {
            winit::event::WindowEvent::CloseRequested => {
                if live.window.try_close() {
                    self.live.remove(index);
                    if self.live.is_empty() {
                        event_loop.exit();
                    }
                }
            }
            winit::event::WindowEvent::Resized(size) => {
                live.resizes.on_resize_event(size.width, size.height);
                live.window.request_redraw();
            }
            winit::event::WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                live.window.scale_factor = scale_factor;
                live.window.request_redraw();
            }
            winit::event::WindowEvent::ModifiersChanged(modifiers) => {
                live.window.interaction_modifiers = modifiers_from_winit(modifiers.state());
            }
            winit::event::WindowEvent::KeyboardInput { event, .. } => {
                let key = key_name(&event);
                let input = if event.state == winit::event::ElementState::Pressed {
                    InputEvent::KeyDown(KeyDownEvent {
                        key,
                        modifiers: live.window.modifiers(),
                        repeat: event.repeat,
                    })
                } else {
                    InputEvent::KeyUp(KeyUpEvent {
                        key,
                        modifiers: live.window.modifiers(),
                    })
                };
                if live.window.handle_input_with_app(input, &mut live.app) {
                    live.window.request_redraw();
                }
            }
            winit::event::WindowEvent::CursorMoved { position, .. } => {
                let point = live
                    .window
                    .logical_point_for_physical(position.x, position.y);
                let event = InputEvent::MouseMove(MouseMoveEvent {
                    position: point,
                    modifiers: live.window.modifiers(),
                    buttons: live.window.mouse_buttons,
                });
                live.window.handle_input_with_app(event, &mut live.app);
                for region in live.window.take_hover_dirty_regions() {
                    live.frame_loop.mark_interaction_dirty(region);
                }
                live.window.request_redraw();
            }
            winit::event::WindowEvent::MouseInput { state, button, .. } => {
                let point = live.window.cursor_position();
                let button = core_mouse_button(button);
                live.window
                    .set_mouse_button(button, state == winit::event::ElementState::Pressed);
                let event = if state == winit::event::ElementState::Pressed {
                    InputEvent::MouseDown(MouseDownEvent {
                        button,
                        position: point,
                        modifiers: live.window.modifiers(),
                        click_count: 1,
                    })
                } else {
                    InputEvent::MouseUp(MouseUpEvent {
                        button,
                        position: point,
                        modifiers: live.window.modifiers(),
                        click_count: 1,
                    })
                };
                live.window.handle_input_with_app(event, &mut live.app);
                live.window.request_redraw();
            }
            winit::event::WindowEvent::MouseWheel { delta, .. } => {
                let delta = match delta {
                    winit::event::MouseScrollDelta::LineDelta(x, y) => [x * 16.0, y * 16.0],
                    winit::event::MouseScrollDelta::PixelDelta(point) => {
                        [point.x as f32, point.y as f32]
                    }
                };
                let event = InputEvent::Scroll(ScrollWheelEvent {
                    position: live.window.cursor_position(),
                    delta,
                    modifiers: live.window.modifiers(),
                });
                live.window.handle_input_with_app(event, &mut live.app);
                live.window.request_redraw();
            }
            winit::event::WindowEvent::RedrawRequested => self.draw(event_loop, window_id),
            winit::event::WindowEvent::CursorEntered { .. } => live.window.request_redraw(),
            winit::event::WindowEvent::CursorLeft { .. } => {
                if live.window.clear_hover_with_app(&mut live.app) {
                    for region in live.window.take_hover_dirty_regions() {
                        live.frame_loop.mark_interaction_dirty(region);
                    }
                }
                live.window.interaction.clear_hover();
                live.window.request_redraw();
            }
            event => log::debug!("unhandled native window event: {event:?}"),
        }
    }

    fn about_to_wait(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        self.create_pending_windows(event_loop);
        for live in &self.live {
            live.window.request_redraw();
        }
    }
}

fn modifiers_from_winit(modifiers: winit::keyboard::ModifiersState) -> Modifiers {
    Modifiers {
        shift: modifiers.shift_key(),
        control: modifiers.control_key(),
        alt: modifiers.alt_key(),
        command: modifiers.super_key(),
    }
}

fn core_mouse_button(button: winit::event::MouseButton) -> CoreMouseButton {
    match button {
        winit::event::MouseButton::Left => CoreMouseButton::Left,
        winit::event::MouseButton::Right => CoreMouseButton::Right,
        winit::event::MouseButton::Middle => CoreMouseButton::Middle,
        winit::event::MouseButton::Back | winit::event::MouseButton::Forward => {
            CoreMouseButton::Other(0)
        }
        winit::event::MouseButton::Other(value) => CoreMouseButton::Other(value),
    }
}

fn key_name(event: &winit::event::KeyEvent) -> String {
    use winit::keyboard::{Key, NamedKey};
    match &event.logical_key {
        Key::Character(character) => character.to_string().to_ascii_lowercase(),
        Key::Named(named) => match named {
            NamedKey::Enter => "enter",
            NamedKey::Escape => "escape",
            NamedKey::Tab => "tab",
            NamedKey::Backspace => "backspace",
            NamedKey::Delete => "delete",
            NamedKey::ArrowLeft => "left",
            NamedKey::ArrowRight => "right",
            NamedKey::ArrowUp => "up",
            NamedKey::ArrowDown => "down",
            NamedKey::Space => "space",
            _ => return format!("{named:?}").to_ascii_lowercase(),
        }
        .to_string(),
        _ => "unknown".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpui_core::geometry::{point, px, size};

    #[test]
    fn window_bounds_override_fallback_size_and_preserve_position() {
        let options = WindowOptions {
            window_bounds: Some(WindowBounds::Windowed(Bounds::new(
                point(px(120.0), px(80.0)),
                size(px(640.0), px(480.0)),
            ))),
            ..WindowOptions::default()
        };
        assert_eq!(initial_bounds(&options).origin, point(px(120.0), px(80.0)));
        assert_eq!(initial_bounds(&options).size, size(px(640.0), px(480.0)));
    }

    #[test]
    fn zero_window_bounds_use_the_explicit_fallback_size() {
        let options = WindowOptions {
            width: 320,
            height: 240,
            window_bounds: Some(WindowBounds::Windowed(Bounds::default())),
            ..WindowOptions::default()
        };
        assert_eq!(initial_bounds(&options).size, size(px(320.0), px(240.0)));
    }
}
