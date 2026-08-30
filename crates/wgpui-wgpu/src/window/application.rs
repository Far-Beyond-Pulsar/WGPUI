//! Native application lifecycle for the retained WGPUI renderer.

use std::sync::Arc;

use wgpui_core::boundary::compositor::CompositeEntry;
use wgpui_core::invalidation::request::FrameSignals;
use wgpui_core::reconcile::description::Description;
use wgpui_core::reconcile::{ElementStateStore, StateKey, StateScope};
use wgpui_core::reconcile::plan::FrameStats;

use crate::render::draw::DrawMode;
use crate::render::frame::RenderTarget;
use crate::window::frame_loop::{FrameLoop, LoopInput};
use crate::window::resize_detector::ResizeDetector;
use crate::window::{Acquired, WindowError, WindowSurface};

/// Options used when creating the native window.
#[derive(Clone, Debug)]
pub struct WindowOptions {
    pub title: String,
    pub width: u32,
    pub height: u32,
    pub resizable: bool,
}

impl Default for WindowOptions {
    fn default() -> Self {
        Self {
            title: "WGPUI".to_string(),
            width: 800,
            height: 600,
            resizable: true,
        }
    }
}

/// A handle to the native window visible to the application callback.
pub struct Window {
    native: Arc<winit::window::Window>,
    scale_factor: f64,
    close_requested: bool,
    last_frame: Option<FrameReport>,
    state: ElementStateStore,
    state_frame: u64,
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
    pub fn scale_factor(&self) -> f64 {
        self.scale_factor
    }
    pub fn request_redraw(&self) {
        self.native.request_redraw();
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
pub struct Application<F> {
    options: WindowOptions,
    build: F,
    max_frames: Option<u64>,
}

impl<F> Application<F>
where
    F: FnMut(&mut Window) -> Description + 'static,
{
    pub fn new(options: WindowOptions, build: F) -> Self {
        Self {
            options,
            build,
            max_frames: None,
        }
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

    pub fn run(self) -> Result<(), ApplicationError> {
        let event_loop = event_loop()?;
        let mut handler = Handler {
            options: self.options,
            build: self.build,
            max_frames: self.max_frames,
            live: None,
            failure: None,
        };
        event_loop
            .run_app(&mut handler)
            .map_err(ApplicationError::from)?;
        handler.failure.map_or(Ok(()), Err)
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
}

struct Handler<F> {
    options: WindowOptions,
    build: F,
    max_frames: Option<u64>,
    live: Option<Live>,
    failure: Option<ApplicationError>,
}

impl<F> Handler<F>
where
    F: FnMut(&mut Window) -> Description + 'static,
{
    fn fail(&mut self, event_loop: &winit::event_loop::ActiveEventLoop, error: ApplicationError) {
        self.failure = Some(error);
        event_loop.exit();
    }

    fn draw(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let Some(live) = self.live.as_mut() else {
            return;
        };
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
        let description = (self.build)(&mut live.window);
        live.window.end_frame();
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
                if live.window.close_requested
                    || self.max_frames.is_some_and(|limit| live.frames >= limit)
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

impl<F> winit::application::ApplicationHandler for Handler<F>
where
    F: FnMut(&mut Window) -> Description + 'static,
{
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.live.is_some() {
            return;
        }
        let attributes = winit::window::Window::default_attributes()
            .with_title(self.options.title.clone())
            .with_resizable(self.options.resizable)
            .with_inner_size(winit::dpi::PhysicalSize::new(
                self.options.width,
                self.options.height,
            ));
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
        });
        if let Some(live) = self.live.as_ref() {
            live.window.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let Some(live) = self.live.as_mut() else {
            return;
        };
        match event {
            winit::event::WindowEvent::CloseRequested => event_loop.exit(),
            winit::event::WindowEvent::Resized(size) => {
                live.resizes.on_resize_event(size.width, size.height);
                live.window.request_redraw();
            }
            winit::event::WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                live.window.scale_factor = scale_factor;
                live.window.request_redraw();
            }
            winit::event::WindowEvent::RedrawRequested => self.draw(event_loop),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Some(live) = self.live.as_ref() {
            live.window.request_redraw();
        }
    }
}
