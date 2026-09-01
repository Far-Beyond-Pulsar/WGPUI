//! Native application lifecycle for the retained WGPUI renderer.

use std::any::{Any, TypeId};
use std::cell::Cell;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

use wgpui_core::app::App;
use wgpui_core::boundary::Pixels;
use wgpui_core::boundary::compositor::CompositeEntry;
use wgpui_core::element::Element as CoreElement;
pub use wgpui_core::element::IntoElement;
use wgpui_core::geometry::{Bounds, Point, Rect, Size, WindowBounds, point, size};
use wgpui_core::invalidation::request::FrameSignals;
use wgpui_core::patch::emit::{Emission, EmitContext};
use wgpui_core::patch::primitive::{Path, Quad};
use wgpui_core::reconcile::description::Description;
use wgpui_core::reconcile::plan::FrameStats;
use wgpui_core::reconcile::{ElementStateStore, StateKey, StateScope};
pub use wgpui_core::window::WindowOptions;
use wgpui_core::window::{
    AnimationClock, ClipboardItem, DragData, FocusEvent, FocusHandle, FocusId, ImeEvent, InputEvent,
    KeyDownEvent,
    KeyUpEvent, Modifiers, ModifiersChangedEvent, MouseButton as CoreMouseButton,
    MouseButtonState, MouseDownEvent, MouseMoveEvent, MouseUpEvent, ScrollWheelEvent,
    TextInputEvent,
};
use wgpui_core::window::{TimerHandle, TimerState};
use wgpui_core::window::{
    WindowAppearance, WindowBackgroundAppearance, WindowDecorations, WindowKind,
};
use wgpui_http_client::{AppHttpClientExt, BoxedHttpClient};

use crate::debug::PerformanceDebug;
use crate::render::draw::DrawMode;
use crate::render::frame::RenderTarget;
use crate::render::surface_registry::SurfaceRegistry;
use crate::window::frame_loop::{FrameLoop, InteractionRegistration, LoopInput};
use crate::window::resize_detector::ResizeDetector;
use crate::window::surface::WgpuSurfaceHandle;
use crate::window::{Acquired, WindowError, WindowSurface};
use wgpui_widgets::assets::Resource;
use wgpui_widgets::assets::{Asset, AssetRegistry, ImageCacheError, RenderImage};
use wgpui_widgets::img::ImgBuilder;
use wgpui_widgets::styled::Styled;

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
type WindowBuildCallback = Box<dyn FnMut(&mut Window, &mut App) -> Description>;

thread_local! {
    static CALLBACK_WINDOW: Cell<*mut Window> = const { Cell::new(std::ptr::null_mut()) };
}

/// Adapt a public-window callback to the backend-neutral widget callback ABI.
///
/// The adapter is only valid while the callback is dispatched by a live WGPU
/// window. Keeping this conversion here avoids making either core or widgets
/// depend on the concrete backend window.
pub fn public_window_callback<E, F>(mut callback: F) -> impl FnMut(
    &E,
    &mut wgpui_core::window::Window,
    &mut App,
) + 'static
where
    F: FnMut(&E, &mut Window, &mut App) + 'static,
{
    move |event, _core_window, app| {
        CALLBACK_WINDOW.with(|slot| {
            let pointer = slot.get();
            assert!(!pointer.is_null(), "public window callback outside WGPU dispatch");
            // SAFETY: the dispatch guard installs the pointer from the
            // currently borrowed Window and restores it before returning.
            unsafe { callback(event, &mut *pointer, app) };
        });
    }
}

/// Adapt an entity listener whose callback receives the public WGPU window.
pub fn public_listener<T: 'static, E: ?Sized>(
    context: &wgpui_core::app::Context<T>,
    callback: impl Fn(&mut T, &E, &mut Window, &mut wgpui_core::app::Context<T>) + 'static,
) -> impl Fn(&E, &mut wgpui_core::window::Window, &mut App) + 'static {
    let entity = context.entity().downgrade();
    move |event, core_window, _app| {
        CALLBACK_WINDOW.with(|slot| {
            let pointer = slot.get();
            assert!(!pointer.is_null(), "public listener outside WGPU dispatch");
            let Some(entity) = entity.upgrade() else {
                return;
            };
            // SAFETY: see `public_window_callback`.
            entity.update_in(core_window, |value, _window, context| unsafe {
                callback(value, event, &mut *pointer, context);
            });
        });
    }
}

/// Adapt an entity listener whose event is passed by value, such as the
/// boolean delivered by hover transitions.
pub fn public_value_listener<T: 'static, E: 'static>(
    context: &wgpui_core::app::Context<T>,
    mut callback: impl FnMut(&mut T, E, &mut Window, &mut wgpui_core::app::Context<T>) + 'static,
) -> impl FnMut(E, &mut wgpui_core::window::Window, &mut App) + 'static {
    let entity = context.entity().downgrade();
    move |event, core_window, _app| {
        CALLBACK_WINDOW.with(|slot| {
            let pointer = slot.get();
            assert!(!pointer.is_null(), "public value listener outside WGPU dispatch");
            let Some(entity) = entity.upgrade() else {
                return;
            };
            // SAFETY: see `public_window_callback`.
            entity.update_in(core_window, |value, _window, context| unsafe {
                callback(value, event, &mut *pointer, context);
            });
        });
    }
}

enum ImmediatePaint {
    Quad(Quad),
    Path(Path),
}

pub struct Window {
    native: Arc<winit::window::Window>,
    gpu_adapter: wgpu::Adapter,
    gpu_device: wgpu::Device,
    gpu_queue: wgpu::Queue,
    surface_registry: Arc<SurfaceRegistry>,
    scale_factor: f64,
    close_requested: Arc<AtomicBool>,
    background_appearance: WindowBackgroundAppearance,
    decorations: bool,
    last_frame: Option<FrameReport>,
    state: ElementStateStore,
    state_frame: u64,
    state_entities: std::collections::HashMap<TypeId, Box<dyn Any>>,
    immediate_paints: Vec<ImmediatePaint>,
    interaction: wgpui_core::window::Window,
    close_handler: Option<CloseHandler>,
    interaction_modifiers: Modifiers,
    mouse_buttons: MouseButtonState,
    cursor: [Pixels; 2],
    cursor_inside: bool,
    interactions: Vec<InteractionRegistration>,
    hovered_interaction: Option<usize>,
    pressed_interaction: Option<usize>,
    pressed_event: Option<MouseDownEvent>,
    active_drag: Option<DragData>,
    drag_hovered: Option<usize>,
    hover_dirty_regions: Vec<Rect>,
    performance_debug: PerformanceDebug,
    animation_clock: AnimationClock,
    animation_frame_requested: AtomicBool,
    last_click: Option<ClickState>,
}

#[derive(Copy, Clone)]
struct ClickState {
    button: CoreMouseButton,
    position: [Pixels; 2],
    time: Instant,
    count: u32,
}

/// A clonable handle for scheduling work on a native window.
#[derive(Clone)]
pub struct WindowHandle {
    native: Arc<winit::window::Window>,
    close_requested: Arc<AtomicBool>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClipboardError {
    message: String,
}

impl std::fmt::Display for ClipboardError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ClipboardError {}

impl From<arboard::Error> for ClipboardError {
    fn from(error: arboard::Error) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}

/// The monitor containing a window, when the platform reports one.
pub type DisplayId = winit::monitor::MonitorHandle;

/// A stateless component rendered with the public WGPU window.
pub trait RenderOnce: 'static + Sized {
    /// Build the element tree for this owned component.
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement;
}

/// A retained view rendered with the public WGPU window.
pub trait Render: 'static + Sized {
    /// Build the element tree for the current frame.
    fn render(
        &mut self,
        window: &mut Window,
        cx: &mut wgpui_core::app::Context<Self>,
    ) -> impl IntoElement;
}

/// Element wrapper generated for [`RenderOnce`] components.
pub struct Component<C: RenderOnce> {
    component: C,
}

/// Element wrapper for an entity implementing the public [`Render`] trait.
pub struct EntityView<T: Render> {
    entity: wgpui_core::app::Entity<T>,
}

pub fn entity_view<T: Render>(entity: wgpui_core::app::Entity<T>) -> EntityView<T> {
    EntityView { entity }
}

impl<T: Render> CoreElement for EntityView<T> {
    fn into_description(self) -> Description {
        Description::deferred(move |window, app| {
            let Some(window) = window.downcast_mut::<Window>() else {
                return Description::new::<Self>();
            };
            let description = self.entity.update((), |value, context| {
                let _scope = wgpui_core::element::enter_contextual_render_scope();
                let element = value.render(window, context);
                let app = app.clone();
                IntoElement::into_description_in(element, window.interaction_mut(), &app)
            });
            description
        })
    }
}

impl<C: RenderOnce> Component<C> {
    /// Wrap an owned component in an element.
    pub fn new(component: C) -> Self {
        Self { component }
    }

    /// Borrow the component before it is consumed during lowering.
    pub fn component(&self) -> &C {
        &self.component
    }
}

impl<C: RenderOnce> CoreElement for Component<C> {
    fn into_description(self) -> Description {
        Description::deferred(move |window, app| {
            let Some(window) = window.downcast_mut::<Window>() else {
                return Description::new::<Self>();
            };
            let element = self.component.render(window, app);
            let app = app.clone();
            IntoElement::into_description_in(element, window.interaction_mut(), &app)
        })
    }
}

/// Public application extension that queues a WGPU-window root view.
pub trait AppWindowExt {
    fn open_window<V: Render>(
        &mut self,
        options: WindowOptions,
        build_root_view: impl FnOnce(&mut Window, &mut App) -> wgpui_core::app::Entity<V> + 'static,
    ) -> Result<(), &'static str>;
}

/// Render a public WGPU view into a description for the current frame.
pub fn render_description<R: Render>(
    view: &mut R,
    window: &mut Window,
    cx: &mut wgpui_core::app::Context<R>,
) -> Description {
    let element = {
        let _scope = wgpui_core::element::enter_contextual_render_scope();
        view.render(window, cx)
    };
    let mut app = cx.app().clone();
    let description = IntoElement::into_description_in(element, window.interaction_mut(), &app);
    let description = description.resolve_deferred(window, &mut app);
    let description = description.resolve_deferred_core_window(window.interaction_mut(), &mut app);
    description.resolve_deferred(window, &mut app)
}

impl AppWindowExt for App {
    fn open_window<V: Render>(
        &mut self,
        options: WindowOptions,
        build_root_view: impl FnOnce(&mut Window, &mut App) -> wgpui_core::app::Entity<V> + 'static,
    ) -> Result<(), &'static str> {
        self.enqueue_window(
            options,
            Box::new(move |app, window| {
                let Some(window) = window.downcast_mut::<Window>() else {
                    return wgpui_core::app::WindowRenderer {
                        render: Box::new(|_, _| Description::new::<()>()),
                    };
                };
                let entity = build_root_view(window, app);
                wgpui_core::app::WindowRenderer {
                    render: Box::new(move |app, window| {
                        let Some(window) = window.downcast_mut::<Window>() else {
                            return Description::new::<()>();
                        };
                        let lower = entity.update((), |value, context| {
                            let element = {
                                let _scope =
                                    wgpui_core::element::enter_contextual_render_scope();
                                value.render(window, context)
                            };
                            Box::new(move |window: &mut Window, app: &mut App| {
                                IntoElement::into_description_in(
                                    element,
                                    window.interaction_mut(),
                                    app,
                                )
                            }) as Box<dyn FnOnce(&mut Window, &mut App) -> Description>
                        });
                        lower(window, app)
                    }),
                }
            }),
        )
    }
}

impl Window {
    /// Access the underlying Winit window for platform-specific integration.
    ///
    /// WGPUI APIs should use this wrapper wherever possible; this escape hatch
    /// is for integrations that specifically require Winit functionality.
    pub fn winit_window(&self) -> &winit::window::Window {
        &self.native
    }

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

    pub fn request_animation_frame(&self) {
        // The redraw is delivered by Winit on a later event-loop turn. The
        // flag makes repeated requests coalesce and keeps this request alive
        // until that redraw is consumed.
        self.animation_frame_requested.store(true, Ordering::Release);
        self.request_redraw();
    }

    /// Queue a retained quad for the current frame.
    pub fn paint_quad(&mut self, quad: Quad) {
        self.immediate_paints.push(ImmediatePaint::Quad(quad));
        self.request_redraw();
    }

    /// Queue a retained, already tessellated path for the current frame.
    pub fn paint_path<C>(&mut self, path: Path, color: C)
    where
        C: Into<[f32; 4]>,
    {
        self.immediate_paints
            .push(ImmediatePaint::Path(path.with_color(color.into())));
        self.request_redraw();
    }

    pub fn refresh(&mut self) {
        tracing::warn!("full repaint forced by Window::refresh; targeted invalidation should generally be preferred");
        self.interaction.refresh();
        self.request_redraw();
    }

    pub fn schedule_timer(&mut self, delay: Duration) -> TimerHandle {
        let timer = self.interaction.schedule_timer(delay);
        self.request_redraw();
        timer
    }

    pub fn cancel_timer(&mut self, timer: TimerHandle) -> bool {
        self.interaction.cancel_timer(timer)
    }

    pub fn timer_state(&self, timer: TimerHandle) -> Option<TimerState> {
        self.interaction.timer_state(timer)
    }

    pub fn next_timer_deadline(&self) -> Option<Instant> {
        self.interaction.next_timer_deadline()
    }

    pub fn take_due_timers(&mut self, now: Instant) -> Vec<wgpui_core::window::TimerId> {
        self.interaction.take_due_timers(now)
    }

    pub fn take_hover_dirty_regions(&mut self) -> Vec<Rect> {
        std::mem::take(&mut self.hover_dirty_regions)
    }

    pub fn clear_hover_with_app(&mut self, app: &mut App) -> bool {
        self.cursor_inside = false;
        let index = self.hovered_interaction.take();
        if let Some(index) = index
            && let Some(interaction) = self.interactions.get(index)
        {
            self.hover_dirty_regions.push(interaction.bounds);
        }
        let event = InputEvent::MouseLeave(MouseMoveEvent {
            position: self.cursor,
            modifiers: self.modifiers(),
            buttons: self.mouse_buttons,
        });
        let mut handled =
            index.is_some_and(|index| self.dispatch_interaction_bubbled(index, &event, app));
        if let (Some(index), Some(data)) = (self.drag_hovered.take(), self.active_drag.clone()) {
            self.mark_interaction_dirty_for(index);
            handled |= self.dispatch_drag_hover(index, false, &data, app);
        }
        handled
    }
    /// Access opt-in visual performance diagnostics for this window.
    pub fn performance_debug(&mut self) -> &mut PerformanceDebug {
        &mut self.performance_debug
    }

    /// Create a producer-owned texture set that can be placed in this window's
    /// retained scene with `WgpuSurface`.
    pub fn create_wgpu_surface(
        &self,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Result<WgpuSurfaceHandle, WindowError> {
        if width == 0 || height == 0 {
            return Err(WindowError::InvalidSurfaceSize { width, height });
        }
        let allowed_usages = self
            .gpu_adapter
            .get_texture_format_features(format)
            .allowed_usages;
        let required_usages =
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        if !allowed_usages.contains(required_usages) {
            return Err(WindowError::UnsupportedSurfaceFormat(format));
        }
        Ok(WgpuSurfaceHandle::new(
            Arc::clone(&self.surface_registry),
            &self.gpu_device,
            &self.gpu_queue,
            width,
            height,
            format,
            {
                let native = Arc::clone(&self.native);
                Arc::new(move || {
                    native.request_redraw();
                })
            },
        ))
    }

    pub fn write_to_clipboard(&self, item: ClipboardItem) -> Result<(), ClipboardError> {
        let text = item.text().ok_or_else(|| ClipboardError {
            message: "native clipboard currently supports text items only".to_string(),
        })?;
        let mut clipboard = arboard::Clipboard::new().map_err(ClipboardError::from)?;
        clipboard.set_text(text).map_err(ClipboardError::from)
    }

    pub fn read_from_clipboard(&self) -> Result<Option<ClipboardItem>, ClipboardError> {
        let mut clipboard = arboard::Clipboard::new().map_err(ClipboardError::from)?;
        match clipboard.get_text() {
            Ok(text) => Ok(Some(ClipboardItem::new_string(text))),
            Err(arboard::Error::ContentNotAvailable) => Ok(None),
            Err(error) => Err(ClipboardError::from(error)),
        }
    }

    pub fn set_ime_allowed(&self, allowed: bool) {
        self.native.set_ime_allowed(allowed);
    }
    pub fn set_title(&self, title: &str) {
        self.native.set_title(title);
    }
    pub fn appearance(&self) -> WindowAppearance {
        match self.native.theme() {
            Some(winit::window::Theme::Dark) => WindowAppearance::Dark,
            Some(winit::window::Theme::Light) | None => WindowAppearance::Light,
        }
    }
    pub fn window_bounds(&self) -> WindowBounds {
        let bounds = self.bounds();
        if self.native.fullscreen().is_some() {
            WindowBounds::Fullscreen(bounds)
        } else if self.native.is_maximized() {
            WindowBounds::Maximized(bounds)
        } else {
            WindowBounds::Windowed(bounds)
        }
    }
    pub fn content_size(&self) -> Size<Pixels> {
        self.bounds().size
    }
    pub fn is_active(&self) -> bool {
        self.native.has_focus()
    }
    pub fn set_background_appearance(&mut self, appearance: WindowBackgroundAppearance) {
        self.native
            .set_transparent(appearance != WindowBackgroundAppearance::Opaque);
        self.native
            .set_blur(appearance == WindowBackgroundAppearance::Blurred);
        #[cfg(target_os = "windows")]
        {
            use winit::platform::windows::{BackdropType, WindowExtWindows};
            let backdrop = match appearance {
                WindowBackgroundAppearance::Opaque | WindowBackgroundAppearance::Transparent => {
                    BackdropType::None
                }
                WindowBackgroundAppearance::Blurred => BackdropType::TransientWindow,
                WindowBackgroundAppearance::MicaBackdrop => BackdropType::MainWindow,
                WindowBackgroundAppearance::MicaAltBackdrop => BackdropType::TabbedWindow,
            };
            self.native.set_system_backdrop(backdrop);
        }
        self.background_appearance = appearance;
    }
    pub fn background_appearance(&self) -> WindowBackgroundAppearance {
        self.background_appearance
    }
    pub fn set_decorations(&mut self, decorations: bool) {
        self.native.set_decorations(decorations);
        self.decorations = decorations;
    }
    pub fn window_decorations(&self) -> WindowDecorations {
        window_decorations_for(self.decorations)
    }

    /// Record the requested client inset without claiming that Winit applied
    /// it. Winit does not expose a cross-platform client-area inset API; code
    /// that needs platform-specific non-client geometry should use
    /// [`Self::winit_window`].
    pub fn set_client_inset(&mut self, inset: Pixels) {
        log::warn!(
            "client window insets are not supported by the WGPU/Winit backend; requested inset: {inset:?}"
        );
    }
    pub fn has_decorations(&self) -> bool {
        self.decorations
    }
    pub fn set_resizable(&self, resizable: bool) {
        self.native.set_resizable(resizable);
    }
    pub fn set_minimizable(&self, minimizable: bool) {
        let mut buttons = self.native.enabled_buttons();
        buttons.set(winit::window::WindowButtons::MINIMIZE, minimizable);
        self.native.set_enabled_buttons(buttons);
    }
    pub fn set_min_inner_size(&self, size: Option<Size<Pixels>>) {
        self.native.set_min_inner_size(
            size.map(|size| winit::dpi::LogicalSize::new(size.width.value(), size.height.value())),
        );
    }
    pub fn set_outer_position(&self, position: Point<Pixels>) {
        self.native
            .set_outer_position(winit::dpi::LogicalPosition::new(
                position.x.value(),
                position.y.value(),
            ));
    }
    pub fn set_visible(&self, visible: bool) {
        self.native.set_visible(visible);
    }
    pub fn focus_window(&self) {
        self.native.focus_window();
    }
    pub fn minimize(&self) {
        self.native.set_minimized(true);
    }
    pub fn zoom(&self) {
        self.native.set_maximized(!self.native.is_maximized());
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
        WindowHandle {
            native: Arc::clone(&self.native),
            close_requested: Arc::clone(&self.close_requested),
        }
    }
    pub fn current_monitor(&self) -> Option<DisplayId> {
        self.native.current_monitor()
    }
    pub fn close(&mut self) {
        self.close_requested.store(true, Ordering::Release);
        self.native.request_redraw();
    }
    pub fn remove_window(&mut self) {
        self.close();
    }
    pub fn close_requested(&self) -> bool {
        self.close_requested.load(Ordering::Acquire)
    }
    pub fn on_close_requested(&mut self, handler: impl FnMut(&mut Self) -> bool + 'static) {
        self.close_handler = Some(Box::new(handler));
    }
    pub fn try_close(&mut self) -> bool {
        if self.close_requested() {
            return true;
        }
        let allowed = self.close_handler.take().is_none_or(|mut handler| {
            let allowed = handler(self);
            self.close_handler = Some(handler);
            allowed
        });
        if allowed {
            self.close();
        }
        allowed
    }
    pub fn interaction(&mut self) -> &mut wgpui_core::window::Window {
        &mut self.interaction
    }

    /// Borrow the backend-neutral interaction window used while lowering
    /// descriptions and dispatching input.
    pub fn interaction_mut(&mut self) -> &mut wgpui_core::window::Window {
        &mut self.interaction
    }

    pub fn focus<A>(&mut self, handle: &FocusHandle, _cx: A) -> bool {
        self.interaction.focus(handle)
    }

    pub fn focus_next<A>(&mut self, _cx: A) -> Option<FocusId> {
        self.interaction.focus_next()
    }

    pub fn focus_prev<A>(&mut self, _cx: A) -> Option<FocusId> {
        self.interaction.focus_previous()
    }

    /// Spawn work through the entity context associated with this window.
    ///
    /// The native window does not own an executor, so this delegates to the
    /// context rather than creating a second task runtime.
    pub fn spawn<T, F, R>(
        &self,
        cx: &wgpui_core::app::Context<T>,
        make: F,
    ) -> wgpui_core::app::Task<R>
    where
        T: 'static,
        F: AsyncFnOnce(&wgpui_core::app::Context<T>) -> R + 'static,
        R: 'static,
    {
        cx.spawn(async move |_entity, context| make(context).await)
    }
    pub fn handle_input(&mut self, event: InputEvent) -> bool {
        self.interaction.handle_input(event)
    }
    pub fn handle_input_with_app(&mut self, event: InputEvent, app: &mut App) -> bool {
        CALLBACK_WINDOW.with(|slot| {
            let previous = slot.replace(self as *mut Window);
            let handled = self.handle_input_with_app_inner(event, app);
            slot.set(previous);
            handled
        })
    }

    fn handle_input_with_app_inner(&mut self, event: InputEvent, app: &mut App) -> bool {
        if self.interactions.is_empty() {
            return self.interaction.handle_input(event);
        }
        match &event {
            InputEvent::KeyDown(key) if key.key.eq_ignore_ascii_case("tab") => {
                let handled = self.interaction.handle_input(event);
                handled | self.apply_focus_transition(app)
            }
            InputEvent::KeyDown(key) => {
                let mut handled = false;
                if let Some(index) = self.focused_interaction() {
                    let result = self.dispatch_interaction_bubbled_result(index, &event, app);
                    handled |= result.handled;
                    if !result.propagate {
                        return handled | self.apply_focus_transition(app);
                    }
                }
                let action = self.interaction.resolve_action(key);
                if let Some(action) = action {
                    if let Some(index) = self.interactions.iter().position(|registration| {
                        registration
                            .focus
                            .is_some_and(|focus| Some(focus.id()) == self.interaction.focused())
                    }) {
                        handled = self.dispatch_action_bubbled(index, &*action, app);
                    }
                    if !handled {
                        handled = app.dispatch_action(&*action);
                    }
                }
                handled | self.apply_focus_transition(app)
            }
            InputEvent::KeyUp(_) | InputEvent::TextInput(_) | InputEvent::Ime(_) => {
                let handled = self
                    .focused_interaction()
                    .is_some_and(|index| self.dispatch_interaction_bubbled(index, &event, app));
                handled | self.apply_focus_transition(app)
            }
            InputEvent::ModifiersChanged(_) => {
                let handled = self
                    .focused_interaction()
                    .is_some_and(|index| self.dispatch_interaction_bubbled(index, &event, app));
                handled | self.apply_focus_transition(app)
            }
            InputEvent::MouseMove(mouse) => {
                self.cursor_inside = true;
                let hit = self.hit_interaction(mouse.position);
                let mut handled = false;
                if self.active_drag.is_none()
                    && self.mouse_buttons.left
                    && let Some(index) = self.pressed_interaction
                    && let Some(registration) = self.interactions.get_mut(index)
                    && let Some(data) = registration.interaction.drag_source()
                {
                    let data = data.with_position(mouse.position);
                    registration
                        .interaction
                        .start_drag(&data, &mut self.interaction, app);
                    self.active_drag = Some(data);
                }
                if self.hovered_interaction != hit {
                    if let Some(previous) = self
                        .hovered_interaction
                        .and_then(|index| self.interactions.get(index))
                    {
                        self.hover_dirty_regions.push(previous.bounds);
                    }
                    if let Some(current) = hit.and_then(|index| self.interactions.get(index)) {
                        self.hover_dirty_regions.push(current.bounds);
                    }
                    if let Some(previous) = self.hovered_interaction {
                        handled |= self.dispatch_interaction_bubbled(
                            previous,
                            &InputEvent::MouseLeave(*mouse),
                            app,
                        );
                    }
                    if let Some(current) = hit {
                        handled |= self.dispatch_interaction_bubbled(
                            current,
                            &InputEvent::MouseEnter(*mouse),
                            app,
                        );
                    }
                    self.hovered_interaction = hit;
                }
                if let Some(current) = hit {
                    handled |= self.dispatch_interaction_bubbled(current, &event, app);
                }
                if let Some(data) = self.active_drag.clone() {
                    let previous_drag = self.drag_hovered;
                    if previous_drag != hit {
                        if let Some(previous) = previous_drag {
                            handled |= self.dispatch_drag_hover(previous, false, &data, app);
                        }
                        if let Some(current) = hit {
                            handled |= self.dispatch_drag_hover(current, true, &data, app);
                        }
                        self.drag_hovered = hit;
                    }
                    self.active_drag = Some(data.with_position(mouse.position));
                }
                handled
            }
            InputEvent::MouseDown(mouse) => {
                self.pressed_interaction = self.hit_interaction(mouse.position);
                self.pressed_event = Some(*mouse);
                let mut handled = false;
                if let Some(index) = self.pressed_interaction {
                    if mouse.is_focusing()
                        && let Some(focus) = self.interactions[index].focus
                        && self.interaction.focus_id(focus.id(), false)
                    {
                        self.mark_interaction_dirty_for(index);
                        handled |= self.apply_focus_transition(app);
                    }
                    handled |= self.dispatch_interaction_bubbled(index, &event, app);
                }
                handled
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
                if let Some(data) = self.active_drag.take() {
                    if let Some(previous) = self.drag_hovered.take() {
                        self.mark_interaction_dirty_for(previous);
                        let mut handled = self.dispatch_drag_hover(previous, false, &data, app);
                        if self.hit_interaction(mouse.position) == Some(previous) {
                            handled |= self.dispatch_drop(previous, &data, app);
                        }
                        return handled;
                    }
                    return false;
                }
                self.mark_interaction_dirty_for(index);
                let mut handled = self.dispatch_interaction_bubbled(index, &event, app);
                if self.hit_interaction(mouse.position) == Some(index) {
                    handled |= self.dispatch_interaction_bubbled(
                        index,
                        &InputEvent::Click(wgpui_core::window::ClickEvent::Mouse(
                            wgpui_core::window::MouseClickEvent { down, up: *mouse },
                        )),
                        app,
                    );
                }
                handled
            }
            InputEvent::Scroll(scroll) => {
                let mut remaining = scroll.delta;
                let mut handled = false;
                for index in self.hit_interactions(scroll.position) {
                    let mut bubbled = *scroll;
                    bubbled.delta = remaining;
                    let event = InputEvent::Scroll(bubbled);
                    handled |= self.dispatch_interaction_bubbled(index, &event, app);
                    if handled {
                        break;
                    }
                    let Some(scroll_root) = self
                        .interactions
                        .get(index)
                        .and_then(|registration| registration.scroll_root.as_ref())
                    else {
                        continue;
                    };
                    let consumed = scroll_root.handle.scroll_by(wgpui_core::geometry::Point::new(
                        Pixels(remaining[0]),
                        Pixels(remaining[1]),
                    ));
                    remaining[0] -= consumed.x.value();
                    remaining[1] -= consumed.y.value();
                    handled |= consumed.x != Pixels::ZERO || consumed.y != Pixels::ZERO;
                    if remaining == [0.0, 0.0] {
                        break;
                    }
                }
                handled
            }
            _ => self.interaction.handle_input(event),
        }
    }
    fn hit_interactions(&self, position: [Pixels; 2]) -> Vec<usize> {
        let point = [
            position[0].value() * self.scale_factor as f32,
            position[1].value() * self.scale_factor as f32,
        ];
        let mut hits: Vec<_> = self
            .interactions
            .iter()
            .enumerate()
            .filter(|(_, registration)| {
                let bounds = registration.bounds;
                point[0] >= bounds.min_x
                    && point[0] < bounds.max_x
                    && point[1] >= bounds.min_y
                    && point[1] < bounds.max_y
            })
            .map(|(index, _)| index)
            .collect();
        hits.sort_unstable_by_key(|index| std::cmp::Reverse(self.interactions[*index].order));
        hits
    }
    fn hit_interaction(&self, position: [Pixels; 2]) -> Option<usize> {
        self.hit_interactions(position).into_iter().next()
    }
    fn dispatch_interaction(&mut self, index: usize, event: &InputEvent, app: &mut App) -> bool {
        self.dispatch_interaction_result(index, event, app).handled
    }
    fn dispatch_interaction_bubbled(
        &mut self,
        index: usize,
        event: &InputEvent,
        app: &mut App,
    ) -> bool {
        let mut current = Some(index);
        let mut handled = false;
        while let Some(index) = current {
            let parent = self
                .interactions
                .get(index)
                .and_then(|registration| registration.parent);
            let result = self.dispatch_interaction_result(index, event, app);
            handled |= result.handled;
            if !result.propagate {
                break;
            }
            current = parent;
        }
        handled
    }
    fn dispatch_interaction_bubbled_result(
        &mut self,
        index: usize,
        event: &InputEvent,
        app: &mut App,
    ) -> wgpui_core::window::EventResult {
        let mut current = Some(index);
        let mut result = wgpui_core::window::EventResult::IGNORED;
        while let Some(index) = current {
            let parent = self.interactions.get(index).and_then(|registration| registration.parent);
            let current_result = self.dispatch_interaction_result(index, event, app);
            if current_result.handled {
                result.handled = true;
            }
            result.propagate = current_result.propagate;
            if !current_result.propagate {
                break;
            }
            current = parent;
        }
        result
    }
    fn dispatch_action_bubbled(
        &mut self,
        index: usize,
        action: &dyn wgpui_core::Action,
        app: &mut App,
    ) -> bool {
        let mut current = Some(index);
        let mut handled = false;
        while let Some(index) = current {
            let parent = self
                .interactions
                .get(index)
                .and_then(|registration| registration.parent);
            let result = self.interactions.get_mut(index).map_or(
                wgpui_core::window::EventResult::IGNORED,
                |registration| {
                    registration
                        .interaction
                        .dispatch_action(action, &mut self.interaction, app)
                },
            );
            handled |= result.handled;
            if !result.propagate {
                break;
            }
            current = parent;
        }
        handled
    }
    fn dispatch_drag_hover(
        &mut self,
        index: usize,
        hovered: bool,
        data: &DragData,
        app: &mut App,
    ) -> bool {
        let mut current = Some(index);
        let mut handled = false;
        while let Some(index) = current {
            let parent = self
                .interactions
                .get(index)
                .and_then(|registration| registration.parent);
            let result = self.interactions.get_mut(index).map_or(
                wgpui_core::window::EventResult::IGNORED,
                |registration| {
                    registration.interaction.dispatch_drag_hover(
                        hovered,
                        data,
                        &mut self.interaction,
                        app,
                    )
                },
            );
            handled |= result.handled;
            if !result.propagate {
                break;
            }
            current = parent;
        }
        handled
    }
    fn dispatch_drop(&mut self, index: usize, data: &DragData, app: &mut App) -> bool {
        let mut current = Some(index);
        let mut handled = false;
        while let Some(index) = current {
            let parent = self
                .interactions
                .get(index)
                .and_then(|registration| registration.parent);
            let result = self.interactions.get_mut(index).map_or(
                wgpui_core::window::EventResult::IGNORED,
                |registration| {
                    registration
                        .interaction
                        .dispatch_drop(data, &mut self.interaction, app)
                },
            );
            handled |= result.handled;
            if !result.propagate {
                break;
            }
            current = parent;
        }
        handled
    }
    fn apply_focus_transition(&mut self, app: &mut App) -> bool {
        let Some(transition) = self.interaction.take_focus_transition() else {
            return false;
        };
        if transition.from == transition.to {
            let Some(id) = transition.to else {
                return false;
            };
            let Some(index) = self
                .interactions
                .iter()
                .position(|registration| registration.focus.is_some_and(|focus| focus.id() == id))
            else {
                return false;
            };
            self.mark_interaction_dirty_for(index);
            return self.dispatch_interaction_bubbled(
                index,
                &InputEvent::Focus(FocusEvent {
                    focused: true,
                    visible: transition.visible,
                }),
                app,
            );
        }
        let mut handled = false;
        for (id, focused, visible) in [
            (transition.from, false, false),
            (transition.to, true, transition.visible),
        ] {
            let Some(id) = id else {
                continue;
            };
            let Some(index) = self
                .interactions
                .iter()
                .position(|registration| registration.focus.is_some_and(|focus| focus.id() == id))
            else {
                continue;
            };
            self.mark_interaction_dirty_for(index);
            handled |= self.dispatch_interaction_bubbled(
                index,
                &InputEvent::Focus(FocusEvent { focused, visible }),
                app,
            );
        }
        self.native.set_ime_allowed(self.interaction.focused().is_some());
        handled
    }
    fn mark_interaction_dirty_for(&mut self, index: usize) {
        if let Some(registration) = self.interactions.get(index) {
            self.hover_dirty_regions.push(registration.bounds);
        }
    }
    fn dispatch_interaction_result(
        &mut self,
        index: usize,
        event: &InputEvent,
        app: &mut App,
    ) -> wgpui_core::window::EventResult {
        let mut interactions = std::mem::take(&mut self.interactions);
        let result = interactions.get_mut(index).map_or(
            wgpui_core::window::EventResult::IGNORED,
            |registration| {
                registration
                    .interaction
                    .dispatch(event, &mut self.interaction, app)
            },
        );
        self.interactions = interactions;
        result
    }
    fn set_interactions(&mut self, interactions: Vec<InteractionRegistration>, app: &mut App) {
        let previous_address = self
            .hovered_interaction
            .and_then(|index| self.interactions.get(index))
            .map(|registration| registration.address);
        let previous_bounds = self
            .hovered_interaction
            .and_then(|index| self.interactions.get(index))
            .map(|registration| registration.bounds);
        let previous_pressed_address = self
            .pressed_interaction
            .and_then(|index| self.interactions.get(index))
            .map(|registration| registration.address);
        let previous_drag_address = self
            .drag_hovered
            .and_then(|index| self.interactions.get(index))
            .map(|registration| registration.address);
        let pressed_event = self.pressed_event;
        let mut previous_interactions = std::mem::replace(&mut self.interactions, interactions);
        self.interaction.retain_focus_handles(
            self.interactions
                .iter()
                .filter_map(|registration| registration.focus),
        );
        for (order, registration) in self.interactions.iter().enumerate() {
            if let Some(focus) = registration.focus {
                self.interaction
                    .register_focus_handle_ordered(focus, order as u64);
            }
        }
        let focused = self.interaction.focused();
        let focus_visible = self.interaction.focus_manager().focus_visible();
        for index in 0..self.interactions.len() {
            if self.interactions[index]
                .focus
                .is_some_and(|focus| Some(focus.id()) == focused)
            {
                self.dispatch_interaction(
                    index,
                    &InputEvent::Focus(FocusEvent {
                        focused: true,
                        visible: focus_visible,
                    }),
                    app,
                );
            }
        }
        let hit = self
            .cursor_inside
            .then(|| self.hit_interaction(self.cursor))
            .flatten();
        let hit_address = hit
            .and_then(|index| self.interactions.get(index))
            .map(|registration| registration.address);
        let hit_bounds = hit
            .and_then(|index| self.interactions.get(index))
            .map(|registration| registration.bounds);
        self.pressed_interaction = previous_pressed_address.and_then(|address| {
            self.interactions
                .iter()
                .position(|registration| registration.address == address)
        });
        self.pressed_event = self.pressed_interaction.and(pressed_event);
        let drag_hit = self.active_drag.as_ref().and(hit);
        let drag_hit_address = drag_hit
            .and_then(|index| self.interactions.get(index))
            .map(|registration| registration.address);
        let old_drag_index = previous_drag_address.and_then(|address| {
            previous_interactions
                .iter()
                .position(|registration| registration.address == address)
        });
        if previous_address != hit_address {
            if let Some(bounds) = previous_bounds {
                self.hover_dirty_regions.push(bounds);
            }
            if let Some(index) = hit
                && let Some(registration) = self.interactions.get(index)
            {
                self.hover_dirty_regions.push(registration.bounds);
            }
            let mouse = MouseMoveEvent {
                position: self.cursor,
                modifiers: self.modifiers(),
                buttons: self.mouse_buttons,
            };
            if let Some(old_address) = previous_address
                && let Some(old_index) = previous_interactions
                    .iter()
                    .position(|registration| registration.address == old_address)
                && let Some(registration) = previous_interactions.get_mut(old_index)
            {
                registration.interaction.dispatch(
                    &InputEvent::MouseLeave(mouse),
                    &mut self.interaction,
                    app,
                );
            }
            if let Some(index) = hit {
                self.dispatch_interaction(index, &InputEvent::MouseEnter(mouse), app);
            }
        } else if let Some(index) = hit {
            self.dispatch_interaction(
                index,
                &InputEvent::MouseEnter(MouseMoveEvent {
                    position: self.cursor,
                    modifiers: self.modifiers(),
                    buttons: self.mouse_buttons,
                }),
                app,
            );
        }
        if previous_address == hit_address && previous_bounds != hit_bounds {
            if let Some(bounds) = previous_bounds {
                self.hover_dirty_regions.push(bounds);
            }
            if let Some(bounds) = hit_bounds {
                self.hover_dirty_regions.push(bounds);
            }
        }
        self.hovered_interaction = hit;
        if self.active_drag.is_some() && previous_drag_address != drag_hit_address {
            if let (Some(index), Some(data)) = (old_drag_index, self.active_drag.clone())
                && let Some(registration) = previous_interactions.get_mut(index)
            {
                registration.interaction.dispatch_drag_hover(
                    false,
                    &data,
                    &mut self.interaction,
                    app,
                );
            }
            if let Some(index) = drag_hit
                && let Some(data) = self.active_drag.clone()
            {
                self.dispatch_drag_hover(index, true, &data, app);
            }
            self.drag_hovered = drag_hit;
        } else {
            self.drag_hovered = previous_drag_address.and_then(|address| {
                self.interactions
                    .iter()
                    .position(|registration| registration.address == address)
            });
        }
    }
    fn focused_interaction(&self) -> Option<usize> {
        let focused = self.interaction.focused()?;
        self.interactions
            .iter()
            .position(|registration| registration.focus.is_some_and(|focus| focus.id() == focused))
    }
    pub fn handle_window_focus(&mut self, focused: bool, app: &mut App) -> bool {
        if focused {
            self.native.set_ime_allowed(self.interaction.focused().is_some());
            return false;
        }
        let mut handled = self.clear_hover_with_app(app);
        self.pressed_interaction = None;
        self.pressed_event = None;
        self.active_drag = None;
        self.drag_hovered = None;
        if self.interaction.blur() {
            handled |= self.apply_focus_transition(app);
        }
        self.native.set_ime_allowed(false);
        handled
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
    fn next_click_count(&mut self, button: CoreMouseButton, position: [Pixels; 2]) -> u32 {
        let now = Instant::now();
        let count = self.last_click.map_or(1, |last| {
            let distance_x = position[0].value() - last.position[0].value();
            let distance_y = position[1].value() - last.position[1].value();
            let close = distance_x.mul_add(distance_x, distance_y * distance_y) <= 16.0;
            if last.button == button
                && now.duration_since(last.time) <= Duration::from_millis(500)
                && close
            {
                last.count.saturating_add(1)
            } else {
                1
            }
        });
        self.last_click = Some(ClickState {
            button,
            position,
            time: now,
            count,
        });
        count
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
    pub fn use_state<T: 'static>(
        &mut self,
        app: &mut App,
        build: impl FnOnce(&mut Self, &mut App) -> T,
    ) -> wgpui_core::app::Entity<T> {
        if let Some(entity) = self
            .state_entities
            .get(&TypeId::of::<T>())
            .and_then(|value| value.downcast_ref::<wgpui_core::app::Entity<T>>())
        {
            return entity.clone();
        }
        let value = build(self, app);
        let entity = app.new_entity(value);
        self.state_entities
            .insert(TypeId::of::<T>(), Box::new(entity.clone()));
        entity
    }

    pub fn use_state_in_scope<T: 'static, R>(
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
    /// Access the underlying Winit window for platform-specific integration.
    pub fn winit_window(&self) -> &winit::window::Window {
        &self.native
    }

    pub fn id(&self) -> winit::window::WindowId {
        self.native.id()
    }
    pub fn inner_size(&self) -> winit::dpi::PhysicalSize<u32> {
        self.native.inner_size()
    }
    pub fn outer_position(&self) -> Option<winit::dpi::PhysicalPosition<i32>> {
        self.native.outer_position().ok()
    }
    pub fn scale_factor(&self) -> f64 {
        self.native.scale_factor()
    }
    pub fn appearance(&self) -> WindowAppearance {
        match self.native.theme() {
            Some(winit::window::Theme::Dark) => WindowAppearance::Dark,
            Some(winit::window::Theme::Light) | None => WindowAppearance::Light,
        }
    }
    pub fn has_focus(&self) -> bool {
        self.native.has_focus()
    }
    pub fn is_visible(&self) -> Option<bool> {
        self.native.is_visible()
    }
    pub fn is_maximized(&self) -> bool {
        self.native.is_maximized()
    }
    pub fn is_fullscreen(&self) -> bool {
        self.native.fullscreen().is_some()
    }
    pub fn request_redraw(&self) {
        self.native.request_redraw();
    }
    pub fn set_title(&self, title: &str) {
        self.native.set_title(title);
    }
    pub fn focus_window(&self) {
        self.native.focus_window();
    }
    pub fn set_visible(&self, visible: bool) {
        self.native.set_visible(visible);
    }
    pub fn set_resizable(&self, resizable: bool) {
        self.native.set_resizable(resizable);
    }
    pub fn set_decorations(&self, decorations: bool) {
        self.native.set_decorations(decorations);
    }
    pub fn set_minimizable(&self, minimizable: bool) {
        let mut buttons = self.native.enabled_buttons();
        buttons.set(winit::window::WindowButtons::MINIMIZE, minimizable);
        self.native.set_enabled_buttons(buttons);
    }
    pub fn set_min_inner_size(&self, size: Option<Size<Pixels>>) {
        self.native.set_min_inner_size(
            size.map(|size| winit::dpi::LogicalSize::new(size.width.value(), size.height.value())),
        );
    }
    pub fn set_outer_position(&self, position: Point<Pixels>) {
        self.native
            .set_outer_position(winit::dpi::LogicalPosition::new(
                position.x.value(),
                position.y.value(),
            ));
    }
    pub fn set_minimized(&self, minimized: bool) {
        self.native.set_minimized(minimized);
    }
    pub fn set_maximized(&self, maximized: bool) {
        self.native.set_maximized(maximized);
    }
    pub fn set_fullscreen(&self, fullscreen: bool) {
        self.native.set_fullscreen(
            fullscreen
                .then(|| winit::window::Fullscreen::Borderless(self.native.current_monitor())),
        );
    }
    pub fn close(&self) {
        self.close_requested.store(true, Ordering::Release);
        self.native.request_redraw();
    }
    pub fn close_requested(&self) -> bool {
        self.close_requested.load(Ordering::Acquire)
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
    http_client: Option<BoxedHttpClient>,
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
            http_client: None,
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
        let mut app = App::create();
        if let Some(client) = self.http_client {
            app.set_http_client(client);
        } else {
            app.install_default_http_client();
        }
        let mut handler = Handler {
            initial: Some((
                self.options,
                Box::new(move |window, app| {
                    IntoElement::into_description_in(
                        (self.build)(window),
                        window.interaction_mut(),
                        app,
                    )
                }),
            )),
            max_frames: self.max_frames,
            initialize: Some(Box::new(initialize)),
            live: Vec::new(),
            failure: None,
            app,
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

    /// Configures the client used by URI resource loading in this application.
    pub fn with_http_client(mut self, client: BoxedHttpClient) -> Self {
        self.http_client = Some(client);
        self
    }

    pub fn run(mut self) -> Result<(), ApplicationError> {
        let event_loop = event_loop()?;
        let mut app = App::create();
        if let Some(client) = self.http_client {
            app.set_http_client(client);
        } else {
            app.install_default_http_client();
        }
        let mut handler = Handler {
            initial: Some((
                self.options,
                Box::new(move |window, app| {
                    IntoElement::into_description_in(
                        (self.build)(window),
                        window.interaction_mut(),
                        app,
                    )
                }),
            )),
            max_frames: self.max_frames,
            initialize: None,
            live: Vec::new(),
            failure: None,
            app,
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

    /// Configures the client used by URI resource loading in this application.
    pub fn with_http_client(self, client: BoxedHttpClient) -> ConfiguredApplication {
        ConfiguredApplication { http_client: Some(client), assets: Arc::new(()) }
    }

    /// Retain the application-builder shape used by asset-backed examples.
    /// Asset resolution is owned by the widget layer; the native lifecycle
    /// does not need to inspect the source while constructing the event loop.
    pub fn with_assets<T: wgpui_widgets::assets::AssetSource>(self, assets: T) -> ConfiguredApplication {
        ConfiguredApplication {
            http_client: None,
            assets: Arc::new(assets),
        }
    }

    pub fn run(self, initialize: impl FnOnce(&mut App) + 'static) -> Result<(), ApplicationError> {
        let event_loop = event_loop()?;
        let mut handler = Handler {
            initial: Some((
                WindowOptions::default(),
                Box::new(|_, _| Description::new::<()>()),
            )),
            max_frames: None,
            initialize: Some(Box::new(initialize)),
            app: {
                let mut app = App::create();
                app.set_global(AssetRegistry::new(Arc::new(())));
                app.set_global(AssetRequests::default());
                app
            },
            live: Vec::new(),
            failure: None,
        };
        event_loop
            .run_app(&mut handler)
            .map_err(ApplicationError::from)?;
        handler.failure.map_or(Ok(()), Err)
    }
}

/// An [`Application`] with an explicitly configured native HTTP client.
pub struct ConfiguredApplication {
    http_client: Option<BoxedHttpClient>,
    assets: Arc<dyn wgpui_widgets::assets::AssetSource>,
}

#[derive(Clone)]
pub struct AssetRef {
    key: String,
    state: Arc<std::sync::Mutex<AssetState>>,
}

#[derive(Clone)]
enum AssetState {
    Loading,
    Ready(Arc<RenderImage>),
    Failed(String),
}

impl AssetRef {
    pub fn is_loading(&self) -> bool {
        match self.state.lock() {
            Ok(state) => matches!(*state, AssetState::Loading),
            Err(poisoned) => matches!(*poisoned.into_inner(), AssetState::Loading),
        }
    }

    pub fn error(&self) -> Option<String> {
        match self.state.lock() {
            Ok(state) => match &*state {
                AssetState::Failed(error) => Some(error.clone()),
                _ => None,
            },
            Err(poisoned) => match &*poisoned.into_inner() {
                AssetState::Failed(error) => Some(error.clone()),
                _ => None,
            },
        }
    }
}

pub enum LegacyImageSource {
    Resource(Resource),
    Deferred(Box<dyn FnOnce(&mut Window, &mut App) -> AssetRef>),
}

pub struct LegacyImgBuilder {
    source: LegacyImageSource,
    image: Option<ImgBuilder>,
    fallback: Option<Box<dyn Fn() -> Description>>,
    loading: Option<Box<dyn Fn() -> Description>>,
    border: bool,
    border_color: Option<wgpui_core::color::Hsla>,
    click_handler: Option<Box<dyn FnMut(&wgpui_core::window::ClickEvent, &mut wgpui_core::window::Window, &mut App)>>,
}

pub trait IntoLegacyImageSource: 'static {
    fn into_legacy_source(self) -> LegacyImageSource;
}

impl IntoLegacyImageSource for Resource {
    fn into_legacy_source(self) -> LegacyImageSource { LegacyImageSource::Resource(self) }
}
impl IntoLegacyImageSource for std::path::PathBuf {
    fn into_legacy_source(self) -> LegacyImageSource { LegacyImageSource::Resource(self.into()) }
}
impl IntoLegacyImageSource for String {
    fn into_legacy_source(self) -> LegacyImageSource { LegacyImageSource::Resource(self.into()) }
}
impl IntoLegacyImageSource for &'static str {
    fn into_legacy_source(self) -> LegacyImageSource { LegacyImageSource::Resource(self.into()) }
}
impl<F> IntoLegacyImageSource for F
where
    F: FnOnce(&mut Window, &mut App) -> AssetRef + 'static,
{
    fn into_legacy_source(self) -> LegacyImageSource { LegacyImageSource::Deferred(Box::new(self)) }
}

pub fn img(source: impl IntoLegacyImageSource) -> LegacyImgBuilder {
    LegacyImgBuilder {
        source: source.into_legacy_source(),
        image: None,
        fallback: None,
        loading: None,
        border: false,
        border_color: None,
        click_handler: None,
    }
}

impl LegacyImgBuilder {
    pub fn id(mut self, id: impl Into<wgpui_core::reconcile::description::ElementId>) -> Self {
        self.image = Some(self.image.take().unwrap_or_else(|| wgpui_widgets::img::img(Resource::Embedded("pending".into()))).id(id)); self
    }
    pub fn size(mut self, size: impl wgpui_widgets::styled::IntoStylePixels) -> Self { self.image = Some(self.image.take().unwrap_or_else(|| wgpui_widgets::img::img(Resource::Embedded("pending".into()))).size(size)); self }
    pub fn size_12(self) -> Self { self.size(48.0) }
    pub fn size_8(self) -> Self { self.size(32.0) }
    pub fn size_16(self) -> Self { self.size(64.0) }
    pub fn size_full(mut self) -> Self { self.image = Some(self.image.take().unwrap_or_else(|| wgpui_widgets::img::img(Resource::Embedded("pending".into()))).size_full()); self }
    pub fn object_fit(mut self, object_fit: wgpui_widgets::img::ObjectFit) -> Self {
        self.image = Some(self.image.take().unwrap_or_else(|| wgpui_widgets::img::img(Resource::Embedded("pending".into()))).object_fit(object_fit));
        self
    }
    pub fn border_1(mut self) -> Self { self.border = true; self }
    pub fn border_color(mut self, color: impl Into<wgpui_core::color::Hsla>) -> Self { self.border_color = Some(color.into()); self }
    pub fn on_click<F>(mut self, handler: F) -> Self
    where F: FnMut(&wgpui_core::window::ClickEvent, &mut wgpui_core::window::Window, &mut App) + 'static { self.click_handler = Some(Box::new(handler)); self }
    pub fn with_fallback<F>(mut self, fallback: F) -> Self where F: Fn() -> Description + 'static { self.fallback = Some(Box::new(fallback)); self }
    pub fn with_loading<F>(mut self, loading: F) -> Self where F: Fn() -> Description + 'static { self.loading = Some(Box::new(loading)); self }
}

impl CoreElement for LegacyImgBuilder {
    fn into_description(self) -> Description { Description::deferred(move |window, app| { let Some(window) = window.downcast_mut::<Window>() else { return Description::new::<Self>(); }; self.resolve(window, app) }) }
    fn into_description_in(self, _window: &mut wgpui_core::window::Window, _app: &App) -> Description { wgpui_core::Element::into_description(self) }
}

impl LegacyImgBuilder {
    fn resolve(self, window: &mut Window, app: &mut App) -> Description {
        let Self { source, image, fallback, loading, border, border_color, click_handler } = self;
        let asset = match source {
            LegacyImageSource::Resource(resource) => {
                let key = format!("resource:{resource:?}");
                let state = match app.global::<AssetRegistry>().and_then(|registry| registry.load_cached(resource).ok()) {
                    Some(image) => AssetState::Ready(image),
                    None => AssetState::Failed("asset could not be resolved".into()),
                };
                AssetRef { key, state: Arc::new(std::sync::Mutex::new(state)) }
            }
            LegacyImageSource::Deferred(resolve) => resolve(window, app),
        };
        let state = match asset.state.lock() {
            Ok(state) => state.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        };
        let description = match state {
            AssetState::Ready(image_data) => {
                let source = wgpui_widgets::img::ImageSourceId::from_raw(crate::window::application::asset_source_id(&asset.key));
                let image = image.unwrap_or_else(|| ImgBuilder::from_decoded(source, image_data));
                wgpui_core::Element::into_description(image)
            }
            AssetState::Loading => loading.map_or_else(|| Description::new::<()>(), |render| render()),
            AssetState::Failed(_) => fallback.map_or_else(|| Description::new::<()>(), |render| render()),
        };
        let description = if border {
            let mut container = wgpui_widgets::div::div().border_1();
            if let Some(color) = border_color {
                container = container.border_color(color);
            }
            wgpui_core::Element::into_description(container.child(description))
        } else {
            description
        };
        if let Some(mut handler) = click_handler { wgpui_core::Element::into_description(wgpui_widgets::div::div().on_click(move |event, window, app| handler(event, window, app)).child(description)) } else { description }
    }
}

fn asset_source_id(key: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new(); key.hash(&mut hasher); hasher.finish().max(1)
}

pub trait AssetImageOutput: Send + 'static {
    fn into_image_result(self) -> Result<Arc<RenderImage>, ImageCacheError>;
}

impl AssetImageOutput for Result<Arc<RenderImage>, ImageCacheError> {
    fn into_image_result(self) -> Result<Arc<RenderImage>, ImageCacheError> { self }
}

struct AssetRequests(std::sync::Mutex<std::collections::HashMap<String, AssetRef>>);

impl Default for AssetRequests {
    fn default() -> Self { Self(std::sync::Mutex::new(std::collections::HashMap::new())) }
}

pub trait AppAssetExt {
    fn remove_asset<A: Asset>(&mut self, source: &A::Source);
}

impl AppAssetExt for App {
    fn remove_asset<A: Asset>(&mut self, source: &A::Source) {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        source.hash(&mut hasher);
        if let Some(requests) = self.global::<AssetRequests>() {
            let key = format!("asset:{}:{}", std::any::type_name::<A>(), hasher.finish());
            requests.0.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).remove(&key);
        }
    }
}

impl Window {
    pub fn use_asset<A>(
        &mut self,
        source: &A::Source,
        app: &mut App,
    ) -> AssetRef
    where
        A: Asset,
        A::Output: AssetImageOutput,
    {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        source.hash(&mut hasher);
        let key = format!("asset:{}:{}", std::any::type_name::<A>(), hasher.finish());
        if app.global::<AssetRequests>().is_none() {
            app.set_global(AssetRequests::default());
        }
        let Some(requests) = app.global::<AssetRequests>() else {
            return AssetRef {
                key,
                state: Arc::new(std::sync::Mutex::new(AssetState::Failed(
                    "asset request registry is unavailable".into(),
                ))),
            };
        };
        let mut entries = requests.0.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(asset) = entries.get(&key) {
            return asset.clone();
        }
        let state = Arc::new(std::sync::Mutex::new(AssetState::Loading));
        let redraw = {
            let native = Arc::clone(&self.native);
            Arc::new(move || native.request_redraw()) as Arc<dyn Fn() + Send + Sync>
        };
        let asset = AssetRef { key: key.clone(), state: Arc::clone(&state) };
        entries.insert(key, asset.clone());
        let mut asset_app = app.clone();
        let future = A::load(source.clone(), &mut asset_app);
        let task = app.background_spawn(async move {
            let result = future.await.into_image_result();
            let new_state = match result {
                Ok(image) => AssetState::Ready(image),
                Err(error) => AssetState::Failed(error.to_string()),
            };
            match state.lock() {
                Ok(mut state) => *state = new_state,
                Err(poisoned) => {
                    let mut state = poisoned.into_inner();
                    *state = new_state;
                }
            }
            redraw();
        });
        task.detach();
        asset
    }

    pub fn remove_asset<A: Asset>(&mut self, source: &A::Source, app: &mut App) {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        source.hash(&mut hasher);
        if let Some(requests) = app.global::<AssetRequests>() {
            let key = format!("asset:{}:{}", std::any::type_name::<A>(), hasher.finish());
            requests.0.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).remove(&key);
        }
    }
}

impl ConfiguredApplication {
    pub fn run(self, initialize: impl FnOnce(&mut App) + 'static) -> Result<(), ApplicationError> {
        let event_loop = event_loop()?;
        let mut app = App::create();
        if let Some(client) = self.http_client {
            app.set_http_client(client);
        } else {
            app.install_default_http_client();
        }
        app.set_global(AssetRegistry::new(self.assets));
        app.set_global(AssetRequests::default());
        let mut handler = Handler {
            initial: Some((
                WindowOptions::default(),
                Box::new(|_, _| Description::new::<()>()),
            )),
            max_frames: None,
            initialize: Some(Box::new(initialize)),
            app,
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
    id: wgpui_core::app::WindowId,
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

fn window_decorations_for(server_decorations: bool) -> WindowDecorations {
    if server_decorations {
        WindowDecorations::Server
    } else {
        WindowDecorations::Client
    }
}

fn append_immediate_paints(
    mut description: Description,
    paints: impl IntoIterator<Item = ImmediatePaint>,
) -> Description {
    for paint in paints {
        description = description.child(
            Description::new::<ImmediatePaint>().emit(
                move |_context: &EmitContext, emission: &mut Emission| match &paint {
                    ImmediatePaint::Quad(quad) => {
                        emission.quad(quad.clone());
                    }
                    ImmediatePaint::Path(path) => {
                        emission.path(path.clone());
                    }
                },
            ),
        );
    }
    description
}

impl Handler {
    fn fail(&mut self, event_loop: &winit::event_loop::ActiveEventLoop, error: ApplicationError) {
        self.failure = Some(error);
        event_loop.exit();
    }

    fn create_window(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        id: wgpui_core::app::WindowId,
        options: WindowOptions,
        build: WindowBuildCallback,
    ) -> Result<(), ApplicationError> {
        let title = options
            .titlebar
            .as_ref()
            .and_then(|titlebar| titlebar.title.as_deref())
            .unwrap_or(&options.title);
        let resizable = options.resizable && options.is_resizable;
        let decorations = options
            .window_decorations
            .map(|decorations| decorations == WindowDecorations::Server)
            .unwrap_or(options.titlebar.is_some());
        if options.window_decorations == Some(WindowDecorations::Client) {
            log::warn!("client window decorations are lowered as a borderless Winit window; WGPUI does not provide native client chrome");
        }
        if !options.is_movable {
            log::warn!("non-movable windows are not supported by the cross-platform Winit boundary; the request is not applied");
        }
        if options.app_id.is_some() {
            log::warn!("WindowOptions::app_id is not supported by the cross-platform Winit boundary; the request is not applied");
        }
        if options.tabbing_identifier.is_some() {
            log::warn!("WindowOptions::tabbing_identifier is not supported by the cross-platform Winit boundary; the request is not applied");
        }
        if options.display_id.is_some() {
            log::warn!("WindowOptions::display_id is not supported by the cross-platform Winit boundary; the request is not applied");
        }
        let attributes = winit::window::Window::default_attributes()
            .with_title(title)
            .with_resizable(resizable)
            .with_visible(options.show)
            .with_decorations(decorations)
            .with_enabled_buttons(enabled_window_buttons(resizable, options.is_minimizable))
            .with_window_level(window_level(&options.kind));
        let attributes = apply_titlebar_attributes(attributes, options.titlebar.as_ref());
        let attributes = apply_background_attributes(attributes, options.window_background);
        let bounds = initial_bounds(&options);
        let mut attributes = attributes.with_inner_size(winit::dpi::LogicalSize::new(
            bounds.size.width.value(),
            bounds.size.height.value(),
        ));
        if let Some(min_size) = options.window_min_size {
            attributes = attributes.with_min_inner_size(winit::dpi::LogicalSize::new(
                min_size.width.value(),
                min_size.height.value(),
            ));
        }
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
        let surface_registry = Arc::new(SurfaceRegistry::new());
        let (width, height) = surface.size();
        let mut resizes = ResizeDetector::new();
        resizes.seed(width, height);
        let mode = DrawMode::best_available(context.indirect);
        let window = Window {
            native,
            gpu_adapter: context.adapter.clone(),
            gpu_device: context.device.clone(),
            gpu_queue: context.queue.clone(),
            surface_registry: Arc::clone(&surface_registry),
            scale_factor,
            close_requested: Arc::new(AtomicBool::new(false)),
            background_appearance: options.window_background,
            decorations,
            last_frame: None,
            state: ElementStateStore::new(),
            state_frame: 0,
            state_entities: std::collections::HashMap::new(),
            immediate_paints: Vec::new(),
            interaction: wgpui_core::window::Window::new(),
            close_handler: None,
            interaction_modifiers: Modifiers::default(),
            mouse_buttons: MouseButtonState::default(),
            cursor: [Pixels::ZERO, Pixels::ZERO],
            cursor_inside: false,
            interactions: Vec::new(),
            hovered_interaction: None,
            pressed_interaction: None,
            pressed_event: None,
            active_drag: None,
            drag_hovered: None,
            hover_dirty_regions: Vec::new(),
            performance_debug: PerformanceDebug::default(),
            animation_clock: AnimationClock::new(),
            animation_frame_requested: AtomicBool::new(false),
            last_click: None,
        };
        let mut frame_loop = FrameLoop::new(&context.device);
        frame_loop.set_surface_registry(surface_registry);
        self.live.push(Live {
            id,
            frame_loop,
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
                let mut request_build = Some(request.build);
                let mut renderer = None;
                let build = Box::new(move |window: &mut Window, app: &mut App| {
                    if renderer.is_none()
                        && let Some(build) = request_build.take()
                    {
                        renderer = Some(build(app, window));
                    }
                    renderer.as_mut().map_or_else(
                        || Description::new::<()>(),
                        |renderer| (renderer.render)(app, window),
                    )
                });
                if let Err(error) =
                    self.create_window(event_loop, request.id, request.options, build)
                {
                    self.app.window_creation_failed(request.id);
                    if first_error.is_none() {
                        first_error = Some(error);
                    }
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
        self.app.run_pending_tasks();
        let all_other_windows_reached_limit = self.max_frames.is_some_and(|limit| {
            self.live
                .iter()
                .enumerate()
                .all(|(other_index, live)| other_index == index || live.frames >= limit)
        });
        let live = &mut self.live[index];
        let animation_frame_requested = live
            .window
            .animation_frame_requested
            .swap(false, Ordering::AcqRel);
        let due_timer_count = live.window.take_due_timers(Instant::now()).len();
        if due_timer_count > 0 {
            log::debug!("window timer deadline reached: {due_timer_count} timer(s)");
        }
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
        let animation_clock = std::mem::take(&mut live.window.animation_clock);
        let (animation_clock, description) =
            wgpui_core::window::animation::with_animation_clock(animation_clock, || {
                (live.build)(&mut live.window, &mut live.app)
            });
        live.window.animation_clock = animation_clock;
        let description = description.resolve_deferred(&mut live.window, &mut live.app);
        let description = description.resolve_deferred_core_window(
            live.window.interaction_mut(),
            &mut live.app,
        );
        let description = description.resolve_deferred(&mut live.window, &mut live.app);
        let description = description.resolve_deferred_core_window(
            live.window.interaction_mut(),
            &mut live.app,
        );
        let description = append_immediate_paints(
            description,
            std::mem::take(&mut live.window.immediate_paints),
        );
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
                live.window
                    .set_interactions(frame.interactions, &mut live.app);
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
                if live.window.close_requested() || live.app.close_requested() {
                    let closed = if live.app.close_requested() {
                        self.live.drain(..).map(|live| live.id).collect::<Vec<_>>()
                    } else {
                        vec![self.live.remove(index).id]
                    };
                    for id in closed {
                        self.app.window_closed(id);
                    }
                    if self.live.is_empty() {
                        event_loop.exit();
                    }
                } else if self.max_frames.is_some_and(|limit| live.frames >= limit)
                    && all_other_windows_reached_limit
                {
                    event_loop.exit();
                } else if animation_frame_requested || self.max_frames.is_some() || frame.needs_redraw {
                    live.window.request_redraw();
                }
            }
            Err(error) => self.fail(event_loop, ApplicationError::Render(error.to_string())),
        }
    }
}

fn enabled_window_buttons(resizable: bool, minimizable: bool) -> winit::window::WindowButtons {
    let mut buttons = winit::window::WindowButtons::CLOSE;
    if minimizable {
        buttons.insert(winit::window::WindowButtons::MINIMIZE);
    }
    if resizable {
        buttons.insert(winit::window::WindowButtons::MAXIMIZE);
    }
    buttons
}

fn apply_titlebar_attributes(
    attributes: winit::window::WindowAttributes,
    titlebar: Option<&wgpui_core::window::TitlebarOptions>,
) -> winit::window::WindowAttributes {
    #[cfg(target_os = "macos")]
    {
        use winit::platform::macos::WindowAttributesExtMacOS;
        attributes.with_titlebar_transparent(
            titlebar.is_some_and(|titlebar| titlebar.appears_transparent),
        )
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = titlebar;
        attributes
    }
}

fn window_level(kind: &WindowKind) -> winit::window::WindowLevel {
    match kind {
        WindowKind::Normal => winit::window::WindowLevel::Normal,
        WindowKind::PopUp | WindowKind::Floating => winit::window::WindowLevel::AlwaysOnTop,
    }
}

fn apply_background_attributes(
    attributes: winit::window::WindowAttributes,
    appearance: WindowBackgroundAppearance,
) -> winit::window::WindowAttributes {
    let attributes = attributes
        .with_transparent(appearance != WindowBackgroundAppearance::Opaque)
        .with_blur(appearance == WindowBackgroundAppearance::Blurred);
    #[cfg(target_os = "windows")]
    {
        use winit::platform::windows::{BackdropType, WindowAttributesExtWindows};
        let backdrop = match appearance {
            WindowBackgroundAppearance::Opaque | WindowBackgroundAppearance::Transparent => {
                BackdropType::None
            }
            WindowBackgroundAppearance::Blurred => BackdropType::TransientWindow,
            WindowBackgroundAppearance::MicaBackdrop => BackdropType::MainWindow,
            WindowBackgroundAppearance::MicaAltBackdrop => BackdropType::TabbedWindow,
        };
        attributes.with_system_backdrop(backdrop)
    }
    #[cfg(not(target_os = "windows"))]
    {
        attributes
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
        if self.live.is_empty() && self.app.close_requested() {
            event_loop.exit();
            return;
        }
        if self.live.is_empty()
            && !self.app.close_requested()
            && let Some((options, build)) = self.initial.take()
        {
            let id = self.app.reserve_window();
            if let Err(error) = self.create_window(event_loop, id, options, build) {
                self.app.window_creation_failed(id);
                self.fail(event_loop, error);
            }
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
        let surface_registry = Arc::new(SurfaceRegistry::new());
        let (width, height) = surface.size();
        let mut resizes = ResizeDetector::new();
        resizes.seed(width, height);
        let mode = DrawMode::best_available(context.indirect);
        let window = Window {
            native,
            gpu_adapter: context.adapter.clone(),
            gpu_device: context.device.clone(),
            gpu_queue: context.queue.clone(),
            surface_registry: Arc::clone(&surface_registry),
            scale_factor,
            close_requested: false,
            last_frame: None,
            state: ElementStateStore::new(),
            state_frame: 0,
            state_entities: std::collections::HashMap::new(),
            immediate_paints: Vec::new(),
            interaction: wgpui_core::window::Window::new(),
            close_handler: None,
            interaction_modifiers: Modifiers::default(),
            mouse_buttons: MouseButtonState::default(),
            cursor: [Pixels::ZERO, Pixels::ZERO],
            cursor_inside: false,
            interactions: Vec::new(),
            hovered_interaction: None,
            pressed_interaction: None,
            pressed_event: None,
            hover_dirty_regions: Vec::new(),
            performance_debug: PerformanceDebug::default(),
            animation_clock: AnimationClock::new(),
            last_click: None,
        };
        let mut frame_loop = FrameLoop::new(&context.device);
        frame_loop.set_surface_registry(surface_registry);
        self.live = Some(Live {
            frame_loop,
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
                    let id = self.live.remove(index).id;
                    self.app.window_closed(id);
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
                let modifiers = modifiers_from_winit(modifiers.state());
                live.window.interaction_modifiers = modifiers;
                if live.window.handle_input_with_app(
                    InputEvent::ModifiersChanged(ModifiersChangedEvent { modifiers }),
                    &mut live.app,
                ) {
                    live.window.request_redraw();
                }
            }
            winit::event::WindowEvent::KeyboardInput { event, .. } => {
                let key = key_name(&event);
                let text = event.text.as_ref().map(ToString::to_string);
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
                if let Some(text) = text
                    && !text.is_empty()
                    && live.window.handle_input_with_app(
                        InputEvent::TextInput(TextInputEvent { text }),
                        &mut live.app,
                    )
                {
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
                let handled = live.window.handle_input_with_app(event, &mut live.app);
                let hover_dirty_regions = live.window.take_hover_dirty_regions();
                for region in hover_dirty_regions.iter().copied() {
                    live.frame_loop.mark_interaction_dirty(region);
                }
                if handled || !hover_dirty_regions.is_empty() {
                    live.window.request_redraw();
                }
            }
            winit::event::WindowEvent::MouseInput { state, button, .. } => {
                let point = live.window.cursor_position();
                let button = core_mouse_button(button);
                live.window
                    .set_mouse_button(button, state == winit::event::ElementState::Pressed);
                let click_count = if state == winit::event::ElementState::Pressed {
                    live.window.next_click_count(button, point)
                } else {
                    live.window
                        .pressed_event
                        .map_or(1, |event| event.click_count)
                };
                let event = if state == winit::event::ElementState::Pressed {
                    InputEvent::MouseDown(MouseDownEvent {
                        button,
                        position: point,
                        modifiers: live.window.modifiers(),
                        click_count,
                    })
                } else {
                    InputEvent::MouseUp(MouseUpEvent {
                        button,
                        position: point,
                        modifiers: live.window.modifiers(),
                        click_count,
                    })
                };
                if live.window.handle_input_with_app(event, &mut live.app) {
                    live.window.request_redraw();
                }
            }
            winit::event::WindowEvent::MouseWheel { delta, .. } => {
                let delta = match delta {
                    winit::event::MouseScrollDelta::LineDelta(x, y) => [x * 16.0, y * 16.0],
                    winit::event::MouseScrollDelta::PixelDelta(point) => {
                        [
                            point.x as f32 / live.window.scale_factor as f32,
                            point.y as f32 / live.window.scale_factor as f32,
                        ]
                    }
                };
                let event = InputEvent::Scroll(ScrollWheelEvent {
                    position: live.window.cursor_position(),
                    delta,
                    modifiers: live.window.modifiers(),
                });
                if live.window.handle_input_with_app(event, &mut live.app) {
                    live.window.request_redraw();
                }
            }
            winit::event::WindowEvent::Ime(ime) => {
                let event = ime_event(ime);
                if live.window.handle_input_with_app(InputEvent::Ime(event), &mut live.app) {
                    live.window.request_redraw();
                }
            }
            winit::event::WindowEvent::Focused(focused) => {
                if focused {
                    live.window.interaction.activate();
                } else {
                    live.window.interaction.deactivate();
                }
                live.window.handle_window_focus(focused, &mut live.app);
                let dirty_regions = live.window.take_hover_dirty_regions();
                for region in dirty_regions {
                    live.frame_loop.mark_interaction_dirty(region);
                }
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
        self.app.run_pending_tasks();
        self.create_pending_windows(event_loop);
        let now = Instant::now();
        let timer_deadline = self
            .live
            .iter()
            .filter_map(|live| live.window.next_timer_deadline())
            .min();
        if self.max_frames.is_some() || self.app.has_pending_tasks() || self.app.close_requested() {
            event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
            for live in &self.live {
                live.window.request_redraw();
            }
        } else if let Some(deadline) = timer_deadline {
            event_loop.set_control_flow(winit::event_loop::ControlFlow::WaitUntil(deadline));
            if deadline <= now {
                for live in &self.live {
                    live.window.request_redraw();
                }
            }
        } else {
            event_loop.set_control_flow(winit::event_loop::ControlFlow::Wait);
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
            CoreMouseButton::Other(match button {
                winit::event::MouseButton::Back => 4,
                _ => 5,
            })
        }
        winit::event::MouseButton::Other(value) => CoreMouseButton::Other(value),
    }
}

fn ime_event(event: winit::event::Ime) -> ImeEvent {
    match event {
        winit::event::Ime::Enabled => ImeEvent::Enabled,
        winit::event::Ime::Preedit(text, cursor) => ImeEvent::Preedit {
            selection: cursor.map(|(start, end)| {
                byte_to_utf16_offset(&text, start)..byte_to_utf16_offset(&text, end)
            }),
            text,
        },
        winit::event::Ime::Commit(text) => ImeEvent::Commit(text),
        winit::event::Ime::Disabled => ImeEvent::Disabled,
    }
}

fn byte_to_utf16_offset(text: &str, byte_offset: usize) -> usize {
    text.get(..byte_offset.min(text.len()))
        .unwrap_or(text)
        .encode_utf16()
        .count()
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

    fn spawn_accepts_a_context_async_closure<T: 'static>(
        window: &Window,
        context: &wgpui_core::app::Context<T>,
    ) {
        std::mem::drop(window.spawn(context, async move |_context| {}));
    }

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

    #[test]
    fn ime_preedit_offsets_are_converted_from_utf8_bytes_to_utf16_units() {
        let event = ime_event(winit::event::Ime::Preedit(
            "a😀中".to_string(),
            Some((1, 6)),
        ));
        assert_eq!(
            event,
            ImeEvent::Preedit {
                text: "a😀中".to_string(),
                selection: Some(1..4),
            }
        );
    }

    #[test]
    fn window_options_translate_to_supported_winit_controls() {
        let buttons = enabled_window_buttons(false, false);
        assert!(buttons.contains(winit::window::WindowButtons::CLOSE));
        assert!(!buttons.contains(winit::window::WindowButtons::MINIMIZE));
        assert!(!buttons.contains(winit::window::WindowButtons::MAXIMIZE));

        assert_eq!(
            window_level(&WindowKind::Normal),
            winit::window::WindowLevel::Normal
        );
        assert_eq!(
            window_level(&WindowKind::PopUp),
            winit::window::WindowLevel::AlwaysOnTop
        );
        assert_eq!(
            window_level(&WindowKind::Floating),
            winit::window::WindowLevel::AlwaysOnTop
        );
    }

    #[test]
    fn native_modifiers_preserve_command_and_alt_without_losing_shift() {
        let state = winit::keyboard::ModifiersState::SUPER
            | winit::keyboard::ModifiersState::ALT
            | winit::keyboard::ModifiersState::SHIFT;
        assert_eq!(
            modifiers_from_winit(state),
            Modifiers {
                shift: true,
                control: false,
                alt: true,
                command: true,
            }
        );
    }

    #[test]
    fn background_options_preserve_transparency_and_blur_contract() {
        for appearance in [
            WindowBackgroundAppearance::Opaque,
            WindowBackgroundAppearance::Transparent,
            WindowBackgroundAppearance::Blurred,
            WindowBackgroundAppearance::MicaBackdrop,
            WindowBackgroundAppearance::MicaAltBackdrop,
        ] {
            let attributes = apply_background_attributes(
                winit::window::Window::default_attributes(),
                appearance,
            );
            assert_eq!(
                attributes.transparent(),
                appearance != WindowBackgroundAppearance::Opaque
            );
        }
    }

    #[test]
    fn decoration_state_reports_the_native_decoration_mode() {
        assert_eq!(
            window_decorations_for(true),
            WindowDecorations::Server
        );
        assert_eq!(
            window_decorations_for(false),
            WindowDecorations::Client
        );
    }

    #[test]
    fn immediate_paints_are_lowered_as_retained_description_children() {
        let path = Path::new(
            vec![
                wgpui_core::patch::primitive::PathVertex {
                    position: [0.0, 0.0],
                    st: [0.0, 1.0],
                },
                wgpui_core::patch::primitive::PathVertex {
                    position: [1.0, 0.0],
                    st: [0.0, 1.0],
                },
                wgpui_core::patch::primitive::PathVertex {
                    position: [0.0, 1.0],
                    st: [0.0, 1.0],
                },
            ],
            [1.0; 4],
        );
        let description = append_immediate_paints(
            Description::new::<()>(),
            [ImmediatePaint::Path(path)],
        );

        assert_eq!(description.child_descriptions().len(), 1);
        assert!(description
            .child_descriptions()
            .first()
            .is_some_and(Description::emits));
    }

    #[test]
    fn public_callback_adapter_matches_core_widget_callback_abi() {
        let _element = wgpui_widgets::div::div().on_click(public_window_callback(
            |_: &wgpui_core::window::ClickEvent, _window: &mut Window, _app: &mut App| {},
        ));
    }
}
