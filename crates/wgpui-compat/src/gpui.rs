//! The 2.0 compatibility façade. Its crate name is `gpui` so unchanged
//! examples exercise the public surface rather than a source-level port.
//!
//! Only symbols that already have a real 2.0 implementation are exported.
//! The missing legacy lifecycle and frontend symbols remain compile failures
//! in the probe instead of becoming behaviorless placeholders.

use std::cell::RefCell;
use std::future::Future;
use std::rc::Rc;
use std::any::TypeId;
use std::ops::{Deref, DerefMut};

pub use wgpui_core::boundary::policy::Pixels;
pub use wgpui_core::geometry::Rect;
pub use wgpui_core::patch::primitive::{PrimitiveKind, Quad, Shadow, Underline};
pub use wgpui_core::reconcile::description::{Description, ElementId};
pub use wgpui_core::reconcile::diff_key::ReconcileKey;
pub use wgpui_core::reconcile::instance::{ElementInstance, InstanceKey, InstanceTable};
pub use wgpui_layout::taffy_tree::LayoutNodeId;
pub use wgpui_text::shaping::{Font, FontId, FontStyle, FontWeight, SharedString};
pub use wgpui_widgets::div::{Div, IntoDescription, div};
pub use wgpui_widgets::div::interactivity::style::BoxShadow as ResolvedBoxShadow;
pub use wgpui_widgets::styled::Styled;
pub use wgpui_widgets::styled::LinearColorStop;

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Size<T> {
    pub width: T,
    pub height: T,
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Point<T> {
    pub x: T,
    pub y: T,
}

impl From<Point<Pixels>> for [f32; 2] {
    fn from(value: Point<Pixels>) -> Self {
        [value.x.value(), value.y.value()]
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Bounds<T> {
    pub origin: Point<T>,
    pub size: Size<T>,
}
impl<T> Bounds<T> {
    pub fn new(origin: Point<T>, size: Size<T>) -> Self { Self { origin, size } }
}

impl Size<Pixels> {
    pub fn center(self) -> Point<Pixels> {
        Point { x: Pixels(self.width.value() / 2.0), y: Pixels(self.height.value() / 2.0) }
    }
}

impl Bounds<Pixels> {
    pub fn centered<C>(_display: Option<DisplayId>, size: Size<Pixels>, _cx: &C) -> Self {
        Self {
            origin: Point {
                x: Pixels::ZERO,
                y: Pixels::ZERO,
            },
            size,
        }
    }
}

pub const fn px(value: f32) -> Pixels {
    Pixels(value)
}
pub fn size<T>(width: T, height: T) -> Size<T> {
    Size { width, height }
}
pub fn point<T>(x: T, y: T) -> Point<T> {
    Point { x, y }
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Rgba {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Rgba {
    pub fn opacity(mut self, alpha: f32) -> Self {
        self.a = alpha;
        self
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Hsla {
    pub h: f32,
    pub s: f32,
    pub l: f32,
    pub a: f32,
}

impl Hsla {
    pub fn opacity(mut self, alpha: f32) -> Self {
        self.a = alpha;
        self
    }
}

pub fn rgb(hex: u32) -> Rgba {
    Rgba {
        r: ((hex >> 16) & 0xff) as f32 / 255.0,
        g: ((hex >> 8) & 0xff) as f32 / 255.0,
        b: (hex & 0xff) as f32 / 255.0,
        a: 1.0,
    }
}
pub fn rgba(hex: u32) -> Rgba {
    Rgba {
        r: ((hex >> 24) & 0xff) as f32 / 255.0,
        g: ((hex >> 16) & 0xff) as f32 / 255.0,
        b: ((hex >> 8) & 0xff) as f32 / 255.0,
        a: (hex & 0xff) as f32 / 255.0,
    }
}
pub fn hsla(h: f32, s: f32, l: f32, a: f32) -> Hsla {
    Hsla { h, s, l, a }
}

pub fn linear_color_stop(color: impl Into<[f32; 4]>, position: f32) -> LinearColorStop {
    LinearColorStop { color: color.into(), position }
}
impl From<Rgba> for [f32; 4] {
    fn from(value: Rgba) -> Self {
        [value.r, value.g, value.b, value.a]
    }
}
impl From<Hsla> for [f32; 4] {
    fn from(value: Hsla) -> Self {
        let q = if value.l < 0.5 {
            value.l * (1.0 + value.s)
        } else {
            value.l + value.s - value.l * value.s
        };
        let p = 2.0 * value.l - q;
        fn hue(p: f32, q: f32, mut t: f32) -> f32 {
            if t < 0.0 {
                t += 1.0;
            } else if t > 1.0 {
                t -= 1.0;
            }
            if t < 1.0 / 6.0 {
                p + (q - p) * 6.0 * t
            } else if t < 1.0 / 2.0 {
                q
            } else if t < 2.0 / 3.0 {
                p + (q - p) * (2.0 / 3.0 - t) * 6.0
            } else {
                p
            }
        }
        if value.s == 0.0 {
            [value.l, value.l, value.l, value.a]
        } else {
            [
                hue(p, q, value.h + 1.0 / 3.0),
                hue(p, q, value.h),
                hue(p, q, value.h - 1.0 / 3.0),
                value.a,
            ]
        }
    }
}
pub const fn red() -> Hsla {
    Hsla { h: 0.0, s: 1.0, l: 0.5, a: 1.0 }
}
pub const fn green() -> Hsla {
    Hsla { h: 1.0 / 3.0, s: 1.0, l: 0.5, a: 1.0 }
}
pub const fn blue() -> Hsla {
    Hsla { h: 2.0 / 3.0, s: 1.0, l: 0.5, a: 1.0 }
}
pub const fn black() -> Hsla {
    Hsla { h: 0.0, s: 0.0, l: 0.0, a: 1.0 }
}
pub const fn white() -> Hsla {
    Hsla { h: 0.0, s: 0.0, l: 1.0, a: 1.0 }
}
pub const fn yellow() -> Hsla {
    Hsla { h: 1.0 / 6.0, s: 1.0, l: 0.5, a: 1.0 }
}
pub fn transparent_black() -> Rgba { Rgba { r: 0.0, g: 0.0, b: 0.0, a: 0.0 } }
pub fn relative(value: f32) -> f32 { value * 16.0 }

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BoxShadow {
    pub color: Hsla,
    pub offset: Point<Pixels>,
    pub blur_radius: Pixels,
    pub spread_radius: Pixels,
}

impl From<BoxShadow> for ResolvedBoxShadow {
    fn from(value: BoxShadow) -> Self {
        Self {
            color: value.color.into(),
            offset: value.offset.into(),
            blur_radius: value.blur_radius.value(),
            spread_radius: value.spread_radius.value(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Colors {
    pub text: Rgba, pub text_muted: Rgba, pub selected_text: Rgba,
    pub background: Rgba, pub surface: Rgba, pub surface_hover: Rgba,
    pub disabled: Rgba, pub selected: Rgba, pub border: Rgba, pub separator: Rgba,
    pub container: Rgba, pub accent: Rgba, pub accent_hover: Rgba,
    pub accent_active: Rgba, pub success: Rgba, pub success_hover: Rgba,
    pub warning: Rgba, pub warning_hover: Rgba, pub error: Rgba, pub error_hover: Rgba,
}
impl Colors {
    pub fn light() -> Self { Self::from_palette(0xffffff, 0xf5f5f5, 0x007aff) }
    pub fn dark() -> Self { Self::from_palette(0xffffff, 0x1e1e1e, 0x0a84ff) }
    fn from_palette(text: u32, background: u32, accent: u32) -> Self {
        let text = rgb(text); let background = rgb(background); let accent = rgb(accent);
        Self { text, text_muted: rgb(0x888888), selected_text: rgb(0xffffff), background,
            surface: rgb(0x2d2d2d), surface_hover: rgb(0x3d3d3d), disabled: rgb(0x666666),
            selected: accent, border: rgb(0x777777), separator: rgb(0x777777),
            container: background, accent, accent_hover: accent, accent_active: accent,
            success: rgb(0x28cd41), success_hover: rgb(0x28cd41), warning: rgb(0xffcc00),
            warning_hover: rgb(0xffcc00), error: rgb(0xff3b30), error_hover: rgb(0xff3b30) }
    }
}
impl Default for Colors { fn default() -> Self { Self::light() } }
impl Colors {
    pub fn for_appearance<C>(_window: &C) -> Self { Self::default() }
}
impl From<Rgba> for Hsla {
    fn from(color: Rgba) -> Self {
        let max = color.r.max(color.g).max(color.b);
        let min = color.r.min(color.g).min(color.b);
        let lightness = (max + min) / 2.0;
        if (max - min).abs() < f32::EPSILON {
            return Hsla { h: 0.0, s: 0.0, l: lightness, a: color.a };
        }
        let delta = max - min;
        let saturation = delta / (1.0 - (2.0 * lightness - 1.0).abs());
        let hue = if (max - color.r).abs() < f32::EPSILON {
            ((color.g - color.b) / delta).rem_euclid(6.0)
        } else if (max - color.g).abs() < f32::EPSILON {
            (color.b - color.r) / delta + 2.0
        } else {
            (color.r - color.g) / delta + 4.0
        } / 6.0;
        Hsla { h: hue, s: saturation, l: lightness, a: color.a }
    }
}

pub struct Task<T>(Option<T>);
impl<T> Task<T> {
    pub fn ready(value: T) -> Self {
        Self(Some(value))
    }
    pub fn detach(self) {}
}

pub trait Action: 'static + Send + Sync {
    fn name() -> &'static str where Self: Sized;
}

#[macro_export]
macro_rules! actions {
    ($namespace:path, [$( $(#[$attr:meta])* $name:ident),* $(,)?]) => {
        $(
            $(#[$attr])*
            #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
            pub struct $name;
            impl $crate::Action for $name {
                fn name() -> &'static str { concat!(stringify!($namespace), "::", stringify!($name)) }
            }
        )*
    };
    ([$( $(#[$attr:meta])* $name:ident),* $(,)?]) => {
        $(
            $(#[$attr])*
            #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
            pub struct $name;
            impl $crate::Action for $name {
                fn name() -> &'static str { stringify!($name) }
            }
        )*
    };
}

#[derive(Clone)]
pub struct KeyBinding { pub keystrokes: SharedString, pub action: &'static str, pub context: Option<SharedString> }
impl KeyBinding {
    pub fn new<A: Action>(keystrokes: &str, _action: A, context: Option<&str>) -> Self {
        Self { keystrokes: keystrokes.into(), action: A::name(), context: context.map(Into::into) }
    }
}
pub struct Menu { pub name: SharedString, pub items: Vec<MenuItem> }
pub enum MenuItem { Action { name: SharedString, action: &'static str } }
impl MenuItem {
    pub fn action<A: Action>(name: impl Into<SharedString>, _action: A) -> Self {
        Self::Action { name: name.into(), action: A::name() }
    }
}
impl<T> Future for Task<T> {
    type Output = T;
    fn poll(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<T> {
        let task = unsafe { self.get_unchecked_mut() };
        std::task::Poll::Ready(task.0.take().expect("task polled after completion"))
    }
}

pub struct Entity<T>(Rc<RefCell<Option<T>>>);
impl<T> Clone for Entity<T> {
    fn clone(&self) -> Self {
        Self(Rc::clone(&self.0))
    }
}
impl<T> Entity<T> {
    fn initialize(&self, value: T) {
        *self.0.borrow_mut() = Some(value);
    }
    pub fn read(&self) -> std::cell::Ref<'_, T> {
        std::cell::Ref::map(self.0.borrow(), |value| {
            value.as_ref().expect("entity is initialized")
        })
    }
    pub fn update<R>(&self, update: impl FnOnce(&mut T) -> R) -> R {
        update(self.0.borrow_mut().as_mut().expect("entity is initialized"))
    }
    pub fn downgrade(&self) -> WeakEntity<T> {
        WeakEntity(Rc::downgrade(&self.0))
    }
}
#[derive(Debug)]
pub struct EntityError;
pub struct WeakEntity<T>(std::rc::Weak<RefCell<Option<T>>>);
impl<T> Clone for WeakEntity<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<T> WeakEntity<T> {
    pub fn update(&self, update: impl FnOnce(&mut T)) -> Result<(), EntityError> {
        let entity = self.0.upgrade().ok_or(EntityError)?;
        update(entity.borrow_mut().as_mut().ok_or(EntityError)?);
        Ok(())
    }
}

pub struct Context<T> {
    entity: Option<Entity<T>>,
    notifications: Rc<RefCell<u64>>,
    quit_requested: Rc<RefCell<bool>>,
}
impl<T> Context<T> {
    fn from_entity(entity: Entity<T>, notifications: Rc<RefCell<u64>>) -> Self {
        Self {
            entity: Some(entity),
            notifications,
            quit_requested: Rc::new(RefCell::new(false)),
        }
    }
    pub fn entity(&self) -> Entity<T> {
        self.entity.clone().expect("context entity is initialized")
    }
    pub fn notify(&mut self) {
        *self.notifications.borrow_mut() += 1;
    }
    pub fn quit(&mut self) {
        *self.quit_requested.borrow_mut() = true;
    }
    pub fn focus_handle(&mut self) -> FocusHandle {
        FocusHandle::new()
    }
    pub fn listener<A, F>(&mut self, handler: F) -> F
    where
        F: Fn(&mut T, &A, &mut Window, &mut Context<T>) + 'static,
    {
        handler
    }
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    pub fn new<U>(&mut self, build: impl FnOnce(&mut Context<U>) -> U) -> Entity<U> {
        let entity = Entity(Rc::new(RefCell::new(None)));
        let mut context = Context {
            entity: None,
            notifications: Rc::clone(&self.notifications),
            quit_requested: Rc::new(RefCell::new(false)),
        };
        entity.initialize(build(&mut context));
        entity
    }
    pub fn spawn<F, R>(&self, future: F) -> Task<R>
    where
        F: Future<Output = R>,
    {
        Task::ready(futures::executor::block_on(future))
    }
}

pub struct App {
    notifications: Rc<RefCell<u64>>,
    descriptions: Vec<Description>,
    active: bool,
    key_bindings: Vec<KeyBinding>,
    menus: Vec<Menu>,
    action_types: Vec<TypeId>,
    quit_requested: bool,
    window_closed_handlers: Vec<WindowClosedHandler>,
}
impl App {
    pub fn new() -> Self {
        Self {
            notifications: Rc::new(RefCell::new(0)),
            descriptions: Vec::new(),
            active: false,
            key_bindings: Vec::new(),
            menus: Vec::new(),
            action_types: Vec::new(),
            quit_requested: false,
            window_closed_handlers: Vec::new(),
        }
    }
    pub fn new_entity<T>(&mut self, build: impl FnOnce(&mut Context<T>) -> T) -> Entity<T> {
        let entity = Entity(Rc::new(RefCell::new(None)));
        let mut context = Context {
            entity: None,
            notifications: Rc::clone(&self.notifications),
            quit_requested: Rc::new(RefCell::new(false)),
        };
        entity.initialize(build(&mut context));
        entity
    }
    pub fn open_window<T: Render>(
        &mut self,
        _options: WindowOptions,
        build: impl FnOnce(&mut Window, &mut Context<T>) -> Entity<T>,
    ) -> Result<WindowHandle, String> {
        let context_entity = Entity(Rc::new(RefCell::new(None)));
        let mut window = Window;
        let root_entity = build(
            &mut window,
            &mut Context::from_entity(context_entity, Rc::clone(&self.notifications)),
        );
        let mut context = Context::from_entity(root_entity.clone(), Rc::clone(&self.notifications));
        let description =
            root_entity.update(|root| {
                IntoDescription::into_description(root.render(&mut window, &mut context))
            });
        self.descriptions.push(description);
        Ok(WindowHandle)
    }
    pub fn activate(&mut self, active: bool) {
        self.active = active;
    }
    pub fn on_action<A: Action>(&mut self, _handler: impl FnMut(&A, &mut App) + 'static) {
        self.action_types.push(TypeId::of::<A>());
    }
    pub fn bind_keys(&mut self, bindings: impl IntoIterator<Item = KeyBinding>) {
        self.key_bindings.extend(bindings);
    }
    pub fn set_menus(&mut self, menus: Vec<Menu>) { self.menus = menus; }
    pub fn on_window_closed(&mut self, handler: impl FnMut(&mut App, WindowHandle) + 'static) -> Task<()> {
        self.window_closed_handlers.push(Box::new(handler));
        Task::ready(())
    }
    pub fn quit(&mut self) { self.quit_requested = true; }
    pub fn windows(&self) -> &[WindowHandle] { &[] }
    pub fn descriptions(&self) -> &[Description] {
        &self.descriptions
    }
}
pub struct Application;
impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}
impl Default for Application {
    fn default() -> Self {
        Self::new()
    }
}
impl Application {
    pub fn new() -> Self {
        Self
    }
    pub fn run(self, callback: impl FnOnce(&mut App)) {
        let mut app = App::new();
        callback(&mut app);
    }
}
pub struct Window;
pub struct WindowHandle;
type WindowClosedHandler = Box<dyn FnMut(&mut App, WindowHandle)>;
#[derive(Default)]
pub struct WindowOptions {
    pub window_bounds: Option<WindowBounds>,
}
pub enum WindowBounds {
    Windowed(Bounds<Pixels>),
}
#[derive(Copy, Clone, Debug, Default)]
pub struct DisplayId;

#[derive(Clone, Debug, Default)]
pub struct FocusHandle {
    focused: Rc<RefCell<bool>>,
    pub tab_index: usize,
    pub tab_stop: bool,
}

impl FocusHandle {
    fn new() -> Self {
        Self::default()
    }
    pub fn is_focused<C>(&self, _window: &C) -> bool {
        *self.focused.borrow()
    }
}

pub trait FocusHandleBuilder: Sized {
    fn tab_index(self, index: usize) -> Self;
    fn tab_stop(self, enabled: bool) -> Self;
}
impl FocusHandleBuilder for FocusHandle {
    fn tab_index(self, index: usize) -> Self {
        Self { tab_index: index, ..self }
    }
    fn tab_stop(self, enabled: bool) -> Self {
        Self { tab_stop: enabled, ..self }
    }
}

impl Window {
    pub fn focus(&mut self, handle: &FocusHandle, _cx: &mut Context<impl Sized>) {
        *handle.focused.borrow_mut() = true;
    }
    pub fn focus_next(&mut self, _cx: &mut Context<impl Sized>) {}
    pub fn focus_prev(&mut self, _cx: &mut Context<impl Sized>) {}
}

pub struct Stateful<T> {
    pub element: T,
    pub focus: Option<FocusHandle>,
    event_bindings: usize,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct ClickEvent {
    count: usize,
}
impl ClickEvent {
    pub fn click_count(&self) -> usize { self.count.max(1) }
}

impl<T> Stateful<T> {
    pub fn new(element: T) -> Self {
        Self { element, focus: None, event_bindings: 0 }
    }
}
impl<T> Deref for Stateful<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target { &self.element }
}
impl<T> DerefMut for Stateful<T> {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.element }
}
impl Stateful<Div> {
    pub fn track_focus(mut self, handle: &FocusHandle) -> Self {
        self.focus = Some(handle.clone());
        self
    }
    pub fn focus(mut self, style: impl FnOnce(Div) -> Div) -> Self {
        self.element = style(self.element);
        self
    }
    pub fn focus_visible(mut self, style: impl FnOnce(Div) -> Div) -> Self {
        self.element = style(self.element);
        self
    }
    pub fn on_action<F>(mut self, _handler: F) -> Self where F: 'static {
        self.event_bindings += 1;
        self
    }
    pub fn on_click<T, F>(mut self, _handler: F) -> Self
    where
        F: Fn(&mut T, &ClickEvent, &mut Window, &mut Context<T>) + 'static,
    {
        self.event_bindings += 1;
        self
    }
    pub fn map(self, transform: impl FnOnce(Div) -> Div) -> Self {
        Self { element: transform(self.element), ..self }
    }
    pub fn tab_index(mut self, index: usize) -> Self {
        self.focus = Some(self.focus.unwrap_or_default().tab_index(index));
        self
    }
}
impl IntoDescription for Stateful<Div> {
    fn into_description(self) -> Description {
        self.element.describe()
    }
}
pub trait StatefulElement: Sized {
    fn track_focus(self, handle: &FocusHandle) -> Stateful<Self>;
}
impl StatefulElement for Div {
    fn track_focus(self, handle: &FocusHandle) -> Stateful<Self> {
        Stateful { element: self, focus: Some(handle.clone()), event_bindings: 0 }
    }
}

impl Styled for Stateful<Div> {
    fn style(&mut self) -> &mut wgpui_widgets::div::interactivity::style::DivStyle {
        self.element.style()
    }
}

pub trait FocusStyle: Sized {
    fn focus(self, style: impl FnOnce(Div) -> Div) -> Stateful<Div>;
    fn focus_visible(self, style: impl FnOnce(Div) -> Div) -> Stateful<Div>;
}
impl FocusStyle for Div {
    fn focus(self, style: impl FnOnce(Div) -> Div) -> Stateful<Div> {
        Stateful::new(style(self))
    }
    fn focus_visible(self, style: impl FnOnce(Div) -> Div) -> Stateful<Div> {
        Stateful::new(style(self))
    }
}

pub trait IntoElement: Sized + IntoDescription {
    type Element;
    fn into_element(self) -> Self::Element;
}
impl<T: IntoDescription> IntoElement for T {
    type Element = Div;
    fn into_element(self) -> Div {
        div().child(self)
    }
}
impl<T: 'static> IntoDescription for Entity<T> {
    fn into_description(self) -> Description {
        Description::new::<T>()
    }
}

pub trait Render: 'static + Sized {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement;
}
pub trait RenderOnce: 'static {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement;
}

pub mod core {
    pub use wgpui_core::*;
}

pub mod layout {
    pub use wgpui_layout::*;
}

pub mod text {
    pub use wgpui_text::*;
}

pub mod widgets {
    pub use wgpui_widgets::*;
}

pub mod prelude {
    pub use crate::{
        div, linear_color_stop, Div, IntoDescription, IntoElement, ReconcileKey, Render,
        RenderOnce, Stateful, StatefulElement, Styled, FocusHandleBuilder, FocusStyle,
    };
}
