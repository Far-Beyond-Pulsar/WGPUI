//! The 2.0 compatibility façade. Its crate name is `gpui` so unchanged
//! examples exercise the public surface rather than a source-level port.
//!
//! Only symbols that already have a real 2.0 implementation are exported.
//! The missing legacy lifecycle and frontend symbols remain compile failures
//! in the probe instead of becoming behaviorless placeholders.

use std::cell::RefCell;
use std::future::Future;
use std::rc::Rc;

pub use wgpui_core::boundary::policy::Pixels;
pub use wgpui_core::geometry::Rect;
pub use wgpui_core::patch::primitive::{PrimitiveKind, Quad, Shadow, Underline};
pub use wgpui_core::reconcile::description::{Description, ElementId};
pub use wgpui_core::reconcile::diff_key::ReconcileKey;
pub use wgpui_core::reconcile::instance::{ElementInstance, InstanceKey, InstanceTable};
pub use wgpui_layout::taffy_tree::LayoutNodeId;
pub use wgpui_text::shaping::{Font, FontId, FontStyle, FontWeight, SharedString};
pub use wgpui_widgets::div::{Div, IntoDescription, div};
pub use wgpui_widgets::styled::Styled;

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

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Bounds<T> {
    pub origin: Point<T>,
    pub size: Size<T>,
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

pub fn px(value: f32) -> Pixels {
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

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Hsla {
    pub h: f32,
    pub s: f32,
    pub l: f32,
    pub a: f32,
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
pub fn red() -> Rgba {
    rgb(0xff0000)
}
pub fn green() -> Rgba {
    rgb(0x00ff00)
}
pub fn blue() -> Rgba {
    rgb(0x0000ff)
}
pub fn black() -> Rgba {
    rgb(0x000000)
}
pub fn white() -> Rgba {
    rgb(0xffffff)
}
pub fn yellow() -> Rgba {
    rgb(0xffff00)
}

pub struct Task<T>(Option<T>);
impl<T> Task<T> {
    pub fn ready(value: T) -> Self {
        Self(Some(value))
    }
    pub fn detach(self) {}
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
}
impl<T> Context<T> {
    fn from_entity(entity: Entity<T>, notifications: Rc<RefCell<u64>>) -> Self {
        Self {
            entity: Some(entity),
            notifications,
        }
    }
    pub fn entity(&self) -> Entity<T> {
        self.entity.clone().expect("context entity is initialized")
    }
    pub fn notify(&mut self) {
        *self.notifications.borrow_mut() += 1;
    }
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    pub fn new<U>(&mut self, build: impl FnOnce(&mut Context<U>) -> U) -> Entity<U> {
        let entity = Entity(Rc::new(RefCell::new(None)));
        let mut context = Context {
            entity: None,
            notifications: Rc::clone(&self.notifications),
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
}
impl App {
    pub fn new() -> Self {
        Self {
            notifications: Rc::new(RefCell::new(0)),
            descriptions: Vec::new(),
            active: false,
        }
    }
    pub fn new_entity<T>(&mut self, build: impl FnOnce(&mut Context<T>) -> T) -> Entity<T> {
        let entity = Entity(Rc::new(RefCell::new(None)));
        let mut context = Context {
            entity: None,
            notifications: Rc::clone(&self.notifications),
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
            root_entity.update(|root| root.render(&mut window, &mut context).into_description());
        self.descriptions.push(description);
        Ok(WindowHandle)
    }
    pub fn activate(&mut self, active: bool) {
        self.active = active;
    }
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
#[derive(Default)]
pub struct WindowOptions {
    pub window_bounds: Option<WindowBounds>,
}
pub enum WindowBounds {
    Windowed(Bounds<Pixels>),
}
#[derive(Copy, Clone, Debug, Default)]
pub struct DisplayId;

pub trait IntoElement: Sized {
    type Element;
    fn into_element(self) -> Self::Element;
    fn into_description(self) -> Description;
}
impl IntoElement for Div {
    type Element = Div;
    fn into_element(self) -> Div {
        self
    }
    fn into_description(self) -> Description {
        IntoDescription::into_description(self)
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
        Div, IntoDescription, IntoElement, ReconcileKey, Render, RenderOnce, Styled, div,
    };
}
