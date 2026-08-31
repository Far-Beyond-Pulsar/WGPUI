//! `wgpui-core` — patch protocol, persistent scene, ambient reconciliation,
//! boundary/tile policy, and invalidation. See docs/gpu-native-architecture.md
//! §3.1 for this crate's file map and role.
//!
//! No live `wgpu::Device` appears anywhere in this crate — it owns the
//! *shapes* of GPU work (shader source as text, buffer/binding layout
//! descriptors as plain Rust structs) so it stays unit-testable headlessly,
//! matching `slab.rs`'s existing "no device, no queue" module doc in the
//! legacy backend (§3.1 intro). `wgpui-wgpu` (§3.5) is what actually creates
//! pipelines and dispatches compute/render work — Phase 0's spike harnesses
//! (§8) live there too, in `crates/wgpui-wgpu/benches/`, since they need a
//! real device.
#![allow(dead_code)]

pub mod app;
pub use app::entity::EntityError;
pub use app::{
    App, Context, Entity, EntityFactory, EntityId, Subscription, Task, TaskError, WeakEntity,
    WindowClosedSubscription, WindowId, WindowList,
};
pub use window::{
    ClipboardItem, ClickEvent, CloseState, DispatchNodeId, DragData, DragHoverEvent, DropEvent, EventResult, FocusEvent, FocusHandle, FocusId, FocusManager, ImeEvent,
    FocusTransition, Focusable, HitTestIndex, Hitbox, HitboxId, InputEvent, KeyBinding,
    KeyDownEvent, KeyParseError, KeyUpEvent, KeyboardButton, KeyboardClickEvent, Keystroke, Menu,
    MenuItem, Modifiers, ModifiersChangedEvent, MouseButton, MouseButtonState, MouseClickEvent,
    MouseDownEvent, MouseMoveEvent, MouseUpEvent, ScrollWheelEvent, TextInputEvent,
    TimerHandle, TimerId, TimerState, TitlebarOptions, WindowAppearance,
    WindowBackgroundAppearance, WindowDecorations, WindowKind,
    ScrollRootHandle, WindowOptions,
};
pub mod boundary;
pub mod color;
pub mod damage;
pub mod diagnostics;
pub mod element;
pub mod geometry;
pub mod hooks;
pub mod indirect;
pub mod invalidation;
pub mod occlusion;
pub mod ordering;
pub mod patch;
pub mod reconcile;
pub mod scene;
pub mod shaders;
pub mod test_support;
pub mod window;
pub use element::{
    Component, Element, IntoElement, Render, RenderOnce, Stateful, render_description,
};
pub mod action;
pub use action::Action;
