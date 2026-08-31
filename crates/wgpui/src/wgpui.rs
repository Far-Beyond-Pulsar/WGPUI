//! The native WGPUI 2.0 public API.
//!
//! This crate is intentionally a direct namespace over the retained core,
//! widgets, text, layout, and WGPU lifecycle. It has no dependency on the
//! frozen compatibility crate or the reference implementation under `old/`.

pub use wgpui_core as core;
pub use wgpui_layout as layout;
pub use wgpui_text as text;
pub use wgpui_wgpu as gpu;
pub use wgpui_widgets as widgets;

pub use wgpui_core::actions;
pub use wgpui_core::boundary;
pub use wgpui_core::boundary::{BoundaryPolicy, Buffering, Retention};
pub use wgpui_core::color::{
    Background, ColorSpace, Colors, GradientStop, Hsla, LinearColorStop, Rgba, black, blue,
    gradient_color_stop, green, hsla, linear_color_stop, linear_gradient, opaque_grey,
    pattern_slash, radial_gradient, red, rgb, rgba, solid_background, transparent_black,
    transparent_white, white, yellow,
};
pub use wgpui_core::geometry;
pub use wgpui_core::geometry::Rect;
pub use wgpui_core::geometry::{
    AbsoluteLength, Bounds, DefiniteLength, Half, Length, Pixels, Point, Rems, Size, WindowBounds,
    phi, point, px, relative, rems, size,
};
pub use wgpui_core::invalidation;
pub use wgpui_core::patch;
pub use wgpui_core::patch::emit::{Emit, EmitContext, Emitter, FrameEmission};
pub use wgpui_core::patch::primitive::{
    AtlasTileId, BackdropFilter, Glyph, GlyphRun, Path, PathVertex, PolySprite, Quad, Shadow,
    Underline,
};
pub use wgpui_core::reconcile;
pub use wgpui_core::reconcile::description::{
    RawText, RawTextKey, TextDecoration, TextOptions,
};
pub use wgpui_core::reconcile::{Description, ElementId, FramePlan, ReconcileKey, Reconciler};
pub use wgpui_core::reconcile::{StateKey, StateScope};
pub use wgpui_core::scene;
pub use wgpui_core::window::{
    AnimationClock, AnimationScheduler, DispatchTree, FocusManager, Keymap, WindowTimers,
};
pub use wgpui_core::{
    Action, App, ClickEvent, CloseState, Context, DispatchNodeId, DragData, DragHoverEvent,
    DropEvent, Entity, EntityError, EntityFactory, EntityId, EventResult, FocusEvent, FocusHandle,
    FocusId, FocusTransition, Focusable, HitTestIndex, Hitbox, HitboxId,
    InputEvent, KeyBinding, KeyDownEvent, KeyParseError, KeyUpEvent, KeyboardButton,
    KeyboardClickEvent, Keystroke, Menu, MenuItem, Modifiers, MouseButton, MouseButtonState,
    MouseClickEvent, MouseDownEvent, MouseMoveEvent, MouseUpEvent, ScrollWheelEvent, Subscription,
    Task, TaskError, TimerHandle, TimerId, TimerState, WeakEntity, WindowClosedSubscription,
    WindowId, WindowList,
};
pub use wgpui_core::{
    Component, Element, IntoElement, Render, RenderOnce, Stateful, render_description,
};
pub use wgpui_layout::{
    AvailableSpace, Dimension, Display, FlexDirection, IntrinsicSize, LayoutSize, LayoutStyle,
    LayoutTree, Measure,
};
pub use wgpui_layout::taffy_tree::{
    AlignContent, AlignItems, BoxSizing, FlexWrap, GridPlacement, GridTemplateComponent,
    LengthPercentage, LengthPercentageAuto, Overflow, Position, TrackSizingFunction,
};
pub use wgpui_macros::IntoElement;
pub use wgpui_text::patch::{RunPlacement, glyph_runs};
pub use wgpui_text::shaping::{
    Font, FontId, FontRun, FontStyle, FontWeight, SharedString, TextMeasurement, TextShaper,
};
pub use wgpui_widgets::animation::{
    Animation, AnimationElement, AnimationExt, AnimationSample, AnimationTimeline, ease_in_out,
    ease_out_quint, linear, quadratic,
};
pub use wgpui_widgets::canvas::{
    Canvas, CanvasContext, IntoCanvasBounds, IntoCanvasColor, PathBuilder, PathStyle, canvas, fill,
};
pub use wgpui_widgets::div::StatefulDiv;
pub use wgpui_widgets::div::interactivity::style::{
    BoxShadow, Corners, DivStyle, Edges, LinearGradient, Pattern, RadialGradient,
};
pub use wgpui_widgets::div::{Div, div};
pub use wgpui_widgets::image_cache::{
    DecodedFrame, DecodedImage, ImageCache, ImageDecodeError, decode, decode_async, decode_svg_at,
};
pub use wgpui_widgets::img::{
    ImageEngine, ImageLoadState, ImageSourceId, ImageStyle, Img, ObjectFit, SharedImageEngine, img,
};
pub use wgpui_widgets::list;
pub use wgpui_widgets::list::uniform_list::{
    UniformItemTransform, UniformList, UniformListState, uniform_list,
};
pub use wgpui_widgets::list::virtual_list::{
    VirtualItemTransform, VirtualList, VirtualListState, virtual_list,
};
pub use wgpui_widgets::scroll::{ScrollHandle, ScrollPhysics, ScrollPhysicsMode};
pub use wgpui_widgets::styled::{
    IntoStyleBackground, IntoStyleColor, IntoStyleDimension, IntoStylePixels, Styled, TextAlign,
    TextOverflow,
};
pub use wgpui_widgets::styled_text::{
    Highlight, HighlightStyle, StrikethroughStyle, StyledText, TextEngine, TextStyle,
    UnderlineStyle,
};
pub use wgpui_widgets::svg::{Svg, SvgKey, load as load_svg, svg};
pub use wgpui_widgets::wgpu_surface::{SurfaceId, SurfaceStyle, WgpuSurface, WgpuSurfaceKey};

pub use wgpui_wgpu::window::application::{
    Application, ApplicationError, DisplayId, FrameReport, NativeApplication, Window, WindowHandle,
    WindowOptions,
};
pub use wgpui_wgpu::window::surface::WgpuSurfaceHandle;

/// Lower a native WGPU surface handle into a retained surface element.
pub fn wgpu_surface(handle: WgpuSurfaceHandle) -> WgpuSurface {
    WgpuSurface::new(SurfaceId::from_raw(handle.id()))
}
pub use wgpui_wgpu::debug::{PerformanceDebug, TileRefreshFlash};

pub mod prelude {
    pub use crate::{
        Application, Description, Div, Element, EntityFactory, IntoElement, Render, RenderOnce,
        Stateful, StatefulDiv, Styled, UniformList, VirtualList, Window, WindowOptions, div,
        uniform_list, virtual_list,
    };
}
