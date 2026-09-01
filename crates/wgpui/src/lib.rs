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
pub use wgpui_http_client as http_client;

pub use wgpui_http_client::{
    AppHttpClientExt, AsyncBody, BoxedHttpClient, Builder, FollowRedirects, HttpBody,
    HttpClient, HttpClientService, HttpClientWithProxy, HttpClientWithUrl, HttpRequestExt, Inner,
    Method, NullHttpClient, RedirectPolicy, Request, Response, Result, StatusCode, Uri, Url,
};

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
    ShadowClip, Underline,
};
pub use wgpui_core::reconcile;
pub use wgpui_core::reconcile::description::{
    RawText, RawTextKey, TextDecoration, TextOptions,
};
pub use wgpui_core::reconcile::{Description, ElementId, FramePlan, ReconcileKey, Reconciler};
pub use wgpui_core::reconcile::{StateKey, StateScope};
pub use wgpui_layout::taffy_tree::{LayoutNodeId as LayoutId, LayoutStyle as Style};
pub use wgpui_text::shaping::ShapedLine;
pub use wgpui_core::reconcile::{
    RetainedElementSnapshot, RetainedFrameSnapshot, RetainedWalk, RetainedWalkNode, TileOwnership,
};
pub use wgpui_core::damage::{DamagePlan, DamageReason, DamageRecord};
pub use wgpui_core::scene;
pub use wgpui_core::window::{
    AnimationClock, AnimationScheduler, DispatchTree, FocusManager, Keymap, WindowTimers,
};
pub use wgpui_core::{
    Action, App, BackgroundExecutor, ClipboardItem, ClickEvent, CloseState, Context, DispatchNodeId, DragData, DragHoverEvent,
    DropEvent, Entity, EntityError, EntityFactory, EntityId, EventResult, FocusEvent, FocusHandle,
    FocusId, FocusTransition, Focusable, HitTestIndex, Hitbox, HitboxId, ImeEvent,
    InputEvent, KeyBinding, KeyDownEvent, KeyParseError, KeyUpEvent, KeyboardButton,
    KeyboardClickEvent, Keystroke, Menu, MenuItem, Modifiers, ModifiersChangedEvent, MouseButton, MouseButtonState,
    MouseClickEvent, MouseDownEvent, MouseMoveEvent, MouseUpEvent, ScrollWheelEvent, Subscription, TextInputEvent,
    Task, TaskError, Timer, TimerHandle, TimerId, TimerState, WeakEntity, WindowClosedSubscription,
    TitlebarOptions, WindowAppearance, WindowBackgroundAppearance, WindowDecorations, WindowId,
    WindowKind, WindowList,
};

/// Compatibility spelling used by the restored examples.
pub type AppContext = App;
pub use wgpui_core::{
    Element, Stateful,
};
pub use wgpui_wgpu::window::application::{
    AppWindowExt, Component, Render, RenderOnce, render_description,
};
pub use wgpui_core::element::IntoElement;
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
    Animation, AnimationElement, AnimationExt, AnimationSample, AnimationTimeline, Transformation,
    bounce, ease_in_out, ease_out_quint, linear, percentage, quadratic,
};
pub use wgpui_widgets::canvas::{
    Canvas, CanvasContext, IntoCanvasBounds, IntoCanvasColor, PathBuilder, PathStyle, canvas, fill,
};
pub use wgpui_widgets::div::StatefulDiv;
pub use wgpui_widgets::div::interactivity::style::{
    BoxShadow, Corners, CursorStyle, DivStyle, Edges, LinearGradient, Pattern, RadialGradient,
};
pub use wgpui_widgets::div::{Div, div};
pub use wgpui_widgets::image_cache::{
    DecodedFrame, DecodedImage, ImageCache, ImageDecodeError, decode, decode_async, decode_svg_at,
};
pub use wgpui_widgets::img::{
    img, img_with_engine, ImageEngine, ImageLoadState, ImageSourceId, ImageStyle, Img,
    ImgBuilder, ObjectFit, SharedImageEngine,
};
pub use wgpui_widgets::list;
pub use wgpui_widgets::list::uniform_list::{
    UniformItemTransform, UniformList, UniformListState, uniform_list,
};
pub use wgpui_widgets::list::virtual_list::{
    VirtualItemTransform, VirtualList, VirtualListScrollController, VirtualListState, vlist,
    virtual_list,
};
pub use wgpui_widgets::scroll::{ScrollHandle, ScrollPhysics, ScrollPhysicsMode, ScrollStrategy};
pub type UniformListScrollHandle = ScrollHandle;

/// Compatibility name for element identity used by custom elements.
pub type GlobalElementId = ElementId;

/// Compatibility name for inspector identity used by custom elements.
pub type InspectorElementId = ElementId;

/// Legacy layer tuning accepted by the retained scrolling examples.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LayerPolicy {
    pub overdraw_margin: Size<Pixels>,
}

impl Default for LayerPolicy {
    fn default() -> Self {
        Self {
            overdraw_margin: Size::ZERO,
        }
    }
}

impl From<LayerPolicy> for BoundaryPolicy {
    fn from(policy: LayerPolicy) -> Self {
        Self {
            buffering: Buffering::Margin(Some(policy.overdraw_margin)),
            ..Self::default()
        }
    }
}

/// Process-wide render counters for diagnostics and examples.
pub mod render_stats {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicI8, Ordering};
    use std::sync::{LazyLock, Mutex};
    use std::time::{Duration, Instant};

    #[derive(Clone, Debug, Default, PartialEq, Eq)]
    pub struct TimerSnapshot {
        pub count: u64,
        pub total: Duration,
        pub max: Duration,
    }

    #[derive(Clone, Debug, Default, PartialEq, Eq)]
    pub struct Snapshot {
        pub counters: BTreeMap<&'static str, u64>,
        pub timers: BTreeMap<&'static str, TimerSnapshot>,
    }

    #[derive(Default)]
    struct Registry {
        counters: BTreeMap<&'static str, u64>,
        timers: BTreeMap<&'static str, TimerSnapshot>,
    }

    static REGISTRY: LazyLock<Mutex<Registry>> = LazyLock::new(|| Mutex::new(Registry::default()));
    static FORCE_ENABLED: AtomicI8 = AtomicI8::new(0);

    fn registry() -> std::sync::MutexGuard<'static, Registry> {
        REGISTRY
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub fn enabled() -> bool {
        match FORCE_ENABLED.load(Ordering::Relaxed) {
            1 => true,
            2 => false,
            _ => std::env::var("WGPUI_RENDER_STATS")
                .map(|value| !value.is_empty() && value != "0")
                .unwrap_or(false),
        }
    }

    pub fn set_force_enabled(enabled: bool) {
        FORCE_ENABLED.store(if enabled { 1 } else { 2 }, Ordering::Relaxed);
    }

    pub fn clear_force_enabled() {
        FORCE_ENABLED.store(0, Ordering::Relaxed);
    }

    pub fn add(name: &'static str, amount: u64) {
        if enabled() {
            registry()
                .counters
                .entry(name)
                .and_modify(|value| *value += amount)
                .or_insert(amount);
        }
    }

    pub fn count(name: &'static str) {
        add(name, 1);
    }

    pub fn record(name: &'static str, duration: Duration) {
        if enabled() {
            let mut registry = registry();
            let timer = registry.timers.entry(name).or_default();
            timer.count += 1;
            timer.total += duration;
            timer.max = timer.max.max(duration);
        }
    }

    pub fn scope(name: &'static str) -> Option<Scope> {
        enabled().then(|| Scope {
            name,
            start: Instant::now(),
        })
    }

    pub fn snapshot() -> Snapshot {
        let registry = registry();
        Snapshot {
            counters: registry.counters.clone(),
            timers: registry.timers.clone(),
        }
    }

    pub fn reset() {
        *registry() = Registry::default();
    }

    pub struct Scope {
        name: &'static str,
        start: Instant,
    }

    impl Drop for Scope {
        fn drop(&mut self) {
            record(self.name, self.start.elapsed());
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn forced_counters_are_visible_in_snapshots() {
            set_force_enabled(true);
            reset();
            count("test: counter");
            assert_eq!(snapshot().counters.get("test: counter"), Some(&1));
            clear_force_enabled();
            reset();
        }
    }
}

pub use wgpui_widgets::styled::{
    IntoStyleBackground, IntoStyleColor, IntoStyleDimension, IntoStylePixels, Styled, TextAlign,
    TextOverflow,
};
pub use wgpui_widgets::styled_text::{
    Highlight, HighlightStyle, StrikethroughStyle, StyledText, TextEngine, TextStyle,
    UnderlineStyle,
};
pub use wgpui_widgets::svg::{
    svg, svg_with_engine, load as load_svg, Svg, SvgBuilder, SvgKey,
};
pub use wgpui_widgets::wgpu_surface::{SurfaceId, SurfaceStyle, WgpuSurface, WgpuSurfaceKey};
pub use wgpui_widgets::assets::{
    AssetLoadError, AssetSource, ImageAssetLoader, ImageSource, Resource, SharedUri,
};

pub use wgpui_wgpu::debug::{PerformanceDebug, TileRefreshFlash};
pub use wgpui_wgpu::window::application::{
    Application, ApplicationError, ClipboardError, ConfiguredApplication, DisplayId, FrameReport,
    NativeApplication, Window, WindowHandle, WindowOptions,
};
pub use wgpui_wgpu::window::surface::WgpuSurfaceHandle;

/// Lower a native WGPU surface handle into a retained surface element.
pub fn wgpu_surface(handle: WgpuSurfaceHandle) -> WgpuSurface {
    WgpuSurface::new(SurfaceId::from_raw(handle.id()))
}

pub mod prelude {
    pub use crate::{
        AppWindowExt, Application, Description, Div, Element, EntityFactory, IntoElement, Render,
        RenderOnce,
        Stateful, StatefulDiv, Styled, UniformList,
        VirtualList, Window, WindowOptions, div, uniform_list, virtual_list,
    };
}
