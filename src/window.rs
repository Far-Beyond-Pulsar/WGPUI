#[cfg(any(feature = "inspector", debug_assertions))]
use crate::Inspector;
use crate::layer::{Layer, LayerCacheKey, LayerItem};
use crate::platform::cross::slab::SlabKind;
use crate::scene_pack::{
    FallbackReason, PackedLayer, RecordedSlabPack, SlabPackPiece, first_unsupported_kind,
    pack_layer_items,
};
use crate::time_ext::Instant;
use crate::util::post_inc;
use crate::util::{ResultExt, measure};
use crate::{
    Action, AnyDrag, AnyElement, AnyImageCache, AnyTooltip, AnyView, App, AppContext, Arena, Asset,
    AsyncWindowContext, AvailableSpace, BackdropFilter, Background, BorderStyle, Bounds, BoxShadow,
    Capslock, Context, Corners, CursorStyle, Decorations, DevicePixels, DispatchActionListener,
    DispatchNodeId, DispatchTree, DisplayId, Edges, Effect, ElementGeometry, Entity, EntityId,
    EventEmitter, FileDropEvent, Filter, FilterBoundary, FontId, Global, GlobalElementId, GlyphId,
    GpuSpecs, Hsla, InputHandler, InstanceKey, IsZero, KeyBinding, KeyContext, KeyDownEvent, KeyEvent,
    KeyUpEvent, Keystroke, KeystrokeEvent, LayerId, LayerKey, LayerPolicy, LayerTransform,
    LayoutId, LineLayoutIndex, Modifiers, ModifiersChangedEvent, MonochromeSprite, MouseButton,
    MouseDownEvent, MouseEvent, MouseExitEvent, MouseMoveEvent, MouseUpEvent, Path, Pixels,
    PlatformAtlas, PlatformDisplay, PlatformInput, PlatformInputHandler, PlatformWindow, Point,
    PolychromeSprite, Priority, PromptButton, PromptLevel, Quad, ReconcileKey, Render,
    RenderGlyphParams,
    RenderImage, RenderImageParams, RenderSvgParams, Replay, ResizeEdge, SMOOTH_SVG_SCALE_FACTOR,
    SUBPIXEL_VARIANTS_X, SUBPIXEL_VARIANTS_Y, ScaledPixels, Scene, ScrollWheelEvent, Shadow,
    SharedString, Size, StrikethroughStyle, Style, SubscriberSet, Subscription, SystemWindowTab,
    SystemWindowTabController, TabStopMap, TaffyLayoutEngine, Task, TextColor, TextStyle,
    TextStyleRefinement, TransformationMatrix, Underline, UnderlineStyle, WindowAppearance,
    WindowBackgroundAppearance, WindowBounds, WindowControls, WindowDecorations, WindowOptions,
    WindowParams, WindowTextSystem, point, prelude::*, px, rems, size, transparent_black,
};
use anyhow::{Context as _, Result, anyhow};
use collections::{FxHashMap, FxHashSet};
use derive_more::{Deref, DerefMut};
use futures::FutureExt;
use futures::channel::oneshot;
use itertools::FoldWhile::{Continue, Done};
use itertools::Itertools;
use parking_lot::RwLock;
use raw_window_handle::{HandleError, HasDisplayHandle, HasWindowHandle};
use refineable::Refineable;
use slotmap::SlotMap;
use smallvec::SmallVec;
use std::{
    any::{Any, TypeId},
    borrow::Cow,
    cell::{Cell, RefCell},
    cmp,
    fmt::{Debug, Display},
    hash::{Hash, Hasher},
    marker::PhantomData,
    mem,
    ops::{DerefMut, Range},
    rc::Rc,
    sync::{
        Arc, Weak,
        atomic::{AtomicUsize, Ordering::SeqCst},
    },
    time::Duration,
};
use uuid::Uuid;

mod prompts;

/// One maximal run of a layer's own packable primitives, plus where nested
/// layers sat between them.
pub(crate) enum SlabSegment {
    /// Item-index range (all primitives) with its packed form.
    Stretch(Range<usize>, Box<PackedLayer>),
    Nested(LayerKey),
}

/// The overscroll-buffer state (#96) of the layer a subtree is being
/// prepainted inside, handed to buffered elements via
/// [`Window::prepaint_layer_buffer`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct PrepaintLayerBuffer {
    pub key: LayerKey,
    pub margin: Size<Pixels>,
    /// The scroll-space position the buffer was rendered at; opaque to the
    /// framework, meaningful to the element that set it.
    pub anchor: Point<Pixels>,
    /// How far the content has scrolled since the buffer was rendered.
    pub content_offset: Point<Pixels>,
    /// Whether the layer will composite this frame (skip per-item work).
    pub will_composite: bool,
    /// Whether this frame's draw will re-record the layer (paint the full
    /// buffer range and re-anchor).
    pub refilling: bool,
    /// Whether the texture covers the full buffer yet; false until the first
    /// buffered record.
    pub buffer_ready: bool,
}

/// Pack one stretch of consecutive primitive items. The result is stored
/// relative to the layer's composited `origin` (see `make_packed_relative`).
///
/// Unsupported kinds were rejected for the whole layer before segmentation
/// (see `build_slab_segments`), and `pack_layer_items` still guards its own
/// contract, so this is the gather-and-sort step only — the pre-slab
/// redundant full validation pass is gone.
fn pack_stretch(
    items: &[LayerItem],
    start: usize,
    end: usize,
    origin: [f32; 2],
) -> Option<Box<PackedLayer>> {
    match pack_layer_items(&items[start..end]) {
        crate::scene_pack::PackOutcome::Packed(mut packed) => {
            crate::platform::cross::slab_gpu::make_packed_relative(&mut packed, origin);
            Some(packed)
        }
        crate::scene_pack::PackOutcome::FellBack(_) => None,
    }
}

/// Whether `previous` and `incoming` cache keys describe the same retained
/// content at two different origins: every non-bounds input matches, and the
/// bounds differ by position alone.
///
/// A resize is deliberately not a transform-only move: the packed instance
/// bytes were built for the old extent, so only a re-record can produce the
/// new geometry.
fn is_transform_only_move(previous: &LayerCacheKey, incoming: &LayerCacheKey) -> bool {
    previous.bounds.size == incoming.bounds.size
        && previous.bounds.origin != incoming.bounds.origin
        && previous.content_mask == incoming.content_mask
        && previous.opacity == incoming.opacity
        && previous.scale_factor == incoming.scale_factor
}

/// Split a layer's items into slab segments: stretches of its own primitives
/// separated by nested-layer references. `None` when nothing is packable —
/// including layers holding a primitive with no slab kind, or items that are
/// all nested references.
///
/// Test-only probe today: production packs at record time via
/// [`pack_layer_at_record`], and the composite path splices cached bytes. It
/// keeps the unsupported-kind rejection side-effect-free so probes stay
/// quiet (a `debug_assert` in this position would fire for the supported
/// backdrop-filter-inside-a-layer configuration, which exists precisely to
/// exercise the fallback); production reporting happens in
/// `pack_layer_at_record`.
#[cfg(test)]
pub(crate) fn build_slab_segments(
    items: &[LayerItem],
    origin: [f32; 2],
) -> Option<Vec<SlabSegment>> {
    // `None` from the scan means "no rejection"; `?` would invert that into
    // rejecting exactly the packable layers.
    if first_unsupported_kind(items).is_some() {
        return None;
    }
    build_slab_stretches(items, origin)
}

/// Segmentation proper, after the caller has handled unsupported kinds.
fn build_slab_stretches(
    items: &[LayerItem],
    origin: [f32; 2],
) -> Option<Vec<SlabSegment>> {
    let mut segments: Vec<SlabSegment> = Vec::new();
    let mut stretch_start: Option<usize> = None;
    for (index, item) in items.iter().enumerate() {
        match item {
            LayerItem::Primitive(_) => {
                stretch_start.get_or_insert(index);
            }
            LayerItem::Nested(nested) => {
                if let Some(start) = stretch_start.take() {
                    let packed = pack_stretch(items, start, index, origin)?;
                    segments.push(SlabSegment::Stretch(start..index, packed));
                }
                segments.push(SlabSegment::Nested(*nested));
            }
        }
    }
    if let Some(start) = stretch_start.take() {
        let packed = pack_stretch(items, start, items.len(), origin)?;
        segments.push(SlabSegment::Stretch(start..items.len(), packed));
    }

    let has_stretch = segments
        .iter()
        .any(|segment| matches!(segment, SlabSegment::Stretch(..)));
    if !has_stretch {
        return None;
    }
    Some(segments)
}

impl RecordedSlabPack {
    /// Absorb `build_slab_segments`' output: hoist the per-kind totals and
    /// each stretch's offset runs out of what used to be the per-frame
    /// emission loop, so a composite only clones them.
    fn from_segments(segments: Vec<SlabSegment>) -> Self {
        let mut totals = [0u32; SlabKind::COUNT];
        for segment in &segments {
            let SlabSegment::Stretch(_, packed) = segment else {
                continue;
            };
            totals[SlabKind::Quads.index()] += packed.quads.len() as u32;
            totals[SlabKind::Shadows.index()] += packed.shadows.len() as u32;
            totals[SlabKind::Paths.index()] += packed.total_path_vertices();
            totals[SlabKind::Underlines.index()] += packed.underlines.len() as u32;
            totals[SlabKind::MonoSprites.index()] += packed.mono_sprites.len() as u32;
            totals[SlabKind::PolySprites.index()] += packed.poly_sprites.len() as u32;
        }
        // Running per-kind offsets: every stretch draws out of the same slab
        // ranges, so a later stretch's runs start where earlier ones' bytes
        // end.
        let mut offsets = [0u32; SlabKind::COUNT];
        // Stretches arrive contiguous and ascending — nested references sit
        // between them, never inside — which the offset accumulation below
        // depends on.
        #[cfg(debug_assertions)]
        let mut stretch_cursor = 0usize;
        let pieces = segments
            .into_iter()
            .map(|segment| match segment {
                SlabSegment::Stretch(range, packed) => {
                    #[cfg(debug_assertions)]
                    {
                        debug_assert!(
                            range.start >= stretch_cursor,
                            "stretch ranges must ascend"
                        );
                        stretch_cursor = range.end;
                    }
                    let runs = packed
                        .runs
                        .iter()
                        .map(|run| crate::scene::SlabRun {
                            kind: run.kind,
                            start: offsets[run.kind.index()] + run.start,
                            count: run.count,
                            texture_id: run.texture_id,
                        })
                        .collect();
                    offsets[SlabKind::Quads.index()] += packed.quads.len() as u32;
                    offsets[SlabKind::Shadows.index()] += packed.shadows.len() as u32;
                    offsets[SlabKind::Paths.index()] += packed.total_path_vertices();
                    offsets[SlabKind::Underlines.index()] += packed.underlines.len() as u32;
                    offsets[SlabKind::MonoSprites.index()] += packed.mono_sprites.len() as u32;
                    offsets[SlabKind::PolySprites.index()] += packed.poly_sprites.len() as u32;
                    SlabPackPiece::Stretch {
                        runs,
                        packed: Arc::from(packed),
                    }
                }
                SlabSegment::Nested(nested) => SlabPackPiece::Nested(nested),
            })
            .collect();
        RecordedSlabPack { totals, pieces }
    }
}

/// Pack a freshly-recorded layer's own content once, for every composite
/// until the next record to splice from.
///
/// `Some(Err)` caches a fallback verdict after reporting it — warn-once plus
/// counter, once per record rather than once per frame. The report is
/// deliberately NOT accompanied by the `debug_assert` that guards
/// `pack_layer_items` itself: an unsupported kind at record time is a
/// supported outcome (the layer composites through the legacy path), not a
/// producer bug. `None` when nothing was packable at all — empty layers and
/// nested-reference-only layers have always composited legacy-style.
fn pack_layer_at_record(
    items: &[LayerItem],
    origin: [f32; 2],
) -> Option<Result<Arc<RecordedSlabPack>, FallbackReason>> {
    profiling::scope!("wgpui: pack layer at record");
    if let Some(reason) = first_unsupported_kind(items) {
        crate::scene_pack::report_rejection(reason);
        return Some(Err(reason));
    }
    let segments = build_slab_stretches(items, origin)?;
    Some(Ok(Arc::new(RecordedSlabPack::from_segments(segments))))
}

/// The pre-slab composite replay: re-emit every retained primitive through
/// `push_retained`, recursing into nested layers.
fn composite_layer_legacy(
    scene: &mut Scene,
    layers: &mut FxHashMap<LayerKey, Layer>,
    key: LayerKey,
    frame: u64,
    scale_factor: f32,
) {
    let Some(layer) = layers.get_mut(&key) else {
        // A nested layer evicted out from under its parent. The parent was
        // judged clean, so this cannot produce wrong pixels on its own — but
        // it does mean content silently vanished, so make it visible rather
        // than merely absent.
        crate::render_stats::count("layer: nested reference missing");
        return;
    };
    layer.last_visited = frame;
    let bounds = layer.cache_key.bounds.scale(scale_factor);
    // Taken out so the recursive call can borrow the map; put back below. A
    // layer never appears twice in one composite tree, so nothing can observe
    // the gap.
    let items = mem::take(&mut layer.items);

    scene.begin_layer(key, bounds, false);
    for item in &items {
        match item {
            LayerItem::Primitive(primitive) => scene.push_retained(primitive),
            LayerItem::Nested(nested) => {
                composite_layer_legacy(scene, layers, *nested, frame, scale_factor)
            }
        }
    }
    scene.end_layer();

    if let Some(layer) = layers.get_mut(&key) {
        layer.items = items;
    }
}

use crate::util::atomic_incr_if_not_zero;
pub use prompts::*;

/// Default window size used when no explicit size is provided.
pub const DEFAULT_WINDOW_SIZE: Size<Pixels> = size(px(1536.), px(1095.));

/// A 6:5 aspect ratio minimum window size to be used for functional,
/// additional-to-main-Zed windows, like the settings and rules library windows.
pub const DEFAULT_ADDITIONAL_WINDOW_SIZE: Size<Pixels> = Size {
    width: Pixels(900.),
    height: Pixels(750.),
};

/// Represents the two different phases when dispatching events.
#[derive(Default, Copy, Clone, Debug, Eq, PartialEq)]
pub enum DispatchPhase {
    /// After the capture phase comes the bubble phase, in which mouse event listeners are
    /// invoked front to back and keyboard event listeners are invoked from the focused element
    /// to the root of the element tree. This is the phase you'll most commonly want to use when
    /// registering event listeners.
    #[default]
    Bubble,
    /// During the initial capture phase, mouse event listeners are invoked back to front, and keyboard
    /// listeners are invoked from the root of the tree downward toward the focused element. This phase
    /// is used for special purposes such as clearing the "pressed" state for click events. If
    /// you stop event propagation during this phase, you need to know what you're doing. Handlers
    /// outside of the immediate region may rely on detecting non-local events during this phase.
    Capture,
}

impl DispatchPhase {
    /// Returns true if this represents the "bubble" phase.
    #[inline]
    pub fn bubble(self) -> bool {
        self == DispatchPhase::Bubble
    }

    /// Returns true if this represents the "capture" phase.
    #[inline]
    pub fn capture(self) -> bool {
        self == DispatchPhase::Capture
    }
}

/// After this many consecutive draws that each ended with deferred
/// invalidations, warn once. A view that invalidates from inside its own render
/// now genuinely re-renders forever; that is arguably correct, but it is also
/// a self-perpetuating frame cost and should be visible rather than silent.
const DEFERRED_NOTIFY_LOOP_WARN_DRAWS: u32 = 120;

/// Identifies one retained element instance, for the purpose of naming it in
/// an explicit [`InvalidationRequest`].
///
/// Defined alongside [`InvalidationScope`] rather than with the machinery that
/// will produce it, so that the scope enum is complete from the start and later
/// phases add behaviour to a variant instead of reshaping the type and every
/// match on it. **Still nothing constructs one** — #92 gave retained element
/// instances a real, address-by-value key, [`InstanceKey`] (`instance.rs`),
/// but reconciliation decides reuse by comparing `diff_key`s at prepaint/paint
/// time, not by consulting an explicit invalidation request naming an
/// `InstanceId`. Wiring an `Instance`-scoped `InvalidationRequest` producer —
/// for a future caller that wants to force one specific instance dirty
/// directly, the way `Window::invalidate_layer` does for a whole layer — is
/// separable follow-up work, not something #92 needed. `LayerKey`, defined the
/// same way as this type once was, is now real — see [`crate::layer`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InstanceId(pub u64);

/// In what respect something stopped being valid.
///
/// The axes are independent, and that is the whole point of naming them: a
/// layer that only moved needs a new composite matrix and no CPU work at all,
/// while a label that changed its text needs repainting without re-laying-out
/// anything around it.
///
/// Axes are *derived by the framework* from what actually changed. They are
/// deliberately not declared at `cx.notify()` sites: correctness would then
/// depend on every call site classifying its own change correctly, and the
/// failure mode of getting it wrong is silently stale UI.
///
/// Hand-rolled rather than a `bitflags!` macro because the crate does not
/// depend on `bitflags`, and one `u8` newtype is not worth a dependency.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Invalidation(u8);

impl Invalidation {
    /// Sizes and positions must be recomputed.
    pub const LAYOUT: Self = Self(1 << 0);
    /// Painted output must be re-emitted.
    pub const DISPLAY: Self = Self(1 << 1);
    /// Hitboxes and dispatch nodes must be re-registered.
    pub const HIT: Self = Self(1 << 2);
    /// Only the composite transform changed, so nothing needs re-rendering.
    /// Nothing sets this until layers can be composited independently.
    pub const TRANSFORM: Self = Self(1 << 3);

    /// No axis at all.
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Every axis.
    pub const fn all() -> Self {
        Self(Self::LAYOUT.0 | Self::DISPLAY.0 | Self::HIT.0 | Self::TRANSFORM.0)
    }

    /// Whether no axis is set.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Whether every axis in `other` is set.
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Whether any axis in `other` is set.
    pub const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    /// The axes set in either operand. `BitOr` in a `const` context.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl std::ops::BitOr for Invalidation {
    type Output = Self;

    fn bitor(self, other: Self) -> Self {
        self.union(other)
    }
}

impl std::ops::BitOrAssign for Invalidation {
    fn bitor_assign(&mut self, other: Self) {
        self.0 |= other.0;
    }
}

/// What an [`InvalidationRequest`] applies to.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum InvalidationScope {
    /// One retained element instance, and nothing above or below it. Nothing
    /// produces this yet.
    Instance(InstanceId),
    /// One retained layer, and nothing above or below it.
    Layer(LayerKey),
    /// Every consumer whose recorded dependency set contains this entity.
    Entity(EntityId),
    /// The window as a whole: device loss, scale factor change, focus moving.
    Window,
}

/// One typed invalidation: what stopped being valid, and in what respect.
///
/// This is the single operation every part of the framework invalidates
/// through. It replaces three mechanisms with incompatible reach — a
/// window-wide `refreshing` boolean, an upward dispatch-tree walk, and a
/// forward dependency-set check — none of which could express the others.
///
/// The fields are framework-internal. The public surface is the deliberately
/// coarse shims [`Window::refresh`] and [`Window::refresh_buffers`], plus
/// `cx.notify()`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct InvalidationRequest {
    scope: InvalidationScope,
    axes: Invalidation,
}

impl InvalidationRequest {
    pub(crate) fn new(scope: InvalidationScope, axes: Invalidation) -> Self {
        Self { scope, axes }
    }

    /// What `cx.notify()` means: this entity's data changed, so anything
    /// rendering it has stale painted output and stale hit geometry.
    pub(crate) fn entity(entity_id: EntityId) -> Self {
        Self::new(
            InvalidationScope::Entity(entity_id),
            Invalidation::DISPLAY.union(Invalidation::HIT),
        )
    }
}

struct WindowInvalidatorInner {
    pub dirty: bool,
    pub draw_phase: DrawPhase,
    pub dirty_views: FxHashSet<EntityId>,
    pub update_count: usize,
    /// Entities notified while a draw was in progress.
    ///
    /// A notify that arrives mid-draw cannot be applied where it happens: the
    /// frame being built has already read whatever it was going to read, and
    /// pushing an `Effect::Notify` from inside prepaint/paint would run
    /// observer callbacks in the middle of an element tree walk. It used to be
    /// dropped instead, which lost two separate things — the dirty flag that
    /// schedules the frame able to answer it, and the effect that runs
    /// `cx.observe` callbacks. Recorded here and applied by
    /// [`WindowInvalidator::flush_deferred_invalidations`] once the draw is
    /// over and `&mut App` is available again.
    ///
    /// A set, so a view notifying the same entity repeatedly inside one draw
    /// costs one entry.
    deferred_notifies: FxHashSet<EntityId>,
    /// Layers named by an [`InvalidationScope::Layer`] request, and the axes
    /// each was named with, accumulated since a draw last took them.
    ///
    /// Held here rather than written straight into `Window::layers` because
    /// [`WindowInvalidator::invalidate`] takes `&self` and may be called
    /// mid-draw, when the window's retained state is being read. The draw that
    /// answers the request folds these into the layers at
    /// `Window::apply_invalidations`.
    dirty_layers: FxHashMap<LayerKey, Invalidation>,
    /// Window-scope axes accumulated since a draw last took them.
    window_axes: Invalidation,
    /// Window-scope axes requested while a draw was in progress.
    ///
    /// Same reasoning as `deferred_notifies`: the frame being built has already
    /// decided which views to rebuild, so a request arriving mid-draw belongs
    /// to the next one. It used to be dropped outright, which is why a
    /// smooth-scroll animation driven from prepaint never scheduled its own
    /// following frame.
    deferred_window_axes: Invalidation,
    /// Consecutive flushes that had something to apply. Only used to decide
    /// whether to emit the loop warning.
    consecutive_deferred_draws: u32,
    warned_about_deferred_loop: bool,
}

#[derive(Clone)]
pub(crate) struct WindowInvalidator {
    inner: Rc<RefCell<WindowInvalidatorInner>>,
}

static NOTIFY_ATTRIBUTION_ENABLED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    std::env::var("WGPUI_NOTIFY_ATTRIBUTION")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false)
});

struct NotifyAttributionState {
    counts: FxHashMap<EntityId, u64>,
    last_report: Instant,
}

static NOTIFY_ATTRIBUTION: std::sync::LazyLock<parking_lot::Mutex<NotifyAttributionState>> =
    std::sync::LazyLock::new(|| {
        parking_lot::Mutex::new(NotifyAttributionState {
            counts: FxHashMap::default(),
            last_report: Instant::now(),
        })
    });

/// Attribute entity invalidations to their source ids when
/// `WGPUI_NOTIFY_ATTRIBUTION` is set, so a notify storm can be traced to the
/// entities causing it instead of only being visible as frame cost. One warn
/// line per second names the top offenders; disabled costs a single atomic
/// load per invalidation.
fn record_notify_attribution(entity: EntityId) {
    if !*NOTIFY_ATTRIBUTION_ENABLED {
        return;
    }
    let mut state = NOTIFY_ATTRIBUTION.lock();
    *state.counts.entry(entity).or_insert(0) += 1;
    if state.last_report.elapsed() >= Duration::from_secs(1) {
        let mut top: Vec<(u64, u64)> = state
            .counts
            .iter()
            .map(|(id, count)| (id.as_u64(), *count))
            .collect();
        top.sort_unstable_by_key(|(_, count)| std::cmp::Reverse(*count));
        top.truncate(10);
        let total: u64 = state.counts.values().sum();
        log::warn!(
            target: "notify_attribution",
            "{} entity invalidations in the last second across {} entities; top: {}",
            total,
            state.counts.len(),
            top.iter()
                .map(|(id, count)| format!("entity {id}: {count}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        state.counts.clear();
        state.last_report = Instant::now();
    }
}

impl WindowInvalidator {
    pub fn new() -> Self {
        WindowInvalidator {
            inner: Rc::new(RefCell::new(WindowInvalidatorInner {
                dirty: true,
                draw_phase: DrawPhase::None,
                dirty_views: FxHashSet::default(),
                update_count: 0,
                deferred_notifies: FxHashSet::default(),
                dirty_layers: FxHashMap::default(),
                window_axes: Invalidation::empty(),
                deferred_window_axes: Invalidation::empty(),
                consecutive_deferred_draws: 0,
                warned_about_deferred_loop: false,
            })),
        }
    }

    /// Record `request`, applying it now or deferring it to the end of the
    /// current draw.
    ///
    /// Returns whether it was applied immediately. `false` means it was
    /// deferred, never that it was dropped: an invalidation is legal in every
    /// draw phase.
    pub fn invalidate(&self, request: InvalidationRequest, cx: &mut App) -> bool {
        match request.scope {
            // `request.axes` is not consulted. This phase implements `Entity`
            // scope with the forward dependency check in
            // `Window::accessed_entity_invalidated`, whose answer is a single
            // bool — a cached view either replays all of its recorded output or
            // rebuilds all of it, so there is no subset for an axis to select.
            // Axes start selecting something once a layer can be invalidated in
            // one respect and left alone in the others.
            InvalidationScope::Entity(entity_id) => self.invalidate_entity(entity_id, cx),
            InvalidationScope::Window => self.invalidate_window(request.axes),
            InvalidationScope::Layer(key) => self.invalidate_layer(key, request.axes),
            // Unreachable today: nothing constructs this scope, because nothing
            // retains the instances it names. Counted rather than asserted so
            // that the phase which produces the first one can see it arriving
            // before it has anywhere to apply it.
            InvalidationScope::Instance(_) => {
                crate::render_stats::count("invalidate: instance");
                true
            }
        }
    }

    /// Record that one retained layer stopped being valid in `axes`, and
    /// nothing above or below it.
    ///
    /// Unlike [`Self::invalidate_entity`] this never defers: a layer's axes are
    /// read at the start of the next draw, so recording them mid-draw is
    /// harmless and the request is answered by the frame after the one in
    /// progress — which is the same frame a deferred notify would reach.
    pub fn invalidate_layer(&self, key: LayerKey, axes: Invalidation) -> bool {
        crate::render_stats::count("invalidate: layer");
        let mut inner = self.inner.borrow_mut();
        inner.update_count += 1;
        *inner
            .dirty_layers
            .entry(key)
            .or_insert(Invalidation::empty()) |= axes;
        inner.dirty = true;
        true
    }

    /// Take the per-layer axes accumulated since the last call, for the draw
    /// that is about to answer them.
    pub fn take_layer_axes(&self) -> FxHashMap<LayerKey, Invalidation> {
        mem::take(&mut self.inner.borrow_mut().dirty_layers)
    }

    /// Record a window-scope invalidation.
    ///
    /// Split out of [`Self::invalidate`] because it is the one scope that needs
    /// no `&mut App`: nothing observes the window itself, so there is no
    /// `Effect::Notify` to push. [`Window::refresh`] and
    /// [`Window::refresh_buffers`] are `&mut Window` methods whose public
    /// signatures carry no context, and they may be called mid-draw, when `App`
    /// is leased out.
    pub fn invalidate_window(&self, axes: Invalidation) -> bool {
        crate::render_stats::count("invalidate: window");
        let mut inner = self.inner.borrow_mut();
        inner.update_count += 1;
        if inner.draw_phase == DrawPhase::None {
            inner.window_axes |= axes;
            inner.dirty = true;
            true
        } else {
            inner.deferred_window_axes |= axes;
            drop(inner);
            crate::render_stats::count("invalidate: deferred to end of draw");
            false
        }
    }

    /// Take the window-scope axes accumulated since the last call, for the draw
    /// that is about to answer them.
    pub fn take_window_axes(&self) -> Invalidation {
        mem::take(&mut self.inner.borrow_mut().window_axes)
    }

    fn invalidate_entity(&self, entity: EntityId, cx: &mut App) -> bool {
        crate::render_stats::count("invalidate: entity");
        record_notify_attribution(entity);
        let mut inner = self.inner.borrow_mut();
        inner.update_count += 1;
        inner.dirty_views.insert(entity);
        #[cfg(feature = "flamegraph")]
        crate::record_entity_invalidated();
        if inner.draw_phase == DrawPhase::None {
            inner.dirty = true;
            // Released before re-entering the app: `push_effect` does not touch
            // the invalidator today, but nothing about its signature promises
            // that, and an active borrow here has no purpose.
            drop(inner);
            cx.push_effect(Effect::Notify { emitter: entity });
            true
        } else {
            let first = inner.deferred_notifies.insert(entity);
            drop(inner);
            if first {
                crate::render_stats::count("invalidate: deferred to end of draw");
            } else {
                crate::render_stats::count("notify: deferred (duplicate, collapsed)");
            }
            false
        }
    }

    /// Apply everything [`Self::invalidate`] deferred while a draw was in
    /// progress: fold the deferred window-scope axes into the ones the next
    /// draw will take, mark the window dirty so a frame is scheduled to answer
    /// the invalidations already sitting in `dirty_views`, and push the
    /// `Effect::Notify`s so `cx.observe` callbacks finally run.
    ///
    /// Must be called after the draw phase has returned to
    /// [`DrawPhase::None`], with `&mut App` available. `Window::draw` is the
    /// only production caller.
    pub fn flush_deferred_invalidations(&self, cx: &mut App) {
        let (deferred, warn) = {
            let mut inner = self.inner.borrow_mut();
            debug_assert_eq!(
                inner.draw_phase,
                DrawPhase::None,
                "deferred invalidations must not be flushed while still drawing"
            );
            let window_axes = mem::take(&mut inner.deferred_window_axes);
            if inner.deferred_notifies.is_empty() && window_axes.is_empty() {
                inner.consecutive_deferred_draws = 0;
                return;
            }
            inner.window_axes |= window_axes;
            inner.dirty = true;
            inner.update_count += 1;
            inner.consecutive_deferred_draws = inner.consecutive_deferred_draws.saturating_add(1);
            let warn = inner.consecutive_deferred_draws >= DEFERRED_NOTIFY_LOOP_WARN_DRAWS
                && !mem::replace(&mut inner.warned_about_deferred_loop, true);
            (mem::take(&mut inner.deferred_notifies), warn)
        };

        crate::render_stats::count("notify: deferred flush scheduled a redraw");
        if warn {
            log::warn!(
                "window has deferred invalidations on {} consecutive draws \
                 (entities: {:?}); something is invalidating from inside its own \
                 render, which now redraws every frame",
                DEFERRED_NOTIFY_LOOP_WARN_DRAWS,
                deferred.iter().take(8).collect::<Vec<_>>(),
            );
        }

        // Borrow released: applying an effect can re-enter the invalidator.
        for entity in deferred {
            cx.push_effect(Effect::Notify { emitter: entity });
        }
    }

    pub fn is_dirty(&self) -> bool {
        self.inner.borrow().dirty
    }

    pub fn set_dirty(&self, dirty: bool) {
        let mut inner = self.inner.borrow_mut();
        inner.dirty = dirty;
        if dirty {
            inner.update_count += 1;
        }
    }

    pub fn set_phase(&self, phase: DrawPhase) {
        self.inner.borrow_mut().draw_phase = phase
    }

    pub fn draw_phase(&self) -> DrawPhase {
        self.inner.borrow().draw_phase
    }

    pub fn update_count(&self) -> usize {
        self.inner.borrow().update_count
    }

    pub fn take_views(&self) -> FxHashSet<EntityId> {
        mem::take(&mut self.inner.borrow_mut().dirty_views)
    }

    /// Return a set previously removed by [`Self::take_views`].
    ///
    /// Merges rather than assigns. The caller drains what it consumed and hands
    /// the (empty) set back, so in practice this adds nothing — but assigning
    /// would silently discard anything inserted by
    /// [`Self::invalidate`] in between, and "in between" is a window that
    /// only ever gets wider.
    pub fn replace_views(&self, views: FxHashSet<EntityId>) {
        if views.is_empty() {
            return;
        }
        self.inner.borrow_mut().dirty_views.extend(views);
    }

    #[track_caller]
    pub fn debug_assert_paint(&self) {
        debug_assert!(
            matches!(self.inner.borrow().draw_phase, DrawPhase::Paint),
            "this method can only be called during paint"
        );
    }

    #[track_caller]
    pub fn debug_assert_prepaint(&self) {
        debug_assert!(
            matches!(self.inner.borrow().draw_phase, DrawPhase::Prepaint),
            "this method can only be called during request_layout, or prepaint"
        );
    }

    #[track_caller]
    pub fn debug_assert_effects(&self) {
        debug_assert!(
            matches!(self.inner.borrow().draw_phase, DrawPhase::Effects),
            "this method can only be called during on_frame"
        );
    }

    /// [`DrawPhase::Effects`] is deliberately **not** accepted here.
    ///
    /// `on_frame` is meant to be cheap by construction: it receives resolved
    /// geometry and returns nothing, so it cannot build an element tree — and
    /// the assertion is what keeps that true rather than merely intended. The
    /// methods gated on this include `with_element_state`, which an effect must
    /// not touch: element state is migrated forward by whichever frame accessed
    /// it, and an effect replayed on behalf of a view that never ran would
    /// register accesses for a subtree that does not exist this frame.
    #[track_caller]
    pub fn debug_assert_paint_or_prepaint(&self) {
        debug_assert!(
            matches!(
                self.inner.borrow().draw_phase,
                DrawPhase::Paint | DrawPhase::Prepaint
            ),
            "this method can only be called during request_layout, prepaint, or paint"
        );
    }
}

type AnyObserver = Box<dyn FnMut(&mut Window, &mut App) -> bool + 'static>;

pub(crate) type AnyWindowFocusListener =
    Box<dyn FnMut(&WindowFocusEvent, &mut Window, &mut App) -> bool + 'static>;

pub(crate) struct WindowFocusEvent {
    pub(crate) previous_focus_path: SmallVec<[FocusId; 8]>,
    pub(crate) current_focus_path: SmallVec<[FocusId; 8]>,
}

impl WindowFocusEvent {
    pub fn is_focus_in(&self, focus_id: FocusId) -> bool {
        !self.previous_focus_path.contains(&focus_id) && self.current_focus_path.contains(&focus_id)
    }

    pub fn is_focus_out(&self, focus_id: FocusId) -> bool {
        self.previous_focus_path.contains(&focus_id) && !self.current_focus_path.contains(&focus_id)
    }
}

/// This is provided when subscribing for `Context::on_focus_out` events.
pub struct FocusOutEvent {
    /// A weak focus handle representing what was blurred.
    pub blurred: WeakFocusHandle,
}

slotmap::new_key_type! {
    /// A globally unique identifier for a focusable element.
    pub struct FocusId;
}

thread_local! {
    /// Fallback arena used when no app-specific arena is active.
    /// In production, each window draw sets CURRENT_ELEMENT_ARENA to the app's arena.
    pub(crate) static ELEMENT_ARENA: RefCell<Arena> = RefCell::new(Arena::new(1024 * 1024));

    /// Points to the current App's element arena during draw operations.
    /// This allows multiple test Apps to have isolated arenas, preventing
    /// cross-session corruption when the scheduler interleaves their tasks.
    static CURRENT_ELEMENT_ARENA: Cell<Option<*const RefCell<Arena>>> = const { Cell::new(None) };
}

/// Allocates an element in the current arena. Uses the app-specific arena if one
/// is active (during draw), otherwise falls back to the thread-local ELEMENT_ARENA.
pub(crate) fn with_element_arena<R>(f: impl FnOnce(&mut Arena) -> R) -> R {
    CURRENT_ELEMENT_ARENA.with(|current| {
        if let Some(arena_ptr) = current.get() {
            // SAFETY: The pointer is valid for the duration of the draw operation
            // that set it, and we're being called during that same draw.
            let arena_cell = unsafe { &*arena_ptr };
            f(&mut arena_cell.borrow_mut())
        } else {
            ELEMENT_ARENA.with_borrow_mut(f)
        }
    })
}

/// RAII guard that sets CURRENT_ELEMENT_ARENA for the duration of a draw operation.
/// When dropped, restores the previous arena (supporting nested draws).
///
/// `pub` (not `pub(crate)`) so that DLL-loaded plugin code — which links its
/// own separate copy of this crate, and therefore has its own separate copy
/// of the `CURRENT_ELEMENT_ARENA`/`ELEMENT_ARENA` thread-locals — can enter
/// the *host's* already-correct scope using the `App` reference it's handed
/// across the FFI boundary: `ElementArenaScope::enter(cx.element_arena())`.
/// See `App::element_arena`'s doc comment for the full leak this closes.
pub struct ElementArenaScope {
    previous: Option<*const RefCell<Arena>>,
}

impl ElementArenaScope {
    /// Enter a scope where element allocations use the given arena.
    pub fn enter(arena: &RefCell<Arena>) -> Self {
        let previous = CURRENT_ELEMENT_ARENA.with(|current| {
            let prev = current.get();
            current.set(Some(arena as *const RefCell<Arena>));
            prev
        });
        Self { previous }
    }
}

impl Drop for ElementArenaScope {
    fn drop(&mut self) {
        CURRENT_ELEMENT_ARENA.with(|current| {
            current.set(self.previous);
        });
    }
}

/// Returned when the element arena has been used and so must be cleared before the next draw.
#[must_use]
pub struct ArenaClearNeeded<'app> {
    arena: &'app RefCell<Arena>,
}

impl<'app> ArenaClearNeeded<'app> {
    /// Create a new ArenaClearNeeded that will clear the given arena.
    pub(crate) fn new(arena: &'app RefCell<Arena>) -> Self {
        Self { arena }
    }

    /// Clear the element arena.
    pub fn clear(self) {
        // SAFETY: The arena reference must be valid and cleared before the next draw.
        self.arena.borrow_mut().clear();
    }
}

pub(crate) type FocusMap = RwLock<SlotMap<FocusId, FocusRef>>;
pub(crate) struct FocusRef {
    pub(crate) ref_count: AtomicUsize,
    pub(crate) tab_index: isize,
    pub(crate) tab_stop: bool,
}

impl FocusId {
    /// Obtains whether the element associated with this handle is currently focused.
    pub fn is_focused(&self, window: &Window) -> bool {
        window.focus == Some(*self)
    }

    /// Obtains whether the element associated with this handle contains the focused
    /// element or is itself focused.
    pub fn contains_focused(&self, window: &Window, cx: &App) -> bool {
        window
            .focused(cx)
            .is_some_and(|focused| self.contains(focused.id, window))
    }

    /// Obtains whether the element associated with this handle is contained within the
    /// focused element or is itself focused.
    pub fn within_focused(&self, window: &Window, cx: &App) -> bool {
        let focused = window.focused(cx);
        focused.is_some_and(|focused| focused.id.contains(*self, window))
    }

    /// Obtains whether this handle contains the given handle in the most recently rendered frame.
    pub(crate) fn contains(&self, other: Self, window: &Window) -> bool {
        window
            .rendered_frame
            .dispatch_tree
            .focus_contains(*self, other)
    }
}

/// A handle which can be used to track and manipulate the focused element in a window.
pub struct FocusHandle {
    pub(crate) id: FocusId,
    handles: Arc<FocusMap>,
    /// The index of this element in the tab order.
    pub tab_index: isize,
    /// Whether this element can be focused by tab navigation.
    pub tab_stop: bool,
}

impl std::fmt::Debug for FocusHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("FocusHandle({:?})", self.id))
    }
}

impl FocusHandle {
    pub(crate) fn new(handles: &Arc<FocusMap>) -> Self {
        let id = handles.write().insert(FocusRef {
            ref_count: AtomicUsize::new(1),
            tab_index: 0,
            tab_stop: false,
        });

        Self {
            id,
            tab_index: 0,
            tab_stop: false,
            handles: handles.clone(),
        }
    }

    pub(crate) fn for_id(id: FocusId, handles: &Arc<FocusMap>) -> Option<Self> {
        let lock = handles.read();
        let focus = lock.get(id)?;
        if atomic_incr_if_not_zero(&focus.ref_count) == 0 {
            return None;
        }
        Some(Self {
            id,
            tab_index: focus.tab_index,
            tab_stop: focus.tab_stop,
            handles: handles.clone(),
        })
    }

    /// Sets the tab index of the element associated with this handle.
    pub fn tab_index(mut self, index: isize) -> Self {
        self.tab_index = index;
        if let Some(focus) = self.handles.write().get_mut(self.id) {
            focus.tab_index = index;
        }
        self
    }

    /// Sets whether the element associated with this handle is a tab stop.
    ///
    /// When `false`, the element will not be included in the tab order.
    pub fn tab_stop(mut self, tab_stop: bool) -> Self {
        self.tab_stop = tab_stop;
        if let Some(focus) = self.handles.write().get_mut(self.id) {
            focus.tab_stop = tab_stop;
        }
        self
    }

    /// Converts this focus handle into a weak variant, which does not prevent it from being released.
    pub fn downgrade(&self) -> WeakFocusHandle {
        WeakFocusHandle {
            id: self.id,
            handles: Arc::downgrade(&self.handles),
        }
    }

    /// Moves the focus to the element associated with this handle.
    pub fn focus(&self, window: &mut Window, cx: &mut App) {
        window.focus(self, cx)
    }

    /// Obtains whether the element associated with this handle is currently focused.
    pub fn is_focused(&self, window: &Window) -> bool {
        self.id.is_focused(window)
    }

    /// Obtains whether the element associated with this handle contains the focused
    /// element or is itself focused.
    pub fn contains_focused(&self, window: &Window, cx: &App) -> bool {
        self.id.contains_focused(window, cx)
    }

    /// Obtains whether the element associated with this handle is contained within the
    /// focused element or is itself focused.
    pub fn within_focused(&self, window: &Window, cx: &mut App) -> bool {
        self.id.within_focused(window, cx)
    }

    /// Obtains whether this handle contains the given handle in the most recently rendered frame.
    pub fn contains(&self, other: &Self, window: &Window) -> bool {
        self.id.contains(other.id, window)
    }

    /// Dispatch an action on the element that rendered this focus handle
    pub fn dispatch_action(&self, action: &dyn Action, window: &mut Window, cx: &mut App) {
        if let Some(node_id) = window
            .rendered_frame
            .dispatch_tree
            .focusable_node_id(self.id)
        {
            window.dispatch_action_on_node(node_id, action, cx)
        }
    }
}

impl Clone for FocusHandle {
    fn clone(&self) -> Self {
        Self::for_id(self.id, &self.handles).unwrap()
    }
}

impl PartialEq for FocusHandle {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for FocusHandle {}

impl Drop for FocusHandle {
    fn drop(&mut self) {
        self.handles
            .read()
            .get(self.id)
            .unwrap()
            .ref_count
            .fetch_sub(1, SeqCst);
    }
}

/// A weak reference to a focus handle.
#[derive(Clone, Debug)]
pub struct WeakFocusHandle {
    pub(crate) id: FocusId,
    pub(crate) handles: Weak<FocusMap>,
}

impl WeakFocusHandle {
    /// Attempts to upgrade the [WeakFocusHandle] to a [FocusHandle].
    pub fn upgrade(&self) -> Option<FocusHandle> {
        let handles = self.handles.upgrade()?;
        FocusHandle::for_id(self.id, &handles)
    }
}

impl PartialEq for WeakFocusHandle {
    fn eq(&self, other: &WeakFocusHandle) -> bool {
        self.id == other.id
    }
}

impl Eq for WeakFocusHandle {}

impl PartialEq<FocusHandle> for WeakFocusHandle {
    fn eq(&self, other: &FocusHandle) -> bool {
        self.id == other.id
    }
}

impl PartialEq<WeakFocusHandle> for FocusHandle {
    fn eq(&self, other: &WeakFocusHandle) -> bool {
        self.id == other.id
    }
}

/// Focusable allows users of your view to easily
/// focus it (using window.focus_view(cx, view))
pub trait Focusable: 'static {
    /// Returns the focus handle associated with this view.
    fn focus_handle(&self, cx: &App) -> FocusHandle;
}

impl<V: Focusable> Focusable for Entity<V> {
    fn focus_handle(&self, cx: &App) -> FocusHandle {
        self.read(cx).focus_handle(cx)
    }
}

/// ManagedView is a view (like a Modal, Popover, Menu, etc.)
/// where the lifecycle of the view is handled by another view.
pub trait ManagedView: Focusable + EventEmitter<DismissEvent> + Render {}

impl<M: Focusable + EventEmitter<DismissEvent> + Render> ManagedView for M {}

/// Emitted by implementers of [`ManagedView`] to indicate the view should be dismissed, such as when a view is presented as a modal.
pub struct DismissEvent;

type FrameCallback = Box<dyn FnOnce(&mut Window, &mut App)>;

pub(crate) type AnyMouseListener =
    Box<dyn FnMut(&dyn Any, DispatchPhase, &mut Window, &mut App) + 'static>;

/// A cross-DLL-safe event type discriminator computed from `std::any::type_name::<T>()`.
/// The type name string is the same across compilation units, so the hash is consistent
/// between the main binary and plugin DLLs.
pub(crate) fn type_name_hash<T: 'static>() -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    std::any::type_name::<T>().hash(&mut hasher);
    hasher.finish()
}

/// Cross-DLL discriminator for actions, computed from the action name (e.g. "editor::SaveCurrentFile").
/// Action names are defined in source and are consistent across compilation units.
pub(crate) fn action_name_hash<T: crate::Action>() -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    T::name_for_type().hash(&mut hasher);
    hasher.finish()
}

/// A mouse listener entry with a type discriminator for cross-DLL event dispatch.
pub(crate) struct MouseListenerEntry {
    pub(crate) discriminator: u64,
    pub(crate) listener: AnyMouseListener,
}

#[derive(Clone)]
pub(crate) struct CursorStyleRequest {
    pub(crate) hitbox_id: Option<HitboxId>,
    pub(crate) style: CursorStyle,
}

#[derive(Default, Eq, PartialEq)]
pub(crate) struct HitTest {
    pub(crate) ids: SmallVec<[HitboxId; 8]>,
    pub(crate) hover_hitbox_count: usize,
}

/// A type of window control area that corresponds to the platform window.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WindowControlArea {
    /// An area that allows dragging of the platform window.
    Drag,
    /// An area that allows closing of the platform window.
    Close,
    /// An area that allows maximizing of the platform window.
    Max,
    /// An area that allows minimizing of the platform window.
    Min,
}

/// An identifier for a [Hitbox] which also includes [HitboxBehavior].
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct HitboxId(u64);

impl HitboxId {
    /// Checks if the hitbox with this ID is currently hovered. Except when handling
    /// `ScrollWheelEvent`, this is typically what you want when determining whether to handle mouse
    /// events or paint hover styles.
    ///
    /// See [`Hitbox::is_hovered`] for details.
    pub fn is_hovered(self, window: &Window) -> bool {
        let hit_test = &window.mouse_hit_test;
        for id in hit_test.ids.iter().take(hit_test.hover_hitbox_count) {
            if self == *id {
                return true;
            }
        }
        false
    }

    /// Checks if the hitbox with this ID contains the mouse and should handle scroll events.
    /// Typically this should only be used when handling `ScrollWheelEvent`, and otherwise
    /// `is_hovered` should be used. See the documentation of `Hitbox::is_hovered` for details about
    /// this distinction.
    pub fn should_handle_scroll(self, window: &Window) -> bool {
        window.mouse_hit_test.ids.contains(&self)
    }

    fn next(mut self) -> HitboxId {
        HitboxId(self.0.wrapping_add(1))
    }
}

/// A rectangular region that potentially blocks hitboxes inserted prior.
/// See [Window::insert_hitbox] for more details.
#[derive(Clone, Debug, Deref)]
pub struct Hitbox {
    /// A unique identifier for the hitbox.
    pub id: HitboxId,
    /// The bounds of the hitbox.
    #[deref]
    pub bounds: Bounds<Pixels>,
    /// The content mask when the hitbox was inserted.
    pub content_mask: ContentMask<Pixels>,
    /// Flags that specify hitbox behavior.
    pub behavior: HitboxBehavior,
    /// The retained layer whose local coordinate space contains these bounds.
    /// `None` identifies legacy window-relative hitboxes.
    pub(crate) layer: Option<LayerKey>,
}

impl Hitbox {
    /// Checks if the hitbox is currently hovered. Except when handling `ScrollWheelEvent`, this is
    /// typically what you want when determining whether to handle mouse events or paint hover
    /// styles.
    ///
    /// This can return `false` even when the hitbox contains the mouse, if a hitbox in front of
    /// this sets `HitboxBehavior::BlockMouse` (`InteractiveElement::occlude`) or
    /// `HitboxBehavior::BlockMouseExceptScroll` (`InteractiveElement::block_mouse_except_scroll`).
    ///
    /// Handling of `ScrollWheelEvent` should typically use `should_handle_scroll` instead.
    /// Concretely, this is due to use-cases like overlays that cause the elements under to be
    /// non-interactive while still allowing scrolling. More abstractly, this is because
    /// `is_hovered` is about element interactions directly under the mouse - mouse moves, clicks,
    /// hover styling, etc. In contrast, scrolling is about finding the current outer scrollable
    /// container.
    pub fn is_hovered(&self, window: &Window) -> bool {
        self.id.is_hovered(window)
    }

    /// Checks if the hitbox contains the mouse and should handle scroll events. Typically this
    /// should only be used when handling `ScrollWheelEvent`, and otherwise `is_hovered` should be
    /// used. See the documentation of `Hitbox::is_hovered` for details about this distinction.
    ///
    /// This can return `false` even when the hitbox contains the mouse, if a hitbox in front of
    /// this sets `HitboxBehavior::BlockMouse` (`InteractiveElement::occlude`).
    pub fn should_handle_scroll(&self, window: &Window) -> bool {
        self.id.should_handle_scroll(window)
    }
}

/// How the hitbox affects mouse behavior.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum HitboxBehavior {
    /// Normal hitbox mouse behavior, doesn't affect mouse handling for other hitboxes.
    #[default]
    Normal,

    /// All hitboxes behind this hitbox will be ignored and so will have `hitbox.is_hovered() ==
    /// false` and `hitbox.should_handle_scroll() == false`. Typically for elements this causes
    /// skipping of all mouse events, hover styles, and tooltips. This flag is set by
    /// [`InteractiveElement::occlude`].
    ///
    /// For mouse handlers that check those hitboxes, this behaves the same as registering a
    /// bubble-phase handler for every mouse event type:
    ///
    /// ```ignore
    /// window.on_mouse_event(move |_: &EveryMouseEventTypeHere, phase, window, cx| {
    ///     if phase == DispatchPhase::Capture && hitbox.is_hovered(window) {
    ///         cx.stop_propagation();
    ///     }
    /// })
    /// ```
    ///
    /// This has effects beyond event handling - any use of hitbox checking, such as hover
    /// styles and tooltips. These other behaviors are the main point of this mechanism. An
    /// alternative might be to not affect mouse event handling - but this would allow
    /// inconsistent UI where clicks and moves interact with elements that are not considered to
    /// be hovered.
    BlockMouse,

    /// All hitboxes behind this hitbox will have `hitbox.is_hovered() == false`, even when
    /// `hitbox.should_handle_scroll() == true`. Typically for elements this causes all mouse
    /// interaction except scroll events to be ignored - see the documentation of
    /// [`Hitbox::is_hovered`] for details. This flag is set by
    /// [`InteractiveElement::block_mouse_except_scroll`].
    ///
    /// For mouse handlers that check those hitboxes, this behaves the same as registering a
    /// bubble-phase handler for every mouse event type **except** `ScrollWheelEvent`:
    ///
    /// ```ignore
    /// window.on_mouse_event(move |_: &EveryMouseEventTypeExceptScroll, phase, window, cx| {
    ///     if phase == DispatchPhase::Bubble && hitbox.should_handle_scroll(window) {
    ///         cx.stop_propagation();
    ///     }
    /// })
    /// ```
    ///
    /// See the documentation of [`Hitbox::is_hovered`] for details of why `ScrollWheelEvent` is
    /// handled differently than other mouse events. If also blocking these scroll events is
    /// desired, then a `cx.stop_propagation()` handler like the one above can be used.
    ///
    /// This has effects beyond event handling - this affects any use of `is_hovered`, such as
    /// hover styles and tooltips. These other behaviors are the main point of this mechanism.
    /// An alternative might be to not affect mouse event handling - but this would allow
    /// inconsistent UI where clicks and moves interact with elements that are not considered to
    /// be hovered.
    BlockMouseExceptScroll,
}

/// An identifier for a tooltip.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct TooltipId(usize);

impl TooltipId {
    /// Checks if the tooltip is currently hovered.
    pub fn is_hovered(&self, window: &Window) -> bool {
        window
            .tooltip_bounds
            .as_ref()
            .is_some_and(|tooltip_bounds| {
                tooltip_bounds.id == *self
                    && tooltip_bounds.bounds.contains(&window.mouse_position())
            })
    }
}

pub(crate) struct TooltipBounds {
    id: TooltipId,
    bounds: Bounds<Pixels>,
}

#[derive(Clone)]
pub(crate) struct TooltipRequest {
    id: TooltipId,
    tooltip: AnyTooltip,
}

pub(crate) struct DeferredDraw {
    current_view: EntityId,
    priority: usize,
    parent_node: DispatchNodeId,
    element_id_stack: SmallVec<[ElementId; 32]>,
    text_style_stack: Vec<TextStyleRefinement>,
    element: Option<AnyElement>,
    absolute_offset: Point<Pixels>,
    prepaint_range: Range<PrepaintStateIndex>,
    paint_range: Range<PaintIndex>,
}

/// A callback registered by [`Element::on_frame`](crate::Element::on_frame).
///
/// `Fn` and `Rc` rather than owned `FnMut` because the callback has to outlive
/// the element that registered it: an element inside a cached view is never
/// built on a frame the view replays, and the whole point of the channel is
/// that its effect still happens.
pub(crate) type FrameEffectCallback = Rc<dyn Fn(ElementGeometry, &mut Window, &mut App)>;

/// One recorded `on_frame` effect: what to run, and the geometry it resolved
/// against.
///
/// Recorded like a hitbox, and replayed by the same kind of index range. The
/// geometry is the one from the frame the element last actually ran on, which
/// is exactly right on a cache hit — a cached view only replays when its bounds
/// and content mask are unchanged, so re-deriving the geometry could not
/// produce a different answer, and deriving it would mean re-rendering the view
/// to get an element tree to derive it from.
#[derive(Clone)]
pub(crate) struct FrameEffect {
    pub(crate) callback: FrameEffectCallback,
    pub(crate) geometry: ElementGeometry,
}

pub(crate) struct Frame {
    pub(crate) focus: Option<FocusId>,
    pub(crate) window_active: bool,
    pub(crate) element_states: FxHashMap<(GlobalElementId, TypeId), ElementStateBox>,
    accessed_element_states: Vec<(GlobalElementId, TypeId)>,
    pub(crate) mouse_listeners: Vec<Option<MouseListenerEntry>>,
    pub(crate) dispatch_tree: DispatchTree,
    pub(crate) scene: Scene,
    pub(crate) hitboxes: Vec<Hitbox>,
    /// `on_frame` effects registered this frame, in the order they ran.
    pub(crate) effects: Vec<FrameEffect>,
    pub(crate) window_control_hitboxes: Vec<(WindowControlArea, Hitbox)>,
    pub(crate) deferred_draws: Vec<DeferredDraw>,
    pub(crate) input_handlers: Vec<Option<PlatformInputHandler>>,
    pub(crate) tooltip_requests: Vec<Option<TooltipRequest>>,
    pub(crate) cursor_styles: Vec<CursorStyleRequest>,
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) debug_bounds: FxHashMap<String, Bounds<Pixels>>,
    // Broadened beyond the plain inspector/debug_assertions gate below: this
    // field backs `Window::build_inspector_element_id`, which flamegraph-only
    // (release) builds also need, to attribute CPU spans in `element.rs`'s
    // `Drawable::request_layout`/`prepaint`/`paint` to a source file:line.
    #[cfg(any(feature = "inspector", debug_assertions, feature = "flamegraph"))]
    pub(crate) next_inspector_instance_ids: FxHashMap<Rc<crate::InspectorElementPath>, usize>,
    #[cfg(any(feature = "inspector", debug_assertions))]
    pub(crate) inspector_hitboxes: FxHashMap<HitboxId, crate::InspectorElementId>,
    #[cfg(any(feature = "inspector", debug_assertions))]
    pub(crate) inspector_element_infos: Vec<crate::InspectorElementInfo>,
    #[cfg(any(feature = "inspector", debug_assertions))]
    pub(crate) inspector_event_listeners:
        FxHashMap<crate::GlobalElementId, Vec<crate::InspectorEventListener>>,
    #[cfg(any(feature = "inspector", debug_assertions))]
    pub(crate) inspector_element_states: FxHashMap<crate::GlobalElementId, Vec<SharedString>>,
    pub(crate) tab_stops: TabStopMap,
}

#[derive(Clone, Default)]
pub(crate) struct PrepaintStateIndex {
    hitboxes_index: usize,
    effects_index: usize,
    tooltips_index: usize,
    deferred_draws_index: usize,
    dispatch_tree_index: usize,
    accessed_element_states_index: usize,
    line_layout_index: LineLayoutIndex,
}

#[derive(Clone, Default)]
pub(crate) struct PaintIndex {
    scene_index: usize,
    mouse_listeners_index: usize,
    input_handlers_index: usize,
    cursor_styles_index: usize,
    accessed_element_states_index: usize,
    tab_handle_index: usize,
    line_layout_index: LineLayoutIndex,
}

impl Frame {
    pub(crate) fn new(dispatch_tree: DispatchTree) -> Self {
        Frame {
            focus: None,
            window_active: false,
            element_states: FxHashMap::default(),
            accessed_element_states: Vec::new(),
            mouse_listeners: Vec::new(),
            dispatch_tree,
            scene: Scene::default(),
            hitboxes: Vec::new(),
            effects: Vec::new(),
            window_control_hitboxes: Vec::new(),
            deferred_draws: Vec::new(),
            input_handlers: Vec::new(),
            tooltip_requests: Vec::new(),
            cursor_styles: Vec::new(),

            #[cfg(any(test, feature = "test-support"))]
            debug_bounds: FxHashMap::default(),

            #[cfg(any(feature = "inspector", debug_assertions, feature = "flamegraph"))]
            next_inspector_instance_ids: FxHashMap::default(),

            #[cfg(any(feature = "inspector", debug_assertions))]
            inspector_hitboxes: FxHashMap::default(),

            #[cfg(any(feature = "inspector", debug_assertions))]
            inspector_element_infos: Vec::new(),

            #[cfg(any(feature = "inspector", debug_assertions))]
            inspector_event_listeners: FxHashMap::default(),

            #[cfg(any(feature = "inspector", debug_assertions))]
            inspector_element_states: FxHashMap::default(),

            tab_stops: TabStopMap::default(),
        }
    }

    pub(crate) fn clear(&mut self) {
        self.element_states.clear();
        self.accessed_element_states.clear();
        self.mouse_listeners.clear();
        self.dispatch_tree.clear();
        self.scene.clear();
        self.input_handlers.clear();
        self.tooltip_requests.clear();
        self.cursor_styles.clear();
        self.hitboxes.clear();
        self.effects.clear();
        self.window_control_hitboxes.clear();
        self.deferred_draws.clear();
        self.tab_stops.clear();
        self.focus = None;

        #[cfg(any(feature = "inspector", debug_assertions, feature = "flamegraph"))]
        self.next_inspector_instance_ids.clear();

        #[cfg(any(feature = "inspector", debug_assertions))]
        {
            self.inspector_hitboxes.clear();
            self.inspector_element_infos.clear();
            self.inspector_event_listeners.clear();
            self.inspector_element_states.clear();
        }
    }

    pub(crate) fn cursor_style(&self, window: &Window) -> Option<CursorStyle> {
        self.cursor_styles
            .iter()
            .rev()
            .fold_while(None, |style, request| match request.hitbox_id {
                None => Done(Some(request.style)),
                Some(hitbox_id) => Continue(
                    style.or_else(|| hitbox_id.is_hovered(window).then_some(request.style)),
                ),
            })
            .into_inner()
    }

    pub(crate) fn hit_test(
        &self,
        position: Point<Pixels>,
        layers: &FxHashMap<LayerKey, Layer>,
    ) -> HitTest {
        let mut set_hover_hitbox_count = false;
        let mut hit_test = HitTest::default();
        for hitbox in self.hitboxes.iter().rev() {
            let position = hitbox
                .layer
                .and_then(|key| layers.get(&key))
                .map_or(position, |layer| layer.transform.invert(position));
            let bounds = hitbox.bounds.intersect(&hitbox.content_mask.bounds);
            if bounds.contains(&position) {
                hit_test.ids.push(hitbox.id);
                if !set_hover_hitbox_count
                    && hitbox.behavior == HitboxBehavior::BlockMouseExceptScroll
                {
                    hit_test.hover_hitbox_count = hit_test.ids.len();
                    set_hover_hitbox_count = true;
                }
                if hitbox.behavior == HitboxBehavior::BlockMouse {
                    break;
                }
            }
        }
        if !set_hover_hitbox_count {
            hit_test.hover_hitbox_count = hit_test.ids.len();
        }
        hit_test
    }

    pub(crate) fn focus_path(&self) -> SmallVec<[FocusId; 8]> {
        self.focus
            .map(|focus_id| self.dispatch_tree.focus_path(focus_id))
            .unwrap_or_default()
    }

    pub(crate) fn finish(&mut self, prev_frame: &mut Self) {
        for element_state_key in &self.accessed_element_states {
            if let Some((element_state_key, element_state)) =
                prev_frame.element_states.remove_entry(element_state_key)
            {
                self.element_states.insert(element_state_key, element_state);
            }
        }

        self.scene.finish();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
enum InputModality {
    Mouse,
    Keyboard,
}

/// Holds the state for a specific window.
pub struct Window {
    pub(crate) handle: AnyWindowHandle,
    pub(crate) invalidator: WindowInvalidator,
    pub(crate) removed: bool,
    pub(crate) platform_window: Box<dyn PlatformWindow>,
    display_id: Option<DisplayId>,
    sprite_atlas: Arc<dyn PlatformAtlas>,
    text_system: Arc<WindowTextSystem>,
    rem_size: Pixels,
    /// The stack of override values for the window's rem size.
    ///
    /// This is used by `with_rem_size` to allow rendering an element tree with
    /// a given rem size.
    rem_size_override_stack: SmallVec<[Pixels; 8]>,
    pub(crate) viewport_size: Size<Pixels>,
    layout_engine: Option<TaffyLayoutEngine>,
    pub(crate) root: Option<AnyView>,
    pub(crate) element_id_stack: SmallVec<[ElementId; 32]>,
    pub(crate) text_style_stack: Vec<TextStyleRefinement>,
    pub(crate) rendered_entity_stack: Vec<EntityId>,
    pub(crate) element_offset_stack: Vec<Point<Pixels>>,
    pub(crate) retained_layer_stack: Vec<LayerKey>,
    pub(crate) hitbox_layer_stack: Vec<(LayerKey, Point<Pixels>, Bounds<Pixels>)>,
    /// Parallel to `element_id_stack`, but pushed only around a `.layer()`
    /// subtree's children (#92) and never read by anything predating this
    /// phase. Each entry is either a child's own `ElementId` or a synthetic
    /// `ElementId::InstanceSlot` for a child that has none, which is what lets
    /// `InstanceKey` address elements — bare `div()`, all of `Text` — that
    /// never call `.id(...)`. Kept separate from `element_id_stack` on purpose:
    /// folding positional segments into that stack would shift every existing
    /// `GlobalElementId` (and therefore every `LayerKey` and `with_element_state`
    /// key) derived from it, for no benefit to systems that don't need instance
    /// identity.
    pub(crate) instance_id_stack: SmallVec<[ElementId; 32]>,
    /// The `request_layout`-time counterpart of `hitbox_layer_stack` (#93):
    /// which `.layer()` ancestor, if any, is currently being visited while
    /// `instance_id_stack` reflects the element whose taffy node is about to
    /// be requested. Pushed by a `.layer()` div's own `request_layout`,
    /// mirroring exactly how `with_layer_hitbox_scope` pushes
    /// `hitbox_layer_stack` during `prepaint`. Kept as its own stack rather
    /// than reusing `hitbox_layer_stack` because the two are live during
    /// different, non-overlapping draw phases (`request_layout` completes,
    /// for the whole tree, before any element's `prepaint` begins) and
    /// conflating them would make either one's invariants harder to state.
    pub(crate) layout_layer_stack: Vec<LayerKey>,
    pub(crate) element_opacity: f32,
    pub(crate) content_mask_stack: Vec<ContentMask<Pixels>>,
    pub(crate) requested_autoscroll: Option<Bounds<Pixels>>,
    pub(crate) image_cache_stack: Vec<AnyImageCache>,
    pub(crate) rendered_frame: Frame,
    pub(crate) next_frame: Frame,
    /// Retained layers, addressed by a stable key rather than by an offset into
    /// a frame array.
    ///
    /// Deliberately not part of [`Frame`]: a layer's whole purpose is to
    /// outlive the frame that built it, and the frames are swapped and cleared
    /// every draw.
    pub(crate) layers: FxHashMap<LayerKey, Layer>,
    /// Monotonic draw counter, for layer eviction. Not a frame *index* — it
    /// counts draws this window performed, so a window that stops drawing stops
    /// ageing its layers.
    pub(crate) layer_frame: u64,
    /// Content tokens handed to the renderer's slab registry: bumped every
    /// time a layer's items are replaced, read at composite time. A token the
    /// registry has not seen means "upload this layer's slab".
    slab_tokens: FxHashMap<LayerKey, u64>,
    next_slab_token: u64,
    next_layer_id: u32,
    next_hitbox_id: HitboxId,
    pub(crate) next_tooltip_id: TooltipId,
    pub(crate) tooltip_bounds: Option<TooltipBounds>,
    next_frame_callbacks: Rc<RefCell<Vec<FrameCallback>>>,
    pub(crate) dirty_views: FxHashSet<EntityId>,
    /// The entities notified since the last frame, as reported, before the
    /// dispatch-tree ancestor walk turns them into `dirty_views`.
    ///
    /// `dirty_views` throws away everything that is not a view: the walk starts
    /// from `view_node_ids` and a model is not in it. Keeping the raw set is
    /// what lets a cached view ask whether anything *it reads* changed, rather
    /// than only whether anything it *contains* changed. See
    /// [`Self::accessed_entity_invalidated`].
    ///
    /// Lives for the duration of one draw, and is cleared alongside
    /// `dirty_views`.
    pub(crate) invalidated_entities: FxHashSet<EntityId>,
    focus_listeners: SubscriberSet<(), AnyWindowFocusListener>,
    pub(crate) focus_lost_listeners: SubscriberSet<(), AnyObserver>,
    default_prevented: bool,
    mouse_position: Point<Pixels>,
    mouse_hit_test: HitTest,
    modifiers: Modifiers,
    capslock: Capslock,
    mouse_button_pressed: Option<MouseButton>,
    scale_factor: f32,
    pub(crate) bounds_observers: SubscriberSet<(), AnyObserver>,
    appearance: WindowAppearance,
    pub(crate) appearance_observers: SubscriberSet<(), AnyObserver>,
    active: Rc<Cell<bool>>,
    hovered: Rc<Cell<bool>>,
    pub(crate) needs_present: Rc<Cell<bool>>,
    pub(crate) last_input_timestamp: Rc<Cell<Instant>>,
    pub(crate) resizing_window: Rc<Cell<bool>>,
    last_input_modality: InputModality,
    /// The window-scope axes this draw is answering, taken from the invalidator
    /// at the top of [`Self::draw`] and cleared at the bottom.
    ///
    /// This replaces the `refreshing` boolean, which conflated "a redraw was
    /// requested" with "no view may reuse its cached output". Those are
    /// different requests, and `refresh_buffers` needs exactly the first
    /// without the second.
    pub(crate) window_invalidation: Invalidation,
    /// Set while a cached view rebuilds, forcing cached views nested inside it
    /// to rebuild too.
    ///
    /// Not invalidation: it says where the element walk currently is, not what
    /// changed. It shared the `refreshing` field with window-scope invalidation
    /// only because both happened to mean "no cache reuse right now". See
    /// `nested_view_cache_enabled` for the opt-in that lifts it.
    pub(crate) nested_view_cache_suppressed: bool,
    pub(crate) activation_observers: SubscriberSet<(), AnyObserver>,
    pub(crate) focus: Option<FocusId>,
    focus_enabled: bool,
    pending_input: Option<PendingInput>,
    pending_modifier: ModifierState,
    pub(crate) pending_input_observers: SubscriberSet<(), AnyObserver>,
    prompt: Option<RenderablePromptHandle>,
    pub(crate) client_inset: Option<Pixels>,
    #[cfg(any(feature = "inspector", debug_assertions))]
    inspector: Option<Entity<Inspector>>,
    /// The flamegraph frame index opened by the most recent `draw()` call, if
    /// a capture is active. Read (and cleared) by `present`/
    /// `present_framebuffer_only`, both of which are `&self` methods, hence
    /// the `Cell` rather than a plain field.
    #[cfg(feature = "flamegraph")]
    flamegraph_open_frame: Cell<Option<u64>>,
}

#[derive(Clone, Debug, Default)]
struct ModifierState {
    modifiers: Modifiers,
    saw_keystroke: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DrawPhase {
    None,
    Effects,
    Prepaint,
    Paint,
    Focus,
}

#[derive(Default, Debug)]
struct PendingInput {
    keystrokes: SmallVec<[Keystroke; 1]>,
    focus: Option<FocusId>,
    timer: Option<Task<()>>,
    needs_timeout: bool,
}

pub(crate) struct ElementStateBox {
    pub(crate) inner: Box<dyn Any>,
    #[cfg(debug_assertions)]
    pub(crate) type_name: &'static str,
}

fn default_bounds(display_id: Option<DisplayId>, cx: &mut App) -> WindowBounds {
    let window_bounds = cx
        .windows()
        .iter()
        .find_map(|w| w.update(cx, |_, window, _| window.window_bounds()).ok());

    const CASCADE_OFFSET: f32 = 25.0;

    let display = display_id
        .map(|id| cx.find_display(id))
        .unwrap_or_else(|| cx.primary_display());

    let default_placement = || Bounds::new(point(px(0.), px(0.)), DEFAULT_WINDOW_SIZE);

    // Use visible_bounds to exclude taskbar/dock areas
    let display_bounds = display
        .as_ref()
        .map(|d| d.visible_bounds())
        .unwrap_or_else(default_placement);

    let (
        Bounds {
            origin: base_origin,
            size: base_size,
        },
        window_bounds_ctor,
    ): (_, fn(Bounds<Pixels>) -> WindowBounds) = match window_bounds {
        Some(bounds) => match bounds {
            WindowBounds::Windowed(bounds) => (bounds, WindowBounds::Windowed),
            WindowBounds::Maximized(bounds) => (bounds, WindowBounds::Maximized),
            WindowBounds::Fullscreen(bounds) => (bounds, WindowBounds::Fullscreen),
        },
        None => (
            display
                .as_ref()
                .map(|d| d.default_bounds())
                .unwrap_or_else(default_placement),
            WindowBounds::Windowed,
        ),
    };

    let cascade_offset = point(px(CASCADE_OFFSET), px(CASCADE_OFFSET));
    let proposed_origin = base_origin + cascade_offset;
    let proposed_bounds = Bounds::new(proposed_origin, base_size);

    let display_right = display_bounds.origin.x + display_bounds.size.width;
    let display_bottom = display_bounds.origin.y + display_bounds.size.height;
    let window_right = proposed_bounds.origin.x + proposed_bounds.size.width;
    let window_bottom = proposed_bounds.origin.y + proposed_bounds.size.height;

    let fits_horizontally = window_right <= display_right;
    let fits_vertically = window_bottom <= display_bottom;

    let final_origin = match (fits_horizontally, fits_vertically) {
        (true, true) => proposed_origin,
        (false, true) => point(display_bounds.origin.x, base_origin.y),
        (true, false) => point(base_origin.x, display_bounds.origin.y),
        (false, false) => display_bounds.origin,
    };
    window_bounds_ctor(Bounds::new(final_origin, base_size))
}

impl Window {
    pub(crate) fn new(
        handle: AnyWindowHandle,
        options: WindowOptions,
        cx: &mut App,
    ) -> Result<Self> {
        let WindowOptions {
            window_bounds,
            titlebar,
            focus,
            show,
            kind,
            is_movable,
            is_resizable,
            is_minimizable,
            display_id,
            window_background,
            app_id,
            window_min_size,
            window_decorations,
            tabbing_identifier,
            app_icon,
        } = options;

        let window_bounds = window_bounds.unwrap_or_else(|| default_bounds(display_id, cx));
        let mut platform_window = cx.platform.open_window(
            handle,
            WindowParams {
                bounds: window_bounds.get_bounds(),
                titlebar,
                kind,
                is_movable,
                is_resizable,
                is_minimizable,
                focus,
                show,
                display_id,
                window_min_size,
                tabbing_identifier,
                window_decorations,
                app_icon,
            },
        )?;

        let tab_bar_visible = platform_window.tab_bar_visible();
        SystemWindowTabController::init_visible(cx, tab_bar_visible);
        if let Some(tabs) = platform_window.tabbed_windows() {
            SystemWindowTabController::add_tab(cx, handle.window_id(), tabs);
        }

        let display_id = platform_window.display().map(|display| display.id());
        let sprite_atlas = platform_window.sprite_atlas();
        let mouse_position = platform_window.mouse_position();
        let modifiers = platform_window.modifiers();
        let capslock = platform_window.capslock();
        let content_size = platform_window.content_size();
        let scale_factor = platform_window.scale_factor();
        let appearance = platform_window.appearance();
        let text_system = Arc::new(WindowTextSystem::new(cx.text_system().clone()));
        let invalidator = WindowInvalidator::new();
        let active = Rc::new(Cell::new(platform_window.is_active()));
        let hovered = Rc::new(Cell::new(platform_window.is_hovered()));
        let needs_present = Rc::new(Cell::new(false));
        let next_frame_callbacks: Rc<RefCell<Vec<FrameCallback>>> = Default::default();
        let last_input_timestamp = Rc::new(Cell::new(Instant::now()));

        platform_window
            .request_decorations(window_decorations.unwrap_or(WindowDecorations::Server));
        platform_window.set_background_appearance(window_background);

        match window_bounds {
            WindowBounds::Fullscreen(_) => platform_window.toggle_fullscreen(),
            WindowBounds::Maximized(_) => platform_window.zoom(),
            WindowBounds::Windowed(_) => {}
        }

        platform_window.on_close(Box::new({
            let window_id = handle.window_id();
            let mut cx = cx.to_async();
            move || {
                let _ = handle.update(&mut cx, |_, window, _| window.remove_window());
                let _ = cx.update(|cx| {
                    SystemWindowTabController::remove_tab(cx, window_id);
                });
            }
        }));
        platform_window.on_request_frame(Box::new({
            let mut cx = cx.to_async();
            let invalidator = invalidator.clone();
            let active = active.clone();
            let needs_present = needs_present.clone();
            let next_frame_callbacks = next_frame_callbacks.clone();
            let last_input_timestamp = last_input_timestamp.clone();
            move |request_frame_options| {
                let next_frame_callbacks = next_frame_callbacks.take();
                if !next_frame_callbacks.is_empty() {
                    handle
                        .update(&mut cx, |_, window, cx| {
                            for callback in next_frame_callbacks {
                                callback(window, cx);
                            }
                        })
                        .log_err();
                }

                // Keep presenting the current scene for 1 extra second since the
                // last input to prevent the display from underclocking the refresh rate.
                let needs_present = request_frame_options.require_presentation
                    || needs_present.get()
                    || (active.get()
                        && last_input_timestamp.get().elapsed() < Duration::from_secs(1));

                if invalidator.is_dirty() || request_frame_options.force_render {
                    #[cfg(feature = "flamegraph")]
                    crate::record_frame_pacing(true);
                    crate::render_stats::count("window: full draw");
                    if request_frame_options.force_render {
                        crate::render_stats::count("window: forced (cache bypassed)");
                    }
                    let _t = crate::render_stats::scope("window: draw + present");
                    measure("frame duration", || {
                        handle
                            .update(&mut cx, |_, window, cx| {
                                if request_frame_options.force_render {
                                    // Bypass cached view reuse so we don't replay stale
                                    // atlas tile references after a GPU device recovery.
                                    window.refresh();
                                }
                                let arena_clear_needed = window.draw(cx);
                                window.present();
                                // drop the arena elements after present to reduce latency
                                arena_clear_needed.clear();
                            })
                            .log_err();
                    })
                } else if needs_present {
                    #[cfg(feature = "flamegraph")]
                    crate::record_frame_pacing(false);
                    crate::render_stats::count("window: present only (no draw)");
                    // Fast path: framebuffer already updated by surface blit, just present it
                    handle
                        .update(&mut cx, |_, window, _| window.present_framebuffer_only())
                        .log_err();
                }

                handle
                    .update(&mut cx, |_, window, _| {
                        window.complete_frame();
                    })
                    .log_err();

                // Drives the once-per-second dump for `WGPUI_RENDER_STATS=1`.
                // Ticked per platform frame rather than per draw, so the
                // "full draw" / "present only" counters above are meaningful
                // relative to it.
                crate::render_stats::tick_frame();
            }
        }));
        platform_window.on_resize(Box::new({
            let mut cx = cx.to_async();
            move |_, _| {
                handle
                    .update(&mut cx, |_, window, cx| window.bounds_changed(cx))
                    .log_err();
            }
        }));
        platform_window.on_moved(Box::new({
            let mut cx = cx.to_async();
            move || {
                handle
                    .update(&mut cx, |_, window, cx| window.bounds_changed(cx))
                    .log_err();
            }
        }));
        platform_window.on_appearance_changed(Box::new({
            let mut cx = cx.to_async();
            move || {
                handle
                    .update(&mut cx, |_, window, cx| window.appearance_changed(cx))
                    .log_err();
            }
        }));
        platform_window.on_active_status_change(Box::new({
            let mut cx = cx.to_async();
            let active_state = active.clone();
            move |active| {
                active_state.set(active);
                handle
                    .update(&mut cx, |_, window, cx| {
                        window.active.set(active);
                        window.modifiers = window.platform_window.modifiers();
                        window.capslock = window.platform_window.capslock();
                        window
                            .activation_observers
                            .clone()
                            .retain(&(), |callback| callback(window, cx));

                        window.bounds_changed(cx);
                        window.refresh();

                        SystemWindowTabController::update_last_active(cx, window.handle.id);
                    })
                    .log_err();
            }
        }));
        platform_window.on_hover_status_change(Box::new({
            let mut cx = cx.to_async();
            move |active| {
                handle
                    .update(&mut cx, |_, window, _| {
                        window.hovered.set(active);
                        window.refresh();
                    })
                    .log_err();
            }
        }));
        platform_window.on_input({
            let mut cx = cx.to_async();
            Box::new(move |event| {
                handle
                    .update(&mut cx, |_, window, cx| window.dispatch_event(event, cx))
                    .log_err()
                    .unwrap_or(DispatchEventResult::default())
            })
        });
        platform_window.on_hit_test_window_control({
            let mut cx = cx.to_async();
            Box::new(move || {
                handle
                    .update(&mut cx, |_, window, _cx| {
                        for (area, hitbox) in &window.rendered_frame.window_control_hitboxes {
                            if window.mouse_hit_test.ids.contains(&hitbox.id) {
                                return Some(*area);
                            }
                        }
                        None
                    })
                    .log_err()
                    .unwrap_or(None)
            })
        });
        platform_window.on_move_tab_to_new_window({
            let mut cx = cx.to_async();
            Box::new(move || {
                handle
                    .update(&mut cx, |_, _window, cx| {
                        SystemWindowTabController::move_tab_to_new_window(cx, handle.window_id());
                    })
                    .log_err();
            })
        });
        platform_window.on_merge_all_windows({
            let mut cx = cx.to_async();
            Box::new(move || {
                handle
                    .update(&mut cx, |_, _window, cx| {
                        SystemWindowTabController::merge_all_windows(cx, handle.window_id());
                    })
                    .log_err();
            })
        });
        platform_window.on_select_next_tab({
            let mut cx = cx.to_async();
            Box::new(move || {
                handle
                    .update(&mut cx, |_, _window, cx| {
                        SystemWindowTabController::select_next_tab(cx, handle.window_id());
                    })
                    .log_err();
            })
        });
        platform_window.on_select_previous_tab({
            let mut cx = cx.to_async();
            Box::new(move || {
                handle
                    .update(&mut cx, |_, _window, cx| {
                        SystemWindowTabController::select_previous_tab(cx, handle.window_id())
                    })
                    .log_err();
            })
        });
        platform_window.on_toggle_tab_bar({
            let mut cx = cx.to_async();
            Box::new(move || {
                handle
                    .update(&mut cx, |_, window, cx| {
                        let tab_bar_visible = window.platform_window.tab_bar_visible();
                        SystemWindowTabController::set_visible(cx, tab_bar_visible);
                    })
                    .log_err();
            })
        });

        if let Some(app_id) = app_id {
            platform_window.set_app_id(&app_id);
        }

        platform_window.map_window().unwrap();

        Ok(Window {
            handle,
            invalidator,
            removed: false,
            platform_window,
            display_id,
            sprite_atlas,
            text_system,
            rem_size: px(16.),
            rem_size_override_stack: SmallVec::new(),
            viewport_size: content_size,
            layout_engine: Some(TaffyLayoutEngine::new()),
            root: None,
            element_id_stack: SmallVec::default(),
            text_style_stack: Vec::new(),
            rendered_entity_stack: Vec::new(),
            element_offset_stack: Vec::new(),
            retained_layer_stack: Vec::new(),
            hitbox_layer_stack: Vec::new(),
            instance_id_stack: SmallVec::default(),
            layout_layer_stack: Vec::new(),
            content_mask_stack: Vec::new(),
            element_opacity: 1.0,
            requested_autoscroll: None,
            rendered_frame: Frame::new(DispatchTree::new(cx.keymap.clone(), cx.actions.clone())),
            next_frame: Frame::new(DispatchTree::new(cx.keymap.clone(), cx.actions.clone())),
            layers: FxHashMap::default(),
            layer_frame: 0,
            slab_tokens: FxHashMap::default(),
            next_slab_token: 0,
            next_layer_id: 0,
            next_frame_callbacks,
            next_hitbox_id: HitboxId(0),
            next_tooltip_id: TooltipId::default(),
            tooltip_bounds: None,
            dirty_views: FxHashSet::default(),
            invalidated_entities: FxHashSet::default(),
            focus_listeners: SubscriberSet::new(),
            focus_lost_listeners: SubscriberSet::new(),
            default_prevented: true,
            mouse_position,
            mouse_hit_test: HitTest::default(),
            modifiers,
            capslock,
            mouse_button_pressed: None,
            scale_factor,
            bounds_observers: SubscriberSet::new(),
            appearance,
            appearance_observers: SubscriberSet::new(),
            active,
            hovered,
            needs_present,
            last_input_timestamp,
            resizing_window: Rc::new(Cell::new(false)),
            last_input_modality: InputModality::Mouse,
            window_invalidation: Invalidation::empty(),
            nested_view_cache_suppressed: false,
            activation_observers: SubscriberSet::new(),
            focus: None,
            focus_enabled: true,
            pending_input: None,
            pending_modifier: ModifierState::default(),
            pending_input_observers: SubscriberSet::new(),
            prompt: None,
            client_inset: None,
            image_cache_stack: Vec::new(),
            #[cfg(any(feature = "inspector", debug_assertions))]
            inspector: None,
            #[cfg(feature = "flamegraph")]
            flamegraph_open_frame: Cell::new(None),
        })
    }

    pub(crate) fn new_focus_listener(
        &self,
        value: AnyWindowFocusListener,
    ) -> (Subscription, impl FnOnce() + use<>) {
        self.focus_listeners.insert((), value)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[expect(missing_docs)]
pub struct DispatchEventResult {
    pub propagate: bool,
    pub default_prevented: bool,
}

/// Indicates which region of the window is visible. Content falling outside of this mask will not be
/// rendered. Currently, only rectangular content masks are supported, but we give the mask its own type
/// to leave room to support more complex shapes in the future.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct ContentMask<P: Clone + Debug + Default + PartialEq> {
    /// The bounds
    pub bounds: Bounds<P>,
}

// See the note on the Pod impls for `Point` in geometry.rs.
unsafe impl<P: Clone + Debug + Default + PartialEq + bytemuck::Pod> bytemuck::Pod
    for ContentMask<P>
{
}
unsafe impl<P: Clone + Debug + Default + PartialEq + bytemuck::Pod> bytemuck::Zeroable
    for ContentMask<P>
{
}

impl ContentMask<Pixels> {
    /// Scale the content mask's pixel units by the given scaling factor.
    pub fn scale(&self, factor: f32) -> ContentMask<ScaledPixels> {
        ContentMask {
            bounds: self.bounds.scale(factor),
        }
    }

    /// Intersect the content mask with the given content mask.
    pub fn intersect(&self, other: &Self) -> Self {
        let bounds = self.bounds.intersect(&other.bounds);
        ContentMask { bounds }
    }
}

impl Window {
    fn mark_view_dirty(&mut self, view_id: EntityId) {
        // Mark ancestor views as dirty. If already in the `dirty_views` set, then all its ancestors
        // should already be dirty.
        //
        // This is the *containment* half of invalidation, and it only reaches
        // entities that own a dispatch node — that is, views that were
        // prepainted. Entities without one (models, and anything a view merely
        // reads) produce an empty path and mark nothing. The *dependency* half
        // lives in [`Self::accessed_entity_invalidated`], which `AnyView`
        // consults with each cached view's own recorded dependency set.
        //
        // The upward walk survives only because it is how
        // [`InvalidationScope::Entity`] is implemented until a reverse index is
        // possible. `Instance` and `Layer` scope name exactly what they
        // invalidate and must never reach here — that is what stops one chatty
        // leaf from defeating every cache above it.
        for view_id in self
            .rendered_frame
            .dispatch_tree
            .view_path_reversed(view_id)
        {
            if !self.dirty_views.insert(view_id) {
                break;
            }
        }
    }

    /// Whether any entity invalidated since the last frame appears in a cached
    /// view's recorded dependency set — that is, whether its stored output is
    /// known to be out of date.
    ///
    /// This is the half of invalidation that `dirty_views` cannot express.
    /// `dirty_views` answers "does this view *contain* something that changed",
    /// via the dispatch-tree ancestor path, and a model has no node on that
    /// path: notifying one marked no view dirty at all, so every cached view
    /// judged itself clean and replayed stale content indefinitely. That is
    /// issue #83.
    ///
    /// Asking each cached view about its own dependency set instead is exact
    /// and needs no extra bookkeeping — `AnyView` already records the set for
    /// window tracking. It also composes with nesting for free: a cached view's
    /// set is cumulative over its whole subtree, so every cached layer between
    /// the root and the view that actually reads the entity fails this test
    /// too, and the inner one gets prepainted at all. That last part is what a
    /// reverse "who depends on this entity" index cannot do on its own — a
    /// nested cached view that replayed last frame is absent from any
    /// per-frame index precisely because it never ran.
    ///
    /// Cost is one membership test per invalidated entity per cached view; the
    /// invalidated set is normally a handful of entries, and the loop runs over
    /// whichever side is smaller.
    pub(crate) fn accessed_entity_invalidated(&self, accessed: &FxHashSet<EntityId>) -> bool {
        if self.invalidated_entities.is_empty() || accessed.is_empty() {
            return false;
        }
        if self.invalidated_entities.len() <= accessed.len() {
            self.invalidated_entities
                .iter()
                .any(|entity_id| accessed.contains(entity_id))
        } else {
            accessed
                .iter()
                .any(|entity_id| self.invalidated_entities.contains(entity_id))
        }
    }

    /// Registers a callback to be invoked when the window appearance changes.
    pub fn observe_window_appearance(
        &self,
        mut callback: impl FnMut(&mut Window, &mut App) + 'static,
    ) -> Subscription {
        let (subscription, activate) = self.appearance_observers.insert(
            (),
            Box::new(move |window, cx| {
                callback(window, cx);
                true
            }),
        );
        activate();
        subscription
    }

    /// Replaces the root entity of the window with a new one.
    pub fn replace_root<E>(
        &mut self,
        cx: &mut App,
        build_view: impl FnOnce(&mut Window, &mut Context<E>) -> E,
    ) -> Entity<E>
    where
        E: 'static + Render,
    {
        let view = cx.new(|cx| build_view(self, cx));
        self.root = Some(view.clone().into());
        self.refresh();
        view
    }

    /// Returns the root entity of the window, if it has one.
    pub fn root<E>(&self) -> Option<Option<Entity<E>>>
    where
        E: 'static + Render,
    {
        self.root
            .as_ref()
            .map(|view| view.clone().downcast::<E>().ok())
    }

    /// Returns the root view without TypeId-based downcasting.
    /// Useful for cross-DLL access where type identities differ.
    pub fn root_view(&self) -> Option<&AnyView> {
        self.root.as_ref()
    }

    /// Read the root entity data without TypeId checking.
    /// # Safety
    /// The caller must ensure the root is of type T.
    pub unsafe fn read_root<'a, T: 'static>(&self, cx: &'a App) -> &'a T {
        let view = self.root_view().expect("window has no root");
        cx.entities.read_unchecked(view.entity_id())
    }

    /// Update the root entity data without TypeId checking.
    /// # Safety
    /// The caller must ensure the root is of type T.
    pub unsafe fn update_root<T: 'static, R>(
        &mut self,
        cx: &mut App,
        f: impl FnOnce(&mut T, &mut Window, &mut Context<T>) -> R,
    ) -> R {
        let view = self.root_view().expect("window has no root");
        let entity_id = view.entity_id();
        let weak = view.downgrade_unchecked::<T>();
        let mut entity = cx.entities.lease_unchecked::<T>(entity_id);
        let result = f(&mut entity, self, &mut Context::new_context(cx, weak));
        cx.entities.end_lease_unchecked(entity);
        result
    }

    /// Obtain a handle to the window that belongs to this context.
    pub fn window_handle(&self) -> AnyWindowHandle {
        self.handle
    }

    /// Mark the window as dirty, scheduling it to be redrawn on the next frame
    /// with no view reusing its cached output.
    ///
    /// Deprecated in favour of invalidating what actually changed: `cx.notify()`
    /// on the entity, or [`Self::request_animation_frame`] for an animation
    /// driver, which notifies only the enclosing view. This is
    /// [`InvalidationScope::Window`] with every axis set, which is the bluntest
    /// request the framework can express — it is correct for device loss or a
    /// scale factor change, and wasteful for everything else. It is not marked
    /// `#[deprecated]` only because it still has dozens of in-crate callers with
    /// nowhere better to go yet.
    ///
    /// Unlike the pre-#87 version, this is legal during a draw: a request made
    /// from prepaint or paint is deferred to the end of the frame rather than
    /// silently dropped.
    pub fn refresh(&mut self) {
        self.invalidator.invalidate_window(Invalidation::all());
    }

    /// Request a frame because externally-rendered buffer contents changed,
    /// without invalidating any view.
    ///
    /// This exists because "the window needs a frame" and "some view produced
    /// stale output" are different questions that the dirty flag otherwise
    /// conflates. A `WgpuSurface` producer only answers yes to the first: its
    /// texture advanced, but nothing in the element tree changed. The scene
    /// GPUI would rebuild is identical to the last one, and cached views replay
    /// their primitives — including the surface quad — so the compositor
    /// promotes the new texture with nothing re-rendered.
    ///
    /// The same shape serves texture-retained layers (#96): a DISPLAY-only
    /// frame leaves every layer's `needs` untouched, so a clean texture-
    /// retained layer re-emits exactly its composite surface and the renderer
    /// promotes whatever its texture now holds. `refresh_buffers` is the
    /// composite trigger for both kinds of retained texture.
    ///
    /// This is [`InvalidationScope::Window`] with [`Invalidation::DISPLAY`] and
    /// nothing else, and that is the whole difference from [`Self::refresh`]:
    /// no view's layout or hit geometry is claimed to be stale, so cached views
    /// keep replaying.
    ///
    /// The alternatives both over-trigger:
    /// * [`refresh`](Self::refresh) sets every axis, disabling view caching for
    ///   the whole window.
    /// * `cx.notify()` marks the view *and every ancestor* dirty
    ///   ([`mark_view_dirty`](Self::mark_view_dirty) walks the ancestor path),
    ///   so a leaf publishing frames re-renders everything above it.
    ///
    /// Because `dirty_views` is a set, this composes with genuine invalidation
    /// in the same frame: whatever else was marked dirty still rebuilds, and a
    /// buffer refresh neither blocks it nor forces anything extra.
    ///
    /// Callers must still ensure the producing view renders when its *layout*
    /// changes — a view that never prepaints never observes new bounds.
    pub fn refresh_buffers(&mut self) {
        self.invalidator.invalidate_window(Invalidation::DISPLAY);
    }

    /// Whether a cached view may replay its recorded output this frame.
    ///
    /// Two independent things forbid it. A window-scope invalidation touching
    /// [`LAYOUT`](Invalidation::LAYOUT) or [`HIT`](Invalidation::HIT) says the
    /// recorded prepaint — layouts, hitboxes, dispatch nodes — describes a
    /// world that no longer exists, and nothing short of re-running the view
    /// produces a new one. Window-scope `DISPLAY` alone deliberately does not:
    /// that is `refresh_buffers`, where the pixels behind a surface quad
    /// advanced but the quad referring to them did not.
    ///
    /// The other is `nested_view_cache_suppressed`, which is about where the
    /// element walk is rather than about what changed.
    pub(crate) fn view_cache_available(&self) -> bool {
        const REPLAY_INVALIDATING: Invalidation = Invalidation::LAYOUT.union(Invalidation::HIT);
        !self.nested_view_cache_suppressed
            && !self.window_invalidation.intersects(REPLAY_INVALIDATING)
    }

    /// Close this window.
    pub fn remove_window(&mut self) {
        self.removed = true;
    }

    /// Obtain the currently focused [`FocusHandle`]. If no elements are focused, returns `None`.
    pub fn focused(&self, cx: &App) -> Option<FocusHandle> {
        self.focus
            .and_then(|id| FocusHandle::for_id(id, &cx.focus_handles))
    }

    /// Move focus to the element associated with the given [`FocusHandle`].
    pub fn focus(&mut self, handle: &FocusHandle, cx: &mut App) {
        if !self.focus_enabled || self.focus == Some(handle.id) {
            return;
        }

        self.focus = Some(handle.id);
        self.clear_pending_keystrokes();

        // Avoid re-entrant entity updates by deferring observer notifications to the end of the
        // current effect cycle, and only for this window.
        let window_handle = self.handle;
        cx.defer(move |cx| {
            window_handle
                .update(cx, |_, window, cx| {
                    window.pending_input_changed(cx);
                })
                .ok();
        });

        self.refresh();
    }

    /// Remove focus from all elements within this context's window.
    pub fn blur(&mut self) {
        if !self.focus_enabled {
            return;
        }

        self.focus = None;
        self.refresh();
    }

    /// Blur the window and don't allow anything in it to be focused again.
    pub fn disable_focus(&mut self) {
        self.blur();
        self.focus_enabled = false;
    }

    /// Move focus to next tab stop.
    pub fn focus_next(&mut self, cx: &mut App) {
        if !self.focus_enabled {
            return;
        }

        if let Some(handle) = self.rendered_frame.tab_stops.next(self.focus.as_ref()) {
            self.focus(&handle, cx)
        }
    }

    /// Move focus to previous tab stop.
    pub fn focus_prev(&mut self, cx: &mut App) {
        if !self.focus_enabled {
            return;
        }

        if let Some(handle) = self.rendered_frame.tab_stops.prev(self.focus.as_ref()) {
            self.focus(&handle, cx)
        }
    }

    /// Accessor for the text system.
    pub fn text_system(&self) -> &Arc<WindowTextSystem> {
        &self.text_system
    }

    /// On-demand snapshot of memory held by WGPUI's own CPU-side subsystems
    /// for this window (Phase 3 of the profiling epic, issue #59): the
    /// per-frame element arena, this window's text caches, the built-in
    /// image cache, and the flamegraph capture engine's own footprint. Cheap
    /// -- a summation over already-allocated caches, not a fresh allocation
    /// pass -- but not tracked per-frame like spans/counters, so call it as
    /// needed rather than every frame.
    #[cfg(feature = "flamegraph")]
    pub fn memory_snapshot(&self, cx: &App) -> crate::MemorySnapshot {
        crate::MemorySnapshot {
            element_arena_bytes: cx.element_arena_capacity_bytes(),
            text_system: self.text_system.memory_snapshot(),
            image_cache_bytes: crate::elements::total_retained_image_cache_memory_usage(cx),
            capture_engine_bytes: crate::capture_engine_memory_usage(),
        }
    }

    /// On-demand GPU memory snapshot for this window's renderer (Phase 3 of
    /// the profiling epic, issue #59). `None` on platforms/backends that
    /// don't use the WGPU renderer, or before the renderer has been created.
    #[cfg(feature = "flamegraph")]
    pub fn gpu_memory_snapshot(&self) -> Option<crate::GpuMemorySnapshot> {
        self.platform_window.gpu_memory_snapshot()
    }

    /// The live `wgpu::Device`/`Queue` backing this window's renderer, so a
    /// GPU deep-capture replay preview (Phase 6 of the profiling epic, issue
    /// #62, `flamegraph_replay::render_deep_capture_step`) can run against
    /// the app's real device instead of creating a second one. `None` on
    /// platforms/backends that don't use the WGPU renderer, or before the
    /// renderer has been created.
    #[cfg(feature = "flamegraph")]
    pub fn gpu_device_and_queue(&self) -> Option<(wgpu::Device, wgpu::Queue)> {
        self.platform_window.gpu_device_and_queue()
    }

    /// The current text style. Which is composed of all the style refinements provided to `with_text_style`.
    pub fn text_style(&self) -> TextStyle {
        let mut style = TextStyle::default();
        for refinement in &self.text_style_stack {
            style.refine(refinement);
        }
        style
    }

    /// Check if the platform window is maximized.
    ///
    /// On some platforms (namely Windows) this is different than the bounds being the size of the display
    pub fn is_maximized(&self) -> bool {
        self.platform_window.is_maximized()
    }

    /// request a certain window decoration (Wayland)
    pub fn request_decorations(&self, decorations: WindowDecorations) {
        self.platform_window.request_decorations(decorations);
    }

    /// Start a window resize operation (Wayland)
    pub fn start_window_resize(&self, edge: ResizeEdge) {
        self.platform_window.start_window_resize(edge);
    }

    /// Return the `WindowBounds` to indicate that how a window should be opened
    /// after it has been closed
    pub fn window_bounds(&self) -> WindowBounds {
        self.platform_window.window_bounds()
    }

    /// Return the `WindowBounds` excluding insets (Wayland and X11)
    pub fn inner_window_bounds(&self) -> WindowBounds {
        self.platform_window.inner_window_bounds()
    }

    /// Dispatch the given action on the currently focused element.
    pub fn dispatch_action(&mut self, action: Box<dyn Action>, cx: &mut App) {
        let focus_id = self.focused(cx).map(|handle| handle.id);

        let window = self.handle;
        cx.defer(move |cx| {
            window
                .update(cx, |_, window, cx| {
                    let node_id = window.focus_node_id_in_rendered_frame(focus_id);
                    window.dispatch_action_on_node(node_id, action.as_ref(), cx);
                })
                .log_err();
        })
    }

    pub(crate) fn dispatch_keystroke_observers(
        &mut self,
        event: &dyn Any,
        action: Option<Box<dyn Action>>,
        context_stack: Vec<KeyContext>,
        cx: &mut App,
    ) {
        let Some(key_down_event) = event.downcast_ref::<KeyDownEvent>() else {
            return;
        };

        cx.keystroke_observers.clone().retain(&(), move |callback| {
            (callback)(
                &KeystrokeEvent {
                    keystroke: key_down_event.keystroke.clone(),
                    action: action.as_ref().map(|action| action.boxed_clone()),
                    context_stack: context_stack.clone(),
                },
                self,
                cx,
            )
        });
    }

    pub(crate) fn dispatch_keystroke_interceptors(
        &mut self,
        event: &dyn Any,
        context_stack: Vec<KeyContext>,
        cx: &mut App,
    ) {
        let Some(key_down_event) = event.downcast_ref::<KeyDownEvent>() else {
            return;
        };

        cx.keystroke_interceptors
            .clone()
            .retain(&(), move |callback| {
                (callback)(
                    &KeystrokeEvent {
                        keystroke: key_down_event.keystroke.clone(),
                        action: None,
                        context_stack: context_stack.clone(),
                    },
                    self,
                    cx,
                )
            });
    }

    /// Schedules the given function to be run at the end of the current effect cycle, allowing entities
    /// that are currently on the stack to be returned to the app.
    pub fn defer(&self, cx: &mut App, f: impl FnOnce(&mut Window, &mut App) + 'static) {
        let handle = self.handle;
        cx.defer(move |cx| {
            handle.update(cx, |_, window, cx| f(window, cx)).ok();
        });
    }

    /// Subscribe to events emitted by a entity.
    /// The entity to which you're subscribing must implement the [`EventEmitter`] trait.
    /// The callback will be invoked a handle to the emitting entity, the event, and a window context for the current window.
    pub fn observe<T: 'static>(
        &mut self,
        observed: &Entity<T>,
        cx: &mut App,
        mut on_notify: impl FnMut(Entity<T>, &mut Window, &mut App) + 'static,
    ) -> Subscription {
        let entity_id = observed.entity_id();
        let observed = observed.downgrade();
        let window_handle = self.handle;
        cx.new_observer(
            entity_id,
            Box::new(move |cx| {
                window_handle
                    .update(cx, |_, window, cx| {
                        if let Some(handle) = observed.upgrade() {
                            on_notify(handle, window, cx);
                            true
                        } else {
                            false
                        }
                    })
                    .unwrap_or(false)
            }),
        )
    }

    /// Subscribe to events emitted by a entity.
    /// The entity to which you're subscribing must implement the [`EventEmitter`] trait.
    /// The callback will be invoked a handle to the emitting entity, the event, and a window context for the current window.
    pub fn subscribe<Emitter, Evt>(
        &mut self,
        entity: &Entity<Emitter>,
        cx: &mut App,
        mut on_event: impl FnMut(Entity<Emitter>, &Evt, &mut Window, &mut App) + 'static,
    ) -> Subscription
    where
        Emitter: EventEmitter<Evt>,
        Evt: 'static,
    {
        let entity_id = entity.entity_id();
        let handle = entity.downgrade();
        let window_handle = self.handle;
        cx.new_subscription(
            entity_id,
            (
                TypeId::of::<Evt>(),
                Box::new(move |event, cx| {
                    window_handle
                        .update(cx, |_, window, cx| {
                            if let Some(entity) = handle.upgrade() {
                                let event = event.downcast_ref().expect("invalid event type");
                                on_event(entity, event, window, cx);
                                true
                            } else {
                                false
                            }
                        })
                        .unwrap_or(false)
                }),
            ),
        )
    }

    /// Register a callback to be invoked when the given `Entity` is released.
    pub fn observe_release<T>(
        &self,
        entity: &Entity<T>,
        cx: &mut App,
        mut on_release: impl FnOnce(&mut T, &mut Window, &mut App) + 'static,
    ) -> Subscription
    where
        T: 'static,
    {
        let entity_id = entity.entity_id();
        let window_handle = self.handle;
        let (subscription, activate) = cx.release_listeners.insert(
            entity_id,
            Box::new(move |entity, cx| {
                let entity = entity.downcast_mut().expect("invalid entity type");
                let _ = window_handle.update(cx, |_, window, cx| on_release(entity, window, cx));
            }),
        );
        activate();
        subscription
    }

    /// Creates an [`AsyncWindowContext`], which has a static lifetime and can be held across
    /// await points in async code.
    pub fn to_async(&self, cx: &App) -> AsyncWindowContext {
        AsyncWindowContext::new_context(cx.to_async(), self.handle)
    }

    /// Schedule the given closure to be run directly after the current frame is rendered.
    pub fn on_next_frame(&self, callback: impl FnOnce(&mut Window, &mut App) + 'static) {
        RefCell::borrow_mut(&self.next_frame_callbacks).push(Box::new(callback));
    }

    /// Schedule a frame to be drawn on the next animation frame.
    ///
    /// This is useful for elements that need to animate continuously, such as a video player or an animated GIF.
    /// It will cause the window to redraw on the next frame, even if no other changes have occurred.
    ///
    /// If called from within a view, it will notify that view on the next frame. Otherwise, it will refresh the entire window.
    pub fn request_animation_frame(&self) {
        if let Some(entity) = self.rendered_entity_stack.last().copied() {
            self.on_next_frame(move |_, cx| cx.notify(entity));
        } else {
            self.on_next_frame(|window, _| window.refresh());
        }
    }

    /// Spawn the future returned by the given closure on the application thread pool.
    /// The closure is provided a handle to the current window and an `AsyncWindowContext` for
    /// use within your future.
    #[track_caller]
    pub fn spawn<AsyncFn, R>(&self, cx: &App, f: AsyncFn) -> Task<R>
    where
        R: 'static,
        AsyncFn: AsyncFnOnce(&mut AsyncWindowContext) -> R + 'static,
    {
        let handle = self.handle;
        cx.spawn(async move |app| {
            let mut async_window_cx = AsyncWindowContext::new_context(app.clone(), handle);
            f(&mut async_window_cx).await
        })
    }

    /// Spawn the future returned by the given closure on the application thread
    /// pool, with the given priority. The closure is provided a handle to the
    /// current window and an `AsyncWindowContext` for use within your future.
    #[track_caller]
    pub fn spawn_with_priority<AsyncFn, R>(
        &self,
        priority: Priority,
        cx: &App,
        f: AsyncFn,
    ) -> Task<R>
    where
        R: 'static,
        AsyncFn: AsyncFnOnce(&mut AsyncWindowContext) -> R + 'static,
    {
        let handle = self.handle;
        cx.spawn_with_priority(priority, async move |app| {
            let mut async_window_cx = AsyncWindowContext::new_context(app.clone(), handle);
            f(&mut async_window_cx).await
        })
    }

    /// Notify the window that its bounds have changed.
    ///
    /// This updates internal state like `viewport_size` and `scale_factor` from
    /// the platform window, then notifies observers. Normally called automatically
    /// by the platform's resize callback, but exposed publicly for test infrastructure.
    pub fn bounds_changed(&mut self, cx: &mut App) {
        let previous_viewport_size = self.viewport_size;
        self.scale_factor = self.platform_window.scale_factor();
        self.viewport_size = self.platform_window.content_size();
        self.display_id = self.platform_window.display().map(|display| display.id());
        self.resizing_window
            .set(previous_viewport_size != self.viewport_size);

        self.refresh();

        self.bounds_observers
            .clone()
            .retain(&(), |callback| callback(self, cx));
    }

    /// Returns the bounds of the current window in the global coordinate space, which could span across multiple displays.
    pub fn bounds(&self) -> Bounds<Pixels> {
        self.platform_window.bounds()
    }

    /// Set the content size of the window.
    pub fn resize(&mut self, size: Size<Pixels>) {
        self.platform_window.resize(size);
    }

    /// Returns whether or not the window is currently fullscreen
    pub fn is_fullscreen(&self) -> bool {
        self.platform_window.is_fullscreen()
    }

    pub(crate) fn appearance_changed(&mut self, cx: &mut App) {
        self.appearance = self.platform_window.appearance();

        self.appearance_observers
            .clone()
            .retain(&(), |callback| callback(self, cx));
    }

    /// Returns the appearance of the current window.
    pub fn appearance(&self) -> WindowAppearance {
        self.appearance
    }

    /// Returns the size of the drawable area within the window.
    pub fn viewport_size(&self) -> Size<Pixels> {
        self.viewport_size
    }

    /// Returns whether this window is focused by the operating system (receiving key events).
    pub fn is_window_active(&self) -> bool {
        self.active.get()
    }

    /// Returns whether this window is considered to be the window
    /// that currently owns the mouse cursor.
    /// On mac, this is equivalent to `is_window_active`.
    pub fn is_window_hovered(&self) -> bool {
        if cfg!(any(
            target_os = "windows",
            target_os = "linux",
            target_os = "freebsd"
        )) {
            self.hovered.get()
        } else {
            self.is_window_active()
        }
    }

    /// Toggle zoom on the window.
    pub fn zoom_window(&self) {
        self.platform_window.zoom();
    }

    /// Opens the native title bar context menu, useful when implementing client side decorations (Wayland and X11)
    pub fn show_window_menu(&self, position: Point<Pixels>) {
        self.platform_window.show_window_menu(position)
    }

    /// Handle window movement for Linux and macOS.
    /// Tells the compositor to take control of window movement (Wayland and X11)
    ///
    /// Events may not be received during a move operation.
    pub fn start_window_move(&self) {
        self.platform_window.start_window_move()
    }

    /// Move the window so its top-left corner is at `position` in logical screen coordinates.
    /// Cross-platform: no-op on platforms that don't implement it.
    pub fn set_window_position(&self, position: Point<Pixels>) {
        self.platform_window.set_window_position(position)
    }

    /// Call `f` with a reference to the underlying `winit::window::Window`.
    /// `f` is not called on platforms that don't use winit (e.g. the test backend).
    pub fn with_winit_window(&self, mut f: impl FnMut(&crate::platform::cross::WinitWindow)) {
        self.platform_window.with_winit_window(&mut f);
    }

    /// When using client side decorations, set this to the width of the invisible decorations (Wayland and X11)
    pub fn set_client_inset(&mut self, inset: Pixels) {
        self.client_inset = Some(inset);
        self.platform_window.set_client_inset(inset);
    }

    /// Returns the client_inset value by [`Self::set_client_inset`].
    pub fn client_inset(&self) -> Option<Pixels> {
        self.client_inset
    }

    /// Returns whether the title bar window controls need to be rendered by the application (Wayland and X11)
    pub fn window_decorations(&self) -> Decorations {
        self.platform_window.window_decorations()
    }

    /// Returns which window controls are currently visible (Wayland)
    pub fn window_controls(&self) -> WindowControls {
        self.platform_window.window_controls()
    }

    /// Updates the window's title at the platform level.
    pub fn set_window_title(&mut self, title: &str) {
        self.platform_window.set_title(title);
    }

    /// Sets the application identifier.
    pub fn set_app_id(&mut self, app_id: &str) {
        self.platform_window.set_app_id(app_id);
    }

    /// Sets the window background appearance.
    pub fn set_background_appearance(&self, background_appearance: WindowBackgroundAppearance) {
        self.platform_window
            .set_background_appearance(background_appearance);
    }

    /// Mark the window as dirty at the platform level.
    pub fn set_window_edited(&mut self, edited: bool) {
        self.platform_window.set_edited(edited);
    }

    /// Determine the display on which the window is visible.
    pub fn display(&self, cx: &App) -> Option<Rc<dyn PlatformDisplay>> {
        cx.platform
            .displays()
            .into_iter()
            .find(|display| Some(display.id()) == self.display_id)
    }

    /// Show the platform character palette.
    pub fn show_character_palette(&self) {
        self.platform_window.show_character_palette();
    }

    /// The scale factor of the display associated with the window. For example, it could
    /// return 2.0 for a "retina" display, indicating that each logical pixel should actually
    /// be rendered as two pixels on screen.
    pub fn scale_factor(&self) -> f32 {
        self.scale_factor
    }

    /// The size of an em for the base font of the application. Adjusting this value allows the
    /// UI to scale, just like zooming a web page.
    pub fn rem_size(&self) -> Pixels {
        self.rem_size_override_stack
            .last()
            .copied()
            .unwrap_or(self.rem_size)
    }

    /// Sets the size of an em for the base font of the application. Adjusting this value allows the
    /// UI to scale, just like zooming a web page.
    pub fn set_rem_size(&mut self, rem_size: impl Into<Pixels>) {
        self.rem_size = rem_size.into();
    }

    /// Acquire a globally unique identifier for the given ElementId.
    /// Only valid for the duration of the provided closure.
    pub fn with_global_id<R>(
        &mut self,
        element_id: ElementId,
        f: impl FnOnce(&GlobalElementId, &mut Self) -> R,
    ) -> R {
        self.with_id(element_id, |this| {
            let global_id = GlobalElementId(Arc::from(&*this.element_id_stack));

            f(&global_id, this)
        })
    }

    /// Calls the provided closure with the element ID pushed on the stack.
    #[inline]
    pub fn with_id<R>(
        &mut self,
        element_id: impl Into<ElementId>,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        self.element_id_stack.push(element_id.into());
        let result = f(self);
        self.element_id_stack.pop();
        result
    }

    /// Executes the provided function with the specified rem size.
    ///
    /// This method must only be called as part of element drawing.
    // This function is called in a highly recursive manner in editor
    // prepainting, make sure its inlined to reduce the stack burden
    #[inline]
    pub fn with_rem_size<F, R>(&mut self, rem_size: Option<impl Into<Pixels>>, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        self.invalidator.debug_assert_paint_or_prepaint();

        if let Some(rem_size) = rem_size {
            self.rem_size_override_stack.push(rem_size.into());
            let result = f(self);
            self.rem_size_override_stack.pop();
            result
        } else {
            f(self)
        }
    }

    /// The line height associated with the current text style.
    pub fn line_height(&self) -> Pixels {
        self.text_style().line_height_in_pixels(self.rem_size())
    }

    /// Call to prevent the default action of an event. Currently only used to prevent
    /// parent elements from becoming focused on mouse down.
    pub fn prevent_default(&mut self) {
        self.default_prevented = true;
    }

    /// Obtain whether default has been prevented for the event currently being dispatched.
    pub fn default_prevented(&self) -> bool {
        self.default_prevented
    }

    /// Determine whether the given action is available along the dispatch path to the currently focused element.
    pub fn is_action_available(&self, action: &dyn Action, cx: &App) -> bool {
        let node_id =
            self.focus_node_id_in_rendered_frame(self.focused(cx).map(|handle| handle.id));
        self.rendered_frame
            .dispatch_tree
            .is_action_available(action, node_id)
    }

    /// Determine whether the given action is available along the dispatch path to the given focus_handle.
    pub fn is_action_available_in(&self, action: &dyn Action, focus_handle: &FocusHandle) -> bool {
        let node_id = self.focus_node_id_in_rendered_frame(Some(focus_handle.id));
        self.rendered_frame
            .dispatch_tree
            .is_action_available(action, node_id)
    }

    /// The position of the mouse relative to the window.
    pub fn mouse_position(&self) -> Point<Pixels> {
        self.mouse_position
    }

    /// The current pressed mouse button, if any.
    pub fn pressed_mouse_button(&self) -> Option<MouseButton> {
        self.mouse_button_pressed
    }

    /// The current state of the keyboard's modifiers
    pub fn modifiers(&self) -> Modifiers {
        self.modifiers
    }

    /// Returns true if the last input event was keyboard-based (key press, tab navigation, etc.)
    /// This is used for focus-visible styling to show focus indicators only for keyboard navigation.
    pub fn last_input_was_keyboard(&self) -> bool {
        self.last_input_modality == InputModality::Keyboard
    }

    /// The current state of the keyboard's capslock
    pub fn capslock(&self) -> Capslock {
        self.capslock
    }

    fn complete_frame(&self) {
        self.platform_window.completed_frame();
        if !self.platform_window.is_resizing()
            && self.platform_window.content_size() == self.viewport_size
        {
            self.resizing_window.set(false);
        }
    }

    /// Returns true if the platform window is currently resizing.
    pub fn is_window_resizing(&self) -> bool {
        self.resizing_window.get()
            || self.platform_window.is_resizing()
            || self.platform_window.content_size() != self.viewport_size
    }

    /// Produces a new frame and assigns it to `rendered_frame`. To actually show
    /// the contents of the new [`Scene`], use [`Self::present`].
    pub fn draw<'app>(&mut self, cx: &'app mut App) -> ArenaClearNeeded<'app> {
        profiling::scope!("wgpui: draw");
        // Set up the per-App arena for element allocation during this draw.
        // This ensures that multiple test Apps have isolated arenas.
        let _arena_scope = ElementArenaScope::enter(&cx.element_arena);

        #[cfg(feature = "flamegraph")]
        let frame_index = crate::open_frame_cpu_side(self.handle.id.as_u64());
        #[cfg(feature = "flamegraph")]
        self.flamegraph_open_frame.set(frame_index);
        #[cfg(feature = "flamegraph")]
        let _draw_span = crate::enter_span(
            crate::SpanName::Static("Window::draw"),
            crate::SpanCategory::WindowFrame,
            None,
        );

        self.apply_invalidations();

        // Slab-driven re-records (spec #94): the renderer discovers atlas
        // evictions under resident layers at frame start and posts them here.
        // Draining before draw_roots is what makes invalidation-before-draw
        // hold — the affected layer rebuilds (fresh tiles, fresh token) before
        // its span is recorded again, so stale tiles can never reach the GPU.
        if crate::scene_pack::slabs_enabled() {
            for layer_key in crate::platform::cross::slab_gpu::take_rerecord_requests() {
                if self.layers.contains_key(&layer_key) {
                    self.invalidator
                        .invalidate_layer(layer_key, Invalidation::all());
                } else {
                    self.slab_tokens.remove(&layer_key);
                }
            }
        }

        cx.entities.clear_accessed();
        debug_assert!(self.rendered_entity_stack.is_empty());
        self.invalidator.set_dirty(false);
        self.requested_autoscroll = None;

        // Restore the previously-used input handler.
        // Place it back into a None slot (left by a previous .take()) so that
        // cached paint_range indices in reuse_paint find the handler at the
        // expected position.
        if let Some(input_handler) = self.platform_window.take_input_handler() {
            if let Some(slot) = self
                .rendered_frame
                .input_handlers
                .iter_mut()
                .rev()
                .find(|h| h.is_none())
            {
                *slot = Some(input_handler);
            } else {
                self.rendered_frame.input_handlers.push(Some(input_handler));
            }
        }
        // Phase 5 (issue #61): consume an armed UI-tree capture request and
        // start recording, but only on the path that is actually about to
        // draw -- a skipped draw has no elements to record, so the request
        // is left armed for a future frame instead of being silently
        // consumed here. See `flamegraph_ui_capture::request_ui_tree_capture`.
        #[cfg(feature = "flamegraph")]
        let mut ui_tree_capture_guard = None;
        if !cx.mode.skip_drawing() {
            #[cfg(feature = "flamegraph")]
            {
                ui_tree_capture_guard = crate::maybe_begin_capture(self.handle.id.as_u64());
            }

            // Layers age by *draws that could have visited them*. Bumping this
            // for a skipped draw would evict a window's layers for not being
            // visited by a frame that never looked at anything.
            self.layer_frame = self.layer_frame.wrapping_add(1);
            self.draw_roots(cx);
            self.evict_stale_layers();

            // Validate mode: logs that occlusion is active. The full
            // double-render-and-diff is deferred — it needs the scene to be
            // finished (which happens below in `next_frame.finish`), and
            // re-running draw_roots requires resetting per-frame state. A
            // future change can save the finished scene, re-run draw_roots
            // with occlusion disabled, finish again, and diff the two scenes.
            if crate::occlusion::validate_enabled() {
                log::info!("occlusion validate mode active");
            }

            #[cfg(any(feature = "inspector", debug_assertions))]
            {
                let element_infos = std::mem::take(&mut self.next_frame.inspector_element_infos);
                if let Some(inspector) = &self.inspector {
                    inspector.update(cx, |inspector, _cx| {
                        inspector.set_element_infos(element_infos);
                    });
                }
            }
        }
        self.dirty_views.clear();
        self.invalidated_entities.clear();
        self.next_frame.window_active = self.active.get();

        // Register requested input handler with the platform window.
        // Use .take() instead of .pop() to preserve Vec length, so that cached
        // paint_range indices remain valid for reuse_paint on the next frame.
        // Search backwards to find the last Some entry, since reuse_paint may
        // have copied None slots from the previous frame. (Fixes #50456)
        if let Some(input_handler) = self
            .next_frame
            .input_handlers
            .iter_mut()
            .rev()
            .find_map(|h| h.take())
        {
            self.platform_window.set_input_handler(input_handler);
        }

        // #93: with persistent layout enabled, `end_frame` sweeps only the
        // nodes this draw didn't touch — every element still visits
        // `request_layout` unconditionally (see `instance.rs`'s module doc),
        // so a node untouched this frame is unambiguously gone, not merely
        // not-yet-reached. `WGPUI_PERSISTENT_LAYOUT=0` falls back to the
        // pre-#93 `clear()`, wiping and recreating the whole tree every frame.
        let layout_engine = self.layout_engine.as_mut().unwrap();
        if crate::taffy::persistent_layout_enabled() {
            layout_engine.end_frame();
        } else {
            layout_engine.clear();
        }
        self.text_system().finish_frame();
        self.next_frame.finish(&mut self.rendered_frame);

        // Phase 5 (issue #61): `next_frame.scene` is now sorted/finished for
        // this draw -- the exact "paint primitive list handed to the
        // renderer" the capture wants. Deep-copy it now, before the swap
        // below hands it to `present`/the renderer and the next frame
        // overwrites it.
        #[cfg(feature = "flamegraph")]
        if let Some(guard) = ui_tree_capture_guard.take() {
            crate::finish_capture(guard, &self.next_frame.scene, self.scale_factor());
        }

        self.invalidator.set_phase(DrawPhase::Focus);
        let previous_focus_path = self.rendered_frame.focus_path();
        let previous_window_active = self.rendered_frame.window_active;
        mem::swap(&mut self.rendered_frame, &mut self.next_frame);
        self.next_frame.clear();
        let current_focus_path = self.rendered_frame.focus_path();
        let current_window_active = self.rendered_frame.window_active;

        if previous_focus_path != current_focus_path
            || previous_window_active != current_window_active
        {
            if !previous_focus_path.is_empty() && current_focus_path.is_empty() {
                self.focus_lost_listeners
                    .clone()
                    .retain(&(), |listener| listener(self, cx));
            }

            let event = WindowFocusEvent {
                previous_focus_path: if previous_window_active {
                    previous_focus_path
                } else {
                    Default::default()
                },
                current_focus_path: if current_window_active {
                    current_focus_path
                } else {
                    Default::default()
                },
            };
            self.focus_listeners
                .clone()
                .retain(&(), |listener| listener(&event, self, cx));
        }

        debug_assert!(self.rendered_entity_stack.is_empty());
        self.record_entities_accessed(cx);
        self.reset_cursor_style(cx);
        self.window_invalidation = Invalidation::empty();
        self.invalidator.set_phase(DrawPhase::None);

        // Anything that invalidated while this draw was running — scrollbar
        // thumb state computed during prepaint, a view notifying a sibling
        // model, a smooth-scroll animation asking for its next frame from
        // prepaint, an observer chain kicked off from paint — was recorded
        // rather than applied. Now that the phase is `None` again and `cx` is
        // in hand, mark the window dirty so a frame arrives to consume the
        // entries already in `dirty_views`, hand the deferred window-scope axes
        // to that frame, and push the notifications so observers run.
        //
        // This must come after `set_phase(DrawPhase::None)`: the focus
        // listeners above run under `DrawPhase::Focus` and can themselves
        // invalidate, and that belongs in this same flush.
        self.invalidator.flush_deferred_invalidations(cx);

        self.needs_present.set(true);

        ArenaClearNeeded::new(&cx.element_arena)
    }

    fn record_entities_accessed(&mut self, cx: &mut App) {
        let mut entities_ref = cx.entities.accessed_entities.get_mut();
        let mut entities = mem::take(entities_ref.deref_mut());
        let handle = self.handle;
        cx.record_entities_accessed(
            handle,
            // Try moving window invalidator into the Window
            self.invalidator.clone(),
            &entities,
        );
        let mut entities_ref = cx.entities.accessed_entities.get_mut();
        mem::swap(&mut entities, entities_ref.deref_mut());
    }

    /// Turn everything recorded since the last draw into the per-draw state the
    /// element walk reads: window-scope axes, and the entity-scope dirty sets.
    fn apply_invalidations(&mut self) {
        profiling::scope!("wgpui: apply_invalidations");
        self.window_invalidation = self.invalidator.take_window_axes();

        // Layer-scope requests mark exactly the layer they name — no ancestor
        // walk, which is the property that kills the chatty-leaf problem.
        for (key, axes) in self.invalidator.take_layer_axes() {
            if let Some(layer) = self.layers.get_mut(&key) {
                layer.needs |= axes;
            }
        }

        // A window-scope request claims every layer's recorded output is stale.
        // `refresh_buffers` — window scope, `DISPLAY` only — deliberately does
        // not: there the pixels behind a surface quad advanced while the quad
        // referring to them did not, so compositing is exactly right.
        const LAYER_INVALIDATING: Invalidation = Invalidation::LAYOUT.union(Invalidation::HIT);
        if self.window_invalidation.intersects(LAYER_INVALIDATING) {
            for layer in self.layers.values_mut() {
                layer.needs |= self.window_invalidation;
            }
        }

        let mut views = self.invalidator.take_views();
        for entity in views.drain() {
            self.mark_view_dirty(entity);
            // Kept in full, unlike `dirty_views`, which only retains the
            // entities that turned out to be views on a dispatch path. Cached
            // views test their own dependency sets against this. Cleared at the
            // end of the draw rather than here, so it describes exactly the
            // invalidations this frame is answering.
            self.invalidated_entities.insert(entity);
        }
        self.invalidator.replace_views(views);
    }

    fn present(&self) {
        profiling::scope!("wgpui: present");
        #[cfg(feature = "flamegraph")]
        let _present_span = crate::enter_span(
            crate::SpanName::Static("Window::present"),
            crate::SpanCategory::WindowFrame,
            None,
        );

        self.platform_window.draw(&self.rendered_frame.scene);
        self.needs_present.set(false);
        profiling::finish_frame!();

        // GPU spans for this frame are left open (attached asynchronously via
        // `attach_gpu_spans` after query readback); only the CPU/background
        // sides are finalized here.
        #[cfg(feature = "flamegraph")]
        crate::close_frame_cpu_side(self.flamegraph_open_frame.take());
    }

    /// Present only the cached framebuffer (fast path - no compositor)
    fn present_framebuffer_only(&self) {
        profiling::scope!("wgpui: present_framebuffer_only");
        #[cfg(feature = "flamegraph")]
        let _present_span = crate::enter_span(
            crate::SpanName::Static("Window::present_framebuffer_only"),
            crate::SpanCategory::WindowFrame,
            None,
        );

        self.platform_window.present_framebuffer_only();
        self.needs_present.set(false);
        profiling::finish_frame!();
    }

    fn draw_roots(&mut self, cx: &mut App) {
        #[cfg(feature = "flamegraph")]
        let _draw_roots_span = crate::enter_span(
            crate::SpanName::Static("Window::draw_roots"),
            crate::SpanCategory::WindowFrame,
            None,
        );

        self.invalidator.set_phase(DrawPhase::Prepaint);
        self.tooltip_bounds.take();

        let viewport_width = self.viewport_size.width;
        #[cfg(any(feature = "inspector", debug_assertions))]
        let inspector_width = {
            if let Some(inspector) = &self.inspector {
                let width = inspector.read(cx).panel_width();
                if width.0 <= 0.0 {
                    rems(30.0).to_pixels(self.rem_size())
                } else {
                    width
                }
            } else {
                px(0.0)
            }
        };
        #[cfg(not(any(feature = "inspector", debug_assertions)))]
        let inspector_width = px(0.0);

        let root_size = {
            let mut size = self.viewport_size;
            if inspector_width > px(0.0) {
                size.width = (size.width - inspector_width).max(px(0.0));
            }
            size
        };

        // Layout all root elements.
        let mut root_element = self.root.as_ref().unwrap().clone().into_any();
        // `prepaint_as_root` split into its two halves so layout and prepaint can
        // be timed apart; the pair is exactly what that method does.
        {
            profiling::scope!("wgpui: layout");
            let _t = crate::render_stats::scope("frame: layout");
            root_element.layout_as_root(root_size.into(), self, cx);
        }

        let _inspector_element;
        let mut sorted_deferred_draws;
        let mut prompt_element;
        let mut active_drag_element;
        let mut tooltip_element;
        {
            profiling::scope!("wgpui: prepaint");
            let prepaint_timer = crate::render_stats::scope("frame: prepaint");
            root_element.prepaint_at(Point::default(), self, cx);

            #[cfg(any(feature = "inspector", debug_assertions))]
            {
                _inspector_element = self.prepaint_inspector(inspector_width, cx);
            }

            sorted_deferred_draws =
                (0..self.next_frame.deferred_draws.len()).collect::<SmallVec<[_; 8]>>();
            sorted_deferred_draws.sort_by_key(|ix| self.next_frame.deferred_draws[*ix].priority);
            self.prepaint_deferred_draws(&sorted_deferred_draws, cx);

            prompt_element = None;
            active_drag_element = None;
            tooltip_element = None;
            if let Some(prompt) = self.prompt.take() {
                let mut element = prompt.view.any_view().into_any();
                element.prepaint_as_root(Point::default(), root_size.into(), self, cx);
                prompt_element = Some(element);
                self.prompt = Some(prompt);
            } else if let Some(active_drag) = cx.active_drag.take() {
                let mut element = active_drag.view.clone().into_any();
                let offset = self.mouse_position() - active_drag.cursor_offset;
                element.prepaint_as_root(offset, AvailableSpace::min_size(), self, cx);
                active_drag_element = Some(element);
                cx.active_drag = Some(active_drag);
            } else {
                tooltip_element = self.prepaint_tooltip(cx);
            }

            self.mouse_hit_test = self.next_frame.hit_test(self.mouse_position, &self.layers);
            drop(prepaint_timer);
        }

        // Now actually paint the elements.
        profiling::scope!("wgpui: paint");
        let paint_timer = crate::render_stats::scope("frame: paint");
        self.invalidator.set_phase(DrawPhase::Paint);
        root_element.paint(self, cx);

        #[cfg(any(feature = "inspector", debug_assertions))]
        self.paint_inspector(_inspector_element, cx);

        self.paint_deferred_draws(&sorted_deferred_draws, cx);

        if let Some(mut prompt_element) = prompt_element {
            prompt_element.paint(self, cx);
        } else if let Some(mut drag_element) = active_drag_element {
            drag_element.paint(self, cx);
        } else if let Some(mut tooltip_element) = tooltip_element {
            tooltip_element.paint(self, cx);
        }

        #[cfg(any(feature = "inspector", debug_assertions))]
        self.paint_inspector_hitbox(cx);

        drop(paint_timer);
    }

    fn prepaint_tooltip(&mut self, cx: &mut App) -> Option<AnyElement> {
        // Use indexing instead of iteration to avoid borrowing self for the duration of the loop.
        for tooltip_request_index in (0..self.next_frame.tooltip_requests.len()).rev() {
            let Some(Some(tooltip_request)) = self
                .next_frame
                .tooltip_requests
                .get(tooltip_request_index)
                .cloned()
            else {
                log::error!("Unexpectedly absent TooltipRequest");
                continue;
            };
            let mut element = tooltip_request.tooltip.view.clone().into_any();
            let mouse_position = tooltip_request.tooltip.mouse_position;
            let tooltip_size = element.layout_as_root(AvailableSpace::min_size(), self, cx);

            let mut tooltip_bounds =
                Bounds::new(mouse_position + point(px(1.), px(1.)), tooltip_size);
            let window_bounds = Bounds {
                origin: Point::default(),
                size: self.viewport_size(),
            };

            if tooltip_bounds.right() > window_bounds.right() {
                let new_x = mouse_position.x - tooltip_bounds.size.width - px(1.);
                if new_x >= Pixels::ZERO {
                    tooltip_bounds.origin.x = new_x;
                } else {
                    tooltip_bounds.origin.x = cmp::max(
                        Pixels::ZERO,
                        tooltip_bounds.origin.x - tooltip_bounds.right() - window_bounds.right(),
                    );
                }
            }

            if tooltip_bounds.bottom() > window_bounds.bottom() {
                let new_y = mouse_position.y - tooltip_bounds.size.height - px(1.);
                if new_y >= Pixels::ZERO {
                    tooltip_bounds.origin.y = new_y;
                } else {
                    tooltip_bounds.origin.y = cmp::max(
                        Pixels::ZERO,
                        tooltip_bounds.origin.y - tooltip_bounds.bottom() - window_bounds.bottom(),
                    );
                }
            }

            // It's possible for an element to have an active tooltip while not being painted (e.g.
            // via the `visible_on_hover` method). Since mouse listeners are not active in this
            // case, instead update the tooltip's visibility here.
            let is_visible =
                (tooltip_request.tooltip.check_visible_and_update)(tooltip_bounds, self, cx);
            if !is_visible {
                continue;
            }

            self.with_absolute_element_offset(tooltip_bounds.origin, |window| {
                element.prepaint(window, cx)
            });

            self.tooltip_bounds = Some(TooltipBounds {
                id: tooltip_request.id,
                bounds: tooltip_bounds,
            });
            return Some(element);
        }
        None
    }

    fn prepaint_deferred_draws(&mut self, deferred_draw_indices: &[usize], cx: &mut App) {
        assert_eq!(self.element_id_stack.len(), 0);

        let mut deferred_draws = mem::take(&mut self.next_frame.deferred_draws);
        for &deferred_draw_ix in deferred_draw_indices {
            let deferred_draw = &mut deferred_draws[deferred_draw_ix];
            self.element_id_stack
                .clone_from(&deferred_draw.element_id_stack);
            self.text_style_stack
                .clone_from(&deferred_draw.text_style_stack);
            self.next_frame
                .dispatch_tree
                .set_active_node(deferred_draw.parent_node);

            let prepaint_start = self.prepaint_index();
            if let Some(element) = deferred_draw.element.as_mut() {
                self.with_rendered_view(deferred_draw.current_view, |window| {
                    window.with_absolute_element_offset(deferred_draw.absolute_offset, |window| {
                        element.prepaint(window, cx)
                    });
                })
            } else {
                self.reuse_prepaint(deferred_draw.prepaint_range.clone());
            }
            let prepaint_end = self.prepaint_index();
            deferred_draw.prepaint_range = prepaint_start..prepaint_end;
        }

        // Process any nested deferred draws that were added during prepaint.
        // A deferred element's child may itself contain deferred() elements,
        // which call defer_draw() during their prepaint.
        // Repeat until no new deferred draws are added.
        while !self.next_frame.deferred_draws.is_empty() {
            let nested = mem::take(&mut self.next_frame.deferred_draws);
            for mut deferred_draw in nested {
                self.element_id_stack
                    .clone_from(&deferred_draw.element_id_stack);
                self.text_style_stack
                    .clone_from(&deferred_draw.text_style_stack);
                self.next_frame
                    .dispatch_tree
                    .set_active_node(deferred_draw.parent_node);

                let prepaint_start = self.prepaint_index();
                if let Some(element) = deferred_draw.element.as_mut() {
                    self.with_rendered_view(deferred_draw.current_view, |window| {
                        window
                            .with_absolute_element_offset(deferred_draw.absolute_offset, |window| {
                                element.prepaint(window, cx)
                            })
                    });
                }
                let prepaint_end = self.prepaint_index();
                deferred_draw.prepaint_range = prepaint_start..prepaint_end;
                deferred_draws.push(deferred_draw);
            }
        }

        self.next_frame.deferred_draws = deferred_draws;
        self.element_id_stack.clear();
        self.text_style_stack.clear();
    }

    fn paint_deferred_draws(&mut self, deferred_draw_indices: &[usize], cx: &mut App) {
        assert_eq!(self.element_id_stack.len(), 0);

        // Deferred draws are overlays (tooltips, popovers, drag images) and must sort above the
        // whole main scene. Raise the order floor so they do — this also keeps a deferred
        // backdrop's order from falling inside a content-filter order range left by the main scene.
        self.next_frame.scene.raise_order_floor();

        let mut deferred_draws = mem::take(&mut self.next_frame.deferred_draws);
        for &deferred_draw_ix in deferred_draw_indices {
            let mut deferred_draw = &mut deferred_draws[deferred_draw_ix];
            self.element_id_stack
                .clone_from(&deferred_draw.element_id_stack);
            self.next_frame
                .dispatch_tree
                .set_active_node(deferred_draw.parent_node);

            let paint_start = self.paint_index();
            if let Some(element) = deferred_draw.element.as_mut() {
                self.with_rendered_view(deferred_draw.current_view, |window| {
                    element.paint(window, cx);
                })
            } else {
                self.reuse_paint(deferred_draw.paint_range.clone());
            }
            let paint_end = self.paint_index();
            deferred_draw.paint_range = paint_start..paint_end;
        }

        // Paint any nested deferred draws that were added during prepaint
        // but are not covered by the original indices.
        for index in deferred_draw_indices.len()..deferred_draws.len() {
            let mut deferred_draw = &mut deferred_draws[index];
            self.element_id_stack
                .clone_from(&deferred_draw.element_id_stack);
            self.next_frame
                .dispatch_tree
                .set_active_node(deferred_draw.parent_node);

            let paint_start = self.paint_index();
            if let Some(element) = deferred_draw.element.as_mut() {
                self.with_rendered_view(deferred_draw.current_view, |window| {
                    element.paint(window, cx);
                })
            } else {
                self.reuse_paint(deferred_draw.paint_range.clone());
            }
            let paint_end = self.paint_index();
            deferred_draw.paint_range = paint_start..paint_end;
        }

        self.next_frame.deferred_draws = deferred_draws;
        self.element_id_stack.clear();
    }

    pub(crate) fn prepaint_index(&self) -> PrepaintStateIndex {
        PrepaintStateIndex {
            hitboxes_index: self.next_frame.hitboxes.len(),
            effects_index: self.next_frame.effects.len(),
            tooltips_index: self.next_frame.tooltip_requests.len(),
            deferred_draws_index: self.next_frame.deferred_draws.len(),
            dispatch_tree_index: self.next_frame.dispatch_tree.len(),
            accessed_element_states_index: self.next_frame.accessed_element_states.len(),
            line_layout_index: self.text_system.layout_index(),
        }
    }

    /// Check that a cached view's stored ranges still fit inside the arrays they
    /// index, returning the name of the first array they don't fit.
    ///
    /// Both `PrepaintStateIndex` and `PaintIndex` are bundles of **absolute
    /// offsets** into per-frame arrays, recorded during an earlier frame and
    /// replayed against `rendered_frame` (and the text system's previous-frame
    /// lists) on a later one. Nothing in the type system ties a stored range to
    /// the array it came from, so a range that has aged — because the view went
    /// unpainted for a frame, because a retryable prepaint truncated the arrays,
    /// or for any reason not yet enumerated — will slice out of bounds and take
    /// the process down.
    ///
    /// Callers must treat a `Some(_)` result as a cache miss and rebuild. That
    /// turns every such case into a slower frame instead of a panic, without
    /// needing to know in advance what made the range stale.
    ///
    /// Both ranges are validated together at prepaint time on purpose: once
    /// prepaint has committed to reusing, paint has no rebuild path left, so a
    /// bad paint range discovered later would be unrecoverable.
    /// Returns `(array name, stored end offset, actual length)` on failure.
    pub(crate) fn invalid_reuse_range(
        &self,
        prepaint: &Range<PrepaintStateIndex>,
        paint: &Range<PaintIndex>,
    ) -> Option<(&'static str, usize, usize)> {
        let frame = &self.rendered_frame;
        let layouts = self.text_system.previous_frame_layout_extent();

        // Every offset in both index types, paired with the array it slices.
        //
        // Listed exhaustively and checked uniformly on purpose: an earlier
        // version enumerated these by hand and silently omitted two of them
        // (`accessed_element_states` and `wrapped_lines_index` on the paint
        // side), which left `reuse_paint` free to slice out of bounds on
        // exactly the fields nobody had thought about. If a field is added to
        // `PrepaintStateIndex` or `PaintIndex`, add it here.
        let checks: [(&'static str, usize, usize, usize); 16] = [
            // (name, start, end, length of the array it indexes)
            (
                "prepaint hitboxes",
                prepaint.start.hitboxes_index,
                prepaint.end.hitboxes_index,
                frame.hitboxes.len(),
            ),
            (
                "prepaint effects",
                prepaint.start.effects_index,
                prepaint.end.effects_index,
                frame.effects.len(),
            ),
            (
                "prepaint tooltip_requests",
                prepaint.start.tooltips_index,
                prepaint.end.tooltips_index,
                frame.tooltip_requests.len(),
            ),
            (
                "prepaint deferred_draws",
                prepaint.start.deferred_draws_index,
                prepaint.end.deferred_draws_index,
                frame.deferred_draws.len(),
            ),
            (
                "prepaint dispatch_tree",
                prepaint.start.dispatch_tree_index,
                prepaint.end.dispatch_tree_index,
                frame.dispatch_tree.len(),
            ),
            (
                "prepaint accessed_element_states",
                prepaint.start.accessed_element_states_index,
                prepaint.end.accessed_element_states_index,
                frame.accessed_element_states.len(),
            ),
            (
                "prepaint line_layouts",
                prepaint.start.line_layout_index.lines_index,
                prepaint.end.line_layout_index.lines_index,
                layouts.lines_index,
            ),
            (
                "prepaint wrapped_line_layouts",
                prepaint.start.line_layout_index.wrapped_lines_index,
                prepaint.end.line_layout_index.wrapped_lines_index,
                layouts.wrapped_lines_index,
            ),
            (
                "paint scene",
                paint.start.scene_index,
                paint.end.scene_index,
                frame.scene.len(),
            ),
            (
                "paint mouse_listeners",
                paint.start.mouse_listeners_index,
                paint.end.mouse_listeners_index,
                frame.mouse_listeners.len(),
            ),
            (
                "paint input_handlers",
                paint.start.input_handlers_index,
                paint.end.input_handlers_index,
                frame.input_handlers.len(),
            ),
            (
                "paint cursor_styles",
                paint.start.cursor_styles_index,
                paint.end.cursor_styles_index,
                frame.cursor_styles.len(),
            ),
            (
                "paint accessed_element_states",
                paint.start.accessed_element_states_index,
                paint.end.accessed_element_states_index,
                frame.accessed_element_states.len(),
            ),
            (
                "paint tab_stops",
                paint.start.tab_handle_index,
                paint.end.tab_handle_index,
                frame.tab_stops.paint_index(),
            ),
            (
                "paint line_layouts",
                paint.start.line_layout_index.lines_index,
                paint.end.line_layout_index.lines_index,
                layouts.lines_index,
            ),
            (
                "paint wrapped_line_layouts",
                paint.start.line_layout_index.wrapped_lines_index,
                paint.end.line_layout_index.wrapped_lines_index,
                layouts.wrapped_lines_index,
            ),
        ];

        for (name, start, end, len) in checks {
            // `start > end` panics just as hard as `end > len`, and both mean
            // the same thing: this range no longer describes the array.
            if start > end {
                return Some((name, start, end));
            }
            if end > len {
                return Some((name, end, len));
            }
        }

        None
    }

    /// Record an [`Element::on_frame`](crate::Element::on_frame) effect on this
    /// frame, so a cached ancestor replaying next frame can re-run it.
    ///
    /// Called by the element as it runs its own effect; the recording is what
    /// makes the effect survive the element not being built at all.
    pub(crate) fn record_frame_effect(
        &mut self,
        callback: FrameEffectCallback,
        geometry: ElementGeometry,
    ) {
        // Registering from anywhere else would put an entry in the frame's
        // effect list that no `on_frame` walk produced, and the ranges a cached
        // view records around its subtree would no longer describe that
        // subtree's effects.
        self.invalidator.debug_assert_effects();
        self.next_frame
            .effects
            .push(FrameEffect { callback, geometry });
    }

    /// Re-run the effects a cached subtree registered when it last rendered,
    /// and re-record them so the range stays valid for the frame after this
    /// one.
    ///
    /// This is what makes caching safe: an element inside a replayed view never
    /// runs, so any side effect living in its closures would silently stop. It
    /// used to be handled by re-rendering the view and calling `on_frame` on
    /// the fresh tree, which ran `render` and a full `layout_as_root` on every
    /// cache hit — the two things the cache exists to skip. Replaying the
    /// recorded effects costs one `Rc` clone and one call each.
    pub(crate) fn replay_frame_effects(&mut self, range: &Range<PrepaintStateIndex>, cx: &mut App) {
        let range = range.start.effects_index..range.end.effects_index;
        if range.is_empty() {
            return;
        }
        let effects = self.rendered_frame.effects[range].to_vec();
        self.next_frame.effects.extend(effects.iter().cloned());

        let phase = self.invalidator.draw_phase();
        self.invalidator.set_phase(DrawPhase::Effects);
        for effect in effects {
            (effect.callback)(effect.geometry, self, cx);
        }
        self.invalidator.set_phase(phase);
    }

    pub(crate) fn reuse_prepaint(&mut self, range: Range<PrepaintStateIndex>) {
        self.next_frame.hitboxes.extend(
            self.rendered_frame.hitboxes[range.start.hitboxes_index..range.end.hitboxes_index]
                .iter()
                .cloned(),
        );
        self.next_frame.tooltip_requests.extend(
            self.rendered_frame.tooltip_requests
                [range.start.tooltips_index..range.end.tooltips_index]
                .iter_mut()
                .map(|request| request.take()),
        );
        self.next_frame.accessed_element_states.extend(
            self.rendered_frame.accessed_element_states[range.start.accessed_element_states_index
                ..range.end.accessed_element_states_index]
                .iter()
                .map(|(id, type_id)| (id.clone(), *type_id)),
        );
        self.text_system
            .reuse_layouts(range.start.line_layout_index..range.end.line_layout_index);

        let reused_subtree = self.next_frame.dispatch_tree.reuse_subtree(
            range.start.dispatch_tree_index..range.end.dispatch_tree_index,
            &mut self.rendered_frame.dispatch_tree,
            self.focus,
        );

        if reused_subtree.contains_focus() {
            self.next_frame.focus = self.focus;
        }

        self.next_frame.deferred_draws.extend(
            self.rendered_frame.deferred_draws
                [range.start.deferred_draws_index..range.end.deferred_draws_index]
                .iter()
                .map(|deferred_draw| DeferredDraw {
                    current_view: deferred_draw.current_view,
                    parent_node: reused_subtree.refresh_node_id(deferred_draw.parent_node),
                    element_id_stack: deferred_draw.element_id_stack.clone(),
                    text_style_stack: deferred_draw.text_style_stack.clone(),
                    priority: deferred_draw.priority,
                    element: None,
                    absolute_offset: deferred_draw.absolute_offset,
                    prepaint_range: deferred_draw.prepaint_range.clone(),
                    paint_range: deferred_draw.paint_range.clone(),
                }),
        );
    }

    pub(crate) fn paint_index(&self) -> PaintIndex {
        PaintIndex {
            scene_index: self.next_frame.scene.len(),
            mouse_listeners_index: self.next_frame.mouse_listeners.len(),
            input_handlers_index: self.next_frame.input_handlers.len(),
            cursor_styles_index: self.next_frame.cursor_styles.len(),
            accessed_element_states_index: self.next_frame.accessed_element_states.len(),
            tab_handle_index: self.next_frame.tab_stops.paint_index(),
            line_layout_index: self.text_system.layout_index(),
        }
    }

    pub(crate) fn reuse_paint(&mut self, range: Range<PaintIndex>) {
        self.reuse_paint_except_scene(&range);
        self.next_frame.scene.replay(
            range.start.scene_index..range.end.scene_index,
            &self.rendered_frame.scene,
        );
    }

    /// Replay just the scene half of a recorded paint range.
    ///
    /// The fallback for a reuse that expected a retained layer and did not find
    /// one.
    pub(crate) fn replay_scene_range(&mut self, range: &Range<PaintIndex>) {
        self.next_frame.scene.replay(
            range.start.scene_index..range.end.scene_index,
            &self.rendered_frame.scene,
        );
    }

    /// Everything [`Self::reuse_paint`] replays except the scene.
    ///
    /// Split out because a retained layer is a better source for the scene half
    /// than a recorded index range: it re-emits primitives with the draw orders
    /// they already have, where `Scene::replay` pushes each one back through
    /// `insert_primitive` and re-derives its order from a `BoundsTree` — the
    /// per-primitive cost the whole retained model exists to remove. The other
    /// arrays still travel by range until #97.
    pub(crate) fn reuse_paint_except_scene(&mut self, range: &Range<PaintIndex>) {
        self.next_frame.cursor_styles.extend(
            self.rendered_frame.cursor_styles
                [range.start.cursor_styles_index..range.end.cursor_styles_index]
                .iter()
                .cloned(),
        );
        self.next_frame.input_handlers.extend(
            self.rendered_frame.input_handlers
                [range.start.input_handlers_index..range.end.input_handlers_index]
                .iter_mut()
                .map(|handler| handler.take()),
        );
        self.next_frame.mouse_listeners.extend(
            self.rendered_frame.mouse_listeners
                [range.start.mouse_listeners_index..range.end.mouse_listeners_index]
                .iter_mut()
                .map(|listener| listener.take()),
        );
        self.next_frame.accessed_element_states.extend(
            self.rendered_frame.accessed_element_states[range.start.accessed_element_states_index
                ..range.end.accessed_element_states_index]
                .iter()
                .map(|(id, type_id)| (id.clone(), *type_id)),
        );
        self.next_frame.tab_stops.replay(
            &self.rendered_frame.tab_stops.insertion_history
                [range.start.tab_handle_index..range.end.tab_handle_index],
        );

        self.text_system.reuse_layouts(
            range.start.line_layout_index.clone()..range.end.line_layout_index.clone(),
        );
    }

    /// Push a text style onto the stack, and call a function with that style active.
    /// Use [`Window::text_style`] to get the current, combined text style. This method
    /// should only be called as part of element drawing.
    // This function is called in a highly recursive manner in editor
    // prepainting, make sure its inlined to reduce the stack burden
    #[inline]
    pub fn with_text_style<F, R>(&mut self, style: Option<TextStyleRefinement>, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        self.invalidator.debug_assert_paint_or_prepaint();
        if let Some(style) = style {
            self.text_style_stack.push(style);
            let result = f(self);
            self.text_style_stack.pop();
            result
        } else {
            f(self)
        }
    }

    /// Updates the cursor style at the platform level. This method should only be called
    /// during the paint phase of element drawing.
    pub fn set_cursor_style(&mut self, style: CursorStyle, hitbox: &Hitbox) {
        self.invalidator.debug_assert_paint();
        self.next_frame.cursor_styles.push(CursorStyleRequest {
            hitbox_id: Some(hitbox.id),
            style,
        });
    }

    /// Updates the cursor style for the entire window at the platform level. A cursor
    /// style using this method will have precedence over any cursor style set using
    /// `set_cursor_style`. This method should only be called during the paint
    /// phase of element drawing.
    pub fn set_window_cursor_style(&mut self, style: CursorStyle) {
        self.invalidator.debug_assert_paint();
        self.next_frame.cursor_styles.push(CursorStyleRequest {
            hitbox_id: None,
            style,
        })
    }

    /// Sets a tooltip to be rendered for the upcoming frame. This method should only be called
    /// during the paint phase of element drawing.
    pub fn set_tooltip(&mut self, tooltip: AnyTooltip) -> TooltipId {
        self.invalidator.debug_assert_prepaint();
        let id = TooltipId(post_inc(&mut self.next_tooltip_id.0));
        self.next_frame
            .tooltip_requests
            .push(Some(TooltipRequest { id, tooltip }));
        id
    }

    /// Invoke the given function with the given content mask after intersecting it
    /// with the current mask. This method should only be called during element drawing.
    // This function is called in a highly recursive manner in editor
    // prepainting, make sure its inlined to reduce the stack burden
    #[inline]
    pub fn with_content_mask<R>(
        &mut self,
        mask: Option<ContentMask<Pixels>>,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        self.invalidator.debug_assert_paint_or_prepaint();
        if let Some(mask) = mask {
            let mask = mask.intersect(&self.content_mask());
            self.content_mask_stack.push(mask);
            let result = f(self);
            self.content_mask_stack.pop();
            result
        } else {
            f(self)
        }
    }

    /// Updates the global element offset relative to the current offset. This is used to implement
    /// scrolling. This method should only be called during the prepaint phase of element drawing.
    pub fn with_element_offset<R>(
        &mut self,
        offset: Point<Pixels>,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        self.invalidator.debug_assert_prepaint();

        if offset.is_zero() {
            return f(self);
        };

        let abs_offset = self.element_offset() + offset;
        self.with_absolute_element_offset(abs_offset, f)
    }

    /// Updates the global element offset based on the given offset. This is used to implement
    /// drag handles and other manual painting of elements. This method should only be called during
    /// the prepaint phase of element drawing.
    pub fn with_absolute_element_offset<R>(
        &mut self,
        offset: Point<Pixels>,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        self.invalidator.debug_assert_prepaint();
        self.element_offset_stack.push(offset);
        let result = f(self);
        self.element_offset_stack.pop();
        result
    }

    pub(crate) fn with_element_opacity<R>(
        &mut self,
        opacity: Option<f32>,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        self.invalidator.debug_assert_paint_or_prepaint();

        let Some(opacity) = opacity else {
            return f(self);
        };

        let previous_opacity = self.element_opacity;
        self.element_opacity = previous_opacity * opacity;
        let result = f(self);
        self.element_opacity = previous_opacity;
        result
    }

    /// Perform prepaint on child elements in a "retryable" manner, so that any side effects
    /// of prepaints can be discarded before prepainting again. This is used to support autoscroll
    /// where we need to prepaint children to detect the autoscroll bounds, then adjust the
    /// element offset and prepaint again. See [`crate::List`] for an example. This method should only be
    /// called during the prepaint phase of element drawing.
    pub fn transact<T, U>(&mut self, f: impl FnOnce(&mut Self) -> Result<T, U>) -> Result<T, U> {
        self.invalidator.debug_assert_prepaint();
        let index = self.prepaint_index();
        let result = f(self);
        if result.is_err() {
            self.next_frame.hitboxes.truncate(index.hitboxes_index);
            self.next_frame
                .tooltip_requests
                .truncate(index.tooltips_index);
            self.next_frame
                .deferred_draws
                .truncate(index.deferred_draws_index);
            self.next_frame
                .dispatch_tree
                .truncate(index.dispatch_tree_index);
            self.next_frame
                .accessed_element_states
                .truncate(index.accessed_element_states_index);
            self.text_system.truncate_layouts(index.line_layout_index);
        }
        result
    }

    /// When you call this method during [`Element::prepaint`], containing elements will attempt to
    /// scroll to cause the specified bounds to become visible. When they decide to autoscroll, they will call
    /// [`Element::prepaint`] again with a new set of bounds. See [`crate::List`] for an example of an element
    /// that supports this method being called on the elements it contains. This method should only be
    /// called during the prepaint phase of element drawing.
    pub fn request_autoscroll(&mut self, bounds: Bounds<Pixels>) {
        self.invalidator.debug_assert_prepaint();
        self.requested_autoscroll = Some(bounds);
    }

    /// This method can be called from a containing element such as [`crate::List`] to support the autoscroll behavior
    /// described in [`Self::request_autoscroll`].
    pub fn take_autoscroll(&mut self) -> Option<Bounds<Pixels>> {
        self.invalidator.debug_assert_prepaint();
        self.requested_autoscroll.take()
    }

    /// Asynchronously load an asset, if the asset hasn't finished loading this will return None.
    /// Your view will be re-drawn once the asset has finished loading.
    ///
    /// Note that the multiple calls to this method will only result in one `Asset::load` call at a
    /// time.
    pub fn use_asset<A: Asset>(&mut self, source: &A::Source, cx: &mut App) -> Option<A::Output> {
        let (task, is_first) = cx.fetch_asset::<A>(source);
        task.clone().now_or_never().or_else(|| {
            if is_first {
                let entity_id = self.current_view();
                self.spawn(cx, {
                    let task = task.clone();
                    async move |cx| {
                        task.await;

                        cx.on_next_frame(move |_, cx| {
                            cx.notify(entity_id);
                        });
                    }
                })
                .detach();
            }

            None
        })
    }

    /// Asynchronously load an asset, if the asset hasn't finished loading or doesn't exist this will return None.
    /// Your view will not be re-drawn once the asset has finished loading.
    ///
    /// Note that the multiple calls to this method will only result in one `Asset::load` call at a
    /// time.
    pub fn get_asset<A: Asset>(&mut self, source: &A::Source, cx: &mut App) -> Option<A::Output> {
        let (task, _) = cx.fetch_asset::<A>(source);
        task.now_or_never()
    }
    /// Obtain the current element offset. This method should only be called during the
    /// prepaint phase of element drawing.
    pub fn element_offset(&self) -> Point<Pixels> {
        self.invalidator.debug_assert_prepaint();
        self.element_offset_stack
            .last()
            .copied()
            .unwrap_or_default()
    }

    /// Obtain the current element opacity. This method should only be called during the
    /// prepaint phase of element drawing.
    #[inline]
    pub(crate) fn element_opacity(&self) -> f32 {
        self.invalidator.debug_assert_paint_or_prepaint();
        self.element_opacity
    }

    /// Obtain the current content mask. This method should only be called during element drawing.
    pub fn content_mask(&self) -> ContentMask<Pixels> {
        self.invalidator.debug_assert_paint_or_prepaint();
        self.content_mask_stack
            .last()
            .cloned()
            .unwrap_or_else(|| ContentMask {
                bounds: Bounds {
                    origin: Point::default(),
                    size: self.viewport_size,
                },
            })
    }

    /// Provide elements in the called function with a new namespace in which their identifiers must be unique.
    /// This can be used within a custom element to distinguish multiple sets of child elements.
    pub fn with_element_namespace<R>(
        &mut self,
        element_id: impl Into<ElementId>,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        self.element_id_stack.push(element_id.into());
        let result = f(self);
        self.element_id_stack.pop();
        result
    }

    /// Use a piece of state that exists as long this element is being rendered in consecutive frames.
    pub fn use_keyed_state<S: 'static>(
        &mut self,
        key: impl Into<ElementId>,
        cx: &mut App,
        init: impl FnOnce(&mut Self, &mut Context<S>) -> S,
    ) -> Entity<S> {
        let current_view = self.current_view();
        self.with_global_id(key.into(), |global_id, window| {
            window.with_element_state(global_id, |state: Option<Entity<S>>, window| {
                if let Some(state) = state {
                    (state.clone(), state)
                } else {
                    let new_state = cx.new(|cx| init(window, cx));
                    cx.observe(&new_state, move |_, cx| {
                        cx.notify(current_view);
                    })
                    .detach();
                    (new_state.clone(), new_state)
                }
            })
        })
    }

    /// Use a piece of state that exists as long this element is being rendered in consecutive frames, without needing to specify a key
    ///
    /// NOTE: This method uses the location of the caller to generate an ID for this state.
    ///       If this is not sufficient to identify your state (e.g. you're rendering a list item),
    ///       you can provide a custom ElementID using the `use_keyed_state` method.
    #[track_caller]
    pub fn use_state<S: 'static>(
        &mut self,
        cx: &mut App,
        init: impl FnOnce(&mut Self, &mut Context<S>) -> S,
    ) -> Entity<S> {
        self.use_keyed_state(
            ElementId::CodeLocation(*core::panic::Location::caller()),
            cx,
            init,
        )
    }

    /// Updates or initializes state for an element with the given id that lives across multiple
    /// frames. If an element with this ID existed in the rendered frame, its state will be passed
    /// to the given closure. The state returned by the closure will be stored so it can be referenced
    /// when drawing the next frame. This method should only be called as part of element drawing.
    pub fn with_element_state<S, R>(
        &mut self,
        global_id: &GlobalElementId,
        f: impl FnOnce(Option<S>, &mut Self) -> (R, S),
    ) -> R
    where
        S: 'static,
    {
        self.invalidator.debug_assert_paint_or_prepaint();

        let key = (global_id.clone(), TypeId::of::<S>());
        self.next_frame.accessed_element_states.push(key.clone());

        if let Some(any) = self
            .next_frame
            .element_states
            .remove(&key)
            .or_else(|| self.rendered_frame.element_states.remove(&key))
        {
            let ElementStateBox {
                inner,
                #[cfg(debug_assertions)]
                type_name,
            } = any;
            // Using the extra inner option to avoid needing to reallocate a new box.
            let mut state_box = inner
                .downcast::<Option<S>>()
                .map_err(|_| {
                    #[cfg(debug_assertions)]
                    {
                        anyhow::anyhow!(
                            "invalid element state type for id, requested {:?}, actual: {:?}",
                            std::any::type_name::<S>(),
                            type_name
                        )
                    }

                    #[cfg(not(debug_assertions))]
                    {
                        anyhow::anyhow!(
                            "invalid element state type for id, requested {:?}",
                            std::any::type_name::<S>(),
                        )
                    }
                })
                .unwrap();

            let state = state_box.take().expect(
                "reentrant call to with_element_state for the same state type and element id",
            );
            let (result, state) = f(Some(state), self);
            state_box.replace(state);
            self.next_frame.element_states.insert(
                key,
                ElementStateBox {
                    inner: state_box,
                    #[cfg(debug_assertions)]
                    type_name,
                },
            );
            result
        } else {
            let (result, state) = f(None, self);
            self.next_frame.element_states.insert(
                key,
                ElementStateBox {
                    inner: Box::new(Some(state)),
                    #[cfg(debug_assertions)]
                    type_name: std::any::type_name::<S>(),
                },
            );
            result
        }
    }

    /// A variant of `with_element_state` that allows the element's id to be optional. This is a convenience
    /// method for elements where the element id may or may not be assigned. Prefer using `with_element_state`
    /// when the element is guaranteed to have an id.
    ///
    /// The first option means 'no ID provided'
    /// The second option means 'not yet initialized'
    pub fn with_optional_element_state<S, R>(
        &mut self,
        global_id: Option<&GlobalElementId>,
        f: impl FnOnce(Option<Option<S>>, &mut Self) -> (R, Option<S>),
    ) -> R
    where
        S: 'static,
    {
        self.invalidator.debug_assert_paint_or_prepaint();

        if let Some(global_id) = global_id {
            self.with_element_state(global_id, |state, cx| {
                let (result, state) = f(Some(state), cx);
                let state =
                    state.expect("you must return some state when you pass some element id");
                (result, state)
            })
        } else {
            let (result, state) = f(None, self);
            debug_assert!(
                state.is_none(),
                "you must not return an element state when passing None for the global id"
            );
            result
        }
    }

    /// Executes the given closure within the context of a tab group.
    #[inline]
    pub fn with_tab_group<R>(&mut self, index: Option<isize>, f: impl FnOnce(&mut Self) -> R) -> R {
        if let Some(index) = index {
            self.next_frame.tab_stops.begin_group(index);
            let result = f(self);
            self.next_frame.tab_stops.end_group();
            result
        } else {
            f(self)
        }
    }

    /// Defers the drawing of the given element, scheduling it to be painted on top of the currently-drawn tree
    /// at a later time. The `priority` parameter determines the drawing order relative to other deferred elements,
    /// with higher values being drawn on top.
    ///
    /// This method should only be called as part of the prepaint phase of element drawing.
    pub fn defer_draw(
        &mut self,
        element: AnyElement,
        absolute_offset: Point<Pixels>,
        priority: usize,
    ) {
        self.invalidator.debug_assert_prepaint();
        let parent_node = self.next_frame.dispatch_tree.active_node_id().unwrap();
        self.next_frame.deferred_draws.push(DeferredDraw {
            current_view: self.current_view(),
            parent_node,
            element_id_stack: self.element_id_stack.clone(),
            text_style_stack: self.text_style_stack.clone(),
            priority,
            element: Some(element),
            absolute_offset,
            prepaint_range: PrepaintStateIndex::default()..PrepaintStateIndex::default(),
            paint_range: PaintIndex::default()..PaintIndex::default(),
        });
    }

    /// The shared composite predicate for one retained layer: every condition
    /// both [`Self::with_retained_layer`]'s paint-time decision and a buffered
    /// element's prepaint-time prediction (#96) must agree on.
    ///
    /// Cache-key equality is deliberately NOT here: the paint-time decision
    /// branches on *how* the key differs (identical → composite, origin-only →
    /// transform-only move), so the caller adds the comparison it needs. The
    /// buffer prediction, which only composites, adds full equality.
    ///
    /// #96: an overscroll buffer (non-zero margin) exempts the pointer
    /// conditions. Scrolling happens under the cursor by definition, and
    /// the shifted composite is precisely the frame a re-render must not
    /// happen on; hover styling inside the buffer goes stale between
    /// refills (half a margin of scroll) by design.
    fn layer_reuse_conditions(
        &self,
        layer: &Layer,
        content_key: Option<u64>,
        mouse_inside: bool,
        view_rebuilt: bool,
    ) -> bool {
        let pointer_exempt = layer.policy.overdraw_margin != size(px(0.), px(0.));
        layer.content_key == content_key
            && layer.has_content()
            && layer.needs.is_empty()
            && !layer.deferred_dirty
            && (pointer_exempt || (!layer.had_mouse && !mouse_inside))
            && !view_rebuilt
            && self.view_cache_available()
            && !self.accessed_entity_invalidated(&layer.accessed_entities)
            // The paint range is an absolute offset into last frame's
            // arrays. It is validated for the same reason
            // `AnyView::cached` validates its own: nothing in the type
            // system ties a stored range to the array it came from, and a
            // range that has aged slices out of bounds and takes the
            // process down. A stale range is a re-render, not a crash.
            && self
                .invalid_reuse_range(
                    &(PrepaintStateIndex::default()..PrepaintStateIndex::default()),
                    &layer.paint_range,
                )
                .is_none()
    }

    /// Paint into the retained layer named by `global_id`, or composite last
    /// frame's primitives for it and skip painting entirely.
    ///
    /// Returns `Some(_)` when `f` ran, `None` when the layer composited. A
    /// caller that needs to know which happened — because it also holds
    /// index-range state for the old replay path — should branch on that.
    ///
    /// # When a layer composites
    ///
    /// Never on the first sight of a key, and afterwards only when *every* one
    /// of these holds:
    ///
    /// - the view the layer belongs to was not itself invalidated this frame.
    ///   A notified view re-runs `render` and produces a *fresh description*,
    ///   which this phase has no way to compare against the one the layer was
    ///   built from — reconciliation is #92. Until then, a rebuilt description
    ///   has to be assumed different, or a panel whose data changed would
    ///   composite last frame's pixels;
    /// - no entity it read while rendering has been invalidated — the same
    ///   dependency test cached views use, which is what makes notifying a model
    ///   reach the layers that display it;
    /// - its bounds, content mask, opacity and scale factor are unchanged;
    /// - its transform continues to describe the layer's local coordinate
    ///   space. Hitboxes are recorded in that space and hit testing inverts the
    ///   transform before applying the usual blocking rules;
    /// - the pointer is outside it, and was outside it last frame. Hover state
    ///   is read during paint and is not an entity, so nothing invalidates a
    ///   layer when the pointer crosses it. Re-rendering under the pointer is
    ///   cheaper than auditing every style for hover-sensitivity, and wrong
    ///   hover is a defect users notice immediately;
    /// - no window-scope invalidation claims the recorded geometry is stale.
    ///
    /// The list is deliberately conservative: every entry is a way a layer
    /// could otherwise show last frame's pixels for this frame's state.
    pub(crate) fn with_retained_layer<R>(
        &mut self,
        global_id: &GlobalElementId,
        bounds: Bounds<Pixels>,
        policy: LayerPolicy,
        content_key: Option<u64>,
        cx: &mut App,
        f: impl FnOnce(&mut Window, &mut App) -> R,
    ) -> Option<R> {
        if !crate::layer::layers_enabled() {
            return Some(f(self, cx));
        }

        let key = LayerKey::from_global_element_id(global_id);
        let cache_key = LayerCacheKey {
            bounds,
            content_mask: self.content_mask(),
            opacity: self.element_opacity,
            scale_factor: self.scale_factor(),
        };
        let mouse_inside = bounds.contains(&self.mouse_position);
        // A rebuilt view means a rebuilt description, and nothing here can tell
        // a rebuilt-but-identical description from a rebuilt-and-different one
        // — that is reconciliation, and it is #92. `dirty_views` is the right
        // set to ask rather than the raw notified set, because it has already
        // been walked up the dispatch tree: a notified descendant marks this
        // view too, so a layer cannot composite over a child that changed.
        //
        // A `layer_keyed` caller has declared what the content is a function
        // of, so a notify that leaves the key alone is a notify this subtree
        // does not care about. That is the whole reason the key exists: the
        // view most worth caching under is the one notified every frame for a
        // reason unrelated to the subtree.
        let view_rebuilt = content_key.is_none() && self.dirty_views.contains(&self.current_view());

        let layer_conditions_hold = self.layers.get(&key).is_some_and(|layer| {
            self.layer_reuse_conditions(layer, content_key, mouse_inside, view_rebuilt)
        });
        let reusable_frame = layer_conditions_hold;

        let composite = reusable_frame
            && self
                .layers
                .get(&key)
                .is_some_and(|layer| layer.cache_key == cache_key);

        // Spec #94's headline: a TRANSFORM-only change should cost one
        // uniform write, not a re-record plus a full instance re-upload.
        // Slabs must be live because the fast path has no legacy fallback —
        // replaying retained primitives at their recorded positions would
        // draw the layer where it was, not where it is.
        let transform_only_move = reusable_frame
            && crate::scene_pack::slabs_enabled()
            && self.layers.get(&key).is_some_and(|layer| {
                is_transform_only_move(&layer.cache_key, &cache_key)
            });

        if composite {
            // Primitives come from the layer; everything else the skipped paint
            // would have registered — mouse listeners, cursor styles, input
            // handlers, tab stops, shaped text, accessed element state — comes
            // from the recorded range. Without this a composited panel renders
            // correctly and stops responding to the mouse.
            //
            // The replacement range is stamped around BOTH halves: a `PaintIndex`
            // spans every array, so only the full reuse+composite extent stays a
            // valid source for the next frame's replay. With slabs, compositing
            // no longer appends the recorded primitives to the frame arrays, and
            // leaving the recording-era range in place would read as out-of-bounds
            // against composited frames' shorter scenes and force an eternal
            // re-render.
            let composite_start = self.paint_index();
            let range = self.layers[&key].paint_range.clone();
            self.reuse_paint_except_scene(&range);
            if self.is_layer_occluded(key) {
                crate::render_stats::count("occlusion: layers culled");
                if let Some(layer) = self.layers.get_mut(&key) {
                    layer.deferred_dirty = false;
                }
                // Occluded frames emit nothing new; the previous range still
                // describes the last visible extent.
            } else if self.layers[&key].texture_retained {
                // The #96 skip condition: the content lives in a persistent
                // texture, so the whole composite is one surface draw. The
                // surface's bounds carry the buffer shift, so this same path
                // covers both the still frame and the scroll frame.
                self.paint_layer_texture_surface(key);
                let composite_end = self.paint_index();
                if let Some(layer) = self.layers.get_mut(&key) {
                    layer.paint_range = composite_start..composite_end;
                }
            } else {
                self.composite_layer(key);
                let composite_end = self.paint_index();
                if let Some(layer) = self.layers.get_mut(&key) {
                    layer.paint_range = composite_start..composite_end;
                }
            }
            return None;
        }

        if transform_only_move
            && self.try_composite_transform_only_move(key, cache_key.clone(), self.layer_frame)
        {
            return None;
        }

        // Deferred dirty: if the layer has existing content and is occluded,
        // skip the re-render and keep old content until it becomes visible.
        if self.layers.get(&key).is_some_and(|layer| layer.has_content())
            && self.is_layer_occluded(key)
        {
            crate::render_stats::count("occlusion: layers culled");
            crate::render_stats::count("occlusion: layers deferred-dirty");
            if let Some(layer) = self.layers.get_mut(&key) {
                layer.deferred_dirty = true;
            }
            let range = self.layers[&key].paint_range.clone();
            self.reuse_paint_except_scene(&range);
            return None;
        }

        let (result, accessed_entities) = cx.detect_accessed_entities(|cx| {
            self.retained_layer_stack.push(key);
            let result = self.record_layer(key, cache_key, policy, |window| f(window, cx));
            self.retained_layer_stack.pop();
            result
        });

        if let Some(layer) = self.layers.get_mut(&key) {
            layer.accessed_entities = accessed_entities;
            layer.content_key = content_key;
            layer.had_mouse = mouse_inside;
        }

        // #96: the record frame painted inline (correct for this frame), and a
        // texture-retained layer must also bake its content into the
        // renderer's persistent texture so the NEXT frame can composite from
        // it. The spans below carry the texture target, so the renderer
        // redirects them into the texture instead of drawing them on screen —
        // the inline primitives above remain this frame's only visible output.
        if self
            .layers
            .get(&key)
            .is_some_and(|layer| layer.texture_retained)
        {
            self.emit_layer_texture_render(key, self.layer_frame);
        }

        Some(result)
    }

    /// Run `f`, keeping every primitive it paints in the retained layer `key`.
    ///
    /// The lower half of [`Self::with_retained_layer`], split out because
    /// [`AnyView::cached`](crate::AnyView::cached) makes its own decision about
    /// whether to reuse — it has an invalidation record of its own that
    /// predates layers — and needs only the retention.
    pub(crate) fn record_layer<R>(
        &mut self,
        key: LayerKey,
        cache_key: LayerCacheKey,
        policy: LayerPolicy,
        f: impl FnOnce(&mut Window) -> R,
    ) -> R {
        profiling::scope!("wgpui: record layer");
        crate::render_stats::count("layer: re-rendered");
        let scaled_bounds = cache_key.bounds.scale(cache_key.scale_factor);
        let paint_start = self.paint_index();
        let id = self.next_layer_id;
        let frame = self.layer_frame;

        // Overscroll buffer (#96): the texture covers bounds + 2 × margin, so
        // content beyond the viewport exists and a scroll within the margin
        // never needs a re-record. Everything downstream — the ordering
        // scope's extent, the occlusion protection band, the pack origin —
        // keys off this rect rather than the visible bounds.
        let margin = policy.overdraw_margin;
        let buffered = margin != size(px(0.), px(0.));
        let texture_bounds = if buffered {
            crate::layer::inflate_bounds(cache_key.bounds, margin)
        } else {
            cache_key.bounds
        };
        let scaled_texture_bounds = texture_bounds.scale(cache_key.scale_factor);

        let was_texture_retained = self
            .layers
            .get(&key)
            .is_some_and(|layer| layer.texture_retained && layer.has_content());

        {
            let layer = self.layers.entry(key).or_insert_with(|| {
                crate::render_stats::count("layer: created");
                Layer::new(LayerId(id), policy, frame)
            });
            layer.opaque_bounds = None;
            layer.poisoned_bounds.clear();
        }

        self.next_frame
            .scene
            .begin_layer(key, scaled_texture_bounds, true);
        let result = f(self);
        let items = self
            .next_frame
            .scene
            .end_layer()
            .expect("a recording layer must return its captured items");

        // Instance-tier occlusion (#95): drop this layer's own primitives that
        // are fully covered by opaque quads painted above them in the same
        // layer, before anything downstream sees them. One decision upstream
        // of both consumers — `pack_layer_at_record` below and the legacy
        // replay in `composite_layer_legacy` — so packed and legacy outputs
        // agree by construction.
        //
        // This runs only while the layer is re-rendering (dirty anyway), and
        // only against same-layer occluders: any change to those occluders
        // re-records the layer through this very path, so a baked decision
        // cannot go stale silently and a clean layer's slab never churns
        // because an occluder moved. The retained items are rebuilt wholesale
        // on every record; reconciled children's owned `ElementInstance::items`
        // were captured during the walk above and stay complete, so a later
        // record replays them intact once they are no longer covered.
        //
        // #96: overdraw regions are exempt — content outside the current clip
        // exists precisely so a later scroll (a transform-only composite) can
        // reveal it, so items reaching into the margin band are never culled
        // and never act as occluders.
        let items = if buffered {
            let viewport = cache_key.bounds.scale(cache_key.scale_factor);
            let buffer = scaled_texture_bounds;
            crate::occlusion::cull_covered_instances_excluding_overdraw(items, viewport, buffer)
        } else {
            crate::occlusion::cull_covered_instances(items)
        };

        let paint_range = paint_start..self.paint_index();
        if let Some(layer) = self.layers.get_mut(&key) {
            if layer.id.0 == id {
                self.next_layer_id = self.next_layer_id.wrapping_add(1);
            }
            layer.policy = policy;
            layer.items = items;
            layer.paint_range = paint_range;
            let bounds = cache_key.bounds;
            layer.cache_key = cache_key;
            layer.transform = LayerTransform {
                offset: bounds.origin,
            };
            // Fresh items mean fresh content: the next composite of this
            // layer must re-upload its slab, which the new token forces.
            self.slab_tokens.insert(key, self.next_slab_token);
            self.next_slab_token += 1;
            // Pack once per content generation: every composite until the
            // next record splices spans straight from this pack, so the
            // validate/gather/sort work runs here instead of per frame.
            // Content is origin-relative to THIS record's origin, which is
            // exactly what a clean composite re-emits. A buffered layer packs
            // relative to the TEXTURE origin instead, so its spans draw in
            // texture space with a zero transform translate.
            let pack_origin = if buffered {
                [
                    scaled_texture_bounds.origin.x.0,
                    scaled_texture_bounds.origin.y.0,
                ]
            } else {
                [scaled_bounds.origin.x.0, scaled_bounds.origin.y.0]
            };
            layer.packed = if crate::scene_pack::slabs_enabled() {
                pack_layer_at_record(&layer.items, pack_origin)
            } else {
                None
            };
            // Texture retention (#96): rasterize packable, nested-free content
            // above the policy threshold — or always, for an overscroll
            // buffer, whose margin content is the point. `WGPUI_LAYERS_RASTERIZE=0`
            // keeps everything primitive-retained.
            let has_nested = layer
                .items
                .iter()
                .any(|item| matches!(item, LayerItem::Nested(_)));
            let rasterizable = crate::layer::rasterization_enabled()
                && crate::scene_pack::slabs_enabled()
                && layer.packed.as_ref().is_some_and(|packed| packed.is_ok())
                && !has_nested
                && (buffered || layer.items.len() > policy.rasterize_above);
            layer.texture_retained = rasterizable;
            layer.texture_bounds = texture_bounds;
            if rasterizable && buffered && was_texture_retained {
                crate::render_stats::count("scroll: buffer refills");
            }
            layer.needs = Invalidation::empty();
            layer.last_visited = frame;
            layer.had_mouse = false;
            layer.deferred_dirty = false;
            layer.poisoned_bounds.clear();

            if crate::layer::layer_debug_enabled() {
                self.paint_layer_debug_tint(key, bounds, true);
            }
        }

        result
    }

    /// Mark the current retained layer as fully opaque over `bounds`.
    /// Callers must only use this for a solid, alpha-one, unrounded region.
    pub(crate) fn mark_current_layer_opaque(&mut self, bounds: Bounds<Pixels>) {
        let Some(key) = self.retained_layer_stack.last().copied() else {
            return;
        };
        if let Some(layer) = self.layers.get_mut(&key) {
            layer.opaque_bounds = Some(bounds);
        }
    }

    /// Mark the current layer as having a backdrop filter or filter group that
    /// reads pixels from behind it within `expanded_bounds`. Layers beneath
    /// this region must not be occluded.
    pub(crate) fn mark_current_layer_poisoned(&mut self, expanded_bounds: Bounds<Pixels>) {
        let Some(key) = self.retained_layer_stack.last().copied() else {
            return;
        };
        if let Some(layer) = self.layers.get_mut(&key) {
            layer.poisoned_bounds.push(expanded_bounds);
        }
    }

    pub(crate) fn is_bounds_occluded(&self, bounds: Bounds<Pixels>) -> bool {
        if !crate::occlusion::enabled() {
            return false;
        }
        let occluders = self
            .layers
            .values()
            .filter_map(|layer| layer.opaque_bounds)
            .collect::<Vec<_>>();
        crate::occlusion::fully_covered(bounds, &occluders)
    }

    fn is_layer_occluded(&self, key: LayerKey) -> bool {
        if !crate::occlusion::enabled() {
            return false;
        }
        let Some(target) = self.layers.get(&key) else {
            return false;
        };
        // Check backdrop filter / filter group poisoning: any layer above the
        // target with poisoned bounds that overlap the target's bounds prevents
        // occlusion. The filter reads the pixels underneath it.
        let target_id = target.id;
        for layer in self.layers.values() {
            if layer.id <= target_id {
                continue;
            }
            for poisoned in &layer.poisoned_bounds {
                let overlap = poisoned.intersect(&target.cache_key.bounds);
                if overlap.size.width > Pixels::ZERO && overlap.size.height > Pixels::ZERO {
                    crate::render_stats::count("occlusion: poisoned by backdrop filter");
                    return false;
                }
            }
        }

        let occluders = self
            .layers
            .values()
            .filter(|layer| layer.id > target.id)
            .filter_map(|layer| layer.opaque_bounds)
            .collect::<Vec<_>>();
        crate::occlusion::fully_covered(target.cache_key.bounds, &occluders)
    }

    /// The [`LayerKey`] and cache key a layer rooted at `global_id` would have
    /// right now.
    pub(crate) fn layer_identity(
        &self,
        global_id: &GlobalElementId,
        bounds: Bounds<Pixels>,
    ) -> (LayerKey, LayerCacheKey) {
        (
            LayerKey::from_global_element_id(global_id),
            LayerCacheKey {
                bounds,
                content_mask: self.content_mask(),
                opacity: self.element_opacity,
                scale_factor: self.scale_factor(),
            },
        )
    }

    /// Composite `key` if it holds retained content, reporting whether it did.
    ///
    /// A caller that decided to reuse before reaching paint has to handle
    /// `false`: the layer may have been evicted, in which case the only
    /// remaining source for its primitives is the old index-range replay.
    pub(crate) fn try_composite_layer(&mut self, key: LayerKey) -> bool {
        if !crate::layer::layers_enabled() {
            return false;
        }
        if !self
            .layers
            .get(&key)
            .is_some_and(|layer| layer.has_content())
        {
            crate::render_stats::count("layer: composite missed (no retained content)");
            return false;
        }
        self.composite_layer(key);
        true
    }

    /// Re-emit `key`'s retained content, and that of every layer nested
    /// inside it, without re-running any element code.
    ///
    /// With slabs enabled and the layer's own items packable, each maximal
    /// stretch of its own primitives is recorded as a scene slab span — no
    /// primitives are re-cloned into the frame arrays, and the renderer draws
    /// them from persistent per-layer buffers. Nested layers split stretches
    /// so their own ordering scopes sit exactly where they were painted. Any
    /// packing rejection takes the legacy replay path for the whole layer,
    /// which is correct rebuild behaviour, not an error.
    fn composite_layer(&mut self, key: LayerKey) {
        profiling::scope!("wgpui: layer composite");
        crate::render_stats::count("layer: composited");
        let _t = crate::render_stats::scope("layer: composite");
        let frame = self.layer_frame;
        let scale_factor = self.scale_factor();
        let bounds = self
            .layers
            .get(&key)
            .map(|layer| layer.cache_key.bounds)
            .unwrap_or_default();

        if crate::scene_pack::slabs_enabled()
            && !self.next_frame.scene.innermost_layer_is_recording()
        {
            let composited_as_slab =
                self.try_composite_layer_into_scene_as_slab(key, frame, scale_factor);
            if composited_as_slab {
                if crate::layer::layer_debug_enabled() {
                    self.paint_layer_debug_tint(key, bounds, false);
                }
                return;
            }
        }

        composite_layer_legacy(
            &mut self.next_frame.scene,
            &mut self.layers,
            key,
            frame,
            scale_factor,
        );

        if crate::layer::layer_debug_enabled() {
            self.paint_layer_debug_tint(key, bounds, false);
        }
    }

    /// The slab splice half of [`Self::composite_layer`]: emit one span per
    /// maximal stretch of `key`'s own primitives from the pack cached at
    /// record time, recursing into nested layers between stretches. Returns
    /// `false` when the layer must composite through the legacy path instead
    /// (nothing cached: nothing packable, a packing rejection, or slabs off
    /// at record time).
    ///
    /// No packing happens here — the bytes were built once at record, so a
    /// clean frame's composite is span emission only and the renderer reads
    /// the unchanged token as Clean (zero uploads).
    fn try_composite_layer_into_scene_as_slab(
        &mut self,
        key: LayerKey,
        frame: u64,
        scale_factor: f32,
    ) -> bool {
        profiling::scope!("wgpui: slab splice");
        // A texture-retained layer's pack is relative to its TEXTURE origin
        // (#96), so splicing it as a plain span would draw it offset by the
        // margin. Its composites go through the surface path instead — the
        // retained-layer decision routes there; this guard covers the direct
        // `composite_layer` callers (nested recursion, `AnyView::cached`'s
        // reuse), which fall back to the legacy replay: correct, just slower.
        if self
            .layers
            .get(&key)
            .is_some_and(|layer| layer.texture_retained)
        {
            return false;
        }
        let Some((scaled_bounds, scaled_origin, pack)) = self.layers.get(&key).and_then(|layer| {
            if !layer.has_content() {
                return None;
            }
            let pack = match &layer.packed {
                Some(Ok(pack)) => Arc::clone(pack),
                _ => return None,
            };
            let scaled_bounds = layer.cache_key.bounds.scale(scale_factor);
            let scaled_origin = [scaled_bounds.origin.x.0, scaled_bounds.origin.y.0];
            Some((scaled_bounds, scaled_origin, pack))
        }) else {
            return false;
        };
        self.emit_layer_slab_spans(
            key,
            frame,
            scale_factor,
            scaled_bounds,
            scaled_origin,
            &pack,
            None,
        )
    }

    /// The #94 headline path: composite a layer whose only change since its
    /// last render is its origin.
    ///
    /// The caller has already established every composite condition except
    /// full `cache_key` equality, and [`is_transform_only_move`] has
    /// established that the bounds differ by origin alone. Packed content is
    /// stored origin-relative, so re-emitting it under the SAME slab token at
    /// the NEW origin leaves the renderer's sync plan Clean: zero instance
    /// bytes move, and the whole relocation costs one 64-byte transform slot.
    /// That slot write happens renderer-side off the span's new origin.
    ///
    /// Returns `false` — having touched nothing, so the caller falls back to
    /// a normal re-record — when the layer turns out unable to take the path:
    ///
    /// - a recording layer is active above this one. The legacy fallback
    ///   inside [`Self::composite_layer`] would replay the retained
    ///   primitives at their recorded positions, which the move just made
    ///   stale; only a re-record repaints them correctly.
    /// - the retained items include nested layer references. Each nested
    ///   layer carries its own origin, refreshed only by its own paint walk,
    ///   which a composited parent skips; splicing them here would draw the
    ///   children at their previous positions inside a moved parent. Whether
    ///   the children moved by exactly the parent's delta is not something
    ///   this path can know, so it declines.
    /// - the retained items do not pack into slabs at the new origin (the
    ///   same rejection any slab composite hits).
    fn try_composite_transform_only_move(
        &mut self,
        key: LayerKey,
        cache_key: LayerCacheKey,
        frame: u64,
    ) -> bool {
        if self.next_frame.scene.innermost_layer_is_recording() {
            return false;
        }
        // A texture-retained layer moves even cheaper (#96): the texture is
        // position-independent, so the move is one surface draw at the new
        // bounds — no spans, no slot writes, no uploads.
        if self
            .layers
            .get(&key)
            .is_some_and(|layer| layer.texture_retained)
        {
            let new_bounds = cache_key.bounds;
            if let Some(layer) = self.layers.get_mut(&key) {
                let delta = new_bounds.origin - layer.cache_key.bounds.origin;
                layer.cache_key = cache_key;
                layer.transform = LayerTransform {
                    offset: new_bounds.origin + layer.content_offset,
                };
                if let Some(opaque) = &mut layer.opaque_bounds {
                    opaque.origin += delta;
                }
                for poisoned in &mut layer.poisoned_bounds {
                    poisoned.origin += delta;
                }
                layer.texture_bounds.origin += delta;
            }
            let composite_start = self.paint_index();
            let range = match self.layers.get(&key) {
                Some(layer) => layer.paint_range.clone(),
                None => return true,
            };
            self.reuse_paint_except_scene(&range);
            if self.is_layer_occluded(key) {
                crate::render_stats::count("occlusion: layers culled");
                if let Some(layer) = self.layers.get_mut(&key) {
                    layer.deferred_dirty = false;
                }
                return true;
            }
            crate::render_stats::count("layer: composited (texture)");
            self.paint_layer_texture_surface(key);
            let composite_end = self.paint_index();
            if let Some(layer) = self.layers.get_mut(&key) {
                layer.paint_range = composite_start..composite_end;
            }
            return true;
        }
        let scale_factor = self.scale_factor();
        let scaled_bounds = cache_key.bounds.scale(scale_factor);
        let scaled_origin = [scaled_bounds.origin.x.0, scaled_bounds.origin.y.0];
        // Packed bytes are origin-relative, so a pure translation leaves them
        // byte-identical: the record-time cache is reused as-is under the
        // SAME token — no repack, no uploads, one transform slot renderer-side.
        let pack = {
            let Some(layer) = self.layers.get(&key) else {
                return false;
            };
            if layer
                .items
                .iter()
                .any(|item| matches!(item, LayerItem::Nested(_)))
            {
                return false;
            }
            let Some(Ok(pack)) = layer.packed.as_ref() else {
                // Never packed or an unsupported-kind fallback: decline so the
                // caller falls back to a full re-record, which rebuilds it.
                return false;
            };
            Arc::clone(pack)
        };

        // Stamp everything a re-record would have stamped, minus the record:
        // the content is unchanged, so the slab token stays untouched and the
        // renderer keeps treating its resident bytes as current.
        let new_bounds = cache_key.bounds;
        if let Some(layer) = self.layers.get_mut(&key) {
            let delta = new_bounds.origin - layer.cache_key.bounds.origin;
            layer.cache_key = cache_key;
            layer.transform = LayerTransform {
                offset: new_bounds.origin,
            };
            // Occluder coverage is recorded in window coordinates; it moves
            // with the layer or culling decisions below and next frame judge
            // the wrong region.
            if let Some(opaque) = &mut layer.opaque_bounds {
                opaque.origin += delta;
            }
            for poisoned in &mut layer.poisoned_bounds {
                poisoned.origin += delta;
            }
        }

        let composite_start = self.paint_index();
        let range = match self.layers.get(&key) {
            Some(layer) => layer.paint_range.clone(),
            None => return true,
        };
        self.reuse_paint_except_scene(&range);
        if self.is_layer_occluded(key) {
            crate::render_stats::count("occlusion: layers culled");
            if let Some(layer) = self.layers.get_mut(&key) {
                layer.deferred_dirty = false;
            }
            // Occluded frames emit nothing new; the previous range still
            // describes the last visible extent.
            return true;
        }

        crate::render_stats::count("layer: composited (transform-only)");
        self.emit_layer_slab_spans(
            key,
            frame,
            scale_factor,
            scaled_bounds,
            scaled_origin,
            &pack,
            None,
        );
        let composite_end = self.paint_index();
        if let Some(layer) = self.layers.get_mut(&key) {
            layer.paint_range = composite_start..composite_end;
        }
        if crate::layer::layer_debug_enabled() {
            self.paint_layer_debug_tint(key, new_bounds, false);
        }
        true
    }

    /// The slab splice half shared by [`Self::try_composite_layer_into_scene_as_slab`]
    /// and [`Self::try_composite_transform_only_move`]: record one span per
    /// stretch of the layer's own primitives from `pack`, recursing into
    /// nested layers between stretches.
    ///
    /// The pack was built at `origin` (record time for a clean composite, the
    /// move for the transform-only path), so spans reference cached bytes
    /// whose reference origin equals the origin they declare — the renderer
    /// reads an unchanged token as Clean and uploads nothing. Everything the
    /// spans carry besides bounds/key/token/origin is precomputed in the
    /// pack; this loop only clones runs and Arcs.
    fn emit_layer_slab_spans(
        &mut self,
        key: LayerKey,
        frame: u64,
        scale_factor: f32,
        bounds: Bounds<ScaledPixels>,
        origin: [f32; 2],
        pack: &RecordedSlabPack,
        texture: Option<crate::scene::LayerTextureTarget>,
    ) -> bool {
        let token = self.slab_tokens.get(&key).copied().unwrap_or(0);

        self.next_frame.scene.begin_layer(key, bounds, false);
        crate::render_stats::count("layer: composited (slab)");
        for piece in &pack.pieces {
            match piece {
                SlabPackPiece::Stretch { runs, packed } => {
                    self.next_frame.scene.push_layer_slab_span(
                        bounds,
                        key,
                        token,
                        origin,
                        pack.totals,
                        runs.clone(),
                        Arc::clone(packed),
                        texture.clone(),
                    );
                }
                SlabPackPiece::Nested(nested) => {
                    self.composite_nested_layer_for_slab(*nested, frame, scale_factor);
                }
            }
        }
        self.next_frame.scene.end_layer();
        if let Some(layer) = self.layers.get_mut(&key) {
            layer.last_visited = frame;
        }
        true
    }

    /// Bake a freshly-recorded texture-retained layer's content into the
    /// renderer's persistent texture (#96).
    ///
    /// One span per stretch of the layer's own primitives, each carrying the
    /// [`crate::scene::LayerTextureTarget`] that redirects the renderer into
    /// the layer's texture. The pack was built at the texture origin, so the
    /// spans draw in texture space with a zero transform translate. Called
    /// only from the record path — the record frame's inline primitives are
    /// the visible output; these spans are the texture's.
    fn emit_layer_texture_render(&mut self, key: LayerKey, frame: u64) {
        profiling::scope!("wgpui: layer texture render");
        let scale_factor = self.scale_factor();
        let Some((scaled_texture_bounds, origin, pack, target)) =
            self.layers.get(&key).and_then(|layer| {
                if !layer.texture_retained || !layer.has_content() {
                    return None;
                }
                let pack = match &layer.packed {
                    Some(Ok(pack)) => Arc::clone(pack),
                    _ => return None,
                };
                let scaled_texture_bounds = layer.texture_bounds.scale(scale_factor);
                let origin = [
                    scaled_texture_bounds.origin.x.0,
                    scaled_texture_bounds.origin.y.0,
                ];
                let target = crate::scene::LayerTextureTarget {
                    layer_id: layer.id,
                    key,
                    content_token: self.slab_tokens.get(&key).copied().unwrap_or(0),
                    texture_bounds: scaled_texture_bounds,
                };
                Some((scaled_texture_bounds, origin, pack, target))
            })
        else {
            return;
        };
        crate::render_stats::count("layer: rasterized");
        self.emit_layer_slab_spans(
            key,
            frame,
            scale_factor,
            scaled_texture_bounds,
            origin,
            &pack,
            Some(target),
        );
    }

    /// The #96 skip condition: composite a texture-retained layer with a
    /// single surface draw.
    ///
    /// The surface samples the layer's persistent texture across its buffer
    /// extent, shifted by the buffered element's [`Layer::content_offset`]
    /// (zero for a still frame), and clips to the layer's visible rect —
    /// margin content exists in the texture but never paints outside the
    /// layer's own extent. Also moves [`Layer::transform`] by the same shift
    /// so hit testing inverts into the coordinate space the recorded hitboxes
    /// live in.
    fn paint_layer_texture_surface(&mut self, key: LayerKey) {
        let scale_factor = self.scale_factor();
        let Some((layer_id, content_offset, texture_bounds, visible_bounds)) =
            self.layers.get(&key).map(|layer| {
                (
                    layer.id,
                    layer.content_offset,
                    {
                        let mut bounds = layer.texture_bounds;
                        bounds.origin += layer.content_offset;
                        bounds
                    },
                    layer
                        .cache_key
                        .bounds
                        .intersect(&layer.cache_key.content_mask.bounds),
                )
            })
        else {
            return;
        };
        if let Some(layer) = self.layers.get_mut(&key) {
            layer.transform = LayerTransform {
                offset: layer.cache_key.bounds.origin + content_offset,
            };
        }

        crate::render_stats::count("layer: composited (texture)");
        use crate::{PaintSurface, scene::SurfaceContent};
        self.next_frame.scene.insert_primitive(PaintSurface {
            order: 0,
            bounds: texture_bounds.scale(scale_factor),
            content_mask: crate::ContentMask {
                bounds: visible_bounds.scale(scale_factor),
            },
            content: SurfaceContent::Layer(layer_id),
        });
    }

    /// Recurse into a nested layer from the slab path. The child decides its
    /// own representation independently — slabbed children splice spans into
    /// their own scope inside the parent's, legacy children replay inline.
    fn composite_nested_layer_for_slab(&mut self, nested: LayerKey, frame: u64, scale_factor: f32) {
        let exists = self
            .layers
            .get(&nested)
            .is_some_and(|layer| layer.has_content());
        if !exists {
            crate::render_stats::count("layer: nested reference missing");
            return;
        }
        if crate::scene_pack::slabs_enabled()
            && !self.next_frame.scene.innermost_layer_is_recording()
            && self.try_composite_layer_into_scene_as_slab(nested, frame, scale_factor)
        {
            return;
        }
        let scene = &mut self.next_frame.scene;
        let layers = &mut self.layers;
        composite_layer_legacy(scene, layers, nested, frame, scale_factor);
    }
    /// Tint `bounds` by the layer's id, at full strength on a frame it
    /// re-rendered.
    ///
    /// The failure this makes visible is a layer that re-renders every frame
    /// while looking correct. Without it that layer is only *slow*, which is
    /// indistinguishable from the layer simply being large — and the sizing
    /// rule layers introduce (separate content by update frequency, not just by
    /// visual grouping) is otherwise impossible to check.
    fn paint_layer_debug_tint(&mut self, key: LayerKey, bounds: Bounds<Pixels>, re_rendered: bool) {
        let Some(layer) = self.layers.get(&key) else {
            return;
        };
        let color = crate::layer::debug_tint(layer.id, re_rendered);
        let scale_factor = self.scale_factor();
        let content_mask = self.content_mask();
        self.next_frame.scene.insert_primitive(Quad {
            order: 0,
            border_style: BorderStyle::Solid,
            bounds: bounds.scale(scale_factor),
            content_mask: content_mask.scale(scale_factor),
            background: color.into(),
            border_color: Hsla { a: 1.0, ..color }.into(),
            corner_radii: Corners::default(),
            border_widths: Edges::all(ScaledPixels(1.)),
        });
    }

    /// Drop retained content for layers no draw has visited recently.
    ///
    /// Mark-and-sweep: every layer that composited or re-rendered this draw
    /// stamped `last_visited`. A layer past its `evict_after_frames` gives up
    /// its primitives but keeps its record — its key, its id, and the fact that
    /// it exists — so a scrolled-away-and-back panel re-materialises into the
    /// same identity instead of being seen as new. The record itself is dropped
    /// after a further interval.
    ///
    /// This is also what bounds retained *instance* memory (#92,
    /// [`Layer::instances`]): instances are owned by layers and die with them,
    /// via `Layer::drop_content` below.
    fn evict_stale_layers(&mut self) {
        profiling::scope!("wgpui: evict_stale_layers");
        let frame = self.layer_frame;
        let mut dropped_content = 0usize;
        let mut dropped_records = 0usize;
        let mut forgotten_keys: Vec<LayerKey> = Vec::new();
        self.layers.retain(|key, layer| {
            let age = frame.saturating_sub(layer.last_visited);
            let evict_after = layer.policy.evict_after_frames as u64;
            if age > evict_after.saturating_mul(2) {
                dropped_records += 1;
                forgotten_keys.push(*key);
                log::trace!(
                    "layer {key:?} ({:?}) forgotten after {age} frames",
                    layer.id
                );
                return false;
            }
            if age > evict_after && layer.has_content() {
                dropped_content += 1;
                log::trace!(
                    "layer {key:?} ({:?}) dropped its content after {age} frames",
                    layer.id
                );
                layer.drop_content();
            }
            true
        });
        for key in &forgotten_keys {
            self.slab_tokens.remove(key);
        }
        for _ in 0..dropped_content + dropped_records {
            crate::render_stats::count("layer: evicted");
        }
    }

    /// Creates a new painting layer for the specified bounds. A "layer" is a batch
    /// of geometry that are non-overlapping and have the same draw order. This is typically used
    /// for performance reasons.
    ///
    /// Unrelated to [`Self::with_retained_layer`]: this is a clip-and-collapse
    /// group within one frame, not a retained one across frames.
    ///
    /// This method should only be called as part of the paint phase of element drawing.
    pub fn paint_layer<R>(&mut self, bounds: Bounds<Pixels>, f: impl FnOnce(&mut Self) -> R) -> R {
        self.invalidator.debug_assert_paint();

        let scale_factor = self.scale_factor();
        let content_mask = self.content_mask();
        let clipped_bounds = bounds.intersect(&content_mask.bounds);
        if !clipped_bounds.is_empty() {
            self.next_frame
                .scene
                .push_layer(clipped_bounds.scale(scale_factor));
        }

        let result = f(self);

        if !clipped_bounds.is_empty() {
            self.next_frame.scene.pop_layer();
        }

        result
    }

    /// Paint one or more drop shadows into the scene for the next frame at the current z-index.
    ///
    /// This method should only be called as part of the paint phase of element drawing.
    pub fn paint_shadows(
        &mut self,
        bounds: Bounds<Pixels>,
        corner_radii: Corners<Pixels>,
        shadows: &[BoxShadow],
    ) {
        self.invalidator.debug_assert_paint();

        let scale_factor = self.scale_factor();
        let content_mask = self.content_mask();
        let opacity = self.element_opacity();
        for shadow in shadows {
            let shadow_bounds = (bounds + shadow.offset).dilate(shadow.spread_radius);
            self.next_frame.scene.insert_primitive(Shadow {
                order: 0,
                blur_radius: shadow.blur_radius.scale(scale_factor),
                bounds: shadow_bounds.scale(scale_factor),
                content_mask: content_mask.scale(scale_factor),
                corner_radii: corner_radii.scale(scale_factor),
                color: shadow.color.opacity(opacity),
            });
        }
    }

    /// Paint one or more quads into the scene for the next frame at the current stacking context.
    /// Quads are colored rectangular regions with an optional background, border, and corner radius.
    /// see [`fill`], [`outline`], and [`quad`] to construct this type.
    ///
    /// This method should only be called as part of the paint phase of element drawing.
    ///
    /// Note that the `quad.corner_radii` are allowed to exceed the bounds, creating sharp corners
    /// where the circular arcs meet. This will not display well when combined with dashed borders.
    /// Use `Corners::clamp_radii_for_quad_size` if the radii should fit within the bounds.
    pub fn paint_quad(&mut self, quad: PaintQuad) {
        self.invalidator.debug_assert_paint();

        let scale_factor = self.scale_factor();
        let content_mask = self.content_mask();
        let opacity = self.element_opacity();
        self.next_frame.scene.insert_primitive(Quad {
            order: 0,
            bounds: quad.bounds.scale(scale_factor),
            content_mask: content_mask.scale(scale_factor),
            background: quad.background.opacity(opacity),
            border_color: quad.border_color.opacity(opacity),
            corner_radii: quad.corner_radii.scale(scale_factor),
            border_widths: quad.border_widths.scale(scale_factor),
            border_style: quad.border_style,
        });
    }

    /// Paint a backdrop filter into the scene for the next frame at the current z-index. The
    /// renderer blurs the content already painted behind `bounds` and composites the result
    /// into the rounded rectangle described by `bounds` and `corner_radii` — the CSS
    /// `backdrop-filter` effect (frosted glass). Typically the element then paints a translucent
    /// background quad on top so its color tints the blurred backdrop.
    ///
    /// Does nothing when `filters` produce no visible blur.
    ///
    /// This method should only be called as part of the paint phase of element drawing.
    pub fn paint_backdrop_filter(
        &mut self,
        bounds: Bounds<Pixels>,
        corner_radii: Corners<Pixels>,
        filters: &[Filter],
    ) {
        self.invalidator.debug_assert_paint();

        let radius = Filter::max_blur_radius(filters);
        if radius <= Pixels::ZERO {
            return;
        }

        let scale_factor = self.scale_factor();
        let content_mask = self.content_mask();
        self.next_frame.scene.insert_primitive(BackdropFilter {
            order: 0,
            bounds: bounds.scale(scale_factor),
            content_mask: content_mask.scale(scale_factor),
            corner_radii: corner_radii.scale(scale_factor),
            blur_radius: radius.scale(scale_factor),
            opacity: self.element_opacity(),
            _pad: 0,
        });

        // Poisoning: the backdrop filter reads content behind it. Layers
        // beneath this region must not be occluded. Expand by the blur radius
        // since the filter samples neighbouring pixels.
        self.mark_current_layer_poisoned(bounds.dilate(radius));
    }

    /// Isolate the painting performed by `f` into a content-filter group: the renderer renders
    /// everything `f` paints into an offscreen target, blurs it as a single layer, and
    /// composites the result back into the rounded rectangle described by `bounds` and
    /// `corner_radii` — the CSS `filter` effect (e.g. blurring an element and its children).
    ///
    /// When `filters` produce no visible blur this simply runs `f` with no offscreen
    /// indirection.
    ///
    /// This method should only be called as part of the paint phase of element drawing.
    pub fn with_filter_layer<R>(
        &mut self,
        bounds: Bounds<Pixels>,
        corner_radii: Corners<Pixels>,
        filters: &[Filter],
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        self.invalidator.debug_assert_paint();

        let radius = Filter::max_blur_radius(filters);
        if radius <= Pixels::ZERO {
            return f(self);
        }

        // Snapshot the (scaled) group parameters once so the start and end markers agree.
        //
        // `opacity` is 1.0 — NOT `element_opacity()`. The group's children/bg/border are painted
        // through the normal paint methods while `element_opacity` is still in effect, so they
        // already carry the element's opacity (consistent with gpui's per-primitive opacity for
        // non-filtered elements). Re-applying it at composite time would double it (e.g.
        // `.blur(r).opacity(0.5)` would render at 0.25 instead of 0.5).
        let scale_factor = self.scale_factor();
        let content_mask = self.content_mask();
        let boundary = FilterBoundary {
            order: 0,
            bounds: bounds.scale(scale_factor),
            content_mask: content_mask.scale(scale_factor),
            corner_radii: corner_radii.scale(scale_factor),
            blur_radius: radius.scale(scale_factor),
            opacity: 1.0,
            is_start: true,
        };

        // Poisoning: filter groups sample neighbouring pixels within
        // `max_blur_radius`. Layers beneath and within that margin must not
        // be occluded, or the filter reads stale pixels.
        self.mark_current_layer_poisoned(bounds.dilate(radius));

        self.next_frame.scene.insert_primitive(boundary);
        let result = f(self);
        self.next_frame.scene.insert_primitive(FilterBoundary {
            is_start: false,
            ..boundary
        });

        result
    }

    /// Paint the given `Path` into the scene for the next frame at the current z-index.
    ///
    /// This method should only be called as part of the paint phase of element drawing.
    pub fn paint_path(&mut self, mut path: Path<Pixels>, color: impl Into<Background>) {
        self.invalidator.debug_assert_paint();

        let scale_factor = self.scale_factor();
        let content_mask = self.content_mask();
        let opacity = self.element_opacity();
        path.content_mask = content_mask;
        let color: Background = color.into();
        path.color = color.opacity(opacity);
        self.next_frame
            .scene
            .insert_primitive(path.scale(scale_factor));
    }

    /// Paint an underline into the scene for the next frame at the current z-index.
    ///
    /// This method should only be called as part of the paint phase of element drawing.
    pub fn paint_underline(
        &mut self,
        origin: Point<Pixels>,
        width: Pixels,
        style: &UnderlineStyle,
    ) {
        self.invalidator.debug_assert_paint();

        let scale_factor = self.scale_factor();
        let height = if style.wavy {
            style.thickness * 3.
        } else {
            style.thickness
        };
        let bounds = Bounds {
            origin,
            size: size(width, height),
        };
        let content_mask = self.content_mask();
        let element_opacity = self.element_opacity();

        self.next_frame.scene.insert_primitive(Underline {
            order: 0,
            pad: 0,
            bounds: bounds.scale(scale_factor),
            content_mask: content_mask.scale(scale_factor),
            color: style.color.unwrap_or_default().opacity(element_opacity),
            thickness: style.thickness.scale(scale_factor),
            wavy: if style.wavy { 1 } else { 0 },
        });
    }

    /// Paint a strikethrough into the scene for the next frame at the current z-index.
    ///
    /// This method should only be called as part of the paint phase of element drawing.
    pub fn paint_strikethrough(
        &mut self,
        origin: Point<Pixels>,
        width: Pixels,
        style: &StrikethroughStyle,
    ) {
        self.invalidator.debug_assert_paint();

        let scale_factor = self.scale_factor();
        let height = style.thickness;
        let bounds = Bounds {
            origin,
            size: size(width, height),
        };
        let content_mask = self.content_mask();
        let opacity = self.element_opacity();

        self.next_frame.scene.insert_primitive(Underline {
            order: 0,
            pad: 0,
            bounds: bounds.scale(scale_factor),
            content_mask: content_mask.scale(scale_factor),
            thickness: style.thickness.scale(scale_factor),
            color: style.color.unwrap_or_default().opacity(opacity),
            wavy: 0,
        });
    }

    /// Paints a monochrome (non-emoji) glyph into the scene for the next frame at the current z-index.
    ///
    /// The y component of the origin is the baseline of the glyph.
    /// You should generally prefer to use the [`ShapedLine::paint`](crate::ShapedLine::paint) or
    /// [`WrappedLine::paint`](crate::WrappedLine::paint) methods in the [`TextSystem`](crate::TextSystem).
    /// This method is only useful if you need to paint a single glyph that has already been shaped.
    ///
    /// This method should only be called as part of the paint phase of element drawing.
    pub fn paint_glyph(
        &mut self,
        origin: Point<Pixels>,
        font_id: FontId,
        glyph_id: GlyphId,
        font_size: Pixels,
        color: TextColor,
    ) -> Result<()> {
        self.invalidator.debug_assert_paint();

        let element_opacity = self.element_opacity();
        let scale_factor = self.scale_factor();
        let glyph_origin = origin.scale(scale_factor);

        let subpixel_variant = Point {
            x: (glyph_origin.x.0.fract() * SUBPIXEL_VARIANTS_X as f32).floor() as u8,
            y: (glyph_origin.y.0.fract() * SUBPIXEL_VARIANTS_Y as f32).floor() as u8,
        };
        let params = RenderGlyphParams {
            font_id,
            glyph_id,
            font_size,
            subpixel_variant,
            scale_factor,
            is_emoji: false,
        };

        let raster_bounds = self.text_system().raster_bounds(&params)?;
        if !raster_bounds.is_zero() {
            let tile = self
                .sprite_atlas
                .get_or_insert_with(&params.clone().into(), &mut || {
                    let (size, bytes) = self.text_system().rasterize_glyph(&params)?;
                    Ok(Some((size, Cow::Owned(bytes))))
                })?
                .expect("Callback above only errors or returns Some");
            let bounds = Bounds {
                origin: glyph_origin.map(|px| px.floor()) + raster_bounds.origin.map(Into::into),
                size: tile.bounds.size.map(Into::into),
            };
            let content_mask = self.content_mask().scale(scale_factor);

            // Debug: Print glyph info
            use std::sync::atomic::{AtomicU32, Ordering};
            static GLYPH_COUNT: AtomicU32 = AtomicU32::new(0);
            let count = GLYPH_COUNT.fetch_add(1, Ordering::Relaxed);

            self.next_frame.scene.insert_primitive(MonochromeSprite {
                order: 0,
                pad: 0,
                bounds,
                content_mask,
                text_color: color.with_opacity(element_opacity),
                tile,
                transformation: TransformationMatrix::unit(),
            });
        }
        Ok(())
    }

    /// Paints an emoji glyph into the scene for the next frame at the current z-index.
    ///
    /// The y component of the origin is the baseline of the glyph.
    /// You should generally prefer to use the [`ShapedLine::paint`](crate::ShapedLine::paint) or
    /// [`WrappedLine::paint`](crate::WrappedLine::paint) methods in the [`TextSystem`](crate::TextSystem).
    /// This method is only useful if you need to paint a single emoji that has already been shaped.
    ///
    /// This method should only be called as part of the paint phase of element drawing.
    pub fn paint_emoji(
        &mut self,
        origin: Point<Pixels>,
        font_id: FontId,
        glyph_id: GlyphId,
        font_size: Pixels,
    ) -> Result<()> {
        self.invalidator.debug_assert_paint();

        let scale_factor = self.scale_factor();
        let glyph_origin = origin.scale(scale_factor);
        let params = RenderGlyphParams {
            font_id,
            glyph_id,
            font_size,
            // We don't render emojis with subpixel variants.
            subpixel_variant: Default::default(),
            scale_factor,
            is_emoji: true,
        };

        let raster_bounds = self.text_system().raster_bounds(&params)?;
        if !raster_bounds.is_zero() {
            let tile = self
                .sprite_atlas
                .get_or_insert_with(&params.clone().into(), &mut || {
                    let (size, bytes) = self.text_system().rasterize_glyph(&params)?;
                    Ok(Some((size, Cow::Owned(bytes))))
                })?
                .expect("Callback above only errors or returns Some");

            let bounds = Bounds {
                origin: glyph_origin.map(|px| px.floor()) + raster_bounds.origin.map(Into::into),
                size: tile.bounds.size.map(Into::into),
            };
            let content_mask = self.content_mask().scale(scale_factor);
            let opacity = self.element_opacity();

            self.next_frame.scene.insert_primitive(PolychromeSprite {
                order: 0,
                pad: 0,
                grayscale: 0,
                bounds,
                corner_radii: Default::default(),
                content_mask,
                tile,
                opacity,
            });
        }
        Ok(())
    }

    /// Paint a monochrome SVG into the scene for the next frame at the current stacking context.
    ///
    /// This method should only be called as part of the paint phase of element drawing.
    pub fn paint_svg(
        &mut self,
        bounds: Bounds<Pixels>,
        path: SharedString,
        mut data: Option<&[u8]>,
        transformation: TransformationMatrix,
        color: Hsla,
        cx: &App,
    ) -> Result<()> {
        self.invalidator.debug_assert_paint();

        let element_opacity = self.element_opacity();
        let scale_factor = self.scale_factor();

        let bounds = bounds.scale(scale_factor);
        let params = RenderSvgParams {
            path,
            size: bounds.size.map(|pixels| {
                DevicePixels::from((pixels.0 * SMOOTH_SVG_SCALE_FACTOR).ceil() as i32)
            }),
        };

        let Some(tile) =
            self.sprite_atlas
                .get_or_insert_with(&params.clone().into(), &mut || {
                    let Some((size, bytes)) = cx.svg_renderer.render_alpha_mask(&params, data)?
                    else {
                        return Ok(None);
                    };
                    Ok(Some((size, Cow::Owned(bytes))))
                })?
        else {
            return Ok(());
        };
        let content_mask = self.content_mask().scale(scale_factor);
        let svg_bounds = Bounds {
            origin: bounds.center()
                - Point::new(
                    ScaledPixels(tile.bounds.size.width.0 as f32 / SMOOTH_SVG_SCALE_FACTOR / 2.),
                    ScaledPixels(tile.bounds.size.height.0 as f32 / SMOOTH_SVG_SCALE_FACTOR / 2.),
                ),
            size: tile
                .bounds
                .size
                .map(|value| ScaledPixels(value.0 as f32 / SMOOTH_SVG_SCALE_FACTOR)),
        };

        self.next_frame.scene.insert_primitive(MonochromeSprite {
            order: 0,
            pad: 0,
            bounds: svg_bounds
                .map_origin(|origin| origin.round())
                .map_size(|size| size.ceil()),
            content_mask,
            text_color: TextColor::from(color).with_opacity(element_opacity),
            tile,
            transformation,
        });

        Ok(())
    }

    /// Paint an image into the scene for the next frame at the current z-index.
    /// This method will panic if the frame_index is not valid
    ///
    /// This method should only be called as part of the paint phase of element drawing.
    pub fn paint_image(
        &mut self,
        bounds: Bounds<Pixels>,
        corner_radii: Corners<Pixels>,
        data: Arc<RenderImage>,
        frame_index: usize,
        grayscale: bool,
    ) -> Result<()> {
        self.invalidator.debug_assert_paint();

        let scale_factor = self.scale_factor();
        let bounds = bounds.scale(scale_factor);
        let params = RenderImageParams {
            image_id: data.id,
            frame_index,
        };

        let tile = self
            .sprite_atlas
            .get_or_insert_with(&params.into(), &mut || {
                Ok(Some((
                    data.size(frame_index),
                    Cow::Borrowed(
                        data.as_bytes(frame_index)
                            .expect("It's the caller's job to pass a valid frame index"),
                    ),
                )))
            })?
            .expect("Callback above only returns Some");
        let content_mask = self.content_mask().scale(scale_factor);
        let corner_radii = corner_radii.scale(scale_factor);
        let opacity = self.element_opacity();

        self.next_frame.scene.insert_primitive(PolychromeSprite {
            order: 0,
            pad: 0,
            grayscale: u32::from(grayscale),
            bounds: bounds
                .map_origin(|origin| origin.floor())
                .map_size(|size| size.ceil()),
            content_mask,
            corner_radii,
            tile,
            opacity,
        });
        Ok(())
    }

    /// Paint a WGPU surface into the scene for the next frame at the current z-index.
    /// The renderer will look up the front buffer texture from the `SurfaceRegistry`
    /// using the given `SurfaceId`.
    ///
    /// This method should only be called as part of the paint phase of element drawing.
    pub fn paint_wgpu_surface(
        &mut self,
        bounds: Bounds<Pixels>,
        surface_id: crate::platform::cross::surface_registry::SurfaceId,
    ) {
        use crate::{PaintSurface, scene::SurfaceContent};

        self.invalidator.debug_assert_paint();

        let scale_factor = self.scale_factor();
        let bounds = bounds.scale(scale_factor);
        let content_mask = self.content_mask().scale(scale_factor);
        self.next_frame.scene.insert_primitive(PaintSurface {
            order: 0,
            bounds,
            content_mask,
            content: SurfaceContent::Wgpu(surface_id),
        });
    }

    /// Create a double-buffered WGPU surface handle for external GPU rendering.
    ///
    /// Returns `None` on platforms that don't use the WGPU renderer.
    /// The returned handle provides `device()` / `queue()` access and a
    /// `back_buffer_view()` you can render into, then call `present()` to
    /// swap buffers and trigger a re-composite (no layout/paint cycle).
    pub fn create_wgpu_surface(
        &self,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Option<crate::WgpuSurfaceHandle> {
        self.platform_window
            .create_wgpu_surface(width, height, format)
    }

    /// Removes an image from the sprite atlas.
    pub fn drop_image(&mut self, data: Arc<RenderImage>) -> Result<()> {
        for frame_index in 0..data.frame_count() {
            let params = RenderImageParams {
                image_id: data.id,
                frame_index,
            };

            self.sprite_atlas.remove(&params.clone().into());
        }

        Ok(())
    }

    /// Add a node to the layout tree for the current frame. Takes the `Style` of the element for which
    /// layout is being requested, along with the layout ids of any children. This method is called during
    /// calls to the [`Element::request_layout`] trait method and enables any element to participate in layout.
    ///
    /// This method should only be called as part of the request_layout or prepaint phase of element drawing.
    #[must_use]
    pub fn request_layout(
        &mut self,
        style: Style,
        children: impl IntoIterator<Item = LayoutId>,
        cx: &mut App,
    ) -> LayoutId {
        self.invalidator.debug_assert_prepaint();

        cx.layout_id_buffer.clear();
        cx.layout_id_buffer.extend(children);
        let rem_size = self.rem_size();
        let scale_factor = self.scale_factor();

        self.layout_engine.as_mut().unwrap().request_layout(
            style,
            rem_size,
            scale_factor,
            &cx.layout_id_buffer,
        )
    }

    /// Add a node to the layout tree for the current frame. Instead of taking a `Style` and children,
    /// this variant takes a function that is invoked during layout so you can use arbitrary logic to
    /// determine the element's size. One place this is used internally is when measuring text.
    ///
    /// The given closure is invoked at layout time with the known dimensions and available space and
    /// returns a `Size`.
    ///
    /// This method should only be called as part of the request_layout or prepaint phase of element drawing.
    pub fn request_measured_layout<F>(&mut self, style: Style, measure: F) -> LayoutId
    where
        F: Fn(Size<Option<Pixels>>, Size<AvailableSpace>, &mut Window, &mut App) -> Size<Pixels>
            + 'static,
    {
        self.invalidator.debug_assert_prepaint();

        let rem_size = self.rem_size();
        let scale_factor = self.scale_factor();
        self.layout_engine
            .as_mut()
            .unwrap()
            .request_measured_layout(style, rem_size, scale_factor, measure)
    }

    /// The retained `LayoutId` this frame may reuse instead of creating a
    /// fresh node, if `diff_key` proves it safe (#93).
    ///
    /// Reads `instance_id_stack`/`layout_layer_stack` exactly as they stand
    /// at the call site — correct only because the caller (an element's own
    /// `request_layout`) is invoked with its own segment already pushed by
    /// its parent's child loop, mirroring the identical convention
    /// `prepaint_reconciled_child` already relies on one phase later. Returns
    /// `None` — meaning "create fresh, exactly as before this phase" — for
    /// every case that isn't a proven-safe reuse: no layer, instances or
    /// persistent layout disabled, first sight of this `InstanceKey`, or a
    /// `diff_key` comparison whose axes include `LAYOUT`.
    fn reusable_layout(&self, diff_key: Option<&dyn ReconcileKey>) -> Option<LayoutId> {
        if !crate::taffy::persistent_layout_enabled() || !crate::instance::instances_enabled() {
            return None;
        }
        let new_key = diff_key?;
        let layer_key = self.current_layout_layer()?;
        let key = self.current_instance_key();
        let retained = self.layers.get(&layer_key)?.instances.get(&key)?;
        if new_key.compare(retained.diff_key.as_ref()).contains(Invalidation::LAYOUT) {
            return None;
        }
        Some(retained.layout)
    }

    /// [`Self::request_layout`], but reusing the retained node for the
    /// current `InstanceKey` instead of creating a fresh one when `diff_key`
    /// proves its `LAYOUT` axis is clean (#93) — the Taffy-level counterpart
    /// to reconciliation's `prepaint`/`paint` skip.
    ///
    /// `diff_key` is computed by the caller (an element's own `request_layout`)
    /// via its own `Element::diff_key`, once for this purpose and again,
    /// independently, wherever the element's parent makes its own
    /// `prepaint`-time reconciliation decision — recomputing rather than
    /// threading a shared value between the two, since both are meant to be
    /// cheap (see `instance.rs`'s module doc) and threading it would couple
    /// two decisions that resolve at genuinely different times against
    /// genuinely different criteria (this one only needs the `LAYOUT` bit;
    /// the other also needs bounds, content mask and entity dependencies that
    /// are not yet known this early).
    #[must_use]
    pub fn request_layout_or_reuse(
        &mut self,
        diff_key: Option<&dyn ReconcileKey>,
        style: Style,
        children: impl IntoIterator<Item = LayoutId>,
        cx: &mut App,
    ) -> LayoutId {
        if let Some(id) = self.reusable_layout(diff_key)
            && self.layout_engine.as_mut().unwrap().reuse(id)
        {
            crate::render_stats::count("taffy: node reused");
            return id;
        }
        self.request_layout(style, children, cx)
    }

    /// [`Self::request_measured_layout`], but reusing the retained node for
    /// the current `InstanceKey` the same way [`Self::request_layout_or_reuse`]
    /// does (#93).
    ///
    /// Reusing a measured node without re-registering `measure` is sound
    /// specifically *because* reuse only happens when `diff_key` proves
    /// nothing this element's own content depends on has changed — the stale
    /// `'static`-closure hazard the design doc's §2.5 describes is a hazard
    /// for retaining a node *despite* a change, which this never does; the
    /// closure captured last time it *did* change is still describing
    /// current content.
    pub fn request_measured_layout_or_reuse<F>(
        &mut self,
        diff_key: Option<&dyn ReconcileKey>,
        style: Style,
        measure: F,
    ) -> LayoutId
    where
        F: Fn(Size<Option<Pixels>>, Size<AvailableSpace>, &mut Window, &mut App) -> Size<Pixels>
            + 'static,
    {
        if let Some(id) = self.reusable_layout(diff_key)
            && self.layout_engine.as_mut().unwrap().reuse(id)
        {
            crate::render_stats::count("taffy: node reused");
            return id;
        }
        self.request_measured_layout(style, measure)
    }

    /// Compute the layout for the given id within the given available space.
    /// This method is called for its side effect, typically by the framework prior to painting.
    /// After calling it, you can request the bounds of the given layout node id or any descendant.
    ///
    /// This method should only be called as part of the prepaint phase of element drawing.
    pub fn compute_layout(
        &mut self,
        layout_id: LayoutId,
        available_space: Size<AvailableSpace>,
        cx: &mut App,
    ) {
        self.invalidator.debug_assert_prepaint();

        let mut layout_engine = self.layout_engine.take().unwrap();
        layout_engine.compute_layout(layout_id, available_space, self, cx);
        self.layout_engine = Some(layout_engine);
    }

    /// Obtain the bounds computed for the given LayoutId relative to the window. This method will usually be invoked by
    /// GPUI itself automatically in order to pass your element its `Bounds` automatically.
    ///
    /// This method should only be called as part of element drawing.
    pub fn layout_bounds(&mut self, layout_id: LayoutId) -> Bounds<Pixels> {
        self.invalidator.debug_assert_prepaint();

        let scale_factor = self.scale_factor();
        let mut bounds = self
            .layout_engine
            .as_mut()
            .unwrap()
            .layout_bounds(layout_id, scale_factor)
            .map(Into::into);
        bounds.origin += self.element_offset();
        bounds
    }

    /// This method should be called during `prepaint`. You can use
    /// the returned [Hitbox] during `paint` or in an event handler
    /// to determine whether the inserted hitbox was the topmost.
    ///
    /// This method should only be called as part of the prepaint phase of element drawing.
    pub fn insert_hitbox(&mut self, bounds: Bounds<Pixels>, behavior: HitboxBehavior) -> Hitbox {
        self.invalidator.debug_assert_prepaint();

        let mut bounds = bounds;
        let mut content_mask = self.content_mask();
        let layer = self.hitbox_layer_stack.last().copied();
        if let Some((_, origin, _)) = layer {
            bounds.origin -= origin;
            content_mask.bounds.origin -= origin;
        }
        let mut id = self.next_hitbox_id;
        self.next_hitbox_id = self.next_hitbox_id.next();
        let hitbox = Hitbox {
            id,
            bounds,
            content_mask,
            behavior,
            layer: layer.map(|(key, _, _)| key),
        };
        self.next_frame.hitboxes.push(hitbox.clone());
        hitbox
    }

    /// Record hitboxes inserted by `f` in `key`'s local coordinate space.
    ///
    /// The layer still paints in window coordinates today, but its input
    /// geometry must not: a later transform-only composite moves the pixels
    /// without re-running prepaint. Hit testing therefore maps the query point
    /// back into this coordinate space once per layer.
    ///
    /// `bounds` travels with the key so a buffered element prepainting inside
    /// the layer (#96) can evaluate the same composite decision
    /// [`Self::with_retained_layer`] will make at paint time — the prediction
    /// and the decision must read the same fresh inputs or a skipped list
    /// gets recorded empty.
    pub(crate) fn with_layer_hitbox_scope<R>(
        &mut self,
        key: LayerKey,
        bounds: Bounds<Pixels>,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        self.invalidator.debug_assert_prepaint();
        self.hitbox_layer_stack.push((key, bounds.origin, bounds));
        let result = f(self);
        self.hitbox_layer_stack.pop();
        result
    }

    /// The layer a `.layer()` div's *children* are currently being prepainted
    /// inside, if any (#92). Read from `hitbox_layer_stack`, which
    /// `with_layer_hitbox_scope` already keeps correctly scoped to exactly
    /// this — see that method's own doc comment.
    pub(crate) fn current_prepaint_layer(&self) -> Option<LayerKey> {
        self.hitbox_layer_stack.last().map(|(key, _, _)| *key)
    }

    /// The overscroll buffer (#96) of the layer the subtree is currently being
    /// prepainted inside, if that layer is texture-retained with a non-zero
    /// margin and its texture actually covers the buffer.
    ///
    /// `will_composite` mirrors the composite decision
    /// [`Self::with_retained_layer`] will make later this frame — both call
    /// [`Self::layer_reuse_conditions`] with the same fresh inputs, so a
    /// virtualized list can skip its per-item layout on frames the layer is
    /// going to composite anyway without the prediction ever disagreeing with
    /// the decision.
    ///
    /// `refilling` is true on the frame a `DISPLAY` invalidation will
    /// re-record the layer: the element must paint its full buffer range and
    /// re-anchor. `buffer_ready` is false until the first buffered record —
    /// the texture then covers the viewport only, and the element should
    /// request a refill rather than trust the margin.
    pub(crate) fn prepaint_layer_buffer(&self) -> Option<PrepaintLayerBuffer> {
        let (key, _, bounds) = self.hitbox_layer_stack.last().copied()?;
        let layer = self.layers.get(&key)?;
        let margin = layer.policy.overdraw_margin;
        if margin == size(px(0.), px(0.)) || !layer.texture_retained {
            return None;
        }
        let fresh_cache_key = LayerCacheKey {
            bounds,
            content_mask: self.content_mask(),
            opacity: self.element_opacity,
            scale_factor: self.scale_factor(),
        };
        let view_rebuilt =
            layer.content_key.is_none() && self.dirty_views.contains(&self.current_view());
        Some(PrepaintLayerBuffer {
            key,
            margin,
            anchor: layer.buffer_anchor,
            content_offset: layer.content_offset,
            will_composite: self.layer_reuse_conditions(
                layer,
                layer.content_key,
                /* mouse_inside */ false,
                view_rebuilt,
            ) && layer.cache_key == fresh_cache_key,
            refilling: !layer.needs.is_empty(),
            buffer_ready: layer.buffer_anchored,
        })
    }

    /// Whether the named layer's `DISPLAY` invalidation will re-record it this
    /// frame — the refill signal for a buffered element's prepaint.
    pub(crate) fn layer_is_refilling(&self, key: LayerKey) -> bool {
        self.layers
            .get(&key)
            .is_some_and(|layer| !layer.needs.is_empty())
    }

    /// Re-anchor a buffered layer's overscroll texture at `anchor` (the
    /// element's scroll-space position the buffer was rendered at). Also marks
    /// the buffer as anchored, the element's signal that the texture covers
    /// the full margin.
    pub(crate) fn set_layer_buffer_anchor(&mut self, key: LayerKey, anchor: Point<Pixels>) {
        if let Some(layer) = self.layers.get_mut(&key) {
            layer.buffer_anchor = anchor;
            layer.buffer_anchored = true;
        }
    }

    /// Record how far a buffered layer's content has scrolled since its
    /// texture was rendered. The composite shifts by this much instead of
    /// re-recording.
    pub(crate) fn set_layer_content_offset(&mut self, key: LayerKey, offset: Point<Pixels>) {
        if let Some(layer) = self.layers.get_mut(&key) {
            layer.content_offset = offset;
        }
    }

    /// Ask for a buffered layer to re-render its texture, re-centred on the
    /// current scroll position. Applied like any layer invalidation: the next
    /// draw answers it, which is why refills are requested at half margin —
    /// the shift never outruns the texture while the refill is in flight.
    pub(crate) fn request_layer_buffer_refill(&self, key: LayerKey) {
        self.invalidator.invalidate_layer(key, Invalidation::DISPLAY);
    }

    // A paint-time counterpart (`current_paint_layer`, reading
    // `retained_layer_stack`) is deliberately not exposed: `paint_reconciled_child`
    // (div.rs, #92) needs the *same* `LayerKey` `prepaint_reconciled_child`
    // already resolved, not a freshly re-queried one — see `ChildReconciliation`'s
    // doc comment for why re-deriving it independently at paint time would be
    // wrong. It travels from prepaint to paint inside `ChildReconciliation`
    // instead.

    /// Run `f` with `key` pushed onto `layout_layer_stack` (#93). Mirrors
    /// `with_layer_hitbox_scope`, one phase earlier — see
    /// `Window::layout_layer_stack`'s doc comment for why they're separate
    /// stacks rather than one.
    pub(crate) fn with_layout_layer<R>(&mut self, key: LayerKey, f: impl FnOnce(&mut Self) -> R) -> R {
        self.layout_layer_stack.push(key);
        let result = f(self);
        self.layout_layer_stack.pop();
        result
    }

    /// The layer a `.layer()` div's *children* are currently requesting
    /// layout inside, if any (#93). See `Window::layout_layer_stack`.
    pub(crate) fn current_layout_layer(&self) -> Option<LayerKey> {
        self.layout_layer_stack.last().copied()
    }

    /// Run `f` with `id` pushed onto `instance_id_stack`, so that any
    /// `InstanceKey` computed inside `f` — including one for a
    /// grandchild several levels deeper — includes this segment in its path
    /// (#92). Mirrors `element_id_stack`'s own push/pop discipline
    /// (`with_id`), kept as a separate stack for the reasons given on
    /// `Window::instance_id_stack`'s doc comment.
    pub(crate) fn with_instance_slot<R>(&mut self, id: ElementId, f: impl FnOnce(&mut Self) -> R) -> R {
        self.instance_id_stack.push(id);
        let result = f(self);
        self.instance_id_stack.pop();
        result
    }

    /// The `InstanceKey` for the element addressed by the current
    /// `instance_id_stack` (#92). Call this right after `with_instance_slot`
    /// has pushed the element's own segment, before recursing further into
    /// it — the stack at that point is exactly this element's path.
    pub(crate) fn current_instance_key(&self) -> InstanceKey {
        InstanceKey::from_path(&self.instance_id_stack)
    }

    /// Re-emit a reconciled `ElementInstance`'s retained items into the layer
    /// currently being (re-)recorded, preserving the layer-local draw orders
    /// they were recorded with (#92).
    ///
    /// Mirrors `composite_layer`'s replay of a *whole* layer, at instance
    /// granularity: a retained primitive goes through `Scene::push_retained`
    /// (no `BoundsTree` insert, no re-derivation of z), and a nested `.layer()`
    /// reference is re-registered as-is so the enclosing recorder still learns
    /// it was nested here. Must only be called while a layer is actively
    /// recording (`Scene::begin_layer(.., record: true)` open) — i.e. from
    /// inside the same paint walk `current_paint_layer` reports as active —
    /// or the re-emitted items are silently dropped on the floor instead of
    /// landing in the new capture, exactly as `push_retained`'s own
    /// capture-awareness requires.
    pub(crate) fn replay_instance_items(&mut self, items: &[LayerItem]) {
        for item in items {
            match item {
                LayerItem::Primitive(primitive) => self.next_frame.scene.push_retained(primitive),
                LayerItem::Nested(key) => self.next_frame.scene.push_captured_item(LayerItem::Nested(*key)),
            }
        }
    }

    /// How many items the innermost recording layer has captured so far
    /// (#92). Bracket a child's own contribution to a layer's retained
    /// `items` — `captured_len` before its `paint`, `captured_len` after —
    /// the same way `paint_index` brackets its contribution to the other
    /// paint-phase arrays. See `Scene::captured_len`.
    pub(crate) fn captured_len(&self) -> usize {
        self.next_frame.scene.captured_len()
    }

    /// Clone the items the innermost recording layer captured within `range`
    /// (#92). Paired with `captured_len`; see `Scene::captured_slice`.
    pub(crate) fn captured_slice(&self, range: Range<usize>) -> Vec<LayerItem> {
        self.next_frame.scene.captured_slice(range)
    }

    /// Set a hitbox which will act as a control area of the platform window.
    ///
    /// This method should only be called as part of the paint phase of element drawing.
    pub fn insert_window_control_hitbox(&mut self, area: WindowControlArea, hitbox: Hitbox) {
        self.invalidator.debug_assert_paint();
        self.next_frame.window_control_hitboxes.push((area, hitbox));
    }

    /// Sets the key context for the current element. This context will be used to translate
    /// keybindings into actions.
    ///
    /// This method should only be called as part of the paint phase of element drawing.
    pub fn set_key_context(&mut self, context: KeyContext) {
        self.invalidator.debug_assert_paint();
        self.next_frame.dispatch_tree.set_key_context(context);
    }

    /// Sets the focus handle for the current element. This handle will be used to manage focus state
    /// and keyboard event dispatch for the element.
    ///
    /// This method should only be called as part of the prepaint phase of element drawing.
    pub fn set_focus_handle(&mut self, focus_handle: &FocusHandle, _: &App) {
        self.invalidator.debug_assert_prepaint();
        if focus_handle.is_focused(self) {
            self.next_frame.focus = Some(focus_handle.id);
        }
        self.next_frame.dispatch_tree.set_focus_id(focus_handle.id);
    }

    /// Sets the view id for the current element, which will be used to manage view caching.
    ///
    /// This method should only be called as part of element prepaint. We plan on removing this
    /// method eventually when we solve some issues that require us to construct editor elements
    /// directly instead of always using editors via views.
    pub fn set_view_id(&mut self, view_id: EntityId) {
        self.invalidator.debug_assert_prepaint();
        self.next_frame.dispatch_tree.set_view_id(view_id);
    }

    /// Get the entity ID for the currently rendering view
    pub fn current_view(&self) -> EntityId {
        self.invalidator.debug_assert_paint_or_prepaint();
        self.rendered_entity_stack.last().copied().unwrap()
    }

    pub(crate) fn with_rendered_view<R>(
        &mut self,
        id: EntityId,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        self.rendered_entity_stack.push(id);
        let result = f(self);
        self.rendered_entity_stack.pop();
        result
    }

    /// Executes the provided function with the specified image cache.
    pub fn with_image_cache<F, R>(&mut self, image_cache: Option<AnyImageCache>, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        if let Some(image_cache) = image_cache {
            self.image_cache_stack.push(image_cache);
            let result = f(self);
            self.image_cache_stack.pop();
            result
        } else {
            f(self)
        }
    }

    /// Sets an input handler, such as [`ElementInputHandler`][element_input_handler], which interfaces with the
    /// platform to receive textual input with proper integration with concerns such
    /// as IME interactions. This handler will be active for the upcoming frame until the following frame is
    /// rendered.
    ///
    /// This method should only be called as part of the paint phase of element drawing.
    ///
    /// [element_input_handler]: crate::ElementInputHandler
    pub fn handle_input(
        &mut self,
        focus_handle: &FocusHandle,
        input_handler: impl InputHandler,
        cx: &App,
    ) {
        self.invalidator.debug_assert_paint();

        if focus_handle.is_focused(self) {
            let cx = self.to_async(cx);
            self.next_frame
                .input_handlers
                .push(Some(PlatformInputHandler::new(cx, Box::new(input_handler))));
        }
    }

    /// Register a mouse event listener on the window for the next frame. The type of event
    /// is determined by the first parameter of the given listener. When the next frame is rendered
    /// the listener will be cleared.
    ///
    /// This method should only be called as part of the paint phase of element drawing.
    pub fn on_mouse_event<Event: MouseEvent>(
        &mut self,
        mut listener: impl FnMut(&Event, DispatchPhase, &mut Window, &mut App) + 'static,
    ) {
        self.invalidator.debug_assert_paint();

        let discriminator = type_name_hash::<Event>();
        self.next_frame
            .mouse_listeners
            .push(Some(MouseListenerEntry {
                discriminator,
                listener: Box::new(
                    move |event: &dyn Any,
                          phase: DispatchPhase,
                          window: &mut Window,
                          cx: &mut App| {
                        // SAFETY: dispatch_mouse_event only calls this listener when the
                        // discriminator matches the dispatched event's type_name_hash.
                        let event = unsafe { &*(event as *const dyn Any as *const Event) };
                        listener(event, phase, window, cx)
                    },
                ),
            }));
    }

    /// Register a key event listener on this node for the next frame. The type of event
    /// is determined by the first parameter of the given listener. When the next frame is rendered
    /// the listener will be cleared.
    ///
    /// This is a fairly low-level method, so prefer using event handlers on elements unless you have
    /// a specific need to register a listener yourself.
    ///
    /// This method should only be called as part of the paint phase of element drawing.
    pub fn on_key_event<Event: KeyEvent>(
        &mut self,
        listener: impl Fn(&Event, DispatchPhase, &mut Window, &mut App) + 'static,
    ) {
        self.invalidator.debug_assert_paint();

        let discriminator = type_name_hash::<Event>();
        self.next_frame.dispatch_tree.on_key_event(
            discriminator,
            Rc::new(
                move |event: &dyn Any, phase, window: &mut Window, cx: &mut App| {
                    // SAFETY: dispatch_key_down_up_event only calls this listener
                    // when its discriminator matches the event's type_name_hash.
                    let event = unsafe { &*(event as *const dyn Any as *const Event) };
                    listener(event, phase, window, cx)
                },
            ),
        );
    }

    /// Register a modifiers changed event listener on the window for the next frame.
    ///
    /// This is a fairly low-level method, so prefer using event handlers on elements unless you have
    /// a specific need to register a global listener.
    ///
    /// This method should only be called as part of the paint phase of element drawing.
    pub fn on_modifiers_changed(
        &mut self,
        listener: impl Fn(&ModifiersChangedEvent, &mut Window, &mut App) + 'static,
    ) {
        self.invalidator.debug_assert_paint();

        self.next_frame.dispatch_tree.on_modifiers_changed(Rc::new(
            move |event: &ModifiersChangedEvent, window: &mut Window, cx: &mut App| {
                listener(event, window, cx)
            },
        ));
    }

    /// Register a listener to be called when the given focus handle or one of its descendants receives focus.
    /// This does not fire if the given focus handle - or one of its descendants - was previously focused.
    /// Returns a subscription and persists until the subscription is dropped.
    pub fn on_focus_in(
        &mut self,
        handle: &FocusHandle,
        cx: &mut App,
        mut listener: impl FnMut(&mut Window, &mut App) + 'static,
    ) -> Subscription {
        let focus_id = handle.id;
        let (subscription, activate) =
            self.new_focus_listener(Box::new(move |event, window, cx| {
                if event.is_focus_in(focus_id) {
                    listener(window, cx);
                }
                true
            }));
        cx.defer(move |_| activate());
        subscription
    }

    /// Register a listener to be called when the given focus handle or one of its descendants loses focus.
    /// Returns a subscription and persists until the subscription is dropped.
    pub fn on_focus_out(
        &mut self,
        handle: &FocusHandle,
        cx: &mut App,
        mut listener: impl FnMut(FocusOutEvent, &mut Window, &mut App) + 'static,
    ) -> Subscription {
        let focus_id = handle.id;
        let (subscription, activate) =
            self.new_focus_listener(Box::new(move |event, window, cx| {
                if let Some(blurred_id) = event.previous_focus_path.last().copied()
                    && event.is_focus_out(focus_id)
                {
                    let event = FocusOutEvent {
                        blurred: WeakFocusHandle {
                            id: blurred_id,
                            handles: Arc::downgrade(&cx.focus_handles),
                        },
                    };
                    listener(event, window, cx)
                }
                true
            }));
        cx.defer(move |_| activate());
        subscription
    }

    fn reset_cursor_style(&self, cx: &mut App) {
        // Set the cursor only if we're the active window.
        if self.is_window_hovered() {
            let style = self
                .rendered_frame
                .cursor_style(self)
                .unwrap_or(CursorStyle::Arrow);
            cx.platform.set_cursor_style(style);
        }
    }

    /// Dispatch a given keystroke as though the user had typed it.
    /// You can create a keystroke with Keystroke::parse("").
    pub fn dispatch_keystroke(&mut self, keystroke: Keystroke, cx: &mut App) -> bool {
        let keystroke = keystroke.with_simulated_ime();
        let result = self.dispatch_event(
            PlatformInput::KeyDown(KeyDownEvent {
                keystroke: keystroke.clone(),
                is_held: false,
                prefer_character_input: false,
            }),
            cx,
        );
        if !result.propagate {
            return true;
        }

        if let Some(input) = keystroke.key_char
            && let Some(mut input_handler) = self.platform_window.take_input_handler()
        {
            input_handler.dispatch_input(&input, self, cx);
            self.platform_window.set_input_handler(input_handler);
            return true;
        }

        false
    }

    /// Return a key binding string for an action, to display in the UI. Uses the highest precedence
    /// binding for the action (last binding added to the keymap).
    pub fn keystroke_text_for(&self, action: &dyn Action) -> String {
        self.highest_precedence_binding_for_action(action)
            .map(|binding| {
                binding
                    .keystrokes()
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .unwrap_or_else(|| action.name().to_string())
    }

    /// Dispatch a mouse or keyboard event on the window.
    #[profiling::function]
    pub fn dispatch_event(&mut self, event: PlatformInput, cx: &mut App) -> DispatchEventResult {
        #[cfg(feature = "flamegraph")]
        crate::record_input_event_dispatched();
        self.last_input_timestamp.set(Instant::now());

        // Track whether this input was keyboard-based for focus-visible styling
        self.last_input_modality = match &event {
            PlatformInput::KeyDown(_) | PlatformInput::ModifiersChanged(_) => {
                InputModality::Keyboard
            }
            PlatformInput::MouseDown(e) if e.is_focusing() => InputModality::Mouse,
            _ => self.last_input_modality,
        };

        // Handlers may set this to false by calling `stop_propagation`.
        cx.propagate_event = true;
        // Handlers may set this to true by calling `prevent_default`.
        self.default_prevented = false;

        let event = match event {
            // Track the mouse position with our own state, since accessing the platform
            // API for the mouse position can only occur on the main thread.
            PlatformInput::MouseMove(mouse_move) => {
                self.mouse_position = mouse_move.position;
                self.modifiers = mouse_move.modifiers;
                PlatformInput::MouseMove(mouse_move)
            }
            PlatformInput::MouseDown(mouse_down) => {
                if std::env::var_os("GPUI_DEBUG_MOUSE").is_some() {
                    eprintln!(
                        "[WGPUI] dispatch MouseDown @ {:?} click_count={} first_mouse={} active={}",
                        mouse_down.position,
                        mouse_down.click_count,
                        mouse_down.first_mouse,
                        self.active.get(),
                    );
                }
                self.mouse_position = mouse_down.position;
                self.modifiers = mouse_down.modifiers;
                self.mouse_button_pressed = Some(mouse_down.button);
                PlatformInput::MouseDown(mouse_down)
            }
            PlatformInput::MouseUp(mouse_up) => {
                if std::env::var_os("GPUI_DEBUG_MOUSE").is_some() {
                    eprintln!(
                        "[WGPUI] dispatch MouseUp @ {:?} click_count={} active={}",
                        mouse_up.position,
                        mouse_up.click_count,
                        self.active.get(),
                    );
                }
                self.mouse_position = mouse_up.position;
                self.modifiers = mouse_up.modifiers;
                if self.mouse_button_pressed == Some(mouse_up.button) {
                    self.mouse_button_pressed = None;
                }
                PlatformInput::MouseUp(mouse_up)
            }
            PlatformInput::MouseExited(mouse_exited) => {
                self.modifiers = mouse_exited.modifiers;
                PlatformInput::MouseExited(mouse_exited)
            }
            PlatformInput::ModifiersChanged(modifiers_changed) => {
                self.modifiers = modifiers_changed.modifiers;
                self.capslock = modifiers_changed.capslock;
                PlatformInput::ModifiersChanged(modifiers_changed)
            }
            PlatformInput::ScrollWheel(scroll_wheel) => {
                self.mouse_position = scroll_wheel.position;
                self.modifiers = scroll_wheel.modifiers;
                PlatformInput::ScrollWheel(scroll_wheel)
            }
            // Translate dragging and dropping of external files from the operating system
            // to internal drag and drop events.
            PlatformInput::FileDrop(file_drop) => match file_drop {
                FileDropEvent::Entered { position, paths } => {
                    self.mouse_position = position;
                    if cx.active_drag.is_none() {
                        cx.active_drag = Some(AnyDrag {
                            value: Arc::new(paths.clone()),
                            view: cx.new(|_| paths).into(),
                            cursor_offset: position,
                            cursor_style: None,
                        });
                    }
                    PlatformInput::MouseMove(MouseMoveEvent {
                        position,
                        pressed_button: Some(MouseButton::Left),
                        modifiers: Modifiers::default(),
                    })
                }
                FileDropEvent::Pending { position } => {
                    self.mouse_position = position;
                    PlatformInput::MouseMove(MouseMoveEvent {
                        position,
                        pressed_button: Some(MouseButton::Left),
                        modifiers: Modifiers::default(),
                    })
                }
                FileDropEvent::Submit { position } => {
                    cx.activate(true);
                    self.mouse_position = position;
                    PlatformInput::MouseUp(MouseUpEvent {
                        button: MouseButton::Left,
                        position,
                        modifiers: Modifiers::default(),
                        click_count: 1,
                    })
                }
                FileDropEvent::Exited => {
                    cx.active_drag.take();
                    PlatformInput::FileDrop(FileDropEvent::Exited)
                }
            },
            PlatformInput::KeyDown(_) | PlatformInput::KeyUp(_) => event,
        };

        if let Some(any_mouse_event) = event.mouse_event() {
            // Determine the event discriminator by checking against known event types.
            // TypeId checks here work because both the event and the check are from the
            // main binary. The resulting type_name_hash is consistent with DLL-computed
            // hashes, so DLL-registered listeners match correctly.
            let discriminator: u64 = if any_mouse_event.is::<MouseDownEvent>() {
                type_name_hash::<MouseDownEvent>()
            } else if any_mouse_event.is::<MouseUpEvent>() {
                type_name_hash::<MouseUpEvent>()
            } else if any_mouse_event.is::<MouseMoveEvent>() {
                type_name_hash::<MouseMoveEvent>()
            } else if any_mouse_event.is::<ScrollWheelEvent>() {
                type_name_hash::<ScrollWheelEvent>()
            } else if any_mouse_event.is::<MouseExitEvent>() {
                type_name_hash::<MouseExitEvent>()
            } else {
                0
            };
            self.dispatch_mouse_event(any_mouse_event, discriminator, cx);
        } else if let Some(any_key_event) = event.keyboard_event() {
            self.dispatch_key_event(any_key_event, cx);
        }

        DispatchEventResult {
            propagate: cx.propagate_event,
            default_prevented: self.default_prevented,
        }
    }

    fn dispatch_mouse_event(&mut self, event: &dyn Any, discriminator: u64, cx: &mut App) {
        let hit_test = self
            .rendered_frame
            .hit_test(self.mouse_position(), &self.layers);
        if std::env::var_os("GPUI_DEBUG_MOUSE").is_some() {
            eprintln!(
                "[WGPUI] dispatch_mouse_event pos={:?} hit_ids={:?} listeners={} active={} hover={}",
                self.mouse_position(),
                hit_test.ids,
                self.rendered_frame.mouse_listeners.len(),
                self.active.get(),
                self.hovered.get(),
            );
        }
        if hit_test != self.mouse_hit_test {
            self.mouse_hit_test = hit_test;
            self.reset_cursor_style(cx);
        }

        #[cfg(any(feature = "inspector", debug_assertions))]
        {
            // Handle inspector resize dragging
            if let Some(inspector) = &self.inspector {
                let is_resizing = inspector.read(cx).is_resizing();
                if is_resizing {
                    let viewport_width = self.viewport_size.width;
                    let handled = inspector.update(cx, |inspector, _cx| {
                        crate::handle_inspector_resize(inspector, event, viewport_width)
                    });
                    if handled {
                        self.refresh();
                        return;
                    }
                }
            }

            if self.is_inspector_picking(cx) {
                self.handle_inspector_mouse_event(event, cx);
                // When inspector is picking, all other mouse handling is skipped.
                return;
            }
        }

        let mut mouse_listeners = mem::take(&mut self.rendered_frame.mouse_listeners);

        // Capture phase, events bubble from back to front.
        // Only listeners whose discriminator matches the dispatched event type are called.
        // The discriminator (type_name_hash) is consistent across compilation units,
        // so DLL-registered listeners match correctly.
        for entry in &mut mouse_listeners {
            if let Some(entry) = entry {
                if entry.discriminator == discriminator {
                    (entry.listener)(event, DispatchPhase::Capture, self, cx);
                    if !cx.propagate_event {
                        break;
                    }
                }
            }
        }

        // Bubble phase, where most normal handlers do their work.
        if cx.propagate_event {
            for entry in mouse_listeners.iter_mut().rev() {
                if let Some(entry) = entry {
                    if entry.discriminator == discriminator {
                        (entry.listener)(event, DispatchPhase::Bubble, self, cx);
                        if !cx.propagate_event {
                            break;
                        }
                    }
                }
            }
        }

        self.rendered_frame.mouse_listeners = mouse_listeners;

        if cx.has_active_drag() {
            if event.is::<MouseMoveEvent>() {
                // If this was a mouse move event, redraw the window so that the
                // active drag can follow the mouse cursor.
                self.refresh();
            } else if event.is::<MouseUpEvent>() {
                // If this was a mouse up event, cancel the active drag and redraw
                // the window.
                cx.active_drag = None;
                self.refresh();
            }
        }
    }

    fn dispatch_key_event(&mut self, event: &dyn Any, cx: &mut App) {
        if self.invalidator.is_dirty() {
            self.draw(cx).clear();
        }

        let node_id = self.focus_node_id_in_rendered_frame(self.focus);
        let dispatch_path = self.rendered_frame.dispatch_tree.dispatch_path(node_id);

        let mut keystroke: Option<Keystroke> = None;

        if let Some(event) = event.downcast_ref::<ModifiersChangedEvent>() {
            if event.modifiers.number_of_modifiers() == 0
                && self.pending_modifier.modifiers.number_of_modifiers() == 1
                && !self.pending_modifier.saw_keystroke
            {
                let key = match self.pending_modifier.modifiers {
                    modifiers if modifiers.shift => Some("shift"),
                    modifiers if modifiers.control => Some("control"),
                    modifiers if modifiers.alt => Some("alt"),
                    modifiers if modifiers.platform => Some("platform"),
                    modifiers if modifiers.function => Some("function"),
                    _ => None,
                };
                if let Some(key) = key {
                    keystroke = Some(Keystroke {
                        key: key.to_string(),
                        key_char: None,
                        modifiers: Modifiers::default(),
                    });
                }
            }

            if self.pending_modifier.modifiers.number_of_modifiers() == 0
                && event.modifiers.number_of_modifiers() == 1
            {
                self.pending_modifier.saw_keystroke = false
            }
            self.pending_modifier.modifiers = event.modifiers
        } else if let Some(key_down_event) = event.downcast_ref::<KeyDownEvent>() {
            self.pending_modifier.saw_keystroke = true;
            keystroke = Some(key_down_event.keystroke.clone());
        }

        let Some(keystroke) = keystroke else {
            self.finish_dispatch_key_event(event, dispatch_path, self.context_stack(), cx);
            return;
        };

        cx.propagate_event = true;
        self.dispatch_keystroke_interceptors(event, self.context_stack(), cx);
        if !cx.propagate_event {
            self.finish_dispatch_key_event(event, dispatch_path, self.context_stack(), cx);
            return;
        }

        let mut currently_pending = self.pending_input.take().unwrap_or_default();
        if currently_pending.focus.is_some() && currently_pending.focus != self.focus {
            currently_pending = PendingInput::default();
        }

        let match_result = self.rendered_frame.dispatch_tree.dispatch_key(
            currently_pending.keystrokes,
            keystroke,
            &dispatch_path,
        );

        if !match_result.to_replay.is_empty() {
            self.replay_pending_input(match_result.to_replay, cx);
            cx.propagate_event = true;
        }

        if !match_result.pending.is_empty() {
            currently_pending.timer.take();
            currently_pending.keystrokes = match_result.pending;
            currently_pending.focus = self.focus;

            let text_input_requires_timeout = event
                .downcast_ref::<KeyDownEvent>()
                .filter(|key_down| key_down.keystroke.key_char.is_some())
                .and_then(|_| self.platform_window.take_input_handler())
                .map_or(false, |mut input_handler| {
                    let accepts = input_handler.accepts_text_input(self, cx);
                    self.platform_window.set_input_handler(input_handler);
                    accepts
                });

            currently_pending.needs_timeout |=
                match_result.pending_has_binding || text_input_requires_timeout;

            if currently_pending.needs_timeout {
                currently_pending.timer = Some(self.spawn(cx, async move |cx| {
                    cx.background_executor.timer(Duration::from_secs(1)).await;
                    cx.update(move |window, cx| {
                        let Some(currently_pending) = window
                            .pending_input
                            .take()
                            .filter(|pending| pending.focus == window.focus)
                        else {
                            return;
                        };

                        let node_id = window.focus_node_id_in_rendered_frame(window.focus);
                        let dispatch_path =
                            window.rendered_frame.dispatch_tree.dispatch_path(node_id);

                        let to_replay = window
                            .rendered_frame
                            .dispatch_tree
                            .flush_dispatch(currently_pending.keystrokes, &dispatch_path);

                        window.pending_input_changed(cx);
                        window.replay_pending_input(to_replay, cx)
                    })
                    .log_err();
                }));
            } else {
                currently_pending.timer = None;
            }
            self.pending_input = Some(currently_pending);
            self.pending_input_changed(cx);
            cx.propagate_event = false;
            return;
        }

        let skip_bindings = event
            .downcast_ref::<KeyDownEvent>()
            .filter(|key_down_event| key_down_event.prefer_character_input)
            .map(|_| {
                self.platform_window
                    .take_input_handler()
                    .map_or(false, |mut input_handler| {
                        let accepts = input_handler.accepts_text_input(self, cx);
                        self.platform_window.set_input_handler(input_handler);
                        // If modifiers are not excessive (e.g. AltGr), and the input handler is accepting text input,
                        // we prefer the text input over bindings.
                        accepts
                    })
            })
            .unwrap_or(false);

        if !skip_bindings {
            for binding in match_result.bindings {
                self.dispatch_action_on_node(node_id, binding.action.as_ref(), cx);
                if !cx.propagate_event {
                    self.dispatch_keystroke_observers(
                        event,
                        Some(binding.action),
                        match_result.context_stack,
                        cx,
                    );
                    self.pending_input_changed(cx);
                    return;
                }
            }
        }

        self.finish_dispatch_key_event(event, dispatch_path, match_result.context_stack, cx);
        self.pending_input_changed(cx);
    }

    fn finish_dispatch_key_event(
        &mut self,
        event: &dyn Any,
        dispatch_path: SmallVec<[DispatchNodeId; 32]>,
        context_stack: Vec<KeyContext>,
        cx: &mut App,
    ) {
        self.dispatch_key_down_up_event(event, &dispatch_path, cx);
        if !cx.propagate_event {
            return;
        }

        self.dispatch_modifiers_changed_event(event, &dispatch_path, cx);
        if !cx.propagate_event {
            return;
        }

        self.dispatch_keystroke_observers(event, None, context_stack, cx);
    }

    fn pending_input_changed(&mut self, cx: &mut App) {
        self.pending_input_observers
            .clone()
            .retain(&(), |callback| callback(self, cx));
    }

    fn dispatch_key_down_up_event(
        &mut self,
        event: &dyn Any,
        dispatch_path: &SmallVec<[DispatchNodeId; 32]>,
        cx: &mut App,
    ) {
        // Compute cross-DLL discriminator from event type. TypeId checks work here
        // because both the event and checks are from the main binary.
        let discriminator: u64 = if event.is::<KeyDownEvent>() {
            type_name_hash::<KeyDownEvent>()
        } else if event.is::<KeyUpEvent>() {
            type_name_hash::<KeyUpEvent>()
        } else if event.is::<ModifiersChangedEvent>() {
            type_name_hash::<ModifiersChangedEvent>()
        } else {
            0
        };

        // Capture phase
        for node_id in dispatch_path {
            let listeners = self
                .rendered_frame
                .dispatch_tree
                .node(*node_id)
                .key_listeners
                .clone();

            for (listener_disc, key_listener) in &listeners {
                if *listener_disc == discriminator {
                    key_listener(event, DispatchPhase::Capture, self, cx);
                    if !cx.propagate_event {
                        return;
                    }
                }
            }
        }

        // Bubble phase
        for node_id in dispatch_path.iter().rev() {
            let listeners = self
                .rendered_frame
                .dispatch_tree
                .node(*node_id)
                .key_listeners
                .clone();

            for (listener_disc, key_listener) in &listeners {
                if *listener_disc == discriminator {
                    key_listener(event, DispatchPhase::Bubble, self, cx);
                    if !cx.propagate_event {
                        return;
                    }
                }
            }
        }
    }

    fn dispatch_modifiers_changed_event(
        &mut self,
        event: &dyn Any,
        dispatch_path: &SmallVec<[DispatchNodeId; 32]>,
        cx: &mut App,
    ) {
        let Some(event) = event.downcast_ref::<ModifiersChangedEvent>() else {
            return;
        };
        for node_id in dispatch_path.iter().rev() {
            let node = self.rendered_frame.dispatch_tree.node(*node_id);
            for listener in node.modifiers_changed_listeners.clone() {
                listener(event, self, cx);
                if !cx.propagate_event {
                    return;
                }
            }
        }
    }

    /// Determine whether a potential multi-stroke key binding is in progress on this window.
    pub fn has_pending_keystrokes(&self) -> bool {
        self.pending_input.is_some()
    }

    pub(crate) fn clear_pending_keystrokes(&mut self) {
        self.pending_input.take();
    }

    /// Returns the currently pending input keystrokes that might result in a multi-stroke key binding.
    pub fn pending_input_keystrokes(&self) -> Option<&[Keystroke]> {
        self.pending_input
            .as_ref()
            .map(|pending_input| pending_input.keystrokes.as_slice())
    }

    fn replay_pending_input(&mut self, replays: SmallVec<[Replay; 1]>, cx: &mut App) {
        let node_id = self.focus_node_id_in_rendered_frame(self.focus);
        let dispatch_path = self.rendered_frame.dispatch_tree.dispatch_path(node_id);

        'replay: for replay in replays {
            let event = KeyDownEvent {
                keystroke: replay.keystroke.clone(),
                is_held: false,
                prefer_character_input: true,
            };

            cx.propagate_event = true;
            for binding in replay.bindings {
                self.dispatch_action_on_node(node_id, binding.action.as_ref(), cx);
                if !cx.propagate_event {
                    self.dispatch_keystroke_observers(
                        &event,
                        Some(binding.action),
                        Vec::default(),
                        cx,
                    );
                    continue 'replay;
                }
            }

            self.dispatch_key_down_up_event(&event, &dispatch_path, cx);
            if !cx.propagate_event {
                continue 'replay;
            }
            if let Some(input) = replay.keystroke.key_char.as_ref().cloned()
                && let Some(mut input_handler) = self.platform_window.take_input_handler()
            {
                input_handler.dispatch_input(&input, self, cx);
                self.platform_window.set_input_handler(input_handler)
            }
        }
    }

    fn focus_node_id_in_rendered_frame(&self, focus_id: Option<FocusId>) -> DispatchNodeId {
        focus_id
            .and_then(|focus_id| {
                self.rendered_frame
                    .dispatch_tree
                    .focusable_node_id(focus_id)
            })
            .unwrap_or_else(|| self.rendered_frame.dispatch_tree.root_node_id())
    }

    fn dispatch_action_on_node(
        &mut self,
        node_id: DispatchNodeId,
        action: &dyn Action,
        cx: &mut App,
    ) {
        let action_registry = cx.actions.clone();
        let dispatch_path = self.rendered_frame.dispatch_tree.dispatch_path(node_id);

        // Capture phase for global actions.
        cx.propagate_event = true;
        if let Some(mut global_listeners) = cx
            .global_action_listeners
            .remove(&action.as_any().type_id())
        {
            for listener in &global_listeners {
                listener(action.as_any(), DispatchPhase::Capture, cx);
                if !cx.propagate_event {
                    break;
                }
            }

            global_listeners.extend(
                cx.global_action_listeners
                    .remove(&action.as_any().type_id())
                    .unwrap_or_default(),
            );

            cx.global_action_listeners
                .insert(action.as_any().type_id(), global_listeners);
        }

        if !cx.propagate_event {
            return;
        }

        // Capture phase for window actions.
        for node_id in &dispatch_path {
            let node = self.rendered_frame.dispatch_tree.node(*node_id);
            let action_type_id = action.as_any().type_id();
            let action_disc = action_registry.discriminator_for_type(&action_type_id);
            for DispatchActionListener {
                action_type,
                action_discriminator,
                listener,
            } in node.action_listeners.clone()
            {
                let any_action = action.as_any();
                if action_type == action_type_id || action_discriminator == action_disc {
                    listener(any_action, DispatchPhase::Capture, self, cx);

                    if !cx.propagate_event {
                        return;
                    }
                }
            }
        }

        // Bubble phase for window actions.
        for node_id in dispatch_path.iter().rev() {
            let node = self.rendered_frame.dispatch_tree.node(*node_id);
            let action_type_id = action.as_any().type_id();
            let action_disc = action_registry.discriminator_for_type(&action_type_id);
            for DispatchActionListener {
                action_type,
                action_discriminator,
                listener,
            } in node.action_listeners.clone()
            {
                let any_action = action.as_any();
                if action_type == action_type_id || action_discriminator == action_disc {
                    cx.propagate_event = false; // Actions stop propagation by default during the bubble phase
                    listener(any_action, DispatchPhase::Bubble, self, cx);

                    if !cx.propagate_event {
                        return;
                    }
                }
            }
        }

        // Bubble phase for global actions.
        if let Some(mut global_listeners) = cx
            .global_action_listeners
            .remove(&action.as_any().type_id())
        {
            for listener in global_listeners.iter().rev() {
                cx.propagate_event = false; // Actions stop propagation by default during the bubble phase

                listener(action.as_any(), DispatchPhase::Bubble, cx);
                if !cx.propagate_event {
                    break;
                }
            }

            global_listeners.extend(
                cx.global_action_listeners
                    .remove(&action.as_any().type_id())
                    .unwrap_or_default(),
            );

            cx.global_action_listeners
                .insert(action.as_any().type_id(), global_listeners);
        }
    }

    /// Register the given handler to be invoked whenever the global of the given type
    /// is updated.
    pub fn observe_global<G: Global>(
        &mut self,
        cx: &mut App,
        f: impl Fn(&mut Window, &mut App) + 'static,
    ) -> Subscription {
        let window_handle = self.handle;
        let (subscription, activate) = cx.global_observers.insert(
            TypeId::of::<G>(),
            Box::new(move |cx| {
                window_handle
                    .update(cx, |_, window, cx| f(window, cx))
                    .is_ok()
            }),
        );
        cx.defer(move |_| activate());
        subscription
    }

    /// Focus the current window and bring it to the foreground at the platform level.
    pub fn activate_window(&self) {
        self.platform_window.activate();
    }

    /// Minimize the current window at the platform level.
    pub fn minimize_window(&self) {
        self.platform_window.minimize();
    }

    /// Toggle full screen status on the current window at the platform level.
    pub fn toggle_fullscreen(&self) {
        self.platform_window.toggle_fullscreen();
    }

    /// Updates the IME panel position suggestions for languages like japanese, chinese.
    pub fn invalidate_character_coordinates(&self) {
        self.on_next_frame(|window, cx| {
            if let Some(mut input_handler) = window.platform_window.take_input_handler() {
                if let Some(bounds) = input_handler.selected_bounds(window, cx) {
                    window.platform_window.update_ime_position(bounds);
                }
                window.platform_window.set_input_handler(input_handler);
            }
        });
    }

    /// Present a platform dialog.
    /// The provided message will be presented, along with buttons for each answer.
    /// When a button is clicked, the returned Receiver will receive the index of the clicked button.
    pub fn prompt<T>(
        &mut self,
        level: PromptLevel,
        message: &str,
        detail: Option<&str>,
        answers: &[T],
        cx: &mut App,
    ) -> oneshot::Receiver<usize>
    where
        T: Clone + Into<PromptButton>,
    {
        let prompt_builder = cx.prompt_builder.take();
        let Some(prompt_builder) = prompt_builder else {
            unreachable!("Re-entrant window prompting is not supported by GPUI");
        };

        let answers = answers
            .iter()
            .map(|answer| answer.clone().into())
            .collect::<Vec<_>>();

        let receiver = match &prompt_builder {
            PromptBuilder::Default => self
                .platform_window
                .prompt(level, message, detail, &answers)
                .unwrap_or_else(|| {
                    self.build_custom_prompt(&prompt_builder, level, message, detail, &answers, cx)
                }),
            PromptBuilder::Custom(_) => {
                self.build_custom_prompt(&prompt_builder, level, message, detail, &answers, cx)
            }
        };

        cx.prompt_builder = Some(prompt_builder);

        receiver
    }

    fn build_custom_prompt(
        &mut self,
        prompt_builder: &PromptBuilder,
        level: PromptLevel,
        message: &str,
        detail: Option<&str>,
        answers: &[PromptButton],
        cx: &mut App,
    ) -> oneshot::Receiver<usize> {
        let (sender, receiver) = oneshot::channel();
        let handle = PromptHandle::new(sender);
        let handle = (prompt_builder)(level, message, detail, answers, handle, self, cx);
        self.prompt = Some(handle);
        receiver
    }

    /// Returns the current context stack.
    pub fn context_stack(&self) -> Vec<KeyContext> {
        let node_id = self.focus_node_id_in_rendered_frame(self.focus);
        let dispatch_tree = &self.rendered_frame.dispatch_tree;
        dispatch_tree
            .dispatch_path(node_id)
            .iter()
            .filter_map(move |&node_id| dispatch_tree.node(node_id).context.clone())
            .collect()
    }

    /// Returns all available actions for the focused element.
    pub fn available_actions(&self, cx: &App) -> Vec<Box<dyn Action>> {
        let node_id = self.focus_node_id_in_rendered_frame(self.focus);
        let mut actions = self.rendered_frame.dispatch_tree.available_actions(node_id);
        for action_type in cx.global_action_listeners.keys() {
            if let Err(ix) = actions.binary_search_by_key(action_type, |a| a.as_any().type_id()) {
                let action = cx.actions.build_action_type(action_type).ok();
                if let Some(action) = action {
                    actions.insert(ix, action);
                }
            }
        }
        actions
    }

    /// Returns key bindings that invoke an action on the currently focused element. Bindings are
    /// returned in the order they were added. For display, the last binding should take precedence.
    pub fn bindings_for_action(&self, action: &dyn Action) -> Vec<KeyBinding> {
        self.rendered_frame
            .dispatch_tree
            .bindings_for_action(action, &self.rendered_frame.dispatch_tree.context_stack)
    }

    /// Returns the highest precedence key binding that invokes an action on the currently focused
    /// element. This is more efficient than getting the last result of `bindings_for_action`.
    pub fn highest_precedence_binding_for_action(&self, action: &dyn Action) -> Option<KeyBinding> {
        self.rendered_frame
            .dispatch_tree
            .highest_precedence_binding_for_action(
                action,
                &self.rendered_frame.dispatch_tree.context_stack,
            )
    }

    /// Returns the key bindings for an action in a context.
    pub fn bindings_for_action_in_context(
        &self,
        action: &dyn Action,
        context: KeyContext,
    ) -> Vec<KeyBinding> {
        let dispatch_tree = &self.rendered_frame.dispatch_tree;
        dispatch_tree.bindings_for_action(action, &[context])
    }

    /// Returns the highest precedence key binding for an action in a context. This is more
    /// efficient than getting the last result of `bindings_for_action_in_context`.
    pub fn highest_precedence_binding_for_action_in_context(
        &self,
        action: &dyn Action,
        context: KeyContext,
    ) -> Option<KeyBinding> {
        let dispatch_tree = &self.rendered_frame.dispatch_tree;
        dispatch_tree.highest_precedence_binding_for_action(action, &[context])
    }

    /// Returns any bindings that would invoke an action on the given focus handle if it were
    /// focused. Bindings are returned in the order they were added. For display, the last binding
    /// should take precedence.
    pub fn bindings_for_action_in(
        &self,
        action: &dyn Action,
        focus_handle: &FocusHandle,
    ) -> Vec<KeyBinding> {
        let dispatch_tree = &self.rendered_frame.dispatch_tree;
        let Some(context_stack) = self.context_stack_for_focus_handle(focus_handle) else {
            return vec![];
        };
        dispatch_tree.bindings_for_action(action, &context_stack)
    }

    /// Returns the highest precedence key binding that would invoke an action on the given focus
    /// handle if it were focused. This is more efficient than getting the last result of
    /// `bindings_for_action_in`.
    pub fn highest_precedence_binding_for_action_in(
        &self,
        action: &dyn Action,
        focus_handle: &FocusHandle,
    ) -> Option<KeyBinding> {
        let dispatch_tree = &self.rendered_frame.dispatch_tree;
        let context_stack = self.context_stack_for_focus_handle(focus_handle)?;
        dispatch_tree.highest_precedence_binding_for_action(action, &context_stack)
    }

    fn context_stack_for_focus_handle(
        &self,
        focus_handle: &FocusHandle,
    ) -> Option<Vec<KeyContext>> {
        let dispatch_tree = &self.rendered_frame.dispatch_tree;
        let node_id = dispatch_tree.focusable_node_id(focus_handle.id)?;
        let context_stack: Vec<_> = dispatch_tree
            .dispatch_path(node_id)
            .into_iter()
            .filter_map(|node_id| dispatch_tree.node(node_id).context.clone())
            .collect();
        Some(context_stack)
    }

    /// Returns a generic event listener that invokes the given listener with the view and context associated with the given view handle.
    pub fn listener_for<T: 'static, E>(
        &self,
        view: &Entity<T>,
        f: impl Fn(&mut T, &E, &mut Window, &mut Context<T>) + 'static,
    ) -> impl Fn(&E, &mut Window, &mut App) + 'static {
        let view = view.downgrade();
        move |e: &E, window: &mut Window, cx: &mut App| {
            view.update(cx, |view, cx| f(view, e, window, cx)).ok();
        }
    }

    /// Returns a generic handler that invokes the given handler with the view and context associated with the given view handle.
    pub fn handler_for<E: 'static, Callback: Fn(&mut E, &mut Window, &mut Context<E>) + 'static>(
        &self,
        entity: &Entity<E>,
        f: Callback,
    ) -> impl Fn(&mut Window, &mut App) + 'static {
        let entity = entity.downgrade();
        move |window: &mut Window, cx: &mut App| {
            entity.update(cx, |entity, cx| f(entity, window, cx)).ok();
        }
    }

    /// Register a callback that can interrupt the closing of the current window based the returned boolean.
    /// If the callback returns false, the window won't be closed.
    pub fn on_window_should_close(
        &self,
        cx: &App,
        f: impl Fn(&mut Window, &mut App) -> bool + 'static,
    ) {
        let mut cx = self.to_async(cx);
        self.platform_window.on_should_close(Box::new(move || {
            cx.update(|window, cx| f(window, cx)).unwrap_or(true)
        }))
    }

    /// Register an action listener on this node for the next frame. The type of action
    /// is determined by the first parameter of the given listener. When the next frame is rendered
    /// the listener will be cleared.
    ///
    /// This is a fairly low-level method, so prefer using action handlers on elements unless you have
    /// a specific need to register a listener yourself.
    ///
    /// This method should only be called as part of the paint phase of element drawing.
    pub fn on_action(
        &mut self,
        action_type: TypeId,
        action_discriminator: u64,
        listener: impl Fn(&dyn Any, DispatchPhase, &mut Window, &mut App) + 'static,
    ) {
        self.invalidator.debug_assert_paint();

        self.next_frame.dispatch_tree.on_action(
            action_type,
            action_discriminator,
            Rc::new(listener),
        );
    }

    /// Register a capturing action listener on this node for the next frame if the condition is true.
    /// The type of action is determined by the first parameter of the given listener. When the next
    /// frame is rendered the listener will be cleared.
    ///
    /// This is a fairly low-level method, so prefer using action handlers on elements unless you have
    /// a specific need to register a listener yourself.
    ///
    /// This method should only be called as part of the paint phase of element drawing.
    pub fn on_action_when(
        &mut self,
        condition: bool,
        action_type: TypeId,
        action_discriminator: u64,
        listener: impl Fn(&dyn Any, DispatchPhase, &mut Window, &mut App) + 'static,
    ) {
        self.invalidator.debug_assert_paint();

        if condition {
            self.next_frame.dispatch_tree.on_action(
                action_type,
                action_discriminator,
                Rc::new(listener),
            );
        }
    }

    /// Read information about the GPU backing this window.
    /// Currently returns None on Mac and Windows.
    pub fn gpu_specs(&self) -> Option<GpuSpecs> {
        self.platform_window.gpu_specs()
    }

    /// Perform titlebar double-click action.
    /// This is macOS specific.
    pub fn titlebar_double_click(&self) {
        self.platform_window.titlebar_double_click();
    }

    /// Gets the window's title at the platform level.
    /// This is macOS specific.
    pub fn window_title(&self) -> String {
        self.platform_window.get_title()
    }

    /// Returns a list of all tabbed windows and their titles.
    /// This is macOS specific.
    pub fn tabbed_windows(&self) -> Option<Vec<SystemWindowTab>> {
        self.platform_window.tabbed_windows()
    }

    /// Returns the tab bar visibility.
    /// This is macOS specific.
    pub fn tab_bar_visible(&self) -> bool {
        self.platform_window.tab_bar_visible()
    }

    /// Merges all open windows into a single tabbed window.
    /// This is macOS specific.
    pub fn merge_all_windows(&self) {
        self.platform_window.merge_all_windows()
    }

    /// Moves the tab to a new containing window.
    /// This is macOS specific.
    pub fn move_tab_to_new_window(&self) {
        self.platform_window.move_tab_to_new_window()
    }

    /// Shows or hides the window tab overview.
    /// This is macOS specific.
    pub fn toggle_window_tab_overview(&self) {
        self.platform_window.toggle_window_tab_overview()
    }

    /// Sets the tabbing identifier for the window.
    /// This is macOS specific.
    pub fn set_tabbing_identifier(&self, tabbing_identifier: Option<String>) {
        self.platform_window
            .set_tabbing_identifier(tabbing_identifier)
    }

    /// Toggles the inspector mode on this window.
    #[cfg(any(feature = "inspector", debug_assertions))]
    pub fn toggle_inspector(&mut self, cx: &mut App) {
        self.inspector = match self.inspector {
            None => {
                let rem_size = self.rem_size();
                let inspector = cx.new(|_| Inspector::new());
                inspector.update(cx, |inspector, _cx| {
                    inspector.init_panel_width(rem_size);
                });
                Some(inspector)
            }
            Some(_) => None,
        };
        self.refresh();
    }

    /// Returns true if the window is in inspector mode.
    pub fn is_inspector_picking(&self, _cx: &App) -> bool {
        #[cfg(any(feature = "inspector", debug_assertions))]
        {
            if let Some(inspector) = &self.inspector {
                return inspector.read(_cx).is_picking();
            }
        }
        false
    }

    /// Executes the provided function with mutable access to an inspector state.
    #[cfg(any(feature = "inspector", debug_assertions))]
    pub fn with_inspector_state<T: 'static, R>(
        &mut self,
        _inspector_id: Option<&crate::InspectorElementId>,
        cx: &mut App,
        f: impl FnOnce(&mut Option<T>, &mut Self) -> R,
    ) -> R {
        if let Some(inspector_id) = _inspector_id
            && let Some(inspector) = &self.inspector
        {
            let inspector = inspector.clone();
            let active_element_id = inspector.read(cx).active_element_id();
            if Some(inspector_id) == active_element_id {
                return inspector.update(cx, |inspector, _cx| {
                    inspector.with_active_element_state(self, f)
                });
            }
        }
        f(&mut None, self)
    }

    #[cfg(any(feature = "inspector", debug_assertions, feature = "flamegraph"))]
    pub(crate) fn build_inspector_element_id(
        &mut self,
        path: crate::InspectorElementPath,
    ) -> crate::InspectorElementId {
        self.invalidator.debug_assert_paint_or_prepaint();
        let path = Rc::new(path);
        let next_instance_id = self
            .next_frame
            .next_inspector_instance_ids
            .entry(path.clone())
            .or_insert(0);
        let instance_id = *next_instance_id;
        *next_instance_id += 1;
        crate::InspectorElementId { path, instance_id }
    }

    #[cfg(any(feature = "inspector", debug_assertions))]
    fn prepaint_inspector(&mut self, inspector_width: Pixels, cx: &mut App) -> Option<AnyElement> {
        if let Some(inspector) = self.inspector.take() {
            let mut inspector_element = AnyView::from(inspector.clone()).into_any_element();
            inspector_element.prepaint_as_root(
                point(self.viewport_size.width - inspector_width, px(0.0)),
                size(inspector_width, self.viewport_size.height).into(),
                self,
                cx,
            );
            self.inspector = Some(inspector);
            Some(inspector_element)
        } else {
            None
        }
    }

    #[cfg(any(feature = "inspector", debug_assertions))]
    fn paint_inspector(&mut self, mut inspector_element: Option<AnyElement>, cx: &mut App) {
        if let Some(mut inspector_element) = inspector_element {
            inspector_element.paint(self, cx);
        };
    }

    /// Registers a hitbox that can be used for inspector picking mode, allowing users to select and
    /// inspect UI elements by clicking on them.
    #[cfg(any(feature = "inspector", debug_assertions))]
    pub fn insert_inspector_hitbox(
        &mut self,
        hitbox_id: HitboxId,
        inspector_id: Option<&crate::InspectorElementId>,
        cx: &App,
    ) {
        self.invalidator.debug_assert_paint_or_prepaint();
        if !self.is_inspector_picking(cx) {
            return;
        }
        if let Some(inspector_id) = inspector_id {
            self.next_frame
                .inspector_hitboxes
                .insert(hitbox_id, inspector_id.clone());
        }
    }

    /// Register an event listener for an element in the inspector.
    #[cfg(any(feature = "inspector", debug_assertions))]
    pub fn register_inspector_event_listener(
        &mut self,
        global_id: &crate::GlobalElementId,
        event_type: SharedString,
        location: SharedString,
    ) {
        self.invalidator.debug_assert_paint_or_prepaint();
        self.next_frame
            .inspector_event_listeners
            .entry(global_id.clone())
            .or_default()
            .push(crate::InspectorEventListener {
                event_type,
                location,
            });
    }

    /// Register an element state for the inspector display.
    #[cfg(any(feature = "inspector", debug_assertions))]
    pub fn register_inspector_element_state(
        &mut self,
        global_id: &crate::GlobalElementId,
        state: SharedString,
    ) {
        self.invalidator.debug_assert_paint_or_prepaint();
        self.next_frame
            .inspector_element_states
            .entry(global_id.clone())
            .or_default()
            .push(state);
    }

    #[cfg(any(feature = "inspector", debug_assertions))]
    fn paint_inspector_hitbox(&mut self, cx: &App) {
        if let Some(inspector) = self.inspector.as_ref() {
            let inspector = inspector.read(cx);
            if let Some((hitbox_id, _)) = self.hovered_inspector_hitbox(inspector, &self.next_frame)
                && let Some(hitbox) = self
                    .next_frame
                    .hitboxes
                    .iter()
                    .find(|hitbox| hitbox.id == hitbox_id)
            {
                self.paint_quad(crate::fill(hitbox.bounds, crate::rgba(0x61afef4d)));
            }
        }
    }

    #[cfg(any(feature = "inspector", debug_assertions))]
    fn handle_inspector_mouse_event(&mut self, event: &dyn Any, cx: &mut App) {
        let Some(inspector) = self.inspector.clone() else {
            return;
        };
        if event.downcast_ref::<MouseMoveEvent>().is_some() {
            inspector.update(cx, |inspector, _cx| {
                if let Some((_, inspector_id)) =
                    self.hovered_inspector_hitbox(inspector, &self.rendered_frame)
                {
                    inspector.hover(inspector_id, self);
                }
            });
        } else if event.downcast_ref::<crate::MouseDownEvent>().is_some() {
            inspector.update(cx, |inspector, _cx| {
                if let Some((_, inspector_id)) =
                    self.hovered_inspector_hitbox(inspector, &self.rendered_frame)
                {
                    inspector.select(inspector_id, self);
                }
            });
        } else if let Some(event) = event.downcast_ref::<crate::ScrollWheelEvent>() {
            // This should be kept in sync with SCROLL_LINES in x11 platform.
            const SCROLL_LINES: f32 = 3.0;
            const SCROLL_PIXELS_PER_LAYER: f32 = 36.0;
            let delta_y = event
                .delta
                .pixel_delta(px(SCROLL_PIXELS_PER_LAYER / SCROLL_LINES))
                .y;
            if let Some(inspector) = self.inspector.clone() {
                inspector.update(cx, |inspector, _cx| {
                    if let Some(depth) = inspector.pick_depth.as_mut() {
                        *depth += f32::from(delta_y) / SCROLL_PIXELS_PER_LAYER;
                        let max_depth = self.mouse_hit_test.ids.len() as f32 - 0.5;
                        if *depth < 0.0 {
                            *depth = 0.0;
                        } else if *depth > max_depth {
                            *depth = max_depth;
                        }
                        if let Some((_, inspector_id)) =
                            self.hovered_inspector_hitbox(inspector, &self.rendered_frame)
                        {
                            inspector.set_active_element_id(inspector_id, self);
                        }
                    }
                });
            }
        }
    }

    #[cfg(any(feature = "inspector", debug_assertions))]
    fn hovered_inspector_hitbox(
        &self,
        inspector: &Inspector,
        frame: &Frame,
    ) -> Option<(HitboxId, crate::InspectorElementId)> {
        if let Some(pick_depth) = inspector.pick_depth {
            let depth = (pick_depth as i64).try_into().unwrap_or(0);
            let max_skipped = self.mouse_hit_test.ids.len().saturating_sub(1);
            let skip_count = (depth as usize).min(max_skipped);
            for hitbox_id in self.mouse_hit_test.ids.iter().skip(skip_count) {
                if let Some(inspector_id) = frame.inspector_hitboxes.get(hitbox_id) {
                    return Some((*hitbox_id, inspector_id.clone()));
                }
            }
        }
        None
    }

    /// For testing: set the current modifier keys state.
    /// This does not generate any events.
    #[cfg(any(test, feature = "test-support"))]
    pub fn set_modifiers(&mut self, modifiers: Modifiers) {
        self.modifiers = modifiers;
    }
}

// #[derive(Clone, Copy, Eq, PartialEq, Hash)]
slotmap::new_key_type! {
    /// A unique identifier for a window.
    pub struct WindowId;
}

impl WindowId {
    /// Converts this window ID to a `u64`.
    pub fn as_u64(&self) -> u64 {
        self.0.as_ffi()
    }
}

impl From<u64> for WindowId {
    fn from(value: u64) -> Self {
        WindowId(slotmap::KeyData::from_ffi(value))
    }
}

/// A handle to a window with a specific root view type.
/// Note that this does not keep the window alive on its own.
#[derive(Deref, DerefMut)]
pub struct WindowHandle<V> {
    #[deref]
    #[deref_mut]
    pub(crate) any_handle: AnyWindowHandle,
    state_type: PhantomData<fn(V) -> V>,
}

impl<V> Debug for WindowHandle<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WindowHandle")
            .field("any_handle", &self.any_handle.id.as_u64())
            .finish()
    }
}

impl<V: 'static + Render> WindowHandle<V> {
    /// Creates a new handle from a window ID.
    /// This does not check if the root type of the window is `V`.
    pub fn new(id: WindowId) -> Self {
        WindowHandle {
            any_handle: AnyWindowHandle {
                id,
                state_type: TypeId::of::<V>(),
            },
            state_type: PhantomData,
        }
    }

    /// Get the root view out of this window.
    ///
    /// This will fail if the window is closed or if the root view's type does not match `V`.
    #[cfg(any(test, feature = "test-support"))]
    pub fn root<C>(&self, cx: &mut C) -> Result<Entity<V>>
    where
        C: AppContext,
    {
        cx.update_window(self.any_handle, |root_view, _, _| {
            root_view
                .downcast::<V>()
                .map_err(|_| anyhow!("the type of the window's root view has changed"))
        })?
    }

    /// Updates the root view of this window.
    ///
    /// This will fail if the window has been closed or if the root view's type does not match
    pub fn update<C, R>(
        &self,
        cx: &mut C,
        update: impl FnOnce(&mut V, &mut Window, &mut Context<V>) -> R,
    ) -> Result<R>
    where
        C: AppContext,
    {
        cx.update_window(self.any_handle, |root_view, window, cx| {
            let view = root_view
                .downcast::<V>()
                .map_err(|_| anyhow!("the type of the window's root view has changed"))?;

            Ok(view.update(cx, |view, cx| update(view, window, cx)))
        })?
    }

    /// Read the root view out of this window.
    ///
    /// This will fail if the window is closed or if the root view's type does not match `V`.
    pub fn read<'a>(&self, cx: &'a App) -> Result<&'a V> {
        let x = cx
            .windows
            .get(self.id)
            .and_then(|window| {
                window
                    .as_deref()
                    .and_then(|window| window.root.clone())
                    .map(|root_view| root_view.downcast::<V>())
            })
            .context("window not found")?
            .map_err(|_| anyhow!("the type of the window's root view has changed"))?;

        Ok(x.read(cx))
    }

    /// Read the root view out of this window, with a callback
    ///
    /// This will fail if the window is closed or if the root view's type does not match `V`.
    pub fn read_with<C, R>(&self, cx: &C, read_with: impl FnOnce(&V, &App) -> R) -> Result<R>
    where
        C: AppContext,
    {
        cx.read_window(self, |root_view, cx| read_with(root_view.read(cx), cx))
    }

    /// Read the root view pointer off of this window.
    ///
    /// This will fail if the window is closed or if the root view's type does not match `V`.
    pub fn entity<C>(&self, cx: &C) -> Result<Entity<V>>
    where
        C: AppContext,
    {
        cx.read_window(self, |root_view, _cx| root_view)
    }

    /// Check if this window is 'active'.
    ///
    /// Will return `None` if the window is closed or currently
    /// borrowed.
    pub fn is_active(&self, cx: &mut App) -> Option<bool> {
        cx.update_window(self.any_handle, |_, window, _| window.is_window_active())
            .ok()
    }
}

impl<V> Copy for WindowHandle<V> {}

impl<V> Clone for WindowHandle<V> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<V> PartialEq for WindowHandle<V> {
    fn eq(&self, other: &Self) -> bool {
        self.any_handle == other.any_handle
    }
}

impl<V> Eq for WindowHandle<V> {}

impl<V> Hash for WindowHandle<V> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.any_handle.hash(state);
    }
}

impl<V: 'static> From<WindowHandle<V>> for AnyWindowHandle {
    fn from(val: WindowHandle<V>) -> Self {
        val.any_handle
    }
}

/// A handle to a window with any root view type, which can be downcast to a window with a specific root view type.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct AnyWindowHandle {
    pub(crate) id: WindowId,
    state_type: TypeId,
}

impl AnyWindowHandle {
    /// Get the ID of this window.
    pub fn window_id(&self) -> WindowId {
        self.id
    }

    /// Attempt to convert this handle to a window handle with a specific root view type.
    /// If the types do not match, this will return `None`.
    pub fn downcast<T: 'static>(&self) -> Option<WindowHandle<T>> {
        if TypeId::of::<T>() == self.state_type {
            Some(WindowHandle {
                any_handle: *self,
                state_type: PhantomData,
            })
        } else {
            None
        }
    }

    /// Updates the state of the root view of this window.
    ///
    /// This will fail if the window has been closed.
    pub fn update<C, R>(
        self,
        cx: &mut C,
        update: impl FnOnce(AnyView, &mut Window, &mut App) -> R,
    ) -> Result<R>
    where
        C: AppContext,
    {
        cx.update_window(self, update)
    }

    /// Read the state of the root view of this window.
    ///
    /// This will fail if the window has been closed.
    pub fn read<T, C, R>(self, cx: &C, read: impl FnOnce(Entity<T>, &App) -> R) -> Result<R>
    where
        C: AppContext,
        T: 'static,
    {
        let view = self
            .downcast::<T>()
            .context("the type of the window's root view has changed")?;

        cx.read_window(&view, read)
    }
}

impl HasWindowHandle for Window {
    fn window_handle(&self) -> Result<raw_window_handle::WindowHandle<'_>, HandleError> {
        self.platform_window.window_handle()
    }
}

impl HasDisplayHandle for Window {
    fn display_handle(
        &self,
    ) -> std::result::Result<raw_window_handle::DisplayHandle<'_>, HandleError> {
        self.platform_window.display_handle()
    }
}

/// An identifier for an [`Element`].
///
/// Can be constructed with a string, a number, or both, as well
/// as other internal representations.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum ElementId {
    /// The ID of a View element
    View(EntityId),
    /// An integer ID.
    Integer(u64),
    /// A string based ID.
    Name(SharedString),
    /// A UUID.
    Uuid(Uuid),
    /// An ID that's equated with a focus handle.
    FocusHandle(FocusId),
    /// A combination of a name and an integer.
    NamedInteger(SharedString, u64),
    /// A path.
    Path(Arc<std::path::Path>),
    /// A code location.
    CodeLocation(core::panic::Location<'static>),
    /// A labeled child of an element.
    NamedChild(Arc<ElementId>, SharedString),
    /// A byte array ID (used for text-anchors)
    OpaqueId([u8; 20]),
    /// A synthetic, framework-assigned positional identity for an element
    /// that has no author-supplied [`ElementId`].
    ///
    /// Framework-internal (#92): pushed only onto [`Window::instance_id_stack`],
    /// never onto [`Window::element_id_stack`], and never constructed by any
    /// public `From` impl — so it can never collide with an author's own
    /// `ElementId::Integer`. It exists so [`InstanceKey`](crate::instance::InstanceKey)
    /// can address the majority of elements (bare `div()`, all of `Text`) that
    /// never call `.id(...)`, the same way [`LayerKey`] addresses elements that
    /// do.
    InstanceSlot(u32),
}

impl ElementId {
    /// Constructs an `ElementId::NamedInteger` from a name and `usize`.
    pub fn named_usize(name: impl Into<SharedString>, integer: usize) -> ElementId {
        Self::NamedInteger(name.into(), integer as u64)
    }
}

impl Display for ElementId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ElementId::View(entity_id) => write!(f, "view-{}", entity_id)?,
            ElementId::Integer(ix) => write!(f, "{}", ix)?,
            ElementId::Name(name) => write!(f, "{}", name)?,
            ElementId::FocusHandle(_) => write!(f, "FocusHandle")?,
            ElementId::NamedInteger(s, i) => write!(f, "{}-{}", s, i)?,
            ElementId::Uuid(uuid) => write!(f, "{}", uuid)?,
            ElementId::Path(path) => write!(f, "{}", path.display())?,
            ElementId::CodeLocation(location) => write!(f, "{}", location)?,
            ElementId::NamedChild(id, name) => write!(f, "{}-{}", id, name)?,
            ElementId::OpaqueId(opaque_id) => write!(f, "{:x?}", opaque_id)?,
            ElementId::InstanceSlot(slot) => write!(f, "#{}", slot)?,
        }

        Ok(())
    }
}

impl TryInto<SharedString> for ElementId {
    type Error = anyhow::Error;

    fn try_into(self) -> anyhow::Result<SharedString> {
        if let ElementId::Name(name) = self {
            Ok(name)
        } else {
            anyhow::bail!("element id is not string")
        }
    }
}

impl From<usize> for ElementId {
    fn from(id: usize) -> Self {
        ElementId::Integer(id as u64)
    }
}

impl From<i32> for ElementId {
    fn from(id: i32) -> Self {
        Self::Integer(id as u64)
    }
}

impl From<SharedString> for ElementId {
    fn from(name: SharedString) -> Self {
        ElementId::Name(name)
    }
}

impl From<String> for ElementId {
    fn from(name: String) -> Self {
        ElementId::Name(name.into())
    }
}

impl From<Arc<str>> for ElementId {
    fn from(name: Arc<str>) -> Self {
        ElementId::Name(name.into())
    }
}

impl From<Arc<std::path::Path>> for ElementId {
    fn from(path: Arc<std::path::Path>) -> Self {
        ElementId::Path(path)
    }
}

impl From<&'static str> for ElementId {
    fn from(name: &'static str) -> Self {
        ElementId::Name(SharedString::new_static(name))
    }
}

impl<'a> From<&'a FocusHandle> for ElementId {
    fn from(handle: &'a FocusHandle) -> Self {
        ElementId::FocusHandle(handle.id)
    }
}

impl From<(&'static str, EntityId)> for ElementId {
    fn from((name, id): (&'static str, EntityId)) -> Self {
        ElementId::NamedInteger(SharedString::new_static(name), id.as_u64())
    }
}

impl From<(&'static str, usize)> for ElementId {
    fn from((name, id): (&'static str, usize)) -> Self {
        ElementId::NamedInteger(SharedString::new_static(name), id as u64)
    }
}

impl From<(SharedString, usize)> for ElementId {
    fn from((name, id): (SharedString, usize)) -> Self {
        ElementId::NamedInteger(name, id as u64)
    }
}

impl From<(&'static str, u64)> for ElementId {
    fn from((name, id): (&'static str, u64)) -> Self {
        ElementId::NamedInteger(SharedString::new_static(name), id)
    }
}

impl From<Uuid> for ElementId {
    fn from(value: Uuid) -> Self {
        Self::Uuid(value)
    }
}

impl From<(&'static str, u32)> for ElementId {
    fn from((name, id): (&'static str, u32)) -> Self {
        ElementId::NamedInteger(SharedString::new_static(name), u64::from(id))
    }
}

impl<T: Into<SharedString>> From<(ElementId, T)> for ElementId {
    fn from((id, name): (ElementId, T)) -> Self {
        ElementId::NamedChild(Arc::new(id), name.into())
    }
}

impl From<&'static core::panic::Location<'static>> for ElementId {
    fn from(location: &'static core::panic::Location<'static>) -> Self {
        ElementId::CodeLocation(*location)
    }
}

impl From<[u8; 20]> for ElementId {
    fn from(opaque_id: [u8; 20]) -> Self {
        ElementId::OpaqueId(opaque_id)
    }
}

/// A rectangle to be rendered in the window at the given position and size.
/// Passed as an argument [`Window::paint_quad`].
#[derive(Clone)]
pub struct PaintQuad {
    /// The bounds of the quad within the window.
    pub bounds: Bounds<Pixels>,
    /// The radii of the quad's corners.
    pub corner_radii: Corners<Pixels>,
    /// The background color of the quad.
    pub background: Background,
    /// The widths of the quad's borders.
    pub border_widths: Edges<Pixels>,
    /// The color of the quad's borders.
    pub border_color: Hsla,
    /// The style of the quad's borders.
    pub border_style: BorderStyle,
}

impl PaintQuad {
    /// Sets the corner radii of the quad.
    pub fn corner_radii(self, corner_radii: impl Into<Corners<Pixels>>) -> Self {
        PaintQuad {
            corner_radii: corner_radii.into(),
            ..self
        }
    }

    /// Sets the border widths of the quad.
    pub fn border_widths(self, border_widths: impl Into<Edges<Pixels>>) -> Self {
        PaintQuad {
            border_widths: border_widths.into(),
            ..self
        }
    }

    /// Sets the border color of the quad.
    pub fn border_color(self, border_color: impl Into<Hsla>) -> Self {
        PaintQuad {
            border_color: border_color.into(),
            ..self
        }
    }

    /// Sets the background color of the quad.
    pub fn background(self, background: impl Into<Background>) -> Self {
        PaintQuad {
            background: background.into(),
            ..self
        }
    }
}

/// Creates a quad with the given parameters.
pub fn quad(
    bounds: Bounds<Pixels>,
    corner_radii: impl Into<Corners<Pixels>>,
    background: impl Into<Background>,
    border_widths: impl Into<Edges<Pixels>>,
    border_color: impl Into<Hsla>,
    border_style: BorderStyle,
) -> PaintQuad {
    PaintQuad {
        bounds,
        corner_radii: corner_radii.into(),
        background: background.into(),
        border_widths: border_widths.into(),
        border_color: border_color.into(),
        border_style,
    }
}

/// Creates a filled quad with the given bounds and background color.
pub fn fill(bounds: impl Into<Bounds<Pixels>>, background: impl Into<Background>) -> PaintQuad {
    PaintQuad {
        bounds: bounds.into(),
        corner_radii: (0.).into(),
        background: background.into(),
        border_widths: (0.).into(),
        border_color: transparent_black(),
        border_style: BorderStyle::default(),
    }
}

/// Creates a rectangle outline with the given bounds, border color, and a 1px border width
pub fn outline(
    bounds: impl Into<Bounds<Pixels>>,
    border_color: impl Into<Hsla>,
    border_style: BorderStyle,
) -> PaintQuad {
    PaintQuad {
        bounds: bounds.into(),
        corner_radii: (0.).into(),
        background: transparent_black().into(),
        border_widths: (1.).into(),
        border_color: border_color.into(),
        border_style,
    }
}

#[cfg(test)]
mod test {
    use crate::{Invalidation, TestAppContext, Window, prelude::*, px, size};

    struct EmptyView;
    impl crate::Render for EmptyView {
        fn render(
            &mut self,
            _window: &mut Window,
            _cx: &mut Context<Self>,
        ) -> impl crate::IntoElement {
            crate::Empty
        }
    }

    /// A stored reuse range that has outlived its array must be reported as
    /// invalid, not sliced. This is the guard standing between a stale range and
    /// the `range end index N out of range for slice of length M` abort in
    /// `reuse_layouts`.
    #[gpui::test]
    fn stale_reuse_ranges_are_rejected_not_sliced(cx: &mut TestAppContext) {
        use crate::{PaintIndex, PrepaintStateIndex};

        let window = cx.open_window(size(px(800.), px(600.)), |_, _| EmptyView);

        window
            .update(cx, |_, this, _| {
                // A freshly-drawn window's arrays are empty, so all-zero ranges
                // are the only ones that fit. They must be accepted.
                let empty_prepaint = PrepaintStateIndex::default()..PrepaintStateIndex::default();
                let empty_paint = PaintIndex::default()..PaintIndex::default();
                assert!(
                    this.invalid_reuse_range(&empty_prepaint, &empty_paint)
                        .is_none(),
                    "a zero-length range fits any array"
                );

                // Every field is checked independently. An earlier version of
                // the validator enumerated these by hand and omitted two on the
                // paint side, which shipped a panic in `reuse_paint`
                // ("range start index 213 out of range for slice of length
                // 211"). If a field is added to either index type, add it here
                // and to `invalid_reuse_range`.
                let overruns: [(&str, PrepaintStateIndex, PaintIndex); 15] = [
                    (
                        "hitboxes",
                        PrepaintStateIndex {
                            hitboxes_index: 9,
                            ..Default::default()
                        },
                        PaintIndex::default(),
                    ),
                    (
                        "dispatch_tree",
                        PrepaintStateIndex {
                            dispatch_tree_index: 9,
                            ..Default::default()
                        },
                        PaintIndex::default(),
                    ),
                    (
                        "line layouts",
                        PrepaintStateIndex {
                            line_layout_index: crate::LineLayoutIndex {
                                lines_index: 9,
                                wrapped_lines_index: 0,
                            },
                            ..Default::default()
                        },
                        PaintIndex::default(),
                    ),
                    (
                        "scene",
                        PrepaintStateIndex::default(),
                        PaintIndex {
                            scene_index: 9,
                            ..Default::default()
                        },
                    ),
                    (
                        "mouse_listeners",
                        PrepaintStateIndex::default(),
                        PaintIndex {
                            mouse_listeners_index: 9,
                            ..Default::default()
                        },
                    ),
                    (
                        "prepaint tooltips",
                        PrepaintStateIndex {
                            tooltips_index: 9,
                            ..Default::default()
                        },
                        PaintIndex::default(),
                    ),
                    (
                        "prepaint deferred_draws",
                        PrepaintStateIndex {
                            deferred_draws_index: 9,
                            ..Default::default()
                        },
                        PaintIndex::default(),
                    ),
                    (
                        "prepaint accessed_element_states",
                        PrepaintStateIndex {
                            accessed_element_states_index: 9,
                            ..Default::default()
                        },
                        PaintIndex::default(),
                    ),
                    (
                        "prepaint wrapped line layouts",
                        PrepaintStateIndex {
                            line_layout_index: crate::LineLayoutIndex {
                                lines_index: 0,
                                wrapped_lines_index: 9,
                            },
                            ..Default::default()
                        },
                        PaintIndex::default(),
                    ),
                    (
                        "paint input_handlers",
                        PrepaintStateIndex::default(),
                        PaintIndex {
                            input_handlers_index: 9,
                            ..Default::default()
                        },
                    ),
                    (
                        "paint cursor_styles",
                        PrepaintStateIndex::default(),
                        PaintIndex {
                            cursor_styles_index: 9,
                            ..Default::default()
                        },
                    ),
                    (
                        // Omitted from the original validator.
                        "paint accessed_element_states",
                        PrepaintStateIndex::default(),
                        PaintIndex {
                            accessed_element_states_index: 9,
                            ..Default::default()
                        },
                    ),
                    (
                        "paint tab_stops",
                        PrepaintStateIndex::default(),
                        PaintIndex {
                            tab_handle_index: 9,
                            ..Default::default()
                        },
                    ),
                    (
                        "paint line layouts",
                        PrepaintStateIndex::default(),
                        PaintIndex {
                            line_layout_index: crate::LineLayoutIndex {
                                lines_index: 9,
                                wrapped_lines_index: 0,
                            },
                            ..Default::default()
                        },
                    ),
                    (
                        // Omitted from the original validator — this is the one
                        // that panicked in `reuse_paint`'s wrapped-lines loop.
                        "paint wrapped line layouts",
                        PrepaintStateIndex::default(),
                        PaintIndex {
                            line_layout_index: crate::LineLayoutIndex {
                                lines_index: 0,
                                wrapped_lines_index: 9,
                            },
                            ..Default::default()
                        },
                    ),
                ];

                for (label, prepaint_end, paint_end) in overruns {
                    let prepaint = PrepaintStateIndex::default()..prepaint_end;
                    let paint = PaintIndex::default()..paint_end;
                    let result = this.invalid_reuse_range(&prepaint, &paint);
                    assert!(
                        result.is_some(),
                        "an over-long {label} range must be rejected, not replayed"
                    );
                    let (_, end, len) = result.unwrap();
                    assert!(
                        end > len,
                        "{label}: reported stored_end {end} should exceed actual_len {len}"
                    );
                }
            })
            .ok();
    }

    // ── Issue #83 scaffolding ────────────────────────────────────────────
    //
    // A cached `AnyView` nested under a root view, where the leaf counts its
    // own renders. `cx.notify()` on the leaf must produce a rebuild of the
    // leaf on the next frame; anything else is the #83 symptom (stale
    // content silently replayed forever).

    struct CacheLeaf {
        renders: std::rc::Rc<std::cell::Cell<usize>>,
    }

    impl crate::Render for CacheLeaf {
        fn render(
            &mut self,
            _window: &mut Window,
            _cx: &mut Context<Self>,
        ) -> impl crate::IntoElement {
            self.renders.set(self.renders.get() + 1);
            crate::div().w(px(10.)).h(px(10.))
        }
    }

    struct CacheRoot {
        leaf: crate::Entity<CacheLeaf>,
        renders: std::rc::Rc<std::cell::Cell<usize>>,
        /// Touch the leaf through `EntityMap` before rendering it, the way a
        /// panel that reads its child's state for a title or a badge would.
        read_leaf_first: bool,
    }

    impl crate::Render for CacheRoot {
        fn render(
            &mut self,
            _window: &mut Window,
            cx: &mut Context<Self>,
        ) -> impl crate::IntoElement {
            self.renders.set(self.renders.get() + 1);
            if self.read_leaf_first {
                let _ = self.leaf.read(cx).renders.get();
            }
            crate::div().size_full().child(
                crate::AnyView::from(self.leaf.clone())
                    .cached(crate::StyleRefinement::default().w(px(10.)).h(px(10.))),
            )
        }
    }

    fn cached_leaf_window(
        cx: &mut TestAppContext,
        read_leaf_first: bool,
    ) -> (
        crate::WindowHandle<CacheRoot>,
        crate::Entity<CacheLeaf>,
        std::rc::Rc<std::cell::Cell<usize>>,
        std::rc::Rc<std::cell::Cell<usize>>,
    ) {
        let leaf_renders = std::rc::Rc::new(std::cell::Cell::new(0));
        let root_renders = std::rc::Rc::new(std::cell::Cell::new(0));
        let leaf = cx.update(|cx| {
            cx.new(|_| CacheLeaf {
                renders: leaf_renders.clone(),
            })
        });
        let leaf_for_root = leaf.clone();
        let root_renders_for_root = root_renders.clone();
        let window = cx.open_window(size(px(800.), px(600.)), move |_, _| CacheRoot {
            leaf: leaf_for_root,
            renders: root_renders_for_root,
            read_leaf_first,
        });
        cx.run_until_parked();
        (window, leaf, leaf_renders, root_renders)
    }

    /// Drive one frame the way an external frame pump does: the window needs
    /// to draw, but nothing is invalidated, so every cached view takes the
    /// reuse path. `refresh_windows`/`Window::refresh` is *not* equivalent —
    /// it sets every invalidation axis, which bypasses the cache entirely and
    /// would hide exactly the bug these tests are looking for.
    fn clean_frame(cx: &mut TestAppContext, window: crate::AnyWindowHandle) {
        window
            .update(cx, |_, window, _| window.refresh_buffers())
            .unwrap();
        cx.run_until_parked();
    }

    /// Run whatever `Window::on_next_frame` queued, the way the platform's
    /// `on_request_frame` hook does, and let any resulting invalidation settle.
    /// Returns whether there was anything to run.
    ///
    /// The test platform's `on_request_frame` discards its callback, so nothing
    /// in the harness ever drains this queue on its own — without this,
    /// `request_animation_frame` is invisible to tests.
    fn pump_next_frame_callbacks(cx: &mut TestAppContext, window: crate::AnyWindowHandle) -> bool {
        let ran = window
            .update(cx, |_, window, cx| {
                let callbacks = window.next_frame_callbacks.take();
                let ran = !callbacks.is_empty();
                for callback in callbacks {
                    callback(window, cx);
                }
                ran
            })
            .unwrap();
        cx.run_until_parked();
        ran
    }

    /// `cx.notify()` on a cached leaf must rebuild that leaf on the next
    /// frame, whether or not an ancestor happened to touch it through
    /// `EntityMap` first.
    #[gpui::test]
    fn notify_rebuilds_cached_view(cx: &mut TestAppContext) {
        for read_leaf_first in [false, true] {
            let (_window, leaf, leaf_renders, root_renders) =
                cached_leaf_window(cx, read_leaf_first);

            let leaf_before = leaf_renders.get();
            let root_before = root_renders.get();

            leaf.update(cx, |_, cx| cx.notify());
            cx.run_until_parked();

            assert!(
                root_renders.get() > root_before,
                "read_leaf_first={read_leaf_first}: notifying the leaf did not even \
                 produce a frame (root renders stuck at {root_before})"
            );
            assert!(
                leaf_renders.get() > leaf_before,
                "read_leaf_first={read_leaf_first}: the frame happened but the cached \
                 leaf replayed its stale prepaint instead of rebuilding \
                 (leaf renders stuck at {leaf_before})"
            );
        }
    }

    /// The same thing, repeated. One successful invalidation is not enough:
    /// the reuse path re-registers a view's dependencies from the *stored*
    /// set, so a set that lost the view's own id goes stale permanently after
    /// the first clean frame.
    #[gpui::test]
    fn notify_rebuilds_cached_view_repeatedly(cx: &mut TestAppContext) {
        for read_leaf_first in [false, true] {
            let (window, leaf, leaf_renders, _) = cached_leaf_window(cx, read_leaf_first);

            for round in 0..4 {
                // A clean frame in between, so the leaf takes the reuse path
                // and re-registers itself from its stored dependency set.
                clean_frame(cx, window.into());

                let before = leaf_renders.get();
                leaf.update(cx, |_, cx| cx.notify());
                cx.run_until_parked();
                assert!(
                    leaf_renders.get() > before,
                    "read_leaf_first={read_leaf_first} round {round}: cached leaf stopped \
                     responding to notify (renders stuck at {before})"
                );
            }
        }
    }

    struct CacheMid {
        leaf: crate::Entity<CacheLeaf>,
        renders: std::rc::Rc<std::cell::Cell<usize>>,
    }

    impl crate::Render for CacheMid {
        fn render(
            &mut self,
            _window: &mut Window,
            _cx: &mut Context<Self>,
        ) -> impl crate::IntoElement {
            self.renders.set(self.renders.get() + 1);
            crate::div().size_full().child(
                crate::AnyView::from(self.leaf.clone())
                    .cached(crate::StyleRefinement::default().w(px(10.)).h(px(10.))),
            )
        }
    }

    struct CacheNestedRoot {
        mid: crate::Entity<CacheMid>,
    }

    impl crate::Render for CacheNestedRoot {
        fn render(
            &mut self,
            _window: &mut Window,
            _cx: &mut Context<Self>,
        ) -> impl crate::IntoElement {
            crate::div().size_full().child(
                crate::AnyView::from(self.mid.clone())
                    .cached(crate::StyleRefinement::default().w(px(10.)).h(px(10.))),
            )
        }
    }

    /// A cached view nested inside another cached view. When the leaf is
    /// notified the middle view must rebuild too, otherwise the leaf never
    /// gets a chance to prepaint at all.
    #[gpui::test]
    fn notify_rebuilds_nested_cached_view(cx: &mut TestAppContext) {
        let leaf_renders = std::rc::Rc::new(std::cell::Cell::new(0));
        let mid_renders = std::rc::Rc::new(std::cell::Cell::new(0));
        let leaf = cx.update(|cx| {
            cx.new(|_| CacheLeaf {
                renders: leaf_renders.clone(),
            })
        });
        let leaf_for_mid = leaf.clone();
        let mid_renders_for_mid = mid_renders.clone();
        let mid = cx.update(|cx| {
            cx.new(|_| CacheMid {
                leaf: leaf_for_mid,
                renders: mid_renders_for_mid,
            })
        });
        let mid_for_root = mid.clone();
        let window = cx.open_window(size(px(800.), px(600.)), move |_, _| CacheNestedRoot {
            mid: mid_for_root,
        });
        cx.run_until_parked();

        for round in 0..4 {
            clean_frame(cx, window.into());
            let leaf_before = leaf_renders.get();
            let mid_before = mid_renders.get();
            leaf.update(cx, |_, cx| cx.notify());
            cx.run_until_parked();
            assert!(
                mid_renders.get() > mid_before,
                "round {round}: notifying the leaf left its cached parent replaying, \
                 so the leaf could never prepaint"
            );
            assert!(
                leaf_renders.get() > leaf_before,
                "round {round}: the leaf replayed stale content after notify"
            );
        }

        // And the parent must stay independently invalidatable.
        for round in 0..4 {
            clean_frame(cx, window.into());
            let mid_before = mid_renders.get();
            mid.update(cx, |_, cx| cx.notify());
            cx.run_until_parked();
            assert!(
                mid_renders.get() > mid_before,
                "round {round}: notifying the middle cached view did nothing"
            );
        }
    }

    /// Direct measurement of the invalidation bookkeeping: after a frame in
    /// which a cached view took the *reuse* path, the window must still be
    /// registered as an invalidation target for that view. If it is not,
    /// `App::notify` falls through to `pending_notifications` and the view is
    /// permanently stuck.
    #[gpui::test]
    fn reused_cached_view_stays_tracked(cx: &mut TestAppContext) {
        let (window, leaf, leaf_renders, _root_renders) = cached_leaf_window(cx, false);
        let leaf_id = leaf.entity_id();
        let window_id = window.window_id();

        for round in 0..4 {
            // Force a frame in which nothing is dirty, so the cached leaf
            // reuses and re-registers itself purely from its stored set.
            let before = leaf_renders.get();
            clean_frame(cx, window.into());
            assert_eq!(
                leaf_renders.get(),
                before,
                "round {round}: the leaf rebuilt, so this frame did not exercise \
                 the reuse path the assertions below are about"
            );

            cx.update(|cx| {
                assert!(
                    cx.tracked_entities
                        .get(&window_id)
                        .is_some_and(|set| set.contains(&leaf_id)),
                    "round {round}: the cached leaf dropped out of `tracked_entities` \
                     after a reuse frame; `App::notify` can no longer find it"
                );
                assert!(
                    cx.window_invalidators_by_entity
                        .get(&leaf_id)
                        .is_some_and(|windows| windows.contains_key(&window_id)),
                    "round {round}: the cached leaf dropped out of \
                     `window_invalidators_by_entity` after a reuse frame"
                );
            });
        }
    }

    /// An *uncached* child view sitting inside nested cached panels — the dock
    /// shape. Notifying the child must reach it through both cached layers.
    #[gpui::test]
    fn notify_reaches_uncached_view_under_cached_panels(cx: &mut TestAppContext) {
        struct Panel {
            child: crate::AnyView,
            cached: bool,
        }
        impl crate::Render for Panel {
            fn render(
                &mut self,
                _window: &mut Window,
                _cx: &mut Context<Self>,
            ) -> impl crate::IntoElement {
                let child = self.child.clone();
                crate::div().size_full().child(if self.cached {
                    child
                        .cached(crate::StyleRefinement::default().w(px(10.)).h(px(10.)))
                        .into_any_element()
                } else {
                    child.into_any_element()
                })
            }
        }

        let leaf_renders = std::rc::Rc::new(std::cell::Cell::new(0));
        let leaf = cx.update(|cx| {
            cx.new(|_| CacheLeaf {
                renders: leaf_renders.clone(),
            })
        });
        // leaf is rendered as a plain `Entity<V>` element (uncached) inside
        // cached B, inside cached A, inside the uncached root.
        let inner = cx.update({
            let leaf = leaf.clone();
            move |cx| {
                cx.new(|_| Panel {
                    child: leaf.into(),
                    cached: false,
                })
            }
        });
        let outer = cx.update({
            let inner = inner.clone();
            move |cx| {
                cx.new(|_| Panel {
                    child: inner.into(),
                    cached: true,
                })
            }
        });
        let outer_for_root = outer.clone();
        let window = cx.open_window(size(px(800.), px(600.)), move |_, _| Panel {
            child: outer_for_root.into(),
            cached: true,
        });
        cx.run_until_parked();

        for round in 0..4 {
            clean_frame(cx, window.into());
            let before = leaf_renders.get();
            leaf.update(cx, |_, cx| cx.notify());
            cx.run_until_parked();
            assert!(
                leaf_renders.get() > before,
                "round {round}: notifying a view nested under cached panels did not \
                 rebuild it (renders stuck at {before})"
            );
        }
    }

    // ── Model-dependency scaffolding ─────────────────────────────────────

    struct DepModel {
        value: usize,
    }

    /// Reads `model` during render and publishes what it saw. Nests an
    /// optional child so the read can be pushed arbitrarily deep inside a
    /// cached view's subtree.
    struct DepReader {
        model: crate::Entity<DepModel>,
        seen: std::rc::Rc<std::cell::Cell<usize>>,
        renders: std::rc::Rc<std::cell::Cell<usize>>,
        child: Option<crate::AnyView>,
        /// Wrap the child in `.cached(..)` rather than rendering it plainly.
        cache_child: bool,
        /// Read the model *before* rendering the child, so the child's own
        /// dependency scope opens with the model already marked accessed.
        read_model: bool,
    }

    impl crate::Render for DepReader {
        fn render(
            &mut self,
            _window: &mut Window,
            cx: &mut Context<Self>,
        ) -> impl crate::IntoElement {
            self.renders.set(self.renders.get() + 1);
            if self.read_model {
                self.seen.set(self.model.read(cx).value);
            }
            let child = self.child.clone().map(|child| {
                if self.cache_child {
                    child
                        .cached(crate::StyleRefinement::default().w(px(10.)).h(px(10.)))
                        .into_any_element()
                } else {
                    child.into_any_element()
                }
            });
            crate::div().size_full().children(child)
        }
    }

    /// A cached view that derives its content from a *model* entity it reads
    /// during render. Notifying the model must rebuild that view.
    ///
    /// `mark_view_dirty` used to walk only `view_path_reversed`, which knows
    /// about entities owning a dispatch node — a model has none, so the walk
    /// yielded nothing, no view was marked dirty, and every cached view
    /// replayed. This is issue #83's mechanism.
    ///
    /// The `parent_reads_model_first` case is the one that also exercises
    /// `App::detect_accessed_entities`: with the old `difference()` capture,
    /// an entity an ancestor had already read was subtracted out of the
    /// child's dependency set, so the child looked like it depended on nothing.
    #[gpui::test]
    fn notify_on_model_rebuilds_cached_reader(cx: &mut TestAppContext) {
        for parent_reads_model_first in [false, true] {
            let seen = std::rc::Rc::new(std::cell::Cell::new(usize::MAX));
            let renders = std::rc::Rc::new(std::cell::Cell::new(0));
            let model = cx.update(|cx| cx.new(|_| DepModel { value: 0 }));

            let reader = cx.update({
                let (model, seen, renders) = (model.clone(), seen.clone(), renders.clone());
                move |cx| {
                    cx.new(|_| DepReader {
                        model,
                        seen,
                        renders,
                        child: None,
                        cache_child: false,
                        read_model: true,
                    })
                }
            });

            let window = cx.open_window(size(px(800.), px(600.)), {
                let (model, reader) = (model.clone(), reader.clone());
                move |_, _| DepReader {
                    model,
                    seen: std::rc::Rc::new(std::cell::Cell::new(0)),
                    renders: std::rc::Rc::new(std::cell::Cell::new(0)),
                    child: Some(reader.into()),
                    cache_child: true,
                    read_model: parent_reads_model_first,
                }
            });
            cx.run_until_parked();

            for round in 1..=4 {
                clean_frame(cx, window.into());
                model.update(cx, |model, cx| {
                    model.value = round;
                    cx.notify();
                });
                cx.run_until_parked();
                assert_eq!(
                    seen.get(),
                    round,
                    "parent_reads_model_first={parent_reads_model_first} round {round}: \
                     the cached view reading this model replayed stale content after the \
                     model was notified ({} renders so far)",
                    renders.get()
                );
            }
        }
    }

    /// The model is read by a view buried under two cached layers, and the
    /// notified entity is neither a view nor on anyone's ancestor path. Every
    /// cached layer above the reader has to rebuild, or the reader never
    /// prepaints.
    #[gpui::test]
    fn notify_on_model_rebuilds_deeply_nested_cached_reader(cx: &mut TestAppContext) {
        let seen = std::rc::Rc::new(std::cell::Cell::new(usize::MAX));
        let renders = std::rc::Rc::new(std::cell::Cell::new(0));
        let model = cx.update(|cx| cx.new(|_| DepModel { value: 0 }));

        let leaf = cx.update({
            let (model, seen, renders) = (model.clone(), seen.clone(), renders.clone());
            move |cx| {
                cx.new(|_| DepReader {
                    model,
                    seen,
                    renders,
                    child: None,
                    cache_child: false,
                    read_model: true,
                })
            }
        });
        let mid = cx.update({
            let (model, leaf) = (model.clone(), leaf.clone());
            move |cx| {
                cx.new(|_| DepReader {
                    model,
                    seen: std::rc::Rc::new(std::cell::Cell::new(0)),
                    renders: std::rc::Rc::new(std::cell::Cell::new(0)),
                    child: Some(leaf.into()),
                    cache_child: true,
                    read_model: false,
                })
            }
        });
        let window = cx.open_window(size(px(800.), px(600.)), {
            let (model, mid) = (model.clone(), mid.clone());
            move |_, _| DepReader {
                model,
                seen: std::rc::Rc::new(std::cell::Cell::new(0)),
                renders: std::rc::Rc::new(std::cell::Cell::new(0)),
                child: Some(mid.into()),
                cache_child: true,
                read_model: false,
            }
        });
        cx.run_until_parked();

        for round in 1..=4 {
            clean_frame(cx, window.into());
            model.update(cx, |model, cx| {
                model.value = round;
                cx.notify();
            });
            cx.run_until_parked();
            assert_eq!(
                seen.get(),
                round,
                "round {round}: a model read two cached layers down did not invalidate \
                 the layers above it"
            );
        }
    }

    /// The dependency index must not make everything rebuild: a model nobody
    /// on screen reads still has to leave cached views alone. Without this the
    /// fix for #83 would just be window-scope invalidation under another name.
    #[gpui::test]
    fn unrelated_notify_leaves_cached_views_alone(cx: &mut TestAppContext) {
        let (window, _leaf, leaf_renders, _root_renders) = cached_leaf_window(cx, false);
        let unrelated = cx.update(|cx| cx.new(|_| DepModel { value: 0 }));

        clean_frame(cx, window.into());
        let before = leaf_renders.get();

        for _ in 0..4 {
            unrelated.update(cx, |model, cx| {
                model.value += 1;
                cx.notify();
            });
            cx.run_until_parked();
            clean_frame(cx, window.into());
        }

        assert_eq!(
            leaf_renders.get(),
            before,
            "a cached view rebuilt because of an entity it never reads"
        );
    }

    // ── Notify-during-draw scaffolding ───────────────────────────────────
    //
    // A root view that notifies a *model* from inside its own `render`, which
    // runs during `DrawPhase::Prepaint`. Real code does this constantly:
    // scrollbar thumb geometry is computed during prepaint/paint, and
    // `CodeEditor::set_language` notifies the `InputState` it owns rather than
    // itself. `WindowInvalidator::invalidate` used to answer such a
    // notify by inserting into `dirty_views` and then doing nothing else — no
    // dirty flag, so no frame was ever scheduled to consume the insertion, and
    // no `Effect::Notify`, so `cx.observe` callbacks never ran at all.

    struct DrawNotifier {
        signal: crate::Entity<DepModel>,
        /// Number of remaining renders that should notify `signal`. Bounded so
        /// a broken fix shows up as a failing assertion rather than a hang.
        pending: std::rc::Rc<std::cell::Cell<usize>>,
        /// Notifies to issue per render, to exercise deduplication.
        per_render: usize,
        renders: std::rc::Rc<std::cell::Cell<usize>>,
    }

    impl crate::Render for DrawNotifier {
        fn render(
            &mut self,
            _window: &mut Window,
            cx: &mut Context<Self>,
        ) -> impl crate::IntoElement {
            self.renders.set(self.renders.get() + 1);
            // Read it, so this window is registered as tracking the entity.
            // Without that `App::notify` takes the "no window tracks this"
            // fallback and pushes the effect regardless of draw phase, which
            // would hide the bug under test.
            let _ = self.signal.read(cx).value;
            if self.pending.get() > 0 {
                self.pending.set(self.pending.get() - 1);
                for _ in 0..self.per_render {
                    self.signal.update(cx, |model, cx| {
                        model.value += 1;
                        cx.notify();
                    });
                }
            }
            crate::div().size_full()
        }
    }

    fn draw_notifier_window(
        cx: &mut TestAppContext,
        per_render: usize,
    ) -> (
        crate::WindowHandle<DrawNotifier>,
        crate::Entity<DepModel>,
        std::rc::Rc<std::cell::Cell<usize>>,
        std::rc::Rc<std::cell::Cell<usize>>,
    ) {
        let pending = std::rc::Rc::new(std::cell::Cell::new(0));
        let renders = std::rc::Rc::new(std::cell::Cell::new(0));
        let signal = cx.update(|cx| cx.new(|_| DepModel { value: 0 }));
        let window = cx.open_window(size(px(800.), px(600.)), {
            let (signal, pending, renders) = (signal.clone(), pending.clone(), renders.clone());
            move |_, _| DrawNotifier {
                signal,
                pending,
                per_render,
                renders,
            }
        });
        cx.run_until_parked();
        (window, signal, pending, renders)
    }

    fn observe_count(
        cx: &mut TestAppContext,
        signal: &crate::Entity<DepModel>,
    ) -> (std::rc::Rc<std::cell::Cell<usize>>, crate::Subscription) {
        let observed = std::rc::Rc::new(std::cell::Cell::new(0usize));
        let subscription = cx.update({
            let (observed, signal) = (observed.clone(), signal.clone());
            move |cx| cx.observe(&signal, move |_, _| observed.set(observed.get() + 1))
        });
        (observed, subscription)
    }

    /// An entity notified while a draw is in progress must still reach its
    /// observers. This is the "a bunch of events get skipped" symptom: the
    /// `Effect::Notify` was simply never pushed.
    #[gpui::test]
    fn notify_during_draw_reaches_observers(cx: &mut TestAppContext) {
        let (window, signal, pending, _renders) = draw_notifier_window(cx, 1);
        let (observed, _sub) = observe_count(cx, &signal);

        pending.set(1);
        clean_frame(cx, window.into());

        assert_eq!(
            observed.get(),
            1,
            "an entity notified during a draw never reached its observers"
        );
    }

    /// ...and the window must schedule a frame to answer that notify, rather
    /// than leaving the entity sitting in `dirty_views` with nothing to
    /// consume it.
    #[gpui::test]
    fn notify_during_draw_schedules_a_follow_up_frame(cx: &mut TestAppContext) {
        let (window, _signal, pending, renders) = draw_notifier_window(cx, 1);

        let before = renders.get();
        pending.set(1);
        clean_frame(cx, window.into());

        assert_eq!(
            renders.get(),
            before + 2,
            "expected the notifying frame plus exactly one follow-up frame"
        );
    }

    /// Deduplication: a view that notifies the same entity many times inside a
    /// single draw must cost one observer call and one follow-up frame, not N.
    #[gpui::test]
    fn repeated_notifies_during_one_draw_collapse(cx: &mut TestAppContext) {
        let (window, signal, pending, renders) = draw_notifier_window(cx, 8);
        let (observed, _sub) = observe_count(cx, &signal);

        let before = renders.get();
        pending.set(1);
        clean_frame(cx, window.into());

        assert_eq!(
            observed.get(),
            1,
            "eight notifies of one entity in one draw should collapse to one"
        );
        assert_eq!(
            renders.get(),
            before + 2,
            "eight notifies of one entity in one draw should cost one extra frame"
        );
    }

    /// The other side of the ledger: a draw that notifies nothing must not
    /// schedule anything. If this fails the deferral is a redraw loop.
    #[gpui::test]
    fn draw_without_notifications_schedules_no_extra_frame(cx: &mut TestAppContext) {
        let (window, signal, _pending, renders) = draw_notifier_window(cx, 1);
        let (observed, _sub) = observe_count(cx, &signal);

        let before = renders.get();
        for _ in 0..4 {
            clean_frame(cx, window.into());
        }

        assert_eq!(
            renders.get(),
            before + 4,
            "the deferred-notify flush scheduled redraws with nothing deferred"
        );
        assert_eq!(observed.get(), 0, "observers fired without any notify");
    }

    /// The hazard the deferral introduces, stated as a measurement rather than
    /// a hope: each draw that notifies begets exactly one more draw. The chain
    /// terminates only because the view stops notifying — a view that notifies
    /// unconditionally from its own render or paint now redraws forever.
    ///
    /// This is why `TextInput`'s paint-time "record what was painted" update in
    /// `wgpui-component` had to become conditional; it notified on every paint.
    #[gpui::test]
    fn each_notifying_draw_begets_exactly_one_more(cx: &mut TestAppContext) {
        for chain in [1usize, 2, 5] {
            let (window, _signal, pending, renders) = draw_notifier_window(cx, 1);

            let before = renders.get();
            pending.set(chain);
            clean_frame(cx, window.into());

            assert_eq!(
                renders.get(),
                before + chain + 1,
                "chain {chain}: expected {chain} notifying draws plus one \
                 settling draw"
            );
            let dirty = window
                .update(cx, |_, window, _| window.invalidator.is_dirty())
                .unwrap();
            assert!(
                !dirty,
                "chain {chain}: the window never settled after the notifies stopped"
            );
        }
    }

    /// A notify issued during `Paint` (not just `Prepaint`) is dropped by the
    /// same code path. Scrollbar thumb state is computed here.
    #[gpui::test]
    fn notify_during_paint_reaches_observers(cx: &mut TestAppContext) {
        struct PaintNotifier {
            signal: crate::Entity<DepModel>,
            pending: std::rc::Rc<std::cell::Cell<usize>>,
        }

        impl crate::Render for PaintNotifier {
            fn render(
                &mut self,
                _window: &mut Window,
                cx: &mut Context<Self>,
            ) -> impl crate::IntoElement {
                let _ = self.signal.read(cx).value;
                let signal = self.signal.clone();
                let pending = self.pending.clone();
                crate::div().size_full().child(crate::canvas(
                    |_, _, _| (),
                    move |_, _, window, cx| {
                        window.invalidator.debug_assert_paint();
                        if pending.get() > 0 {
                            pending.set(pending.get() - 1);
                            signal.update(cx, |model, cx| {
                                model.value += 1;
                                cx.notify();
                            });
                        }
                    },
                ))
            }
        }

        let pending = std::rc::Rc::new(std::cell::Cell::new(0));
        let signal = cx.update(|cx| cx.new(|_| DepModel { value: 0 }));
        let window = cx.open_window(size(px(800.), px(600.)), {
            let (signal, pending) = (signal.clone(), pending.clone());
            move |_, _| PaintNotifier { signal, pending }
        });
        cx.run_until_parked();

        let (observed, _sub) = observe_count(cx, &signal);
        pending.set(1);
        clean_frame(cx, window.into());

        assert_eq!(
            observed.get(),
            1,
            "an entity notified during paint never reached its observers"
        );
    }

    /// The animation drivers in `virtual_list`, `uniform_list` and `h_list` run
    /// inside `prepaint`. They used to call `Window::refresh`, which the
    /// `not_drawing` guard swallowed outright, so smooth scroll only advanced
    /// when something unrelated kept the window dirty.
    ///
    /// This pins the replacement: `request_animation_frame` from prepaint
    /// queues a frame callback, schedules exactly the frames asked for, and
    /// then settles. That it *queues a callback* is the part that matters —
    /// #87 later made a mid-draw `refresh` defer rather than drop, but a
    /// deferred refresh only raises the dirty flag, so it still cannot carry an
    /// animation on its own. The deferral half is covered by
    /// [`refresh_during_draw_is_deferred_not_dropped`].
    #[gpui::test]
    fn request_animation_frame_from_prepaint_drives_frames(cx: &mut TestAppContext) {
        struct PrepaintDriver {
            /// Remaining prepaints that should ask for another frame. Bounded so
            /// a runaway shows up as a failed assertion rather than a hang.
            pending: std::rc::Rc<std::cell::Cell<usize>>,
            prepaints: std::rc::Rc<std::cell::Cell<usize>>,
        }

        impl crate::Render for PrepaintDriver {
            fn render(
                &mut self,
                _window: &mut Window,
                _cx: &mut Context<Self>,
            ) -> impl crate::IntoElement {
                let pending = self.pending.clone();
                let prepaints = self.prepaints.clone();
                crate::div().size_full().child(crate::canvas(
                    move |_, window, _| {
                        prepaints.set(prepaints.get() + 1);
                        if pending.get() > 0 {
                            pending.set(pending.get() - 1);
                            window.request_animation_frame();
                        }
                    },
                    |_, _, _, _| (),
                ))
            }
        }

        let requested_frames = 3;
        {
            let pending = std::rc::Rc::new(std::cell::Cell::new(0));
            let prepaints = std::rc::Rc::new(std::cell::Cell::new(0));
            let window = cx.open_window(size(px(800.), px(600.)), {
                let (pending, prepaints) = (pending.clone(), prepaints.clone());
                move |_, _| PrepaintDriver { pending, prepaints }
            });
            cx.run_until_parked();

            let before = prepaints.get();
            pending.set(requested_frames);
            // One externally-pumped frame, then nothing else touches the
            // window except the frame callbacks the draw queued for itself —
            // which is exactly the situation an animation has to survive.
            clean_frame(cx, window.into());

            let mut pumps = 0;
            while pump_next_frame_callbacks(cx, window.into()) {
                pumps += 1;
                assert!(
                    pumps <= requested_frames + 1,
                    "the driver never stopped asking for frames"
                );
            }

            assert_eq!(
                prepaints.get(),
                before + 1 + requested_frames,
                "request_animation_frame from prepaint: expected the pumped \
                 frame plus {requested_frames} self-scheduled frames"
            );
            let dirty = window
                .update(cx, |_, window, _| window.invalidator.is_dirty())
                .unwrap();
            assert!(
                !dirty,
                "the window never settled after the driver stopped asking"
            );
            assert_eq!(
                pending.get(),
                0,
                "every frame request should have been consumed by a prepaint"
            );
        }
    }

    /// A window-scope invalidation issued mid-draw used to be dropped by a
    /// `not_drawing()` guard. That is why the smooth-scroll drivers in
    /// `virtual_list`, `uniform_list` and `h_list` — all of which call
    /// `refresh()` from prepaint while an animation is in flight — never
    /// scheduled their own next frame, and animated only when some unrelated
    /// invalidation happened to keep the window dirty.
    ///
    /// The request must be recorded and carried to the next draw with its axes
    /// intact, and the chain must terminate once the requests stop.
    #[gpui::test]
    fn refresh_during_draw_is_deferred_not_dropped(cx: &mut TestAppContext) {
        struct RefreshDuringDraw {
            leaf: crate::Entity<CacheLeaf>,
            /// Remaining renders that should refresh. Bounded so a lost
            /// deferral shows up as a failing assertion rather than a hang.
            pending: std::rc::Rc<std::cell::Cell<usize>>,
        }

        impl crate::Render for RefreshDuringDraw {
            fn render(
                &mut self,
                window: &mut Window,
                _cx: &mut Context<Self>,
            ) -> impl crate::IntoElement {
                if self.pending.get() > 0 {
                    self.pending.set(self.pending.get() - 1);
                    window.refresh();
                }
                crate::div().size_full().child(
                    crate::AnyView::from(self.leaf.clone())
                        .cached(crate::StyleRefinement::default().w(px(10.)).h(px(10.))),
                )
            }
        }

        let leaf_renders = std::rc::Rc::new(std::cell::Cell::new(0));
        let pending = std::rc::Rc::new(std::cell::Cell::new(0));
        let leaf = cx.update({
            let leaf_renders = leaf_renders.clone();
            move |cx| {
                cx.new(|_| CacheLeaf {
                    renders: leaf_renders,
                })
            }
        });
        let window = cx.open_window(size(px(800.), px(600.)), {
            let pending = pending.clone();
            move |_, _| RefreshDuringDraw { leaf, pending }
        });
        cx.run_until_parked();

        clean_frame(cx, window.into());
        let leaf_before = leaf_renders.get();

        pending.set(1);
        clean_frame(cx, window.into());
        assert_eq!(
            leaf_renders.get(),
            leaf_before,
            "the refresh arrived mid-draw, too late for the frame already \
             being built"
        );

        // Reading the flag is also what drives the frame that answers it:
        // leaving `update` flushes effects, and that draws every dirty window.
        let dirty = window
            .update(cx, |_, window, _| window.invalidator.is_dirty())
            .unwrap();
        assert!(dirty, "a refresh() issued mid-draw was dropped");
        cx.run_until_parked();

        assert!(
            leaf_renders.get() > leaf_before,
            "the follow-up frame replayed the cached leaf, so the deferred \
             window-scope axes were lost on the way to it"
        );
        let still_dirty = window
            .update(cx, |_, window, _| window.invalidator.is_dirty())
            .unwrap();
        assert!(
            !still_dirty,
            "the window never settled after the refreshes stopped"
        );
    }

    #[gpui::test]
    fn test_set_app_id_via_options(cx: &mut TestAppContext) {
        let window = cx.open_window(size(px(800.), px(600.)), |_, _| EmptyView);

        window
            .update(cx, |_, this, _| {
                this.set_app_id("com.example.test-app");
            })
            .ok();
    }

    #[gpui::test]
    fn test_set_app_id_via_method(cx: &mut TestAppContext) {
        let window = cx.open_window(size(px(800.), px(600.)), |_, _| EmptyView);

        window
            .update(cx, |_, this, _| {
                this.set_app_id("com.example.another-app");
            })
            .ok();
    }

    #[gpui::test]
    fn test_set_app_id_update(cx: &mut TestAppContext) {
        let window = cx.open_window(size(px(800.), px(600.)), |_, _| EmptyView);

        window
            .update(cx, |_, this, _| {
                this.set_app_id("com.example.initial");
            })
            .ok();

        window
            .update(cx, |_, this, _| {
                this.set_app_id("com.example.updated");
            })
            .ok();
    }

    #[gpui::test]
    #[should_panic(expected = "this method can only be called during paint")]
    fn on_frame_cannot_call_paint_methods(cx: &mut TestAppContext) {
        use crate::{div, fill};

        struct PaintFromOnFrame;

        impl Render for PaintFromOnFrame {
            fn render(
                &mut self,
                _window: &mut Window,
                _cx: &mut Context<Self>,
            ) -> impl IntoElement {
                div().on_frame(|geom, window, _cx| {
                    window.paint_quad(fill(geom.bounds, crate::red()));
                })
            }
        }

        cx.open_window(size(px(800.), px(600.)), |_, _| PaintFromOnFrame);
        cx.run_until_parked();
    }

    /// `on_frame` receives resolved geometry precisely so that it never needs
    /// to build anything. Requesting layout from it would put element-tree
    /// construction back on a path that also runs for cached subtrees, which is
    /// the whole thing the channel is designed to avoid.
    #[gpui::test]
    #[should_panic(expected = "this method can only be called during request_layout, or prepaint")]
    fn on_frame_cannot_request_layout(cx: &mut TestAppContext) {
        use crate::{Style, div};

        struct LayoutFromOnFrame;

        impl Render for LayoutFromOnFrame {
            fn render(
                &mut self,
                _window: &mut Window,
                _cx: &mut Context<Self>,
            ) -> impl IntoElement {
                div().on_frame(|_geom, window, cx| {
                    window.request_layout(Style::default(), None, cx);
                })
            }
        }

        cx.open_window(size(px(800.), px(600.)), |_, _| LayoutFromOnFrame);
        cx.run_until_parked();
    }

    /// Element state is keyed by `GlobalElementId` and migrated forward by
    /// whichever frame accessed it. An effect replayed on behalf of a view that
    /// never ran this frame would register accesses for a subtree that does not
    /// exist, so `with_element_state` is out of bounds from `on_frame` too.
    #[gpui::test]
    #[should_panic(
        expected = "this method can only be called during request_layout, prepaint, or paint"
    )]
    fn on_frame_cannot_touch_element_state(cx: &mut TestAppContext) {
        use crate::div;

        struct ElementStateFromOnFrame;

        impl Render for ElementStateFromOnFrame {
            fn render(
                &mut self,
                _window: &mut Window,
                _cx: &mut Context<Self>,
            ) -> impl IntoElement {
                div()
                    .on_frame(|_geom, window: &mut Window, _cx| {
                        let id = crate::GlobalElementId(std::sync::Arc::from(
                            [crate::ElementId::Name("root".into())].as_slice(),
                        ));
                        window.with_element_state::<usize, _>(&id, |state, _| {
                            ((), state.unwrap_or(0))
                        });
                    })
                    .id("root")
            }
        }

        cx.open_window(size(px(800.), px(600.)), |_, _| ElementStateFromOnFrame);
        cx.run_until_parked();
    }

    /// A view whose `on_frame` records its own bounds, nested under two plain
    /// divs with different bounds.
    struct GeometryStasher {
        seen: std::rc::Rc<std::cell::RefCell<Vec<crate::Bounds<crate::Pixels>>>>,
    }

    impl crate::Render for GeometryStasher {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            let seen = self.seen.clone();
            crate::div().size_full().child(
                crate::div().w(px(400.)).h(px(300.)).child(
                    crate::div()
                        .w(px(100.))
                        .h(px(50.))
                        .on_frame(move |geom, _, _| seen.borrow_mut().push(geom.bounds))
                        .id("stasher"),
                ),
            )
        }
    }

    /// The effect must fire once per frame, with *this* element's geometry.
    ///
    /// It used to fire once per ancestor as well, because `Div::on_frame`
    /// recursed into its children while `Drawable::prepaint` was already
    /// calling `on_frame` on every element it walked. The ancestor's call ran
    /// last and passed the *ancestor's* bounds, so an element stashing
    /// `geom.bounds` ended up holding the size of whatever div was furthest up
    /// the tree — which is exactly the stale-geometry defect this channel was
    /// added to close, reintroduced by the channel itself.
    #[gpui::test]
    fn on_frame_runs_once_per_element_with_its_own_geometry(cx: &mut TestAppContext) {
        let seen = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let seen_for_view = seen.clone();
        let _window = cx.open_window(size(px(800.), px(600.)), move |_, _| GeometryStasher {
            seen: seen_for_view,
        });
        cx.run_until_parked();

        let recorded = seen.borrow().clone();
        assert_eq!(
            recorded.len(),
            1,
            "on_frame fired {} times for one element in one frame: {recorded:?}",
            recorded.len()
        );
        assert_eq!(recorded[0].size, size(px(100.), px(50.)));
    }

    /// The acceptance condition for the effects channel: a cached view replays
    /// without rendering, *and* the effects inside it still fire, with the
    /// geometry they resolved against.
    struct CachedStasherRoot {
        stasher: crate::Entity<GeometryStasher>,
    }

    impl crate::Render for CachedStasherRoot {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            crate::div().size_full().child(
                crate::AnyView::from(self.stasher.clone())
                    .cached(crate::StyleRefinement::default().w(px(400.)).h(px(300.))),
            )
        }
    }

    #[gpui::test]
    fn on_frame_effects_survive_a_cached_view_reusing(cx: &mut TestAppContext) {
        let seen = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let seen_for_view = seen.clone();
        let stasher = cx.update(|cx| {
            cx.new(move |_| GeometryStasher {
                seen: seen_for_view,
            })
        });
        let window = cx.open_window(size(px(800.), px(600.)), {
            let stasher = stasher.clone();
            move |_, _| CachedStasherRoot { stasher }
        });
        cx.run_until_parked();

        let after_first = seen.borrow().len();
        assert_eq!(after_first, 1, "the first frame renders and stashes once");

        for round in 0..3 {
            clean_frame(cx, window.into());
            assert_eq!(
                seen.borrow().len(),
                after_first + round + 1,
                "round {round}: the cached view replayed but its on_frame effect did not run"
            );
            assert_eq!(
                seen.borrow().last().copied().unwrap().size,
                size(px(100.), px(50.)),
                "round {round}: the replayed effect got the wrong geometry"
            );
        }
    }

    // -------------------------------------------------------------------
    // Retained layers
    // -------------------------------------------------------------------

    /// A view with a `.layer()` div wrapping a canvas that counts its own
    /// paints.
    ///
    /// The canvas is the observable: `paint` runs only when the layer actually
    /// re-emits its content, so the counter distinguishes "composited" from
    /// "re-rendered" — which nothing else about a correct layer does, because a
    /// composite is meant to be indistinguishable in its output.
    struct LayerView {
        paints: std::rc::Rc<std::cell::Cell<usize>>,
    }

    impl crate::Render for LayerView {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            let paints = self.paints.clone();
            crate::div().size_full().child(
                crate::div()
                    .id("panel")
                    .layer()
                    // Positioned away from the origin so the test platform's
                    // mouse position (0,0) is outside it: a layer under the
                    // pointer always re-renders.
                    .absolute()
                    .left(px(200.))
                    .top(px(200.))
                    .w(px(100.))
                    .h(px(100.))
                    .cursor_pointer()
                    .bg(crate::red())
                    .child(
                        crate::canvas(
                            |_, _, _| (),
                            move |bounds, _, window, _| {
                                paints.set(paints.get() + 1);
                                window.paint_quad(crate::fill(bounds, crate::blue()));
                            },
                        )
                        .w(px(10.))
                        .h(px(10.)),
                    )
                    // Interactivity registered during the subtree's paint —
                    // the part a composite skips and has to replay.
                    .child(
                        crate::div()
                            .w(px(10.))
                            .h(px(10.))
                            .cursor_pointer()
                            .on_mouse_down(crate::MouseButton::Left, |_, _, _| {}),
                    ),
            )
        }
    }

    /// The layer tests assert on retained state that `WGPUI_LAYERS=0` is
    /// defined to remove. Skipping under the kill switch is what makes "the
    /// switch reverts to the old path" a claim the suite actually checks: with
    /// it set, every remaining test must still pass.
    fn layers_off() -> bool {
        !crate::layer::layers_enabled()
    }

    fn layer_window(
        cx: &mut TestAppContext,
    ) -> (
        crate::WindowHandle<LayerView>,
        std::rc::Rc<std::cell::Cell<usize>>,
    ) {
        let paints = std::rc::Rc::new(std::cell::Cell::new(0));
        let paints_for_view = paints.clone();
        let window = cx.open_window(size(px(800.), px(600.)), move |_, _| LayerView {
            paints: paints_for_view,
        });
        cx.run_until_parked();
        (window, paints)
    }

    /// The phase's headline acceptance: a layer with unchanged content
    /// composites across frames instead of re-emitting its primitives.
    #[gpui::test]
    fn an_unchanged_layer_composites_instead_of_re_rendering(cx: &mut TestAppContext) {
        if layers_off() {
            return;
        }
        let (window, paints) = layer_window(cx);
        let painted_once = paints.get();
        assert_eq!(painted_once, 1, "the first frame renders the layer");

        let key = window
            .update(cx, |_, this, _| {
                assert_eq!(this.layers.len(), 1, "the `.layer()` div created a layer");
                *this.layers.keys().next().unwrap()
            })
            .unwrap();

        for round in 0..4 {
            clean_frame(cx, window.into());
            window
                .update(cx, |_, this, _| {
                    let layer = this.layers.get(&key).expect("layer survived the frame");
                    assert!(
                        layer.has_content(),
                        "round {round}: the layer lost its retained primitives"
                    );
                    assert_eq!(
                        layer.last_visited, this.layer_frame,
                        "round {round}: the layer was not visited this frame, so it \
                         neither composited nor re-rendered"
                    );
                    assert!(
                        layer.needs.is_empty(),
                        "round {round}: a clean frame left the layer invalidated"
                    );
                })
                .unwrap();
            assert_eq!(
                paints.get(),
                painted_once,
                "round {round}: the layer re-emitted its content instead of compositing"
            );
        }
    }

    #[gpui::test]
    fn a_layer_hitbox_moves_with_its_transform(cx: &mut TestAppContext) {
        if layers_off() {
            return;
        }
        let (window, _) = layer_window(cx);

        window
            .update(cx, |_, this, _| {
                let key = *this.layers.keys().next().expect("the layer was recorded");
                let layer = this.layers.get_mut(&key).expect("the layer still exists");
                layer.transform.offset = crate::Point::new(px(220.), px(210.));

                let hit_test = this
                    .rendered_frame
                    .hit_test(crate::Point::new(px(225.), px(215.)), &this.layers);
                assert!(
                    !hit_test.ids.is_empty(),
                    "the child hitbox should move with its layer transform"
                );

                let previous_position = this
                    .rendered_frame
                    .hit_test(crate::Point::new(px(205.), px(205.)), &this.layers);
                assert!(
                    previous_position.ids.is_empty(),
                    "the hitbox must not remain at the layer's previous position"
                );
            })
            .unwrap();
    }

    // -------------------------------------------------------------------
    // Transform-only composites (#94)
    // -------------------------------------------------------------------

    /// Serializes the stats-based test below against its own siblings. The
    /// counter registry is process-global; only tests that trigger the
    /// transform-only path bump its counter, and these tests are the only
    /// ones that do.
    static TRANSFORM_STATS_ORDERING: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

    /// A `.layer_keyed(..)` panel whose position and size are driven through
    /// shared state, so a frame can move it through real layout while its
    /// content key holds. This is the exact shape the transform-only
    /// composite path exists for: a panel translated by its surroundings,
    /// whose own subtree did not change.
    ///
    /// The interactive children exist so the layer carries real hitboxes —
    /// a move must carry them along via [`LayerTransform`] — and so the
    /// mouse-inside gate has something to be inside of.
    struct MovableLayerView {
        origin: std::rc::Rc<std::cell::Cell<crate::Point<crate::Pixels>>>,
        extent: std::rc::Rc<std::cell::Cell<crate::Size<crate::Pixels>>>,
        version: u64,
        paints: std::rc::Rc<std::cell::Cell<usize>>,
    }

    impl crate::Render for MovableLayerView {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            let origin = self.origin.get();
            let extent = self.extent.get();
            let paints = self.paints.clone();
            crate::div().size_full().child(
                crate::div()
                    .id("panel")
                    .layer_keyed(self.version)
                    // Away from the test platform's (0,0) mouse position: a
                    // layer under the pointer always re-renders.
                    .absolute()
                    .left(origin.x)
                    .top(origin.y)
                    .w(extent.width)
                    .h(extent.height)
                    .bg(crate::red())
                    .cursor_pointer()
                    .child(
                        crate::canvas(
                            |_, _, _| (),
                            move |bounds, _, window, _| {
                                paints.set(paints.get() + 1);
                                window.paint_quad(crate::fill(bounds, crate::blue()));
                            },
                        )
                        .w(px(50.))
                        .h(px(50.)),
                    )
                    .child(
                        crate::div()
                            .w(px(10.))
                            .h(px(10.))
                            .on_mouse_down(crate::MouseButton::Left, |_, _, _| {}),
                    ),
            )
        }
    }

    fn movable_layer_window(
        cx: &mut TestAppContext,
    ) -> (
        crate::WindowHandle<MovableLayerView>,
        std::rc::Rc<std::cell::Cell<usize>>,
        std::rc::Rc<std::cell::Cell<crate::Point<crate::Pixels>>>,
        std::rc::Rc<std::cell::Cell<crate::Size<crate::Pixels>>>,
    ) {
        let origin = std::rc::Rc::new(std::cell::Cell::new(crate::point(px(200.), px(200.))));
        let extent = std::rc::Rc::new(std::cell::Cell::new(size(px(100.), px(100.))));
        let paints = std::rc::Rc::new(std::cell::Cell::new(0));
        let origin_for_view = origin.clone();
        let extent_for_view = extent.clone();
        let paints_for_view = paints.clone();
        let window = cx.open_window(
            size(px(800.), px(600.)),
            move |_, _| MovableLayerView {
                origin: origin_for_view,
                extent: extent_for_view,
                version: 0,
                paints: paints_for_view,
            },
        );
        cx.run_until_parked();
        (window, paints, origin, extent)
    }

    /// Translate the panel through shared state and notify, the way an
    /// ancestor-layout change would.
    fn move_movable_layer(
        window: &crate::WindowHandle<MovableLayerView>,
        origin: &std::rc::Rc<std::cell::Cell<crate::Point<crate::Pixels>>>,
        cx: &mut TestAppContext,
        dx: f32,
        dy: f32,
    ) {
        let old = origin.get();
        origin.set(crate::Point::new(old.x + px(dx), old.y + px(dy)));
        window.update(cx, |_, _, cx| cx.notify()).unwrap();
        cx.run_until_parked();
    }

    /// `(content_token, totals, origin)` for every slab span of `key` in the
    /// rendered frame. With slabs live, the token staying put while the
    /// origin moves is precisely the contract the renderer side pins (see
    /// slab_gpu's write-log tests and the GPU-tier transform readback):
    /// Clean plan, zero instance bytes, one transform slot written.
    fn slab_spans_for(
        window: &Window,
        key: crate::LayerKey,
    ) -> Vec<(u64, [u32; crate::platform::cross::slab::SlabKind::COUNT], [f32; 2])> {
        window
            .rendered_frame
            .scene
            .layer_slab_spans
            .iter()
            .filter(|span| span.key == key)
            .map(|span| (span.content_token, span.totals, span.origin))
            .collect()
    }

    /// The #94 headline acceptance: translating a keyed layer whose content
    /// did not change skips the re-record entirely. The retained packed bytes
    /// are re-emitted under the SAME slab token at the NEW origin, which the
    /// renderer resolves as Clean — zero instance bytes, one uniform slot —
    /// and the following idle frame goes silent again.
    #[gpui::test]
    fn a_transform_only_move_skips_the_record_and_reemits_the_same_token(
        cx: &mut TestAppContext,
    ) {
        if layers_off() {
            return;
        }
        let slabs = crate::scene_pack::slabs_enabled();
        let (window, paints, origin, _) = movable_layer_window(cx);
        let painted_once = paints.get();
        assert_eq!(painted_once, 1, "the first frame renders the layer");

        // Spans exist only from a layer's second frame on: the first one
        // records, it does not composite yet.
        clean_frame(cx, window.into());

        let (key, scale_factor, first_span, first_token) = window
            .update(cx, |_, this, _| {
                let key = *this.layers.keys().next().expect("the layer was recorded");
                (
                    key,
                    this.scale_factor(),
                    slab_spans_for(this, key),
                    this.slab_tokens.get(&key).copied(),
                )
            })
            .unwrap();
        if slabs {
            assert!(
                !first_span.is_empty(),
                "the layer must composite as slab spans"
            );
        }
        let expected_origin = |x: f32, y: f32| [x * scale_factor, y * scale_factor];

        move_movable_layer(&window, &origin, cx, 40., 30.);

        window
            .update(cx, |_, this, _| {
                let layer = &this.layers[&key];
                assert_eq!(
                    layer.cache_key.bounds.origin,
                    crate::point(px(240.), px(230.)),
                    "the layer's cache key must be re-stamped to the new position"
                );
                assert_eq!(
                    layer.transform.offset,
                    crate::point(px(240.), px(230.)),
                    "hit-test inversion depends on the transform following the move"
                );
                if slabs {
                    assert_eq!(
                        slab_spans_for(this, key),
                        vec![(
                            first_span[0].0,
                            first_span[0].1,
                            expected_origin(240., 230.)
                        )],
                        "the moved frame must re-emit the SAME token and totals at \
                         the NEW origin"
                    );
                    assert_eq!(
                        this.slab_tokens.get(&key).copied(),
                        first_token,
                        "a transform-only move must not bump the slab token"
                    );
                }
            })
            .unwrap();

        assert_eq!(
            paints.get(),
            if slabs { painted_once } else { painted_once + 1 },
            "with slabs live the move must not re-record; without them the old \
             full-re-render behaviour stands"
        );

        // The frame after the move is an ordinary clean frame: nothing may
        // regress back into re-recording, and nothing may move again.
        clean_frame(cx, window.into());
        window
            .update(cx, |_, this, _| {
                let layer = &this.layers[&key];
                assert_eq!(layer.transform.offset, crate::point(px(240.), px(230.)));
                if slabs {
                    assert_eq!(
                        slab_spans_for(this, key),
                        vec![(
                            first_span[0].0,
                            first_span[0].1,
                            expected_origin(240., 230.)
                        )],
                        "the idle frame must be byte-identical to the moved frame"
                    );
                }
            })
            .unwrap();
        assert_eq!(
            paints.get(),
            if slabs { painted_once } else { painted_once + 1 },
            "the idle frame re-rendered the layer"
        );
    }

    /// A transform-only move must take the layer's input geometry with it:
    /// the hitboxes were registered relative to the layer's origin, and the
    /// updated [`LayerTransform`] is what maps a world-space query back in.
    /// A stale offset would leave clicks landing one move behind the pixels.
    #[gpui::test]
    fn a_transform_moved_layer_hit_tests_at_its_new_position(cx: &mut TestAppContext) {
        if layers_off() {
            return;
        }
        let (window, paints, origin, _) = movable_layer_window(cx);
        let painted_once = paints.get();

        move_movable_layer(&window, &origin, cx, 40., 30.);
        assert_eq!(
            paints.get(),
            if crate::scene_pack::slabs_enabled() {
                painted_once
            } else {
                painted_once + 1
            },
            "setup: the move behaved unexpectedly"
        );

        window
            .update(cx, |_, this, _| {
                // The layer used to span (200..300); it now spans
                // (240..340). Probe points that belong to exactly one of the
                // two regions, since the move is smaller than the layer.
                let at_new_only = this
                    .rendered_frame
                    .hit_test(crate::Point::new(px(330.), px(320.)), &this.layers);
                assert!(
                    !at_new_only.ids.is_empty(),
                    "the layer's hitboxes must follow its transform-only move"
                );
                let at_old_only = this
                    .rendered_frame
                    .hit_test(crate::Point::new(px(205.), px(205.)), &this.layers);
                assert!(
                    at_old_only.ids.is_empty(),
                    "the hitboxes must not remain at the layer's previous position"
                );
            })
            .unwrap();
    }

    /// Resizing is NOT a transform-only move: the packed instance bytes
    /// describe the old extent, so the layer must go through a full
    /// re-record — which also bumps its slab token, forcing the renderer to
    /// upload the new geometry.
    #[gpui::test]
    fn a_resize_re_records_instead_of_taking_the_transform_path(cx: &mut TestAppContext) {
        if layers_off() {
            return;
        }
        let slabs = crate::scene_pack::slabs_enabled();
        let (window, paints, origin, extent) = movable_layer_window(cx);
        let painted_once = paints.get();

        move_movable_layer(&window, &origin, cx, 40., 30.);
        let paints_after_move = paints.get();
        assert_eq!(
            paints_after_move,
            if slabs { painted_once } else { painted_once + 1 },
            "setup: the pure move behaved unexpectedly"
        );

        let (key, token_after_move) = window
            .update(cx, |_, this, _| {
                let key = *this.layers.keys().next().expect("the layer was recorded");
                (key, this.slab_tokens.get(&key).copied())
            })
            .unwrap();

        extent.set(size(px(140.), px(100.)));
        window.update(cx, |_, _, cx| cx.notify()).unwrap();
        cx.run_until_parked();

        window
            .update(cx, |_, this, _| {
                let layer = &this.layers[&key];
                assert_eq!(layer.cache_key.bounds.size.width, px(140.));
                if slabs {
                    assert_ne!(
                        this.slab_tokens.get(&key).copied(),
                        token_after_move,
                        "a re-record must hand the renderer a fresh token"
                    );
                }
            })
            .unwrap();
        assert_eq!(
            paints.get(),
            paints_after_move + 1,
            "the resize must re-record regardless of the kill-switch state"
        );
    }

    /// Changing the layer's content key while it moves is the caller
    /// withdrawing the claim the fast path rests on: full re-record.
    #[gpui::test]
    fn a_content_key_change_re_records_even_while_the_layer_moves(cx: &mut TestAppContext) {
        if layers_off() {
            return;
        }
        let (window, paints, origin, _) = movable_layer_window(cx);
        let painted_once = paints.get();

        // Move AND change the key in the same frame: whatever the geometry
        // did, the key change alone forbids reuse.
        let old = origin.get();
        origin.set(crate::Point::new(old.x + px(40.), old.y + px(30.)));
        window
            .update(cx, |view, _, cx| {
                view.version += 1;
                cx.notify();
            })
            .unwrap();
        cx.run_until_parked();

        assert_eq!(
            paints.get(),
            painted_once + 1,
            "a changed content key must force a re-record"
        );
    }

    /// The documented conservatism carries over verbatim: a layer under the
    /// pointer re-renders, because hover state is read during paint and
    /// nothing invalidates for it.
    #[gpui::test]
    fn a_moved_layer_under_the_pointer_still_re_records(cx: &mut TestAppContext) {
        if layers_off() {
            return;
        }
        let (window, paints, origin, _) = movable_layer_window(cx);
        let painted_once = paints.get();

        let old = origin.get();
        origin.set(crate::Point::new(old.x + px(40.), old.y + px(30.)));
        window
            .update(cx, |_, this, cx| {
                // The panel now sits at (240, 230)..(340, 330).
                this.mouse_position = crate::Point::new(px(250.), px(240.));
                cx.notify();
            })
            .unwrap();
        cx.run_until_parked();

        assert_eq!(
            paints.get(),
            painted_once + 1,
            "mouse-inside must fall back to the normal record path"
        );
        window
            .update(cx, |_, this, _| {
                let key = *this.layers.keys().next().expect("the layer was recorded");
                assert!(
                    this.layers[&key].had_mouse,
                    "the re-record must stamp had_mouse like any other"
                );
            })
            .unwrap();
    }

    /// A layer carrying a backdrop filter poisons the pixels behind it and
    /// cannot express its content as slab instances at all. It must never
    /// take the fast path — the fallback is the ordinary re-record.
    #[gpui::test]
    fn a_backdrop_poisoned_layer_re_records_when_moved(cx: &mut TestAppContext) {
        if layers_off() {
            return;
        }
        struct PoisonedMovableLayer {
            origin: std::rc::Rc<std::cell::Cell<crate::Point<crate::Pixels>>>,
            paints: std::rc::Rc<std::cell::Cell<usize>>,
        }

        impl crate::Render for PoisonedMovableLayer {
            fn render(
                &mut self,
                _window: &mut Window,
                _cx: &mut Context<Self>,
            ) -> impl IntoElement {
                let origin = self.origin.get();
                let paints = self.paints.clone();
                crate::div().size_full().child(
                    crate::div()
                        .id("frosted")
                        .layer_keyed(0u64)
                        .absolute()
                        .left(origin.x)
                        .top(origin.y)
                        .w(px(100.))
                        .h(px(100.))
                        .bg(crate::red())
                        .child(
                            crate::canvas(
                                |_, _, _| (),
                                move |bounds, _, window, _| {
                                    paints.set(paints.get() + 1);
                                    window.paint_quad(crate::fill(bounds, crate::blue()));
                                },
                            )
                            .w(px(50.))
                            .h(px(50.)),
                        )
                        .child(crate::div().w(px(60.)).h(px(60.)).backdrop_blur(px(4.))),
                )
            }
        }

        let origin = std::rc::Rc::new(std::cell::Cell::new(crate::point(px(200.), px(200.))));
        let paints = std::rc::Rc::new(std::cell::Cell::new(0));
        let origin_for_view = origin.clone();
        let paints_for_view = paints.clone();
        let window = cx.open_window(
            size(px(800.), px(600.)),
            move |_, _| PoisonedMovableLayer {
                origin: origin_for_view,
                paints: paints_for_view,
            },
        );
        cx.run_until_parked();
        let painted_once = paints.get();

        // Note: the operative gate here is pack rejection — a BackdropFilter
        // primitive has no slab kind, so `build_slab_segments` refuses the
        // layer and the fast path declines. (The window-side poisoned_bounds
        // bookkeeping is cleared at the end of every record, so it is not an
        // observable here.)
        let old = origin.get();
        origin.set(crate::Point::new(old.x + px(40.), old.y + px(30.)));
        window.update(cx, |_, _, cx| cx.notify()).unwrap();
        cx.run_until_parked();

        assert_eq!(
            paints.get(),
            painted_once + 1,
            "the poisoned layer must re-record instead of transforming"
        );
    }

    /// A layer whose retained content includes nested layer references
    /// declines the fast path: each nested layer's origin is refreshed only
    /// by its own paint walk, which a composited parent skips, so splicing
    /// them under a moved parent could draw them at their previous
    /// positions. Ambiguity falls back to the record path.
    #[gpui::test]
    fn a_layer_with_nested_layers_re_records_when_moved(cx: &mut TestAppContext) {
        if layers_off() {
            return;
        }
        struct OuterWithNestedLayer {
            origin: std::rc::Rc<std::cell::Cell<crate::Point<crate::Pixels>>>,
            outer_paints: std::rc::Rc<std::cell::Cell<usize>>,
            inner_paints: std::rc::Rc<std::cell::Cell<usize>>,
        }

        impl crate::Render for OuterWithNestedLayer {
            fn render(
                &mut self,
                _window: &mut Window,
                _cx: &mut Context<Self>,
            ) -> impl IntoElement {
                let origin = self.origin.get();
                let outer_paints = self.outer_paints.clone();
                let inner_paints = self.inner_paints.clone();
                crate::div().size_full().child(
                    crate::div()
                        .id("outer")
                        .layer_keyed(0u64)
                        .absolute()
                        .left(origin.x)
                        .top(origin.y)
                        .w(px(100.))
                        .h(px(100.))
                        .bg(crate::red())
                        .child(
                            crate::canvas(
                                |_, _, _| (),
                                move |bounds, _, window, _| {
                                    outer_paints.set(outer_paints.get() + 1);
                                    window.paint_quad(crate::fill(bounds, crate::blue()));
                                },
                            )
                            .w(px(20.))
                            .h(px(20.)),
                        )
                        .child(
                            crate::div()
                                .id("inner")
                                .layer()
                                .absolute()
                                .left(px(30.))
                                .top(px(30.))
                                .w(px(40.))
                                .h(px(40.))
                                .child(
                                    crate::canvas(
                                        |_, _, _| (),
                                        move |bounds, _, window, _| {
                                            inner_paints.set(inner_paints.get() + 1);
                                            window.paint_quad(crate::fill(bounds, crate::green()));
                                        },
                                    )
                                    .size_full(),
                                ),
                        ),
                )
            }
        }

        let origin = std::rc::Rc::new(std::cell::Cell::new(crate::point(px(150.), px(150.))));
        let outer_paints = std::rc::Rc::new(std::cell::Cell::new(0));
        let inner_paints = std::rc::Rc::new(std::cell::Cell::new(0));
        let origin_for_view = origin.clone();
        let outer_paints_for_view = outer_paints.clone();
        let inner_paints_for_view = inner_paints.clone();
        let window = cx.open_window(
            size(px(800.), px(600.)),
            move |_, _| OuterWithNestedLayer {
                origin: origin_for_view,
                outer_paints: outer_paints_for_view,
                inner_paints: inner_paints_for_view,
            },
        );
        cx.run_until_parked();
        let outer_once = outer_paints.get();
        let inner_once = inner_paints.get();

        let old = origin.get();
        origin.set(crate::Point::new(old.x + px(40.), old.y + px(30.)));
        window.update(cx, |_, _, cx| cx.notify()).unwrap();
        cx.run_until_parked();

        assert_eq!(
            outer_paints.get(),
            outer_once + 1,
            "the outer layer must decline the fast path and re-record"
        );
        assert_eq!(
            inner_paints.get(),
            inner_once + 1,
            "re-recording the parent repaints the nested layer's subtree"
        );
    }

    /// Occluder bookkeeping must follow a transform-only move. The moved
    /// foreground's opaque coverage is recorded in window coordinates; if it
    /// did not shift with the layer, the background would stay culled (or
    /// deferred-dirty) against coverage at the foreground's OLD position.
    #[gpui::test]
    fn moving_a_transform_only_occluder_updates_what_it_covers(cx: &mut TestAppContext) {
        if layers_off() || occlusion_off() {
            return;
        }
        let slabs = crate::scene_pack::slabs_enabled();
        struct MovableOccluderView {
            fg_origin: std::rc::Rc<std::cell::Cell<crate::Point<crate::Pixels>>>,
            bg_paints: std::rc::Rc<std::cell::Cell<usize>>,
            fg_paints: std::rc::Rc<std::cell::Cell<usize>>,
        }

        impl crate::Render for MovableOccluderView {
            fn render(
                &mut self,
                _window: &mut Window,
                _cx: &mut Context<Self>,
            ) -> impl IntoElement {
                let fg_origin = self.fg_origin.get();
                let bg_paints = self.bg_paints.clone();
                let fg_paints = self.fg_paints.clone();
                crate::div()
                    .size_full()
                    .child(
                        crate::div()
                            .id("bg")
                            .layer_keyed(0u64)
                            .absolute()
                            .left(px(200.))
                            .top(px(200.))
                            .w(px(200.))
                            .h(px(200.))
                            .bg(crate::green())
                            .child(crate::canvas(
                                |_, _, _| (),
                                move |bounds, _, window, _| {
                                    bg_paints.set(bg_paints.get() + 1);
                                    window.paint_quad(crate::fill(bounds, crate::blue()));
                                },
                            )),
                    )
                    .child(
                        crate::div()
                            .id("fg")
                            .layer_keyed(0u64)
                            .absolute()
                            .left(fg_origin.x)
                            .top(fg_origin.y)
                            .w(px(200.))
                            .h(px(200.))
                            .bg(crate::red())
                            .child(crate::canvas(
                                |_, _, _| (),
                                move |bounds, _, window, _| {
                                    fg_paints.set(fg_paints.get() + 1);
                                    window.paint_quad(crate::fill(bounds, crate::blue()));
                                },
                            )),
                    )
            }
        }

        let fg_origin = std::rc::Rc::new(std::cell::Cell::new(crate::point(px(200.), px(200.))));
        let bg_paints = std::rc::Rc::new(std::cell::Cell::new(0));
        let fg_paints = std::rc::Rc::new(std::cell::Cell::new(0));
        let fg_origin_for_view = fg_origin.clone();
        let bg_paints_for_view = bg_paints.clone();
        let fg_paints_for_view = fg_paints.clone();
        let window = cx.open_window(
            size(px(800.), px(600.)),
            move |_, _| MovableOccluderView {
                fg_origin: fg_origin_for_view,
                bg_paints: bg_paints_for_view,
                fg_paints: fg_paints_for_view,
            },
        );
        cx.run_until_parked();
        let bg_once = bg_paints.get();
        let fg_once = fg_paints.get();

        // Both layers fully overlap, so the foreground covers the
        // background. Both are keyed, so view notifies never rebuild them:
        // every transition below is driven by exactly one mechanism.
        let move_fg = |window: &crate::WindowHandle<MovableOccluderView>,
                       cx: &mut TestAppContext,
                       dx: f32,
                       dy: f32| {
            let old = fg_origin.get();
            fg_origin.set(crate::Point::new(old.x + px(dx), old.y + px(dy)));
            window.update(cx, |_, _, cx| cx.notify()).unwrap();
            cx.run_until_parked();
        };

        // A first clean frame settles occlusion: the background is culled
        // under the foreground's opaque region.
        clean_frame(cx, window.into());

        // Reveal: move the foreground off the background. The frame that
        // moves the occluder still judges the background against its
        // previous-frame coverage (paint order) — the documented cross-frame
        // cost the record-path occluder tests also pin — so the reveal lands
        // on the following clean frame, where the background must COMPOSITE
        // (visited). Culling is what stale coverage would keep producing; it
        // only stops if the foreground's opaque region moved away with it.
        move_fg(&window, cx, 300., 300.);
        assert_eq!(
            fg_paints.get(),
            if slabs { fg_once } else { fg_once + 1 },
            "the foreground itself must not re-record when slabs are live"
        );

        clean_frame(cx, window.into());
        let revealed = window
            .update(cx, |_, this, _| {
                let bg_key = this
                    .layers
                    .iter()
                    .min_by_key(|(_, layer)| layer.id)
                    .expect("both layers exist")
                    .0;
                (
                    this.layers[&bg_key].last_visited == this.layer_frame,
                    bg_paints.get(),
                )
            })
            .unwrap();
        assert!(
            revealed.0,
            "the background must stop being culled once the occluder's \
             coverage has moved away with it"
        );
        assert_eq!(
            revealed.1, bg_once,
            "a revealed-and-clean background composites; it does not re-render"
        );

        // Cover again and invalidate ONLY the background: covered and dirty,
        // it must go deferred-dirty rather than rebuild unseen.
        move_fg(&window, cx, -300., -300.);
        window
            .update(cx, |_, this, _| {
                let bg_key = this
                    .layers
                    .iter()
                    .min_by_key(|(_, layer)| layer.id)
                    .expect("both layers exist")
                    .0;
                this.invalidator
                    .invalidate_layer(*bg_key, Invalidation::all());
            })
            .unwrap();
        cx.run_until_parked();
        window
            .update(cx, |_, this, _| {
                let bg_key = this
                    .layers
                    .iter()
                    .min_by_key(|(_, layer)| layer.id)
                    .expect("both layers exist")
                    .0;
                assert!(
                    this.layers[&bg_key].deferred_dirty,
                    "setup: the invalidated background must be deferred-dirty under \
                     the covering foreground"
                );
            })
            .unwrap();

        // Reveal again. The deferred-dirty background may only rebuild once
        // the moved coverage no longer hides it — the same shift-dependent
        // decision, now through the deferred-dirty gate.
        let bg_before_release = bg_paints.get();
        move_fg(&window, cx, 300., 300.);
        clean_frame(cx, window.into());

        assert!(
            bg_paints.get() > bg_before_release,
            "the deferred-dirty background must rebuild once the moved \
             coverage no longer hides it"
        );
        window
            .update(cx, |_, this, _| {
                let bg_key = this
                    .layers
                    .iter()
                    .min_by_key(|(_, layer)| layer.id)
                    .expect("both layers exist")
                    .0;
                assert!(
                    !this.layers[&bg_key].deferred_dirty,
                    "deferred_dirty must clear after the revealed rebuild"
                );
            })
            .unwrap();
    }
    // -------------------------------------------------------------------
    // Instance-tier occlusion culling (#95)
    // -------------------------------------------------------------------

    /// Quads of a layer's retained items whose background matches `color`.
    fn quads_with_background(items: &[crate::layer::LayerItem], color: crate::Hsla) -> usize {
        items
            .iter()
            .filter(|item| {
                matches!(
                    item,
                    crate::layer::LayerItem::Primitive(crate::scene::Primitive::Quad(quad))
                        if quad.background.solid == color
                )
            })
            .count()
    }

    /// A layer holding two canvases; an opaque sibling painted between them
    /// fully covers the first and leaves the second alone.
    struct CoveredInstanceView {
        paints: std::rc::Rc<std::cell::Cell<usize>>,
    }

    impl crate::Render for CoveredInstanceView {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            let paints = self.paints.clone();
            crate::div().size_full().child(
                crate::div()
                    .id("panel")
                    .layer()
                    .absolute()
                    .left(px(200.))
                    .top(px(200.))
                    .w(px(200.))
                    .h(px(200.))
                    // The covered cullee.
                    .child(
                        crate::canvas(
                            |_, _, _| (),
                            move |bounds, _, window, _| {
                                paints.set(paints.get() + 1);
                                window.paint_quad(crate::fill(bounds, crate::blue()));
                            },
                        )
                        .absolute()
                        .left(px(0.))
                        .top(px(0.))
                        .w(px(40.))
                        .h(px(40.)),
                    )
                    // Opaque sibling painted after it.
                    .child(
                        crate::div()
                            .absolute()
                            .left(px(0.))
                            .top(px(0.))
                            .w(px(60.))
                            .h(px(60.))
                            .bg(crate::red()),
                    )
                    // Control: same kind of content, outside the cover.
                    .child(
                        crate::canvas(
                            |_, _, _| (),
                            move |bounds, _, window, _| {
                                window.paint_quad(crate::fill(bounds, crate::green()));
                            },
                        )
                        .absolute()
                        .left(px(100.))
                        .top(px(100.))
                        .w(px(40.))
                        .h(px(40.)),
                    ),
            )
        }
    }

    /// Occluded instances inside a dirty layer emit no primitives on the next
    /// record, and the reserved counter demonstrates it.
    #[gpui::test]
    fn occluded_instances_inside_a_dirty_layer_emit_no_primitives(cx: &mut TestAppContext) {
        if layers_off() || occlusion_off() {
            return;
        }
        let paints = std::rc::Rc::new(std::cell::Cell::new(0));
        let paints_for_view = paints.clone();
        let window = cx.open_window(
            size(px(800.), px(600.)),
            move |_, _| CoveredInstanceView {
                paints: paints_for_view,
            },
        );
        cx.run_until_parked();
        let painted_once = paints.get();
        assert_eq!(painted_once, 1, "setup: the canvas painted once");

        let _ordering = TRANSFORM_STATS_ORDERING.lock();
        let counter = "occlusion: instances culled";
        crate::render_stats::set_force_enabled(true);
        let before = crate::render_stats::snapshot();
        let before_culled = before.counters.get(counter).copied().unwrap_or(0);

        // Dirty the layer without touching its content: the record runs again,
        // and the sweep drops the covered canvas's quad.
        window.update(cx, |_, _, cx| cx.notify()).unwrap();
        cx.run_until_parked();

        let after = crate::render_stats::snapshot();
        crate::render_stats::set_force_enabled(false);
        let culled = after
            .counters
            .get(counter)
            .copied()
            .unwrap_or(0)
            - before_culled;

        assert_eq!(
            paints.get(),
            painted_once + 1,
            "setup: the dirty layer re-rendered"
        );
        assert!(culled >= 1, "the sweep must report the covered instance");

        window
            .update(cx, |_, window, _| {
                assert_eq!(window.layers.len(), 1, "setup: exactly one layer");
                let layer = window.layers.values().next().expect("the layer exists");
                assert_eq!(
                    quads_with_background(&layer.items, crate::blue()),
                    0,
                    "the fully covered canvas must not emit"
                );
                assert!(
                    quads_with_background(&layer.items, crate::red()) >= 1,
                    "the occluder itself still emits"
                );
                assert!(
                    quads_with_background(&layer.items, crate::green()) >= 1,
                    "the uncovered control still emits"
                );
            })
            .unwrap();
    }

    /// A static bottom layer under an animating opaque top layer: the trap
    /// this phase exists to avoid. The animation churns the TOP layer's slab
    /// every frame while the bottom layer's bytes never move — it neither
    /// re-renders nor repacks, so the renderer keeps judging its slab Clean
    /// and uploads zero bytes for it (the CPU-decision half of `slab: bytes
    /// uploaded`; the renderer-side half is pinned by slab_gpu's own tests).
    ///
    /// The two layers live under separate views: a notified view re-renders
    /// every non-deferred layer beneath it, so sharing one view would make
    /// the bottom layer re-record for reasons unrelated to occlusion.
    struct TrapStaticBottomView {
        paints: std::rc::Rc<std::cell::Cell<usize>>,
    }

    impl crate::Render for TrapStaticBottomView {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            let paints = self.paints.clone();
            crate::div()
                .id("bottom")
                .layer()
                .absolute()
                .left(px(150.))
                .top(px(150.))
                .w(px(300.))
                .h(px(300.))
                .bg(crate::green())
                .child(crate::canvas(
                    |_, _, _| (),
                    move |bounds, _, window, _| {
                        paints.set(paints.get() + 1);
                        window.paint_quad(crate::fill(bounds, crate::blue()));
                    },
                ))
        }
    }

    struct TrapAnimatedOccluderView {
        origin: std::rc::Rc<std::cell::Cell<crate::Point<crate::Pixels>>>,
        paints: std::rc::Rc<std::cell::Cell<usize>>,
    }

    impl crate::Render for TrapAnimatedOccluderView {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            let origin = self.origin.get();
            let paints = self.paints.clone();
            crate::div()
                .id("top")
                .layer()
                .absolute()
                .left(origin.x)
                .top(origin.y)
                .w(px(120.))
                .h(px(120.))
                .bg(crate::red())
                .child(crate::canvas(
                    |_, _, _| (),
                    move |bounds, _, window, _| {
                        paints.set(paints.get() + 1);
                        window.paint_quad(crate::fill(bounds, crate::blue()));
                    },
                ))
        }
    }

    struct TrapRootView {
        bottom: crate::Entity<TrapStaticBottomView>,
        occluder: crate::Entity<TrapAnimatedOccluderView>,
    }

    impl crate::Render for TrapRootView {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            crate::div()
                .size_full()
                .child(self.bottom.clone())
                .child(self.occluder.clone())
        }
    }

    /// THE trap test: an occluder animating over a clean layer never forces
    /// that layer's repack or upload, even though the covered regions shift
    /// every frame.
    #[gpui::test]
    fn an_animated_occluder_never_churns_the_clean_layer_it_moves_over(
        cx: &mut TestAppContext,
    ) {
        if layers_off() || occlusion_off() {
            return;
        }
        let bottom_paints = std::rc::Rc::new(std::cell::Cell::new(0));
        let top_paints = std::rc::Rc::new(std::cell::Cell::new(0));
        let origin = std::rc::Rc::new(std::cell::Cell::new(crate::point(px(160.), px(160.))));
        let origin_for_view = origin.clone();
        let bottom_paints_for_view = bottom_paints.clone();
        let top_paints_for_view = top_paints.clone();

        let occluder = cx.update(|cx| {
            cx.new(|_| TrapAnimatedOccluderView {
                origin: origin_for_view,
                paints: top_paints_for_view,
            })
        });
        let occluder_handle = occluder.clone();
        let bottom = cx.update(|cx| {
            cx.new(|_| TrapStaticBottomView {
                paints: bottom_paints_for_view,
            })
        });
        let window =
            cx.open_window(size(px(800.), px(600.)), move |_, _| TrapRootView { bottom, occluder });
        cx.run_until_parked();

        // The bottom layer is the wider one (300px vs the 120px occluder).
        fn read_bottom(
            window: &crate::WindowHandle<TrapRootView>,
            cx: &mut TestAppContext,
        ) -> (u64, Option<u64>, usize, usize) {
            window
                .update(cx, |_, window, _| {
                    let key = *window
                        .layers
                        .iter()
                        .find(|(_, layer)| layer.cache_key.bounds.size.width == px(300.))
                        .map(|(key, _)| key)
                        .expect("the static bottom layer was recorded");
                    let layer = &window.layers[&key];
                    let pack_ptr = match &layer.packed {
                        Some(Ok(pack)) => std::sync::Arc::as_ptr(pack) as *const () as usize,
                        _ => 0,
                    };
                    (
                        layer.last_visited,
                        window.slab_tokens.get(&key).copied(),
                        pack_ptr,
                        layer.cache_key.bounds.size.width.0 as usize,
                    )
                })
                .unwrap()
        }

        let bottom_painted_once = bottom_paints.get();
        let top_painted_once = top_paints.get();
        let (_, token_before, pack_before, _) = read_bottom(&window, cx);

        for _round in 0..4 {
            let old = origin.get();
            origin.set(crate::Point::new(old.x + px(30.), old.y + px(20.)));
            // Notify only the animating view: the bottom layer's owner stays
            // clean, exactly like a real animation driving one subtree.
            occluder_handle.update(cx, |_, cx| cx.notify());
            cx.run_until_parked();

            assert_eq!(
                bottom_paints.get(),
                bottom_painted_once,
                "the animated occluder must never force the clean layer to re-render"
            );
        }

        let (_, token_after, pack_after, _) = read_bottom(&window, cx);
        window
            .update(cx, |_, window, _| {
                let key = *window
                    .layers
                    .iter()
                    .find(|(_, layer)| layer.cache_key.bounds.size.width == px(300.))
                    .map(|(key, _)| key)
                    .expect("the static bottom layer was recorded");
                let layer = &window.layers[&key];
                assert!(
                    layer.needs.is_empty(),
                    "the clean bottom layer must stay fully valid"
                );
                assert!(
                    !layer.deferred_dirty,
                    "partial coverage must not trigger the deferred-dirty path"
                );
            })
            .unwrap();
        assert_eq!(
            token_before, token_after,
            "a stable token is what makes the renderer treat the bottom \
             slab as Clean (zero uploads)"
        );
        if crate::scene_pack::slabs_enabled() {
            assert_ne!(pack_before, 0, "the bottom layer must have packed");
            assert_eq!(
                pack_before, pack_after,
                "the bottom layer's packed bytes were replaced: its slab would \
                 have to re-upload"
            );
        }
        assert_eq!(
            top_paints.get(),
            top_painted_once + 4,
            "setup: the animating layer re-recorded every frame"
        );
    }

    /// An instance culled while covered comes back exactly once when the
    /// occluder moves away inside the same layer: the move dirties the layer
    /// through ordinary element invalidation, one re-record recomputes the
    /// sweep with fresh geometry, and the cullee's primitives are emitted
    /// again — identical to what a full rebuild would have painted.
    struct InstanceRevealView {
        occluder_origin: std::rc::Rc<std::cell::Cell<crate::Point<crate::Pixels>>>,
        paints: std::rc::Rc<std::cell::Cell<usize>>,
    }

    impl crate::Render for InstanceRevealView {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            let occluder_origin = self.occluder_origin.get();
            let paints = self.paints.clone();
            crate::div().size_full().child(
                crate::div()
                    .id("panel")
                    .layer()
                    .absolute()
                    .left(px(150.))
                    .top(px(150.))
                    .w(px(300.))
                    .h(px(300.))
                    .child(
                        crate::canvas(
                            |_, _, _| (),
                            move |bounds, _, window, _| {
                                paints.set(paints.get() + 1);
                                window.paint_quad(crate::fill(bounds, crate::blue()));
                            },
                        )
                        .absolute()
                        .left(px(20.))
                        .top(px(20.))
                        .w(px(40.))
                        .h(px(40.)),
                    )
                    .child(
                        crate::div()
                            .absolute()
                            .left(occluder_origin.x)
                            .top(occluder_origin.y)
                            .w(px(80.))
                            .h(px(80.))
                            .bg(crate::red()),
                    ),
            )
        }
    }

    #[gpui::test]
    fn an_instance_culled_under_a_moving_occluder_comes_back_when_revealed(
        cx: &mut TestAppContext,
    ) {
        if layers_off() || occlusion_off() {
            return;
        }
        let paints = std::rc::Rc::new(std::cell::Cell::new(0));
        let origin = std::rc::Rc::new(std::cell::Cell::new(crate::point(px(10.), px(10.))));
        let origin_for_view = origin.clone();
        let paints_for_view = paints.clone();
        let window = cx.open_window(
            size(px(800.), px(600.)),
            move |_, _| InstanceRevealView {
                occluder_origin: origin_for_view,
                paints: paints_for_view,
            },
        );
        cx.run_until_parked();
        let painted_once = paints.get();

        fn blue_quads(window: &Window) -> usize {
            window
                .layers
                .values()
                .map(|layer| quads_with_background(&layer.items, crate::blue()))
                .sum()
        }

        // Covered: the canvas quad is baked out of the retained record.
        window
            .update(cx, |_, window, _| {
                assert_eq!(
                    blue_quads(window),
                    0,
                    "setup: the covered cullee must be culled on record"
                );
            })
            .unwrap();

        // Move the occluder away inside the layer; ordinary invalidation
        // re-records it once.
        origin.set(crate::point(px(220.), px(220.)));
        window.update(cx, |_, _, cx| cx.notify()).unwrap();
        cx.run_until_parked();

        assert_eq!(
            paints.get(),
            painted_once + 1,
            "the reveal must cost exactly one re-record"
        );
        window
            .update(cx, |_, window, _| {
                assert_eq!(
                    blue_quads(window),
                    1,
                    "revealed: the previously culled instance emits again"
                );
            })
            .unwrap();
    }

    /// Visual occlusion is not hit occlusion (#95 §"Must not skip hit
    /// registration"). Two identical buttons in one layer; an opaque sibling
    /// fully covers the first (its primitives are culled) and leaves the
    /// second alone. Hitboxes, listeners and dispatch nodes must behave
    /// identically for both — the internal differential for the Phase 5
    /// hit-test pattern, run with culling enabled.
    ///
    /// This is also the spec's explicit tab-panel case: a nested panel whose
    /// sibling covers it emits nothing yet stays fully clickable while its
    /// covering overlay carries no `BlockMouse`.
    struct ClickableUnderOverlayView {
        clicks_a: std::rc::Rc<std::cell::Cell<usize>>,
        clicks_b: std::rc::Rc<std::cell::Cell<usize>>,
    }

    impl crate::Render for ClickableUnderOverlayView {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            let clicks_a = self.clicks_a.clone();
            let clicks_b = self.clicks_b.clone();
            crate::div().size_full().child(
                crate::div()
                    .id("panel")
                    .layer()
                    .absolute()
                    .left(px(150.))
                    .top(px(150.))
                    .w(px(400.))
                    .h(px(300.))
                    // Button A sits under a deeply nested "tab panel".
                    .child(
                        crate::div()
                            .absolute()
                            .left(px(0.))
                            .top(px(0.))
                            .w(px(200.))
                            .h(px(200.))
                            .child(
                                crate::div().child(
                                    crate::div()
                                        .id("button-a")
                                        .absolute()
                                        .left(px(20.))
                                        .top(px(20.))
                                        .w(px(60.))
                                        .h(px(60.))
                                        .bg(crate::red())
                                        .on_mouse_down(crate::MouseButton::Left,
                                            move |_, _, _| clicks_a.set(clicks_a.get() + 1)),
                                ),
                            ),
                    )
                    // Opaque overlay painted after A: covers it completely.
                    .child(
                        crate::div()
                            .absolute()
                            .left(px(0.))
                            .top(px(0.))
                            .w(px(200.))
                            .h(px(200.))
                            .bg(crate::blue()),
                    )
                    // Button B is never covered.
                    .child(
                        crate::div()
                            .id("button-b")
                            .absolute()
                            .left(px(240.))
                            .top(px(20.))
                            .w(px(60.))
                            .h(px(60.))
                            .bg(crate::red())
                            .on_mouse_down(crate::MouseButton::Left,
                                move |_, _, _| clicks_b.set(clicks_b.get() + 1)),
                    ),
            )
        }
    }

    #[gpui::test]
    fn culling_leaves_hit_testing_and_clicks_identical(cx: &mut TestAppContext) {
        if layers_off() || occlusion_off() {
            return;
        }
        let clicks_a = std::rc::Rc::new(std::cell::Cell::new(0));
        let clicks_b = std::rc::Rc::new(std::cell::Cell::new(0));
        let window = cx.open_window(
            size(px(800.), px(600.)),
            {
                let clicks_a = clicks_a.clone();
                let clicks_b = clicks_b.clone();
                move |_, _| ClickableUnderOverlayView { clicks_a, clicks_b }
            },
        );
        cx.run_until_parked();

        window
            .update(cx, |_, window, _| {
                // Both buttons are red; the covered one's quad must be gone,
                // the uncovered one's must remain.
                // Quads are recorded in scaled pixels; buttons sit at
                // panel-local x 20 (covered) and 240 (control).
                fn quads_at(window: &Window, window_x: f32) -> usize {
                    let scale = window.scale_factor();
                    let target = window_x * scale;
                    window
                        .layers
                        .values()
                        .map(|layer| {
                            layer
                                .items
                                .iter()
                                .filter(|item| {
                                    matches!(
                                        item,
                                        crate::layer::LayerItem::Primitive(
                                            crate::scene::Primitive::Quad(quad),
                                        ) if quad.bounds.origin.x.0 == target
                                    )
                                })
                                .count()
                        })
                        .sum::<usize>()
                }
                assert_eq!(
                    quads_at(window, 170.),
                    0,
                    "setup: the covered button's quad is culled"
                );
                assert!(
                    quads_at(window, 390.) >= 1,
                    "setup: the uncovered control still emits"
                );
            })
            .unwrap();

        // Click the centers of both buttons: covered A at window (200, 200),
        // uncovered B at (420, 200).
        for position in [
            crate::point(px(200.), px(200.)),
            crate::point(px(420.), px(200.)),
        ] {
            window
                .update(cx, |_, window, cx| {
                    window.dispatch_event(
                        crate::PlatformInput::MouseDown(crate::MouseDownEvent {
                            button: crate::MouseButton::Left,
                            position,
                            modifiers: Default::default(),
                            click_count: 1,
                            first_mouse: false,
                        }),
                        cx,
                    );
                })
                .unwrap();
        }
        cx.run_until_parked();

        assert_eq!(
            clicks_a.get(),
            1,
            "the culled-but-not-blocked button must still receive the click"
        );
        assert_eq!(
            clicks_b.get(),
            1,
            "the uncovered control must behave as always"
        );

        // The Phase 5 differential half: hitboxes stay registered for culled
        // content at the same positions.
        window
            .update(cx, |_, window, _| {
                let hit_over_culled = window.rendered_frame.hit_test(
                    crate::point(px(200.), px(200.)),
                    &window.layers,
                );
                let hit_over_control = window.rendered_frame.hit_test(
                    crate::point(px(420.), px(200.)),
                    &window.layers,
                );
                assert!(
                    !hit_over_culled.ids.is_empty(),
                    "culled content keeps its hitboxes"
                );
                assert!(
                    !hit_over_control.ids.is_empty(),
                    "setup: the control registers hitboxes"
                );
            })
            .unwrap();
    }

    /// The window-side half of the headline metric, counted: one move costs
    /// exactly one fast-path composite, and the following idle frame costs
    /// zero. The renderer-side half — zero instance bytes and exactly one
    /// transform slot for that composite — is unreachable from window tests
    /// (the test platform has no renderer) and is pinned by slab_gpu's
    /// write-log tests plus the GPU-tier transform readback.
    ///
    /// `FORCE_ENABLED` is process-global and render_stats' own tests flip it
    /// concurrently; an interleaving can silence our counters mid-frame
    /// without anything being wrong. The structural half (the paint counter
    /// above) proves the fast path ran, so only the counter read is retried.
    #[gpui::test]
    fn a_transform_only_move_counts_exactly_one_fast_path_hit(cx: &mut TestAppContext) {
        if layers_off() || !crate::scene_pack::slabs_enabled() {
            return;
        }
        let _ordering = TRANSFORM_STATS_ORDERING.lock();
        let hit_counter = "layer: composited (transform-only)";

        for _attempt in 0..25 {
            let (window, paints, origin, _) = movable_layer_window(cx);
            let painted_once = paints.get();

            crate::render_stats::set_force_enabled(true);
            let before = crate::render_stats::snapshot();
            let before_hits = before.counters.get(hit_counter).copied().unwrap_or(0);

            move_movable_layer(&window, &origin, cx, 40., 30.);
            clean_frame(cx, window.into());

            let after = crate::render_stats::snapshot();
            crate::render_stats::set_force_enabled(false);
            let hits = after
                .counters
                .get(hit_counter)
                .copied()
                .unwrap_or(0)
                - before_hits;

            match paints.get() {
                value if value == painted_once => {}
                other => panic!(
                    "the transform-only move fell back to re-recording \
                     (paints {painted_once} -> {other})"
                ),
            }
            if hits == 1 {
                return;
            }
        }
        panic!("the transform-only counter never recorded its single hit");
    }

    // -------------------------------------------------------------------
    // Element instances + reconciliation (#92)
    // -------------------------------------------------------------------

    /// A view with a `.layer()` div wrapping a reconciled `Div` child — a
    /// leaf with static-shaped, no-pseudo-state styling, so it is eligible
    /// for `diff_key` and, unlike a `canvas` (which has no `diff_key` at all
    /// and, per the recursion fix, would force every ancestor above it to
    /// always be treated as changed), can legitimately be judged reusable.
    struct InstanceView {
        /// Drives the reconciled child's own style, so changing it is a real
        /// content change from that child's point of view.
        visible: bool,
        /// Bumped and notified to force the *layer* to re-render — without
        /// touching anything the reconciled child's `diff_key` reads — so a
        /// round can distinguish "notified but unchanged" from "changed".
        unrelated: usize,
    }

    impl crate::Render for InstanceView {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            let opacity = if self.visible { 1.0 } else { 0.5 };
            crate::div().size_full().child(
                crate::div()
                    .id("panel")
                    .layer()
                    // Positioned away from the test platform's (0,0) mouse
                    // position, the same reason `LayerView` does this: a
                    // layer under the pointer always re-renders.
                    .absolute()
                    .left(px(200.))
                    .top(px(200.))
                    .w(px(100.))
                    .h(px(100.))
                    .child(
                        crate::div()
                            .opacity(opacity)
                            .w(px(50.))
                            .h(px(50.))
                            .bg(crate::red()),
                    ),
            )
        }
    }

    fn instance_view_window(cx: &mut TestAppContext) -> crate::WindowHandle<InstanceView> {
        let window = cx.open_window(size(px(800.), px(600.)), move |_, _| InstanceView {
            visible: true,
            unrelated: 0,
        });
        cx.run_until_parked();
        window
    }

    fn instances_off() -> bool {
        !crate::instance::instances_enabled()
    }

    /// The address of the `diff_key` allocation the window's one
    /// `.layer()` subtree's single `ElementInstance` currently holds.
    ///
    /// A mechanical, unambiguous reuse signal — unlike comparing recorded
    /// *values* (a paint range, say), which can coincidentally match across
    /// a rebuild whenever two frames happen to paint the same amount of
    /// content at the same scene offset. `Layer::instances`'s entry for a
    /// reused child is never touched at all (see `prepaint_reconciled_child`'s
    /// `Reused` branch — no `.insert()` call on that path), so its `diff_key`
    /// `Box` survives with the same address; a rebuild always replaces the
    /// entry with a freshly allocated one.
    fn the_instance_diff_key_address(
        window: crate::WindowHandle<InstanceView>,
        cx: &mut TestAppContext,
    ) -> usize {
        window
            .update(cx, |_, this, _| {
                let key = *this.layers.keys().next().expect("the layer exists");
                let instance = this.layers[&key]
                    .instances
                    .values()
                    .next()
                    .expect("the reconciled child retained an ElementInstance");
                std::ptr::from_ref(instance.diff_key.as_ref()) as *const () as usize
            })
            .unwrap()
    }

    /// The phase's headline acceptance: inside a layer that *is* re-rendering
    /// (because its view was notified), a child whose own description did not
    /// change skips `prepaint`/`paint` entirely — its `ElementInstance` entry
    /// is left untouched, not merely rebuilt to an equivalent value — while a
    /// child whose description genuinely changed gets a freshly recorded one.
    #[gpui::test]
    fn an_unchanged_child_inside_a_re_rendering_layer_skips_paint(cx: &mut TestAppContext) {
        if layers_off() || instances_off() {
            return;
        }
        let window = instance_view_window(cx);
        window
            .update(cx, |_, this, _| {
                assert_eq!(this.layers.len(), 1, "the `.layer()` div created a layer");
                assert!(
                    !this.layers.values().next().unwrap().instances.is_empty(),
                    "the first frame must retain an ElementInstance for the reconciled child"
                );
            })
            .unwrap();
        let address_after_first_frame = the_instance_diff_key_address(window, cx);

        // Force the layer to re-render — not composite — by notifying its
        // owning view, without touching anything the reconciled child's
        // `diff_key` reads (`visible` stays put).
        window
            .update(cx, |view, _, cx| {
                view.unrelated += 1;
                cx.notify();
            })
            .unwrap();
        cx.run_until_parked();

        assert_eq!(
            the_instance_diff_key_address(window, cx),
            address_after_first_frame,
            "an unrelated notify that forces the layer to re-render must not repaint a \
             child whose own description is unchanged — its ElementInstance entry must \
             be left untouched, not merely rebuilt to an equivalent value"
        );

        // Now change what the reconciled child's own style actually is, and
        // notify again — this must repaint, recording a fresh range.
        window
            .update(cx, |view, _, cx| {
                view.visible = false;
                cx.notify();
            })
            .unwrap();
        cx.run_until_parked();

        assert_ne!(
            the_instance_diff_key_address(window, cx),
            address_after_first_frame,
            "a real change to the reconciled child's own style must still repaint it, \
             recording a freshly allocated ElementInstance entry"
        );
    }

    /// `WGPUI_INSTANCES=0` must reproduce pre-#92 behaviour exactly: no
    /// `ElementInstance` is ever retained, notified or not.
    #[gpui::test]
    fn disabling_instances_rebuilds_every_child_on_every_notify(cx: &mut TestAppContext) {
        if layers_off() || !instances_off() {
            return;
        }
        let window = instance_view_window(cx);

        window
            .update(cx, |_, this, _| {
                let key = *this.layers.keys().next().unwrap();
                assert!(
                    this.layers[&key].instances.is_empty(),
                    "WGPUI_INSTANCES=0 must retain no ElementInstances at all"
                );
            })
            .unwrap();

        window
            .update(cx, |view, _, cx| {
                view.unrelated += 1;
                cx.notify();
            })
            .unwrap();
        cx.run_until_parked();

        window
            .update(cx, |_, this, _| {
                let key = *this.layers.keys().next().unwrap();
                assert!(
                    this.layers[&key].instances.is_empty(),
                    "WGPUI_INSTANCES=0 must never start retaining ElementInstances"
                );
            })
            .unwrap();
    }

    // -------------------------------------------------------------------
    // Persistent Taffy layout tree (#93)
    // -------------------------------------------------------------------

    fn persistent_layout_off() -> bool {
        !crate::taffy::persistent_layout_enabled()
    }

    /// The phase's headline acceptance: a reconciled child whose description
    /// did not change keeps the *exact same* Taffy node across frames — not
    /// merely an equal one — proving `request_layout_or_reuse` really skipped
    /// node creation rather than recreating an indistinguishable node.
    #[gpui::test]
    fn an_unchanged_reconciled_child_keeps_its_taffy_node(cx: &mut TestAppContext) {
        if layers_off() || instances_off() || persistent_layout_off() {
            return;
        }
        let window = instance_view_window(cx);
        let layout_before = window
            .update(cx, |_, this, _| {
                let key = *this.layers.keys().next().unwrap();
                this.layers[&key].instances.values().next().unwrap().layout
            })
            .unwrap();

        window
            .update(cx, |view, _, cx| {
                view.unrelated += 1;
                cx.notify();
            })
            .unwrap();
        cx.run_until_parked();

        let layout_after = window
            .update(cx, |_, this, _| {
                let key = *this.layers.keys().next().unwrap();
                this.layers[&key].instances.values().next().unwrap().layout
            })
            .unwrap();

        assert_eq!(
            layout_before, layout_after,
            "an unrelated notify must not replace a reconciled child's Taffy node"
        );
    }

    /// `TaffyLayoutEngine::end_frame` must not leak: rendering many frames,
    /// alternating between an unrelated notify (everything reconciles) and a
    /// real change (the changed subtree rebuilds), must settle rather than
    /// grow the live node count without bound.
    #[gpui::test]
    fn taffy_node_count_does_not_grow_unboundedly(cx: &mut TestAppContext) {
        if layers_off() || instances_off() || persistent_layout_off() {
            return;
        }
        let window = instance_view_window(cx);

        let node_count = |cx: &mut TestAppContext| {
            window
                .update(cx, |_, this, _| {
                    this.layout_engine.as_ref().unwrap().live_node_count()
                })
                .unwrap()
        };

        let after_first_frame = node_count(cx);

        for round in 0..40 {
            window
                .update(cx, |view, _, cx| {
                    if round % 2 == 0 {
                        view.unrelated += 1;
                    } else {
                        view.visible = !view.visible;
                    }
                    cx.notify();
                })
                .unwrap();
            cx.run_until_parked();
        }

        let after_many_rounds = node_count(cx);
        assert_eq!(
            after_many_rounds, after_first_frame,
            "the live Taffy node count must return to its steady-state value, not \
             accumulate orphaned nodes across frames"
        );
    }

    /// A real content change still produces geometrically correct output —
    /// reuse is never allowed to let a stale size/position through. Resizing
    /// the reconciled child (a genuine `LAYOUT`-axis change, unlike
    /// `InstanceView`'s opacity-only change elsewhere in this module) must be
    /// reflected in its resolved bounds.
    struct ResizingView {
        width: crate::Pixels,
    }

    impl crate::Render for ResizingView {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            crate::div().size_full().child(
                crate::div()
                    .id("panel")
                    .layer()
                    .absolute()
                    .left(px(200.))
                    .top(px(200.))
                    .w(px(200.))
                    .h(px(100.))
                    .child(
                        crate::div()
                            .w(self.width)
                            .h(px(20.))
                            .bg(crate::red()),
                    ),
            )
        }
    }

    #[gpui::test]
    fn a_layout_change_still_resizes_the_reused_node(cx: &mut TestAppContext) {
        if layers_off() || instances_off() || persistent_layout_off() {
            return;
        }
        let window = cx.open_window(size(px(800.), px(600.)), move |_, _| ResizingView {
            width: px(20.),
        });
        cx.run_until_parked();

        let width_before = window
            .update(cx, |_, this, _| {
                let key = *this.layers.keys().next().unwrap();
                let layout = this.layers[&key].instances.values().next().unwrap().layout;
                this.layout_engine.as_mut().unwrap().layout_bounds(layout, 1.0).size.width
            })
            .unwrap();
        assert_eq!(width_before, px(20.));

        window
            .update(cx, |view, _, cx| {
                view.width = px(60.);
                cx.notify();
            })
            .unwrap();
        cx.run_until_parked();

        let width_after = window
            .update(cx, |_, this, _| {
                let key = *this.layers.keys().next().unwrap();
                let layout = this.layers[&key].instances.values().next().unwrap().layout;
                this.layout_engine.as_mut().unwrap().layout_bounds(layout, 1.0).size.width
            })
            .unwrap();
        assert_eq!(
            width_after,
            px(60.),
            "a genuine size change must resize the node, not replay the old size"
        );
    }

    /// `WGPUI_PERSISTENT_LAYOUT=0` must reproduce pre-#93 behaviour exactly:
    /// `Window::draw` calls `clear()`, never `end_frame()`, so the live node
    /// count after any draw reflects only what that single draw created —
    /// nothing survives from the frame before.
    #[gpui::test]
    fn disabling_persistent_layout_clears_every_frame(cx: &mut TestAppContext) {
        if layers_off() || !persistent_layout_off() {
            return;
        }
        let window = instance_view_window(cx);
        let count_after_first = window
            .update(cx, |_, this, _| {
                this.layout_engine.as_ref().unwrap().live_node_count()
            })
            .unwrap();
        assert!(count_after_first > 0, "the first frame creates nodes");

        clean_frame(cx, window.into());

        let count_after_second = window
            .update(cx, |_, this, _| {
                this.layout_engine.as_ref().unwrap().live_node_count()
            })
            .unwrap();
        assert_eq!(
            count_after_first, count_after_second,
            "clear()-then-fully-rebuild must land on the same steady-state count, \
             never accumulating across draws"
        );
    }

    /// Regression test for the bug `DivDiffKey`'s recursive children fixes: a
    /// `Div` with no id and static style, wrapping a `Text` child whose
    /// content is driven by view state. Before the fix, `DivDiffKey` only
    /// snapshotted the wrapper's own style plus a per-slot identity/type
    /// shape — it never looked at what each child's *own* description
    /// contained, so a content-only change several levels down (here: the
    /// text) went undetected and the whole subtree replayed stale.
    struct GrandchildTextView {
        text: std::rc::Rc<std::cell::RefCell<crate::SharedString>>,
        wrapper_paints: std::rc::Rc<std::cell::Cell<usize>>,
    }

    impl crate::Render for GrandchildTextView {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            let text = self.text.borrow().clone();
            let wrapper_paints = self.wrapper_paints.clone();
            crate::div().size_full().child(
                crate::div()
                    .id("panel")
                    .layer()
                    .absolute()
                    .left(px(200.))
                    .top(px(200.))
                    .w(px(100.))
                    .h(px(100.))
                    .child(
                        // The reconciled unit under test: static style, no
                        // id, one text child whose *content* varies. The
                        // canvas sibling counts every time this div's own
                        // `paint` actually runs — the same technique
                        // `LayerView`/`InstanceView` already use, just one
                        // level deeper.
                        crate::div().w(px(90.)).h(px(90.)).child(text).child(
                            crate::canvas(
                                |_, _, _| (),
                                move |bounds, _, window, _| {
                                    wrapper_paints.set(wrapper_paints.get() + 1);
                                    window.paint_quad(crate::fill(bounds, crate::blue()));
                                },
                            )
                            .w(px(1.))
                            .h(px(1.)),
                        ),
                    ),
            )
        }
    }

    #[gpui::test]
    fn a_grandchild_content_change_is_not_missed(cx: &mut TestAppContext) {
        if layers_off() || instances_off() {
            return;
        }
        let text = std::rc::Rc::new(std::cell::RefCell::new(crate::SharedString::from("AAAA")));
        let wrapper_paints = std::rc::Rc::new(std::cell::Cell::new(0));
        let text_for_view = text.clone();
        let wrapper_paints_for_view = wrapper_paints.clone();
        let window = cx.open_window(size(px(800.), px(600.)), move |_, _| GrandchildTextView {
            text: text_for_view,
            wrapper_paints: wrapper_paints_for_view,
        });
        cx.run_until_parked();
        let painted_once = wrapper_paints.get();
        assert_eq!(painted_once, 1, "the first frame paints the wrapper once");

        *text.borrow_mut() = crate::SharedString::from("this is a much longer string of text");
        window.update(cx, |view, _, cx| { let _ = view; cx.notify(); }).unwrap();
        cx.run_until_parked();

        assert_eq!(
            wrapper_paints.get(),
            painted_once + 1,
            "a text grandchild's content change must still repaint its ancestor \
             even when the ancestor's own style and per-slot identity/type shape \
             are unchanged"
        );
    }

    /// A layer-scope invalidation marks exactly the layer it names, and that
    /// layer re-renders on the next frame.
    #[gpui::test]
    fn invalidating_a_layer_marks_only_that_layer(cx: &mut TestAppContext) {
        if layers_off() {
            return;
        }
        let (window, _paints) = layer_window(cx);

        window
            .update(cx, |_, this, _| {
                let key = *this.layers.keys().next().unwrap();
                this.invalidator
                    .invalidate_layer(key, Invalidation::DISPLAY);
                this.invalidator
                    .invalidate_layer(crate::LayerKey(0xdead_beef), Invalidation::DISPLAY);
            })
            .unwrap();
        cx.run_until_parked();

        window
            .update(cx, |_, this, _| {
                assert_eq!(
                    this.layers.len(),
                    1,
                    "naming a key no layer holds must not create one"
                );
                let layer = this.layers.values().next().unwrap();
                assert!(
                    layer.needs.is_empty(),
                    "the layer was invalidated but did not re-render in answer"
                );
                assert!(layer.has_content());
            })
            .unwrap();
    }

    /// Eviction is mark-and-sweep on draw age. A layer that stops being visited
    /// gives up its primitives first and its record later, so a panel that
    /// scrolls away and back re-materialises into the same identity.
    #[gpui::test]
    fn stale_layers_lose_content_before_they_lose_identity(cx: &mut TestAppContext) {
        if layers_off() {
            return;
        }
        let (window, _paints) = layer_window(cx);

        let (key, id) = window
            .update(cx, |_, this, _| {
                let (key, layer) = this.layers.iter().next().unwrap();
                (*key, layer.id)
            })
            .unwrap();

        // Age the layer past `evict_after_frames` without visiting it.
        window
            .update(cx, |_, this, _| {
                let evict_after = this.layers[&key].policy.evict_after_frames as u64;
                this.layer_frame += evict_after + 1;
                this.evict_stale_layers();
            })
            .unwrap();

        window
            .update(cx, |_, this, _| {
                let layer = this
                    .layers
                    .get(&key)
                    .expect("the record must outlive its content");
                assert!(!layer.has_content(), "content should have been dropped");
                assert_eq!(layer.id, id, "identity must survive content eviction");
                assert_eq!(
                    layer.needs,
                    Invalidation::all(),
                    "an emptied layer must not be judged clean when next visited"
                );

                // Age it past the second interval: now the record goes too.
                this.layer_frame += layer.policy.evict_after_frames as u64 * 2;
                this.evict_stale_layers();
                assert!(this.layers.is_empty(), "the record should be gone");
            })
            .unwrap();
    }

    /// A notified view re-runs `render`, so the description behind its layers
    /// is new and may say something different. Nothing in this phase can
    /// compare descriptions — that is #92 — so a rebuilt view must re-render
    /// its layers.
    ///
    /// Without this the failure is silent and severe: a panel whose data
    /// changed keeps compositing last frame's pixels for as long as its bounds
    /// hold still, which for a docked panel is forever. It cannot be caught by
    /// the entity-dependency test either, because a view driven by state
    /// outside the entity graph — an `Arc<RwLock<_>>` polled by a frame pump,
    /// which is how the level editor works — reads no entity during paint at
    /// all.
    #[gpui::test]
    fn a_layer_in_a_notified_view_re_renders(cx: &mut TestAppContext) {
        if layers_off() {
            return;
        }
        let (window, paints) = layer_window(cx);
        let painted_once = paints.get();

        // Baseline: without a notify, it composites.
        clean_frame(cx, window.into());
        assert_eq!(paints.get(), painted_once, "precondition: it composites");

        for round in 0..3 {
            window.update(cx, |_, _, cx| cx.notify()).unwrap();
            cx.run_until_parked();
            assert_eq!(
                paints.get(),
                painted_once + round + 1,
                "round {round}: the view was notified and re-rendered, but its layer \
                 composited the description from before the change"
            );
        }
    }

    /// A view with a `layer_keyed` panel whose key is driven by `version`,
    /// separate from the notifies the view receives.
    struct KeyedLayerView {
        version: u64,
        paints: std::rc::Rc<std::cell::Cell<usize>>,
    }

    impl crate::Render for KeyedLayerView {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            let paints = self.paints.clone();
            crate::div().size_full().child(
                crate::div()
                    .id("panel")
                    .layer_keyed(self.version)
                    .absolute()
                    .left(px(200.))
                    .top(px(200.))
                    .w(px(100.))
                    .h(px(100.))
                    .child(
                        crate::canvas(
                            |_, _, _| (),
                            move |bounds, _, window, _| {
                                paints.set(paints.get() + 1);
                                window.paint_quad(crate::fill(bounds, crate::blue()));
                            },
                        )
                        .w(px(10.))
                        .h(px(10.)),
                    ),
            )
        }
    }

    /// `layer_keyed` exists for the case a plain `.layer()` cannot serve: a view
    /// notified every frame for a reason that has nothing to do with the
    /// subtree. The level editor's viewport is notified on every engine frame
    /// because its texture advanced, while the chrome over it is unchanged.
    ///
    /// So the key must hold across notifies — and must stop holding the moment
    /// it changes, or the declaration would be unenforceable.
    #[gpui::test]
    fn a_keyed_layer_composites_across_notifies_until_its_key_changes(cx: &mut TestAppContext) {
        if layers_off() {
            return;
        }
        let paints = std::rc::Rc::new(std::cell::Cell::new(0));
        let paints_for_view = paints.clone();
        let window = cx.open_window(size(px(800.), px(600.)), move |_, _| KeyedLayerView {
            version: 0,
            paints: paints_for_view,
        });
        cx.run_until_parked();
        let painted_once = paints.get();
        assert_eq!(painted_once, 1);

        // Notifying without touching the key must not re-render: this is
        // exactly what a plain `.layer()` would (correctly) refuse to do.
        for round in 0..3 {
            window.update(cx, |_, _, cx| cx.notify()).unwrap();
            cx.run_until_parked();
            assert_eq!(
                paints.get(),
                painted_once,
                "round {round}: the key was unchanged, so the layer should have composited"
            );
        }

        // Changing the key must re-render, or the declaration means nothing.
        window
            .update(cx, |view, _, cx| {
                view.version += 1;
                cx.notify();
            })
            .unwrap();
        cx.run_until_parked();
        assert_eq!(
            paints.get(),
            painted_once + 1,
            "the content key changed and the layer composited stale content anyway"
        );
    }

    /// Compositing skips the layer's subtree paint, and everything except
    /// primitives is registered *during* paint. A layer that composites while
    /// losing its mouse listeners looks perfectly correct and stops responding
    /// to clicks, which is the failure mode this whole epic is trying to stop
    /// reintroducing.
    #[gpui::test]
    fn a_composited_layer_keeps_its_subtrees_interactivity(cx: &mut TestAppContext) {
        if layers_off() {
            return;
        }
        let (window, _paints) = layer_window(cx);

        let baseline = window
            .update(cx, |_, this, _| {
                (
                    this.rendered_frame.mouse_listeners.len(),
                    this.rendered_frame.cursor_styles.len(),
                )
            })
            .unwrap();

        for round in 0..3 {
            clean_frame(cx, window.into());
            let after = window
                .update(cx, |_, this, _| {
                    (
                        this.rendered_frame.mouse_listeners.len(),
                        this.rendered_frame.cursor_styles.len(),
                    )
                })
                .unwrap();
            assert_eq!(
                after, baseline,
                "round {round}: compositing dropped what the skipped paint had registered"
            );
        }
    }

    /// A cached view's primitives come from its layer, and a frame that reuses
    /// must produce the same scene as the frame that rendered it.
    #[gpui::test]
    fn a_reused_cached_view_composites_the_same_scene(cx: &mut TestAppContext) {
        // The debug overlay adds primitives on purpose, so scene equality is
        // not a meaningful assertion with it on.
        if layers_off() || crate::layer::layer_debug_enabled() {
            return;
        }
        let (window, _leaf, _leaf_renders, _root_renders) = cached_leaf_window(cx, false);

        let rendered: Vec<u32> = window
            .update(cx, |_, this, _| {
                this.rendered_frame
                    .scene
                    .quads
                    .iter()
                    .map(|q| q.order)
                    .collect()
            })
            .unwrap();

        clean_frame(cx, window.into());

        let composited: Vec<u32> = window
            .update(cx, |_, this, _| {
                this.rendered_frame
                    .scene
                    .quads
                    .iter()
                    .map(|q| q.order)
                    .collect()
            })
            .unwrap();

        assert_eq!(
            rendered, composited,
            "compositing the cached view's layer produced a different scene than \
             painting it did"
        );
    }

    // -------------------------------------------------------------------
    // Occlusion culling
    // -------------------------------------------------------------------

    /// A two-layer view: a background layer behind a foreground layer of the
    /// same size. The foreground has a solid opaque background, so it ought to
    /// occlude the background.
    struct TwoLayerOcclusionView {
        bg_paints: std::rc::Rc<std::cell::Cell<usize>>,
        fg_paints: std::rc::Rc<std::cell::Cell<usize>>,
        /// If true, the foreground has `border_radius` which prevents full occlusion.
        fg_rounded: bool,
        /// If true, the foreground has non-opaque `.opacity(0.5)`.
        fg_translucent: bool,
        /// If true, the foreground has a backdrop filter that poisons the bg.
        fg_backdrop_filter: bool,
    }

    impl crate::Render for TwoLayerOcclusionView {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            let bg_paints = self.bg_paints.clone();
            let fg_paints = self.fg_paints.clone();

            let mut fg_div = crate::div()
                .id("fg")
                .layer()
                .absolute()
                .left(px(200.))
                .top(px(200.))
                .w(px(200.))
                .h(px(200.))
                .bg(crate::red());

            if self.fg_rounded {
                fg_div = fg_div.rounded(px(20.));
            }
            if self.fg_translucent {
                fg_div = fg_div.opacity(0.5);
            }

            let mut fg = fg_div.child(crate::canvas(
                |_, _, _| (),
                move |bounds, _, window, _| {
                    fg_paints.set(fg_paints.get() + 1);
                    window.paint_quad(crate::fill(bounds, crate::blue()));
                },
            ));

            if self.fg_backdrop_filter {
                fg = fg.child(
                    crate::div()
                        .w(px(200.))
                        .h(px(200.))
                        .backdrop_blur(px(10.)),
                );
            }

            crate::div().size_full().child(
                crate::div()
                    .id("bg")
                    .layer()
                    .absolute()
                    .left(px(200.))
                    .top(px(200.))
                    .w(px(200.))
                    .h(px(200.))
                    .bg(crate::green())
                    .child(
                        crate::canvas(
                            |_, _, _| (),
                            move |bounds, _, window, _| {
                                bg_paints.set(bg_paints.get() + 1);
                                window.paint_quad(crate::fill(bounds, crate::blue()));
                            },
                        )
                        .w(px(10.))
                        .h(px(10.)),
                    ),
            )
            .child(fg)
        }
    }

    fn occlusion_off() -> bool {
        !crate::occlusion::enabled()
    }

    fn two_layer_occlusion_window(
        cx: &mut TestAppContext,
        fg_rounded: bool,
        fg_translucent: bool,
        fg_backdrop_filter: bool,
    ) -> (
        crate::WindowHandle<TwoLayerOcclusionView>,
        std::rc::Rc<std::cell::Cell<usize>>,
        std::rc::Rc<std::cell::Cell<usize>>,
    ) {
        let bg_paints = std::rc::Rc::new(std::cell::Cell::new(0));
        let fg_paints = std::rc::Rc::new(std::cell::Cell::new(0));
        let bg_paints_for_view = bg_paints.clone();
        let fg_paints_for_view = fg_paints.clone();
        let window = cx.open_window(
            size(px(800.), px(600.)),
            move |_, _| TwoLayerOcclusionView {
                bg_paints: bg_paints_for_view,
                fg_paints: fg_paints_for_view,
                fg_rounded,
                fg_translucent,
                fg_backdrop_filter,
            },
        );
        cx.run_until_parked();
        (window, bg_paints, fg_paints)
    }

    /// The foreground occludes the background: on a clean frame the background
    /// layer is culled (not composited).
    #[gpui::test]
    fn occlusion_culls_covered_layer_on_clean_frame(cx: &mut TestAppContext) {
        if layers_off() || occlusion_off() {
            return;
        }
        let (window, bg_paints, fg_paints) =
            two_layer_occlusion_window(cx, false, false, false);
        let bg_painted_once = bg_paints.get();
        let fg_painted_once = fg_paints.get();
        assert_eq!(bg_painted_once, 1, "first frame renders both layers");
        assert_eq!(fg_painted_once, 1, "first frame renders both layers");

        // Find the two layer keys and verify the foreground occludes the background.
        let (bg_key, fg_key) = window
            .update(cx, |_, this, _| {
                let mut keys: Vec<_> = this.layers.keys().copied().collect();
                keys.sort(); // bg has lower id
                (keys[0], keys[1])
            })
            .unwrap();

        // On a clean frame, the foreground composites normally, the background
        // is culled.
        clean_frame(cx, window.into());
        window
            .update(cx, |_, this, _| {
                let bg = this.layers.get(&bg_key).unwrap();
                let fg = this.layers.get(&fg_key).unwrap();
                assert!(bg.has_content(), "bg retains its content");
                assert!(fg.has_content(), "fg retains its content");
                assert!(
                    bg.opaque_bounds.is_none() || fg.opaque_bounds.is_some(),
                    "the foreground should have opaque_bounds set from the solid bg"
                );
            })
            .unwrap();
        assert_eq!(
            bg_paints.get(),
            bg_painted_once,
            "the occluded bg must not re-render"
        );
        assert_eq!(
            fg_paints.get(),
            fg_painted_once,
            "the foreground must composite without re-rendering"
        );
    }

    /// A layer notified while occluded enters deferred_dirty and does not
    /// re-render until it becomes visible again.
    #[gpui::test]
    fn occlusion_deferred_dirty_when_notified_while_occluded(cx: &mut TestAppContext) {
        if layers_off() || occlusion_off() {
            return;
        }
        // The foreground's position is driven through shared state so the move
        // below goes through real layout. Editing the retained record directly
        // (`layers[key].cache_key.bounds`) would change no painted geometry —
        // that field is derived from layout on every re-render.
        let fg_origin = std::rc::Rc::new(std::cell::Cell::new(crate::point(px(200.), px(200.))));
        let bg_paints = std::rc::Rc::new(std::cell::Cell::new(0));
        let fg_paints = std::rc::Rc::new(std::cell::Cell::new(0));
        let fg_origin_for_view = fg_origin.clone();
        let bg_paints_for_view = bg_paints.clone();
        let fg_paints_for_view = fg_paints.clone();
        let window = cx.open_window(
            size(px(800.), px(600.)),
            move |_, _| RevealedOcclusionView {
                fg_origin: fg_origin_for_view,
                bg_paints: bg_paints_for_view,
                fg_paints: fg_paints_for_view,
            },
        );
        cx.run_until_parked();
        let bg_painted_once = bg_paints.get();
        let fg_painted_once = fg_paints.get();

        // Notify the view — this marks the layers dirty.
        window.update(cx, |_, _, cx| cx.notify()).unwrap();
        cx.run_until_parked();

        // The background is dirty but occluded, so it goes deferred dirty.
        // The foreground is dirty and visible, so it re-renders.
        window
            .update(cx, |_, this, _| {
                let bg_key = this
                    .layers
                    .iter()
                    .min_by_key(|(_, layer)| layer.id)
                    .expect("both layers exist")
                    .0;
                let bg = &this.layers[&bg_key];
                assert!(
                    bg.deferred_dirty,
                    "the occluded layer must be marked deferred_dirty; has_content={}, opaque_bounds={:?}",
                    bg.has_content(),
                    this.layers.iter().map(|(k, l)| (k.0, l.opaque_bounds)).collect::<Vec<_>>(),
                );
            })
            .unwrap();

        assert_eq!(
            bg_paints.get(),
            bg_painted_once,
            "the occluded bg must not re-render despite being notified"
        );
        assert_eq!(
            fg_paints.get(),
            fg_painted_once + 1,
            "the visible fg must re-render when notified"
        );

        // Move the foreground to (400, 400) so it no longer covers the
        // background at (200, 200), and notify.
        fg_origin.set(crate::point(px(400.), px(400.)));
        window.update(cx, |_, _, cx| cx.notify()).unwrap();
        cx.run_until_parked();

        // The frame that moves the occluder still judges the background
        // against the foreground's previous-frame opaque region: layers are
        // visited in paint order, so the background decides before the
        // foreground has re-recorded its new geometry. Staying culled for
        // exactly that frame is the documented cost of cross-frame occluder
        // data; the invariant is that the layer comes back (next frame).
        assert_eq!(
            bg_paints.get(),
            bg_painted_once,
            "the frame that moves the occluder still sees last frame's coverage"
        );

        clean_frame(cx, window.into());

        assert!(
            bg_paints.get() > bg_painted_once,
            "the bg must re-render once fresh occluder data shows it visible"
        );
        window
            .update(cx, |_, this, _| {
                let bg_key = this
                    .layers
                    .iter()
                    .min_by_key(|(_, layer)| layer.id)
                    .expect("both layers exist")
                    .0;
                let bg = &this.layers[&bg_key];
                assert!(
                    !bg.deferred_dirty,
                    "deferred_dirty must be cleared after re-rendering"
                );
            })
            .unwrap();
    }

    /// A two-layer view whose foreground can be moved at runtime: a background
    /// layer behind a fully-covering foreground layer, both with solid opaque
    /// backgrounds.
    struct RevealedOcclusionView {
        fg_origin: std::rc::Rc<std::cell::Cell<crate::Point<crate::Pixels>>>,
        bg_paints: std::rc::Rc<std::cell::Cell<usize>>,
        fg_paints: std::rc::Rc<std::cell::Cell<usize>>,
    }

    impl crate::Render for RevealedOcclusionView {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            let fg_origin = self.fg_origin.get();
            let bg_paints = self.bg_paints.clone();
            let fg_paints = self.fg_paints.clone();

            crate::div().size_full().child(
                crate::div()
                    .id("bg")
                    .layer()
                    .absolute()
                    .left(px(200.))
                    .top(px(200.))
                    .w(px(200.))
                    .h(px(200.))
                    .bg(crate::green())
                    .child(
                        crate::canvas(
                            |_, _, _| (),
                            move |bounds, _, window, _| {
                                bg_paints.set(bg_paints.get() + 1);
                                window.paint_quad(crate::fill(bounds, crate::blue()));
                            },
                        )
                        .w(px(10.))
                        .h(px(10.)),
                    ),
            )
            .child(
                crate::div()
                    .id("fg")
                    .layer()
                    .absolute()
                    .left(fg_origin.x)
                    .top(fg_origin.y)
                    .w(px(200.))
                    .h(px(200.))
                    .bg(crate::red())
                    .child(crate::canvas(
                        |_, _, _| (),
                        move |bounds, _, window, _| {
                            fg_paints.set(fg_paints.get() + 1);
                            window.paint_quad(crate::fill(bounds, crate::blue()));
                        },
                    )),
            )
        }
    }

    /// A backdrop filter poisons layers behind it — they must not be occluded.
    #[gpui::test]
    fn occlusion_backdrop_filter_poisons_layer_below(cx: &mut TestAppContext) {
        if layers_off() || occlusion_off() {
            return;
        }
        let (window, bg_paints, fg_paints) =
            two_layer_occlusion_window(cx, false, false, true);
        let bg_painted_once = bg_paints.get();
        let fg_painted_once = fg_paints.get();

        // Both layers rendered on the first frame.
        assert_eq!(bg_painted_once, 1);
        assert_eq!(fg_painted_once, 1);

        // On a clean frame, the foreground (with backdrop filter) must NOT
        // occlude the background — the backdrop filter reads behind it.
        clean_frame(cx, window.into());
        assert_eq!(
            bg_paints.get(),
            bg_painted_once,
            "the bg behind a backdrop filter must not be culled (but should \
             still composite normally)"
        );
        assert_eq!(
            fg_paints.get(),
            fg_painted_once,
            "the foreground composites normally"
        );
    }

    /// Rounded corners prevent full occlusion — the foreground's conservative
    /// opaque region is inset by the corner radius, so it doesn't fully cover
    /// the background.
    #[gpui::test]
    fn occlusion_rounded_corners_prevent_full_occlusion(cx: &mut TestAppContext) {
        if layers_off() || occlusion_off() {
            return;
        }
        let (window, bg_paints, fg_paints) =
            two_layer_occlusion_window(cx, true, false, false);
        let bg_painted_once = bg_paints.get();
        let fg_painted_once = fg_paints.get();

        assert_eq!(bg_painted_once, 1, "first frame renders both layers");
        assert_eq!(fg_painted_once, 1, "first frame renders both layers");

        // The rounded foreground's opaque_bounds, if set, would be inset by
        // 20px and would not cover the full 200x200 bg. But the key
        // behavioural assertion is that the bg is NOT culled.
        clean_frame(cx, window.into());
        assert_eq!(
            bg_paints.get(),
            bg_painted_once,
            "the bg behind a rounded fg must not be culled (it composites)"
        );
        assert_eq!(
            fg_paints.get(),
            fg_painted_once,
            "the foreground composites without re-rendering"
        );
    }

    /// Translucent element opacity prevents occlusion.
    #[gpui::test]
    fn occlusion_translucent_does_not_occlude(cx: &mut TestAppContext) {
        if layers_off() || occlusion_off() {
            return;
        }
        let (window, bg_paints, fg_paints) =
            two_layer_occlusion_window(cx, false, true, false);
        let bg_painted_once = bg_paints.get();
        let fg_painted_once = fg_paints.get();

        // The foreground has `opacity(0.5)`, so element_opacity < 1.0 and
        // `compute_opaque_region` returns None. The background should not be
        // occluded.
        let fg_key = window
            .update(cx, |_, this, _| {
                let mut keys: Vec<_> = this.layers.keys().copied().collect();
                keys.sort();
                let fg_key = keys[1];
                let fg = this.layers.get(&fg_key).unwrap();
                assert!(
                    fg.opaque_bounds.is_none(),
                    "a translucent foreground must not set opaque_bounds"
                );
                fg_key
            })
            .unwrap();

        let _ = fg_key;

        clean_frame(cx, window.into());
        assert_eq!(
            bg_paints.get(),
            bg_painted_once,
            "the bg behind a translucent fg must not be culled (composites)"
        );
        assert_eq!(
            fg_paints.get(),
            fg_painted_once,
            "the foreground composites"
        );
    }

    /// Hitboxes registered by an occluded layer survive the occlusion — visual
    /// occlusion is not hit occlusion.
    #[gpui::test]
    fn occlusion_hitboxes_survive_culled_layer(cx: &mut TestAppContext) {
        if layers_off() || occlusion_off() {
            return;
        }
        let (window, _bg_paints, _fg_paints) =
            two_layer_occlusion_window(cx, false, false, false);

        // After the initial render, count mouse listeners and cursor styles.
        let baseline = window
            .update(cx, |_, this, _| {
                (
                    this.rendered_frame.mouse_listeners.len(),
                    this.rendered_frame.cursor_styles.len(),
                )
            })
            .unwrap();

        // On a clean frame, the bg is occluded (culled). But its hitboxes must
        // survive — the cull path still calls `reuse_paint_except_scene`.
        for round in 0..3 {
            clean_frame(cx, window.into());
            let after = window
                .update(cx, |_, this, _| {
                    (
                        this.rendered_frame.mouse_listeners.len(),
                        this.rendered_frame.cursor_styles.len(),
                    )
                })
                .unwrap();
            assert_eq!(
                after, baseline,
                "round {round}: occlusion culling dropped hitboxes from the occluded layer"
            );
        }
    }

    /// Partial overlap — foreground leaves part of the background uncovered.
    #[gpui::test]
    fn occlusion_partial_overlap_does_not_cull(cx: &mut TestAppContext) {
        if layers_off() || occlusion_off() {
            return;
        }
        // A foreground layer at (100, 0, 100, 200) only covers half of the
        // background at (0, 0, 200, 200).
        let bg_paints = std::rc::Rc::new(std::cell::Cell::new(0));
        let fg_paints = std::rc::Rc::new(std::cell::Cell::new(0));
        let bg_paints_for_view = bg_paints.clone();
        let fg_paints_for_view = fg_paints.clone();
        let window = cx.open_window(
            size(px(800.), px(600.)),
            move |_, _| PartialOverlapView {
                bg_paints: bg_paints_for_view,
                fg_paints: fg_paints_for_view,
            },
        );
        cx.run_until_parked();
        let bg_painted_once = bg_paints.get();
        let fg_painted_once = fg_paints.get();

        assert_eq!(bg_painted_once, 1);
        assert_eq!(fg_painted_once, 1);

        // On a clean frame the bg is NOT culled — the fg only covers its right half.
        clean_frame(cx, window.into());
        assert_eq!(
            bg_paints.get(),
            bg_painted_once,
            "the partially-covered bg must not be culled (it composites)"
        );
        assert_eq!(fg_paints.get(), fg_painted_once, "the fg composites");
    }

    struct PartialOverlapView {
        bg_paints: std::rc::Rc<std::cell::Cell<usize>>,
        fg_paints: std::rc::Rc<std::cell::Cell<usize>>,
    }

    impl crate::Render for PartialOverlapView {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            let bg_paints = self.bg_paints.clone();
            let fg_paints = self.fg_paints.clone();
            crate::div().size_full().child(
                crate::div()
                    .id("bg")
                    .layer()
                    .absolute()
                    .left(px(200.))
                    .top(px(200.))
                    .w(px(200.))
                    .h(px(200.))
                    .bg(crate::green())
                    .child(crate::canvas(
                        |_, _, _| (),
                        move |bounds, _, window, _| {
                            bg_paints.set(bg_paints.get() + 1);
                            window.paint_quad(crate::fill(bounds, crate::blue()));
                        },
                    )),
            )
            .child(
                crate::div()
                    .id("fg")
                    .layer()
                    .absolute()
                    .left(px(300.))
                    .top(px(200.))
                    .w(px(100.))
                    .h(px(200.))
                    .bg(crate::red())
                    .child(crate::canvas(
                        |_, _, _| (),
                        move |bounds, _, window, _| {
                            fg_paints.set(fg_paints.get() + 1);
                            window.paint_quad(crate::fill(bounds, crate::blue()));
                        },
                    )),
            )
        }
    }

    /// Two occluders that together fully cover the target — one covers x 0-50,
    /// the other covers x 50-100.
    #[gpui::test]
    fn occlusion_two_occluders_combine_to_cover(cx: &mut TestAppContext) {
        if layers_off() || occlusion_off() {
            return;
        }
        let bg_paints = std::rc::Rc::new(std::cell::Cell::new(0));
        let fg1_paints = std::rc::Rc::new(std::cell::Cell::new(0));
        let fg2_paints = std::rc::Rc::new(std::cell::Cell::new(0));
        let bg_paints_for_view = bg_paints.clone();
        let fg1_paints_for_view = fg1_paints.clone();
        let fg2_paints_for_view = fg2_paints.clone();
        let window = cx.open_window(
            size(px(800.), px(600.)),
            move |_, _| TwoOccluderView {
                bg_paints: bg_paints_for_view,
                fg1_paints: fg1_paints_for_view,
                fg2_paints: fg2_paints_for_view,
            },
        );
        cx.run_until_parked();
        let bg_painted_once = bg_paints.get();

        // On a clean frame, the bg should be culled because fg1+fg2 cover it.
        clean_frame(cx, window.into());
        assert_eq!(
            bg_paints.get(),
            bg_painted_once,
            "the bg covered by two fg layers must be culled"
        );
    }

    struct TwoOccluderView {
        bg_paints: std::rc::Rc<std::cell::Cell<usize>>,
        fg1_paints: std::rc::Rc<std::cell::Cell<usize>>,
        fg2_paints: std::rc::Rc<std::cell::Cell<usize>>,
    }

    impl crate::Render for TwoOccluderView {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            let bg_paints = self.bg_paints.clone();
            let fg1_paints = self.fg1_paints.clone();
            let fg2_paints = self.fg2_paints.clone();
            crate::div().size_full().child(
                // Background layer at (200,200,100,200)
                crate::div()
                    .id("bg")
                    .layer()
                    .absolute()
                    .left(px(200.))
                    .top(px(200.))
                    .w(px(100.))
                    .h(px(200.))
                    .bg(crate::green())
                    .child(crate::canvas(
                        |_, _, _| (),
                        move |bounds, _, window, _| {
                            bg_paints.set(bg_paints.get() + 1);
                            window.paint_quad(crate::fill(bounds, crate::blue()));
                        },
                    )),
            )
            .child(
                // Foreground 1 covers left half at (200,200,50,200)
                crate::div()
                    .id("fg1")
                    .layer()
                    .absolute()
                    .left(px(200.))
                    .top(px(200.))
                    .w(px(50.))
                    .h(px(200.))
                    .bg(crate::red())
                    .child(crate::canvas(
                        |_, _, _| (),
                        move |bounds, _, window, _| {
                            fg1_paints.set(fg1_paints.get() + 1);
                            window.paint_quad(crate::fill(bounds, crate::blue()));
                        },
                    )),
            )
            .child(
                // Foreground 2 covers right half at (250,200,50,200)
                crate::div()
                    .id("fg2")
                    .layer()
                    .absolute()
                    .left(px(250.))
                    .top(px(200.))
                    .w(px(50.))
                    .h(px(200.))
                    .bg(crate::red())
                    .child(crate::canvas(
                        |_, _, _| (),
                        move |bounds, _, window, _| {
                            fg2_paints.set(fg2_paints.get() + 1);
                            window.paint_quad(crate::fill(bounds, crate::blue()));
                        },
                    )),
            )
        }
    }

    /// A zero-sized layer is trivially "covered" (see `occlusion.rs`'s
    /// zero-size rule) and emits nothing: every primitive it paints has empty
    /// clipped bounds and is dropped by the scene, so the layer retains no
    /// items and there is nothing to composite or cull. Its paint closures do
    /// re-run on clean frames — the reuse paths are gated on `has_content`,
    /// which stays false — but that re-recording is invisible work, not a
    /// correctness hazard. What must hold is no crash and no scene output.
    #[gpui::test]
    fn occlusion_zero_sized_layer_is_always_covered(cx: &mut TestAppContext) {
        if layers_off() || occlusion_off() {
            return;
        }
        let paints = std::rc::Rc::new(std::cell::Cell::new(0));
        let paints_for_view = paints.clone();
        let window = cx.open_window(
            size(px(800.), px(600.)),
            move |_, _| ZeroSizedLayerView {
                paints: paints_for_view,
            },
        );
        cx.run_until_parked();
        let painted_once = paints.get();
        assert_eq!(painted_once, 1, "the first frame runs the layer's paint");

        window
            .update(cx, |_, this, _| {
                let key = *this
                    .layers
                    .keys()
                    .next()
                    .expect("the `.layer()` div created a layer");
                assert!(
                    !this.layers[&key].has_content(),
                    "a zero-sized layer retains no items — the scene drops \
                     primitives whose clipped bounds are empty"
                );
            })
            .unwrap();

        let quads_after_first = quad_count(window.into(), cx);
        for _ in 0..3 {
            clean_frame(cx, window.into());
        }
        assert!(
            paints.get() > painted_once,
            "with nothing retained there is nothing to cull, so the layer's \
             paint re-runs each clean frame"
        );
        assert_eq!(
            quad_count(window.into(), cx),
            quads_after_first,
            "the zero-sized layer contributes no primitives to any frame"
        );
    }

    fn quad_count(window: crate::AnyWindowHandle, cx: &mut TestAppContext) -> usize {
        window
            .update(cx, |_, this, _| this.rendered_frame.scene.quads.len())
            .unwrap()
    }

    struct ZeroSizedLayerView {
        paints: std::rc::Rc<std::cell::Cell<usize>>,
    }

    impl crate::Render for ZeroSizedLayerView {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            let paints = self.paints.clone();
            crate::div().size_full().child(
                crate::div()
                    .id("zero")
                    .layer()
                    .absolute()
                    .left(px(200.))
                    .top(px(200.))
                    .w(px(0.))
                    .h(px(0.))
                    .bg(crate::red())
                    .child(crate::canvas(
                        |_, _, _| (),
                        move |bounds, _, window, _| {
                            paints.set(paints.get() + 1);
                            window.paint_quad(crate::fill(bounds, crate::blue()));
                        },
                    )),
            )
        }
    }

    /// An occluded layer that is evicted must not leave dangling deferred_dirty
    /// state — eviction clears the layer's content.
    #[gpui::test]
    fn occlusion_eviction_clears_deferred_dirty(cx: &mut TestAppContext) {
        if layers_off() || occlusion_off() {
            return;
        }
        let (window, _bg_paints, _fg_paints) =
            two_layer_occlusion_window(cx, false, false, false);

        let bg_key = window
            .update(cx, |_, this, _| {
                let mut keys: Vec<_> = this.layers.keys().copied().collect();
                keys.sort();
                keys[0]
            })
            .unwrap();

        // Mark the bg as deferred_dirty
        window
            .update(cx, |_, this, _| {
                let bg = this.layers.get_mut(&bg_key).unwrap();
                bg.deferred_dirty = true;
            })
            .unwrap();

        // Evict the bg by aging it past its eviction threshold.
        window
            .update(cx, |_, this, _| {
                let evict_after = this.layers[&bg_key].policy.evict_after_frames as u64;
                this.layer_frame += evict_after + 1;
                this.evict_stale_layers();
            })
            .unwrap();

        window
            .update(cx, |_, this, _| {
                let bg = this.layers.get(&bg_key).unwrap();
                assert!(
                    !bg.deferred_dirty,
                    "eviction must clear deferred_dirty"
                );
                assert!(
                    !bg.has_content(),
                    "eviction must drop retained content"
                );
                assert_eq!(
                    bg.needs,
                    Invalidation::all(),
                    "evicted layer needs a full rebuild"
                );
            })
            .unwrap();
    }

    /// A layer occluded by a non-opaque element (e.g. a background gradient)
    /// must not be culled.
    #[gpui::test]
    fn occlusion_non_solid_background_does_not_occlude(cx: &mut TestAppContext) {
        if layers_off() || occlusion_off() {
            return;
        }
        let bg_paints = std::rc::Rc::new(std::cell::Cell::new(0));
        let fg_paints = std::rc::Rc::new(std::cell::Cell::new(0));
        let bg_paints_for_view = bg_paints.clone();
        let fg_paints_for_view = fg_paints.clone();
        let window = cx.open_window(
            size(px(800.), px(600.)),
            move |_, _| NonSolidBgView {
                bg_paints: bg_paints_for_view,
                fg_paints: fg_paints_for_view,
            },
        );
        cx.run_until_parked();
        let bg_painted_once = bg_paints.get();

        // On a clean frame, the non-solid fg does not occlude the bg.
        clean_frame(cx, window.into());
        assert_eq!(
            bg_paints.get(),
            bg_painted_once,
            "the bg behind a non-solid fg must not be culled (it composites)"
        );
    }

    struct NonSolidBgView {
        bg_paints: std::rc::Rc<std::cell::Cell<usize>>,
        fg_paints: std::rc::Rc<std::cell::Cell<usize>>,
    }

    impl crate::Render for NonSolidBgView {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            let bg_paints = self.bg_paints.clone();
            let fg_paints = self.fg_paints.clone();
            crate::div().size_full().child(
                crate::div()
                    .id("bg")
                    .layer()
                    .absolute()
                    .left(px(200.))
                    .top(px(200.))
                    .w(px(200.))
                    .h(px(200.))
                    .bg(crate::green())
                    .child(crate::canvas(
                        |_, _, _| (),
                        move |bounds, _, window, _| {
                            bg_paints.set(bg_paints.get() + 1);
                            window.paint_quad(crate::fill(bounds, crate::blue()));
                        },
                    )),
            )
            .child(
                crate::div()
                    .id("fg")
                    .layer()
                    .absolute()
                    .left(px(200.))
                    .top(px(200.))
                    .w(px(200.))
                    .h(px(200.))
                    // Linear gradient background — NOT a solid color, so
                    // it does NOT qualify as an occluder.
                    .bg(crate::linear_gradient(
                        0.,
                        crate::gradient_color_stop(crate::red(), 0.0),
                        crate::gradient_color_stop(crate::blue(), 1.0),
                    ))
                    .child(crate::canvas(
                        |_, _, _| (),
                        move |bounds, _, window, _| {
                            fg_paints.set(fg_paints.get() + 1);
                            window.paint_quad(crate::fill(bounds, crate::blue()));
                        },
                    )),
            )
        }
    }

    // -------------------------------------------------------------------
    // Texture-retained layers and overscroll buffers (#96)
    // -------------------------------------------------------------------

    fn rasterization_off() -> bool {
        !crate::layer::rasterization_enabled()
    }

    /// A layer holding three canvases; the rasterize threshold is shared so a
    /// test can move it through the real render path.
    struct RasterizedView {
        paints: std::rc::Rc<std::cell::Cell<usize>>,
        threshold: std::rc::Rc<std::cell::Cell<usize>>,
    }

    impl crate::Render for RasterizedView {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            let paints = self.paints.clone();
            crate::div().size_full().child(
                crate::div()
                    .id("rasterized")
                    .layer_with_policy(crate::LayerPolicy {
                        rasterize_above: self.threshold.get(),
                        ..Default::default()
                    })
                    // Away from the origin: the test platform's mouse sits at
                    // (0,0), and a layer under the pointer always re-renders.
                    .absolute()
                    .left(px(200.))
                    .top(px(200.))
                    .w(px(200.))
                    .h(px(200.))
                    .child(crate::canvas(
                        |_, _, _| (),
                        move |bounds, _, window, _| {
                            paints.set(paints.get() + 1);
                            window.paint_quad(crate::fill(bounds, crate::blue()));
                        },
                    )
                    .absolute()
                    .left(px(0.))
                    .top(px(0.))
                    .size(px(40.))
                    .h(px(40.)))
                    .child(
                        crate::div()
                            .size(px(20.))
                            .h(px(20.))
                            .absolute()
                            .left(px(10.))
                            .top(px(10.))
                            .bg(crate::red()),
                    )
                    .child(
                        crate::div()
                            .size(px(20.))
                            .h(px(20.))
                            .absolute()
                            .left(px(40.))
                            .top(px(40.))
                            .bg(crate::green()),
                    ),
            )
        }
    }

    /// The #96 skip condition: a layer above `rasterize_above` bakes its
    /// content into a texture at record time, and a clean frame composites it
    /// with one `SurfaceContent::Layer` surface instead of slab spans.
    #[gpui::test]
    fn a_texture_retained_layer_composites_from_a_surface(cx: &mut TestAppContext) {
        if layers_off() || rasterization_off() {
            return;
        }
        let paints = std::rc::Rc::new(std::cell::Cell::new(0));
        let paints_for_view = paints.clone();
        let threshold = std::rc::Rc::new(std::cell::Cell::new(2usize));
        let window = cx.open_window(
            size(px(800.), px(600.)),
            move |_, _| RasterizedView {
                paints: paints_for_view,
                threshold: threshold.clone(),
            },
        );
        cx.run_until_parked();
        assert_eq!(paints.get(), 1, "setup: the record frame painted inline");

        window
            .update(cx, |_, this, _| {
                let layer = this.layers.values().next().expect("the layer exists");
                let kinds: Vec<&'static str> = layer
                    .items
                    .iter()
                    .map(|item| match item {
                        crate::layer::LayerItem::Primitive(crate::scene::Primitive::Quad(_)) => {
                            "quad"
                        }
                        crate::layer::LayerItem::Primitive(_) => "other",
                        crate::layer::LayerItem::Nested(_) => "nested",
                    })
                    .collect();
                assert!(
                    layer.texture_retained,
                    "three packable primitives exceed rasterize_above: 2 \
                     (packed_ok={:?}, items={} {kinds:?}, nested={}, slabs={}, raster_env={})",
                    layer.packed.as_ref().map(|packed| packed.is_ok()),
                    layer.items.len(),
                    layer
                        .items
                        .iter()
                        .any(|item| matches!(item, crate::layer::LayerItem::Nested(_))),
                    crate::scene_pack::slabs_enabled(),
                    crate::layer::rasterization_enabled(),
                );
                assert_eq!(
                    layer.texture_bounds, layer.cache_key.bounds,
                    "no margin: the texture covers exactly the layer bounds"
                );
            })
            .unwrap();

        let _ordering = TRANSFORM_STATS_ORDERING.lock();
        let counter = "layer: rasterized";
        crate::render_stats::set_force_enabled(true);
        let before = crate::render_stats::snapshot();
        let before_rasterized = before.counters.get(counter).copied().unwrap_or(0);

        clean_frame(cx, window.into());

        let after = crate::render_stats::snapshot();
        crate::render_stats::set_force_enabled(false);

        assert_eq!(
            paints.get(),
            1,
            "the clean frame must composite instead of re-rendering"
        );
        assert_eq!(
            after.counters.get(counter).copied().unwrap_or(0),
            before_rasterized,
            "a clean frame must not re-bake the texture"
        );

        window
            .update(cx, |_, this, _| {
                let scene = &this.rendered_frame.scene;
                assert!(
                    !scene.layer_slab_spans.is_empty()
                        || scene
                            .surfaces
                            .iter()
                            .any(|surface| matches!(surface.content, crate::scene::SurfaceContent::Layer(_))),
                    "the composite frame must carry the layer surface"
                );
                let layer_surfaces = scene
                    .surfaces
                    .iter()
                    .filter(|surface| {
                        matches!(surface.content, crate::scene::SurfaceContent::Layer(_))
                    })
                    .count();
                assert_eq!(
                    layer_surfaces, 1,
                    "the skip condition emits exactly one surface for the layer"
                );
            })
            .unwrap();
    }

    /// Below `rasterize_above` a layer stays primitive-retained: no texture
    /// decision, composites as slab spans, no surfaces in the scene.
    #[gpui::test]
    fn a_small_layer_stays_primitive_retained(cx: &mut TestAppContext) {
        if layers_off() || rasterization_off() {
            return;
        }
        let paints = std::rc::Rc::new(std::cell::Cell::new(0));
        let paints_for_view = paints.clone();
        let threshold = std::rc::Rc::new(std::cell::Cell::new(1000usize));
        let window = cx.open_window(
            size(px(800.), px(600.)),
            move |_, _| RasterizedView {
                paints: paints_for_view,
                threshold: threshold.clone(),
            },
        );
        cx.run_until_parked();

        window
            .update(cx, |_, this, _| {
                let layer = this.layers.values().next().expect("the layer exists");
                assert!(
                    !layer.texture_retained,
                    "three primitives against rasterize_above: 1000 must stay \
                     primitive-retained"
                );
            })
            .unwrap();

        clean_frame(cx, window.into());

        window
            .update(cx, |_, this, _| {
                let layer_surfaces = this
                    .rendered_frame
                    .scene
                    .surfaces
                    .iter()
                    .filter(|surface| {
                        matches!(surface.content, crate::scene::SurfaceContent::Layer(_))
                    })
                    .count();
                assert_eq!(
                    layer_surfaces, 0,
                    "a primitive-retained layer never composites through a surface"
                );
            })
            .unwrap();
    }

    /// A virtualized list inside an overscroll-buffer layer: scrolling within
    /// half the margin shifts the composite without re-recording or laying
    /// out items; scrolling past it refills exactly once.
    struct BufferedListView {
        paints: std::rc::Rc<std::cell::Cell<usize>>,
        handle: crate::ScrollHandle,
        controller: crate::VirtualListScrollController,
    }

    impl crate::Render for BufferedListView {
        fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
            let paints = self.paints.clone();
            let heights: std::rc::Rc<Vec<crate::Pixels>> =
                std::rc::Rc::new((0..100).map(|_| px(28.)).collect());
            crate::div().size_full().child(
                crate::div()
                    .id("buffered-list")
                    .layer_keyed("buffered-list-content")
                    .layer_with_policy(crate::LayerPolicy {
                        overdraw_margin: crate::size(px(0.), px(50.)),
                        ..Default::default()
                    })
                    .size_full()
                    .child(crate::vlist(
                        cx.entity(),
                        "buffered-vlist",
                        heights,
                        self.handle.clone(),
                        self.controller.clone(),
                        move |this: &mut Self, range: std::ops::Range<usize>, _, _| {
                            this.paints.set(this.paints.get() + range.len());
                            range
                                .map(|ix| {
                                    crate::div()
                                        .w_full()
                                        .h(px(28.))
                                        .bg(crate::blue())
                                        .child(format!("item {ix}"))
                                })
                                .collect::<Vec<_>>()
                        },
                    )),
            )
        }
    }

    #[gpui::test]
    fn an_overscroll_buffer_shifts_without_rerecording_then_refills(
        cx: &mut TestAppContext,
    ) {
        if layers_off() || rasterization_off() {
            return;
        }
        let paints = std::rc::Rc::new(std::cell::Cell::new(0));
        let handle = crate::ScrollHandle::new();
        let controller = crate::VirtualListScrollController::new();
        // Deterministic scroll positions: no smooth-scroll lag.
        controller
            .state
            .borrow_mut()
            .smooth_scroll
            .set_mode(crate::SmoothScrollMode::Disabled);
        let view = BufferedListView {
            paints: paints.clone(),
            handle: handle.clone(),
            controller: controller.clone(),
        };
        let window = cx.open_window(size(px(400.), px(300.)), move |_, _| view);
        cx.run_until_parked();

        // Settle the buffer: frame 1 records (viewport range), the next frame
        // notices the un-anchored buffer and requests a refill, and the refill
        // frame re-renders covering viewport + margin.
        clean_frame(cx, window.into());
        clean_frame(cx, window.into());
        clean_frame(cx, window.into());

        let key = window
            .update(cx, |_, this, _| {
                let (key, layer) = this.layers.iter().next().expect("the layer exists");
                assert!(layer.texture_retained, "a buffered layer rasterizes");
                assert!(layer.buffer_anchored, "the refill anchored the buffer");
                assert_eq!(
                    layer.texture_bounds.size.height,
                    px(300.) + px(50.) + px(50.),
                    "the texture covers viewport + 2 × margin"
                );
                *key
            })
            .unwrap();

        let paints_before_scroll = paints.get();
        let _ordering = TRANSFORM_STATS_ORDERING.lock();
        let refill_counter = "scroll: buffer refills";
        crate::render_stats::set_force_enabled(true);
        let before = crate::render_stats::snapshot();
        let before_refills = before.counters.get(refill_counter).copied().unwrap_or(0);

        // A small scroll: within half the margin (25px), so the frame shifts
        // the composite instead of re-recording, and no items are laid out.
        window
            .update(cx, |_, this, _| {
                handle.set_offset(crate::Point::new(px(0.), px(-20.)));
                this.refresh_buffers();
            })
            .unwrap();
        cx.run_until_parked();

        let after_shift = crate::render_stats::snapshot();
        assert_eq!(
            paints.get(),
            paints_before_scroll,
            "a buffered scroll frame must not lay out items"
        );

        window
            .update(cx, |_, this, _| {
                let layer = &this.layers[&key];
                assert_eq!(
                    layer.content_offset,
                    crate::Point::new(px(0.), px(-20.)),
                    "the shift is recorded as the content offset"
                );
                assert_eq!(
                    layer.transform.offset,
                    crate::Point::new(px(0.), px(0.) + px(-20.)),
                    "the layer transform shifts with the content for hit testing"
                );
            })
            .unwrap();

        // Scrolling past half the margin requests a refill; the next frame
        // re-renders the buffer and re-anchors at the new position. (The
        // harness needs an explicit frame for it: in production the wheel
        // listener's notify supplies the next frame.)
        window
            .update(cx, |_, this, _| {
                handle.set_offset(crate::Point::new(px(0.), px(-60.)));
                this.refresh_buffers();
            })
            .unwrap();
        cx.run_until_parked();
        clean_frame(cx, window.into());

        let after = crate::render_stats::snapshot();
        crate::render_stats::set_force_enabled(false);

        assert!(
            paints.get() > paints_before_scroll,
            "the refill frame lays out the buffer range"
        );
        assert_eq!(
            after
                .counters
                .get(refill_counter)
                .copied()
                .unwrap_or(0)
                - before_refills,
            1,
            "exactly one refill for one margin-crossing scroll"
        );
        window
            .update(cx, |_, this, _| {
                let layer = &this.layers[&key];
                assert_eq!(
                    layer.content_offset,
                    crate::Point::default(),
                    "the refill re-centres the buffer: no residual shift"
                );
                assert_eq!(layer.buffer_anchor.y, px(-60.));
            })
            .unwrap();
    }

    /// Overdraw regions are exempt from instance-tier occlusion culling
    /// (#96): content in the margin band exists so a later scroll can reveal
    /// it, so an occluder covering it must not suppress emission.
    struct OverdrawOccludedView {
        paints: std::rc::Rc<std::cell::Cell<usize>>,
    }

    impl crate::Render for OverdrawOccludedView {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            let paints = self.paints.clone();
            crate::div().size_full().child(
                crate::div()
                    .id("overdraw-occluded")
                    .layer_with_policy(crate::LayerPolicy {
                        overdraw_margin: crate::size(px(0.), px(50.)),
                        rasterize_above: 2,
                        ..Default::default()
                    })
                    .absolute()
                    .left(px(100.))
                    .top(px(100.))
                    .w(px(200.))
                    .h(px(200.))
                    // The cullee: sits in the margin band above the layer.
                    .child(crate::canvas(
                        |_, _, _| (),
                        move |bounds, _, window, _| {
                            paints.set(paints.get() + 1);
                            window.paint_quad(crate::fill(bounds, crate::blue()));
                        },
                    )
                    .absolute()
                    .left(px(0.))
                    .top(px(-30.))
                    .w(px(40.))
                    .h(px(40.)))
                    // An opaque sibling covering the margin item's window
                    // position exactly.
                    .child(
                        crate::div()
                            .absolute()
                            .left(px(0.))
                            .top(px(-30.))
                            .size(px(60.))
                            .bg(crate::red()),
                    ),
            )
        }
    }

    #[gpui::test]
    fn overdraw_content_survives_instance_culling(cx: &mut TestAppContext) {
        if layers_off() || rasterization_off() || occlusion_off() {
            return;
        }
        let paints = std::rc::Rc::new(std::cell::Cell::new(0));
        let paints_for_view = paints.clone();
        let window = cx.open_window(
            size(px(800.), px(600.)),
            move |_, _| OverdrawOccludedView {
                paints: paints_for_view,
            },
        );
        cx.run_until_parked();

        window
            .update(cx, |_, this, _| {
                let layer = this.layers.values().next().expect("the layer exists");
                let blue_quads = layer
                    .items
                    .iter()
                    .filter(|item| {
                        matches!(
                            item,
                            crate::layer::LayerItem::Primitive(
                                crate::scene::Primitive::Quad(quad)
                            ) if quad.background.solid == crate::blue()
                        )
                    })
                    .count();
                assert_eq!(
                    blue_quads, 1,
                    "the margin-band item must survive culling even under an \
                     opaque sibling: a scroll can reveal it"
                );
                assert!(
                    layer.texture_retained,
                    "the buffered layer rasterizes regardless of the threshold"
                );
            })
            .unwrap();
    }
}
