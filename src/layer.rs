//! Retained layers: explicit, named, independently invalidated units of caching.
//!
//! A layer is the first piece of genuinely *retained* state in the renderer. It
//! is modelled on `CALayer`: created explicitly, addressed by a stable key,
//! independently invalidated, and independently composited.
//!
//! ```ignore
//! div().id("properties-panel").layer().child(expensive_content)
//! ```
//!
//! [`Element::layer`](crate::InteractiveElement::layer) requires an
//! [`ElementId`](crate::ElementId), and that requirement is the point. A
//! layer's entire value is surviving across frames, so it must have a name that
//! survives across frames too. The cache this replaces was implicit and
//! anonymous, addressed by raw offsets into per-frame arrays — which is why a
//! range that aged by one frame could slice out of bounds, and why
//! `Window::invalid_reuse_range` exists as a hand-maintained fifteen-field
//! guard. A layer is addressed by [`LayerKey`], so there is nothing to age.
//!
//! # What is retained in this phase
//!
//! Primitives, at layer granularity: a composited layer re-emits its recorded
//! primitives with their recorded *layer-local* draw orders, which is what
//! lets it skip both primitive emission and the per-primitive `BoundsTree`
//! insert that dominates `Scene::insert_primitive`.
//!
//! As of #92, a layer that *is* re-rendering can additionally reconcile at
//! element granularity — see [`Layer::instances`] and [`crate::instance`] —
//! so one changed child no longer forces every sibling through `prepaint`/
//! `paint` again. Textures (#96) still live inside a layer later; that one
//! doesn't exist yet.
//!
//! Hitboxes, dispatch nodes, tooltips and shaped text still travel through the
//! old index-range replay path, which stays until #97 — see
//! [`Layer::paint_range`].
//!
//! # Ordering
//!
//! Each layer paints into its own ordering scope with its own `BoundsTree`
//! starting at zero (see [`crate::scene::Scene`]). Inter-layer ordering is
//! decided by where the layer was entered in its parent's scope, and the
//! composite step at `Scene::finish` maps every scope's local orders into a
//! global order range reserved for it.
//!
//! **Order invalidation is per-layer, not per-primitive.** If any content in a
//! layer changes bounds, that layer's tree re-inserts and its primitives
//! re-sort; content in other layers is untouched. This implies a sizing rule
//! that is easy to get wrong: **layer boundaries should separate content by
//! update frequency, not only by visual grouping.** A layer holding one
//! 120Hz-animating element and a thousand static ones re-sorts all thousand
//! every frame, and will look fine while being slower than no layer at all.
//! `WGPUI_LAYER_DEBUG=1` is how that gets diagnosed.

use crate::{
    Bounds, ContentMask, EntityId, GlobalElementId, InstanceKey, Invalidation, Pixels, Point,
    Primitive, px, size,
};
use crate::instance::ElementInstance;
use collections::{FxHashMap, FxHashSet};
use std::hash::{Hash, Hasher};

/// Identifies one retained layer for as long as the window keeps it.
///
/// Derived by hashing a [`GlobalElementId`], which is the whole path of
/// [`ElementId`](crate::ElementId)s down to the layer's element — so the key is
/// stable across frames for as long as the element keeps its place in the tree,
/// and distinct for two elements that share a local id under different parents.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LayerKey(pub u64);

impl LayerKey {
    /// Derive the key for the layer rooted at `id`.
    pub fn from_global_element_id(id: &GlobalElementId) -> Self {
        let mut hasher = collections::FxHasher::default();
        id.hash(&mut hasher);
        // Reserve 0 so a defaulted key is never a live layer.
        LayerKey(hasher.finish() | 1)
    }
}

/// A dense, per-window index for a layer, assigned on first sight.
///
/// [`LayerKey`] is a hash and is what the layer is addressed by;
/// `LayerId` is small, ordered by first appearance, and exists so the debug
/// overlay can pick a stable tint per layer without hashing a `u64` into a
/// colour space every frame.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LayerId(pub u32);

/// How a layer sits relative to its parent.
///
/// Translation from a layer's local coordinate space into window coordinates.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LayerTransform {
    /// Offset from the parent's coordinate space.
    pub offset: Point<Pixels>,
}

impl Default for LayerTransform {
    fn default() -> Self {
        LayerTransform {
            offset: Point::default(),
        }
    }
}

impl LayerTransform {
    /// Whether this transform moves nothing.
    pub fn is_identity(&self) -> bool {
        self.offset.x == px(0.) && self.offset.y == px(0.)
    }

    /// Transform a point from the layer's local coordinate space into window
    /// coordinates.
    pub fn apply(&self, point: Point<Pixels>) -> Point<Pixels> {
        point + self.offset
    }

    /// Transform a point from window coordinates into the layer's local
    /// coordinate space.
    pub fn invert(&self, point: Point<Pixels>) -> Point<Pixels> {
        point - self.offset
    }
}

/// Tuning for one layer.
///
/// `rasterize_above` and `overdraw_margin` are defined here and read by nothing
/// yet: they are inputs to texture-backed layers (#96) and overscroll buffers.
/// Defining them now means those phases add behaviour to a field rather than
/// reshaping this struct and every construction of it.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LayerPolicy {
    /// Below this primitive count the layer stays primitive-retained: it keeps
    /// its recorded primitives and re-emits them, with no texture. A layer
    /// holding twelve quads is cheaper to re-emit than to composite through an
    /// offscreen target.
    ///
    /// Unused until #96.
    pub rasterize_above: usize,
    /// How far outside its bounds the layer renders, so that a later scroll can
    /// reveal already-rendered content without re-rendering.
    ///
    /// Unused until #96.
    pub overdraw_margin: crate::Size<Pixels>,
    /// How many consecutive frames the layer may go unvisited before its
    /// retained content is dropped.
    pub evict_after_frames: u32,
}

impl Default for LayerPolicy {
    fn default() -> Self {
        LayerPolicy {
            rasterize_above: 256,
            overdraw_margin: size(px(0.), px(0.)),
            evict_after_frames: 60,
        }
    }
}

impl LayerPolicy {
    /// The policy [`AnyView::cached`](crate::AnyView::cached) gets.
    ///
    /// Compat means every axis is invalidated together: a cached view either
    /// replays all of its recorded output or rebuilds all of it, which is the
    /// behaviour those call sites already depend on. It is deliberately not the
    /// default, so that a policy gaining per-axis behaviour later does not
    /// silently change what `cached` does.
    pub fn compat() -> Self {
        LayerPolicy::default()
    }
}

/// One item of a layer's retained content, in paint order.
///
/// Nested layers are recorded as a reference rather than inlined, so a
/// composite of the outer layer reproduces the inner layer's *own* ordering
/// scope. Inlining the inner layer's primitives would mix two local order
/// spaces into one and silently reorder them.
///
/// `Clone` since #92: an `ElementInstance`'s own retained items are a cloned
/// sub-slice of what a `.layer()` subtree captured, not a range into the
/// layer's own `items` — see `ElementInstance::items`'s doc comment for why.
#[derive(Clone)]
pub(crate) enum LayerItem {
    /// A primitive carrying its layer-local draw order.
    Primitive(Primitive),
    /// A layer painted inside this one.
    Nested(LayerKey),
}

/// What a layer needs to be worth compositing rather than re-rendering.
///
/// Compared as a whole: any difference re-renders. These are the inputs to
/// paint that are not entities, so no invalidation request names them.
#[derive(Clone, Default, PartialEq)]
pub(crate) struct LayerCacheKey {
    pub bounds: Bounds<Pixels>,
    pub content_mask: ContentMask<Pixels>,
    pub opacity: f32,
    pub scale_factor: f32,
}

/// An explicit, retained, independently invalidated unit of caching.
///
/// Lives in `Window::layers`, keyed by [`LayerKey`] — deliberately **not** in
/// the frame arrays. That is the structural fix for the stale-range problem: a
/// layer's cached state is addressed by a stable name, not by an offset into an
/// array that no longer exists.
///
/// The key is not stored on the record: the map key is the single source of
/// truth for a layer's identity, and a copy of it inside the value is one more
/// thing that can disagree with it.
///
/// Nor are hitboxes, which the design sketch for this phase listed. They would
/// be write-only here — hit geometry still travels through the index-range
/// replay path, and the absolute bounds recorded today are not what #90 wants
/// anyway, since it makes hitboxes layer-relative and hit-tests by transforming
/// the query point. Recording them now would cost a clone on every re-render to
/// hold data that has to be thrown away.
pub(crate) struct Layer {
    /// Dense index, for the debug overlay.
    pub id: LayerId,
    /// Retained primitives and nested-layer references, in paint order,
    /// carrying layer-local draw orders.
    ///
    /// Empty means the layer has been evicted of content but is still known —
    /// a scrolled-away-and-back panel re-materialises into the same record.
    pub items: Vec<LayerItem>,
    /// Everything the layer's paint registered *besides* primitives — mouse
    /// listeners, cursor styles, input handlers, tab stops, shaped text, and
    /// accessed element state — as an index range into the previous frame.
    ///
    /// A composited layer does not run its subtree's paint, and all of that is
    /// registered during paint. Dropping it would leave a composited panel
    /// looking correct and silently unclickable, with its element state
    /// (scroll offsets, focus) garbage-collected out from under it. Primitives
    /// come from [`Self::items`]; the rest still travels by range until #97
    /// retires that path.
    pub paint_range: std::ops::Range<crate::PaintIndex>,
    /// Entities read while the layer last rendered. An invalidation naming any
    /// of these re-renders it.
    pub accessed_entities: FxHashSet<EntityId>,
    /// What the caller declared the content to be a function of, if anything.
    ///
    /// `None` is a plain `.layer()`, which re-renders whenever its view is
    /// notified. `Some` is `.layer_keyed(..)`, which composites across a notify
    /// while the key holds — the caller's claim, not the framework's inference.
    /// Compared as part of the composite decision, so switching a layer from
    /// keyed to unkeyed (or changing what is hashed) re-renders rather than
    /// reusing content recorded under different rules.
    pub content_key: Option<u64>,
    /// The non-entity paint inputs, compared as a whole.
    pub cache_key: LayerCacheKey,
    /// Where the layer sits. Must be the identity to composite; see
    /// [`LayerTransform`].
    pub transform: LayerTransform,
    /// Axes invalidated since the layer last rendered.
    pub needs: Invalidation,
    pub policy: LayerPolicy,
    /// The frame counter value when the layer was last visited by a draw.
    pub last_visited: u64,
    /// Whether the mouse was inside the layer when it last rendered.
    ///
    /// Hover state is read during paint and is not an entity, so nothing
    /// invalidates a layer when the pointer crosses it. Rather than guess which
    /// styles are hover-sensitive, a layer under the pointer — or one that was
    /// under it last frame — simply re-renders.
    pub had_mouse: bool,
    /// A conservative opaque coverage region in window coordinates.
    pub opaque_bounds: Option<Bounds<Pixels>>,
    /// Set when invalidated while visually occluded; content is rebuilt when revealed.
    pub deferred_dirty: bool,
    /// Bounds of backdrop filters and filter groups above this layer that
    /// read pixels behind them. Any layer whose content falls within these
    /// bounds must not be occluded, or the filter would sample stale pixels.
    pub poisoned_bounds: Vec<Bounds<Pixels>>,
    /// Retained per-element state for this layer's subtree, keyed by
    /// [`crate::InstanceKey`] (#92).
    ///
    /// Lives here rather than in a window-global map so instance memory is
    /// bounded by the same mark-and-sweep eviction that already bounds
    /// `items`: instances are owned by their layer and die with it. Cleared
    /// alongside `items` in `drop_content`, and — unlike `items`, which is
    /// unconditionally overwritten every time this layer re-renders —
    /// individual entries here are only overwritten for the children that
    /// actually rebuilt; a reconciled child's entry is left untouched.
    pub instances: FxHashMap<InstanceKey, ElementInstance>,
    /// Whether the renderer holds (or, within this same frame's `draw()`,
    /// is about to build) a persistent texture backing this layer's current
    /// content (Phase 11, `docs/retained-layers.md` §3.3).
    ///
    /// Set by `Window::record_layer` whenever a fresh re-render crosses
    /// [`LayerPolicy::rasterize_above`] with `WGPUI_LAYERS_RASTERIZE=1` *and*
    /// every primitive in it is one rasterization supports (see
    /// `Window::flatten_for_rasterize`); cleared by [`Self::drop_content`] on
    /// eviction. `Window::composite_layer` reads this to decide whether a
    /// clean composite can stand in with a single
    /// `SurfaceContent::Layer` primitive instead of replaying `items`.
    pub(crate) rasterized: bool,
    /// Rolling history of recent re-render timestamps, most recent last,
    /// bounded to [`RERENDER_HISTORY_LEN`] entries. Feeds
    /// [`Self::rerender_rate_hz`], which powers the Inspector's optional
    /// per-layer FPS labels (`hud::is_layer_fps_enabled`) — display only,
    /// never consulted for anything that affects what gets drawn.
    pub(crate) rerender_times: std::collections::VecDeque<crate::time_ext::Instant>,
}

/// Cap on [`Layer::rerender_times`]. Large enough that a layer re-rendering
/// once or twice a second still produces a stable rate (the oldest and
/// newest samples span several seconds), small enough that a 120Hz-animating
/// layer's history is trivial to keep and doesn't need trimming by age, only
/// by count.
const RERENDER_HISTORY_LEN: usize = 20;

impl Layer {
    pub fn new(id: LayerId, policy: LayerPolicy, frame: u64) -> Self {
        Layer {
            id,
            items: Vec::new(),
            paint_range: crate::PaintIndex::default()..crate::PaintIndex::default(),
            accessed_entities: FxHashSet::default(),
            content_key: None,
            cache_key: LayerCacheKey::default(),
            transform: LayerTransform::default(),
            // A layer that has never rendered needs everything.
            needs: Invalidation::all(),
            policy,
            last_visited: frame,
            had_mouse: false,
            opaque_bounds: None,
            deferred_dirty: false,
            poisoned_bounds: Vec::new(),
            instances: FxHashMap::default(),
            rasterized: false,
            rerender_times: std::collections::VecDeque::with_capacity(RERENDER_HISTORY_LEN),
        }
    }

    /// Whether the layer holds content it could composite from.
    pub fn has_content(&self) -> bool {
        !self.items.is_empty()
    }

    /// Whether this layer's current `items` are worth handing the renderer a
    /// [`crate::scene::LayerRasterizeJob`] for, per
    /// [`LayerPolicy::rasterize_above`] (or the Utilities tab's override —
    /// see [`effective_rasterize_above`]). `Window::composite_layer` reads
    /// [`Self::rasterized`], set from this, to decide whether a clean
    /// composite can stand in with a single `SurfaceContent::Layer`
    /// primitive.
    pub(crate) fn should_rasterize(&self) -> bool {
        layers_rasterize_enabled()
            && Self::crosses_rasterize_threshold(
                self.items.len(),
                effective_rasterize_above(&self.policy),
            )
    }

    /// Record that this layer just re-rendered, for
    /// [`Self::rerender_rate_hz`]. Called only from the dirty (re-render)
    /// path in `Window::record_layer` — a clean composite is, by
    /// definition, not a new data point.
    pub(crate) fn record_rerender(&mut self, now: crate::time_ext::Instant) {
        self.rerender_times.push_back(now);
        while self.rerender_times.len() > RERENDER_HISTORY_LEN {
            self.rerender_times.pop_front();
        }
    }

    /// This layer's re-render rate in Hz, estimated from the span between
    /// its oldest and newest recorded re-renders — `None` with fewer than
    /// two samples (a span needs two points), or when the newest sample is
    /// more than five seconds old (the layer has gone quiet; a rate
    /// computed from a stale window would understate how idle it actually
    /// is). Display-only: see [`Self::rerender_times`].
    pub(crate) fn rerender_rate_hz(&self, now: crate::time_ext::Instant) -> Option<f32> {
        let newest = *self.rerender_times.back()?;
        if now.saturating_duration_since(newest) > std::time::Duration::from_secs(5) {
            return None;
        }
        let oldest = *self.rerender_times.front()?;
        let span = newest.saturating_duration_since(oldest).as_secs_f32();
        if span <= 0.0 {
            return None;
        }
        Some((self.rerender_times.len() - 1) as f32 / span)
    }

    /// Pure comparison behind [`Self::should_rasterize`], split out so the
    /// threshold logic is unit-testable independent of the process-global
    /// `WGPUI_LAYERS_RASTERIZE` env read (which, like `layers_enabled`'s own
    /// `LazyLock`, is read once per process and so cannot be toggled
    /// per-test). `pub(crate)` because `Window::record_layer` checks the
    /// threshold against the freshly captured item list before it becomes
    /// `self.items` (see that call site for why).
    pub(crate) fn crosses_rasterize_threshold(item_count: usize, rasterize_above: usize) -> bool {
        item_count > rasterize_above
    }

    /// Drop retained content, keeping the record so the layer can
    /// re-materialise into the same key and id.
    pub fn drop_content(&mut self) {
        self.items.clear();
        self.items.shrink_to_fit();
        self.paint_range = crate::PaintIndex::default()..crate::PaintIndex::default();
        self.needs = Invalidation::all();
        self.deferred_dirty = false;
        // #92: an evicted layer's ElementInstances describe content that no
        // longer exists. Left in place they would be dead weight at best; at
        // worst a later InstanceKey collision (extremely unlikely, but the
        // failure mode matters) could reuse one against unrelated content.
        self.instances.clear();
        // Whatever texture the renderer built for the old content is about
        // to be freed by the eviction report `Window::evict_stale_layers`
        // sends alongside this call; a re-materialised layer starts unknown
        // again rather than assuming stale backing survived.
        self.rasterized = false;
        // A re-materialised layer's re-render rate starts from nothing
        // rather than carrying a stale span across the gap.
        self.rerender_times.clear();
    }
}

/// Whether `.layer()` and the layer-backed path for `AnyView::cached` are live.
///
/// `WGPUI_LAYERS=0` makes `.layer()` a no-op passthrough and sends `cached`
/// back through the index-range replay it has always used. Following the
/// `WGPUI_NESTED_VIEW_CACHE` precedent: this is the phase that changes the
/// public API surface, so the old path stays reachable without a rebuild until
/// #97 deletes it.
///
/// Read once, at first use.
pub(crate) fn layers_enabled() -> bool {
    static ENABLED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var("WGPUI_LAYERS")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(true)
    });
    *ENABLED
}

/// Runtime switch for the layer diagnostic overlay, defaulting off.
///
/// Was a `LazyLock` reading `WGPUI_LAYER_DEBUG` once at process start, then
/// briefly a hardcoded `true` — both wrong for the same reason: a debug
/// visualizer belongs behind a switch a user can flip while the app is
/// running, not an env var fixed for the process's whole lifetime. The
/// Inspector's Utilities tab is that switch; see [`set_layer_debug_enabled`].
static LAYER_DEBUG_ENABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Whether to draw the layer diagnostic overlay.
///
/// Tints every composite by its [`LayerId`] and flashes the tint on a frame
/// the layer re-rendered. The failure this exists to catch is a layer that is
/// silently re-rendering every frame: without it that layer is merely slow,
/// which is indistinguishable from the layer being large.
pub(crate) fn layer_debug_enabled() -> bool {
    LAYER_DEBUG_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Turn the layer diagnostic overlay on or off at runtime.
///
/// Public so a host UI — the Inspector's Utilities tab, in this codebase —
/// can wire it to a switch. Takes effect on the next frame; nothing needs
/// re-rendering to pick it up, since every composite/re-render checks
/// [`layer_debug_enabled`] fresh.
pub fn set_layer_debug_enabled(enabled: bool) {
    LAYER_DEBUG_ENABLED.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

/// Current state of the layer diagnostic overlay switch.
pub fn is_layer_debug_enabled() -> bool {
    layer_debug_enabled()
}

/// Whether a layer that crosses [`LayerPolicy::rasterize_above`] queues a
/// [`crate::scene::LayerRasterizeJob`] for the renderer to build into a
/// persistent texture (Phase 11, `docs/retained-layers.md` §3.3) — see
/// `Window::flatten_for_rasterize` and `WgpuRenderer::process_layer_rasterize_requests`
/// for the two ends of that pipeline.
///
/// Unconditionally on — was gated behind `WGPUI_LAYERS_RASTERIZE` while the
/// renderer side didn't exist yet (turning it on would only have grown
/// `Scene::rasterize_requests` for no effect); both ends are wired now, and
/// the flag was removed at the project's request.
pub(crate) fn layers_rasterize_enabled() -> bool {
    true
}

// ---- Runtime tunables (Inspector → Utilities → Layers) --------------------
//
// Every call site still builds its own `LayerPolicy` with its own opinion of
// `rasterize_above`/`evict_after_frames` — these don't change that. They sit
// *in front of* it: `effective_rasterize_above`/`effective_evict_after_frames`
// are what `Layer::should_rasterize`, `Window::record_layer` and
// `Window::evict_stale_layers` actually call, and an override there wins over
// every policy in the window at once. `0` is the sentinel for "no override"
// in both `AtomicUsize`/`AtomicU32` stores below, since a real threshold of
// `0` is already expressible as [`is_force_rasterize_all`] and a real evict
// window of `0` frames would evict content the instant it stopped being
// visited, which nothing wants.

static RASTERIZE_ABOVE_OVERRIDE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
static FORCE_RASTERIZE_ALL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static EVICT_AFTER_FRAMES_OVERRIDE: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

/// Override [`LayerPolicy::rasterize_above`] for every layer in the process,
/// regardless of what policy its own call site built. `None` clears the
/// override, returning each layer to its own policy's value.
pub fn set_rasterize_above_override(threshold: Option<usize>) {
    RASTERIZE_ABOVE_OVERRIDE.store(threshold.unwrap_or(0), std::sync::atomic::Ordering::Relaxed);
}

/// Current `rasterize_above` override, if any is set.
pub fn rasterize_above_override() -> Option<usize> {
    match RASTERIZE_ABOVE_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        0 => None,
        n => Some(n),
    }
}

/// Force every non-empty layer to rasterize, bypassing
/// `rasterize_above`/its override entirely. The blunt instrument next to
/// [`set_rasterize_above_override`]'s scalpel — useful for finding a layer
/// whose texture path has a bug that only shows up once it exists, without
/// hunting for how big it needs to get first.
pub fn set_force_rasterize_all(enabled: bool) {
    FORCE_RASTERIZE_ALL.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

/// Whether [`set_force_rasterize_all`] is currently on.
pub fn is_force_rasterize_all() -> bool {
    FORCE_RASTERIZE_ALL.load(std::sync::atomic::Ordering::Relaxed)
}

/// The `rasterize_above` threshold actually in effect for `policy`, folding
/// in both overrides above: "force all" wins outright (an empty layer still
/// never rasterizes — there is nothing to put in a texture — so this reads
/// as "threshold zero", not "always"), otherwise an explicit numeric
/// override replaces the call site's own value.
pub(crate) fn effective_rasterize_above(policy: &LayerPolicy) -> usize {
    if is_force_rasterize_all() {
        return 0;
    }
    rasterize_above_override().unwrap_or(policy.rasterize_above)
}

/// Override [`LayerPolicy::evict_after_frames`] for every layer in the
/// process. `None` clears the override.
pub fn set_evict_after_frames_override(frames: Option<u32>) {
    EVICT_AFTER_FRAMES_OVERRIDE.store(frames.unwrap_or(0), std::sync::atomic::Ordering::Relaxed);
}

/// Current `evict_after_frames` override, if any is set.
pub fn evict_after_frames_override() -> Option<u32> {
    match EVICT_AFTER_FRAMES_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        0 => None,
        n => Some(n),
    }
}

/// The tint for `layer_id`, and whether it should be drawn at full strength
/// because the layer re-rendered this frame.
pub(crate) fn debug_tint(id: LayerId, re_rendered: bool) -> crate::Hsla {
    // Golden-ratio hue stepping keeps adjacent ids visually distinct.
    let hue = (id.0 as f32 * 0.618_034) % 1.0;
    crate::Hsla {
        h: hue,
        s: 0.9,
        l: 0.55,
        a: if re_rendered { 0.35 } else { 0.12 },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ElementId;
    use std::sync::Arc;

    fn global_id(path: &[&'static str]) -> GlobalElementId {
        GlobalElementId(Arc::from(
            path.iter()
                .map(|name| ElementId::Name((*name).into()))
                .collect::<Vec<_>>(),
        ))
    }

    #[test]
    fn layer_key_is_stable_and_path_sensitive() {
        let a = LayerKey::from_global_element_id(&global_id(&["root", "panel"]));
        let b = LayerKey::from_global_element_id(&global_id(&["root", "panel"]));
        assert_eq!(a, b, "the same path must produce the same key every frame");

        let under_other_parent = LayerKey::from_global_element_id(&global_id(&["other", "panel"]));
        assert_ne!(
            a, under_other_parent,
            "the same local id under a different parent is a different layer"
        );
    }

    #[test]
    fn layer_key_is_never_zero() {
        // `LayerKey(0)` is reserved so that a defaulted or sentinel key can
        // never collide with a live layer.
        for path in [
            vec!["a"],
            vec!["a", "b"],
            vec!["panel", "list", "row"],
            vec![],
        ] {
            assert_ne!(LayerKey::from_global_element_id(&global_id(&path)).0, 0);
        }
    }

    #[test]
    fn dropping_content_keeps_identity_and_forces_a_rebuild() {
        let mut layer = Layer::new(LayerId(3), LayerPolicy::default(), 0);
        layer.items.push(LayerItem::Nested(LayerKey(9)));
        layer.needs = Invalidation::empty();

        layer.drop_content();

        assert_eq!(layer.id, LayerId(3));
        assert!(!layer.has_content());
        assert_eq!(
            layer.needs,
            Invalidation::all(),
            "an evicted layer must not be judged clean when it is next visited"
        );
    }

    #[test]
    fn layer_transform_round_trips_points() {
        let transform = LayerTransform {
            offset: Point::new(px(17.), px(-23.)),
        };
        let local = Point::new(px(4.), px(9.));

        assert_eq!(transform.invert(transform.apply(local)), local);
    }

    #[test]
    fn rasterize_threshold_is_strictly_greater_than() {
        // Default policy: 256. At the threshold a layer stays
        // primitive-retained; one item past it, it qualifies.
        assert!(!Layer::crosses_rasterize_threshold(256, 256));
        assert!(Layer::crosses_rasterize_threshold(257, 256));
        assert!(!Layer::crosses_rasterize_threshold(0, 0));
        assert!(Layer::crosses_rasterize_threshold(1, 0));
    }

    #[test]
    fn dropping_content_clears_the_rasterized_flag() {
        let mut layer = Layer::new(LayerId(1), LayerPolicy::default(), 0);
        layer.rasterized = true;

        layer.drop_content();

        assert!(
            !layer.rasterized,
            "a re-materialised layer must not claim a texture the eviction just freed"
        );
    }

    #[test]
    fn dropping_content_clears_rerender_history() {
        let mut layer = Layer::new(LayerId(1), LayerPolicy::default(), 0);
        let now = crate::time_ext::Instant::now();
        layer.record_rerender(now);
        layer.record_rerender(now + std::time::Duration::from_millis(16));

        layer.drop_content();

        assert!(
            layer.rerender_times.is_empty(),
            "a re-materialised layer must not report a rate spanning the gap before it existed again"
        );
    }

    #[test]
    fn rerender_history_is_bounded() {
        let mut layer = Layer::new(LayerId(1), LayerPolicy::default(), 0);
        let start = crate::time_ext::Instant::now();
        for i in 0..(RERENDER_HISTORY_LEN * 3) {
            layer.record_rerender(start + std::time::Duration::from_millis(i as u64 * 16));
        }
        assert_eq!(layer.rerender_times.len(), RERENDER_HISTORY_LEN);
    }

    #[test]
    fn rerender_rate_needs_at_least_two_samples() {
        let mut layer = Layer::new(LayerId(1), LayerPolicy::default(), 0);
        let now = crate::time_ext::Instant::now();
        assert_eq!(layer.rerender_rate_hz(now), None, "no samples yet");

        layer.record_rerender(now);
        assert_eq!(
            layer.rerender_rate_hz(now),
            None,
            "one sample has no span to compute a rate from"
        );
    }

    #[test]
    fn rerender_rate_matches_a_steady_cadence() {
        let mut layer = Layer::new(LayerId(1), LayerPolicy::default(), 0);
        let start = crate::time_ext::Instant::now();
        // 60Hz: 16 samples spaced 1/60s apart span 15/60s, so the rate is
        // 15 intervals over that span — exactly 60Hz again, by construction.
        for i in 0..16u64 {
            layer.record_rerender(start + std::time::Duration::from_secs_f64(i as f64 / 60.0));
        }
        let now = start + std::time::Duration::from_secs_f64(15.0 / 60.0);
        let rate = layer.rerender_rate_hz(now).expect("has samples");
        assert!(
            (rate - 60.0).abs() < 0.01,
            "expected ~60Hz, got {rate}"
        );
    }

    #[test]
    fn rerender_rate_goes_stale_after_five_seconds_of_silence() {
        let mut layer = Layer::new(LayerId(1), LayerPolicy::default(), 0);
        let start = crate::time_ext::Instant::now();
        layer.record_rerender(start);
        layer.record_rerender(start + std::time::Duration::from_millis(16));

        let still_fresh = start + std::time::Duration::from_secs(4);
        assert!(layer.rerender_rate_hz(still_fresh).is_some());

        let gone_quiet = start + std::time::Duration::from_secs(6);
        assert_eq!(
            layer.rerender_rate_hz(gone_quiet),
            None,
            "a layer that hasn't re-rendered in 5s should read as idle, not a stale rate"
        );
    }
}
