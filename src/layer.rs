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
//! `paint` again. As of #96 a layer can additionally be *texture-retained*:
//! its content lives in a persistent renderer-side texture and a clean frame
//! composites it with a single surface draw — see [`Layer::texture_retained`].
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
    Primitive, px, point, size,
};
use crate::instance::ElementInstance;
use crate::scene_pack::{FallbackReason, RecordedSlabPack};
use collections::{FxHashMap, FxHashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

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
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LayerPolicy {
    /// Below this primitive count the layer stays primitive-retained: it keeps
    /// its recorded primitives and re-emits them, with no texture. A layer
    /// holding twelve quads is cheaper to re-emit than to composite through an
    /// offscreen target. A layer with a non-zero [`Self::overdraw_margin`]
    /// rasterizes regardless — the overscroll buffer is the point.
    pub rasterize_above: usize,
    /// How far outside its bounds the layer renders, so that a later scroll can
    /// reveal already-rendered content without re-rendering.
    ///
    /// Non-zero marks the layer as an *overscroll buffer* (#96): the texture
    /// covers `bounds + 2 × margin`, scrolling shifts the composite without
    /// re-recording, and the layer re-renders (refills) once accumulated
    /// scroll passes half the margin.
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

    /// Whether this policy turns the layer into an overscroll buffer (#96):
    /// non-zero margin means the texture extends past the viewport and
    /// scrolling shifts the composite instead of re-recording.
    pub fn buffers_scroll(&self) -> bool {
        self.overdraw_margin != crate::size(px(0.), px(0.))
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
    /// This layer's own content, packed into slab-ready form once at record
    /// time (`Some(Ok)`), or why packing fell back to the legacy composite
    /// path (`Some(Err)`). `None` when slabs are off or nothing was packable.
    ///
    /// Derived from `items` at the origin that record used, and cleared with
    /// them in [`Self::drop_content`]. The pack clones every primitive into
    /// per-kind arrays, so caching it keeps a second copy of the layer's
    /// geometry alive between records — the price of taking validate/gather/
    /// sort out of every composite. That copy is bounded by the same
    /// mark-and-sweep eviction that bounds `items` (it dies with the layer's
    /// content, and a re-record replaces it wholesale), exactly like
    /// `instances`; a layer is only worth compositing if it survives long
    /// enough to amortise one extra copy.
    pub packed: Option<Result<Arc<RecordedSlabPack>, FallbackReason>>,
    /// Whether this layer's content is baked into a persistent renderer-side
    /// texture (#96): clean frames composite it with a single
    /// `SurfaceContent::Layer` draw instead of re-emitting slab spans.
    ///
    /// Decided at record time — rasterization needs packable, nested-free
    /// content — and consumed by the composite paths, which emit the surface
    /// and skip the spans entirely. The texture itself lives in the renderer's
    /// layer-texture cache; when the renderer drops one (resize, eviction) it
    /// posts a re-record request and the next frame re-bakes it.
    pub texture_retained: bool,
    /// The buffer extent the texture was rendered at, in window coordinates:
    /// [`LayerCacheKey::bounds`] inflated by [`LayerPolicy::overdraw_margin`].
    ///
    /// The composite surface samples the texture across these bounds (shifted
    /// by [`Self::content_offset`]) while clipping to the layer's visible
    /// rect, so margin content exists offscreen but never paints outside the
    /// layer's own extent.
    pub texture_bounds: Bounds<Pixels>,
    /// How far the buffered content has scrolled since the texture was
    /// rendered: the difference between the current scroll position and
    /// [`Self::buffer_anchor`], maintained by the buffered element (a
    /// virtualized list) via `Window::set_layer_content_offset`.
    ///
    /// The composite shifts by this much instead of re-recording; the element
    /// requests a refill through `Window::request_layer_buffer_refill` once it
    /// passes half the margin, so the shift never outruns the texture.
    pub content_offset: Point<Pixels>,
    /// The scroll-space position the buffer was rendered at, set by the
    /// buffered element at refill time. Opaque to the framework: only the
    /// element that reads it back gives it meaning.
    pub buffer_anchor: Point<Pixels>,
    /// Whether [`Self::buffer_anchor`] reflects a record that actually painted
    /// the buffer range. False until the buffered element's first refill, so
    /// the element knows the texture still covers the viewport only.
    pub buffer_anchored: bool,
}

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
            packed: None,
            texture_retained: false,
            texture_bounds: Bounds::default(),
            content_offset: Point::default(),
            buffer_anchor: Point::default(),
            buffer_anchored: false,
        }
    }

    /// Whether the layer holds content it could composite from.
    pub fn has_content(&self) -> bool {
        !self.items.is_empty()
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
        // The pack describes exactly the dropped items; keeping it would pin
        // the cloned geometry with nothing left to splice it against.
        self.packed = None;
        // The texture (if any) outlives this record only as stale pixels; the
        // renderer's cache is invalidated through the re-record path.
        self.texture_retained = false;
        self.content_offset = Point::default();
        self.buffer_anchor = Point::default();
        self.buffer_anchored = false;
    }
}

/// Inflate `bounds` by `margin` on every side.
///
/// The buffer rect for an overscroll layer: `bounds + 2 × margin` per axis,
/// so content up to `margin` beyond the viewport exists in the texture and a
/// scroll of up to `margin` never outruns it.
pub(crate) fn inflate_bounds(bounds: Bounds<Pixels>, margin: crate::Size<Pixels>) -> Bounds<Pixels> {
    Bounds {
        origin: point(bounds.origin.x - margin.width, bounds.origin.y - margin.height),
        size: size(
            bounds.size.width + margin.width + margin.width,
            bounds.size.height + margin.height + margin.height,
        ),
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

/// Whether layers may be rasterized into persistent textures (#96).
///
/// `WGPUI_LAYERS_RASTERIZE=0` forces every layer to stay primitive-retained
/// (slab spans, no offscreen texture), which isolates GPU-side texture
/// compositing problems from CPU-side ones: the two modes must render
/// identically, only slower.
///
/// Read once, at first use.
pub(crate) fn rasterization_enabled() -> bool {
    static ENABLED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var("WGPUI_LAYERS_RASTERIZE")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(true)
    });
    *ENABLED
}

/// Whether to draw the layer diagnostic overlay.
///
/// `WGPUI_LAYER_DEBUG=1` tints every composite by its [`LayerId`] and flashes
/// the tint on a frame the layer re-rendered. The failure this exists to catch
/// is a layer that is silently re-rendering every frame: without it that layer
/// is merely slow, which is indistinguishable from the layer being large.
///
/// Read once, at first use.
pub(crate) fn layer_debug_enabled() -> bool {
    static ENABLED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var("WGPUI_LAYER_DEBUG")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(false)
    });
    *ENABLED
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
}
