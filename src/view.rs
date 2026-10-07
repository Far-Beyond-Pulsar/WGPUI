use crate::{
    AnyElement, AnyEntity, AnyWeakEntity, App, Bounds, ContentMask, Context, Element,
    ElementGeometry, ElementId, Entity, EntityId, GlobalElementId, InspectorElementId, IntoElement,
    LayerPolicy, LayoutId, PaintIndex, Pixels, PrepaintStateIndex, Render, Style, StyleRefinement,
    TextStyle, WeakEntity,
};
use crate::{Empty, Window, ElementArenaScope};
use anyhow::Result;
use collections::FxHashSet;
use refineable::Refineable;
use std::rc::Rc;
use std::{any::TypeId, fmt, ops::Range};

struct AnyViewState {
    prepaint_range: Range<PrepaintStateIndex>,
    paint_range: Range<PaintIndex>,
    cache_key: ViewCacheKey,
    accessed_entities: FxHashSet<EntityId>,
    /// Set while an off-screen auto-height view skipped its prepaint. Its
    /// recorded ranges then describe an older frame, so it must rebuild before
    /// it can ever be reused or replayed again.
    culled: bool,
}

/// The content height an [`AnyView::cached_auto_height`] view was last measured
/// at, kept in element state so it follows the view's place in the tree.
#[derive(Clone, Copy)]
struct RememberedViewHeight(Pixels);

/// Read the remembered height without disturbing it. Also marks the state as
/// used this frame, which is what keeps it from being swept.
fn remembered_view_height(id: &GlobalElementId, window: &mut Window) -> Option<Pixels> {
    let mut height = None;
    window.with_element_state::<Option<RememberedViewHeight>, _>(id, |state, _| {
        let state = state.flatten();
        height = state.map(|state| state.0);
        ((), state)
    });
    height
}

fn remember_view_height(id: &GlobalElementId, height: Pixels, window: &mut Window) {
    window.with_element_state::<Option<RememberedViewHeight>, _>(id, |_, _| {
        ((), Some(RememberedViewHeight(height)))
    });
}

/// Replace the dependency set a cached view stored when it last rebuilt. Used
/// after an isolated view re-renders in place, so that its stored state names
/// what it reads now and not what it read the last time the walk rebuilt it.
pub(crate) fn refresh_view_dependencies(
    id: &GlobalElementId,
    accessed: FxHashSet<EntityId>,
    window: &mut Window,
) {
    window.with_element_state::<AnyViewState, _>(id, |state, _| {
        let state = state.map(|mut state| {
            state.accessed_entities = accessed;
            state
        });
        // No stored state means the view was never walked as a cache; there is
        // nothing to refresh, and `with_element_state` needs a value back.
        match state {
            Some(state) => ((), state),
            None => ((), AnyViewState {
                prepaint_range: PrepaintStateIndex::default()..PrepaintStateIndex::default(),
                paint_range: PaintIndex::default()..PaintIndex::default(),
                cache_key: ViewCacheKey::default(),
                accessed_entities: FxHashSet::default(),
                culled: true,
            }),
        }
    });
}

#[derive(Default)]
struct ViewCacheKey {
    bounds: Bounds<Pixels>,
    content_mask: ContentMask<Pixels>,
    text_style: TextStyle,
}

impl<V: Render> Element for Entity<V> {
    type RequestLayoutState = AnyElement;
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        Some(ElementId::View(self.entity_id()))
    }

    fn source_location(&self) -> Option<&'static std::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        // Redundant inside `Window::draw`, whose scope is already visible to
        // every gpui copy, including plugin DLLs; kept so rendering a view
        // always targets this `App`'s arena. Nesting is free:
        // `ElementArenaScope` restores the previous arena on drop.
        let _arena_scope = ElementArenaScope::enter(cx.element_arena());
        let mut element = self.update(cx, |view, cx| view.render(window, cx).into_any_element());
        let layout_id = window.with_rendered_view(self.entity_id(), |window| {
            element.request_layout(window, cx)
        });
        (layout_id, element)
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _: Bounds<Pixels>,
        element: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) {
        window.set_view_id(self.entity_id());
        window.with_rendered_view(self.entity_id(), |window| element.prepaint(window, cx));
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _: Bounds<Pixels>,
        element: &mut Self::RequestLayoutState,
        _: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        window.with_rendered_view(self.entity_id(), |window| element.paint(window, cx));
    }
}

/// Whether a cached view rebuilding is allowed to leave the cached views nested
/// inside it reusing, rather than forcing the whole subtree to rebuild.
///
/// This is what makes "no state change = no op" actually hold for nested
/// panels. Upstream forces the whole nested subtree to rebuild whenever any
/// ancestor cached view rebuilds, which in the level editor meant one genuinely
/// dirty view produced five collateral cache misses — measured at roughly 600ms
/// of every 890ms spent on the UI thread. With this on, a reuse costs ~0.003ms
/// against ~0.64ms for the rebuild it replaces.
///
/// Set `WGPUI_NESTED_VIEW_CACHE=0` to fall back to upstream behaviour.
///
/// **Why the escape hatch exists.** Enabling this repeatedly aborted the process
/// in `LineLayoutCache::reuse_layouts` — a stored reuse range outliving the
/// array it indexes. The ranges in `PrepaintStateIndex`/`PaintIndex` are
/// absolute offsets into per-frame arrays with nothing tying them to the array
/// they were recorded against. `Window::invalid_reuse_range` now bounds-checks
/// every one of them before a single byte is copied and treats a bad range as a
/// cache miss, so the failure mode is a slower frame rather than a crash. That
/// guard is what makes this safe to leave on; if something still goes wrong, the
/// env var reverts the behaviour without a rebuild.
///
/// Read once, at first use.
fn nested_view_cache_enabled() -> bool {
    static ENABLED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var("WGPUI_NESTED_VIEW_CACHE")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(true)
    });
    *ENABLED
}

/// Report the first cached-view reuse whose stored range had outlived the array
/// it indexes, then stay quiet.
///
/// One line is enough to identify the cause — which array, how far past the end,
/// and which view — without a per-frame flood. Everything after the first is
/// silently handled as a cache miss.
fn log_stale_reuse_range(entity: EntityId, array: &'static str, end: usize, len: usize) {
    // Counted on every occurrence, logged only on the first: the count is what
    // tells you whether this is a one-off at startup or happening continuously,
    // so it must not sit behind the log-once guard.
    crate::render_stats::count("view cache: stale range (rebuilt)");

    static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    log::warn!(
        "[VIEW CACHE] stale reuse range discarded: entity={:?} array={} stored_end={} actual_len={}. \
         Rebuilding instead of replaying. This is the condition that used to panic in \
         `reuse_layouts`; further occurrences are handled silently.",
        entity,
        array,
        end,
        len,
    );
}

/// A dynamically-typed handle to a view, which can be downcast to a [Entity] for a specific type.
#[derive(Clone, Debug)]
pub struct AnyView {
    entity: AnyEntity,
    render: fn(&AnyView, &mut Window, &mut App) -> AnyElement,
    cached_style: Option<Rc<StyleRefinement>>,
    /// Whether the cached root size takes its height from the previous
    /// frame's measured content rather than from `cached_style`. See
    /// [`AnyView::cached_auto_height`].
    auto_height: bool,
    /// See [`AnyView::isolated`].
    isolated: bool,
    /// Set by `request_layout` each frame: this auto-height view had nothing
    /// remembered yet and was rendered inline, so `paint` must not take the
    /// cached path (there is no cached state for it to replay).
    rendered_inline: bool,
    /// The concrete view type, for naming profiler spans. Static, so it costs
    /// one pointer-pair per view handle and nothing per frame.
    type_name: &'static str,
}

impl<V: Render> From<Entity<V>> for AnyView {
    fn from(value: Entity<V>) -> Self {
        AnyView {
            entity: value.into_any(),
            render: any_view::render::<V>,
            cached_style: None,
            auto_height: false,
            isolated: false,
            rendered_inline: false,
            type_name: std::any::type_name::<V>(),
        }
    }
}

impl AnyView {
    /// Indicate that this view should be cached when using it as an element.
    /// When using this method, the view's previous layout and paint will be recycled from the previous frame if [Context::notify] has not been called since it was rendered.
    /// The one exception is when [Window::refresh] is called, in which case caching is ignored.
    pub fn cached(mut self, style: StyleRefinement) -> Self {
        self.cached_style = Some(style.into());
        self
    }

    /// [`Self::cached`] for a view whose height is decided by its content.
    ///
    /// A cached view is laid out as a childless leaf with `style` as its size,
    /// so it can only be cached when the caller can name that size up front.
    /// This variant instead measures the view and remembers the result: the
    /// height of the cached node is the content height measured the last time
    /// the view actually rebuilt, and `style` supplies everything else (usually
    /// just the width, e.g. `w_full()`).
    ///
    /// Behaviour, in order of frames:
    ///
    /// - The first frame there is nothing remembered, so the view is rendered
    ///   inline exactly like an uncached view and its resulting height is
    ///   recorded.
    /// - From then on it is cached. A notify on the view (or on anything it read)
    ///   rebuilds only this view; its siblings reuse their previous layout,
    ///   prepaint and paint.
    /// - If a rebuild measures a different height than the one it was given, the
    ///   new height is remembered and the view invalidates itself, so the
    ///   ancestors re-lay-out with the correct size on the next frame. Content
    ///   that changes height is therefore briefly one frame stale; content that
    ///   keeps its height never is.
    ///
    /// The remembered height lives in element state keyed by the view's position
    /// in the tree, so a view that moves to a different place in the tree starts
    /// over with an inline frame instead of ever being given a wrong size.
    ///
    /// Intended for rows, cards and sections of inspector-like panels, where one
    /// hover, caret blink or edit would otherwise rebuild a whole panel.
    pub fn cached_auto_height(mut self, style: StyleRefinement) -> Self {
        self.cached_style = Some(style.into());
        self.auto_height = true;
        self
    }

    /// Make this cached view an *isolation boundary*: it refreshes by itself.
    ///
    /// A notify on an ordinary cached view dirties every view above it and
    /// invalidates every cached ancestor whose dependency set (cumulative over
    /// its subtree) contains the notified entity. A view that changes ten times
    /// a second therefore rebuilds the panel, the dock and everything else
    /// above it ten times a second.
    ///
    /// An isolated view is the exception. Its dependencies are its own: they are
    /// not folded into its ancestors' sets, and an invalidation stops at it.
    /// When it is invalidated, `Window::draw` re-renders just this view, in
    /// place, into its own retained layer, before the element walk; the
    /// ancestors then replay and composite the new content. Nothing above it
    /// renders, lays out or prepaints.
    ///
    /// What that costs, and what it asks of the view:
    ///
    /// - It must be **display-only**. Its hitboxes, listeners, tooltips and
    ///   focus handles are recorded when it is last visited by the walk, and are
    ///   not refreshed by an in-place re-render.
    /// - Its **size must not depend on what changes**. If a re-render measures a
    ///   different height than the cached node was given, the view falls back to
    ///   the ordinary path for that frame (ancestors included) so they can
    ///   lay out again.
    /// - Anything that makes in-place re-rendering unsafe (a window-wide layout
    ///   or hit invalidation, a texture-retained layer, a layer that has
    ///   no content yet) also falls back to the ordinary path. Isolation is
    ///   never a reason for a view to be stale.
    ///
    /// Has no effect unless the view is also [`Self::cached`] or
    /// [`Self::cached_auto_height`].
    pub fn isolated(mut self) -> Self {
        self.isolated = true;
        self
    }

    /// Run this view's `render` and box the result.
    pub(crate) fn render_element(&self, window: &mut Window, cx: &mut App) -> AnyElement {
        (self.render)(self, window, cx)
    }

    /// Whether this view caches *and* asked to be an isolation boundary.
    pub(crate) fn is_isolation_boundary(&self) -> bool {
        self.isolated && self.cached_style.is_some()
    }

    /// Convert this to a weak handle.
    pub fn downgrade(&self) -> AnyWeakView {
        AnyWeakView {
            entity: self.entity.downgrade(),
            render: self.render,
            type_name: self.type_name,
        }
    }

    /// Convert this to a [Entity] of a specific type.
    /// If this handle does not contain a view of the specified type, returns itself in an `Err` variant.
    pub fn downcast<T: 'static>(self) -> Result<Entity<T>, Self> {
        match self.entity.downcast() {
            Ok(entity) => Ok(entity),
            Err(entity) => Err(Self {
                entity,
                render: self.render,
                cached_style: self.cached_style,
                auto_height: self.auto_height,
                isolated: self.isolated,
                rendered_inline: false,
                type_name: self.type_name,
            }),
        }
    }

    /// Gets the [TypeId] of the underlying view.
    pub fn entity_type(&self) -> TypeId {
        self.entity.entity_type
    }

    /// Gets the entity id of this handle.
    pub fn entity_id(&self) -> EntityId {
        self.entity.entity_id()
    }

    /// # Safety
    /// The caller must ensure the underlying entity is of type T.
    pub unsafe fn downgrade_unchecked<T: 'static>(&self) -> WeakEntity<T> {
        WeakEntity::from_raw(self.entity.downgrade())
    }
}

impl PartialEq for AnyView {
    fn eq(&self, other: &Self) -> bool {
        self.entity == other.entity
    }
}

impl Eq for AnyView {}

impl Element for AnyView {
    type RequestLayoutState = Option<AnyElement>;
    type PrepaintState = Option<AnyElement>;

    fn id(&self) -> Option<ElementId> {
        Some(ElementId::View(self.entity_id()))
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        window.with_rendered_view(self.entity_id(), |window| {
            // Disable caching when inspecting so that mouse_hit_test has all hitboxes.
            let caching_disabled = window.is_inspector_picking(cx);
            // An auto-height view has nothing to give its cached node until it
            // has been measured once; until then it renders inline.
            let remembered_height = if self.auto_height && !caching_disabled {
                _id.and_then(|id| remembered_view_height(id, window))
            } else {
                None
            };
            self.rendered_inline = self.auto_height && remembered_height.is_none();
            match self.cached_style.as_ref() {
                Some(style) if !caching_disabled && !self.rendered_inline => {
                    let mut root_style = Style::default();
                    root_style.refine(style);
                    if let Some(height) = remembered_height {
                        root_style.size.height = height.into();
                    }
                    let layout_id = window.request_layout(root_style, None, cx);
                    (layout_id, None)
                }
                _ => {
                    // Same rationale as `Entity<V>::request_layout`.
                    let _arena_scope = ElementArenaScope::enter(cx.element_arena());
                    let mut element = {
                        let _t = crate::render_stats::scope("frame: render");
                        wgpui_scope_dyn!(format!("render {}", self.type_name));
                        (self.render)(self, window, cx)
                    };
                    let layout_id = element.request_layout(window, cx);
                    (layout_id, Some(element))
                }
            }
        })
    }

    fn prepaint(
        &mut self,
        global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        element: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<AnyElement> {
        window.set_view_id(self.entity_id());
        window.with_rendered_view(self.entity_id(), |window| {
            if let Some(mut element) = element.take() {
                element.prepaint(window, cx);
                // Rendered inline, so `bounds` is this view's real laid-out
                // size. Remember it: from the next frame on the view caches.
                if self.rendered_inline
                    && self.cached_style.is_some()
                    && let Some(id) = global_id
                {
                    remember_view_height(id, bounds.size.height, window);
                }
                return Some(element);
            }

            window.with_element_state::<AnyViewState, _>(
                global_id.unwrap(),
                |element_state, window| {
                    let content_mask = window.content_mask();
                    let text_style = window.text_style();

                    // Stored ranges are absolute offsets into per-frame arrays.
                    // Verify they still fit before anything is copied — a stale
                    // range would otherwise slice out of bounds and abort the
                    // process. Both ranges are checked here, at prepaint, because
                    // once prepaint commits to reusing, paint has no rebuild path
                    // left. A failure is just a cache miss: we fall through and
                    // rebuild, costing a frame's work rather than the process.
                    let stale_range = element_state.as_ref().and_then(|state| {
                        window.invalid_reuse_range(&state.prepaint_range, &state.paint_range)
                    });
                    if let Some((array, end, len)) = stale_range {
                        log_stale_reuse_range(self.entity_id(), array, end, len);
                    }

                    // `dirty_views` alone is not enough. It is built by walking
                    // the dispatch tree upward from each notified entity, so it
                    // only ever contains entities that own a dispatch node —
                    // views that were prepainted. Notifying anything else (a
                    // model, or any entity a view merely reads) marks nothing,
                    // and every cached view rendering that entity's data judges
                    // itself clean and replays, indefinitely. That is issue #83.
                    //
                    // So also ask whether anything in this view's *own* recorded
                    // dependency set changed. Nesting is covered for free: the
                    // set is cumulative over the whole subtree, so every cached
                    // layer above the view that actually reads the entity fails
                    // this test too and rebuilds, which is what gets the inner
                    // one prepainted at all.
                    let dependency_invalidated = element_state.as_ref().is_some_and(|state| {
                        window.accessed_entity_invalidated(&state.accessed_entities)
                            // Already re-rendered in place this frame, before the walk.
                            && !window.isolated_view_is_fresh(self.entity_id())
                    });
                    let layer_invalidated = crate::layer::layers_enabled()
                        && window.cached_layer_invalidated(
                            window.layer_identity(global_id.unwrap(), bounds).0,
                        );

                    // An auto-height view entirely outside the clip (a row scrolled
                    // out of an inspector) has nothing to show: skip rebuilding,
                    // replaying and painting it. Its remembered height still
                    // keeps the scroll extent right. It rebuilds when it comes
                    // back into view.
                    if self.auto_height
                        && bounds.size.height > crate::px(0.)
                        && !content_mask.bounds.intersects(&bounds)
                    {
                        crate::render_stats::count("view cache: culled (off-screen)");
                        let empty_range = {
                            let index = window.prepaint_index();
                            index.clone()..index
                        };
                        let mut state = element_state.unwrap_or_else(|| AnyViewState {
                            prepaint_range: empty_range.clone(),
                            paint_range: PaintIndex::default()..PaintIndex::default(),
                            cache_key: ViewCacheKey::default(),
                            accessed_entities: FxHashSet::default(),
                            culled: true,
                        });
                        state.culled = true;
                        state.prepaint_range = empty_range;
                        return (None, state);
                    }

                    if let Some(mut element_state) = element_state
                        && stale_range.is_none()
                        && !element_state.culled
                        && element_state.cache_key.bounds == bounds
                        && element_state.cache_key.content_mask == content_mask
                        && element_state.cache_key.text_style == text_style
                        && !window.dirty_views.contains(&self.entity_id())
                        && !dependency_invalidated
                        && !layer_invalidated
                        && window.view_cache_available()
                    {
                        crate::render_stats::count("view cache: reused");
                        let _t = crate::render_stats::scope("view cache: reuse_prepaint");
                        let prepaint_start = window.prepaint_index();
                        window.reuse_prepaint(element_state.prepaint_range.clone());
                        if self.is_isolation_boundary() {
                            // Its dependencies stay its own; see `AnyView::isolated`.
                            if let Some(id) = global_id {
                                window.note_isolated_view(
                                    self,
                                    id,
                                    bounds,
                                    content_mask.clone(),
                                    element_state.accessed_entities.clone(),
                                    self.auto_height,
                                    false,
                                );
                            }
                        } else {
                            cx.entities
                                .extend_accessed(&element_state.accessed_entities);
                        }

                        // `on_frame` effects still run on a cache hit — that is
                        // the whole point of the channel: side effects that must
                        // fire every frame regardless of caching. They are
                        // replayed from what the subtree recorded when it last
                        // rendered, *not* by rebuilding the subtree to find
                        // them again. Rebuilding would run `render` and a full
                        // `layout_as_root` on every reuse, which is most of
                        // what the cache is here to skip.
                        window.replay_frame_effects(&element_state.prepaint_range, cx);

                        let prepaint_end = window.prepaint_index();
                        element_state.prepaint_range = prepaint_start..prepaint_end;

                        return (None, element_state);
                    }

                    // Cache miss. If this fires every frame for a view whose
                    // content is static, something is calling `cx.notify()` on
                    // it or on one of its descendants — `mark_view_dirty` walks
                    // the ancestor path, so a chatty leaf invalidates every
                    // cached view above it.
                    crate::render_stats::count("view cache: rebuilt");
                    if dependency_invalidated {
                        // Counted separately because this is the class of
                        // rebuild the #83 fix added. If it dominates, some
                        // entity read across a whole subtree is being notified
                        // every frame, and the fix to make is at that call site
                        // rather than here.
                        crate::render_stats::count("view cache: rebuilt (dependency changed)");
                        crate::render_stats::count_tagged(
                            "view cache: rebuilt (dependency changed): ",
                            self.type_name,
                        );
                    }
                    let _t = crate::render_stats::scope("view cache: rebuild");
                    // Name the rebuilt view and why the cache missed, so the
                    // hot, constantly-rebuilding views identify themselves.
                    wgpui_scope_dyn!(format!(
                        "view rebuild ({}): {}",
                        if dependency_invalidated {
                            "dependency changed"
                        } else if stale_range.is_some() {
                            "stale range"
                        } else {
                            "dirty/bounds/first"
                        },
                        self.type_name
                    ));

                    // Rebuilding this view normally forces every cached view
                    // nested inside it to rebuild too. See
                    // `nested_view_cache_enabled` for why, and for the opt-in
                    // that lifts it.
                    let nested_cache_suppressed = window.nested_view_cache_suppressed;
                    if !nested_view_cache_enabled() {
                        window.nested_view_cache_suppressed = true;
                    }

                    let prepaint_start = window.prepaint_index();
                    let _arena_scope = ElementArenaScope::enter(cx.element_arena());
                    let ((mut element, measured_height), accessed_entities) = cx.detect_accessed_entities_with(!self.is_isolation_boundary(), |cx| {
                        // Split three ways: building the element tree is usually
                        // trivial next to laying it out and prepainting it, and
                        // conflating them hides which one to go after.
                        let mut element = {
                            let _t = crate::render_stats::scope("  rebuild: render");
                            // Also counted into the whole-frame bucket. A cached
                            // view renders from prepaint rather than from
                            // request_layout, so this is the one place where
                            // `frame: render` nests under `frame: prepaint`.
                            let _frame_render = crate::render_stats::scope("frame: render");
                            wgpui_scope_dyn!(format!("render {}", self.type_name));
                            (self.render)(self, window, cx)
                        };
                        let mut measured_height = None;
                        {
                            let _t = crate::render_stats::scope("  rebuild: layout");
                            if self.auto_height {
                                // Measure the content at this width with the height
                                // unconstrained, so a root that fills its parent
                                // (`h_full`) reports its content rather than echoing
                                // back the height it was given.
                                let available = crate::size(
                                    crate::AvailableSpace::Definite(bounds.size.width),
                                    crate::AvailableSpace::MaxContent,
                                );
                                measured_height =
                                    Some(element.layout_as_root(available, window, cx).height);
                            } else {
                                element.layout_as_root(bounds.size.into(), window, cx);
                            }
                        }
                        {
                            let _t = crate::render_stats::scope("  rebuild: prepaint");
                            element.prepaint_at(bounds.origin, window, cx);
                        }
                        (element, measured_height)
                    });

                    // A rebuild that measured a different height than the cached
                    // node was given: remember the new one and invalidate, so the
                    // ancestors lay out with it on the next frame.
                    if let (Some(height), Some(id)) = (measured_height, global_id)
                        && (height - bounds.size.height).abs() > crate::px(0.5)
                    {
                        remember_view_height(id, height, window);
                        if self.is_isolation_boundary() {
                            // The ancestors must lay out again, so this notify may
                            // not stop at the isolation boundary.
                            window.request_isolated_relayout(self.entity_id());
                        }
                        cx.notify(self.entity_id());
                    }

                    let prepaint_end = window.prepaint_index();
                    window.nested_view_cache_suppressed = nested_cache_suppressed;

                    if self.is_isolation_boundary()
                        && let Some(id) = global_id
                    {
                        window.note_isolated_view(
                            self,
                            id,
                            bounds,
                            content_mask.clone(),
                            accessed_entities.clone(),
                            self.auto_height,
                            true,
                        );
                    }

                    (
                        Some(element),
                        AnyViewState {
                            accessed_entities,
                            prepaint_range: prepaint_start..prepaint_end,
                            paint_range: PaintIndex::default()..PaintIndex::default(),
                            cache_key: ViewCacheKey {
                                bounds,
                                content_mask,
                                text_style,
                            },
                            culled: false,
                        },
                    )
                },
            )
        })
    }

    fn paint(
        &mut self,
        global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _: &mut Self::RequestLayoutState,
        element: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        window.with_rendered_view(self.entity_id(), |window| {
            let caching_disabled = window.is_inspector_picking(cx);
            if self.cached_style.is_some() && !caching_disabled && !self.rendered_inline {
                let global_id = global_id.unwrap();
                // `cached` is a layer with a compat policy: every axis
                // invalidated together, primitive-retained. The decision about
                // *whether* to reuse was made in prepaint by `AnyViewState`,
                // which predates layers and reaches things layers cannot see
                // yet (recorded bounds, text style, the dispatch subtree). The
                // layer supplies the retained geometry; stacking is always
                // resolved from the current tree, including when a cached
                // subtree skips individual primitive emission through slabs.
                let (layer_key, layer_cache_key) = window.layer_identity(global_id, bounds);
                let layers_enabled = crate::layer::layers_enabled();

                window.with_element_state::<AnyViewState, _>(global_id, |element_state, window| {
                    let mut element_state = element_state.unwrap();

                    let paint_start = window.paint_index();

                    if let Some(element) = element {
                        // Paired with the prepaint path above.
                        let nested_cache_suppressed = window.nested_view_cache_suppressed;
                        if !nested_view_cache_enabled() {
                            window.nested_view_cache_suppressed = true;
                        }
                        if layers_enabled {
                            window.record_layer(
                                layer_key,
                                layer_cache_key,
                                LayerPolicy::compat(),
                                |window| element.paint(window, cx),
                            );
                        } else {
                            element.paint(window, cx);
                        }
                        window.nested_view_cache_suppressed = nested_cache_suppressed;
                    } else {
                        if element_state.culled {
                            // Off-screen and skipped in prepaint: nothing to replay.
                            let paint_end = window.paint_index();
                            element_state.paint_range = paint_start..paint_end;
                            return ((), element_state);
                        }
                        window.reuse_paint_except_scene(&element_state.paint_range);
                        // The layer can be gone even though prepaint committed
                        // to reusing — eviction is driven by draw age, and this
                        // view's element state outlives it. Falling back to the
                        // recorded scene range keeps that a slower frame rather
                        // than a missing panel.
                        if !window.try_composite_layer(layer_key) {
                            window.replay_scene_range(&element_state.paint_range);
                        }
                    }

                    let paint_end = window.paint_index();
                    element_state.paint_range = paint_start..paint_end;

                    ((), element_state)
                })
            } else {
                element.as_mut().unwrap().paint(window, cx);
            }
        });
    }
}

impl<V: 'static + Render> IntoElement for Entity<V> {
    type Element = Entity<V>;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl IntoElement for AnyView {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

/// A weak, dynamically-typed view handle that does not prevent the view from being released.
pub struct AnyWeakView {
    entity: AnyWeakEntity,
    render: fn(&AnyView, &mut Window, &mut App) -> AnyElement,
    type_name: &'static str,
}

impl AnyWeakView {
    /// Convert to a strongly-typed handle if the referenced view has not yet been released.
    pub fn upgrade(&self) -> Option<AnyView> {
        let entity = self.entity.upgrade()?;
        Some(AnyView {
            entity,
            render: self.render,
            cached_style: None,
            auto_height: false,
            isolated: false,
            rendered_inline: false,
            type_name: self.type_name,
        })
    }
}

impl<V: 'static + Render> From<WeakEntity<V>> for AnyWeakView {
    fn from(view: WeakEntity<V>) -> Self {
        AnyWeakView {
            entity: view.into(),
            render: any_view::render::<V>,
            type_name: std::any::type_name::<V>(),
        }
    }
}

impl PartialEq for AnyWeakView {
    fn eq(&self, other: &Self) -> bool {
        self.entity == other.entity
    }
}

impl std::fmt::Debug for AnyWeakView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnyWeakView")
            .field("entity_id", &self.entity.entity_id)
            .finish_non_exhaustive()
    }
}

mod any_view {
    use crate::{AnyElement, AnyView, App, IntoElement, Render, Window};

    pub(crate) fn render<V: 'static + Render>(
        view: &AnyView,
        window: &mut Window,
        cx: &mut App,
    ) -> AnyElement {
        let view = view.clone().downcast::<V>().unwrap();
        view.update(cx, |view, cx| view.render(window, cx).into_any_element())
    }
}

/// A view that renders nothing
pub struct EmptyView;

impl Render for EmptyView {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        Empty
    }
}
