use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    AtlasTextureId, AtlasTile, Background, Bounds, ContentMask, Corners, Edges, Hsla, LayerId,
    LayerKey, Pixels, Point, Radians, ScaledPixels, Size, TextColor, bounds_tree::BoundsTree,
    layer::LayerItem, platform::cross::surface_registry::SurfaceId, point,
};
use std::{
    fmt::Debug,
    iter::Peekable,
    mem,
    ops::{Add, Range, Sub},
    slice,
    sync::Arc,
};
use crate::platform::cross::slab::SlabKind;
use crate::scene_pack::PackedLayer;

#[allow(non_camel_case_types, unused)]
pub(crate) type PathVertex_ScaledPixels = PathVertex<ScaledPixels>;

pub(crate) type DrawOrder = u32;

/// One ordering scope: a `BoundsTree` starting at zero, plus where the scope
/// sits in its parent.
///
/// The root scope is the window. Every retained layer paints into a scope of
/// its own, which is what makes a layer's draw orders independent of everything
/// painted outside it. Before this existed, `insert_primitive` assigned `order`
/// from a single global tree, so a primitive's z — and therefore, downstream,
/// its byte offset in the GPU buffer — was a function of every other primitive
/// in the window.
struct OrderScope {
    tree: BoundsTree<ScaledPixels>,
    /// The scope this one was entered from, and the local order it was entered
    /// at. `None` for the root.
    parent: Option<(usize, DrawOrder)>,
    /// Scopes entered from this one, in entry order.
    children: Vec<usize>,
    /// The highest local order handed out in this scope.
    max_local: DrawOrder,
    /// Whether entering this scope pushed a synthetic clip entry that leaving
    /// it has to pop. See [`Scene::begin_scope`].
    synthetic_clip: bool,
    /// The union of everything actually painted in this scope, including nested
    /// scopes. A layer's content is not confined to its bounds — shadows,
    /// overflowing children and outlines all paint outside — so this, not the
    /// declared bounds, is what the parent has to record.
    painted: Option<Bounds<ScaledPixels>>,
    /// Local order -> global order. Built by [`Scene::resolve_orders`].
    global: Vec<DrawOrder>,
}

impl OrderScope {
    fn new(parent: Option<(usize, DrawOrder)>) -> Self {
        OrderScope {
            tree: BoundsTree::default(),
            parent,
            children: Vec::new(),
            max_local: 0,
            synthetic_clip: false,
            painted: None,
            global: Vec::new(),
        }
    }
}

/// A clip group opened by [`Scene::push_layer`], tagged with the ordering scope
/// it belongs to.
///
/// The scope matters because a retained layer may be painted inside a clip
/// group: the clip's order is a *local* order in the outer scope and means
/// nothing in the inner one.
#[derive(Copy, Clone)]
struct ClipEntry {
    scope: usize,
    order: DrawOrder,
}

/// The length of every primitive array, captured at a scope boundary.
///
/// Primitives are appended in paint order, so the stretch of each array a scope
/// contributed is exactly the span between two of these. Recording nine lengths
/// twice per layer is what lets `finish` rewrite orders per scope without
/// tagging every individual primitive with the scope it came from.
#[derive(Clone, Copy, Default)]
struct ArrayLens {
    shadows: usize,
    backdrop_filters: usize,
    filter_boundaries: usize,
    quads: usize,
    paths: usize,
    underlines: usize,
    monochrome_sprites: usize,
    polychrome_sprites: usize,
    surfaces: usize,
}

/// One uninterrupted stretch of one scope's primitives.
struct ScopeRun {
    scope: usize,
    start: ArrayLens,
    end: ArrayLens,
}

pub(crate) struct Scene {
    pub(crate) paint_operations: Vec<PaintOperation>,
    /// Every ordering scope this frame. `scopes[0]` is the root and always
    /// exists.
    scopes: Vec<OrderScope>,
    /// The chain of open scopes; the last is where primitives land.
    scope_stack: Vec<usize>,
    /// Contiguous stretches of primitives, one per uninterrupted period a scope
    /// was active. Closed and reopened at every scope boundary.
    runs: Vec<ScopeRun>,
    clip_stack: Vec<ClipEntry>,
    /// Layers currently being painted, innermost last.
    ///
    /// The `Vec` is present when the layer is *recording* — re-rendering, and
    /// keeping what it emits. A compositing layer is on the stack with no
    /// recording buffer, so that an enclosing recorder still learns it was
    /// nested here and can replay it by reference next time.
    capture_stack: Vec<(LayerKey, Option<Vec<LayerItem>>)>,
    pub(crate) shadows: Vec<Shadow>,
    pub(crate) backdrop_filters: Vec<BackdropFilter>,
    pub(crate) filter_boundaries: Vec<FilterBoundary>,
    pub(crate) quads: Vec<Quad>,
    pub(crate) paths: Vec<Path<ScaledPixels>>,
    pub(crate) underlines: Vec<Underline>,
    pub(crate) monochrome_sprites: Vec<MonochromeSprite>,
    pub(crate) polychrome_sprites: Vec<PolychromeSprite>,
    pub(crate) surfaces: Vec<PaintSurface>,
    /// Spliced slab references recorded this frame, in reservation order;
    /// sorted by resolved global order in [`Scene::finish`]. The renderer
    /// resolves each against its own registry at draw time — the scene never
    /// holds GPU state, only the marker plus the packed bytes that back it.
    pub(crate) layer_slab_spans: Vec<LayerSlabSpan>,
}

impl Default for Scene {
    fn default() -> Self {
        Scene {
            paint_operations: Vec::new(),
            scopes: vec![OrderScope::new(None)],
            scope_stack: vec![0],
            runs: vec![ScopeRun {
                scope: 0,
                start: ArrayLens::default(),
                end: ArrayLens::default(),
            }],
            clip_stack: Vec::new(),
            capture_stack: Vec::new(),
            shadows: Vec::new(),
            backdrop_filters: Vec::new(),
            filter_boundaries: Vec::new(),
            quads: Vec::new(),
            paths: Vec::new(),
            underlines: Vec::new(),
            monochrome_sprites: Vec::new(),
            polychrome_sprites: Vec::new(),
            surfaces: Vec::new(),
            layer_slab_spans: Vec::new(),
        }
    }
}

impl Scene {
    pub fn clear(&mut self) {
        self.paint_operations.clear();
        self.scopes.clear();
        self.scopes.push(OrderScope::new(None));
        self.scope_stack.clear();
        self.scope_stack.push(0);
        self.runs.clear();
        self.runs.push(ScopeRun {
            scope: 0,
            start: ArrayLens::default(),
            end: ArrayLens::default(),
        });
        self.clip_stack.clear();
        self.capture_stack.clear();
        self.paths.clear();
        self.shadows.clear();
        self.backdrop_filters.clear();
        self.filter_boundaries.clear();
        self.quads.clear();
        self.underlines.clear();
        self.monochrome_sprites.clear();
        self.polychrome_sprites.clear();
        self.surfaces.clear();
        self.layer_slab_spans.clear();
    }

    pub fn len(&self) -> usize {
        self.paint_operations.len()
    }

    fn array_lens(&self) -> ArrayLens {
        ArrayLens {
            shadows: self.shadows.len(),
            backdrop_filters: self.backdrop_filters.len(),
            filter_boundaries: self.filter_boundaries.len(),
            quads: self.quads.len(),
            paths: self.paths.len(),
            underlines: self.underlines.len(),
            monochrome_sprites: self.monochrome_sprites.len(),
            polychrome_sprites: self.polychrome_sprites.len(),
            surfaces: self.surfaces.len(),
        }
    }

    /// Close the open run and start one for `scope`.
    fn switch_run(&mut self, scope: usize) {
        let at = self.array_lens();
        if let Some(run) = self.runs.last_mut() {
            run.end = at;
        }
        self.runs.push(ScopeRun {
            scope,
            start: at,
            end: at,
        });
    }

    fn active_scope(&self) -> usize {
        self.scope_stack.last().copied().unwrap_or(0)
    }

    /// Hand out the next local order in the active scope, recording it as the
    /// scope's high-water mark and widening the scope's painted extent.
    fn note_order(&mut self, order: DrawOrder, bounds: Bounds<ScaledPixels>) -> DrawOrder {
        let scope = &mut self.scopes[self.scope_stack.last().copied().unwrap_or(0)];
        scope.max_local = scope.max_local.max(order);
        scope.painted = Some(match scope.painted {
            Some(painted) => painted.union(&bounds),
            None => bounds,
        });
        order
    }

    /// Open an ordering scope for a layer occupying `bounds`.
    ///
    /// The layer takes an order strictly above everything already painted in
    /// the parent scope, and its whole contents are numbered into the gap
    /// between that order and the next. Above-all rather than overlap-based
    /// because a layer's content is not confined to its bounds — shadows,
    /// overflowing children and outlines all paint outside — so an
    /// overlap-derived order could place content beneath something it was
    /// painted after. This mirrors how content-filter boundaries have always
    /// been ordered.
    fn begin_scope(&mut self, bounds: Bounds<ScaledPixels>) -> usize {
        let parent = self.active_scope();
        let entry_order = self.scopes[parent].tree.insert_above_all(bounds);
        self.scopes[parent].max_local = self.scopes[parent].max_local.max(entry_order);

        let index = self.scopes.len();
        self.scopes.push(OrderScope::new(Some((parent, entry_order))));
        self.scopes[parent].children.push(index);
        self.scope_stack.push(index);

        // A clip group open in an enclosing scope collapses everything inside
        // it to one order. That order belongs to the outer scope, so re-express
        // it here: open a clip in the new scope too, and the collapse carries
        // across the boundary instead of being silently dropped.
        if self
            .clip_stack
            .last()
            .is_some_and(|clip| clip.scope != index)
        {
            let order = self.scopes[index].tree.insert(bounds);
            self.scopes[index].max_local = self.scopes[index].max_local.max(order);
            self.clip_stack.push(ClipEntry {
                scope: index,
                order,
            });
            self.scopes[index].synthetic_clip = true;
        }

        self.switch_run(index);
        index
    }

    /// Close the scope opened by [`Self::begin_scope`], widening the parent's
    /// record of the layer to whatever it actually painted.
    ///
    /// The parent recorded the layer's *bounds* on entry, but content escapes
    /// those bounds routinely. Re-inserting the union at the same order means
    /// content painted afterwards that overlaps the overflow still sorts above
    /// the layer, which an entry-time insert alone would not guarantee.
    fn end_scope(&mut self) {
        let index = self.active_scope();
        if self.scopes[index].synthetic_clip {
            self.clip_stack.pop();
        }
        self.scope_stack.pop();

        let painted = self.scopes[index].painted;
        if let (Some((parent, entry_order)), Some(painted)) = (self.scopes[index].parent, painted) {
            crate::render_stats::count("layer: order tree reinsert");
            self.scopes[parent].tree.insert_at_order(painted, entry_order);
            let parent_painted = &mut self.scopes[parent].painted;
            *parent_painted = Some(match *parent_painted {
                Some(existing) => existing.union(&painted),
                None => painted,
            });
        }

        let parent = self.active_scope();
        self.switch_run(parent);
    }

    /// Open an ordering scope for the layer `key`.
    ///
    /// `record` asks for everything painted inside to be kept, which is what a
    /// re-rendering layer wants. A compositing layer passes `false`: it is
    /// replaying content it already holds, and re-recording it would only
    /// duplicate it.
    pub fn begin_layer(&mut self, key: LayerKey, bounds: Bounds<ScaledPixels>, record: bool) {
        self.begin_scope(bounds);
        self.capture_stack
            .push((key, record.then(Vec::new)));
    }

    /// Close the scope opened by [`Self::begin_layer`], returning what it
    /// captured — in paint order, carrying layer-local draw orders — if it was
    /// recording.
    pub fn end_layer(&mut self) -> Option<Vec<LayerItem>> {
        let (key, items) = self
            .capture_stack
            .pop()
            .expect("end_layer without a matching begin_layer");
        self.end_scope();
        // An enclosing recorder stores the nested layer by reference. Inlining
        // its primitives would merge two local order spaces into one and
        // silently reorder them, and it would also defeat the nested layer's
        // own independent invalidation.
        if let Some((_, Some(parent_items))) = self.capture_stack.last_mut() {
            parent_items.push(LayerItem::Nested(key));
        }
        items
    }

    /// Re-emit a retained primitive, keeping the layer-local order it was
    /// recorded with.
    ///
    /// This is the saving the whole phase is for: no `BoundsTree` insert, and
    /// no re-derivation of z from whatever else happens to be painted this
    /// frame.
    pub fn push_retained(&mut self, primitive: &Primitive) {
        let mut primitive = primitive.clone();
        let bounds = primitive
            .bounds()
            .intersect(&primitive.content_mask().bounds);
        let order = self.note_order(primitive_order(&primitive), bounds);
        set_primitive_order(&mut primitive, order);
        // Path ids are indices into `self.paths`, so they mean nothing outside
        // the scene being built and have to be reassigned on every replay.
        if let Primitive::Path(path) = &mut primitive {
            path.id = PathId(self.paths.len());
        }
        count_primitive(&primitive);
        self.push_to_array(&primitive);
        // Symmetric with `insert_primitive`'s own capture-awareness (#92): if
        // the innermost layer is actively recording, this replayed primitive
        // has to land in its new item list too, or the layer's next composite
        // would be missing it. `composite_layer`, this method's original and
        // still only caller before #92, always calls `begin_layer(record:
        // false)`, so `capture_stack.last()`'s items is `None` there and this
        // is a no-op for it — this only fires for `Window::replay_instance_items`,
        // called from inside an actively-recording layer while a reconciled
        // child's paint is being skipped and its retained items re-emitted.
        if let Some((_, Some(items))) = self.capture_stack.last_mut() {
            items.push(LayerItem::Primitive(primitive.clone()));
        }
        self.paint_operations
            .push(PaintOperation::Primitive(primitive));
    }

    /// Re-register a raw [`LayerItem`] — specifically a nested-layer reference
    /// — in the innermost recording layer's item list, without touching any
    /// primitive array or draw order (#92).
    ///
    /// The counterpart to `push_retained` for the other half of
    /// `Window::replay_instance_items`: a reconciled child's own subtree may
    /// contain a nested `.layer()` div, which contributes a
    /// `LayerItem::Nested` reference (see `end_layer`) rather than a
    /// primitive. Re-registering it here — rather than visiting the nested
    /// layer to re-derive it — is what lets a reconciled parent skip its
    /// child's `paint` entirely; the nested layer's own record persists
    /// independently in `Window::layers` regardless.
    pub(crate) fn push_captured_item(&mut self, item: LayerItem) {
        if let Some((_, Some(items))) = self.capture_stack.last_mut() {
            items.push(item);
        }
    }

    /// How many items the innermost recording layer has captured so far
    /// (#92). Bracketing a reconciled child's contribution — `captured_len`
    /// before its `paint`, `captured_len` after — is how `Div`'s child loop
    /// learns which of the capture's items are this child's own, without a
    /// second, parallel list.
    pub(crate) fn captured_len(&self) -> usize {
        self.capture_stack
            .last()
            .and_then(|(_, items)| items.as_ref())
            .map(Vec::len)
            .unwrap_or(0)
    }

    /// Whether an enclosing layer is actively recording. Slab splicing must
    /// not run inside a recording context: a recording parent captures its
    /// nested composite's primitives as its own future content, and spans
    /// would leave that capture empty.
    pub(crate) fn innermost_layer_is_recording(&self) -> bool {
        self.capture_stack
            .last()
            .is_some_and(|(_, items)| items.is_some())
    }

    /// Clone the items the innermost recording layer captured within `range`
    /// (#92). Paired with `captured_len`; see its doc comment.
    pub(crate) fn captured_slice(&self, range: Range<usize>) -> Vec<LayerItem> {
        self.capture_stack
            .last()
            .and_then(|(_, items)| items.as_ref())
            .map(|items| items[range].to_vec())
            .unwrap_or_default()
    }

    pub fn push_layer(&mut self, bounds: Bounds<ScaledPixels>) {
        let scope = self.active_scope();
        let order = self.scopes[scope].tree.insert(bounds);
        self.note_order(order, bounds);
        self.clip_stack.push(ClipEntry { scope, order });
        self.paint_operations
            .push(PaintOperation::StartLayer(bounds));
    }

    pub fn pop_layer(&mut self) {
        self.clip_stack.pop();
        self.paint_operations.push(PaintOperation::EndLayer);
    }

    /// Record a slab splice marker: the layer's content is drawn from the
    /// renderer's per-layer slabs instead of this frame's primitive arrays.
    ///
    /// The marker reserves one draw-order slot strictly above everything
    /// painted so far in the active scope — the same reservation a layer
    /// boundary gets — so surrounding content sorts around it exactly where
    /// the layer would have sat. `bounds` should be the layer's composited
    /// bounds; they only shape the order slot, never clip anything.
    ///
    /// A clean layer's composite emits one of these per maximal run of its own
    /// primitives (nested layers split it), carrying the packed bytes as ground
    /// truth: the renderer compares `content_token` against its registry to
    /// decide between zero uploads and re-uploading exactly this layer's
    /// ranges.
    pub fn push_layer_slab_span(
        &mut self,
        bounds: Bounds<ScaledPixels>,
        key: LayerKey,
        content_token: u64,
        origin: [f32; 2],
        totals: [u32; SlabKind::COUNT],
        runs: Vec<SlabRun>,
        packed: Arc<PackedLayer>,
        texture: Option<LayerTextureTarget>,
    ) {
        let scope = self.active_scope();
        let local_order = self.scopes[scope].tree.insert_above_all(bounds);
        self.note_order(local_order, bounds);
        self.layer_slab_spans.push(LayerSlabSpan {
            key,
            content_token,
            origin,
            totals,
            runs,
            packed,
            texture,
            reservation_bounds: bounds,
            order_scope: scope,
            local_order,
            order: 0,
        });
        let span = self.layer_slab_spans.last().expect("just pushed").clone();
        self.paint_operations.push(PaintOperation::LayerSlab(span));
    }

    /// How many slab spans this frame carries.
    pub fn slab_span_count(&self) -> usize {
        self.layer_slab_spans.len()
    }

    /// Raise the draw-order floor so every primitive inserted afterwards sorts above everything
    /// inserted before. Called before painting deferred draws so overlays (tooltips, popovers,
    /// drag images) sort above the main scene — and a deferred backdrop's order can't fall inside
    /// a content-filter (`filter`) order range left behind by the main scene.
    ///
    /// Scoped to the active ordering scope. Deferred draws are painted at the
    /// top level, so in practice that is the root — but a layer that hoists its
    /// own deferred content hoists it within itself, which is the layer-granular
    /// behaviour this phase is for.
    pub fn raise_order_floor(&mut self) {
        let scope = self.active_scope();
        let floor = self.scopes[scope].tree.max_order() + 1;
        self.scopes[scope].tree.set_order_floor(floor);
    }

    pub fn insert_primitive(&mut self, primitive: impl Into<Primitive>) {
        let mut primitive = primitive.into();
        let clipped_bounds = primitive
            .bounds()
            .intersect(&primitive.content_mask().bounds);

        // Content-filter boundaries must always be inserted as matched pairs — dropping one
        // (e.g. for an empty clipped region) would orphan its partner and corrupt the renderer's
        // target stack. Each marker takes an order strictly above ALL prior content, so the start
        // sorts after everything painted before it and the element's own children (which overlap
        // the marker bounds) sort strictly above the start. This keeps a marker's order range from
        // colliding with unrelated non-overlapping content that reuses low orderings (e.g. a
        // background grid), which would otherwise sweep that content into the group.
        let is_filter_boundary = matches!(primitive, Primitive::FilterBoundary(_));

        if clipped_bounds.is_empty() && !is_filter_boundary {
            return;
        }

        // A degenerate filter boundary has no clipped extent, but it still has
        // to widen the scope's painted region — an enclosing layer's recorded
        // extent must cover the marker pair or a later sibling could sort
        // between them.
        let extent = if clipped_bounds.is_empty() {
            *primitive.bounds()
        } else {
            clipped_bounds
        };

        let scope = self.active_scope();
        let order = {
            let _t = crate::render_stats::scope("frame: bounds tree");
            if is_filter_boundary {
                self.scopes[scope].tree.insert_above_all(extent)
            } else {
                self.clip_stack
                    .last()
                    .filter(|clip| clip.scope == scope)
                    .map(|clip| clip.order)
                    .unwrap_or_else(|| self.scopes[scope].tree.insert(clipped_bounds))
            }
        };
        let order = self.note_order(order, extent);
        set_primitive_order(&mut primitive, order);
        if let Primitive::Path(path) = &mut primitive {
            path.id = PathId(self.paths.len());
        }
        count_primitive(&primitive);
        self.push_to_array(&primitive);
        if let Some((_, Some(items))) = self.capture_stack.last_mut() {
            items.push(LayerItem::Primitive(primitive.clone()));
        }
        self.paint_operations
            .push(PaintOperation::Primitive(primitive));
    }

    /// Append `primitive` to the array for its kind, without touching its order.
    fn push_to_array(&mut self, primitive: &Primitive) {
        match primitive {
            Primitive::Shadow(shadow) => self.shadows.push(shadow.clone()),
            Primitive::BackdropFilter(filter) => self.backdrop_filters.push(*filter),
            Primitive::FilterBoundary(boundary) => self.filter_boundaries.push(*boundary),
            Primitive::Quad(quad) => self.quads.push(quad.clone()),
            Primitive::Path(path) => self.paths.push(path.clone()),
            Primitive::Underline(underline) => self.underlines.push(underline.clone()),
            Primitive::MonochromeSprite(sprite) => self.monochrome_sprites.push(sprite.clone()),
            Primitive::PolychromeSprite(sprite) => self.polychrome_sprites.push(sprite.clone()),
            Primitive::Surface(surface) => self.surfaces.push(surface.clone()),
        }
    }

    pub fn replay(&mut self, range: Range<usize>, prev_scene: &Scene) {
        for operation in &prev_scene.paint_operations[range] {
            match operation {
                PaintOperation::Primitive(primitive) => self.insert_primitive(primitive.clone()),
                PaintOperation::StartLayer(bounds) => self.push_layer(*bounds),
                PaintOperation::EndLayer => self.pop_layer(),
                PaintOperation::LayerSlab(span) => {
                    // Re-reserve the order slot in this frame's scope layout;
                    // everything else about the marker is frame-independent.
                    let mut span = span.clone();
                    let scope = self.active_scope();
                    let local_order = self
                        .scopes[scope]
                        .tree
                        .insert_above_all(span.reservation_bounds);
                    self.note_order(local_order, span.reservation_bounds);
                    span.order_scope = scope;
                    span.local_order = local_order;
                    self.layer_slab_spans.push(span);
                }
            }
        }
    }

    /// Map every scope's local orders into a global range reserved for it, and
    /// rewrite the orders recorded on primitives to match.
    ///
    /// Depth-first, in entry order: a scope entered at parent-local order `o`
    /// consumes a contiguous run of global orders sitting strictly between the
    /// global orders of `o` and `o + 1`. That is exactly the nesting property
    /// the local trees were built on — content inside a layer sorts above
    /// everything painted before the layer in its parent, and below everything
    /// the parent painted afterwards that overlaps it.
    fn resolve_orders(&mut self) {
        profiling::scope!("wgpui: resolve_orders");
        fn assign(scopes: &mut Vec<OrderScope>, index: usize, counter: &mut DrawOrder) {
            let max_local = scopes[index].max_local;
            // Taken out so the recursive call can borrow `scopes` mutably;
            // restored before returning.
            let children = mem::take(&mut scopes[index].children);

            let mut global = vec![0; max_local as usize + 1];
            let mut next_child = 0;
            for local in 1..=max_local {
                *counter += 1;
                global[local as usize] = *counter;
                while next_child < children.len() {
                    let child = children[next_child];
                    // Entry orders are handed out by `insert_above_all`, so
                    // they only ever increase and this scan is linear.
                    match scopes[child].parent {
                        Some((_, entry)) if entry == local => {
                            assign(scopes, child, counter);
                            next_child += 1;
                        }
                        _ => break,
                    }
                }
            }
            // Defensive: a child whose entry order somehow exceeded the
            // parent's high-water mark still has to be numbered, or its
            // primitives would keep unmapped local orders.
            while next_child < children.len() {
                assign(scopes, children[next_child], counter);
                next_child += 1;
            }

            scopes[index].global = global;
            scopes[index].children = children;
        }

        let mut counter = 0;
        assign(&mut self.scopes, 0, &mut counter);

        for run in &self.runs {
            let global = &self.scopes[run.scope].global;
            let map = |order: &mut DrawOrder| {
                *order = global
                    .get(*order as usize)
                    .copied()
                    .unwrap_or(DrawOrder::MAX);
            };
            for shadow in &mut self.shadows[run.start.shadows..run.end.shadows] {
                map(&mut shadow.order);
            }
            for filter in
                &mut self.backdrop_filters[run.start.backdrop_filters..run.end.backdrop_filters]
            {
                map(&mut filter.order);
            }
            for boundary in
                &mut self.filter_boundaries[run.start.filter_boundaries..run.end.filter_boundaries]
            {
                map(&mut boundary.order);
            }
            for quad in &mut self.quads[run.start.quads..run.end.quads] {
                map(&mut quad.order);
            }
            for path in &mut self.paths[run.start.paths..run.end.paths] {
                map(&mut path.order);
            }
            for underline in &mut self.underlines[run.start.underlines..run.end.underlines] {
                map(&mut underline.order);
            }
            for sprite in &mut self.monochrome_sprites
                [run.start.monochrome_sprites..run.end.monochrome_sprites]
            {
                map(&mut sprite.order);
            }
            for sprite in &mut self.polychrome_sprites
                [run.start.polychrome_sprites..run.end.polychrome_sprites]
            {
                map(&mut sprite.order);
            }
            for surface in &mut self.surfaces[run.start.surfaces..run.end.surfaces] {
                map(&mut surface.order);
            }
        }

        // Slab spans order exactly like a one-primitive scope entry: through
        // the same local→global mapping their reserving scope received.
        for span in &mut self.layer_slab_spans {
            span.order = self
                .scopes
                .get(span.order_scope)
                .and_then(|scope| scope.global.get(span.local_order as usize))
                .copied()
                .unwrap_or(DrawOrder::MAX);
        }
    }

    pub fn finish(&mut self) {
        profiling::scope!("wgpui: scene finish");
        let _t = crate::render_stats::scope("frame: scene finish");
        debug_assert_eq!(
            self.scope_stack.len(),
            1,
            "a layer was left open when the frame finished"
        );
        if let Some(run) = self.runs.last_mut() {
            run.end = ArrayLens {
                shadows: self.shadows.len(),
                backdrop_filters: self.backdrop_filters.len(),
                filter_boundaries: self.filter_boundaries.len(),
                quads: self.quads.len(),
                paths: self.paths.len(),
                underlines: self.underlines.len(),
                monochrome_sprites: self.monochrome_sprites.len(),
                polychrome_sprites: self.polychrome_sprites.len(),
                surfaces: self.surfaces.len(),
            };
        }
        self.resolve_orders();
        self.shadows.sort_by_key(|shadow| shadow.order);
        self.quads.sort_by_key(|quad| quad.order);
        self.paths.sort_by_key(|path| path.order);
        self.underlines.sort_by_key(|underline| underline.order);
        self.monochrome_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.tile_id));
        self.polychrome_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.tile_id));
        self.surfaces.sort_by_key(|surface| surface.order);
        self.backdrop_filters.sort_by_key(|filter| filter.order);
        // Markers normally get distinct, monotonically-increasing orders (children overlap
        // their group bounds and so sort strictly between the start and end). The `!is_start`
        // tiebreak only matters for a degenerate empty group whose start and end tie: it keeps
        // the start (false = 0) ahead of the end (true = 1) so the pair stays well-formed.
        self.filter_boundaries
            .sort_by_key(|boundary| (boundary.order, !boundary.is_start));
        self.layer_slab_spans
            .sort_by_key(|span| span.order);
    }

    // Production draws through `frame_batches`; this remains the legacy
    // oracle consumed by scene_pack's and slab-splice's golden-model tests.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn batches(&self) -> impl Iterator<Item = PrimitiveBatch<'_>> {
        BatchIterator {
            shadows: &self.shadows,
            shadows_start: 0,
            shadows_iter: self.shadows.iter().peekable(),
            quads: &self.quads,
            quads_start: 0,
            quads_iter: self.quads.iter().peekable(),
            paths: &self.paths,
            paths_start: 0,
            paths_iter: self.paths.iter().peekable(),
            underlines: &self.underlines,
            underlines_start: 0,
            underlines_iter: self.underlines.iter().peekable(),
            monochrome_sprites: &self.monochrome_sprites,
            monochrome_sprites_start: 0,
            monochrome_sprites_iter: self.monochrome_sprites.iter().peekable(),
            polychrome_sprites: &self.polychrome_sprites,
            polychrome_sprites_start: 0,
            polychrome_sprites_iter: self.polychrome_sprites.iter().peekable(),
            surfaces: &self.surfaces,
            surfaces_start: 0,
            surfaces_iter: self.surfaces.iter().peekable(),
            backdrop_filters: &self.backdrop_filters,
            backdrop_filters_start: 0,
            backdrop_filters_iter: self.backdrop_filters.iter().peekable(),
            filter_boundaries: &self.filter_boundaries,
            filter_boundaries_start: 0,
            filter_boundaries_iter: self.filter_boundaries.iter().peekable(),
        }
    }

    /// The frame's draw stream with [`LayerSlabSpan`]s spliced in at the
    /// positions their reservations earned — one iterator covering both the
    /// legacy primitive batches and the slab references, ordered by global
    /// draw order.
    pub(crate) fn frame_batches(&self) -> FrameBatchIterator<'_> {
        FrameBatchIterator {
            queued: std::collections::VecDeque::new(),
            legacy: BatchIterator {
                shadows: &self.shadows,
                shadows_start: 0,
                shadows_iter: self.shadows.iter().peekable(),
                quads: &self.quads,
                quads_start: 0,
                quads_iter: self.quads.iter().peekable(),
                paths: &self.paths,
                paths_start: 0,
                paths_iter: self.paths.iter().peekable(),
                underlines: &self.underlines,
                underlines_start: 0,
                underlines_iter: self.underlines.iter().peekable(),
                monochrome_sprites: &self.monochrome_sprites,
                monochrome_sprites_start: 0,
                monochrome_sprites_iter: self.monochrome_sprites.iter().peekable(),
                polychrome_sprites: &self.polychrome_sprites,
                polychrome_sprites_start: 0,
                polychrome_sprites_iter: self.polychrome_sprites.iter().peekable(),
                surfaces: &self.surfaces,
                surfaces_start: 0,
                surfaces_iter: self.surfaces.iter().peekable(),
                backdrop_filters: &self.backdrop_filters,
                backdrop_filters_start: 0,
                backdrop_filters_iter: self.backdrop_filters.iter().peekable(),
                filter_boundaries: &self.filter_boundaries,
                filter_boundaries_start: 0,
                filter_boundaries_iter: self.filter_boundaries.iter().peekable(),
            },
            pending: None,
            next_span: 0,
            scene: self,
        }
    }
}

/// One entry of the merged draw stream: either a legacy primitive batch or a
/// spliced slab reference.
#[derive(Debug)]
pub(crate) enum SceneBatch<'a> {
    Primitives(PrimitiveBatch<'a>),
    LayerSlab(usize),
}

pub(crate) struct FrameBatchIterator<'a> {
    legacy: BatchIterator<'a>,
    /// Head-of-line legacy batches waiting to be yielded, plus any tails
    /// produced when a span splits a same-kind batch around its position.
    queued: std::collections::VecDeque<SceneBatch<'a>>,
    pending: Option<PrimitiveBatch<'a>>,
    next_span: usize,
    scene: &'a Scene,
}

/// Split a slice-carrying batch at the first element whose order is
/// `>= boundary`, returning `(head, tail)`. Single-element batches
/// (`FilterBoundary`) cannot straddle a span and are returned whole.
fn split_batch_at<'a>(
    batch: PrimitiveBatch<'a>,
    boundary: DrawOrder,
    filter_boundaries: &[FilterBoundary],
) -> (Option<PrimitiveBatch<'a>>, Option<PrimitiveBatch<'a>>) {
    let partition = |orders: &dyn Fn(usize) -> DrawOrder, len: usize| {
        let mut point = 0;
        while point < len && orders(point) < boundary {
            point += 1;
        }
        (point, len)
    };
    match batch {
        PrimitiveBatch::Shadows(slice) => {
            let (point, _) = partition(&|i| slice[i].order, slice.len());
            (
                Some(PrimitiveBatch::Shadows(&slice[..point])),
                Some(PrimitiveBatch::Shadows(&slice[point..])),
            )
        }
        PrimitiveBatch::Quads(slice) => {
            let (point, _) = partition(&|i| slice[i].order, slice.len());
            (
                Some(PrimitiveBatch::Quads(&slice[..point])),
                Some(PrimitiveBatch::Quads(&slice[point..])),
            )
        }
        PrimitiveBatch::Paths(slice) => {
            let (point, _) = partition(&|i| slice[i].order, slice.len());
            (
                Some(PrimitiveBatch::Paths(&slice[..point])),
                Some(PrimitiveBatch::Paths(&slice[point..])),
            )
        }
        PrimitiveBatch::Underlines(slice) => {
            let (point, _) = partition(&|i| slice[i].order, slice.len());
            (
                Some(PrimitiveBatch::Underlines(&slice[..point])),
                Some(PrimitiveBatch::Underlines(&slice[point..])),
            )
        }
        PrimitiveBatch::MonochromeSprites { texture_id, sprites } => {
            let (point, _) = partition(&|i| sprites[i].order, sprites.len());
            (
                Some(PrimitiveBatch::MonochromeSprites {
                    texture_id,
                    sprites: &sprites[..point],
                }),
                Some(PrimitiveBatch::MonochromeSprites {
                    texture_id,
                    sprites: &sprites[point..],
                }),
            )
        }
        PrimitiveBatch::PolychromeSprites { texture_id, sprites } => {
            let (point, _) = partition(&|i| sprites[i].order, sprites.len());
            (
                Some(PrimitiveBatch::PolychromeSprites {
                    texture_id,
                    sprites: &sprites[..point],
                }),
                Some(PrimitiveBatch::PolychromeSprites {
                    texture_id,
                    sprites: &sprites[point..],
                }),
            )
        }
        PrimitiveBatch::Surfaces(slice) => {
            let (point, _) = partition(&|i| slice[i].order, slice.len());
            (
                Some(PrimitiveBatch::Surfaces(&slice[..point])),
                Some(PrimitiveBatch::Surfaces(&slice[point..])),
            )
        }
        PrimitiveBatch::BackdropFilters(slice) => {
            let (point, _) = partition(&|i| slice[i].order, slice.len());
            (
                Some(PrimitiveBatch::BackdropFilters(&slice[..point])),
                Some(PrimitiveBatch::BackdropFilters(&slice[point..])),
            )
        }
        // One marker, one order: it either precedes the span or follows it.
        PrimitiveBatch::FilterBoundary(index) => {
            let order = filter_boundaries[index].order;
            if order < boundary {
                (Some(PrimitiveBatch::FilterBoundary(index)), None)
            } else {
                (None, Some(PrimitiveBatch::FilterBoundary(index)))
            }
        }
    }
}

impl<'a> FrameBatchIterator<'a> {
    fn batch_lead_order(batch: &PrimitiveBatch<'a>) -> Option<DrawOrder> {
        match batch {
            PrimitiveBatch::Shadows(s) => s.first().map(|p| p.order),
            PrimitiveBatch::Quads(q) => q.first().map(|p| p.order),
            PrimitiveBatch::Paths(p) => p.first().map(|p| p.order),
            PrimitiveBatch::Underlines(u) => u.first().map(|p| p.order),
            PrimitiveBatch::MonochromeSprites { sprites, .. } => sprites.first().map(|p| p.order),
            PrimitiveBatch::PolychromeSprites { sprites, .. } => sprites.first().map(|p| p.order),
            PrimitiveBatch::Surfaces(s) => s.first().map(|p| p.order),
            PrimitiveBatch::BackdropFilters(f) => f.first().map(|p| p.order),
            PrimitiveBatch::FilterBoundary(_) => None,
        }
    }
}

impl<'a> Iterator for FrameBatchIterator<'a> {
    type Item = SceneBatch<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(queued) = self.queued.pop_front() {
                return Some(queued);
            }
            if self.pending.is_none() {
                self.pending = self.legacy.next();
            }

            let next_span = self
                .scene
                .layer_slab_spans
                .get(self.next_span)
                .map(|span| (self.next_span, span.order));

            match (self.pending.take(), next_span) {
                (None, Some((index, _))) => {
                    self.next_span += 1;
                    return Some(SceneBatch::LayerSlab(index));
                }
                (None, None) => return None,
                (Some(batch), Some((index, span_order))) => {
                    // A span's reservation owns its own global order value, so
                    // it never ties with surrounding content. When it lands
                    // strictly inside a same-kind batch, the batch splits
                    // around it so every element draws at its own position.
                    let lead = Self::batch_lead_order(&batch);
                    if lead.is_none_or(|lead| lead >= span_order) {
                        self.next_span += 1;
                        self.pending = Some(batch);
                        return Some(SceneBatch::LayerSlab(index));
                    }
                    let (head, tail) =
                        split_batch_at(batch, span_order, &self.scene.filter_boundaries);
                    self.next_span += 1;
                    self.queued.push_back(SceneBatch::LayerSlab(index));
                    if let Some(tail) = tail {
                        self.queued.push_back(SceneBatch::Primitives(tail));
                    }
                    if let Some(head) = head {
                        return Some(SceneBatch::Primitives(head));
                    }
                }
                (Some(batch), None) => return Some(SceneBatch::Primitives(batch)),
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Default)]
pub(crate) enum PrimitiveKind {
    // Lowest discriminant: at an equal order, a content-filter group-start is emitted before
    // the group's own content so the renderer redirects rendering before any child draws.
    FilterBoundaryStart,
    Shadow,
    #[default]
    Quad,
    Path,
    Underline,
    MonochromeSprite,
    PolychromeSprite,
    Surface,
    BackdropFilter,
    // Highest discriminant: at an equal order, a group-end is emitted after the group's content
    // so the renderer composites the filtered group only once every child has been drawn.
    FilterBoundaryEnd,
}

pub(crate) enum PaintOperation {
    Primitive(Primitive),
    StartLayer(Bounds<ScaledPixels>),
    EndLayer,
    /// A spliced slab reference. Replayed by re-reserving an order slot and
    /// re-registering the same marker — the packed bytes ride along behind an
    /// `Arc`, so replay costs one reference count, not a re-pack.
    LayerSlab(LayerSlabSpan),
}

/// One instanced draw the renderer issues from a layer's slab, with `start`
/// already offset to point into the layer's concatenated per-kind stream (all
/// of a layer's stretches share its slab ranges).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SlabRun {
    pub kind: SlabKind,
    pub start: u32,
    pub count: u32,
    pub texture_id: Option<AtlasTextureId>,
}

/// A splice marker plus everything the renderer needs to draw it without
/// consulting the window: identity, content token, transform origin,
/// allocation totals, per-kind runs, and the packed bytes themselves.
///
/// Cloning is cheap on purpose (`Arc` over the pack): paint operations replay
/// these verbatim when a recorded index range is reused.
#[derive(Clone)]
pub(crate) struct LayerSlabSpan {
    pub key: LayerKey,
    pub content_token: u64,
    /// The layer's composited origin in scaled pixels; the renderer uploads
    /// this into the layer's 64-byte transform slot and the shaders translate
    /// by it in the vertex stage.
    pub origin: [f32; 2],
    /// Per-kind instance totals across ALL spans of this layer, so whichever
    /// span arrives first can size the layer's reservations.
    pub totals: [u32; SlabKind::COUNT],
    /// This stretch's draws, `start` offset into the layer-wide streams.
    pub runs: Vec<SlabRun>,
    /// This stretch's packed arrays — the upload source of truth.
    pub packed: Arc<PackedLayer>,
    /// Set when this span renders into a texture-retained layer's persistent
    /// texture (#96) instead of the framebuffer. The packed coordinates are
    /// texture-relative (the pack was built at the texture origin), so the
    /// renderer draws them with a zero transform translate through a viewport
    /// remapped onto the texture.
    pub texture: Option<LayerTextureTarget>,
    reservation_bounds: Bounds<ScaledPixels>,
    order_scope: usize,
    local_order: DrawOrder,
    /// Resolved global draw order, filled in during [`Scene::finish`].
    order: DrawOrder,
}

impl LayerSlabSpan {
    pub fn order(&self) -> DrawOrder {
        self.order
    }
}

#[derive(Clone)]
pub(crate) enum Primitive {
    Shadow(Shadow),
    Quad(Quad),
    Path(Path<ScaledPixels>),
    Underline(Underline),
    MonochromeSprite(MonochromeSprite),
    PolychromeSprite(PolychromeSprite),
    Surface(PaintSurface),
    BackdropFilter(BackdropFilter),
    FilterBoundary(FilterBoundary),
}

impl Primitive {
    pub fn bounds(&self) -> &Bounds<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.bounds,
            Primitive::Quad(quad) => &quad.bounds,
            Primitive::Path(path) => &path.bounds,
            Primitive::Underline(underline) => &underline.bounds,
            Primitive::MonochromeSprite(sprite) => &sprite.bounds,
            Primitive::PolychromeSprite(sprite) => &sprite.bounds,
            Primitive::Surface(surface) => &surface.bounds,
            Primitive::BackdropFilter(filter) => &filter.bounds,
            Primitive::FilterBoundary(boundary) => &boundary.bounds,
        }
    }

    pub fn content_mask(&self) -> &ContentMask<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.content_mask,
            Primitive::Quad(quad) => &quad.content_mask,
            Primitive::Path(path) => &path.content_mask,
            Primitive::Underline(underline) => &underline.content_mask,
            Primitive::MonochromeSprite(sprite) => &sprite.content_mask,
            Primitive::PolychromeSprite(sprite) => &sprite.content_mask,
            Primitive::Surface(surface) => &surface.content_mask,
            Primitive::BackdropFilter(filter) => &filter.content_mask,
            Primitive::FilterBoundary(boundary) => &boundary.content_mask,
        }
    }
}

fn primitive_order(primitive: &Primitive) -> DrawOrder {
    match primitive {
        Primitive::Shadow(shadow) => shadow.order,
        Primitive::Quad(quad) => quad.order,
        Primitive::Path(path) => path.order,
        Primitive::Underline(underline) => underline.order,
        Primitive::MonochromeSprite(sprite) => sprite.order,
        Primitive::PolychromeSprite(sprite) => sprite.order,
        Primitive::Surface(surface) => surface.order,
        Primitive::BackdropFilter(filter) => filter.order,
        Primitive::FilterBoundary(boundary) => boundary.order,
    }
}

fn set_primitive_order(primitive: &mut Primitive, order: DrawOrder) {
    match primitive {
        Primitive::Shadow(shadow) => shadow.order = order,
        Primitive::Quad(quad) => quad.order = order,
        Primitive::Path(path) => path.order = order,
        Primitive::Underline(underline) => underline.order = order,
        Primitive::MonochromeSprite(sprite) => sprite.order = order,
        Primitive::PolychromeSprite(sprite) => sprite.order = order,
        Primitive::Surface(surface) => surface.order = order,
        Primitive::BackdropFilter(filter) => filter.order = order,
        Primitive::FilterBoundary(boundary) => boundary.order = order,
    }
}

fn count_primitive(primitive: &Primitive) {
    match primitive {
        Primitive::Shadow(_) => crate::render_stats::count("frame: primitives emitted (shadow)"),
        Primitive::Quad(_) => crate::render_stats::count("frame: primitives emitted (quad)"),
        Primitive::Path(_) => crate::render_stats::count("frame: primitives emitted (path)"),
        Primitive::Underline(_) => {
            crate::render_stats::count("frame: primitives emitted (underline)")
        }
        Primitive::MonochromeSprite(_) | Primitive::PolychromeSprite(_) => {
            crate::render_stats::count("frame: primitives emitted (sprite)")
        }
        Primitive::Surface(_) => crate::render_stats::count("frame: primitives emitted (surface)"),
        Primitive::BackdropFilter(_) | Primitive::FilterBoundary(_) => {}
    }
}

struct BatchIterator<'a> {
    shadows: &'a [Shadow],
    shadows_start: usize,
    shadows_iter: Peekable<slice::Iter<'a, Shadow>>,
    quads: &'a [Quad],
    quads_start: usize,
    quads_iter: Peekable<slice::Iter<'a, Quad>>,
    paths: &'a [Path<ScaledPixels>],
    paths_start: usize,
    paths_iter: Peekable<slice::Iter<'a, Path<ScaledPixels>>>,
    underlines: &'a [Underline],
    underlines_start: usize,
    underlines_iter: Peekable<slice::Iter<'a, Underline>>,
    monochrome_sprites: &'a [MonochromeSprite],
    monochrome_sprites_start: usize,
    monochrome_sprites_iter: Peekable<slice::Iter<'a, MonochromeSprite>>,
    polychrome_sprites: &'a [PolychromeSprite],
    polychrome_sprites_start: usize,
    polychrome_sprites_iter: Peekable<slice::Iter<'a, PolychromeSprite>>,
    surfaces: &'a [PaintSurface],
    surfaces_start: usize,
    surfaces_iter: Peekable<slice::Iter<'a, PaintSurface>>,
    backdrop_filters: &'a [BackdropFilter],
    backdrop_filters_start: usize,
    backdrop_filters_iter: Peekable<slice::Iter<'a, BackdropFilter>>,
    filter_boundaries: &'a [FilterBoundary],
    filter_boundaries_start: usize,
    filter_boundaries_iter: Peekable<slice::Iter<'a, FilterBoundary>>,
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = PrimitiveBatch<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut orders_and_kinds = [
            (
                self.shadows_iter.peek().map(|s| s.order),
                PrimitiveKind::Shadow,
            ),
            (self.quads_iter.peek().map(|q| q.order), PrimitiveKind::Quad),
            (self.paths_iter.peek().map(|q| q.order), PrimitiveKind::Path),
            (
                self.underlines_iter.peek().map(|u| u.order),
                PrimitiveKind::Underline,
            ),
            (
                self.monochrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::MonochromeSprite,
            ),
            (
                self.polychrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::PolychromeSprite,
            ),
            (
                self.surfaces_iter.peek().map(|s| s.order),
                PrimitiveKind::Surface,
            ),
            (
                self.backdrop_filters_iter.peek().map(|f| f.order),
                PrimitiveKind::BackdropFilter,
            ),
            (
                self.filter_boundaries_iter.peek().map(|b| b.order),
                // The same vec yields both start and end markers; the discriminant decides
                // where the next marker sorts relative to draw batches at an equal order
                // (start before content, end after).
                match self.filter_boundaries_iter.peek() {
                    Some(boundary) if boundary.is_start => PrimitiveKind::FilterBoundaryStart,
                    _ => PrimitiveKind::FilterBoundaryEnd,
                },
            ),
        ];
        orders_and_kinds.sort_by_key(|(order, kind)| (order.unwrap_or(u32::MAX), *kind));

        let first = orders_and_kinds[0];
        let second = orders_and_kinds[1];
        let (batch_kind, max_order_and_kind) = if first.0.is_some() {
            (first.1, (second.0.unwrap_or(u32::MAX), second.1))
        } else {
            return None;
        };

        match batch_kind {
            PrimitiveKind::Shadow => {
                let shadows_start = self.shadows_start;
                let mut shadows_end = shadows_start + 1;
                self.shadows_iter.next();
                while self
                    .shadows_iter
                    .next_if(|shadow| (shadow.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    shadows_end += 1;
                }
                self.shadows_start = shadows_end;
                Some(PrimitiveBatch::Shadows(
                    &self.shadows[shadows_start..shadows_end],
                ))
            }
            PrimitiveKind::Quad => {
                let quads_start = self.quads_start;
                let mut quads_end = quads_start + 1;
                self.quads_iter.next();
                while self
                    .quads_iter
                    .next_if(|quad| (quad.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    quads_end += 1;
                }
                self.quads_start = quads_end;
                Some(PrimitiveBatch::Quads(&self.quads[quads_start..quads_end]))
            }
            PrimitiveKind::Path => {
                let paths_start = self.paths_start;
                let mut paths_end = paths_start + 1;
                self.paths_iter.next();
                while self
                    .paths_iter
                    .next_if(|path| (path.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    paths_end += 1;
                }
                self.paths_start = paths_end;
                Some(PrimitiveBatch::Paths(&self.paths[paths_start..paths_end]))
            }
            PrimitiveKind::Underline => {
                let underlines_start = self.underlines_start;
                let mut underlines_end = underlines_start + 1;
                self.underlines_iter.next();
                while self
                    .underlines_iter
                    .next_if(|underline| (underline.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    underlines_end += 1;
                }
                self.underlines_start = underlines_end;
                Some(PrimitiveBatch::Underlines(
                    &self.underlines[underlines_start..underlines_end],
                ))
            }
            PrimitiveKind::MonochromeSprite => {
                let texture_id = self.monochrome_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.monochrome_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.monochrome_sprites_iter.next();
                while self
                    .monochrome_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.monochrome_sprites_start = sprites_end;
                Some(PrimitiveBatch::MonochromeSprites {
                    texture_id,
                    sprites: &self.monochrome_sprites[sprites_start..sprites_end],
                })
            }
            PrimitiveKind::PolychromeSprite => {
                let texture_id = self.polychrome_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.polychrome_sprites_start;
                let mut sprites_end = self.polychrome_sprites_start + 1;
                self.polychrome_sprites_iter.next();
                while self
                    .polychrome_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.polychrome_sprites_start = sprites_end;
                Some(PrimitiveBatch::PolychromeSprites {
                    texture_id,
                    sprites: &self.polychrome_sprites[sprites_start..sprites_end],
                })
            }
            PrimitiveKind::Surface => {
                let surfaces_start = self.surfaces_start;
                let mut surfaces_end = surfaces_start + 1;
                self.surfaces_iter.next();
                while self
                    .surfaces_iter
                    .next_if(|surface| (surface.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    surfaces_end += 1;
                }
                self.surfaces_start = surfaces_end;
                Some(PrimitiveBatch::Surfaces(
                    &self.surfaces[surfaces_start..surfaces_end],
                ))
            }
            PrimitiveKind::BackdropFilter => {
                let backdrop_filters_start = self.backdrop_filters_start;
                let mut backdrop_filters_end = backdrop_filters_start + 1;
                self.backdrop_filters_iter.next();
                while self
                    .backdrop_filters_iter
                    .next_if(|filter| (filter.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    backdrop_filters_end += 1;
                }
                self.backdrop_filters_start = backdrop_filters_end;
                Some(PrimitiveBatch::BackdropFilters(
                    &self.backdrop_filters[backdrop_filters_start..backdrop_filters_end],
                ))
            }
            // Boundaries are emitted one at a time (never merged) so the renderer can switch
            // render targets at exactly the right point in the batch stream.
            PrimitiveKind::FilterBoundaryStart | PrimitiveKind::FilterBoundaryEnd => {
                let index = self.filter_boundaries_start;
                self.filter_boundaries_iter.next();
                self.filter_boundaries_start = index + 1;
                Some(PrimitiveBatch::FilterBoundary(index))
            }
        }
    }
}

#[derive(Debug)]
pub(crate) enum PrimitiveBatch<'a> {
    Shadows(&'a [Shadow]),
    Quads(&'a [Quad]),
    Paths(&'a [Path<ScaledPixels>]),
    Underlines(&'a [Underline]),
    MonochromeSprites {
        texture_id: AtlasTextureId,
        sprites: &'a [MonochromeSprite],
    },
    PolychromeSprites {
        texture_id: AtlasTextureId,
        sprites: &'a [PolychromeSprite],
    },
    Surfaces(&'a [PaintSurface]),
    BackdropFilters(&'a [BackdropFilter]),
    /// A single content-filter group boundary; index into [`Scene::filter_boundaries`]. Read
    /// `is_start` to tell whether this opens the group (switch render target) or closes it
    /// (filter the offscreen target and composite it back).
    FilterBoundary(usize),
}

#[derive(Default, Debug, Clone, Copy, bytemuck::NoUninit)]
#[repr(C)]
pub(crate) struct Quad {
    pub order: DrawOrder,
    pub border_style: BorderStyle,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub background: Background,
    pub border_color: Hsla,
    pub corner_radii: Corners<ScaledPixels>,
    pub border_widths: Edges<ScaledPixels>,
}

// Stride expected by `array<Quad>` in quads.wgsl's storage buffer.
const _: () = assert!(std::mem::size_of::<Quad>() == 168);
const _: () = assert!(std::mem::offset_of!(Quad, background) == 40);

impl From<Quad> for Primitive {
    fn from(quad: Quad) -> Self {
        Primitive::Quad(quad)
    }
}

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub(crate) struct Underline {
    pub order: DrawOrder,
    pub pad: u32, // align to 8 bytes
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub thickness: ScaledPixels,
    pub wavy: u32,
}

// Stride expected by `array<Underline>` in underlines.wgsl's storage buffer.
const _: () = assert!(std::mem::size_of::<Underline>() == 64);

impl From<Underline> for Primitive {
    fn from(underline: Underline) -> Self {
        Primitive::Underline(underline)
    }
}

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub(crate) struct Shadow {
    pub order: DrawOrder,
    pub blur_radius: ScaledPixels,
    pub bounds: Bounds<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
}

// Stride expected by `array<Shadow>` in shadows.wgsl's storage buffer.
const _: () = assert!(std::mem::size_of::<Shadow>() == 72);

impl From<Shadow> for Primitive {
    fn from(shadow: Shadow) -> Self {
        Primitive::Shadow(shadow)
    }
}

/// A backdrop filter blurs (and may otherwise filter) the content already rendered behind
/// `bounds`, compositing the result into a rounded rectangle — the frosted-glass effect.
/// Emitted by [`crate::Window::paint_backdrop_filter`]; produces the CSS `backdrop-filter` effect.
#[derive(Default, Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub(crate) struct BackdropFilter {
    pub order: DrawOrder,
    /// The largest blur radius among the element's backdrop filters, in scaled (device) pixels.
    ///
    /// Placed directly after `order` (rather than next to the other filter parameters) so the
    /// four bytes naturally pad the struct to an 8-byte boundary before `bounds`: WGSL aligns
    /// `Bounds` (built from `vec2<f32>`s) to 8 bytes in the storage buffer, and without an
    /// explicit 4-byte field here the compiler-inserted padding would desync this `#[repr(C)]`
    /// struct's layout from the shader's `BackdropFilter` struct, corrupting every field read on
    /// the GPU.
    pub blur_radius: ScaledPixels,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    /// Element opacity captured at paint time, multiplied into the composited result.
    pub opacity: f32,
    /// Pads the struct to 64 bytes (a multiple of 8) to match the storage-buffer stride WGSL
    /// derives for `array<BackdropFilter>` — see the note on `blur_radius` above.
    pub _pad: u32,
}

// Stride expected by `array<BackdropFilter>` in backdrop_blur.wgsl's storage buffer.
const _: () = assert!(std::mem::size_of::<BackdropFilter>() == 64);

impl From<BackdropFilter> for Primitive {
    fn from(filter: BackdropFilter) -> Self {
        Primitive::BackdropFilter(filter)
    }
}

/// The start or end marker of a content-filter (`filter`) isolation group. The element's
/// subtree is painted between a matched start/end pair; the renderer redirects that span into
/// an offscreen target, filters it, and composites it back at `bounds`. Produces the CSS
/// `filter` effect (e.g. blurring the element and its children as a single group).
#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub(crate) struct FilterBoundary {
    pub order: DrawOrder,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub blur_radius: ScaledPixels,
    pub opacity: f32,
    /// `true` for the start marker (opens the group), `false` for the end marker (closes it).
    pub is_start: bool,
}

impl From<FilterBoundary> for Primitive {
    fn from(boundary: FilterBoundary) -> Self {
        Primitive::FilterBoundary(boundary)
    }
}

/// The style of a border.
#[derive(
    Default,
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    Deserialize,
    JsonSchema,
    bytemuck::NoUninit,
)]
// Fixed integer repr: `Quad` is `bytemuck::NoUninit`, and bytemuck only
// accepts fieldless enums with an explicit integer representation (not
// `repr(C)`, whose size is platform-chosen).
#[repr(u32)]
pub enum BorderStyle {
    /// A solid border.
    #[default]
    Solid = 0,
    /// A dashed border.
    Dashed = 1,
}

/// A data type representing a 2 dimensional transformation that can be applied to an element.
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct TransformationMatrix {
    /// 2x2 matrix containing rotation and scale,
    /// stored row-major
    pub rotation_scale: [[f32; 2]; 2],
    /// translation vector
    pub translation: [f32; 2],
}

impl Eq for TransformationMatrix {}

impl TransformationMatrix {
    /// The unit matrix, has no effect.
    pub fn unit() -> Self {
        Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [0.0, 0.0],
        }
    }

    /// Move the origin by a given point
    pub fn translate(mut self, point: Point<ScaledPixels>) -> Self {
        self.compose(Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [point.x.0, point.y.0],
        })
    }

    /// Clockwise rotation in radians around the origin
    pub fn rotate(self, angle: Radians) -> Self {
        self.compose(Self {
            rotation_scale: [
                [angle.0.cos(), -angle.0.sin()],
                [angle.0.sin(), angle.0.cos()],
            ],
            translation: [0.0, 0.0],
        })
    }

    /// Scale around the origin
    pub fn scale(self, size: Size<f32>) -> Self {
        self.compose(Self {
            rotation_scale: [[size.width, 0.0], [0.0, size.height]],
            translation: [0.0, 0.0],
        })
    }

    /// Perform matrix multiplication with another transformation
    /// to produce a new transformation that is the result of
    /// applying both transformations: first, `other`, then `self`.
    #[inline]
    pub fn compose(self, other: TransformationMatrix) -> TransformationMatrix {
        if other == Self::unit() {
            return self;
        }
        // Perform matrix multiplication
        TransformationMatrix {
            rotation_scale: [
                [
                    self.rotation_scale[0][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][0],
                    self.rotation_scale[0][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][1],
                ],
                [
                    self.rotation_scale[1][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][0],
                    self.rotation_scale[1][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][1],
                ],
            ],
            translation: [
                self.translation[0]
                    + self.rotation_scale[0][0] * other.translation[0]
                    + self.rotation_scale[0][1] * other.translation[1],
                self.translation[1]
                    + self.rotation_scale[1][0] * other.translation[0]
                    + self.rotation_scale[1][1] * other.translation[1],
            ],
        }
    }

    /// Apply transformation to a point, mainly useful for debugging
    pub fn apply(&self, point: Point<Pixels>) -> Point<Pixels> {
        let input = [point.x.0, point.y.0];
        let mut output = self.translation;
        for (i, output_cell) in output.iter_mut().enumerate() {
            for (k, input_cell) in input.iter().enumerate() {
                *output_cell += self.rotation_scale[i][k] * *input_cell;
            }
        }
        Point::new(output[0].into(), output[1].into())
    }
}

impl Default for TransformationMatrix {
    fn default() -> Self {
        Self::unit()
    }
}

#[derive(Clone, Debug, Copy, bytemuck::NoUninit)]
#[repr(C)]
pub(crate) struct MonochromeSprite {
    pub order: DrawOrder,
    pub pad: u32, // align to 8 bytes
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub text_color: TextColor,
    pub tile: AtlasTile,
    pub transformation: TransformationMatrix,
}

// Stride expected by `array<MonochromeSprite>` in mono_sprites.wgsl's storage buffer.
const _: () = assert!(std::mem::size_of::<MonochromeSprite>() == 168);
const _: () = assert!(std::mem::offset_of!(MonochromeSprite, tile) == 112);
const _: () = assert!(std::mem::offset_of!(MonochromeSprite, transformation) == 144);

impl From<MonochromeSprite> for Primitive {
    fn from(sprite: MonochromeSprite) -> Self {
        Primitive::MonochromeSprite(sprite)
    }
}

#[derive(Clone, Debug, Copy, bytemuck::NoUninit)]
#[repr(C)]
pub(crate) struct PolychromeSprite {
    pub order: DrawOrder,
    pub pad: u32, // align to 8 bytes
    /// Stored as `u32` because the WGSL struct reads it as a `u32` at this
    /// offset (and masks the low byte, `grayscale & 0xFFu`); as a `bool` it
    /// would leave three padding bytes here, which bytemuck rejects for
    /// `NoUninit` types. Layout is unchanged.
    pub grayscale: u32,
    pub opacity: f32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub tile: AtlasTile,
}

// Stride expected by `array<PolychromeSprite>` in poly_sprites.wgsl's storage buffer.
const _: () = assert!(std::mem::size_of::<PolychromeSprite>() == 96);
const _: () = assert!(std::mem::offset_of!(PolychromeSprite, tile) == 64);

impl From<PolychromeSprite> for Primitive {
    fn from(sprite: PolychromeSprite) -> Self {
        Primitive::PolychromeSprite(sprite)
    }
}

/// The backing content for a painted surface.
#[derive(Clone, Debug)]
pub(crate) enum SurfaceContent {
    /// A WGPU surface managed by the SurfaceRegistry.
    Wgpu(SurfaceId),
    /// A texture-retained layer's persistent texture (#96). The renderer
    /// samples it from its layer-texture cache; the surface's bounds select
    /// the sub-rect (the texture covers the layer's buffer extent) and the
    /// content mask clips to the layer's visible rect.
    Layer(LayerId),
}

/// Renderer-side target carried by a texture-retained layer's slab spans
/// (#96): the spans redirect into this layer's persistent texture instead of
/// the framebuffer, and the following `PaintSurface` composites the result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LayerTextureTarget {
    pub layer_id: LayerId,
    /// The owning layer, so the renderer can request a re-record if it has to
    /// drop the texture (resize, eviction).
    pub key: LayerKey,
    /// The content generation baked into the texture; matches the span's own
    /// token, carried again here so the cache entry is self-describing.
    pub content_token: u64,
    /// The buffer extent in scaled window pixels. The texture is created at
    /// exactly this size and the composite surface samples it across these
    /// bounds.
    pub texture_bounds: Bounds<ScaledPixels>,
}

#[derive(Clone, Debug)]
pub(crate) struct PaintSurface {
    pub order: DrawOrder,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub content: SurfaceContent,
}

impl From<PaintSurface> for Primitive {
    fn from(surface: PaintSurface) -> Self {
        Primitive::Surface(surface)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PathId(pub(crate) usize);

/// A line made up of a series of vertices and control points.
#[derive(Clone, Debug)]
pub struct Path<P: Clone + Debug + Default + PartialEq> {
    pub(crate) id: PathId,
    pub(crate) order: DrawOrder,
    pub(crate) bounds: Bounds<P>,
    pub(crate) content_mask: ContentMask<P>,
    pub(crate) vertices: Vec<PathVertex<P>>,
    pub(crate) color: Background,
    start: Point<P>,
    current: Point<P>,
    contour_count: usize,
}

impl Path<Pixels> {
    /// Create a new path with the given starting point.
    pub fn new(start: Point<Pixels>) -> Self {
        Self {
            id: PathId(0),
            order: DrawOrder::default(),
            vertices: Vec::new(),
            start,
            current: start,
            bounds: Bounds {
                origin: start,
                size: Default::default(),
            },
            content_mask: Default::default(),
            color: Default::default(),
            contour_count: 0,
        }
    }

    /// Scale this path by the given factor.
    pub fn scale(&self, factor: f32) -> Path<ScaledPixels> {
        Path {
            id: self.id,
            order: self.order,
            bounds: self.bounds.scale(factor),
            content_mask: self.content_mask.scale(factor),
            vertices: self
                .vertices
                .iter()
                .map(|vertex| vertex.scale(factor))
                .collect(),
            start: self.start.map(|start| start.scale(factor)),
            current: self.current.scale(factor),
            contour_count: self.contour_count,
            color: self.color,
        }
    }

    /// Move the start, current point to the given point.
    pub fn move_to(&mut self, to: Point<Pixels>) {
        self.contour_count += 1;
        self.start = to;
        self.current = to;
    }

    /// Draw a straight line from the current point to the given point.
    pub fn line_to(&mut self, to: Point<Pixels>) {
        self.contour_count += 1;
        if self.contour_count > 1 {
            self.push_triangle(
                (self.start, self.current, to),
                (point(0., 1.), point(0., 1.), point(0., 1.)),
            );
        }
        self.current = to;
    }

    /// Draw a curve from the current point to the given point, using the given control point.
    pub fn curve_to(&mut self, to: Point<Pixels>, ctrl: Point<Pixels>) {
        self.contour_count += 1;
        if self.contour_count > 1 {
            self.push_triangle(
                (self.start, self.current, to),
                (point(0., 1.), point(0., 1.), point(0., 1.)),
            );
        }

        self.push_triangle(
            (self.current, ctrl, to),
            (point(0., 0.), point(0.5, 0.), point(1., 1.)),
        );
        self.current = to;
    }

    /// Push a triangle to the Path.
    pub fn push_triangle(
        &mut self,
        xy: (Point<Pixels>, Point<Pixels>, Point<Pixels>),
        st: (Point<f32>, Point<f32>, Point<f32>),
    ) {
        self.bounds = self
            .bounds
            .union(&Bounds {
                origin: xy.0,
                size: Default::default(),
            })
            .union(&Bounds {
                origin: xy.1,
                size: Default::default(),
            })
            .union(&Bounds {
                origin: xy.2,
                size: Default::default(),
            });

        self.vertices.push(PathVertex {
            xy_position: xy.0,
            st_position: st.0,
            content_mask: Default::default(),
        });
        self.vertices.push(PathVertex {
            xy_position: xy.1,
            st_position: st.1,
            content_mask: Default::default(),
        });
        self.vertices.push(PathVertex {
            xy_position: xy.2,
            st_position: st.2,
            content_mask: Default::default(),
        });
    }
}

impl<T> Path<T>
where
    T: Clone + Debug + Default + PartialEq + PartialOrd + Add<T, Output = T> + Sub<Output = T>,
{
    #[allow(unused)]
    pub(crate) fn clipped_bounds(&self) -> Bounds<T> {
        self.bounds.intersect(&self.content_mask.bounds)
    }
}

impl From<Path<ScaledPixels>> for Primitive {
    fn from(path: Path<ScaledPixels>) -> Self {
        Primitive::Path(path)
    }
}

#[derive(Clone, Debug)]
#[repr(C)]
pub(crate) struct PathVertex<P: Clone + Debug + Default + PartialEq> {
    pub(crate) xy_position: Point<P>,
    pub(crate) st_position: Point<f32>,
    pub(crate) content_mask: ContentMask<P>,
}

impl PathVertex<Pixels> {
    pub fn scale(&self, factor: f32) -> PathVertex<ScaledPixels> {
        PathVertex {
            xy_position: self.xy_position.scale(factor),
            st_position: self.st_position,
            content_mask: self.content_mask.scale(factor),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Point, Size};

    fn sp(value: f32) -> ScaledPixels {
        ScaledPixels(value)
    }

    /// All test primitives cover the same region so the bounds tree assigns strictly
    /// increasing orders in insertion order — making the expected batch order deterministic.
    fn full_bounds() -> Bounds<ScaledPixels> {
        Bounds {
            origin: Point {
                x: sp(0.0),
                y: sp(0.0),
            },
            size: Size {
                width: sp(100.0),
                height: sp(100.0),
            },
        }
    }

    fn mask() -> ContentMask<ScaledPixels> {
        ContentMask {
            bounds: full_bounds(),
        }
    }

    fn quad() -> Quad {
        Quad {
            bounds: full_bounds(),
            content_mask: mask(),
            ..Default::default()
        }
    }

    fn boundary(is_start: bool) -> FilterBoundary {
        FilterBoundary {
            order: 0,
            bounds: full_bounds(),
            content_mask: mask(),
            corner_radii: Corners::default(),
            blur_radius: sp(8.0),
            opacity: 1.0,
            is_start,
        }
    }

    fn backdrop() -> BackdropFilter {
        BackdropFilter {
            bounds: full_bounds(),
            content_mask: mask(),
            corner_radii: Corners::default(),
            blur_radius: sp(20.0),
            opacity: 1.0,
            ..Default::default()
        }
    }

    fn batch_kinds(scene: &mut Scene) -> Vec<&'static str> {
        scene.finish();
        scene
            .batches()
            .map(|batch| match batch {
                PrimitiveBatch::Quads(_) => "quad",
                PrimitiveBatch::BackdropFilters(_) => "backdrop",
                PrimitiveBatch::FilterBoundary(ix) => {
                    if scene.filter_boundaries[ix].is_start {
                        "start"
                    } else {
                        "end"
                    }
                }
                _ => "other",
            })
            .collect()
    }

    #[test]
    fn content_filter_group_brackets_its_children() {
        let mut scene = Scene::default();
        // Background painted before the filtered element.
        scene.insert_primitive(quad());
        // A content-filtered element: start marker, its child, end marker.
        scene.insert_primitive(boundary(true));
        scene.insert_primitive(quad());
        scene.insert_primitive(boundary(false));

        // The start must precede the group's child and the end must follow it, so the
        // renderer can redirect rendering for exactly the group's span.
        assert_eq!(
            batch_kinds(&mut scene),
            vec!["quad", "start", "quad", "end"]
        );
    }

    // Note: this validates only the *scene ordering* of nested filter boundaries (start/child/
    // end interleaving), not that a renderer actually isolates both levels — that depends on the
    // backend's group-texture pool (see MAX_FILTER_DEPTH) and is exercised by the `blur` example.
    #[test]
    fn nested_content_filters_emit_well_nested_ordering() {
        let mut scene = Scene::default();
        scene.insert_primitive(boundary(true)); // outer start
        scene.insert_primitive(quad()); // outer child
        scene.insert_primitive(boundary(true)); // inner start
        scene.insert_primitive(quad()); // inner child
        scene.insert_primitive(boundary(false)); // inner end
        scene.insert_primitive(boundary(false)); // outer end

        assert_eq!(
            batch_kinds(&mut scene),
            vec!["start", "quad", "start", "quad", "end", "end"]
        );
    }

    #[test]
    fn backdrop_filter_sorts_before_a_later_overlapping_quad() {
        let mut scene = Scene::default();
        // Content behind the frosted panel.
        scene.insert_primitive(quad());
        // The panel: its backdrop snapshot, then its (translucent) background quad on top.
        scene.insert_primitive(backdrop());
        scene.insert_primitive(quad());

        assert_eq!(batch_kinds(&mut scene), vec!["quad", "backdrop", "quad"]);
    }

    // ---------------------------------------------------------------------
    // Ordering equivalence: layer-local ordering against the global tree.
    //
    // This is the correctness gate for the retained-layer phase. Layer-local
    // ordering is the thing everything downstream — per-layer slabs, occlusion,
    // transform-only scrolling — is built on; get it wrong here and every later
    // phase inherits it, invisibly.
    //
    // What is asserted is *relative* order between overlapping primitives, not
    // equality of the order integers. Those cannot match and should not: the
    // global tree reuses low orders for non-overlapping content, and a layer
    // deliberately takes an order above everything painted before it so that
    // content escaping its bounds still sorts correctly. Two primitives that do
    // not overlap have no visual relationship, so any relative order is
    // equally correct for them. Two that *do* overlap have exactly one correct
    // answer, and it must be the same answer the global tree gives.
    // ---------------------------------------------------------------------

    fn rect(x: f32, y: f32, w: f32, h: f32) -> Bounds<ScaledPixels> {
        Bounds {
            origin: Point { x: sp(x), y: sp(y) },
            size: Size {
                width: sp(w),
                height: sp(h),
            },
        }
    }

    fn quad_at(bounds: Bounds<ScaledPixels>) -> Quad {
        Quad {
            bounds,
            content_mask: ContentMask {
                bounds: rect(-1000., -1000., 10_000., 10_000.),
            },
            ..Default::default()
        }
    }

    /// One step of a reference scene, replayed through both ordering schemes.
    #[derive(Clone, Copy)]
    enum Step {
        Quad(Bounds<ScaledPixels>),
        /// Content-filter group markers, which take an order above all prior
        /// content and must keep their contents strictly between them.
        FilterStart(Bounds<ScaledPixels>),
        FilterEnd(Bounds<ScaledPixels>),
        /// A retained layer boundary. Ignored by the reference, which has only
        /// the one global tree — which is exactly the equivalence being tested.
        BeginLayer(LayerKey, Bounds<ScaledPixels>),
        EndLayer,
        /// A clip group: everything inside collapses to one order.
        PushClip(Bounds<ScaledPixels>),
        PopClip,
        /// What `paint_deferred_draws` does before painting overlays.
        RaiseFloor,
    }

    /// The reference scene, exercising every ordering mechanism at once:
    /// overlapping and non-overlapping content, layers nested two deep, a layer
    /// whose content escapes its declared bounds, a filter group containing a
    /// layer, a clip group, and deferred draws hoisted above everything.
    fn reference_scene() -> Vec<Step> {
        let panel = rect(0., 0., 200., 400.);
        let inner = rect(10., 10., 180., 100.);
        vec![
            // Background grid: non-overlapping, so the global tree reuses low
            // orders for it. This is the content a badly-chosen layer order
            // range would sweep into a filter group.
            Step::Quad(rect(0., 500., 50., 50.)),
            Step::Quad(rect(100., 500., 50., 50.)),
            Step::Quad(rect(200., 500., 50., 50.)),
            // Main content behind the panel.
            Step::Quad(rect(0., 0., 800., 600.)),
            // A retained layer, with a layer nested inside it.
            Step::BeginLayer(LayerKey(1), panel),
            Step::Quad(panel),
            Step::BeginLayer(LayerKey(2), inner),
            Step::Quad(inner),
            Step::Quad(rect(20., 20., 60., 60.)),
            Step::EndLayer,
            // Overlaps the nested layer, painted after it.
            Step::Quad(rect(10., 60., 180., 60.)),
            // Escapes the layer's declared bounds — the case `end_scope`'s
            // re-insert exists for.
            Step::Quad(rect(150., 380., 300., 60.)),
            Step::EndLayer,
            // Painted after the layer and overlapping only its overflow.
            Step::Quad(rect(300., 390., 100., 40.)),
            // A filter group containing a layer.
            Step::FilterStart(rect(400., 0., 200., 200.)),
            Step::Quad(rect(400., 0., 200., 200.)),
            Step::BeginLayer(LayerKey(3), rect(420., 20., 160., 160.)),
            Step::Quad(rect(420., 20., 160., 160.)),
            Step::EndLayer,
            Step::FilterEnd(rect(400., 0., 200., 200.)),
            // A clip group, collapsing its contents to one order.
            Step::PushClip(rect(0., 200., 300., 100.)),
            Step::Quad(rect(0., 200., 150., 100.)),
            Step::Quad(rect(100., 200., 150., 100.)),
            Step::PopClip,
            // Deferred draws: overlays hoisted above the whole main scene.
            Step::RaiseFloor,
            Step::Quad(rect(50., 50., 100., 100.)),
            Step::BeginLayer(LayerKey(4), rect(60., 60., 80., 80.)),
            Step::Quad(rect(60., 60., 80., 80.)),
            Step::EndLayer,
        ]
    }

    /// Replay `steps` through `Scene`, returning each quad's final z-position,
    /// indexed by the order the quads were painted in.
    fn layered_ranks(steps: &[Step]) -> Vec<usize> {
        let mut scene = Scene::default();
        for step in steps {
            match *step {
                Step::Quad(bounds) => scene.insert_primitive(quad_at(bounds)),
                Step::FilterStart(bounds) => scene.insert_primitive(FilterBoundary {
                    bounds,
                    is_start: true,
                    ..boundary(true)
                }),
                Step::FilterEnd(bounds) => scene.insert_primitive(FilterBoundary {
                    bounds,
                    is_start: false,
                    ..boundary(false)
                }),
                Step::BeginLayer(key, bounds) => scene.begin_layer(key, bounds, true),
                Step::EndLayer => {
                    scene.end_layer();
                }
                Step::PushClip(bounds) => scene.push_layer(bounds),
                Step::PopClip => scene.pop_layer(),
                Step::RaiseFloor => scene.raise_order_floor(),
            }
        }
        // Tag each quad with its paint index before sorting, so the sorted
        // position can be mapped back. `corner_radii.top_left` is unused by
        // ordering and survives the sort untouched.
        for (index, quad) in scene.quads.iter_mut().enumerate() {
            quad.corner_radii.top_left = sp(index as f32);
        }
        scene.finish();

        let mut ranks = vec![0; scene.quads.len()];
        for (rank, quad) in scene.quads.iter().enumerate() {
            ranks[quad.corner_radii.top_left.0 as usize] = rank;
        }
        ranks
    }

    /// Replay `steps` through a single global `BoundsTree`, reproducing exactly
    /// what `insert_primitive` did before layers existed.
    fn global_ranks(steps: &[Step]) -> Vec<usize> {
        let mut tree = BoundsTree::<ScaledPixels>::default();
        let mut clip: Vec<DrawOrder> = Vec::new();
        // (paint index, order), for quads only.
        let mut quads: Vec<(usize, DrawOrder)> = Vec::new();
        let mut painted = 0;

        for step in steps {
            match *step {
                Step::Quad(bounds) => {
                    let order = clip
                        .last()
                        .copied()
                        .unwrap_or_else(|| tree.insert(bounds));
                    quads.push((painted, order));
                    painted += 1;
                }
                Step::FilterStart(bounds) | Step::FilterEnd(bounds) => {
                    tree.insert_above_all(bounds);
                }
                // Layers do not exist in the reference. That is the point: the
                // same paint sequence, ordered by one global tree.
                Step::BeginLayer(..) | Step::EndLayer => {}
                Step::PushClip(bounds) => clip.push(tree.insert(bounds)),
                Step::PopClip => {
                    clip.pop();
                }
                Step::RaiseFloor => tree.set_order_floor(tree.max_order() + 1),
            }
        }

        // Stable sort by order, matching `Scene::finish`.
        let mut sorted = quads.clone();
        sorted.sort_by_key(|(_, order)| *order);
        let mut ranks = vec![0; quads.len()];
        for (rank, (index, _)) in sorted.iter().enumerate() {
            ranks[*index] = rank;
        }
        ranks
    }

    #[test]
    fn layer_local_ordering_matches_the_global_tree_on_the_reference_scene() {
        let steps = reference_scene();
        let layered = layered_ranks(&steps);
        let global = global_ranks(&steps);
        assert_eq!(layered.len(), global.len());

        let bounds: Vec<Bounds<ScaledPixels>> = steps
            .iter()
            .filter_map(|step| match step {
                Step::Quad(bounds) => Some(*bounds),
                _ => None,
            })
            .collect();

        let mut compared = 0;
        for a in 0..bounds.len() {
            for b in (a + 1)..bounds.len() {
                if !bounds[a].intersects(&bounds[b]) {
                    continue;
                }
                compared += 1;
                assert_eq!(
                    layered[a] < layered[b],
                    global[a] < global[b],
                    "quads {a} and {b} overlap, and layer-local ordering put them in the \
                     opposite z-order to the global tree. layered={:?} global={:?}",
                    (layered[a], layered[b]),
                    (global[a], global[b]),
                );
            }
        }
        assert!(
            compared > 10,
            "the reference scene must actually exercise overlap; only {compared} \
             overlapping pairs were compared"
        );
    }

    #[test]
    fn a_layers_orders_do_not_depend_on_what_is_painted_outside_it() {
        // The property that makes retained primitives reusable at all: a
        // layer's local orders are a function of its own content and nothing
        // else. Under the old global tree they were a function of every
        // primitive painted before it in the window.
        let panel = rect(0., 0., 200., 200.);
        let capture = |before: &[Bounds<ScaledPixels>]| -> Vec<DrawOrder> {
            let mut scene = Scene::default();
            for bounds in before {
                scene.insert_primitive(quad_at(*bounds));
            }
            scene.begin_layer(LayerKey(1), panel, true);
            scene.insert_primitive(quad_at(panel));
            scene.insert_primitive(quad_at(rect(10., 10., 100., 100.)));
            scene.insert_primitive(quad_at(rect(50., 50., 100., 100.)));
            let items = scene.end_layer().unwrap();
            items
                .iter()
                .map(|item| match item {
                    LayerItem::Primitive(primitive) => primitive_order(primitive),
                    LayerItem::Nested(_) => unreachable!("no nested layers here"),
                })
                .collect()
        };

        let alone = capture(&[]);
        let crowded = capture(&[
            rect(0., 0., 800., 600.),
            rect(0., 0., 400., 300.),
            rect(0., 0., 200., 200.),
            rect(0., 0., 100., 100.),
        ]);
        assert_eq!(
            alone, crowded,
            "the same layer content produced different local orders depending on what \
             was painted before it"
        );
        assert_eq!(alone, vec![1, 2, 3]);
    }

    #[test]
    fn compositing_a_layer_reproduces_the_order_it_recorded() {
        let panel = rect(0., 0., 200., 200.);
        let mut scene = Scene::default();
        scene.insert_primitive(quad_at(rect(0., 0., 800., 600.)));
        scene.begin_layer(LayerKey(1), panel, true);
        scene.insert_primitive(quad_at(panel));
        scene.insert_primitive(quad_at(rect(10., 10., 100., 100.)));
        let items = scene.end_layer().unwrap();
        scene.finish();
        let recorded: Vec<DrawOrder> = scene.quads.iter().map(|quad| quad.order).collect();

        // Replay the same frame, compositing the layer instead of painting it.
        let mut replayed = Scene::default();
        replayed.insert_primitive(quad_at(rect(0., 0., 800., 600.)));
        replayed.begin_layer(LayerKey(1), panel, false);
        for item in &items {
            match item {
                LayerItem::Primitive(primitive) => replayed.push_retained(primitive),
                LayerItem::Nested(_) => unreachable!(),
            }
        }
        replayed.end_layer();
        replayed.finish();

        assert_eq!(
            recorded,
            replayed
                .quads
                .iter()
                .map(|quad| quad.order)
                .collect::<Vec<_>>(),
            "a composited layer produced different global orders than the paint it replaced"
        );
    }

    #[test]
    fn a_nested_layer_keeps_its_own_order_space() {
        let outer = rect(0., 0., 400., 400.);
        let inner = rect(0., 0., 200., 200.);
        let mut scene = Scene::default();
        scene.begin_layer(LayerKey(1), outer, true);
        scene.insert_primitive(quad_at(outer));
        scene.begin_layer(LayerKey(2), inner, true);
        scene.insert_primitive(quad_at(inner));
        scene.insert_primitive(quad_at(inner));
        let inner_items = scene.end_layer().unwrap();
        let outer_items = scene.end_layer().unwrap();

        // The outer layer holds one primitive and a reference, not three
        // primitives: inlining would merge the two local order spaces.
        assert_eq!(outer_items.len(), 2);
        assert!(matches!(outer_items[0], LayerItem::Primitive(_)));
        assert!(matches!(outer_items[1], LayerItem::Nested(LayerKey(2))));

        // The inner layer's own orders start from 1 regardless of the outer's.
        let inner_orders: Vec<DrawOrder> = inner_items
            .iter()
            .map(|item| match item {
                LayerItem::Primitive(primitive) => primitive_order(primitive),
                LayerItem::Nested(_) => unreachable!(),
            })
            .collect();
        assert_eq!(inner_orders, vec![1, 2]);
    }
}

#[cfg(test)]
mod slab_splice_tests {
    use super::*;
    use crate::layer::LayerItem;
    use crate::platform::cross::slab::SlabKind;
    use crate::window::build_slab_segments;
    use crate::{px, TileId};

    fn sp(value: f32) -> ScaledPixels {
        ScaledPixels(value)
    }

    fn rect(x: f32, y: f32, w: f32, h: f32) -> Bounds<ScaledPixels> {
        Bounds {
            origin: Point { x: sp(x), y: sp(y) },
            size: Size { width: sp(w), height: sp(h) },
        }
    }

    /// Wide enough that nothing inserted is clipped empty.
    fn mask() -> ContentMask<ScaledPixels> {
        ContentMask {
            bounds: rect(-1000., -1000., 10_000., 10_000.),
        }
    }

    fn quad_marked(bounds: Bounds<ScaledPixels>, marker: u32) -> Quad {
        Quad {
            bounds,
            content_mask: mask(),
            corner_radii: Corners { top_left: sp(marker as f32), ..Corners::default() },
            ..Default::default()
        }
    }

    fn shadow_marked(bounds: Bounds<ScaledPixels>, marker: u32) -> Shadow {
        Shadow {
            order: 0,
            blur_radius: sp(4.),
            bounds,
            corner_radii: Corners { top_left: sp(marker as f32), ..Corners::default() },
            content_mask: mask(),
            color: Hsla::default(),
        }
    }

    fn underline_marked(bounds: Bounds<ScaledPixels>, marker: u32) -> Underline {
        Underline {
            order: 0,
            pad: 0,
            bounds,
            content_mask: mask(),
            color: Hsla::default(),
            thickness: sp(marker as f32),
            wavy: 0,
        }
    }

    fn mono_sprite_marked(bounds: Bounds<ScaledPixels>, texture_index: u32, tile_number: u32, marker: u32) -> MonochromeSprite {
        MonochromeSprite {
            order: 0,
            pad: 0,
            bounds,
            content_mask: mask(),
            text_color: crate::TextColor {
                tag: crate::TextColorTag::Solid,
                color_space: crate::ColorSpace::Srgb,
                solid: Hsla { h: marker as f32, s: 1., l: 0.5, a: 1. },
                gradient_angle_or_reserved: 0.,
                colors: Default::default(),
                pad: 0,
            },
            tile: crate::AtlasTile {
                texture_id: AtlasTextureId { index: texture_index, kind: crate::AtlasTextureKind::Monochrome },
                tile_id: TileId(tile_number),
                padding: 0,
                bounds: Bounds {
                    origin: point(crate::DevicePixels(0), crate::DevicePixels(0)),
                    size: crate::Size { width: crate::DevicePixels(8), height: crate::DevicePixels(8) },
                },
            },
            transformation: TransformationMatrix::unit(),
        }
    }

    fn path_marked(marker: u32, origin: (f32, f32), triangle_count: usize) -> Path<ScaledPixels> {
        let (x, y) = origin;
        let mut pixels_path = Path::new(point(px(x), px(y)));
        pixels_path.content_mask = ContentMask {
            bounds: Bounds {
                origin: point(px(-1000.), px(-1000.)),
                size: Size { width: px(10_000.), height: px(10_000.) },
            },
        };
        pixels_path.move_to(point(px(x), px(y)));
        for step in 0..triangle_count {
            let offset = (step + 2) as f32 * 20.;
            let lift = if step % 2 == 0 { 15. } else { 0. };
            pixels_path.line_to(point(px(x + offset), px(y + lift)));
        }
        let mut scaled = pixels_path.scale(1.0);
        scaled.color = Background::default();
        scaled.color.solid.h = marker as f32;
        scaled.vertices[0].st_position.x = marker as f32;
        scaled
    }

    /// One identified element of a draw stream: slab kind plus unique marker.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Entry {
        Quad(u32),
        Shadow(u32),
        Underline(u32),
        Mono(u32),
        Path(u32),
    }

    fn quad_entry(quad: &Quad) -> Entry {
        Entry::Quad(quad.corner_radii.top_left.0 as u32)
    }

    fn shadow_entry(shadow: &Shadow) -> Entry {
        Entry::Shadow(shadow.corner_radii.top_left.0 as u32)
    }

    fn underline_entry(underline: &Underline) -> Entry {
        Entry::Underline(underline.thickness.0 as u32)
    }

    fn mono_entry(sprite: &MonochromeSprite) -> Entry {
        Entry::Mono(sprite.text_color.solid.h as u32)
    }

    fn path_entry(path: &Path<ScaledPixels>) -> Entry {
        Entry::Path(path.color.solid.h as u32)
    }

    /// Expand the legacy batch stream to identified entries.
    fn legacy_stream(scene: &Scene) -> Vec<Entry> {
        let mut stream = Vec::new();
        for batch in scene.batches() {
            match batch {
                PrimitiveBatch::Quads(quads) => stream.extend(quads.iter().map(quad_entry)),
                PrimitiveBatch::Shadows(shadows) => stream.extend(shadows.iter().map(shadow_entry)),
                PrimitiveBatch::Paths(paths) => stream.extend(paths.iter().map(path_entry)),
                PrimitiveBatch::Underlines(underlines) => {
                    stream.extend(underlines.iter().map(underline_entry))
                }
                PrimitiveBatch::MonochromeSprites { sprites, .. } => {
                    stream.extend(sprites.iter().map(mono_entry))
                }
                _ => {}
            }
        }
        stream
    }

    /// Expand `frame_batches` — spliced spans included — to identified
    /// entries, walking each span's runs against its packed arrays.
    fn spliced_stream(scene: &Scene) -> Vec<Entry> {
        let mut stream = Vec::new();
        for batch in scene.frame_batches() {
            match batch {
                SceneBatch::Primitives(primitives) => match primitives {
                    PrimitiveBatch::Quads(quads) => stream.extend(quads.iter().map(quad_entry)),
                    PrimitiveBatch::Shadows(shadows) => {
                        stream.extend(shadows.iter().map(shadow_entry))
                    }
                    PrimitiveBatch::Paths(paths) => stream.extend(paths.iter().map(path_entry)),
                    PrimitiveBatch::Underlines(underlines) => {
                        stream.extend(underlines.iter().map(underline_entry))
                    }
                    PrimitiveBatch::MonochromeSprites { sprites, .. } => {
                        stream.extend(sprites.iter().map(mono_entry))
                    }
                    _ => {}
                },
                SceneBatch::LayerSlab(index) => {
                    let span = &scene.layer_slab_spans[index];
                    let packed = &span.packed;
                    for run in &span.runs {
                        // Runs address the layer-wide concatenated streams;
                        // this single-stretch test layer starts at zero, so
                        // local slicing recovers the same elements.
                        let (start, count) = (run.start as usize, run.count as usize);
                        match run.kind {
                            SlabKind::Quads => stream.extend(
                                packed.quads[start..start + count].iter().map(quad_entry),
                            ),
                            SlabKind::Shadows => stream.extend(
                                packed.shadows[start..start + count].iter().map(shadow_entry),
                            ),
                            SlabKind::Underlines => stream.extend(
                                packed.underlines[start..start + count]
                                    .iter()
                                    .map(underline_entry),
                            ),
                            SlabKind::MonoSprites => stream.extend(
                                packed.mono_sprites[start..start + count].iter().map(mono_entry),
                            ),
                            SlabKind::PolySprites => {}
                            SlabKind::Paths => {
                                // Walk the flattened vertex stream's prefix
                                // sums to recover whole paths.
                                let mut offset = 0u32;
                                for path in &packed.paths {
                                    let len = path.vertices.len() as u32;
                                    if len > 0
                                        && offset < run.start + run.count
                                        && offset + len > run.start
                                    {
                                        stream.push(path_entry(path));
                                    }
                                    offset += len;
                                }
                            }
                        }
                    }
                }
            }
        }
        stream
    }

    /// Record one scene whose layers hold marked primitives, then composite
    /// it twice from the SAME captures: once through the legacy replay, once
    /// through the slab splice. Returns both finished scenes.
    fn twin_scenes() -> (Scene, Scene) {
        let panel = rect(0., 0., 400., 400.);
        let mut recording = Scene::default();
        // Root content before the layer.
        recording.insert_primitive(quad_marked(rect(0., 0., 800., 600.), 900));
        // The layer: mixed kinds, overlapping so orders increase with paint.
        recording.begin_layer(LayerKey(11), panel, true);
        recording.insert_primitive(quad_marked(panel, 901));
        recording.insert_primitive(shadow_marked(panel, 902));
        recording.insert_primitive(path_marked(903, (0., 0.), 1));
        recording.insert_primitive(underline_marked(panel, 904));
        recording.insert_primitive(mono_sprite_marked(panel, 0, 1, 905));
        recording.insert_primitive(quad_marked(rect(10., 10., 200., 200.), 906));
        let items = recording.end_layer().unwrap();
        // Root content painted after the layer, overlapping its overflow.
        recording.insert_primitive(quad_marked(rect(300., 300., 200., 200.), 907));
        recording.finish();
        assert!(matches!(items[5], LayerItem::Primitive(_)));

        // Legacy twin: push_retained replay.
        let mut legacy = Scene::default();
        legacy.insert_primitive(quad_marked(rect(0., 0., 800., 600.), 900));
        legacy.begin_layer(LayerKey(11), panel, false);
        for item in &items {
            match item {
                LayerItem::Primitive(primitive) => legacy.push_retained(primitive),
                LayerItem::Nested(_) => unreachable!(),
            }
        }
        legacy.end_layer();
        legacy.insert_primitive(quad_marked(rect(300., 300., 200., 200.), 907));
        legacy.finish();

        // Spliced twin: one span per maximal stretch of the layer's items.
        let mut spliced = Scene::default();
        spliced.insert_primitive(quad_marked(rect(0., 0., 800., 600.), 900));
        spliced.begin_layer(LayerKey(11), panel, false);
        let segments = build_slab_segments(&items, [0.; 2]).expect("the layer must be packable");
        let mut totals = [0u32; SlabKind::COUNT];
        for segment in &segments {
            if let crate::window::SlabSegment::Stretch(_, packed) = segment {
                totals[SlabKind::Quads.index()] += packed.quads.len() as u32;
                totals[SlabKind::Shadows.index()] += packed.shadows.len() as u32;
                totals[SlabKind::Paths.index()] += packed.total_path_vertices();
                totals[SlabKind::Underlines.index()] += packed.underlines.len() as u32;
                totals[SlabKind::MonoSprites.index()] += packed.mono_sprites.len() as u32;
                totals[SlabKind::PolySprites.index()] += packed.poly_sprites.len() as u32;
            }
        }
        for segment in segments {
            let crate::window::SlabSegment::Stretch(_, packed) = segment else {
                continue;
            };
            let mut offsets = [0u32; SlabKind::COUNT];
            let mut runs = Vec::new();
            for run in &packed.runs {
                let index = run.kind.index();
                runs.push(SlabRun {
                    kind: run.kind,
                    start: offsets[index] + run.start,
                    count: run.count,
                    texture_id: run.texture_id,
                });
            }
            spliced.push_layer_slab_span(
                panel,
                LayerKey(11),
                1,
                [panel.origin.x.0, panel.origin.y.0],
                totals,
                runs,
                Arc::from(packed),
            None,);
        }
        spliced.end_layer();
        spliced.insert_primitive(quad_marked(rect(300., 300., 200., 200.), 907));
        spliced.finish();

        (legacy, spliced)
    }

    #[test]
    fn spliced_mixed_scene_matches_the_legacy_batch_stream_entry_for_entry() {
        let (legacy, spliced) = twin_scenes();

        // Sanity: the splice actually happened and no primitives leaked into
        // the frame arrays.
        assert_eq!(spliced.slab_span_count(), 1);
        assert_eq!(spliced.quads.len(), 2, "only the root quads remain");
        assert_eq!(spliced.shadows.len(), 0);

        assert_eq!(
            spliced_stream(&spliced),
            legacy_stream(&legacy),
            "the spliced draw stream must equal the legacy batch stream"
        );
    }

    #[test]
    fn a_span_sorts_exactly_where_its_layer_was_painted() {
        // Overlapping-after root content must draw ABOVE every span entry;
        // before-content BELOW. The equivalence test covers this implicitly;
        // here it is asserted directly on global orders.
        let (_, spliced) = twin_scenes();
        let span_order = spliced.layer_slab_spans[0].order();
        let after_quad = spliced.quads.last().unwrap();
        assert!(
            span_order < after_quad.order,
            "content painted after the layer must sort above the span"
        );
        let first_root_quad = &spliced.quads[0];
        assert!(
            first_root_quad.order < span_order,
            "content painted before the layer must sort below the span"
        );
    }

    #[test]
    fn nested_layers_split_a_layer_into_stretches_that_keep_their_positions() {
        let outer = rect(0., 0., 500., 500.);
        let inner = rect(20., 20., 100., 100.);
        let mut recording = Scene::default();
        recording.begin_layer(LayerKey(21), outer, true);
        recording.insert_primitive(quad_marked(outer, 801));
        recording.insert_primitive(quad_marked(outer, 802));
        recording.begin_layer(LayerKey(22), inner, true);
        recording.insert_primitive(shadow_marked(inner, 803));
        let inner_items = recording.end_layer().unwrap();
        recording.insert_primitive(underline_marked(outer, 804));
        let outer_items = recording.end_layer().unwrap();
        recording.finish();
        assert!(matches!(outer_items[2], LayerItem::Nested(LayerKey(22))));

        // Spliced composite of the OUTER layer only, with the nested child's
        // shadow composited legacy-style between the stretches.
        let mut spliced = Scene::default();
        spliced.begin_layer(LayerKey(21), outer, false);
        let mut cursor = 0usize;
        let segments = build_slab_segments(&outer_items, [0.; 2]).expect("packable");
        let mut totals = [0u32; SlabKind::COUNT];
        for segment in &segments {
            if let crate::window::SlabSegment::Stretch(_, packed) = segment {
                totals[SlabKind::Quads.index()] += packed.quads.len() as u32;
                totals[SlabKind::Underlines.index()] += packed.underlines.len() as u32;
            }
        }
        for segment in segments {
            match segment {
                crate::window::SlabSegment::Stretch(range, packed) => {
                    assert_eq!(range.start, cursor, "stretches are contiguous");
                    cursor = range.end;
                    let mut offsets = [0u32; SlabKind::COUNT];
                    let mut runs = Vec::new();
                    for run in &packed.runs {
                        let index = run.kind.index();
                        runs.push(SlabRun {
                            kind: run.kind,
                            start: offsets[index] + run.start,
                            count: run.count,
                            texture_id: run.texture_id,
                        });
                        offsets[index] += match run.kind {
                            SlabKind::Quads => packed.quads.len() as u32,
                            SlabKind::Underlines => packed.underlines.len() as u32,
                            _ => unreachable!("test uses two kinds"),
                        };
                    }
                    spliced.push_layer_slab_span(
                        outer,
                        LayerKey(21),
                        1,
                        [0.; 2],
                        totals,
                        runs,
                        Arc::from(packed),
                    None,);
                }
                crate::window::SlabSegment::Nested(key) => {
                    cursor += 1;
                    spliced.begin_layer(key, inner, false);
                    for item in &inner_items {
                        match item {
                            LayerItem::Primitive(primitive) => spliced.push_retained(primitive),
                            LayerItem::Nested(_) => unreachable!(),
                        }
                    }
                    spliced.end_layer();
                }
            }
        }
        spliced.end_layer();
        spliced.finish();

        // Expected interleaving: quads 801/802, then the child's shadow 803,
        // then the parent's underline 804 — exactly what the legacy replay
        // produces.
        let entries = spliced_stream(&spliced);
        assert_eq!(
            entries,
            vec![
                Entry::Quad(801),
                Entry::Quad(802),
                Entry::Shadow(803),
                Entry::Underline(804),
            ],
            "nested content must sit between its parent's stretches"
        );
    }

    #[test]
    fn replaying_a_range_reregisters_slab_spans() {
        // A recorded range that contains a span op must survive the
        // index-range replay path: the fallback after a failed try_composite
        // depends on it.
        let panel = rect(0., 0., 200., 200.);
        let mut source = Scene::default();
        source.insert_primitive(quad_marked(panel, 701));
        source.begin_layer(LayerKey(31), panel, true);
        source.insert_primitive(quad_marked(panel, 702));
        source.insert_primitive(shadow_marked(panel, 703));
        let items = source.end_layer().unwrap();

        let mut recorded = Scene::default();
        recorded.insert_primitive(quad_marked(panel, 701));
        recorded.begin_layer(LayerKey(31), panel, false);
        let start = recorded.len();
        let segments = build_slab_segments(&items, [0.; 2]).expect("packable");
        for segment in segments {
            let crate::window::SlabSegment::Stretch(_, packed) = segment else {
                continue;
            };
            let mut runs = Vec::new();
            for run in &packed.runs {
                runs.push(SlabRun {
                    kind: run.kind,
                    start: run.start,
                    count: run.count,
                    texture_id: run.texture_id,
                });
            }
            recorded.push_layer_slab_span(
                panel,
                LayerKey(31),
                1,
                [0.; 2],
                [
                    packed.quads.len() as u32,
                    packed.shadows.len() as u32,
                    0,
                    0,
                    0,
                    0,
                ],
                runs,
                Arc::from(packed),
            None,);
        }
        recorded.end_layer();
        let end = recorded.len();
        recorded.finish();
        assert_eq!(recorded.slab_span_count(), 1);

        // Replay the range into a fresh scene.
        let mut replayed = Scene::default();
        replayed.replay(start..end, &recorded);
        replayed.finish();

        assert_eq!(
            replayed.slab_span_count(),
            1,
            "replay must re-register the span"
        );
        assert!(
            replayed.layer_slab_spans[0].order() >= 1,
            "the re-reserved slot must resolve through the new scope layout"
        );

        // And packing still rejects unsupported primitives through the same
        // window helper the composite hook uses.
        let mut rejected = Scene::default();
        rejected.begin_layer(LayerKey(41), panel, true);
        rejected.insert_primitive(crate::scene::FilterBoundary {
            order: 0,
            bounds: panel,
            content_mask: mask(),
            corner_radii: Corners::default(),
            blur_radius: sp(8.),
            opacity: 1.,
            is_start: true,
        });
        rejected.insert_primitive(quad_marked(panel, 704));
        rejected.insert_primitive(crate::scene::FilterBoundary {
            order: 0,
            bounds: panel,
            content_mask: mask(),
            corner_radii: Corners::default(),
            blur_radius: sp(8.),
            opacity: 1.,
            is_start: false,
        });
        let boundary_items = rejected.end_layer().unwrap();
        assert!(build_slab_segments(&boundary_items, [0.; 2]).is_none());
    }
}

