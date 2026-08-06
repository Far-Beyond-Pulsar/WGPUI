# Retained Rendering for wgpui

Status: proposal. Supersedes the earlier eight-proposal performance plan and the
first draft of this document.

wgpui is an immediate-mode renderer with a bolted-on view cache. This proposes
converting it to a retained-mode renderer with an immediate-mode *API* — the
programming model stays exactly as it is, and everything underneath it stops
being rebuilt every frame.

There are three pillars. They are not independent proposals and cannot be
ranked against each other, because each one's central obstacle is removed by
another. §1 establishes that coupling before anything else, because it is the
reason this is one architecture rather than three features.

---

## 0. Root cause: what is wrong today

### 0.1 Everything is rebuilt every frame

Per frame, unconditionally:

- every view's `render()` runs, allocating a fresh element tree into a bump
  arena that is cleared at the end (`arena.rs`, `element.rs:693`);
- `TaffyLayoutEngine::clear()` (`taffy.rs:50`) drops the entire layout tree, and
  every node is recreated with `new_leaf`;
- every primitive is re-inserted into a global `BoundsTree` to compute its
  z-order (`scene.rs:78`);
- `Scene::finish` (`scene.rs:163`) re-sorts nine arrays;
- `WgpuRenderer::draw` (`renderer.rs:1814-1970`) re-uploads every buffer in
  full.

### 0.2 The one cache that exists is unsafe, and known to be

`AnyView::cached` reuses a view by **replaying recorded output** — index ranges
copied forward by `reuse_prepaint`/`reuse_paint` (`window.rs:3130`, `3191`) and
`Scene::replay` (`scene.rs:153`). The view's closures never run. Three
consequences:

**Side effects are silently skipped.** Element closures are impure by design.
From commit `9112ec5`:

> The level editor viewport writes `element_bounds` in a prepaint closure and its
> click/move handlers read it to normalise the cursor to [0,1] before calling
> into the engine; with the panel cached that write stopped happening and
> left-click drag went skippy.

The mitigation shipped was `Panel::cacheable`, a manual opt-out. That makes
correctness depend on every author auditing their own closures, with silently
wrong behaviour as the failure mode. It is why `.cached()` appears at four call
sites in the entire repository.

**Cache entries are validated against the world, not by construction.**
`PrepaintStateIndex`/`PaintIndex` are raw `usize` offsets into fifteen `Vec`s.
A range that ages by one frame slices out of bounds and aborts the process,
which is why `invalid_reuse_range` (`window.rs:3003`) exists as a
hand-maintained fifteen-field bounds check whose own comment records that an
earlier version silently omitted two fields.

**Invalidation is coarse upward and blind sideways.** `mark_view_dirty`
(`window.rs:1625`) walks the dispatch tree upward, so one chatty leaf
invalidates every cached view above it. And it reaches only entities owning a
dispatch node, so notifying a model marked nothing — requiring a second,
unrelated mechanism (`accessed_entity_invalidated`, `window.rs:1670`) to be
added alongside.

### 0.3 `refresh()` is a window-wide boolean that silently no-ops

```rust
pub fn refresh(&mut self) {
    if self.invalidator.not_drawing() {   // silently no-op mid-draw
        self.refreshing = true;           // kills caching for the entire window
        self.invalidator.set_dirty(true);
    }
}
```

Four animation drivers are dead because of the guard: `virtual_list.rs:226`,
`uniform_list.rs:484`, `h_list.rs:376` (prepaint) and `text.rs:813,826` (paint).
Smooth scroll does not schedule its own next frame — it animates only when some
unrelated invalidation happens to keep the window dirty.

---

## 1. Why this is one architecture

The three pillars are:

- **I. Element instances** — the element tree becomes retained state, reconciled
  against a cheap per-frame description, instead of being rebuilt.
- **II. Layers** — explicit, retained, independently composited regions with
  their own backing stores and invalidation axes.
- **III. Persistent GPU slabs** — per-layer suballocated buffers that are
  patched, not re-uploaded.

Each one is blocked on its own by something another removes:

**III is blocked by global ordering.** `insert_primitive` (`scene.rs:78`) assigns
`order` from a global `BoundsTree` — a primitive's z depends on overlap against
everything painted before it. `Scene::finish` then sorts by that order, and the
renderer draws instanced from the sorted, contiguous result. So a primitive's
*byte offset in the GPU buffer is a function of every other primitive in the
window.* Insert one quad in the middle and every subsequent quad shifts. No
delta-upload scheme survives that. **Layers fix it**: ordering becomes
layer-local (each layer owns a `BoundsTree` starting at 0), and inter-layer
order is the layer z-order. Now a layer's bytes are stable under changes
elsewhere, and per-layer slabs become possible.

**I is blocked by identity.** Reconciliation needs to match this frame's
description nodes to last frame's instances, and `GlobalElementId` exists only
for elements that were given an `ElementId` — most `div`s are anonymous.
**Layers fix it**: a layer is a reconciliation root with a stable key, so
identity inside it only has to be stable *relative to the layer*, which
positional identity provides. Layer eviction also bounds instance lifetime,
which is what keeps retained instances from being an unbounded memory leak.

**II is blocked by cost granularity.** Layers alone give "skip everything for
clean regions." But a layer that *is* dirty still redoes full layout, prepaint
and paint of its entire subtree because one label changed — which is exactly
today's behaviour, just scoped smaller. **I fixes it**: inside a dirty layer,
reconciliation confines the work to the nodes that actually changed.

**And the earlier plan's Taffy rejection was wrong for the same reason.** I
previously argued a persistent Taffy tree is impossible because `LayoutId` *is*
the taffy `NodeId` (`taffy.rs:265`), minted inside `request_layout`, so it can't
key a cross-frame map. True — and irrelevant once instances exist, because *the
instance is the stable key* and it simply holds its `LayoutId` from last frame.

So: III needs II, I needs II, II is only worth building because of I, and I
unlocks persistent layout. Building any one alone yields a fraction of its
value.

---

## 2. Pillar I — Element instances

### 2.1 The split

wgpui conflates three things that have three different lifetimes:

| | What it is | Lifetime today | Should be |
|---|---|---|---|
| **Description** | the value `render()` produces (`Div { style, children }`) | per-frame, arena | per-frame, arena — correct already |
| **Instance** | layout id, computed bounds, dispatch node, prepaint state, emitted primitives, shaped text | per-frame, dropped | **retained, keyed** |
| **State** | user state via `with_element_state` | **retained, keyed** — already correct | unchanged |

Today Description and Instance are the same object: `Drawable<E>` holds
`element: E` alongside phase-carried `RequestLayoutState`/`PrepaintState`
(`element.rs:308-340`), arena-allocated and dropped each frame.

State is *already* retained and keyed by `(GlobalElementId, TypeId)`, migrated
forward by `Frame::finish` (`window.rs:1028`). Pillar I is that exact pattern,
applied to the instance itself. There is no new concept here — only an existing
one applied to the object that actually costs something.

```rust
/// Retained per-node state. Lives in the owning layer, keyed by InstanceKey.
struct ElementInstance {
    key: InstanceKey,
    type_id: TypeId,

    description: Box<dyn AnyDescription>,   // last frame's, for diffing
    layout: Option<LayoutId>,               // retained Taffy node
    bounds: Bounds<Pixels>,
    dispatch_node: Option<DispatchNodeId>,
    prepaint: Box<dyn Any>,                 // E::PrepaintState
    primitives: PrimitiveRange,             // slab range in the owning layer

    needs: Invalidation,
    children: SmallVec<[InstanceId; 4]>,
}
```

### 2.2 Identity

```rust
struct InstanceKey {
    parent: InstanceId,
    slot: ChildSlot,     // Keyed(ElementId) | Positional(u16)
    type_id: TypeId,
}
```

Reconciliation is per-parent. New children are matched to old by explicit
`ElementId` first, then by position. `TypeId` must match — without it,
`if cond { div() } else { img() }` at the same position would reuse a `Div`
instance as an `Img`.

This is React's reconciliation without keys, and its failure mode is the right
one: a mismatch causes a **subtree rebuild** — one slow frame, never incorrect
output. Contrast with today's reuse ranges, whose failure mode was a process
abort requiring a fifteen-field guard.

Existing `ElementId`s become real keys that survive reordering, which is a
meaningful upgrade: a reordered keyed list currently rebuilds everything and
would then move instances.

### 2.3 Reconciliation, and the migration story

```rust
pub trait Element {
    /// Update this retained instance from a freshly-built description.
    /// Returns which invalidation axes the change affects.
    fn reconcile(&mut self, new: Self) -> Invalidation
    where
        Self: Sized,
    {
        *self = new;
        Invalidation::all()
    }
}
```

**The default implementation is the entire migration plan.** Every existing
element compiles untouched and behaves exactly as it does today — full rebuild,
zero savings, zero risk. Elements opt into cheapness one at a time, starting
with the four that dominate any real frame: `Div`, `Text`, `Img`, `Svg`. This is
what makes the pillar shippable incrementally rather than as a big-bang rewrite,
and it is what the previous draft failed to offer.

### 2.4 What the diff compares — and what it must not

This is the design point that makes reconciliation tractable at all.

**Listeners are never compared.** They can't be; closures aren't `PartialEq`.
They also don't need to be: *a listener affects neither layout nor paint
output.* Swap them in unconditionally, contributing `Invalidation::empty()`.
That disposes of the standard objection to diffing a closure-carrying element
tree.

**Style is split by what it affects.** `StyleRefinement` already derives
`Refineable`, so the split can be generated rather than hand-written:

| Field group | Axis |
|---|---|
| size, padding, margin, flex, position, display, gap | `LAYOUT` |
| background, border color, corner radii, shadow, opacity, text color | `DISPLAY` |
| hitbox behavior, cursor style | `HIT` |

**Text compares by `SharedString`.** `SharedString` wraps `SmolStr`
(`shared_string.rs:14`), heap-backed by an `Arc` — equality short-circuits on
pointer equality for shared clones. An unchanged label costs a pointer compare
instead of a full shaping pass. Combined with unchanged available width, the
instance reuses its `LineLayout` directly, rather than through
`LineLayoutCache`'s range replay (which is one of the two things that has
actually aborted this process).

**Effects get an explicit home.** The `9112ec5` defect is closed by giving
skipped-closure side effects somewhere legal to live:

```rust
impl Element {
    /// Runs on every frame this element participates in, reconciled or not,
    /// with resolved geometry. Geometry stashing and external-state
    /// publication belong here.
    fn on_frame(&mut self, geom: ElementGeometry, window: &mut Window, cx: &mut App) {}
}
```

The level-editor viewport moves its `element_bounds` write into `on_frame` and
becomes fully cacheable. `Panel::cacheable` and its opt-out list are deleted.
A debug assertion enforces that `on_frame` does not call `request_layout`,
`paint_*`, or `with_element_state`.

### 2.5 Persistent layout falls out

With instances retained, `TaffyLayoutEngine::clear()` (`taffy.rs:50`) stops
being unconditional. An instance whose reconcile returned no `LAYOUT` keeps its
`LayoutId`, and Taffy's own internal dirty propagation handles the rest — the
library already supports incremental layout; it was never the obstacle.

The real obstacle was the other one I named: `request_measured_layout`
(`taffy.rs:79`) stores a `'static` closure capturing per-frame state, so
retaining the tree would retain stale measurement inputs. Instances fix this
too — the closure captures the **instance id** and resolves live state at
measure time instead of closing over a snapshot. Staleness becomes
unrepresentable rather than merely avoided.

> **As shipped (phase 8, #93):** the closure still captures content directly,
> not an instance id — reuse is offered to `request_measured_layout_or_reuse`
> only when `diff_key` has already proven the content that closure captured
> is still current, so there is nothing for it to become stale *against*.
> Capturing an instance id and resolving live state at measure time would
> additionally cover content that changes *independent of* reconciliation
> (none does, today) — a strictly more general mechanism than this phase
> needed, left for if that stops being true.

### 2.6 Honest accounting

**`render()` still runs.** Reconciliation does not skip description building —
only layers do. Pillar I skips *layout, prepaint, paint, shaping, and primitive
emission*. That division is the whole reason both pillars are needed:

- Layers: skip everything for clean *regions*.
- Instances: skip nearly everything inside a *dirty* region.

Description building is genuinely cheap (constructing structs into a bump
arena); layout, shaping and paint are not. But this is an assumption worth
measuring in Phase 0 rather than asserting: if `render()` dominates for some
view, that view wants a layer, not better reconciliation.

---

## 3. Pillar II — Layers

Modelled on `CALayer`: explicitly created, retained, independently invalidated,
independently composited, with its own backing store.

```rust
pub struct Layer {
    key: LayerKey,               // stable: derived from GlobalElementId
    instances: InstanceArena,    // Pillar I lives inside here
    order_tree: BoundsTree,      // layer-LOCAL z-ordering
    slabs: LayerSlabs,           // Pillar III lives inside here
    hitboxes: Vec<Hitbox>,       // layer-relative
    transform: LayerTransform,
    needs: Invalidation,
    policy: LayerPolicy,
}
```

### 3.1 Explicit creation

```rust
div().id("properties-panel").layer().child(expensive_content)
```

`.layer()` requires an `ElementId`. That is deliberate: a layer's entire value
is surviving across frames, so it must have a stable name. Anonymous caching is
what made the current system's identity story fragile.

### 3.2 Four invalidation axes

`CALayer` splits `setNeedsLayout` from `setNeedsDisplay`. wgpui needs four:

```rust
bitflags! {
    pub struct Invalidation: u8 {
        const LAYOUT    = 1 << 0;  // re-run layout
        const DISPLAY   = 1 << 1;  // re-run paint into the backing store
        const HIT       = 1 << 2;  // re-register hitboxes / dispatch nodes
        const TRANSFORM = 1 << 3;  // composite-only: new matrix, zero CPU work
    }
}
```

`TRANSFORM` alone is the scroll case and the reason for the architecture: a 1px
scroll sets one flag on one layer. No render, no reconcile, no layout, no
prepaint, no paint, no upload — one changed matrix.

Axes are **derived by the framework** from what actually changed (§2.4), never
hand-declared at `cx.notify()` sites. Hand-classified invalidation has stale UI
as its silent failure mode, which is disqualifying for a cache.

### 3.3 Backing stores generalize existing machinery

The renderer already does layer-backed offscreen compositing.
`Window::with_filter_layer` (`window.rs:3798`) pushes `FilterBoundary` markers
into the scene; `renderer.rs:2197-2456` maintains a pool of offscreen render
targets (`renderer.rs:1433-1500`), redirects every draw between the markers into
one, and composites the result. A retained layer is that, plus:

1. **Persistent rather than pooled textures**, sized to
   `bounds + policy.overdraw_margin`. The existing pool becomes the allocator.
2. **A skip condition** — a layer dirty only in `TRANSFORM` emits a single
   `Primitive::Surface` instead of the marker pair and its contents:
   ```rust
   pub(crate) enum SurfaceContent {
       Wgpu(SurfaceId),
       Layer(LayerId),   // new
   }
   ```
   `SurfaceRegistry` (`platform/cross/surface_registry.rs`) already provides
   double-buffered, generation-tracked textures with exactly the needed API
   (`swap_ready_display_if_new`, `frame_generation`, `should_composite_swap`).
3. **`refresh_buffers()` as the composite trigger.** Its docstring
   (`window.rs:1798`) already describes this precise case: "the texture
   advanced, but nothing in the element tree changed."

**Not every layer should be rasterized.** A layer holding twelve quads is
cheaper to re-emit than to composite through a texture.

```rust
pub struct LayerPolicy {
    /// Below this primitive count the layer is primitive-retained: it keeps its
    /// slab and re-emits with the transform folded in, no texture. Default 256.
    rasterize_above: usize,
    overdraw_margin: Size<Pixels>,
    evict_after_frames: u32,   // default 60
}
```

### 3.4 Eviction

Mark-and-sweep at end of draw. A layer unvisited for `evict_after_frames`
returns its texture to the pool and drops its instance arena; the `Layer` record
survives a further interval so a scrolled-away-and-back panel re-materialises
without a full rebuild. This is what bounds Pillar I's memory: **instances are
owned by layers and die with them.**

---

## 4. Pillar III — Persistent GPU slabs

### 4.1 The problem, precisely

Today: primitives live in per-kind `Vec<T>` sorted by a globally-assigned
`order`; the renderer uploads each kind as one contiguous blob and draws
instanced with a monotonically advancing `first_instance`
(`renderer.rs:2213`). A primitive's byte offset is therefore a function of the
global sort. This is why the earlier plan's `DirtyRangeSet` sketch could not
work: there are no stable ranges to mark dirty.

### 4.2 Per-layer suballocation

Each layer owns a stable slab range per primitive kind:

```rust
struct LayerSlabs {
    quads: SlabRange,        // stable byte range in the global quads buffer
    shadows: SlabRange,
    mono_sprites: SlabRange,
    ...
    generation: u64,         // bumped when this layer's contents change
}
```

- Within a layer, primitives are ordered by the **layer-local** `BoundsTree`.
- Between layers, ordering is layer z-order.
- Draw becomes: for each layer in z-order, for each kind,
  `pass.draw(0..4, slab.base + local_range)`.

Consequences:

- A clean layer's bytes are **already resident**. Upload nothing.
- A `TRANSFORM`-only layer updates one per-layer transform uniform — 64 bytes,
  not a buffer.
- A dirty layer re-uploads only its own slab. If its primitive count changed,
  the allocator reallocates that layer's range; every other layer is untouched.
- `Scene::finish`'s nine global sorts become per-layer sorts, which are cached:
  a clean layer does not sort at all. Likewise the global `BoundsTree` insert
  per primitive disappears for clean layers.

### 4.3 Costs and mitigations, stated plainly

**More draw calls.** Batches no longer merge across layers, so the count becomes
roughly (layers × kinds present) rather than (kind runs). For a realistic editor
UI that is perhaps 20–60 draw calls — far below any driver concern, and each is
still one instanced call regardless of primitive count. Mitigation where it
matters: the slab allocator packs z-adjacent layers contiguously so runs with
identical pipeline state can be merged back into one `draw`.

**Fragmentation.** Layers resizing repeatedly fragment the slab arena. Standard
answer: size-class buckets with generational compaction during idle frames,
triggered when live/reserved falls below a threshold. Compaction rewrites slab
bases and bumps every affected layer's generation; correctness does not depend
on it happening.

**`Path` ids are index-dependent.** `insert_primitive` sets
`path.id = PathId(self.paths.len())` (`scene.rs:120`), which is a global index.
Path ids become layer-local and are remapped at composite time. Small but real,
and easy to miss.

**Atlas tile references.** Sprites carry `tile.tile_id` into the sort key. A
retained slab holds tile references that the atlas may evict. Layers must
subscribe to atlas eviction and take `DISPLAY` when a tile they reference is
dropped — this is the same hazard `force_render` handles today after device
recovery (`window.rs:1345`).

---

## 5. Where the pillars meet

Two junction points carry most of the correctness risk. Both need to be designed
up front, not discovered.

### 5.1 Ordering

An instance's retained primitives keep valid orders only while their *ordering
context* is stable. With per-layer `BoundsTree`s, that context is the layer. So:

> **Order invalidation is per-layer, not per-instance.** If any instance in a
> layer changes bounds, the layer's tree re-inserts and the layer's slab is
> re-sorted and re-uploaded. Instances elsewhere are unaffected.

This is the correct granularity and it is what ties the three pillars into one
mechanism. It also implies a sizing rule worth stating: a layer containing one
high-frequency animating element and a thousand static ones will re-sort all
thousand every frame. Layer boundaries should separate content by *update
frequency*, not only by visual grouping. `WGPUI_LAYER_DEBUG=1` (§7) is how that
gets diagnosed.

### 5.2 Hit testing

`Window::hit_test` (`window.rs:998`) walks a flat `Vec<Hitbox>` of absolute
bounds. Composite a layer at a new transform without re-prepainting and every
hitbox inside it is at last frame's position — hover, click and cursor desync
from the pixels. This is what makes `TRANSFORM`-only scrolling unshippable
without a fix, and the earlier plan did not address it.

Hitboxes become layer-relative, and hit testing transforms the **query point**
into each layer's space rather than transforming every hitbox:

```rust
fn hit_test(&self, position: Point<Pixels>) -> HitTest {
    for layer in self.layers_front_to_back() {
        if !layer.clip_bounds.contains(&position) { continue; }
        let local = layer.transform.invert().apply(position);
        for hitbox in layer.hitboxes.iter().rev() { /* ... */ }
    }
}
```

One inverse-transform per layer instead of one per hitbox, exact at any offset.
The same treatment applies to `mouse_listeners`, `cursor_styles` and tooltip
bounds, which all record absolute positions today.

---

## 6. Invalidation

Delete `Window::refreshing`. Delete the `not_drawing()` guards. Replace both
`mark_view_dirty` and `accessed_entity_invalidated` with one typed operation.

```rust
pub struct InvalidationRequest {
    scope: InvalidationScope,
    axes: Invalidation,
}

pub enum InvalidationScope {
    Instance(InstanceId),   // one node; propagates to its layer, not past it
    Layer(LayerKey),
    Entity(EntityId),       // every layer whose dependency set contains it
    Window,                 // device loss, scale-factor change
}
```

Three properties, each closing a §0.3 defect:

**Legal in every phase.** Requests during prepaint/paint append to
`pending_invalidations`, applied at end of draw — the mechanism
`flush_deferred_notifications` (`window.rs:193`) already implements. Nothing
silently no-ops. `refresh()` becomes a deprecated shim for
`Window` scope with all axes.

**No unconditional upward propagation.** `Instance`/`Layer` scope marks exactly
what it names. Ancestors are marked `LAYOUT` only when a re-laid-out child's
resulting size differs from what the parent recorded — standard dirty-layout
propagation, which terminates.

**Dependency-driven by default.** `Entity` scope resolves through a reverse
index:

```rust
entity_dependents: FxHashMap<EntityId, SmallVec<[LayerKey; 4]>>,
```

The current code cannot use a reverse index because a nested cached view that
replayed is absent from any per-frame index (see the comment at
`window.rs:1660`). Layers fix that: a layer is in the index whether or not it
rendered, because it is retained state rather than frame state. `cx.notify()`
becomes `Entity(id)` with `DISPLAY | HIT`.

**Animation.** `request_animation_frame` (`window.rs:2133`) already resolves to
the enclosing rendered view and defers correctly; the four dead `refresh()`
calls become `request_animation_frame()` immediately as Phase 1, independent of
everything else. It later gains `request_animation_frame_for(layer, axes)`, so a
smooth-scroll glide marks `TRANSFORM` only and runs at display rate with zero
CPU cost.

---

## 7. Overscroll buffers

With §2–§6 in place these are not a system, they are a policy:

```rust
LayerPolicy { overdraw_margin: size(px(0.), viewport.height * 0.5), ..default() }
```

The layer renders at viewport + margin. Scrolling sets `TRANSFORM`. When
accumulated offset exceeds the margin, the layer takes `DISPLAY | HIT` and
re-renders centred on the new position.

`VirtualList::prepaint` (`virtual_list.rs:232-263`) currently recomputes the
visible range and re-lays-out every visible item on every scroll frame. Under
layers the range is computed against the *buffer*, items are laid out once per
refill rather than once per frame, and between refills scrolling costs one
matrix.

**Refill must not stutter.** Refilling exactly at the boundary puts one
expensive frame mid-glide. Refill at 50% margin consumption so there is always a
frame of slack; once the layer render path is off the composite's critical path,
refill concurrently with the composite that does not depend on it.

---

## 8. Occlusion culling

Under an immediate-mode renderer, occlusion culling is a per-frame cost paid to
avoid per-frame work, which blunts it. Under a retained one it is qualitatively
better: **coverage and culling decisions are cached like everything else**, so
for static UI the cost amortizes to zero and the saving persists. It is worth a
phase here in a way it was not worth one before.

It also reuses structure the other pillars already build — the per-layer
`BoundsTree` (§4.2) is an AABB tree over exactly the primitives a coverage query
needs, so no separate spatial index is required. The earlier proposal's
"interval tree on the X-axis with Y-intervals per X-slice" is unnecessary.

### 8.1 Two tiers

**Layer-level (coarse, cheap, largest win).** A layer entirely covered by opaque
layers above it in z-order is skipped: its draws are not issued, and — more
valuably — **it does not re-render even when dirty.** A dirty but fully occluded
layer takes `deferred_dirty` and materialises its work only when it becomes
visible again.

That second half is the real prize for this application. A modal dialog or a
maximized panel covering the editor means the entire editor layer tree stops
rendering *and* stops re-rendering in response to notifications, rather than
rendering full frames into pixels nobody sees. Docked-but-hidden panels, stacked
tabs, and collapsed docks all fall out of the same rule.

**Instance-level (fine, within a layer).** Inside a dirty layer, skip primitive
emission for instances whose bounds are fully covered by opaque instances above
them in the layer's local order.

### 8.2 Cull at the right moment

Persistent slabs (§4) change where culling belongs, and getting this wrong would
churn the thing the slabs exist to keep stable.

If culling changed a *clean* layer's primitive set, the layer's slab would need
rewriting — so an occluder moving every frame would force its occludee to
re-upload every frame, which is worse than not culling. So:

- **Layer tier culls at draw time.** Primitives stay resident in the slab; the
  draw range is simply not issued. A fully-occluded layer costs nothing and its
  bytes never move. Trivially contiguous, since a layer's slab is contiguous by
  construction.
- **Instance tier culls at emission time, and only for layers that are dirty
  anyway.** Those instances are being re-emitted regardless, so filtering them
  is free. A *clean* layer is never re-emitted merely because occlusion changed
  — instead, an occlusion-state change marks the layer `DISPLAY`, and for static
  UI that never fires.

This keeps culling a strict optimization on top of the slab lifecycle rather
than an input to it.

### 8.3 What counts as an occluder

The earlier proposal said "elements with `background: Some(Solid(color))` where
`color.alpha() == 1.0`." That is necessary and nowhere near sufficient in this
renderer. A conservative opaque region requires all of:

- **Solid background.** `Background` may be a gradient or pattern; only the solid
  variant qualifies without further analysis.
- **`element_opacity == 1.0`.** Opacity is a separate multiplier applied at paint
  time (`Window::element_opacity`), independent of the color's own alpha.
- **Corner radii inset.** A rounded rect does **not** cover its bounds. The
  opaque region is the bounds inset by the corner radius on each affected side —
  omitting this produces visible corner artifacts, and they will be subtle
  enough to ship.
- **Border opacity.** A translucent border over an opaque fill leaves the border
  band non-opaque; inset by border width unless the border is itself opaque and
  the same color.
- **No backdrop filter above it.** `BackdropFilter` primitives *read* what is
  behind them. Anything beneath a backdrop filter must never be culled, however
  occluded it appears. The occlusion sweep must treat a backdrop filter as
  poisoning everything below it within its bounds.
- **Blur margin.** `FilterBoundary` groups (`with_filter_layer`) sample
  neighbouring pixels, so content within `max_blur_radius` of a filtered group's
  edge is still read. Occluders adjacent to a filter group must be shrunk by
  that radius, or the group's bounds expanded, before either participates.

Viewport/clip culling already exists and is not part of this: `insert_primitive`
early-returns on an empty clipped bounds (`scene.rs:93`).

**Overdraw regions are exempt.** A layer with `overdraw_margin` (§7) renders
content outside its current clip precisely so a later `TRANSFORM` can reveal it.
Culling within the margin would defeat the buffer. Occlusion applies to a
layer's visible region only.

### 8.4 Culling must never skip hit registration

This is the same defect class as `9112ec5` (§0.2), and it would be easy to
reintroduce here.

Visual occlusion is **not** hit occlusion. An opaque quad painted over a button
does not stop that button being hovered or clicked in wgpui today — blocking
mouse input is a separate, explicit opt-in
(`HitboxBehavior::BlockMouse` / `InteractiveElement::occlude`). So culling an
element because it is visually covered must not remove it from hit testing, or
overlays would start silently swallowing input that previously passed through.

The constraint maps cleanly onto the invalidation axes, which is a good sign the
seam is in the right place:

> **Occlusion culling suppresses `DISPLAY` work only. Never `HIT`, never
> `LAYOUT`, and `on_frame` (§2.4) still runs for culled elements.**

Hitboxes, dispatch nodes, listeners, cursor styles and tooltip bounds are all
registered for occluded content exactly as before. Only primitive emission and
draw issuance are skipped.

### 8.5 It must be provably a no-op

Culling is the one optimization here whose bugs are invisible in the common case
and catastrophic in the rare one. Two mechanisms, both required:

- `WGPUI_OCCLUSION=0` disables it entirely.
- `WGPUI_OCCLUSION=validate` renders each frame twice — culled and unculled —
  and compares the resulting scenes, logging any divergence with the offending
  layer and instance. Slow by design; run it in CI over a scripted UI walk
  alongside the hit-test differential test from Phase 5.

Counters: `occlusion: layers culled`, `occlusion: layers deferred-dirty`,
`occlusion: instances culled`, `occlusion: poisoned by backdrop filter`.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| Reconciliation costs more than it saves for shallow/cheap subtrees | `reconcile` returns `Invalidation`; the framework tracks a per-subtree payoff ratio and marks persistently-unprofitable subtrees `AlwaysRebuild`, skipping the diff. Adaptive and measured, not guessed. |
| Retained instances leak memory | Instances are owned by layers; layer eviction (§3.4) bounds them. A window with no layers retains nothing. |
| Layer boundaries drawn wrongly (one animating element in a large static layer re-sorts everything, §5.1) | `WGPUI_LAYER_DEBUG=1` tints composites by `LayerId` and flashes on re-render, so a secretly-rebuilding layer is visible rather than merely slow. Plus `layer: re-rendered` counters. |
| Atlas eviction invalidates retained sprite slabs | Layers subscribe to atlas eviction and take `DISPLAY` (§4.3). |
| Slab fragmentation | Size-class buckets, generational idle compaction (§4.3). Correctness independent of it running. |
| Hit-test regressions from transform math | Differential test: for random transforms, layer hit-test result must equal a full re-prepaint at that offset. Lands with Phase 5, before any `TRANSFORM`-only path ships. |
| Occlusion culls something visible (rounded corners, backdrop filters, blur margins — §8.3) | `WGPUI_OCCLUSION=validate` renders culled and unculled and diffs the scenes; run in CI over a scripted UI walk. Occluder classification is conservative by construction — when in doubt, not an occluder. |
| Occlusion silently swallows mouse input (§8.4) | Culling suppresses `DISPLAY` only; hitbox/dispatch/listener registration and `on_frame` are untouched by definition, and the hit-test differential test from Phase 5 covers the regression. |
| Deferred-dirty layers (§8.1) never materialise, leaving stale content when revealed | Becoming visible is itself a `DISPLAY` invalidation; a layer cannot leave deferred-dirty without re-rendering. Counter `occlusion: layers deferred-dirty` should return to zero when an overlay closes. |
| The whole thing destabilizes the crate | `WGPUI_LAYERS=0` / `WGPUI_INSTANCES=0` / `WGPUI_LAYERS_RASTERIZE=0` / `WGPUI_OCCLUSION=0` kill switches from the first commit, following the `WGPUI_NESTED_VIEW_CACHE` precedent (`view.rs:103`). Old paths stay functional until the final phase. |

---

## 10. Phasing

| Phase | Work | Gate to proceed |
|---|---|---|
| **0** | Baseline. `WGPUI_RENDER_STATS=1` already counts cache hits/misses and draw-vs-present. Add per-phase timing: render / reconcile-equivalent / layout / prepaint / paint / sort / upload. **Specifically measure whether `render()` is cheap relative to layout+paint (§2.6) — Pillar I's premise.** | Numbers exist for the workloads that matter. |
| **1** | Four dead `refresh()` → `request_animation_frame()`. Independent of all below. | Smooth scroll completes a glide with no other input. |
| **2** | Typed invalidation (§6): `InvalidationRequest`, phase-legal deferral, reverse entity index. Delete `refreshing`. Old cache still in place, driven by new requests. | `view cache: rebuilt` per frame drops; no new `stale range` counts. |
| **3** | `Element::on_frame` effects channel + migrate the geometry-stashing panels. | Level-editor viewport cacheable with correct drag. `Panel::cacheable` list empty. |
| **4** | `Layer` primitive: identity, eviction, primitive-retained backing only. Layer-local `BoundsTree`. `.layer()` API. | A layer composites unchanged across frames; layer-local ordering matches global ordering on a reference scene. |
| **5** | Layer-relative hitboxes + point-transform hit testing (§5.2). | Differential hit-test test passes at arbitrary transforms. |
| **6** | **Layer-tier occlusion culling** (§8.1): opaque-layer classification (§8.3), front-to-back sweep, skip draws, `deferred_dirty` for occluded dirty layers. Cull at draw time (§8.2). `WGPUI_OCCLUSION=validate`. | With a modal open, the layers behind it issue no draws and re-render zero times while notifying. Validate mode reports no divergence over a scripted UI walk. |
| **7** | `ElementInstance` + `Element::diff_key` with the conservative default (`None`). Migrated `Div`, `SharedString`/`&str`, `Svg`; `Img` deliberately not migrated (its rendered output depends on per-element async-loading/animation state a `&self`-only `diff_key` cannot observe safely — see `img.rs`'s note on `impl Element for Img`). Implemented as `Element::diff_key(&self, window: &Window) -> Option<Box<dyn ReconcileKey>>` rather than the `reconcile(&mut self, new: Self) -> Invalidation` sketched in §2.3 — comparison happens against a small owned fingerprint (`ReconcileKey`), not the live description, which sidesteps having to retain arena-allocated children across the frame boundary (see `instance.rs`'s module doc). `InstanceKey` is address-by-path like `LayerKey`, not the `{parent, slot, type_id}` triple sketched in §2.2 — type mismatches are caught by a failed downcast inside `ReconcileKey::compare` instead. Scoped to content inside a `.layer()` subtree only, and skips `prepaint`/`paint` only — `request_layout` (and Taffy node creation) still runs unconditionally every frame; see §2.5, still phase 8's job. | Inside a re-rendering (non-composited) layer, a child whose `diff_key` compares unchanged skips `prepaint`/`paint` entirely — tested directly in `window.rs`'s `an_unchanged_child_inside_a_re_rendering_layer_skips_paint`. `WGPUI_INSTANCES=0` reproduces the exact pre-#92 behaviour (`disabling_instances_rebuilds_every_child_on_every_notify`). |
| **8** | Persistent Taffy: `ElementInstance` retains `layout: LayoutId` for `Div`, `Svg` and (`SharedString`/`&str`, via `request_measured_layout_or_reuse`) `Text`; `Window::request_layout_or_reuse` reuses a retained node outright — no `set_style`/`set_children` incremental patching in this phase, since the recursive `diff_key` check (fixed alongside this phase, see `fix/div-diff-key-recursion`) already proves the whole subtree's `LAYOUT` axis is clean before reuse is offered at all. `TaffyLayoutEngine::clear()` is replaced by `end_frame()`, a touched-this-frame sweep rather than a root-reachability walk — sound because `request_layout` is never skipped at the Rust level (§2.6), so every live node is touched by *something* every frame, reused or freshly created. `WGPUI_PERSISTENT_LAYOUT=0` reverts to `clear()`. `StyledText` and `Img` are not migrated (`StyledText` has no `diff_key`; `Img`'s absence of one is phase 7's own, see `img.rs`). | `clear()` no longer called with the switch on; an unchanged reconciled child keeps the *same* `LayoutId` across frames (not merely an equal one — see `window.rs`'s `an_unchanged_reconciled_child_keeps_its_taffy_node`); live node count returns to steady state over many frames rather than growing (`taffy_node_count_does_not_grow_unboundedly`); a genuine size change still resizes correctly (`a_layout_change_still_resizes_the_reused_node`). |
| **9** | Per-layer GPU slabs (§4): suballocation, per-layer transform uniforms, per-layer sorts. | A `TRANSFORM`-only frame issues zero `write_buffer` beyond one uniform. Idle window issues zero. |
| **10** | **Instance-tier occlusion culling** (§8.1): coverage queried against the layer's local `BoundsTree`, culled at emission time for already-dirty layers only (§8.2). | Occluded instances inside a dirty layer emit no primitives; a clean layer's slab does not churn when an occluder moves. Validate mode still clean. |
| **11** | Texture-retained layers via generalized `with_filter_layer` + `SurfaceContent::Layer`. Overscroll buffers as `LayerPolicy` (§7). | Scroll frame cost independent of item count between refills. |
| **12** | Delete the old cache: `invalid_reuse_range`, reuse ranges on `PrepaintStateIndex`/`PaintIndex`, `Scene::replay`, `WGPUI_NESTED_VIEW_CACHE`, `AnyView::cached`'s replay path. | No known workload needs a kill switch. |

Phases 1–3 are independently valuable and commit to nothing structural. **Phase
4 is the decision point** — where the API surface changes. Phase 7 is the
largest single piece of work and depends on 4 landing cleanly. Phases 9–11 are
pure performance and only pay off once 4–8 are correct.

The two occlusion phases are deliberately split and deliberately separated.
Phase 6 needs only layers and the visual/hit separation established in Phase 5,
and it carries most of the win for this application — so it lands early, well
before the larger instance work. Phase 10 needs both instances (7) and slabs (9)
to exist, because §8.2's emission-time rule is only well-defined once slab
lifetime is. Running them as one phase would force the cheap, high-value half to
wait on the expensive half.

---

## 11. Rejected

- **Hand-declared invalidation version stamps** (`layout_version` /
  `paint_version` / `children_version` set at each `cx.notify()` site).
  Correctness would depend on every call site classifying its own change
  correctly, with stale UI as the silent failure mode. The axes here are derived
  by the framework from an actual diff (§2.4).
- **Per-element reuse ranges in the current scheme** — extending
  `PrepaintStateIndex`/`PaintIndex` to element granularity. This multiplies §0.2's
  failure surface by ~100x on a mechanism that has already aborted the process.
  Note this is a rejection of the *mechanism*, not the goal: Pillar I delivers
  per-element reuse via keyed retained instances, whose failure mode is a
  rebuild rather than an out-of-bounds slice.
- **A separate spatial index for occlusion** — the earlier proposal's quadtree of
  opaque regions, or its "interval tree on the X-axis with Y-intervals per
  X-slice." Each layer already maintains a `BoundsTree` (§4.2) over exactly the
  primitives a coverage query needs. Building a second structure would duplicate
  it and could disagree with it.
- **Alpha-threshold or per-pixel occlusion.** Coverage is decided from geometry
  and material only (§8.3), conservatively. Anything requiring pixel readback or
  a depth prepass costs a GPU round trip to save CPU work, which is the wrong
  trade in a UI renderer whose bottleneck is the CPU side.
