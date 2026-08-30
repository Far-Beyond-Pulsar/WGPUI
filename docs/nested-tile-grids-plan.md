# WGPUI 2.0 Nested Tile Grids and Regional Damage Plan

## Purpose

Extend the current tiled boundary implementation into a hierarchy of retained
scroll roots. Every scroll root may own an independent two-axis tile grid, and
child grids may replace the parent's damage and rasterization responsibility
for the pixels they cover. Scrolling an already-resident root must remain a
compositor transform whenever possible; crossing a tile boundary must produce
only the newly exposed work.

This plan addresses three separate concerns that must not be conflated:

1. **Geometry:** layout, scroll offsets, clips, and hitboxes must use the same
   coordinate chain.
2. **Damage:** a changed primitive or interaction must invalidate only the
   smallest affected tile set.
3. **Presentation:** parent and child tile grids must composite in the correct
   order without double drawing or leaking through clips.

The browser comparison is precedent, not an implementation dependency:
Chromium rasterizes large content layers as tiles and scrolls retained tiles on
the compositor thread; WebRender uses picture-cache slices and tile damage per
scroll root. WGPUI will retain the same useful separation while keeping its
GPU-driven scene and indirect-draw architecture.

## Non-negotiable invariants

- A scroll tick inside the resident range changes transforms and visibility
  metadata only; it does not reshape, re-layout, re-emit, or upload unchanged
  content.
- A clip resize invalidates nodes whose effective clip changed, even if their
  own layout rectangle and primitive bytes did not change.
- A hover transition invalidates the union of the old and new hit regions,
  mapped to the affected tile grids. It never invalidates the whole viewport by
  default.
- A child tile grid owns its pixels. Parent damage must not cause child tiles to
  be rerasterized; child damage must not force unrelated parent tiles to be
  rerasterized.
- A tile is eligible for drawing only when it intersects the visible viewport
  and its ancestor clip chain. Retention outside the visible range is a memory
  policy, not permission to draw.
- Hit testing uses the same nested scroll transforms and clips as rendering.
  Fully clipped controls cannot receive input.
- Every retained address remains stable when sibling content is unchanged.
- A failed or unsupported tile configuration falls back to a correct untiled
  retained boundary without panicking or changing application semantics.
- GPU uploads remain delta-only. A damage record names exact primitive slots or
  tile textures; no full-scene upload is permitted as a convenience fallback.

## Target model

### Scroll roots

Introduce a retained `ScrollRootId` for every element that establishes a
scrolling clip. A root stores:

- parent scroll-root identity;
- viewport rectangle in parent coordinates;
- content-space origin and extent;
- current and previous scroll offsets;
- clip policy and overscan policy;
- optional `TileGrid` and its resident-tile budget;
- compositor transform state;
- child scroll-root list;
- damage accumulated since the last presented frame.

The existing `.boundary()` remains the public fast-path API. Normal overflow
containers should be able to become scroll roots automatically when their
content or policy warrants it; explicit IDs remain useful for stable identity,
debugging, and state association, but must not be required for correctness.

### Tile ownership

Use a hierarchical key:

```text
(ScrollRootId, TileCoord, generation)
```

The generation prevents a released tile from aliasing a new tile at the same
coordinate. Each tile owns its retained primitive slots, texture/cache token,
visibility state, and last-used priority. A child grid is not flattened into
the parent grid; it is a separate ownership domain with a compositing edge.

### Damage representation

Represent damage as a set of rectangles tagged with their owning scroll root:

```text
Damage {
    root: ScrollRootId,
    content_rect: Rect,
    reason: Content | Hover | Clip | ScrollReveal | Resource,
}
```

Damage is propagated down the hierarchy by coordinate conversion. When a child
root covers a region, parent raster damage is clipped around the child's
coverage. Parent compositing damage still includes the child's transformed
quad, but parent rasterization does not include the child's content.

## Implementation sequence

### Step 1: Establish a single transform/clip walk

Refactor emission, interaction collection, tile visibility, and debug-region
mapping to consume one shared retained walk result. The result must provide,
for each node:

- absolute origin;
- accumulated scroll translation;
- effective clip;
- owning scroll root;
- visible bounds;
- stable instance address.

Acceptance criteria:

- nested vertical and horizontal scroll roots produce identical render and
  hit-test rectangles;
- resizing any ancestor updates descendants' effective clips;
- a clipped control cannot receive hover, click, or scroll input;
- no duplicate coordinate arithmetic remains in the native input path.

### Step 2: Make scroll input root-aware

Route wheel, touch, and future pan events through the topmost hit registration,
then bubble through scroll ancestors until one consumes the event. The event
must carry the root-local position and delta. A scroll handler that consumes a
delta invalidates only its root's transform/visibility state; it does not
invalidate content bytes.

Acceptance criteria:

- a scroll container works without `.id()`;
- nested containers consume input in inner-to-outer order;
- an inner container at its limit bubbles the remaining delta outward;
- pointer movement over unchanged hitboxes schedules no frame;
- hover enter/leave dispatches exactly once per identity transition.

### Step 3: Introduce hierarchical tile ownership

Generalize `LayerKey::tiled` and tile residency so each scroll root has an
independent grid. Preserve the existing single-grid behavior as a special case
and keep the current GPU visibility/indirect-argument path. Add parent/root
generation checks to eviction and texture retention.

Acceptance criteria:

- two independent scroll roots can retain overlapping tile coordinates without
  aliasing;
- nested roots can be evicted independently;
- a parent scroll transform does not rewrite child tile primitives;
- crossing a child tile boundary reveals only child tiles, not parent tiles;
- memory budgets are enforced independently and reported when infeasible.

### Step 4: Implement damage ownership and subtraction

Build a damage planner that maps primitive changes, hover regions, clip changes,
resource uploads, and scroll reveals to tile sets. Subtract child-root coverage
from parent raster damage. Keep compositing damage separate from raster damage.

For a hover transition, compute:

```text
affected = old_hit_region ∪ new_hit_region
```

Then intersect it with each relevant root's visible tile plane. A changed
primitive may touch multiple tiles; those tiles are the only ones eligible for
re-emission or upload.

Acceptance criteria:

- hover over a control flashes and updates only intersecting tiles;
- hover out restores only the old region;
- adjacent tiles remain visibly distinguishable in diagnostics;
- a content change inside a child grid never flashes unrelated parent tiles;
- a parent background change does not rebuild child-grid content.

### Step 5: Resize without whole-scene rebuilds

Treat viewport and clip changes as geometry damage. Recompute layout only where
the layout constraints actually changed, and re-emit only nodes whose bounds,
effective clip, or inherited transform changed. Recompute tile visibility on the
GPU from the new viewport and retain already-valid tiles.

Use a short resize policy:

- coalesce native resize events;
- configure the surface once per accepted size;
- render the latest size;
- preserve resident tile content where its content-space coverage is still
  valid;
- prioritize newly visible edge tiles;
- avoid synchronous readback or full atlas rebuild during the drag.

Acceptance criteria:

- repeated resize events coalesce;
- shrinking and growing back updates clips in both directions;
- a resize with unchanged primitive bytes produces zero primitive uploads;
- only tiles entering the viewport are newly rasterized;
- frame-time instrumentation shows no full-scene CPU walk beyond the shared
  retained geometry walk.

### Step 6: GPU compositing and nested clip enforcement

Represent each scroll root's transform, clip, and child-root coverage in GPU
buffers. The compositor should resolve root transforms and tile visibility in
compute, then issue indirect draws for the surviving tile/layer records.

Primitive shaders must enforce the effective clip for all primitive families,
including blurred shadows, paths, sprites, glyphs, and underlines. Rectangular
UV crops are acceptable for sprites and glyphs; effects whose falloff depends
on the original geometry require shader clip tests rather than geometry
distortion.

Acceptance criteria:

- no primitive leaks through any ancestor clip;
- nested clips intersect correctly under arbitrary two-axis offsets;
- child content composites above/below parent content according to paint order;
- all available draw modes produce identical pixels;
- indirect paths do not require CPU instance counts.

## Diagnostics and observability

Extend `TileRefreshFlash` into a region-aware diagnostic mode with distinct
colors or borders for:

- content upload;
- hover damage;
- clip/resize damage;
- scroll reveal;
- child-grid ownership.

Expose counters for:

- layout nodes recomputed;
- nodes re-emitted;
- primitive slots updated;
- tile textures regenerated;
- tile transforms changed;
- parent damage subtracted by child grids;
- hover hit transitions;
- frames requested without a meaningful state change.

The diagnostic mode must itself be retained and must never alter normal render
behavior when disabled.

## Required correctness gates

1. **Nested geometry differential:** compare shared-walk output against an
   independent CPU oracle for three nested roots, both axes, offsets including
   negative coordinates, and resize in both directions.
2. **Scroll fast path:** a resident-range scroll changes transforms/visibility
   only, with zero shaping, layout emission, primitive upload, or atlas upload.
3. **Boundary crossing:** crossing one tile edge creates exactly the newly
   revealed tile set; parent and sibling grids remain untouched.
4. **Regional hover:** enter and leave a button changes exactly the old/new
   tile intersection and dispatches one enter/leave pair.
5. **Nested damage isolation:** mutate parent, child, and overlapping sibling
   content separately and assert the exact dirty tile sets.
6. **Resize clip gate:** shrink, grow, and nested-root resize; assert no pixels
   outside the effective clip and no stale clip after returning to the original
   size.
7. **Pixel differential:** render the same nested scene through all draw modes
   and compare against a CPU reference compositor with zero tolerance where
   formats permit exact comparison.
8. **Stress gate:** thousands of roots/tiles, rapid alternating scroll and
   hover, bounded memory, no stale generation references, and no panics.

## Performance gates

- Resident-range scroll: zero primitive bytes and zero atlas bytes uploaded.
- Hover transition: upload and draw work proportional to affected tiles, not
  viewport area.
- Resize: no synchronous full-scene upload; frame pacing remains bounded while
  edge tiles are prioritized.
- Nested grids: child-grid scrolling does not scale CPU work with parent tile
  count.
- Diagnostics off: no additional allocations or GPU passes in the steady state.

## Deliberate exclusions

- This plan does not make arbitrary vector tessellation GPU-generated; paths
  remain CPU-produced until a separate path-generation phase proves a GPU
  implementation faster and compatible.
- It does not require every element to be tiled. Small or frequently changing
  content may remain an untiled retained layer.
- It does not make developer IDs mandatory. Stable automatic identity remains
  the default, with IDs used as an optimization and state/debugging aid.
- It does not use a full-viewport damage fallback except when correctness
  requires it and the fallback is explicitly reported by diagnostics.

## Definition of done

Nested scroll roots work with existing application code, including the current
examples, without API changes beyond optional explicit IDs or cache boundaries.
Scrolling, hover, and resize update only the pixels and retained records they
actually affect. The correctness gates pass across all supported draw modes,
the performance counters demonstrate the fast paths, and the implementation
has a documented untiled fallback for unsupported or over-budget cases.

## Implementation boundary

The native backend now implements the shared retained walk, nested root and
tile visibility/residency metadata, regional damage calculations, scroll
bubbling, delta-only scene updates, layer translations in every native
primitive shader, and the untiled fallback. The last presented frame is held
in one presentation buffer when the target supports `COPY_DST`; tile metadata
then restricts the damage raster region, rather than becoming a second scene
or render cache. The existing GPU tile-visibility and indirect-argument
passes remain covered by their differential integration tests.

Two pieces remain deliberately isolated because the current public frame
protocol has no data path for them. `ScrollRootTable` is not yet the source of
truth for ordinary overflow discovery: `PlannedNode::declared_boundary` is
still populated only by an explicit boundary description. The GPU tile pass is
also not yet driven by per-root descriptors from `FrameRenderer`; the native
frame path currently uses ordinary indirect visibility passes plus a single
union scissor over the presentation damage. This is conservative for
disjoint regions and does not create per-tile scene layers or a secondary tile
render cache. Wiring exact per-tile GPU visibility further requires extending
the frame input protocol. Targets that do not offer `COPY_DST` use the untiled
full-render fallback, preserving correctness while the presentation buffer
cannot be retained.
