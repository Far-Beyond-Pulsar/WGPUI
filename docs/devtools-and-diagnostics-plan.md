# WGPUI Diagnostics, Profiling, and Inspector Plan

## Current findings

The native backend has useful building blocks but not a complete diagnostics
system:

- `wgpui-core::InstrumentationHooks` provides optional CPU spans, counters,
  frame-present notifications, and a GPU timestamp seam.
- `FrameStats`, retained node outcomes, invalidation axes, damage records, tile
  visits, and upload statistics already expose much of the information a
  profiler needs.
- `wgpui-devtools` contains small render-stat, hook, flamegraph, replay, and
  inspector shells.
- The legacy backend contains substantially more flamegraph and inspector
  behavior, but it cannot be used as the native backend's runtime dependency.

The legacy application also has a network seam that the native crates do not
currently expose. `old::App::with_http_client`, `http_client`, and
`set_http_client` store an `Arc<dyn gpui_http_client::HttpClient>`. The trait's
primary operation is
`send(Request<AsyncBody>) -> BoxFuture<Result<Response<AsyncBody>>>`, with
helpers for GET, JSON POST, redirects, proxy, user-agent, and multipart
requests. The legacy image asset loader uses that client directly. This is a
real extension point, not a browser-network implementation: DNS, connection,
TLS, protocol, and cache phases are only observable when the supplied client
reports them.

Two user-visible correctness issues must be handled before profiling results
are trusted:

1. The shadow showcase needs a pixel-level native regression proving that its
   shadow layers are visible, correctly blurred, and clipped. A successful
   launch or primitive-count assertion is insufficient.
2. `.overflow_y_scroll()` currently sets overflow/clipping state but does not
   by itself connect wheel input to a scroll state. An explicit ID is not the
   correct prerequisite under the 2.0 API: identity is positional unless an
   author supplies an ID. Automatic overflow-to-scroll-root discovery, plus
   compatibility for `track_scroll`, is required.

The diagnostics work must observe the retained pipeline and its actual damage;
it must not introduce a second scene, render cache, or a full-frame redraw.

## Design invariants

- Disabled diagnostics are effectively free: no per-element allocation, no
  global lock on the frame hot path, and no extra GPU pass.
- A capture is a coherent single presented frame. It is armed before a frame,
  freezes collection at the presentation boundary, and is immutable afterward.
- Element records use retained instance addresses and generation numbers, not
  pointer addresses. They remain meaningful across frames and cannot alias
  after an element is released.
- CPU work, GPU work, memory accounting, damage, and event-listener metadata
  are separate streams with explicit timestamps and ownership IDs.
- The profiler reports what happened; enabling it must not alter reconciliation,
  tile ownership, clipping, scroll behavior, or primitive ordering.
- Memory inspection exposes typed ownership and byte ranges that the framework
  owns. It never dereferences arbitrary application pointers or pretends a
  retained snapshot is the live scene.
- Listener inspection reports event kind, owner element, registration order,
  capture/bubble phase, and whether a handler is present. Closure internals are
  intentionally opaque.
- The external inspector consumes a versioned, length-delimited snapshot
  protocol. It must work from a file capture when no inspector is connected.
- Runtime builds do not install an instrumented client, trace recorder, GPU
  query pool, allocation observer, or inspector transport. Capture mode may
  replace/adapt those components at an explicit safe boundary and may pay for
  readback, command recording, serialization, and synchronization only while
  the capture is active.

## Capture-only session model

The backend uses an explicit state machine:

```text
Disabled -> Armed -> Collecting -> Frozen -> Exported
              ^          |            |
              +----------+------------+
                   discard / reset
```

`Disabled` is the normal state. The fast path must not call a recorder, acquire
a mutex, allocate an event object, or branch through a feature-rich observer.
`Armed` is a small atomic request that becomes `Collecting` at a frame or
network safe point. `Collecting` enables the slow hooks and any GPU command
markers/readback needed by the requested capture. At the presentation boundary
the session becomes `Frozen`; all data is then immutable and can be serialized
or inspected without holding the application or renderer live. A capture may
request one frame, a frame plus preceding input/network context, or a bounded
sequence, but the default remains one frame.

This distinction matters for GPU captures. A RenderDoc-like capture cannot be
reconstructed from ordinary retained scene bytes after the fact: command
encoders, render passes, bind groups, pipeline state, indirect arguments,
resource versions, and synchronization points must be recorded while the
capture is active. The native renderer therefore needs a capture adapter around
its command submission path, not a permanent shadow command log.

## Inspector scope

The external inspector backend should expose the following framework-native
equivalents. The UI may resemble RenderDoc and Chrome DevTools, but the backend
must report unsupported concepts instead of manufacturing them.

| Inspector area | Native data the backend must provide |
| --- | --- |
| Frame/event timeline | CPU spans, element attribution, input, invalidation, damage, uploads, command encoders, render passes, submits, presents, and calibrated GPU timestamps |
| Render passes | Attachments, load/store operations, viewport/scissor, pipeline, bind groups, vertex/index/indirect buffers, draw/dispatch arguments, and pass ordering |
| Resource browser | Primitive slabs, indirect buffers, atlases, layer/tile textures, surfaces, staging/readback buffers, generations, ownership, and last-use frame |
| Texture/buffer viewer | Format-aware metadata, bounded readback, mip/layer data, tile occupancy, byte ranges, and safe typed decoding |
| Pipeline/shader view | Stable pipeline IDs, shader stage/source identifiers, WGSL source when available, specialization/configuration, and bind-layout metadata |
| Element/DOM view | Retained element tree, type and source metadata, IDs, layout/box model, transforms, clips, scroll roots, boundaries, tile ownership, computed style, and paint records |
| Interaction view | Hitboxes, z/order, focus/hover/active state, listeners, capture/bubble phase, dispatch ancestry, and the reason an input was clipped or rejected |
| Performance view | Frame stages, inclusive/exclusive element time, rebuild reasons, invalidation axes, damage regions, tile work, upload ranges, CPU/GPU overlap, and dropped events |
| Memory view | Owned CPU allocations and GPU resources, live/capacity/high-water bytes, allocation maps, residency, evictions, generations, and bounded content readback |
| Capture/replay | Immutable frame inputs, resource versions, command/state records, retained scene/patch data, presentation target metadata, and deterministic replay diagnostics |

Browser-only concepts such as JavaScript heap objects, CSS source maps, or
service workers are not silently mapped to unrelated WGPUI objects. If an
application supplies an equivalent subsystem, it may register a typed provider
with the same snapshot protocol.

## Network diagnostics and the legacy HTTP seam

The native API should recover the legacy HTTP client seam behind a small core
trait or an optional compatibility crate, without forcing a particular HTTP
implementation into WGPUI. The minimum compatibility surface is:

- an owned request/response type with method, URL, headers, body metadata,
  redirect policy, status, and streaming body boundaries;
- an async `send` operation matching the legacy `HttpClient` semantics;
- application configuration equivalent to `with_http_client`, `http_client`,
  and `set_http_client`;
- asset loaders using the configured client rather than bypassing it;
- optional capability hooks for DNS, socket connect, proxy, TLS, protocol,
  cache, redirect, request queue, upload, download, and decompression phases.

Capture mode wraps the configured client with a recording decorator only after
capture is armed. The decorator records request start, headers subject to
redaction, body chunk timing/size, response headers/status, body chunk timing,
errors, redirects, and cancellation. It can capture bounded body bytes for a
preview, but never assumes that credentials or arbitrary response data are safe
to export. A transport that exposes phase hooks can add the Chrome-style
waterfall breakdown; otherwise the inspector marks those phases unavailable.

Network requests already in flight when capture starts are reported as
`partially observed` unless the client itself supports a capture-safe tap. The
backend must not buffer every request or add a permanent wrapper just to make
this case look complete.

## Target architecture

```text
application / native backend
        |
        v
core TraceHooks + stable IDs + frame boundary
        |
        +--> bounded trace recorder --> CPU element flamegraph
        |                         \--> merged CPU/GPU frame timeline
        +--> retained frame snapshot --> external inspector protocol
        |                           \--> element/layout/clip/listener view
        +--> allocation registry --> memory and GPU-buffer views
        +--> damage/tile records --> update-scope and tile diagnostics
```

The core-facing contract should remain small. Backend-specific facilities such
as wgpu timestamp queries, mapped buffers, surface textures, and atlas pages
belong in adapters. `wgpui-devtools` owns recording, serialization, and
presentation-independent analysis.

## Work phases

Each phase is split into three worktree tasks. Agents must add focused tests
for the behavior they touch and must not modify `old/` except for narrowly
scoped reference fixtures or compatibility evidence.

### Phase 0 — unblock correctness and establish baselines

1. **Shadow rendering correctness**
   - Reproduce the native shadow showcase at several sizes and scales.
   - Trace style-to-primitive-to-shader data for one and multiple shadow layers.
   - Add a native pixel/reference gate covering opacity, blur, spread, offset,
     rounded corners, clipping, and occlusion.
   - Fix the actual native path, preserving the public shadow API.

2. **Automatic scroll-root compatibility**
   - Make overflow containers discoverable as scroll roots when they have
     scrollable content, without requiring `.id()`.
   - Preserve explicit `ScrollHandle`/`track_scroll` behavior and the existing
     position-based identity rule.
   - Route wheel input through the hit target and scroll ancestors, using the
     shared transform/clip walk and bubbling remaining deltas.
   - Add integration tests for the shadow example shape, nested roots, resize
     while scrolled, and returning to the top.

3. **Performance baseline**
   - Measure the current native frame stages with diagnostics disabled and
     enabled: description build, reconcile, layout, shared walk, emission,
     damage planning, upload, visibility, and present.
   - Identify continuous invalidation and duplicate work before optimizing it.
   - Add a repeatable benchmark fixture with N siblings, deep nesting, a
     scrolled boundary, shadows, text, and one continuously updating surface.

### Phase 1 — zero-cost trace foundation

1. **Stable trace contract**
   - Replace string-only spans with a versioned event model containing frame ID,
     thread/queue, span ID, parent span, element address, boundary/root ID,
     tile coordinate, and monotonic timestamps.
   - Keep `NoopHooks` and a cheap disabled path source-compatible.

2. **Bounded frame recorder**
   - Implement a lock-free or per-thread buffered recorder with configurable
     event/byte budgets and explicit dropped-event counters.
   - Add frame snapshots, reset, export, and tests for nesting, overflow,
     concurrent producers, and frame-boundary atomicity.

3. **Element attribution**
   - Thread stable element metadata through reconciliation, shared walking,
     emission, interaction registration, and damage planning.
   - Record outcome, invalidation reason, layout/paint skip, effective clip,
     owning root, and tile ownership without changing rendering decisions.

### Phase 2 — single-frame flamegraphs and GPU correlation

1. **Capture controller**
   - Add an API to arm a capture for the next frame, or a selected frame ID.
   - Freeze only capture collection at the presentation boundary; do not stop
     the event loop or mutate the application tree.
   - Serialize an immutable capture with schema version, clock calibration,
     dropped-event status, and all relevant frame inputs.

2. **CPU element flamegraph**
   - Build hierarchical views for build, reconciliation, layout, prepaint,
     emission, interaction, damage, and upload preparation.
   - Support inclusive/exclusive time, element ancestry, rebuild reasons,
     reused nodes, and damage/tile contribution.
   - Export a compact JSON and a folded-stack format suitable for an external
     inspector.

3. **GPU timestamps and merged timeline**
   - Implement wgpu query-pool allocation, resolve, delayed readback, and
     device-capability fallback.
   - Correlate GPU scopes with frame and element/boundary IDs where the backend
     can prove the association; report unknown attribution explicitly.
   - Add tests for timestamp availability, query exhaustion, device loss, and
     CPU/GPU clock conversion.

### Phase 3 — retained element inspector

1. **Versioned inspector snapshot**
   - Define a transport-neutral snapshot containing the element tree, stable
     addresses, type names, source locations when available, layout bounds,
     transforms, clips, scroll roots, boundary/tile ownership, invalidation,
     and last-presented state.
   - Keep capture snapshots usable without a live application connection.

2. **Interaction and scroll inspection**
   - Enumerate listeners by event family and dispatch phase, hitboxes,
     focusability, hover/active/focus state, scroll handles, scroll extents,
     current offsets, and bubbling order.
   - Include enough metadata to explain why a clipped element did or did not
     receive input.

3. **Inspector query and selection API**
   - Add queries by stable element address, explicit ID, source location,
     bounds, boundary, scroll root, and tile.
   - Support selecting an element without causing a layout or full-scene
     rebuild; selection overlays are diagnostic damage only.

### Phase 4 — memory and GPU resource inspection

1. **Owned allocation registry**
   - Register retained slabs, primitive arenas, description buffers, layout
     storage, event registrations, trace buffers, and per-frame scratch data.
   - Report live bytes, capacity, high-water mark, allocation count, and owner
     category with no arbitrary pointer exposure.

2. **GPU resource registry**
   - Register primitive buffers, indirect argument/count buffers, atlas pages,
     layer textures, tile metadata, query buffers, and surface resources.
   - Expose byte ranges, formats, dimensions, residency, generation, upload
     history, and last-use frame; provide safe readback only when supported.

3. **Buffer visualization data**
   - Produce a stable snapshot format for hex/typed views, tile occupancy,
     slab allocation maps, atlas packing, and indirect-draw records.
   - Add redaction and size limits so a large application cannot accidentally
     create an unbounded inspector payload.

### Phase 5 — damage, tile, and visual diagnostics

1. **Authoritative update scopes**
   - Drive refresh borders from actual presentation damage, tile visits, and
     primitive uploads, never from a secondary cache or timer alone.
   - Select the outermost meaningful updating element, except when a child has
     a demonstrably faster cadence; keep the existing two-pixel yellow border
     and non-shading overlay behavior.

2. **Tile and scroll visualization**
   - Show tile ownership, visibility, residency, clip, transform-only scroll,
     newly exposed tiles, and parent/child damage subtraction on demand.
   - Make the overlay itself opt-in, bounded, and excluded from its own update
     statistics.

3. **Frame comparison and replay evidence**
   - Add last-presented versus current damage maps, primitive-slot diffs,
     upload ranges, and a replayable single-frame input record.
   - Add regression fixtures for scrolled roots, resize, shadows, text, and
     continuously updating surfaces.

### Phase 6 — external inspector and release gates

1. **Inspector transport**
   - Implement file export first, then an opt-in local IPC transport with
     authentication/endpoint ownership appropriate for a developer tool.
   - Support snapshot requests, capture arm/stop, resource readback requests,
     and capability negotiation.

2. **External viewer contract**
   - Provide schemas and a minimal reference viewer or fixtures that render the
     element tree, flamegraph, timeline, memory maps, listeners, damage, and
     tile ownership.
   - Keep the viewer independent of the renderer process and able to display a
     frozen capture after the application exits.

3. **Hardening and acceptance**
   - Run correctness tests with diagnostics off and on and compare frame
     outputs, damage, uploads, and scroll behavior.
   - Add overhead budgets for disabled mode, normal enabled mode, and capture
     mode; add stress tests for thousands of elements/listeners/tiles.
   - Run `cargo check`, targeted tests, and warnings-denied clippy. Document
     unsupported GPU timestamp/readback paths instead of silently faking data.

### Phase 7 — network waterfall, replay, and provider capabilities

1. **Native HTTP compatibility seam**
   - Introduce the native equivalent of the legacy `HttpClient` configuration
     and accessors, keeping the existing application API source-compatible
     where it already exists.
   - Route asset loading through that seam and add fake-client tests for
     redirects, failures, streaming bodies, cancellation, and response status.
   - Keep the HTTP implementation out of the core framework contract.

2. **Capture-only network recorder**
   - Add a decorator that is installed only for an armed capture and emits
     request/response metadata, timing, transfer sizes, redirects, errors, and
     bounded redacted body previews.
   - Add optional transport phase callbacks for DNS/connect/TLS/protocol/cache
     timing, with explicit unavailable markers when the user client cannot
     provide those phases.
   - Test that disabled mode makes no request recording calls or allocations,
     and that in-flight requests are correctly marked partially observed.

3. **Network inspector and deterministic replay**
   - Add a Chrome-style request list/waterfall schema with initiator, resource
     type, headers, status, transfer/cache information, timing, and body
     preview metadata.
   - Record replay fixtures without silently replaying credentials or mutating
     live application state.
   - Integrate network events into the same frozen capture bundle and external
     inspector transport.

## Dispatch order and dependencies

Dispatch Phase 0 first. Its three tasks are independent enough to run in
parallel, but the performance-baseline task should consume the correctness
fixtures from the shadow and scroll tasks before it publishes measurements.

Dispatch Phase 1 only after Phase 0 has a reproducible scrolled/shadowed
fixture. Phase 2 depends on the stable trace contract and element attribution;
Phase 3 can begin after attribution but should consume the same snapshot schema;
Phase 4 depends on resource-registration seams in the native backend; Phase 5
depends on authoritative damage records; Phase 6 is the integration and release
gate.

The first three worktree prompts should therefore be:

1. Fix and prove native shadow rendering.
2. Fix automatic scroll-root discovery and scroll bubbling without requiring an
   ID, preserving `track_scroll` compatibility.
3. Build the diagnostics-off/on benchmark and identify the current continuous
   invalidation cost centers.

No profiler result should be called authoritative until the first two tasks
pass their pixel and interaction/resize gates.

The advanced RenderDoc/DevTools work is not a license to add permanent
instrumentation. Every task must include an off-mode assertion or benchmark;
capture-mode overhead is expected and must be reported as part of the capture
metadata.
