# WGPUI Rendering-Perf Port Plan (single-crate Pulsar)

> ## PORT PROGRESS (live)
> **Committed to `main` (build + clippy green, behavior unchanged unless noted):**
> - Phase 1 — SceneChunk tracking + incremental sort (`ac56f89`)
> - Phase 2 — scene-side damage computation (`5f9fc50`, dormant)
> - Phase 4 — per-view layout caching (`13e8ce73`, **active perf win**, unvalidated)
> - Phase 6 — scene batch counting (`1f13dd73`, metric)
> - Phase 0 — frame-metrics collection (`39f0332`, slim; perf crate recreated)
>
> **Implemented in this working tree (safe path, still needs downstream visual validation):**
> - Phase 3 — per-type dirty-range upload support via scene generation + persisted
>   changed ranges, with full-upload fallback on size changes and empty writes
>   skipped safely.
> - Phase 5 — safe row element identity cache for `list` + `uniform_list`
>   without reusing painted `AnyElement` handles.
> - Phase 7 — conservative buffer compaction by recreating buffers when more
>   than 30% of the current allocation is unused.
>
> **Genuinely live in production (with focused tests):**
> - Measured-layout cache: exact (per-frame, per-node) keying on known
>   dimensions + available space in the real `taffy` layout path
>   (`src/gpui/taffy.rs`, `src/gpui/retained/measurement.rs`).
> - WGPU-surface bounds-cache pruning of destroyed/unseen surfaces in the live
>   compositor draw, with a full-redraw fallback (`renderer.rs`).
> - View prepaint/paint reuse (`ViewState`) and scene-chunk identity
>   (`SceneSegmentPool`).
> - **Retained fiber identity (this session):** every id-bearing element is
>   mirrored into the `FiberTree` on the live layout walk
>   (`src/gpui/element/drawable.rs`) — universally, with no opt-in and no
>   "legacy vs retained" branch — carried forward across cached frames via the
>   view-reuse path, and consumed by `FrameMetrics` (`retained_fiber_count` /
>   `retained_dirty_fiber_count`). Cross-frame fiber identity and clean-on-reuse
>   dirty tracking are covered by `src/gpui/retained/identity_bridge_tests.rs`.
>   This is a read-only side table: it does **not** yet drive rendering.
>
> **UI refresh-heat overlay (this session, wired into main — no feature gate):**
> A 2D analogue of Unreal's shader-complexity view. Per-view layout/prepaint cost
> is timed in `src/gpui/view/{entity_element,any_element}.rs`, mapped to a
> `RefreshHeat` level by `src/gpui/retained/profiling.rs`, and—when the overlay is
> on—painted as a translucent heat quad over each view plus a bottom legend
> (`Window::paint_refresh_overlay`). Toggled via the public
> `Window::toggle_refresh_overlay`/`set_refresh_overlay` API and the bindable
> `ToggleRefreshOverlay` action; while on, the window forces a full repaint each
> frame so the heat stays live. Hottest level is surfaced in
> `FrameMetrics::refresh_heat`. Headless tests cover heat mapping, current-frame
> regions, distinct colors, toggle state, and quad injection. A demo lives at
> `examples/refresh_overlay.rs` (`cargo run --example refresh_overlay`, F12
> toggles). Display QA of the colored overlay is the maintainer's pre-merge step
> (no display here).
>
> **Partial redraw is no longer env-gated:** the `GPUI_PARTIAL_REDRAW` opt-out was
> removed; partial redraw is unconditionally the path, keeping only the per-frame
> correctness fallbacks (no persistent framebuffer / full redraw needed / damage
> cannot form a scissor -> full clear).
>
> **Scaffolding removed (this session — no dead code, no legacy):** the opt-in
> `RenderNode` trait + `UpdateResult` + per-phase payload structs + the
> companion-node lifecycle (`update_retained_node`/`_tree`, `create/update_render_node`,
> `render_node_type_id`, `fiber_children`) + `LegacyRenderNode` + the intrinsic
> sizing module + the render-node test fixtures were all deleted. They were never
> consumed by a real frame (payloads were always `None` in production), so rather
> than keep dead scaffolding they were removed. The `retained/` subsystem now has
> **zero `expect(dead_code)` gates**; every remaining item is live in production or
> `#[cfg(test)]` (compiled out of release builds entirely). The
> `render_node_type_id().is_none()` "legacy" branch is gone — fiber reconciliation
> is universal.
>
> **Further dead-code sweep (this session):** also removed the inert scene
> transform subsystem (`Transform2D`/`TransformId`/`TransformTable`,
> `insert_transform`, `begin_view_chunk_with_transform`, per-chunk `transform_id`)
> which only ever held identity transforms; the dead `Scene::mark_chunk_dirty`/
> `is_chunk_dirty`/`has_changes`; the `taffy` perf-debug helpers; the redundant
> `WgpuSurfaceHandle::resize` (the element uses `request_resize`/`defer_resize`);
> the unread `WindowInvalidator::update_count`; and a stale `allow(dead_code)` on
> `Priority::probability`. Also removed the redundant private smooth-scroll
> duplicate helpers (the feature is wired via the list paint's direct
> `smooth_scroll.update()`/`current()`), the never-read accessibility-mode flag
> (no accesskit integration exists), and the dead `WgpuSurfaceRegistry::size`/
> `format` duplicates of the public `WgpuSurfaceHandle` API. The only remaining
> `expect/allow(dead_code)` are **not scaffolding**: GPU-resource RAII
> (`atlas`/`renderer`/`surface_registry` triple-buffer `textures`), cross-platform
> backend stubs (`platform/mod.rs`), and executor queue internals — each kept
> because removal would break resource lifetime or non-Linux builds.
>
> **Multi-crate render-primitive plugin system (this session — user-authorized
> override of the maintainer's "hold off"):** new `crates/gpui-render-primitive`
> (the `RenderPrimitive` trait + `PrimitiveRegistry` + per-type `PrimitiveBatches`)
> makes primitive types extensible from their own crates. Zero runtime overhead via
> **per-batch, not per-instance, dispatch**: instances are appended as raw `Pod`
> bytes (a memcpy) and drawn one batched call per type. `crates/gpui-prim-solid` is
> an example primitive living in its own crate, depending only on the api crate
> (editing it recompiles only it). The core re-exports the api, collects instances
> via `Window::paint_custom_primitive` into `Scene::custom_primitives`, and reports
> `FrameMetrics::custom_primitive_count`; an end-to-end test and a bench scenario
> (50/500/2000 instances) cover the collection path headlessly.
>
> **GPU draw + registration wired in (this session):** the renderer now ends its
> final pass with a guarded `PrimitiveRegistry::draw_batch` per collected type
> (one batched GPU call each), passing the device/queue/surface-format/globals
> so plugins build their pipeline lazily and cache it. The draw is **additive and
> guarded** by `if !scene.custom_primitives.is_empty()`, so a frame with no plugin
> primitives is byte-identical to before. Registration flows
> `Window::register_render_primitive` → `PlatformWindow::register_render_primitive`
> (default no-op; test platform drops it) → cross window (registers on the live
> renderer, or buffers in `pending_primitives` and flushes the instant the renderer
> is created) → `WgpuRenderer::register_primitive` → `PrimitiveRegistry`. REMAINING
> (maintainer, display-side): pixel-level confirmation that registered plugin
> primitives paint correctly on a real GPU/surface — the CPU collection + dispatch
> wiring is complete and tested headlessly.
>
> **Deferred per maintainer ("hold off for now"):** a future retained render-node
> layer that genuinely drives partial redraw (to be built as real code,
> display-validated, when that work is scheduled).
>
> **Correctness fixes this session:** FiberId allocation collision, unbounded
> `mark_dirty` ancestor walk, and the intrinsic/measurement `known_dimensions`
> cache-collision are fixed with regression tests.
>
> **Display QA:** not re-run this session (no display available); the dramatic
> renderer change and the overlay are explicitly validated by the maintainer
> against the examples and the local game-engine UI before merge.
>
> **Headless perf bench (this session):** `benches/render_pipeline.rs`
> (`cargo bench --features test-support --bench render_pipeline`) drives real
> frames through the test platform (GPU present is a no-op) and reports per-frame
> CPU cost (p50/min/mean/max) for cold draw, clean redraw (view reuse), dirty
> rebuild, and overlay, across 100/500/1500 rows, plus the scene/retained metrics.
> It quantifies the perf characteristics headlessly — e.g. view-reuse reaches
> ~3.7x at 1500 rows, while the retained carry-forward's per-frame reconcile cost
> (GlobalElementId allocations) makes it a slight loss for very small trees and is
> a clear optimization target. This is the quantitative complement to the
> maintainer's on-display GPU/visual validation.
>
> **Implemented default-on policy:** persistent-framebuffer swapchain copy is wired
> for full frames, and partial redraw/scissor defaults on while still falling back
> to full redraw whenever framebuffer/damage invariants are not satisfied.
>
> **Full parity target:** retained-fiber rendering and partial redraw should become
> the default path after the tests in `.omo/plans/rendering-fiber-parity.md` prove
> retained identity, cached replay, scene segments, transforms, and damage/framebuffer
> invariants. Wayland-specific `damage_buffer` is still not a separate backend path;
> any platform lesson must flow through WGPUI's unified `wgpu` + `winit` renderer.
>
> **Retained-fiber parity decision:** implement the full retained architecture in
> this branch, but adapt it to WGPUI's single-crate `wgpu`/`winit` backend instead
> of blindly copying Zed's split platform stack. Preserve compatibility shims for
> existing GPUI-facing APIs where technically possible; if an API cannot be
> retained, prove the migration with tests and notes.
>
> Next step: finish final review synthesis, then validate against downstream
> real applications, including the local game-engine UI, and expand the
> feature-gated refresh profiler into an interactive overlay with gradient legend
> and shortcut binding.



Plan to bring the fiber / partial-redraw / persistent-GPU-buffer work that was
prototyped in **coleleavitt/WGPUI** (a 14-crate workspace fork) into
**Far-Beyond-Pulsar/WGPUI** (this repo — a single `gpui-ce` crate), **keeping
Pulsar's single-crate structure**. Reference issue: Far-Beyond-Pulsar/WGPUI#4
("Perf improvements via fiber rendering"). Maintainer's stated bottleneck:
*"the whole UI must refresh each time"* (canvas/full-frame refresh).

> Status: LIVE FULL-PARITY PORT. The safe bridge is already implemented in stages:
> committed phases are listed above; current working-tree changes cover Phase 3
> dirty-range uploads, Phase 5 row identity caching, Phase 7 compaction,
> persistent-framebuffer swapchain copy, default-on partial redraw/scissor with
> correctness fallbacks, retained scene segment identity, transform-scoped
> damage tests, retained child-node traversal, UpdateResult-driven clean replay,
> hitbox/focus/click replay, WGPU-surface primitive replay, `FiberTree` identity
> reconciliation, retained lifecycle payload ownership, exact measured-layout
> cache integration, WGPU-surface cache pruning, and feature-gated refresh
> profiling scaffolding. The
> active target is now retained-fiber parity with default-on retained/partial
> rendering after tests prove the invariants.

---

## 1. The two repos

| | coleleavitt/WGPUI (reference) | Far-Beyond-Pulsar/WGPUI (target) |
|---|---|---|
| Structure | 14-crate workspace (`gpui`, `gpui_wgpu`, `gpui_linux`, …) | **single crate `gpui-ce`** |
| Core gpui | `gpui/src/` | `src/gpui/` |
| wgpu backend | `gpui_wgpu/src/` (`wgpu_renderer.rs`, `wgpu_atlas.rs`) | `src/gpui/platform/cross/` (`renderer.rs`, `atlas.rs`) |
| Perf work | Phases 0–6 + buffer compaction + Wayland damage | Safe bridge now present: Phases 0/1/2/3/4-layout/5/6/7 plus retained scene segment/transform tests and default-on persistent-framebuffer partial redraw with fallbacks |
| Outcome | **partial redraw DISABLED at HEAD** (`561980f`, "until scene chunk tracking is reliable") | Retained/partial policy is default-on with per-frame framebuffer/damage safety fallbacks |

**Path mapping for the port:** `gpui/src/<x>` → `src/gpui/<x>`;
`gpui_wgpu/src/<x>` → `src/gpui/platform/cross/<x>`. Cole's files are larger
(scene.rs 922 / window.rs 5919 / wgpu_renderer.rs 1956 lines) than Pulsar's, but
same lineage, so changes translate; line numbers do **not**.

---

## 2. Historical baseline and current residual gaps

```
on_request_frame (vsync)                       platform/cross/platform.rs:1137
  → Window::draw(cx)                            src/window.rs
      → invalidate_entities()                   collect dirty views
      → draw_roots: Prepaint then Paint         re-renders the whole tree
      → next_frame.finish()                     sort ALL primitives by order
      → swap(rendered_frame, next_frame); next_frame.clear()   wipes EVERYTHING
  → Window::present()
      → platform_window.draw(&scene)            src/platform/cross/renderer.rs:1622
          → queue.write_buffer(globals)
          → queue.write_buffer(instances)       ALL instance data, every frame
          → per-type batch draws; submit; present
```

### The three core problems (and the current working-tree status)
1. **Full scene rebuild every frame** — historical problem. The working tree now
   records per-view `SceneChunk`s (`src/scene.rs:41`), reuses view paint ranges
   from `AnyViewState` (`src/view.rs:220`), and computes changed ranges before
   clearing dirty flags (`src/scene.rs:368`). The residual gap is that WGPUI is
   still not a fully retained fiber tree: custom elements still execute the
   existing `Element::request_layout` / `prepaint` / `paint` lifecycle, and there
   is no persistent `RenderNode` layer.
2. **Full GPU re-upload every frame** — mostly addressed in the safe path.
   `WgpuRenderer::upload_primitive_buffer` (`src/platform/cross/renderer.rs:2033`)
   skips unchanged generation+size pairs, writes dirty byte ranges when size is
   stable, records empty changes without writing, and falls back to a full upload
   on size changes. The remaining parity gap is segment-level GPU caching and
   transform-buffer diffing, not per-type buffer persistence.
3. **No damage scoping** — scene-side damage exists (`src/scene.rs:432`) and the
   renderer can choose a partial `LoadOp::Load` scissored pass
   (`src/platform/cross/renderer.rs:1602`), but it is still explicitly gated by
   `GPUI_PARTIAL_REDRAW=1`, a matching persistent framebuffer, and small damage.
   Default behavior remains full clear/full draw for correctness.

### Foundation already in WGPUI (build on this, don't reinvent)
- **View dirty tracking:** `mark_view_dirty()` clears the view's layout cache,
  marks the view dirty, and walks rendered ancestors into `dirty_descendants`
  (`src/window.rs:1554`).
- **View/layout caching:** `AnyView::request_layout` reuses
  `cached_layout_for_view` for clean views and stores new layout ids afterward
  (`src/view.rs:184`, `src/window.rs:3818`). This is layout reuse, not a full
  retained-node tree.
- **Paint reuse and chunking:** `AnyView::paint` wraps reused or freshly painted
  view output in `begin_view_chunk` / `end_view_chunk` (`src/view.rs:289`), so
  scene chunks and changed ranges align with view paint reuse.
- **Per-type GPU buffers:** `render_context.rs` owns the persistent buffers and
  compacts/reallocates them when more than 30% unused (`src/platform/cross/render_context.rs:196`).
- **Demand-driven redraw** *(already landed on branch `perf/idle-redraw`)* —
  removes the idle busy-loop so the loop sleeps unless dirty; this is the idle
  half of "whole UI refreshes each time".

The remaining gap is no longer "add scene chunks / persistent buffers". It is
the larger retained-pipeline gap: stable per-element identities, retained render
nodes, persistent scene segments, transform-scoped cached primitives, and
compatibility fallbacks for legacy/custom `Element` implementations.

---

## 3. Cole's phased approach (and how each ended)

| Phase | Commit | What it did | Stability |
|---|---|---|---|
| 0 Instrumentation | `d233746` | frame-timing, scene-size/upload counters, benchmarks | ✅ stable, foundational |
| 1 SceneChunk | `74edbb0` | per-view scene chunks + incremental sort | ✅ but reliability is the crux (see §4) |
| 2 Damage regions | `c7285ff` | track changed rects, scissor the pass | ⚠ couples to partial redraw |
| 3 Persistent GPU buffers | `2cc240ef` | per-type buffers + diff uploads (only changed ranges) | ✅ the big GPU win, mostly independent |
| 4 Fiber partial redraw + layout cache | `803e921` | skip re-render of unchanged views | ❌ **disabled at HEAD** — unstable until chunk tracking is reliable |
| 5+6 List/element cache + batch merge | `88b2aa7` | cache list items, merge batches | ✅ big win for large lists |
| 7 Buffer compaction | `604c0d0` | compact persistent buffers at >30% fragmentation | ✅ depends on Phase 3 |
| Wayland damage | `fc12aee`→revert→reapply | `damage_buffer` + opaque region on resize | ⚠ churned; Linux-specific |

**Key lesson (from commit history + `.sisyphus` notes):** the *fiber partial
redraw* (Phase 4) is only safe once *SceneChunk tracking* (Phase 1) is provably
reliable — cole shipped with Phase 4 **off**. The persistent-GPU-buffer win
(Phase 3) and list/batch caching (5+6) are largely **independent** of the risky
partial-redraw and deliver most of the GPU savings on their own.

---

## 4. Strategy from here: implement full retained parity, tests first

The safe bridge is already implemented in this working tree. The strategy now is
to build the retained architecture on top of it and make retained/partial rendering
the default once tests prove the invariants:

1. **Red tests first.** Add tests for retained identity, dirty ancestor propagation,
   cached prepaint/paint replay, scene segment lifecycle, transform-only scroll,
   measurement cache invalidation, legacy render-node fallback, and default-on
   partial redraw decisions.
2. **Retained core.** Port/adapt `FiberTree`, dirty flags, render node trait,
   legacy fallback, cached payloads, measurement cache, and intrinsic sizing into
   cohesive WGPUI modules.
3. **Scene/transform integration.** Add stable scene segments and transform IDs
   behind WGPUI's primitive set (`backdrop_blurs`, no `subpixel_sprites`).
4. **Window/element integration.** Wire retained reconciliation, layout,
   prepaint, paint, hitbox/focus, deferred draw, text, canvas, list, and
   uniform-list behavior through the current public API surface with shims where
   possible.
5. **Renderer default-on policy.** Make retained/partial redraw the default path,
   with per-frame correctness fallbacks when target size, framebuffer availability,
   damage, or backend capability invariants fail.
6. **Verification.** Run Rust gates, downstream/examples, and display QA where a
   display is available. Lack of display blocks only display evidence, not the
   implementation itself.

`Integrate + gate` for the safe bridge remains: WGPUI `cargo build`,
`./script/clippy`, `cargo test --workspace --all-targets`, downstream inspector
build, and real display validation before marking display-sensitive work done.

### Avoid while still implementing everything
- **Do not use `should_skip_view` as full subtree skip by itself.** It is currently
  a layout-cache guard. Full skip must come from retained fibers with cached
  prepaint/paint payloads and legacy fallback.
- **Do not silently break GPUI-facing APIs.** Keep shims for `AnyView::cached()`,
  `Stateful`, `canvas()`, `deferred()`, wrapper elements, text constructors, and
  `Window::draw()` behavior where feasible, even if retained internals become the
  default.
- **Do not add a Wayland-specific renderer fork.** Use platform references to make
  WGPUI's unified `wgpu`/`winit` backend correct across backends.

---

## 5. Per-file status map (target = WGPUI `src/`)

> **The symbol-level map is in Appendix A** (from the analysis pass). High-level
> summary below; Appendix A has the per-phase source→target file/symbol mapping,
> portability/stability per phase, the refined tiered order, the exact HEAD-bug
> root cause, and the 5 riskiest adaptations. **Two findings change the plan:**
> (a) WGPUI's renderer **already has** per-type persistent buffers
> (`render_context.rs`) and a wired full-frame `persistent_framebuffer` copy —
> so Phase 3 / `66ee` are *adaptations of existing infra*, not new infra; (b)
> a **primitive-set delta**
> (source has `subpixel_sprites` / no `backdrop_blurs`; Pulsar is the reverse)
> must be threaded through every per-type change.

- **Phase 0** → `src/profiler.rs`, `src/window.rs`,
  `src/platform/cross/renderer.rs`: frame metrics are present; keep using them
  for before/after safe-bridge validation.
- **Phase 1** → `src/scene.rs`, `src/view.rs`, `src/window.rs`: per-view
  `SceneChunk`s, generation, changed ranges, incremental sort, and view paint
  chunking are present.
- **Phase 2** → `src/scene.rs`, `src/platform/cross/renderer.rs`: scene-side
  damage rects and renderer scissor mode are present, but renderer use remains
  gated and default-off.
- **Phase 3** → `src/platform/cross/renderer.rs`,
  `src/platform/cross/render_context.rs`: per-type persistent buffers, dirty byte
  ranges, full-upload fallback on size change, and empty-write record-only
  behavior are present.
- **Phase 4** → `src/window.rs`, `src/view.rs`: view layout-cache reuse is
  active. Full retained subtree skip is **not** present and must not be inferred
  from `should_skip_view` alone.
- **Phase 5+6** → `src/elements/list.rs`, `src/elements/uniform_list.rs`,
  `src/scene.rs`: list/uniform-list identity caches and batch counting/iteration
  hardening are present; still display-validate with downstream list churn.
- **Phase 7** → `src/platform/cross/render_context.rs`: buffer compaction is
  present via buffer recreation when more than 30% unused.

---

## 6. Verification

Implementation is not complete until behavior is proven by tests and, where the
environment allows it, real rendering:

- **Red tests first:** retained identity, dirty ancestor propagation, cached
  prepaint/paint replay, scene segment lifecycle, transform-only scroll,
  measurement cache invalidation, legacy fallback, and default-on renderer policy.
- **Agent side:** `cargo fmt --all`, `./script/clippy`,
  `cargo test --workspace --all-targets`, and targeted example builds.
- **Display side:** run representative examples or the downstream inspector and
  verify draw, scroll, text input, focus, IME, animation, resize, list churn,
  `uniform_list` y-flipped reuse, backdrop blur, persistent framebuffer copy, and
  embedded `WgpuSurface` behavior. If no display is available, record the exact
  blocker and do not claim display validation.

Definition of done for full parity: retained fibers are default-on, partial redraw
is default-on with correctness fallbacks, compatibility shims or explicit migration
tests cover existing public APIs, all Rust gates pass, and display/manual QA is
either completed or blocked with exact evidence.

---

## 7. Open risks
1. **Partial redraw correctness** — the exact reason cole disabled it. Needs the
   chunk-tracking invariant to hold across layered draws, deferred draws, and
   atlas tile reuse. Highest risk; gated.
2. **Atlas tile reuse vs caching** — cached views hold `AtlasTile` UVs; combined
   with glyph-atlas eviction this can replay stale tiles (separate known issue).
   Don't combine atlas eviction with partial redraw blindly.
3. **Structural translation** — cole's multi-crate code references crate paths
   that collapse into Pulsar's single crate; watch `pub(crate)` visibility.
4. **wgpu fork divergence** — both pin different custom wgpu forks; buffer/bind
   APIs may differ in Phase 3.
5. **No display in CI/agent** — headless tests prove invariants, but display
   confidence still requires actual example/downstream runs.

---

## Appendix A — Detailed port map (symbol-level)

### Structural reality
- Core gpui files port ~1:1: `scene.rs`, `window.rs`, `view.rs`, `element.rs`
  line up symbol-for-symbol (cole `gpui/src/*` → Pulsar `src/*`).
- **Renderer diverged.** Pulsar already had, in `src/platform/cross/`:
  per-type persistent buffers in `render_context.rs` (`quads_buffer`,
  `shadows_buffer`, `underlines_buffer`, `mono_sprites_buffer`,
  `poly_sprites_buffer`, `backdrop_blurs_buffer`, `paths_vertices_buffer`, each
  `Mutex<wgpu::Buffer>`). The current safe path adds generation/range upload
  guards, dirty-range writes with full-upload fallback, buffer compaction, and a
  full-frame persistent-framebuffer swapchain copy to that infrastructure.
- **Primitive-set delta (applies to every per-type change):** source has
  `subpixel_sprites` and NO `backdrop_blurs`; Pulsar has `backdrop_blurs` and NO
  `subpixel_sprites`. When porting `SceneChunk` fields / `ChangedRanges` /
  `chunk_bounds` / persistent buffers / `BatchCounter` / sort keys: **drop
  subpixel, add backdrop_blurs**. An omitted `backdrop_blurs` range = blurs never
  marked dirty = stale/missing blur once partial paths are live.
- Imports: Pulsar uses `collections::{FxHashMap,FxHashSet}` (not `rustc_hash`).
- No `perf` crate in Pulsar → Phase-0 metrics go in `src/profiler.rs`
  (re-exported via `src/gpui.rs`). No `crates/perf`, no Wayland surface API.

### Port-map table

| Phase / commit | SOURCE | TARGET | Portability | Stability |
|---|---|---|---|---|
| 0 `d233746` Instrumentation | `perf/`, `gpui/window.rs`, `gpui_wgpu/wgpu_renderer.rs`, `perf_overlay.rs`, `examples/bench_render.rs` | `src/profiler.rs`, `src/window.rs`, `src/platform/cross/renderer.rs`, `src/elements/perf_overlay.rs`(new) | NEEDS-ADAPTATION | Stable |
| 1 `74edbb0` SceneChunk+incr sort | `gpui/scene.rs`,`view.rs`,`window.rs` | `src/scene.rs`,`src/view.rs`,`src/window.rs` | CLEAN→ADAPT (prim-set) | DS stable; dirty-tracking = HEAD-bug root |
| 2 `c7285ff` Damage+scissor | `gpui/scene.rs`,`gpui_wgpu/wgpu_renderer.rs` | `src/scene.rs`,`src/platform/cross/renderer.rs` | NEEDS-ADAPTATION | Renderer path UNSTABLE — gate OFF |
| 3 `2cc240e` Persistent buf+diff upload | `gpui_wgpu/wgpu_renderer.rs` | `src/platform/cross/render_context.rs`+`renderer.rs` (buffers EXIST) | ADAPTED SAFE SUBSET | Per-type dirty-range writes + full-upload fallback; needs display validation |
| 4 `803e921` Fiber skip+layout cache | `gpui/window.rs`,`view.rs`,`element.rs` | same `src/*` | PARTIAL ONLY | Per-view layout cache landed; `should_skip_view`/full retained tree remain gated |
| 5+6 `88b2aa7` List cache+batch merge | `gpui/scene.rs`,`elements/uniform_list.rs`,`list.rs`,`window.rs`,`wgpu_renderer.rs` | same-named `src/*` | NEEDS-ADAPTATION (uniform_list sig; prim-set) | **Stable, safest wins** |
| `604c0d0` Compaction | `gpui_wgpu/wgpu_renderer.rs` | `render_context.rs`+`renderer.rs` | ADAPTED SAFE SUBSET | Recreate buffers when more than 30% unused |
| `fc12`/`2b11` Wayland damage | `gpui_linux/wayland/window.rs` | **NO TARGET** (winit) | SKIP | Churned (revert+reapply) |
| `66eea08` Persistent frame tex | `gpui_wgpu/wgpu_renderer.rs` | `renderer.rs` (`persistent_framebuffer` existed) | ADAPTED GATED | Full-frame copy wired; partial redraw still opt-in |
| `561980f` (HEAD) Disable partial redraw | `gpui_wgpu/wgpu_renderer.rs` | `renderer.rs` | CLEAN | **Canonical safe config** |

### Current execution order (supersedes the old port order)
- **Tier 1 (already in tree):** Phase **5+6** list/uniform-list identity caching
  and batch-counting/iteration hardening; Phase **0** instrumentation.
- **Tier 2 (already in tree):** Phase **1** `SceneChunk` + `generation` +
  incremental sort, with full-sort fallback.
- **Tier 3 (already in tree, needs display validation):** Phase **4** view
  layout-cache reuse. `should_skip_view` is active only as a layout-cache guard;
  do not expand it into full subtree skip without retained-node fallback work.
- **Tier 4 (already in tree, needs display validation):** Phase **3** dirty-range
  upload safe subset: bind full buffers, write dirty ranges only when size is
  stable, and fall back to full upload on size changes.
- **Implementation target:** retained Zed fiber-tree parity, segment GPU cache,
  transform-buffer diff uploads, and compatibility-shimmed API migrations are in
  scope for this full-parity work.
- **No platform fork:** Wayland-only damage remains a reference, not a separate
  backend path; WGPUI keeps one `winit`/`wgpu` renderer.

### Why partial redraw needs retained invariants before default-on
Historical root cause from the reference line: **`dirty_views` (window) can
desync from `SceneChunk.dirty` / damage (scene)**. If a view calls `cx.notify()`
but the scene path yields empty damage, a partial-redraw renderer can preserve the
old framebuffer and make newly typed text or changed rows invisible.

WGPUI's current safe bridge avoids that failure by defaulting to full clear/full
draw and by making `render_pass_mode` fall back to `FullClear` unless partial
redraw is explicitly enabled, a persistent framebuffer exists, full redraw is not
needed, and damage produces a valid scissor. Full parity must keep those
correctness fallbacks while flipping the normal path to retained/partial by
default after tests prove keystrokes and list churn produce coherent dirty flags,
changed ranges, damage rects, and visible framebuffer updates every frame.

### 5 riskiest remaining validation/adaptation points
1. **Phase 3 dirty-range writes on persistent buffers** — already adapted onto
   WGPUI's `Mutex<Buffer>` infrastructure. Validate that range writes, record-only
   empty changes, and full-upload fallback behave under resize, atlas churn, and
   primitive count changes.
2. **Primitive-set mismatch** threaded through every safe path (drop subpixel,
   add backdrop_blurs everywhere). Missing `backdrop_blurs` in `SceneChunk`,
   `ChangedRanges`, damage, upload, or batching means stale/missing blur once
   partial paths are live.
3. **Persistent framebuffer copy** — now wired as a full-frame copy from the
   offscreen target to the swapchain after rendering; validate black/garbled
   window risks on resize, target-size mismatch, backdrop blur, and transparency.
4. **Phase 4 layout-cache invalidation** across resize/scroll/animation/focus/IME
   — `mark_view_dirty` clears per-view layout cache, but real-window validation is
   required before treating layout reuse as fully safe.
5. **List/uniform-list identity caches** — tests cover cache id math, but display
   validation must cover append/insert/remove/reset/reload, focus retention,
   smooth scrolling, and y-flipped `uniform_list` reuse.

---

## Appendix B — Zed fiber-tree parity audit (`/tmp/zed-gpui-fiber`)

The sparse Zed reference checkout was initialized from `maddythewisp/zed:fiber`
with `crates/gpui` and `crates/gpui_macros` indexed by CodeGraph. Compared with
`zed-industries/zed:main`, the GPUI-side fiber branch changes 68 files with
about 28.7k inserted lines and 6.7k deleted lines. Treat it as an architecture
branch, not a renderer patch.

### What Zed fiber actually adds

- **Persistent fiber runtime:** `FiberTree` stores fibers, dirty flags, retained
  render nodes, children, view roots, cached paint output, cached hitbox state,
  cached layout state, cached effects/listeners, scene segments, per-fiber
  element state, focus ids, and active overlay roots. This subsumes much of the
  current per-frame dispatch/hitbox/layout bookkeeping.
- **Retained render-node layer:** `RenderNode` provides layout/prepaint/paint
  begin/end hooks, intrinsic sizing hooks, child-bound requirements, focus and
  interactivity capabilities, and a `LegacyNode` fallback for non-retained
  elements.
- **Persistent scene segments:** `SceneSegmentPool` allocates stable segment ids
  and owns a `TransformTable`, so cached paint output can be replayed and moved
  by transform changes without regenerating primitives.
- **Sizing infrastructure:** `MeasurementCache`, `IntrinsicSize`, `SizingInput`,
  `SizingCtx`, and `SizeQuery` make layout measurement cacheable and comparable
  across frames.
- **Macro/API plumbing:** `derive_fiber_element` emits `Element` impls whose
  legacy `request_layout` / `prepaint` / `paint` methods are unreachable because
  the retained-node path owns those phases.

### Why it conflicts with the current WGPUI branch

WGPUI's current public contract is still the GPUI-compatible `Element` lifecycle:
`request_layout`, `prepaint`, and `paint` methods run on custom elements, and
applications can use the existing `AnyView::cached()`, `Stateful`, `canvas()`,
`deferred()`, wrapper elements, text constructors, and `Window::draw()` behavior.
Zed's PR description intentionally migrates several of those APIs: automatic view
caching replaces `AnyView::cached()`, `window.clear()`/`ArenaClearNeeded` change,
`Stateful` disappears, `canvas()` paint receives `&mut T`, `deferred()` becomes
`.z_index()`, wrapper elements become fluent methods, and text construction moves
to a `TextElement` trait. Those are reasonable in a retained architecture, but
they are not safe to include in the same PR as WGPUI's renderer upload/damage
bridge.

### Compatibility-first full-parity order

1. **Legacy fallback and identity bridge.** Introduce a retained identity shell
   that can host current `Element` implementations as always-dirty legacy nodes.
   Acceptance gate: all existing examples and downstream apps compile without API
   changes.
2. **Scene segment bridge.** Add stable segment IDs behind the current `Scene`
   representation before changing public element contracts. Acceptance gate:
   cached/replayed segments produce the same draw order, content masks, deferred
   draw ordering, and hit testing as the current scene.
3. **Transform table behind current primitives.** Add transform IDs only if they
   can move cached primitives without changing `Element` APIs. Acceptance gate:
   scroll, clipped content masks, hitboxes, and embedded `WgpuSurface` bounds stay
   correct.
4. **Retained nodes element-by-element.** Convert built-in elements one at a time
   to `RenderNode`-like retained nodes while custom elements keep the legacy path.
   Acceptance gate: each conversion has old/new behavior parity and can be
   disabled or bypassed if a downstream custom element depends on legacy timing.
5. **Sizing/measurement cache.** Add intrinsic sizing only after retained node
   lifecycles are stable. Acceptance gate: text wrapping, image sizing, list
   measurement, and resize behavior match the legacy path.
6. **API migrations with shims.** Public migrations like `canvas()` state sharing
   or `deferred()` → `.z_index()` are in scope, but old APIs should remain as
   shims when technically feasible. If a shim is impossible, add a compile/runtime
   test proving the new behavior and document the migration.

### Full-parity implementation requirements

1. Retained fibers own stable identity, dirty flags, cached layout/prepaint/paint
   payloads, hitbox/focus/effect state, and scene segment lists.
2. Legacy/custom elements have a correctness-first fallback so existing apps keep
   rendering while built-in elements become retained nodes.
3. Scene segments and transforms are integrated with WGPUI's existing primitive
   set and renderer buffers.
4. Default-on retained/partial rendering uses correctness fallbacks for empty
   damage, framebuffer mismatch, resize, backend limitations, and first-frame
   conditions.
5. Tests and display QA cover text input, focus, IME, scroll, resize, animation,
   backdrop blur, `WgpuSurface`, and list/uniform-list churn.
