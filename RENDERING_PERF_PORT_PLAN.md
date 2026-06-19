# WGPUI Rendering-Perf Port Plan (single-crate Pulsar)

Plan to bring the fiber / partial-redraw / persistent-GPU-buffer work that was
prototyped in **coleleavitt/WGPUI** (a 14-crate workspace fork) into
**Far-Beyond-Pulsar/WGPUI** (this repo — a single `gpui-ce` crate), **keeping
Pulsar's single-crate structure**. Reference issue: Far-Beyond-Pulsar/WGPUI#4
("Perf improvements via fiber rendering"). Maintainer's stated bottleneck:
*"the whole UI must refresh each time"* (canvas/full-frame refresh).

> Status: PLAN. No renderer changes landed yet. The phase tasks (#16–#23) track
> the work. A background analysis is producing a finer per-file port map to
> append to §5.

---

## 1. The two repos

| | coleleavitt/WGPUI (reference) | Far-Beyond-Pulsar/WGPUI (target) |
|---|---|---|
| Structure | 14-crate workspace (`gpui`, `gpui_wgpu`, `gpui_linux`, …) | **single crate `gpui-ce`** |
| Core gpui | `gpui/src/` | `src/` |
| wgpu backend | `gpui_wgpu/src/` (`wgpu_renderer.rs`, `wgpu_atlas.rs`) | `src/platform/cross/` (`renderer.rs`, `atlas.rs`) |
| Perf work | Phases 0–6 + buffer compaction + Wayland damage | none (baseline) |
| Outcome | **partial redraw DISABLED at HEAD** (`561980f`, "until scene chunk tracking is reliable") | — |

**Path mapping for the port:** `gpui/src/<x>` → `src/<x>`;
`gpui_wgpu/src/<x>` → `src/platform/cross/<x>`. Cole's files are larger
(scene.rs 922 / window.rs 5919 / wgpu_renderer.rs 1956 lines) than Pulsar's, but
same lineage, so changes translate; line numbers do **not**.

---

## 2. Pulsar's current pipeline (baseline)

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

### The three core problems (same as cole's analysis)
1. **Full scene rebuild every frame** — `Frame::clear()` (`window.rs:813-824`)
   wipes `scene` + all element/hitbox/dispatch state; the whole element tree is
   re-prepainted/painted and `scene.finish()` re-sorts everything.
2. **Full GPU re-upload every frame** — `renderer.draw()` `write_buffer`s all
   instance data regardless of what changed (`renderer.rs:~1650-1696`).
3. **No damage scoping** — the main pass is `LoadOp::Clear` over the whole
   framebuffer, then a full draw, even for a one-pixel change.

### Foundation already in Pulsar (build on this, don't reinvent)
- **View dirty tracking:** `WindowInvalidator { dirty, dirty_views }`
  (`window.rs:101-148`), `mark_view_dirty()` walks ancestors (`window.rs:1439`).
- **View caching:** `AnyView.cached_style`, `AnyViewState { prepaint_range,
  paint_range, cache_key }`, `ViewCacheKey { bounds, content_mask, text_style }`.
- **Paint reuse:** `reuse_prepaint`/`reuse_paint` with cached `PrepaintStateIndex`
  / `PaintIndex` ranges (`window.rs:2452, 2511`) — used today for deferred draws.
- **Demand-driven redraw** *(already landed on branch `perf/idle-redraw`)* —
  removes the idle busy-loop so the loop sleeps unless dirty; this is the idle
  half of "whole UI refreshes each time".

The gap cole filled is the **scene + GPU layer**: extend the existing
view-level reuse down into scene chunking, persistent buffers, and damage so a
non-dirty view costs ~0 CPU/GPU instead of being re-copied and re-uploaded.

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

## 4. Strategy: land the safe wins first, gate the risky one

Order chosen to maximize value while isolating the known instability:

1. **Phase 0 (instrument)** — land first. Gives before/after numbers so every
   later phase is measured, not guessed. Low risk. *(task #16)*
2. **Phase 3 (persistent GPU buffers + diff uploads)** — biggest GPU win,
   independent of partial redraw, directly answers Trident's "avoid full
   per-frame re-uploads". Land before the fiber work. *(task #19)*
3. **Phase 5+6 (list/element cache + batch merge)** — big CPU win for large
   lists (the axon-mind transcript/file tree), low risk. *(task #21)*
4. **Phase 1 (SceneChunk tracking)** — the prerequisite for safe partial redraw;
   land + validate its correctness on its own (no behavior change yet). *(task #17)*
5. **Phase 2 (damage + scissor)** — once chunks are trusted. *(task #18)*
6. **Phase 4 (fiber partial redraw)** — port **behind a default-OFF feature
   flag / runtime toggle**; only flip on after Phase 1 chunk tracking is proven
   reliable on a real display. This is where cole's version broke. *(task #20)*
7. **Phase 7 (buffer compaction)** — after Phase 3. *(task #22)*

`Integrate + gate` (task #23) runs after every phase: WGPUI `cargo build` +
`cargo clippy` green (the crate denies `redundant_clone` etc.), and the
downstream **axon-mind inspector still builds** against it.

### Keep DISABLED / avoid (from cole's outcome)
- **Do not enable fiber partial redraw (Phase 4) by default.** Ship it off;
  gate behind a flag until Phase 1 is validated. This is the documented
  instability ("disable partial redraw until scene chunk tracking is reliable").
- Treat the **Wayland damage_buffer** change as Linux-only and optional — it was
  reverted/reapplied (churn). Land it last, isolated.

---

## 5. Per-file port map (target = Pulsar `src/`)

> **The symbol-level map is in Appendix A** (from the analysis pass). High-level
> summary below; Appendix A has the per-phase source→target file/symbol mapping,
> portability/stability per phase, the refined tiered order, the exact HEAD-bug
> root cause, and the 5 riskiest adaptations. **Two findings change the plan:**
> (a) Pulsar's renderer **already has** per-type persistent buffers
> (`render_context.rs`) *and* a `persistent_framebuffer` (offscreen) with the
> swapchain blit stubbed at `renderer.rs:2342` — so Phase 3 / `66ee` are
> *adaptations of existing infra*, not new infra; (b) a **primitive-set delta**
> (source has `subpixel_sprites` / no `backdrop_blurs`; Pulsar is the reverse)
> must be threaded through every per-type change.

- **Phase 0** → `src/platform/cross/renderer.rs` (frame counters), `src/window.rs`
  (`measure("frame duration", …)` already exists — extend it), maybe a small
  `src/scene.rs` size counter. Optionally a `benches/` harness.
- **Phase 1** → `src/scene.rs` (add per-view chunk ranges + incremental sort;
  Scene currently has *no* change tracking), `src/window.rs` (feed dirty_views
  into chunk invalidation at `draw_roots`).
- **Phase 2** → `src/platform/cross/renderer.rs` (scissor rects on the main
  pass instead of full-frame `LoadOp::Clear`), `src/scene.rs` (damage rects).
- **Phase 3** → `src/platform/cross/renderer.rs` (`draw()` /
  `write_to_instance_buffer`: per-type persistent buffers + dirty-range diff
  uploads) and `src/platform/cross/render_context.rs` (`ensure_buffer_size`).
- **Phase 4** → `src/window.rs` (view-level skip in `draw_roots`/reuse path) +
  `src/view.rs`-equivalent caching; **flag-gated**.
- **Phase 5+6** → `src/elements/list.rs`, `src/elements/uniform_list.rs`,
  `src/scene.rs` (batch merge in `batches()`).
- **Phase 7** → `src/platform/cross/renderer.rs` / `render_context.rs`.

---

## 6. Verification (the headless constraint)

The agent that authored this cannot open a window (no display), so **runtime
validation is yours** on the ThinkPad. Per phase:
- Agent side: `cargo build` + `cargo clippy` green; reasoned-correct; downstream
  axon-mind inspector builds.
- Your side: run a representative app (the axon-mind inspector or a WGPUI
  example), compare Phase-0 frame-time / upload-byte counters before vs after,
  and check visual correctness (esp. when toggling the Phase-4 flag).

Definition of done for the issue: Phases 0,3,5+6 landed + measured win; Phases
1/2 landed and validated; Phase 4 present but off-by-default with a documented
toggle and a green light only after chunk tracking proves reliable.

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
5. **No display in CI/agent** — everything is compile-validated only until you
   test.

---

## Appendix A — Detailed port map (symbol-level)

### Structural reality
- Core gpui files port ~1:1: `scene.rs`, `window.rs`, `view.rs`, `element.rs`
  line up symbol-for-symbol (cole `gpui/src/*` → Pulsar `src/*`).
- **Renderer diverged.** Pulsar already has, in `src/platform/cross/`:
  per-type persistent buffers in `render_context.rs` (`quads_buffer`,
  `shadows_buffer`, `underlines_buffer`, `mono_sprites_buffer`,
  `poly_sprites_buffer`, `backdrop_blurs_buffer`, `paths_vertices_buffer`, each
  `Mutex<wgpu::Buffer>`) doing **full reupload** via `ensure_buffer_size` +
  `write_buffer(..,0,data)`; and a `persistent_framebuffer` (+`_view`) offscreen
  target whose **swapchain blit is a TODO** (`renderer.rs:2342`, `:2547`).
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
| 3 `2cc240e` Persistent buf+diff upload | `gpui_wgpu/wgpu_renderer.rs` | `src/platform/cross/render_context.rs`+`renderer.rs` (buffers EXIST) | RISKY | Build stable; correctness ∝ Phase 1 |
| 4 `803e921` Fiber skip+layout cache | `gpui/window.rs`,`view.rs`,`element.rs` | same `src/*` | NEEDS-ADAPTATION | High-risk (cache invalidation; subtree skip) |
| 5+6 `88b2aa7` List cache+batch merge | `gpui/scene.rs`,`elements/uniform_list.rs`,`list.rs`,`window.rs`,`wgpu_renderer.rs` | same-named `src/*` | NEEDS-ADAPTATION (uniform_list sig; prim-set) | **Stable, safest wins** |
| `604c0d0` Compaction | `gpui_wgpu/wgpu_renderer.rs` | `render_context.rs`+`renderer.rs` | RISKY (after P3) | Stable; defer |
| `fc12`/`2b11` Wayland damage | `gpui_linux/wayland/window.rs` | **NO TARGET** (winit) | SKIP | Churned (revert+reapply) |
| `66eea08` Persistent frame tex | `gpui_wgpu/wgpu_renderer.rs` | `renderer.rs` (`persistent_framebuffer` partly exists; blit stubbed) | RISKY | Infra sound; policy was the bug |
| `561980f` (HEAD) Disable partial redraw | `gpui_wgpu/wgpu_renderer.rs` | `renderer.rs` | CLEAN | **Canonical safe config** |

### Refined PORT ORDER (supersedes §4)
- **Tier 1 (land first, low-risk, real wins):** Phase **5+6** (list/uniform_list
  caching + batch-merge + the `BatchIterator` `peek().unwrap()`→`let Some else`
  hardening — take that fix *unconditionally*); then Phase **0** instrumentation.
- **Tier 2 (foundation, behavior unchanged):** Phase **1** SceneChunk +
  `generation` + incremental sort (safe: full-sort fallback). Unlocks 2/3/4, no
  output change yet.
- **Tier 3 (CPU caching, no GPU partial-redraw):** Phase **4 layout-caching half
  only** + the `refresh()`/`refreshing` guards; **skip/gate `should_skip_view`**.
- **Tier 4 (GPU, generation-skip only):** Phase **3** in post-`88b2` form
  (bind full buffer + `draw(0..4, range)`); enable **generation-skip +
  full-reupload-on-len-change** only; per-range upload stays gated.
- **Gate OFF / defer:** Phase **2** renderer scissor/skip (scene-side APIs only),
  `66ee` blit + `561980f` together with `partial_redraw_enabled=false`,
  `604c0d0` compaction (after P3 verified), **SKIP** Wayland.

### Why partial redraw was disabled at HEAD (and the safe config)
Root cause: **`dirty_views` (window) desyncs from `SceneChunk.dirty` (scene).**
When a view calls `cx.notify()` on itself, `dirty_views=1` but the chunk's
damage computation can yield `damage_rects=0`; with partial redraw on, the
renderer's empty-damage early-return **skips the draw**, so newly typed text
never reaches the framebuffer (the TextInput "text never appears" bug).
**Safe config (HEAD):** `partial_redraw_enabled=false` → unconditional full
`LoadOp::Clear` redraw each frame; all chunk/damage/generation tracking stays
**computed but inert**. **Do not** re-introduce: the empty-damage early-return,
default-on damage scissor, per-range GPU upload as the *sole* path, or
`should_skip_view` subtree-skip — until a keystroke is verified to yield
`dirty_views≥1` AND non-empty `changed_ranges` for that view's chunk every frame.

### 5 riskiest adaptations
1. **Phase 3 onto Pulsar's existing `Mutex<Buffer>` per-type buffers** — graft
   `last_*_len`/`last_generation`/`upload_ranges` without breaking
   `ensure_buffer_size` reallocation; use post-`88b2` form (bind full buffer +
   `draw(0..4, range.start..range.end)`, not byte-offset bind slices).
2. **Primitive-set mismatch** threaded through Phases 1/2/3/5+6 (drop subpixel,
   add backdrop_blurs everywhere).
3. **`66ee` blit vs Pulsar's stubbed `persistent_framebuffer`** — completing the
   swapchain blit (`renderer.rs:2342`) correctly; wrong = black/garbled window.
4. **Phase 4 layout-cache invalidation** across resize/scroll/animation/focus/IME
   — re-place `layout_cache.clear()`/`layout_engine.clear()` at Pulsar's winit
   resize sites (`cross/window.rs:120/201`), else stale layout after resize.
5. **uniform_list cache signature** — Pulsar `render_items` returns
   `SmallVec<[AnyElement;64]>`; rework the iterate/zip/`with_element_state`
   plumbing; off-by-one in `cache_range`/`ITEM_CACHE_BUFFER` = wrong-item/pop-in.
