# WGPUI 2.0

### GPU-Native Rearchitecture

Status: **proposal — planning phase, no code written yet.** This is the spec
for WGPUI 2.0: a from-the-ground-up GPU-native rearchitecture, kept
backward-compatible at the frontend (§7) and versioned as a major bump
because almost nothing below that frontend stays the same. It follows up on
`docs/retained-layers.md` ("R-N") and `docs/scroll-free-by-default.md`
("SFD") — WGPUI 1.x's own retained-rendering work — and does not repeat
their vocabulary (`Layer`, `Invalidation`, `LayerPolicy`, `ElementInstance`,
`diff_key`, overscroll buffers), citing them as "R-N §M" / "SFD §M"
throughout. Read those two first if you haven't; this document assumes
them. Self-references below ("2.0 does X", "2.0's job is...") mean this
spec, the same way the other two documents refer to themselves as "R-N" and
"SFD." This revision unifies everything decided so far into one coherent
pass — ambient reconciliation as the actual default (§4.0), `.boundary()`
narrowed to a pure compositing policy (§4.1), the `.uncached()` escape
hatch (§4.2), tile-based buffering for freeform 2D content (§4.3), the
explicit delta-upload contract (§5.0), the `WgpuSurface` fast path (§5.5),
and — new in this revision — the full workspace file map (§3), so every
mechanism above has a concrete, growable home instead of a crate-level
sketch.

**Contents:** §0 Constraints · §1 Current state · §2 Target shape · §3 File
map · §4 Reconciliation & boundaries (4.0 ambient reconciliation · 4.1
`.boundary()` · 4.2 `.uncached()` · 4.3 tile-based buffering) · §5 GPU
compute pipeline (5.0 upload granularity · 5.1 ordering · 5.2 occlusion ·
5.3 indirect draw · 5.4 invalidation · 5.5 `WgpuSurface`) · §6 What stays on
the CPU (6.1 regular-content layout · 6.2 the `diff_key` invariant) · §7
Frontend contract · §8 Phasing · §9 Risks · §10 Rejected/deferred · §11
Immediate next actions.

## 0. What this is actually asking for

R-N and SFD made WGPUI's renderer *retained* — CPU state persists across
frames instead of being rebuilt, so unchanged content costs nothing to
re-walk. That work is real, merged (`#98`–`#148`), and mostly on by default.
It is also, honestly, a CPU architecture: every mechanism it added — layers,
instances, per-layer slabs, occlusion culling, scroll buffers — is CPU code
that *produces* GPU buffers, not GPU code that *computes* the UI. The
ordering sort, the occlusion sweep, the layout of a ten-thousand-row list,
the decision about which primitives are dirty: all of it still happens on
one CPU thread, one frame at a time, in a 14,278-line file.

The request this document answers is the next step past that: stop treating
the GPU as a rasterizer that CPU code hands finished triangles to, and start
treating it as the machine that *computes the UI* — ordering, culling,
compositing, and (for the content shapes that admit it) layout — with the
CPU's job shrunk to running user code and describing what changed. Down to
indirect dispatch where indirect dispatch is the right tool, not as a
slogan.

Three constraints were set explicitly and are treated as hard requirements
throughout, not preferences:

1. **The public API does not change**, except the caching primitives
   (explicitly carved out). `Render`, `Entity<T>`, `Context<T>`, the
   `Styled`/Tailwind-style DSL, actions, keymaps, and every element
   constructor (`div()`, `img()`, `svg()`, `uniform_list()`, ...) compile and
   behave identically. This document treats "identically" as testable, not
   aspirational — see §7.
2. **Native-first, WASM best-effort.** Every GPU-driven mechanism is designed
   for native wgpu (Vulkan/DX12/Metal) without compromise; WASM gets a
   CPU-executed fallback for whatever WebGPU can't yet do, shipped later and
   never gating native progress.
3. **Built as a parallel workspace crate, cut over at parity.** The existing
   code keeps working, unmodified in behavior, as the default, until a new
   crate reaches feature parity and an explicit cutover phase retires the
   old path. No third option (in-place flag ladder) — this repo already has
   nine `WGPUI_*` kill switches from R-N/SFD, and SFD §0.1 documents directly
   what that costs: a mechanism that exists, is fast, and is used in one of
   thirty-seven call sites in the real application, because reaching it
   required knowing about three unrelated pieces of API. A parallel crate
   with one public backend choice is the fix for that failure mode, not a
   repeat of it.
4. **The R-N/SFD object model is the backbone, redesigned to be GPU-computed
   from the start** — not thrown away, not merely accelerated in place.
   `Layer`, `ElementInstance`, the four `Invalidation` axes, and per-layer
   slabs got the *concepts* right (retained, address-stable, per-region
   invalidation). What's being replaced is the *mechanics*: CPU sort → GPU
   compute sort; CPU occlusion sweep → GPU compute occlusion; CPU-computed
   draw ranges → GPU-computed indirect draw args; CPU Taffy walk for regular
   content → GPU layout kernel. Where content genuinely can't move to the
   GPU (arbitrary user code, heterogeneous flexbox, text shaping), it stays
   exactly where it is, on purpose — see §6.
5. **The default assumption, everywhere, is "retained unless a diff proves
   otherwise" — never "rebuilt unless something opted into caching."** This
   is stated as its own constraint because today's shipped architecture,
   despite building the exact mechanism this needs (`diff_key`/
   `InstanceKey` reconciliation, R-N Phase 7), still gets this backwards at
   the one seam that matters most: reconciliation only runs *inside* a
   `.layer()` subtree (R-N's own Phase 7 gate: "scoped to content inside a
   `.layer()` subtree only"). Everything else — which, per SFD §0.1's own
   count, is most of a real application's element tree — gets full
   `prepaint`/`paint`/layout on every notification of its owning view,
   regardless of whether anything an observer could see actually changed.
   That is opt-in retention with rebuild as the fallback, and it is exactly
   the shape SFD §0.1 found produces near-zero real-world adoption (1 of 37
   call sites), because the fast path is invisible until someone goes
   looking for it. 2.0 inverts this at the architecture level, not just at
   one call site — see §4. Inverting a default doesn't mean removing the
   ability to override it: content that is provably always-dirty every
   frame gets an explicit, equally simple way to say so (`.uncached()`,
   §4.2) — the assumption is retained-unless-proven-otherwise, and an
   up-front developer assertion that it isn't counts as proof.

---

## 1. Where we actually are (grounded, not asserted)

| Fact | Evidence |
|---|---|
| `window.rs` is 14,278 lines, one `impl Window` block spans ~5,800 lines, 450 methods total | `wc -l src/window.rs`; `impl Window` at `window.rs:2322`–`8167` |
| `app.rs` is 2,802 lines; `impl App` alone is ~1,724 lines | `app.rs:718`–`2442` |
| `elements/div.rs` is 4,528 lines — the single largest element, doing interactivity, style resolution, scroll, and layer/boundary registration in one file | `wc -l src/elements/div.rs` |
| The embedded profiler/replay system (`flamegraph.rs`, `flamegraph_gpu.rs`, `flamegraph_replay.rs`, `flamegraph_ui_capture.rs`) is ~9,000 lines — larger than the entire GPU renderer plus every shader combined | line counts, §4 below |
| `geometry.rs` (3,961), `scene.rs` (2,941), `scene_pack.rs` (1,971), `platform.rs` (1,948), `style.rs` (1,578) round out the files over 1,500 lines | `wc -l src/*.rs` |
| R-N's own status table says phases 4, 6, 7, 8, 9, 11 are shipped and default-on; phase 12 (delete the old cache) is **not started**, and the old replay path is still load-bearing — `Layer::paint_range` routes through it | `docs/scroll-free-by-default.md` §0 |
| Of 37 hand-rolled scroll containers in the actual consuming application, 1 uses the fast (layered, buffered) path | SFD §0.1 |
| The crate already requests `wgpu::Features::INDIRECT_FIRST_INSTANCE` and `MULTI_DRAW_INDIRECT_COUNT` at device creation — hard-required on native outside macOS, best-effort on macOS — and documents a real, already-hit driver gotcha about the former for *externally embedded* content (`WgpuSurfaceHandle`/Helio). **Nothing in the crate issues an indirect draw or a compute dispatch.** A crate-wide search for `dispatch_workgroups`, `draw_indirect`, `multi_draw_indirect`, and `create_compute_pipeline` returns zero matches. | `README.md` "Custom Device Gotcha"; `render_context.rs:104-176`; crate-wide grep |
| Per-instance GPU data is already **storage-buffer "vertex pulling"**, not vertex-attribute instancing — every render pipeline binds `vertex.buffers: &[]` and indexes a bound storage buffer with `@builtin(instance_index)` in the shader, drawn as `pass.draw(0..4, first_instance..first_instance+count)`. This is already the right shape for GPU-computed content: a compute pass that writes into the same storage buffer needs no pipeline change. | `renderer.rs:1332-1548` (pipeline construction), draw call sites e.g. `renderer.rs:3788-3789` |
| Per-layer GPU transforms are applied by **runtime string-splicing**: `slab_transform.wgsl` has no entry points at all — it's textually spliced into every other shader's source at pipeline-build time, rewriting one known vertex-position expression per shader (plus a matching fragment-stage undo for four shaders that re-read world-space geometry), with an `assert!` that each rewrite matched *exactly once* so a drifted shader fails to compile rather than silently losing the transform | `renderer.rs:99-143` (`slab_shader_source`, `FRAGMENT_TRANSLATE_EDITS`) |
| The fourth `Invalidation` axis — `TRANSFORM`, the one R-N designed specifically so a scroll tick costs "one changed matrix, zero everything else" (R-N §3.2) — is **dead code today**: the bitflag exists but the crate's own comment on it says nothing sets it yet, "until layers can be composited independently." (`Invalidation` is a hand-rolled `u8` newtype living in `window.rs`, not the `bitflags` crate, and not in `layer.rs` despite R-N's own sketch placing it there.) | `window.rs:402-461` |
| `wgpu = "30"` is pinned, with `dx12`, `vulkan`, `webgpu`, `wgsl` features on native and `web`, `webgpu` on WASM; `taffy = "=0.9.0"` is pinned; `bytemuck` (with GPU-cast derives) and `etagere` (atlas bin-packing) are already dependencies | `Cargo.toml` |
| Eight real shader files exist (`quads`, `shadows`, `mono_sprites`, `poly_sprites`, `paths`, `underlines`, `backdrop_blur`, `surfaces`) — all vertex/fragment pairs driving instanced draws, with draw-call coalescing already merging byte-contiguous same-layer/kind runs (`OpenSlabRun`, `renderer.rs:1690-1723`) — plus the `slab_transform.wgsl` splice fragment above. Per-layer suballocation (`slab.rs`, 1,538 lines; `slab_gpu.rs`, 1,437 lines) is a pure CPU-side allocator (size-class free lists, generation counters) that computes byte ranges and hands them to `write_buffer`; module doc states this directly: "no device, no queue." | file listing, `src/platform/cross/shaders/*.wgsl`, `slab.rs:1-31`, `slab_gpu.rs:1-28` |
| **The upload granularity today is per-layer, not per-primitive.** A dirty layer's `write_buffer` call covers that layer's *entire* byte range for a given primitive kind — a 10,000-quad layer with one changed quad still uploads all 10,000 quads' worth of bytes for that kind. This is a real improvement over the pre-R-N global re-upload, but it is not a delta upload, and nothing in R-N/SFD claims it is. §5.0 below makes the distinction load-bearing. | `slab_gpu.rs`'s own module doc: "the renderer executes a plan as one `write_buffer` per kind, byte offset = `SlabRange.base * stride`" — one call per (layer, kind), not one per changed primitive |
| The crate is a single `[package]`, not a `[workspace]` — everything above lives in one compilation unit | `Cargo.toml:1` |

None of this is a criticism of R-N/SFD in isolation — measured against an
immediate-mode CPU renderer, it's a large, real improvement, and the
instrumentation culture it left behind (`render_stats`, the flamegraph/replay
system, `WGPUI_OCCLUSION=validate`-style differential testing) is exactly the
discipline this plan intends to keep using. The point of this table is
narrower: the ceiling on "retain CPU state better" has essentially been
reached by the existing effort, and the honest next lever is moving
computation off the CPU, not finding one more CPU cache to add.

One more thing worth being explicit about, because it changes how this plan
should be read: the renderer isn't starting from zero on the GPU side. The
storage-buffer vertex-pulling model, the indirect-draw device features, and
the per-layer slab byte-range concept are already there — provisioned,
correctly negotiated across platforms, and completely unused. §5 below is,
to a real extent, "turn on the capability that's already been asked for,"
not "invent a new one."

---

## 2. Target shape, in one picture

```
                    ┌─────────────────────────────────────────────┐
                    │  Frontend (frozen surface — §7)              │
                    │  Render / Entity<T> / Context<T> / Styled DSL│
                    │  actions, keymap, elements/* constructors    │
                    └───────────────────┬───────────────────────────┘
                                         │ render() — arbitrary CPU code,
                                         │ runs exactly as it does today
                                         ▼
                    ┌─────────────────────────────────────────────┐
                    │  Description (per-frame, arena — unchanged)  │
                    └───────────────────┬───────────────────────────┘
                                         │ reconcile against retained
                                         │ instance tree (R-N Pillar I,
                                         │ already exists — instance.rs)
                                         ▼
                    ┌─────────────────────────────────────────────┐
                    │  Patch list — the ONE frontend/backend       │
                    │  boundary. Pure data: inserts/updates/       │
                    │  removals of primitives, layout inputs,      │
                    │  hitboxes, dispatch nodes. No control flow.  │
                    └───────────────────┬───────────────────────────┘
                                         │
                     ┌───────────────────┼───────────────────┐
                     ▼                   ▼                   ▼
             ┌───────────────┐  ┌────────────────┐  ┌─────────────────┐
             │ CPU: Taffy for │  │ CPU: cosmic-text│  │ CPU: hit-test,  │
             │ heterogeneous  │  │ shaping (§6 —   │  │ focus, actions, │
             │ layout only    │  │ not GPU-portable)│  │ input dispatch  │
             └───────┬───────┘  └────────┬────────┘  └─────────────────┘
                     │                   │
                     ▼                   ▼
          ┌───────────────────────────────────────────────────┐
          │  Persistent GPU-resident scene (per-layer slabs,   │
          │  patched not rebuilt — R-N Pillar III, extended)   │
          └───────────────────────┬─────────────────────────────┘
                                   │ compute passes, every frame,
                                   │ over resident + newly-patched data
                                   ▼
     ┌─────────────────────────────────────────────────────────────┐
     │ GPU compute: regular-content layout (§6) · ordering (§5.1) · │
     │ occlusion culling (§5.2) · indirect draw-arg generation (§5.3)│
     └───────────────────────────────┬─────────────────────────────┘
                                      ▼
                    indirect multi-draw, instanced, per layer
```

Note what the "reconcile against retained instance tree" box in this
picture is *not* scoped to: it is not drawn as living inside a `.boundary()`
subtree, because it isn't one. Per constraint 5 (§0), reconciliation is
ambient — every element in the window goes through it, every frame,
independent of whether any `.boundary()` exists anywhere in the tree. A
`.boundary()` (§4) is a request layered on top of this diagram for specific
subtrees — independent compositing, texture retention, overdraw buffering —
never the thing that switches reconciliation on in the first place. This is
the single largest substantive difference from what shipped for R-N/SFD, so
it's worth being this explicit about it before §4 makes it concrete.

The load-bearing idea: **the patch list is the only interface between "what
the app described" and "how the GPU computes and draws it."** It is data,
never a callback or a control-flow handoff, which is exactly what makes the
backend swappable while the frontend that produces the patch list (elements,
`Render`, event handlers) stays untouched. R-N already built the two pieces
either side of this boundary — `ElementInstance`/reconciliation on the
frontend side (R-N §2, shipped) and per-layer slabs on the backend side (R-N
§4, shipped) — but the seam between them today is "CPU code copies bytes
into a `Vec` that CPU code later `write_buffer`s in full or in a computed
*per-layer* range — not a per-primitive delta (§5.0 makes this precise)."
2.0's job is to make that seam a real patch protocol the GPU consumes
directly, and to move everything downstream of it onto the GPU.

---

## 3. Workspace layout

Convert the repo to a Cargo workspace. The root package (`gpui-ce`, published
name, `gpui` lib name) stays the crate external consumers depend on — its
git URL and import paths do not change. New members are added alongside it;
nothing about how `Pulsar-Native` (or any other consumer) depends on this
repo changes until the explicit cutover in **Phase 8** (§8) — not Phase 7,
which is devtools extraction; getting this wrong once in an earlier draft of
this document is exactly the kind of drift a unifying pass exists to catch.

```
WGPUI/
├── Cargo.toml                  # [workspace], root package = gpui-ce (unchanged public surface)
├── src/                        # unchanged during migration; becomes the
│                                # legacy backend, then a thin re-export (Phase 8)
├── crates/
│   ├── wgpui-core/             # §3.1 — patch protocol, scene, reconciliation,
│   │                           #   boundary/tile policy, invalidation. No
│   │                           #   windowing, no live wgpu::Device.
│   ├── wgpui-layout/           # §3.2 — Taffy integration, isolated.
│   ├── wgpui-text/             # §3.3 — cosmic-text shaping, isolated.
│   ├── wgpui-widgets/          # §3.4 — div/text/img/svg/list/... elements.
│   ├── wgpui-wgpu/             # §3.5 — the only crate that touches a live
│   │                           #   wgpu::Device: pipelines, compute dispatch,
│   │                           #   atlas, textures, winit windowing.
│   └── wgpui-devtools/         # §3.6 — flamegraph/replay/inspector, moved
│                               #   wholesale behind a small hook trait.
└── docs/
```

Every subsection below is a real, growable module tree — the point of this
section, per the ask this revision answers, is that no future addition to
any of this needs to land in a 1,000+-line file the way `window.rs`,
`div.rs`, and the flamegraph quartet do today. Three things are true across
all of them, stated once instead of six times:

- **Not everything needs a redesign to get a home.** `app/entity_map.rs`,
  `app/context.rs`, `app/async_context.rs` (already split out of `app.rs`
  today), `window/prompts.rs` (already split out of `window.rs`), and every
  file under `text_system/` (`line.rs`, `line_wrapper.rs`, `line_layout.rs`,
  `font_features.rs`, `font_fallbacks.rs`) are already the right shape and
  size. These move essentially as-is into their new crate; the file map
  below calls this out explicitly (**moved, not rebuilt**) rather than
  implying everything is being rewritten from scratch.
- **The god files get named, targeted splits**, not a mechanical chop.
  `window.rs` (14,278 lines, one 450-method `impl Window` block), `div.rs`
  (4,528 lines), `app.rs`'s core `impl App` (~1,724 lines), and the
  flamegraph quartet (~9,000 lines across four files) each get a split
  derived from seams the current code already shows (§3.4 quotes `div.rs`'s
  own four seams directly) or from the module boundaries this document
  itself defines (§3.1's split *is* patch/scene/reconcile/invalidation/
  boundary, because those stopped being one concern the moment ambient
  reconciliation, §4.0, removed the single recursive CPU walk that used to
  justify combining them).
- **No file target below is a hard rule enforced by tooling in this
  revision** — it's a design intent stated concretely enough to check by
  eye (`wc -l`) rather than left as "keep files short."

### 3.1 `wgpui-core` — patch, scene, reconciliation, boundary policy

No live `wgpu::Device` anywhere in this crate — it owns the *shapes* of GPU
work (shader source as text, buffer/binding layout descriptors as plain
Rust structs) so it stays unit-testable headlessly, the same reason
`slab.rs`'s module doc already commits to "no device, no queue" today.
`wgpui-wgpu` (§3.5) is what actually creates pipelines and dispatches.

```
crates/wgpui-core/src/
├── lib.rs
├── patch/
│   ├── mod.rs              # Patch, PatchList (§2, §5.0)
│   ├── primitive.rs        # per-kind payloads: update-in-place vs insert/remove
│   └── apply.rs            # apply a PatchList to a Scene — Phase 1's round-trip gate lives here
├── scene/
│   ├── mod.rs
│   ├── layer.rs            # Layer record: id, key, policy, invalidation state (R-N's Layer, 2.0's mechanics)
│   ├── tile.rs             # TileCoord, (boundary, TileCoord) addressing (§4.3)
│   ├── slab.rs             # size-class allocator, ported/cleaned from today's slab.rs
│   └── slab_range.rs       # byte-offset math, generation counters, per-primitive slot addressing (§5.0)
├── reconcile/
│   ├── mod.rs
│   ├── instance.rs         # ElementInstance, InstanceKey — ambient, §4.0
│   ├── diff_key.rs         # ReconcileKey trait; §6.2's invariant is enforced against this file's default
│   └── uncached.rs         # the scope flag threaded through prepaint/paint (§4.2)
├── invalidation/
│   ├── mod.rs
│   ├── axes.rs             # LAYOUT/DISPLAY/HIT/TRANSFORM — finally wired live, §5.4
│   ├── request.rs          # InvalidationRequest/InvalidationScope (R-N §6, unchanged)
│   └── reason.rs           # Reason::Scroll vs Reason::DataChanged (§5.4, §4.1)
├── boundary/
│   ├── mod.rs
│   ├── policy.rs           # BoundaryPolicy, Buffering enum (§4.1, §4.3)
│   └── identity.rs         # positional-identity fallback (SFD §1.0), reused by WgpuSurface (§5.5, Gap 1)
├── occlusion/
│   ├── mod.rs
│   └── coverage.rs         # conservative opaque-region test (R-N §8.3) — CPU reference impl AND
│                            #   the oracle `validate` mode diffs the compute path (§5.2) against
├── shaders/                 # WGSL *source* + Rust-side layout descriptors — text and data, no device
│   ├── ordering.wgsl        # §5.1
│   ├── occlusion.wgsl       # §5.2
│   ├── layout_uniform.wgsl  # §6.1
│   ├── tile_visibility.wgsl # §4.3
│   └── indirect_args.wgsl   # §5.3
├── window/                  # window.rs's actual successors, one concern each
│   ├── mod.rs               # Window struct assembly only
│   ├── focus.rs             # FocusHandle/FocusId, tab stops
│   ├── hitbox.rs            # Hitbox, point-transform hit-testing (R-N §5.2, kept as-is)
│   ├── dispatch.rs          # DispatchTree, action dispatch, key_context
│   ├── input.rs             # keyboard/mouse event routing
│   ├── animation.rs         # request_animation_frame(_for), §5.4's TRANSFORM-only glide path
│   └── prompts.rs           # moved, not rebuilt — already its own file today
├── app/
│   ├── mod.rs
│   ├── entity.rs            # moved, not rebuilt — today's app/entity_map.rs
│   ├── context.rs           # moved, not rebuilt — today's app/context.rs
│   ├── async_context.rs     # moved, not rebuilt — today's app/async_context.rs
│   ├── effects.rs           # deferred notifications, flush_deferred_invalidations
│   └── globals.rs
└── test_support/
    └── mod.rs               # headless patch/reconcile/window testing, mirrors today's platform/test
```

### 3.2 `wgpui-layout` — Taffy, isolated

```
crates/wgpui-layout/src/
├── lib.rs
├── taffy_tree.rs        # persistent TaffyTree wrapper — today's taffy.rs, made ambient per §4.0
├── measure.rs           # measured-layout closures (text/intrinsic-size leaves)
├── containment.rs       # estimated_size / layout containment (SFD §0.-3)
└── regular.rs           # detects §6.1-eligible ("regular") content; everything else stays here
```

### 3.3 `wgpui-text` — cosmic-text shaping, isolated (§6: not moving to the GPU)

Mostly a move, not a rewrite — today's `text_system/` is already close to
this shape.

```
crates/wgpui-text/src/
├── lib.rs
├── shaping.rs           # cosmic-text integration — today's text_system.rs core
├── line.rs              # moved, not rebuilt
├── line_wrapper.rs      # moved, not rebuilt
├── line_layout.rs       # moved, not rebuilt
├── fonts/
│   ├── features.rs      # moved, not rebuilt — today's font_features.rs
│   └── fallbacks.rs     # moved, not rebuilt — today's font_fallbacks.rs
└── patch.rs             # shaped-run → patch conversion (Phase 5, closes §6.2's Img/StyledText gap)
```

### 3.4 `wgpui-widgets` — elements, split along `div.rs`'s own seams

`div.rs` (4,528 lines) already shows four fairly distinct concerns living in
one file — quoted directly so the split below isn't arbitrary: a ~456-line
event-binding builder trait (`InteractiveElement`), a ~1,140-line
`Interactivity` state-and-paint engine (the single largest block, interleaving
style application, hitbox/dispatch registration, scroll handling, and layer
paint), `Div`'s own `Element` impl plus its reconciliation fingerprint
(~600 lines combined), and scroll/click retained-state types (~235 lines).

```
crates/wgpui-widgets/src/
├── lib.rs
├── div/
│   ├── mod.rs               # Div, DivFrameState/DivPrepaintState — the small remainder
│   ├── events.rs            # InteractiveElement (today's ~456-line trait block)
│   ├── interactivity/
│   │   ├── mod.rs
│   │   ├── style.rs         # style application + classify_style_change (§6.2's engine)
│   │   ├── hitbox.rs        # hitbox/dispatch-node registration
│   │   └── layer_paint.rs   # boundary/paint plumbing (today's layer-paint slice of the ~1,140-line block)
│   ├── diff.rs               # DivDiffKey / ReconcileKey impl
│   └── scroll_state.rs        # ScrollAnchor/ScrollHandle (today's ~235-line block)
├── text.rs
├── styled_text.rs             # gets diff_key here (§6.2 invariant, Phase 5)
├── img.rs                     # gets diff_key here (§6.2 invariant, Phase 5)
├── svg.rs
├── canvas.rs
├── list/
│   ├── mod.rs
│   ├── list.rs
│   ├── uniform_list.rs        # the CPU special case §6.1's GPU kernel generalizes
│   ├── virtual_list.rs
│   └── h_list.rs
├── wgpu_surface.rs             # real identity + trivial diff_key (§5.5, Gap 1)
├── animation.rs
├── overlay/
│   ├── anchored.rs
│   └── deferred.rs
├── scroll/
│   ├── smooth_scroll.rs
│   └── scroll_buffer.rs
├── surface.rs
└── image_cache.rs
```

### 3.5 `wgpui-wgpu` — the only crate touching a live device

Two subtrees, genuinely distinct concerns that today live intermixed under
`platform/cross/`: windowing (winit, input plumbing, OS integration) and
rendering (device, pipelines, compute dispatch, atlas, textures).

```
crates/wgpui-wgpu/src/
├── lib.rs
├── window/
│   ├── mod.rs               # winit window creation/event loop glue — today's platform/cross/window.rs
│   ├── dispatcher.rs        # moved, not rebuilt
│   ├── keyboard.rs          # moved, not rebuilt
│   ├── resize_detector.rs   # moved, not rebuilt
│   └── app_menu.rs          # moved, not rebuilt
└── render/
    ├── mod.rs
    ├── device.rs            # device/queue creation, feature negotiation — today's render_context.rs
    ├── buffers/
    │   ├── slab_buffers.rs  # GPU-side buffer per slab kind
    │   └── upload.rs        # delta-upload adjacency coalescing (§5.0)
    ├── compute/
    │   ├── ordering_pass.rs         # dispatches wgpui-core's ordering.wgsl (§5.1)
    │   ├── occlusion_pass.rs        # dispatches occlusion.wgsl (§5.2)
    │   ├── layout_pass.rs           # dispatches layout_uniform.wgsl (§6.1)
    │   ├── tile_visibility_pass.rs  # dispatches tile_visibility.wgsl (§4.3)
    │   └── indirect_args_pass.rs    # dispatches indirect_args.wgsl (§5.3)
    ├── pipelines.rs         # render pipeline creation per kind (today's quads/shadows/.../paths)
    ├── draw.rs              # indirect draw issuance + coalescing (today's OpenSlabRun logic)
    ├── readback.rs          # CPU-readback fallback for indirect draw (§5.3) — macOS best-effort + WASM
    ├── atlas.rs             # glyph/sprite atlas, etagere bin-packing — moved, not rebuilt
    ├── textures/
    │   ├── layer_texture.rs     # boundary texture-retention pool (today's LayerTextureEntry)
    │   └── external_surface.rs  # unified WgpuSurface consumer entry (§5.5, Gap 2) — same type as layer_texture
    ├── surface_registry.rs  # producer-side triple-buffer — UNCHANGED (§5.5, §9's explicit "don't touch this")
    └── shaders/              # the hand-written render shaders, moved as-is
        ├── quads.wgsl
        ├── shadows.wgsl
        ├── mono_sprites.wgsl
        ├── poly_sprites.wgsl
        ├── paths.wgsl
        ├── underlines.wgsl
        ├── backdrop_blur.wgsl
        └── surfaces.wgsl
```

### 3.6 `wgpui-devtools` — moved wholesale, behind one small hook trait

~9,000 lines across four files today — more than the renderer and every
shader combined — of genuinely valuable, genuinely separable tooling
(per-frame CPU/GPU flamegraphs, a RenderDoc-style capture-and-replay engine
for the whole UI tree). None of it needs to live in the same compilation
unit as the engine it profiles; it needs a handful of stable hook points
(span push/pop, GPU timestamp write, frame-capture trigger) that
`wgpui-core`/`wgpui-wgpu` expose as a small trait, the same shape `profiling`
(already a dependency) uses for its own backend-agnostic design. Pure
move-and-decouple, zero behavior change — Phase 7's gate (§8) is precisely
that `wgpui-core` builds and runs with this crate absent entirely.

```
crates/wgpui-devtools/src/
├── lib.rs
├── hooks.rs             # the small trait wgpui-core/wgpui-wgpu expose into
├── flamegraph/
│   ├── cpu.rs           # moved, not rebuilt — today's flamegraph.rs
│   ├── gpu.rs           # moved, not rebuilt — today's flamegraph_gpu.rs
│   ├── replay.rs        # moved, not rebuilt — today's flamegraph_replay.rs
│   └── ui_capture.rs    # moved, not rebuilt — today's flamegraph_ui_capture.rs
├── render_stats.rs      # moved, not rebuilt
├── inspector.rs         # moved, not rebuilt
└── perf_ab_tests.rs     # moved, not rebuilt
```

### 3.7 The root crate's fate

`src/` is not restructured during the migration — it is frozen (bugfixes
only, per the Phase 1 risk-table entry) as the legacy backend, exactly as it
exists today, until Phase 8's cutover collapses it to a thin re-export of
`wgpui-core` + `wgpui-widgets` (plus whichever of `wgpui-layout`/`-text`/
`-wgpu`/`-devtools` the public surface needs to name types from). No target
file map for it here: by construction, at cutover it should be small enough
not to need one.

---

## 4. Ambient reconciliation, and two symmetric manual primitives

Three separate mechanisms, deliberately, where R-N/SFD had effectively one
— that conflation is exactly what constraint 5 (§0) says to undo: reconciliation
is always on (§4.0, no API), `.boundary()` opts a subtree *into* independent
compositing (§4.1), and `.uncached()` opts a subtree *out of* reconciliation
for the one content shape where diffing never pays off (§4.2).

### 4.0 Reconciliation is ambient — ships in Phase 1, no API at all

Every element in the window is reconciled via `diff_key`/`InstanceKey`
(R-N §2, §2.3) every frame, whether or not it sits inside a `.boundary()`.
A `div` three levels deep in an ordinary, unboundaried panel that renders
identically to last frame skips `request_layout` (keeping its retained
Taffy node — R-N §2.5/Phase 8's mechanism, made ambient), `prepaint`, and
`paint`, exactly as if it were inside today's `.layer()`. There is no
opt-in call, because there is nothing to opt into: this is simply what the
window does with the description `render()` produces, the same way a
browser diffs its DOM/style/layout state whether or not an element has
`will-change: transform`. §6's "what stays on the CPU" content — Taffy for
heterogeneous layout, hit-test/dispatch registration — is retained under
this rule too, since retention there was always gated on the identical
`diff_key`-proves-clean condition R-N Phase 7/8 already built; ambient
reconciliation just stops fencing that condition to `.layer()` subtrees.

This is also what actually resolves a case §4.1 (below) would otherwise
need a dedicated mechanism for: a hover-driven style change on one element, under
ambient reconciliation, produces exactly one instance's `DISPLAY`
invalidation (via `classify_style_change`, `div.rs:2285-2353`) — ancestors
above it were never rebuilding on that notification's account in the first
place, because they were always subject to the same diff. No separate
"auto sub-boundary for hover" mechanism is needed; it falls out of the
general rule rather than requiring a special case reserved for later.

**What this changes about R-N's own numbers.** R-N §2.6 called "is
`render()` cheap relative to layout+paint" an assumption worth measuring,
not proving. Under ambient reconciliation it stops being an assumption:
regardless of whether `render()` reruns for a notified view (it always
does — arbitrary user code can't be skipped, §6), the diff immediately
downstream of it is what decides whether any layout/paint/GPU work happens,
for *every* element it touches, not only elements an author remembered to
wrap.

**Risk this takes on, honestly.** Making reconciliation the ambient default
means a reconciliation bug's blast radius is the whole application on day
one, not one opted-in subtree — SFD's own kill-switch culture
(`WGPUI_INSTANCES=0`, precedent at `view.rs:103`) exists for exactly this
class of risk, and 2.0 keeps a kill switch for the same reason, but now
gating the *default* path rather than an edge feature. See the risk table
(§9).

### 4.1 The single cache-boundary primitive

`.boundary()` is what's left once reconciliation is no longer its job: a
*compositing and buffering* policy for subtrees that benefit from being
independently rasterized — a scrollable panel, a viewport, anything the
old `.layer()`/`.layer_keyed()`/`.layer_with_policy()` triad targeted. It
answers "does this region get its own GPU texture, an overdraw margin, and
its own occlusion tier," never "does this region's content get diffed" —
that question was already answered yes, for everything, in §4.0.

Today, getting onto the fast path requires, together, on every scrollable or
expensive subtree individually (SFD §0.1):

```rust
div()
    .id("scroller")                          // silently required — SFD §0.2
    .layer_keyed(content_key)                // a key you derive and keep correct
    .layer_with_policy(LayerPolicy { .. })    // a margin you tune
    .overflow_y_scroll()
    .track_scroll(&handle)
```

plus `Panel::cacheable` as a separate manual opt-out mechanism for the old
cache (R-N §0.2), plus `diff_key` as a separate per-element-type opt-in (R-N
§2.3). Four mechanisms, each independently learnable, each independently
capable of silently no-oping (SFD §0.2's finding: `.layer()` without `.id()`
compiles, runs, and does nothing).

2.0 replaces all four with one element:

```rust
div().id("properties-panel").boundary().child(expensive_content)
```

`.boundary()` takes an optional `BoundaryPolicy` for tuning only — never for
correctness:

```rust
pub struct BoundaryPolicy {
    /// Below this primitive count, stay primitive-retained (no texture).
    /// Same role as R-N's `rasterize_above`; still complexity-driven,
    /// still automatic below/above this line.
    pub rasterize_above: usize,
    /// How this boundary buffers ahead of scroll/pan. `Margin` is R-N/SFD's
    /// existing mechanism, generalized only in name here — §4.3 is where the
    /// second variant, and the reason this is an enum and not still a single
    /// `Option<Size<Pixels>>`, actually gets used.
    pub buffering: Buffering,
}

pub enum Buffering {
    /// No buffer beyond the visible bounds.
    None,
    /// One rectangular region sized to viewport + margin, refilled wholesale
    /// when scrolled past it (R-N §7/SFD's overscroll buffer, unchanged).
    /// Right for linear content: lists, columns, anything scrolling along
    /// one or two bounded axes with a defined content extent. Auto-sized
    /// from the viewport when unset (SFD §7's own recommendation).
    Margin(Option<Size<Pixels>>),
    /// A grid of independently cached tiles (§4.3). Right for freeform,
    /// arbitrarily-positioned, pannable-in-any-direction content — node
    /// graphs, canvases, 2D level views — where `Margin` would have to grow
    /// multiplicatively in both axes to avoid frequent whole-region refills.
    Tiled { tile_size: Size<Pixels>, retain_radius: u32 },
}
```

What makes a bare `.boundary()` safe by default, closing every gap SFD found
in the opt-in version — note none of these are about *whether* content is
diffed (§4.0 already answers that unconditionally); they're about a
compositing boundary correctly locating and invalidating itself:

- **Identity never requires `.id()`.** Positional identity (SFD §1.0 —
  `ElementId::InstanceSlot(child_index)` for a boundary root, the same
  mechanism `instance_id_stack` already applies one level down for a
  layer's children) is the *only* identity path; an explicit `.id()` refines
  it for cross-frame stability under reordering, it doesn't gate it. Where
  R-N/SFD's `.id()` requirement gated reconciliation itself (so a forgotten
  `.id()` meant silently no cheaper than a full rebuild), here a
  forgotten `.id()` only costs a boundary its independent-compositing
  benefit — the content underneath is reconciled regardless (§4.0), so the
  failure mode shrinks from "silently as slow as no caching at all" to
  "silently missing one specific optimization."
- **Scroll is a distinguishable signal, not an inferred one, from day one.**
  SFD §1.1 found this had to be built as `notify_scroll()` — a tagged
  notification layers recognize as transform-only — because a generic
  `cx.notify()` can't be told apart from "the data changed" after the fact.
  2.0 builds this into the invalidation system's vocabulary from the start
  (§5.4) rather than retrofitting it once `.boundary()` is already deployed
  everywhere, which is the order SFD explicitly recommends against (SFD §1
  intro: don't broadcast a known gap to every call site at once).
- **Invalidation axes are always derived from the diff**, never
  hand-declared (R-N §2.4, already shipped and kept unchanged in spirit) —
  a `.boundary()` with no policy at all still gets exactly-correct
  transform/display/hit/layout invalidation, because the reconciler tells it
  what changed; the policy struct only ever tunes *how* a boundary that is
  known to be dirty gets rasterized, never *whether* it's considered dirty.
- **Hover/animation-driven content no longer needs a dedicated boundary
  mechanism at all.** SFD §1.0 flagged this as "reserved, not built yet,"
  reserved specifically because under R-N/SFD's model a hover-triggered
  `cx.notify()` on the owning view was indistinguishable from any other
  notify, and the only tool available to contain the damage was promoting
  the hovered element to its own layer. Under §4.0's ambient reconciliation
  this isn't a gap to close with a new mechanism — a hover-driven style
  change already produces exactly one instance's `DISPLAY` invalidation via
  `classify_style_change`, with or without a `.boundary()` anywhere nearby.
  `.boundary()` remains available for a *different*, legitimate reason to
  promote hovered content — e.g. a hover effect on a texture-retained panel
  where recompositing (not rediffing) is the actual cost — but it is no
  longer required for correctness the way SFD's finding implied.

`Panel::cacheable` and its opt-out list are deleted outright — `on_frame`
(R-N §2.4) already gives skipped-closure side effects a legal home, so there
is no longer a category of element that needs to opt *out* of caching.

### 4.2 The escape hatch: `.uncached()` for content that never benefits

Ambient reconciliation (§4.0) is a bet that diffing pays for itself, and for
almost all UI content it does — R-N §2.6's own accounting (description
building is cheap; layout/paint/shaping are what's worth skipping) is why.
The bet fails for one specific, identifiable shape: a subtree whose content
is guaranteed to differ every single frame — a live telemetry HUD, an audio
waveform, a per-frame physics/debug overlay, a node-graph editor's live
value readouts during a simulation scrub. For that content, `diff_key`
comparison is not "usually free, occasionally expensive" — it is
*unconditionally* wasted work, because the outcome is always "rebuild
anyway," and worse, the framework has been maintaining a retained
`ElementInstance` + fingerprint for each of those elements for no reason:
memory held for a comparison that will never once succeed.

`.uncached()` is the deliberate, symmetric complement to `.boundary()` —
where `.boundary()` opts a subtree *into* additional compositing benefits on
top of always-on reconciliation, `.uncached()` opts a subtree *out of*
reconciliation itself:

```rust
div().uncached().child(live_telemetry_panel)
```

Mechanically this is not new machinery — it's making an existing code path
selectable on purpose. Every element type's `diff_key` already defaults to
`None` (`element.rs:136-138`), and R-N §2.3 already specifies exactly what
that means: "full rebuild, zero savings, zero risk," the same unconditional
`request_layout`/`prepaint`/`paint` path the framework used before
reconciliation existed at all. `.uncached()` pushes a scope flag (the same
shape as the existing content-mask/text-style stacks threaded through
`window.rs`) that forces every element inside it onto that path regardless
of what its own `diff_key` impl would otherwise report — no `ElementInstance`
is allocated for them, no fingerprint is retained, no comparison runs. It is
strictly less bookkeeping than reconciling-and-always-losing, not merely a
skip.

**What it does not touch.** State retention (`use_state`, entity reads,
focus, tab stops) is a separate mechanism keyed by `(GlobalElementId,
TypeId)` (R-N §2.1's table draws this line explicitly: State is "already
retained... unchanged" by any of Pillar I's mechanics). `.uncached()` only
suppresses `ElementInstance`/`diff_key` reconciliation; a slider or text
input living inside an `.uncached()` panel keeps its interactive state
exactly as it would anywhere else. Occlusion culling (§5.2) and the GPU
patch protocol (§2) are similarly untouched — an `.uncached()` subtree still
emits ordinary patches every frame (always a full replace, never a delta),
and the GPU-side pipeline has no way to tell, or reason to care, whether a
patch arrived because a diff proved change or because diffing was skipped
entirely. `.uncached()`'s effect is confined to the CPU reconciliation step;
nothing below it changes.

**Composes with `.boundary()`, and the combination matters.** These answer
different questions — "should this be diffed" vs. "should this be its own
compositing unit" — so both can apply to the same subtree. This is exactly
the shape R-N §5.1's own sizing rule warns about: "a layer containing one
high-frequency animating element and a thousand static ones will re-sort
all thousand every frame." A live viewport HUD dropped into a large static
panel wants both: `.boundary()` isolates its churn into its own
independently-ordered, independently-composited region so it stops forcing
the static content around it to re-sort, and `.uncached()` stops the
framework from wasting a diff on content that was always going to redraw.
Neither alone solves what the pair solves together.

**Relationship to R-N's own deferred idea.** R-N §9's risk table already
proposed an *adaptive* version of this — the framework tracking a per-subtree
payoff ratio and automatically marking persistently-unprofitable subtrees
`AlwaysRebuild` — but left it as a future mitigation, not a shipped
mechanism. `.uncached()` is the deterministic, developer-asserted version:
faster to build, correct on day one for the case a developer already knows
about with certainty, and it does not preclude the adaptive heuristic from
being added later as a *default* for subtrees nobody annotated — the two
are complementary, not competing, the same relationship the original ask
draws between "manual caching primitives for developers to assert" and "the
framework can handle itself without developer intervention." Building the
adaptive version is out of scope here (§10) precisely because the manual
primitive is what a developer facing this problem today can reach for
immediately, without waiting on a heuristic to prove itself trustworthy.

### 4.3 Two-axis content: tile-based buffering

`Buffering::Margin` (§4.1) is R-N/SFD's overscroll buffer, and it is the
right mechanism for what it was built for: a list or column with a defined
content extent, scrolling along one bounded axis. It is the wrong mechanism
for content with no linear index at all — a node/blueprint graph editor, a
whiteboard, an infinite canvas, a 2D level or world view — where children
are placed at arbitrary positions on a plane the user can pan freely in
*any* direction, not just up/down a list. Two concrete reasons a bigger
margin doesn't fix this:

- **The area grows multiplicatively, not additively.** A vertical list's
  margin only has to extend along the scroll axis; a freely-pannable plane's
  margin has to extend on both axes to avoid frequent refills, so a 50%
  margin on each axis buffers 2.25× the visible area (`1.5²`), not 1.5×, and
  the multiplier gets worse the more generously it's sized to avoid
  refilling on a fast diagonal pan.
- **"Refill" still means re-rendering the whole buffered rectangle**, per
  R-N §7's own protocol: cross the margin and the *entire* viewport+margin
  region re-renders, centred on the new position. For a list that's cheap
  because list rows are cheap to re-lay-out (SFD's own containment work,
  §0.-3, made this cheaper still). For a graph editor with thousands of
  nodes, that's the exact "lay out the buffer range means laying out
  everything in it" problem SFD §0.-1.3 already diagnosed for plain divs —
  except here there's no `uniform_list`-style escape hatch to fall back on,
  because the content has no uniform index to virtualize by.

**`Buffering::Tiled`** is what browser compositors actually do for this
case — divide the content's (potentially unbounded) plane into a grid of
fixed-size tiles, each independently rendered and cached, so panning in any
direction only requires producing the tiles newly revealed at the leading
edge, never re-rendering what's already resident. This is the mechanism the
original brief named directly ("browser rendering engines that use tile
based scroll buffers"), and it turns out to need almost no new machinery:

- **A tile is just a `Layer`, addressed one dimension further.** Everything
  a tile needs — its own instance arena (§4.0), its own slab (§5.0), its own
  place in occlusion culling (§5.2) and indirect draw issuance (§5.3) — is
  what a `Layer` already is. The only new concept is the address:
  `LayerKey` extends from "one key per boundary" to "one key per
  `(boundary, TileCoord)` pair," a mechanical generalization of the same
  positional-identity scheme §4.1 already builds for boundary roots.
- **Visibility is a compute-shader-shaped problem, almost by definition.**
  "Which tile coordinates intersect (viewport ∪ retain radius) at the
  current pan offset" is a handful of integer divisions per tile — the kind
  of cheap, uniform, parallel computation §5.3's indirect-draw-arg
  generation already exists to drive. A compute pass computes tile
  visibility directly from a pan-offset uniform and writes indirect draw
  args only for in-range tiles; the CPU never enumerates tile candidates.
  This is arguably the single cleanest example in this entire document of
  "down to indirect dispatch" actually paying for itself, rather than being
  reached for on principle.
- **Panning is `TRANSFORM`-only for every resident tile** (§5.4) — sliding
  the view within the currently-buffered grid recomposites existing tiles
  at new offsets, zero render/reconcile/layout work. Crossing into a new
  tile triggers `DISPLAY` for *that tile alone* (ambient reconciliation,
  §4.0, applies inside it exactly as inside any other boundary) — not the
  whole buffered region, which is the concrete advance over `Margin`'s
  refill-everything behavior.
- **Eviction is R-N §3.4's mark-and-sweep, spatially triggered.** A tile
  whose coordinate falls outside (viewport ∪ retain radius) for
  `evict_after_frames` returns its slab/texture to the pool — same
  mechanism, same bounding argument, applied per tile instead of per layer.

**Costs, stated honestly, not glossed over:**

- **Tile size is a real tuning knob with a real tradeoff**, the same shape
  as §5.0's write-granularity tradeoff: too small and per-tile overhead
  (layer bookkeeping, draw-call count) dominates; too large and refill cost
  approaches `Margin`'s whole-region-refill cost for content that happens
  to sit in one still-large tile. There's no principled default without
  measuring — Phase 0's spike discipline applies here too, picking a
  starting size from common compositor practice (roughly 256–512px) and
  validating it against a representative node-graph workload, not asserting
  it.
- **Content spanning multiple tiles needs a rule.** A wire/connection
  between two nodes in different tiles either gets clipped and rasterized
  into each tile it crosses (what browser tiling actually does for an
  element spanning tile boundaries) or lives on an unbuffered overlay layer
  above the tile grid — the same named pattern SFD §2 already proposes for
  hover-resolved content that can't cleanly live inside a buffer. Reuse
  that pattern rather than inventing a second one.
- **Tile count needs its own memory budget, not just a per-tile timer.** An
  erratic pan pattern (fast diagonal movement, frequent direction reversal)
  can keep many tiles within "recently visited" simultaneously, which
  `evict_after_frames` alone doesn't bound — a freeform 2D plane can have
  far more live tiles than a typical UI ever had layers. A total resident-
  tile cap with LRU eviction beyond it is the added mitigation R-N §3.4
  didn't need because it never had more than one buffer per layer.

Phase 4.5 (§8) is deliberately after indirect draw (Phase 4) and
independent of text/regular-layout work (Phases 5–6.1): tiling needs
addressable layers and GPU-computed draw-arg generation to be real, and it
needs neither text shaping nor the uniform-content layout kernel.

---

## 5. GPU compute pipeline

### 5.0 Upload granularity: what a patch actually guarantees

This has to be stated as an explicit contract, not left implicit in "patches
instead of full-range rewrites" (§2), because §1's table finding is exactly
the trap: today's shipped mechanism already scopes `write_buffer` to a dirty
*layer's* byte range, which reads like a delta upload and isn't one. A
10,000-quad layer with one changed quad re-uploads all 10,000 quads' worth
of bytes for that kind, because dirtiness is tracked per-layer, not
per-primitive. That's real per-region invalidation — a genuine improvement
over the pre-R-N global re-upload — but "only what changed inside a dirty
region" and "only what changed" are different claims, and 2.0 needs to
commit to the second one, not quietly settle for the first and call it
done.

**The commitment:** an update to a single primitive's value uploads O(1)
bytes — one `write_buffer(offset, size)` call scoped to that primitive's own
stable slot in the slab, never widened to cover its layer-mates. This
requires naming a GPU-side address per primitive, not just a per-layer
range — a small extension of the `InstanceKey` → slot mapping §4.0's
reconciliation already needs, carried one step further into the patch list
(§2) instead of stopping at "this layer is dirty."

Three cases, stated honestly rather than as one blanket guarantee:

- **Value updates** (a primitive's fields change, it keeps its slot): O(1)
  — one small `write_buffer` per changed primitive, or one call covering
  several if the changed primitives happen to be byte-adjacent in the slab
  (the same adjacency-coalescing `OpenSlabRun` already does for draw calls,
  `renderer.rs:1690-1723`, applied to writes instead of draws — reused
  logic, not new logic).
- **Insert/remove that forces the allocator to relocate a primitive** (a
  size-class change or compaction, per `slab.rs`'s existing free-list/
  generation model) costs a wider write for the primitives actually moved —
  bounded by the layer's own slab, never the whole scene, but genuinely not
  O(1). Disclosed here rather than glossed over, the same treatment R-N gave
  its own fragmentation risk (R-N §4.3).
- **A clean layer uploads zero bytes** — not "a small range," zero. Already
  true today (R-N's shipped mechanism) and does not regress.

**Risk this introduces, stated up front:** a burst of many small, scattered,
non-adjacent primitive updates in one frame could regress into many tiny
`write_buffer` calls, trading bytes-transferred for driver call-count
overhead — a real cost on some backends. The mitigation is the same
adjacency-coalescing rule as the value-update case above, and where
coalescing doesn't help (genuinely scattered, unrelated primitives changing
in the same frame), that's measured in Phase 0's spike alongside the
ordering/occlusion/layout spikes, not assumed away.

**Gate (Phase 1, §8):** changing one primitive's color in a 10,000-primitive
layer issues exactly one `write_buffer` call sized to that primitive's
stride — measured directly via a byte-count/call-count counter
(`render_stats`-style, ported per §8's Phase 4 note), not inferred from
"the test passed."

### 5.1 Ordering

Today: `BoundsTree::insert` (`bounds_tree.rs:61`) is a CPU AABB tree, one per
ordering scope — each `.layer()` gets its own tree starting at 0, which is
R-N §4.2's per-layer narrowing, already shipped. `Scene::finish`
(`scene.rs:712-752`) then does nine separate `sort_by_key` calls, one per
primitive array (`quads`, `shadows`, `paths`, `underlines`,
`monochrome_sprites`, `polychrome_sprites`, `surfaces`,
`backdrop_filters`/`filter_boundaries`, `layer_slab_spans`) — a standard
single-threaded stable sort, run every frame a layer (or the legacy
per-frame scene) is dirty.

2.0 target: a compute pass over a dirty layer's slab computes pairwise
overlap and emits a sort key per primitive (same information the CPU
`BoundsTree` computes today, restated as a parallel bitonic/radix sort over
GPU buffers already sitting in the storage-buffer layout the renderer uses
today — §1's finding that instancing is already vertex-pulling means this
compute pass writes into the exact buffers the existing draw calls already
read, no pipeline restructuring required). A clean layer's sort key buffer
is untouched and reused as-is, exactly matching R-N §5.1's rule ("order
invalidation is per-layer") — 2.0 changes *where* the sort runs when it does
run, not when it runs.

### 5.2 Occlusion culling

R-N §8 already designed this as a two-tier, provably-a-no-op system built on
each layer's `BoundsTree` (layer-tier at composite time, instance-tier at
emission time for dirty layers only), with a `WGPUI_OCCLUSION=validate`
differential mode as the correctness backstop. That design does not change —
it was already correctly scoped to avoid churning clean layers (R-N §8.2).
What changes is that the instance-tier coverage test (R-N §8.3's conservative
opaque-region test: solid background, opacity 1.0, corner-radius inset,
border-opacity inset, no backdrop filter above, blur-margin exemption) runs
as a compute pass over the layer's resident primitive buffer instead of a
CPU loop, for the same reason as §5.1: it's the same computation, restated
as data-parallel, run only when a layer is dirty (per R-N §8.2's rule, which
2.0 keeps unchanged). The validate-mode safety net (R-N §8.5) is kept exactly
as designed — for GPU-computed culling it's not optional, it's the only
practical way to catch a compute-shader coverage bug, since there is no
CPU-side result to eyeball.

### 5.3 Indirect draw

Today: the renderer computes `first_instance`/count ranges on the CPU
(`quads_first_instance` and its siblings, `renderer.rs:3486-3492`) and issues
one `pass.draw(0..4, first_instance..first_instance+count)` per (layer,
primitive kind), already coalescing byte-contiguous same-layer runs into one
call (`OpenSlabRun`, `renderer.rs:1690-1723`). 2.0 target: §5.1/§5.2's
compute passes write those same two numbers — instance count, first
instance — into a GPU buffer instead of a CPU local; the CPU issues a
**fixed** sequence of `draw_indirect`/`multi_draw_indirect` calls — one per
(layer, kind) slot that *could* be populated — every frame, regardless of
how many are actually zero. A clean window's CPU-side draw-issuing cost
becomes O(layer slots), not O(resident primitives) and not O(dirty layers)
— the GPU decides how much work each indirect call expands to, including
"none," without the CPU ever finding out the count.

This is a smaller lift than it would be in most codebases: `render_context.rs:104-176`
already requests and negotiates `INDIRECT_FIRST_INSTANCE` (hard-required on
native outside macOS) and `MULTI_DRAW_INDIRECT_COUNT` (best-effort on
macOS, hard-required elsewhere) at device creation, correctly split by
platform, and the crate already has hard-won, documented experience with
exactly the failure mode that matters — drivers silently dropping indirect
draws with nonzero `firstInstance` (`README.md`, "Custom Device Gotcha",
originally hit via externally-embedded Helio content). §1's table's
sharpest finding is that this negotiation code has shipped and worked in
production for a feature the crate's own primitive renderer has never used.
2.0's job in this phase is narrower than "solve indirect dispatch" — it's
"finally call the draw functions the device was already asked to support,"
plus the one genuinely new piece: a CPU-readback fallback (compute writes
the args, CPU reads them back and issues direct draws) for the macOS
best-effort case and for WASM, which are the same fallback path per
decision 2 (§0) and therefore one piece of work, not two.

### 5.4 Invalidation vocabulary, extended — and one axis actually turned on

R-N's four axes (`LAYOUT`/`DISPLAY`/`HIT`/`TRANSFORM`, R-N §3.2) and its
typed `InvalidationRequest`/`InvalidationScope` (R-N §6) are kept unchanged —
they're the right granularity and nothing above changes what gets
invalidated, only how the resulting work is executed. But `TRANSFORM`
specifically needs more than "keep it": per §1's table, it's a live bitflag
(`Invalidation::TRANSFORM`, `window.rs:409-416`) that nothing in the crate
sets — the crate's own comment says as much. That's not a rounding error;
it's the specific bit that was supposed to make a scroll tick cost one
matrix, and it has never fired. 2.0's compositing (§5.1–5.3, all
GPU-resident and patch-driven) is what finally makes "independently
composited" true in the sense R-N's own comment is waiting for, so wiring
real `TRANSFORM`-only invalidation is folded into Phase 2 (§8) rather than
left as a fifth thing to remember.

One addition on top, pulled forward from SFD §1.1's "needs its own change,
not bundled into scroll" note: a fifth *signal* kind (distinct from the four
invalidation *axes* above), **`Reason::Scroll`**, distinguishable from
`Reason::DataChanged` at the point invalidation is raised — not inferred
after the fact by a key someone remembered to write, which is exactly what
made this hard to retrofit safely in SFD's own telling. `.boundary()` (§4)
consumes this directly to decide whether a notification can resolve to
`TRANSFORM` alone; every other consumer of `cx.notify()` is unaffected.

### 5.5 The obvious fast path this opens up: `WgpuSurface`

`wgpu_surface()`/`WgpuSurfaceHandle` (`elements/wgpu_surface.rs`, 545 lines)
is what the README calls "the unique capability that makes WGPUI suitable
as a shell for 3D applications" — an external render thread renders into a
triple-buffered texture at its own pace, and the compositor samples
whatever's latest-ready whenever it draws. It is also, structurally, the
single best-fitting case for everything §4/§5 build, and today it uses none
of it. Two concrete gaps, both closeable, one of them essentially free once
§5's general mechanism exists:

**Gap 1 — it has no identity, so it gets none of §4.0's retention for free.**
`WgpuSurface::id()` is hardcoded to return `None`
(`wgpu_surface.rs:449-451`) — not a fundamental limitation, a specific
historical choice — which means it can never be addressed by
`InstanceKey`/`GlobalElementId` and so never participates in reconciliation,
Taffy-node reuse, or `.boundary()` at all. Concretely, `request_layout`
builds a fresh `Style::default()` and calls `window.request_layout`
unconditionally every single frame (`wgpu_surface.rs:457-468`) — the *exact*
"no `diff_key`, full rebuild" path §6.2 flags as acceptable only for
third-party elements, except here it's forced, not chosen, because there is
no identity to hang a `diff_key` on in the first place.

This element is actually the cleanest possible case for reconciliation to
handle, once it has identity: its pixel *content* is never part of the CPU
description at all — it's produced by someone else's render loop and the
compositor always samples whatever's currently `ready`, so the framework
never needs to ask "did the texture change" the way it asks that question
for a `div`'s children. A `diff_key` comparing only `(bounds, style,
surface_id)` is sufficient and correct by construction, because those are
the only three things that affect *its own* composite entry (Taffy leaf,
order-tree position, indirect-draw slot). Give it real (positional, per
SFD §1.0's mechanism) identity and that trivial `diff_key`, and an
unmoved, unresized 3D viewport panel — the common case, e.g. sitting still
while the user edits an unrelated inspector panel — costs the framework
nothing per frame beyond confirming "unchanged," instead of rebuilding a
style and re-registering a Taffy leaf unconditionally, forever. Panning a
panel that contains a live viewport becomes a `TRANSFORM`-only, one-matrix
update (§5.4) rather than touching layout/prepaint/paint every frame
regardless of whether the panel moved. This is a small, self-contained fix
— folded into Phase 2 (§8) alongside the rest of the positional-identity
work it depends on — not a new mechanism.

**Gap 2 — it composites through a second, parallel pipeline that duplicates
what §5 already has to build.** `WgpuSurface::paint` pushes into
`Scene.surfaces` via `window.paint_wgpu_surface` (`wgpu_surface.rs:517-530`),
drawn through the dedicated `surfaces` pipeline backed by `SurfaceRegistry`
(`surface_registry.rs`, 772 lines) — a triple-buffered, atomically-swapped,
generation-tracked texture handoff built specifically for a render thread
that runs independently of the compositor. That machinery is *correct and
necessary* for its actual job (cross-thread producer/consumer pacing,
backpressure via `has_unconsumed_frame`, GPU-synced swaps) and nothing here
proposes touching it. But the *consuming* half — composite an
already-rendered, externally-owned texture into the ordered scene, at the
right z-position, GPU-side-transformed, occlusion-aware, via one indirect
draw entry — is exactly the general mechanism §5.1–§5.3 already has to
build for `.boundary()`'s texture-retained layers (today's
`LayerTextureEntry`, `renderer.rs:2190-2201`, a *second*, single-buffered,
framework-baked texture pool doing conceptually the same compositing job).
Two composite pipelines exist today for one operation. 2.0 doesn't need a
third — a `WgpuSurface` becomes the degenerate case of a compositing
boundary: "here is a texture, produced externally instead of baked by the
rasterizer, composite it exactly like a boundary's baked texture." Once
that unification lands (folded into Phase 4, §8, where the general
indirect-draw compositing entry is built), `WgpuSurface` content gets
layer-tier occlusion culling for free (§5.2) — a 3D viewport fully covered
by a modal stops being drawn at all, which it cannot today, since
`Scene.surfaces` participates in ordering but not in the conservative-
opaque-region occlusion sweep, which requires the layer-local `BoundsTree`
identity Gap 1 is what actually blocks.

Both gaps trace back to the same root cause — `id() -> None` — which is
worth landing first (Phase 2) precisely because Gap 2's unification becomes
easy to state and test once Gap 1 gives `WgpuSurface` a real place in the
layer/order system to unify *into*.

---

## 6. What stays on the CPU, on purpose

Not everything belongs on the GPU, and pretending otherwise produces the
same failure class R-N explicitly rejected for other reasons (R-N §11):
mechanisms whose correctness depends on an assumption nobody validated.

- **Heterogeneous flexbox/grid layout stays on Taffy, on the CPU.** Taffy's
  algorithm is recursive over children of arbitrary, data-dependent size —
  that's inherently sequential in the general case and a poor compute-shader
  target. 2.0 does not attempt to reimplement general flexbox in WGSL. What
  moves to the GPU is narrower and concrete (§6.1).
- **Text shaping stays on the CPU**, via `cosmic-text`, unchanged. Shaping
  is branch-heavy, font/cache-dependent, and not a good data-parallel target
  with today's tooling; GPU text-shaping techniques exist in research but
  are not mature enough to gate this plan on (§9, Rejected). What *is*
  data-parallel and already GPU-appropriate — placing already-shaped glyphs
  as instanced sprites — stays exactly that, patched through the same
  persistent-slab protocol as every other primitive (§2). This closes R-N's
  self-documented gap (`Img`/`StyledText` never got `diff_key`, R-N Phase 7
  table; SFD §3) as an ordinary consequence of giving every primitive kind
  the same patch path, not as a special case.
- **Hit-testing, focus, actions, and input dispatch stay on the CPU.** These
  are small, latency-critical (must resolve within one input event, not one
  frame), and already cheap — R-N §5.2's point-transform hit test (inverse-
  transform the query point per layer, not every hitbox) is the right
  design and is kept as-is.
- **User code — `Render::render`, event handlers, `on_frame` — stays exactly
  where it is and runs exactly as it does today.** This is not a limitation
  to route around; it's the actual constraint that makes "frontend doesn't
  change" possible at all. Arbitrary Rust closures cannot execute on a GPU
  compute pipeline, and the frontend/backend boundary (§2) is deliberately
  drawn *after* all such code has already run, so this requirement and "the
  backend can do whatever it wants" are compatible by construction rather
  than in tension.

### 6.1 What does move: regular-content layout

The scoped, real GPU-layout target is content whose position is a function
computable independently per item, in parallel — exactly the case
`uniform_list`/`h_list` already special-case on the CPU today (bypassing
Taffy entirely, per `uniform_list.rs`'s own doc comment: "rather than use
the full taffy layout system, uniform_list simply measures the first element
and lays out all remaining elements in a line"). That CPU special case
*already proves the content shape is regular enough* — 2.0's job is to give
it a compute kernel instead of a CPU loop: item count, a per-item size
(uniform, or from a size buffer computed once), and container bounds go in;
final positions are written directly into the instance buffer, in parallel,
with zero Taffy node creation. The same applies to a flex row/column of
same-sized children generally, not just the two hand-built element types
that special-case it today — a strict generalization, not a new concept.

Any content that breaks the "position is independent per item" property
(auto/content-dependent sizing that shifts later siblings, nested regular
layouts inside heterogeneous ones) falls back to Taffy on the CPU, exactly
as it does today for non-uniform content. This mirrors SFD §0.-3's own
containment design (`estimated_size` → `None` falls back to real layout,
never guessed) — the fallback is always "do it for real on the CPU," never
a wrong GPU answer.

### 6.2 The `diff_key`/`estimated_size` invariant

Constraint 5 (§0) only pays off for an element type that actually
implements `diff_key` — the trait's own default (`None`, `element.rs:136-
138`) means "assume changed, rebuild," which is the correct, unavoidable
default for a *third-party* `Element` impl (its purity can't be proven from
outside), but today it is also the state of two of the framework's own
built-in types: `Img` and `StyledText` never got one (R-N Phase 7's own
table; SFD §3). Under R-N/SFD that was a bounded, documented gap. Under
ambient reconciliation (§4.0) it stops being bounded — every `Img` or
`StyledText` anywhere in the tree, boundaried or not, is now the one kind
of node that still forces a full rebuild of itself on every notification of
its owning view, which is precisely the two element kinds SFD §3 already
identified as dominating real list rows (avatars, thumbnails, rich text).

So this is elevated from a one-off fix (previously scoped to Phase 5) to a
standing rule: **every first-party element type ships with `diff_key`
implemented, and `estimated_size`/`on_frame` wherever they apply** — it is
part of what "adding an element to this framework" means, checked the same
way a new element's `Element` impl is checked for panics today, not treated
as a follow-up someone gets to eventually. The permissive `None` default on
the trait itself does not change — it stays exactly right for foreign code
— only the bar for what ships inside `wgpui-widgets` does.

---

## 7. Frontend contract, made testable

"The public API doesn't change" is enforced, not just stated:

- Every existing example under `examples/learn/`, `examples/bench/`, and
  `examples/legacy/` must compile and run unmodified against `wgpui-core`
  once it's wired in as an alternate backend — not "mostly," not "with the
  caveats listed here." A CI job builds every example against both backends.
- Where feasible, a pinned snapshot of the real consuming application's UI
  code (the one SFD's grep already sampled — "32 of 37 hand-rolled scroll
  containers") compiles against both backends with zero source changes
  beyond the dependency line. This is the test SFD's own findings say
  matters: the framework's *usage* in a real app, not just its examples.
- The only source-visible changes anywhere are the caching-primitive
  replacement in §4 (`.layer()`/`.layer_keyed()`/`.layer_with_policy()`/
  `Panel::cacheable` → `.boundary()` with its `BoundaryPolicy`/`Buffering`
  enum (§4.1, §4.3), plus the new `.uncached()`, §4.2), which was explicitly
  called out as in-scope. Everything else — `div()`, the `Styled` DSL, `Render`,
  `Entity<T>`, `Context<T>`, actions, keymaps, `uniform_list()`/`list()`/
  `virtual_list()`, `canvas()`, `wgpu_surface()` — is byte-for-byte the same
  surface, verified by the compile check above, not by inspection.

---

## 8. Phasing

Each phase has a falsifiable gate, following R-N's own discipline (R-N §10).
Phases 1–3 are almost entirely additive (new crates, no behavior change to
the default backend); the switch to `wgpui-core` as default happens only at
Phase 8, after parity is demonstrated, not assumed.

| Phase | Work | Gate |
|---|---|---|
| **0** | Workspace scaffold (`crates/wgpui-core`, `-layout`, `-text`, `-widgets`, `-devtools`, `-wgpu`, empty or thin). Port `flamegraph_gpu.rs`'s GPU timestamp capture as the baseline instrumentation. **Spike, not build**: a synthetic 100K-quad scene's ordering+occlusion as a compute pass vs. today's CPU `BoundsTree`; a 10,000-row uniform grid's layout as a compute kernel vs. today's CPU loop. | Numbers exist, on real target hardware, for the two spikes that decide whether Phases 3 and 6.1 are worth building at all. If a spike doesn't win, the corresponding phase is rescoped or dropped here, not discovered mid-build. |
| **1** | `wgpui-core::patch` — the patch-list protocol (§2): insert/update/remove for primitives, layout inputs, hitboxes, dispatch nodes. `wgpui-core::scene` — persistent per-layer slabs (R-N Pillar III's concept, 2.0's mechanics), accepting patches instead of full-range rewrites. **Ambient reconciliation (§4.0) ships here, window-wide, with no `.boundary()` involved at all** — `diff_key`/`InstanceKey` reconciliation and Taffy node reuse apply to every element in the tree by construction, not fenced to any subtree. **`.uncached()` (§4.2) ships in this same phase, not later** — shipping the default without its escape hatch would leave high-frequency-update UIs (the game-engine-editor workloads this crate targets) with no way to opt out of reconciliation bookkeeping that provably never pays off for them. No compute yet; CPU-computed draw ranges, same as today, just through the new protocol. | A round-trip test: apply a patch sequence, read back the resident buffer, matches an equivalent full-rebuild reference exactly. **Separately, and just as load-bearing**: a plain, unboundaried three-level-deep div that renders identically to last frame keeps the same `LayoutId` and skips `prepaint`/`paint` — with zero `.boundary()`, zero `.id()`, zero API touched anywhere in the test. **A third, symmetric check**: a `.uncached()` subtree allocates no `ElementInstance` and its children's state (`use_state`, focus) survives across frames identically to a reconciled subtree's — proving the two mechanisms are actually decoupled, not just documented as such. **A fourth check, per §5.0**: changing one primitive's value inside a large layer issues one `write_buffer` call sized to that primitive's stride, not the layer's full range — the actual delta-upload guarantee, checked from Phase 1 rather than assumed to fall out of the compute phases later. |
| **2** | `.boundary()` (§4.1) implemented as a pure compositing/buffering policy on top of Phase 1's already-ambient reconciliation: independent GPU texture retention, auto positional identity for the boundary root, and the `Reason::Scroll` signal (§5.4) from day one — not retrofitted. Old `.layer()`/`.layer_keyed()` etc. keep working unchanged in the legacy backend; `.boundary()` only exists in `wgpui-core`. **`WgpuSurface` gets real (positional) identity + a trivial `(bounds, style, surface_id)` `diff_key` here too (§5.5, Gap 1)** — it depends on the same positional-identity work this phase already builds, and unblocks Phase 4's compositing unification. | `.boundary()` with zero policy arguments reaches R-N's fast path (transform-only recomposite on scroll) on a plain `overflow_y_scroll` div with no other API touched — the exact test SFD Pass A's gate specifies, now true by default rather than by opt-in. Additionally: removing `.boundary()` from that same test case degrades the scroll case to a per-tick recomposite (no independent texture) but does **not** reintroduce full rebuild — confirming boundary and reconciliation are actually decoupled, not just documented as such. **`WgpuSurface` check**: an unmoved, unresized `wgpu_surface()` element skips `request_layout`/`prepaint`/`paint` across frames exactly like a reconciled `div` would. |
| **3** | GPU compute ordering + occlusion (§5.1, §5.2) over Phase 1's slabs. `WGPUI_OCCLUSION=validate`-equivalent differential harness ported and run against the compute path. | Culled/unculled scenes match exactly over a scripted UI walk (R-N §8.5's bar, unchanged). Spike numbers from Phase 0 reproduced on the real pipeline, not just the synthetic case. |
| **4** | Indirect draw-arg generation + `multi_draw_indirect` (§5.3), with the CPU-readback fallback for drivers/WASM that can't take it. **`WgpuSurface`/`SurfaceRegistry` compositing unification (§5.5, Gap 2)**: the *consuming* half of `SurfaceRegistry`'s composite path is folded into the same indirect-draw entry mechanism `.boundary()`'s texture-retained layers use; `SurfaceRegistry`'s producer-side triple-buffer, atomic generation tracking, and `gpu_submit_lock` cross-thread synchronization are untouched — nothing about how the external render thread paces itself changes. | A clean window's CPU-side draw-issuing work is O(layer slots), independent of resident primitive count, measured directly (same `render_stats`-style counters R-N used, ported into `wgpui-core`). **`WgpuSurface` check**: a viewport panel fully covered by a modal (occlusion-culled per §5.2/Phase 3) issues zero draws for its embedded 3D content, and `WgpuSurfaceHandle`'s existing concurrency tests (`submit_guard`, backpressure via `has_unconsumed_frame`) pass unmodified against the unified consumer path. |
| **4.5** | `Buffering::Tiled` (§4.3): `LayerKey` extended to `(boundary, TileCoord)` addressing; tile-visibility compute pass driving indirect draw-arg generation for the in-range tile set (reusing Phase 4's mechanism directly); spatial mark-and-sweep eviction plus a total resident-tile budget with LRU eviction beyond it. | Panning a node-graph-style canvas across a tile boundary renders only the newly-revealed tile(s) — measured directly, not inferred — while panning within the resident grid costs one `TRANSFORM` update per visible tile and zero render/reconcile/layout work anywhere. |
| **5** | `wgpui-text`: shaped-run → patch conversion; `Img`/`StyledText` get the `diff_key` R-N Phase 7 left undone — the first two elements checked off against §6.2's standing invariant, not a one-off fix (closing SFD §3). Atlas-eviction subscription for GPU-resident glyph/sprite tiles (R-N §4.3's hazard, same fix, new home). | Scroll-content-heavy scenes (avatars, multi-run text — SFD §3's stated motivation) hit the fast path with no per-refill shaping cost for unchanged rows, under ambient reconciliation (Phase 1), not because they're inside a `.boundary()`. |
| **6.1** | GPU compute layout kernel for regular content (§6.1): uniform lists/grids and same-sized flex runs. Heterogeneous content is untouched, still Taffy/CPU. | `uniform_list`/`h_list`'s existing CPU special case and the new kernel produce identical positions on the same input (differential test, same philosophy as R-N's hit-test differential test) across a range of item counts; CPU cost for layout is O(1) dispatches, not O(item count). |
| **7** | `wgpui-devtools` extraction (move, don't rewrite, the flamegraph/replay/inspector system onto a small hook trait `wgpui-core` exposes). File breakup of what remains monolithic in `wgpui-core`'s own modules (target: no file over ~1,000 lines). Can run in parallel with 1–6 — it's orthogonal risk — sequenced last here only because it's cheapest once most call sites have already been touched by the phases above. | `wgpui-core` compiles and runs with `wgpui-devtools` absent entirely (feature-gated), proving the dependency is genuinely one-directional. |
| **8** | Parity checklist (§7) passes: every example, plus the pinned real-app snapshot where available, compiles and renders equivalently on both backends. `wgpui-core` becomes the default; root `gpui-ce` becomes a thin re-export. Legacy immediate-mode paths, `AnyView::cached`'s replay mechanism, and the `WGPUI_*` flag ladder are deleted — this is R-N's own never-finished Phase 12, finally safe to do because there is no longer a CPU path underneath it that anything still depends on. | No known workload needs the legacy backend as anything but a documented rollback tag. |

Phases 1–2 are independently valuable and low-risk (new crate, no default
change). **Phase 0 is the actual decision point** — if either spike doesn't
show a real win on real hardware, that's the moment to descope, not five
phases later. Phase 8 is the only phase that changes what a consumer gets by
default, and it's gated on parity being demonstrated, not scheduled.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| GPU layout kernel diverges from Taffy's exact rounding/min-max/gap semantics for edge cases | Differential test against Taffy's own CPU output is the Phase 6.1 gate, not a follow-up; any regular-content case the kernel can't reproduce exactly falls back to Taffy, same discipline as SFD's `estimated_size` containment fallback (SFD §0.-3). |
| Indirect-dispatch driver/feature gaps (older GPUs, some Vulkan/Linux drivers, WASM/WebGPU) — the crate has *already* hit this class of bug for externally-embedded content (`README.md`'s `INDIRECT_FIRST_INSTANCE` note) | Feature-detect at device creation; CPU-readback fallback (§5.3) is built as a first-class path from Phase 4, not a patch bolted on later, and doubles as the WASM path. |
| GPU-resident state is much harder to introspect than a CPU `Vec` when something's wrong | The existing DeepCapture/replay system (`flamegraph_replay.rs`, "RenderDoc for our UI framework") already solves the actual problem — extend it to read back compute-written buffers, don't rebuild a debugging story from zero. This is the strongest argument for moving devtools (§3.6) rather than deleting any of it. |
| Two backends live simultaneously for the length of the migration | Phase-gated parity checklist (§7/§8) with a hard, explicit cutover phase, not an indefinite dual-maintenance state; legacy backend is frozen (bugfixes only) once Phase 1 starts, so it's not a moving target. |
| Scope creep — R-N's CPU-side retained rendering alone took from `#98` to `#148`, ~7 months of merged phase work, for a narrower goal than this one | Every phase above has a single falsifiable gate; Phase 0's spikes exist specifically to kill scope before it's committed to, not after. |
| Public API compatibility silently breaks because an internal type (`LayerPolicy`, `Interactivity`, `ScrollHandle`) leaks GPU-shaped fields into the public surface | §7's compile-check CI job runs on every PR touching `wgpui-core`, not as a pre-release gate — catches the break at the commit that introduces it. |
| **Ambient reconciliation (§4.0/constraint 5) means a `diff_key`/reconciliation bug's blast radius is the entire application from Phase 1 onward, not one opted-in subtree** — this is the direct cost of the correction that makes 2.0's default posture right | Kill switch from Phase 1 (`WGPUI_INSTANCES=0`'s precedent, `view.rs:103`, now gating the *default* path rather than an edge feature); the Phase 1 gate's second clause (a plain unboundaried div reusing its `LayoutId`) is a targeted regression test for exactly this, run continuously, not just once at Phase 1's landing. |
| Unifying `WgpuSurface`'s composite path with `.boundary()`'s (§5.5, Gap 2) accidentally touches `SurfaceRegistry`'s cross-thread producer-side synchronization (`gpu_submit_lock`, atomic generation tracking, GPU-synced buffer swaps) — hard-won, carefully-documented concurrency code that has nothing to do with the bug being fixed | Scope Phase 4's change explicitly to the *consumer* side only — how an already-produced texture enters the ordered scene and gets drawn. `SurfaceRegistry`'s producer API (`back_buffer_view`, `present_synced`, `submit_guard`, backpressure) is untouched, and its existing tests are the gate (Phase 4's row, §8), not a new test suite reverse-engineered from the concurrency doc comments. |
| `Buffering::Tiled` (§4.3) picks a tile size that's wrong for a given workload — too small inflates per-tile/draw-call overhead, too large approaches `Margin`'s whole-region-refill cost — and an erratic pan pattern keeps more tiles "recently visited" than a per-tile eviction timer alone bounds | Tile size starts from measured common-compositor practice (~256–512px) and is validated against a representative node-graph workload in Phase 4.5, not asserted (same Phase 0 spike discipline); a total resident-tile budget with LRU eviction is a first-class part of the mechanism, not a follow-up, exactly because R-N §3.4's per-layer timer alone was never designed for grid-many tiles. |

---

## 10. Rejected / explicitly deferred

- **Reimplementing general flexbox/grid layout as a compute shader.**
  Taffy's algorithm is recursive over heterogeneous, data-dependent content
  — a poor data-parallel fit with current techniques, and reimplementing a
  mature, correctness-tested layout engine from scratch is a multi-year
  research project on its own, not a phase. Scope GPU layout to the regular
  case (§6.1) where the payoff is real and the risk is bounded.
- **GPU text shaping.** Real published techniques exist (SDF/MSDF-adjacent,
  parallel shaping research), but they're not production-mature enough to
  gate this plan on, and `cosmic-text`'s CPU shaping is not, on current
  evidence, the bottleneck this crate has (R-N §2.6's own caveat: measure
  before assuming a stage dominates). Revisit only if Phase 5's
  instrumentation shows shaping actually dominating a real frame.
- **A new layout/styling DSL.** Out of scope by explicit instruction — the
  Tailwind-style `Styled` API and Taffy-based flexbox/grid model are the
  frontend contract (§7), not implementation detail up for renegotiation.
- **An automatic "always-dirty subtree" heuristic** (R-N §9's own sketch: the
  framework tracks a per-subtree payoff ratio and marks persistently-
  unprofitable subtrees `AlwaysRebuild` without being told). Real idea, not
  dismissed — but it's an adaptive layer on top of the manual primitive
  (§4.2), not a substitute for it, and it needs its own measurement story
  (what payoff ratio, over what window, before auto-marking) before it's
  trustworthy. Ship `.uncached()` first, since it's what a developer facing
  this problem today can use immediately with certainty; revisit the
  adaptive version once there's field data on how often developers actually
  reach for it, which is also the data that would validate a heuristic.
- **Forking Taffy.** Reuse the crate for the CPU-necessary cases (§6);
  narrow its call sites, don't reimplement or fork it.
- **Zoom-level/multi-resolution tiling** (browser compositors' practice of
  caching the same tile at multiple mip-like resolutions so zooming doesn't
  force a full re-render at the new scale). Real technique, genuinely more
  than §4.3 needs to solve the problem as asked — 2-axis pan buffering, not
  pan-and-zoom buffering. `Buffering::Tiled` re-renders at the current scale
  on a zoom change, same as `Margin` does today; multi-resolution caching is
  a legitimate later addition to the same tile-grid mechanism if zoom-heavy
  workloads (e.g. a level editor's minimap) show it's worth the added
  complexity, not a day-one requirement.
- **In-place flag-ladder migration** (the R-N/SFD pattern of one
  `WGPUI_*` env var per mechanism). Explicitly rejected per the workspace
  decision in §0 — it's the mechanism SFD's own §0.1 finding blames for low
  real-world adoption of a feature that already works.
- **Gating native releases on WASM parity.** Per §0's decision, WASM
  follows each phase with a CPU fallback; it never blocks native.
- **Gating reconciliation/persistent layout behind an opt-in primitive**,
  the R-N/SFD shape where `diff_key`-based skipping only applies inside a
  `.layer()`/`.boundary()` subtree. Rejected directly by constraint 5 (§0):
  opt-in retention with rebuild as the fallback is the exact shape SFD §0.1
  measured as producing near-zero real adoption, and there is no correctness
  reason reconciliation needs a subtree boundary — R-N's own `diff_key`
  mechanism is unconditionally applicable per-element; §4.0 removes the
  fence around it rather than widening the fence.

---

## 11. Immediate next actions

This document is the plan; nothing above has been built. The concrete first
steps, in order:

1. Stand up the empty workspace (`[workspace]` in the root `Cargo.toml`,
   empty `crates/wgpui-core` etc.) — mechanical, zero risk, unblocks
   everything else being developed in parallel without touching `src/`.
2. Run Phase 0's two spikes (GPU compute ordering+occlusion vs. CPU
   `BoundsTree`; GPU layout kernel vs. `uniform_list`'s CPU loop) on
   representative hardware, and write down the numbers before anything else
   in this plan is built on top of them.
3. Only after both spikes report back: start Phase 1 (the patch protocol),
   since every later phase depends on it and it's independently low-risk
   regardless of what the spikes show.
