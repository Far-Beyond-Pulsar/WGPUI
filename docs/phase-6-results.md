# Phase 6 Results — Window and Present Integration

**Branch**: `wgpui-2.0/phase-6-window-present`, off `2.0` at `1bf4a553b3`
(Phases 0–5.6 complete).

**The phase §11 called the largest genuinely open item.** Not a §8 table row
either — §8 has no Phase 6. The scope comes from §11's action 1 ("winit event
loop, surface/swapchain configuration, resize handling, an actual runnable
entry point", "sized like a real phase, comparable to Phase 1, not a wrap-up
task") and from §9's risk row that Phase 5.6 surfaced: *`2.0` has no
window/present path anywhere; nothing in seven-plus phases of real, verified
work has ever been looked at running*. This document does not edit
`docs/gpu-native-architecture.md`; updating §8, §9 and §11 is a separate act.

**Hardware**: every result below ran on an NVIDIA GeForce RTX 4060 Laptop GPU,
Vulkan, driver 561.03, `INDIRECT_FIRST_INSTANCE=true`,
`MULTI_DRAW_INDIRECT_COUNT=true`. Verified directly by
`examples/window_probe.rs` at wrap-up time rather than carried over from an
earlier phase's report. The swapchain resolved to `Rgba8Unorm` and present mode
`Immediate`. One machine, one driver, one backend — §11's action 2 still
stands, and this phase adds a second axis to it: one *window system*.

**A real window was confirmed on screen in this environment.** No
offscreen-fallback proof was needed. `examples/window_probe.rs` reports
`EventLoop::new: OK`, `create_window: OK`, `is_visible = Some(true)` before and
after presenting, and the swapchain offers both `Rgba8Unorm` and `COPY_SRC` —
so the pipelines draw straight into the presented image and that image's own
bytes are what every comparison below reads. Why that matters is §4.

---

## 1. How this phase was written, and why this report is a wrap-up

Five commits, and they were not all made by the same author. This matters
enough to state before any result:

| Commit | Author | What |
|---|---|---|
| `5d38242680` | agent | Milestone A: window, surface, swapchain, present, magenta |
| `dac18a781e` | agent | Milestones B and C: resize, and the real pipeline behind it |
| `58225a793a` | agent | Milestone D: `tests/window_present.rs`, the byte-exact gate |
| `db78d228bb` | **the project owner, by hand** | `LoopInput` parameter grouping, `uploaded_bytes` accounting |
| `c66b3ac58b` | **the project owner, by hand** | `present_mode`: `Fifo` → `Immediate` |

**The implementing agent's process was killed before it ran final verification
or wrote this document.** The three commit messages it left assert results it
never got to re-check in the tree it left behind. The owner then made two
further manual commits while personally testing frame-loop behaviour. Nobody
had run the branch in that final state.

That is the whole reason this wrap-up exists, and it earned its keep: **the
gate test did not pass in the state the branch was pushed in.** See §5.

---

## 2. What shipped, and where

| File | Lines | What |
|---|---|---|
| `crates/wgpui-wgpu/src/window.rs` | +500 | `WindowSurface`: surface creation, adapter selection against it, swapchain configuration, acquire-with-recovery, present, `SurfaceStats` |
| `crates/wgpui-wgpu/src/window/frame_loop.rs` | +626 | `FrameLoop`: the whole of §2's picture on retained state, plus `ReferenceScene`/`SolidFill`/`PlacedGlyphs` |
| `crates/wgpui-wgpu/src/window/resize_detector.rs` | +167 | Event coalescing — no longer the Phase 0 stub |
| `crates/wgpui-wgpu/src/render/frame.rs` | 129 ± | `RenderTarget` extracted so a swapchain image is a legal draw target |
| `crates/wgpui-wgpu/src/render/readback.rs` | +88 | `read_texture_rgba8`, `OffscreenTarget::read_pixels`'s body moved verbatim |
| `crates/wgpui-wgpu/src/render/device.rs` | 40 ± | `instance()` / `context_for(instance, Some(&surface))`; `ComputeContext` keeps its `Adapter` |
| `crates/wgpui-wgpu/tests/window_present.rs` | +866 | The gate: 19 checks against the presented image |
| `crates/wgpui-wgpu/examples/phase6_window.rs` | +784 | The runnable entry point, and a gate in its own right (non-zero exit) |
| `crates/wgpui-wgpu/examples/window_probe.rs` | +186 | Phase 0's adapter probe, one level up: does a window open here at all |

3,322 insertions, 76 deletions, 11 files. **`src/`, `docs/gpu-native-architecture.md`
and the root `Cargo.toml` are untouched** — confirmed by
`git diff 1bf4a553b3..HEAD -- src/ docs/gpu-native-architecture.md Cargo.toml`
returning empty, not by inspection.

Two architectural moves are worth naming because they touched code four phases
had already proved:

- **`RenderTarget`.** Phase 4's `FrameRenderer::render` took an
  `&OffscreenTarget`, which it could, because the only thing a frame had ever
  been drawn into was a texture this crate allocated. A swapchain image is
  allocated by the presentation engine and cannot be owned here. `render_to`
  now takes a view plus extent plus clear colour, and `render` is a one-line
  delegation through `OffscreenTarget::target()`. Every Phase 4/5.6 offscreen
  test drives the identical body — that is what makes "the offscreen test and
  the window draw the same frame" a property of the code rather than a claim.
- **`read_texture_rgba8`.** `OffscreenTarget::read_pixels`'s body, moved
  unchanged to a free function over `&wgpu::Texture`, so the swapchain image
  and an offscreen texture are unpacked by the same row-depadding code. A
  comparison between the two is only a comparison if both sides were unpacked
  the same way.

Neither changes behaviour for any earlier phase's tests, and all of them still
pass (§8).

---

## 3. The two manual commits, read rather than routed around

### `db78d228bb` "save" — sound, and it fixed something the agent could not have

Three changes across `frame_loop.rs`, `phase6_window.rs`, `window_present.rs`:

1. **`FrameLoop::draw`'s eight parameters became one `LoopInput<'a>` struct.**
   This is not cosmetic and it is not arbitrary: `draw` took `&mut self,
   device, queue, description, atlas, target, mode, signals, composites` —
   **nine arguments, against clippy's `too_many_arguments` threshold of
   seven**, which `clippy.toml` does not raise. Checked rather than assumed: a
   nine-argument method run through `clippy-driver` reports *"this function has
   too many arguments (9/7)"* under `#[warn(clippy::too_many_arguments)]`, which
   is on by default. With `--deny warnings`, the branch as the agent left it
   would not have passed its own clippy gate. The owner hit it, and fixed it the
   way the codebase already solves the same problem one level up (`FrameInput`
   in `frame.rs`). The doc comment the owner left says exactly that. This is a
   correct fix, in the right shape, to a real defect the agent never got far
   enough to see.
2. **`uploaded_bytes` accumulated and printed by the example.** This is the
   measurable half of the agent's own fingerprint finding, which until then was
   only observable as a boolean (`was_idle`). It turns a claim into a number,
   and the number is stark — see §6.
3. A `cargo fmt` pass over the touched regions.

**Assessment: sound, and an improvement.** No behaviour change, no race, no
leak. The grouping is a pure signature refactor; every field is passed straight
through to the same place it went before, which I checked hunk by hunk.

### `c66b3ac58b` "Update window.rs" — right intent, one real defect

One line: `present_mode: Fifo` → `Immediate`, with a comment explaining that
`Immediate` presents without waiting for vsync so a window can be resized at
the rate the OS delivers `WM_SIZE` events rather than at the display refresh.

**The intent is right and the effect on this machine is right.** It is also
what every resize number in this report was measured under, and it is a genuine
improvement for the thing the owner was testing.

**The defect is that it was named outright rather than chosen from the
capability list, and `wgpu` does not fall back for it.** From
`wgpu-core-30.0.0/src/device/surface_config.rs`: only `AutoVsync` and
`AutoNoVsync` are resolved against `caps.present_modes`; any other mode the
surface does not offer returns
`ConfigureSurfaceError::UnsupportedPresentMode`, which
`backend/wgpu_core.rs`'s `configure` routes to `handle_error_nolabel` — a
panic under the default handler, **inside `WindowSurface::new`, where there is
no `Result` left for a caller to do anything with**. `Immediate` is not
universal: WebGPU exposes only `Fifo`, and some Wayland/Mesa configurations
offer `Mailbox` without `Immediate`. On such a machine Milestone A does not
degrade — it aborts at window creation.

It is also the one field in that struct literal that was not capability-checked.
`format` is checked and has its own named error (`WindowError::NoTargetFormat`);
`alpha_mode` is picked from `capabilities.alpha_modes.first()`. The manual edit
introduced the only exception.

**Fixed** (§5.3), preserving the intent exactly: this machine still gets
`Immediate`, confirmed by the example printing
`surface: Rgba8Unorm Target Immediate at 800x500`.

---

## 4. Milestone A — a window opens, presents, and the magenta is real

**Status: met.**

`PROOF_MAGENTA` is `[255, 0, 255, 255]` in `Rgba8Unorm`, chosen because black is
what an uncleared attachment, a failed draw and a device-lost swapchain all look
like, so black on screen proves nothing. Magenta is produced by no default path
in this crate.

Evidence, re-run at wrap-up:

- `examples/window_probe.rs`: a real window opens (`is_visible = Some(true)`),
  a surface is created from it, `request_adapter` with `compatible_surface`
  picks a presentable adapter out of the several this machine enumerates, the
  swapchain configures, eight frames present.
- `cargo run -p wgpui-wgpu --example phase6_window -- --verify --frames 20`:
  **20 frames drawn, 20 verified, `configures: 1, acquires: 20, suboptimal: 0,
  retries: 0, skipped: 0, lost: 0, presents: 20`, exit `OK`.** Each of those 20
  is the swapchain image read back before present and compared pixel by pixel
  against `[255, 0, 255, 255]`.
- `tests/window_present.rs`: **8/8 presented images were exactly the clear
  colour, 3,200,000 pixels compared.**

The readback is of the image handed to `Queue::present`, taken immediately
before presenting it, which is possible because the surface is configured with
`COPY_SRC` and because its format is `TARGET_FORMAT` itself. Both were checked
by the probe, not assumed. So this is **not** the offscreen-parallel-render
fallback the phase brief allowed for: nothing renders the scene twice and
compares the copies.

---

## 5. Milestone B — resize is real, and the two things verification found

**Status: met, after fixing a defect in its own gate.**

### 5.1 The strong form: real window-manager resizes

`cargo run -p wgpui-wgpu --example phase6_window -- --scene --resize --verify`
drives `request_inner_size`, waits for the window manager's answer to arrive as
a genuine `WindowEvent::Resized`, and reconfigures on that. Down-then-up
repeatedly, since shrinking is the direction that frees swapchain images:

```
surface stats:           SurfaceStats { configures: 10, acquires: 66, suboptimal: 0,
                                        retries: 0, skipped: 0, lost: 0, presents: 66 }
resize events seen:      12
reconfigurations:        9
sizes presented:         [(800,500), (900,560), (640,400), (320,200), (160,100),
                          (1100,700), (200,140), (1280,800), (240,160), (800,500)]
frames drawn:            66
frames verified:         66
OK
```

**66 of 66 frames verified off the swapchain, 0 lost, 0 skipped, 0 suboptimal,
0 retried, across nine real reconfigurations at ten distinct sizes down to
160×100 and up to 1280×800.** `retries: 0` is the number worth stating: the
loop takes the pending resize and configures *before* it acquires, so the
`Outdated` a naive order would hit on the frame after every resize never
happens.

`ResizeDetector` stops being the Phase 0 stub. It is deliberately **not** the
legacy detector — that one polls global mouse state through `device_query` to
answer "is the user still dragging", a policy this phase has no layout to
apply. It keeps the mechanical half: a drag emits resize events far faster than
frames are drawn and `Surface::configure` waits for the device to go idle, so
one configure per *frame* rather than per *event* is the difference between a
resize and a stall. Twelve events became nine reconfigurations above; four unit
tests cover the burst, the no-op resize back to the configured size, the
minimize (Windows reports 0×0, which `configure` rejects, so it is counted and
dropped rather than deferred), and shrink-then-grow.

### 5.2 What verification found: the gate test had never passed

**`tests/window_present.rs` failed on the branch as pushed.** Not marginally —
one of its eighteen checks:

```
FAILED: one `Surface::configure` per size change, no more: 15 for 8 sizes
```

Commit `58225a793a`'s own message claims "8 resize sizes down to 64x48 and back
to 800x500, each presenting correctly, **exactly one `Surface::configure` per
size**, 0 lost". That clause was asserted, not observed; the agent was killed
before it could run what it had just written.

**The bug is in the harness's premise, not in `WindowSurface`.** The test drives
`WindowSurface::resize` directly — deliberately, and the file documents why: a
real WM resize depends on what the window manager does with the request, which
is right for a harness a human runs and wrong for a deterministic test. But that
means it configures the *swapchain* to eight extents while the *window's client
area* stays 800×500. On Vulkan an extent that disagrees with the window's is
precisely what `VK_ERROR_OUT_OF_DATE_KHR` reports, so the next
`get_current_texture` is `Outdated` and `acquire`'s documented recovery —
reconfigure once, ask once more — answers it. 8 resizes + 7 recoveries = 15.

I did not infer this. Two experiments settled it:

- **Reverting only the manual `Immediate` commit and re-running gave byte-identical
  numbers** — 15 configures, 7 retries. So the failure predates both manual
  commits and belongs to `58225a793a`.
- **The fixed test now names which sizes needed recovery**, and it is exactly the
  seven that differ from the window: `[(900,560), (400,260), (160,100),
  (64,48), (1100,700), (200,140), (1280,800)]`. The eighth, `(800,500)`, equals
  the client area and needed none. That is the hypothesis, confirmed by the
  mechanism rather than by the count happening to fit.

**Fix.** The claim the check exists to make — `resize` does not double-configure
— is made against the difference, since `SurfaceStats` already counts recoveries
separately, and the recoveries are bounded and named rather than absorbed:

```
`WindowSurface::resize` configured exactly once per size change:
  15 configures less 7 recovery reconfigures is 8 for 8 sizes
each stale swapchain cost at most one reconfigure-and-retry: 7 across 8 sizes,
  at [(900,560), (400,260), (160,100), (64,48), (1100,700), (200,140), (1280,800)]
  — the extents this harness deliberately disagrees with the window on
```

This is a correction to the *check*, not a weakening of the milestone: every one
of the eight sizes still presents byte-exact magenta, `lost` is still asserted
zero, and the unqualified `retries: 0` form of the claim is the example's, where
the window actually moves. Both are cited above, and the test's doc comment now
says which is which so the next reader does not re-derive it.

### 5.3 The present-mode fix

`present_mode` is now chosen from `capabilities.present_modes` —
`Immediate`, else `Mailbox`, else `Fifo`, which WebGPU requires every surface to
support, so the function cannot return a mode the surface will refuse. The
owner's preference is preserved and still selected here; `WindowSurface::present_mode()`
reports which one a run actually got, and the example prints it, so a report
quoting frame counts can say what paced them.

---

## 6. Milestone C — the real pipeline drives the loop

**Status: met. Not a bypassed hardcoded scene.**

`window/frame_loop.rs` runs the whole of §2's picture per frame:

```
Description → Reconciler → FramePlan → Emitter::emit → ScenePatch → Scene::apply
  → ordering/occlusion compute → indirect args → QuadPipeline/MonoSpritePipeline
  → swapchain → present
```

Nothing in that chain is new machinery; every step is a call into a stage an
earlier phase built and proved. What is new is that they run one after another,
on retained state, once per displayed frame. The quad is a `SolidFill` emitter
over whatever rectangle layout gives it — not a constant — and the text is real
`cosmic-text` shaping through `wgpui-text`, rasterised into a real `etagere`
atlas, uploaded, and drawn from its own tiles. The draw mode is
`DrawMode::best_available`, which on this adapter resolves to
`MultiDrawIndirectCount`, the strongest indirect path.

Observed: `resident=32 draws=2 slots=2 plan_builds=1/1` — 31 glyphs plus one
quad, two draw calls (one per kind), and the slot-base plans built **once each**
across the whole run, which is Phase 4's O(layer-slots) gate still holding
through nine reconfigurations.

### The finding this milestone produced, and the number the manual commit added

**A steady window never settles without a `diff_key`.** `Description::new`
attaches no fingerprint, and no fingerprint means R-N §2.3's permissive default:
assume changed, rebuild. Reconciliation still reuses the instance and its layout
node — §4.0's ambient guarantee holds — but `Emitter::emit` only takes its skip
path for a node the plan marked fully reused, and `reconcile_records` pushes an
update op per record without comparing the value it replaces. So an
unfingerprinted element re-emits and re-uploads **every frame, forever**.

Invisible to every phase before this one, because a test that draws one frame or
six cannot see it. Measured here, both arms, same scene, 20 frames — and the
second column is what `db78d228bb` made observable:

| | idle frames | bytes uploaded |
|---|---|---|
| fingerprinted | 19 / 20 | 1,552 |
| `--no-fingerprint` | 0 / 20 | 31,040 |

31,040 is exactly 20 × 1,552: a settled window re-uploading its entire scene at
display rate. The boolean said "not idle"; the byte count says how much that
costs.

---

## 7. Milestone D — proven at pixel level, and one latent bug found by reading

**Status: met.**

The gate is `tests/window_present.rs`, and it is the strong form: the bytes
compared are the bytes in the swapchain image handed to `Queue::present`, read
back through `COPY_SRC` immediately before presenting it. Nineteen checks, all
passing:

- **3,200,000 pixels** across 8 presented clear frames, byte-exact.
- **8 resize sizes** down to 64×48 and back to 800×500, each presenting
  correctly, `lost == 0`, configure accounting as §5.2.
- **57,600 quad pixels** exactly `[64, 160, 240, 255]` over the rectangle the
  scene *emitted*, ending exactly at `x=320, y=180` with no spill either way.
- **4,373 glyph texels** of the presented image byte-exact against their own
  atlas tiles, 2,494 of them inked, 0 skipped as shared between overlapping
  rasters. White-on-black reduces a rendered pixel to an identity with its
  source texel, so this asserts equality, not a threshold — Phase 5.6's
  discipline, one level up.
- A fingerprinted scene settles: 5 of 6 frames changed nothing.
- Slot bases built once each across 6 frames, not per frame.
- 30 acquired, 30 presented, 0 lost across the whole run.

The comparison cannot pass vacuously: it asserts `compared > 1500` **and**
`inked * 4 > compared`, so a blank frame fails, and the 8th check asserts the
quad's emitted rectangle equals what the description asked for so a zero-sized
quad cannot slip through a zero-iteration pixel loop.

### The check that was wrong before the renderer was

`wrong_scene_pixel` originally compared against the constant the quad was
*described* with. At 320×200 the column's children want 244px of a 200px box,
taffy's default `flex_shrink` applies, and the quad is legitimately 147px tall —
so the check failed on a correct frame. It now reads the emitted quad out of the
scene, which is what "matches what was described" has to mean once layout is in
the loop. The agent found and fixed this itself, and it is recorded because it
is the same class of error as §5.2's.

### A latent bug, found by reading the rule against `frame.rs`'s contract

`FrameLoop::draw` takes dirtiness from the patch: a layer no op names has the
same primitives in the same slots, and its previous compute results are still in
the argument buffers, so an empty patch yields the clean-frame path rather than
`Dirty::All`. That rule is right about *scene content* and **incomplete on its
own**.

Occlusion is computed against `FrameInput::clip`, and the clip is the window's
rectangle. `frame.rs`'s own module doc is explicit that a clean layer's results
from a previous frame are still sitting there. So: resize a window without
changing a single primitive, the patch is legitimately empty, step 2 is skipped,
and **the indirect arguments still describe the old rectangle**. A window shrunk
and then grown back would keep drawing the shrunk frame's cull decisions —
content that left the small viewport never comes back.

The viewport is a second, independent source of dirtiness, and the patch cannot
report it. Fixed: a frame whose clip moved is `Dirty::All` whatever the patch
says, and `LoopFrame::was_idle` now requires the viewport to be unchanged too.

**How strong this is, stated plainly.** This was found by reading, not by a
failing test, and Phase 6's own reference scene *cannot* exhibit it — its text
element is `width: 100%`, so every resize re-emits its runs and the patch is
never empty across one. I did not manufacture a second scene shape to falsify
it, so the symptom is reasoned, not observed. What *is* observed is the fix's
mechanism, via a new `FrameLoop::viewport_recomputes()` counter asserted in both
directions: the gate test checks it is exactly 1 across 6 frames at one size (a
loop that raised it every frame would be rebuilding a settled window forever),
and the example fails if it is not exactly one per distinct size presented — 10
for 10, with idle frames unchanged at 56 and `plan_builds` still 1/1, so the fix
costs a settled window nothing.

---

## 8. Check, test, and clippy status

All run at wrap-up, in the final state, on the hardware named at the top.

- **`cargo check --workspace`**: exit 0. 72 warnings, all from `gpui-ce` — the
  same baseline every phase since Phase 2 has reported, unchanged. **Zero
  warnings from any 2.0 crate.**
- **`cargo metadata --locked`**: consistent. (Phase 4's missing-lock-entry class
  of bug, checked rather than assumed.)
- **Tests — 506 passing, 0 failed, 0 ignored, 0 skipped**, scoped to the 2.0
  crates. `cargo test --workspace` was **not** run: it pulls in `gpui-ce`'s
  legacy suite, confirmed 10+ minutes without completing and unmodified by this
  branch.

  | Crate | Tests |
  |---|---|
  | `wgpui-core` | 320 |
  | `wgpui-wgpu` | 93 |
  | `wgpui-text` | 54 |
  | `wgpui-widgets` | 23 |
  | `wgpui-layout` | 6 |
  | `wgpui-devtools` | 6 |
  | **total** | **506** |

  The GPU-dependent tests were confirmed to have actually run rather than
  skipped for want of an adapter: `--nocapture` output names the real RTX 4060
  adapter 40 times, and no test reports itself skipped.

- **Clippy**: `cargo clippy -p wgpui-core -p wgpui-wgpu --all-targets --
  --deny warnings` — clean, from a genuine cold build (`cargo clean -p` removed
  253 files / 3.0 GiB first). Re-run forced over all six 2.0 crates
  (`-core`, `-wgpu`, `-text`, `-layout`, `-widgets`, `-devtools`) after touching
  every source file, so nothing was served from cache: clean. **Zero
  suppressions added.** `clippy.toml`'s conventions were read first; nothing in
  this phase trips its `disallowed-methods` list.

  Note on convention: `AGENTS.md` prefers `./script/clippy`, which runs
  `--release --all-features` across the whole workspace including `gpui-ce`.
  The per-crate form above is what every prior phase's report used and what this
  phase's brief specified; the difference is scope and profile, not lint set.

- **Formatting**: `cargo fmt` is *not* clean across `wgpui-wgpu` as a whole — it
  wants to reformat 20 files this phase never touched. Those were reverted. The
  four files this phase owns are now rustfmt-clean, which accounts for four
  small hunks in the wrap-up diff that are not mine semantically.

---

## 9. What is still open

### Deferred by name, and real

- **`poly_sprites` / images.** §11 calls this "needs a new primitive kind,
  larger", and having checked, that undersells it. There is no image path in
  `2.0` **at any layer**: `PrimitiveKind` has exactly two variants (`Quad`,
  `GlyphRun`) with no image kind; `poly_sprites.wgsl` is still a two-line
  placeholder with no pipeline behind it; `AtlasKind::Polychrome` exists as an
  enum variant with a texel width and **nothing produces a polychrome tile**;
  and there is no image *loading or decoding* anywhere — no `image` crate, no
  PNG/JPEG dependency in any 2.0 `Cargo.toml`, no decode, no upload.
  `wgpui-widgets`' `Img` is a `Description` carrying an opaque `ImageSourceId`
  and a `diff_key` (Phase 5's work), and nothing resolves that id to pixels.
  So the gap is four missing layers — load, decode, polychrome rasterise/upload,
  render pipeline — not one. Worth stating precisely, because "no render
  pipeline for `Img`" reads as much smaller than it is.
- **`shadows` / `underlines`** — flagged as `QuadPipeline`-shaped and cheap;
  still two-line placeholders.
- **`paths` / `backdrop_blur`** — still two-line placeholders, still not scoped.
- **The rest of `window/`.** `keyboard.rs`, `dispatcher.rs` and `app_menu.rs`
  are still exactly the Phase 0 stubs they have always been. **There is no input
  plumbing in this phase at all.** §11's action 1 named four things — event
  loop, surface/swapchain configuration, resize handling, a runnable entry point
  — and this phase is those four and nothing else. A window that opens, draws
  and resizes is not a window a user can interact with.

### Disclosed by this phase

- **The stale-clip fix is unfalsified in its symptom** (§7). The mechanism is
  asserted both directions; the missing-content symptom was reasoned from
  `frame.rs`'s contract, not reproduced. A scene with no viewport-dependent
  layout, resized down and back up with a pixel comparison, would close this.
- **The gate test's resize arm costs recoveries by construction** (§5.2), and
  the unqualified `retries: 0` claim lives in the example rather than the test.
  The example is a gate (non-zero exit) but is not run by `cargo test`.
- **macOS has no coverage from the test at all.** `winit::EventLoop::new()` does
  not return an error off the main thread — it *panics*, calling the situation
  "a significant cross-platform compatibility hazard". Cargo runs every test on
  a spawned thread. Windows and both Linux backends provide `with_any_thread`;
  **macOS does not and cannot**, because AppKit genuinely requires the main
  thread. There the test reports itself SKIPPED and names the example as the
  on-screen gate instead of pretending to coverage it does not have. A second
  structural constraint — one `EventLoop` per process — is why this is one test
  function running nineteen checks rather than nineteen `#[test]`s, which would
  give eighteen failures and no information.
- **One machine, one driver, one backend, and now one window system.** Every
  number here is Windows 11 / Vulkan / NVIDIA. §11's action 2 already carried
  this; this phase widens what it applies to.
- Scale factor 1 only, and glyph positions rounded by the caller, because
  `wgpui-text` still does not floor the pen the way the legacy renderer does —
  Phase 5.6's disclosure, unchanged and unnarrowed.

### Carried forward, unchanged by this phase

- **Phase 6.1's fate is still undecided** — the fused-dispatch follow-up spike
  (§8's rescoped row) has not been run, and the phase has neither been executed
  nor dropped from the table. §11's action 3.
- **§6.2 is still half-discharged**: neither `Img` nor `StyledText` has
  `estimated_size`; both have `diff_key`.
- **`wgpui-devtools` extraction (Phase 7)** has not started.
- **Final cutover (Phase 8)** has not started. `wgpui-core` is not the default;
  `src/` is still the legacy backend, frozen.
- `PrimitiveStore::reflow`'s O(n²) bulk-build cost; GPU occlusion's 1.30× loss
  on low-visibility scenes; the transcription-oracle limit on Phase 5.5's
  differential; the deliberately-preserved 2×-scale sub-pixel aliasing quirk.
  All named and measured, none yet mitigated, none yet needed.
- `gpui-ce`'s legacy test binary still has not been confirmed to finish.

---

## 10. Honest read

The four milestones hold, and the branch as pushed did not.

That is the whole shape of this wrap-up. Three of the four milestones were
genuinely built and genuinely proved by the implementing agent, and the fourth's
gate — the byte-exact swapchain comparison that is the point of the entire phase
— was written correctly in every respect except one check whose premise its
author never got to test, because the process died first. One check out of
eighteen, asserting a number that was never observed. The pattern Phase 1
(a commit that did not compile), Phase 2 (a layer leak), Phase 4 (a caching bug
that undermined its own gate's premise) and Phase 4.5 (another layer leak) all
found, found again here, at the phase whose directive was to get it right
because everything rides on it.

The two manual commits were not noise. One of them fixed a clippy failure that
would have blocked the branch and the agent never saw; the other made a real
improvement with one real portability hole in it. Neither introduced a race or a
leak — there is no threading in this phase to race: `WindowSurface`'s own doc
notes that `SurfaceRegistry`'s producer side is not wired to a window yet, so
the exclusive lock the legacy backend takes around `configure` has no second
party to lock out, and adding one would be a guess about a mechanism that does
not exist here.

What this phase actually closes is exact and worth not overstating: **`2.0` now
has a window, and what is on it has been proven correct at the byte level rather
than looked at and pronounced fine.** What it does not close is equally exact:
nothing can be typed into that window, nothing can be clicked in it, and the only
things it can draw are solid quads and monochrome text. The largest open item in
§11 is closed. It was not the last one.
