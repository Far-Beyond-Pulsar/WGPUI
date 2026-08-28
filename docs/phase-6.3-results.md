# Phase 6.3 Results — `ShadowPipeline` and `UnderlinePipeline`

Branch: `wgpui-2.0/phase-6.3-shadows-underlines`, off `2.0` at `d768f383ca`
(Phase 6.2's landing). Not merged, no PR opened.

`docs/gpu-native-architecture.md` is not edited by this report; §8's `6.3` row is
quoted here, not amended there.

## 0. The gate, and whether it was met

§8's row states it:

> Both pipelines byte-exact against legacy output, same discipline, reusing the
> now-three-times-proven pattern rather than re-deriving it.

**Met, byte-exact, with no tolerance defined and none consumed.** On the
reference adapter (NVIDIA GeForce RTX 4060 Laptop GPU, Vulkan, DiscreteGpu,
driver `561.03`, `INDIRECT_FIRST_INSTANCE=true MULTI_DRAW_INDIRECT_COUNT=true`
— read off `examples/adapter_probe`, run, not assumed):

```
phase_6_3_shadow_gate:    315392 of 315392 pixels byte-exact,
                          95268 of them painted by at least one arm
phase_6_3_underline_gate: 196608 of 196608 pixels byte-exact,
                          7011 of them painted by at least one arm,
                          across 5 wavy cases
every shadow drawn identically across 4 draw modes
every underline drawn identically across 4 draw modes
```

Unlike Phase 6.2, no ±1 tolerance was even designed in. Neither shader does the
`f32`/`f64` multiply that made one prudent there: both arms of each differential
are the *same* arithmetic on the *same* GPU, because the oracle is the legacy
shader itself rather than a CPU model of it (§2).

---

## 1. The three questions this phase was told to answer, up front

### 1.1 Were the shader files real ports or placeholders? **Placeholders. Both.**

Verified by reading them, not by trusting the comment — which is exactly what
the brief asked, because Phase 5.6 found the identical comment false for
`mono_sprites.wgsl`.

| File | Before | Legacy source |
|---|---|---|
| `crates/wgpui-wgpu/src/render/shaders/shadows.wgsl` | **2 lines** | 189 lines |
| `crates/wgpui-wgpu/src/render/shaders/underlines.wgsl` | **2 lines** | 143 lines |

Both said, verbatim, "Placeholder — moved as-is from
`src/platform/cross/shaders/{shadows,underlines}.wgsl` in a later phase." Both
were two-line comment stubs with no WGSL in them at all.

**That comment has now been false three times** (`mono_sprites` in Phase 5.6,
these two in Phase 6.3). The remaining files carrying it are `paths.wgsl` and
`backdrop_blur.wgsl` — Phase 6.4 should assume they are placeholders and be
pleasantly surprised, not the other way round.

### 1.2 Did "`QuadPipeline`-shaped" hold? **For underlines yes, end to end. For shadows, at the pipeline only.**

- **`UnderlinePipeline` is `QuadPipeline`-shaped in every respect.** Same two
  bind group layouts over the same three frame resources, same empty vertex
  buffer list, same triangle strip, same blend, and — the part the phrase does
  not obviously cover — the same *compute* treatment: an underline paints inside
  its own rectangle, so the ordering pass gets `origin`/`size` like every kind
  before it, and it is an ordinary occlusion `cullee`, which is the legacy
  sweep's own classification (`src/occlusion.rs:262` lists `Underline` beside
  `Quad`).
- **`ShadowPipeline` is `QuadPipeline`-shaped at the pipeline and not at the
  primitive.** The pipeline really is another shader against the quad layout —
  `render/pipelines.rs` needed no new mechanism. But a shadow is the first
  primitive kind in 2.0 whose **drawn extent is larger than its own rectangle**:
  the shader expands the bounds by `3 × blur_radius` per side before projecting
  them, and integrates the Gaussian across that margin. That is real
  fragment-shader mathematics well beyond `Quad`'s flat fill (an `erf`
  approximation, a four-sample numerical integration of a Gaussian, and a
  curved-corner chord term), and it forced two decisions no earlier kind needed:
  `Shadow::drawn_bounds()` for the ordering pass, and
  `CoverageItem::uncullable` rather than `cullee`.

  **And then both of those turned out to be inert.** See §4 — this is the
  phase's main finding and it is not the flattering one.

The honest summary: the phase table's "genuinely cheap" was right about the
*volume* of work and understated the *shape* of it for shadows. Underlines cost
what the table predicted. Shadows cost a real shader port plus a correct
decision whose correctness cannot yet be demonstrated.

### 1.3 Is the differential proof byte-exact? **Yes, and against a stronger oracle than any prior phase's.**

§2.

---

## 2. What the differentials actually compare

`crates/wgpui-wgpu/tests/legacy_shadow_differential.rs` and
`crates/wgpui-wgpu/tests/legacy_underline_differential.rs`.

**The oracle is the legacy shader file itself.** Not a transcription (Phase
5.5's rasteriser differential had to transcribe, because
`TextSystem::rasterize_glyph` is `pub(crate)` in a frozen crate), and not a CPU
model. Each test `include_str!`s
`src/platform/cross/shaders/{shadows,underlines}.wgsl` — so deleting or moving
the frozen file breaks the build rather than silently weakening the test —
compiles it, feeds it a buffer in the legacy struct's own layout (72 bytes for
`Shadow`, 64 for `Underline`, the numbers `src/scene.rs:1488` and `:1468`
assert), and renders it through `wgpu::BlendState::ALPHA_BLENDING`, which is
what `renderer.rs:890` selects for a non-premultiplied surface and is
field-for-field `render/pipelines.rs`'s `ALPHA_OVER`.

The other arm is 2.0's real `FrameRenderer::render_to`: patch → apply → upload →
GPU ordering → GPU occlusion → indirect-argument generation → indirect draw.
Nothing is stubbed and no path is bypassed.

### Rendering the legacy file unwrapped is legitimate, and checked

The legacy renderer does not hand these files to `create_shader_module`
directly: `renderer.rs:99`'s `slab_shader_source` prepends `slab_transform.wgsl`
and rewrites two expressions per shader to thread a per-layer translate through.
Three things make this a no-op here rather than a gap:

1. Both rewrites are the identity at a zero translate — `position + vec2(0.0)`
   in the vertex stage, `layer_world_position(p)` in the fragment stage — and
   zero is the only transform these tests use.
2. The files are *designed* to be rendered unwrapped. `slab_shader_source`'s own
   doc says "the `.wgsl` files themselves stay byte-pristine: `flamegraph_replay`
   renders them against its own bind-group layouts." This differential is a
   second such consumer, not a novel use — and `flamegraph_replay.rs:596` sets
   `premultiplied_alpha: 0` for the same offscreen situation, which is what
   these tests do too.
3. `the_legacy_source_still_has_the_shape_this_test_relies_on` asserts, in both
   files, that each rewrite pattern still matches **exactly once** — the same
   assertion `slab_shader_source` makes at load. A drift in the frozen shader
   fails this test loudly instead of quietly making the wrapper non-trivial.

### The one transcription, and it is checked

The legacy structs carry HSLA and convert in the vertex shader; 2.0 carries
straight RGBA. Every case uses a colour whose conversion involves nothing but
0, 0.5, 1, 2, 3 and 6 — values every IEEE-754 implementation represents and
combines exactly — and `the_colour_transcription_is_checked_rather_than_assumed`
runs the legacy `hsla_to_rgba` transcribed into Rust and asserts it produces
exactly the bytes handed to 2.0. **Colour-space conversion is not what these
gates prove; it is what they hold fixed so they can prove something else.**

### The clear colour, and the bug the vacuity guard caught

The first run of the shadow gate reported **39,424 of 39,424 pixels byte-exact
while painting nothing.** `OffscreenTarget::target()` clears to opaque black —
which every test before this one wants, because Phase 5.6's white-text proof
depends on it — and a 50%-alpha *black* shadow composited over opaque black is
`rgb = 0`, `alpha = 1`, which is the clear colour again.

The gate's own "must have painted something" guard caught it. Both arms now
clear to an asymmetric mid colour (`0.25, 0.5, 0.75`), asymmetric so a red/blue
swap cannot hide behind a grey shadow, and the clear byte is **measured** off an
empty render rather than predicted — `CLEAR_COLOR`'s components are not exact
multiples of 1/255, so predicting the byte would mean predicting the driver's
rounding. `measured_clear_pixel` doubles as an assertion that both arms clear
identically, which is a precondition of the whole comparison and would otherwise
have been assumed.

### Where the proof is strong

- **The shadow cases move each field independently** and are deliberately
  fractional: positions at `.25` and `.75`, sub-pixel blur radii, a radius larger
  than half the rectangle, a heavy blur wider than the rectangle it comes from,
  and one reaching past the top-left corner of the viewport. A comparison that
  only ever landed on integers would be testing far less than it looks like.
- **The underline cases are 5-of-8 wavy**, asserted (`wavy_cases >= 4`), because
  the wavy branch is the whole of the interesting fragment mathematics — a sine
  SDF with a derivative-based distance correction — and the straight branch is
  two lines.
- **Both gates have been watched failing.** The shadow gate rejects a 0.4% change
  in blur radius at 2,276 pixels, first at (61, 11), one byte apart in blue. The
  underline gate is falsified twice, once per branch: a 2.5% thicker wave
  disagrees at 543 pixels, and turning `wavy` off disagrees at 2,420. One
  perturbation would only have proved the comparison discriminates on the branch
  it happened to hit.
- **All four draw modes** — per-slot indirect, multi-draw indirect, multi-draw
  indirect count, and the CPU-readback fallback — produce byte-identical output
  for both kinds.

### Where it is narrow, stated rather than left to be found

- **Per-fragment clipping is outside both proofs.** The legacy arms are given a
  content mask far larger than the viewport, because 2.0 has no per-primitive
  clip at all — the frame's clip reaches the occlusion pass instead (§5.2). The
  clip-distance path in both legacy shaders therefore never rejects a fragment.
- **Per-corner radii are outside the shadow proof.** 2.0 carries one uniform
  radius (`Quad`'s convention, which `PolySprite` already follows), so the legacy
  `pick_corner_radius` selects the same value in every quadrant and its branch
  collapses. The legacy shader can express four different radii; 2.0's `Shadow`
  cannot.
- **One machine, one driver, one backend.** Windows 11 / Vulkan / NVIDIA, as
  every prior phase. Unchanged and unimproved.
- **Nothing has been looked at on a screen.** Both gates are offscreen readback.
  A window path exists since Phase 6, and pointing it at a shadow was not in
  scope.

---

## 3. Two legacy behaviours ported as-is rather than fixed

Phase 5.5 set the precedent explicitly when it ported the 2×-scale sub-pixel
aliasing quirk: reproducing legacy output exactly is the goal while both
backends coexist, and a quirk quietly "improved" out from under a parity gate is
worse than a quirk. Both of these are flagged in the shader source and here.

### 3.1 A zero blur radius makes both sides draw nothing

Legacy `gaussian(x, sigma)` is
`exp(-(x*x) / (2*sigma*sigma)) / (sqrt(2*PI)*sigma)`. At `sigma == 0` that is
`exp(-0/0) / 0` — `exp(NaN)`, so NaN — and the integration's `step` is also
zero, so `alpha` comes out NaN and the shadow does not appear.

A `blur_radius` of zero is a perfectly ordinary CSS box-shadow, so this reads as
a real legacy bug. It is reproduced, kept as a gate case, asserted to agree
between the two arms, and excluded *by name* from the "must have painted
something" guard rather than by lowering that guard for every case.

### 3.2 An underline's alpha is squared

`fs_underline` ends `blend_color(input.color, input.color.a)`, and
`blend_color`'s body is `alpha = color.a * alpha_factor` — so the factor *is*
the colour's alpha and the result is `a²`. The wavy branch does the same one
step further along. A 50%-alpha underline therefore paints at 25%.

`the_legacy_alpha_really_is_squared` measures this rather than asserting it: the
red channel reads **112**, against a squared-alpha prediction of **111.8** and an
unsquared one of **159.5**, and the test fails if those two predictions ever
stop differing — so it cannot pass vacuously. Fully opaque underlines are
unaffected, which is presumably why this has survived, and fully opaque is what
almost every real underline is.

**Both should be revisited at Phase 8**, when legacy stops being the thing 2.0
is measured against.

---

## 4. The claim this phase made about itself, and then falsified

This is the finding.

Layers 1–3 asserted — in a commit message and in three doc comments — that a
shadow's two adjustments were load-bearing: `drawn_bounds()` reaching the
ordering pass, and `CoverageItem::uncullable` instead of `cullee`. A frame-level
test was written to prove it end to end, and then, following this project's own
rule that a gate nobody has watched fail is a gate nobody knows works, the
mechanisms were removed to watch it fail.

**It did not fail. Nothing did.**

| Perturbation | Result |
|---|---|
| `uncullable` → `cullee` | every shadow test passes, gate included |
| ordering fed `origin`/`size` instead of `drawn_bounds` | same |
| both at once | same — 315,392 of 315,392 pixels still byte-exact |

The reason is a limitation `frame.rs` already documents for glyphs and that
bites harder here: **occlusion dispatches per primitive kind.** So:

1. The quad that would cover a shadow is in a *different dispatch* and can never
   cull it, whatever flag the shadow carries.
2. No shadow can occlude another shadow — a shadow's interior is a blurred
   gradient, so it has no opaque region at all and never qualifies as an
   occluder.
3. `occlusion.rs`'s `keep_item` *keeps* an item whose visible rectangle is empty
   rather than dropping it, so even a shadow entirely outside the clip survives.

Nothing in 2.0 can cull a shadow today. Both adjustments are **correct, matching
legacy, and currently inert.** They are kept because they are right, and because
the day cross-kind occlusion exists a shadow culled against its unblurred
rectangle would lose falloff that was never covered — the exact failure
`src/occlusion.rs:255` describes. The experiment is recorded at the code
(`patch/primitive.rs`, `render/frame.rs`, and the test's own doc) rather than
only here, so a future phase cannot come to depend on a mechanism nobody had
exercised without reading that it had never been exercised.

The test is kept on narrower and honest terms. It proves the composite a real
drop shadow produces — shadow under card, falloff visible around the card,
`Shadow` sorting below `Quad` — comes out of the real frame path. That is what
`window_shadow` and the `shadow` bench are actually about.

---

## 5. What was built, per layer

Five commits, each pushed before the next began.

| Commit | Layer |
|---|---|
| `7abb8b0cdc` | 1 — `Shadow` as a fourth primitive kind (`wgpui-core`) |
| `13f8bf92a4` | 2 — `shadows.wgsl` + `ShadowPipeline`, the fifth pipeline |
| `03126ee2cf` | 3 — the shadow gate |
| `001c7c816f` | 4 — `Underline`, `underlines.wgsl`, `UnderlinePipeline`, its gate |
| `6dac0acccb` | 5 — verification, and §4's falsification |

Shadow first, and deliberately: it is the one whose "cheap" claim was in doubt,
and finding out early whether the doubt was real was worth more than doing the
easy one first. It was real, though not in the way expected (§4).

### `wgpui-core`

`patch/primitive.rs`'s module doc has claimed since Phase 1 that adding a kind
is "a `PrimitiveKind` variant, a payload type, one `PrimitiveStore` field, and
the `match` arms the compiler points at". Phase 6.2 tested that once. **This
phase tested it twice more and it held both times** — nothing in the slab
allocator, the patch protocol, the upload planner or the indirect-draw slot
table needed a per-kind line.

Both payloads are 40 bytes padded to 48, `Quad`'s layout reasoning one field set
over. `Shadow` carries `origin`, `size`, `color`, `corner_radius`, `blur_radius`;
`Underline` carries `origin`, `size`, `color`, `thickness`, `wavy`. Every field
is asserted at its own byte offset, because the only reader is WGSL where
nothing checks a layout.

**The kind order is the legacy renderer's own tie-break, not a preference.**
`src/scene.rs:1015` reads `Shadow, Quad, Path, Underline, MonochromeSprite,
PolychromeSprite`, and at an equal draw order that discriminant decides which of
two primitives paints on top. 2.0's `PrimitiveKind::ALL` is now
`Shadow, Quad, Underline, GlyphRun, PolySprite` — the same sequence with `Path`
(Phase 6.4) removed — so 2.0's kind grouping and the legacy sorter agree about
relative paint order by construction rather than by coincidence. A test asserts
the four relations.

Inserting a kind *below* `Quad` broke one test, correctly:
`slots_name_a_reservation_and_never_an_instance_count` asserted
`draw_slots().first()` was the quad slot. It now addresses its slot by kind,
which is what it meant all along.

`CoverageItem::uncullable` gives Phase 3's already-documented `cullable: false`
case its first constructor. Its doc has named shadows specifically since Phase 3.

`Emitter`'s per-kind `KindOperations` parameters are grouped into one
`PendingOperations` struct. With five kinds, `sweep_departed`'s parameter list
was about to cross clippy's `too_many_arguments` threshold; grouping is what that
pressure was pointing at, not a suppression of it.

### `wgpui-wgpu`

Both shaders are genuine ports of the legacy *mathematics* — `gaussian`, `erf`,
`blur_along_x` and the four-sample integration for shadows; the sine SDF with
its derivative-based distance correction for underlines — wrapped in
`quads.wgsl`'s vertex-pulling and indirection shape. What is replaced rather
than transcribed is listed at the top of each file: the arena lookup goes
through `visible[slot.base + instance]`, colour arrives as straight RGBA, and
there is no per-fragment clip.

`to_device_position_impl` keeps the *legacy spelling* rather than `quads.wgsl`'s.
The two are algebraically identical and bit-identical in exact IEEE arithmetic,
but they offer a compiler different fused multiply-add opportunities, and a
single-ULP difference in a clip coordinate can move which side of a pixel centre
an edge lands on. A shader that exists to be compared against a specific
expression keeps that expression.

**`issue_quads` became `issue_instanced` over a `&wgpu::RenderPipeline`.** This
is Phase 6.2's `issue_sprites` finding a second time: the shadow pass needed the
body unchanged — same bind group indices, same dynamic offsets, same four modes
— so the quad name came off rather than the body being copied. Verified in the
source, not in the message: one definition at `draw.rs:491`, three call sites at
`frame.rs:1209`, `:1218`, `:1230`. A duplicate would have hidden that the fifth
and sixth pipelines cost zero lines there.

### One hazard caught while writing, not by a failure

The first draft of `underlines.wgsl` packed `wavy` into a trailing `vec4<f32>`
and bit-cast it out. `Underline::encode` writes the boolean as the word `1`,
whose bit pattern read as `f32` is a **denormal** — and a GPU is free to flush
denormals to zero on load, which would draw every wavy underline straight on
exactly the hardware that does it, with no error anywhere and nothing to
attribute it to. The slot declares `wavy: u32` and a test asserts the shader
still spells it that way.

### One gap in an older test, closed

`CoverageItem::cullable` has existed since Phase 3 and **no test had ever set it
false**: every item in `compute_differential.rs` comes from
`quad_coverage_item`, which always sets it true. So the occlusion shader's
`FLAG_CULLABLE` branch was three phases old and unexercised.
`the_shader_honours_the_uncullable_flag_exactly_as_the_cpu_does` closes it,
comparing the same geometry with the flag both ways so the assertion is about the
flag rather than about the geometry. (That the flag then turns out not to matter
in a real frame is §4; the shader implementing it correctly is still worth
knowing.)

### Two assertions in `indirect_draw.rs`, corrected rather than relaxed

Both read `LAYERS` where they meant `LAYERS * NON_ATLAS_KINDS`, a multiplier that
was silently `1` while `Quad` was the only texture-free kind. It is now named and
derived (`DRAWN_KINDS - SPRITE_KINDS`), so the seventh kind will not need this
edit again.

---

## 6. Check, test and clippy status

All on the reference adapter, scoped to the touched crates. `cargo test
--workspace` was **not** run: it pulls in `gpui-ce`'s legacy suite, confirmed in
a prior phase to exceed 10 minutes without completing and unrelated to any 2.0
branch.

**Tests — `cargo test -p wgpui-core -p wgpui-wgpu -p wgpui-widgets`: 517 passed,
0 failed, 0 ignored.** (493 at Phase 6.2's landing.)

| Target | Result |
|---|---|
| `wgpui-core` unit | 332 passed |
| `wgpui-wgpu` unit | 72 passed |
| `compute_differential` | 6 passed |
| `glyph_atlas_upload` | 5 passed |
| `glyph_sprite_draw` | 6 passed |
| `image_sprite_draw` | 6 passed |
| `indirect_args_differential` | 8 passed |
| `indirect_draw` | 5 passed |
| `legacy_image_differential` | 3 passed |
| `legacy_shadow_differential` | **7 passed** |
| `legacy_underline_differential` | **6 passed** |
| `surface_registry_consumer` | 4 passed |
| `tile_visibility` | 4 passed |
| `window_present` | 1 passed |
| `wgpui-widgets` unit | 46 passed |
| `scroll_content_gate` | 6 passed |

**Check — clean.** `cargo check -p wgpui-core -p wgpui-wgpu -p wgpui-widgets
--all-targets`: no errors, no warnings.

**Clippy — clean, on a genuine cold build, zero suppressions.**
`cargo clean -p wgpui-core -p wgpui-wgpu -p wgpui-widgets` (removed 338 files,
4.2 GiB) then
`cargo clippy -p wgpui-core -p wgpui-wgpu -p wgpui-widgets --all-targets --
--deny warnings`: exit 0, and the log shows all five crates `Checking` and no
warning of any kind. Phase 6.2's standard is that "a clean exit code on an
unread log isn't evidence", so the log was read *and* the run was confirmed to
have done work: a second invocation finishes in 0.32s with no `Checking` lines
at all, which it would not if the first had been a cache hit.

`clippy.toml` was read first. Nothing in this phase touches its
`disallowed-methods` (`std::process::Command`, `serde_json::from_reader`) or its
`disallowed-types` (all commented out) — confirmed by grepping the diff, not by
recollection. The diff adds **zero** `#[allow]` or `#[expect]` attributes.

**`src/`, `docs/gpu-native-architecture.md`, `Cargo.toml` and `Cargo.lock` are
all untouched** — an empty `git diff --stat origin/2.0..HEAD` against those
paths. This phase therefore adds no dependency at all, not even a dev-dependency
edge (Phase 6.2 added five).

---

## 7. What is still open

### Named by §8's own table

- **`paths` / `backdrop_blur` — Phase 6.4.** Still two-line placeholders, still
  not scoped. Given §1.1, treat their "moved as-is" comments as false until
  read. `paths` additionally needs the tessellation vertex-buffer machinery no
  phase has built, which is a different shape from every pipeline so far —
  Phase 6.4 is *not* two more of these.
- **Phase 6.1's fate is still undecided.** Spike B killed it as originally
  scoped (~1000–1150× slower); the rescoped fused-dispatch follow-up spike has
  still not been run, and the phase has neither been executed nor dropped from
  the table. Unchanged by this phase.
- **The animation driver — Phase 6.5** — has not started.
  `crates/wgpui-widgets/src/animation.rs` and `window/animation.rs` are still
  3-line stubs. Re-verified by reading them, not carried over on faith.
- **§6.2's `estimated_size` half is still not closed.** No trait hook exists
  anywhere in 2.0; `wgpui-layout/src/containment.rs` is still a 3-line file.
  Unchanged by this phase, which touched neither.
- **`wgpui-devtools` extraction (Phase 7)** has not started.
- **Final cutover (Phase 8) and the legacy alias crate** have not started.
  `wgpui-core` is not the default; `src/` is still the frozen legacy backend.

### Opened or sharpened by this phase

- **Cross-kind occlusion does not exist, and a second kind now depends on that
  being fixed eventually.** Phase 5.6 recorded it for glyphs as a limit on what
  could be culled. §4 shows it also makes a *correctness* mechanism inert. When
  cross-kind occlusion lands, `Shadow`'s `uncullable` and `drawn_bounds` become
  load-bearing on the same day, and there is currently no test that would catch
  getting them wrong.
- **Example parity is unblocked for shadows and underlines, not achieved.** §8's
  `6.3` row names `shadow` (bench), `window_shadow`, and "any text with
  underline/strikethrough styling". The pipelines those need now exist and are
  byte-exact. **None of those examples has been ported to 2.0**, and text
  styling has a further gap: `wgpui-text` does not emit an `Underline` primitive
  for anything. `StyledText` has no underline or strikethrough path — the kind
  and the pipeline exist; nothing produces one from a style. That is a
  `wgpui-text` change, not a renderer one, and it is open.
- **Nothing emits a `Shadow` either.** `Emission::shadow()` exists and no widget
  calls it; `BoxShadow` (`src/style.rs:316`) has no 2.0 counterpart, and neither
  does the multi-shadow list a real `box_shadow` style carries. 2.0 can *draw* a
  shadow byte-exactly and no element *asks* for one.
- **Two legacy quirks are now reproduced in 2.0** (§3): zero-blur shadows draw
  nothing, and underline alpha is squared. Both are deliberate and both should be
  revisited at Phase 8.
- **Per-corner radii are not representable** for `Quad`, `PolySprite` or
  `Shadow`. Three kinds now share the uniform-radius convention, which makes it
  cheaper to keep and more expensive to change; the legacy shaders all accept
  four.
- **`SlotBasePlan`'s `for_*` constructors are now the only per-kind code left in
  `render/draw.rs`** — five near-identical functions extracting
  `(slot_layout, slot_stride)`. Not fixed here (it would be a refactor unrelated
  to the gate), noted because it is the thing that will grow with kind six.

### Carried forward, unchanged

Scale factor 1 only and caller-rounded glyph positions (Phase 5.6); nearest-
neighbour scaled images (Phase 6.2); GIFs decode but do not animate;
`PrimitiveStore::reflow`'s O(n²) bulk-build cost; GPU occlusion's 1.30× loss on
low-visibility scenes; the transcription-oracle limit on Phase 5.5's
differential; the 2×-scale sub-pixel aliasing quirk; input plumbing
(`keyboard.rs`, `dispatcher.rs`, `app_menu.rs`) still Phase 0 stubs; one machine
/ one driver / one backend for every number in every report so far; `gpui-ce`'s
legacy test binary still not confirmed to finish.

---

## 8. Honest read

Two pipelines, two primitive kinds, two byte-exact gates against the legacy
shaders themselves rather than against a model of them. The pattern the phase
table said was proven three times over was proven twice more and held: nothing
in the slab allocator, patch protocol, upload planner or slot table needed a
per-kind line, and `issue_instanced` serves three pipelines where Phase 4 wrote
one.

The phase table's "genuinely cheap" was right for underlines and half-right for
shadows. A shadow's fragment shader is real numerical work — an `erf`
approximation and a four-sample Gaussian integration — and it is the first
primitive in 2.0 that paints outside its own rectangle, which forced two
decisions the phrase does not cover.

The most useful thing this phase produced is not either gate. It is §4: those
two decisions were asserted to be load-bearing, in a commit message and three
doc comments, and then measured and found inert. 2.0's occlusion dispatches per
kind, so nothing can cull a shadow today whatever flag it carries. The
adjustments are still right and are still there; what changed is that the code
now says what is true about them, and a future phase enabling cross-kind
occlusion will find a note saying that the mechanism it is about to start
depending on has never once been exercised. Every phase in this project has
found at least one thing under actual verification. This one's was a claim the
phase had made about itself an hour earlier.

What this phase does not do is make shadows or underlines *usable*. No element
emits either primitive: `BoxShadow` has no 2.0 counterpart and `StyledText` has
no underline path. The renderer can draw both byte-exactly and nothing asks it
to. That is named here rather than folded into "shadows and underlines work".
