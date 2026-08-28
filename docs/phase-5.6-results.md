# Phase 5.6 Results — The Glyph Sprite Pipeline

**Branch**: `wgpui-2.0/phase-5.6-glyph-sprite-pipeline`, off `2.0` at
`cdba053d00` (Phases 0–5.5 complete).

**Not in §8's phase table.** Like Phase 5.5 before it, this closes a gap the
plan *disclosed* rather than implementing a pre-written row. §9's risk table
named it and §11 called it "the single most load-bearing open item". This
document does not edit `docs/gpu-native-architecture.md`; updating §8, §9 and
§11 is a separate act.

**Hardware**: every rendering result below ran on an NVIDIA GeForce RTX 4060
Laptop GPU, Vulkan, driver 561.03, `INDIRECT_FIRST_INSTANCE=true`,
`MULTI_DRAW_INDIRECT_COUNT=true`. Verified by `examples/adapter_probe` before
any result was claimed, per Phase 0's standard. One machine, one driver, one
backend — §11's action 2 still stands.

---

## 1. The gap, and what it actually was

§9's risk row, quoted in full because the phase is defined by it:

> Text is now shaped, positioned, reconciled, rasterized, and correctly
> present in GPU atlas texture memory (proven by an actual differential and a
> real page readback) — but nothing yet draws it: there is no sprite render
> pipeline and no sprite primitive kind consuming those atlas tiles.

Two halves. The second half — "no sprite primitive kind" — turned out to be
**already false**, and finding that out early is what kept this phase small.
`GlyphRun`/`Glyph` has been the sprite primitive kind since Phase 1: it carries
`position`, `atlas_origin`, `atlas_size`, a colour, and (since Phase 5) an
`AtlasTileId` naming page and slot. Phase 5.5 filled those tiles with real
texels. Nothing about the primitive kind needed adding, and **no byte of
`patch/primitive.rs` was changed** — the layout was already exactly what a
vertex-pulling sprite shader wants.

The first half was true and total. There was no pipeline.

### One thing the brief got wrong, worth stating

The task described `crates/wgpui-wgpu/src/render/shaders/mono_sprites.wgsl` as
"already exists, unused until now — read it for what it expects: an atlas
texture binding, glyph position/size/tile-coordinate instance data, text
colour."

It was two lines:

```wgsl
// Placeholder — moved as-is from src/platform/cross/shaders/mono_sprites.wgsl in a later phase.
// See docs/gpu-native-architecture.md §3.5.
```

Every file under `shaders/` except `quads.wgsl` and `surfaces.wgsl` is that
same placeholder. So the shader was written, not ported, and its design is
this phase's rather than the legacy backend's. That is a scope difference from
the framing, in the direction of more work, and it is the reason the shader
carries as much justification in its header as it does.

---

## 2. What shipped, and where

| File | +/− | What |
|---|---|---|
| `render/shaders/mono_sprites.wgsl` | +169/−2 | The shader, written from nothing. |
| `render/pipelines.rs` | +254/−19 | `MonoSpritePipeline`; `slot_base_bind_group` shared out of `QuadPipeline`. |
| `render/draw.rs` | +213/−12 | `SlotBasePlan` (was `QuadDrawPlan`), `GlyphDraw`, `issue_glyphs`, three `DrawStats` counters. |
| `render/frame.rs` | +273/−16 | The glyph arena, args, compute, plan, page bind groups, and issue step. |
| `render/atlas_upload.rs` | +25/−0 | `page_kind`, `pages_of_kind`. |
| `tests/glyph_sprite_draw.rs` | +692/−0 | Six tests. The gate. |
| `tests/indirect_draw.rs` | +24/−5 | Two Phase 4 assertions whose premise this phase removed. |
| `examples/phase4_draw_issuance_bench.rs` | +1/−0 | `FrameInput`'s new field. |

Total: **1,651 insertions, 54 deletions across 8 files.** Nothing under `src/`
was touched, and nothing in `wgpui-core` or `wgpui-text` was changed at all —
including `patch/primitive.rs`, whose layout this phase consumes unmodified.

---

## 3. The design, and the one decision that made it a phase

Structurally, `mono_sprites.wgsl` is `quads.wgsl` with a texture. Instance
addressing is identical and deliberately so: `visible[slot.base + instance]`,
the same `SlotBase` uniform under both `FirstInstance` encodings, the same
four-vertex triangle strip, `vertex.buffers: &[]`. A text layer is a slot like
any other and takes §5.3's fixed sequence unmodified.

**The texture is the whole of the difference, and it is not a small one.** A
glyph names its atlas tile as a packed `(page, slot)` word; the page decides
which `wgpu::Texture` holds its texels. A bind group cannot change inside a
draw call. So a slot whose glyphs span two pages cannot be one draw, and the
CPU cannot know which pages a slot's *surviving* glyphs reference without
reading back the very instance counts §5.3 exists to avoid learning.

Three ways out:

1. **Bind every page at once** as a binding array. Needs a device feature
   WebGPU does not guarantee. Rejected — it would make the default path
   depend on a capability the fallback story is specifically built to not
   need.
2. **Sort glyphs by page on the CPU.** Requires walking primitives per frame,
   which is the exact work Phase 4 removed.
3. **Draw the slot once per page, and let the shader drop the glyphs that are
   not on the bound one.** Chosen.

Under (3), `page.index` is a uniform bound alongside the texture; a glyph whose
tile names a different page collapses to a degenerate triangle strip, exactly
as an unused instance already did. Each glyph is rasterised into the
framebuffer exactly once, by the pass that bound its own page — so this is
correct, not merely harmless, and `glyphs_on_several_atlas_pages_all_draw_and_
none_draws_twice` is the test that says so.

The honest cost: this pipeline's CPU draw-issuing work is **O(layer slots ×
live monochrome pages)**, not O(layer slots) flat. That is reported rather than
buried — `DrawStats::atlas_pages_bound` carries the multiplier — and it is the
same shape `issue_composites` already has for the same reason (a bind group
cannot change inside a `multi_draw_indirect`). In the common case there is one
monochrome page, and the multiplier is 1.

### Bind groups

| Group | Contents | Same as `QuadPipeline`? |
|---|---|---|
| 0 | globals, `GlyphRun` arena (storage), indirection buffer (storage) | Yes, over a different arena |
| 1 | slot base, by dynamic offset | Yes — literally the same helper |
| 2 | page index (uniform), page texture | New |

`QuadDrawPlan` was never quad-specific: the slot base is
`wgpui_core::indirect`'s notion, not a shader's. It is now `SlotBasePlan` and
both pipelines build one, rather than two copies that could drift.

### `textureLoad`, not a sampler

The atlas already holds `SUBPIXEL_VARIANTS_X = 4` horizontal rasters of every
glyph — the legacy design's way of carrying sub-pixel positioning in the
*raster* rather than in the *sample*. A glyph quad is therefore meant to blit
one texel to one pixel, and a 1:1 blit needs no filtering and no normalised
coordinates. `textureLoad` at an integer address is what it needs. There is no
sampler in this pipeline at all, and the texture binding is declared
non-filterable.

This is also what makes §4's proof possible: a filtered sample would make
"identical" depend on interpolation, the same reason `quads.wgsl` chose hard
edges over a coverage ramp.

### The `GlyphSlot` alignment trap

`Glyph`'s colour starts at byte 28, which is not a `vec4<f32>` alignment. WGSL
would have aligned a `vec4<f32>` member to 32 and silently read the wrong bytes
for every field after it — including the atlas tile, which would have sent
every glyph to the wrong page. The shader spells the colour as four `f32`
scalars, which align to 4 and land where `GlyphRun::encode` put them. A unit
test in `pipelines.rs` asserts the shader still contains those four fields and
the sentinel constants, because nothing else can check a WGSL string.

---

## 4. How visual correctness was proven, and how strong that is

**Strong. This is a byte-exact comparison against the atlas's own texels, not
a smoke test.**

White text on black through this pipeline's straight-alpha `over` blend reduces
to an identity. The shader emits `rgb = 1` and `alpha = colour.a * coverage`.
The blend computes `src.rgb * srcAlpha + dst.rgb * (1 - srcAlpha)`, which with
`dst = 0` and `src.rgb = 1` is `srcAlpha`, which is `coverage`, which is the
atlas texel over 255. Written back to `Rgba8Unorm` it is the texel byte again.

So a rendered pixel is not *similar* to its atlas texel — it is the same byte,
and the tests assert equality rather than a threshold. This extends Phase 5.5's
"every glyph claiming a tile finds ink in it" exactly one level up, as the task
asked: **every glyph drawn on screen carries that tile's own texels at that
glyph's own screen position.**

`tests/glyph_sprite_draw.rs`, six tests, all passing:

| Test | What it establishes |
|---|---|
| `every_glyph_draws_its_own_tile_texels_at_its_own_position` | **The gate.** Real shaped text (`Hamburgefonstiv 0123`, 24px, IBM Plex Sans) through the real rasteriser and real atlas. 3,097 texels compared, 1,796 of them inked, every one byte-exact. |
| `text_draws_without_being_rounded_first` | The same line drawn exactly as `wgpui_text::patch::glyph_runs` produced it, nothing doctored: 19 of 19 inked glyphs put ink inside their own box. |
| `glyphs_on_several_atlas_pages_all_draw_and_none_draws_twice` | 64-texel pages force a genuine 2-page split. 2 pages bound, 3,920 texels byte-exact across both. A glyph drawn by the wrong page's pass would read another glyph's texels; one drawn twice would blend over itself and come out brighter. |
| `every_draw_mode_draws_the_same_text` | All four `DrawMode`s produce byte-identical framebuffers, Phase 4's discipline applied to the new pipeline. |
| `a_blank_glyph_and_a_missing_atlas_are_both_ordinary` | Whitespace holds its slot and paints nothing; a frame with no atlas renders, binds nothing, and paints nothing, rather than erroring. |
| `a_ramp_raster_lands_texel_for_texel_with_no_row_shift` | A synthetic 13×11 raster whose every texel is distinct. Real glyphs are smooth, so a shifted row still looks like a glyph; here a one-texel error is a byte mismatch. Also asserts the sprite occupies *exactly* its raster's extent — 143 painted pixels, no spill. |

### What the proof does not cover, stated plainly

- **Monochrome only.** Colour glyphs (`AtlasKind::Polychrome`) are never bound
  and never drawn. See §6.
- **Scale factor 1.** `Glyph::position` is in layer space and `atlas_size` is
  in device texels; at `scale_factor != 1` those disagree and the slot carries
  no scale to reconcile them. Every test runs at 1×. See §6.
- **The gate rounds positions.** A 1:1 blit is only texel-exact when the quad's
  corners are whole pixels, and `wgpui_text::patch::glyph_runs` does not floor
  the pen position the way the legacy `Window::paint_glyph` does. The gate
  rounds; `text_draws_without_being_rounded_first` then shows the undoctored
  path paints correctly to within the pixel that flooring would remove. The
  flooring belongs in `wgpui-text`, not here. See §6.
- **Nobody has looked at it.** Every claim here is a readback comparison. No
  human has seen this text on a screen in a window, because 2.0 has no window
  path yet — that is the cutover, not this phase.

---

## 5. What verification found

Three things the tests found rather than confirmed. All three are recorded
because each was a plausible wrong belief.

**1. The target's alpha is not the coverage.** The first version of the gate
asserted `pixel.a == coverage` and failed at 255 vs 196. The pass clears to
opaque black and the alpha blend is `One`/`OneMinusSrcAlpha`, so
`a = srcAlpha + 1·(1 − srcAlpha) = 1` whatever the coverage was. A framebuffer
that keeps its opacity under text is the correct outcome. The assertion now
says `a == 255` with the derivation written next to it, because "alpha equals
coverage" is the plausible wrong expectation and someone will have it again.

**2. Adjacent glyph rasters genuinely overlap.** The multi-page test failed at
one pixel — glyph 16 spans x 187–199 and glyph 17 starts at x 198. A raster is
wider than its advance wherever a letter leans, so the framebuffer at a shared
column holds the *blend* of two tiles and no single tile's texels describe it.
The comparison now runs only over pixels exactly one glyph claims (24 of 3,944
were shared in that test, 0 of 3,097 in the gate). **That is a restriction on
what can be checked, not a tolerance** — on the pixels it does check, equality
is still exact. It also means painter order within a text run is load-bearing,
which is a reason the glyph path goes through the real ordering pass rather
than an identity permutation.

**3. `SlabBuffer::upload` never filtered `UploadRange` by kind.** With one
arena that was invisible; with two it would apply a glyph's byte span to the
quad buffer and overwrite an unrelated primitive. `frame.rs` now filters per
kind before applying. This is a latent Phase 4 bug that this phase's second
arena would have activated — found by reading the code before wiring it, not
by a failing test, because no existing test passes a non-empty `uploads` list.

### Two Phase 4 assertions changed, and exactly what changed about them

`tests/indirect_draw.rs` had two assertions that encoded *"nothing draws the
`GlyphRun` half of the slot table"* — one of them saying so in its own failure
message, citing "Phase 4 built one instanced pipeline". That is the premise
this phase removed, not the gate.

- `slots_visited == LAYERS` → `LAYERS * PrimitiveKind::COUNT`. Both halves of
  the table are now walked.
- `slots_skipped + draw_calls_issued == slots_visited` gained a third term,
  `glyph_slots_unavailable`.

**Gate 1 itself is untouched and still passes**: at 290 and 18,026 resident
primitives, `draw_calls_issued` is 6 and 6, `bind_group_binds` is 7 and 7,
`slots_visited` is 12 and 12, `instances_known_to_cpu` is `None`,
`readback_words` is 0. Draw-issuing work still does not grow with the primitive
count.

`glyph_slots_unavailable` is new and mirrors the existing
`composite_entries_unavailable`: a glyph slot with no atlas page to bind cannot
be issued at all, because there is no such thing as a draw call without a bound
texture. Calling that "skipped" would claim the CPU decided something it did
not; calling it "drawn" would be false. It is a third outcome, and counting it
is what keeps every slot the fixed sequence named accounted for.

---

## 6. What is still open

### Explicitly deferred, by name

- **`poly_sprites` — colour glyphs and images.** Not attempted, per the task's
  scoping. A polychrome page is `Rgba8Unorm` and this shader reads one coverage
  channel; binding a colour page here would sample an emoji's red channel as if
  it were coverage, so `atlas_page_bind_groups` filters colour pages out
  rather than drawing them wrong. This needs its own shader, its own bind
  group layout, and — unlike glyphs — **a primitive kind that does not exist
  yet**: nothing in `patch/primitive.rs` carries an image. That makes it
  larger than this phase was, not smaller, and it is the next honest
  candidate for a phase of its own.
- **`shadows`, `underlines`.** Both are `QuadPipeline`'s shape exactly — same
  bind group layout, same draw call, a different shader. Cheap, and blocked on
  nothing but a primitive kind to carry them.
- **`paths`.** Needs a real vertex buffer and tessellation machinery no phase
  has built. `backdrop_blur` needs a second pass over the framebuffer. Neither
  is close.

### Disclosed by this phase

- **Sub-pixel positions are not floored.** `wgpui_text::patch::glyph_runs`
  keeps the fractional pen position, while the atlas already carries the
  fraction as one of four sub-pixel variants. The legacy `Window::paint_glyph`
  floors; 2.0 does not yet. Until it does, a glyph at a fractional position
  blits up to a texel off. One line in `wgpui-text`, and a `wgpui-text` test
  is the right place for it — not fixed here because this phase deliberately
  did not touch the conversion.
- **No scale factor in the glyph slot.** `position` is layer space,
  `atlas_size` is device texels. At 1× they agree; at 2× the sprite quad would
  be twice the size it should be. Fixing it is either a scale field on
  `GlyphRun` (48 bytes has no room; the run's `color` is already replicated per
  glyph, so a run-level uniform is the more likely shape) or positions carried
  in device space. Naming it rather than guessing.
- **Cross-kind occlusion is not expressible.** Glyphs go through the real
  ordering and occlusion passes — a glyph is a `CoverageItem::cullee` and never
  an occluder, which is correct, since a coverage mask is not an opaque
  rectangle. But the occlusion dispatch is per kind, so a glyph behind an
  opaque quad is not culled by it: the glyph dispatch sees only glyphs, all of
  them cullees, and nothing to cull them against. Poison regions still apply.
  This is a structural property of the per-kind dispatch that Phase 3 and 4
  established, not something this phase introduced, and it is a correctness-
  preserving inefficiency (text draws when it could have been skipped) rather
  than a wrong picture.
- **Page bind groups are rebuilt every frame.** A bind group names a texture
  view, and an atlas page destroyed between frames leaves a cached one pointing
  at a dead texture. The 16-byte uniform behind it *is* cached, since its value
  is the page index. Cost is O(live monochrome pages) bind-group creations per
  frame, typically one. Cheap, but it is per-frame CPU work and it is named
  here rather than left to be discovered.

### Carried forward from earlier phases, unchanged

- **§6.2's `estimated_size` half**: neither `Img` nor `StyledText` has one.
  Untouched.
- **Phase 6.1's fate**: still undecided. Run the fused-dispatch spike or drop
  the row; this phase says nothing about it.
- **Devtools extraction (§3.6)**: not started.
- **Final cutover (§8's last phase)**: not started, and nothing here brings it
  closer than one pipeline's worth. There is still no window path in 2.0.
- **Non-NVIDIA/non-Windows validation**: §11's action 2. Every number in this
  document is one machine.
- `PrimitiveStore::reflow`'s O(n²) bulk build; GPU occlusion's 1.30× loss on
  low-visibility scenes; Phase 5.5's transcription-oracle limit; the
  deliberately-preserved 2×-scale sub-pixel aliasing. All still open, none
  touched.

---

## 7. Check, test, and clippy status

```
cargo check -p wgpui-wgpu --all-targets            clean
cargo test  -p wgpui-core -p wgpui-wgpu            408 passed, 0 failed
cargo clippy -p wgpui-core -p wgpui-wgpu \
    --all-targets -- --deny warnings               clean
```

The clippy run was a genuine cold one: `cargo clean -p wgpui-core -p
wgpui-wgpu` first, then both crates recompiled from scratch. `clippy.toml`'s
conventions were read before writing (`disallowed-methods`,
`disallowed-types`, `avoid-breaking-exported-api = false` — which is what makes
the `QuadDrawPlan` → `SlotBasePlan` rename an ordinary change rather than a
break).

Test breakdown across the two crates:

| Suite | Tests |
|---|---|
| `wgpui-core` lib | 320 |
| `wgpui-wgpu` lib | 52 |
| `compute_differential` | 5 |
| `glyph_atlas_upload` | 4 |
| **`glyph_sprite_draw`** | **6** |
| `indirect_args_differential` | 8 |
| `indirect_draw` | 5 |
| `surface_registry_consumer` | 4 |
| `tile_visibility` | 4 |

`cargo test --workspace` was **not** run, per the task's instruction: it
includes `gpui-ce`'s legacy suite, confirmed 10+ minutes without completing and
unrelated to any 2.0 branch. That binary still has not been confirmed to
finish — §11's action 4, still carried.

---

## 8. Honest read

**This phase did what it said and it is smaller than it sounds.** One pipeline,
one primitive kind, one shader, one page loop. The primitive kind was already
there and did not move a byte. The indirect-draw machinery was already there
and took the new kind without modification — which is the strongest single
piece of evidence that Phases 1–4's architecture is what it claimed to be, and
it is worth stating that this was *tested*, not assumed: the glyph path goes
through the same `IndirectArgsPass`, the same `SlotBasePlan`, the same four
`DrawMode`s, and the same `DrawStats` counters, with no per-kind branch
anywhere in `render/draw.rs`'s issuing logic except the page loop.

**What genuinely diverged from the framing** was the shader: it did not exist
to be read, so it was designed here. That is more work than the task expected
and it is why the shader header carries an argument rather than a description.

**What this does not do** is make WGPUI 2.0 draw a window. §11's sentence —
"the first point across seven phases where 'every gate passed' and 'something
appears on a screen' have genuinely diverged" — is now narrower but not gone.
Text renders, correctly, byte for byte, into an offscreen texture in a test. A
window, an event loop, and a swapchain are still the cutover phase's work, and
nothing here shortens that. The honest summary is: **the divergence between
"proven" and "visible" is now one phase's worth of window plumbing plus
`poly_sprites`, rather than the whole of the sprite path.**

The proof is the part worth defending. It is not "the draw call did not error".
It is 7,000+ texels of real shaped text compared byte for byte against the
atlas pages they came out of, across one and two atlas pages, across four draw
modes, with a synthetic distinct-texel raster proving the addressing that
smooth glyphs would hide. Two of the three things verification found were
wrong beliefs of mine, not bugs in the code — which is the outcome a real
proof is supposed to produce when the code happens to be right.
