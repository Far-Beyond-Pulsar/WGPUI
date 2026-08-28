# Phase 6.2 Results — Image Loading, Decoding, and Rendering

Branch: `wgpui-2.0/phase-6.2-images`. Not merged, no PR opened.

`docs/gpu-native-architecture.md` is not edited by this report; §8's `6.2` row
and §6.2's invariant text are quoted here, not amended there.

## 0. The gate, and whether it was met

§8's row states it:

> A real image file loads, decodes, uploads, and renders byte-exact against the
> legacy renderer's output for the same source — same differential-proof
> discipline as Phase 5.6, one primitive kind over, now genuinely built from zero
> rather than extending something partial.

**Met, byte-exact, with no tolerance consumed.** On the reference adapter
(NVIDIA GeForce RTX 4060 Laptop GPU, Vulkan, DiscreteGpu, driver `561.03`,
`INDIRECT_FIRST_INSTANCE=true MULTI_DRAW_INDIRECT_COUNT=true`):

```
phase_6_2_png_gate: 262144 of 262144 pixels byte-exact, 0 within one unit;
  source texels: 161368 opaque, 8540 translucent, 92236 transparent
phase_6_2_gif_gate: 250000 of 250000 pixels byte-exact, 0 within one unit;
  source texels: 250000 opaque, 0 translucent, 0 transparent;
  25 frames decoded, frame 0 drawn
```

The precise strength of that claim is §3. It is stronger than "byte-exact on an
opaque asset" and weaker than "byte-exact on every format at every scale", and
both halves of that sentence matter.

---

## 1. How this phase was written, and what this continuation inherited

Four layer commits, then one commit explicitly marked UNVERIFIED:

| Commit | Layer |
|---|---|
| `788020e138` | 1 — `PolySprite` primitive kind |
| `d6b08e5eab` | 2 — polychrome atlas-tile producer |
| `4655a2f550` | 3 — `PolySpritePipeline`, the fourth pipeline |
| `5c9057d85e` | 4 — decode, `ImageCache`/`ImageEngine`, `Img` drawing real pixels |
| `746b61a1dd` | the differential test, committed mid-fix, **UNVERIFIED** |

The prior agent hit an API session limit immediately after diagnosing that the
gate's failure was in the test's own oracle rather than in the renderer, and
committed the partial fix with a note saying so rather than leaving it
uncommitted. This continuation verified that diagnosis independently (§2),
finished and tightened the test (§3), re-read all five diffs against their
commit messages (§5), and ran the checks (§6).

---

## 2. The oracle bug: diagnosis confirmed, from the blend state rather than from the note

The prior agent's one-line diagnosis was: *the comparison target clears to
opaque black, so the alpha-`over` blend always yields 255, and an oracle that
expected the source texel's alpha to survive produced false mismatches on every
transparent pixel.*

That was verified from the code rather than accepted, because a self-diagnosis
of a self-inflicted bug is exactly the claim least worth taking on trust.

**The blend state.** `render/pipelines.rs:67` — `ALPHA_OVER`:

```rust
color: src_factor: SrcAlpha,  dst_factor: OneMinusSrcAlpha, operation: Add
alpha: src_factor: One,       dst_factor: OneMinusSrcAlpha, operation: Add
```

**The fragment output.** `shaders/poly_sprites.wgsl:197` returns
`vec4<f32>(color.rgb, alpha)` — *straight* alpha, not premultiplied, with
`alpha = color.a * sprite.opacity * coverage`.

**The destination.** `OffscreenTarget::target()` (`render/frame.rs:281`) clears
to `wgpu::Color::BLACK`, which is `{0, 0, 0, a: 1.0}` — opaque.

Working the two channels separately, with `a` the source alpha:

- Colour: `src.rgb * a + dst.rgb * (1 - a)` = `src.rgb * a + 0` = `texel.rgb * a`.
- Alpha: `src.a * 1 + dst.a * (1 - a)` = `a + 1 * (1 - a)` = **1, for every
  value of `a`**.

So the diagnosis holds exactly, and for the reason given. The destination alpha
is 1 and the destination's contribution to alpha is `One`-weighted, so there is
no transparency in the target for a transparent source to preserve. An oracle
predicting `texel.a` in the output alpha channel is wrong on every pixel whose
alpha is not already 255 — which on the PNG under test is 100,776 of 262,144
pixels, and reads as a catastrophic renderer failure while being a one-line
arithmetic error in the test.

Two further conditions had to hold for the corrected oracle to be right, and
both were checked rather than assumed:

- **`TARGET_FORMAT` is `Rgba8Unorm`, not `Rgba8UnormSrgb`** (`pipelines.rs:63`).
  Had it been sRGB, blending would happen in linear space after a transfer
  function and none of the byte arithmetic above would survive. Phase 5.6's
  byte-exact text proof depends on the same fact.
- **The discard path reproduces, rather than special-cases, the fully
  transparent texel.** The shader discards at `alpha <= 0.0`, leaving the clear
  colour; the oracle's `expected_over_black([r, g, b, 0])` evaluates to
  `[0, 0, 0, 255]`, which is that clear colour. The two agree without the oracle
  needing an `if`.
- **Coverage is exactly 1.0 at every pixel centre**, so it does not perturb the
  comparison. With `corner_radius == 0`, `rounded_rect_distance` reduces to
  `max(|x - w/2| - w/2, |y - h/2| - h/2)`, which at a pixel centre in
  `[0.5, size - 0.5]` is at most `-0.5`, so `saturate(0.5 - d) == 1.0` — at the
  outermost pixel column too, where it evaluates to exactly `-0.5` and not less.

**The state of the fix as committed.** Contrary to what the UNVERIFIED note
implied, the corrected `expected_over_black` in `746b61a1dd` was already
complete and correct; the commit was interrupted after the fix, not partway
through it. Running it was all that was needed to establish that, and it had
never been run. This is recorded because "the fix was already right" is a
different finding from "the fix needed finishing", and the note asserted the
second.

### What this continuation changed

The fix was right; the *assertions built on it* were weak in a way that would
have let a much less impressive result be reported in the same words.

1. **The alpha classes of the source are now counted and printed** — opaque,
   translucent, fully transparent. "Byte-exact" over an asset that happens to be
   entirely opaque is a substantially weaker sentence than it sounds, because
   the alpha multiply the whole comparison is about never runs. Without this
   count a report cannot tell the two cases apart. It turned out to matter: the
   GIF asset is 100% opaque (§3).
2. **The PNG gate now asserts its asset contains translucent texels.** If a
   future asset swap makes it opaque, the gate says so instead of passing
   vacuously.
3. **Both gates now assert `exact == total`.** The PNG's previous bar was
   `exact > total / 2` and the GIF's was `exact > 0` — the latter proves only
   that something matched. Since every pixel of both assets does in fact agree
   byte-for-byte, the bar is set where the behaviour actually is.
4. The ±1 tolerance is retained as a *diagnostic*, not as the pass condition
   (§3).

---

## 3. What the differential actually proves

### The two links, and why chaining them matters

`crates/wgpui-wgpu/tests/legacy_image_differential.rs` compares against a side
that shares no code with the thing under test, twice:

1. **Decode.** Our `DecodedFrame` is compared against `image::guess_format` then
   `load_from_memory_with_format(..).into_rgba8()` — which is
   `src/elements/img.rs`'s `ImageAssetLoader::load`, *called* rather than
   transcribed. Unlike Phase 5.5's rasteriser oracle this is not a transcription
   risk: that half of the legacy path is the `image` crate's own public API at
   the version the root crate pins.
2. **Draw.** Every pixel of the rendered framebuffer is compared against what
   the blend derived in §2 produces from the oracle's own bytes.

Chaining them is what makes "the pixels on screen are the ones the legacy
decoder produced" one checked claim rather than two half-claims that could both
pass while disagreeing with each other in the middle.

### Nothing in the path is stubbed

The gate drives a real `Img` through `Reconciler` → `LayoutTree` → `Emitter` →
`apply` → `AtlasTextures::sync` → `FrameRenderer` → framebuffer readback, and
asserts `report.skipped == 0` (every queued rectangle reached a texture),
`sprites_resident == 1`, and `atlas_pages(Polychrome) == 1`. The source files
are read off disk from `examples/legacy/image/` rather than embedded, because
"a real image file" is the gate's own wording and a byte array pasted into a
test is neither real nor a file.

### Byte-exact, and the tolerance that was designed in and then not needed

The honest complication that glyph rendering did not have: a translucent texel
composites to `round(rgb * a / 255)`, which the GPU computes in `f32` and the
oracle computes in `f64`. Those are permitted to disagree by one unit at a
rounding boundary. Phase 5.6's pure-coverage case had no such multiply.

So the test computes a ±1 classification — **and does not pass on it.**
Measured, every one of the 512,144 compared pixels across both assets agreed
byte-for-byte, translucent ones included, and both tests assert exactly that.
The ±1 machinery survives only so that a one-unit divergence dies with a message
naming the byte-exact bar, rather than as a raw mismatch, and can be read and
judged instead of silently absorbed.

**Stated plainly: this is byte-exact, not byte-exact-within-a-tolerance. No
tolerance was consumed. The tolerance exists, is documented, and is currently
dead code on this hardware.**

### Where it is strong

- **The translucent blend is genuinely exercised.** `app-icon.png` carries
  **8,540 translucent texels** (0 < a < 255) alongside 161,368 opaque and 92,236
  fully transparent. Every one of the 8,540 — the population where the `f32`/
  `f64` multiply could have diverged, and the population the *old* broken oracle
  was wrong about — matched exactly. This is the part of the result that is
  actually interesting.
- **Two different legacy decode arms.** A GIF takes
  `GifDecoder::into_frames()`, not `load_from_memory_with_format`. A differential
  that only ever exercises one arm proves one arm.
- **The gate has been watched failing.** `the_comparison_actually_detects_a_wrong_pixel`
  corrupts one opaque texel of the oracle by +64 and confirms the comparison
  rejects it. Observed rejecting at pixel (234, 50). A gate nobody has seen fail
  is a gate nobody knows works.

### Where it is narrow, stated rather than left to be found

- **The GIF asset is 100% opaque** — 250,000 opaque texels, zero translucent,
  zero transparent. So the GIF arm proves the *decode arm* and the 1:1 blit; it
  contributes nothing to the blend proof. The blend proof rests entirely on the
  PNG's 8,540 translucent texels. This is why the alpha classes are printed.
- **Natural size only.** Both gates draw at `ObjectFit::Fill` at the decoded
  size, opacity 1.0, corner radius 0. That is deliberate — it is the condition
  under which the comparison is an equality rather than a tolerance — but it
  means *scaling, rounding, grayscale and opacity are not covered by the
  byte-exact claim.* They are covered by `tests/image_sprite_draw.rs`'s own
  assertions, which are not differential against legacy.
- **A scaled image is nearest-neighbour here and interpolated in legacy.** The
  shader uses `textureLoad`, not `textureSample`, for the reason Phase 5.6
  recorded: an integer-address load is exactly comparable against the CPU-side
  page bytes. So a downscaled photograph is visibly harsher in 2.0 than in
  legacy. `a_scaled_sprite_samples_the_nearest_texel` pins this down as a
  behaviour rather than leaving it as a surprise. It is a self-contained change
  — a sampler, a bind-group entry, normalised coordinates — held back only
  because making it now would have cost this phase the exactness its gate is
  built on. **This is a real, open fidelity gap.**
- **Two formats.** PNG and GIF. JPEG, WebP, BMP, ICO and TIFF decode through the
  same `into_rgba8()` arm and are not separately differentially tested.
- **One machine, one driver, one backend.** Windows 11 / Vulkan / NVIDIA, as
  every prior phase.

---

## 4. What was built, per layer

### Layer 1 — `PolySprite`, a third primitive kind (`788020e138`)

`patch/primitive.rs`'s module doc has claimed since Phase 1 that adding a kind
is "implement `Primitive`, add a `PrimitiveKind` variant, add one
`PrimitiveStore` field". Nothing had ever tested that. **The claim held**: the
slab allocator, the patch protocol, the upload planner and the indirect-draw
slot table needed no per-kind line — only the three `match` arms the compiler
pointed at.

The payload is `Quad`'s shape with `Glyph`'s atlas reference: 48 bytes exactly
filled, no padding, every field asserted at its own byte offset, because the
only reader is WGSL where nothing checks a layout.

`Scene::layers_referencing` now scans sprites as well as glyph runs —
**verified in the source**, not just in the message (`scene/atlas.rs:452`,
which walks both `glyph_runs` and `poly_sprites`). This is not bookkeeping: a
sprite left pointing at a freed rectangle draws whatever was allocated over it,
which is R-N §4.3's hazard exactly.

Disclosed by that commit rather than hidden: `tests/indirect_draw.rs` failed on
this change, correctly, and its constant said so by name until layer 3 built
the third pipeline.

### Layer 2 — the polychrome tile producer (`d6b08e5eab`)

Phase 6's report recorded honestly that `AtlasKind::Polychrome` existed as an
enum variant with a texel width and **nothing produced a polychrome tile**.
This is the producer, and the claim that the atlas was already generic held —
no page management, bin packing or upload code changed.

What was missing was a *name*. `GlyphAtlas`'s maps were keyed by
`GlyphRasterKey`, so there was no way to look up a tile that is not a glyph.
`AtlasKey { Glyph, Image }` is that name, and it is the line the legacy atlas
draws too (`src/platform.rs`'s `AtlasKey::Image`). The two key spaces share one
page numbering deliberately: a page index must identify a page globally or an
`AtlasEviction::Page` would be ambiguous.

`ImageTile` is `GlyphTile` minus the bearing, because that field would be a lie
for an image — a glyph's ink sits at an offset from its pen; an image has no pen
— and a field that is always zero invites a caller to add it to a position and
get the right answer for the wrong reason.

`RasterizedImage` documents straight alpha *at the type*, which is the field the
SVG path gets wrong downstream (§4, layer 4).

### Layer 3 — `PolySpritePipeline`, and one function where there were two (`4655a2f550`)

`poly_sprites.wgsl` stops being a two-line placeholder. It is `mono_sprites`
over a colour page, with three real differences, each a decision:

1. **Quad size and tile size are two numbers, not one.** A glyph blits its tile
   one texel to one pixel. An image does not — layout decides how big the
   picture is, the decode decides how big its bitmap is, and they agree only at
   the natural size. The fragment shader maps the quad's local coordinate
   through the ratio, guarded against a zero-sized rectangle (`PolySprite::ZERO`
   is representable and would otherwise divide by zero) and clamped against
   reading a neighbouring image's texels at the far edge.
2. **The corner radius is the legacy `quad_sdf`**, transcribed with its
   `saturate(0.5 - distance)` edge term, so a rounded avatar's antialiased rim
   is the legacy rim rather than an approximation of it.
3. **Grayscale uses the legacy Rec. 709 weights.**

The bigger finding is in `render/draw.rs`. `issue_glyphs` needed *nothing*
changed to serve a second sprite kind — same bind group indices, same page loop,
same four draw modes — so it lost the glyph name and became `issue_sprites` over
a `&wgpu::RenderPipeline` rather than being copied. **Verified in the source**:
one `issue_sprites` at `draw.rs:570`, called twice from `frame.rs:951` and
`frame.rs:967`. A duplicate would have hidden the result that Phase 4's "nothing
here is written per kind" claim survived its first real test.

`DrawStats::glyph_draws_issued`/`glyph_slots_unavailable` accordingly became
`sprite_draws_issued`/`sprite_slots_unavailable` and now count both passes.

### Layer 4 — decode, and `Img` drawing real pixels (`5c9057d85e`)

`image_cache.rs` and `svg.rs` stop being two-line stubs. The decode is
`src/elements/img.rs`'s dispatch ported expression for expression —
`guess_format`, the GIF and animated-WebP frame paths, `into_rgba8()` for
everything else — plus `src/svg_renderer.rs`'s resvg path.

**No new package enters the lockfile — verified, not asserted.** The `Cargo.lock`
diff across the whole phase adds *zero* `name =` entries; it adds only five
dependency edges (`image` to `wgpui-wgpu`'s dev-deps, `wgpui-widgets` to the
same, and `image`/`resvg`/`usvg` to `wgpui-widgets`). `image`, `resvg` and
`usvg` were already in the graph at the versions the root crate pins, which is
what lets the two decoders be compared byte for byte at all.

The `wgpui-wgpu` → `wgpui-widgets` edge is a **dev**-dependency deliberately:
§3.5 makes `wgpui-wgpu` "the only crate touching a live device" and §3.4 puts
elements the other side of that line, so an ordinary edge would invert the
split. A test binary is not the library, and the graph stays acyclic either way.

Two deliberate divergences from legacy, both asserted rather than described:

1. **SVG output is un-premultiplied at the decode.** resvg hands back a
   premultiplied pixmap; legacy uploads it as-is and leaves the correction to a
   shader flag it sets only when the *surface's* composite alpha mode is
   `PreMultiplied` — a property of the window, not of the image — so on an
   Opaque surface a translucent SVG comes out darker than it should in legacy.
   2.0 corrects at the decode, so it composites the same way on every surface.
   **This is a deliberate deviation from byte-parity with legacy, in 2.0's
   favour.**
2. **`ObjectFit::fit` guards a zero-area image or box.** The legacy expression
   divides by zero there and produces NaN bounds.

`Img` now emits a real `PolySprite` instead of a white placeholder quad, holds a
`SharedImageEngine` the way `StyledText` holds a text engine, and — the concrete
payoff of having real dimensions — an *unsized* image asks layout for its own
natural size instead of vanishing into a zero-sized box.

Verification on the way through found a real latent trap: `ImageStyle` derived
`Default`, which made `opacity` **0.0**, so any caller writing
`..ImageStyle::default()` would have got an invisible image. Harmless until this
phase because nothing read the field; now written out by hand with that reason
recorded at the impl.

`svg.rs` is thin on purpose, and that is the finding: the only thing an SVG has
that a bitmap does not is the absence of a natural pixel size, so `load` is the
whole loading half and everything downstream is `Img`'s. **The tinted alpha-mask
`svg()` path — monochrome atlas, different pipeline — is genuinely separate work
and is open, not done.**

---

## 5. Reading the commits against what they claim

Every phase's wrap-up has found at least one thing a commit message asserted
that the diff did not support. This one is in `5c9057d85e`, and it is about
`wgpui-widgets/tests/scroll_content_gate.rs`.

**The claim:** *"Phase 5's gate still holds, and holds harder: its forty avatars
are now real decoded frames going through the whole path, not element-shaped
placeholders."*

**What the diff does:** the forty avatars are `DecodedFrame`s **constructed by
hand** — `texels: vec![index as u8; 40 * 40 * 4]` — held in a real `ImageCache`
and drawn through a real `ImageEngine` and a real `Img`, but the tile source is
`GateImageTiles`, a **substitute** atlas that hands out one synthetic tile per
key.

So the claim overstates on two counts:

- "real decoded frames" — nothing decodes a file there; the frames are synthetic
  buffers. What became real is the *cache → engine → `Img` → `PolySprite`* path.
- "going through the whole path" — the atlas is a stub.

**The substitution itself is correct and correctly disclosed.** That gate
measures reconciliation cost, not pixels, and the test's own new doc comment
says exactly that: *"the gate measures reconciliation, so the atlas is a
substitute here and the real one is exercised in `wgpui-wgpu`'s own tests."* The
code is honest; the commit message is not. And the gate *did* genuinely get
stronger — before this phase `Img::new(content.avatar)` had no engine at all and
emitted a placeholder quad. It just did not get as much stronger as the message
says.

Nothing else diverged. The three load-bearing claims spot-checked in the source
rather than read in prose — `layers_referencing` scanning `poly_sprites`, one
`issue_sprites` serving two pipelines, and zero new lockfile packages — all
held.

---

## 6. Check, test and clippy status

All on the reference adapter, scoped to the touched crates. `cargo test
--workspace` was **not** run: it pulls in `gpui-ce`'s legacy suite, confirmed
in a prior phase to exceed 10 minutes without completing and unrelated to any
2.0 branch.

**Tests — `cargo test -p wgpui-core -p wgpui-wgpu -p wgpui-widgets`: 493 passed,
0 failed, 0 ignored.**

| Target | Result |
|---|---|
| `wgpui-core` unit | 325 passed |
| `wgpui-wgpu` unit | 69 passed |
| `compute_differential` | 5 passed |
| `glyph_atlas_upload` | 5 passed |
| `glyph_sprite_draw` | 6 passed |
| `image_sprite_draw` | 6 passed |
| `indirect_args_differential` | 8 passed |
| `indirect_draw` | 5 passed |
| `legacy_image_differential` | **3 passed** |
| `surface_registry_consumer` | 4 passed |
| `tile_visibility` | 4 passed |
| `window_present` | 1 passed |
| `wgpui-widgets` unit | 46 passed |
| `scroll_content_gate` | 6 passed |

**Clippy — clean, on a genuine cold build.** `cargo clean -p wgpui-core -p
wgpui-wgpu -p wgpui-widgets` (removed 168 files, 3.4 GiB) then `cargo clippy -p
wgpui-core -p wgpui-wgpu -p wgpui-widgets --all-targets -- --deny warnings`:
exit 0, no warnings. `wgpui-core` was then cleaned and linted a second time on
its own, because the first run's captured output had been truncated and its
`Checking wgpui-core` line was not visible — a clean exit code on an unread log
is not evidence. It lints clean cold.

`clippy.toml` was read first; nothing in this phase touches its
`disallowed-methods` (`std::process::Command`, `serde_json::from_reader`) or its
`disallowed-types` (all commented out).

---

## 7. Two questions §6.2 asked, answered plainly

### 7.1 Animated GIF support: decoded, not animated

**Every frame decodes, with its delay.** `decode()`'s GIF arm walks
`GifDecoder::into_frames()` and produces a `DecodedFrame` per frame, each
carrying its `delay: Duration` — because the decoder already produces both and
dropping them would mean decoding again to recover them. The gate's asset yields
25 frames. `DecodedImage::is_animated()` is real. `Img::frame_index` is a real
builder parameter and a real `diff_key` field, so changing it correctly
invalidates `DISPLAY`.

**Nothing advances that index over time.** 2.0 has no animation driver at all:
`crates/wgpui-widgets/src/animation.rs` is a **3-line stub**, and so is
`window/animation.rs`. Verified by reading them, not inferred.

So the scoping decision, stated plainly and not expanded: **an animated source
renders the frame it is asked for, and a caller that ticks the index itself gets
animation. "GIF decoding works" is true. "GIFs animate" is false.** No timer,
no frame scheduler, no `on_frame`. The `gif_viewer` example remains blocked on
an animation driver, not on decode.

This is static-first-frame-*rendering* with full-animation *decoding* — which is
a slightly better position than "static-only", and worth naming precisely
because the two are easy to conflate in either direction.

### 7.2 `estimated_size`: not closed, and could not have been

§6.2 records the invariant as `diff_key`-complete but not invariant-complete:
Phase 5 gave `Img` and `StyledText` real `diff_key` implementations; neither got
`estimated_size`.

**This phase did not close it, and the gap is larger than "`Img` is missing a
method".** `estimated_size` has **no implementation site anywhere in 2.0**: a
grep across `crates/` finds exactly two occurrences of the identifier, both in
prose — a doc comment in `img.rs` explaining why `layout_size()` is *not* it,
and line 1 of `crates/wgpui-layout/src/containment.rs`, which is a **3-line
file** consisting of a module doc and `#![allow(dead_code)]`. There is no trait
hook in `wgpui-core` for an element to implement. So `Img` could not have
implemented `estimated_size` in this phase even incidentally; the containment
machinery it would feed does not exist yet.

What *did* fall out naturally is a neighbouring but distinct thing, and the
distinction is the point: `Img::layout_size()` now returns the decoded natural
size when no explicit size was requested, so an unsized image no longer collapses
to a zero-sized box. That is a real improvement in what an `Img` asks layout for
— but it is a *known* size for a *decoded* image, not an *estimate* for an
*undecoded* one, which is precisely what `estimated_size` exists to provide.
An undecoded source still asks for zero. `img.rs:478-495` says so at the method.

**§6.2 remains half-discharged, unchanged by this phase.**

---

## 8. What is still open

### Named by §8's own table

- **`shadows` / `underlines` — Phase 6.3.** Still two-line placeholders. Flagged
  as `QuadPipeline`-shaped and cheap; nothing in this phase touched them.
- **`paths` / `backdrop_blur` — Phase 6.4.** Still two-line placeholders, still
  not scoped.
- **Phase 6.1's fate is still undecided.** Spike B killed it as originally
  scoped (~1000–1150× slower); the rescoped fused-dispatch follow-up spike has
  still not been run, and the phase has neither been executed nor dropped from
  the table. Unchanged by this phase.
- **`wgpui-devtools` extraction (Phase 7)** has not started.
- **Final cutover (Phase 8) and the legacy alias crate** have not started.
  `wgpui-core` is not the default; `src/` is still the frozen legacy backend.

### Opened or sharpened by this phase

- **Example parity is unblocked but not achieved.** §8's `6.2` row says the four
  missing layers "directly block example parity — `image_loading`, `gif_viewer`,
  `svg`, and any example using `img()`/`svg()` cannot run without all four
  layers existing." All four now exist. **None of those examples has been ported
  to 2.0** — `crates/*/examples/` contains only probes and phase benches. They
  are additionally blocked on the input plumbing Phase 6 disclosed is entirely
  missing (`keyboard.rs`, `dispatcher.rs`, `app_menu.rs` are still Phase 0
  stubs), and `gif_viewer` on the animation driver (§7.1).
- **Scaled images are nearest-neighbour, legacy interpolates** (§3). A real
  fidelity gap, self-contained to fix, deliberately not fixed here because it
  would have cost the gate its exactness.
- **The tinted alpha-mask `svg()` path does not exist** — monochrome atlas plus
  a different pipeline. `svg()` today loads and draws an SVG as a colour bitmap.
- **The byte-exact proof covers two formats at natural size only** (§3). JPEG,
  WebP, BMP, ICO, TIFF share the `into_rgba8()` arm and are untested
  differentially; scaling, grayscale, opacity and corner radius are tested but
  not against legacy.
- **The GIF differential arm proves decode, not blend** — its asset is 100%
  opaque (§3).
- **`scroll_content_gate`'s avatars are synthetic frames over a substitute
  atlas** (§5), correctly so for what that gate measures, but the phrase "real
  decoded frames going through the whole path" should not be repeated from that
  commit message.

### Carried forward, unchanged

Scale factor 1 only and caller-rounded glyph positions (Phase 5.6);
`PrimitiveStore::reflow`'s O(n²) bulk-build cost; GPU occlusion's 1.30× loss on
low-visibility scenes; the transcription-oracle limit on Phase 5.5's
differential; the 2×-scale sub-pixel aliasing quirk; one machine / one driver /
one backend for every number in every report so far; `gpui-ce`'s legacy test
binary still not confirmed to finish.

---

## 9. Honest read

The four layers hold end to end, and the gate is the evidence rather than the
claim: a real file on disk is decoded by `wgpui-widgets`, keyed into a
polychrome atlas page by `wgpui-core`, uploaded to an `Rgba8Unorm` texture,
drawn by a fourth pipeline through the same `issue_sprites` that serves glyphs,
read back, and found byte-identical to what the legacy decoder's own bytes
composite to — including on 8,540 translucent pixels where the alpha multiply
had every opportunity to disagree and did not.

The phase's own weakest link was never the renderer. It was a test that clears
to opaque black and then expected transparency to survive, which is worth
recording as a pattern: a differential's oracle is code too, it is written
faster than the thing it checks, and it fails in the direction of looking like a
catastrophic failure of the subject. The prior agent diagnosed it correctly and
the fix it committed was already right; what was actually missing was that
nobody had run it, and that the assertions around the correct fix were loose
enough (`exact > 0`) to have reported a far weaker result in the same words.

What this phase does not do is make images *good*. Scaled images are harsher
than legacy's, animated GIFs decode but do not animate, the alpha-mask SVG path
is absent, and no example exercises any of it. Those are named above rather than
folded into "images work".
