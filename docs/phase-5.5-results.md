# Phase 5.5 Results — Glyph Rasterisation

Status: **The gap is closed, and it was closed by porting rather than
designing.** `2.0` now turns a font outline into pixels, puts those pixels in
the atlas tile the allocator reserved for them, and copies the page into a real
`wgpu::Texture` that reads back byte-identical. The port is proven to agree with
the legacy rasteriser over 2,592 requests, byte for byte, and the agreement was
watched to fail before it was believed.

This is not a row in `docs/gpu-native-architecture.md`'s phase table. It is the
gap §9's newest risk row and §11's first next-action both name, given the
"explicit phase now rather than discovering it when something is expected to
visibly draw and doesn't" that §9 offers as one of its two options. **This
document does not edit that spec**; a later editor can add the row.

Work lives on branch `wgpui-2.0/phase-5.5-glyph-rasterization`, off `2.0` at
`6cd1b717f6`, pushed to origin, not merged, no PR.

**Nothing under `src/` changed, `docs/gpu-native-architecture.md` was not
edited, and the root `Cargo.toml` is untouched.** `git diff 2.0..HEAD --stat --
src/ docs/gpu-native-architecture.md Cargo.toml` is empty, checked by running it
rather than asserted.

**No new dependency, and no new package in `Cargo.lock`.** `cosmic-text` was
already `wgpui-text`'s dependency (an integration test does not inherit
`[dependencies]`, so the differential's oracle needed it as a dev-dependency
too, at the same version). The lockfile gained exactly one line — the
`wgpui-wgpu → wgpui-text` **dev**-dependency edge §5 explains. `cargo metadata
--locked` exits 0.

**Contents:** §1 What was actually missing · §2 What shipped, and where ·
§3 The rasterisation design · §4 Caching · §5 The atlas half · §6 The device
half · §7 The gate · §8 The one design change, and why it was necessary ·
§9 What verification found · §10 Check, test, and clippy status · §11 Honest
read · §12 What is open

---

## 1. What was actually missing

Quoting §9's risk row in full, because the scope of this phase is exactly the
scope of that paragraph:

> Text is fully shaped, positioned, reconciled, and its atlas tiles are
> allocated — but nothing anywhere in `2.0` yet rasterizes a glyph's font
> outline into the pixels that allocated tile is supposed to hold. `wgpui-text`
> produces positions and tile *requests*; `wgpui-wgpu`'s atlas allocates
> *space*; the CPU-side rasterization step in between […] was out of Phase 5's
> stated scope and was never separately scoped into any phase.

Phase 5 left the seam open in a specific and deliberate shape, which is why this
phase is small: `wgpui_core::scene::atlas::GlyphTileSource` was already the
trait, `wgpui-wgpu`'s `AtlasTileSource` already took its rasteriser as a closure
parameter, and Phase 5's own report said what the missing implementation was
for. Three things had to be built into that shape:

1. **A rasteriser.** `cosmic_text::SwashCache` was already in the dependency
   graph and the legacy crate already calls it; the work is calling it at the
   right point with the right key.
2. **Somewhere for the pixels to go.** The atlas allocated rectangles and held
   nothing in them.
3. **A copy onto the GPU.** Phase 5 named this as "mechanical once something
   draws", and it is.

**It was slightly smaller than the framing suggested in one place and slightly
larger in another**, and both are worth saying plainly since this project's
discipline depends on that holding. Smaller: no third cache was needed, no
rasterisation policy had to be invented, and the whole `swash` call is about
forty lines. Larger: `GlyphRasterKey` turned out not to carry enough information
to reproduce the legacy bitmap at any scale factor other than 1× (§8), which is
a change to a Phase 5 type in `wgpui-core` rather than purely additive work in
the two crates the brief named.

---

## 2. What shipped, and where

| File | Δ | Role |
|---|---|---|
| `wgpui-text/src/raster.rs` | +587 (new) | `GlyphRasterizer`, `RasterizedGlyph` production, `RasterError`, the `SwashContent` conversions |
| `wgpui-text/src/test_fonts.rs` | +37 (new) | One embedded face, for tests that assert about real pixels |
| `wgpui-text/tests/legacy_raster_differential.rs` | +554 (new) | The gate: the transcribed legacy rasteriser, and 2,592 comparisons |
| `wgpui-wgpu/src/render/atlas_upload.rs` | +290 (new) | `AtlasTextures` — texture creation, row padding, `write_texture` |
| `wgpui-wgpu/tests/glyph_atlas_upload.rs` | +363 (new) | Four GPU tests, including shape→raster→pack→upload→read back |
| `wgpui-wgpu/src/render/atlas.rs` | +520 / −43 | Page texel buffers, `get_or_insert_raster`, `PendingUpload`, `tile_texels` |
| `wgpui-core/src/scene/atlas.rs` | +78 / −1 | `RasterizedGlyph`, `AtlasKind::bytes_per_pixel`, `GlyphRasterKey::scale_factor_bits` |
| `wgpui-text/src/shaping.rs` | +35 | `LoadedFont::weight`, `raster_face`, `font_system_mut` |
| `wgpui-text/src/wgpui_text.rs`, `patch.rs`, `wgpui-core/src/scene.rs`, both manifests | +36 | module wiring, the new key field, the dev-dependency |

In total +2,500 / −44 under `crates/`, of which 917 lines are the two
integration-test files — not counting the `#[cfg(test)]` modules inside
`raster.rs` and `atlas.rs`, which are a good deal more.

**Two files outside §3's map**, both recorded rather than glossed:

- `wgpui-text/src/raster.rs` — §3.3's file map for `wgpui-text` names
  `shaping.rs`, `line*.rs`, `fonts/`, and `patch.rs`, and none of them is where
  a rasteriser belongs. `shaping.rs`'s own module doc says the file "never
  rasterises a glyph", which was true and is a statement about *shaping*, not a
  claim on the crate. Rasterising lives in `wgpui-text` because that is the
  crate that owns `cosmic-text` and therefore `swash`; it lives in its own file
  because folding it into `shaping.rs` would put a `SwashCache` on every caller
  that only measures text.
- `wgpui-wgpu/src/render/atlas_upload.rs` — §3.5's map names `atlas.rs` and
  nothing else for this. Splitting it is what keeps `atlas.rs` device-free,
  which is the property Phase 5 built it for and §5 below argues is worth
  keeping.

---

## 3. The rasterisation design

There is not much design here, and that is the point. The reference is
`src/platform/cross/text_system.rs`'s `CosmicTextSystemState::raster_bounds` and
`rasterize_glyph`, plus `src/text_system.rs`'s `TextSystem::rasterize_glyph`
wrapper, and `raster.rs` follows them expression for expression:

1. Look the face up: `(fontdb::ID, fontdb::Weight)` out of the shaper's own
   loaded-face table.
2. Turn the sub-pixel *variant* back into a sub-pixel *offset*:
   `variant / SUBPIXEL_VARIANTS / scale_factor`, with `.trunc()` on the vertical
   component, exactly as the legacy writes it.
3. Build a `cosmic_text::CacheKey` from face, glyph index, device font size,
   that offset, weight and empty flags.
4. `SwashCache::get_image`.
5. Size is `placement.width/height`; bearing is `(placement.left,
   -placement.top)`, which is literally what the legacy `raster_bounds` returns
   as its bounds origin.
6. Convert `swash`'s content type into one of the atlas's two texel formats,
   through the same six-armed match the legacy has — including the Rec. 709
   luminance weights it uses to flatten a sub-pixel mask, and the two "cross"
   arms (a colour bitmap requested as a mask keeps its *alpha*, not its
   luminance; a mask requested as colour widens to `[255, 255, 255, alpha]`).

**One thing is fused relative to the legacy, and it is not a change.** The
legacy calls `get_image` twice per glyph — once through `glyph_raster_bounds`
for the size, once through `rasterize_glyph` for the pixels, with a
`HashMap<RenderGlyphParams, Bounds<DevicePixels>>` in front of the first. Both
land on the same `SwashCache::image_cache` entry, so the second was already a
lookup; the two-step exists because the legacy `PlatformTextSystem` trait hands
the atlas a size before it will call the build closure. `GlyphRasterizer::
rasterize` returns size, bearing and pixels together, which is what the atlas
here actually wants, and the legacy's `raster_bounds` cache has nothing left to
cache — the atlas answers a resident glyph without reaching the rasteriser at
all.

**Errors are ordinary, not exceptional.** `RasterError` has five variants and
every one of them is reached by normal text — a space has no coverage, an
unmapped codepoint has no outline. `AtlasTileSource` maps all of them to `None`,
which `wgpui-text`'s `patch` module already turns into a positioned glyph
carrying `AtlasTileId::NONE`. One bad glyph degrades to a blank, never to a
failed frame, which is the legacy's behaviour too.

**Three things were deliberately not touched**, per the brief's scope: sub-pixel
positioning refinements beyond the four legacy variants, hinting quality
(`Hinting::default()`, as `shaping.rs` already passes to cosmic-text), and
colour-emoji handling beyond what the legacy already does. The colour path is
ported; it is not extended.

---

## 4. Caching

Two layers, neither of them new, and no third one added:

- **`SwashCache::image_cache`**, inside `cosmic-text`, memoises outline → bitmap
  per `CacheKey`. That is the expensive half, and it is the same cache the
  legacy crate relies on. `GlyphRasterizer::cached_bitmap_count` reports its
  size, which is the count half of what the legacy reports under its
  `flamegraph` feature.
- **`GlyphAtlas`'s `tiles_by_key` map**, which is where a `GlyphRasterKey`
  becomes a tile. `AtlasTileSource` consults it *before* calling the rasteriser,
  so the `swash` → atlas format conversion runs once per distinct raster rather
  than once per glyph occurrence: a paragraph's forty `e`s cost one conversion,
  and the end-to-end test measures exactly that (41 glyphs shaped, 30 rasters,
  5 atlas cache hits, 6 blanks).

A `HashMap` in `raster.rs` would sit between two caches that already cover the
same key and would need invalidating whenever the font database changed. It
would be a maintenance cost against no measurement, so it is not there — and the
module doc says so, rather than leaving its absence to look like an oversight.

---

## 5. The atlas half

`GlyphAtlas` allocated rectangles and held nothing in them. Now each page owns
its texels (`page_size²` × `AtlasKind::bytes_per_pixel` bytes), and
`get_or_insert_raster` blits a `RasterizedGlyph` into the rectangle the bin
packer chose.

**It still opens no device**, which is the same argument Phase 5's `atlas.rs`
makes one step earlier, moved along: the blit, the page stride, the row offsets,
the two texel widths, and the eviction interactions are all asserted headlessly,
on any machine. An atlas whose blitting can only be checked on hardware is an
atlas whose blitting does not get checked. Six new headless tests cover the
placement itself — including one that counts the texels a blit touched (`32`,
for an 8×4 mask) so a stride bug shows up as a count rather than as a shrug.

**Uploads are recorded per written tile, not coalesced per page.** The bin
packer scatters small glyphs, so the bounding box of a frame's writes is very
nearly the whole page and uploading it would move megabytes to change kilobytes.
The legacy makes the same choice — one `write_texture` per tile, in
`WgpuAtlasState::upload_texture`.

**Two refusals rather than trust.** A bitmap whose byte count disagrees with its
declared size, and a bitmap whose `AtlasKind` disagrees with the key's, are both
refused before any page is opened. The blit walks rows, so a bitmap one row
short is not a rendering artefact — it is a row of the next glyph.

**The page-buffer cost is real and disclosed.** 1 MiB per monochrome 1024² page,
4 MiB per colour one, held for the page's lifetime; the legacy keeps no CPU
copy. It is kept because it is what makes the path testable without a device and
because `write_texture` reads from it anyway. If a workload ever shows the
resident cost mattering, dropping a page's buffer after its last upload is a
self-contained change behind `page_texels`. Not done now, per this document's
own measure-before-building discipline.

**One consequence is asserted rather than commented.** Freeing a tile does not
blank its texels — the legacy does not either — so space reserved through the
old `get_or_insert` can read back as the glyph that used to live there. That is
safe only because the *eviction event* is what makes a stale reference visible,
and a test now pins both halves (`a_fresh_page_is_transparent_and_a_reused_
rectangle_keeps_its_old_texels`) so a future change that starts clearing on
eviction has to notice this was the reason it did not before.

---

## 6. The device half

`render/atlas_upload.rs` is the port Phase 5 named as deliberately absent and
mechanical: `WgpuAtlasState::push_texture`'s texture creation and
`Monochrome`/`Polychrome` → `R8Unorm`/`Rgba8Unorm` mapping, and
`upload_texture`'s `COPY_BYTES_PER_ROW_ALIGNMENT` row padding and
`queue.write_texture` — including the legacy's own recorded reason for
`write_texture` over a staging buffer ("Work around driver issues […] see
helio/ship_flight repro").

`AtlasTextures::sync` creates textures for new pages, destroys them eagerly for
pages the atlas no longer has, and copies every queued rectangle. It reports
what it did (`UploadReport`), because "the upload happened" and "there was
nothing to upload" are otherwise indistinguishable — and the second is what a
broken drain looks like.

It is a separate type from `GlyphAtlas` rather than fields on it, specifically
so `atlas.rs` keeps the headless property §5 argues for. The two are joined by
`drain_uploads`, a plain list of rectangles: the CPU side does not know a device
exists, and this side does not know how anything was packed.

**A destroyed page's queued uploads are dropped with it**, in `destroy_page`. An
upload names a page and a rectangle and the uploader reads texels back out of
the page, so a queued upload for a page that no longer exists is either a silent
no-op or a read of the wrong page depending on how carefully the uploader was
written — and that is not a thing to leave to the uploader.

---

## 7. The gate

> Port what's there, prove it produces the same pixels as the legacy path for a
> representative set of glyphs.

**Met**, over 2,592 comparisons, and falsified three ways.

### 7.1 Methodology, and the oracle's one real limitation

`crates/wgpui-text/tests/legacy_raster_differential.rs`. Both arms shape and
rasterise against the *same single embedded face* — IBM Plex Sans Regular, the
file the legacy backend already bundles for WASM — loaded into two separate
`fontdb::Database`s containing nothing else, so "they resolved different faces"
is not a way for this test to pass or fail by accident. Both arms assert which
face they got before comparing anything.

The oracle is a **transcription** of `CosmicTextSystemState::{raster_bounds,
rasterize_glyph}` and the face-loading in `load_family` that feeds them, not a
call into `gpui-ce`. It would be better to call the real thing; it is not
reachable. `RenderGlyphParams` is `pub(crate)`, `PlatformTextSystem` is a
private trait, and `TextSystem::rasterize_glyph` is `pub(crate)` too, so no code
outside the root crate can invoke the legacy rasteriser at all — and making it
reachable means changing `src/`, which every phase from 1 onward is forbidden
and which §9 freezes.

**That is weaker than calling the real thing in exactly one way**: it cannot
catch the legacy file changing underneath it. It is not weaker in the way that
matters — the two arms share no state, no cache and no code path, so an
agreement is a real agreement. The test file says this in its own module doc
rather than only here.

### 7.2 Result

| Arm | Requests | Produced a bitmap | Declined | Disagreements |
|---|---|---|---|---|
| Monochrome, 3 sizes × 3 scales × 4 variants | 2,592 | 2,556 | 36 | **0** |
| Colour (the legacy `is_emoji` path), 3 sizes | 216 | 216 | 0 | **0** |

The sample is every distinct glyph the face shapes for
`"Hamburgefonstiv HAMBURGEFONSTIV 0123456789 ,.;:!?'\"()[]{}-_+=@#$%&*/\\<>|~^`"`
— 72 glyph indices, taken through real shaping rather than written down, because
glyph indices are font-local and a hand-written list would silently stop
covering the face if the face changed. Sizes are 12/16/24 px; scale factors are
1.0/1.5/2.0.

**Both halves of the count are asserted, so neither can quietly become the
whole.** A differential where nothing had ink would agree perfectly and prove
nothing, so the test requires that more than half the sample rasterises *and*
that at least one request declines (the space glyph, at every size and scale —
36 of them).

The colour arm exercises the `Mask` → RGBA widening rather than a real colour
bitmap, because the embedded face has no colour glyphs. That is the arm a real
emoji font reaches for any codepoint it has no colour form for, and it is the
arm a machine with no emoji font can still check. The `Color`-content arms are
covered by a direct unit test against the legacy expressions instead, named as
such.

### 7.3 The GPU half

`crates/wgpui-wgpu/tests/glyph_atlas_upload.rs`, four tests, on a real adapter —
**NVIDIA GeForce RTX 4060 Laptop GPU, Vulkan, driver 561.03**, the same single
machine every phase since Phase 0 has measured on. The adapter was verified
present by running `examples/adapter_probe.rs` rather than assumed; a missing
one is reported and skipped through `device::context_or_report`, per Phase 3's
rule.

- A mixed monochrome-and-colour atlas (16 rasters, 2 pages) uploaded and read
  back through `copy_texture_to_buffer`, compared **byte for byte** against
  `GlyphAtlas::page_texels`, with a non-blank assertion so the comparison cannot
  be vacuous.
- A second `sync` on an unchanged atlas uploads **nothing** (`UploadReport ==
  default`), and one new glyph is one copy and no new page.
- A destroyed page drops its texture and its view on the next `sync`.
- **End to end, with nothing synthetic in it:**

```
shaped 41 glyphs into 1 run: 35 tiles, 6 blanks
atlas: 1 page, 30 tiles, 30 allocations, 5 cache hits
rasteriser: 30 rasterized, 7 declined
upload: 30 rectangles, 3,963 texel bytes, 1 page created, 0 skipped
```

Every page read back identical to the CPU side, and **every glyph that claims a
tile finds ink in it** — asserted, 35 of 35. That last assertion is the one that
would have failed at every point in `2.0` before this phase.

### 7.4 Falsified, three ways

A gate that passes on the first run has proved a number is what you expected.
Three separate breaks were made and the gate watched to fail:

| Break | Result |
|---|---|
| `bearing`'s sign flipped in `raster.rs` (by hand, then reverted) | **2,556 of 2,592 disagreed**; all four differential tests failed |
| The scale factor collapsed out of the request, as a key without `scale_factor_bits` would force (a standing test) | **142 of 216 requests differ** |
| The upload origin shifted by one texel in `atlas_upload.rs` (by hand, then reverted) | Two of four GPU tests failed, "differs from the CPU side at texel 65" |

The two hand-made breaks were reverted and the suites re-run green. The third is
a permanent test, because it is about the one real decision this port had to
make (§8) and it is invisible at 1×.

A fourth, smaller check (`perturbing_a_single_texel_is_caught`) confirms the
comparison reaches the texels rather than stopping at the dimensions — the
failure mode where a differential compares two sizes and calls that agreement.

---

## 8. The one design change, and why it was necessary

`GlyphRasterKey` gained a `scale_factor_bits: u32` field. This is a change to a
Phase 5 type in `wgpui-core`, so it deserves its own section rather than a line
in a table.

The legacy turns a sub-pixel variant into a sub-pixel offset by dividing by the
device-pixel ratio:

```rust
let subpixel_shift = point(
    params.subpixel_variant.x as f32 / SUBPIXEL_VARIANTS_X as f32 / params.scale_factor,
    params.subpixel_variant.y as f32 / SUBPIXEL_VARIANTS_Y as f32 / params.scale_factor,
);
```

So the scale factor decides *the bitmap*, not only its size. Phase 5's key
folded the scale into `font_size_bits` (`(font_size * scale_factor).to_bits()`),
which is exactly right for the size and silently wrong for the offset: a 16px
glyph at 2× and a 32px glyph at 1× are the same device size, two different
rasters, and — under Phase 5's key — one tile. The legacy's own
`RenderGlyphParams` hashes `font_size` and `scale_factor` as separate fields,
which is the same conclusion reached from the other direction.

**Two honest notes about it.** First, the collision is narrow — it needs one
process rendering the same face and glyph at two scale factors, i.e. a
mixed-DPI multi-monitor setup — which is presumably why Phase 5 didn't hit it;
nothing in Phase 5 rasterised, so nothing could. Second, **the legacy division
by `scale_factor` looks like a bug, and it was ported anyway.** At 2× the four
variants map to offsets 0, 0.125, 0.25, 0.375, and `cosmic_text::SubpixelBin`
puts 0.125 and 0.25 in the *same* bin — so a 2× display gets three distinct
rasters where it asked for four. Reproducing that is the point: both backends
have to agree about which bitmap a variant means while both exist, and "fix the
legacy's sub-pixel quantisation" is a different piece of work with a different
gate. It is named here so whoever does that work finds this paragraph rather
than rediscovering the arithmetic.

`AtlasKind::bytes_per_pixel` was added in the same file, and `RasterizedGlyph`
beside `GlyphTileSource`, for the reason Phase 5 gives for the trait's home:
`wgpui-text` produces one and `wgpui-wgpu` consumes one, and neither may name
the other.

---

## 9. What verification found

**Two real clippy findings, both fixed rather than suppressed.** A nine-argument
`compare` function in the differential (the same `too_many_arguments` class
Phase 4 hit and fixed), restructured into a `Request` struct and a method on the
two-sided fixture — which is better than a suppression for a reason specific to
this test: a differential whose two arms are fed by two separate argument lists
can compare two *different* requests and call it agreement, and one struct
consumed by both arms removes that failure mode. And a `useless_conversion` in
the format test.

**One test was wrong on its first run**, caught by running it: a check that a
reserved-but-unwritten tile keeps a previously-evicted glyph's texels was
written to reuse a 16×16 rectangle with a 4×4 request, and `etagere`'s bucketed
allocator put the small request somewhere else entirely, so the test asserted
non-zero texels against a fresh region and failed. Rewritten to assert both
halves deterministically (a fresh page is zeroed; a same-size reuse keeps the
old texels), which is a better test than the one that was intended.

**Two `#[derive]`s that did not compile**, both caught by building rather than
reading: `Eq` on a struct holding `[f32; 2]`, and `Default` on a struct holding
`SwashCache`.

**No gate-supporting test was found to be a no-op.** Every falsification in §7.4
was actually run and its output read; the two hand-made ones were reverted and
the suites re-run.

---

## 10. Check, test, and clippy status

- `cargo check --workspace` — passes. `gpui-ce` generates **72 warnings**,
  exactly the baseline Phase 2 recorded and Phases 3–5 carried unchanged,
  including the 5 pre-existing `E0133`s. Read rather than assumed.
- `cargo metadata --locked` — exits 0. `Cargo.lock` gained **one line and no new
  package**: the `wgpui-wgpu → wgpui-text` dev-dependency edge.
- **Tests: 493 passing, 0 failed, 0 ignored, 0 skipped**, across the five
  workspace crates:

| Crate | Target | Tests |
|---|---|---|
| `wgpui-core` | lib | 320 |
| `wgpui-layout` | lib | 6 |
| `wgpui-text` | lib | 54 |
| `wgpui-text` | `legacy_raster_differential` | 4 |
| `wgpui-wgpu` | lib | 50 |
| `wgpui-wgpu` | 6 integration targets (incl. GPU) | 30 |
| `wgpui-widgets` | lib | 23 |
| `wgpui-widgets` | `scroll_content_gate` | 6 |

  On Phase 5's exact counted set (which excluded `wgpui-layout`) this is 487
  against its 459 — **+28**. All four `glyph_atlas_upload` tests were confirmed
  to actually *run* on the adapter rather than skip, by reading their
  `context_or_report` lines under `--nocapture`.

  `cargo test --workspace` was **not** run — it includes `gpui-ce`'s legacy
  suite, confirmed by earlier phases to run 10+ minutes without completing and
  unrelated to any 2.0 branch.
- **Clippy: clean from a genuine cold build.** `cargo clean` first (the final
  run emptied a 360 MiB `target/`; an earlier one removed 6.2 GiB of test
  builds), then `cargo clippy -p wgpui-core -p wgpui-text -p wgpui-wgpu
  -p wgpui-widgets --all-targets -- --deny warnings`. Exit 0, zero warnings,
  zero errors — 113 units checked or compiled from empty, with all five
  workspace crates named in the output, so "clean" is not an incremental run
  reporting nothing because it did nothing. Verified by reading the captured log
  rather than by the exit code alone.
  **Zero suppressions added** — `git diff` for added `#[allow]`/`#[expect]`
  lines under `crates/` is empty, checked by running it. `clippy.toml`'s
  conventions were checked first: its `disallowed-methods` list covers
  `std::process::Command` and `serde_json::from_reader`, none of which this
  phase touches.

  Note `script/clippy` adds `--release --all-features`; the command above
  follows the brief. No crate here has features, so the difference is the
  profile only.

---

## 11. Honest read

**The gap named in §9 is closed, and closing it was mostly transcription.** The
rasterisation itself is forty lines of `swash` call and a six-armed match, both
lifted from a file that already worked. That is what the brief predicted and it
is what happened. The interesting work was in the two places where the port
*could not* be a transcription: `GlyphRasterKey` needed a field it did not have
(§8), and the atlas needed somewhere for pixels to live (§5).

**What "it renders" does and does not mean now.** This is the first phase where
`2.0` produces real pixels and puts them on a GPU, and it is worth being exact
about how far that goes:

- ✅ A font outline becomes a bitmap that agrees with the legacy backend's
  bitmap, byte for byte.
- ✅ That bitmap lands in the atlas rectangle reserved for it, at the right
  coordinates, in the right format.
- ✅ That atlas page reaches a real `wgpu::Texture` and reads back identical.
- ❌ **Nothing draws it.** There is still no sprite pipeline
  (`render/pipelines.rs` names its own unbuilt work), no sprite primitive kind,
  and nothing that binds the atlas texture in a render pass. Phase 5's report
  named both; neither is this phase's scope, and neither got smaller.

So the honest statement is: **`2.0` now has correct glyph pixels on the GPU and
still does not put them on a screen.** That is a genuinely different position
from where Phase 5 left it — the missing step is now "bind and draw", which is
ordinary pipeline work with an obvious shape, rather than "produce the pixels at
all", which had no owner. But nobody has pointed at a screen yet, and §11's
first next-action asked for the gap to be given a home, not for the whole
picture.

**Where this phase is thinner than it sounds.** Three things:

1. **One face.** The differential runs against IBM Plex Sans Regular and nothing
   else. That is a deliberate improvement over Phase 5 (whose shaping tests ran
   against whatever the machine had, and could therefore assert nothing about
   metrics), but it means hinting or content-type behaviour peculiar to another
   face — a bitmap-only CJK face, a real colour emoji face, a variable font — is
   untested. The `SwashContent::Color` arms are covered by direct unit tests
   against the legacy expressions rather than by a face that produces them.
2. **One machine, again.** RTX 4060 / Vulkan / Windows, same as every phase
   since Phase 0. The upload path's row-alignment arithmetic is exactly the kind
   of thing that differs by backend, and it has been checked on one.
3. **The oracle is a transcription** (§7.1), so it cannot notice `src/` changing.
   §9 freezes the legacy backend, which is why that is acceptable rather than
   merely tolerable — but they are different words and this is the second one
   unless someone re-reads the legacy file when it moves.

---

## 12. What is open

**Directly downstream of this phase:**

- **The sprite pipeline and the sprite primitive kind.** The two absences in
  §11. `AtlasTextures::view` already hands out the `wgpu::TextureView` a bind
  group needs, and `Glyph` already carries `atlas_origin`/`atlas_size`, so the
  remaining work is a shader, a bind group layout, and the three-step change
  `patch/primitive.rs`'s own doc describes for adding a kind. Nothing here makes
  it harder.
- **Dropping page buffers after upload**, if the resident MiB ever matter (§5).
  Self-contained behind `page_texels`; not done, because nothing has measured
  it.
- **The legacy sub-pixel-shift quirk** (§8): at scale factors other than 1×, two
  of the four variants collapse into one `SubpixelBin`. Ported faithfully; worth
  fixing on both backends at once, or on neither, and not worth fixing quietly
  on one.
- **A second face in the differential**, especially a colour emoji face, which
  would exercise the `SwashContent::Color` arms for real rather than by
  transcription (§11.1).

**Carried forward from Phase 5, unchanged by this phase:**

- **`estimated_size` on neither `Img` nor `StyledText`.** §6.2 is still
  `diff_key`-complete and not invariant-complete. Deliberately untouched — the
  brief scoped this phase to the visible half of Phase 5's gap and said to leave
  this where it is. It was not trivially in scope: `StyledText`'s
  `estimated_size` needs a measurement path that does not exist, and nothing in
  this phase went near either element.
- **§6.2 as a standing rule** is still two elements down out of a
  `wgpui-widgets` that is mostly Phase 0 placeholders.
- **`SmolStr`'s inline-small-string half**, and `line_wrapper`'s narrowed CJK
  break opportunities.

**Still open in the plan at large:**

- **Phase 6.1's fate** is still undecided — a fused-dispatch follow-up spike, or
  drop the row. Phase 0's Spike B already argued against the standalone form by
  ~1000×. Nothing in this phase bears on it.
- **`wgpui-devtools` extraction (Phase 7)**, unstarted.
- **The cutover (Phase 8)**, where the frozen `Pixels`/`Point`/`Size` geometry
  and the frozen `TextStyle` come across. `raster.rs` is `f32`- and `[u32; 2]`-
  shaped for the same reason the rest of `wgpui-text` is: the swap should be a
  type substitution, not a redesign. `test_fonts`'s ~200 KB embedded face should
  get a feature gate at that point; it is public today because an integration
  test cannot see a `#[cfg(test)]` module, and nothing ships this crate yet.
- **Breadth**, item 2 of §11: every number in every phase report so far, this
  one included, is one machine, one driver, one backend.
