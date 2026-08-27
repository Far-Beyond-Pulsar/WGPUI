# Phase 5 Results — `wgpui-text`, `Img`/`StyledText` `diff_key`, and Atlas-Eviction Subscription

Status: **Phase 5 executed, the gate met and falsified both ways.** This
documents what was built, the shaping and patch-conversion design, the two
`diff_key` decisions and the legacy blocker one of them had to clear, the
atlas-eviction mechanism and its regression test, the gate's real result with
its methodology, check/test/clippy status, and an honest read on what is open.
It follows `docs/gpu-native-architecture.md` ("2.0" below) §3.3, §6, §6.2, and
§8's Phase 5 row, plus `docs/retained-layers.md` ("R-N") §4.3.

Work lives on branch `wgpui-2.0/phase-5-text`, pushed to origin, not merged, no
PR.

**Nothing under `src/` changed, and `docs/gpu-native-architecture.md` was not
edited.** `git diff 1213bd366f..HEAD -- src/ docs/gpu-native-architecture.md`
is empty, checked by running it rather than asserted. The root `Cargo.toml` is
also untouched — every manifest change is under `crates/`.

**Two dependencies were added**, both to crates that had none of them and both
already pinned by the root crate at the same version: `cosmic-text = "0.18.2"`
(to `wgpui-text`) and `etagere = "0.2"` (to `wgpui-wgpu`). `Cargo.lock` gained
six lines and no new package: both were already in the graph via `gpui-ce`, so
the lockfile only records the new edges. `cargo metadata --locked` exits 0 —
Phase 4's lockfile trap was checked for, not assumed absent.

**Contents:** §1 What shipped, and where · §2 The shaping design · §3 The
patch-conversion design · §4 `Img`'s `diff_key`, and the legacy blocker ·
§5 `StyledText`'s `diff_key` · §6 The atlas allocator · §7 Atlas eviction, and
its regression test · §8 The gate · §9 What verification found · §10 Check,
test, and clippy status · §11 Honest read · §12 What is open

---

## 1. What shipped, and where

| File | Lines | Role |
|---|---|---|
| `wgpui-text/src/shaping.rs` | 1,013 | `TextShaper` (cosmic-text), `SharedString`, the font vocabulary, `ShapedLine`/`ShapedRun`/`ShapedGlyph`, the shaping cache |
| `wgpui-text/src/patch.rs` | 455 | Shaped-run → `GlyphRun`/`Glyph` conversion; sub-pixel bucketing |
| `wgpui-text/src/line.rs` | 322 | `WrappedLine` — one shaped line plus its boundaries, and its emission |
| `wgpui-text/src/line_layout.rs` | 245 | index ↔ x mapping over a shaped line |
| `wgpui-text/src/line_wrapper.rs` | 202 | Greedy word wrapping over an already-shaped line |
| `wgpui-text/src/fonts/features.rs` | 124 | `FontFeatures` and its cosmic-text conversion |
| `wgpui-text/src/fonts/fallbacks.rs` | 23 | `FontFallbacks` |
| `wgpui-core/src/scene/atlas.rs` | 424 | `AtlasKind`, `GlyphRasterKey`, `GlyphTile`, `GlyphTileSource`, `AtlasEviction`, and `Scene::evict_atlas` |
| `wgpui-core/src/patch/primitive.rs` | +184 / −6 | `AtlasTileId`, `Glyph::atlas_tile`, `GlyphRun::atlas_tiles` |
| `wgpui-wgpu/src/render/atlas.rs` | 850 | `GlyphAtlas` (etagere bin-packing), `AtlasTileSource` |
| `wgpui-widgets/src/img.rs` | 503 | `ImgKey` and the `Img` description shape |
| `wgpui-widgets/src/styled_text.rs` | 730 | `StyledTextKey`, `TextEngine`, the `StyledText` description shape |
| `wgpui-widgets/tests/scroll_content_gate.rs` | 525 | The gate |

Every §3.3 file map entry for `wgpui-text` is now real; all seven were three-line
Phase 0 placeholders. So were `wgpui-widgets/src/{img,styled_text}.rs` and
`wgpui-wgpu/src/render/atlas.rs`.

**Two files outside §3's map**, both recorded rather than glossed:

- `wgpui-core/src/scene/atlas.rs` — the eviction subscription needs a home in
  `wgpui-core`, since it is the crate that owns layers and their invalidation,
  and §3.1's map does not name one. Putting it in `scene/tile.rs` would have put
  two unrelated meanings of "tile" (Phase 4.5's buffering grid, and an atlas
  allocation) in one file.
- `wgpui-widgets/tests/scroll_content_gate.rs` — the gate spans three crates, so
  it cannot live inside any one module's `#[cfg(test)]`.

**One §3.3 entry is smaller than "moved, not rebuilt" implies**, and §2/§3 below
say exactly where and why. `line_layout.rs` is 245 lines against the legacy
file's 763; the difference is almost entirely `LineLayoutCache`, deliberately
not ported.

---

## 2. The shaping design

### 2.1 cosmic-text does the shaping, unchanged

§6 is unambiguous — "Text shaping stays on the CPU, via `cosmic-text`,
unchanged" — and `TextShaper::shape_line_uncached` is the legacy
`CosmicTextSystem::layout_line` recipe, step for step: build an `AttrsList` with
one span per font run carrying `metadata(font_id)`, `ShapeLine::new(…,
Shaping::Advanced, 4)`, `layout_to_buffer(…, Wrap::None, Ellipsize::None, …)`,
then group the resulting glyphs into runs by face. The same version is pinned as
the root crate uses, so both backends agree about glyph ids, metrics, and
fallback while both exist.

Three legacy behaviours were kept because dropping them would be a visible
regression, and each is a line of code that looks arbitrary without the reason:

- **Fallback substitution is detected and the substituted face is loaded.** A
  glyph whose `font_id` does not match the face its run asked for gets a real
  `FontId` for the face it actually came from — otherwise its raster would be
  requested from the wrong face's atlas entry.
- **Wrapping is `Wrap::None`.** cosmic-text can wrap; this crate wraps itself
  (§2.4), as the legacy system does, because the wrapper has to agree with
  truncation and boundary rules cosmic-text does not know about.
- **Emoji glyph 3 is skipped.** cosmic-text reports a missing-glyph box from an
  emoji face when a codepoint has no colour form; drawing it produces visible
  tofu where the legacy backend draws nothing.

### 2.2 Geometry is bare `f32`

No `Pixels`, `Point<T>`, `Size<T>`. `wgpui-core/src/geometry.rs` set this
convention in Phase 3 and gave the reason: the real geometry surface is frozen
by §7, still lives in the legacy crate, and §3 gives the workspace nowhere to
move it yet. `Glyph::position` in `patch/primitive.rs` is already `[f32; 2]`, so
the conversion in §3 is a move rather than a unit change.

### 2.3 `SharedString` is `Arc<str>`, and its `PartialEq` short-circuits

R-N §2.4 asks a key comparison to short-circuit on unchanged shared clones, so
`PartialEq` is `Arc::ptr_eq(…) || bytes equal`, and `Hash` is by content — the
two have to disagree about what they look at or a rebuilt-but-equal row would
miss the shaping cache. Both directions have a test, and both tests first assert
which case they built (`is_clone_of`) so neither can silently drift into testing
the other.

The legacy `SharedString` is a `SmolStr`, which additionally stores short strings
inline. That half is not here. It is a real difference (a 12-character label
allocates in 2.0 and does not in the legacy backend), and it is the half that
does *not* matter to reconciliation, which is why it was not the priority.

### 2.4 What was deliberately not ported

- **`LineLayoutCache`** (most of the legacy `line_layout.rs`): a frame-indexed
  arena with `reuse_layouts`/`truncate_layouts` bookkeeping, existing because the
  legacy renderer re-lays-out the window every frame and needed somewhere to
  avoid re-shaping. 2.0 does not re-lay-out the window every frame — that is what
  Phase 1 removed — and the shaping it does reach is memoised by the shaper's own
  cache. Porting it would be porting a workaround for a problem the architecture
  removed.
- **The per-character advance cache in `line_wrapper`.** The legacy wrapper runs
  *before* shaping, so per-character advances are all it has; it keeps a
  `HashMap<char, Pixels>` to make that affordable. `wgpui-text` wraps an
  already-shaped line, so every advance is exact — including the ones a
  per-character cache gets wrong, which is not a small set (kerning, ligatures,
  any script where a cluster's width is not the sum of its characters'). Simpler
  *and* more correct, at no extra cost, because the line is shaped anyway.
- **`serde`/`schemars` on `FontFeatures`/`FontFallbacks`** — ~110 lines of
  hand-written deserialisation so a feature map can be written in a settings
  file. That is a configuration concern, and carrying it would make a crate whose
  whole job is "call cosmic-text" depend on a JSON schema generator.
- **CJK and punctuation break opportunities in `line_wrapper`.** The legacy
  `is_word_char` is wider than this one, which handles ASCII whitespace only.
  Narrow-and-correct was chosen over wide-and-approximate: a missed break
  opportunity wraps a line later than ideal, while a wrong one wraps inside a
  word that should have stayed whole. Widening it is a change to one function.

---

## 3. The patch-conversion design

`wgpui-text/src/patch.rs` is §3.3's stated job for that file: a `ShapedLine`
becomes `GlyphRun`/`Glyph` patch payloads. Three decisions carry weight.

**One `GlyphRun` per shaped run, not per line.** A shaped line is already
segmented by face, because fallback can substitute one mid-line. That split is
kept: a face decides which atlas a raster comes from (a colour emoji is not a
coverage mask), and a draw call cannot mix texture formats. Flattening now would
mean re-deriving the split in the sprite pipeline later, from data that no longer
records it.

**Every shaped glyph gets a slab slot, including the blanks.** A space shapes to
a positioned glyph with a real advance and no coverage. It could be dropped — it
costs a slot and draws nothing — and it is not, because `line_layout`'s
index-to-position mapping walks glyphs to answer "where is byte 12", and a run
with holes answers wrong. Blanks carry `AtlasTileId::NONE`, which
`GlyphRun::atlas_tiles()` filters out, so a blank never subscribes its layer to
an eviction it does not care about. Both halves have a test.

**The tile-source seam is a trait in `wgpui-core`.** §6's accounting —
"`wgpui-text` produces glyph positions and atlas tile *requests*; `wgpui-wgpu`'s
atlas allocator turns requests into actual tile coordinates; neither owns the
other's job" — is `GlyphTileSource`, which `wgpui-text` calls and `wgpui-wgpu`
implements. It lives in `wgpui-core` so it costs no dependency edge in either
direction.

Sub-pixel bucketing (`SUBPIXEL_VARIANTS_X = 4`, `Y = 1`, and the `fract() * N`
`.floor()` expression) is the legacy `Window::paint_glyph` code kept character
for character, so a glyph lands in the same variant under either backend.

---

## 4. `Img`'s `diff_key`, and the legacy blocker it had to clear

This is the more interesting of the two, because `Img` having no `diff_key` was
not an oversight. `src/elements/img.rs` documents the reason in a twelve-line
comment above its `Element` impl, and the reason is sound:

> What `paint` shows for an `Img` depends on `ImgState` (per-element,
> `with_optional_element_state`-keyed: `frame_index`, `started_loading`,
> `last_frame_time`) and `ImgLayoutState.replacement` (a fallback/loading
> `AnyElement` substituted in when `request_layout` finds no data yet) — neither
> of which is reachable from `Img::diff_key(&self, _)`.

That is a statement about *ordering*, not about images. In the legacy element the
animation frame and the load phase are discovered during
`request_layout`/`paint`, which run strictly after `diff_key` is asked for its
answer. A key over `source`/`style` alone would report "unchanged" across a GIF
advancing a frame or a pending load resolving, and paint would replay stale
content. Opting out unconditionally was the correct call under that ordering.

2.0 does not have that ordering. An element contributes a `Description` built
from a value that already holds its resolved state — the same way
`WgpuSurface` (Phase 2) already holds its resolved `surface_id`. So `ImgKey`
carries `frame_index` and `load_state` directly, and the two transitions the
legacy comment names are exactly the two the key reports. **The fix is the state
becoming addressable, not a cleverer comparison** — which is worth stating
plainly, because it is easy to read the closed gap as the legacy comment having
been wrong.

`ImgKey`'s five fields and their axes:

| Field | Axes | Why |
|---|---|---|
| `source` | `DISPLAY` | Different resource, same box |
| `frame_index` | `DISPLAY` | An animated source advancing |
| `load_state` | `LAYOUT` + `DISPLAY` | Swaps a replacement subtree in or out — the one case where being conservative is required, not merely defensible |
| `requested_size` | `LAYOUT` + `DISPLAY` | Moves the Taffy leaf and repaints |
| `style` | `DISPLAY` | Grayscale, opacity, corner radius, object-fit all decide how an already-decided box is filled |

**Not the decoded pixels, and nothing requiring a decode.** §6.2's whole point is
that the key is cheap enough to take every frame for every element; hashing an
image's texels would cost more than the rebuild it avoids. Source *identity* plus
frame index is complete regardless: two different pixel buffers cannot share one
source identity at one frame index without the cache substituting content behind
a handle, which it does not do — a reloaded source gets a new `ImageSourceId`.

**A limitation to name:** `ImageStyle` here omits the legacy `loading`/`fallback`
closures. Closures are not comparable and, per R-N §2.4, are never compared;
`ImageLoadState` carries the part of them that is observable. That is correct for
the fingerprint, but it does mean a caller that swaps *which* fallback closure it
supplies without changing the load state gets no invalidation. The legacy
element has the same property for the same reason.

---

## 5. `StyledText`'s `diff_key`

The expensive case, and the reason the key has to be careful rather than merely
present: reconciling a `div` saves a style comparison and a Taffy node;
reconciling a `StyledText` saves a shaping pass — the one piece of per-frame work
§6 explicitly declines to move to the GPU.

**Both expensive fields are compared by identity first.** The text is a
`SharedString` (§2.3). The highlight runs are an `Arc<[(Range, HighlightStyle)]>`,
compared with `Arc::ptr_eq` for the same reason — a syntax-highlighted line
carries dozens of runs and they are almost always the same `Arc` frame to frame,
because whatever produced them did not re-run. Both fall back to a real
comparison when the pointers differ, so a rebuilt-but-identical row is still
reported unchanged; the pointer check is a fast path, never the answer.

**Style is compared whole, except colour.** The legacy `TextDiffKey` already made
this call and it is right: almost every `TextStyle` field affects shaping, so a
finer split buys little. Colour is the exception and is split out, because a
`GlyphRun` carries its colour as a value — so a recolour is a `DISPLAY` update
over the same glyph positions and must not re-shape. Selection, hover, and search
highlight are exactly that case, and §8's third test measures it: recolouring a
row emits 2 nodes and shapes 0 lines.

Highlight comparison is one walk, not two, and reports `(differ, reshapes)`:

| Change | Axes |
|---|---|
| Different `Arc` but equal contents | none |
| A highlight's colour | `DISPLAY` |
| A highlight's weight or slant | `LAYOUT` + `DISPLAY` (a different face is a different shape) |
| A moved or resized range | `LAYOUT` + `DISPLAY` (re-partitions the text into different font runs) |
| A different number of runs | `LAYOUT` + `DISPLAY` |

**`with_highlights` takes the shared handle, not an iterator.** Rebuilding the
`Arc` every frame from an equal `Vec` would silently defeat the short-circuit;
making that visible at the call site beats paying for it invisibly.

**Malformed highlight ranges are skipped, not asserted.** The legacy element
`debug_assert!`s on overlapping or out-of-bounds ranges, which in a release build
means shaping against a run list whose lengths do not add up — and `shape_line`
refuses the whole line for exactly that, so one bad highlight would blank the
row. Skipping degrades one highlight, which is visible and recoverable.

---

## 6. The atlas allocator

`wgpui-wgpu/src/render/atlas.rs` ports the legacy atlas's bin-packing half, using
the same `etagere::BucketedAtlasAllocator` at the same version, so packing
behaviour is shared rather than re-derived.

**It opens no device.** Texture creation, `write_texture` upload batching, and
the `Monochrome`/`Polychrome` format mapping are all real work in the legacy file
and are all deliberately absent, because they are *separable* and separating them
buys something specific: every packing assertion runs headlessly on any machine.
An atlas whose packing can only be checked on hardware is an atlas whose packing
does not get checked. This is named in the module doc, not left as a silent gap.

**Two departures from legacy behaviour, both argued in place:**

- An oversized raster is **reported** rather than growing the page to fit it
  (`min_size.max(&DEFAULT_ATLAS_SIZE)` in the legacy). Growing means a page whose
  size is a function of the largest thing ever put in it.
- A destroyed page's index is **never reissued** — see §9, this was a bug in the
  first version, caught by its own test.

**`AtlasTileId` is one packed `u32`**, 8 bits of page and 24 of slot, and it
lands in the exact four bytes of tail padding Phase 1 left in `Glyph` (44 bytes
of payload in a 48-byte slot). So giving glyphs a real atlas identity costs zero
extra bytes per glyph and `GlyphRun::SLOT_STRIDE` is unchanged — asserted by a
test that checks both the stride and that every field Phase 1 wrote is still at
the offset Phase 1 wrote it to. Out-of-range page or slot values are refused
rather than masked: a truncated page index would make one page's eviction poison
another page's layers, which is the hazard itself.

**`GlyphRasterKey` is compared field by field, never hashed to a `u64`.** An
atlas keyed by a hash is one collision away from silently drawing the wrong glyph
in a way that reproduces only for one user with one font at one zoom level. A
test asserts each of the five fields independently produces a distinct tile.

**Slots are reused from a free list**, so the 24-bit slot field bounds *peak live
tiles* rather than lifetime allocations — the difference between a limit no atlas
reaches and one a long-running editor reaches in an afternoon.

**The rasteriser is a closure parameter** (`AtlasTileSource::new(atlas,
rasterize)`), not a type in this crate. Rasterising means `swash` plus decisions
about hinting, gamma, and colour-emoji handling whose shape is set by what draws
the result — and nothing draws glyphs in 2.0 yet (`render/pipelines.rs` names its
own missing sprite pipeline). Writing one now would mean writing it against an
imagined consumer. A declined raster or a refused allocation degrades to a blank
glyph, never to a failed frame.

---

## 7. Atlas eviction, and its regression test

R-N §4.3's last unaddressed hazard, quoted:

> **Atlas tile references.** Sprites carry `tile.tile_id` into the sort key. A
> retained slab holds tile references that the atlas may evict. Layers must
> subscribe to atlas eviction and take `DISPLAY` when a tile they reference is
> dropped.

Under an immediate-mode renderer this cannot bite: every frame re-records every
sprite. The whole point of a persistent slab is that it does not — a layer that
reconciled clean keeps last frame's bytes, tile coordinates included, and if the
allocator has since handed those texels to a different glyph the layer draws the
wrong picture with no error anywhere. Nothing about the layer changed, so nothing
invalidates it. **Phase 5 is the first phase where this is real rather than
theoretical, because it is the first phase in which anything in 2.0 references an
atlas tile at all.**

**The mechanism.** `GlyphAtlas` accumulates `AtlasEviction::{Tile, Page}` events
and `drain_evictions()` hands them over destructively (one event, one subscriber
— the same contract as the legacy `drain_destroyed_pages`).
`Scene::evict_atlas_batch` takes them, finds every layer holding a glyph in an
evicted page or tile, and invalidates it.

**`DISPLAY`, not `all()`.** Nothing about the layer's layout, hit geometry, or
composite position changed — its glyphs are in the same places at the same sizes,
and only the texels those glyphs point at are gone. Re-emitting re-requests a
tile and rewrites the same slots. Over-invalidating would turn a rare atlas event
into a full relayout of every text-bearing layer on screen, which is exactly the
sledgehammer `force_render` is in the legacy backend and exactly what R-N's axis
vocabulary exists to avoid.

**Residency is scanned, not indexed — on purpose.** The obvious implementation is
a `HashMap<AtlasTileId, HashSet<LayerId>>` updated on every patch. That index has
five update sites (insert, update, remove, layer removal, slab relocation) and a
missed one is *silent*: it produces exactly the stale-texels bug the mechanism
exists to prevent, only now with a mechanism in place that claims to prevent it.
Deriving the answer from the resident primitives makes "what does this layer
reference" true by construction. The cost lands where there is room for it: a
scan is `O(resident glyphs)` once per eviction, against an index's small cost on
every patch — the per-frame path §5.0 spends its whole design budget keeping
cheap. Evictions happen when a page fills or a device is lost. If a profile ever
shows the scan, the index is a self-contained change behind
`Scene::layers_referencing`; building it now would be paying a real maintenance
cost against a measurement nobody has taken.

**The regression test, and confirming it fails without the fix.** Six tests in
`wgpui-core/src/scene/atlas.rs` plus one end-to-end test in `wgpui-wgpu` that
drives a real `GlyphAtlas` into a real `Scene`. Following Phase 4.5's discipline
for a correctness fix, the fix was reverted and the tests re-run:

```
---- scene::atlas::tests::evicting_a_page_invalidates_only_the_layers_referencing_it stdout ----
assertion `left == right` failed: an evicted tile is a repaint, never a relayout
  left: Some(Invalidation(0))
 right: Some(Invalidation(2))
```

Two of six fail with the `invalidate` call removed. They pass with it. The other
four cover cases the removal does not reach — one-tile-versus-page granularity,
blanks not subscribing, a removed layer, and a re-emitted run moving its
subscription — and each is a separate way the mechanism could be wrong while
still invalidating *something*.

---

## 8. The gate

> Scroll-content-heavy scenes (avatars, multi-run text — SFD §3's stated
> motivation) hit the fast path with no per-refill shaping cost for unchanged
> rows, under ambient reconciliation (Phase 1), not because they're inside a
> `.boundary()`.

**Met.**

### 8.1 Methodology

`wgpui-widgets/tests/scroll_content_gate.rs`. Forty list rows, each holding what
SFD §3 names as dominating real rows: an `Img` avatar and two `StyledText`
elements, the title carrying two highlight runs so it shapes as several font runs
rather than one. Driven through the real `Reconciler`, the real `Emitter`, and a
real `Scene` — the same three types every prior phase's gates ran against — with
a real `cosmic-text` shaper reading the machine's own font database.

Not a scroll harness, and deliberately so: §8's Phase 5 row is a claim about
*reconciliation*, and the scroll framing says which workload makes the claim
matter, not which mechanism proves it. Same shape as Phase 1's and Phase 2's
gates — build the frame twice, measure the second.

Counters read per frame: `TextShaper::stats().lines_shaped` (cosmic-text
invocations), `.cache_hits`, `EmissionStats::nodes_emitted`, reused node count
from the `FramePlan`, and `UploadPlan::byte_count()`.

### 8.2 Result

| Frame | Lines shaped | Cache hits | Nodes emitted | Nodes reused | Upload bytes |
|---|---|---|---|---|---|
| 1 (cold) | 80 | 0 | 120 | — | > 0 |
| 2 (identical) | **0** | **0** | **0** | 161 | **0** |
| 3 (identical) | 0 | 0 | 0 | 161 | 0 |

Frame 1 is asserted at exactly 80 and 120 first, because if the first frame does
not do the work, the second frame's zero means nothing. Frame 3 exists because
"costs nothing once" and "costs nothing every frame" are different claims.

**The zero cache hits is the load-bearing number.** It is what distinguishes
"reconciliation skipped the work" from "a memoisation cache absorbed it" — the
shaper is not reached *at all*, not reached and answered cheaply.

### 8.3 The two clauses checked mechanically, not trusted

- **"not because they're inside a `.boundary()`"** — `no_boundary_anywhere` walks
  the whole description tree and asserts `!is_boundary()` on every node,
  including inside each element's own `describe()`. Phase 2 learned this the hard
  way with a test that silently no-op'd itself.
- **Identity is positional** — a companion test walks the same tree asserting
  `element_id() == None` everywhere. Nothing in the gate is ever named.

### 8.4 Falsified both ways

The gate passing on the first run is not evidence it can fail. Both halves of the
phase's `diff_key` work were removed in turn and the gate re-run:

| Removed | Gate's `lines_shaped` | Gate's `cache_hits` | `nodes_emitted` | Tests failing |
|---|---|---|---|---|
| `StyledText`'s `diff_key` | 0 | **80** | 120 | 4 of 6 |
| `Img`'s `diff_key` | 0 | 0 | **40** | 4 of 6 |

The `StyledText` case is the more informative one: `lines_shaped` stays 0 because
the *shaping cache* catches it — and `cache_hits` goes from 0 to 80, which is
precisely the distinction §8.2 says the gate is measuring. The test that asserts
zero cache hits is what makes the gate a statement about reconciliation rather
than about memoisation.

### 8.5 The three supporting tests

- **One changed row reshapes exactly one line.** Editing row 7's title: 1 line
  shaped, 1 node emitted, 160 reused. The avatar and subtitle beside it are
  untouched. This is the other half of the same claim — the saving is real
  because the work is real, and a row that changes pays for exactly itself.
- **A recolour repaints without reshaping.** 2 nodes emitted, 0 lines shaped, 2
  cache hits. This is the case §5's colour split exists for, measured rather than
  asserted.
- **Without reconciliation, the cache alone is measurably weaker.** Same content,
  same shaper, but a fresh `Reconciler`/`Emitter`/`Scene` each frame: 0 lines
  shaped (the cache holds) but 80 cache hits, 120 emissions, 120 conversions, and
  a full upload. The cache cannot skip emission, conversion, or upload;
  reconciliation can.

### 8.6 Magnitude

The gate's number is a zero, so a zero needs something beside it. Shaping the
same 80 lines from cold, on this machine:

| Build | 80 lines | Per line |
|---|---|---|
| Release | 1.42 ms | 17.7 µs |
| Debug | 28.1 ms | 351 µs |

That is what one refill of unchanged rows would otherwise cost. At 40 rows and
60 Hz, 1.42 ms is ~8.5% of a frame budget in release, spent on content that did
not change — and this is a small list. The test asserts only that the time is
non-zero, never against a threshold: a timing threshold in a test is a flake
waiting for a slower CI box.

**One machine.** Same caveat as every prior phase: these are Windows 11 numbers
against the machine's own font database (Segoe UI resolving through
`FontSystem::new()`). Shaping cost varies with font complexity and text content;
the *ratio* the gate asserts is exact and hardware-independent, the magnitude is
not.

---

## 9. What verification found

**One real bug, caught by its own test rather than by review.** The first version
of `GlyphAtlas::open_page` numbered pages from `self.pages.len()`. Because
`destroy_page` removes the page from the list, destroying page 0 handed the next
page the index 0 again — so a retained slab still holding a tile in the destroyed
page would silently start sampling the *new* page's real texels instead of being
caught by its own eviction. That is R-N §4.3's failure reintroduced one level
down, inside the mechanism built to prevent it, and the module's own comment at
the time asserted the opposite ("its index is never reissued"). Fixed with a
monotonic `next_page_index`; `a_destroyed_pages_index_is_never_reissued` guards
it and was what found it.

**Two smaller corrections during the build**, both from running rather than
reading: `Emission`/`PatchList` signatures in the first draft of
`scene/atlas.rs`'s tests were written from memory and did not compile, and a
`let _ = substituted;` left over from an abandoned idea was removed rather than
shipped (AGENTS.md forbids exactly that pattern).

**No gate-supporting test was found to be a no-op**, which was checked the way
Phase 2 learned to check it — every falsification in §8.4 was actually run, and
every "must actually differ from the control" assertion in the `diff_key` tests
is present and asserted before the case is exercised.

---

## 10. Check, test, and clippy status

- `cargo check --workspace` — passes. `gpui-ce` generates 72 warnings, which is
  exactly the baseline Phase 2 recorded and Phases 3–4.5 carried unchanged,
  including the 5 pre-existing `E0133`s. Nothing in this branch touches `src/`,
  so an unchanged count is what it should be, and it was read rather than
  assumed.
- `cargo metadata --locked` — exits 0. Both added dependencies were already in
  the lockfile via the root crate, so `Cargo.lock` gained six lines and **no new
  package entry** — only the three new edges (`wgpui-text → cosmic-text`,
  `wgpui-text → wgpui-core`, `wgpui-wgpu → etagere`, `wgpui-widgets →
  wgpui-text`).
- **Tests: 459 passing, 0 failed, 0 ignored, 0 skipped**, across the four touched
  crates:

| Crate | Target | Tests |
|---|---|---|
| `wgpui-core` | lib | 320 |
| `wgpui-text` | lib | 44 |
| `wgpui-wgpu` | lib | 40 |
| `wgpui-wgpu` | 5 integration targets (incl. GPU) | 26 |
| `wgpui-widgets` | lib | 23 |
| `wgpui-widgets` | `scroll_content_gate` | 6 |

  Phase 4.5 reported 360; Phase 5 adds 99. `cargo test --workspace` was **not**
  run — it includes `gpui-ce`'s legacy suite, confirmed by earlier phases to run
  10+ minutes without completing and unrelated to any 2.0 branch.
- **Clippy: clean from a genuine cold build** (`cargo clean` first),
  `cargo clippy -p wgpui-core -p wgpui-text -p wgpui-widgets -p wgpui-wgpu
  --all-targets -- --deny warnings`. Exit 0, zero warnings, zero errors.
  **Zero suppressions added** — `git diff` for added `#[allow]`/`#[expect]`
  lines under `crates/` is empty, checked by running it. (`wgpui-widgets`'s
  crate root carries a pre-existing `#![allow(dead_code)]` from Phase 0's
  scaffold; it is not this phase's and was not relied on.) `clippy.toml`'s
  conventions were checked first: its `disallowed-methods` list is about
  `std::process::Command` and `serde_json::from_reader`, none of which this
  phase touches.

  Note `script/clippy` adds `--release --all-features`; the command above follows
  the phase brief. Neither crate has features, so the difference is the profile
  only.

---

## 11. Honest read

**The gate is met and the falsification is what makes it worth believing.** A
gate that passes on the first run has proved that a number is what you expected;
a gate that also fails when the mechanism is removed has proved the number is
*about* the mechanism. §8.4 did that twice, and the `StyledText` case in
particular showed the gate distinguishing reconciliation from memoisation, which
is the exact confusion this gate would otherwise be vulnerable to.

**Where this phase is thinner than it sounds.** Three things worth stating
plainly:

1. **Nothing draws a glyph.** `wgpui-wgpu` has no sprite pipeline, so the whole
   path from `GlyphRun` to pixels is untested by pixels. Everything here is
   checked by data — slot counts, tile ids, byte offsets, upload sizes — which is
   the same standard Phase 1 and Phase 2 held to, but it is a weaker standard
   than Phase 3's pixel comparison. A rendering bug in glyph placement would not
   be caught by anything in this branch.
2. **Nothing rasterises a glyph.** `AtlasTileSource` takes the rasteriser as a
   parameter and every test supplies a substitute. The atlas's *packing* is
   tested for real; the raster metrics feeding it are not, because they do not
   exist yet.
3. **The shaping tests are structural.** They assert glyph counts, monotonic
   advance, run grouping, and cache behaviour — never a specific glyph id or
   advance width, because the tests shape against whatever fonts the machine has.
   An advance-width regression in cosmic-text would not be caught here. Embedding
   a test font would fix this and was not done; `TextShaper::with_font_system`
   exists specifically so it can be, without touching anything else.

**One structural note about the branch.** The local branch
`wgpui-2.0/phase-5-text` was already checked out in an abandoned worktree from an
earlier attempt at this phase, so this work was done on a differently-named local
branch and pushed to `origin wgpui-2.0/phase-5-text` (which did not exist on the
remote). The remote branch is correct and complete; the stale local branch and
its worktree can be deleted.

---

## 12. What is open

**For a later phase, named rather than left to be rediscovered:**

- **The sprite pipeline and the glyph rasteriser.** The two absences in §11.
  `render/pipelines.rs` already names the first; the second plugs into
  `AtlasTileSource` with no other change.
- **Atlas texture creation and upload.** The legacy `atlas.rs` has both;
  `wgpui-wgpu`'s port deliberately does not (§6). Mechanical once something
  draws.
- **A sprite primitive kind.** `Img` currently emits a `Quad` standing in for a
  polychrome sprite, because 2.0 has two primitive kinds and no sprite kind.
  Adding one is the three-step change `patch/primitive.rs`'s own doc describes.
  The eviction mechanism is already kind-agnostic — it scans glyph runs today
  only because they are the only thing referencing tiles.
- **`§6.2`'s standing invariant is two elements down, not finished.** `Img` and
  `StyledText` were the two named exceptions and both are closed. But §6.2 is a
  *standing rule* — "every first-party element type ships with `diff_key`
  implemented" — and most of `wgpui-widgets` is still Phase 0 placeholders
  (`div.rs` is 14 lines, `svg.rs` 2, `canvas.rs` 2, `image_cache.rs` 2). The rule
  is now demonstrated on the two hardest cases; applying it to the rest is
  ordinary work for whichever phase builds those elements.
- **`estimated_size` on neither.** §6.2 says "and `estimated_size`/`on_frame`
  wherever they apply." Neither is implemented for `Img` or `StyledText` here.
  For `Img` it plainly applies (a known intrinsic size); for `StyledText` it
  needs a measurement path this phase did not build. This is a real, partial
  discharge of §6.2 and is reported as one rather than counted as complete.
- **`SmolStr`'s inline-small-string half** (§2.3), and the CJK break
  opportunities `line_wrapper` narrows away (§2.4). Both are small, both are
  named in their module docs.
- **Phase 6.1** is a spike, not a build, and Phase 0's Spike B already argued
  against it; **Phase 7** is the devtools extraction; **Phase 8** is the cutover,
  where the frozen `Pixels`/`Point`/`Size` geometry and the frozen `TextStyle`
  come across and this crate's placeholder vocabulary is replaced by the real
  one. Nothing in this phase makes that harder: `wgpui-text`'s types are
  `f32`-shaped exactly so the swap is a type substitution rather than a redesign.
