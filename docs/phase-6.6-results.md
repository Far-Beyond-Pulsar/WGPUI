# Phase 6.6 — `Div`'s real implementation: styling, layout, children

Branch: `wgpui-2.0/phase-6.6-div-emission`, off `2.0` at `4cfef84c91`
(Phases 0–6.3, merged). Not merged, no PR.

Adapter for every rendering result below, verified rather than assumed:
**NVIDIA GeForce RTX 4060 Laptop GPU, Vulkan, driver 561.03**,
`INDIRECT_FIRST_INSTANCE=true`, `MULTI_DRAW_INDIRECT_COUNT=true`. Same single
machine as every prior phase; the breadth caveat §11 already carries is
unimproved by this one.

**This was the second attempt.** The first died to an API session-limit error
during its own first milestone, mid-edit on `Quad`'s field extension, with
nothing committed. It lost nothing because it had built nothing. This attempt
pushed after each milestone rather than at the end, for that reason.

---

## 0. The gate, and whether it was met

> **Gate:** A real, non-trivial styled `div()` tree — background, border,
> rounded corners, box shadow, nested children laid out via Taffy, in one
> example — reconciles, emits, and renders byte-exact against the legacy
> renderer for the same content.

**Met.** `crates/wgpui-wgpu/tests/legacy_div_differential.rs`'s
`phase_6_6_div_tree_gate`: a shadowed, bordered, rounded card holding three
rows, each holding a pill — three levels deep, nine elements, flexbox with
border box, padding, gap and cross-axis stretch — renders **53,248 of 53,248
pixels byte-exact** against the legacy renderer painting the same content, with
50,078 of those pixels painted by at least one arm. 2.0 emits 8 quads and 1
shadow; the legacy arm draws 12 primitives for the same picture.

Four supporting gates, all met, all on the same adapter:

| Gate | Result |
|---|---|
| `phase_6_6_quad_gate` — the quad shader against the frozen legacy `fs_quad` | **551,936 / 551,936** pixels byte-exact, 14 cases, 4 draw modes |
| `phase_6_6_div_gate` — a childless `div()` against `Style::paint`'s 5-draw sequence | **315,392 / 315,392** pixels byte-exact, 8 cases |
| `phase_6_3_shadow_gate` — extended with a per-corner case | **354,816 / 354,816** pixels byte-exact, 9 cases (was 315,392 / 8) |
| `phase_6_6_styled_text_underline_gate` — bands emitted by a real `StyledText` | **73,728 / 73,728** pixels byte-exact, 3 bands |

Every one of those was watched failing before being trusted. §5 lists the
perturbations and what each one moved.

---

## 1. Three answers up front, because they change how the rest reads

### 1.1 Did the Tailwind-style DSL already exist in `2.0`? **No. Not a partial one, not a stub.**

`crates/wgpui-widgets/src/styled.rs` **did not exist**. Nothing anywhere in the
`2.0` crates had a `bg()`, a `border_1()`, a `rounded_md()`, or a `Styled`
trait; a grep for any of them across `crates/` returned nothing at all. §7 lists
"the `Styled` DSL" among the surfaces that are "byte-for-byte the same," which
is a statement about the eventual alias crate (§3.7, Phase 8) and was never a
claim that `2.0` presented one yet — but it reads like one, and it is worth
being explicit that the starting point here was zero.

What this phase built is a **hand-written subset**: 89 methods in
`crates/wgpui-widgets/src/styled.rs`. The bound on the subset is concrete rather
than aesthetic — every method is one `wgpui-core`'s primitive vocabulary and
Taffy's style type can honour today.

**The full surface is materially larger and mostly macro-generated**, and that
is the honest reason it is not ported here. The legacy `Styled` trait
(`src/styled.rs`, 993 lines) expands ten proc-macro invocations —
`gpui_macros::style_helpers!()`, `margin_style_methods!()`,
`padding_style_methods!()`, `position_style_methods!()`,
`overflow_style_methods!()`, `cursor_style_methods!()`,
`border_style_methods!()`, `box_shadow_style_methods!()`,
`visibility_style_methods!()`, `derive_inspector_reflection` — into several
hundred methods across a spacing scale, a colour scale, and every side/corner
combination. Reproducing that faithfully means porting a proc-macro crate, which
is Phase 8's alias-crate work. Doing half of it by hand and calling the API
"unchanged" would be exactly the kind of claim these reports exist to prevent.

What *is* faithful: the shapes and names are the legacy ones, and the numeric
values of the Tailwind scales are transcribed from `gpui-macros`' own expansion
rather than guessed — all eight `shadow_*` presets, both layers each where the
legacy has two, to the digit.

One deliberate signature change, and it is not cosmetic: **colours are
straight-alpha `[f32; 4]`, not `Hsla`.** Every `2.0` primitive carries RGBA as a
value and the conversion happens before a colour reaches a slot; that is what
lets a recolour be a `DISPLAY` update over unchanged geometry. Putting an
`Hsla`→RGBA conversion in `wgpui-widgets` would put a colour space in a crate
that otherwise names none. The alias crate is where an `Hsla` argument becomes
an RGBA field.

### 1.2 Was the emission the only thing missing? **No. The quad shader could not have passed this gate.**

§8's Phase 6.6 row says to bake corner radius and border width "into the shader
inputs `Quad` already carries, or extended if it doesn't — check rather than
assume." Checked, and both halves of that turned out to need work — but the
second one, which the row does not mention at all, was the blocking one.

`crates/wgpui-wgpu/src/render/shaders/quads.wgsl` was **not** a port of the
legacy quad shader. It was a deliberately hard-edged rounded-rectangle SDF, 118
lines against the legacy file's 810, with this comment on the discard:

> Hard-edged rather than antialiased on purpose: every comparison this shader
> takes part in is a bit-exact one between two draw paths, and a coverage ramp
> would make "identical" depend on rasterization order.

That reasoning was correct for the comparisons Phases 1–6.3 actually ran — all
of them between 2.0's own four draw modes — and it is fatal to this phase's
gate. The legacy `fs_quad` antialiases through `saturate(0.5 - outer_sdf)` and
produces a coverage ramp several pixels wide at every rounded corner. **No
amount of care on the emitting side could have made a hard-edged shader
byte-exact against it.** This is the fourth phase in a row to find a shader in
`render/shaders/` was not what a comment or a phase row implied (Phase 5.6:
`mono_sprites`; Phase 6.3: `shadows` and `underlines`; now `quads`, which was
not a placeholder but was not a port either).

So the fragment stage is now a transcription of `fs_quad` —
`pick_corner_radius`, `quad_sdf_impl`, `quarter_ellipse_sdf`, `over`, the
`reduced_border` substitution for zero-width sides, both background fast paths,
and the final `mix`/`saturate` composite. What is deliberately **not**
transcribed is listed in the shader's own header and in §4 below.

### 1.3 Did children and layout need something new? **No — and that is the surprise.**

The brief flagged milestone 3 as "the one most likely to reveal something
genuinely new is needed." It did not. `Description::children()` has carried a
real child list through reconciliation since Phase 1, `patch/emit.rs`'s walk
already accumulates ancestor origins and folded scroll offsets, and
`LayoutTree::request_layout` already takes a child list. A `Div` that builds
`Description`s and hands them to `.children()` gets correct nested layout with
no new mechanism anywhere.

What milestone 3 revealed instead is a **paint-order divergence** that has
nothing to do with children being carried and everything to do with when a
parent's own primitives are emitted relative to theirs. §3 has it, with numbers.

---

## 2. What was built, per layer

Diff against the base, code only: 25 files, **+5,741 / −167** (26 files and
+6,340 with this report). `src/`, `docs/gpu-native-architecture.md`,
the root `Cargo.toml` and `Cargo.lock` are untouched — confirmed by
`git diff --stat 4cfef84c91 -- src/ docs/ Cargo.toml Cargo.lock`, which is
empty. **Zero new dependencies.**

### `wgpui-core`

**`patch/primitive.rs`** — `Quad` widened from one uniform corner radius and one
uniform border width to `corner_radii: [f32; 4]` and `border_widths: [f32; 4]`,
in the legacy `Corners`/`Edges` field order (top-left, top-right, bottom-right,
bottom-left; top, right, bottom, left). `SLOT_STRIDE` 64 → 80, which is already
a multiple of 16 and so drops the tail padding Phase 1 needed. `Shadow` widened
the same way for the same reason, `SLOT_STRIDE` 48 → 64. Both gain a per-word
encoding assertion, because a transposed radius pair rounds the wrong corner and
produces an entirely plausible picture.

Why widen rather than keep the uniform convention: the Tailwind surface is
per-corner and per-side (`rounded_t_md`, `border_b_1`), and the alternative —
emitting extra plain quads to fake one rounded side or one bordered edge — draws
something the legacy renderer does not, so it could never be byte-exact. This
also closes a limitation Phase 6.3 had to disclose about its own shadow proof
("**Per-corner radii are outside this proof**"); that sentence is now deleted
rather than restated, and the test that carried it moves all four radii
independently.

`occlusion.rs`'s `quad_coverage_item` insets by `Quad::max_corner_radius()` /
`max_border_width()` — the worst corner and the widest side, which is the only
sound direction for a conservative opaque region.
`test_support/raster.rs`'s CPU oracle rounds by the same max, for the same
reason, stated in the code.

**`patch/apply.rs`** — one gate assertion (`layer_byte_count == 640_000`)
derived from `Quad::SLOT_STRIDE` instead of written out. What that gate measures
is the *ratio* between one slot and a 10,000-quad layer; a literal there means a
phase widening `Quad` discovers it broke an unrelated gate rather than that it
changed a constant.

### `wgpui-layout`

**`taffy_tree.rs`** — the re-export list grew from three names to ten
(`AlignContent`, `AlignItems`, `BoxSizing`, `FlexWrap`, `LengthPercentage`,
`LengthPercentageAuto`, `Overflow`, `Position` added), plus a
`LayoutSides<T> = taffy::geometry::Rect<T>` alias for the four-sided value
`padding`/`margin`/`border`/`inset` are expressed as. Named `LayoutSides` rather
than `LayoutRect` because that name already means the *computed* rectangle.
Nothing about §3.2's "don't leak `taffy`" policy changed — the leak is closed in
the same place it always was, over a wider surface.

### `wgpui-wgpu`

**`render/shaders/quads.wgsl`** — the port described in §1.2. 118 → 296 lines.

**`render/shaders/shadows.wgsl`** — regains the legacy `pick_corner_radius`
branch that Phase 6.3 collapsed, now that `Shadow` carries four radii.

**`render/pipelines.rs`** — the two stride/field-drift assertions updated, and
the quad one strengthened from "contains `struct QuadSlot`" to a per-field check
matching what the shadow and sprite ones already do.

### `wgpui-widgets`

| File | Before | After |
|---|---|---|
| `div.rs` | 14 lines (a fieldless `pub struct Div;`) | 511 |
| `div/interactivity/style.rs` | 3 lines | 586 |
| `div/diff.rs` | 3 lines | 132 |
| `styled.rs` | **did not exist** | 762 |
| `styled_text.rs` | 730 | 1,270 |

**`div/interactivity/style.rs`** — `DivStyle`, `Corners`, `Edges`, `BoxShadow`,
`DivStyle::paint`, and `classify_style_change` (§6.2's engine). `paint` is a
transcription of `Style::paint` (`src/style.rs:683`) in its order: every
`box-shadow` layer, then the background quad, then the border quad. Including
the detail that looks like decoration and is not — the background quad carries
its *own* colour with the alpha zeroed as its border colour, which is what makes
the shader's `over(background, border_color)` a no-op near a rounded corner.

`DivStyle` is a **resolved** style, not a `StyleRefinement`. A refinement is a
sparse `Option`-per-field overlay whose whole purpose is cascading — base,
`:hover`, `:active`, group state — and §8's Phase 6.6 row scopes interactive
states out explicitly. Building cascade machinery with exactly one layer in it
would be inventing the shape of a problem this phase does not have. When
interactive states land, the cascade goes *above* this type and nothing here
changes.

**`div/diff.rs`** — `DivDiffKey`, delegating the whole style comparison to
`classify_style_change` rather than comparing by equality, so a hover recolour
reports `DISPLAY` and not `LAYOUT`. It carries the child *count* and never the
children: folding a child's fingerprint into its parent's would make a leaf's
change rebuild the whole ancestry, which is the exact behaviour ambient
reconciliation exists to avoid. This is the first first-party element with
children, so it is where `diff_key.rs`'s standing rule first has something to
bite on.

**`styled.rs`** — §1.1.

**`div.rs`** — `Div`, `div()`, `IntoDescription`, `describe()`. `describe`
consumes `self` because a `Description` is not `Clone` (it owns a
`Box<dyn ReconcileKey>` and a `Box<dyn Emit>`), which matches `RenderOnce::render`'s
own shape rather than departing from it. An element that paints nothing gets no
emitter at all rather than one writing an empty emission — the distinction is
load-bearing, because `Emitter::emit` counts an element with an emitter as
visited-and-skipped and one without as not emitting, and a grouping `div` (most
of a real tree) should be the second.

**`styled_text.rs`** — `UnderlineStyle`, `StrikethroughStyle`, two new
`HighlightStyle` fields, and `emit_decorations`. §5.4 has the details.

---

## 3. The one real divergence, measured rather than argued

`Style::paint` paints a parent's border **after** its children — the border draw
sits on the far side of the `continuation` the children are painted in. In 2.0 a
parent's whole emission is appended before any child's, so a parent's border
lands *under* its children.

This could have been left as a sentence in a comment. Instead
`an_overflowing_child_is_where_the_paint_order_difference_becomes_visible`
builds both shapes and measures both:

- A child absolutely positioned over its parent's 12px border band:
  **1,296 of 53,248 pixels disagree.**
- The same tree with the child moved 4px inside the border:
  **0 of 53,248 disagree** — byte-exact.

So the divergence is real and it is *scoped*: it affects a child that overflows
into its parent's border band, and nothing else. Every child of a laid-out flex
box that stays inside its padding — which is every child in the gate's own tree,
and in the overwhelming majority of real UI — is unaffected.

The test asserts the disagreement rather than asserting its absence, and says so
in its own doc: a future phase that fixes the ordering watches it fail and
inverts it. The fix belongs to §5.1's per-layer ordering pass, which is what
decides z-order, not to `DivStyle::paint`.

One smaller, related note: `DivStyle::paint` emits the border as **one**
unclipped quad where `Style::paint` draws the same full-bounds quad four times,
each clipped by a `ContentMask` to one edge band (2.0 has no per-primitive clip
— §5.2 sends the frame's clip to the occlusion pass). That the two are
equivalent is an argument, so `phase_6_6_div_gate` renders the legacy
four-clipped-draw sequence and compares pixels rather than asserting it. Eight
cases, 315,392 / 315,392 exact, including a 30px border thicker than half the
box and a `rounded_full()` pill.

---

## 4. What the differentials prove, and what they explicitly do not

The oracle is Phase 6.3's, extended: every legacy arm `include_str!`s the
**frozen shader file itself** and compiles it, rather than transcribing it.
`legacy_div_differential.rs` compiles two of them at once and replays
`Style::paint`'s per-element sequence against them.

The layout half has its own discipline, and it is the part most easily got
wrong: **the legacy arm's geometry is computed independently, not read out of
2.0's scene.** Every rectangle comes from flex arithmetic written out in the
test file — border box, padding, gap, cross-axis stretch —
and `the_layout_oracle_matches_what_taffy_actually_computed` then confronts that
arithmetic with the primitives 2.0 actually emitted (not with the layout tree,
so the check covers the emit walk's ancestor-origin accumulation too). Reading
positions out of the 2.0 scene would have proved the shaders agree and proved
nothing at all about layout, which is half of what this phase built.

**Outside these proofs**, stated rather than discovered later:

1. **Gradient, pattern and radial backgrounds.** 2.0's `Quad.background` is one
   solid straight-alpha RGBA, so `fs_quad`'s `gradient_color` branches for tags
   1/2/3 are unreachable from a 2.0 quad and are not transcribed. Blocks
   `text_gradients` and any example using `linear_gradient`/`pattern_slash`.
2. **Dashed borders.** No `border_style` field on `Quad`; the ~180-line dashed
   branch of `fs_quad` is not transcribed. `border_dashed()` has no 2.0
   counterpart.
3. **Per-fragment clipping.** No content mask on any 2.0 primitive. Both arms
   are given an effectively infinite mask.
4. **`filter` / `backdrop_filter`.** `Style::paint` wraps its box in a filter
   layer and paints a backdrop filter before the background; neither exists in
   2.0 (Phase 6.4).
5. **Element opacity.** `Window::paint_quad` multiplies every colour by
   `element_opacity()`. 2.0 has no opacity stack; a caller folds it into the
   colour, exactly as `quad_coverage_item` already documented.
6. **Scale factor.** Every legacy paint scales by `window.scale_factor()`.
   Everything here is scale factor 1, matching Phase 5.6's disclosed limit.
7. **Colour-space conversion.** Every colour under test converts through nothing
   but 0, 0.5, 1, 2, 3 and 6, and each differential asserts its own transcribed
   `hsla_to_rgba` produces exactly the bytes 2.0 is handed. This is what the
   gates hold *fixed* so they can prove something else.
8. **Where a line of text sits inside its element.** See §5.4.

---

## 5. Per-milestone status, with the falsifications

### 5.1 Milestone 1 — a styled childless `div()` emits `Quad`s byte-exactly. **Met.**

Commit `85cc7c7050`.

`phase_6_6_quad_gate`: 551,936 / 551,936 pixels across 14 cases chosen to reach
each branch of `fs_quad` a 2.0 quad can reach — the unrounded/unbordered fast
path, the inner-straight-border fast path, the circular inner edge, the
quarter-ellipse arm, the `-1.0` arm for a border thicker than half the box, the
`reduced_border` substitution for zero-width sides, four different radii, a
translucent border over an opaque fill, fractional geometry, and a quad reaching
past the viewport corner. All four draw modes identical.

`phase_6_6_div_gate`: 315,392 / 315,392 across 8 real `div()`s, each positioned
by Taffy and each compared against `Style::paint`'s full 5-draw sequence.

**Watched failing, two ways**, chosen to hit different halves of the
transcription:

- A sub-pixel radius change (`20.0` → `20.25`): **60 of 39,424 pixels
  disagree.** The old hard-edged shader could not have seen this at all — it is
  purely a change in antialiased coverage — which is itself the measurement that
  §1.2's rewrite was necessary rather than tidy.
- A transposed corner-radius pair: **314 of 39,424 disagree.** This is the bug
  `pick_corner_radius`'s quadrant order can hide, and it is a bug no
  uniform-radius `Quad` could ever have exhibited.

### 5.2 Milestone 2 — `box-shadow` emits `Shadow` patches. **Met.**

Commit `8b09f170fc`. `Shadow` widened to per-corner radii (§2), the shadow
shader regains `pick_corner_radius`, and `phase_6_3_shadow_gate` grows a
per-quadrant case: **354,816 / 354,816** pixels across 9 cases, 22,832 painted on
the new one alone.

`DivStyle`'s shadow conversion transcribes `Window::paint_shadows`: the
rectangle is the element's, displaced by the layer's offset and dilated by its
spread (`dilate` moves the origin in by the amount and grows the size by twice
it, so a negative spread — which every multi-layer Tailwind shadow uses —
shrinks it). It reuses the *unspread* box's clamped radii, because that is what
`Style::paint` hands every layer however far that layer's own spread moved its
rectangle; recomputing the clamp against the spread rectangle would be more
principled and would not match.

Watched failing: removing the box-shadow from the gate's tree disagrees at
**15,236 of 53,248 pixels.**

### 5.3 Milestone 3 — children and real Taffy layout. **Met.**

Same commit. The headline gate (§0). `Description::children()` needed nothing
(§1.3); the paint-order divergence is §3.

`div.rs`'s own tests cover the reconciliation half without a device: an
identical second frame reuses everything and emits nothing (`nodes_emitted == 0`,
`layout_nodes_created == 0`, `layout_nodes_reused == 1`); a recolour updates two
records in place at the same `RecordKey`s and inserts none; gaining a background
inserts one record *and* updates one, because the background is emitted before
the border and so shifts it to ordinal 1; and two named children survive a
sibling reorder as `NodeOutcome::Reused`.

Watched failing: a one-pixel gap change moves every row below the first and
disagrees at **692 of 53,248 pixels.**

### 5.4 Milestone 4 — `StyledText` emits `Underline` patches. **Met.**

Commit `e80e1708a9`. `phase_6_6_styled_text_underline_gate`: a real
`StyledText`, shaped by real `cosmic-text`, laid out by Taffy, emitted and
applied through the real patch path, produces three bands — a wavy spelling
squiggle, a straight underline, and a strikethrough whose colour falls back to
its run's — and all three render **73,728 / 73,728 pixels byte-exact**.

The placement is `paint_line`'s (`src/text_system/line.rs`), transcribed as
offsets *from the baseline*:

```
underline_y     = baseline + descent * 0.618
strikethrough_y = baseline + ((ascent * 0.5 + baseline_offset) * 0.5) - baseline_offset
where baseline_offset = (line_height - ascent - descent) / 2 + ascent
```

and a wavy rule gets `thickness * 3` of vertical box, which is
`Window::paint_underline`'s own rule. `TextEngine::shape_and_convert` split into
`shape` and `convert` so the element can see the line's ascent, descent and
per-glyph pen positions — the same values `paint_line` reads.

Two behaviours that look like polish and are not:

- **Adjacent identically-styled ranges are merged into one band.** `paint_line`
  only finishes a decoration when the style changes. Two abutting bands blend
  their shared boundary column twice under straight-alpha `over` and would not
  be byte-exact against one.
- **Overlapping and out-of-range highlights are skipped by exactly the rule
  `font_runs` already applies**, so a decoration can never be drawn under a span
  that was never shaped as its own font run.

Neither decoration affects shaping, so `StyledTextKey` reports a decoration
change as `DISPLAY` and never re-runs the shaper — asserted in
`adding_an_underline_repaints_without_reshaping`, not assumed.

**What this milestone deliberately does not claim**, and it is a pre-existing
gap rather than a new one: 2.0's underline does not land on the same *screen*
pixel as the legacy element's, because `StyledText` places a line's baseline at
its element's `bounds.y` where `paint_line` places the top of the line *box*
there and derives the baseline from ascent and padding. `docs/phase-5.6-results.md`
already disclosed that disagreement for glyphs ("positions rounded, since
`wgpui-text` doesn't floor the pen the way the legacy renderer does"); nothing
about decorations is the right place to fix it. What is proved is that a
decoration sits correctly relative to *its own text*, which is a decoration's
whole job and the part a reader would notice if it were off by a pixel.

Ten emission tests in `styled_text.rs`, each asserting against an independent
transcription of `paint_line`'s arithmetic rather than against whatever the
implementation computed.

### 5.5 Milestones, adjusted

The four milestones in the brief were kept. One thing was added inside milestone
1 that the brief did not anticipate and that turned out to be the phase's
largest single piece of work: the `fs_quad` port (§1.2). One thing was added
inside milestone 2 for consistency: widening `Shadow` alongside `Quad`.

---

## 6. Interactive-state boundaries: were they clean?

**Yes, cleanly, and in a way that will not need undoing.**

The boundary held at exactly one seam: `DivStyle` is a *resolved* style, and
everything interactive is a question of what *produces* one. Nothing in
`div/interactivity/style.rs`, `div/diff.rs` or `div.rs` has an `Option`-per-field
overlay, a state enum, a hover flag, or a hit region. `div/events.rs`,
`div/interactivity/hitbox.rs`, `div/interactivity/layer_paint.rs` and
`div/scroll_state.rs` are still the original 3-line Phase 0 placeholders,
untouched.

The one place the boundary was *nearly* crossed and deliberately was not:
`Div::scroll_offset` exposes `Description::scroll_offset`, which has carried a
displacement since Phase 1 and which Phase 2's boundary gates depend on. It is
the raw mechanism only — nothing decides what the offset should be, subscribes
to a wheel event, or clamps to a content extent. A `Div` that could not reach it
would have made Phase 2's gates unreachable from a real element; a `Div` that
managed it would have been a scroll container, which is out of scope.

`Div::boundary()` and `Div::uncached()` are exposed for the same reason and with
the same restraint.

---

## 7. Check, test and clippy status

- `cargo check -p wgpui-core -p wgpui-layout -p wgpui-text -p wgpui-widgets -p wgpui-wgpu --all-targets`: clean.
- `cargo test` across those five crates: **628 passed, 0 failed, 0 ignored,
  0 skipped.** Phase 6.3 reported 517 across a set it did not enumerate, so
  the delta is not stated as a measured one here.
- `cargo clippy -p wgpui-core -p wgpui-wgpu -p wgpui-widgets --all-targets -- --deny warnings`,
  after `cargo clean -p` on all five crates so the build was genuinely cold:
  **clean, exit 0, zero suppressions.** Three real fixes, none of them a
  `#[allow]`: a derivable `Default` impl on `DivStyle` replaced by the derive; a
  `bool::then(|| …)` with a non-lazy body changed to `then_some`; and a
  four-deep nested tuple in the div differential's oracle replaced by a named
  `ShadowLayer` struct.
- `cargo test --workspace` was **not** run, per the brief: it includes
  `gpui-ce`'s legacy suite, confirmed by several prior phases to run 10+ minutes
  without completing and unrelated to any `2.0` branch.
- `src/`, `docs/gpu-native-architecture.md`, the root `Cargo.toml` and
  `Cargo.lock` are untouched — verified by an empty
  `git diff --stat 4cfef84c91 -- src/ docs/ Cargo.toml Cargo.lock`, not by
  inspection. Zero new dependencies.

All four GPU gates were confirmed actually running (not silently skipped by
`context_or_report`) by reading their `--nocapture` adapter lines, quoted in §0.

---

## 8. What is still open

### Closed by this phase

- §8's Phase 6.6 row, at its stated gate.
- Phase 6.3's disclosed limit that nothing in `2.0` emitted a `Shadow` or an
  `Underline`. Both are now emitted by real elements.
- Phase 6.3's shadow proof's own disclosed limit ("per-corner radii are outside
  this proof").
- §6.2's `diff_key` invariant for `Div` — the third first-party element to get
  one, after `StyledText` and `Img`.

### Opened or sharpened by this phase

- **Paint order: a parent's border draws under its children, not over them.**
  §3, measured at 1,296 / 53,248 pixels on the pathological case and 0 on the
  ordinary one. Belongs to §5.1's ordering pass.
- **The Tailwind DSL is a hand-written 89-method subset**, against a legacy
  surface of several hundred macro-generated methods. §1.1. This is Phase 8's
  alias-crate work, but it is now a *known quantity* rather than an unexamined
  line in §7.
- **`Quad` still has no gradient/pattern background and no `border_style`.**
  Both are reachable from the frozen DSL (`bg(linear_gradient(…))`,
  `border_dashed()`) and neither has a 2.0 counterpart. `text_gradients` and
  `pattern` are blocked on the first.
- **No element opacity anywhere in 2.0.** Every legacy paint multiplies by
  `element_opacity()`. The `opacity` example is blocked on this and it is not
  Phase 6.4's backdrop-blur work, though the two are usually mentioned together.
- **No content mask on any primitive.** `overflow_hidden()` has no effect. This
  matters more now that real content exists to clip.

### Carried forward, unchanged

- **6.4** — `PathPipeline` + `BackdropFilterPipeline`. And, for the fifth time
  of asking: **do not trust `paths.wgsl`/`backdrop_blur.wgsl`'s "moved as-is"
  comments.** That claim has now been false or misleading four times running
  (`mono_sprites`, `shadows`, `underlines`, and — differently — `quads`).
- **6.5** — the animation driver. `widgets/animation.rs` and
  `window/animation.rs` are still 3-line Phase 0 stubs. Also closes GIF frame
  advancement (Phase 6.2's decode-without-animate gap).
- **6.1's fate is still undecided.** The fused-dispatch follow-up spike has not
  been run. A performance question, not a parity blocker.
- **The cross-kind occlusion gap.** Occlusion dispatches per primitive kind, so
  a `Quad` occluder cannot cull a `Shadow` or a `PolySprite`. Phase 6.3 found
  this by falsifying its own claim; nothing here changes it, and this phase
  makes it *more* reachable, since real `div()` trees now produce mixed-kind
  scenes where a real occluder sits over real shadows.
- **§6.2's `estimated_size` half.** Still not closed. No trait hook exists
  anywhere in `2.0`; `wgpui-layout::containment.rs` is still a stub. `Div` got a
  `diff_key` and no `estimated_size`, exactly as `Img` and `StyledText` did.
- **Interactive states, hit-testing, event binding, scroll containers.**
  Explicitly deferred (§6). This is the largest single block of `div.rs`'s
  legacy line count still unbuilt — roughly 1,600 of its 4,528 lines, across
  `InteractiveElement` (~456) and the state/hitbox/dispatch part of
  `Interactivity` (~1,140).
- **Devtools extraction (7)**, the legacy alias crate and cutover (8).
- **Breadth.** One machine, one driver, one backend, one OS, for every number in
  this document and every prior one.

---

## 9. Honest read

The gate is met and the evidence is strong: the oracle is the frozen legacy
shader source compiled and run, the layout arithmetic is independent of the
implementation it checks, and every gate has been watched rejecting a
perturbation aimed at the specific thing it exists to prove.

Three things are worth saying plainly about what that does and does not mean.

**The phase's largest piece of work was not in its brief.** §8's row is about
emission, and emission turned out to be the easy half. The quad shader — which
four prior phases had rendered through without ever comparing it to legacy —
could not have passed this gate under any emission logic, and finding that out
required checking a file whose comment said it was deliberate rather than
provisional. The comment was honest about *why* it was hard-edged; nobody had
revisited whether the reason still held.

**"Byte-exact against the legacy renderer" is a narrower claim than it sounds,
and §4 is where the narrowing lives.** Gradients, dashed borders, per-fragment
clipping, filters, element opacity and non-unit scale factors are all real parts
of what `Style::paint` does and none of them are compared here, because none of
them exist in 2.0 to compare. A `div()` that uses only the properties 2.0 can
express renders exactly. A `div()` that uses `bg(linear_gradient(…))` renders
nothing at all.

**Example parity moved, but it is worth being precise about how far.** Before
this phase, `div()` produced no GPU primitives beyond Phase 2's proof-of-concept
and five working pipelines were reachable by almost nothing. After it, a styled,
nested, shadowed `div()` tree renders correctly through the real pipeline. That
is the largest single step toward §7 that any phase has taken. It is also not
the same as "examples run": no example in the repository compiles against
`wgpui-widgets` today, because the surface they compile against is `gpui-ce`'s,
and the adapter that presents it is Phase 8. What this phase changed is that
Phase 8 now has something real to adapt *to*.
