# Scroll-free-by-default

Status: **Pass 0 and Pass A shipped and verified; Pass B deliberately not
started.** Summary for anyone catching up:

| Pass | Status | Evidence |
|---|---|---|
| 0 — fix what's measured broken | **Shipped** | Idle "present only" loop: 15,000–33,000/s → ~220/s, measured on the same repro (`platform/cross/platform.rs`'s `about_to_wait`). Same fix collapsed 11 full-tree relayouts across two resize gestures down to 3, as a side effect of coalescing (`invalidator.set_dirty` is idempotent; the bottleneck is now the ~1.3s/relayout cost itself, not pass count — see §0.-1.3). Example fixed to build without `--features test-support`. Debug-time warning added for buffered plain divs over 500 children (the footgun §0.-1.3 describes). |
| A — automatic layering | **Shipped, narrower than originally scoped** | `.track_scroll(..)` now promotes to a plain, unkeyed `.layer()` automatically (`WGPUI_AUTO_LAYERS=0` to revert), verified by two new tests. **The originally-planned positional-`LayerKey`-for-id-less-roots mechanism turned out to be unnecessary**: `.track_scroll`/`.overflow_*_scroll` are `StatefulInteractiveElement` methods, so a scroll container always already has `.id(..)` and therefore a valid `GlobalElementId` — there is no id-less case to solve for this specific use. **The auto-generated dependency key from §1.1 was not implemented** — see "What changed on implementation" below for why, and what would be needed to add it safely. |
| B — hover-in-buffer, glide-via-`request_animation_frame_for`, delete old cache | **Not started, on purpose** | All three require correctness-sensitive changes to shared invalidation/animation plumbing (`App::notify`, `dirty_views`, the layer-composite predicate) that need the kind of differential testing (`WGPUI_OCCLUSION=validate`-style) this session couldn't build and run. Old-cache deletion specifically was *checked and found unsafe*: `Layer::paint_range` still routes through `invalid_reuse_range`/the replay path today — deleting it now would break the current layer system, not just remove dead code. |

## What changed on implementation (read before extending Pass A)

§1.1 as originally written proposed a `notify_scroll()` "tagged notification"
that layers would recognize as transform-only without a caller-supplied key,
avoiding the correctness burden of `.layer_keyed(..)`. Implementing Pass A
surfaced why that's harder than it sounds, and it's worth recording precisely
so nobody re-derives it the hard way:

`with_retained_layer`'s actual rule (`window.rs`) is `view_rebuilt =
content_key.is_none() && dirty_views.contains(view)` — so *any* non-`None`
key, including a constant one, already suppresses "notified → must rebuild."
The key's real job is narrower than "prove nothing changed" — it's "opt out
of the coarse view-level rebuild rule," while a *separate*, already-existing
mechanism (`accessed_entity_invalidated`, tracking `Entity<T>` reads made
during the layer's last render) independently catches changes that go
through tracked entities. So a constant auto-key looked, briefly, like it
might already be safe.

It isn't, for the common case that actually matters: a view whose scrollable
content is a plain struct field (`self.rows: Vec<Row>`, exactly what
`plain_scroll_10k.rs` uses) rather than a tracked `Entity<T>`. Changing that
field and calling `cx.notify()` on the owning view is *indistinguishable*,
at the point `with_retained_layer` runs, from the scroll wheel's own
`cx.notify(current_view)` — both are "this view is in `dirty_views`." A
constant key would silently treat the first case as compositable-only,
which is precisely the silent-stale-UI failure this architecture exists to
prevent. Distinguishing them for real needs the scroll path to stop routing
through the generic `App::notify`/`dirty_views` mechanism at all — a change
to shared invalidation plumbing used by every view in the crate, not a
scroll-local one, and not something to land without the differential test
harness this session didn't have time to build.

**What shipped instead is the safe subset**: auto-promotion to a plain
`.layer()` (no key), which is exactly as safe as the existing "safe default"
`.layer()` already documented for manual use — it re-renders on every notify,
same as unlayered content, so it makes no dependency claim on the caller's
behalf. Real win (instance reconciliation, persistent layout, local
ordering for the subtree), zero new correctness risk. The texture-retained
overscroll buffer — the part that makes a scroll *tick* cost nothing — still
requires the caller's explicit `.layer_keyed(..)` + `.layer_with_policy(..)`,
exactly as before. If a genuinely automatic key is wanted later, it needs
`notify_scroll()` (or equivalent) built and tested as its own change against
`App::notify`, not bundled into this one.

---

Continues `docs/retained-layers.md` from its Phase 11
checkpoint — read that document first; this one does not repeat its
vocabulary (`Layer`, `Invalidation`, `LayerPolicy`, the overscroll-buffer
protocol) and cites it throughout as "R-N §M".

No public API is removed here. Every change is either (a) a default that
makes existing code faster with no call-site edits, (b) a diagnostic that
turns a silent performance cliff into a visible one, or (c) deletion of a
superseded internal mechanism that nothing public depends on.

---

## 0.-1 Observed, not inferred

Everything below this point was originally written from static analysis. It
undersold the problem. Running the actual demo the pending diff ships
(`examples/bench/plain_scroll_10k.rs`) and instrumenting the window with
Win32 calls while it ran produced direct, measured evidence that changes the
diagnosis:

- **The example does not compile as staged.** `render_stats::snapshot` /
  `set_force_enabled` are gated `#[cfg(any(test, feature = "test-support"))]`;
  the example calls them unconditionally. `cargo run --example
  plain_scroll_10k` fails with `E0425` until built with `--features
  test-support`. Small, but it means nobody has actually run this demo since
  it was written.
- **First paint took 21.17 seconds.** The window is real, responding, and
  titled "GPUI" for that entire span — and shows nothing, matching "I see
  nothing" exactly. `render_stats`' own log: `8` separate full `frame: layout`
  passes before the first present, mean `1484.5ms` each, `11876.01ms` total —
  each pass creates ~50,000 fresh Taffy nodes (`taffy nodes created: 400,319`
  over the 8 passes), i.e. **every one of the 10,000 rows, laid out again,
  eight times, before one pixel reaches the screen.** `instance: rebuilt
  (stale range)`: 149,751 of 199,751 rebuilds (75%) — the reuse/cache path is
  thrashing, not hitting.
- **A single native resize call blocked for 4.3 seconds and left the process
  in `Responding: False`.** Measured directly: `SetWindowPos` (a synchronous
  Win32 call) took 4332ms to return; `Get-Process` immediately after reported
  `Responding: False`; the OS-level window title changed to `GPUI (Not
  Responding)`. This is the literal Windows "not responding" ghost state, not
  a metaphor.
- **Two resize operations produced eleven full-tree relayouts.** The
  `render_stats` window spanning both resizes: `24.40s, 10849 frames`, `11`
  layout passes, mean `1134ms`, total `12475.89ms` — again ~50,000 Taffy nodes
  and ~50,000 "stale range" instance rebuilds *per pass*. Windows delivers a
  stream of `WM_SIZE`/`WM_WINDOWPOSCHANGING` messages during a single resize
  gesture (one per intermediate size, not one at the end), and nothing
  coalesces them — each raw message drives a full, unbounded layout of all
  10,000 real rows. This is the direct, measured mechanism behind "stupid
  laggy to move the window": it is not "slow," it is *eleven-times-repeated
  full-list layout*, synchronously, on the thread the OS is waiting on to
  answer the resize.
- **Idle costs a full CPU core, continuously.** With nothing changing —
  window not focused, no input — `render_stats` logs a `window: present only
  (no draw)` block **every single second**, at 15,000–33,000 "frames" in it.
  That's not vsync-paced idle; it's an unthrottled present loop, and the
  process's cumulative CPU time climbed from 102s to 180s across roughly 15
  wall-clock seconds of me doing nothing but taking screenshots — i.e. it was
  consuming more than one full core the entire time, competing with whatever
  thread the OS needs responsive during a drag.

**Revised diagnosis.** The `.layer()`-is-opt-in finding below (§0.1–0.3) is
still true and still worth fixing, but it is not what's making this demo
unusable. The actual mechanism, measured: **the overscroll-buffer protocol
was built for virtualized lists, where "lay out the buffer range" means a
few dozen synthesized rows. Applied to a plain, non-virtualized div (exactly
what this demo and the pending `div.rs` diff do), "lay out the buffer range"
still means laying out every real child in the div's `children` vector — all
10,000 of them — because a plain div has no concept of a range; the children
already exist as elements.** Every refill (first mount, every resize, every
scroll past the margin) pays the full 10,000-row cost, and Windows' resize
message stream turns *one* user drag into a dozen of them. The buffer makes
*shift* frames free exactly as designed; it was never able to make *refill*
frames cheap for non-virtualized content, and nothing before this document
noticed because the only prior test used 40 items.

This means the highest-priority fix is narrower and more concrete than "make
layering automatic," and comes before it:

1. **Debounce resize.** Coalesce the `WM_SIZE` stream to one layout per
   settled size (a short trailing-edge debounce, or driving relayout off the
   platform's own resize-end signal where one exists) — turns eleven full
   relayouts per drag into one. This alone should remove the "Not Responding"
   state, independent of anything else in this document.
2. **Fix the idle loop.** `window: present only (no draw)` should not be
   possible to observe at 20,000+/sec — an idle window should present at
   most once per vsync, and arguably not at all when nothing changed and
   nothing is animating. This is a straightforward, high-value, low-risk fix
   on its own, separate from the scroll story.
3. **Either bound refill cost for non-virtualized content, or stop offering
   the overscroll-buffer combination for it.** A plain div wrapped in
   `.layer_keyed` + `.layer_with_policy { overdraw_margin }` cannot cheaply
   "lay out the buffer range" the way a virtualized list can, structurally —
   its children are not a range, they're all there. Two honest options, not
   a hybrid: (a) teach the buffered-div path to virtualize itself — synthesize
   only rows inside the buffer range and diff against the full data, i.e.
   quietly become `uniform_list` under the hood when wrapped this way; or (b)
   stop presenting `.layer_keyed` + `overdraw_margin` as a solution for plain
   divs with large real child counts, and let the existing virtualized list
   types own that case exclusively, with the plain-div integration (the
   pending `div.rs` diff) scoped explicitly to *small* real child counts where
   "lay out everything" was always going to be cheap. (a) is more work and
   strictly better; (b) is honest and immediate. Recommend shipping (b) now —
   correct the example and the div.rs doc comment to say so explicitly — and
   treat (a) as a real, separate, later feature (an internal virtualization
   fallback for buffered plain divs) rather than something this document's
   Pass A should silently assume works today.

Everything in §§1–6 below is still correct and still worth doing — automatic
layering, positional identity, killing the old cache, the dead-code sweep —
but none of it fixes what's actually broken in this demo. §7 folds the three
items above into the phasing as the work that has to land first.

## §0.-2 The blank window is a separate bug, isolated but not yet root-caused

Fixing §0.-1's idle-loop and resize-coalescing issues did **not** fix the
demo's blank-window symptom. Full isolation trail, so nobody re-walks it:

1. **Not the environment.** `hello_world` renders correctly (text, color
   swatches, all visible) in the same sandbox that shows `plain_scroll_10k`
   as a black window with one stray white rectangle.
2. **Not layers/the overscroll buffer.** A minimal repro with the identical
   `.layer_keyed(..)` + `.layer_with_policy(..)` + `.track_scroll(..)` shape
   renders correctly through the real GPU renderer at 40, 2,000, 5,000, and
   50,000 rows — including with per-row structure (two text runs, random
   symbol/price content) matching `plain_scroll_10k.rs` closely. Row count
   and content variety are both ruled out.
3. **Not my Pass 0/A changes.** `git stash` back to the untouched original
   code reproduces the identical failure, byte-for-byte.
4. **Not `render_stats` — recording or reading, in any combination, checked
   exhaustively:**
   - HUD calls `render_stats::snapshot()` synchronously every `render()`,
     `set_force_enabled(true)` on → broken (the original).
   - Same, but the read throttled to once per 200ms → still broken (rules
     out call *frequency*).
   - `set_force_enabled(true)` on, but the read moved onto a background
     timer so `render()` never touches `render_stats` at all → still broken
     (rules out *where* the read happens).
   - Read on the background timer, `set_force_enabled` **not called at all**
     → still broken (rules out the instrumentation-recording path too —
     every `count()`/`scope()` call site in the whole draw pipeline).
5. **The only variant that renders correctly**: building the example
   *without* `--features test-support` at all — which, because
   `render_stats::snapshot`/`set_force_enabled` are gated
   `#[cfg(any(test, feature = "test-support"))]`, is also the only variant
   where none of that code is even compiled in.

Since every *runtime* use of `render_stats` was individually ruled out (§4
above), the remaining candidate is something the `test-support` **Cargo
feature** enables at compile time, independent of anything this demo's own
code does. The prime suspect: `test-support = ["leak-detection", ...]`, and
`leak-detection = ["backtrace"]` — `EntityMap` captures a full
`backtrace::Backtrace` per entity handle under that feature
(`app/entity_map.rs`). Backtrace capture is a known severe-cost operation
(stack unwinding + symbol resolution); this demo only creates one `Entity`
itself, but if any framework-internal path creates entities/handles at a
rate that scales with element count during the 300,000+-node startup burst,
that would explain a failure that is specific to both the feature flag and
to scale, matching everything observed. **Not yet confirmed** — this is
where the next session should start, with `--features leak-detection` in
isolation (would need `StatsTotals::read`'s cfg gate loosened to compile) or
by instrumenting `EntityMap`'s handle-creation path directly.

Practical upshot in the meantime: `plain_scroll_10k.rs` now samples
`render_stats` off the `render()` call path on a background timer (a real
improvement, kept regardless), and the demo works and renders correctly for
its default, no-flags `cargo run` — which is what almost every actual
consumer of this crate will ever build, since `test-support` is a
test/dev-only feature no application ships with. The failure is real,
confirmed pre-existing (independent of anything in this document's Pass
0/A), and now precisely bounded — but not yet fixed at the root.

## §0.-3 Layout containment: the actual fix for "arbitrary scrollable content"

§0.-1.3 originally recommended scoping buffered plain divs down to small
child counts and steering large lists to `uniform_list`. That's the wrong
answer for genuinely arbitrary scrollable content — heterogeneous children
whose position can't be computed by `index × row height` the way a uniform
list's can. **Shipped instead**: layout containment, the direct analog of CSS
`content-visibility: auto` + `contain-intrinsic-size` — the same technique
browsers use for exactly this case, per the brief this section replaces.

- `Element::estimated_size(&self, window: &Window) -> Option<Size<Pixels>>`
  (`element.rs`): a cheap, static size available without running
  `request_layout` at all. Default `None` (opts out of containment — the
  safe, "measure for real" default, following the same migration pattern as
  `diff_key`/`on_frame`). `Div` implements it: `Some` only when both axes are
  an absolute length (`px`/`rem`, and `%` for width only, resolved against
  the viewport as a stand-in for "the parent" — width has no bearing on which
  children are in range, so it can be approximate; height must be exact or
  revealing a child snaps to its real size, the same layout shift a wrong
  `contain-intrinsic-size` produces in a browser). `Auto`/content-dependent
  height stays `None` — never guessed.
- `scroll_buffer::plan_child_containment` (`elements/scroll_buffer.rs`):
  given each child's `estimated_size` in order, accumulates position by
  summing declared heights and marks each child `Contained(size)` or `Real`
  against the buffer's visible+margin window. The first child with no
  `estimated_size` breaks the accumulation — real position for it and
  everything after it isn't knowable without laying it out, so the whole
  remaining suffix falls back to `Real`, exactly what it would have gotten
  anyway. Disclosed limitation, not a correctness gap: `Real` is always safe.
- `Div::request_layout`/`prepaint`/`paint` (`elements/div.rs`): a `Contained`
  child gets a placeholder Taffy leaf (its `estimated_size`, no recursion)
  instead of `child.request_layout`, and is skipped entirely in
  `prepaint`/`paint` — no instance reconciliation, no style resolution, no
  hitbox, no primitives. `ChildReconciliation::Contained` threads the
  decision from `prepaint` to `paint` the same way `Reused`/`Rebuilt` already
  do.

**Measured effect on the actual repro**: startup went from 8 full-tree
relayouts (~18s total, ~2s each) to 1 (~5s) — the unavoidable cold-start pass,
before any prior-frame viewport measurement exists to contain against. Every
invalidation after that resolves without a second full pass. Verified two
ways: the render_stats trace on the real 10,000-row demo, and a new
deterministic test,
`contained_children_outside_the_buffer_window_are_never_painted_and_visible_ones_always_are`
(`window.rs`), which tracks per-row paint counts directly and asserts a row
199 rows from the top is never painted while scrolled to the top, and stays
unpainted when scrolling reveals a *different* distant region — not just "it
looks right," but "the skipped code path was actually skipped." 298/298 tests
pass, zero regressions, including every existing shift/refill/hover/occlusion
test.

**What's still open**: `estimated_size` only covers static, declared sizes.
A child with auto/content-dependent height breaks containment for itself and
every later sibling, exactly like a browser without `contain-intrinsic-size`
falls back to real layout. A natural follow-up — not implemented here — is
falling back to a child's *last known real bounds* (already tracked by
`ElementInstance.bounds`) when `estimated_size` returns `None`, which would
recover most of that suffix once at least one real layout has happened, the
same way a browser's `content-visibility: auto` behaves for
previously-rendered content.

---

## 0. Where we actually are

This is not a request to design retained rendering from scratch — that
architecture exists, is merged, and is on by default. Checked directly against
the code, not the plan:

| R-N phase | Claims to be | Actually is |
|---|---|---|
| 4 (layers) | shipped | `WGPUI_LAYERS` defaults on ([layer.rs:429](../src/layer.rs)) |
| 6 (layer-tier occlusion) | shipped | `occlusion.rs` present, default enabled |
| 7 (instances/`diff_key`) | shipped, `Div`/text/`Svg` only | `WGPUI_INSTANCES` defaults on ([instance.rs:178](../src/instance.rs)); `Img`/`StyledText` explicitly not migrated |
| 8 (persistent Taffy) | shipped | `WGPUI_PERSISTENT_LAYOUT` defaults on ([taffy.rs:66](../src/taffy.rs)) |
| 9 (per-layer slabs) | shipped | referenced live in `perf_ab_tests.rs`'s `WGPUI_SLABS` knob |
| 11 (texture-retained + overscroll buffers) | shipped | `WGPUI_LAYERS_RASTERIZE` defaults on; `scroll_buffer.rs` is real and tested |
| 12 (delete old cache) | **not started** | `AnyView::cached`'s replay path, `invalid_reuse_range`, `PrepaintStateIndex`/`PaintIndex`, `Scene::replay` are all still live and still load-bearing — `Layer::paint_range` itself routes through the old replay path today ([layer.rs:34-36](../src/layer.rs)) |

So the mechanism the user wants ("scroll essentially free") **already exists,
is fast, and is tested** — `scroll_buffer.rs`'s protocol turns a scrolled
frame into one shifted texture composite, zero layout, zero paint, between
refills. The pending diff (`div.rs`, `scroll_buffer.rs`,
`examples/bench/plain_scroll_10k.rs`) is a fourth, correct integration of that
exact mechanism, generalizing it from virtualized lists to a plain
`.overflow_scroll()` div.

That the fourth integration required touching three files to hand-wire one
more call site is the actual finding. Below is why, with numbers, and what to
do about it.

### 0.1 The mechanism is real. Almost nothing in the app uses it.

Reaching the fast path requires all three of, together, on every scrollable
div individually:

```rust
div()
    .id("scroller")                          // 1: required, silently — see §0.2
    .layer_keyed(content_key)                // 2: a key YOU derive and keep correct
    .layer_with_policy(LayerPolicy {          // 3: a margin YOU tune
        overdraw_margin: size(px(0.), px(160.)),
        ..Default::default()
    })
    .overflow_y_scroll()
    .track_scroll(&handle)
```

Grepping the actual application (Pulsar-Native, not the wgpui crate itself):

```
37   call sites using overflow_{x,y}_scroll / track_scroll
 1   of those also uses .layer_keyed(...)   (the level-editor viewport)
 4   call sites using uniform_list (buffered internally, no user action needed)
```

**32 of 37 hand-rolled scroll containers in the actual product get none of
phases 4–11.** Their wheel handler calls `cx.notify(current_view)`
unconditionally ([div.rs:3007](../src/elements/div.rs),
[3382](../src/elements/div.rs), [3434](../src/elements/div.rs)); without a
keyed layer a notify is just "rebuild," so every wheel tick re-runs `render`,
full layout, full prepaint, full paint of that view's subtree. This is very
likely the "insanely inefficient" scrolling the user is observing —
not a hole in the architecture, a hole in its defaults. The fast path is
invisible at the one call site (`overflow_y_scroll` / `track_scroll`) where a
developer would reach for it, and requires knowing about a mechanism
(`.layer_keyed`) documented in a different module for a different purpose
(caching arbitrary subtrees, not scrolling specifically).

### 0.2 The opt-in fails silently

`layer_key` is only computed when the div has an id at all:

```rust
// div.rs:1706
let layer_key = self.interactivity.layer.as_ref()
    .zip(global_id)
    .map(|(_, global_id)| LayerKey::from_global_element_id(global_id));
```

`.layer()` / `.layer_keyed()` on a div without `.id(...)` compiles, runs, and
does **nothing** — no layer is ever registered, no warning, no debug
assertion. A developer who copies the pattern above but forgets `.id(...)`
gets exactly today's full-rebuild behavior with no signal anything is wrong.
This is the same failure class R-N §0.2 identifies in the *old* cache
(silent-wrong is the failure mode) reappearing in the new one, at a much
smaller scale — worth closing before generalizing the mechanism further,
since generalizing it multiplies the number of places it can silently no-op.

### 0.3 Three self-documented v1 gaps

R-N's own phase-11 row lists what shipped short of complete:

> buffered elements must sit under `.layer_keyed` (a plain `.layer()`
> re-records on the scroll notify), hover styling inside a buffer goes stale
> between refills, and the glide still re-runs the view's `render`

These matter more once §1 below makes buffering the default outcome of
`track_scroll` rather than an opt-in three-piece incantation — generalizing a
mechanism to 32 more call sites multiplies whatever is still broken in it by
32.

---

## 1. The change with the highest leverage: default it

**Make `.track_scroll(&handle)` on an `Overflow::Scroll` div buffer by
default.** No new public call required for the common case; `.layer_keyed` /
`.layer_with_policy` keep working unchanged for callers who want to hand-tune
a margin or fold scrolling into a wider caching key.

### 1.0 "Automatic" is a real, scoped mechanism change, not a default flip

This is worth being precise about, because the framework already has *half*
of the "browsers decide automatically" model and it's easy to conflate the
two halves:

- **Rasterize-or-not is already automatic and already complexity-based.**
  `LayerPolicy::rasterize_above` (default 256 primitives,
  [layer.rs:137](../src/layer.rs)) decides, for a layer that exists, whether
  it's cheaper to keep re-emitting primitives or to composite through a
  texture. This is exactly the "decide based on complexity" half of the ask,
  and it already runs unconditionally once a layer exists. Nothing to build
  here.
- **Layer creation itself is 100% manual, and not for a stylistic reason —
  today's identity model can't do better without a change.** A `LayerKey` is
  derived from a `GlobalElementId`
  ([layer.rs:76](../src/layer.rs)), and `GlobalElementId` is only ever
  constructed when an element's own `.id()` returns `Some`
  ([element.rs:441](../src/element.rs): `self.element.id().map(|element_id|
  ...)`). A bare `div()` — no `.id(...)` — has no `GlobalElementId` at any
  point in the stack, full stop. That's not a missing default; it's a missing
  *identity source* for the common case (most divs are anonymous).

  The fix already exists in the codebase, one level too shallow.
  `instance_id_stack` ([window.rs:1704](../src/window.rs)) solves exactly
  this problem already — for a layer's *children*: each gets a real
  `ElementId` if it has one, or a synthetic positional
  `ElementId::InstanceSlot(n)` if it doesn't
  ([instance.rs:63](../src/instance.rs)), which is what lets `InstanceKey`
  address plain, anonymous `div()`s at all. It's deliberately scoped to
  "pushed only around a `.layer()` subtree's children"
  ([window.rs:1694-1699](../src/window.rs)) — it only starts once a layer's
  own identity is already resolved through the ordinary, `.id()`-gated path.

  **Extend that same positional-fallback pattern one level up, to the layer
  root itself.** A div deciding whether to auto-promote derives its
  `LayerKey` from its position in its parent's child list (an
  `ElementId::InstanceSlot(child_index)`) when it has no explicit `.id()`,
  the same way its own children already do one level down. This is not a new
  concept — it's applying #92's own mechanism to the one place it stopped
  short. Once this lands, "requires `.id(...)`" simply stops being true, and
  §0.2's silent-no-op class of bug is closed by construction rather than by
  an assertion.

With that piece in place, the actual automatic-promotion signal is narrower
and more correct than raw size: browsers don't promote a layer because a
subtree is *big*, they promote it because a subtree changes for a reason
**independent of its surroundings** — scroll offset, a running animation, a
hovered/pressed state. R-N itself already says this (§5.1: "layer boundaries
should separate content by update frequency, not visual grouping"). So the
auto-classifier's trigger is:

- **is a scroll container** (`Overflow::Scroll` + `track_scroll` present) —
  unambiguous, and the case this document is about;
- reserved, not built yet: **has hover/active/animation-driven style
  resolution** — the other kind of "notified independent of ancestors."
  Flagged here because the classifier should be built to take this signal
  later without a second identity mechanism, not built here — promoting
  animated content needs its own validation pass (a
  `WGPUI_OCCLUSION=validate`-style differential check, per R-N §8.5's
  precedent) before it's safe to default on.

Complexity (`rasterize_above`) still decides, downstream and unchanged,
whether a promoted layer bothers with a texture. Two separate questions,
answered by two separate mechanisms, exactly as they are today — the only
thing moving is who's allowed to ask the first one.

### 1.1 Auto key: a tagged scroll-notify, not a hand-written key

The correctness burden for `.layer_keyed` today is: the key must cover
"everything this subtree's content depends on, **except** the scroll offset"
— get that wrong and the failure is stale UI, silently. That burden is
avoidable, not irreducible: the reason a key is needed at all is that
`cx.notify()` is one undifferentiated signal, so a layer can't tell "notified
because scroll moved" from "notified because the data changed." Fix the
signal instead of asking every call site to describe its own data
dependencies:

```rust
/// Notify this view because its scroll offset changed. Layers recognize this
/// as TRANSFORM-only and composite instead of re-recording, with no key
/// required. A regular cx.notify() from the same view still invalidates
/// normally — the two are not the same signal and must not be conflated.
pub fn notify_scroll(&self, view: EntityId) { .. }
```

`ScrollHandle`'s own wheel/drag path (the three `cx.notify(current_view)`
call sites in §0.1) switches to this. A `track_scroll`ed div with
`Overflow::Scroll` gets an implicit `LayerPolicy` (margin sized off the
viewport, per R-N §7's own suggestion: `overdraw_margin: viewport * 0.5` on
the scrolling axis) and an implicit key that does not need to name the
content at all, because the signal it's keyed against already says "this is
just scroll."

This is strictly narrower than the general `.layer_keyed` mechanism — it only
ever fires for the one notification kind that is provably transform-only by
construction — which is what makes it safe to turn on unconditionally instead
of asking for an opt-in.

Explicit `.layer_keyed(...)` on a scroll container still works and still
means what it means today (for the case where scrolling *and* some other
notified change should share one layer); the default only changes what
happens when nobody writes it.

### 1.2 Cheap content stays cheap content

None of this overrides `rasterize_above` (R-N §3.3): a scroll container whose
visible + margin content is under the primitive threshold stays
primitive-retained (no texture) exactly as any other layer would. The auto
policy is "use the same mechanism every other layer already uses, tuned for
scroll," not a new code path.

### 1.3 Payoff

Zero call-site changes across all 32 currently-unbuffered scroll containers in
the app. This is the actual "doing less" instance of the whole overhaul: the
framework does the work of recognizing a scroll notification instead of every
screen author learning a three-piece API.

---

## 2. Close the silent-failure gaps before generalizing further

- **Diagnose id-less layers.** `debug_assert!`/a one-time `log::warn!` when
  `Interactivity::layer.is_some()` and `global_id` is `None` at prepaint
  (div.rs:1706's `.zip`) — "`.layer()` has no effect without `.id(..)`." Same
  spirit as `WGPUI_LAYER_DEBUG=1` (R-N §9 risk table): a secretly-inert
  optimization should be loud, not silent.
- **Hover inside a buffer.** Don't attempt general hover-in-texture in v1;
  formalize the pattern `plain_scroll_10k.rs` already uses ad hoc for its
  scrollbar thumb — "content resolved from `:hover`/`:active` state lives on
  an unbuffered overlay layer painted after the composite" — as a documented,
  named pattern (`.layer_overlay()`?) rather than a one-off in an example.
- **Glide re-runs `render`.** Wire smooth-scroll's frame driver through the
  planned `request_animation_frame_for(layer, TRANSFORM)` (R-N §6, named but
  not yet built) instead of `window.refresh()` — this is R-N Phase 1's
  pattern ("four dead `refresh()` → `request_animation_frame()`") applied to
  the fifth one that phase didn't cover because it didn't exist yet.

These three land *before* §1's default flip, on the existing opt-in call
sites (`virtual_list`, `uniform_list`, `h_list`, the level-editor viewport),
so §1 doesn't broadcast known gaps to 32 more places at once.

---

## 3. What determines refill cost: `Img` and `StyledText`

The buffer protocol makes *shift* frames free; it does not change what a
*refill* frame costs, and refill cost is what determines whether a fast fling
ever drops a frame. R-N Phase 7/8 explicitly left `Img` and `StyledText`
without a `diff_key` — every other element type got cheaper reconciliation,
these two didn't. Scroll content is disproportionately avatars, thumbnails,
and rich (multi-run) text — exactly the content dominating real list rows.
Extending `diff_key` to both is orthogonal to §1 and can land in parallel;
it's listed here because it's the other half of "scroll feels free," not
because it blocks it.

---

## 4. Delete the superseded cache (R-N Phase 12)

`AnyView::cached`'s replay path, `invalid_reuse_range`
([window.rs:3860](../src/window.rs)), `PrepaintStateIndex`/`PaintIndex`
ranges, `Scene::replay` ([scene.rs:581](../src/scene.rs)) are all still
present. They're not dead code — `Layer::paint_range` itself still routes
through the old replay mechanism (`layer.rs:34-36`'s own doc: "stays until
#97") — but everything that made this necessary (anonymous, offset-addressed,
age-sensitive caching) has a named, address-stable successor that has been
merged and default-on for multiple phases now. This is the literal "less
code" half of the ask: one caching mechanism instead of two, and removal of
the exact failure class (a range that ages by one frame slices out of bounds
and aborts the process — R-N §0.2) the new architecture exists to retire.
Do this last, once nothing (including `Layer::paint_range`) still depends on
it — same dependency order R-N's own Phase 12 already states.

---

## 5. Mechanical cleanup: one decision point, not four

`scroll_buffer::prepare_scroll_buffer` returns a 3-armed enum
(`Skip`/`Buffer`/`Viewport`) that `virtual_list.rs`, `uniform_list.rs`,
`h_list.rs`, and (per the pending diff) `div.rs` each match on independently,
repeating the same three-arm shape with slightly different bodies per arm.
Collapse to one combinator:

```rust
/// Runs `layout_range` only when the buffer needs content laid out
/// (`Buffer`/`Viewport`); returns immediately on `Skip`. Callers stop
/// hand-matching the three arms individually.
pub(crate) fn with_scroll_buffer(
    window: &mut Window,
    scroll: Point<Pixels>,
    layout_range: impl FnOnce(&mut Window, ScrollBufferFrame) -> R,
) -> Option<R>
```

Small, mechanical, zero behavior change — folded into whichever phase below
touches those four files next, so it isn't its own release.

---

## 6. Phasing: one coordinated pass, not a flag ladder

The straight-line reading of §§1–5 is six sequential, individually-flagged
phases. That's the wrong shape for this specific change, and worth saying
why: phases 12a/13/16 below all touch the *same* code (`div.rs`'s prepaint,
`layer.rs`'s key derivation, the four `prepare_scroll_buffer` call sites) for
related reasons. Sequencing them weeks apart means maintaining the old
manual-`.id()` path, the new positional-identity path, *and* the pre-#96
replay cache all live at once for longer than necessary — which is exactly
the "convoluted, two-systems-at-once" state the ask was to get out of, not
prolong. Where the work is genuinely the same diff, do it in the same pass;
gate on correctness (tests), not on artificial phase boundaries.

Three passes. Pass 0 didn't exist in the original version of this document —
it's what §0.-1's measurements demand, and it comes first because it's the
difference between the demo being usable at all and not.

**Pass 0 — fix what's actually measured broken.**
- Debounce/coalesce resize to one relayout per settled size (§0.-1.1).
- Cap the idle present loop — no unthrottled `window: present only` spin
  (§0.-1.2).
- Fix the example's build (`test-support` feature) so this is checkable by
  anyone, not just someone who happens to pass the right flag.
- Decide and document, explicitly, whether a buffered plain div virtualizes
  its refill internally or is scoped to small child counts (§0.-1.3, option
  (b) recommended for now) — and correct `plain_scroll_10k.rs` and
  `scroll_buffer.rs`'s doc comments to match reality rather than aspiration.
- **Gate:** the exact repro above — build, run, resize twice — no longer
  produces a `Responding: False` state or a multi-second `SetWindowPos`
  block; `render_stats` shows one layout pass per settled resize, not eleven.

**Pass A — make it automatic.**
- Positional `LayerKey` derivation for a layer root with no explicit `.id()`
  (§1.0) — the actual unlock.
- Scroll-container auto-promotion (§1.1) built directly on top of it, since
  there's no reason to land positional identity and then separately wire the
  one consumer it exists for.
- Collapse the four hand-matched `prepare_scroll_buffer` call sites (§5) into
  one combinator in the same diff — touching all four to remove the manual
  `.layer_keyed`/`.layer_with_policy` requirement (§1.1) already means
  editing every one of them; do the mechanical cleanup there rather than as a
  separate pass over the same lines later.
- `notify_scroll` (§1.1) replacing the three unconditional
  `cx.notify(current_view)` scroll call sites.
- Single `WGPUI_AUTO_LAYERS=0` kill switch, not one flag per sub-piece.
- **Gate:** `plain_scroll_10k`-style `perf_ab_tests` case using a *plain*
  `overflow_y_scroll` div with **no** `.layer_keyed`/`.layer_with_policy`
  anywhere in it — red before this pass, green after, with the zero-call-site
  property (§1's payoff) asserted directly: the test app is byte-for-byte
  what a real screen looks like today.

**Pass B — retire what Pass A makes provably unreachable.**
- Once scroll containers auto-promote, survey `Layer::paint_range`'s
  remaining dependency on the old replay path (`layer.rs:34-36`) against
  what's actually still reachable — Pass A likely shrinks that surface
  directly, since a plain scroll container is the single largest source of
  "content under no layer at all" in a typical screen. Delete
  `AnyView::cached`'s replay path, `invalid_reuse_range`,
  `PrepaintStateIndex`/`PaintIndex` ranges, `Scene::replay` (§4) as soon as
  the survey confirms nothing depends on them — not deferred to a symbolic
  "last phase," but not forced before it's actually true either.
- id-less-layer diagnostic (§2) becomes moot for the *auto* path (there's no
  id to forget) but still applies to explicit `.layer()`/`.layer_keyed()`
  call sites, which remain `.id()`-gated by choice, same as today.
- Hover-overlay pattern + glide via `request_animation_frame_for` (§2's other
  two items) — needed now, not deferred, because Pass A is what turns these
  from "known limits of one opt-in call site" into "known limits of every
  scroll container in the app."
- `diff_key` for `Img`/`StyledText` (§3) — independent of A/B, lands whenever
  convenient; listed here because it's the same kind of "close a gap the
  existing phases left" work.
- **Gate:** no known workload needs `WGPUI_LAYERS=0`/`WGPUI_INSTANCES=0` as
  anything but a historical kill switch; `WGPUI_OCCLUSION=validate`-style
  differential run (culled vs a full non-auto-promoted render) over a
  scripted UI walk shows no divergence before the old path is actually
  deleted.

Reserved, explicitly not in this pass: hover/animation-driven auto-promotion
(§1.0's second bullet). It needs its own validation story before it's safe to
default on, and nothing in Pass A or B is designed in a way that blocks
adding it later.
