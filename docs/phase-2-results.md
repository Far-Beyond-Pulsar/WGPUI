# Phase 2 Results — `.boundary()`, Positional Boundary Identity, and the Emission Arrow

Status: **Phase 2 executed.** This documents what was built, what each of the
three gates actually asserts, what passed, and what a human should treat as
still open before calling this gate closed. It follows
`docs/gpu-native-architecture.md` ("2.0" below) §4.1, §4.3, §5.4, §5.5, and
§8's Phase 2 row. Nothing under `src/` changed — the legacy backend is frozen
(§8's own constraint) and `gpui-ce`'s warning count is unchanged from the
Phase 0/1 baseline. Work lives on branch `wgpui-2.0/phase-2-boundary`; most of
it is already merged to `2.0` via PRs #152 and #153.

**Contents:** §1 What shipped and where · §2 The `FramePlan` → `ScenePatch`
emission design · §3 Deviations from §3.1's file map, and why · §4 The three
gates · §5 Check, test, and clippy status · §6 What was found broken, and
fixed · §7 Gate assessment — honest read

---

## 1. What shipped, and where

Phase 2 adds one new module tree (`boundary/compositor.rs`), one new module
(`patch/emit.rs`), a `LayerTransform` on the retained layer, the description
and plan fields the two need, and the first real element shape in
`wgpui-widgets`. Still no `wgpu` dependency anywhere in `wgpui-core`,
`wgpui-layout`, or `wgpui-widgets`; still nothing wired into `src/`.

| File | Lines | Role |
|---|---|---|
| `wgpui-core/src/patch/emit.rs` | 1,495 | `Emit`, `Emission`, `EmitContext`, `Emitter`, `FrameEmission`, `EmissionStats` — the `FramePlan` → `ScenePatch` walk. Gates #1 and #2 |
| `wgpui-core/src/boundary/compositor.rs` | 445 | `Composite`, `BoundaryState`, `BoundaryComposite`, `Compositor` — the per-frame compositing decision |
| `wgpui-core/src/boundary/policy.rs` | 263 | `BoundaryPolicy`, `Buffering`, `Retention`, `Pixels`, `Size` |
| `wgpui-core/src/boundary/identity.rs` | 140 | `BoundaryIdentity::from_path` — SFD §1.0's positional fallback, one level up |
| `wgpui-core/src/invalidation/request.rs` | 255 | `FrameSignals` added: what a *frame* means for a given layer |
| `wgpui-core/src/scene/layer.rs` | 461 | `LayerTransform` added, plus `LayerTable::set_transform` |
| `wgpui-core/src/reconcile/description.rs` | 345 | `.boundary()`, `.boundary_with_policy()`, `.scroll_offset()`, `.emit()` |
| `wgpui-core/src/reconcile/plan.rs` | 445 | `PlannedNode` gains `address`/`boundary`/`declared_boundary`/`boundary_policy`/`scroll_offset`; `FramePlan` gains the emitter table |
| `wgpui-core/src/reconcile/reconciler.rs` | 877 | The walk carries the enclosing boundary and records it — and reads it nowhere |
| `wgpui-widgets/src/wgpu_surface.rs` | 507 | `WgpuSurface`, `WgpuSurfaceKey`, `SurfaceId`, `SurfaceStyle` — §5.5 Gap 1. Gate #3 |
| `wgpui-layout/src/taffy_tree.rs` | 394 | Re-exports (`Dimension`, `FlexDirection`, `LayoutSize`, `AvailableSpace`) and `definite()` so callers need not name `taffy` |

Two dependency edges were added, both one-way: `wgpui-widgets → wgpui-core`
and `wgpui-widgets → wgpui-layout`. Neither core crate names anything in
`wgpui-widgets`. Combined with Phase 1's `wgpui-core → wgpui-layout`, the
workspace graph is still a DAG pointing the direction §3 draws.

One cleanup unrelated to the gates: `wgpui-widgets/src/list/list.rs` was folded
into `list.rs`. §3.4's file map draws `list/mod.rs` plus `list/list.rs`, and
`AGENTS.md` forbids `mod.rs` paths, so the literal translation left a
`list::list` submodule named after its own parent. One file, no `mod.rs`, same
content.

---

## 2. The `FramePlan` → `ScenePatch` emission design

`docs/phase-1-results.md` §6 named this as the largest genuinely open thing at
the end of Phase 1:

> the arrow from a `FramePlan` to a `ScenePatch` does not exist, because
> deciding *which primitives an element emits* requires an element vocabulary
> that §3.4 puts in `wgpui-widgets`.

Phase 2 could not avoid closing it. `.boundary()`'s entire observable claim is
"a scroll tick uploads nothing," and there is no way to check that against a
crate where nothing has ever uploaded anything. So `patch/emit.rs` exists, and
it is deliberately the *smallest* thing that closes the arrow without inventing
the element vocabulary.

**The seam is a trait an element supplies, not a set of element types the
framework knows.** `Emit` has one method: given an `EmitContext` (resolved
absolute bounds, layer, boundary), write primitives into a reused `Emission`
buffer. A blanket impl means a closure is a valid emitter. `wgpui-core` never
learns what a `div` or an `Img` is; §3.4's vocabulary stays entirely ahead of
it, and when it arrives it implements one method rather than being wired in.

Three shape decisions are worth stating, because each is a real trade:

1. **A trait, not `Box<dyn Fn>`.** A real element emits from state it already
   holds. A closure type would force a capture-by-move closure to be
   constructed per element per frame — the per-frame allocation §4.2 objects to
   elsewhere. The emitter also writes into a buffer the walk owns and reuses, so
   a thousand-element frame allocates once rather than a thousand times.
2. **The emitter table lives on `FramePlan`.** The reconciler consumes the
   `Description`, and the emit walk runs *after* layout has been computed, so an
   element's emitter has to survive the gap between the two. Attaching it to the
   plan the emit walk already consumes is the shortest such path. **The stated
   cost:** a boxed trait object is neither `Clone` nor `PartialEq`, so
   `FramePlan` is no longer either, and it was both in Phase 1. Nothing this
   loses is load-bearing — §2's "the seam is pure data, never a callback" claim
   is about `ScenePatch`, which is still exactly that and is what actually
   crosses into the backend — but it is a real narrowing of a public type and
   should be read as one.
3. **A record's cross-frame address is `(element, ordinal within its kind)`.**
   The *n*-th quad an element emits is the same record every frame, so a value
   change takes §5.0's O(1) in-place update instead of an insert/remove churn.
   The obligation this creates is stated in `Emission`'s own doc: an element's
   emission order must be stable across frames.

### The one rule that makes `.boundary()` observable

Everything gate #1 and gate #2 measure falls out of a single condition, rather
than being special-cased at either end:

> An element re-emits when the plan did not mark it reused, **or** when its
> resolved absolute bounds are not the ones it was last emitted with, **or**
> when it moved to a different layer.

The middle clause is the load-bearing one. A reconciler diffs *descriptions*;
it has no opinion about where computed layout put an element. So:

- A scroll container that is **not** a boundary folds its offset into its
  children's positions. Every visible child moves, so every visible child
  re-emits — even though all of them reconciled perfectly clean.
- A scroll container that **is** a boundary hands that displacement to its
  layer's transform instead. Its children's absolute bounds do not move, so
  none of them re-emits.

That is the entire difference between the two gates, and neither branch of the
walk knows it is implementing a gate.

### The compositing decision, and why it takes two inputs

`Compositor::resolve` reaches `Composite::TransformOnly` only when **both** the
content is measurably clean *and* the signal that woke the frame was
`Reason::Scroll`. Under §4.0's ambient reconciliation the first condition is a
fact the frame already established — the reconciler re-diffed everything inside
the boundary whether or not the boundary exists — which is a genuine change
from SFD §1.1, where the tagged notification was the *only* evidence available
and a wrong key meant silently stale UI.

Requiring the signal as well is therefore deliberately conservative, and the
cost is stated rather than hidden: a pure-scroll frame signalled as
`DataChanged` costs one ordinary recomposite it did not have to. What it buys
is that a bug in any element's `diff_key` can only ever produce a slow frame,
never a frame that slid stale content into view. §4.1 asks for the scroll signal
"from day one — not retrofitted"; this is what consuming it looks like once the
diff underneath it is ambient.

---

## 3. Deviations from §3.1's file map, and why

Two files exist that §3.1 does not name, in the same spirit as Phase 1's six.
Both are recorded in their own module docs; collected here:

1. **`patch/emit.rs`** — §2's diagram draws the arrow but §3.1 gives it no
   file, because in the legacy backend the arrow *is* `Element::paint` pushing
   into `Scene` directly. Making it a separate walk over a plan is what lets
   "which elements needed re-emitting" be a value a test asserts on rather than
   an effect it has to detect, which is what both of Phase 2's first two gates
   actually assert.
2. **`boundary/compositor.rs`** — §3.1 gives `boundary/` a policy file (what an
   author may tune) and an identity file (how a boundary finds itself), and no
   home for the thing that consumes both plus `Reason` once per frame. In R-N/SFD
   that decision had no separate home either: it was interleaved into
   `Interactivity`'s paint block inside `div.rs`. §3.4 lists breaking that block
   apart as one of the four seams the widgets crate splits along, and this is the
   half of it that is not any element type's business.

Two smaller judgment calls:

3. **`Pixels` and `Size` are declared in `boundary/policy.rs`.** §4.1 spells
   `Buffering::Margin` as `Option<Size<Pixels>>`, and both types belong to the
   frontend geometry surface §7 freezes — which still lives in the legacy crate
   and has no home in the workspace (§3's file map gives `wgpui-core` no geometry
   module). Declaring them minimally here beat both widening the signature to a
   bare `[f32; 2]` and pulling the legacy crate across the boundary §3 draws.
   Whichever phase moves geometry into the workspace deletes them.
4. **`wgpui-layout` re-exports four `taffy` types and adds `definite()`.**
   `LayoutTree::compute_layout`'s signature already obliged every caller to name
   `taffy::style::AvailableSpace` itself, which is a leak §3.2 does not intend.
   This is the narrowest way to close it without wrapping the style type, which
   §3.2 explicitly rules out ("a third representation of the same thing with
   nothing to say for itself").

---

## 4. The three gates

All three are automated `#[test]` functions, named for the gate they satisfy.
All three pass. Each is described below with what it actually asserts —
including, where the spec's literal wording does not map onto a workspace with
no renderer in it, exactly how it was mapped.

### Gate #1 — `.boundary()` reaches the fast path with nothing else touched

> *"`.boundary()` with zero policy arguments reaches R-N's fast path
> (transform-only recomposite on scroll) on a plain `overflow_y_scroll` div with
> no other API touched."*

**Test:** `patch::emit::tests::gate_1_a_bare_boundary_recomposites_a_scroll_transform_only`
**Result: passes.**

One scroll container holding 300 rows, driven through three real frames — build,
idle, scroll — with a `Window` struct holding a live `Reconciler`, `LayoutTree`,
`Emitter`, and `Scene` across all three, so the gate asserts on what a frame
*produced* rather than on an intermediate value.

On the scroll frame, with one `signals.scrolled(layer)` and one changed offset:

- `Composite::TransformOnly`, `Invalidation::TRANSFORM`, and a transform of
  exactly `translated(0.0, -60.0)`.
- **The observable consequence, which is what makes this a gate rather than an
  assertion about the decision that produced it:** the patch is empty, uploaded
  bytes are 0, upload calls are 0, `nodes_emitted` is 0, and `nodes_skipped` is
  301 — every element that emits anything skipped emitting.
- The scene agrees independently: the layer's transform moved, its invalidation
  is `TRANSFORM` and nothing else, and its 300 quads are still resident.
- `retention` is `Retention::Texture`, because 300 is over the default
  `rasterize_above` of 256 — so the gate exercises the "independent GPU texture
  retention" half of §8's Phase 2 row and not only the transform half. See the
  qualification below on what that word means here.

The "no other API touched" clause is checked mechanically, not by reading the
helper: a recursive `assert_only_boundary_is_touched` walks the description tree
and asserts every node's `element_id()` is `None`, `is_uncached()` is false, and
the one boundary that exists carries exactly `BoundaryPolicy::default()`. The
test also asserts the *idle* frame is `Composite::Clean` with an empty layer
invalidation, so "transform-only" is distinguished from "did nothing."

**One honest qualification about `Retention::Texture`.** It is a *decision*,
recorded and observable, and no texture is allocated, pooled, or drawn anywhere
in this phase. §3.1 puts every live `wgpu::Device` in `wgpui-wgpu` and §8 puts
the compositing entry that would consume this decision in Phase 4. A reviewer
who reads §8's "independent GPU texture retention" as requiring a real texture
should treat this gate as covering the policy and the transform and not the
texture — and §7 below lists it among the deferred items for exactly that
reason.

**Mapping note:** "a plain `overflow_y_scroll` div" is a `Description` with a
`.scroll_offset()` and a column style. There is no `div()` in this workspace and
no `Styled` trait yet (§3.4, Phase 5+). What the gate needs from
`overflow_y_scroll` is a container that displaces its children, which is exactly
what `.scroll_offset()` is — deliberately expressed as *what it does* rather
than as a scroll position, because both consumers consume it identically.

### Gate #2 — boundary and reconciliation are decoupled

> *"Removing `.boundary()` from that same test case degrades the scroll case to
> a per-tick recomposite (no independent texture) but does **not** reintroduce
> full rebuild."*

**Test:** `patch::emit::tests::gate_2_removing_the_boundary_costs_a_recomposite_and_not_a_rebuild`
**Result: passes.**

Two live windows side by side, identical trees except for the one `.boundary()`
call, each driven through the same two frames with the same offset change. This
is the load-bearing gate: Phase 1's whole claim was that reconciliation is
ambient and owes nothing to any boundary, and this is where that gets proved
*under* a boundary by taking the boundary away.

1. **Reconciliation is identical.** Not "similar," not "also fast": the entire
   `FrameStats` value is asserted equal between the two windows, and so is the
   full ordered list of `LayoutNodeId`s. Both are fully reused; the unboundaried
   one rebuilt 0 elements, created 0 layout nodes, swept 0 nodes and 0 instances;
   both instance tables are the same size.
2. **Compositing is not.** The boundaried case emits nothing at all (the
   control). The unboundaried one re-emits all 300 rows and reports
   `Composite::Redisplay`.
3. **It is a recomposite and not a rebuild.** Every one of the 300 operations
   is a value update in place: `records_inserted` is 0, `records_removed` is 0,
   and every patch in the list is asserted to be `PatchOp::Update`. Uploaded
   bytes are exactly `300 × Quad::SLOT_STRIDE` — the moved rows' bytes and
   nothing wider.
4. **The container's own paint does not scroll with its contents** in either
   case, which is a correctness property the two-layer split has to get right:
   the boundaried root layer holds 1 quad and the boundary layer holds 300,
   while the unboundaried root layer holds all 301.

Four supporting tests bound the decision from the other sides: a boundary
signalled `DataChanged` folds the offset in and is refused the fast path (and
does not leave a stale transform behind); a content change inside a
scroll-signalled boundary redisplays it rather than sliding stale pixels; a
boundary holding two quads stays `Retention::Primitives`; and an evicted
boundary releases its layer.

### Gate #3 — `WgpuSurface` gets real identity

> *"An unmoved, unresized `wgpu_surface()` element skips
> `request_layout`/`prepaint`/`paint` across frames exactly like a reconciled
> `div` would."*

**Test:** `wgpu_surface::tests::gate_3_an_unmoved_unresized_surface_skips_work_like_a_reconciled_element_would`
**Result: passes.**

A surface and a plain reconciled element as siblings, neither named. The plain
sibling is the control: every assertion about the surface is made about it too,
so "exactly like" is measured rather than asserted.

- The surface's `element_id()` is asserted to still be `None` — §5.5's Gap 1 is
  that `id()` returns `None`, and it still does. What changed is that this no
  longer costs the element its identity, which is the whole point.
- On frame 2 the surface reports `NodeOutcome::Reused`, its `LayoutNodeId` is
  bit-identical to frame 1's, `layout_nodes_created` is 0,
  `layout_nodes_reused` is 3, and `instances_swept` is 0.
- `skipped(surface)` is asserted **equal to** `skipped(panel)`, and both are
  `true`.

Three supporting tests: each of the three fingerprint fields is separately shown
to be the only thing that rebuilds the surface, reporting exactly the axes it
affects (a resize is `LAYOUT | DISPLAY`, a style or handle change is `DISPLAY`
alone) and never disturbing the sibling; a key compared against a different
element type is a full invalidation; an identical key reports nothing stale. And
the *paint* half has its own test,
`an_unchanged_surface_is_never_asked_to_emit_again`: driven through the real
emitter, the second frame calls the surface's `Emit` zero times and uploads zero
bytes.

**Mapping note, stated rather than glossed.** "Skips `request_layout`" is
`LayoutTree::reuse` rather than `LayoutTree::request_layout` — a real distinction
in this workspace, since `request_layout` is what creates a Taffy node and
`reuse` is what marks a retained one present without recreating it. "Skips
`prepaint`/`paint`" is `NodeOutcome::Reused` plus, in the supporting test, the
emitter genuinely not being called. This is the same mapping Phase 1's gate #2
made, for the same reason.

**The much larger scope note, stated up front in the module's own doc.** What
`wgpui-widgets/src/wgpu_surface.rs` contains is the *shape* a `WgpuSurface`
presents to reconciliation: positional identity, no children, and a fingerprint
over exactly `(bounds, style, surface_id)`. What it does **not** contain is the
element: no `WgpuSurfaceHandle`, no `SurfaceRegistry`, no triple buffer, no
external render thread, no texture, no `wgpu` dependency of any kind. `SurfaceId`
is an opaque `u64` standing in for the real handle, and `SurfaceStyle` carries
two representative fields standing in for the frozen `Style` (§7). Both are
marked as placeholders in the source. Wiring the real ones means wiring
`wgpui-widgets` to `wgpui-wgpu`, which §8 places in Phase 4 alongside Gap 2's
compositing unification. A reviewer should read this gate as "the identity and
fingerprint half of Gap 1 is closed and proved" — which is what §8's Phase 2 row
scopes it to — and not as "`WgpuSurface` is ported."

---

## 5. Check, test, and clippy status

```
cargo check --workspace --offline               → Finished, 0 errors
  ↳ within it, `gpui-ce` (lib)                  → 72 warnings, the identical
                                                   count Phase 0 and Phase 1
                                                   recorded as the baseline;
                                                   5 of them are E0133
                                                   (unsafe_op_in_unsafe_fn) in
                                                   src/app/entity_map.rs,
                                                   src/view.rs, src/window.rs
cargo test -p wgpui-core --offline              → 171 passed, 0 failed
cargo test -p wgpui-layout --offline            → 6 passed, 0 failed
cargo test -p wgpui-widgets --offline           → 6 passed, 0 failed
cargo clippy -p wgpui-core -p wgpui-layout
  -p wgpui-widgets --release --all-targets
  --all-features -- --deny warnings             → Finished, 0 warnings
```

The clippy invocation is exactly what `script/clippy.ps1` runs for a
`-p`-scoped call, which `AGENTS.md` names as this repo's convention.
`clippy.toml` at the repo root was checked first, as it was in Phase 1: its
`disallowed-methods` list is about `std::process::Command` and
`serde_json::from_reader`, its `disallowed-types` list is entirely commented
out, and its one `ignore-interior-mutability` entry names an `agent_ui` type.
None of it applies to these three crates, and **no suppression was added
anywhere** — `#[allow]` appears nowhere in this phase's code. The one crate-level
`#![allow(dead_code)]` in each new crate is Phase 0 scaffold, present because
nothing outside the workspace calls into these crates yet.

Phase 2 added 43 tests to `wgpui-core` (128 → 171) and the first 6 to
`wgpui-widgets`. `wgpui-layout`'s 6 are unchanged.

**On `cargo test --workspace`, stated precisely rather than as a checkmark.**
It was **not** run to completion, and this is the same situation Phase 1
recorded. `gpui-ce`'s own test binary is long-running on this machine — Phase 1's
verification observed it CPU-bound for several minutes without completing, and
Phase 0/1 both recorded the same. That binary is unmodified by this branch and
provably so: `git diff` from Phase 1's tip to this branch's head touches nothing
under `src/`, does not touch the root `Cargo.toml`, and the only `Cargo.lock`
change is the two new `wgpui-widgets` path edges, which `gpui-ce` does not depend
on. **A human should still run `cargo test -p gpui-ce` to completion once** and
confirm its result matches whatever it was before this branch. I am recording
that I did not, rather than implying I did — carried forward from Phase 1
unchanged, because it is unchanged.

---

## 6. What was found broken, and fixed

Everything the six commits claimed was, on reading the tests rather than the
commit messages, actually there: all three gates are real automated tests
driving real frames, each with a live control rather than an asserted constant,
and each with supporting tests that bound the decision from the other side. Two
things needed fixing.

**1. An evicted boundary leaked its layer. Real defect, fixed.**
`Compositor::sweep` implements R-N §3.4's mark-and-sweep and drops a boundary's
retained state after `evict_after_frames`. `Emitter::sweep_departed` retires the
records of every element that left the tree, so a departed boundary's *residency*
went immediately and correctly. But nothing ever called `Scene::remove_layer` —
the method existed, fully implemented, with zero call sites outside its own test
— so the `Layer` entry itself (key, slab handles, invalidation, generation) stayed
in the `LayerTable` for the life of the process, for every boundary identity ever
seen. Small per entry; unbounded in aggregate.

The fix is not simply "call `remove_layer` in the sweep," and the reason is worth
recording. `Emitter::emit` produces a patch and its *caller* applies it
afterwards, so a layer released at the end of the call that produced the patch
could still be named by ops in that very patch — reachable in practice with a
policy of `evict_after_frames: 0`. `Compositor::sweep` now returns the evicted
boundaries' `LayerId`s instead of a count, and the emitter parks them in
`pending_layer_removals` and releases them at the *start* of the next frame, by
which point the caller has applied the patch that retired their records.
Regression test:
`patch::emit::tests::an_evicted_boundary_stops_costing_the_scene_a_layer`, which
asserts all three stages — residency goes immediately, the layer record survives
the eviction interval (so a panel that comes straight back keeps where it was),
and then the layer record goes too.

**2. One gate-supporting test was misleading, though not wrong.**
`each_of_the_three_fields_is_the_only_thing_that_rebuilds_the_surface` built its
"a different surface handle" case as a value identical to the control and then
silently replaced it inside the loop body on `index == 0`, so the array read as
though the first case tested nothing. Rewritten so each case differs from the
control in exactly one field at its declaration site, with an added
`assert_ne!(changed.diff_key(), base().diff_key())` so a future edit that
accidentally makes a case a no-op fails loudly instead of passing vacuously.

---

## 7. Gate assessment — honest read

**All three gates are met, as automated passing tests, with the mapping notes in
§4 stated rather than papered over.** The one a reviewer is most likely to want
to argue with is gate #1's `Retention::Texture`: §8's Phase 2 row says
"independent GPU texture retention," and what exists is the decision, made per
boundary from that boundary's own primitive count, recorded and observable — not
a texture. I think that is the correct reading of a phase whose crates are
forbidden a `wgpu::Device` by §3.1 and whose compositing entry §8 explicitly
schedules for Phase 4. But a reviewer who disagrees should treat gate #1 as
*partially* met — the transform half proved, the texture half decided but not
built — and I would rather they make that call knowing exactly what was checked.

**Explicitly deferred, and not this phase's job.** Named here so nobody mistakes
them for oversights:

1. **Real `WgpuSurface` wiring** — `WgpuSurfaceHandle`, `SurfaceRegistry`, the
   triple buffer, the external render thread. §8 puts it in Phase 4 with Gap 2's
   compositing unification; §9's risk table specifically ring-fences
   `SurfaceRegistry`'s producer-side concurrency from being touched. What Phase 2
   owed was Gap 1 — identity and a `diff_key` — and that is what it delivered.
2. **Real GPU texture retention.** `Retention::Texture` is a decision.
   No texture pool, no rasterize-to-texture, no composite entry. Phase 4.
3. **Tile-based buffering.** `Buffering::Tiled` is declared, carries its
   `tile_size`/`retain_radius`, and reports `is_implemented() == false` — with a
   test asserting exactly that, and asserting that a boundary declaring it falls
   back to *more* buffering rather than to none. The visibility pass, spatial
   eviction, and resident-tile budget are Phase 4.5.
4. **Ordering and occlusion.** Paint order within a layer is append order: a
   record keeps the slot it was first inserted at. §5.1's per-layer ordering pass
   is Phase 3, over these same slabs. Emitting a correct *set* of primitives into
   a correct *layer* is what this phase needs and all it claims.

Smaller things a human should know:

5. **`FramePlan` is no longer `Clone` or `PartialEq`** (§2, decision 2). A real
   narrowing of a Phase 1 public type, caused by the boxed emitter. `ScenePatch`
   — the thing that actually crosses into the backend — is unaffected.
6. **`LayerTransform` holds a translation and nothing else.** Deliberate: §5.4's
   claim is that a scroll tick costs one changed matrix, and the two motions that
   reach a boundary today (scroll, pan) are translations. Widening it to a full
   affine changes this type and nothing that consumes it, because every consumer
   asks it the same question.
7. **The emitter owns each boundary layer's per-frame invalidation.**
   `Emitter::begin_boundary` marks an already-existing layer clean at the top of
   every frame, on the principle that a frame's invalidation is what *that frame*
   made stale. The consequence, which is not obvious from the call site: axes
   raised out of band via `LayerTable::invalidate` between two frames are cleared
   rather than accumulated. Correct for the current single driver; worth
   revisiting when something other than the emit walk starts invalidating layers.
8. **Positional boundary identity costs a boundary its state across a sibling
   reorder**, and only that. `BoundaryIdentity::from_path` hashes the same
   `&[ElementId]` slice `InstanceKey::from_path` does, domain-separated by a
   `"boundary"` prefix so three identities derived from one path cannot collide,
   and with bit 0 forced set so a derived identity can never alias
   `BoundaryId::ROOT`. A test asserts the non-collision directly.
9. **Two primitive kinds, not seven**, and `Quad`/`GlyphRun`'s field sets are
   protocol exercisers rather than GPU layouts — both carried forward from Phase
   1 (`docs/phase-1-results.md` §2) and unchanged here.
10. **The §9 kill switch still has no env-var read.**
    `Reconciler::with_reconciliation_disabled()` exists and is tested; nothing
    reads `WGPUI_INSTANCES` because `wgpui-core` still has no startup path.
    Carried forward from Phase 1 §3, still owed by whichever phase gives the crate
    a window.
11. **`PrimitiveStore::reflow` is still O(records-in-layer) per insert/remove**,
    so a bulk build is O(N²) — Phase 1 §6 flagged it, and Phase 2 is the first
    phase that drives real bulk inserts through it (gate #1 seeds 300 records,
    which is fine; a real list will not be). The value-update path the gates
    measure does not go through it and is genuinely O(1). A phase that starts
    driving large trees should make `reflow` incremental first.

**Nothing in `src/` was touched, and `gpui-ce`'s warning count is unchanged from
Phase 0's baseline**, so §8's "legacy backend is frozen once Phase 1 starts"
constraint holds by inspection as well as by intent.
