# Phase 1 Results — Patch Protocol, Persistent Scene, Ambient Reconciliation

Status: **Phase 1 executed.** This documents what was built, what each of the
four gates actually asserts, what passed, and what a human should treat as
still open before calling this gate closed. It follows
`docs/gpu-native-architecture.md` ("2.0" below) §2, §3.1, §3.2, §4.0, §4.2,
§5.0, and §8's Phase 1 row. Nothing under `src/` changed — the legacy backend
is frozen (§8's own constraint) and `gpui-ce`'s standalone check is unchanged
from the Phase 0 baseline. Work lives on branch
`wgpui-2.0/phase-1-patch-protocol`.

**Contents:** §1 What shipped and where · §2 The primitive-kind scoping
decision · §3 Deviations from §3.1's file map, and why · §4 The four gates
· §5 Check, test, and clippy status · §6 Gate assessment — honest read

---

## 1. What shipped, and where

Phase 1 turns four Phase 0 stub trees into working, headless, testable code.
No `wgpu` dependency was added to `wgpui-core` or `wgpui-layout`; nothing
renders; nothing is wired into `src/`.

| File | Lines | Role |
|---|---|---|
| `wgpui-core/src/patch.rs` | 362 | `RecordKey`, `PatchOp<T>`, `Patch<T>`, `PatchList<T>`, `PatchError` |
| `wgpui-core/src/patch/primitive.rs` | 421 | `Primitive` trait, `PrimitiveKind`, `Quad`, `GlyphRun`, byte encoding |
| `wgpui-core/src/patch/apply.rs` | 802 | `ScenePatch`, `apply`, `UploadPlan`, `compare_to_rebuild` — gates #1 and #4 |
| `wgpui-core/src/scene.rs` | 194 | `Scene` assembly; `draw_ranges` (CPU-computed, §8's "same as today") |
| `wgpui-core/src/scene/layer.rs` | 348 | `BoundaryId`, `LayerKey`, `LayerId`, `Layer`, `LayerTable` |
| `wgpui-core/src/scene/slab.rs` | 534 | Buddy-allocator `SlabAllocator`, `Reallocation` |
| `wgpui-core/src/scene/slab_range.rs` | 339 | `SlabRange`, size classes, `UploadRange`, `coalesce_uploads` |
| `wgpui-core/src/scene/primitive_store.rs` | 727 | Resident bytes, per-record placement, the delta-upload contract |
| `wgpui-core/src/scene/record.rs` | 350 | `RecordStore<T>` plus `LayoutInput`/`Hitbox`/`DispatchNode` |
| `wgpui-core/src/scene/tile.rs` | 46 | `TileCoord` — the *address* only; §4.3's mechanism is Phase 4.5 |
| `wgpui-core/src/reconcile/description.rs` | 256 | `Description`, `ElementId` — the per-frame value |
| `wgpui-core/src/reconcile/diff_key.rs` | 197 | `ReconcileKey`, `compare_by_equality`, `AlwaysDirty` |
| `wgpui-core/src/reconcile/instance.rs` | 361 | `InstanceKey`, `ElementInstance`, `RetainedElement`, `InstanceTable` |
| `wgpui-core/src/reconcile/plan.rs` | 333 | `FramePlan`, `PlannedNode`, `NodeOutcome`, `RebuildReason`, `FrameStats` |
| `wgpui-core/src/reconcile/reconciler.rs` | 916 | The ambient walk — gates #2 and #3 |
| `wgpui-core/src/reconcile/state.rs` | 239 | `StateScope`, `StateKey`, `ElementStateStore` — what `.uncached()` must not touch |
| `wgpui-core/src/reconcile/uncached.rs` | 122 | `UncachedScope`, the §4.2 scope flag |
| `wgpui-core/src/invalidation/{axes,reason,request}.rs` | 276 | The four axes, `Reason::Scroll`, `InvalidationRequest` |
| `wgpui-layout/src/taffy_tree.rs` | 428 | Persistent `LayoutTree`, `LayoutNodeId`, create/reuse/sweep |

One dependency edge was added: `wgpui-core → wgpui-layout`, because a retained
instance record holds the layout node its element reuses. The edge runs one way
only — `wgpui-layout` names nothing from `wgpui-core`. Phase 0's results doc
deferred wiring any inter-crate edge until "Phase 1 is where `wgpui-core`'s
types other crates need to name" exist; this is that moment, and it turned out
to be the reverse direction from the one Phase 0 anticipated.

---

## 2. The primitive-kind scoping decision

**Two primitive kinds ship, not seven.** The legacy renderer has seven
instanced kinds (quads, shadows, paths, underlines, mono sprites, poly sprites,
surfaces). Phase 1 implements `Quad` and `GlyphRun` and stops.

The reasoning is in `patch/primitive.rs`'s own module doc, and it is about what
the protocol has to *serve*, not about effort:

- **`Quad` is the fixed-size shape** — exactly one slab slot, always. Quads,
  shadows, underlines, and both sprite kinds are all structurally this.
- **`GlyphRun` is the variable-size shape** — one slot per glyph, so a run's
  slot count changes with its content and can cross a size class between
  frames. Text runs and paths are structurally this, and it is the only shape
  that exercises the allocator's fall-up/relocate path and §5.0's
  "insert/remove that forces the allocator to relocate" disclosure at all.

Adding a third kind is: implement `Primitive`, add a `PrimitiveKind` variant,
add one `PrimitiveStore` field to `Scene`. Nothing in `patch`, `scene::slab`,
`scene::slab_range`, or the upload machinery is written per-kind — the protocol
is generic over `P: Primitive` and monomorphised. Porting the other five now
would be five repetitions of the same twenty lines and would test the
architecture no further.

**What this does mean, stated plainly:** the field sets of `Quad` and
`GlyphRun` are *not* ports of the legacy renderer's structs. They carry the
subset that exercises the protocol. A later phase that actually draws these
replaces the field sets and the `SLOT_STRIDE` constants; it does not touch the
protocol around them. Nobody should read `Quad` here as the shipping GPU
layout for a quad.

A second scoping decision worth naming: **encoding is byte-oriented, not
`bytemuck`-cast.** That keeps `wgpui-core` free of the dependency, makes each
kind's GPU layout an explicit reviewable decision rather than a consequence of
Rust field order, and — the load-bearing one — lets a headless test compare
resident bytes for exact equality, which is what gate #1 is.

---

## 3. Deviations from §3.1's file map, and why

§3.1 draws `reconcile/` as three files and `scene/` as four. Six files exist
that it does not name. Each is listed in its own module doc with the same
reasoning; collected here:

1. **`reconcile/description.rs`** — §3.1 gives `instance.rs` the *retained*
   side of R-N §2.1's split and names no home for the *per-frame* side,
   because in the legacy backend the two are the same object (`Drawable<E>`).
   Separating them is the whole point of Pillar I.
2. **`reconcile/reconciler.rs`** — §3.1 names the fingerprint trait, the
   retained record, and the scope flag, but no file for the walk that uses all
   three, because in the legacy backend that walk *is* `Div::prepaint`'s child
   loop inside `div.rs`. Ambient reconciliation is precisely the claim that
   this is not one element type's business.
3. **`reconcile/plan.rs`** — the legacy reconciler has no equivalent because
   it *is* the draw walk: it calls `prepaint`/`paint` inline and the only
   evidence of a skip is a counter. §2's premise is that the seam is pure
   data, so the reconciler produces a plan and a caller executes it. This is
   also what makes gate #2's question a value to assert on rather than an
   effect to detect.
4. **`reconcile/state.rs`** — R-N §2.1 lists State as "already retained…
   unchanged," so neither R-N nor §3.1 gives it a file. §4.2 makes it
   load-bearing anyway: "a slider inside an `.uncached()` panel keeps its
   state" needs a mechanism to be true *of*, and gate #3 is a test that it is.
5. **`scene/primitive_store.rs`** — §3.1's `scene/` split has no home for "the
   resident bytes themselves plus the per-record bookkeeping that turns a
   patch into an upload," and putting it in `scene.rs` would have made that
   file the one thing §3 exists to prevent.
6. **`scene/record.rs`** — §2 names four things the patch list carries and
   §3.1 gives a home to one of them. Giving the other three a shared generic
   store is what keeps "primitives, layout inputs, hitboxes, dispatch nodes" a
   single protocol rather than four.

Two more, smaller:

7. **The allocator is a buddy allocator, not a port of the legacy free-list
   design.** Because every size class is already a power-of-two multiple of
   `MIN_CLASS`, keeping every block aligned to its own size makes a block's
   buddy one XOR away, so coalescing on free is exact and immediate. That
   collapses three legacy mechanisms (adjacency scan, advisory compaction
   plan, reserved-block index) into one and removes the residual risk R-N §4.3
   discloses — fragmentation reclaimed only when someone remembers to ask. The
   disclosed cost is that alignment can leave a gap ahead of a large
   allocation; those gaps are split into aligned blocks and pushed onto the
   free lists immediately rather than lost.
8. **The §9 kill switch is a constructor, not an environment variable.**
   §9 asks for one "following `WGPUI_INSTANCES=0`'s precedent."
   `Reconciler::with_reconciliation_disabled()` is that switch and is tested
   (`the_kill_switch_reverts_to_unconditional_rebuild`), but nothing reads an
   env var: `wgpui-core` has no window, no app, and no startup path to read
   one in yet. Whoever wires the reconciler into a real window in a later
   phase owns adding the env-var read on top of this constructor. **This is a
   genuine, if small, piece of §9's mitigation still outstanding.**

---

## 4. The four gates

All four are automated `#[test]` functions in the crate, named for the gate
they satisfy. All four pass. Each is described below with what it actually
asserts — including, where the spec's literal wording does not map onto a
crate with no renderer in it, exactly how it was mapped.

### Gate #1 — round-trip

> *"Apply a patch sequence, read back the resident buffer, matches an
> equivalent full-rebuild reference exactly."*

**Test:** `patch::apply::tests::gate_1_a_patch_sequence_round_trips_to_a_full_rebuild`
**Result: passes.**

Four frames across two layers and both primitive kinds, exercising every case
the protocol has: appends, interior inserts, in-place value updates, a
variable-size update that changes slot count, forty interior removals, and
enough regrowth to cross a size class and force a relocation (the test asserts
the relocation actually happened, rather than hoping it did). A `LayerOracle`
maintained by the test tracks the intended content independently and is
asserted against the scene record-for-record after every frame; the reference
scene is then built from the oracle, not from the scene under test — deriving
it from the scene would have made the gate self-fulfilling, proving only that
the encoder is deterministic.

**One honest qualification about what "exactly" means.** It means every layer's
**occupied bytes** are byte-identical, read back out of the arena at that
layer's own address. It deliberately does **not** mean the two arenas are
identical end to end, and this is the protocol's own contract rather than a
weakening of the gate: `PatchOp::Insert` states that slot placement is the
scene's decision and callers must not depend on it, which is exactly what makes
relocation and compaction legal (§5.0's second case). A scene that reached its
state through inserts, growth, and removals has legitimately placed a layer at
a different base than a fresh build would, and its arena holds vacated blocks a
fresh build never allocated. Asserting whole-arena equality would assert that
the allocator has *no* history-dependence — a property the design explicitly
rejects.

The comparison is proven capable of failing: `the_round_trip_comparison_actually_detects_a_difference`
constructs a wrong-value reference, a wrong-order reference, and a wrong-layer-set
reference, and asserts each produces the corresponding `ResidencyMismatch`.

### Gate #2 — ambient reconciliation

> *"A plain, unboundaried three-level-deep div that renders identically to last
> frame keeps the same `LayoutId` and skips `prepaint`/`paint` — with zero
> `.boundary()`, zero `.id()`, zero API touched anywhere in the test."*

**Test:** `reconcile::reconciler::tests::gate_2_an_unboundaried_three_level_tree_keeps_its_nodes_and_skips_prepaint_and_paint`
**Result: passes.**

A six-element tree, three levels deep (asserted: the plan's maximum depth is
2). On the identical second frame every element reports `NodeOutcome::Reused`,
`layout_nodes_created` is 0, `layout_nodes_reused` is 6, nothing is swept, and
every element's `LayoutNodeId` is bit-identical to the one it held in frame 1.

The "zero API" clause is checked mechanically, not by reading the helper: a
recursive `assert_names_nothing` walks the description tree and asserts every
node's `element_id()` is `None` and `is_uncached()` is false. The boundary half
is enforced by an absence — `Reconciler::reconcile` takes a description tree and
a layout tree, and has no layer, boundary, subtree, or scope parameter a caller
*could* fence it with. There is no `.boundary()` in this crate to touch.

**Mapping note, stated rather than glossed:** "skips `prepaint`/`paint`" is
`NodeOutcome::Reused` on a `FramePlan`, surfaced as
`PlannedNode::skipped_prepaint_and_paint()`. `wgpui-core` has no `prepaint` or
`paint` — those are `Element` trait methods that take `Window` and `App`, which
§3 puts in a different crate. The reconciler's product is a plan saying which
elements need work; "skipped" means the plan says this element needs none. That
is the same claim the spec is making, checked one step earlier in the pipeline,
and it is checkable *because* §2's data-not-callbacks seam moved it there. It
is not the same as observing a real element's methods not being called — that
observation belongs to whichever phase gives `wgpui-widgets` real elements.

`LayoutId` maps to `LayoutNodeId`, the identity `wgpui-layout` hands out.

### Gate #3 — `.uncached()`

> *"A `.uncached()` subtree allocates no `ElementInstance` and its children's
> state (`use_state`, focus) survives across frames identically to a
> reconciled subtree's — proving the two mechanisms are actually decoupled."*

**Test:** `reconcile::reconciler::tests::gate_3_uncached_allocates_no_instance_while_state_survives_identically`
**Result: passes.**

Two structurally identical panels side by side under one root — one reconciled,
one `.uncached()` — so "identically" is measured against a live control rather
than an asserted constant. Over three frames:

- The uncached panel and its leaf contribute **zero** records: the instance
  table holds exactly 3 (root plus the reconciled panel's two elements), and
  both uncached nodes report `instance: None` and `NodeOutcome::Uncached`.
- Both leaves' state is visited every frame through `ElementStateStore`. The
  visit counts are asserted **equal to each other** and equal to the frame
  number, and `sweep` reclaims nothing — an uncached element visits its state
  like any other, so it survives for the same reason.
- The control direction is asserted too: the reconciled leaf reports
  `skipped_prepaint_and_paint()`, the uncached one does not. State behaved
  identically while reconciliation did not, which is what "decoupled" means.

Three supporting tests: entering `.uncached()` drops the records an element
already had immediately rather than at the next sweep; leaving it restores full
reconciliation; nested uncached subtrees stay suppressed past the inner one.

**Mapping note:** `use_state` and `focus` do not exist as APIs in this crate
yet — `focus.rs` is still a Phase 0 stub and there is no `use_state` because
there is no `Window`. What exists is the mechanism both would be built on:
state addressed by `StateKey = hash(path, TypeId)`, which is R-N §2.1's own
description of how state is keyed. The decoupling claim is structural and
checkable today: `state.rs` does not import `instance.rs`, `instance.rs` does
not import `state.rs`, and the reconciler carries a `StateScope` on every
planned node — including uncached ones — without ever consulting it. The two
mechanisms share an input (the element's path) and nothing else.

### Gate #4 — delta upload

> *"Changing one primitive's value inside a large layer issues one
> `write_buffer` call sized to that primitive's stride, not the layer's full
> range."*

**Test:** `patch::apply::tests::gate_4_one_changed_primitive_in_a_ten_thousand_primitive_layer_uploads_one_slot`
**Result: passes.**

A layer holding 10,000 quads (640,000 resident bytes). One `Update` patch for
one primitive produces an `UploadPlan` with `len() == 1` and `byte_count() ==
64` — exactly `Quad::SLOT_STRIDE` — and the test asserts the entry's byte span
equals `record_byte_range(layer, target)`, so it is not merely *one small*
write but a write addressed to that specific primitive's own slot. It also
asserts `byte_count() * 10_000 == layer_byte_count`, i.e. the delta really is
one ten-thousandth of the layer.

Per the task's own framing this is plain headless data — an `UploadRange {
kind, byte_offset, byte_length }` — with no `wgpu::Buffer` and no
`write_buffer` anywhere in this phase. §5.0's gate wording says "measured
directly via a byte-count/call-count counter, not inferred from 'the test
passed'"; `UploadPlan::len()` and `UploadPlan::byte_count()` are that counter,
and the test asserts on both, because either alone can be gamed by the other.

Two supporting tests: a clean frame produces zero entries and zero bytes (§5.0's
third case — "not a small range, zero"), and scattered updates stay three
separate entries while byte-adjacent ones coalesce into one (§5.0's stated
mitigation, never widened to cover bytes that did not change).

**One thing the test surfaced that is worth recording rather than hiding.**
Building that 10,000-quad layer by 10,000 appends uploads 1,684,480 bytes, not
640,000 — 2.6× the layer's size. Each crossing of a size class relocates the
layer and rewrites it, exactly as §5.0's second case discloses; the total is
amortised the way a `Vec`'s doubling is. This is the *build* cost, not the
steady-state cost the gate measures, and it is bounded by the layer's own slab
rather than the scene. The test asserts the floor (`>= 10,000 slots`) and says
so in a comment rather than asserting a number that reads like a guarantee.

---

## 5. Check, test, and clippy status

```
cargo check --workspace --offline               → Finished, 0 errors
cargo check -p gpui-ce --offline                → Finished, 72 warnings
                                                   (identical count to the
                                                    Phase 0 baseline — §8's
                                                    "legacy backend is frozen"
                                                    constraint holds)
cargo test -p wgpui-core --offline              → 128 passed, 0 failed
cargo test -p wgpui-layout --offline            → 6 passed, 0 failed
cargo test --workspace --offline                → every target compiles; see
                                                   the note below on gpui-ce's
                                                   own suite
cargo clippy -p wgpui-core -p wgpui-layout
  --release --all-targets --all-features
  -- --deny warnings                            → Finished, 0 warnings
```

The clippy invocation is exactly what `script/clippy.ps1` runs for a
`-p`-scoped call. `clippy.toml` at the repo root was checked first: its
`disallowed-methods` list is about `std::process::Command` and
`serde_json::from_reader`, and its `disallowed-types` list is entirely
commented out — none of it applies to these two crates, and no suppression
convention there needed following.

**On `cargo test --workspace`, stated precisely rather than as a checkmark.**
Every target in the workspace compiles under the test profile — all six new
crates, `gpui-ce`, and its 44 examples. `wgpui-core` (128) and `wgpui-layout`
(6) pass in full, repeatedly, in about eleven seconds. `gpui-ce`'s **own** test
binary is long-running on this machine: it was still CPU-bound (30 threads,
steadily accumulating CPU time — running, not deadlocked) after several minutes
and was not waited out to completion during this session. That binary is
unmodified by this branch and provably so: `git diff origin/2.0...HEAD` touches
nothing under `src/`, does not touch the root `Cargo.toml`, and the only
`Cargo.lock` change is the new `wgpui-core → wgpui-layout` path edge, which
`gpui-ce` does not depend on. **A human should run `cargo test -p gpui-ce` to
completion once and confirm its result matches whatever it was before this
branch** — I am recording that I did not, rather than implying I did.

**Three real clippy findings were fixed, not suppressed:**

- `InstanceTable::store` took eight positional arguments, four of them
  structurally similar collections a caller could silently transpose. Replaced
  with a `RetainedElement` value.
- `Arena::align_frontier_to` hand-rolled `x % y != 0`; now `is_multiple_of`.
- Gate #1's rebuild helper took a very complex tuple type — fixing it is what
  produced the `LayerOracle` above, which turned out to fix a real weakness in
  the gate rather than a cosmetic one.

**What was found broken on arrival.** The prior session's commit did not
compile. Three things were missing rather than wrong: `wgpui-core`'s
`Cargo.toml` never declared the `wgpui-layout` dependency every reconcile
module imports; `scene.rs` was still the Phase 0 stub and never declared
`primitive_store` or `record`, so 1,075 lines of the commit were not compiled
at all; and `patch/apply.rs` — where §3.1 explicitly puts the round-trip gate —
was an empty stub. `SlotWriter`'s bounds checks also failed to borrow-check
(reading `destination.len()` inside the `ok_or` of a `get_mut`). Everything
that *was* wired compiled and passed once those were fixed.

---

## 6. Gate assessment — honest read

**All four gates are met, as automated passing tests, with the two mapping
notes in §4 stated rather than papered over** (gate #2's "skips
`prepaint`/`paint`" is a plan outcome because this crate has no `prepaint`;
gate #3's "`use_state`, focus" is the `(path, TypeId)` state mechanism both
would be built on, because neither API exists yet). Both mappings are, I think,
the correct reading of what the gate is asking — but a reviewer who disagrees
should treat those two as *partially* met, not fully, and I would rather they
make that call knowing exactly what was checked.

**The largest genuinely open thing, stated plainly: nothing produces a
`Description` or a `ScenePatch` from a real element yet, and the reconciler is
not connected to the scene.** §2's diagram runs `render()` → Description →
reconcile → patch list → scene. Phase 1 built the Description, the
reconciliation, the patch list, and the scene, and tested each against the
gate that covers it — but the arrow from a `FramePlan` to a `ScenePatch` does
not exist, because deciding *which primitives an element emits* requires an
element vocabulary that §3.4 puts in `wgpui-widgets` and that Phase 1's row
does not include. This is not a gap in the gates (none of the four asks for
it) and building a speculative bridge now would fix the emission API before
any element exists to constrain it. But it does mean **the protocol is
end-to-end tested and not yet end-to-end driven**, and whoever picks up the
next phase should know that seam is where they will land.

Smaller things a human should know:

1. **`test_support.rs` is still a Phase 0 stub.** §3.1 names it for "headless
   patch/reconcile/window testing." Everything built here turned out to be
   testable headlessly without a support layer — every test in this phase is a
   plain `#[test]` against plain values — so nothing was invented to fill the
   file. If a later phase's tests start repeating scaffolding, that is the
   moment to write it, not now.
2. **The §9 kill switch has no env-var read** (§3, deviation 8). The
   constructor exists and is tested; the wiring is owed by whichever phase
   gives `wgpui-core` a startup path.
3. **`Quad` and `GlyphRun`'s field sets are protocol exercisers, not GPU
   layouts** (§2). Do not port them forward as-is.
4. **Two primitive kinds, not seven** (§2). Deliberate, reversible in ~20
   lines per kind, and the reasoning is about which structural shapes the
   protocol must serve rather than about time.
5. **`tile.rs` defines an address and nothing else.** §4.3's tile grid,
   visibility pass, and LRU eviction are Phase 4.5. `LayerKey` carrying an
   optional `TileCoord` from the start costs one field and means Phase 4.5
   extends a mechanism instead of reshaping every layer's identity.
6. **`InvalidationScope` has no `Entity` variant.** R-N §6's fourth variant
   needs `EntityId`, which belongs to `wgpui-core::app` — still a Phase 0
   stub. Adding a placeholder would have put a type in the public surface that
   nothing can produce.
7. **The invalidation axes a frame raises are derived from which patch
   categories a layer's patches touched** (`patch/apply.rs`), never declared
   by the caller — which is `invalidation/axes.rs`'s standing rule, but it is
   worth noting that Phase 1 is the first place that rule has a call site to
   be true at, and there is exactly one.
8. **`PrimitiveStore::reflow` is O(records-in-layer) per insert or remove**,
   so building a layer from nothing by N appends is O(N²). This is correctness
   code, not tuned code: `reflow` recomputes every record's slot offset from
   scratch because keeping that in one auditable function is what makes §5.0's
   three cases reviewable side by side, and the value-update path — the one
   the gate measures and the one a real frame overwhelmingly takes — does not
   go through it at all and is genuinely O(1). But it does mean gate #4's
   10,000-append seed is the slowest thing in the test suite by a wide margin,
   and a phase that starts driving real bulk inserts should make `reflow`
   incremental (only records at or after the edit need new offsets) before it
   makes anything else faster.

**Nothing in `src/` was touched, and `gpui-ce`'s standalone warning count is
unchanged from Phase 0's baseline**, so §8's "legacy backend is frozen once
Phase 1 starts" constraint holds by inspection as well as by intent.
