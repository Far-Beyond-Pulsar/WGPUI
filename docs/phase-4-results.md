# Phase 4 Results — Indirect Draw-Arg Generation, `multi_draw_indirect`, and the `WgpuSurface` Composite Unification

Status: **Phase 4 executed, both gates met.** This documents what was built,
what each gate actually asserts, what was measured on real hardware, what the
wrap-up pass found broken and fixed, and what a human should still treat as
open. It follows `docs/gpu-native-architecture.md` ("2.0" below) §5.3, §5.5, and
§8's Phase 4 row, and §9's risk table, which names this phase's specific failure
mode. Work lives on branch `wgpui-2.0/phase-4-indirect-draw`, pushed to origin,
not merged, no PR.

**Nothing under `src/` changed.** The legacy backend is frozen per §8's own
constraint; `git diff origin/2.0..HEAD -- src/` is empty, and the whole branch
diff against `2.0` touches 24 files, all of them under `crates/`, plus one line
of `Cargo.lock` and this file. That was checked by running the diff, not
assumed — and it matters more here than in earlier phases, because §9's risk
table's concern is precisely that this phase would reach into legacy
concurrency code.

**Contents:** §1 The wrap-up pass, and what it found · §2 What shipped, and
where · §3 The indirect-arg and draw-issuance design · §4 The
`WgpuSurface`/`SurfaceRegistry` unification, and what it deliberately left
untouched · §5 Gate #1 — draw issuing is O(layer slots) · §6 Gate #2 — a covered
viewport, and the concurrency evidence · §7 GPU adapter honesty check · §8
Check, test, and clippy status · §9 Gate assessment — honest read · §10 What is
open for later phases

---

## 1. The wrap-up pass, and what it found

Phase 4's seven commits were written in a session that hit an API limit and
stopped. The brief for this pass said the session was cut off *after* core
functionality, while doing additional strengthening work — specifically
"extend gate 1 to sweep every available draw mode" — and asked whether that
in-flight work was worth finishing.

**It was already finished.** Commit `2196448` contains it, and
`tests/indirect_draw.rs`'s gate-1 test does sweep: it loops over
`modes(&context)` — every mode `DrawMode::is_available` allows on the open
device — and asserts draw calls, bind counts, and slot counts equal at both
primitive counts *per mode*, with an extra assertion pinning the per-slot path
to exactly one call per slot, which is the form of the claim that cannot be
satisfied by a mode collapsing everything into one call. On this adapter all
four modes are available and all four are swept. Nothing was owed here.

What the pass did find, by reading the diffs rather than the commit messages,
was four things, all fixed on this branch.

**1. `Cargo.lock` never recorded the `log` dependency this phase added — the
branch as pushed fails any locked build.** `crates/wgpui-wgpu/Cargo.toml` gained
`log = "0.4.16"` in commit `1e42831` (deliberately, so
`render/surface_registry.rs` could stay a mechanical copy of a file that logs —
see §4), but the corresponding `Cargo.lock` line was never committed. This is
the "trivial `Cargo.lock` diff" the interrupted session left behind; it was not
trivial. Verified rather than argued: with the lockfile as pushed,
`cargo metadata --locked` exits 101; with the one-line addition, it exits 0. A
plain `cargo build` regenerates the lockfile silently and so never noticed, which
is exactly why this survived seven commits and a full test run.

**2. `QuadDrawPlan` was rebuilt every frame, contradicting its own doc, on the
one path gate 1 measures.** Its doc comment says it is "built once per
slot-table change rather than per frame," with a paragraph explaining that
"per slot-table change is doing real work in that sentence." `FrameRenderer::render`
built a fresh one unconditionally — a `create_buffer`, a `write_buffer`, and a
`create_bind_group` on every clean frame. Not a gate failure (the work is
`O(slots)` either way, and it sits outside the gate's clock), but a documented
property the code did not have. `FrameRenderer` now holds the plan and rebuilds
it only when the slot table changes, with a `draw_plan_builds()` counter and an
assertion that a 20-frame steady loop leaves it at 1 — a counter rather than an
intention, in the style the rest of the phase uses.

**3. `render/device.rs`'s module doc contradicted the type six lines below it.**
It said Phase 4 "negotiates the three" indirect features, naming
`MULTI_DRAW_INDIRECT` among them, while `IndirectSupport`'s own doc — directly
underneath — explains at length that *there is no `MULTI_DRAW_INDIRECT` feature
in wgpu 30*, which is one of the two genuinely surprising findings this phase
recorded (§3). Two features are negotiated, not three; the doc now says so and
says why there is no third.

**4. `render/readback.rs` claimed the crate had no logger, in the same phase
that gave it one.** `log_dropped_readback`'s doc said "`wgpui-wgpu` has no
logging dependency yet … until the crate gains one," and printed to stderr. The
crate gained one in this phase's own commit `1e42831`. Both stderr
acknowledgements (this one and `frame.rs`'s frame-readback equivalent) now go
through `log::warn!`, and the doc says what actually happened.

A fifth correction came out of re-running the benchmark rather than reading:
**the fallback's readback cost was attributed to the wrong variable.**
`render/draw.rs` recorded it as "187µs at 8 slots and 1.71ms at 128 slots,"
which reads as a function of slot count. It is not. Both benchmark sweeps show
otherwise (§5): at a *fixed* 8 slots the readback climbs 446µs → 1.72ms as the
primitive count rises, and at a fixed primitive count it climbs 853µs → 6.40ms
as the layer count rises. The doc's own explanation was already right —
`Device::poll(wait_indefinitely)` waits for everything already submitted — but
its numbers framed the cost as slot-shaped when the mechanism it describes makes
it frame-GPU-work-shaped. Corrected, with both sweeps cited.

---

## 2. What shipped, and where

| File | Lines | Role |
|---|---|---|
| `wgpui-core/src/indirect.rs` | 581 | The slot table, the argument record, `FirstInstance`, and the CPU reference the WGSL transcribes |
| `wgpui-core/src/shaders/indirect_args.wgsl` | 189 | `clear_visible`, `compact`, `pack` — the transcription |
| `wgpui-core/src/scene.rs` | +148 | `Scene::draw_slots` (`draw_ranges`' successor) and `Scene::arena_slots` |
| `wgpui-core/src/boundary/compositor.rs` | +391 | `CompositeEntry`, `CompositeSource`, `ExternalSurfaceId`, `visible_composites` (R-N §8.1's layer tier), `BoundaryComposite::composite_entry` |
| `wgpui-core/src/test_support/ui_walk.rs` | +120 | `MultiLayerSceneDriver` — the same walk driven into several layers |
| `wgpui-wgpu/src/render/frame.rs` | 646 | The assembly point: upload, compute, arguments, issue, as a value a test can read |
| `wgpui-wgpu/src/render/draw.rs` | 595 | `DrawMode`'s four paths, `DrawStats`, `issue_quads`, `issue_composites` |
| `wgpui-wgpu/src/render/pipelines.rs` | 389 | `QuadPipeline` and `CompositePipeline` — two, not eight, and why |
| `wgpui-wgpu/src/render/compute/indirect_args_pass.rs` | 526 | The dispatch, plus `scatter`: the whole of the wiring from Phase 3's output |
| `wgpui-wgpu/src/render/device.rs` | +211 | `IndirectSupport` — feature negotiation, best-effort, reported off the device |
| `wgpui-wgpu/src/render/surface_registry.rs` | 682 | The legacy 772-line file, moved with four accounted-for differences (§4) |
| `wgpui-wgpu/src/render/textures/external_surface.rs` | 373 | §5.5's Gap 2: both producers through one `plan_composites` |
| `wgpui-wgpu/src/render/textures/layer_texture.rs` | 352 | `Retention::Texture`'s actual texture pool |
| `wgpui-wgpu/src/render/buffers/slab_buffers.rs` | 144 | The first real `wgpu::Buffer` behind Phase 1's upload instructions |
| `wgpui-wgpu/src/render/readback.rs` | +117 | `StagingReader` — the reused staging buffer the fallback needs |
| `wgpui-wgpu/src/render/shaders/quads.wgsl` | 118 | Instanced quads, pulling through the indirection buffer |
| `wgpui-wgpu/src/render/shaders/surfaces.wgsl` | 82 | The one composite shader both producers reach |
| `wgpui-wgpu/tests/indirect_args_differential.rs` | 577 | The WGSL/Rust differential |
| `wgpui-wgpu/tests/indirect_draw.rs` | 668 | Both gates, end to end |
| `wgpui-wgpu/tests/surface_registry_consumer.rs` | 365 | Gate 2's concurrency half |
| `wgpui-wgpu/examples/phase4_draw_issuance_bench.rs` | 240 | Gate 1, measured |

Two deviations from §3's file map, in the same shape the three previous phases
recorded theirs, and both argued in the files' own module docs:
`wgpui-core/src/indirect.rs` (§3.1 gives the shader a home and the computation
none, because in the legacy backend the computation *is* `quads_first_instance`
inline in `renderer.rs`'s draw loop), and `wgpui-wgpu/src/render/frame.rs` (§3.5
lists every stage and no home for the thing that runs them in order; §8's gate
is a claim about what a frame *did*, which is only checkable if a frame is a
value something returns). That is the third and fourth such deviation across
four phases.

---

## 3. The indirect-arg and draw-issuance design

### The slot table is a fact about residency, never about contents

§5.3 asks for "a **fixed** sequence of `draw_indirect`/`multi_draw_indirect`
calls — one per (layer, kind) slot that *could* be populated — every frame,
regardless of how many are actually zero."

`Scene::draw_slots` is that sequence, and it is `draw_ranges`' successor with
one difference that is the whole of the gate: `draw_ranges` reports how many
instances each slot draws; `draw_slots` reports only where each slot's
reservation lives. It reads one `SlabRange` per (layer, kind) pair and never
touches a primitive. Every live layer contributes a slot for every kind,
including kinds it holds nothing of — omitting the empty ones would reshape the
sequence the moment a layer gained its first glyph run, which is exactly the
per-frame CPU re-planning the phase exists to stop.

### The indirection buffer is arena-shaped, so no base is ever read back

Per-instance data is storage-buffer vertex pulling already (§1), so the shader
indexes its arena with `@builtin(instance_index)`. Culling removes an arbitrary
subset and ordering permutes what is left, and neither is a contiguous range —
so the pass writes an indirection buffer where `visible[i]` holds the arena slot
the *i*-th drawn instance reads. That buffer mirrors the arena exactly: slot
`(layer, kind)` owns `[base, base + count)` in it, the same range its
`SlabRange` owns in the arena. Three consequences, all load-bearing: every
slot's run base is the number the CPU already has (nothing is read back to learn
it), compaction is per-slot and needs no global coordination, and the buffer is
sized once alongside the arena.

Compaction is **order-preserving** — a chunked Hillis-Steele scan over 64 lanes
with a running offset between chunks — rather than an unordered atomic append.
An unordered append is shorter and destroys the painter order the Phase 3
relaxation just spent iterations computing. Unwritten entries hold
`UNUSED_INSTANCE = u32::MAX` rather than zero, because zero is a legitimate
arena slot and a shader reading past a slot's `instance_count` would then
silently draw primitive 0 many times over instead of producing something
obviously wrong; `quads.wgsl` degenerates such a vertex to zero area.

### `firstInstance` is a choice because of a bug this crate already hit

`README.md`'s "Custom Device Gotcha" records drivers silently dropping an
indirect draw whose `firstInstance` is nonzero without `INDIRECT_FIRST_INSTANCE`.
So `FirstInstance` is explicit: `Zero` (every record carries `0`, the base
reaches the shader through a per-slot uniform with a dynamic offset the CPU
already knows — no device feature at all, legal on WebGPU, and structurally
incapable of producing the input the bug triggers on) is the **default**;
`SlotBase` (the base rides in the record) is chosen only where
`INDIRECT_FIRST_INSTANCE` was actually negotiated, because a
`multi_draw_indirect` covers many records with no bind-group change between
them and has no other way to address per-entry ranges. One shader serves both:
it computes `slot_base + instance_index`, and which half is zero is the
encoding.

### Two things `wgpu = "30"` says that §5.3 does not

Both were found by writing `wgpu::Features::MULTI_DRAW_INDIRECT` and having it
not exist, and both change what the code can honestly claim:

1. **There is no `MULTI_DRAW_INDIRECT` feature.** `multi_draw_indirect` is
   always callable, and where the backend cannot do it natively wgpu *emulates
   it as a CPU-side loop of `draw_indirect` inside `wgpu-core`*. So calling it
   is never wrong and never fails — but it only stops being the same per-slot
   CPU loop when `MULTI_DRAW_INDIRECT_COUNT` is present, whose own
   documentation says exactly that. That single feature is therefore what
   decides whether multi-draw saves the CPU anything, and
   `IndirectSupport::supports_native_multi_draw()` requires it *and*
   `INDIRECT_FIRST_INSTANCE` — the second because without it a multi-draw is
   not merely pointless but wrong.
2. **README's gotcha is now wgpu's own rule.**
   `InstanceFlags::VALIDATION_INDIRECT_CALL`, on by default, turns an indirect
   draw with a nonzero `first_instance` into a **no-op** when
   `INDIRECT_FIRST_INSTANCE` is absent. The failure the README describes as a
   driver habit is reproducible on purpose at the API layer. That is the direct
   justification for `FirstInstance::Zero` being the default rather than an
   accommodation.

Feature negotiation is **best-effort on every platform**, deliberately weaker
than the legacy path's hard requirement outside macOS. Three reasons, all in
`device.rs`: Phase 3's principle that requesting a feature a phase cannot
exercise makes a working device fail to open; the fallback exists precisely so
a missing feature is a slower path rather than a failure, and a mandatory
feature would make it unreachable and therefore untested; and the per-slot path
needs none of it and is what a device reporting nothing takes — which is also
what WebGPU reports.

### Four modes, and the counter that carries the gate

`DrawMode` has `PerSlotIndirect` (default, no features), `MultiDrawIndirect`,
`MultiDrawIndirectCount`, and `CpuReadback` (§5.3's fallback for the macOS
best-effort case and for WASM, which are one piece of work per decision 2).
The fallback is also this crate's own **reference arm**: `tests/indirect_draw.rs`
renders every available mode and compares framebuffers against it bit-exactly. A
fallback nothing exercises is a fallback that does not work.

`DrawStats::instances_known_to_cpu` is an `Option<u32>`, and that is the design
decision the gate rests on. On every indirect path it is `None` — not zero, not
unknown-but-guessable: the CPU issued the draws without the number existing on
its side of the bus. §5.3's wording is "without the CPU ever finding out the
count," and an `Option` is what turns that into something a test asserts.
`DrawStats::merge` makes unknown contagious on purpose: a frame that took an
indirect path anywhere did not learn its own instance count, and reporting the
sum of the parts it *did* learn would be a smaller lie but a lie all the same.

**Coalescing (`OpenSlabRun`) is deliberately not ported.** In the legacy
renderer merging byte-contiguous same-layer runs is a real optimisation
*because* its ranges are CPU-computed and it can see that two abut. Here it is
neither possible (whether two slots' live instances abut is a fact about the
GPU's compaction the CPU deliberately does not have) nor needed (the sequence is
already one call per slot, which is what coalescing was reducing *to*, and
`MultiDrawIndirectCount` collapses a whole kind into one call — further than
coalescing ever got).

### The seam Phase 3 named, closed

`docs/phase-3-results.md` §2 stated it: "the compute passes write orders, a
draw permutation, and a keep mask; nothing yet consumes them into a draw call."
`IndirectArgsPass::scatter` is the whole of the wiring — one
`copy_buffer_to_buffer` per layer moving that layer's `OrderingOutput::draw_order`
and `OcclusionOutput::culled` into its own arena range. No readback, no
re-encode, no CPU walk over primitives. Phase 3's per-layer outputs stay
layer-local (`[0, count)`) on purpose: that is what lets them be *copied* into
place rather than rewritten.

---

## 4. The `WgpuSurface`/`SurfaceRegistry` unification, and what it left untouched

§5.5's Gap 2 is precise: "Two composite pipelines exist today for one
operation." In the legacy renderer `SurfaceContent::Wgpu(surface_id)` and
`SurfaceContent::Layer(layer_id)` are two separate ~180-line branches, each
fetching its own texture, building its own params buffer and bind group, and
issuing its own `pass.draw(0..4, 0..1)`.

### The device-free half

`wgpui-core::boundary::compositor` gained `CompositeEntry` — where a texture
lands, what clips it, its opacity and corner radius, whether its *source* is
opaque, and a content token — plus `CompositeSource`, which is the whole of the
distinction between the two producers as far as the compositor is concerned.
`source_is_opaque` is **never inferred**: a boundary's baked texture is
transparent wherever its content is, and an external producer's contents are not
the framework's to know at all (§5.5: "its pixel *content* is never part of the
CPU description"), so an entry occludes only when its producer says it does, and
the conservative constructor says `false`.

`visible_composites` is R-N §8.1's **layer tier**, and it is deliberately CPU
work — `crate::occlusion`'s own module doc already said why ("tens of items, not
tens of thousands — so it is not a compute problem") and already named
`coverage::fully_covered` as "the routine it will reuse when the compositor
grows a per-layer opaque region to feed it." This is that compositor growing it,
calling exactly that routine rather than a second copy of the rule. It is
conservative the same two ways the instance tier is: an entry clipped to nothing
is kept, and only the first `MAX_OCCLUDERS` qualifying occluders are considered,
which can only ever *miss* a cull.

### The device half

`render/textures/external_surface.rs` is one function, `plan_composites`, and
the difference between the two producers survives in exactly one expression —
`CompositeConsumer::view`'s `match`. Everything after it (the parameter block,
the bind group, the argument record, the draw call, the shader) is common code
that cannot tell them apart. `render/textures/layer_texture.rs` is where
`.boundary()`'s `Retention::Texture` stops being a decision:
`docs/phase-2-results.md` §7 said explicitly that Phase 2 owed a decision and
Phase 4 owed the texture. It is `LayerTextureEntry` ported with both of its
load-bearing rules kept — the content *token* is compared rather than the
pixels, and eviction returns the evicted boundaries so the obligation to re-bake
is a return value rather than a side channel.

### What is untouched, verified by reading rather than assumed

§9's risk table names this phase's specific failure mode: that the unification
"accidentally touches `SurfaceRegistry`'s cross-thread producer-side
synchronization (`gpu_submit_lock`, atomic generation tracking, GPU-synced
buffer swaps) — hard-won, carefully-documented concurrency code that has nothing
to do with the bug being fixed."

**`render/surface_registry.rs` is the legacy file copied, with exactly four
differences.** That claim is in its module doc, and this pass checked it the
only way worth checking it — by running `diff -u
src/platform/cross/surface_registry.rs
crates/wgpui-wgpu/src/render/surface_registry.rs` and accounting for every hunk.
There are four, and they are the four the doc lists:

1. The module doc itself.
2. The `#[cfg(feature = "flamegraph")]` members are absent —
   `front_texture_snapshot`, `memory_usage`, `SurfaceTextureSnapshot`, and the
   `flamegraph_tests` module. They call `super::render_context::texel_size` /
   `texture_memory_bytes`, which live in the legacy crate; §3.6/§8's Phase 7 is
   what moves devtools.
3. `SurfaceId` gains `from_raw`/`as_raw`. The field itself is unchanged and
   still `pub(crate)`; the legacy consumer reaches in as `surface_id.0` from a
   sibling module, and a consumer in another crate needs a name for the same
   thing.
4. `impl Default for SurfaceRegistry`, calling `new`, which `--deny warnings`
   requires.

The diff contains **nothing else**. The atomic state packing, both
compare-exchange swap loops, the generation gating, the backpressure rule, the
resize skip-while-compositing guard, every `Ordering` on every load and store,
and all six model tests are byte-identical. The six tests came across with the
file, were not touched, and pass.

**`gpu_submit_lock` and `submit_guard` were not ported at all**, which is
stronger than "untouched": they live in `src/elements/wgpu_surface.rs` and
`src/platform/cross/render_context.rs`, and `git diff origin/2.0..HEAD -- src/`
is empty. Nothing in `wgpui-wgpu` names either of them (checked by grep, not by
recollection). The consumer side calls exactly two registry methods, in the
order the legacy surfaces batch calls them —
`swap_ready_display_if_new` then `front_view` — and nothing else: not
`swap_rendering_ready`, not `present_synced`, not `resize`, not
`set_redraw_pending`, not even `has_unconsumed_frame`.

**One honest note about the gate's own wording.** §8's Phase 4 row asks that
"`WgpuSurfaceHandle`'s existing concurrency tests (`submit_guard`, backpressure
via `has_unconsumed_frame`) pass unmodified against the unified consumer path."
`src/elements/wgpu_surface.rs` has **no tests at all** — `submit_guard` has never
had test coverage in this repository, so there is no such existing test to pass
or fail. The existing tests that *do* exist are `surface_registry.rs`'s six
model tests, and those are what §6 reports. This is recorded so nobody reads the
gate's phrasing as evidence that a `submit_guard` test ran.

### The one behavioural change, and it is the point

`plan_composites` runs the layer tier **first** and fetches nothing for an entry
it drops. So a covered external surface's view is never fetched, no bind group
is built for it, and — the observable form — its producer's frame is never
consumed. §5.5 promises "a 3D viewport fully covered by a modal stops being
drawn at all, which it cannot today"; this is what that costs on the consumer
side, and §6 asserts it so it cannot be mistaken for a broken consumer.

---

## 5. Gate #1 — a clean window's draw issuing is O(layer slots)

> §8's wording: *A clean window's CPU-side draw-issuing work is O(layer slots),
> independent of resident primitive count, measured directly (same
> `render_stats`-style counters R-N used, ported into `wgpui-core`).*

### The structural half

`render/draw.rs` is written so the claim is structurally true rather than true
by measurement luck: nothing in it reads a primitive, a record, an upload, or a
count. `QuadDrawPlan` is built from a `SlotTable`, which is built from each
layer's `SlabRange`, and issuing is a loop over slots whose body is a constant
number of calls. Two `wgpui-core` tests state the same claim where no device is
needed and it cannot be skipped: two scenes with the same four layers and a
10,000× difference in primitive count produce slot tables of the *same length*,
and a slot names a reservation and never an instance count.

### The measured half — counters

`tests/indirect_draw.rs::gate_1_a_clean_windows_draw_issuing_work_is_independent_of_primitive_count`,
on the adapter named in §7. Two scenes over the same six layers, one frame dirty
to do the uploads and compute, then clean frames. Every draw-issuing counter is
asserted **equal**, not similar. Swept across every mode the device allows:

| mode | 290 primitives | 18,026 primitives | draw calls | binds | best draw-issue |
|---|---|---|---|---|---|
| per-slot `draw_indirect` | 6 slots | 6 slots | 6 → 6 | 7 → 7 | 1.5µs → 0.7µs |
| `multi_draw_indirect` | 6 slots | 6 slots | 1 → 1 | 2 → 2 | 0.7µs → 0.4µs |
| `multi_draw_indirect_count` | 6 slots | 6 slots | 1 → 1 | 2 → 2 | 0.4µs → 0.2µs |
| CPU readback + direct draw | 6 slots | 6 slots | 6 → 6 | 7 → 7 | 1.0µs → 1.0µs |

The per-slot row is the load-bearing one: it is the mode a featureless device
takes, it is the one that *cannot* satisfy "the same at both counts" by
collapsing to a single call, and the test asserts separately that it issues
exactly `LAYERS` calls. Alongside: a clean frame recomputes zero layers'
ordering or occlusion, `instances_known_to_cpu` is `None`, and `readback_words`
is 0. The slot table itself is asserted to name every (layer, kind) pair —
`LAYERS × PrimitiveKind::COUNT` — including the `GlyphRun` slots nothing draws.

The mirror-image test is what makes that counter mean something: the
CPU-readback fallback *does* report a count (`Some(290)`), *does* read words
back, and is the only mode that can decline to issue a call for an emptied
layer's slot (`slots_skipped == 1`) — the one thing an indirect path cannot do,
because it does not know.

### The measured half — clocks

`examples/phase4_draw_issuance_bench.rs`, same adapter, 2 warm-up runs
discarded and disclosed, 12 timed, median and best both reported (lower middle
for an even count, so every printed number was actually observed). Timed: CPU
command encoding of the fixed draw sequence — set pipeline, set bind group,
issue draw. Excluded and named rather than quietly omitted: pipeline
construction, uploads, the compute passes, and argument generation, every one of
which runs only on a frame where something changed.

**Sweep A — 8 layers, primitive count rising 400×. The gate's line, and it is
flat.**

| primitives | slots | calls | per-slot median/best | multi-draw-count median/best |
|---|---|---|---|---|
| 256 | 8 | 8 / 1 | 1.80µs / 1.40µs | 0.50µs / 0.30µs |
| 2,560 | 8 | 8 / 1 | 2.30µs / 1.80µs | 0.50µs / 0.40µs |
| 25,600 | 8 | 8 / 1 | 1.70µs / 1.40µs | 0.60µs / 0.40µs |
| 102,400 | 8 | 8 / 1 | 1.60µs / 1.30µs | 0.40µs / 0.30µs |

**Sweep B — ~25,600 primitives, layer count rising. This line should rise, and
does.**

| layers | slots | per-slot median/best | multi-draw-count median/best |
|---|---|---|---|
| 2 | 2 | 0.50µs / 0.40µs | 0.60µs / 0.40µs |
| 8 | 8 | 1.30µs / 1.10µs | 0.60µs / 0.40µs |
| 32 | 32 | 3.50µs / 3.20µs | 0.80µs / 0.70µs |
| 128 | 128 | 8.40µs / 7.60µs | 0.50µs / 0.30µs |

Both sweeps exist because either alone answers half the question. Flat is the
gate; rising is what makes the gate a claim about a cost rather than about its
absence — a benchmark showing only the flat line would be equally consistent
with the cost being zero, which it is not. The per-slot path rises roughly
linearly in slots and is independent of primitives, which is precisely
"O(layer slots), not O(resident primitives)." The multi-draw path is flat in
*both* sweeps, because one call covers the whole kind.

**The column nobody asked for, and the most useful thing in the table: the
fallback's price.** The benchmark reports the readback clock beside the
draw-issue one, and the gap is three to four orders of magnitude — 446µs at the
smallest case, 6.40ms at the largest, against 0.3–18µs of actual draw issuing.
It is not the 2KB of argument records, and it is not the slot count:
`Device::poll(wait_indefinitely)` waits for *everything already submitted*, so
reading the arguments back also waits for the compute dispatches that wrote
them and for the previous frame's rendering to drain. Both sweeps show that
directly — at a fixed 8 slots the readback still climbs 446µs → 1.72ms with the
primitive count, and at a fixed primitive count it climbs 853µs → 6.40ms with
the layer count. §5.3 describes this path as the WASM path and the macOS
best-effort path without pricing it. It is a correct path and a slow one: it
does not merely add a copy, it serializes the CPU against the whole frame's GPU
work once per frame. Any device that has `draw_indirect` at all — which is every
WebGPU device — should be taking `PerSlotIndirect` instead, and
`DrawMode::best_available` never returns `CpuReadback`.

Timing caveats, stated rather than glossed: these are sub-microsecond to
low-microsecond numbers on a shared laptop, so individual cells are noisy —
Sweep A's per-slot median wobbles non-monotonically, and Sweep B's 128-layer
multi-draw cell reads faster than its 32-layer cell. The counters, not the
clocks, are what the gate is asserted on; the clocks are corroboration.

### The correctness claim both gates rest on and neither states

`every_draw_mode_renders_the_same_picture`: all four available modes render
**bit-identical** framebuffers (153,600 painted pixels, first difference
`None`), with a guard that each mode painted more than an eighth of the window
so the comparison is not between two blank images. A gate about how cheaply the
CPU issues draws is worth nothing if the draws differ.

### And the transcription is checked, Phase 3's discipline unchanged

`tests/indirect_args_differential.rs`: the compute pass's argument records **and
its indirection buffer** equal `wgpui_core::indirect::indirect_args`'s exactly,
in both `first_instance` encodings, over the scripted UI walk. Comparing the
buffer as well as the records is deliberate — a compaction that produced the
right *counts* from the wrong *instances* would draw the wrong primitives in the
right quantity, which a count-only comparison misses. Across the walk: 156 slot
records, 78 of them empty (so §5.3's "regardless of how many are actually zero"
is exercised, not just asserted), 3,527 instances drawn. The multi-layer harness
compacted only **1** instance away across the whole walk, which is why the
single-layer variant exists and asserts `> 100`: instance-tier occlusion is
scoped per layer (R-N §8.2), so splitting a scene across more layers *reduces*
how much of it is culled. The single-layer arm compacted 189 away — that is
where an off-by-one in the running offset between chunks would show up.

---

## 6. Gate #2 — a covered viewport, and the concurrency evidence

> §8's wording: *a viewport panel fully covered by a modal (occlusion-culled per
> §5.2/Phase 3) issues zero draws for its embedded 3D content, and
> `WgpuSurfaceHandle`'s existing concurrency tests … pass unmodified against the
> unified consumer path.*

### The covered viewport

`gate_2_a_covered_viewport_issues_no_draws_and_consumes_no_produced_frame`. A
real `SurfaceRegistry` surface, a producer presenting a frame through the
untouched producer path, and a texture-retained boundary with a real baked
texture as the modal — so both composite entries are real and the only
difference between the two cases is the layer tier's decision.

**Covered** (modal over the whole window): 2 entries considered, 1 culled, 0
unavailable, **1 composite draw issued** — only the modal. And
`registry.has_unconsumed_frame(surface)` is still `true`: the covered viewport
did not consume the frame its producer presented, which is what "stops being
drawn at all" means on the consumer side.

**Uncovered** (the same modal moved clear): 0 culled, 2 draws issued, and
`has_unconsumed_frame` is now `false` — a viewport that is actually composited
consumes the frame, exactly as the legacy surfaces batch does. Moving the modal
restores both, so the culled case is measuring an absence rather than a path
that never worked.

A separate control test proves the same for a boundary texture at the pixel
level: an uncovered entry paints exactly its own 128×128 rectangle (16,384
pixels), and covering it removes exactly those pixels.

### The concurrency half — three layers of evidence

§9's risk table asks that the existing tests be the gate, "not a new test suite
reverse-engineered from the concurrency doc comments." So:

1. **The six existing model tests pass unmodified.** They came across with the
   file (§4) and were not touched: `should_composite_swap_only_on_new_generation`,
   `indices_stay_a_permutation_across_swaps`,
   `ungated_compositor_regresses_to_stale_frame`,
   `gated_compositor_holds_latest_frame_on_unpaired_paints`,
   `gated_compositor_tracks_new_frames`,
   `gated_compositor_shows_latest_when_producer_outruns_compositor`.
2. **A differential.** The same 13-step producer/consumer script — a paired 1:1
   run, a producer outrunning the consumer, a consumer painting with nothing
   new, and frames where the surface is not drawn at all — driven through the
   legacy consumer sequence (`swap_ready_display_if_new` then `front_view`,
   spelled out as `renderer.rs` spells it) and through `plan_composites`, with
   `(frame_generation, has_unconsumed_frame)` sampled after every step. **All 13
   steps identical.** This is the direct form of "unaffected": not an argument
   that the code was not edited, a measurement that its behaviour did not
   change. The script is guarded to actually reach a backpressure state and
   actually clear it again, so it cannot pass by never exercising the property.
3. **A real cross-thread run.** A producer thread following the backpressure
   loop `has_unconsumed_frame`'s own documentation prescribes ("skip producing
   while it returns true"), against the unified consumer: **200 composites, 172
   frames produced, 6,191 production attempts skipped for backpressure**,
   generation monotonic throughout, and the consumer asserted not to have
   stalled. Backpressure is asserted to have actually engaged.

A fourth test pins the producer API as a compile-time fact rather than a diff
reading: the five calls an external render thread makes, in the order
`WgpuSurfaceHandle`'s own doc example makes them, still compile and behave
(`lock_and_get_back_with_size`, `back_view`, `swap_rendering_ready_no_sync`,
`set_redraw_pending`/`get_pending_surfaces`/`clear_redraw_pending`, `resize`,
`remove`).

And the one behavioural difference is asserted so it cannot be mistaken for a
regression: `a_culled_entry_leaves_the_producers_frame_unconsumed` shows that a
culled entry leaves the registry's observable state *byte-for-byte where it
was* — not the generation, not the composited generation, nothing.

---

## 7. GPU adapter honesty check

Per Phase 0's discipline, checked before any number above was trusted:
`cargo run -p wgpui-wgpu --release --example adapter_probe`.

```
== Adapters enumerated on VULKAN | DX12 ==
  name="NVIDIA GeForce RTX 4060 Laptop GPU" backend=Vulkan device_type=DiscreteGpu driver_info="561.03"
  name="NVIDIA GeForce RTX 4060 Laptop GPU" backend=Dx12   device_type=DiscreteGpu driver="32.0.15.6103"  (x3)
  name="Microsoft Basic Render Driver"      backend=Dx12   device_type=Cpu
== Picking the first adapter and requesting a device ==
  Selected: NVIDIA GeForce RTX 4060 Laptop GPU (Vulkan, DiscreteGpu)
  software/CPU-fallback adapter: no
  request_device: OK
```

Every test and benchmark in this phase reports its adapter *and its negotiated
features* before printing anything:
`[INDIRECT_FIRST_INSTANCE=true MULTI_DRAW_INDIRECT_COUNT=true]`. **This is the
same machine Phase 0 and Phase 3 ran on** — verified against
`docs/phase-3-results.md` §6 rather than assumed — so its caveats carry over
verbatim and are not restated as if re-checked: one adapter, one driver version,
one OS, one laptop-class discrete NVIDIA part.

**A breadth caveat this phase adds that Phase 3 did not have.** This adapter has
both indirect features, so `DrawMode::best_available` picks
`MultiDrawIndirectCount` and every mode is available to sweep. That is the
*best* case for this phase and the worst case for coverage: the featureless
path, the macOS best-effort path, and WASM are all exercised here only by
`PerSlotIndirect` and `CpuReadback` running on a device that did not need them.
Nothing here says the fallback works on a device that actually lacks the
features — only that it produces the same picture on one that has them.

---

## 8. Check, test, and clippy status

```
cargo check -p wgpui-core -p wgpui-wgpu -p wgpui-widgets --all-targets  → clean
cargo test  -p wgpui-core  --release                                    → 279 passed, 0 failed
cargo test  -p wgpui-wgpu  --release                                    → 25 + 5 + 8 + 5 + 4
                                                                          = 47 passed, 0 failed
cargo test  -p wgpui-widgets --release                                  → 6 passed, 0 failed
                                                                          (untouched by this phase)
cargo metadata --locked                                                 → OK (see §1, item 1)
cargo clippy -p wgpui-core -p wgpui-wgpu -p wgpui-widgets --all-targets -- --deny warnings
                                                                        → clean, cold build,
                                                                          zero suppressions
```

332 tests total across the three crates (326 in the two this phase touched). No
test is skipped on this machine:
every GPU-dependent test goes through `device::context_or_report`, which prints
the adapter it got or prints a plain SKIPPED line saying which half of the gate
did not run — never silence that could pass for coverage. A dedicated test
(`a_missing_adapter_is_reported_rather_than_passing_silently`) guards that
behaviour itself.

`cargo test --workspace` was **not** run: it pulls in `gpui-ce`'s legacy suite,
which Phase 1 recorded as running 11+ minutes without completing and which no
2.0 branch modifies. Tests are scoped to the crates this phase touches, and
`git diff` confirms nothing outside `crates/` changed except one `Cargo.lock`
line and this file.

Clippy was re-run cold (`cargo clean -p` on all three crates first) rather than
trusted from the branch's own "fixed rather than suppressed" commit. That commit
is real — `issue_composites` took nine arguments, three of which were the fields
of one `CompositePlan`, and taking the plan is both what gets the signature
inside the limit and what stops a caller pairing one frame's prepared entries
with another frame's culled count; same class of finding and same resolution as
Phase 3's `data_group`. Note that `AGENTS.md` prefers `./script/clippy`, which
adds `--release --all-features` and runs `cargo-machete`/`typos` when installed;
the command above is the narrower one the phase brief specified, so the
release-profile and all-features variants of these lints have not been run.

---

## 9. Gate assessment — honest read

**Gate #1 — draw issuing is O(layer slots), independent of resident primitive
count: met.** Asserted three ways that do not depend on each other: structurally
(nothing in the issuing path reads a primitive), by counter (every draw-issuing
counter equal at a 62× primitive difference, across all four available modes,
with `instances_known_to_cpu == None` on every indirect path), and by clock (a
flat Sweep A and a rising Sweep B on named hardware). The counters are the
proof; the clocks are corroboration, and are noisy enough at this scale to be
worth reading as corroboration only.

**Gate #2 — a covered viewport issues zero draws, and `SurfaceRegistry`'s
concurrency is unaffected: met.** The covered viewport issues zero composite
draws and does not consume its producer's frame, with the uncovered case as a
live control and a pixel-level control beside it. The concurrency clause is met
in the form §9's risk table asked for — the existing tests, unmodified, plus a
step-for-step differential showing identical observable registry state, plus a
real cross-thread run. **With one wording caveat, in §4:** the gate names
`submit_guard` tests that do not exist anywhere in this repository, so what
"existing concurrency tests pass" means here is the six `SurfaceRegistry` model
tests, not a `submit_guard` test. A reviewer who reads the gate's parenthetical
as requiring a `submit_guard` test should treat that clause as satisfied by the
stronger structural fact instead: `gpu_submit_lock` and `submit_guard` were not
ported, are not referenced anywhere in `wgpui-wgpu`, and live in a directory
this branch's diff does not touch.

**Both of §8's Phase 4 clauses are therefore satisfied.** Three things are worth
recording as genuinely open rather than met:

- **Every measurement is from a device that has both indirect features.** The
  fallback and the featureless path are exercised, and produce bit-identical
  pictures — but on hardware that did not need them (§7).
- **The unification is proven at the seam, not through a running application.**
  There is no window loop yet; `FrameRenderer` is the assembly point a test can
  drive, and nothing yet builds `CompositeEntry`s from real elements (§10).
- **One machine.** Identical to Phase 0's and Phase 3's caveat, unimproved.

---

## 10. What is open for later phases

Named here so none of it is rediscovered as a surprise, and separated into what
is somebody else's phase and what is a loose end of this one.

### Already scheduled, and correctly not done here

1. **Tile-based buffering is Phase 4.5.** `Buffering::Tiled` still reports
   `is_implemented() == false`. Phase 4.5 extends `LayerKey` to
   `(boundary, TileCoord)` and drives indirect-arg generation from a
   tile-visibility pass — reusing this phase's mechanism directly, which is why
   it sits immediately after it.
2. **Text, `Img`, and `StyledText` `diff_key` are Phase 5.** Concretely visible
   here: `Scene::draw_slots` names every layer's `GlyphRun` slot, the compute
   pass generates its argument record, the differential asserts those records
   are correct — and **nothing issues their draws**, because
   `render/pipelines.rs` builds one instanced pipeline and glyph runs have no
   shader yet. `FrameRenderer::render`'s doc says so at the top rather than
   leaving it to be found as a missing draw call.
3. **The regular-content layout kernel is Phase 6.1**, and is a rescoped
   *spike*, not a build — Phase 0's Spike B measured a standalone dispatch
   losing by ~1000×. Phase 3's results doc noted the follow-up (fused-dispatch)
   spike is now answerable; Phase 4 does not change that either way.

### Loose ends of this phase

4. **Nothing drives `LayerTexturePool::acquire`/`sweep` in the frame loop.**
   `FrameRenderer::render` calls `begin_frame()` and then reads the pool as a
   *texture source* for composite entries; it never acquires, never bakes, and
   never sweeps. The pool's own tests and gate 2's test acquire manually. So
   `Retention::Texture` now has a real texture and a real eviction policy, and
   still has no producer inside a frame — the rasterize-to-texture step belongs
   with whatever phase brings a real window loop. No gate asked for it; it is
   named because "Phase 4 built the texture pool" could otherwise be read as
   more than it is.
5. **Per-frame allocations remain in the argument stage.**
   `IndirectArgsPass::run` creates a params buffer, a slot buffer, and a bind
   group on every call (twice per frame), and `plan_composites` creates a params
   buffer and a bind group per visible composite entry per frame. Both are
   outside the gate's clock and both are `O(slots)`/`O(entries)`, so neither
   threatens the gate — but the phase's own argument for caching `QuadDrawPlan`
   (§1, item 2) applies to them too, and they were left alone rather than fixed,
   to keep this pass's diff to what it could verify.
6. **Phase 3's Scene A occlusion loss is only half-addressed.**
   `docs/phase-3-results.md` §10 asked Phase 4 to revisit it: "the GPU currently
   pays upload and dispatch for 99k primitives to cull 1,296 of them; indirect
   draw is what makes the CPU stop paying to learn that answer." The CPU has
   indeed stopped paying to learn it (`instances_known_to_cpu == None`). The
   *upload and dispatch* cost of running occlusion over a scene where 3% is
   visible is completely unchanged, and this phase did not measure it. §9's risk
   table already lists that as unmitigated with two candidate directions; it
   still is.
7. **`--locked` was broken on this branch and nothing caught it** (§1, item 1).
   Worth a CI job: `cargo metadata --locked` or `cargo check --locked` would
   have failed on every one of the seven commits.

### What a human should do before calling this closed

1. **Run both gates on hardware without the indirect features** — ideally an
   integrated GPU, macOS/Metal (the actual best-effort case §5.3 names), and a
   WASM/WebGPU build. `cargo test -p wgpui-wgpu --release` and
   `cargo run -p wgpui-wgpu --release --example phase4_draw_issuance_bench`.
   Both name their adapter and its features; neither will silently report a
   software rasterizer's numbers as hardware. This is the single largest gap in
   the phase's evidence: the fallback exists *for* devices this machine is not.
2. **Re-run the benchmark on a machine that is not also running a desktop.**
   The draw-issue clocks are sub-microsecond and visibly noisy; the counters are
   not, and the gate rests on the counters.
3. **Let `gpui-ce`'s legacy suite finish once** — outstanding since Phase 1,
   still not something any 2.0 branch modifies.
