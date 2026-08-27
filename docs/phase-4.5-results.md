# Phase 4.5 Results — `Buffering::Tiled`, Tile Visibility, and Resident-Tile Eviction

Status: **Phase 4.5 executed, the gate met — with one clause met exactly and a
third layer disclosed beside it.** This documents what was built, what the gate
actually asserts, what was measured on real hardware, what verification found
broken and fixed, and what a human should still treat as open. It follows
`docs/gpu-native-architecture.md` ("2.0" below) §4.3 and §8's Phase 4.5 row, and
§9's risk table, which names this phase's specific failure mode. Work lives on
branch `wgpui-2.0/phase-4.5-tiled-buffering`, pushed to origin, not merged, no
PR.

**Nothing under `src/` changed.** `git diff origin/2.0..HEAD -- src/` is empty,
checked by running it. The whole branch diff touches 11 files, all under
`crates/`, plus this one. No dependency was added, so `cargo metadata --locked`
exits 0 — Phase 4's lockfile trap was checked for rather than assumed absent.

**Contents:** §1 What shipped, and where · §2 §4.3's claim, tested · §3 The
tile-visibility design · §4 The multi-tile content rule, and the measurement that
changed it · §5 Eviction and the resident-tile budget · §6 The gate · §7 Tile
size, measured · §8 What verification found · §9 GPU adapter honesty · §10 Check,
test, and clippy status · §11 Gate assessment — honest read · §12 What is open

---

## 1. What shipped, and where

| File | Lines | Role |
|---|---|---|
| `wgpui-core/src/scene/tile.rs` | 1,275 | `TileGrid`, `TileSpan`, `TilePlacement`, `TileResidency`, `tile_visibility` — the grid, the predicate, the placement rule, the budget |
| `wgpui-core/src/shaders/tile_visibility.wgsl` | 97 | The predicate, transcribed |
| `wgpui-core/src/boundary/compositor.rs` | +146 | `Compositor::visit_tiled`, `TiledVisit`, and the `sweep` fix (§8) |
| `wgpui-core/src/boundary/policy.rs` | +123 | Both Phase 2 placeholders closed; `resident_tile_budget` |
| `wgpui-core/src/test_support/ui_walk.rs` | +691 | `TiledCanvasDriver`, `NodeGraphSpec`, `TiledFrameStats` — the gate's harness |
| `wgpui-wgpu/src/render/compute/tile_visibility_pass.rs` | 429 | The dispatch, and the route into Phase 4's argument generation |
| `wgpui-wgpu/src/render/compute/indirect_args_pass.rs` | +96 | `run_with_slots` — the seam a GPU-written slot table enters through |
| `wgpui-wgpu/tests/tile_visibility.rs` | 396 | The WGSL/Rust differential, end to end into draw arguments |
| `wgpui-wgpu/examples/phase45_tiling_bench.rs` | 301 | §4.3's tile-size tradeoff, measured |

Both Phase 0 placeholders named in §3's file map are now real:
`shaders/tile_visibility.wgsl` was a one-line comment and
`render/compute/tile_visibility_pass.rs` was three lines of module doc.

**No deviation from §3's file map this phase**, which is the first time across
six phases — Phase 1 recorded six, Phase 2 two, Phase 3 four, Phase 4 two. §3
gives `tile.rs` to "`TileCoord`, `(boundary, TileCoord)` addressing (§4.3)" and
everything built here is that module growing into its own remit. `TiledVisit`
went into `boundary/compositor.rs` because it is the per-frame compositing
decision that file already owns.

---

## 2. §4.3's claim, tested rather than trusted

§4.3 says tiling "turns out to need almost no new machinery," and the brief for
this phase asked that the claim be verified by reading what exists rather than
assumed. It is **substantially true, with one qualification worth stating.**

**True, and load-bearing.** Phase 1's groundwork is real and was read before
being relied on: `scene/tile.rs` shipped `TileCoord` signed-on-purpose with
tests, `LayerKey { boundary, tile: Option<TileCoord> }` already carried the
address, and `LayerKey::tiled` already existed commented "§4.3, Phase 4.5".
**`LayerKey` did not change in this phase at all** — that is the concrete
evidence the bet paid, and it is why a tile could be a `Layer` rather than a new
kind of thing. Every one of §4.3's four bullets held up:

- A tile *is* a `Layer`. `LayerTable::insert` starting a new layer at
  `Invalidation::all()` is what makes a newly-revealed tile dirty; nothing in
  this phase marks a tile dirty, and `Compositor::visit_tiled` creates no layer,
  allocates no slab, and raises no invalidation.
- Panning is `TRANSFORM`-only via `LayerTable::set_transform`, live since Phase 2
  and used here unmodified.
- The visibility computation is genuinely small — 97 lines of WGSL, one
  invocation per tile, no new pipeline.
- Eviction is R-N §3.4's mark-and-sweep with "unvisited" reinterpreted spatially.

**The qualification: §4.3's "the CPU never enumerates tile candidates" is true of
the draw path and not of residency, and cannot be.** A newly-revealed tile needs
content rendered into it, which is CPU work no visibility kernel can do. So the
predicate exists twice — once in `TileResidency` deciding what stays resident,
once in `tile_visibility.wgsl` deciding what draws — and they are checked against
each other rather than assumed to agree (§3). That is not a defect in the design;
it is a sentence in §4.3 that reads as stronger than what the mechanism can
deliver, and the code says so where a reader will hit it.

---

## 3. The tile-visibility design

### The pass writes Phase 4's slot table, and that is the whole integration

`tile_visibility.wgsl` writes one `vec4<u32>` per resident tile:
`[base, count, 0, 0]`. That is not a convention this phase invented — it is
byte-for-byte `wgpui_core::indirect::encode_slots`' layout, the input
`indirect_args.wgsl`'s `compact` has read since Phase 4, asserted identical by a
test in `scene/tile.rs` that encodes the same records both ways and compares the
bytes.

So an out-of-range tile is a slot with a zero count. Phase 4's `compact` turns
that into a zero-instance argument record; `pack` drops it from a
`multi_draw_indirect_count` entirely. **There is no tile draw path, no tile
pipeline, and nothing new deciding what draws.** A tile stops being drawn because
the existing mechanism finds nothing in it to draw.

`IndirectArgsPass::run_with_slots` is the seam. The existing `run` became a thin
wrapper that uploads a CPU slot table and calls the same dispatch, so the
GPU-driven path and the CPU-driven one are the same code below the slot buffer.

**One consequence, handled rather than dropped.** `run` validates each slot's
reservation against the arena before dispatching, and a GPU-written table cannot
be read without exactly the readback this path exists to avoid. So the check
moved onto the tile descriptors the CPU still writes, in
`TileVisibilityPass::run`. Dropping it would hand the compaction an out-of-range
base, which is an uncaptured device error and by default aborts the process.

### The dilation is exact, which is what lets the two halves agree

The shader dilates the viewport *rectangle* by `retain_radius × tile_size`;
`TileResidency` dilates the tile *span* by `retain_radius` coordinates. These
select the same tiles exactly, because tile edges sit at exact multiples of the
tile size, so `floor((min − r·w)/w)` *is* `floor(min/w) − r`. A test sweeps 40
viewport positions × 3 radii and asserts the two agree per tile rather than
trusting the algebra.

The edge rule is `Rect::intersects`, strict on every side — a region ending
exactly on a tile boundary does not reach into the next tile, the same strictness
every other predicate in this crate uses.

### The differential

`tests/tile_visibility.rs`, on the adapter named in §9, four tests, **all four
confirmed running on the GPU rather than skipping** (checked with `--nocapture`,
not inferred from a pass):

1. **The transcription is exact.** 31 pan steps over a 49-tile resident set
   spanning both signs of both coordinates, compared slot-for-slot and
   flag-for-flag against `scene::tile_visibility`. Guarded against passing
   vacuously: a script where every tile was always in range would agree perfectly
   and prove nothing, so the test asserts both answers actually occurred.
2. **The arguments generated from the GPU-written table** equal what
   `indirect::indirect_args` computes for the same tiles — records *and* the
   indirection buffer — with no slot table reaching the CPU in between. This is
   what makes the integration a reuse of Phase 4 rather than a lookalike.
3. **An out-of-range tile draws zero instances and leaves its whole indirection
   range at `UNUSED_INSTANCE`**, asserted separately, because a wrong count with
   live instances behind it would draw the wrong thing in the right quantity.
4. **Bookkeeping bugs are refused**: a tile reserving past the arena, a malformed
   descriptor table, more tiles than the buffers hold.

Negative tile coordinates are exercised deliberately — they are bit-cast through
a `u32` on the way to the shader, which is precisely where a sign would be lost.

---

## 4. The multi-tile content rule, and the measurement that changed it

§4.3 requires *a* rule and offers two: clip a spanning primitive into each tile,
or put it on an unbuffered overlay layer, "the same named pattern SFD §2 already
proposes," with the instruction to "reuse that pattern rather than inventing a
second one."

**Clipping is not available in this phase.** It needs a per-primitive clip
rectangle, and `Quad` does not have one — `docs/phase-1-results.md` §2 already
recorded that absence. Clipping *geometrically* instead is a different operation:
shrinking a rounded, bordered quad's rectangle moves its corners and border
inward, so a node body straddling a tile edge would render as two
differently-shaped halves. Not merely more work — not yet expressible, and the
expressible version is wrong.

**So: the overlay. But the obvious form of it does not survive its own gate.**

"Any primitive touching two tiles goes on the overlay" was built first. On the
node-graph workload — 130×70 nodes on a 256px grid — a node straddles a tile edge
for most origins, and the crossing gate measured **73% of the scene's primitives
on the unbuffered layer: 144 overlay primitives written against 52 tile ones.**
The mechanism was working and buying almost nothing, because the layer that is
*not* tile-culled held most of the content.

**The rule as shipped:** a primitive no larger than a tile is **anchored** to the
tile holding its top-left corner, overhang and all. Only genuinely oversized
content — a wire spanning several columns, a group box around a subgraph —
reaches the overlay. This is still the overlay pattern rather than a second one:
the overlay exists, it holds what cannot be anchored, and it is
`LayerKey::untiled`, the same layer an untiled boundary already has. What changed
is which content needs it.

**Anchoring's one obligation, checkable rather than remembered.** An anchored
primitive reaches at most one tile past its own, so a tile must stay resident
while its overhang is on screen. `TileGrid::overhang_is_covered(retain_radius)`
requires a radius of one or more, and a test shows that at radius zero the
overhang genuinely does go missing along the leading edge — so the constraint is
demonstrated, not asserted.

**What the overlay costs now, measured across a distribution rather than one
frame.** The single crossing the gate samples wrote **nothing** to the overlay,
which would be a misleading number to publish alone — it means no oversized wire
entered the visible region on that frame, not that the overlay is free. Over
twenty consecutive crossings: **70 overlay primitives on 5 of 20 frames, against
629 written into tiles** — about 10% of pan work, on a quarter of frames.

The overlay's remaining price stands: what sits there is not tile-culled, so a
graph made mostly of very long wires still grows an always-resident layer.
Per-tile clipping becomes the better answer the moment `Quad` gains a content
mask, and that is where the rule should be revisited (§12).

---

## 5. Eviction and the resident-tile budget

Two mechanisms, because §4.3 and §9's risk table both say one was never enough:
`evict_after_frames` is a per-tile timer, and "an erratic pan pattern … can keep
many tiles within 'recently visited' simultaneously, which `evict_after_frames`
alone doesn't bound."

1. **Spatial mark-and-sweep.** A tile out of range for longer than the boundary's
   `evict_after_frames` is evicted. R-N §3.4 unchanged, with "unvisited" meaning
   "not in range" instead of "not in the tree".
2. **A total resident-tile budget with LRU eviction** beyond it
   (`BoundaryPolicy::resident_tile_budget`, default 256).

**A tile in range this frame is never evicted by either rule.** Evicting a tile
the same frame is about to render would trade a memory bound for unbounded work.
The consequence is stated rather than hidden: a budget below the viewport's own
in-range tile count cannot be met, and `TileResidency::over_budget()` reports the
shortfall.

LRU order comes from a monotonic touch stamp, not from the frame number: a whole
span shares one frame, so an eviction order tying across a hundred tiles would be
decided by hash iteration order, which is to say not decided.

**The budget is proven by comparison, not by a small number.** The erratic-pan
test runs the same 400-frame diagonal-with-reversal walk twice — once with a
budget, once with one large enough to be inert — and asserts the timer-only arm
peaks at more than twice what the bounded arm holds. A test that only checked
"the resident count stayed under the budget" would pass equally well against a
pan that never had many tiles live, which would leave the budget untested.

---

## 6. The gate

> §8's wording: *Panning a node-graph-style canvas across a tile boundary renders
> only the newly-revealed tile(s) — measured directly, not inferred — while
> panning within the resident grid costs one `TRANSFORM` update per visible tile
> and zero render/reconcile/layout work anywhere.*

### The harness, and why it measures rather than restates

`TiledCanvasDriver` builds a node graph on an unbounded content plane and drives
it through the real patch protocol into a real `Scene`. **Every frame offers
every visible tile its content, not only the revealed ones.** That is deliberate
and it is what makes the gate a measurement: if the harness skipped resident
tiles, "panning costs zero render work" would be true because the harness
declined to do any. Instead the work is offered every frame and the patch
protocol's own cross-frame addressing (§5.0) is what reduces it to nothing.

### Clause 2 — panning within the resident grid

Eight consecutive 24px pans on a 256px grid. Per frame, asserted and printed:

```
gate (within grid): 42 visible tiles, 43 TRANSFORM updates,
                    0 primitives written, 0 upload bytes, 0 layers displayed
```

43 = one per visible tile plus the overlay, which is the gate's "one `TRANSFORM`
update per visible tile" exactly. Every layer the frame touched is asserted to
carry `TRANSFORM` **and nothing else** — not merely to contain it. Zero
primitives written, zero bytes uploaded, zero layers left needing re-display,
zero layers created. **Met.**

One detail that is load-bearing rather than cosmetic: the baseline pan is 8px,
not zero. A 1024×768 viewport sits on exact multiples of 256, so it starts
perfectly tile-aligned and the first pixel of any pan reveals a whole new column
— which is a *crossing*, the other clause. The test asserts the no-crossing
condition per step rather than assuming it.

### Clause 1 — crossing a tile boundary

One full-tile pan, with the measurement printed:

```
gate (crossing): 6 of 42 tiles revealed, 75 primitives written
                 (75 in tiles, 0 on the overlay), against 432 resident before
```

"Measured directly" is the load-bearing phrase, so the assertion is not that the
count is small. It is that **the set of tile layers carrying `DISPLAY` after the
frame is exactly the set belonging to the revealed tiles** — a set equality, no
more and no fewer. 75 primitives against a 432-primitive whole-region refill is
17%. **Met.**

**The third layer, disclosed rather than folded in.** A crossing can also dirty
the overlay, when oversized content enters or leaves the visible region. On the
sampled frame it wrote nothing; over twenty crossings it wrote 70 primitives on 5
frames (§4). So the gate's "only the newly-revealed tile(s)" is exactly true of
the tile grid, and the boundary has a third layer that is not a tile and is not
covered by that phrasing. The test asserts the tile set exactly and bounds the
overlay's share separately, rather than widening the first assertion to fit.

---

## 7. Tile size, measured rather than asserted

§4.3 asks for a starting size "from common compositor practice (roughly
256–512px)" **validated against a representative node-graph workload, not
asserted**. `examples/phase45_tiling_bench.rs` sweeps 128 through 1024 —
deliberately outside the band at both ends, because a sweep covering only the
band could not show the band is the right place to be.

Adapter as §9. 1600×900 viewport, retain radius 1, budget 256, 2 warm-up runs
discarded, 12 timed.

| edge | tiles | slots | crossing | refill | ratio | overlay | vis med | vis best | buffered area |
|---|---|---|---|---|---|---|---|---|---|
| 128 | 160 | 151 | 86 | 409 | 4.8× | 21.8% | 165.5µs | 160.4µs | 1.49× viewport |
| **256** | **60** | **55** | **50** | **566** | **11.3×** | **2.8%** | **164.3µs** | **158.8µs** | **2.07× viewport** |
| 384 | 40 | 36 | 132 | 814 | 6.2× | 2.7% | 1206.2µs | 654.1µs | 2.74× viewport |
| 512 | 28 | 25 | 205 | 1006 | 4.9× | 3.6% | 342.7µs | 229.8µs | 3.51× viewport |
| 1024 | 15 | 13 | 481 | 1934 | 4.0× | 0.0% | 283.6µs | 241.2µs | 7.47× viewport |

**`TileGrid::DEFAULT_EDGE` is 256 because of this table**, not because 256 is a
round number.

The table was produced three times across the phase. Every counter column —
tiles, slots, crossing, refill, overlay, area — was **identical on every run**;
only the two timing columns moved. That is the expected split (the counters are
deterministic properties of the scene, the clocks are a shared laptop) and it is
stated because it is the reason the argument above rests on the counters.

**§9's risk table predicted the "too small" failure as inflated draw-call
overhead. That is visible — 151 slots against 55 — but it is not the binding
constraint.** The binding one is the overlay: at 128px a 130×70 node is *larger
than a tile*, cannot be anchored, and goes to the unbuffered layer — 21.8% of the
scene against 2.8% at 256. Tile size interacts with the workload's natural
content size directly, and a tile smaller than that unit degrades much faster
than the draw-call argument alone predicts.

**Two caveats the benchmark prints beneath itself, both of which change how the
table should be read.**

1. **The `refill` baseline is not constant across rows.** A retain radius of one
   buffers a ring one tile wide, so the buffered region grows with the tile size
   — 1.49× the viewport at 128, 7.47× at 1024. `Buffering::Margin(None)` buffers
   2.25× (§4.1), so only rows near that area are like-for-like. That is the 256
   row, at 2.07×. Large-tile rows are compared against a bigger buffer than
   `Margin` would have kept, and their ratio flatters tiling. Left in with the
   area disclosed, because the shape of the tradeoff is the point of the sweep.
2. **The `vis` column is a ceiling, not the kernel's cost.** It ends in
   `Device::poll(wait_indefinitely)`, which waits for everything already
   submitted — the same effect `docs/phase-4-results.md` §5 records for the
   readback path. The column is flat across a 10× tile-count range because what
   it mostly measures is one submit-and-synchronize round trip. A real frame
   pipelines this dispatch and never polls it, so no number there is a per-frame
   cost. The 384 row's median against its own best (1206µs vs 654µs) is the
   scale of the noise. It is reported to show the pass is not accidentally
   expensive, which it is not.

---

## 8. What verification found

Re-reading this branch's own commits rather than trusting their messages.

**1. A real leak: `Compositor::sweep` returned only the boundary's own layer.**
It pushed `state.layer` — the untiled/overlay layer — which was the complete
answer for as long as a boundary had exactly one layer, and stopped being
complete the moment this phase gave it one per resident tile. An evicted tiled
boundary drops its `BoundaryState`, taking its `TileResidency` with it, so that
call is the last moment anything knows those tiles existed; every tile's slab
reservation then outlived the scene. The method's own doc is the argument for why
it had to include them: the layer entry "is the compositor's to release, and
nothing else knows when the interval has elapsed." Fixed, and **the test pinning
it was confirmed to actually catch it by reverting the fix and watching it
fail**, rather than by assuming a new test tests something.

**2. The multi-tile content rule as first built put 73% of the scene on the
unbuffered layer** (§4). Found by running the gate, not by reading the code.

**3. `set_transform` before the boundary exists is inert, and it silently broke
the gate harness's first frame.** `Compositor::set_transform` is documented to
report `false` rather than create a boundary. The driver moved the canvas before
declaring it, so frame one resolved its tile span at the identity, and frame
two's ordinary 24px pan then looked like a tile crossing. Fixed in the driver,
and the ordering hazard written into `visit_tiled`'s doc where the next caller
will read it.

**4. `TileSpan::tiles()` was a hang, not an error, on a directly-built huge
span.** Unreachable through `visible_span`, which refuses a span above
`MAX_TILES` outright — but `TileSpan` is two public coordinate pairs a caller can
construct, and a billion-tile span would have looped rather than failed. Now
truncates at the same cap, with a test.

**5. Two of this phase's own first tests were wrong, and both were replaced with
stronger ones.** The eviction interval is anchored on the frame a tile was last
visited, and the first test was off by one against `Compositor::sweep`'s own
boundary. And a budget below the viewport's in-range tile count cannot be met at
all — so the erratic-pan test now proves the budget by comparison against a
timer-only arm rather than by a number staying small, and the unmet case is
asserted separately.

**Not fixed, recorded:** a boundary switched from `Tiled` back to `Margin` keeps
its tile layers until it is swept. Releasing them inside `visit_tiled` would mean
that method returning layers on the path where it reports having no tile set at
all. No caller in this phase changes a boundary's buffering mid-life; the
behaviour is in `visit_tiled`'s doc and in §12.

---

## 9. GPU adapter honesty check

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

**This is the same machine Phases 0, 3, and 4 ran on** — verified against
`docs/phase-4-results.md` §7 rather than assumed — so its caveats carry over
verbatim: one adapter, one driver version, one OS, one laptop-class discrete
NVIDIA part. Every GPU test and the benchmark report the adapter and its
negotiated features before printing anything
(`[INDIRECT_FIRST_INSTANCE=true MULTI_DRAW_INDIRECT_COUNT=true]`).

**A caveat specific to this phase, smaller than Phase 4's.** The tile-visibility
kernel uses no optional device feature at all — no indirect features, no
subgroups, nothing a device could lack. So unlike Phase 4's fallback paths, there
is no untested arm here that exists *for* hardware this machine is not. What is
still one-machine is the timing column in §7, and that column is already
disclosed as a synchronization ceiling rather than a cost.

---

## 10. Check, test, and clippy status

```
cargo check  -p wgpui-core -p wgpui-wgpu --all-targets   → clean, zero warnings
cargo test   -p wgpui-core                               → 309 passed, 0 failed
cargo test   -p wgpui-wgpu                               → 25 + 5 + 8 + 5 + 4 + 4
                                                           = 51 passed, 0 failed
cargo metadata --locked                                  → exit 0
cargo clippy -p wgpui-core -p wgpui-wgpu --all-targets -- --deny warnings
                                                         → clean from a cold build,
                                                           zero suppressions
```

360 tests across the two crates this phase touches. No test is skipped on this
machine: every GPU-dependent test goes through `device::context_or_report`, which
prints the adapter it got or a plain SKIPPED line — never silence that could pass
for coverage. All four tile-visibility tests were confirmed to print an adapter
line rather than skip.

`cargo test --workspace` was **not** run: it pulls in `gpui-ce`'s legacy suite,
recorded since Phase 1 as running 10+ minutes without completing, which no 2.0
branch modifies. Tests are scoped to the touched crates.

Clippy was run cold (`cargo clean -p` on both crates first) rather than trusted
from an incremental cache. Note that `AGENTS.md` prefers `./script/clippy`, which
adds `--release --all-features` and runs `cargo-machete`/`typos` when installed;
the command above is the narrower one the phase brief specified, so the
release-profile and all-features variants have not been run — the same disclosure
Phase 4 made.

**Zero suppressions in this phase's diff** — `git diff origin/2.0..HEAD | grep
'^+.*allow('` is empty. Clippy found three things and all three were fixed rather
than silenced, which is the standard Phases 3 and 4 both set:

1. **`neg_cmp_op_on_partial_ord`** on `TileGrid::new`, which spelled its NaN
   rejection as `!(width >= MIN_EDGE)`. The codebase already had the right idiom
   for exactly this — `Rect::is_empty` uses `partial_cmp` and documents why —
   and the fix follows it. A real readability finding on a real NaN branch, not
   a false positive.
2. **`too_many_arguments`** on `IndirectArgsPass::run_with_slots` (8/7). Fixed by
   introducing `GeneratedSlots { buffer, count }`, which is the same resolution
   Phase 4 applied to `issue_composites` and buys the same thing beyond the lint:
   a count from one pass can no longer be paired with another pass's buffer,
   which would generate arguments for records that are not there, silently.
3. **`too_many_arguments`** on `TileVisibilityPass::run_into_args` (10). Fixed by
   `ArgsTarget`, grouping the argument-generation stage's pass, buffers, and
   record encoding. `vertex_count` and `first_instance` together *are* the
   encoding, and splitting them across a call site is how a record ends up
   carrying a base the shader is not expecting.

The third of those had been suppressed with an `#[allow]` mid-phase and was
un-suppressed during this pass rather than left.

---

## 11. Gate assessment — honest read

**Clause 2 — panning within the resident grid costs one `TRANSFORM` update per
visible tile and zero render/reconcile/layout work: met, exactly.** 42 visible
tiles, 43 `TRANSFORM` updates (tiles plus overlay), zero primitives written, zero
bytes uploaded, zero layers displayed, zero layers created, across eight
consecutive frames — with every touched layer asserted to carry `TRANSFORM` and
nothing else, and with the harness offering the work rather than declining it.

**Clause 1 — crossing a tile boundary renders only the newly-revealed tiles,
measured directly: met for the tile grid, with a third layer disclosed.** The set
of tile layers carrying `DISPLAY` is exactly the revealed set — a set equality,
directly measured. 75 primitives against a 432-primitive refill.

The honest qualification is that a tiled boundary has three kinds of layer, not
two: resident tiles, evicted tiles, and one unbuffered overlay. The gate's
phrasing covers the first two. The overlay can also re-render on a crossing —
measured at 70 primitives across 5 of 20 crossings, about 10% of pan work — and
that is a property of the multi-tile content rule this phase chose, not a defect
in the tiling. A reviewer who reads "only the newly-revealed tile(s)" as covering
the whole boundary should treat the overlay as an additional, measured, bounded
cost rather than as a met-or-not clause.

Three things worth recording as genuinely open rather than met:

- **The gate is measured through `wgpui-core`'s scene, not a running
  application.** There is still no window loop; `TiledCanvasDriver` drives the
  real patch protocol into a real `Scene`, and `FrameRenderer` is not wired to
  tiled boundaries (§12). The GPU half is proven at the pass level by
  differential, not by a frame that drew a tiled canvas end to end.
- **One workload.** The tile-size table is one node-graph shape at one viewport.
  A whiteboard with a few enormous strokes, or a level editor with dense small
  sprites, would sit differently against the anchoring rule in particular.
- **One machine**, identical to every prior phase's caveat, unimproved.

---

## 12. What is open for later phases

### Already scheduled, and correctly not done here

1. **Text, `Img`, and `StyledText` `diff_key` are Phase 5.** Unchanged by this
   phase and unblocked by it. A tile's `GlyphRun` slot is generated with every
   other layer's and nothing issues its draws, exactly as
   `docs/phase-4-results.md` §10 recorded.
2. **The regular-content layout kernel is Phase 6.1**, and is still a rescoped
   *spike* gated on its own fused-dispatch follow-up per §8's row — Phase 0's
   Spike B measured a standalone dispatch losing by ~1000×. Nothing here changes
   that either way.
3. **Zoom/multi-resolution tiling stays rejected (§10).** `Buffering::Tiled`
   re-renders at the current scale on a zoom change, the same as `Margin`.
   `TileGrid` holds a tile size and no scale, so there is no half-built
   multi-resolution path to remove later.

### Loose ends of this phase

4. **`FrameRenderer` does not know about tiled boundaries.** `render/frame.rs`
   builds its slot table from `Scene::draw_slots` — every layer, tiles included,
   which is correct but does not route through `TileVisibilityPass`. The two are
   proven to agree at the pass level (§3) and are not yet wired into one frame.
   That wiring belongs with whatever phase brings a real window loop, alongside
   Phase 4's own item 4 (nothing drives `LayerTexturePool::acquire`/`sweep` in
   the frame loop either).
5. **Per-tile clipping is the better multi-tile rule once `Quad` has a content
   mask** (§4). The anchoring rule shipped here is correct and measured, and it
   leaves oversized content un-culled on the overlay. The moment a per-primitive
   clip rect exists, `TilePlacement` is the one place that decision lives.
6. **Switching a live boundary from `Tiled` to `Margin` strands its tile layers**
   until the boundary is swept (§8). Documented, not fixed.
7. **Per-frame allocations in the tile pass**, matching Phase 4's item 5
   unchanged: `TileVisibilityPass::run` creates a params buffer, a descriptor
   buffer, and a bind group per call. `O(tiles)` and outside any gate's clock,
   but the same argument that justified caching `QuadDrawPlan` applies, and the
   same choice was made — left alone rather than fixed, to keep this phase's diff
   to what it could verify.
8. **Phase 3's Scene A occlusion loss remains unaddressed**, and this phase is
   where §9's risk table said it would become relevant: "a zoomed-out node graph
   is the obvious candidate, tying directly to §4.3's tiling motivation." Tiling
   in fact *helps* the shape of that problem — an out-of-range tile's primitives
   are never uploaded or dispatched at all — but nothing here measured the
   interaction, and the 3%-visible scene Phase 3 measured is not the same as a
   tiled one. Still open, now with a plausible mitigation nobody has tested.

### What a human should do before calling this closed

1. **Run the gate on a second workload** — a whiteboard (few, very large strokes)
   and a sprite grid (many, very small) are the two shapes that would stress the
   anchoring rule from opposite directions. The bench takes a `NodeGraphSpec`
   today; a second generator is the work.
2. **Re-run the tile-size sweep on a machine that is not also running a
   desktop**, and on hardware without a discrete GPU. The `vis` column is
   already disclosed as a synchronization ceiling, but the crossing/refill
   columns are CPU-side counts and would be worth confirming unchanged.
3. **Let `gpui-ce`'s legacy suite finish once** — outstanding since Phase 1,
   still not something any 2.0 branch modifies.
