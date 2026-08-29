# Phase 3 Results — GPU Compute Ordering + Occlusion, and the Differential Harness

Status: **Phase 3 executed, both gates met.** This documents what was built,
what each gate actually asserts, what was measured on real hardware, what was
found broken along the way, and what a human should still treat as open. It
follows `docs/gpu-native-architecture.md` ("2.0" below) §5.1, §5.2, and §8's
Phase 3 row, and `docs/retained-layers.md` ("R-N" below) §8. Nothing under
`src/` changed — the legacy backend is frozen per §8's own constraint, and
`git diff` over `src/`, the root `Cargo.toml`, `build.rs`, `examples/`, and
`tests/` is empty across the whole phase. Work lives on branch
`wgpui-2.0/phase-3-ordering-occlusion`, pushed to origin, not merged, no PR.

**Contents:** §1 The unverified commit, and what happened to it · §2 What
shipped, and where · §3 The two computations, written once · §4 Gate #1 —
culled and unculled match exactly · §5 Gate #2 — Spike A's numbers on the real
pipeline · §6 GPU adapter honesty check · §7 What was found broken, and fixed ·
§8 Check, test, and clippy status · §9 Gate assessment — honest read · §10 What
a human should do before calling this closed

---

## 1. The unverified commit, and what happened to it

This phase was picked up mid-flight. Four commits existed on the branch: three
verified, and `a33bd4ea7f`, committed by a human as-is when the prior agent hit
an API session limit mid-edit, explicitly labelled UNVERIFIED and explicitly
flagged as "not known to compile, not known to be a good direction." 881 lines
across five files. The continuation brief's first instruction was to assess it
honestly rather than rescue it out of sunk-cost reflex.

**Verdict: salvaged, deliberately, after reading it — not by default.** It did
not compile. But the three errors were mechanical, and the design underneath
them was sound and well-argued:

| Error | Nature |
|---|---|
| `ordering_pass.rs` used `item_groups` at two call sites after renaming the binding to `relax_groups` at its definition | An unfinished rename. Two lines. |
| `ordering.wgsl` declared `var best = best;` shadowing a function parameter, which WGSL forbids | The shader had never been compiled *once*. Caught by running the differential test, not by `cargo check` — WGSL is a string until a device parses it. |
| `compute_differential.rs` asserted a 300-deep chain needs more than one submission | Not an error in the new code at all: an assertion pinning the *old* kernel's behaviour, which the rework's whole purpose was to change. |

What made this worth finishing rather than reverting was the third error's
shape. It is the signature of a change that is working — a test failing because
the thing it measured got better. The two real errors were both "stopped
typing," not "wrong idea."

**The design was checked, not taken on faith.** The rework's claim is that the
relaxation may skip any block that settled in the previous iteration, because a
settled block's contribution is already folded into each primitive's current
value. That is a real correctness argument and it holds:

> Invariant. At every iteration `t`, for every `i` and every `j < i` whose
> bounds overlap `i`'s: either `block(j)` is flagged changed at `t`, or
> `order^t[i] ≥ 1 + order^t[j]`.
>
> It holds at `t = 0` because every flag starts set. It is preserved because a
> block unflagged at `t+1` did not move, so its value is what it was at `t`; if
> it *was* flagged at `t` then `i`'s scan saw it and took the max, and if it was
> not, the invariant at `t` already gave the inequality, which survives because
> `order[i]` only rises.
>
> Termination with no flags set therefore means the whole system satisfies
> `order[i] ≥ 1 + order[j]` pairwise. The iteration starts at the recurrence's
> floor and each step is monotone and bounded above by the least fixed point, so
> the value it stops at *is* the least fixed point — the same one
> `BoundsTree::insert` computes.

The same argument covers the within-block Gauss-Seidel reads (a fresh value is
still ≤ the least fixed point, so reading it early can only accelerate
convergence, never overshoot). This is why the shader's answers can be — and
are, per §4 — bit-identical to the CPU tree's rather than merely close.

**What was then changed on top of it, and why.** Finishing the commit was not
the end of the assessment. Running gate #2's benchmark for the first time (the
prior agent never got to run it) showed the rework had traded one bottleneck
for another: see §5. Two further changes followed, both measured, both in §5.

Had any of this gone the other way — had the errors been structural, or the
soundness argument failed, or the measurement shown no path to a win — the
instruction was to revert to `9cb3f2dab4` and take a cleaner path. That was a
live option throughout and is recorded here because the decision not to take it
should be as legible as taking it would have been.

---

## 2. What shipped, and where

| File | Lines | Role |
|---|---|---|
| `wgpui-core/src/geometry.rs` | 217 | `Rect` — the one rectangle type the Rust and WGSL sides share |
| `wgpui-core/src/occlusion.rs` | 1,008 | `Mode` (R-N §8.5's `WGPUI_OCCLUSION` switch), `CoverageItem`, `PoisonRegion`, `CoverageHierarchy`, `keep_mask`, `keep_mask_exhaustive`, the GPU encoders |
| `wgpui-core/src/occlusion/coverage.rs` | 590 | R-N §8.3's five conditions (`opaque_region`) and the coverage sweep (`fully_covered`) |
| `wgpui-core/src/ordering.rs` | 215 | The painter-order recurrence stated directly, plus its GPU encoder |
| `wgpui-core/src/ordering/bounds_tree.rs` | 381 | `src/bounds_tree.rs` ported, answering the same recurrence fast |
| `wgpui-core/src/shaders/ordering.wgsl` | 386 | Hierarchy build, relaxation, bitonic sort |
| `wgpui-core/src/shaders/occlusion.wgsl` | 393 | `keep_item` and `fully_covered`, transcribed |
| `wgpui-core/src/test_support/ui_walk.rs` | 755 | An editor-shaped scene and a scripted walk, driven through a real `Scene` |
| `wgpui-core/src/test_support/raster.rs` | 322 | A reference rasterizer, so "match exactly" is a claim about pixels |
| `wgpui-wgpu/src/render/device.rs` | 146 | Headless compute context that reports a missing adapter rather than panicking |
| `wgpui-wgpu/src/render/readback.rs` | 113 | The one synchronous buffer read both passes need |
| `wgpui-wgpu/src/render/compute/ordering_pass.rs` | 723 | Pipelines built once; relaxation run to a checked fixed point |
| `wgpui-wgpu/src/render/compute/occlusion_pass.rs` | 297 | One dispatch per dirty layer |
| `wgpui-wgpu/tests/compute_differential.rs` | 380 | Gate #1's compute arm |
| `wgpui-wgpu/examples/phase3_ordering_occlusion_bench.rs` | 411 | Gate #2 |

`wgpui-core` still has no `wgpu` dependency and no live device — the constraint
that its coverage logic be pure, portable Rust holds, and §3 is why that
constraint is what makes the differential harness possible at all.

Indirect draw is untouched, per §8's explicit deferral to Phase 4. The compute
passes write orders, a draw permutation, and a keep mask; nothing yet consumes
them into a draw call.

---

## 3. The two computations, written once

The phase's organising constraint — 2.0 §5.2's "the same computation, restated
as data-parallel" — is load-bearing rather than stylistic, and it is worth being
precise about what it buys.

Each computation is written **once in Rust**, tested headlessly against a
brute-force definition, and then transcribed into WGSL. The differential harness
does not check that two independent implementations happen to agree; it checks
that a transcription is faithful. That is a much narrower claim, and it is the
only one a compute shader can practically support, because — as 2.0 §5.2 puts
it — there is no CPU-side result to eyeball.

Three layers of reference, each checking the one above it:

1. **The definition.** `ordering::painter_orders` is the recurrence written as
   an `O(n²)` double loop. `occlusion::keep_mask_exhaustive` is the coverage
   rule with every item scanning every item above it.
2. **The fast Rust.** `ordering::painter_orders_via_tree` is `src/bounds_tree.rs`
   ported. `occlusion::keep_mask` walks a two-level AABB hierarchy. Randomised
   tests assert each equals its definition — a pruning bug that makes the fast
   path miss a candidate is invisible against itself, so it is never checked
   against itself.
3. **The WGSL.** Checked against layer 2 on a real device, per §4.

**One deliberate divergence from `src/occlusion.rs`, applied identically on both
sides:** the occluder set is bounded (`MAX_OCCLUDERS`), where the legacy sweep's
is unbounded. A bounded set can only *miss* a cull, never make a wrong one, so
it stays inside R-N §8.3's "conservative by construction" rule. Because both
sides apply the same cap in the same ascending order, they cap identically — a
test asserts exactly this under a 300-occluder pile-up.

**One disclosed gap in the mapping, not the logic:** Phase 1's `Quad` has no
dashed-border style, so the port cannot reject dashed borders the way
`src/occlusion.rs`'s `quad_opaque_region` does. `quad_coverage_item` documents
this and says a widened `Quad` widens that one function. Nothing silently
assumes a border style that does not exist yet.

---

## 4. Gate #1 — culled and unculled match exactly

> §8's wording: *Culled/unculled scenes match exactly over a scripted UI walk
> (R-N §8.5's bar).*

The gate has two arms. Both pass.

**The CPU arm** (`wgpui-core`, no device, not skippable):
`gate_1_culled_and_unculled_scenes_are_pixel_identical_over_a_scripted_walk`
runs the scripted walk, rasterizes each frame twice — once emitting everything,
once emitting only what `keep_mask` keeps — and compares framebuffers.

**The compute arm** (`wgpui-wgpu/tests/compute_differential.rs`, needs a device)
asserts three things per frame, not one:

1. GPU painter orders and the draw permutation equal the CPU `BoundsTree`'s
   exactly.
2. The GPU keep mask equals the CPU reference's exactly.
3. Rasterizing with and without culling, both through the **GPU's own** draw
   order, gives bit-identical framebuffers.

Claims 1 and 2 are what make the compute path a port; claim 3 is what makes
culling provably an optimization. Asserting only the strongest would let the
shader and the reference agree on being wrong.

**Result on NVIDIA GeForce RTX 4060 Laptop GPU (Vulkan, driver 561.03): 5/5
compute-differential tests pass**, 189 of 3,528 primitives culled across the
walk. `gate_1_compute_arm` reports and returns rather than failing when no
adapter is present, and says which half of the gate did not run.

**What the walk actually exercises**, checked by its own tests rather than
asserted: a translucent primitive (the alpha rule), a rounded opaque primitive
(the corner-radius inset), an opaque primitive with a translucent border (the
border inset), and a filter region (poisoning and the blur margin) — R-N §8.3's
rejection reasons, each present in the scene, each with a named test that fails
if the scene stops containing one. A separate test pins one concrete number: a
14px icon with a 4px radius has a 6px opaque core.

**What this arm does not cover.** R-N §8.4 — that culling must never suppress
hit registration — has no coverage here, correctly: `wgpui-core` has no hitboxes
or dispatch nodes yet, so there is nothing to accidentally cull. That constraint
lands with whichever phase brings hit testing, and is named here so it is not
mistaken for something Phase 3 checked.

---

## 5. Gate #2 — Spike A's numbers on the real pipeline

> §8's wording: *Spike numbers from Phase 0 reproduced on the real pipeline, not
> just the synthetic case.*

`crates/wgpui-wgpu/examples/phase3_ordering_occlusion_bench.rs`. The scene is
not Spike A's 100,000-quad cluster grid — that grid's non-overlapping clusters
gave the GPU kernels a free neighbour-search bound, which
`docs/phase-0-results.md` flagged in its own §4 as real work Phase 3 still owed.
This scene is `test_support::ui_walk`'s editor — chrome, a scrolling tree panel,
a node-graph viewport, a docked inspector, a modal — built through the real
`Scene`/`Layer`/`PrimitiveStore` by real `ScenePatch` appends and read back out
of the resident store, not out of the generator's `Vec`.

**Two shapes, because either alone answers half the question.** At normal zoom a
hundred-thousand-primitive editor has most of its content below the window — a
retained layer holds its whole list and its whole graph (§5.0) — so occlusion
early-outs on primitives that clip to nothing and the measurement is really
about ordering. Zoomed out, the same primitive count is packed into the visible
area, mostly overlapping, and occlusion has real work to do.

### Methodology, and one place it deliberately runs against its own interest

- **Timed window**: input encoding, buffer creation, uploads, every dispatch,
  submit, poll, and the ordering pass's convergence readbacks — end-to-end
  wall-clock from the CPU's point of view, on Phase 0's own terms. Pipeline
  construction is outside it and reported separately (~36–52ms once), because a
  real frame builds pipelines at startup and Spike A's write-up blames its own
  2–3× variance on having built them inside the window. Reading the *results*
  back is also outside it, for Spike A's other stated reason.
- **Two warm-up runs, discarded on both sides, disclosed in the output.** Spike
  A reported a 14.7× first run against a 5.5–6.9× cluster and argued the outlier
  away afterwards, because every run was a fresh process. Here the runs share
  one, so the warm-up can be excluded and *named* instead. Post-warm-up variance
  is under 2% on the GPU side.
- **Median and best both reported**, so one statistic cannot stand in for the
  distribution. For an even count the median is the lower middle, so every
  printed number is a duration that was actually observed.
- **The CPU occlusion side is given the accelerated algorithm.** `keep_mask`
  walks the same two-level hierarchy the shader does. The renderer's actual
  sweep (`src/occlusion.rs`) has no such structure and is quadratic at this
  scale; measuring against that would flatter the GPU enormously. So the
  occlusion comparison is CPU-versus-GPU execution of *one* algorithm. The
  ordering comparison is not levelled this way and does not need to be —
  `BoundsTree` is already the fast structure and it is what ships.

### Three changes the benchmark forced

The prior agent never ran this benchmark. Running it produced the phase's most
useful result, which was initially a bad one.

| Change | Scene B ordering |
|---|---|
| `a33bd4ea7f` as written (one invocation per block) | 375ms — **1.9× slower than the CPU** |
| Relax kernel: one *workgroup* per block, 64 lanes running the earlier-block scan in parallel, lane 0 alone resolving the internal chain | 220ms |
| Follow-up relaxation batches double (8, 16, 32, … capped at 128) instead of a flat 48 | 175ms — **1.16× faster than the CPU** |

The first change is the interesting one. Collapsing a block's chain into one
invocation cut the iteration count by more than an order of magnitude, which was
the right diagnosis — but it did so by serializing 64 primitives into one
thread, dropping a 97k-primitive dispatch to 1,520 invocations (~24 workgroups
on a 24-SM part). Splitting the kernel across a barrier recovers that: the scan
over earlier blocks is independent per primitive and is where nearly all the
time goes, while the within-block chain is a genuine dependency but touches only
cached values. Same computation, same fixed point; the parallel half made
parallel again.

The second is a plain measurement error in a constant. A flat 48-iteration
follow-up batch overshot badly — Scene B settles in the low twenties and was
dispatching 64 iterations plus a redundant bitonic sort. Geometric growth bounds
wasted iterations by roughly the number genuinely needed while keeping round
trips logarithmic, without either constant being fitted to a particular scene.

### Results

NVIDIA GeForce RTX 4060 Laptop GPU (Vulkan, driver 561.03). 4 timed runs after
2 warm-up. Every timed run reported **0 mismatches** against the CPU reference
on painter orders, draw permutation, and keep mask.

**Scene A — normal zoom, 99,054 primitives resident, deepest painter order 99.**
2,980 clip to a non-empty visible region; 1,296 culled (43.5% of the visible
ones). Converges in one submission.

| | CPU | GPU | |
|---|---|---|---|
| ordering (tree+sort vs. relax+bitonic) | 2.532s | 27.4ms | GPU **92.2× faster** |
| occlusion (coverage sweep) | 2.01ms | 2.61ms | GPU 1.30× slower |
| **both** | **2.534s** | **30.1ms** | GPU **84.2× faster** |

**Scene B — zoomed out, 97,254 primitives, deepest painter order 577.** All
97,254 visible; 50,136 culled (51.6%). 24 relax iterations, 3 submissions.

| | CPU | GPU | |
|---|---|---|---|
| ordering | 203.8ms | 175.4ms | GPU **1.16× faster** |
| occlusion | 48.2ms | 3.53ms | GPU **13.7× faster** |
| **both** | **252.2ms** | **179.2ms** | GPU **1.41× faster** |

### Reading these honestly

**Scene A's 92× is not 92× of GPU brilliance.** It is dominated by the CPU
`BoundsTree` taking 2.5 seconds for 99k inserts — ~25µs/insert, worse even than
the ~7.4µs/insert `docs/phase-0-results.md` §4 measured and flagged. Phase 0
guessed why: the SAH-style half-perimeter heuristic does not keep spatially
distant content apart in the tree, so `find_max_ordering`'s pruning degrades on
content spread across a wide canvas. Scene A at normal zoom — a list and a graph
extending far below the window — is exactly that shape, more extremely than
Spike A's grid was. Phase 0 said this is an argument *for* Phase 3 rather than a
benchmark artifact, and this measurement is the same finding on a real scene
rather than a synthetic one. But the number should be quoted as "the CPU
structure degrades badly here and the GPU does not," not as a throughput ratio.

**Scene B is the conservative read, and the one to quote.** There `BoundsTree`
behaves reasonably (~204ms) and the GPU still wins overall by 1.41×, with the
margin coming almost entirely from occlusion (13.7×) rather than ordering
(1.16×). A 1.16× ordering win is thin — thin enough that a different GPU, a
different driver, or a scene with deeper chains could plausibly flip it.

**Scene B is also deliberately adversarial about chain depth.** Its node grid
uses a column pitch of 0.55× node width, so every node overlaps its horizontal
neighbour and the painter-order chain runs 577 deep. Real node-graph editors do
not usually overlap nodes that heavily. This was introduced by `a33bd4ea7f` to
stress exactly the case the relaxation is worst at, and it is the right thing to
measure — but it should not be read as "the typical scene."

**Occlusion loses on Scene A, by 1.30×.** With only 2,980 of 99,054 primitives
visible, the CPU sweep early-outs on 97% of the input while the GPU still pays
dispatch and upload for all of it. This is a real, small loss on a real, common
shape, and it is not worth hiding: the compute path's occlusion win requires
content actually being on screen. It is also exactly the case indirect draw
(Phase 4) changes, since the CPU currently pays to *learn* the answer.

**Comparison to Phase 0.** Spike A's reproducible number was ~6–7× end-to-end on
a synthetic scene with a free spatial bound. This phase reproduces a win on the
real pipeline in both scene shapes — far above it in one, well below it in the
other — with the spatial bound now earned by a real two-level hierarchy rather
than granted by the benchmark's layout. §8's gate asks that the spike's numbers
be reproduced on the real pipeline, and the honest statement is that the
*direction* reproduces robustly while the *magnitude* is entirely scene-shaped
and ranges from 1.4× to 84× across two defensible scenes.

---

## 6. GPU adapter honesty check

Per Phase 0's discipline, checked before any number above was trusted:
`cargo run -p wgpui-wgpu --example adapter_probe`.

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

**A real adapter was available for this run and every number in §5 came off
it.** It is the same machine Phase 0 ran on, so the caveats there carry over
verbatim and are not restated as if they had been re-checked: one adapter, one
driver version, one OS, one laptop-class discrete NVIDIA part. Nothing here says
anything about integrated GPUs, macOS/Metal, Linux/Mesa, older discrete parts,
or WebGPU. The benchmark refuses to print timings without naming its adapter and
warns loudly if it ever selects a software rasterizer.

---

## 7. What was found broken, and fixed

Beyond `a33bd4ea7f`'s three errors (§1):

1. **The relax kernel had destroyed its own parallelism.** §5. Found by
   measurement, not by reading — the code's own comment correctly *stated* the
   trade ("the cost is parallelism") and correctly judged it worth making; what
   neither the comment nor a reading catches is that the trade was avoidable.
2. **`RELAX_BATCH = 48` overshot convergence by ~40 iterations plus a redundant
   sort on any scene that missed the first batch.** §5.
3. **Clippy had never been run on this branch, and `--all-targets` on
   `wgpui-wgpu` had not been run since Phase 0.** Eight findings, all fixed
   rather than suppressed — including two `!(a > b)` comparisons that are *not*
   equivalent to `a <= b`. The negation was load-bearing: a NaN edge must read
   as "empty" and "clamps to zero." Rewriting through `partial_cmp` keeps the
   behaviour and now states the reason, which the negated form never did.
   `a33bd4ea7f` had also dropped `data_group`'s
   `#[allow(clippy::too_many_arguments)]` without getting the function under the
   limit; fixed by naming what the two bind groups actually share.
4. **The benchmark's median was warm-up-polluted.** With 4 runs and
   `values[len/2]`, a single slow first run landed *on* the median (88.9ms
   against a true steady state of 27.3ms). §5's methodology change.

---

## 8. Check, test, and clippy status

```
cargo check -p wgpui-core -p wgpui-wgpu --all-targets   → clean
cargo test  -p wgpui-core --release                     → 257 passed, 0 failed
cargo test  -p wgpui-wgpu --release                     → 3 + 5 passed, 0 failed
cargo clippy -p wgpui-core -p wgpui-wgpu --all-targets -- --deny warnings
                                                        → clean, cold build,
                                                          zero suppressions
```

262 tests total across the two crates. `cargo test --workspace` was **not** run:
it pulls in `gpui-ce`'s legacy suite, which Phase 1 already recorded as running
11+ minutes without completing and which no 2.0 branch modifies. Tests are
scoped to the crates this phase touches, and `git diff` confirms nothing outside
`crates/` changed.

`clippy.toml`'s conventions were checked first; nothing in this phase trips
`disallowed-methods` or `disallowed-types`. Note that `AGENTS.md` prefers
`./script/clippy`, which adds `--release --all-features` and runs
`cargo-machete`/`typos` when installed; the command above is the narrower one
the phase brief specified, so the release-profile and all-features variants of
these lints have not been run.

---

## 9. Gate assessment — honest read

**Gate #1 — culled/unculled match exactly over a scripted UI walk: met.** Both
arms. The CPU arm needs no device and is not skippable; the compute arm ran on a
named real adapter and asserts three things per frame including a pixel
comparison through the GPU's own draw order. The claim being made is that the
WGSL is a faithful transcription of Rust that was itself checked against a
brute-force definition — three layers, each checked against the one above.

**Gate #2 — Spike A's numbers reproduced on the real pipeline: met, with the
magnitude heavily scene-dependent and stated as such.** A realistic large scene
built through the actual `Scene`/`Layer`/`PrimitiveStore` shows the compute path
winning end-to-end in both shapes measured — 84× at normal zoom, 1.41× zoomed
out — on named real hardware, with exact correctness on every timed run. The
1.41× is the number to plan against. The 84× is real but is as much a statement
about `BoundsTree` degrading on widely-spread content as about compute
throughput.

**Both of §8's Phase 3 clauses are therefore satisfied.** Two things are worth
recording as genuinely open rather than met:

- **The ordering win alone is thin on the dense scene (1.16×).** If a future
  scene shape or a slower GPU flips that sign, the phase's value would rest
  almost entirely on occlusion. Worth re-measuring whenever the hierarchy or the
  relaxation changes.
- **One machine.** Identical to Phase 0's caveat, unimproved by this phase.

---

## 10. What a human should do before calling this closed

1. **Re-run both gates on non-NVIDIA hardware** — at minimum one integrated GPU,
   ideally macOS/Metal or Linux/Mesa. `cargo test -p wgpui-wgpu --release` and
   `cargo run -p wgpui-wgpu --example phase3_ordering_occlusion_bench --release`.
   Both name their adapter; neither will silently report a software rasterizer's
   numbers as hardware.
2. **Decide whether Scene B's 0.55× node pitch is the right adversarial case**,
   or whether a real node-graph capture should replace it. It currently drives
   the deepest-chain figure the relaxation is tuned against.
3. **Let `gpui-ce`'s legacy suite finish once**, still outstanding from Phase 1
   and still not something any 2.0 branch modifies.
4. **Phase 4 should revisit Scene A's occlusion loss.** The GPU currently pays
   upload and dispatch for 99k primitives to cull 1,296 of them; indirect draw is
   what makes the CPU stop paying to learn that answer.
5. **Phase 6.1's rescoped spike is now answerable.** Phase 0 deferred it pending
   "Phase 3 infrastructure that doesn't exist yet." It exists: `OrderingPass`
   dispatches per layer with a live encoder, and folding a per-item position
   computation into it is now a concrete change rather than a hypothetical.
