# Phase 0 Results — Workspace Scaffold + the Two Spikes

Status: **Phase 0 executed.** This documents what was built, what was
measured, and what a human still needs to check before treating this gate as
closed. It follows `docs/gpu-native-architecture.md` ("2.0" below) §0, §3,
§8's Phase 0 row, and §11. Nothing under `src/` changed in behavior; this
work lives entirely in a new `[workspace]` member set alongside it, on
branch `wgpui-2.0/phase-0-scaffold-and-spikes`.

**Contents:** §1 Workspace scaffold · §2 Deviations from §3's file map, and
why · §3 GPU adapter honesty check · §4 Spike A (ordering + occlusion) · §5
Spike B (uniform-list layout) · §6 Gate assessment — is Phase 0 done?

---

## 1. Workspace scaffold

The root `Cargo.toml` gained a `[workspace]` table (root package
`gpui-ce`/lib `gpui` listed as a member, per 2.0 §3's requirement that its
public surface not change) alongside six new members:

```toml
[workspace]
members = [
    ".",
    "crates/wgpui-core",
    "crates/wgpui-layout",
    "crates/wgpui-text",
    "crates/wgpui-widgets",
    "crates/wgpui-wgpu",
    "crates/wgpui-devtools",
]
```

Each new crate follows this repo's own `AGENTS.md` conventions rather than
2.0 §3's literal `mod.rs`-based trees: `[lib] name = "wgpui_core", path =
"src/wgpui_core.rs"` (mirroring the root crate's own `gpui-ce` → `gpui` →
`src/gpui.rs` pattern), and every module that 2.0 draws as `foo/mod.rs` is
instead `foo.rs` sitting next to a `foo/` directory of submodules — the
"prefer `src/some_module.rs` over `src/some_module/mod.rs`" rule. §2 below
lists every place this required a judgment call. All 130 files under
`crates/` are placeholder stubs: a module doc comment citing the exact
2.0 section it implements, `#![allow(dead_code)]`, and no `todo!()` /
`unimplemented!()` / `panic!()` anywhere (this repo's own clippy lints
already `deny` `todo!()`). Each crate's `Cargo.toml` has empty
`[dependencies]` except where a stub genuinely needs a type to compile —
that turned out to be only `wgpui-wgpu`, and only in `[dev-dependencies]`/
its own `[dependencies]` for the Phase 0 spike harnesses (§2 below).

**Verification performed:**

```
cargo check --workspace --offline     → Finished, 0 errors (6 new crates + gpui-ce)
cargo check -p gpui-ce --offline      → Finished, 72 warnings (identical count/content
                                          to the pre-workspace baseline — confirmed by
                                          running the same check before touching Cargo.toml)
cargo check -p wgpui-wgpu --all-targets --offline
                                        → Finished, includes the 3 new examples
```

The `gpui-ce` warning count (72) was diffed against a baseline run captured
*before* any change in this branch — identical, confirming constraint 3
("existing code keeps working, unmodified in behavior") held for the
scaffold step. `cargo check --workspace` builds `libwgpui_core`,
`libwgpui_layout`, `libwgpui_text`, `libwgpui_widgets`, `libwgpui_wgpu`,
`libwgpui_devtools` (confirmed via `target/debug/.fingerprint/wgpui-*` and
`target/debug/deps/libwgpui_*.rmeta` build artifacts) with zero warnings of
their own.

---

## 2. Deviations from §3's file map, and why

1. **No-`mod.rs` convention (`AGENTS.md`) applied throughout.** Every
   `foo/mod.rs` in 2.0 §3.1–§3.6 became `foo.rs` + `foo/` (submodules).
   Mechanical, and it's the repo's own standing rule, not a judgment call —
   listed here for completeness rather than as a real deviation.
2. **`wgpui-text/src/fonts/`** needed a declaring file 2.0 §3.3's tree
   doesn't show (it lists `fonts/features.rs` and `fonts/fallbacks.rs` with
   no `fonts/mod.rs`) — added `fonts.rs` since a directory of submodules
   needs one under this repo's convention.
3. **Shader files under `wgpui-core/src/shaders/*.wgsl` and
   `wgpui-wgpu/src/render/shaders/*.wgsl` are placeholder text**, not the
   literal move-as-is 2.0 §3.5 describes for the eight hand-written render
   shaders. Moving ~/platform/cross/shaders/*.wgsl verbatim is mechanical
   but out of scope for a stub pass whose job is the module skeleton, not
   the content; each placeholder file states this directly. Both crates
   gained a thin `shaders.rs` (not in 2.0's literal tree, which treats
   `shaders/` as a bare asset directory) exposing each file via
   `include_str!` — needed so the directory is reachable Rust-side at all
   without adding a real shader-compilation dependency yet.
4. **No inter-crate path dependencies wired yet.** 2.0 §3 implies
   `wgpui-widgets` will eventually depend on `wgpui-core`/`-layout`/`-text`,
   and `wgpui-wgpu` on `wgpui-core`. None of that exists yet — every stub
   file compiles standalone, so nothing forced the dependency graph to be
   real before Phase 1 needs it. Deliberate, not an oversight: Phase 1 is
   where `wgpui-core::patch`/`scene` get real types other crates need to
   name, and wiring the dependency edges now against empty modules would
   only be discovered-wrong later.
5. **`wgpui-wgpu` depends on `wgpu`/`pollster`/`bytemuck` already** (version
   pins matching the root crate's own `Cargo.toml`), specifically because
   Phase 0's spike harnesses live in this crate's `examples/` and need a
   real device — the one deliberate exception to "empty deps at this
   stage," and it's exactly the crate 2.0 §3.5 names as "the only crate that
   touches a live `wgpu::Device`," so the dependency is architecturally
   correct even though it arrived earlier than its first real caller.
   `wgpui-core` (§3.1) stayed fully dependency-free — the "no live
   `wgpu::Device` anywhere in this crate" rule is about the crate's shipped
   library surface, and putting the wgpu-touching spike code in
   `wgpui-wgpu` instead keeps that rule intact from day one rather than
   needing a carve-out later.
6. **Spikes are `examples/`, not `benches/`.** The task brief allowed
   either. Plain `fn main()` examples avoid pulling in `criterion` (adds a
   plotting/HTML-report dependency chain not otherwise needed) and match
   this crate's own `examples/bench/*.rs` convention already used for
   `plain_scroll_10k`, `paths_bench`, etc. Each spike is one command:
   `cargo run -p wgpui-wgpu --example spike_a_ordering_occlusion --release`.

None of these change any behavior under `src/`; all are additive,
new-crate-only decisions.

---

## 3. GPU adapter honesty check

Per the task's explicit instruction, this was checked **before** trusting
any spike number: `crates/wgpui-wgpu/examples/adapter_probe.rs`, run via
`cargo run -p wgpui-wgpu --example adapter_probe --offline`.

**Result: a real discrete GPU is present on this machine.** This is not a
headless CI sandbox — it's the developer machine this session ran on
(Windows 11), and it has a real, driver-installed NVIDIA GPU:

```
== Adapters enumerated on VULKAN | DX12 ==
  name="NVIDIA GeForce RTX 4060 Laptop GPU" backend=Vulkan device_type=DiscreteGpu driver="NVIDIA" driver_info="561.03" vendor=0x10de device_id=0x28e0
  name="NVIDIA GeForce RTX 4060 Laptop GPU" backend=Dx12   device_type=DiscreteGpu driver="32.0.15.6103" vendor=0x10de device_id=0x28e0   (x3, one per DX12 feature-level probe)
  name="Microsoft Basic Render Driver"      backend=Dx12   device_type=Cpu        driver="10.0.26100.8972" vendor=0x1414 device_id=0x8c

== Picking the first adapter (this crate's established pattern) and requesting a device ==
  Selected adapter: name="NVIDIA GeForce RTX 4060 Laptop GPU" backend=Vulkan device_type=DiscreteGpu driver_info="561.03"
  software/CPU-fallback adapter: no
  request_device: OK
```

Both spikes below picked the same adapter (first entry from
`enumerate_adapters`, matching this codebase's own established pattern in
`src/platform/cross/render_context.rs` / `src/flamegraph_gpu.rs` — no
`compatible_surface` exists here to justify `request_adapter` instead).

**What this does and doesn't establish.** This *is* real target hardware in
the sense §8's gate means — a native Vulkan device on a real, currently-sold
discrete GPU, not WARP/llvmpipe/a software rasterizer (the "Microsoft Basic
Render Driver" `Cpu`-type adapter is enumerated but not selected by either
spike). That said, it is **one adapter, one driver version, one OS, one
class of GPU** (a laptop-class discrete NVIDIA part). It says nothing about:

- Integrated GPUs (Intel/AMD APUs), which are a large fraction of real
  desktop/laptop deployments and have materially different compute
  throughput and driver behavior.
- macOS/Metal or Linux/Vulkan-on-Mesa, which this crate also ships to.
- Older or lower-end discrete GPUs.
- The actual WASM/WebGPU path (§0 constraint 2's "best-effort" target),
  which wasn't touched by this spike at all.

A human should re-run both examples (`spike_a_ordering_occlusion`,
`spike_b_uniform_layout`) on at least one integrated-GPU machine and one
non-Windows target before treating the numbers below as representative of
"real target hardware" in the full multi-platform sense, not just "not a
software fallback."

---

## 4. Spike A — ordering + occlusion, GPU compute vs. CPU `BoundsTree`

**Code:** `crates/wgpui-wgpu/examples/spike_a_ordering_occlusion.rs`. Run:
`cargo run -p wgpui-wgpu --example spike_a_ordering_occlusion --release --offline`.

**Scene:** 100,000 quads as 200 spatially disjoint "clusters" (a 20×10 grid
of 200×200px regions), 500 quads/cluster: one opaque background quad, ~479
small randomly placed/sized quads (80% translucent / 20% opaque — genuine
local nesting and overlap), and 20 larger opaque "occluder" quads inserted
last per cluster (so they sort above the nested content, and several
genuinely cover multiple smaller quads beneath them). Clusters never overlap
each other, so all overlap/occlusion structure is local — see §4's own
"honest limitation" note below.

**CPU path**: a faithful, standalone port of `src/bounds_tree.rs`'s
`BoundsTree::insert` (identical AABB dynamic-tree structure and
max-intersecting-order-plus-one rule), fed one quad at a time exactly as the
real `Scene` does, then a `sort_by_key(order)` — the same shape as
`Scene::finish` (`scene.rs:734`). Occlusion is a simplified CPU pass (single
fully-containing, later-drawn, opaque rectangle — not R-N §8.3's full
corner-radius/border/backdrop-filter-aware test, since this synthetic scene
has none of those properties to test).

**GPU path**, one command encoder, one submit: (1) `relax` — a WGSL compute
pass solving `order[i] = 1 + max(order[j] : j<i in the same cluster,
overlapping)` by fixed-point Jacobi relaxation over ping-ponged buffers
(128 iterations); this is *the same recurrence* the CPU tree computes, not
an approximation of it — verified below by exact readback comparison, not
assumed. (2) `bitonic` — an in-place bitonic sort (153 stages for the
131,072-padded array) replacing the CPU `sort_by_key`. (3) `cull` — the same
per-cluster occluder containment test as the CPU reference, one invocation
per quad. Reported GPU time spans buffer creation, the initial upload, all
282 compute passes, and the final `poll(Wait)` — an end-to-end wall-clock
number, not an isolated on-device kernel time. Readback for correctness
validation happens *after* this window closes (a real Phase 3 consumer
wouldn't round-trip through the CPU either).

**Honest limitation of this synthetic scene, stated directly**: because
clusters are non-overlapping by construction, the GPU relax/cull kernels
bound their neighbor search to "the 500 quads in my own cluster" for free.
A generic scene without this structure would need real spatial partitioning
(a uniform grid, or literally the production `BoundsTree`) to get the same
bound — that's real engineering work Phase 3 still has to do, not something
this spike proves is free.

**Results** (4 runs, same machine, same binary):

| Run | CPU total | GPU total (end-to-end) | Speedup |
|---|---|---|---|
| 1 | 775.6 ms | 52.7 ms | 14.7x |
| 2 | 803.6 ms | 117.1 ms | 6.9x |
| 3 | 727.6 ms | 121.0 ms | 6.0x |
| 4 | 746.6 ms | 135.1 ms | 5.5x |

CPU breakdown (representative run): `BoundsTree` insert 739.6 ms,
`sort_by_key` 0.8 ms, occlusion cull 8.2 ms — **the tree insert utterly
dominates.** 41,896 / 100,000 quads (41.9%) were occlusion-culled, matching
between CPU and GPU exactly.

**Correctness, checked not assumed:** every run reports `order[]` and
`culled[]` matching the CPU reference **bit-for-bit, 0 mismatches out of
100,000**, and the relaxation's own convergence counter reads exactly 0 on
its last iteration (the ping-ponged relaxation had fully settled well
within the 128-iteration budget — the scene's actual max painter order was
73).

**Reading the variance honestly:** run 1's 52.7 ms is an outlier; runs 2–4
cluster around 117–135 ms. This is almost certainly first-process shader/
pipeline compilation and OS/driver cache state varying between separate
`cargo run` invocations (each run is a fresh process — nothing here shares
a pipeline cache across runs) rather than a real 2–3x swing in the
underlying compute cost. **Treat ~6–7x as the reproducible number, not
14x.** A production implementation, with pipelines built once at startup
and reused every frame, should land close to or better than the low end of
this range, not the outlier.

**One more honest observation:** 739.6 ms for 100,000 `BoundsTree` inserts
(~7.4 µs/insert) is slower than a simple O(n log n) estimate would suggest.
This wasn't debugged further — the port was validated as *faithful* (exact
match against the brute-force definition, same technique
`bounds_tree.rs`'s own `test_random_iterations` uses), not tuned — but it's
a real, reproducible property of the production algorithm applied to a
scene shaped like this one (many small, spatially separated clusters across
a wide canvas): the SAH-style half-perimeter heuristic doesn't obviously
keep spatially distant clusters apart in the tree, so `find_max_ordering`'s
pruning may be less effective than it would be for a single dense region.
If real UIs with many independent panels/widgets across a wide canvas hit
this same shape, it's an argument *for* Phase 3, not a benchmark artifact
to explain away.

---

## 5. Spike B — uniform-list layout, GPU compute kernel vs. CPU loop

**Code:** `crates/wgpui-wgpu/examples/spike_b_uniform_layout.rs`. Run:
`cargo run -p wgpui-wgpu --example spike_b_uniform_layout --release --offline`.

**CPU path**: the exact formula `uniform_list`'s `prepaint` uses today
(`src/elements/uniform_list.rs:551-553`, `item_origin = padded_bounds.origin
+ visual_scroll_offset + point(0, item_height * ix)`), computed for all
10,000 rows. (In production, `uniform_list` only computes the
scrolled-into-view subset — this spike deliberately computes the full item
count on *both* sides so the comparison is "N positions computed" on equal
terms, not one crediting the CPU side for a virtualization optimization the
GPU side isn't being asked to replicate.)

**GPU path**: one compute pass, `ceil(10000/64)` = 157 workgroups, each
invocation computing the identical formula from a 6-word uniform buffer
(item count, item height, container origin, scroll offset). Timed
end-to-end the same way as Spike A: buffer creation, the parameter upload,
the single dispatch, submit, and `poll(Wait)`.

**Results** (3 runs):

| Run | CPU total | GPU total (end-to-end) | Ratio |
|---|---|---|---|
| 1 | 28.8 µs | 31.5 ms | GPU **1095x slower** |
| 2 | 26.8 µs | 30.8 ms | GPU **1149x slower** |
| 3 | 20.9 µs | 22.2 ms | GPU **1063x slower** |

**Correctness:** `position[]` matches the CPU reference bit-for-bit,
0 mismatches out of 10,000, on every run.

**This spike does not win, and that is the honest, useful answer.** The
per-item computation is a single multiply-add — nanoseconds of real work
per row, ~20-29 µs total on the CPU for all 10,000 rows. A GPU dispatch's
fixed cost (buffer/pipeline setup, driver command submission, the
`poll(Wait)` round-trip to know the result is ready) is 20-32 **milliseconds**
on this hardware — three orders of magnitude larger than the work being
parallelized. No plausible per-item workload increase closes a 1000x gap;
this isn't "the kernel needs tuning," it's "a single isolated dispatch for
this problem shape cannot pay for itself here."

**What this does and doesn't rule out for Phase 6.1.** It rules out
"dispatch a GPU kernel per uniform-list layout, standalone, once per frame"
as its own operation — that's a net loss at this item count on this
hardware, almost certainly at any item count a real UI list actually
reaches (the fixed dispatch cost doesn't shrink, and the CPU cost is already
sub-microsecond-per-row). It does **not** rule out folding uniform-content
layout into a compute pass that's *already resident and already dispatching
anyway* — e.g. as one more pass inside the same command encoder Spike A's
ordering/occlusion work already builds, paying zero additional dispatch
overhead because the round-trip already exists for other reasons. That's a
materially different question from the one this spike answers, and Phase
6.1 should be scoped around it rather than around a standalone kernel.

---

## 6. Gate assessment

§8's Phase 0 gate: *"Numbers exist, on real target hardware, for the two
spikes that decide whether Phases 3 and 6.1 are worth building at all. If a
spike doesn't win, the corresponding phase is rescoped or dropped here, not
discovered mid-build."*

**Spike A (feeds Phase 3, ordering + occlusion): wins, with real
correctness validation, on real (if singular) hardware.** ~6-7x end-to-end
speedup reproducibly, up to 14.7x on one run; exact bit-for-bit agreement
with the production algorithm on both order and occlusion. Phase 3 is
justified going into it. The caveat is breadth, not direction: this is one
GPU, one driver, one OS — see §3's honesty note — and the synthetic scene's
neighbor-search bound (via non-overlapping clusters) is a simplification
that a real Phase 3 spatial-partitioning implementation still has to earn.

**Spike B (feeds Phase 6.1, regular-content layout): does not win as a
standalone operation, decisively (~1000x) and reproducibly, on the one
piece of hardware tested.** Per §8's own instruction, this is the moment to
rescope, not to discover this mid-build. Recommendation: **do not build a
standalone GPU dispatch for uniform-list/regular-content layout.** Phase
6.1 should instead be rescoped around folding regular-content position
computation into the *same* compute pass/encoder Phase 3's ordering or
indirect-arg-generation work already dispatches — the marginal cost of one
more pass inside an already-justified round-trip is a fundamentally
different (and still open) question this spike doesn't answer either way.
If that folded-in version is worth building is a question for whoever scopes
Phase 6.1's design, informed by Phase 3's actual shape once it exists — not
answerable by a Phase 0 spike in isolation, since it depends on Phase 3
infrastructure that doesn't exist yet.

**Before Phase 1 starts, a human should:**

1. Re-run both spikes (`cargo run -p wgpui-wgpu --example spike_a_ordering_occlusion --release`,
   `...spike_b_uniform_layout --release`) on at least one non-discrete-NVIDIA
   target — an integrated GPU and, ideally, macOS/Metal or Linux/Vulkan —
   to confirm Spike A's win and Spike B's loss both hold outside this one
   laptop's hardware/driver combination.
2. Treat Spike A's reproducible ~6-7x, not the 14.7x outlier, as the number
   that should inform Phase 3 scoping conversations.
3. Decide, with Spike B's result in hand, whether Phase 6.1 is worth
   scoping at all in its "folded into an existing pass" form, or whether
   it should be dropped from the plan entirely until/unless Phase 3's shape
   makes that folding concrete enough to spike again.

**Workspace scaffold (Part 1): done, verified, zero deviation in behavior.**
`cargo check --workspace` and the standalone `cargo check -p gpui-ce` both
pass, with `gpui-ce`'s warning output byte-for-byte unchanged from the
pre-workspace baseline. Phase 1 (the patch-list protocol, §8) can start
without anything further here.
