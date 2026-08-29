# Phase 6.4 Results — Lyon Paths and Backdrop Blur

Branch: `wgpui-2.0/phase-6.4-paths-backdrop-blur`.

The phase started from the already-pushed core/protocol and GPU-pipeline work
in `2d09bc3017` and `c82c1c9773`. The preserved test/dependency checkpoint was
`d418c1644f`; the completed differential harness and shader validation fix were
pushed in `263e0c3670`.

## 0. Gate result

**Met for the covered cases: exact legacy-shader differential output on a real
adapter and readback.**

Reference adapter:

```text
NVIDIA GeForce RTX 4060 Laptop GPU
Vulkan / DiscreteGpu / driver 561.03
INDIRECT_FIRST_INSTANCE=true
MULTI_DRAW_INDIRECT_COUNT=true
Windows 11 Home 25H2, x86_64
```

The Phase 6.4 gate rendered 6,144 pixels per arm for each primitive and compared
all 24,576 RGBA bytes, with no tolerance:

```text
Lyon path versus compiled legacy paths.wgsl:       6,144 / 6,144 pixels exact
Backdrop blur versus compiled legacy backdrop:     6,144 / 6,144 pixels exact
```

The path case uses Lyon's real `FillTessellator` to produce a triangle-list
buffer, applies it through `ScenePatch`, uploads it through the path arena, and
renders it through `FrameRenderer` before readback. The comparison arm compiles
the frozen legacy `vs_path`/`fs_path` entry points and feeds them the equivalent
legacy vertex layout.

The backdrop case renders a red base quad, snapshots the rendered target, and
then compares the new two-pass `FrameRenderer` result with a separately encoded
legacy `vs_backdrop_filter`/`fs_backdrop_filter` pass. It exercises a rounded
4-pixel filter region with a 4-pixel blur at the red/black boundary.

## 1. Verification commands

All of these passed on the reference adapter unless noted:

| Command | Result |
|---|---|
| `cargo test -p wgpui-wgpu --test paths_backdrop_differential -- --nocapture` | 4 passed, 0 failed; both differential arms ran on the RTX 4060/Vulkan adapter |
| `cargo test -p wgpui-core --lib` | 332 passed, 0 failed |
| `cargo test -p wgpui-wgpu --test indirect_draw -- --nocapture` | 5 passed, 0 failed |
| `cargo test -p wgpui-wgpu --tests` | 140 passed, 0 failed across all library/integration test binaries |
| `git diff --check` | passed before the implementation checkpoint |
| `rustfmt --edition 2024 crates/wgpui-wgpu/tests/paths_backdrop_differential.rs` | passed |
| `./script/clippy.ps1` | Cargo clippy reported 130 pre-existing `gpui-ce` warnings-as-errors under `--deny warnings`; the wrapper returned 0 without propagating Cargo's failure |

`cargo fmt --all -- --check` could not be used as a repository-wide pass/fail
signal: the existing tree contains unrelated formatting drift and the command
also reports the missing `src/_ownership_and_data_flow.rs` module. The new test
file itself was formatted directly. The final clippy invocation was run through
the repository-required wrapper with no lint suppressions. It emitted errors
such as unused imports, redundant clones, `unsafe_op_in_unsafe_fn`, and other
warnings-as-errors throughout the existing root `src/` tree, ending with
`could not compile gpui-ce (lib test) due to 130 previous errors`. The wrapper
does not `exit $LASTEXITCODE`, so PowerShell reported process exit code 0 even
though Cargo's clippy run failed. No Phase 6.4-specific diagnostic was observed
in that output; the phase's targeted and package tests remain green.

## 2. What the tests prove

`crates/wgpui-wgpu/tests/paths_backdrop_differential.rs` now has four gates:

1. A Lyon-generated path reaches the real GPU, is rendered by the new path
   pipeline, and matches a real render through the frozen legacy path shader.
2. A backdrop filter reaches the real snapshot-copy pass and matches a real
   render through the frozen legacy backdrop shader.
3. The renderer returns `FrameError::BackdropSourceUnavailable` when a filter
   is requested without a copyable target source.
4. The legacy shader files are the actual sources used by the differential
   arms; the comparison arms compile their entry points, so a placeholder
   comment cannot make the gate pass.

The legacy backdrop record was derived from its WGSL layout rather than from
the new protocol layout: `order` occupies bytes 0–3, `blur_radius` bytes 4–7,
`bounds` starts at byte 8, and the remaining fields follow through the 64-byte
record. The first draft incorrectly supplied the new layout to the legacy
shader and failed at the rounded edge. That oracle error was corrected from
the legacy struct declaration; assertions were not weakened.

The first real-adapter run also caught a production WGSL validation error:
the new backdrop shader's integer `filter_index` varying lacked explicit flat
interpolation. The shader now declares `@interpolate(flat)`, matching the
legacy contract and WGSL validation requirements.

Phase 6.4 adds direct-draw statistics for paths and backdrop filters to the
CPU-readback accounting. Those helpers now report known direct instance counts
instead of turning the complete fallback frame into `None`. The existing
`indirect_draw` expectations were updated for the two conditional/direct kinds:
a no-backdrop scene visits 36 slots and issues 18 per-slot instanced calls,
while the full scene slot table still contains 42 `(layer, kind)` entries.

## 3. Changed files

| File | Change |
|---|---|
| `Cargo.lock` | Lock the Lyon dependency used by the GPU differential test |
| `crates/wgpui-wgpu/Cargo.toml` | Add Lyon as a dev dependency |
| `crates/wgpui-wgpu/src/render/shaders/backdrop_blur.wgsl` | Declare the integer filter index as flat-interpolated |
| `crates/wgpui-wgpu/src/render/draw.rs` | Preserve known CPU instance counts for direct path/backdrop draws |
| `crates/wgpui-wgpu/tests/paths_backdrop_differential.rs` | Compile legacy shaders, render both arms on a real adapter, and compare full readbacks |
| `crates/wgpui-wgpu/tests/indirect_draw.rs` | Account for conditional backdrop visitation and direct path/backdrop kinds |
| `docs/phase-6.4-results.md` | Record exact results, caveats, changed files, and remaining gaps |

The earlier Phase 6.4 commits also contain the core `Path`/
`BackdropFilter` protocol, scene stores, upload plumbing, pipelines, snapshot
copy, and shader ports.

## 4. Coverage limits and remaining gaps

- The path differential exercises Lyon fill tessellation and a filled triangle.
  It does not yet exercise Lyon stroke tessellation, curved-edge `st` SDF
  vertices, holes, self-intersections, or a clipped path. `Path::from_lyon`
  currently assigns `[0, 1]` to every tessellated vertex, as required for the
  fill case.
- Both differential cases use a full-viewport content clip, one layer, and one
  path/filter. Per-primitive clipping, multiple layers, multiple filters in a
  layer, and interactions with text/sprites/composites remain open.
- Backdrop coverage includes a nonzero Gaussian blur and rounded corners. The
  zero/small-radius sampling branch, opacity less than one, asymmetric corner
  radii, filter regions at viewport boundaries, and more varied source images
  are not differentially covered.
- The oracle is the frozen legacy shader compiled and rendered independently
  through a separate test pipeline. It is therefore stronger than a copied
  comment or a CPU transcription, but it does not establish that either arm
  matches an independent CPU raster/blur model.
- Verification is one Windows/NVIDIA/Vulkan adapter. Other backends and
  software adapters have not been run here.
- No higher-level widget/emitter integration was added in this phase; the
  primitive protocol and GPU path are directly testable, but production callers
  still need to emit these kinds deliberately.

`docs/gpu-native-architecture.md` was not edited; the parent task owns the
architecture-spec update.
