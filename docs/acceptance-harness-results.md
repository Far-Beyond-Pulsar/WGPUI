# Cross-system acceptance harness

The cross-system harness is in
[`crates/wgpui-wgpu/tests/acceptance_harness.rs`](../crates/wgpui-wgpu/tests/acceptance_harness.rs).
It exercises the production `ScenePatch`, `Scene`, `Compositor`, input indexes,
and WGPU frame renderer together. The GPU portions use the repository's
headless adapter helper; when no adapter is available they report the reason
and skip instead of manufacturing a result.

## Covered behavior

The harness currently verifies:

- diagnostics-disabled, diagnostics-enabled, and capture-mode frame work;
  disabled and capture modes have no debug overlay work, while the enabled
  mode has a bounded overlay budget;
- identical scene work across diagnostics modes, exact output-pixel equality
  for disabled versus capture, and intentional pixel difference for the debug
  overlay;
- delta patch uploads, including the exact changed primitive range and byte
  count;
- stable primitive ordering and equivalent output across every draw mode
  supported by the current device, including CPU-readback fallback;
- tiled reveal and scroll damage, transform-only scroll resolution, sibling
  tile-layer separation, tile-count bounds, and invalid tile configuration;
- thousands of tile records and input listeners/dispatch nodes;
- indirect-draw capability selection and fallback when required device
  features are absent; and
- separation of content invalidation from input-only hitbox/dispatch changes.

The budgets are deterministic count/byte budgets rather than wall-clock
budgets. Timing thresholds are intentionally not asserted: adapter, driver,
debugger, and CI scheduling variance would make such thresholds flaky. The
disabled budget is zero debug tiles and zero capture slots; enabled permits one
debug tile and no capture slots; capture permits no debug tiles and at most two
primitive records per primitive kind.

## Explicit unsupported paths

The requested `docs/devtools-and-diagnostics-plan.md` is not present in this
checkout. A direct read of that exact path failed with PowerShell's
`Cannot find path ... because it does not exist` error. This acceptance file
therefore records the evidence available from the checked-in architecture and
does not infer requirements from a missing document.

There is no production deep capture/replay runtime in this repository. The
capture mode in the harness is deliberately test-local: it records the
renderer-produced slot table for comparison and adds no public API or
per-frame capture work when diagnostics are disabled. The existing
`wgpui-devtools` boundary is backend-neutral and bounded, but it is not a
WGPUI frame recorder. Adding a recorder would require a separate API and
storage/serialization design, so it is outside this acceptance-only change.

Nested scroll-root ownership is also not implemented by the current
architecture. `Compositor::visit_tiled` and `Compositor::resolve` operate on a
single `BoundaryId` at a time; there is no parent/child scroll-root ownership
graph or nested clipping contract to exercise. The harness tests independent
sibling roots and documents this gap rather than pretending that they are
nested roots.

The current patch pipeline reports exact changed layer upload ranges, but it
does not expose a region-to-tile damage planner. The harness consequently
checks exact primitive upload deltas and compositor tile reveal/scroll damage
separately. A full nested regional-damage acceptance test must wait for the
missing ownership and planner APIs.

The harness validates device-feature selection and the renderer's draw-mode
fallbacks. It does not synthesize a real swapchain surface capability list:
headless WGPU contexts do not expose a window surface, and constructing one
would make the test platform-specific. Surface-format/present-mode fallback
coverage remains in the existing `wgpui-wgpu` window tests.

Only the locally available NVIDIA GeForce RTX 4060 Laptop GPU/Vulkan adapter
was observed while validating this work. On machines without a usable WGPU
adapter, the GPU-dependent acceptance tests are skipped with the helper's
concrete adapter error; the core assertions remain deterministic and are still
compiled as part of the integration test.

No files under `old/` or legacy backends were changed.
