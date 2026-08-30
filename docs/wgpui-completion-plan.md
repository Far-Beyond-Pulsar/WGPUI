# WGPUI 2.0 completion plan

This is the execution plan for taking the native backend from the current
phase-tested renderer to a production framework capable of replacing the
legacy implementation. The source of truth remains
[`gpu-native-architecture.md`](gpu-native-architecture.md); this document
turns its open items and the examples compiler report into implementation
workstreams.

## Definition of done

WGPUI 2.0 is not complete when the crates compile or when a synthetic GPU gate
passes. It is complete when all of the following are true:

1. Every target in `wgpui-examples-2` compiles against `wgpui`.
2. Every example launches and presents non-black, expected content.
3. Existing application code requires at most the documented crate-name and
   explicitly approved breaking changes.
4. Every supported legacy behavior has a native implementation, a deliberate
   documented incompatibility, or a tested fallback. No compatibility method
   silently does nothing.
5. Unchanged frames retain their scene, layout, atlas, GPU buffers, and cached
   boundaries. Updates upload only the affected ranges and invalidate only the
   affected work.
6. Device loss, resize, surface loss, atlas eviction, malformed input, and
   unsupported GPU features produce recoverable errors rather than panics.
7. Correctness is demonstrated by native-vs-legacy differential tests and by
   runtime tests, not only by unit tests of intermediate data structures.
8. The release candidate passes cold `check`, focused tests, strict clippy,
   example compilation, and the cross-platform capability matrix.

## Current baseline

The native rendering core already has substantial verified functionality:
retained patch application, ambient reconciliation, cache boundaries and
`.uncached()`, Taffy integration, GPU ordering/occlusion, indirect drawing,
two-axis tiled buffering, glyph/image atlases, quad/sprite/shadow/underline
pipelines, and a real Winit/WGPU window path.

The current example corpus is the more useful readiness metric: **45 targets,
2 compiling, 43 failing, 204 normalized compiler errors**. The first failure
layer is public framework plumbing, not the GPU quad path.

## Execution order and ownership

Each workstream must keep its write set narrow, add focused acceptance tests,
and commit independently. Agents must not add compile-only or no-op shims.

### Workstream A — application and entity lifecycle

Files: `wgpui-core/src/app`, `wgpui-core/src/window`,
`wgpui-wgpu/src/window/application.rs`.

Implement the native equivalent of the legacy application model:

- application construction and run-loop entry points;
- one or more live windows and their retained frame loops;
- `open_window`, close requests, quit, activation, menus, prompts, and window
  lifecycle callbacks;
- `Entity`, `Context`, listener, observer, notification, and task plumbing;
- render/view ownership and safe callback lifetimes;
- redraw coalescing and input-to-invalidation flow.

Acceptance: `hello_world`, `creating_components`, `window`,
`on_window_close_quit`, and `async_tasks` compile and launch; close/quit and a
two-window test complete without leaked windows or re-entrant entity updates.

### Workstream B — native text and content lowering

Files: `wgpui-core/src/element.rs`, `wgpui-text`, `wgpui-widgets/src/text.rs`,
`styled_text.rs`, and the application-owned atlas seam.

Implement raw `String` and `&'static str` as real text elements. The path must
shape, measure, rasterize, allocate atlas tiles, emit glyph runs, and use the
existing mono-sprite pipeline. The application must own shared text shaping,
rasterization, atlas residency, GPU texture synchronization, and invalidation
across frames. Add intrinsic sizing, wrapping, scale-factor handling, and
delta-only atlas uploads.

Acceptance: raw strings produce nonzero layout and glyph primitives; text-only
windows present readable pixels; repeated frames shape/rasterize/upload zero
new work; `text`, `emoji_display`, all karaoke examples, and the text portions
of the remaining examples run visibly.

### Workstream C — interaction and focus

Files: `div/events.rs`, `div/interactivity/hitbox.rs`,
`div/scroll_state.rs`, core window dispatch/focus modules.

Implement event registration and propagation for click, mouse, hover, keyboard,
focus, actions, capture, bubbling, and cancellation. Connect hit testing to
laid-out retained nodes and invalidation to the style cascade. Implement focus
handles, tab order, focus traversal, and listener closures without retaining
strong cycles.

Acceptance: interaction examples click and hover visibly, keyboard actions
fire once, focus traversal is deterministic, hitboxes follow resize and scroll,
and a stress test exercises thousands of handlers without per-frame rebuilds.

### Workstream D — complete style and element surface

Files: `styled.rs`, resolved style types, `div`, text, image, surface, overlay,
list, and scroll modules.

Port the generated legacy style API and implement every method against a real
resolved field, layout mutation, emission path, or interaction state. Close
the known gaps: gradients, patterns, dashed borders, opacity, overflow masks,
no-wrap, ellipsis, tracking, text-decoration fields, shadows, and backdrop
blur. Port the proc-macro-generated helpers rather than maintaining a hand-
written subset.

Acceptance: compile-time API coverage compares the legacy public method set to
the native set; every method has a behavior test; styling, layout, shadow,
gradient, pattern, opacity, and blur examples render expected output.

### Workstream E — scrolling, lists, and buffering

Files: `scroll`, `list`, `overlay`, tile-buffer integration.

Implement scroll handles, scroll physics/anchors, clipping, scrollbar state,
virtual and uniform list realization, two-axis tile residency, LRU eviction,
and retained transforms. CPU layout is allowed where Taffy is required; GPU
compute must remain responsible for visibility, ordering, and indirect draw
preparation where it wins.

Acceptance: 10k-row and virtual-list examples remain interactive, panning a
resident tile grid causes transform-only updates, crossing tile boundaries
emits only newly revealed content, and eviction never leaves stale GPU layers.

### Workstream F — images, SVG, animation, and surfaces

Files: image/cache/animation modules, `wgpu_surface`, `surface_registry`, atlas
upload and frame integration.

Close scaled-image filtering, animated GIF advancement, SVG lifetime, loading
and error states, custom WGPU surface identity, and the producer/consumer
surface synchronization path. Preserve the explicit WGPU surface fast path:
unchanged external textures must composite without rebuilding element content.

Acceptance: image, GIF, SVG, WGPU-surface, and surface-stress examples run for
30 seconds; resize/device/surface loss recovers; scaled image differential
tests include nearest-vs-interpolated edge cases; no surface producer frame is
consumed or dropped incorrectly.

### Workstream G — paths, canvas, gradients, and backdrop filters

Files: path and backdrop shaders/pipelines, canvas/path element modules, atlas
or intermediate-target management.

Audit every shader's actual contents before porting. Implement Lyon tessellation
only where the path API requires CPU geometry; keep fill, compositing, blur,
and repeated evaluation on the GPU. Add gradient/pattern data to the primitive
protocol with stable layouts and partial updates.

Acceptance: path and blur examples render through the native frame path; shader
inputs are validated at pipeline creation; differential tests cover edge,
rounded, transformed, clipped, translucent, and resized cases.

### Workstream H — GPU performance and retention

Files: scene stores, compute passes, upload planner, frame renderer, device and
surface lifecycle.

Implement bulk scene insertion/reservation, cross-kind occlusion where it is
profitable, sparse-visibility heuristics, fused regular-layout experiments
only when benchmarked, and complete retained-resource accounting. Keep CPU
fallbacks first-class for unsupported features and small workloads.

Acceptance: performance tests report CPU-known instance counts only on fallback
paths; steady frames have zero uploads and zero plan rebuilds; one primitive
change uploads one primitive range; large scene population is not O(n²); GPU
and fallback output are pixel-identical.

### Workstream I — platform and production hardening

Files: all native crates, device/surface configuration, platform input and
window modules, devtools.

Replace production panic paths with typed errors or explicit invariant checks;
remove stale placeholder claims and unjustified `dead_code` allowances;
complete app menus, prompts, clipboard, IME, accessibility hooks, cursor and
keyboard semantics, device-loss recovery, and devtools capture. Validate WGPU
feature negotiation on Vulkan, DX12, Metal, integrated GPUs, and fallback
adapters.

Acceptance: a fault-injection suite covers malformed patches, allocation
failure, atlas exhaustion, lost surfaces, device loss, and failed callbacks;
all supported platforms pass the capability matrix without hardcoded optional
features or present modes.

### Workstream J — compatibility and cutover

Files: `wgpui` public exports, legacy alias crate, macros, examples, manifests.

Generate a public API inventory from the legacy crate and compare it against
the native crate. Add only explicitly approved breaking changes to
`docs/wgpui-breaking-changes.md`. Keep the old implementation available as a
rollback tag, but make the new backend the default and remove facade methods
that only conceal missing functionality.

Acceptance: every example compiles and runs with only the permitted crate-name
change; public signatures and macro expansions are covered; output snapshots
and behavior traces have documented intentional differences only.

## Verification gates

Every workstream must pass its focused gate before the next wave consumes it.
The final gate runs, in order:

1. `cargo metadata --locked` and dependency/download audit.
2. Native source/shader scan for empty files, stale placeholders, TODOs, and
   unimplemented production paths.
3. Cold `cargo check --workspace --all-targets`.
4. Focused unit/integration/differential tests.
5. Strict cold clippy with the repository script.
6. Compile all 45 examples and consolidate zero errors.
7. Launch every example for a bounded runtime smoke test, capturing frame
   reports and screenshots.
8. Run retained-frame, resize, scrolling, eviction, device-loss, and
   multi-window stress tests.
9. Repeat the GPU gates on at least Vulkan, DX12, and Metal or document the
   unavailable adapter explicitly.

## Non-negotiable rules

- A method that cannot be implemented yet must remain absent or return a
  typed, visible error; it must not silently return `self`.
- A passing compile check is never evidence that a window rendered.
- A passing pixel test that only compares clear color is not a rendering gate.
- Every claim about retention includes a steady-frame counter and upload-byte
  assertion.
- Every GPU optimization has a measured fallback and a correctness oracle.
- The legacy implementation is an oracle and rollback path, not a permanent
  second implementation to maintain indefinitely.
