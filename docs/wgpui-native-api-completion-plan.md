# WGPUI Native API Completion Plan

Status: planning baseline for the native `wgpui` crate.

This document supersedes compatibility-façade work as the implementation plan
for the public API. The source of truth for the renderer and retention model
remains `docs/gpu-native-architecture.md`; this document applies that model to
the user-facing API that is still missing.

## 1. Decision: native API first

`wgpui` is the canonical public crate. It must expose the real implementation
in `wgpui-core`, `wgpui-layout`, `wgpui-text`, `wgpui-widgets`, and
`wgpui-wgpu`.

`wgpui-compat` is frozen. It may remain in the repository as a historical
migration experiment, but no new functionality, shims, or example work should
be added to it. It is not a dependency of `wgpui`, and the native crates must
never depend on the old implementation under `old/`.

The old package remains under `old/` only for differential tests, reference
behavior, and temporary example source. It is not part of the native runtime.

The intended migration target is:

```text
use wgpui::*;
     │
     ├── public frontend contract
     ├── retained Description tree
     ├── CPU layout only where justified
     ├── delta ScenePatch
     └── GPU compute, indirect draw, tile buffers, and presentation
```

## 2. Compatibility policy

Each legacy public symbol must be classified before implementation:

1. **Same API, same meaning** — preserve the spelling and signature.
2. **Same API, new retained implementation** — preserve the spelling while
   changing the internal execution model.
3. **Small intentional change** — change only where the old contract exposes
   an immediate/CPU assumption that conflicts with retained GPU execution.
4. **Removed or replaced** — only for an API that would make correctness or
   lifetime guarantees impossible; provide a documented migration path.

No API should be omitted merely because its backend implementation is not
written yet. Conversely, no API should be added as a no-op just to make an
example compile.

The compatibility report for every workstream must include the old signature,
the new signature, the semantic difference, and a source-level migration
example. Public API tests must compile representative applications, not just
type-check isolated declarations.

## 3. Workstreams

The workstreams are ordered by dependency, but independent groups may be
implemented concurrently when their write sets do not overlap.

### W1 — Application, windows, and presentation

Build the real native lifecycle in `wgpui-wgpu` and expose it through
`wgpui`:

- `Application::new`, `run`, and shutdown behavior.
- `App`, `Window`, `WindowHandle`, `WindowOptions`, `DisplayId`.
- Winit event dispatch, redraw scheduling, resize, scale-factor changes,
  close requests, and device/surface loss recovery.
- The complete frame path: render description, reconcile, patch, scene,
  compute passes, indirect draw, and swapchain presentation.
- A continuous frame loop that does not rebuild unchanged content.

The lifecycle must use the existing `WindowSurface` and `FrameLoop`, not a
second renderer. A window callback returning must not silently terminate the
application unless the caller requested shutdown.

Gate: a real native window renders a retained `div` tree, responds to resize
and close, and reports frame/retention counters. The test must detect a missing
window or a no-op frame loop.

### W2 — Entities, contexts, state, and tasks

Connect the existing core state/entity mechanisms to the public frontend:

- `Entity<T>`, `WeakEntity<T>`, `Context<T>`, `App` access.
- `read`, `read_with`, `update`, and `update_in` borrowing contracts.
- `cx.notify`, observers, subscriptions, and entity events.
- `Task<T>`, foreground async tasks, background tasks, cancellation, detach,
  and error propagation.
- `Window::use_state` and retained state identity.

The foreground UI remains single-threaded. Background work must not mutate the
scene directly; it returns data that is applied through the normal invalidation
and patch path.

Gate: an example updates entity state at high frequency while an unrelated
subtree remains retained; counters prove only the affected description,
primitive slots, and upload ranges change.

### W3 — Render traits and declarative element contract

Define the native equivalents of `Render`, `RenderOnce`, `IntoElement`, and
component composition:

- Preserve the legacy method shapes wherever they do not encode immediate
  painting.
- Make `render` produce `Description` data or an element that lowers to it.
- Support children, conditional children, keyed children, fragments, and
  component-local state.
- Complete the derive macro without a dependency on old macros.
- Preserve stable element identity across render calls.

The derive macro must generate real element lowering, not merely allocate an
empty description with the type name.

Gate: a three-level component tree with keyed insertion, removal, stateful
children, and unchanged siblings passes reconciliation identity and pixel
correctness tests.

### W4 — Style system and layout surface

Complete the public style API over the existing resolved style and Taffy
layout:

- Dimensions, spacing, flex, grid, alignment, positioning, and overflow.
- Backgrounds, borders, per-side widths, dashed borders, radii, gradients,
  patterns, opacity, and clipping.
- Font, line-height, text alignment, wrapping, decorations, and highlights.
- `when`, `when_some`, responsive/conditional style composition.
- `estimated_size` and intrinsic sizing for unresolved content.

Style values should be resolved into retained state. A style-only change must
not reshape text or rebuild unrelated primitives.

Gate: legacy style cases are compared against an independent oracle for cold,
steady-state, resize, opacity, clipping, gradient, and per-corner cases.

### W5 — Events, hit testing, focus, and actions

Implement input without forcing a full scene rebuild:

- Pointer movement, click, press, release, drag, drop, scroll, and gestures.
- Hit-test acceleration over retained layout bounds.
- Focus handles, focus scopes, tab traversal, focus-visible state.
- Keyboard events, actions, key bindings, dispatch, and listener helpers.
- Pointer capture and event propagation/cancellation.

Hit testing may use a CPU spatial index when that is faster and more reliable;
it must consume retained layout/coverage data rather than repainting to answer
an input query.

Gate: a stress scene with 100,000 non-interactive nodes and a small interactive
set resolves input without visiting or rebuilding the whole tree.

### W6 — Text and font API completion

Finish the public text contract over the existing cosmic-text pipeline:

- Font registration, fallback, features, weights, styles, and metrics.
- `SharedString`, styled runs, highlights, underline, strikethrough, wrapping,
  measurement, and selection geometry.
- Correct scale-factor and subpixel cache keys.
- Atlas residency, eviction invalidation, and delta uploads.
- Text inputs and IME composition hooks needed by examples.

Shaping remains CPU-side because it is irregular and latency-sensitive. Glyph
raster data and rendering remain GPU-resident after upload.

Gate: differential shaping/raster tests, atlas eviction tests, IME/input tests,
and a long scrolling text scene with zero steady-state shaping or uploads.

### W7 — Images, SVG, animation, and assets

Expose the existing image and animation mechanisms as a stable native API:

- Image loading, decoding, caching, natural size, scaling, and object fit.
- PNG, JPEG, GIF, WebP, and SVG behavior with explicit failure reporting.
- Animated frame advancement and frame timing.
- Polychrome atlas residency and eviction.
- Retained `Img`/`Svg` identity and delta-only texture uploads.

Scaling and blending must be compared against legacy before choosing an
intentional semantic difference. Asset loading must not block the UI thread.

Gate: real files render through the new window path, animated images advance,
and repeated frames produce no decode or upload work when unchanged.

### W8 — Drawing, paths, gradients, blur, and custom content

Complete the remaining drawing capabilities with a declarative retained
resource model:

- Path descriptions and tessellation policy.
- GPU path rendering where practical; CPU tessellation only when it wins or is
  required by the selected GPU path algorithm.
- Gradients, patterns, dashed geometry, shadows, underlines, and opacity.
- Backdrop blur and filter regions.
- A custom drawing API that records a portable description rather than accepts
  arbitrary immediate GPU commands.

Each primitive needs an explicit cache/residency policy and an independent
shader differential. No legacy Lyon builder should leak into the public API by
accident.

Gate: byte-exact or formally bounded differential tests for each primitive,
mixed primitive ordering, transparency, clipping, and device feature fallback.

### W9 — Lists, scrolling, and surfaces

Complete high-volume UI primitives:

- Virtual and uniform lists with retained item identity.
- Two-axis scrolling and tile-based scroll buffers.
- Scroll anchoring, scrollbar behavior, overscroll, and smooth scrolling.
- `WgpuSurface` embedding, generation/backpressure, resize, and the obvious
  covered-surface fast path.
- Explicit cache boundaries and `.uncached()` on all relevant elements.

The list API should preserve legacy builders while lowering only visible or
newly revealed content. Panning resident tiles must remain transform-only.

Gate: 100,000-item scrolling, two-axis tile panning, tile eviction, embedded
surface occlusion, and unchanged-content upload counters.

### W10 — Platform, menus, devtools, and test support

Finish the integration surface needed by real applications:

- Native window positioning, menus, clipboard, cursor, scale factor, and
  platform options.
- Accessibility hooks where the legacy contract exposes them.
- Devtools inspection, captures, replay, flamegraphs, and render counters.
- Test application/window harnesses, deterministic clocks, and input drivers.
- WASM/WebGPU and non-Windows backend feature selection.

Platform behavior belongs behind traits so the core retained model stays
portable. Test support must be able to run headless without pretending that a
presenting test passed.

Gate: platform-specific behavior is explicitly marked, deterministic headless
tests pass, and at least one real native backend exercises the full path.

## 4. Cross-cutting correctness requirements

Every workstream must preserve these invariants:

- Unchanged descriptions retain identity and do not re-layout, re-emit,
  re-upload, or redraw without a reason.
- Only changed byte ranges upload; a one-primitive update must not upload the
  surrounding slab.
- `.boundary()` recomposites retained content; `.uncached()` forces the
  documented subtree behavior without destroying unrelated state.
- Resize, scale-factor changes, atlas eviction, tile eviction, device loss,
  and surface generation changes invalidate exactly the affected resources.
- CPU and GPU indirect/direct/fallback paths agree on output and draw order.
- No uninitialized, stale, magenta, transparent-clearing, or torn pixels.
- No hidden dependency on `old/`, `gpui-compat`, or legacy macros.
- No public method silently succeeds without performing its documented action.

## 5. Team execution model

Each agent owns one workstream and its tests. Shared public types must be
changed through a short design checkpoint before parallel agents edit them.

Required deliverables for every workstream:

1. Implementation and focused modules with no placeholder files.
2. API inventory entries and migration examples.
3. Independent correctness tests that can fail when the implementation is
   removed or bypassed.
4. Retention/delta-upload counters for cold versus steady state.
5. Targeted `cargo check`, tests, release tests, and strict clippy.
6. A report naming unsupported platforms and unresolved differences.

Agents must commit and push coherent checkpoints. A passing compile matrix is
not sufficient if the binary exits without creating a window or if the code
routes through `old/`.

## 6. Completion gates

The native API is complete only when all of the following are true:

- `wgpui` is the only supported application-facing crate.
- `wgpui-compat` has received no new implementation work and can be deleted
  without affecting `wgpui`.
- All declared examples compile against `wgpui` with only the crate-name
  change, or have an explicitly reviewed migration diff.
- Representative examples run through a real window and present frames.
- All public API groups in this plan have implementation and differential
  coverage.
- The full Phase 9 dependency, file-integrity, rendering, retention,
  determinism, stress, and cross-platform audit passes.
- Strict clippy is clean for native crates with no warning suppressions added
  to conceal unfinished behavior.

Until these gates pass, a compatibility count must be reported as a compile
count only and must never be described as runtime or backend parity.
