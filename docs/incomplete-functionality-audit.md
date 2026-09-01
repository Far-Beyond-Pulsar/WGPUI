# WGPUI incomplete-functionality audit

**Audit date:** 2026-09-01  
**Audited checkout:** `codex/wgpui-native-integration` in the main WGPUI checkout  
**Scope:** current `2.0`/native crates under `crates/`, the public WGPU application boundary, and the `wgpui-examples-2` runtime. The `old/` backend is legacy and was not treated as a current-native implementation target.

## Executive summary

The native crates compile and the example source set compiles, but the implementation is not feature-complete. The important remaining work is concentrated in invalidation, asset loading, error propagation, layout infrastructure, and several advertised widget surfaces. The scan found:

- 16 confirmed native behavioral or integration gaps, including 6 high-priority items;
- 11 files that are empty structural placeholders, of which 8 correspond to still-missing functionality and 3 duplicate functionality implemented elsewhere;
- 5 explicit platform or capability limitations that return a documented error or warning rather than silently pretending to work;
- several stale phase documents and source comments that incorrectly describe completed work as a stub;
- legacy/test-only placeholders that are intentionally outside this native audit.

This report distinguishes “not implemented”, “implemented elsewhere but the module is stale”, “explicitly unsupported”, and “implemented but incorrect”. A marker such as `return None` is not considered a gap by itself: it is included only where the surrounding public contract promises behavior that is not delivered.

## Confirmed native gaps

The priority is an engineering order, not a statement about API compatibility. These findings can be closed without moving the examples back to `wgpui_core::window::Window`; the public render-time boundary remains `wgpui_wgpu::window::application::Window`.

| ID | Priority | Location | Finding and impact | Ideal closure |
|---|---|---|---|---|
| APP-01 | P0 | `crates/wgpui-core/src/app.rs:168-183`; `crates/wgpui-wgpu/src/window/application.rs:2707-2755`; `crates/wgpui-wgpu/src/window/frame_loop.rs:649` | Entity notifications are delivered only to core observers. The native handler does not subscribe to them or translate them into a redraw/invalidation request. `FrameLoop::draw` requests another frame for animation/layout/debug tiles, but an ordinary state mutation can leave a window stale or white. | Add an app-to-window invalidation bridge at the public WGPU boundary. Coalesce entity changes per event-loop turn, mark the affected window dirty, and request redraw. Keep core backend-neutral; pass `interaction_mut()` into core lowering only at the boundary. Add a test that updates an entity, calls `cx.notify()`, and observes a native redraw request. |
| APP-02 | P0 | `crates/wgpui-core/src/app/context.rs:23,67-80`; `crates/wgpui-core/src/app/entity.rs:81-113` | `Context::notify()` only flips a flag that is never read. `observe_window_bounds` and `observe_window_appearance` ignore both arguments and return an empty entity subscription. `Entity::update`/`update_in` notify after every update regardless of whether `cx.notify()` was called. The public API therefore has the wrong notification semantics and two behaviorless observation methods. | Store and consume the notification state in the update path; preserve the documented distinction between an update and an explicit render invalidation. Implement window observers against the public window event stream, or remove the methods only through an explicitly versioned API decision. Add callback and “no callback without the matching change” tests. |
| INV-01 | P0 | `crates/wgpui-core/src/invalidation/request.rs:18-25,88-90`; `crates/wgpui-wgpu/src/window/application.rs:2707` | `InvalidationScope` has no `Entity(EntityId)` variant, and the frame creates `FrameSignals::new()` without populating it from app/entity changes. Retained reconciliation cannot receive typed entity invalidation from the runtime. | Add entity-scoped invalidation and an app-owned drain/coalescing queue. Populate `FrameSignals` for the frame that consumes those requests and preserve window/layer/instance filtering. Test entity, layer, and window scope independently. |
| ASSET-01 | P0 | `crates/wgpui-widgets/src/assets.rs:103-111`; `crates/wgpui-widgets/src/img.rs:579-600`; `crates/wgpui-widgets/src/svg.rs:218-236` | Deferred `img("…")` and `svg().path("…")` resolution calls `AssetRegistry::load_cached`, which hard-codes `NullHttpClient`; URI misses therefore cannot use the configured HTTP client. The cache-miss path also uses `futures::executor::block_on` during description building, blocking the render thread. | Make deferred resolution an asynchronous, app-owned asset request using the configured client and a loading state; never block the render/event thread. Keep source-ID/engine constructors authoritative and make deferred URI builders resolve through the same registry/client instead of inventing a second loader. Add HTTP, cache-hit, loading, failure, and cancellation tests. |
| ASSET-02 | P1 | `crates/wgpui-widgets/src/img.rs:715-732`; `crates/wgpui-widgets/src/svg.rs:251-271` | Direct `from_resource(Resource::Embedded(...))` treats the embedded identifier as a filesystem path and calls `std::fs::read`, bypassing `AssetSource` and configured embedded assets. | Route embedded resources through the configured `AssetSource`; retain filesystem access only for an explicitly filesystem-backed resource kind. Test an in-memory embedded source and verify no filesystem lookup is required. |
| TEXT-01 | P0 | `crates/wgpui-widgets/src/styled_text.rs:400-407,618-631` | A shaping failure panics only in tests and is silently discarded in release (`let _unreported = error`), causing the text element to emit no content and hiding the actual error. | Return a render/materialization error where the contract allows it, or emit a visible deterministic fallback and log the structured error with source context. Add a release-mode behavior test; do not silently discard shaping failures. |
| LAYOUT-01 | P1 | `crates/wgpui-layout/src/containment.rs`; `crates/wgpui-layout/src/wgpui_layout.rs:9-15`; `crates/wgpui-widgets/src/div.rs:527-548` | The containment/`estimated_size` module is empty. `Div::estimated_size` is an ad-hoc widget field, not a general element measurement contract, so unresolved content cannot participate in the promised fast layout path. | Define the element-level estimated-size contract, propagate it through layout, and fall back to exact layout whenever an estimate is absent or invalid. Add differential tests for estimated and exact paths, including nested content and changes to the estimate. |
| LAYOUT-02 | P1 | `crates/wgpui-layout/src/regular.rs`; `crates/wgpui-core/src/shaders/layout_uniform.wgsl:1-2`; `crates/wgpui-wgpu/src/render/compute/layout_pass.rs` | Regular-content GPU layout is an empty module/shader placeholder. The current implementation must continue using the CPU/Taffy path; there is no GPU regular-content kernel or dispatch. | Either implement the full kernel/dispatch with exact Taffy-equivalent rounding, min/max, gaps, and fallback behavior, or remove the advertised GPU-layout surface until it exists. Gate it by measured workload and test CPU-vs-GPU differential output before enabling it. |
| WIDGET-01 | P1 | `crates/wgpui-widgets/src/text.rs`; `crates/wgpui-widgets/src/list.rs`; `crates/wgpui-widgets/src/list/h_list.rs`; `crates/wgpui/src/lib.rs:122-127`/public reexports | The advertised standalone `text()` element is absent. `list()` and `h_list()` are advertised but there is no general list builder/element and `h_list.rs` contains state/transform types only; public reexports expose uniform/virtual list but not the complete advertised surface. Raw string children and `StyledText` do not close this API gap. | Implement each advertised public builder with layout, interaction, lowering, and virtualization semantics, or remove/deprecate the names from the public pre-1.0 surface. Export only APIs that have a complete render path. Add compile tests and focused behavior tests for empty, keyed, resized, and scrolled lists. |
| OVERLAY-01 | P1 | `crates/wgpui-widgets/src/overlay/deferred.rs`; `crates/wgpui-widgets/src/overlay/anchored.rs` | Deferred overlays are an empty module. Anchored positioning types exist, but there is no deferred overlay element or resolution/render path. | Implement deferred overlay ownership, anchor resolution, z-order, dismissal/focus routing, and invalidation; test anchor movement, window resize, and entity teardown. If the feature is intentionally postponed, remove it from public claims and document the supported anchored subset. |
| IMAGE-01 | P1 | `crates/wgpui-widgets/src/svg.rs` module docs and `crates/wgpui-widgets/src/img.rs:784-785`; `crates/wgpui-widgets/src/image_cache.rs:257-264` | The legacy tinted SVG alpha-mask path is unsupported. `Svg::text_color` delegates to `Img::tint`, which mutates cached RGBA frame data through a shared cache and ignores the tint operation's return value. One user's tint can therefore affect another user of the same source. | Keep source-ID/engine image construction authoritative, but represent tint as per-instance metadata or a derived immutable GPU/CPU view. Implement alpha-mask SVG tint separately if compatibility requires it. Add two-instance same-source/different-tint tests. |
| IMAGE-02 | P2 | `crates/wgpui-widgets/src/image_cache.rs:257-264`; `crates/wgpui-wgpu/src/render/shaders/poly_sprites.wgsl:26-36` | Native image scaling uses nearest-neighbor coordinate selection where the legacy renderer interpolates. This is a visible fidelity difference for scaled images. | Choose and document the native sampling contract, then implement linear interpolation (or an explicit compatibility mode) and compare representative opaque, translucent, and transparent assets. |
| SVG-01 | P2 | `crates/wgpui-widgets/src/svg.rs:332-349` | The public `overflow_hidden()` builder is a no-op. Its docs defer clipping to the parent description, so callers cannot rely on the method to establish a local clipping boundary. | Either lower the SVG clip into the retained scene or remove the method from the SVG builder and document that clipping is parent-owned. Test clipped path pixels and nested overflow behavior. |
| OCC-01 | P2 | `crates/wgpui-core/src/occlusion.rs:34` | Primitive instance occlusion exists, but layer-tier occlusion (skipping wholly covered layers during compositing) is explicitly absent. | Add a conservative layer coverage analysis keyed by clip/transform and preserve ordering; fall back to current compositing whenever coverage is uncertain. Test opaque, translucent, clipped, and transformed layers. |
| PLATFORM-01 | P2 | `crates/wgpui-wgpu/src/window/application.rs:886-912,900-912,2468-2477,3465` | Several public window operations are deliberately unavailable at the cross-platform Winit boundary: custom prompt labels/non-Windows native prompts, client insets (warning only), and options such as non-movable, app id, tabbing identifier, and display id. These are not silent stubs, but the capability surface is incomplete. | Model capabilities explicitly and return typed unsupported errors where the operation cannot be honored. Add platform adapters for supported options rather than logging and ignoring them; document the availability matrix and test every unsupported result. |
| RUNTIME-01 | P0 (example) | `crates/wgpui-examples-2/examples/learn/wgpu_surface.rs`; `crates/wgpui-wgpu/src/render/device.rs` | The full native example run leaves `wgpu_surface` crashing in Helio with a WGPU validation error: `TEXTURE_BINDING_ARRAY` is required by Helio's `GBuffer BGL 1` but was not enabled on the device. This is a third-party integration capability mismatch, not a missing WGPUI method. | Negotiate required WGPU features before device creation and fail with a clear capability error, or have Helio select a supported fallback path. Do not globally enable a feature without checking adapter support. Add a regression test around feature negotiation and make the example report the unsupported adapter instead of panicking. |

## Empty or structural placeholder modules

The following files contain documentation/`allow(dead_code)` but no implementation. A module being empty is itself a maintenance problem, but not every empty file is a distinct runtime gap.

| File | Classification | Required action |
|---|---|---|
| `crates/wgpui-core/src/app/effects.rs` | Missing deferred notification/effect implementation | Implement the queue/flush semantics or remove the module and its public claims. |
| `crates/wgpui-core/src/app/async_context.rs` | Missing/replaced async-context module | Reconcile the module boundary with the actual async context implementation; remove the dead placeholder if no public API depends on it. |
| `crates/wgpui-core/src/app/globals.rs` | Duplicate structural module; global logic currently lives inline in `app.rs` | Consolidate or delete the placeholder; do not leave an advertised module with no contents. |
| `crates/wgpui-layout/src/containment.rs` | Missing functionality; see LAYOUT-01 | Implement the generic estimate/containment contract. |
| `crates/wgpui-layout/src/regular.rs` | Missing functionality; see LAYOUT-02 | Implement or explicitly remove the regular-content GPU layout phase. |
| `crates/wgpui-wgpu/src/render/compute/layout_pass.rs` | Missing functionality; see LAYOUT-02 | Add the dispatch/lowering path or remove the exported empty module. |
| `crates/wgpui-wgpu/src/render/buffers/upload.rs` | Duplicate structural module; upload logic is in `buffers/slab_buffers.rs:132` | Consolidate naming/module ownership or remove the empty file. |
| `crates/wgpui-widgets/src/surface.rs` | Missing public surface widget implementation | Implement the advertised surface element or remove the module from the public API. |
| `crates/wgpui-widgets/src/text.rs` | Missing standalone text element; see WIDGET-01 | Implement the public element or remove the stale module. |
| `crates/wgpui-widgets/src/div/interactivity/layer_paint.rs` | Duplicate structural module; paint is implemented in `style.rs`/`Div::describe` | Consolidate or remove the empty module. |
| `crates/wgpui-widgets/src/overlay/deferred.rs` | Missing functionality; see OVERLAY-01 | Implement deferred overlays or make the anchored-only support explicit. |

The module-level documentation in `crates/wgpui-layout/src/wgpui_layout.rs` says `measure` is still a Phase 0 stub, but `measure.rs` contains a real `IntrinsicSize`/`Measure`/`LayoutSize` implementation. That statement should be corrected rather than used as evidence of a missing measure implementation.

## Intentional unsupported or fallback behavior

These are not “unfinished” in the same sense as a behaviorless method, but they must remain visible to users and tests:

- Cross-platform native prompts, custom labels, client insets, and several Winit window options are explicitly unsupported as listed under PLATFORM-01. They currently log or return typed errors; they should not be silently broadened.
- GPU timestamp/readback capture can return an unavailable/unsupported result. `wgpui-wgpu/src/render/capture.rs` models this explicitly.
- Devtools memory capture, inactive flamegraph capture, `NoopHooks`, and the first-paint surface-registry path are intentional no-op or unavailable states (`crates/wgpui-devtools/src/memory.rs`, `crates/wgpui-devtools/src/flamegraph/capture.rs:720`, `crates/wgpui-core/src/hooks.rs:349`, `crates/wgpui-wgpu/src/render/surface_registry.rs:519`). They are not production rendering gaps.
- `Scene::draw_ranges` is documented as a CPU placeholder (`crates/wgpui-core/src/patch/emit.rs:59`). It is a working CPU path, not a no-op; GPU ordering is a performance/architecture follow-up.
- Deferred URI image/SVG resolution is not safely supported by the current loader path. It must become asynchronous/configured-client-backed before being claimed as supported; silently falling back to a null client is not an acceptable closure.
- `old/` contains legacy test-platform and replay placeholders. It is excluded from the native implementation scope unless a compatibility requirement specifically reopens it.

## Already implemented; do not regress while closing gaps

The following items were checked specifically because their old documentation or earlier failures suggested they might still be missing:

- `Window::refresh()` exists in both `crates/wgpui-core/src/window.rs:271` and `crates/wgpui-wgpu/src/window/application.rs:677`; it emits a tracing warning that a full-window repaint should be avoided. It is not an API gap.
- The public WGPU render-time `Window` exposes `winit_window()` and `interaction_mut()`. Examples should keep using this public type; core remains backend-neutral.
- Animation metadata/advancement is implemented in `crates/wgpui-widgets/src/animation.rs` and `crates/wgpui-core/src/window/animation.rs`; stale image-cache docs still call those modules stubs.
- Div background/gradient/pattern/shadow/backdrop/border painting is implemented in `crates/wgpui-widgets/src/div/interactivity/style.rs` and lowered by `Div::describe`.
- Native shader files, including shadows and underlines, contain real shader code. Their top comments still describe the historical two-line placeholders.
- The atlas borrow/reuse failure that caused the prior group of example crashes has focused coverage and was fixed before this audit. It is not counted as an outstanding gap.

## Documentation drift

The following documents should be updated after the implementation priorities are agreed; they currently overstate old failures or understate completed work:

- `docs/wgpui-rc-audit.md` still says all 45 examples fail to compile. The examples now compile; the remaining native runtime issue observed in the consolidated run is `wgpu_surface`'s Helio feature mismatch.
- `docs/phase-6.2-results.md` and `docs/phase-6.6-results.md` still describe animation modules as three-line stubs.
- `docs/gpu-native-architecture.md` contains historical phase-table entries that describe old placeholder states as current. Preserve the history, but mark completed/deferred items with current status.
- `crates/wgpui-layout/src/wgpui_layout.rs`, `crates/wgpui-widgets/src/image_cache.rs`, and the comments in `render/shaders/shadows.wgsl`/`underlines.wgsl` contain stale placeholder language described above.

## Legacy and test-only scan results

The repository-wide marker scan also found `unimplemented!()` and no-op paths in legacy test platforms, compile-test action definitions, replay helpers, and diagnostics. Those are not current native release gaps:

- `old/src/platform/test/platform.rs` and `old/src/platform/test/window.rs` contain test backend placeholders.
- `tests/action_macros.rs` uses `unimplemented!()` in action compile fixtures.
- `old/src/flamegraph_replay.rs` intentionally uses checkerboard placeholders for replay surfaces.

If the compatibility contract requires those paths to become functional, they need a separate legacy-backend work item. Mixing them into the native WGPU completion work would obscure the public API boundary and risk changing preserved legacy behavior.

## Recommended closure order

1. Close APP-01, APP-02, and INV-01 together, with an end-to-end entity-update/redraw test. This addresses stale/white windows and establishes the invalidation contract needed by later work.
2. Close ASSET-01, ASSET-02, TEXT-01, and IMAGE-01 with asynchronous error-propagating tests. These are correctness and data-isolation issues, not cosmetic compatibility work.
3. Decide the public scope for LAYOUT-01, WIDGET-01, OVERLAY-01, and the empty modules. Implement the contract fully or remove/document the names before adding more examples.
4. Treat LAYOUT-02, IMAGE-02, and OCC-01 as measured performance/fidelity phases with differential tests and conservative fallbacks.
5. Add explicit capability negotiation for RUNTIME-01 and PLATFORM-01, then update the acceptance and phase documentation.

## Evidence and limitations

- The native example compilation pass completed; this audit does not claim that compilation proves runtime correctness.
- The consolidated release runtime pass launched 45 examples: 44 exited successfully after graceful close and `wgpu_surface` was the one remaining crash. Its stderr was captured under `target/examples-2-run/20260901-140743-378/` in `summary.json`, `summary.txt`, and `wgpu_surface.stderr.log`.
- Focused atlas tests (37) and the WGPU library tests (105) passed in the preceding verification.
- This is a source and targeted-runtime audit. It does not prove full parity for accessibility, IME, clipboard, menus, device loss/recovery, every platform adapter, or every interaction sequence. Those require platform-specific harnesses and should be added to the acceptance matrix rather than inferred from marker scans.
