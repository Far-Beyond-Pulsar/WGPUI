# W1 — Native application lifecycle

W1 promotes the real WGPU window path into the native public API. The native
`wgpui` crate now exports `Application`, `Window`, `WindowHandle`,
`WindowOptions`, `DisplayId`, `FrameReport`, and `ApplicationError` directly
from `wgpui-wgpu`; it does not depend on `wgpui-compat` or `old/`.

`Application::run` creates a winit event loop and a real OS window, opens a
`WindowSurface`, keeps one `FrameLoop` alive for the window lifetime, and on
each redraw executes `Description -> Reconciler -> layout -> Emitter ->
ScenePatch -> Scene -> compute/indirect render -> present`. The redraw loop is
continuous until close, callback shutdown, or an explicit frame limit. Resize
events are coalesced by `ResizeDetector`, scale-factor changes update the
native window state, and `WindowSurface::acquire` handles stale/lost surface
images by reconfiguring and retrying.

## Behavioral gate

`crates/wgpui-wgpu/tests/application_lifecycle.rs` runs the native application
with a two-frame limit and an atomic callback counter. It passed on the real
Windows desktop adapter, proving that `run` does not return immediately and
that the callback is invoked for multiple presented frames. The existing
`window_present` gate continues to provide pixel-level swapchain validation.

Device loss is surfaced as a render error rather than hidden behind a fake
frame. Reopening a lost device requires rebuilding all device-owned pipelines,
buffers, atlas textures, and surface state and is intentionally a later
recovery workstream; surface-outdated/lost image recovery is implemented here.

## Verification

- Native lifecycle gate: passed (real window, two frames).
- `cargo check -p wgpui-wgpu`: passed before unrelated shared-checkout W2/W4
  edits made the workspace graph fail.
- Current combined check is blocked by pre-existing shared edits: duplicate
  `TextMeasurement`, duplicate `decode_async`, and an `EntityError` re-export
  mismatch. No W1 file causes those diagnostics.
