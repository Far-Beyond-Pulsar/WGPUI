# Native platform support

The native 2.0 path currently provides a winit window, a validated wgpu
surface, resize handling, rendering, and the core retained input model.

The native input seam now normalizes keyboard, text, IME, modifier, mouse,
wheel, focus, and click-count events, and exposes text clipboard read/write
with typed failures. IME preedit selections are converted to UTF-16 offsets as
expected by native text systems. This seam does not provide a default text
editor or input method UI; applications still own text-buffer behavior.

The following OS integrations remain unimplemented: application menus,
prompts, image/file clipboard formats, accessibility, and cursor shape hooks.
Clipboard access is synchronous and text-only, and IME positioning/candidate
window integration is not exposed. Applications must not treat these missing
integrations as successful platform support; each needs an explicit capability
or error contract before it is documented as available.

Surface creation is supported only when the adapter/surface pair offers the
pipeline's fixed `Rgba8Unorm` format, `COPY_SRC`, at least one alpha mode, and
at least one present mode. A surface that does not meet those requirements is
rejected with a typed error. Device loss and swapchain loss remain recoverable
at the frame boundary: the current frame is skipped or reported lost and the
caller can recreate the window/device according to its lifecycle policy.

The native path has been exercised on Windows with Vulkan and DX12. Other
platforms are not claimed as supported by this document; backend availability,
surface formats, and OS integration requirements must be verified on the
target platform.
