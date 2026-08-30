# Native platform support

The native 2.0 path currently provides a winit window, a validated wgpu
surface, resize handling, rendering, and the core retained input model.

The following OS integrations are not implemented: application menus, native
keyboard dispatch, prompts, clipboard, IME, accessibility, and cursor shape
hooks. They are intentionally not exported as placeholder APIs. Applications
must not treat their absence as successful platform support; a future
workstream must add each integration with an explicit error or capability
report before it is documented as available.

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
