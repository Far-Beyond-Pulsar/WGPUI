# WGPUI 2.0 breaking changes and migration policy

This document records the intentional source-level changes accepted for the
native WGPUI 2.0 API. It is the companion to the GPU architecture plan and
the examples migration corpus. The old implementation remains under `old/`
for comparison and recovery, but new framework code must not depend on it.

## Decisions

### Examples may use direct GPU dependencies

Examples that demonstrate custom rendering may depend directly on `wgpu`,
`bytemuck`, `helio`, and other explicitly required GPU crates. These
dependencies belong to the example crate and are not automatically promoted
to the core public API. Examples must still use the native WGPUI surface and
frame lifecycle where the example is demonstrating framework integration.

### Examples migrate to native names

The migration corpus uses the native naming and module organization. In
particular, `App` replaces the legacy `AppContext`, and examples should not
retain compatibility imports solely to preserve old spelling. Similar names
may be retained only when they describe the same public contract and do not
force the new backend to expose legacy implementation structure.

### The complete element framework is native

`Render`, `RenderOnce`, `IntoElement`, `Element`, the `IntoElement` derive,
stateful elements, and the supporting proc-macro surface are first-class
native implementations. They must feed the description/reconciliation/patch
pipeline directly. A compile-only shim, no-op trait, or re-export from the
legacy crate is not an acceptable implementation.

### Interaction and window APIs are fully implemented

Focus, hit testing, mouse and keyboard events, actions, menus, timers,
scrolling, lists, window controls, input handling, and close behavior must be
implemented against the native retained/GPU pipeline. Examples that exercise
these features are acceptance tests for behavior, not merely compile probes.

### Application construction and window ownership

The legacy application entry point is now `Application::new().run(|cx| { ... })`.
`App::open_window` queues a real native window request and the WGPU event loop
materializes it before rendering the root entity. The former direct retained
constructor cannot share the zero-argument `Application::new` name in Rust, so
it is deliberately preserved as `NativeApplication::new(options, builder)`.
This is an approved WGPUI 2.0 source break; code using the direct constructor
should migrate to `NativeApplication` (or `NativeApplication::with_window`).

## Migration rules

1. Port example source to `wgpui` and native names before adding framework API.
2. Add a native API only when its behavior and ownership model are defined.
3. Preserve legacy behavior where it does not conflict with the retained GPU
   architecture; document deliberate semantic differences here.
4. Do not add compatibility methods that silently discard events, drawing,
   state, or errors.
5. Every completed API family must gain focused correctness tests and at least
   one example must compile and run through the native window path.
6. The examples matrix remains red until all registered examples either pass
   or are explicitly removed with a documented replacement and rationale.

## Accepted source changes

The following are accepted breaking changes for the first native migration:

- `AppContext` becomes `App`.
- Legacy implementation-module paths are replaced by public native modules.
- Direct custom-GPU examples may use their own GPU dependencies.
- Types and methods may be reorganized when the public behavior is preserved
  and the new ownership/lifetime model is materially clearer.

No other source-level break should be introduced silently. Add it to this
document with a migration example before changing the examples corpus.

## Completion requirement

This document is not a waiver for missing functionality. The final native
release requires the full examples matrix to compile, the interactive examples
to run through real event dispatch, and the rendering results to be covered by
the correctness and differential tests described in the architecture plan.
