# Phase 8 compatibility probe

Status: probe and first façade slice implemented on `wgpui-2.0/phase-8-compatibility-probe`.

## Scope and method

The base is the latest `phase-6.5-animation` checkout (`30317dfbbd`), which
contains the merged 2.0 work through Phase 6.4 plus the Phase 6.5 animation
changes. The root `gpui-ce` package, its legacy backend, and its examples were
not cut over or edited.

`crates/wgpui-compat` is a separate package whose library target is named
`gpui`. Its examples point directly at the root package's explicitly declared
example files. The example sources are unchanged; only the dependency's crate
provider changes. Direct example dependencies are declared in the harness so
diagnostics measure the façade rather than accidental manifest omissions.

The coverage test compares the root and harness `[[example]]` name sets, so a
new root example makes the compatibility test fail until it is added to the
probe. Run the matrix with:

```text
cargo check --manifest-path crates/wgpui-compat/Cargo.toml --examples
cargo test -p wgpui-compat --test facade_exports --test manifest_coverage
```

## Coverage matrix

The repository declares 41 examples. On the final probe, 21 compiled and 20
failed. “Pass” means Rust type-checking completed; no example was run, so this
is not runtime compatibility.

Pass: `karaoke_text`, `text_gradients`, `creating_components`, `layout`,
`async_tasks`, `animation`, `emoji_display`, `wgpu_surface`,
`wgpu_surface_quad`, `wgpu_surface_stress`, `virtual_list`, `pattern`,
`focus_visible`, `gif_viewer`, `gradient`, `input`, `opacity`, `scrollable`,
`tab_stop`, `uniform_list`, `window_shadow`.

Fail: `karaoke_app`, `interactive_elements`, `styling`, `custom_drawing`,
`text`, `wgpu_surface_basic`, `mouse_events`, `blur_showcase`,
`smooth_scrolling`, `data_table`, `plain_scroll_10k`, `paths_bench`, `shadow`,
`hello_world`, `image_loading`, `on_window_close_quit`, `svg`, `tree`,
`window`, `window_positioning`.

## Failure classification

| Class | Evidence | State |
| --- | --- | --- |
| Missing type | `App`, `Application`, `Context`, `Bounds`, `WindowBounds`, `WindowOptions`, `Hsla`, input/focus/window types, and widget handles | Open; these are the main blockers |
| Missing trait | `Render`, `RenderOnce`, and legacy `IntoElement` | Open; no compatible 2.0 trait contract exists yet |
| Missing macro | `IntoElement` derives and `actions!`-provided action symbols | Open; the legacy proc-macro expansion cannot be reused without a façade contract |
| Missing method/signature | Not reached consistently because imports and traits fail first | Requires a second probe after façade lifecycle types exist |
| Feature | No feature-gated compiler failure remained after the harness declared the repository's example dependencies | No current evidence |
| Dependency | The initial harness exposed `wgpu`, Helio, `anyhow`, `rand`, `glam`, `bytemuck`, `env_logger`, and `unicode-segmentation` omissions; all are now declared in the harness | Harness issue resolved |
| Runtime integration | Not tested. `wgpui-core::app` and `wgpui-core::window` are still documented placeholder assemblies, and `wgpui-wgpu` has no frontend application loop | Open; deliberately not hidden behind no-op stubs |

## Implemented slice

The façade now exports named, real 2.0 implementations for retained geometry,
patch primitives, descriptions and element identity, reconciliation instances,
text primitives, `Div`, the styling trait, and the existing core `Window` type.
It also provides stable `core`, `layout`, `text`, `widgets`, and `prelude`
module paths. `facade_exports.rs` exercises the exports through a real
description and instance-key construction.

No `App`, `Application`, `Context`, `Render`, `RenderOnce`, legacy geometry
aliases, event system, action macro, or native window loop was invented. The
current 2.0 source does not provide those behaviors, so adding aliases or
no-op implementations would make the compatibility result misleading.

## Compatibility foundation continuation

The continuation adds `Application`/`App` window lifecycle ownership,
`Context` entity creation and notifications, a real single-completion `Task`,
and `Render`/`RenderOnce`/`IntoElement` adapters. Opened roots are rendered
into retained 2.0 `Description` values; the existing `wgpui-wgpu` frame loop
remains the real GPU integration point and the legacy root backend remains
available. Legacy geometry/color constructors (`px`, `size`, `point`,
`Bounds`, `Rgba`, `Hsla`, `rgb`, `rgba`, `hsla`) are covered by focused tests.

The compile-only matrix was rerun across all 41 examples. Lifecycle and
constructor names now resolve; remaining failures are concentrated in action
and derive macros, interaction/focus, specialized widgets, and native window
surface ownership. No additional example is counted as passing until the
per-example command completes successfully after those surfaces are implemented.

Phase 7 devtools work was treated as an adjacent consumer boundary; no
devtools files were changed by this compatibility slice.

## Verification

The 2.0 crates and compatibility library compile. The two compatibility tests
are compile/run checks. The complete example command is expected to fail with
the 20 listed targets until the next façade slice lands. Targeted cold clippy
is run separately for the implemented façade/tests with `--deny warnings`; a
full workspace test is intentionally not used.
