# Phase 8 compatibility probe

Status: continuation in progress from `0fdee0f617`; this is not a Phase 8
completion claim. The next continuation boundary is the interaction/text and
native surface adapters listed below.

## Continuation 2 live probe

The continuation-2 report's 21/41 list was stale. Rechecking the committed
foundation parent (`0fdee0f617`) one example at a time produced **0/41**. The
reported 21/41 was from an uncommitted intermediate worktree and was not a
reproducible branch base. Rechecking `3bb9460545` one example at a time
produced **3/41** (`text_gradients`, `emoji_display`, `tree`), not 2/41. The
aggregate `cargo check --examples` command is retained as a build check, but
its single failure status is not used as a per-example count.

This continuation adds real Taffy grid templates and grid placement, overflow
axes, retained text-style metadata, common legacy spacing/size and border-side
aliases, `Pixels` arithmetic/formatting/conversion helpers, and color-stop
adapters. Focused widget and façade adapter tests pass. The matrix still cannot
be called complete: frontend interaction/focus/listener contracts, image/SVG
and WGPU-surface ownership, path/canvas/gradient rendering, and native window
options/lifecycle remain compiler blockers. No no-op compatibility methods were
added for those missing behaviors.

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

Pass: `text_gradients`, `emoji_display`, `tree`, `shadow`.

Fail: `karaoke_text`, `karaoke_app`, `interactive_elements`, `creating_components`,
`layout`, `styling`, `async_tasks`, `custom_drawing`, `animation`, `text`,
`wgpu_surface`, `wgpu_surface_basic`, `wgpu_surface_quad`, `wgpu_surface_stress`,
`mouse_events`, `blur_showcase`, `smooth_scrolling`, `virtual_list`, `data_table`,
`plain_scroll_10k`, `paths_bench`, `pattern`, `focus_visible`, `gif_viewer`,
`gradient`, `hello_world`, `image_loading`, `input`, `on_window_close_quit`,
`opacity`, `scrollable`, `svg`, `tab_stop`, `uniform_list`, `window`,
`window_positioning`, `window_shadow`.

## Continuation 3 geometry checkpoint

The branch starts at `3bb9460545` with a clean worktree. This checkpoint adds
real `Pixels` arithmetic/conversion, logical-pixel border and radius builders,
resolved background opacity, and conversion from the legacy `BoxShadow` shape
to the 2.0 RGBA shadow primitive. Focused façade tests pass; the individual
probe recovers `shadow`, reaching **4/41**.

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

The compile-only matrix was rerun after the style and action adapter groups:
21/41 pass and 20/41 fail (unchanged count, although the failure diagnostics
are now further downstream). No example is counted as passing unless its
per-example command succeeds.

The continuation added real `Pixels`-accepting style sizing, legacy spacing and
size aliases, alpha-preserving color adapters, RGBA-to-HSLA conversion,
`Colors`, `Bounds::new`, `px` const construction, and an `actions!`/
`Action`/`KeyBinding`/menu registration surface. Focused tests cover the color
and action adapters. The legacy backend remains available.

Current exact blockers from the authoritative matrix are: text style and text
element state (`text_xs`, `text_sm`, `text_lg`, `text_color`, and related text
rendering); grid/overflow/blur/gradient behavior; event, focus, hit-testing,
listener, `Stateful`, and scroll APIs; uniform/virtual list and canvas
adapters; image/SVG/WGPU surface ownership; and remaining window lifecycle
types and methods. The derive `IntoElement` macro is still unavailable, and
the `relative` adapter currently only covers the scalar sizing cases.

Phase 7 devtools work was treated as an adjacent consumer boundary; no
devtools files were changed by this compatibility slice.

## Verification

The 2.0 crates and compatibility library compile. The two compatibility tests
are compile/run checks. The complete example command is expected to fail with
the 20 listed targets until the next façade slice lands. Targeted cold clippy
is run separately for the implemented façade/tests with `--deny warnings`; a
full workspace test is intentionally not used.
