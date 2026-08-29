# Phase 8 compatibility status

Status: **not complete**. The requested gate is not met on
`wgpui-2.0/phase-8-compatibility-continuation-3` (`e08c92d6bf`). No
`phase-8-complete` branch was created or pushed.

## Reproducible compile matrix

Each row was checked independently with:

```text
cargo check --manifest-path crates/wgpui-compat/Cargo.toml --example <name> --locked --offline
```

| Example | Result |
| --- | --- |
| karaoke_text | FAIL: `Timer` |
| karaoke_app | FAIL: gradient/animation façade |
| text_gradients | PASS |
| interactive_elements | FAIL: mouse/hover/focus façade |
| creating_components | FAIL: component/element façade |
| layout | FAIL: `row_span` |
| styling | FAIL: hover/group cascade |
| async_tasks | FAIL: task/context signatures |
| custom_drawing | FAIL: path/canvas/mouse façade |
| animation | FAIL: SVG/animation façade |
| text | FAIL: `StyledText`/text overflow |
| emoji_display | PASS |
| wgpu_surface | FAIL: surface ownership |
| wgpu_surface_basic | FAIL: surface ownership |
| wgpu_surface_quad | FAIL: surface ownership |
| wgpu_surface_stress | FAIL: surface ownership |
| mouse_events | FAIL: event/listener façade |
| blur_showcase | FAIL: canvas/scroll/mouse façade |
| smooth_scrolling | FAIL: uniform-list/scroll façade |
| virtual_list | FAIL: virtual-list/scroll façade |
| data_table | FAIL: canvas/uniform-list/derive façade |
| plain_scroll_10k | FAIL: scroll façade |
| paths_bench | FAIL: path/canvas/gradient façade |
| pattern | FAIL: gradient façade |
| shadow | PASS |
| focus_visible | FAIL: hover/stateful façade |
| gif_viewer | FAIL: image façade |
| gradient | FAIL: gradient/path/canvas façade |
| hello_world | FAIL: dashed border and pixel sizing |
| image_loading | FAIL: asset/image façade |
| input | FAIL: input/paint/element façade |
| on_window_close_quit | FAIL: lifecycle façade |
| opacity | FAIL: asset/image/SVG façade |
| scrollable | FAIL: hover/scroll façade |
| svg | FAIL: asset/SVG façade |
| tab_stop | FAIL: hover/click façade |
| tree | PASS |
| uniform_list | FAIL: uniform-list/processor façade |
| window | FAIL: native window options/lifecycle |
| window_positioning | FAIL: native window options |
| window_shadow | FAIL: native window/paint façade |

**Compile result: 4/41 pass, 37/41 fail.** The four passing examples are
`text_gradients`, `emoji_display`, `shadow`, and `tree`. Manifest coverage
confirms all 41 root examples are represented.

## Implemented and verified

The branch contains real 2.0 adapters for geometry and colors,
`Description`/reconciliation identity, `Div` layout and paint, retained
shadows, text-gradient metadata, basic `App`/`Context`/`Entity`/`Task`/
`Render` façade plumbing, focus-handle identity, actions, key bindings, and
the existing 2.0 image/SVG/cache, text, list, canvas, surface, and WGPU
building blocks where their direct 2.0 contracts exist.

Focused checks passed:

```text
cargo test --manifest-path crates/wgpui-compat/Cargo.toml --offline --test facade_exports --test manifest_coverage
  7 façade tests + 1 coverage test passed
cargo test -p wgpui-core --offline --lib      335 tests passed
cargo test -p wgpui-layout --offline --lib      6 tests passed
cargo test -p wgpui-widgets --offline --lib   84 tests passed
```

## Runtime result and blocking evidence

No compatibility example was run as a successful runtime acceptance test,
because 37 do not compile. The existing 2.0 WGPU surface has a real present
path, but its frontend event loop, input dispatch, and application-to-window
ownership are not wired to it. The widget sources similarly identify missing
hit-test/event cascade and scroll-state contracts. A compile-failing example
cannot establish runtime parity.

The remaining work is architectural: a real frontend event/cascade and input
system, native application loop with WGPU surface ownership, asset resolution,
path and gradient primitives, text element emission, and list/virtualization
contracts. Immediate-return methods or discarded callbacks would make the
matrix green while violating the requirement that adapters route through
Description → reconcile → Scene → GPU behavior.

The legacy backend remains intact and available. The completion gate remains
open; this report intentionally does not claim Phase 8 completion.
