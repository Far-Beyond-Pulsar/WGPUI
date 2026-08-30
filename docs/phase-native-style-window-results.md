# Workstream B: native style and window results

## Scope

This workstream changed only native implementation files:

- `crates/wgpui-widgets/src/styled.rs`
- `crates/wgpui-core/src/app.rs`
- `crates/wgpui-core/src/window.rs`
- `crates/wgpui-wgpu/src/window/application.rs`

No dependency was added. The platform application adapter is in `wgpui-wgpu`; no direct surface implementation was changed.

The style DSL now has the commonly used numeric conversions, baseline alignment, row-span, and right-inset aliases. The conversions lower into the existing layout fields. Core application and window state now tracks activation, quit requests, menus, bounds, and active state. The native platform application applies `WindowOptions` bounds, fallback dimensions, visibility, focus, maximized/fullscreen state, and windowed position through winit. Native resize returns winit's applied-size result instead of discarding it.

## Analyzer counts

The requested first inspection was:

```text
target/examples-2-analysis/report.md
```

That report records the pre-Workstream-A baseline: 45 total examples, 2 passing, 43 failing, 204 normalized unique errors.

The Workstream-A report used as the Workstream-B starting point was:

```text
target/examples-2-analysis-after/report.md
```

It records 45 total, 2 passing, 43 failing, 178 normalized unique errors.

Exact Workstream-B command:

```powershell
.\script\analyze-examples-2-errors.ps1 -Offline -Locked -OutputDirectory target/examples-2-analysis-workstream-b -KeepRawOutput
```

Result in `target/examples-2-analysis-workstream-b/report.json` and `report.md`:

| Count | Before Workstream B | After Workstream B |
| --- | ---: | ---: |
| Total examples | 45 | 45 |
| Passing examples | 2 | 2 |
| Failing examples | 43 | 43 |
| Normalized unique errors | 178 | 166 |

The analyzer exit code is 1 because 43 examples still fail. The count requirement of five newly compiling examples was not met; the architectural blocker is documented below with direct compiler evidence.

## Focused behavior tests

These focused commands passed:

```powershell
cargo test --locked --offline -p wgpui-widgets --lib styled::tests
# 11 passed

cargo test --locked --offline -p wgpui-core --lib window
# 14 passed

cargo test --locked --offline -p wgpui-wgpu --lib application::tests
# 2 passed
```

The new tests cover layout and numeric style mutation, core window bounds/resize/activation/close state, application activation/quit/menu state, and explicit/fallback native window bounds. Existing retained-description, invalidation, and GPU-emission tests also remain passing.

Full native library test command:

```powershell
cargo test --locked --offline -p wgpui-core -p wgpui-widgets -p wgpui-wgpu -p wgpui --lib
# all test binaries passed: 363 core tests, 92 widgets tests, 74 wgpu tests, and 0 wgpui tests
```

## Locked offline and clippy gates

Affected native crates and their tests:

```powershell
cargo check --locked --offline -p wgpui-core -p wgpui-widgets -p wgpui-wgpu -p wgpui --tests
# passed
```

Cold strict clippy was run after cleaning the affected packages:

```powershell
cargo clean -p wgpui-core -p wgpui-widgets -p wgpui-wgpu -p wgpui
.\script\clippy.ps1 -p wgpui-core -p wgpui-widgets -p wgpui-wgpu -p wgpui --locked --offline
# passed; release, all-targets, all-features, --deny warnings
```

Relevant migrated probes were checked with the same locked offline flags:

```powershell
cargo check --locked --offline --message-format=short --manifest-path crates/wgpui-examples-2/Cargo.toml --example native_elements
# passed
cargo check --locked --offline --message-format=short --manifest-path crates/wgpui-examples-2/Cargo.toml --example native_interaction
# passed
```

## Remaining architectural blocker

The legacy migrated examples require an application/context API that is intentionally not the native API documented by the breaking-change policy. Direct evidence from:

```powershell
cargo check --locked --offline --message-format=short --manifest-path crates/wgpui-examples-2/Cargo.toml --example hello_world
```

is:

```text
error[E0050]: method `render` has 3 parameters but the declaration in trait `wgpui::Render::render` has 1
error[E0061]: this function takes 2 arguments but 0 arguments were supplied
error[E0599]: no method named `open_window` found for mutable reference `&mut App` in the current scope
error[E0061]: this method takes 0 arguments but 1 argument was supplied
```

The native `Application::new(WindowOptions, builder)` and `Render::render(&mut self)` contracts cannot be overloaded in Rust. Changing either to accept the legacy context-bearing signatures would break the already compiling native examples and weaken the deliberate native type boundary. Adding an `open_window` facade without integrating it into the existing event loop and retained renderer would be a no-op/shim, so it was not added. The remaining context/window lifecycle, low-level GPU, canvas, list, and asset diagnostics are outside this focused native style/window implementation or require the subsequent architectural migration workstream.

## Scope verification

The final working-tree diff was explicitly checked with:

```powershell
git diff --quiet -- old gpui-ce crates/wgpui-compat
# exit code 0

$changed = @(git diff --name-only)
$changed | Where-Object { $_ -match '(^|/)(old|gpui-ce|wgpui-compat)(/|$)' }
$changed | Where-Object { $_ -match 'wgpui-examples-2|/list|/assets|/canvas' }
git diff -- Cargo.toml Cargo.lock crates/wgpui-core/Cargo.toml crates/wgpui-widgets/Cargo.toml crates/wgpui-wgpu/Cargo.toml crates/wgpui/Cargo.toml crates/wgpui-macros/Cargo.toml
```

Both forbidden-path queries produced no output, the explicit protected-tree diff returned exit code 0, and the dependency diff was empty. The workstream therefore stops with the native behavior/tests implemented and the documented compile blocker, rather than claiming the unmet five-example gate.
