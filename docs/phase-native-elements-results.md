# Native element and lifecycle results

Branch: `codex/wgpui-native-integration`

## Design decision

Native `Render` is:

```rust
fn render(&mut self) -> impl IntoElement + 'static
```

Native `RenderOnce` is:

```rust
fn render(self) -> impl IntoElement + 'static
```

Neither method receives `Window`, `App`, or `Context`. Element construction is
the value-to-description frontend step; retained services and scene work stay
in reconciliation and emission. Only the native probe uses these signatures.
The legacy example corpus was not mass-migrated.

## Implementation

- `Element`, `IntoElement`, `RenderOnce`, `Component`, and `Stateful` now
  compose through owned `Description` values, including derived generic
  components and nested children.
- `Div` child conversion uses the native `IntoElement` lowering path through
  its existing internal widget adapter. The adapter remains available to the
  untouched compatibility crate, but it is not re-exported from `wgpui`.
- Existing native image, SVG, styled-text, and animation elements now lower
  through their real `describe`/emission paths.
- `Styled::with_opacity` delegates to the real opacity style field, and
  `Styled` is available from the native prelude.
- Focused tests cover nested children, render-once ownership, derived generic
  elements, state identity/retention, style invalidation, and the existing
  retained emission path.

`wgpui-compat` and `wgpui-compat-macros` were not modified. No compatibility
shim, no-op method, or empty implementation was added.

## Analyzer

Command used before the changes:

```text
.\script\analyze-examples-2-errors.ps1 -Offline -Locked -OutputDirectory target/examples-2-analysis-baseline
```

Command used after the changes:

```text
.\script\analyze-examples-2-errors.ps1 -Offline -Locked -OutputDirectory target/examples-2-analysis-after
```

| Run | Total | Passed | Failed | Unique diagnostics |
| --- | ---: | ---: | ---: | ---: |
| Before | 45 | 2 | 43 | 204 |
| After | 45 | 2 | 43 | 178 |

The passing probes after the changes are `native_elements` and
`native_interaction`; both import and exercise native `wgpui`, not the facade.

Largest diagnostic reductions by occurrence:

| Diagnostic | Before | After |
| --- | ---: | ---: |
| `E0599` missing `Div::flex` | 140 | 0 |
| `E0599` missing `Div::text_xs` | 52 | 0 |
| `E0599` missing `Div::shadow` | 36 | 0 |
| `E0599` missing `Div::text_sm` | 32 | 0 |
| `E0599` missing `Div::text_color` | 31 | 0 |
| `E0599` missing `Div::flex_1` | 30 | 0 |
| `E0599` missing `Div::absolute` | 21 | 0 |
| `E0599` missing `Div::size_full` | 20 | 0 |

The remaining failures are outside this workstream, chiefly legacy
context-bearing render/application APIs, interaction and window APIs, lists,
assets, and GPU-surface dependencies. The native three-parameter render
diagnostic remains by design for those unported examples.

## Verification

Passed:

```text
cargo check --locked --offline -p wgpui-core -p wgpui-text -p wgpui-macros -p wgpui-widgets -p wgpui --tests
cargo clippy --locked --offline -p wgpui-core -p wgpui-text -p wgpui-macros -p wgpui-widgets -p wgpui --tests --all-features -- --deny warnings
cargo test --locked --offline -p wgpui --test elements       # 6 passed
cargo test --locked --offline -p wgpui-core --lib element   # 18 passed
cargo test --locked --offline -p wgpui-widgets --lib div::tests # 7 passed
```

The repository command `.\script\clippy.ps1` was also run cold with its
`--deny warnings` configuration. Its Cargo phase reached the examples and
failed on the pre-existing out-of-scope corpus errors; the PowerShell wrapper
reported exit code 0 because it does not propagate the native Cargo failure.

The touched-file rustfmt check was run. It reports pre-existing formatting
differences in `wgpui-widgets/src/animation.rs`, `div.rs`, `styled.rs`,
`styled_text.rs`, `svg.rs`, and `wgpui/src/wgpui.rs`; those files contain
unrelated existing formatting drift that was deliberately not rewritten. A
whole-repository `cargo fmt --all -- --check` additionally reports existing
differences in `wgpui-compat`, `old`, examples, and GPU-surface files, which
were also left untouched.

## Blockers left for later workstreams

Completing the remaining examples requires APIs explicitly outside this
workstream: context-aware retained interaction/window behavior, event and
focus handling, list/scroll components, asset loading, canvas/path APIs, and
GPU-surface/dependency wiring. Implementing those here would violate the
requested file scope, so they remain recorded rather than papered over.
