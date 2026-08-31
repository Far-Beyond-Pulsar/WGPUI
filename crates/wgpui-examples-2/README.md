# WGPUI 2.0 examples

This crate is a native-backend migration probe. Its examples are copied from
`old/examples` and use `wgpui` directly. The source tree is intentionally
kept separate so the legacy examples remain an immutable comparison corpus.

Use `pwsh ./script/compile-examples-2.ps1` to compile every example
independently and classify compiler failures. Compilation does not imply that
an example creates a window or presents a frame; runtime candidates are listed
in `docs/examples-2-results.md`.

Bounded launch candidates are declared in `examples/smoke-tests.toml`. The
metadata is consumed by the migration workflow after compilation succeeds; it
does not turn a compile failure into a runtime pass.
