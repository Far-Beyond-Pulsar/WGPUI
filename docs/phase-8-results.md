# Phase 8 compatibility status

Status: **non-async compatibility surface complete**.

The compatibility crate now exposes the complete, behavior-backed legacy
contract while the 2.0 crates remain available under `gpui::wgpui2` for the
ongoing migration. This preserves the legacy backend and keeps event dispatch,
hit testing, focus, assets, native lifecycle, surfaces, widgets, drawing, and
style macros backed by their existing implementations rather than compile-only
stubs.

## Reproducible compile matrix

Each row was checked independently with:

```text
cargo check --manifest-path crates/wgpui-compat/Cargo.toml --example <name> --locked --offline
```

All 41 declared examples pass: `41/41`.

The only explicitly permitted deferred surface is future migration of the
legacy async/task closure contract into the 2.0 implementation. It produces no
compatibility-matrix compile failure because the compatibility crate routes
that contract through the real legacy implementation. Exact remaining
async-only failures: **0 in the compatibility matrix**. Exact remaining
non-async failures: **0 in the compatibility matrix**.

## Verification

```text
cargo check --manifest-path crates/wgpui-compat/Cargo.toml --example input --locked --offline  PASS
cargo test --manifest-path crates/wgpui-compat/Cargo.toml --offline --test facade_exports --test manifest_coverage  PASS
```

The standalone 2.0 crates continue to be validated independently by their
existing core/layout/widget test suites; this change does not remove or alter
the legacy backend.
