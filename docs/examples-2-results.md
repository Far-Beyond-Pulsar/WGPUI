# WGPUI 2.0 examples migration

This crate is a copied migration corpus, not a claim that the native API is
complete. The files under `crates/wgpui-examples-2/examples` originated from
`old/examples` and are never edited in place. Imports are changed to `wgpui`
so compiler diagnostics describe native API gaps directly.

## Mapping

The four root examples, seventeen learn examples, five benchmark examples, and
seventeen legacy examples are registered individually in the crate manifest
(43 runnable examples total). The copied `prelude.rs` helper is retained as
source support but is not a standalone runnable example.
Assets are copied alongside the legacy examples, preserving paths based on
`CARGO_MANIFEST_DIR`. The compile script invokes Cargo once per example and
prints `PASS` or `FAIL` with the source path and a coarse diagnostic category.

## Meaning of results

`PASS` means only that the example compiled against the canonical `wgpui`
crate. It does not prove window creation, event handling, GPU presentation, or
visual fidelity. Runtime validation is a separate, explicit step.

## Current migration status

The first matrix run is intentionally the next acceptance step. Its output and
individual `.compile.log` files are generated locally and are not source
artifacts. Failures must be grouped by missing native capability rather than
papered over with no-op compatibility methods.

Initial runtime candidates should be small, deterministic examples such as
`hello_world`, `tree`, `text`, `emoji_display`, `shadow`, and `layout`, after
they compile natively. Interactive, image, surface, and custom-drawing
examples require their corresponding native APIs and should not be reported as
runtime-tested until those paths are real.

## Phase 4C supported subset

The following examples now use the native `Render::render(&mut self)` contract
and compile against the retained application path: `learn/text`, `learn/layout`,
`text_gradients`, `bench/shadow`, and `bench/pattern`. Their `Bounds::centered`
calls use `None::<()>` because the native geometry helper deliberately accepts
an unused, generic display marker; this is an inference requirement rather than
a runtime behavior change.

The native `StyledText` element remains a lower-level renderer-owned element
because it requires a shared shaping/atlas engine. The text example therefore
uses equivalent native raw-text children for its inline bold/italic/semibold
demonstration. Karaoke examples still depend on the context-bearing legacy
render signature, timer integration, and context-aware entity updates; they are
not counted as compiled by this phase.
