---
id: RENDER-04
title: "[WGPUI 2.0] RENDER-04: SVG overflow clipping"
state: open
labels: wgpui-2.0,render
---
## Scope

crates/wgpui-widgets/src/svg.rs; crates/wgpui-core/src/reconcile

## Required outcome

Lower SVG overflow_hidden to a real local retained clip, including nested and transformed cases.

## Non-negotiable architecture

- Retained reconciliation and minimal patch and upload behavior remain authoritative.
- Keep wgpui-core backend-neutral and use the public WGPU Window boundary.
- Do not use Window::refresh() for ordinary updates, add silent no-op shims, or regress source-ID and engine APIs.

## Acceptance criteria

- Add focused behavior or differential tests for the affected contract.
- Run the relevant crate tests and cargo check -p wgpui-examples-2 --examples.
- Record exact failures or platform limitations rather than hiding them.

Status: open; implementation and verification are still required.

