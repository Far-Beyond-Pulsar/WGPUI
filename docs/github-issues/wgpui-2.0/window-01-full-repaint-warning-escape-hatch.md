---
id: WINDOW-01
title: "[WGPUI 2.0] WINDOW-01: Full repaint warning escape hatch"
state: closed
labels: wgpui-2.0,window
---
## Scope

crates/wgpui-wgpu/src/window/application.rs

## Required outcome

Keep Window::refresh() as an explicit full repaint API and emit a tracing warning when it is used.

## Non-negotiable architecture

- Retained reconciliation and minimal patch and upload behavior remain authoritative.
- Keep wgpui-core backend-neutral and use the public WGPU Window boundary.
- Do not use Window::refresh() for ordinary updates, add silent no-op shims, or regress source-ID and engine APIs.

## Acceptance criteria

- Add focused behavior or differential tests for the affected contract.
- Run the relevant crate tests and cargo check -p wgpui-examples-2 --examples.
- Record exact failures or platform limitations rather than hiding them.

Status: completed and verified in the current WGPUI 2.0 branch. Implementation commit: dcff46d41c.

