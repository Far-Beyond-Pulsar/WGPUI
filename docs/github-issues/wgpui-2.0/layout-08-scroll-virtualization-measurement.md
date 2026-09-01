---
id: LAYOUT-08
title: "[WGPUI 2.0] LAYOUT-08: Scroll virtualization measurement"
state: open
labels: wgpui-2.0,layout
---
## Scope

crates/wgpui-widgets/src/list; crates/wgpui-widgets/src/scroll

## Required outcome

Make virtualized measurement stable through viewport changes, unknown sizes, and keyed reordering.

## Non-negotiable architecture

- Retained reconciliation and minimal patch and upload behavior remain authoritative.
- Keep wgpui-core backend-neutral and use the public WGPU Window boundary.
- Do not use Window::refresh() for ordinary updates, add silent no-op shims, or regress source-ID and engine APIs.

## Acceptance criteria

- Add focused behavior or differential tests for the affected contract.
- Run the relevant crate tests and cargo check -p wgpui-examples-2 --examples.
- Record exact failures or platform limitations rather than hiding them.

Status: open; implementation and verification are still required.

