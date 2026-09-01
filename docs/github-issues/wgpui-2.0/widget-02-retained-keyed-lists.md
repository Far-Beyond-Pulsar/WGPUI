---
id: WIDGET-02
title: "[WGPUI 2.0] WIDGET-02: Retained keyed lists"
state: closed
labels: wgpui-2.0,widget
---
## Scope

crates/wgpui-widgets/src/list.rs; crates/wgpui-widgets/src/list/h_list.rs

## Required outcome

Support keyed vertical and horizontal lists with retained identity, scrolling, and viewport updates.

## Non-negotiable architecture

- Retained reconciliation and minimal patch and upload behavior remain authoritative.
- Keep wgpui-core backend-neutral and use the public WGPU Window boundary.
- Do not use Window::refresh() for ordinary updates, add silent no-op shims, or regress source-ID and engine APIs.

## Acceptance criteria

- Add focused behavior or differential tests for the affected contract.
- Run the relevant crate tests and cargo check -p wgpui-examples-2 --examples.
- Record exact failures or platform limitations rather than hiding them.

Status: completed and verified in the current WGPUI 2.0 branch. Implementation commit: d4cf68fe2a.

