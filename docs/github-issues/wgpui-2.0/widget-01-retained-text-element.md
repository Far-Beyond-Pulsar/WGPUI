---
id: WIDGET-01
title: "[WGPUI 2.0] WIDGET-01: Retained text element"
state: closed
labels: wgpui-2.0,widget
---
## Scope

crates/wgpui-widgets/src/text.rs; crates/wgpui/src/lib.rs

## Required outcome

Expose standalone text lowering through the retained StyledText and shaping path.

## Non-negotiable architecture

- Retained reconciliation and minimal patch and upload behavior remain authoritative.
- Keep wgpui-core backend-neutral and use the public WGPU Window boundary.
- Do not use Window::refresh() for ordinary updates, add silent no-op shims, or regress source-ID and engine APIs.

## Acceptance criteria

- Add focused behavior or differential tests for the affected contract.
- Run the relevant crate tests and cargo check -p wgpui-examples-2 --examples.
- Record exact failures or platform limitations rather than hiding them.

Status: completed and verified in the current WGPUI 2.0 branch. Implementation commit: d4cf68fe2a.

