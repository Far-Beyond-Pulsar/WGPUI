---
id: TEXT-04
title: "[WGPUI 2.0] TEXT-04: Multiline truncation"
state: open
labels: wgpui-2.0,text
---
## Scope

crates/wgpui-widgets/src/text.rs; crates/wgpui-widgets/src/styled_text.rs

## Required outcome

Implement measured multiline truncation and ellipsis behavior through the retained text path.

## Non-negotiable architecture

- Retained reconciliation and minimal patch and upload behavior remain authoritative.
- Keep wgpui-core backend-neutral and use the public WGPU Window boundary.
- Do not use Window::refresh() for ordinary updates, add silent no-op shims, or regress source-ID and engine APIs.

## Acceptance criteria

- Add focused behavior or differential tests for the affected contract.
- Run the relevant crate tests and cargo check -p wgpui-examples-2 --examples.
- Record exact failures or platform limitations rather than hiding them.

Status: open; implementation and verification are still required.

