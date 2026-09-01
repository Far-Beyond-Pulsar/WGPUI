---
id: LAYOUT-01
title: "[WGPUI 2.0] LAYOUT-01: Generic estimated-size contract"
state: closed
labels: wgpui-2.0,layout
---
## Scope

crates/wgpui-layout/src/containment.rs; crates/wgpui-layout/src/measure.rs

## Required outcome

Provide validated estimate-aware layout with exact fallback for absent or invalid estimates.

## Non-negotiable architecture

- Retained reconciliation and minimal patch and upload behavior remain authoritative.
- Keep wgpui-core backend-neutral and use the public WGPU Window boundary.
- Do not use Window::refresh() for ordinary updates, add silent no-op shims, or regress source-ID and engine APIs.

## Acceptance criteria

- Add focused behavior or differential tests for the affected contract.
- Run the relevant crate tests and cargo check -p wgpui-examples-2 --examples.
- Record exact failures or platform limitations rather than hiding them.

Status: completed and verified in the current WGPUI 2.0 branch. Implementation commit: 33e4e1302a.

