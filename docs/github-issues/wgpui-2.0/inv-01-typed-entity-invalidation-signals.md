---
id: INV-01
title: "[WGPUI 2.0] INV-01: Typed entity invalidation signals"
state: closed
labels: wgpui-2.0,inv
---
## Scope

crates/wgpui-core/src/invalidation/request.rs

## Required outcome

Represent entity-scoped invalidation and deterministically coalesce and consume entity signals.

## Non-negotiable architecture

- Retained reconciliation and minimal patch and upload behavior remain authoritative.
- Keep wgpui-core backend-neutral and use the public WGPU Window boundary.
- Do not use Window::refresh() for ordinary updates, add silent no-op shims, or regress source-ID and engine APIs.

## Acceptance criteria

- Add focused behavior or differential tests for the affected contract.
- Run the relevant crate tests and cargo check -p wgpui-examples-2 --examples.
- Record exact failures or platform limitations rather than hiding them.

Status: completed and verified in the current WGPUI 2.0 branch. Implementation commit: dbb8cdee78.

