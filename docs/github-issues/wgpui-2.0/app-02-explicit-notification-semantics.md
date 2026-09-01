---
id: APP-02
title: "[WGPUI 2.0] APP-02: Explicit notification semantics"
state: closed
labels: wgpui-2.0,app
---
## Scope

crates/wgpui-core/src/app/context.rs; crates/wgpui-core/src/app/entity.rs

## Required outcome

Require cx.notify() for render invalidation while preserving observer/drop behavior and no-notify updates.

## Non-negotiable architecture

- Retained reconciliation and minimal patch and upload behavior remain authoritative.
- Keep wgpui-core backend-neutral and use the public WGPU Window boundary.
- Do not use Window::refresh() for ordinary updates, add silent no-op shims, or regress source-ID and engine APIs.

## Acceptance criteria

- Add focused behavior or differential tests for the affected contract.
- Run the relevant crate tests and cargo check -p wgpui-examples-2 --examples.
- Record exact failures or platform limitations rather than hiding them.

Status: completed and verified in the current WGPUI 2.0 branch. Implementation commit: 8677fc37b2.

