---
id: ENTITY-05
title: "[WGPUI 2.0] ENTITY-05: Entity observer lifetime and drop"
state: open
labels: wgpui-2.0,entity
---
## Scope

crates/wgpui-core/src/app.rs; crates/wgpui-core/src/app/entity.rs

## Required outcome

Test observer deregistration, weak handles, duplicate observers, and teardown ordering.

## Non-negotiable architecture

- Retained reconciliation and minimal patch and upload behavior remain authoritative.
- Keep wgpui-core backend-neutral and use the public WGPU Window boundary.
- Do not use Window::refresh() for ordinary updates, add silent no-op shims, or regress source-ID and engine APIs.

## Acceptance criteria

- Add focused behavior or differential tests for the affected contract.
- Run the relevant crate tests and cargo check -p wgpui-examples-2 --examples.
- Record exact failures or platform limitations rather than hiding them.

Status: open; implementation and verification are still required.

