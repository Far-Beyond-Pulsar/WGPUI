---
id: OVERLAY-01
title: "[WGPUI 2.0] OVERLAY-01: Retained deferred overlays"
state: closed
labels: wgpui-2.0,overlay
---
## Scope

crates/wgpui-widgets/src/overlay/deferred.rs; crates/wgpui-widgets/src/overlay/anchored.rs

## Required outcome

Implement deferred overlay ownership, anchor tracking, z-order, dismissal, and invalidation.

## Non-negotiable architecture

- Retained reconciliation and minimal patch and upload behavior remain authoritative.
- Keep wgpui-core backend-neutral and use the public WGPU Window boundary.
- Do not use Window::refresh() for ordinary updates, add silent no-op shims, or regress source-ID and engine APIs.

## Acceptance criteria

- Add focused behavior or differential tests for the affected contract.
- Run the relevant crate tests and cargo check -p wgpui-examples-2 --examples.
- Record exact failures or platform limitations rather than hiding them.

Status: completed and verified in the current WGPUI 2.0 branch. Implementation commit: dcff46d41c.

