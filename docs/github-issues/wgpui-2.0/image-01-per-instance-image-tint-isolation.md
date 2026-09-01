---
id: IMAGE-01
title: "[WGPUI 2.0] IMAGE-01: Per-instance image tint isolation"
state: closed
labels: wgpui-2.0,image
---
## Scope

crates/wgpui-widgets/src/image_cache.rs; crates/wgpui-widgets/src/img.rs; crates/wgpui-widgets/src/svg.rs

## Required outcome

Keep tint metadata instance-local while retaining shared decoded image and cache data.

## Non-negotiable architecture

- Retained reconciliation and minimal patch and upload behavior remain authoritative.
- Keep wgpui-core backend-neutral and use the public WGPU Window boundary.
- Do not use Window::refresh() for ordinary updates, add silent no-op shims, or regress source-ID and engine APIs.

## Acceptance criteria

- Add focused behavior or differential tests for the affected contract.
- Run the relevant crate tests and cargo check -p wgpui-examples-2 --examples.
- Record exact failures or platform limitations rather than hiding them.

Status: completed and verified in the current WGPUI 2.0 branch. Implementation commit: e803fce0fc.

