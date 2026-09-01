---
id: TEST-05
title: "[WGPUI 2.0] TEST-05: Clippy and warning cleanup"
state: open
labels: wgpui-2.0,test
---
## Scope

crates/wgpui-core; crates/wgpui-wgpu; crates/wgpui-widgets

## Required outcome

Make the prescribed clippy wrapper pass without suppressing correctness warnings or changing legacy behavior.

## Non-negotiable architecture

- Retained reconciliation and minimal patch and upload behavior remain authoritative.
- Keep wgpui-core backend-neutral and use the public WGPU Window boundary.
- Do not use Window::refresh() for ordinary updates, add silent no-op shims, or regress source-ID and engine APIs.

## Acceptance criteria

- Add focused behavior or differential tests for the affected contract.
- Run the relevant crate tests and cargo check -p wgpui-examples-2 --examples.
- Record exact failures or platform limitations rather than hiding them.

Status: open; implementation and verification are still required.

