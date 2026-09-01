---
id: ASSET-08
title: "[WGPUI 2.0] ASSET-08: SVG lifetime and reuse"
state: open
labels: wgpui-2.0,asset
---
## Scope

crates/wgpui-widgets/src/svg.rs; crates/wgpui-widgets/src/image_cache.rs

## Required outcome

Preserve SVG source lifetime and cache reuse through retained reconciliation and window teardown.

## Non-negotiable architecture

- Retained reconciliation and minimal patch and upload behavior remain authoritative.
- Keep wgpui-core backend-neutral and use the public WGPU Window boundary.
- Do not use Window::refresh() for ordinary updates, add silent no-op shims, or regress source-ID and engine APIs.

## Acceptance criteria

- Add focused behavior or differential tests for the affected contract.
- Run the relevant crate tests and cargo check -p wgpui-examples-2 --examples.
- Record exact failures or platform limitations rather than hiding them.

Status: open; implementation and verification are still required.

