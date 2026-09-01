---
id: RENDER-03
title: "[WGPUI 2.0] RENDER-03: Linear sprite filtering"
state: open
labels: wgpui-2.0,render
---
## Scope

crates/wgpui-widgets/src/image_cache.rs; crates/wgpui-wgpu/src/render/shaders/poly_sprites.wgsl

## Required outcome

Replace nearest-neighbor scaled image sampling with documented linear interpolation and alpha-correct tests.

## Non-negotiable architecture

- Retained reconciliation and minimal patch and upload behavior remain authoritative.
- Keep wgpui-core backend-neutral and use the public WGPU Window boundary.
- Do not use Window::refresh() for ordinary updates, add silent no-op shims, or regress source-ID and engine APIs.

## Acceptance criteria

- Add focused behavior or differential tests for the affected contract.
- Run the relevant crate tests and cargo check -p wgpui-examples-2 --examples.
- Record exact failures or platform limitations rather than hiding them.

Status: open; implementation and verification are still required.

