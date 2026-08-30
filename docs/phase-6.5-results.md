# Phase 6.5 — animation driver

Status: **implemented on `2.0`**.

The 2.0 animation seam now has real duration sampling, chained definitions,
legacy-compatible easing names, repeat behavior, and a coalescing
`request_animation_frame` queue. `AnimationElement::describe_at` applies a
sample to an element and returns an ordinary `Description`; changing a sampled
style therefore follows the existing reconciliation and scene path without an
animation-specific renderer.

The existing image model also supports timing-based playback. `DecodedImage`
selects a looping frame from its decoded per-frame delays, `ImageEngine` exposes
that selection, and `Img::frame_at` turns it into the existing `frame_index`
description state. Still images and malformed all-zero timing remain safe.

## Gates

- Easing and chained timeline sampling: passed by focused unit tests.
- Repeating timeline requests a subsequent frame: passed.
- Sampled widget output changes an ordinary `Description` key: passed.
- Decoded-frame delay playback and loop boundary: passed.
- Legacy `src/` untouched: confirmed by `git diff`.

## Boundary and caveat

2.0 does not yet contain the frontend `Window`/`Render` assembly or a native
platform event loop. Consequently the core scheduler records and coalesces a
request; `wgpui-wgpu` must consume it when its future window assembly drives a
frame. The driver is deterministic when sampled with an explicit `Instant`,
which keeps reconciliation tests independent of wall-clock sleeps.
