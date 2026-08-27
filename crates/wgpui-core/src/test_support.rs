//! Headless patch/reconcile/window testing, mirrors today's
//! `src/platform/test`. See docs/gpu-native-architecture.md §3.1.
//!
//! Phase 1's results doc left this file a stub on purpose — "everything built
//! here turned out to be testable headlessly without a support layer... if a
//! later phase's tests start repeating scaffolding, that is the moment to write
//! it, not now." Phase 3 is that moment, and specifically because the
//! scaffolding cannot live in a `#[cfg(test)]` module: §5.2's differential
//! harness has to drive *the same scene* through the CPU reference (here) and
//! through the compute passes (in `wgpui-wgpu`), and a test-only module is not
//! reachable from another crate.
//!
//! Two things live here, and both exist to make a claim falsifiable rather than
//! to make a test shorter:
//!
//! - [`ui_walk`] builds a scripted UI walk — a realistic editor-shaped scene
//!   driven through frames of scrolling, selection, and a modal opening — as
//!   real primitives applied to a real [`crate::scene::Scene`] through the real
//!   patch protocol. R-N §8.5 asks for exactly this ("run it in CI over a
//!   scripted UI walk"), and §8's Phase 3 gate asks that the performance case
//!   be a scene "built through `wgpui-core`'s actual `Scene`/`Layer`/
//!   `PrimitiveStore` APIs (not Spike A's synthetic quad buffer)."
//! - [`raster`] is a reference rasterizer. Without one, "culled and unculled
//!   scenes match exactly" can only be checked as "the two primitive lists
//!   agree where they overlap," which is a restatement of the culling rule
//!   rather than a test of it. With one, the gate is what it says: the pixels
//!   are the same either way.

#![allow(dead_code)]

pub mod raster;
pub mod ui_walk;
