//! `wgpui-text` — cosmic-text shaping, isolated.
//! See docs/gpu-native-architecture.md §3.3, §6.
//!
//! §6 is the design contract, and it cuts in one place: shaping is CPU work
//! and stays CPU work; placing already-shaped glyphs as instanced sprites is
//! GPU work and goes through the same patch protocol as every other primitive.
//! This crate lives entirely on the CPU side of that cut. It produces glyph
//! positions and atlas tile *requests*; `wgpui-wgpu`'s allocator turns requests
//! into tile coordinates. Neither owns the other's job, and neither names the
//! other's types.

//! Phase 5.5 adds the one step that was missing between the two: [`raster`]
//! turns a glyph outline into the pixels an allocated atlas tile holds. It is
//! still CPU work and still opens no device — `wgpui-wgpu` copies the bytes into
//! a texture, and that copy is the only part of the path that needs one.

pub mod fonts;
pub mod engine;
pub mod line;
pub mod line_layout;
pub mod line_wrapper;
pub mod patch;
pub mod raster;
pub mod shaping;
/// One embedded face for tests that assert about real glyph pixels.
///
/// Public rather than `#[cfg(test)]` because the differential gate in `tests/`
/// has to shape and rasterise against the *same* face as the unit tests, and an
/// integration test cannot see a `#[cfg(test)]` module. It costs the ~200 KB of
/// the embedded TTF in any binary linking this crate — worth gating behind a
/// feature at Phase 8's cutover, and not worth a feature flag before then, when
/// nothing ships this crate at all.
pub mod test_fonts;

pub use engine::{SharedTextEngine, SharedTextShaper, TextEngine};
