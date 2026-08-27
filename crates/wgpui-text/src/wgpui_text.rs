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

pub mod fonts;
pub mod line;
pub mod line_layout;
pub mod line_wrapper;
pub mod patch;
pub mod shaping;
