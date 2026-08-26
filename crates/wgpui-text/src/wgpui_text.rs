//! `wgpui-text` — cosmic-text shaping, isolated. Text shaping stays on the
//! CPU on purpose (§6); this crate is mostly a move, not a rewrite, of
//! today's `src/text_system/`. See docs/gpu-native-architecture.md §3.3, §6.
#![allow(dead_code)]

pub mod fonts;
pub mod line;
pub mod line_layout;
pub mod line_wrapper;
pub mod patch;
pub mod shaping;
