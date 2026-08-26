//! Font feature/fallback handling. Not shown as its own `mod.rs` in
//! §3.3's tree, but the `fonts/` directory needs a declaring file under
//! this repo's no-`mod.rs` convention (see AGENTS.md). See
//! docs/gpu-native-architecture.md §3.3.
#![allow(dead_code)]

pub mod fallbacks;
pub mod features;
