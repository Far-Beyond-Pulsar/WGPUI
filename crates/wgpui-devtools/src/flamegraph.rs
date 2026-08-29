//! Per-frame CPU/GPU flamegraphs and the RenderDoc-style capture-and-replay
//! engine — ~9,000 lines across four files in the legacy backend today.
//! See docs/gpu-native-architecture.md §3.6, §1.
#![allow(dead_code)]

pub mod cpu;
pub mod gpu;
pub mod replay;
pub mod ui_capture;
