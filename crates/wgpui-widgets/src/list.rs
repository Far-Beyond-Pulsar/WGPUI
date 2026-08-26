//! List elements: `list()`, `uniform_list()`, `virtual_list()`, `h_list()`.
//! `uniform_list`/`h_list` are §6.1's CPU special case — the content shape
//! the GPU layout kernel generalizes. See docs/gpu-native-architecture.md
//! §3.4, §6.1.
#![allow(dead_code)]

pub mod h_list;
pub mod list;
pub mod uniform_list;
pub mod virtual_list;
