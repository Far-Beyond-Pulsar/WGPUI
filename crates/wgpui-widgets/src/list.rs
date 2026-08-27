//! List elements: `list()`, `uniform_list()`, `virtual_list()`, `h_list()`.
//! `uniform_list`/`h_list` are §6.1's CPU special case — the content shape
//! the GPU layout kernel generalizes. See docs/gpu-native-architecture.md
//! §3.4, §6.1.
//!
//! This file is both the module root and the home of the general
//! (non-uniform) `list()` element itself — §3.4's map draws those as
//! `list/mod.rs` and `list/list.rs`, and they collapse into one file here
//! because `AGENTS.md` forbids `mod.rs` paths, which would otherwise leave a
//! `list::list` submodule named after its own parent.
#![allow(dead_code)]

pub mod h_list;
pub mod uniform_list;
pub mod virtual_list;
