//! WGPUI as one shared library.
//!
//! This crate contains `gpui` (WGPUI), `wgpu` and every crate they link. A binary or
//! plugin library that links it (`use wgpui_dylib as _;`) gets those crates from this
//! library and does not link its own copy: rustc takes a crate a linked dylib contains
//! from the dylib. So the host and each plugin library share one copy of the framework's
//! `static`s, `thread_local!`s and `TypeId`s, which is what makes an `Entity<T>`, a global, an
//! image id or a surface id mean the same thing on both sides.
//!
//! Needs `-C prefer-dynamic` (this repository's `.cargo/config.toml`), which makes `std` a shared
//! library too: `std-<hash>.dll` ships beside the executable. Every crate that links this library
//! must be built with the same compiler, flags and feature set (see the dependency list in
//! `Cargo.toml`).

// Linked, not just depended on: a crate this library shares with the library below it must come
// from that library, or it would be in both (rustc: "only shows up once").
use profiling_dylib as _;

pub use gpui;
pub use wgpu;

// zlib's C API, exported for the crates outside this library that call it (build.rs).
include!(concat!(env!("OUT_DIR"), "/native_exports.rs"));
