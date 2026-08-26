//! WGSL *source* + Rust-side layout descriptors for the compute passes this
//! crate describes — text and data only, no device, no queue (the module
//! doc at the top of `wgpui_core.rs` explains why). `wgpui-wgpu` (§3.5) is
//! what actually compiles these into pipelines and dispatches them.
//! See docs/gpu-native-architecture.md §3.1, §5.1-§5.3, §4.3, §6.1.
#![allow(dead_code)]

/// §5.1 — GPU compute ordering pass source.
pub const ORDERING_WGSL: &str = include_str!("shaders/ordering.wgsl");
/// §5.2 — GPU compute occlusion pass source.
pub const OCCLUSION_WGSL: &str = include_str!("shaders/occlusion.wgsl");
/// §6.1 — GPU compute layout kernel for regular (uniform) content.
pub const LAYOUT_UNIFORM_WGSL: &str = include_str!("shaders/layout_uniform.wgsl");
/// §4.3 — tile-visibility compute pass for `Buffering::Tiled`.
pub const TILE_VISIBILITY_WGSL: &str = include_str!("shaders/tile_visibility.wgsl");
/// §5.3 — indirect draw-arg generation.
pub const INDIRECT_ARGS_WGSL: &str = include_str!("shaders/indirect_args.wgsl");
