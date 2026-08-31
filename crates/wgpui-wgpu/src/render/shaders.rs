//! The hand-written render shaders, moved as-is from
//! `src/platform/cross/shaders/*.wgsl`. See docs/gpu-native-architecture.md
//! §3.5, §1.
#![allow(dead_code)]

pub const QUADS_WGSL: &str = include_str!("shaders/quads.wgsl");
pub const SHADOWS_WGSL: &str = include_str!("shaders/shadows.wgsl");
pub const MONO_SPRITES_WGSL: &str = include_str!("shaders/mono_sprites.wgsl");
pub const POLY_SPRITES_WGSL: &str = include_str!("shaders/poly_sprites.wgsl");
pub const PATHS_WGSL: &str = include_str!("shaders/paths.wgsl");
pub const UNDERLINES_WGSL: &str = include_str!("shaders/underlines.wgsl");
pub const BACKDROP_BLUR_WGSL: &str = include_str!("shaders/backdrop_blur.wgsl");
pub const SURFACES_WGSL: &str = include_str!("shaders/surfaces.wgsl");
pub const DEBUG_TILES_WGSL: &str = include_str!("shaders/debug_tiles.wgsl");
pub const DAMAGE_CLEAR_WGSL: &str = include_str!("shaders/damage_clear.wgsl");
