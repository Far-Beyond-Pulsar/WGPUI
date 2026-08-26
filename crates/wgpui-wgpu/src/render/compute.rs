//! The GPU compute passes: ordering, occlusion, regular-content layout,
//! tile visibility, indirect draw-arg generation. Each dispatches the
//! corresponding WGSL source from `wgpui_core::shaders`.
//! See docs/gpu-native-architecture.md §5.1-§5.3, §4.3, §6.1.
#![allow(dead_code)]

pub mod indirect_args_pass;
pub mod layout_pass;
pub mod occlusion_pass;
pub mod ordering_pass;
pub mod tile_visibility_pass;
