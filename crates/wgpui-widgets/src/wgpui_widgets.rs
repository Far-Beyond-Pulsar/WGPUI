//! `wgpui-widgets` — elements, split along `div.rs`'s own seams.
//! See docs/gpu-native-architecture.md §3.4.
#![allow(dead_code)]

pub mod animation;
pub mod assets;
pub mod canvas;
pub mod div;
pub mod image_cache;
pub mod img;
pub mod list;
pub mod overlay;
pub mod scroll;
pub mod styled;
pub mod styled_text;
pub mod surface;
pub mod svg;
pub mod text;
pub mod wgpu_surface;

pub use div::interactivity::style::{
    BoxShadow, Corners, DivStyle, Edges, LinearGradient, Pattern, RadialGradient,
};
