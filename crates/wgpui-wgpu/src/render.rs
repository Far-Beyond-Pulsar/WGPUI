//! Device/queue creation, pipelines, compute dispatch, atlas, textures,
//! draw issuance. See docs/gpu-native-architecture.md §3.5.
pub mod atlas;
pub mod atlas_upload;
pub mod buffers;
pub mod compute;
pub mod device;
pub mod draw;
pub mod frame;
pub mod pipelines;
pub mod readback;
pub mod resources;
pub mod shaders;
pub mod surface_registry;
pub mod textures;
