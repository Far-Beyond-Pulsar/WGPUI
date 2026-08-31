//! Native-resource capture adapter.
//!
//! This type is intentionally cheap when the `devtools` feature or a GPU
//! capture is disabled. The renderer can call it at resource boundaries
//! without adding a second set of allocation and upload paths.

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct NativeResourceId(u64);

impl NativeResourceId {
    pub const INVALID: Self = Self(0);

    pub const fn raw(self) -> u64 {
        self.0
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NativeResourceRole {
    PrimitiveBuffer,
    IndirectArguments,
    IndirectCount,
    Visibility,
    SlotTable,
    AtlasPage,
    LayerTexture,
    TileTexture,
    Surface,
    Staging,
    Readback,
    Query,
    Uniform,
    Other,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct NativeResourceDimensions {
    pub width: u32,
    pub height: u32,
    pub depth_or_array_layers: u32,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct NativeResourceRegistry;

impl NativeResourceRegistry {
    pub fn begin_frame(&self, frame: u64) {
        #[cfg(feature = "devtools")]
        wgpui_devtools::gpu_resources::set_frame(frame);
        #[cfg(not(feature = "devtools"))]
        let _ = frame;
    }

    pub fn current_frame(&self) -> u64 {
        #[cfg(feature = "devtools")]
        {
            wgpui_devtools::gpu_resources::current_frame()
        }
        #[cfg(not(feature = "devtools"))]
        {
            0
        }
    }

    pub fn register_buffer(
        &self,
        label: &str,
        role: NativeResourceRole,
        size: u64,
        usage: u64,
        generation: u64,
    ) -> NativeResourceId {
        #[cfg(feature = "devtools")]
        {
            from_devtools(wgpui_devtools::gpu_resources::register(
                wgpui_devtools::gpu_resources::ResourceDescriptor {
                    kind: wgpui_devtools::gpu_resources::ResourceKind::Buffer,
                    role: role.into(),
                    label: label.to_string(),
                    format: None,
                    dimensions: None,
                    byte_size: size,
                    usage,
                    generation,
                },
            ))
        }
        #[cfg(not(feature = "devtools"))]
        {
            let _ = (label, role, size, usage, generation);
            NativeResourceId::INVALID
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn register_texture(
        &self,
        label: &str,
        role: NativeResourceRole,
        dimensions: NativeResourceDimensions,
        format: &str,
        byte_size: u64,
        usage: u64,
        generation: u64,
    ) -> NativeResourceId {
        #[cfg(feature = "devtools")]
        {
            from_devtools(wgpui_devtools::gpu_resources::register(
                wgpui_devtools::gpu_resources::ResourceDescriptor {
                    kind: wgpui_devtools::gpu_resources::ResourceKind::Texture,
                    role: role.into(),
                    label: label.to_string(),
                    format: Some(wgpui_devtools::gpu_resources::ResourceFormat::new(format)),
                    dimensions: Some(dimensions.into()),
                    byte_size,
                    usage,
                    generation,
                },
            ))
        }
        #[cfg(not(feature = "devtools"))]
        {
            let _ = (
                label, role, dimensions, format, byte_size, usage, generation,
            );
            NativeResourceId::INVALID
        }
    }

    pub fn register_query_set(
        &self,
        label: &str,
        query_type: &str,
        count: u32,
        generation: u64,
    ) -> NativeResourceId {
        #[cfg(feature = "devtools")]
        {
            from_devtools(wgpui_devtools::gpu_resources::register(
                wgpui_devtools::gpu_resources::ResourceDescriptor {
                    kind: wgpui_devtools::gpu_resources::ResourceKind::QuerySet,
                    role: wgpui_devtools::gpu_resources::ResourceRole::Query,
                    label: label.to_string(),
                    format: Some(wgpui_devtools::gpu_resources::ResourceFormat::new(
                        query_type,
                    )),
                    dimensions: Some(wgpui_devtools::gpu_resources::ResourceDimensions {
                        width: count,
                        height: 1,
                        depth_or_array_layers: 1,
                    }),
                    byte_size: 0,
                    usage: 0,
                    generation,
                },
            ))
        }
        #[cfg(not(feature = "devtools"))]
        {
            let _ = (label, query_type, count, generation);
            NativeResourceId::INVALID
        }
    }

    pub fn mark_used(&self, id: NativeResourceId, frame: u64) {
        #[cfg(feature = "devtools")]
        wgpui_devtools::gpu_resources::mark_used(to_devtools(id), frame);
        #[cfg(not(feature = "devtools"))]
        let _ = (id, frame);
    }

    pub fn record_buffer_upload(&self, id: NativeResourceId, offset: u64, size: u64, frame: u64) {
        #[cfg(feature = "devtools")]
        wgpui_devtools::gpu_resources::record_buffer_upload(
            to_devtools(id),
            wgpui_devtools::gpu_resources::ByteRange { offset, size },
            frame,
        );
        #[cfg(not(feature = "devtools"))]
        let _ = (id, offset, size, frame);
    }

    pub fn record_buffer_readback(&self, id: NativeResourceId, offset: u64, size: u64, frame: u64) {
        #[cfg(feature = "devtools")]
        wgpui_devtools::gpu_resources::record_buffer_readback(
            to_devtools(id),
            wgpui_devtools::gpu_resources::ByteRange { offset, size },
            frame,
        );
        #[cfg(not(feature = "devtools"))]
        let _ = (id, offset, size, frame);
    }

    pub fn record_texture_upload(
        &self,
        id: NativeResourceId,
        origin: [u32; 3],
        size: [u32; 3],
        bytes: u64,
        frame: u64,
    ) {
        #[cfg(feature = "devtools")]
        wgpui_devtools::gpu_resources::record_texture_upload(
            to_devtools(id),
            wgpui_devtools::gpu_resources::TextureRegion { origin, size },
            bytes,
            frame,
        );
        #[cfg(not(feature = "devtools"))]
        let _ = (id, origin, size, bytes, frame);
    }

    pub fn evict(&self, id: NativeResourceId, frame: u64) {
        #[cfg(feature = "devtools")]
        wgpui_devtools::gpu_resources::evict(to_devtools(id), frame);
        #[cfg(not(feature = "devtools"))]
        let _ = (id, frame);
    }
}

#[cfg(feature = "devtools")]
fn from_devtools(id: wgpui_devtools::gpu_resources::ResourceId) -> NativeResourceId {
    NativeResourceId(id.raw())
}

#[cfg(feature = "devtools")]
fn to_devtools(id: NativeResourceId) -> wgpui_devtools::gpu_resources::ResourceId {
    wgpui_devtools::gpu_resources::ResourceId::from_raw(id.raw())
}

#[cfg(feature = "devtools")]
impl From<NativeResourceRole> for wgpui_devtools::gpu_resources::ResourceRole {
    fn from(role: NativeResourceRole) -> Self {
        use wgpui_devtools::gpu_resources::ResourceRole;
        match role {
            NativeResourceRole::PrimitiveBuffer => ResourceRole::PrimitiveBuffer,
            NativeResourceRole::IndirectArguments => ResourceRole::IndirectArguments,
            NativeResourceRole::IndirectCount => ResourceRole::IndirectCount,
            NativeResourceRole::Visibility => ResourceRole::Visibility,
            NativeResourceRole::SlotTable => ResourceRole::SlotTable,
            NativeResourceRole::AtlasPage => ResourceRole::AtlasPage,
            NativeResourceRole::LayerTexture => ResourceRole::LayerTexture,
            NativeResourceRole::TileTexture => ResourceRole::TileTexture,
            NativeResourceRole::Surface => ResourceRole::Surface,
            NativeResourceRole::Staging => ResourceRole::Staging,
            NativeResourceRole::Readback => ResourceRole::Readback,
            NativeResourceRole::Query => ResourceRole::Query,
            NativeResourceRole::Uniform => ResourceRole::Uniform,
            NativeResourceRole::Other => ResourceRole::Other,
        }
    }
}

#[cfg(feature = "devtools")]
impl From<NativeResourceDimensions> for wgpui_devtools::gpu_resources::ResourceDimensions {
    fn from(dimensions: NativeResourceDimensions) -> Self {
        Self {
            width: dimensions.width,
            height: dimensions.height,
            depth_or_array_layers: dimensions.depth_or_array_layers,
        }
    }
}

#[cfg(all(test, feature = "devtools"))]
mod tests {
    use super::*;
    use wgpui_devtools::{CaptureRequest, ResourceKind, ResourceRole, start_capture, stop_capture};

    #[test]
    fn native_adapter_preserves_resource_categories_and_transfer_metadata() {
        assert!(start_capture(CaptureRequest { include_gpu: true }));
        let registry = NativeResourceRegistry;
        registry.begin_frame(12);
        let primitive = registry.register_buffer(
            "primitive arena",
            NativeResourceRole::PrimitiveBuffer,
            256,
            7,
            4,
        );
        let surface = registry.register_texture(
            "surface",
            NativeResourceRole::Surface,
            NativeResourceDimensions {
                width: 32,
                height: 16,
                depth_or_array_layers: 1,
            },
            "Rgba8Unorm",
            32 * 16 * 4,
            3,
            1,
        );
        let query = registry.register_query_set("timestamps", "Timestamp", 8, 0);
        registry.record_buffer_upload(primitive, 64, 32, 13);
        registry.record_texture_upload(surface, [2, 3, 0], [4, 5, 1], 80, 13);
        registry.mark_used(surface, 14);
        registry.evict(query, 15);

        let snapshot = stop_capture().expect("the capture was started");
        assert_eq!(snapshot.resources.len(), 3);
        assert_eq!(snapshot.resources[0].descriptor.kind, ResourceKind::Buffer);
        assert_eq!(
            snapshot.resources[0].descriptor.role,
            ResourceRole::PrimitiveBuffer
        );
        assert_eq!(snapshot.resources[0].descriptor.generation, 4);
        assert_eq!(snapshot.resources[0].uploads[0].bytes, 32);
        assert_eq!(snapshot.resources[1].descriptor.role, ResourceRole::Surface);
        assert_eq!(
            snapshot.resources[1]
                .descriptor
                .dimensions
                .map(|dimensions| dimensions.width),
            Some(32)
        );
        assert_eq!(snapshot.resources[1].last_use_frame, Some(14));
        assert_eq!(snapshot.resources[2].descriptor.role, ResourceRole::Query);
        assert!(!snapshot.resources[2].resident);
    }
}
