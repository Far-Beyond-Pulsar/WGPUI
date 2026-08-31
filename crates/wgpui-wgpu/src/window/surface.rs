use std::sync::Arc;

use crate::render::surface_registry::{SurfaceId, SurfaceRegistry};

struct SurfaceInner {
    registry: Arc<SurfaceRegistry>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    id: SurfaceId,
    format: wgpu::TextureFormat,
}

impl Drop for SurfaceInner {
    fn drop(&mut self) {
        self.registry.remove(self.id);
    }
}

/// A handle for rendering into a producer-owned surface that the retained
/// compositor samples as an external texture.
#[derive(Clone)]
pub struct WgpuSurfaceHandle {
    inner: Arc<SurfaceInner>,
}

impl WgpuSurfaceHandle {
    pub(crate) fn new(
        registry: Arc<SurfaceRegistry>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Self {
        let id = registry.create(device, width, height, format);
        Self {
            inner: Arc::new(SurfaceInner {
                registry,
                device: Arc::new(device.clone()),
                queue: Arc::new(queue.clone()),
                id,
                format,
            }),
        }
    }

    /// The opaque surface id used by [`crate::render::surface_registry`].
    pub fn id(&self) -> u64 {
        self.inner.id.as_raw()
    }

    /// The device that owns this surface's textures.
    pub fn device(&self) -> &wgpu::Device {
        &self.inner.device
    }

    /// The queue that owns submissions written into this surface.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.inner.queue
    }

    /// The surface texture format.
    pub fn format(&self) -> wgpu::TextureFormat {
        self.inner.format
    }

    /// The current back buffer and its physical dimensions.
    pub fn back_view_with_size(&self) -> Option<(wgpu::TextureView, (u32, u32))> {
        self.inner
            .registry
            .lock_and_get_back_with_size(self.inner.id)
    }

    /// Publish the buffer most recently rendered by this handle.
    ///
    /// The native example renders and publishes on the same queue as the UI
    /// compositor, so queue ordering makes the no-sync swap safe and avoids a
    /// device-wide poll on every animation frame.
    pub fn swap_buffers(&self) {
        self.inner
            .registry
            .swap_rendering_ready_no_sync(self.inner.id);
    }

    /// Alias for [`Self::swap_buffers`].
    pub fn present(&self) {
        self.swap_buffers();
    }
}
