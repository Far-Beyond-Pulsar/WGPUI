use std::sync::Arc;

use crate::render::surface_registry::{SurfaceId, SurfaceRegistry};

/// Why a producer-owned surface could not be resized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceResizeError {
    /// A WGPU texture cannot have a zero-sized extent.
    ZeroSize,
    /// The surface is currently being consumed by the compositor.
    Busy,
    /// The handle no longer refers to a registered surface.
    NotFound,
}

impl std::fmt::Display for SurfaceResizeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroSize => formatter.write_str("surface dimensions must be non-zero"),
            Self::Busy => formatter.write_str("surface is busy with compositor presentation"),
            Self::NotFound => formatter.write_str("surface is no longer registered"),
        }
    }
}

impl std::error::Error for SurfaceResizeError {}

struct SurfaceInner {
    registry: Arc<SurfaceRegistry>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    id: SurfaceId,
    format: wgpu::TextureFormat,
    request_redraw: Arc<dyn Fn() + Send + Sync>,
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
        request_redraw: Arc<dyn Fn() + Send + Sync>,
    ) -> Self {
        let id = registry.create(device, width, height, format);
        Self {
            inner: Arc::new(SurfaceInner {
                registry,
                device: Arc::new(device.clone()),
                queue: Arc::new(queue.clone()),
                id,
                format,
                request_redraw,
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

    /// The current producer texture dimensions in physical pixels.
    pub fn size(&self) -> Option<(u32, u32)> {
        self.inner.registry.size(self.inner.id)
    }

    /// Resize all producer buffers while preserving this handle's identity.
    ///
    /// A resize is rejected while the compositor is consuming a published
    /// frame. The caller should retry after the next presentation callback.
    pub fn resize(&self, width: u32, height: u32) -> Result<(), SurfaceResizeError> {
        if width == 0 || height == 0 {
            return Err(SurfaceResizeError::ZeroSize);
        }
        if self.size().is_none() {
            return Err(SurfaceResizeError::NotFound);
        }
        if self
            .inner
            .registry
            .resize(self.device(), self.inner.id, width, height)
        {
            Ok(())
        } else {
            Err(SurfaceResizeError::Busy)
        }
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

    /// Publish a frame and request exactly one native redraw until the frame is
    /// consumed. Multiple producer updates between display frames are folded
    /// into the latest ready buffer.
    pub fn present(&self) {
        self.inner
            .registry
            .swap_rendering_ready_no_sync(self.inner.id);
        if !self.inner.registry.set_redraw_pending(self.inner.id) {
            (self.inner.request_redraw)();
        }
    }

    /// Publish a frame with its queue submission recorded for cross-thread
    /// pacing diagnostics.
    pub fn present_synced(&self, submission_index: wgpu::SubmissionIndex) {
        self.inner
            .registry
            .swap_rendering_ready(self.inner.id, submission_index);
        if !self.inner.registry.set_redraw_pending(self.inner.id) {
            (self.inner.request_redraw)();
        }
    }

    /// Publish a frame without scheduling a native redraw.
    pub fn present_synced_silent(&self, submission_index: wgpu::SubmissionIndex) {
        self.inner
            .registry
            .swap_rendering_ready(self.inner.id, submission_index);
    }

    /// Whether a published frame is waiting for the compositor.
    pub fn has_unconsumed_frame(&self) -> bool {
        self.inner.registry.has_unconsumed_frame(self.inner.id)
    }
}
