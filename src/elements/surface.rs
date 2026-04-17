use crate::{
    App, Bounds, DevicePixels, Element, ElementId, GlobalElementId, InspectorElementId,
    IntoElement, LayoutId, ObjectFit, Pixels, Size, Style, StyleRefinement, Styled, Window,
};
#[cfg(target_os = "macos")]
use core_video::pixel_buffer::CVPixelBuffer;
use refineable::Refineable;
use std::sync::Arc;

/// A source of a [`Surface`] element's content.
///
/// Variants are platform-specific:
/// - macOS uses CoreVideo pixel buffers.
/// - Linux/FreeBSD use type-erased GPU textures.
/// - Windows uses shared texture handles.
pub enum SurfaceSource {
    /// A macOS image buffer from CoreVideo
    #[cfg(target_os = "macos")]
    ImageBuffer(CVPixelBuffer),
    /// Windows shared texture handle
    #[cfg(target_os = "windows")]
    SharedTexture {
        /// Native handle to the shared texture
        nt_handle: isize,
        /// Width of the texture in pixels
        width: u32,
        /// Height of the texture in pixels
        height: u32,
    },
    /// Linux DMA-BUF file descriptor
    #[cfg(target_os = "linux")]
    DmaBuf {
        /// File descriptor for the DMA-BUF
        fd: i32,
        /// Width of the texture in pixels
        width: u32,
        /// Height of the texture in pixels
        height: u32,
    },
    /// A GPU texture handle for use with the WGPU renderer (Linux/FreeBSD).
    ///
    /// GPUI keeps this as `Any` to avoid a hard dependency on `wgpu` in this crate.
    /// Callers should pass an `Arc<wgpu::Texture>` created from the renderer's device.
    /// Renderers that don't recognize the concrete type will skip drawing this surface.
    #[cfg(any(target_os = "linux", target_os = "freebsd"))]
    Texture {
        /// Type-erased GPU texture handle (expected to be `Arc<wgpu::Texture>`).
        texture: Arc<dyn std::any::Any + Send + Sync>,
        /// Dimensions of the texture in device pixels.
        size: Size<DevicePixels>,
    },
}

impl Clone for SurfaceSource {
    fn clone(&self) -> Self {
        match self {
            #[cfg(target_os = "macos")]
            Self::ImageBuffer(surface) => Self::ImageBuffer(surface.clone()),
            #[cfg(target_os = "windows")]
            Self::SharedTexture {
                nt_handle,
                width,
                height,
            } => Self::SharedTexture {
                nt_handle: *nt_handle,
                width: *width,
                height: *height,
            },
            #[cfg(target_os = "linux")]
            Self::DmaBuf { fd, width, height } => Self::DmaBuf {
                fd: *fd,
                width: *width,
                height: *height,
            },
            #[cfg(any(target_os = "linux", target_os = "freebsd"))]
            Self::Texture { texture, size } => Self::Texture {
                texture: Arc::clone(texture),
                size: *size,
            },
        }
    }
}

impl std::fmt::Debug for SurfaceSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(target_os = "macos")]
            Self::ImageBuffer(surface) => f.debug_tuple("ImageBuffer").field(surface).finish(),
            #[cfg(target_os = "windows")]
            Self::SharedTexture {
                nt_handle,
                width,
                height,
            } => f
                .debug_struct("SharedTexture")
                .field("nt_handle", nt_handle)
                .field("width", width)
                .field("height", height)
                .finish(),
            #[cfg(target_os = "linux")]
            Self::DmaBuf { fd, width, height } => f
                .debug_struct("DmaBuf")
                .field("fd", fd)
                .field("width", width)
                .field("height", height)
                .finish(),
            #[cfg(any(target_os = "linux", target_os = "freebsd"))]
            Self::Texture { size, .. } => f
                .debug_struct("Texture")
                .field("size", size)
                .finish_non_exhaustive(),
        }
    }
}

#[cfg(target_os = "macos")]
impl From<CVPixelBuffer> for SurfaceSource {
    fn from(value: CVPixelBuffer) -> Self {
        SurfaceSource::ImageBuffer(value)
    }
}

/// A surface element.
pub struct Surface {
    source: SurfaceSource,
    object_fit: ObjectFit,
    style: StyleRefinement,
}

/// Create a new surface element.
pub fn surface(source: impl Into<SurfaceSource>) -> Surface {
    Surface {
        source: source.into(),
        object_fit: ObjectFit::Contain,
        style: Default::default(),
    }
}

impl Surface {
    /// Set the object fit for the image.
    pub fn object_fit(mut self, object_fit: ObjectFit) -> Self {
        self.object_fit = object_fit;
        self
    }
}

impl Element for Surface {
    type RequestLayoutState = ();
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let mut style = Style::default();
        style.refine(&self.style);
        let layout_id = window.request_layout(style, [], cx);
        (layout_id, ())
    }

    fn prepaint(
        &mut self,
        _global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        _window: &mut Window,
        _cx: &mut App,
    ) -> Self::PrepaintState {
    }

    fn paint(
        &mut self,
        _global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _: &mut Self::RequestLayoutState,
        _: &mut Self::PrepaintState,
        window: &mut Window,
        _: &mut App,
    ) {
        match &self.source {
            #[cfg(target_os = "macos")]
            SurfaceSource::ImageBuffer(surface) => {
                // Drawing CVPixelBuffer directly is no longer supported by the
                // backend; the old `window.paint_surface` helper was removed
                // in favor of `paint_wgpu_surface` which takes a `SurfaceId`.
                // Proper support would require registering the buffer as a WGPU
                // surface and copying its contents into the registry.
                // For now we simply ignore the buffer so the library compiles.
                let _ = surface;
            }
            #[cfg(any(target_os = "linux", target_os = "freebsd"))]
            SurfaceSource::Texture { texture, size } => {
                let new_bounds = self.object_fit.get_bounds(bounds, *size);
                window.paint_surface(new_bounds, Arc::clone(texture), *size);
            }
            #[allow(unreachable_patterns)]
            _ => {}
        }
    }
}

impl IntoElement for Surface {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Styled for Surface {
    fn style(&mut self) -> &mut StyleRefinement {
        &mut self.style
    }
}

#[cfg(all(test, any(target_os = "linux", target_os = "freebsd")))]
mod tests {
    use super::SurfaceSource;
    use crate::{DevicePixels, size};
    use std::sync::Arc;

    #[test]
    fn texture_source_is_cloneable_and_debuggable() {
        let source = SurfaceSource::Texture {
            texture: Arc::new(123_u32),
            size: size(DevicePixels(64), DevicePixels(32)),
        };

        let cloned = source.clone();
        let debug_output = format!("{cloned:?}");

        assert!(debug_output.contains("Texture"));
        assert!(debug_output.contains("size"));
    }
}
