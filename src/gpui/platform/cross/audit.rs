//! Real-GPU rendering audit support: copy a render-target texture back to the
//! CPU as RGBA8, and a self-contained offscreen harness that exercises the
//! custom render-primitive plugin GPU draw path without a window.
//!
//! This module must live inside the `cross` module because it touches
//! [`WgpuContext`]'s `pub(super)` device/queue. All GPU work here runs on the
//! caller's thread, which must be the renderer/device-owning thread (polling the
//! device from another thread risks driver contention / device-lost).

use anyhow::Result;

use crate::PixelBuffer;
#[cfg(test)]
use crate::platform::cross::render_context::WgpuContext;

/// Copy `texture` back to the CPU as tightly packed RGBA8.
///
/// Handles the mandatory 256-byte `bytes_per_row` padding and swizzles BGRA
/// targets into canonical RGBA so assertions are channel-correct regardless of
/// the adapter's preferred surface format. Must run on the device-owning thread,
/// after the work that wrote `texture` has been submitted.
pub(crate) fn read_texture_rgba8(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    format: wgpu::TextureFormat,
) -> Result<PixelBuffer> {
    let size = texture.size();
    let (width, height) = (size.width, size.height);
    let bytes_per_pixel = 4u32;
    let unpadded_bytes_per_row = width * bytes_per_pixel;
    let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("audit_readback"),
        size: u64::from(padded_bytes_per_row) * u64::from(height),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("audit_readback_encoder"),
    });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_result| {});
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|error| anyhow::anyhow!("device poll failed during readback: {error:?}"))?;

    // Every render target the renderer picks is 4 bytes/pixel, but the channel
    // packing differs. Decode each into canonical RGBA8 rather than assuming a
    // byte layout (a naive RGBA8 read garbles 10-bit and BGRA targets).
    let decode = PixelDecode::for_format(format)?;
    let mapped = slice.get_mapped_range();
    let mut pixels = Vec::with_capacity((unpadded_bytes_per_row * height) as usize);
    for row in 0..height as usize {
        let start = row * padded_bytes_per_row as usize;
        let line = &mapped[start..start + unpadded_bytes_per_row as usize];
        for texel in line.chunks_exact(4) {
            pixels.extend_from_slice(&decode.to_rgba8(texel));
        }
    }
    drop(mapped);
    readback.unmap();

    Ok(PixelBuffer::new(width, height, pixels))
}

/// How a 4-byte texel of a given surface format maps to canonical RGBA8.
#[derive(Clone, Copy)]
enum PixelDecode {
    /// Bytes already in R, G, B, A order.
    Rgba8,
    /// Bytes in B, G, R, A order (swap to RGBA).
    Bgra8,
    /// 10/10/10/2 bits packed little-endian into a u32 (R low, A high).
    Rgb10a2,
}

impl PixelDecode {
    fn for_format(format: wgpu::TextureFormat) -> Result<Self> {
        match format {
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => {
                Ok(Self::Rgba8)
            }
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => {
                Ok(Self::Bgra8)
            }
            wgpu::TextureFormat::Rgb10a2Unorm => Ok(Self::Rgb10a2),
            other => Err(anyhow::anyhow!(
                "readback does not support surface format {other:?}"
            )),
        }
    }

    fn to_rgba8(self, texel: &[u8]) -> [u8; 4] {
        match self {
            Self::Rgba8 => [texel[0], texel[1], texel[2], texel[3]],
            Self::Bgra8 => [texel[2], texel[1], texel[0], texel[3]],
            Self::Rgb10a2 => {
                let value = u32::from_le_bytes([texel[0], texel[1], texel[2], texel[3]]);
                // 10-bit channels -> 8-bit (drop low 2 bits); 2-bit alpha -> 0/85/170/255.
                let red = ((value & 0x3ff) >> 2) as u8;
                let green = (((value >> 10) & 0x3ff) >> 2) as u8;
                let blue = (((value >> 20) & 0x3ff) >> 2) as u8;
                let alpha = (((value >> 30) & 0x3) * 85) as u8;
                [red, green, blue, alpha]
            }
        }
    }
}

/// A windowless GPU harness that renders a single custom-primitive batch into an
/// offscreen texture and reads the result back, so the real plugin pipeline
/// (`build` + `draw`) can be pixel-validated deterministically without a winit
/// surface. Construction returns `Ok(None)` when no suitable GPU/adapter is
/// available (e.g. a headless CI runner), so callers can skip rather than fail.
#[cfg(test)]
pub(crate) struct OffscreenGpu {
    context: WgpuContext,
    format: wgpu::TextureFormat,
    target: wgpu::Texture,
    target_view: wgpu::TextureView,
    globals_layout: wgpu::BindGroupLayout,
    globals_bind_group: wgpu::BindGroup,
    _globals_buffer: wgpu::Buffer,
}

#[cfg(test)]
impl OffscreenGpu {
    /// Create an offscreen target of `width` x `height`. Returns `Ok(None)` if a
    /// real GPU device cannot be created on this machine.
    pub(crate) fn new(width: u32, height: u32) -> Result<Option<Self>> {
        let context = match WgpuContext::new() {
            Ok(context) => context,
            // No suitable adapter/device (e.g. headless CI): skip gracefully.
            Err(_) => return Ok(None),
        };

        // Non-sRGB, RGBA channel order: matches the renderer's deliberate non-sRGB
        // choice (shaders emit sRGB bytes) and needs no readback swizzle.
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let device = &context.device;

        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("audit_offscreen_target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());

        // Globals: viewport_size in the first two floats, matching the SolidQuads
        // shader and the renderer's globals layout (renderer.rs globals_bind_group).
        let globals_data: [f32; 4] = [width as f32, height as f32, 0.0, 0.0];
        let globals_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("audit_globals"),
            size: std::mem::size_of_val(&globals_data) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        context
            .queue
            .write_buffer(&globals_buffer, 0, bytemuck::cast_slice(&globals_data));

        let globals_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("audit_globals_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let globals_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("audit_globals_bind_group"),
            layout: &globals_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals_buffer.as_entire_binding(),
            }],
        });

        Ok(Some(Self {
            context,
            format,
            target,
            target_view,
            globals_layout,
            globals_bind_group,
            _globals_buffer: globals_buffer,
        }))
    }

    /// The offscreen target's color format. Pixel tests must pack instances and
    /// expect colors consistent with this (non-sRGB) format.
    pub(crate) fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    /// Render one custom-primitive batch over `clear`, then read the result back.
    /// This drives the real [`gpui_render_primitive::PrimitiveRegistry::draw_batch`]
    /// path on a real GPU.
    pub(crate) fn render_primitive_batch(
        &mut self,
        registry: &mut gpui_render_primitive::PrimitiveRegistry,
        id: gpui_render_primitive::PrimitiveTypeId,
        instances: &[u8],
        instance_count: u32,
        clear: wgpu::Color,
    ) -> Result<PixelBuffer> {
        let device = &self.context.device;
        let queue = &self.context.queue;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("audit_offscreen_encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("audit_offscreen_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.target_view,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear),
                        store: wgpu::StoreOp::Store,
                    },
                    resolve_target: None,
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            registry.draw_batch(
                id,
                instances,
                instance_count,
                device,
                queue,
                self.format,
                &self.globals_layout,
                &self.globals_bind_group,
                &mut pass,
            );
        }
        queue.submit(Some(encoder.finish()));

        read_texture_rgba8(device, queue, &self.target, self.format)
    }
}

#[cfg(test)]
mod tests {
    use gpui_prim_solid::{SolidQuad, SolidQuads};
    use gpui_render_primitive::PrimitiveRegistry;

    use super::{OffscreenGpu, PixelDecode};

    // GPU-free regression tests for the readback decoder. The windowed surface on
    // some adapters is Rgb10a2Unorm (10/10/10/2 packed), which a naive RGBA8 read
    // garbles; these lock the per-format decode without needing a real display.

    #[test]
    fn pixel_decode_rgb10a2_unpacks_packed_channels() {
        let decode = PixelDecode::for_format(wgpu::TextureFormat::Rgb10a2Unorm).unwrap();
        // The actual framebuffer value captured for #113355: R=68,G=205,B=340 (10-bit), A=3.
        let value: u32 = 68 | (205 << 10) | (340 << 20) | (3 << 30);
        assert_eq!(decode.to_rgba8(&value.to_le_bytes()), [17, 51, 85, 255]);
    }

    #[test]
    fn pixel_decode_bgra8_swizzles_to_rgba() {
        let decode = PixelDecode::for_format(wgpu::TextureFormat::Bgra8Unorm).unwrap();
        assert_eq!(decode.to_rgba8(&[85, 51, 17, 255]), [17, 51, 85, 255]);
    }

    #[test]
    fn pixel_decode_rgba8_passthrough() {
        let decode = PixelDecode::for_format(wgpu::TextureFormat::Rgba8Unorm).unwrap();
        assert_eq!(decode.to_rgba8(&[17, 51, 85, 255]), [17, 51, 85, 255]);
    }

    #[test]
    fn pixel_decode_rejects_unsupported_format() {
        assert!(PixelDecode::for_format(wgpu::TextureFormat::R8Unorm).is_err());
    }

    /// The custom render-primitive plugin GPU draw, validated at the pixel level
    /// on a real GPU: a red SolidQuad painted over a black clear must produce red
    /// pixels inside the quad and black pixels outside it. Skips when no GPU is
    /// available so the suite still passes on headless runners.
    #[test]
    fn solid_quad_plugin_draws_red_pixels_on_gpu() {
        let Some(mut gpu) = OffscreenGpu::new(64, 64).expect("offscreen gpu init") else {
            eprintln!("solid_quad_plugin_draws_red_pixels_on_gpu: no GPU available, skipping");
            return;
        };

        let mut registry = PrimitiveRegistry::new();
        registry.register(Box::new(SolidQuads));

        // Opaque red quad covering the interior [16, 48) x [16, 48).
        let quad = SolidQuad::new([16.0, 16.0], [32.0, 32.0], [1.0, 0.0, 0.0, 1.0]);
        let image = gpu
            .render_primitive_batch(
                &mut registry,
                SolidQuads::TYPE,
                quad.as_bytes(),
                1,
                wgpu::Color::BLACK,
            )
            .expect("render + readback");

        assert_eq!(image.width, 64);
        assert_eq!(image.height, 64);

        // Center is inside the quad -> red.
        let center = image.pixel_at(32, 32);
        assert!(
            center[0] > 200 && center[1] < 60 && center[2] < 60 && center[3] > 200,
            "center pixel should be opaque red, got {center:?}"
        );
        // A corner is outside the quad -> black clear.
        let corner = image.pixel_at(2, 2);
        assert!(
            corner[0] < 30 && corner[1] < 30 && corner[2] < 30,
            "corner pixel should be black clear, got {corner:?}"
        );
    }

    /// The harness reports the non-sRGB format the plugin pipeline is built
    /// against, so pixel expectations stay consistent with production.
    #[test]
    fn offscreen_target_is_non_srgb() {
        let Some(gpu) = OffscreenGpu::new(8, 8).expect("offscreen gpu init") else {
            return;
        };
        assert_eq!(gpu.format(), wgpu::TextureFormat::Rgba8Unorm);
    }
}
