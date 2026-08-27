//! Render pipeline creation per primitive kind (today's
//! quads/shadows/.../paths pipeline construction in `src/renderer.rs`).
//! See docs/gpu-native-architecture.md §3.5.
//!
//! # Two pipelines, not eight, and why that is not a shortcut
//!
//! The legacy renderer has eight (`quads`, `shadows`, `mono_sprites`,
//! `poly_sprites`, `paths`, `underlines`, `backdrop_blur`, `surfaces`). Phase 4
//! builds two: [`QuadPipeline`] and [`CompositePipeline`].
//!
//! That is the same ratio Phase 1 chose for primitive kinds and for the same
//! reason (`patch/primitive.rs`'s module doc: two kinds because they are the
//! two structurally different shapes, not because seven was too many to type).
//! The two here are the two structurally different *draw* shapes §5.3 and §5.5
//! ask about:
//!
//! - **[`QuadPipeline`]** is an instanced pipeline pulling per-instance data
//!   out of a storage buffer, drawn through an indirect argument record whose
//!   instance count the GPU decided. Adding `shadows` or `underlines` is
//!   another shader with the same bind group layout and the same draw call;
//!   nothing in `render/draw.rs` is written per kind.
//! - **[`CompositePipeline`]** draws one already-rendered texture into the
//!   scene, and is where §5.5's Gap 2 lands: both producers bind the same
//!   layout and take the same call.
//!
//! What is genuinely absent is the atlas-backed sprite path and `paths`' own
//! vertex buffer, both of which need machinery (`render/atlas.rs`,
//! path tessellation) that no phase has built yet.

use wgpui_core::patch::primitive::Primitive;

/// Colour format every Phase 4 target uses.
///
/// `Rgba8Unorm` rather than a float format: it is blendable everywhere without
/// a device feature, and every comparison this phase makes is between two draw
/// paths writing the same format, so quantisation is common to both sides
/// rather than a source of disagreement.
pub const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Straight-alpha `over`, the blend the legacy renderer's instanced pipelines
/// use.
const ALPHA_OVER: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::SrcAlpha,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
};

/// The frame-constant uniform both pipelines read.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Globals {
    /// Framebuffer size in pixels.
    pub viewport: [f32; 2],
}

impl Globals {
    /// The 16 bytes the shader's `Globals` struct expects.
    pub fn to_bytes(self) -> [u8; 16] {
        let mut bytes = [0u8; 16];
        bytes[0..4].copy_from_slice(&self.viewport[0].to_le_bytes());
        bytes[4..8].copy_from_slice(&self.viewport[1].to_le_bytes());
        bytes
    }
}

/// The instanced quad pipeline plus the bind groups its draws need.
pub struct QuadPipeline {
    /// The pipeline itself.
    pub pipeline: wgpu::RenderPipeline,
    /// Globals, arena, and indirection buffer.
    pub frame_layout: wgpu::BindGroupLayout,
    /// The per-slot base index, addressed by dynamic offset.
    pub slot_layout: wgpu::BindGroupLayout,
    /// Byte stride between two slots' entries in the slot-base buffer, which is
    /// the device's own `min_uniform_buffer_offset_alignment`.
    pub slot_stride: u32,
}

impl QuadPipeline {
    /// Build the pipeline. Once per device, never per frame.
    pub fn new(device: &wgpu::Device) -> QuadPipeline {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wgpui quads"),
            source: wgpu::ShaderSource::Wgsl(super::shaders::QUADS_WGSL.into()),
        });
        let frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("quads frame"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT, 16, false),
                storage_entry(1, wgpu::ShaderStages::VERTEX_FRAGMENT),
                storage_entry(2, wgpu::ShaderStages::VERTEX),
            ],
        });
        let slot_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("quads slot base"),
            entries: &[uniform_entry(0, wgpu::ShaderStages::VERTEX, 16, true)],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("quads"),
            bind_group_layouts: &[Some(&frame_layout), Some(&slot_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("quads"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vertex_main"),
                compilation_options: Default::default(),
                // §1: "every render pipeline binds `vertex.buffers: &[]` and
                // indexes a bound storage buffer with `@builtin(instance_index)`
                // in the shader." Unchanged, deliberately.
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fragment_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: TARGET_FORMAT,
                    blend: Some(ALPHA_OVER),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: Default::default(),
            cache: None,
        });
        QuadPipeline {
            pipeline,
            frame_layout,
            slot_layout,
            slot_stride: device
                .limits()
                .min_uniform_buffer_offset_alignment
                .max(16),
        }
    }

    /// The bind group holding this frame's globals, quad arena, and indirection
    /// buffer.
    pub fn frame_bind_group(
        &self,
        device: &wgpu::Device,
        globals: &wgpu::Buffer,
        arena: &wgpu::Buffer,
        visible: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("quads frame"),
            layout: &self.frame_layout,
            entries: &[
                buffer_entry(0, globals),
                buffer_entry(1, arena),
                buffer_entry(2, visible),
            ],
        })
    }

    /// The bind group over a slot-base buffer, read one slot at a time through
    /// a dynamic offset.
    pub fn slot_bind_group(&self, device: &wgpu::Device, bases: &wgpu::Buffer) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("quads slot base"),
            layout: &self.slot_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: bases,
                    offset: 0,
                    size: std::num::NonZeroU64::new(16),
                }),
            }],
        })
    }

    /// Bytes one quad occupies in the arena, restated from `wgpui-core` so a
    /// drift between the shader's `QuadSlot` and the protocol's `Quad` fails a
    /// test rather than rendering garbage.
    pub const fn arena_slot_stride() -> usize {
        wgpui_core::patch::primitive::Quad::SLOT_STRIDE
    }
}

/// The one composite pipeline both producers draw through (§5.5, Gap 2).
pub struct CompositePipeline {
    /// The pipeline itself.
    pub pipeline: wgpu::RenderPipeline,
    /// Globals alone.
    pub frame_layout: wgpu::BindGroupLayout,
    /// One entry's parameters, texture, and sampler.
    pub entry_layout: wgpu::BindGroupLayout,
    /// The sampler every entry uses.
    pub sampler: wgpu::Sampler,
}

impl CompositePipeline {
    /// Bytes one entry's parameter block occupies: three `vec4<f32>`.
    pub const PARAMS_SIZE: u64 = 48;

    /// Build the pipeline. Once per device, never per frame.
    pub fn new(device: &wgpu::Device) -> CompositePipeline {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wgpui composite"),
            source: wgpu::ShaderSource::Wgsl(super::shaders::SURFACES_WGSL.into()),
        });
        let frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composite frame"),
            entries: &[uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT, 16, false)],
        });
        let entry_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composite entry"),
            entries: &[
                uniform_entry(
                    0,
                    wgpu::ShaderStages::VERTEX_FRAGMENT,
                    Self::PARAMS_SIZE,
                    false,
                ),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("composite"),
            bind_group_layouts: &[Some(&frame_layout), Some(&entry_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("composite"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vertex_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fragment_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: TARGET_FORMAT,
                    blend: Some(ALPHA_OVER),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: Default::default(),
            cache: None,
        });
        CompositePipeline {
            pipeline,
            frame_layout,
            entry_layout,
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("composite"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
        }
    }

    /// The bind group holding this frame's globals.
    pub fn frame_bind_group(&self, device: &wgpu::Device, globals: &wgpu::Buffer) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite frame"),
            layout: &self.frame_layout,
            entries: &[buffer_entry(0, globals)],
        })
    }

    /// One entry's bind group.
    ///
    /// The only place a boundary's baked texture and an external surface's
    /// triple-buffered one differ is which `view` reaches this function, which
    /// is §5.5's Gap 2 stated as a signature.
    pub fn entry_bind_group(
        &self,
        device: &wgpu::Device,
        params: &wgpu::Buffer,
        view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite entry"),
            layout: &self.entry_layout,
            entries: &[
                buffer_entry(0, params),
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }
}

fn buffer_entry<'a>(index: u32, buffer: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding: index,
        resource: buffer.as_entire_binding(),
    }
}

fn uniform_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    size: u64,
    dynamic: bool,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: dynamic,
            min_binding_size: std::num::NonZeroU64::new(size),
        },
        count: None,
    }
}

fn storage_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shaders_agree_with_the_protocol_about_a_quad_slot() {
        // `quads.wgsl`'s `QuadSlot` is four `vec4<f32>`. If `Quad::SLOT_STRIDE`
        // ever changes without the shader changing with it, every instance past
        // the first reads the wrong bytes and renders plausible garbage — so
        // the agreement is asserted rather than left to a comment.
        assert_eq!(QuadPipeline::arena_slot_stride(), 4 * 16);
        assert!(super::super::shaders::QUADS_WGSL.contains("struct QuadSlot"));
    }

    #[test]
    fn the_composite_parameter_block_is_three_vec4s() {
        assert_eq!(CompositePipeline::PARAMS_SIZE, 3 * 16);
    }
}
