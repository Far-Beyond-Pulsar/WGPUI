//! Render pipeline creation per primitive kind (today's
//! quads/shadows/.../paths pipeline construction in `src/renderer.rs`).
//! See docs/gpu-native-architecture.md §3.5.
//!
//! # Four pipelines, not eight, and why that is not a shortcut
//!
//! The legacy renderer has eight (`quads`, `shadows`, `mono_sprites`,
//! `poly_sprites`, `paths`, `underlines`, `backdrop_blur`, `surfaces`). Phase 4
//! built two — [`QuadPipeline`] and [`CompositePipeline`] — Phase 5.6 added the
//! third, [`MonoSpritePipeline`], and Phase 6.2 the fourth,
//! [`PolySpritePipeline`].
//!
//! That is the same ratio Phase 1 chose for primitive kinds and for the same
//! reason (`patch/primitive.rs`'s module doc: two kinds because they are the
//! two structurally different shapes, not because seven was too many to type).
//! The three here are the three structurally different *draw* shapes §5.3, §5.5
//! and §9's glyph row ask about:
//!
//! - **[`QuadPipeline`]** is an instanced pipeline pulling per-instance data
//!   out of a storage buffer, drawn through an indirect argument record whose
//!   instance count the GPU decided. Adding `shadows` or `underlines` is
//!   another shader with the same bind group layout and the same draw call;
//!   nothing in `render/draw.rs` is written per kind.
//! - **[`MonoSpritePipeline`]** is that same instanced shape *plus a texture*:
//!   one glyph per instance, its coverage mask read out of the atlas page its
//!   `AtlasTileId` names. The texture is the whole of the difference, and it is
//!   the difference that made a sprite pipeline a phase of its own rather than
//!   another shader against [`QuadPipeline`]'s layout — a bind group cannot
//!   change inside a draw call, so "which page" has to become part of how the
//!   draw sequence is issued. See `shaders/mono_sprites.wgsl` for the resolution
//!   and [`crate::render::draw::issue_glyphs`] for the sequence.
//! - **[`PolySpritePipeline`]** is [`MonoSpritePipeline`]'s shape over a colour
//!   page: one image per instance, four channels instead of one. It is the
//!   fourth pipeline and the *first* that needed no new mechanism at all — the
//!   bind-group shape, the page loop, and the slot-base plan were all already
//!   there, which is the strongest form of the claim `render/draw.rs` has been
//!   making since Phase 4 that nothing in it is written per kind.
//! - **[`CompositePipeline`]** draws one already-rendered texture into the
//!   scene, and is where §5.5's Gap 2 lands: both producers bind the same
//!   layout and take the same call.
//!
//! - **[`ShadowPipeline`]** (Phase 6.3) is the first that needed nothing new at
//!   *this* layer either — same layouts, same draw, another shader — which is
//!   what §8's "`QuadPipeline`-shaped" claim was about and where it holds. The
//!   two places it does not hold are outside this file entirely (a shadow paints
//!   past its own rectangle, and is never culled); see [`ShadowPipeline`]'s own
//!   doc, which names them rather than letting "cheap" stand unqualified.
//!
//! - **[`UnderlinePipeline`]** (Phase 6.3) is [`ShadowPipeline`]'s construction
//!   over another arena and another shader, with no qualification anywhere: it
//!   is the one kind for which "`QuadPipeline`-shaped" is true end to end.
//!
//! What is still genuinely absent is `backdrop_blur` and `paths`' own vertex
//! buffer, which needs tessellation machinery no phase has built (Phase 6.4).
//!
//! **Colour glyphs are not drawn either, and that is not the same absence.**
//! [`PolySpritePipeline`] samples exactly the page a colour emoji's raster
//! lands in, so the pipeline that would draw one now exists. What is missing is
//! that `wgpui-text`'s conversion emits emoji into a [`MonoSpritePipeline`]-only
//! `GlyphRun`, and routing them to this kind instead is a `wgpui-text` change,
//! not a renderer one. Recorded here because "poly_sprites exists" would
//! otherwise read as "colour emoji work."

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
pub(crate) const ALPHA_OVER: wgpu::BlendState = wgpu::BlendState {
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
            slot_stride: device.limits().min_uniform_buffer_offset_alignment.max(16),
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
        slot_base_bind_group(device, &self.slot_layout, bases)
    }

    /// Bytes one quad occupies in the arena, restated from `wgpui-core` so a
    /// drift between the shader's `QuadSlot` and the protocol's `Quad` fails a
    /// test rather than rendering garbage.
    pub const fn arena_slot_stride() -> usize {
        wgpui_core::patch::primitive::Quad::SLOT_STRIDE
    }
}

/// The instanced shadow pipeline: a blurred, rounded rectangle per instance.
///
/// **[`QuadPipeline`]'s shape exactly** — the same two bind group layouts over
/// the same three frame resources, the same empty vertex buffer list, the same
/// triangle strip, the same blend state — over the `Shadow` arena and a
/// different shader. §8's `6.3` row calls this "`QuadPipeline`-shaped" and at
/// *this* layer the claim holds without qualification; the two places it does
/// not are both outside this file and are recorded at
/// [`crate::render::frame::FrameRenderer::render_to`] and on
/// [`wgpui_core::patch::primitive::Shadow`]:
///
/// 1. The shader draws a rectangle *larger* than the primitive's own, so the
///    ordering pass is fed [`wgpui_core::patch::primitive::Shadow::drawn_bounds`]
///    rather than `origin`/`size`.
/// 2. A shadow is never culled
///    ([`wgpui_core::occlusion::CoverageItem::uncullable`]), where every prior
///    kind is a `cullee`.
///
/// Neither changes anything here, which is the point of recording them here:
/// the *pipeline* really is another shader against the quad layout, and the
/// real work of the phase was elsewhere.
pub struct ShadowPipeline {
    /// The pipeline itself.
    pub pipeline: wgpu::RenderPipeline,
    /// Globals, arena, and indirection buffer.
    pub frame_layout: wgpu::BindGroupLayout,
    /// The per-slot base index, addressed by dynamic offset.
    pub slot_layout: wgpu::BindGroupLayout,
    /// Byte stride between two slots' entries in the slot-base buffer.
    pub slot_stride: u32,
}

impl ShadowPipeline {
    /// Build the pipeline. Once per device, never per frame.
    pub fn new(device: &wgpu::Device) -> ShadowPipeline {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wgpui shadows"),
            source: wgpu::ShaderSource::Wgsl(super::shaders::SHADOWS_WGSL.into()),
        });
        let frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadows frame"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT, 16, false),
                storage_entry(1, wgpu::ShaderStages::VERTEX_FRAGMENT),
                storage_entry(2, wgpu::ShaderStages::VERTEX),
            ],
        });
        let slot_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadows slot base"),
            entries: &[uniform_entry(0, wgpu::ShaderStages::VERTEX, 16, true)],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadows"),
            bind_group_layouts: &[Some(&frame_layout), Some(&slot_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadows"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vertex_main"),
                compilation_options: Default::default(),
                // §1's rule, unchanged for this kind too.
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
        ShadowPipeline {
            pipeline,
            frame_layout,
            slot_layout,
            slot_stride: device.limits().min_uniform_buffer_offset_alignment.max(16),
        }
    }

    /// The bind group holding this frame's globals, shadow arena, and
    /// indirection buffer.
    pub fn frame_bind_group(
        &self,
        device: &wgpu::Device,
        globals: &wgpu::Buffer,
        arena: &wgpu::Buffer,
        visible: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadows frame"),
            layout: &self.frame_layout,
            entries: &[
                buffer_entry(0, globals),
                buffer_entry(1, arena),
                buffer_entry(2, visible),
            ],
        })
    }

    /// The bind group over a slot-base buffer.
    pub fn slot_bind_group(&self, device: &wgpu::Device, bases: &wgpu::Buffer) -> wgpu::BindGroup {
        slot_base_bind_group(device, &self.slot_layout, bases)
    }

    /// Bytes one shadow occupies in the arena, restated from `wgpui-core` so a
    /// drift between the shader's `ShadowSlot` and the protocol's `Shadow` fails
    /// a test rather than rendering garbage.
    pub const fn arena_slot_stride() -> usize {
        wgpui_core::patch::primitive::Shadow::SLOT_STRIDE
    }
}

/// The instanced underline pipeline: a straight or wavy rule per instance.
///
/// [`ShadowPipeline`]'s construction exactly, over the `Underline` arena and
/// `underlines.wgsl`. Unlike [`ShadowPipeline`] there is no qualification to
/// attach: an underline paints inside its own rectangle and is an ordinary
/// occlusion cullee, so §8's "`QuadPipeline`-shaped" claim holds for this kind
/// end to end — pipeline, ordering, and occlusion alike. It is the cheapest
/// primitive kind 2.0 has added, and saying so is only worth anything beside
/// the shadow row that qualifies the same phrase.
pub struct UnderlinePipeline {
    /// The pipeline itself.
    pub pipeline: wgpu::RenderPipeline,
    /// Globals, arena, and indirection buffer.
    pub frame_layout: wgpu::BindGroupLayout,
    /// The per-slot base index, addressed by dynamic offset.
    pub slot_layout: wgpu::BindGroupLayout,
    /// Byte stride between two slots' entries in the slot-base buffer.
    pub slot_stride: u32,
}

/// The Lyon-tessellated path pipeline.
pub struct PathPipeline {
    /// The pipeline itself.
    pub pipeline: wgpu::RenderPipeline,
    /// Globals and path arena.
    pub frame_layout: wgpu::BindGroupLayout,
    /// The base vertex for the current layer's flattened stream.
    pub slot_layout: wgpu::BindGroupLayout,
    /// Dynamic-uniform alignment in bytes.
    pub slot_stride: u32,
}

impl PathPipeline {
    /// Build the path pipeline. Paths use a triangle list and one direct draw
    /// per layer because one primitive owns a variable number of vertices.
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wgpui paths"),
            source: wgpu::ShaderSource::Wgsl(super::shaders::PATHS_WGSL.into()),
        });
        let frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("paths frame"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT, 16, false),
                storage_entry(1, wgpu::ShaderStages::VERTEX_FRAGMENT),
            ],
        });
        let slot_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("paths slot base"),
            entries: &[uniform_entry(0, wgpu::ShaderStages::VERTEX, 16, true)],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("paths"),
            bind_group_layouts: &[Some(&frame_layout), Some(&slot_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("paths"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vertex_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
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
        Self {
            pipeline,
            frame_layout,
            slot_layout,
            slot_stride: device.limits().min_uniform_buffer_offset_alignment.max(16),
        }
    }

    /// Bind globals and the flattened path arena.
    pub fn frame_bind_group(
        &self,
        device: &wgpu::Device,
        globals: &wgpu::Buffer,
        arena: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("paths frame"),
            layout: &self.frame_layout,
            entries: &[buffer_entry(0, globals), buffer_entry(1, arena)],
        })
    }
}

/// The framebuffer-sampling backdrop-filter pipeline.
pub struct BackdropPipeline {
    /// The pipeline itself.
    pub pipeline: wgpu::RenderPipeline,
    /// Globals and filter arena.
    pub frame_layout: wgpu::BindGroupLayout,
    /// The filter record base for the current layer.
    pub slot_layout: wgpu::BindGroupLayout,
    /// Source snapshot texture and sampler.
    pub texture_layout: wgpu::BindGroupLayout,
    /// Dynamic-uniform alignment in bytes.
    pub slot_stride: u32,
}

impl BackdropPipeline {
    /// Build the backdrop-filter pipeline.
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wgpui backdrop blur"),
            source: wgpu::ShaderSource::Wgsl(super::shaders::BACKDROP_BLUR_WGSL.into()),
        });
        let frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("backdrop filters frame"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT, 16, false),
                storage_entry(1, wgpu::ShaderStages::VERTEX_FRAGMENT),
            ],
        });
        let slot_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("backdrop filters slot base"),
            entries: &[uniform_entry(
                0,
                wgpu::ShaderStages::VERTEX_FRAGMENT,
                16,
                true,
            )],
        });
        let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("backdrop filters texture"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("backdrop filters"),
            bind_group_layouts: &[
                Some(&frame_layout),
                Some(&slot_layout),
                Some(&texture_layout),
            ],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("backdrop filters"),
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
        Self {
            pipeline,
            frame_layout,
            slot_layout,
            texture_layout,
            slot_stride: device.limits().min_uniform_buffer_offset_alignment.max(16),
        }
    }

    /// Bind globals and the filter arena.
    pub fn frame_bind_group(
        &self,
        device: &wgpu::Device,
        globals: &wgpu::Buffer,
        arena: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("backdrop filters frame"),
            layout: &self.frame_layout,
            entries: &[buffer_entry(0, globals), buffer_entry(1, arena)],
        })
    }

    /// Bind the current framebuffer snapshot.
    pub fn texture_bind_group(
        &self,
        device: &wgpu::Device,
        view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("backdrop filters texture"),
            layout: &self.texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        })
    }
}

impl UnderlinePipeline {
    /// Build the pipeline. Once per device, never per frame.
    pub fn new(device: &wgpu::Device) -> UnderlinePipeline {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wgpui underlines"),
            source: wgpu::ShaderSource::Wgsl(super::shaders::UNDERLINES_WGSL.into()),
        });
        let frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("underlines frame"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT, 16, false),
                storage_entry(1, wgpu::ShaderStages::VERTEX_FRAGMENT),
                storage_entry(2, wgpu::ShaderStages::VERTEX),
            ],
        });
        let slot_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("underlines slot base"),
            entries: &[uniform_entry(0, wgpu::ShaderStages::VERTEX, 16, true)],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("underlines"),
            bind_group_layouts: &[Some(&frame_layout), Some(&slot_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("underlines"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vertex_main"),
                compilation_options: Default::default(),
                // §1's rule, unchanged for this kind too.
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
        UnderlinePipeline {
            pipeline,
            frame_layout,
            slot_layout,
            slot_stride: device.limits().min_uniform_buffer_offset_alignment.max(16),
        }
    }

    /// The bind group holding this frame's globals, underline arena, and
    /// indirection buffer.
    pub fn frame_bind_group(
        &self,
        device: &wgpu::Device,
        globals: &wgpu::Buffer,
        arena: &wgpu::Buffer,
        visible: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("underlines frame"),
            layout: &self.frame_layout,
            entries: &[
                buffer_entry(0, globals),
                buffer_entry(1, arena),
                buffer_entry(2, visible),
            ],
        })
    }

    /// The bind group over a slot-base buffer.
    pub fn slot_bind_group(&self, device: &wgpu::Device, bases: &wgpu::Buffer) -> wgpu::BindGroup {
        slot_base_bind_group(device, &self.slot_layout, bases)
    }

    /// Bytes one underline occupies in the arena, restated from `wgpui-core` so
    /// a drift between the shader's `UnderlineSlot` and the protocol's
    /// `Underline` fails a test rather than rendering garbage.
    pub const fn arena_slot_stride() -> usize {
        wgpui_core::patch::primitive::Underline::SLOT_STRIDE
    }
}

/// A bind group over a slot-base buffer, read one slot at a time through a
/// dynamic offset.
///
/// Shared by every instanced pipeline rather than written per kind: the slot
/// base is `wgpui_core::indirect`'s, not any one shader's, and
/// [`crate::render::draw::SlotBasePlan`] builds one of these whichever pipeline
/// is going to consume it.
pub fn slot_base_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    bases: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("slot base"),
        layout,
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

/// The instanced monochrome-sprite pipeline: one glyph per instance, its
/// coverage mask read out of one atlas page.
///
/// Bind groups, and why there are three rather than [`QuadPipeline`]'s two:
///
/// 0. **frame** — globals, the `GlyphRun` arena, the indirection buffer. Exactly
///    [`QuadPipeline`]'s, over a different arena.
/// 1. **slot** — the per-slot base index, by dynamic offset. Identical, and
///    built by the same [`slot_base_bind_group`].
/// 2. **page** — which atlas page is bound, and the page's texture. The one
///    thing a quad has no equivalent of, and the reason
///    [`crate::render::draw::issue_glyphs`] has an outer loop that
///    [`crate::render::draw::issue_instanced`] does not.
///
/// **Monochrome pages only.** A colour glyph's raster is an `Rgba8Unorm` page
/// and this shader reads a single coverage channel; binding a colour page here
/// would sample its red channel and paint nonsense. Colour glyphs are the
/// deferred `poly_sprites` work — see this module's doc.
pub struct MonoSpritePipeline {
    /// The pipeline itself.
    pub pipeline: wgpu::RenderPipeline,
    /// Globals, glyph arena, and indirection buffer.
    pub frame_layout: wgpu::BindGroupLayout,
    /// The per-slot base index, addressed by dynamic offset.
    pub slot_layout: wgpu::BindGroupLayout,
    /// The bound page's index and its texture.
    pub page_layout: wgpu::BindGroupLayout,
    /// Byte stride between two slots' entries in the slot-base buffer.
    pub slot_stride: u32,
}

impl MonoSpritePipeline {
    /// Bytes one atlas-page parameter block occupies: one `vec4<u32>`, for the
    /// alignment reason `mono_sprites.wgsl` records against `SlotBase`.
    pub const PAGE_PARAMS_SIZE: u64 = 16;

    /// Build the pipeline. Once per device, never per frame.
    pub fn new(device: &wgpu::Device) -> MonoSpritePipeline {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wgpui mono sprites"),
            source: wgpu::ShaderSource::Wgsl(super::shaders::MONO_SPRITES_WGSL.into()),
        });
        let frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mono sprites frame"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT, 16, false),
                storage_entry(1, wgpu::ShaderStages::VERTEX_FRAGMENT),
                storage_entry(2, wgpu::ShaderStages::VERTEX),
            ],
        });
        let slot_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mono sprites slot base"),
            entries: &[uniform_entry(0, wgpu::ShaderStages::VERTEX, 16, true)],
        });
        let page_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mono sprites atlas page"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::VERTEX, Self::PAGE_PARAMS_SIZE, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        // Not filterable, and no sampler at all: the shader
                        // reads texels at integer addresses. See
                        // `shaders/mono_sprites.wgsl`.
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mono sprites"),
            bind_group_layouts: &[Some(&frame_layout), Some(&slot_layout), Some(&page_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mono sprites"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vertex_main"),
                compilation_options: Default::default(),
                // §1's rule, unchanged for this kind too.
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
        MonoSpritePipeline {
            pipeline,
            frame_layout,
            slot_layout,
            page_layout,
            slot_stride: device.limits().min_uniform_buffer_offset_alignment.max(16),
        }
    }

    /// The bind group holding this frame's globals, glyph arena, and
    /// indirection buffer.
    pub fn frame_bind_group(
        &self,
        device: &wgpu::Device,
        globals: &wgpu::Buffer,
        arena: &wgpu::Buffer,
        visible: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mono sprites frame"),
            layout: &self.frame_layout,
            entries: &[
                buffer_entry(0, globals),
                buffer_entry(1, arena),
                buffer_entry(2, visible),
            ],
        })
    }

    /// The bind group over a slot-base buffer.
    pub fn slot_bind_group(&self, device: &wgpu::Device, bases: &wgpu::Buffer) -> wgpu::BindGroup {
        slot_base_bind_group(device, &self.slot_layout, bases)
    }

    /// One atlas page's bind group: which page it is, and its texture.
    pub fn page_bind_group(
        &self,
        device: &wgpu::Device,
        params: &wgpu::Buffer,
        view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mono sprites atlas page"),
            layout: &self.page_layout,
            entries: &[
                buffer_entry(0, params),
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(view),
                },
            ],
        })
    }

    /// Bytes one glyph occupies in the arena, restated from `wgpui-core` so a
    /// drift between the shader's `GlyphSlot` and the protocol's `Glyph` fails a
    /// test rather than rendering garbage.
    pub const fn arena_slot_stride() -> usize {
        wgpui_core::patch::primitive::GlyphRun::SLOT_STRIDE
    }
}

/// The instanced polychrome-sprite pipeline: one image per instance, its colour
/// bitmap read out of one atlas page.
///
/// [`MonoSpritePipeline`]'s bind-group shape exactly — frame, slot, page — over
/// the `PolySprite` arena and an `Rgba8Unorm` page. That it *is* exactly the
/// same shape is the finding, not a coincidence to tidy away later: the third
/// instanced pipeline needed a shader and a layout and no new mechanism, which
/// is what `render/draw.rs` was claiming when it said nothing there is written
/// per kind.
///
/// **Colour pages only**, and the mirror image of the mono pipeline's rule: this
/// shader reads four channels and binding a coverage page here would sample its
/// single channel as RGBA and paint nonsense. [`crate::render::frame`] selects
/// pages by [`wgpui_core::scene::atlas::AtlasKind`] for exactly this reason.
pub struct PolySpritePipeline {
    /// The pipeline itself.
    pub pipeline: wgpu::RenderPipeline,
    /// Globals, sprite arena, and indirection buffer.
    pub frame_layout: wgpu::BindGroupLayout,
    /// The per-slot base index, addressed by dynamic offset.
    pub slot_layout: wgpu::BindGroupLayout,
    /// The bound page's index and its texture.
    pub page_layout: wgpu::BindGroupLayout,
    /// Byte stride between two slots' entries in the slot-base buffer.
    pub slot_stride: u32,
    /// Linear sampler used only for sprites whose drawn size differs from the
    /// resident bitmap size.
    pub sampler: wgpu::Sampler,
}

impl PolySpritePipeline {
    /// Bytes one atlas-page parameter block occupies: one `vec4<u32>`, the same
    /// block [`MonoSpritePipeline`] uses and for the same alignment reason.
    pub const PAGE_PARAMS_SIZE: u64 = MonoSpritePipeline::PAGE_PARAMS_SIZE;

    /// Build the pipeline. Once per device, never per frame.
    pub fn new(device: &wgpu::Device) -> PolySpritePipeline {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wgpui poly sprites"),
            source: wgpu::ShaderSource::Wgsl(super::shaders::POLY_SPRITES_WGSL.into()),
        });
        let frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("poly sprites frame"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT, 16, false),
                storage_entry(1, wgpu::ShaderStages::VERTEX_FRAGMENT),
                storage_entry(2, wgpu::ShaderStages::VERTEX),
            ],
        });
        let slot_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("poly sprites slot base"),
            entries: &[uniform_entry(0, wgpu::ShaderStages::VERTEX, 16, true)],
        });
        let page_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("poly sprites atlas page"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::VERTEX, Self::PAGE_PARAMS_SIZE, false),
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
            label: Some("poly sprites"),
            bind_group_layouts: &[Some(&frame_layout), Some(&slot_layout), Some(&page_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("poly sprites"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vertex_main"),
                compilation_options: Default::default(),
                // §1's rule, unchanged for this kind too.
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
        PolySpritePipeline {
            pipeline,
            frame_layout,
            slot_layout,
            page_layout,
            slot_stride: device.limits().min_uniform_buffer_offset_alignment.max(16),
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("poly sprite filtering"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
        }
    }

    /// The bind group holding this frame's globals, sprite arena, and
    /// indirection buffer.
    pub fn frame_bind_group(
        &self,
        device: &wgpu::Device,
        globals: &wgpu::Buffer,
        arena: &wgpu::Buffer,
        visible: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("poly sprites frame"),
            layout: &self.frame_layout,
            entries: &[
                buffer_entry(0, globals),
                buffer_entry(1, arena),
                buffer_entry(2, visible),
            ],
        })
    }

    /// The bind group over a slot-base buffer.
    pub fn slot_bind_group(&self, device: &wgpu::Device, bases: &wgpu::Buffer) -> wgpu::BindGroup {
        slot_base_bind_group(device, &self.slot_layout, bases)
    }

    /// One atlas page's bind group: which page it is, and its texture.
    pub fn page_bind_group(
        &self,
        device: &wgpu::Device,
        params: &wgpu::Buffer,
        view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("poly sprites atlas page"),
            layout: &self.page_layout,
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

    /// Bytes one sprite occupies in the arena, restated from `wgpui-core` so a
    /// drift between the shader's `SpriteSlot` and the protocol's `PolySprite`
    /// fails a test rather than rendering garbage.
    pub const fn arena_slot_stride() -> usize {
        wgpui_core::patch::primitive::PolySprite::SLOT_STRIDE
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
            entries: &[uniform_entry(
                0,
                wgpu::ShaderStages::VERTEX_FRAGMENT,
                16,
                false,
            )],
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
    pub fn frame_bind_group(
        &self,
        device: &wgpu::Device,
        globals: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
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
        // `quads.wgsl`'s `QuadSlot` is nine `vec4<f32>` since the material path
        // `Quad` to per-corner radii and per-side border widths. If
        // `Quad::SLOT_STRIDE` ever changes without the shader changing with it,
        // every instance past the first reads the wrong bytes and renders
        // plausible garbage — so the agreement is asserted rather than left to
        // a comment.
        assert_eq!(QuadPipeline::arena_slot_stride(), 9 * 16);
        let shader = super::super::shaders::QUADS_WGSL;
        assert!(shader.contains("struct QuadSlot"));
        assert!(shader.contains("material_kind: vec4<u32>"));
        for field in [
            "origin_size",
            "background",
            "border_color",
            "corner_radii",
            "border_widths",
            "material_kind",
            "material_first",
            "material_second",
            "material_parameters",
        ] {
            assert!(shader.contains(field), "the shader dropped `{field}`");
        }
    }

    #[test]
    fn the_shader_agrees_with_the_protocol_about_a_shadow_slot() {
        // `QuadSlot`'s drift hazard, one kind over.
        assert_eq!(ShadowPipeline::arena_slot_stride(), 4 * 16);
        let shader = super::super::shaders::SHADOWS_WGSL;
        assert!(shader.contains("struct ShadowSlot"));
        for field in ["origin_size", "color", "corner_radii", "blur"] {
            assert!(shader.contains(field), "the shader dropped `{field}`");
        }
    }

    #[test]
    fn the_shadow_shaders_blur_margin_is_the_protocols_own() {
        // Transcribed into WGSL, where nothing checks it against the Rust
        // constant. If the two ever disagree the ordering pass reserves a
        // different rectangle from the one the strip covers, and the outermost
        // band of the falloff is clipped by a triangle edge — a subtly wrong
        // shadow, with no error anywhere.
        assert_eq!(
            wgpui_core::patch::primitive::Shadow::BLUR_MARGIN_SIGMAS,
            3.0
        );
        assert!(
            super::super::shaders::SHADOWS_WGSL.contains("const BLUR_MARGIN_SIGMAS: f32 = 3.0;")
        );
    }

    #[test]
    fn the_shader_agrees_with_the_protocol_about_an_underline_slot() {
        assert_eq!(UnderlinePipeline::arena_slot_stride(), 48);
        let shader = super::super::shaders::UNDERLINES_WGSL;
        assert!(shader.contains("struct UnderlineSlot"));
        for field in ["origin_size", "color", "thickness", "wavy"] {
            assert!(shader.contains(field), "the shader dropped `{field}`");
        }
        // `wavy` must be a real `u32` member. Packed into a trailing
        // `vec4<f32>` and bit-cast, the word `1` is a denormal `f32` and a GPU
        // that flushes denormals on load would draw every wavy underline
        // straight — a wrong picture with no error anywhere.
        assert!(
            shader.contains("wavy: u32,"),
            "the wavy flag must be declared as a word, not bit-cast out of an f32"
        );
    }

    #[test]
    fn the_composite_parameter_block_is_three_vec4s() {
        assert_eq!(CompositePipeline::PARAMS_SIZE, 3 * 16);
    }

    #[test]
    fn the_shader_agrees_with_the_protocol_about_a_glyph_slot() {
        // The same drift hazard `QuadSlot` has, and a worse one: `GlyphSlot`'s
        // colour starts at byte 28, which is not a `vec4<f32>` alignment, so the
        // shader spells it as four scalars. If the stride or the field order
        // moves without the shader moving with it, every glyph past the first
        // reads a neighbour's bytes.
        assert_eq!(MonoSpritePipeline::arena_slot_stride(), 48);
        let shader = super::super::shaders::MONO_SPRITES_WGSL;
        assert!(shader.contains("struct GlyphSlot"));
        for field in ["color_r", "color_g", "color_b", "color_a", "atlas_tile"] {
            assert!(shader.contains(field), "the shader dropped `{field}`");
        }
    }

    #[test]
    fn the_shader_agrees_with_the_protocol_about_a_sprite_slot() {
        // The same drift hazard `QuadSlot` and `GlyphSlot` have. `SpriteSlot`
        // is the one of the three whose members all land on their natural
        // alignment, so it can be spelled the obvious way — which makes it
        // *more* important to assert, not less: nothing about the spelling
        // would look wrong if the encoder's field order moved.
        assert_eq!(PolySpritePipeline::arena_slot_stride(), 48);
        let shader = super::super::shaders::POLY_SPRITES_WGSL;
        assert!(shader.contains("struct SpriteSlot"));
        for field in [
            "origin",
            "size",
            "atlas_origin",
            "atlas_size",
            "corner_radius",
            "opacity",
            "grayscale",
            "atlas_tile",
        ] {
            assert!(shader.contains(field), "the shader dropped `{field}`");
        }
    }

    #[test]
    fn both_sprite_pipelines_transcribe_the_same_tile_packing() {
        // Two shaders now hard-code `AtlasTileId`'s page/slot split, and WGSL
        // checks neither against the Rust constant. A shader that disagreed
        // would filter its sprites onto the wrong page and silently draw
        // nothing — the failure mode with no error anywhere.
        for shader in [
            super::super::shaders::MONO_SPRITES_WGSL,
            super::super::shaders::POLY_SPRITES_WGSL,
        ] {
            assert!(shader.contains("const UNUSED_INSTANCE: u32 = 0xffffffffu;"));
            assert!(shader.contains("const NO_TILE: u32 = 0xffffffffu;"));
            assert!(shader.contains("const TILE_SLOT_BITS: u32 = 24u;"));
        }
    }

    #[test]
    fn the_shader_uses_the_protocols_own_sentinel_values() {
        // Three constants live in `wgpui-core` and are transcribed into WGSL,
        // where nothing can check them at compile time. `AtlasTileId::NONE` and
        // `UNUSED_INSTANCE` are both `u32::MAX`, and the tile's page/slot split
        // is 24 bits — a shader that disagreed about the split would filter
        // glyphs onto the wrong page and silently draw nothing.
        let shader = super::super::shaders::MONO_SPRITES_WGSL;
        assert_eq!(wgpui_core::indirect::UNUSED_INSTANCE, u32::MAX);
        assert!(wgpui_core::patch::primitive::AtlasTileId::NONE.is_none());
        assert!(shader.contains("const UNUSED_INSTANCE: u32 = 0xffffffffu;"));
        assert!(shader.contains("const NO_TILE: u32 = 0xffffffffu;"));
        assert!(shader.contains("const TILE_SLOT_BITS: u32 = 24u;"));
        // And the split the shader hard-codes is the one the protocol packs: a
        // tile in page 5 must read back as page 5 after a 24-bit shift.
        let tile = wgpui_core::patch::primitive::AtlasTileId::new(5, 1)
            .expect("page 5 slot 1 is in range");
        assert_eq!(tile.as_raw() >> 24, 5);
        assert_eq!(tile.page(), Some(5));
    }
}
