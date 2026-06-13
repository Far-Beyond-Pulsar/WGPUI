//! Render a GPU scene and composite a headless GPUI overlay on top.
//!
//! Run with:
//! `cargo run --example headless_gpui_overlay`

#[path = "../prelude.rs"]
mod example_prelude;

use anyhow::{Context as _, Result};
use bytemuck::{Pod, Zeroable};
use example_prelude::init_example;
use gpui::{
    App, Application, Bounds, Context, Hsla, QuitMode, Render, WgpuOutputHandle, WgpuSurfaceHandle,
    Window, WindowBackgroundAppearance, WindowBounds, WindowOptions, div, prelude::*, px, rgb,
    size, transparent_black, wgpu_surface,
};
use std::{
    sync::{Arc, mpsc},
    time::Instant,
};
use wgpu::util::DeviceExt;

const SURFACE_WIDTH: u32 = 1280;
const SURFACE_HEIGHT: u32 = 720;
const SURFACE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

const SCENE_SHADER: &str = r#"
struct Uniforms {
    mvp: mat4x4<f32>,
    tint: vec4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.clip_position = uniforms.mvp * vec4<f32>(input.position, 1.0);
    output.color = input.color * uniforms.tint.rgb;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(input.color, uniforms.tint.a);
}
"#;

const OVERLAY_SHADER: &str = r#"
@group(0) @binding(0) var overlay_texture: texture_2d<f32>;
@group(0) @binding(1) var overlay_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(-1.0, 3.0),
        vec2<f32>(3.0, -1.0),
    );
    let uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, -1.0),
        vec2<f32>(2.0, 1.0),
    );

    var output: VertexOutput;
    output.clip_position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    output.uv = uvs[vertex_index];
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(overlay_texture, overlay_sampler, input.uv);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

impl Vertex {
    fn layout<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SceneUniform {
    mvp: [[f32; 4]; 4],
    tint: [f32; 4],
}

#[rustfmt::skip]
const CUBE_VERTICES: &[Vertex] = &[
    Vertex { position: [-0.5, -0.5,  0.5], color: [0.90, 0.20, 0.20] }, Vertex { position: [ 0.5, -0.5,  0.5], color: [0.90, 0.20, 0.20] },
    Vertex { position: [ 0.5,  0.5,  0.5], color: [1.00, 0.50, 0.50] }, Vertex { position: [-0.5,  0.5,  0.5], color: [1.00, 0.50, 0.50] },
    Vertex { position: [ 0.5, -0.5, -0.5], color: [0.20, 0.80, 0.20] }, Vertex { position: [-0.5, -0.5, -0.5], color: [0.20, 0.80, 0.20] },
    Vertex { position: [-0.5,  0.5, -0.5], color: [0.50, 1.00, 0.50] }, Vertex { position: [ 0.5,  0.5, -0.5], color: [0.50, 1.00, 0.50] },
    Vertex { position: [-0.5, -0.5, -0.5], color: [0.20, 0.20, 0.90] }, Vertex { position: [-0.5, -0.5,  0.5], color: [0.20, 0.20, 0.90] },
    Vertex { position: [-0.5,  0.5,  0.5], color: [0.50, 0.50, 1.00] }, Vertex { position: [-0.5,  0.5, -0.5], color: [0.50, 0.50, 1.00] },
    Vertex { position: [ 0.5, -0.5,  0.5], color: [0.90, 0.90, 0.20] }, Vertex { position: [ 0.5, -0.5, -0.5], color: [0.90, 0.90, 0.20] },
    Vertex { position: [ 0.5,  0.5, -0.5], color: [1.00, 1.00, 0.50] }, Vertex { position: [ 0.5,  0.5,  0.5], color: [1.00, 1.00, 0.50] },
    Vertex { position: [-0.5,  0.5,  0.5], color: [0.20, 0.90, 0.90] }, Vertex { position: [ 0.5,  0.5,  0.5], color: [0.20, 0.90, 0.90] },
    Vertex { position: [ 0.5,  0.5, -0.5], color: [0.50, 1.00, 1.00] }, Vertex { position: [-0.5,  0.5, -0.5], color: [0.50, 1.00, 1.00] },
    Vertex { position: [-0.5, -0.5, -0.5], color: [0.90, 0.20, 0.90] }, Vertex { position: [ 0.5, -0.5, -0.5], color: [0.90, 0.20, 0.90] },
    Vertex { position: [ 0.5, -0.5,  0.5], color: [1.00, 0.50, 1.00] }, Vertex { position: [-0.5, -0.5,  0.5], color: [1.00, 0.50, 1.00] },
];

#[rustfmt::skip]
const CUBE_INDICES: &[u16] = &[
     0,  1,  2,   0,  2,  3,
     4,  5,  6,   4,  6,  7,
     8,  9, 10,   8, 10, 11,
    12, 13, 14,  12, 14, 15,
    16, 17, 18,  16, 18, 19,
    20, 21, 22,  20, 22, 23,
];

#[rustfmt::skip]
const PLANE_VERTICES: &[Vertex] = &[
    Vertex { position: [-1.0, 0.0, -1.0], color: [0.20, 0.24, 0.30] },
    Vertex { position: [ 1.0, 0.0, -1.0], color: [0.20, 0.24, 0.30] },
    Vertex { position: [ 1.0, 0.0,  1.0], color: [0.12, 0.14, 0.20] },
    Vertex { position: [-1.0, 0.0,  1.0], color: [0.12, 0.14, 0.20] },
];

#[rustfmt::skip]
const PLANE_INDICES: &[u16] = &[0, 1, 2, 0, 2, 3];

static DEBUG_FRAME_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DEBUG_OVERLAY_FRAME_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

struct HeadlessOverlay {
    start_time: Instant,
}

impl Render for HeadlessOverlay {
    fn render(&mut self, window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        let n = DEBUG_OVERLAY_FRAME_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if n % 60 == 0 {
            eprintln!("[overlay] render #{n}");
        }
        let elapsed_seconds = self.start_time.elapsed().as_secs_f32();
        let progress = ((elapsed_seconds * 0.85).sin() * 0.5 + 0.5).clamp(0.0, 1.0);
        let pulse = ((elapsed_seconds * 2.4).sin() * 0.5 + 0.5).clamp(0.0, 1.0);
        let progress_width = px(280.0 * progress.max(0.08));
        let pulse_color = Hsla::from(rgb(0x67d4ff)).opacity(0.35 + pulse * 0.55);

        window.request_animation_frame();

        div()
            .size_full()
            .bg(transparent_black())
            .text_color(rgb(0xf4fbff))
            .child(
                div()
                    .absolute()
                    .top(px(24.0))
                    .left(px(24.0))
                    .w(px(380.0))
                    .p_5()
                    .rounded_xl()
                    .border_1()
                    .border_color(Hsla::from(rgb(0xb9ddff)).opacity(0.24))
                    .bg(Hsla::from(rgb(0x081524)).opacity(0.74))
                    .child(
                        div()
                            .text_xl()
                            .font_weight(gpui::FontWeight::BOLD)
                            .child("Headless GPUI Overlay"),
                    )
                    .child(
                        div()
                            .mt_2()
                            .text_sm()
                            .text_color(Hsla::from(rgb(0xd3ebff)).opacity(0.94))
                            .child("This UI is rendered offscreen by a separate GPUI application and then composited over the scene as a GPU texture."),
                    )
                    .child(
                        div()
                            .mt_4()
                            .h(px(8.0))
                            .rounded_full()
                            .bg(Hsla::from(rgb(0x93d6ff)).opacity(0.18))
                            .child(
                                div()
                                    .h_full()
                                    .w(progress_width)
                                    .rounded_full()
                                    .bg(Hsla::from(rgb(0x59d4ff)).opacity(0.92)),
                            ),
                    )
                    .child(
                        div()
                            .mt_3()
                            .text_xs()
                            .text_color(rgb(0x8bc8ee))
                            .child("Semi-transparent cards make the scene remain visible underneath."),
                    ),
            )
            .child(
                div()
                    .absolute()
                    .right(px(28.0))
                    .bottom(px(28.0))
                    .w(px(280.0))
                    .p_4()
                    .rounded_xl()
                    .border_1()
                    .border_color(Hsla::from(rgb(0xffffff)).opacity(0.16))
                    .bg(Hsla::from(rgb(0x02060c)).opacity(0.64))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_2()
                            .child(div().size_3().rounded_full().bg(pulse_color))
                            .child(
                                div()
                                    .text_base()
                                    .font_weight(gpui::FontWeight::BOLD)
                                    .child("Overlay live"),
                            ),
                    )
                    .child(
                        div()
                            .mt_2()
                            .text_sm()
                            .text_color(Hsla::from(rgb(0xd5dbe7)).opacity(0.92))
                            .child("The scene is drawn into a `WgpuSurface`. This card comes from a headless window's `wgpu_output()`."),
                    ),
            )
            .child(
                div()
                    .absolute()
                    .left(px(24.0))
                    .right(px(24.0))
                    .bottom(px(26.0))
                    .h(px(1.0))
                    .bg(Hsla::from(rgb(0xdcefff)).opacity(0.14)),
            )
    }
}

struct SceneState {
    scene_pipeline: wgpu::RenderPipeline,
    overlay_pipeline: wgpu::RenderPipeline,
    cube_vertex_buffer: wgpu::Buffer,
    cube_index_buffer: wgpu::Buffer,
    plane_vertex_buffer: wgpu::Buffer,
    plane_index_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    scene_bind_group: wgpu::BindGroup,
    overlay_bind_group_layout: wgpu::BindGroupLayout,
    overlay_sampler: wgpu::Sampler,
    depth_view: wgpu::TextureView,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    start_time: Instant,
    width: u32,
    height: u32,
}

impl SceneState {
    fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        width: u32,
        height: u32,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        let scene_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("headless_overlay_scene_shader"),
            source: wgpu::ShaderSource::Wgsl(SCENE_SHADER.into()),
        });
        let overlay_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("headless_overlay_composite_shader"),
            source: wgpu::ShaderSource::Wgsl(OVERLAY_SHADER.into()),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("headless_overlay_uniforms"),
            size: std::mem::size_of::<SceneUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let scene_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("headless_overlay_scene_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("headless_overlay_scene_bind_group"),
            layout: &scene_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let scene_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("headless_overlay_scene_pipeline_layout"),
                bind_group_layouts: &[Some(&scene_bind_group_layout)],
                immediate_size: 0,
            });

        let scene_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("headless_overlay_scene_pipeline"),
            layout: Some(&scene_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &scene_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &scene_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let overlay_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("headless_overlay_composite_bind_group_layout"),
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

        let overlay_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("headless_overlay_composite_pipeline_layout"),
                bind_group_layouts: &[Some(&overlay_bind_group_layout)],
                immediate_size: 0,
            });

        let overlay_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("headless_overlay_composite_pipeline"),
            layout: Some(&overlay_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &overlay_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &overlay_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let overlay_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("headless_overlay_composite_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let cube_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("headless_overlay_cube_vertex_buffer"),
            contents: bytemuck::cast_slice(CUBE_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let cube_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("headless_overlay_cube_index_buffer"),
            contents: bytemuck::cast_slice(CUBE_INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });
        let plane_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("headless_overlay_plane_vertex_buffer"),
            contents: bytemuck::cast_slice(PLANE_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let plane_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("headless_overlay_plane_index_buffer"),
            contents: bytemuck::cast_slice(PLANE_INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        let depth_view = Self::create_depth_view(&device, width, height);

        Self {
            scene_pipeline,
            overlay_pipeline,
            cube_vertex_buffer,
            cube_index_buffer,
            plane_vertex_buffer,
            plane_index_buffer,
            uniform_buffer,
            scene_bind_group,
            overlay_bind_group_layout,
            overlay_sampler,
            depth_view,
            device,
            queue,
            start_time: Instant::now(),
            width,
            height,
        }
    }

    fn create_depth_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("headless_overlay_depth_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.depth_view = Self::create_depth_view(&self.device, width, height);
    }

    fn draw_object(
        &self,
        render_pass: &mut wgpu::RenderPass<'_>,
        model: glam::Mat4,
        tint: [f32; 4],
        view_projection: glam::Mat4,
        vertex_buffer: &wgpu::Buffer,
        index_buffer: &wgpu::Buffer,
        index_count: u32,
    ) {
        let uniform = SceneUniform {
            mvp: (view_projection * model).to_cols_array_2d(),
            tint,
        };
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
        render_pass.set_bind_group(0, &self.scene_bind_group, &[]);
        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.draw_indexed(0..index_count, 0, 0..1);
    }

    fn render(
        &mut self,
        view: &wgpu::TextureView,
        overlay_view: Option<&wgpu::TextureView>,
    ) -> wgpu::SubmissionIndex {
        let elapsed_seconds = self.start_time.elapsed().as_secs_f32();
        let aspect = self.width as f32 / self.height.max(1) as f32;
        let projection =
            glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect, 0.1, 100.0);
        let camera_position = glam::Vec3::new(
            elapsed_seconds.cos() * 7.5,
            4.2,
            elapsed_seconds.sin() * 7.5,
        );
        let view_matrix = glam::Mat4::look_at_rh(
            camera_position,
            glam::Vec3::new(0.0, 1.0, 0.0),
            glam::Vec3::Y,
        );
        let view_projection = projection * view_matrix;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("headless_overlay_scene_encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("headless_overlay_scene_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.03,
                            g: 0.04,
                            b: 0.06,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            render_pass.set_pipeline(&self.scene_pipeline);

            let plane_model = glam::Mat4::from_scale(glam::Vec3::new(12.0, 1.0, 12.0));
            self.draw_object(
                &mut render_pass,
                plane_model,
                [0.8, 0.9, 1.0, 1.0],
                view_projection,
                &self.plane_vertex_buffer,
                &self.plane_index_buffer,
                PLANE_INDICES.len() as u32,
            );

            let cube_1 = glam::Mat4::from_translation(glam::Vec3::new(0.0, 1.15, 0.0))
                * glam::Mat4::from_rotation_y(elapsed_seconds * 1.1)
                * glam::Mat4::from_rotation_x(elapsed_seconds * 0.6)
                * glam::Mat4::from_scale(glam::Vec3::splat(1.7));
            self.draw_object(
                &mut render_pass,
                cube_1,
                [1.0, 0.92, 0.92, 1.0],
                view_projection,
                &self.cube_vertex_buffer,
                &self.cube_index_buffer,
                CUBE_INDICES.len() as u32,
            );

            let orbit_angle = elapsed_seconds * 0.8;
            let cube_2 = glam::Mat4::from_translation(glam::Vec3::new(
                orbit_angle.cos() * 3.1,
                0.85,
                orbit_angle.sin() * 3.1,
            )) * glam::Mat4::from_rotation_y(-elapsed_seconds * 1.7)
                * glam::Mat4::from_rotation_z(elapsed_seconds * 0.9)
                * glam::Mat4::from_scale(glam::Vec3::splat(0.95));
            self.draw_object(
                &mut render_pass,
                cube_2,
                [0.82, 0.96, 1.0, 1.0],
                view_projection,
                &self.cube_vertex_buffer,
                &self.cube_index_buffer,
                CUBE_INDICES.len() as u32,
            );

            let cube_3 = glam::Mat4::from_translation(glam::Vec3::new(
                (elapsed_seconds * 0.6).sin() * -3.7,
                1.0 + (elapsed_seconds * 1.5).sin().abs() * 0.45,
                -2.4,
            )) * glam::Mat4::from_rotation_x(elapsed_seconds * 1.1)
                * glam::Mat4::from_rotation_y(elapsed_seconds * 0.7)
                * glam::Mat4::from_scale(glam::Vec3::new(0.7, 1.8, 0.7));
            self.draw_object(
                &mut render_pass,
                cube_3,
                [0.92, 1.0, 0.88, 1.0],
                view_projection,
                &self.cube_vertex_buffer,
                &self.cube_index_buffer,
                CUBE_INDICES.len() as u32,
            );
        }

        if let Some(overlay_view) = overlay_view {
            let overlay_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("headless_overlay_composite_bind_group"),
                layout: &self.overlay_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(overlay_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.overlay_sampler),
                    },
                ],
            });

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("headless_overlay_composite_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            render_pass.set_pipeline(&self.overlay_pipeline);
            render_pass.set_bind_group(0, &overlay_bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()))
    }
}

struct HeadlessOverlayExample {
    surface: WgpuSurfaceHandle,
    overlay_output: WgpuOutputHandle,
    state: Option<SceneState>,
}

impl Render for HeadlessOverlayExample {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        if let Some((view, (width, height))) = self.surface.back_view_with_size() {
            let state = self.state.get_or_insert_with(|| {
                let device = Arc::new(self.surface.device().clone());
                let queue = Arc::new(self.surface.queue().clone());
                SceneState::new(device, queue, width, height, self.surface.format())
            });

            if state.width != width || state.height != height {
                state.resize(width, height);
            }

            let overlay_view = self.overlay_output.front_buffer_view();
            DEBUG_FRAME_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if DEBUG_FRAME_COUNT.load(std::sync::atomic::Ordering::Relaxed) % 60 == 1 {
                eprintln!(
                    "[main] overlay_view.is_some()={} has_pending_frame={} size={:?}",
                    overlay_view.is_some(),
                    self.overlay_output.has_pending_frame(),
                    self.overlay_output.size(),
                );
            }
            let submission_index = state.render(&view, overlay_view.as_ref());
            drop(overlay_view);
            drop(view);

            self.surface.present_synced(submission_index);
        }

        cx.notify();

        div()
            .w(px(SURFACE_WIDTH as f32))
            .h(px(SURFACE_HEIGHT as f32))
            .bg(rgb(0x05070a))
            .child(wgpu_surface(self.surface.clone()).absolute().inset_0())
    }
}

fn spawn_headless_overlay() -> Result<WgpuOutputHandle> {
    let (sender, receiver) = mpsc::sync_channel::<Result<WgpuOutputHandle>>(1);

    std::thread::Builder::new()
        .name("gpui-headless-overlay".to_string())
        .spawn(move || {
            Application::headless()
                .with_quit_mode(QuitMode::Explicit)
                .run(move |cx: &mut App| {
                    let result = (|| -> Result<WgpuOutputHandle> {
                        let bounds = Bounds::centered(
                            None,
                            size(px(SURFACE_WIDTH as f32), px(SURFACE_HEIGHT as f32)),
                            cx,
                        );

                        let window = cx.open_window(
                            WindowOptions {
                                window_bounds: Some(WindowBounds::Windowed(bounds)),
                                window_background: WindowBackgroundAppearance::Transparent,
                                ..Default::default()
                            },
                            |window: &mut Window, cx: &mut App| {
                                window.set_background_appearance(
                                    WindowBackgroundAppearance::Transparent,
                                );
                                cx.new(|_| HeadlessOverlay {
                                    start_time: Instant::now(),
                                })
                            },
                        )?;

                        window
                            .wgpu_output(cx)?
                            .context("headless window did not expose wgpu_output")
                    })();

                    if sender.send(result).is_err() {
                        cx.quit();
                    }
                });
        })
        .context("failed to spawn headless GPUI overlay thread")?;

    receiver
        .recv()
        .context("headless GPUI overlay thread terminated before producing output")?
}

fn main() -> Result<()> {
    let overlay_output = spawn_headless_overlay()?;

    Application::new().run(move |cx: &mut App| {
        init_example(cx, "Headless GPUI Overlay");

        let bounds = Bounds::centered(
            None,
            size(px(SURFACE_WIDTH as f32), px(SURFACE_HEIGHT as f32)),
            cx,
        );

        let overlay_output = overlay_output.clone();
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            move |window: &mut Window, cx: &mut App| {
                let surface = window
                    .create_wgpu_surface(SURFACE_WIDTH, SURFACE_HEIGHT, SURFACE_FORMAT)
                    .expect("WgpuSurface not supported on this platform");

                let overlay_output = overlay_output.clone();
                cx.new(|_| HeadlessOverlayExample {
                    surface,
                    overlay_output,
                    state: None,
                })
            },
        )
        .expect("failed to open window");
    });

    Ok(())
}
