use std::sync::Arc;

/// Example: WgpuSurface with a spinning cube
/// Demonstrates raw wgpu rendering within a gpui WgpuSurface — no third-party renderers.
use gpui::{
    App,
    Application,
    Context,
    Render,
    WgpuSurfaceHandle,
    Window,
    WindowOptions,
    div,
    prelude::*,
    rgb,
    wgpu_surface,
};
use wgpu::util::DeviceExt;

const SHADER: &str = r#"
struct Uniforms {
    mvp: mat4x4<f32>,
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
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.mvp * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
"#;

// 24 vertices: 4 per face x 6 faces. Each vertex: [x, y, z, r, g, b].
#[rustfmt::skip]
const VERTICES: &[[f32; 6]] = &[
    // Front  (+Z) — red
    [-0.5, -0.5,  0.5,  0.90, 0.20, 0.20], [ 0.5, -0.5,  0.5,  0.90, 0.20, 0.20],
    [ 0.5,  0.5,  0.5,  1.00, 0.50, 0.50], [-0.5,  0.5,  0.5,  1.00, 0.50, 0.50],
    // Back   (-Z) — green
    [ 0.5, -0.5, -0.5,  0.20, 0.80, 0.20], [-0.5, -0.5, -0.5,  0.20, 0.80, 0.20],
    [-0.5,  0.5, -0.5,  0.50, 1.00, 0.50], [ 0.5,  0.5, -0.5,  0.50, 1.00, 0.50],
    // Left   (-X) — blue
    [-0.5, -0.5, -0.5,  0.20, 0.20, 0.90], [-0.5, -0.5,  0.5,  0.20, 0.20, 0.90],
    [-0.5,  0.5,  0.5,  0.50, 0.50, 1.00], [-0.5,  0.5, -0.5,  0.50, 0.50, 1.00],
    // Right  (+X) — yellow
    [ 0.5, -0.5,  0.5,  0.90, 0.90, 0.20], [ 0.5, -0.5, -0.5,  0.90, 0.90, 0.20],
    [ 0.5,  0.5, -0.5,  1.00, 1.00, 0.50], [ 0.5,  0.5,  0.5,  1.00, 1.00, 0.50],
    // Top    (+Y) — cyan
    [-0.5,  0.5,  0.5,  0.20, 0.90, 0.90], [ 0.5,  0.5,  0.5,  0.20, 0.90, 0.90],
    [ 0.5,  0.5, -0.5,  0.50, 1.00, 1.00], [-0.5,  0.5, -0.5,  0.50, 1.00, 1.00],
    // Bottom (-Y) — magenta
    [-0.5, -0.5, -0.5,  0.90, 0.20, 0.90], [ 0.5, -0.5, -0.5,  0.90, 0.20, 0.90],
    [ 0.5, -0.5,  0.5,  1.00, 0.50, 1.00], [-0.5, -0.5,  0.5,  1.00, 0.50, 1.00],
];

#[rustfmt::skip]
const INDICES: &[u16] = &[
     0,  1,  2,   0,  2,  3,  // Front
     4,  5,  6,   4,  6,  7,  // Back
     8,  9, 10,   8, 10, 11,  // Left
    12, 13, 14,  12, 14, 15,  // Right
    16, 17, 18,  16, 18, 19,  // Top
    20, 21, 22,  20, 22, 23,  // Bottom
];

struct CubeRenderState {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    depth_view: wgpu::TextureView,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    start_time: std::time::Instant,
    width: u32,
    height: u32,
}

impl CubeRenderState {
    fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        width: u32,
        height: u32,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cube_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cube_uniforms"),
            size: 64, // mat4x4<f32> = 64 bytes
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cube_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cube_bg"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cube_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cube_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 24, // 6 x f32 = 24 bytes
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
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
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

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cube_vb"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cube_ib"),
            contents: bytemuck::cast_slice(INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        let depth_view = Self::make_depth_view(&device, width, height);

        Self {
            pipeline,
            vertex_buffer,
            index_buffer,
            uniform_buffer,
            bind_group,
            depth_view,
            device,
            queue,
            start_time: std::time::Instant::now(),
            width,
            height,
        }
    }

    fn make_depth_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cube_depth"),
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
        self.depth_view = Self::make_depth_view(&self.device, width, height);
    }

    fn render(&mut self, view: &wgpu::TextureView) {
        let t = self.start_time.elapsed().as_secs_f32();

        let aspect = self.width as f32 / self.height.max(1) as f32;
        let proj = glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect, 0.1, 100.0);
        let camera = glam::Mat4::look_at_rh(
            glam::Vec3::new(0.0, 0.9, 2.5),
            glam::Vec3::ZERO,
            glam::Vec3::Y,
        );
        let model = glam::Mat4::from_rotation_y(t * 1.1) * glam::Mat4::from_rotation_x(t * 0.65);
        let mvp: [[f32; 4]; 4] = (proj * camera * model).to_cols_array_2d();

        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&mvp));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cube_encoder"),
            });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("cube_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.05,
                            b: 0.08,
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
            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            rpass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            rpass.draw_indexed(0..INDICES.len() as u32, 0, 0..1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
    }
}

/// State for the spinning-cube example.
///
/// The cube is rendered directly inside `Render::render()` on the GPUI main thread.
/// This guarantees that the wgpu GPU commands for the cube are submitted *before*
/// the compositor's commands in the same frame, so the DX12 resource barriers
/// wgpu generates are always in the correct order — no cross-thread sync needed.
struct SurfaceExample {
    surface: WgpuSurfaceHandle,
    /// Lazily initialised once the surface has been laid out and sized.
    state: Option<CubeRenderState>,
    frame_count: u32,
    last_fps_update: std::time::Instant,
    display_fps: f64,
}

impl Render for SurfaceExample {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // Render the cube into the back buffer this frame.
        if let Some((view, (w, h))) = self.surface.back_view_with_size() {
            let state = self.state.get_or_insert_with(|| {
                let device = Arc::new(self.surface.device().clone());
                let queue = Arc::new(self.surface.queue().clone());
                let format = self.surface.format();
                CubeRenderState::new(device, queue, w, h, format)
            });

            if state.width != w || state.height != h {
                state.resize(w, h);
            }

            state.render(&view);
            drop(view);

            // Swap rendering→ready. The compositor (running right after this
            // method returns) calls swap_ready_display() and samples the texture
            // we just rendered into, within the same GPU submission sequence.
            self.surface.swap_buffers();

            self.frame_count = self.frame_count.wrapping_add(1);
            let now = std::time::Instant::now();
            if now.duration_since(self.last_fps_update) >= std::time::Duration::from_secs(1) {
                self.display_fps = self.frame_count as f64;
                self.frame_count = 0;
                self.last_fps_update = now;
            }
        }

        // Mark the entity dirty so GPUI keeps scheduling redraws at vsync rate.
        cx.notify();

        div()
            .w(gpui::px(800.0))
            .h(gpui::px(600.0))
            .bg(rgb(0x0d0d14))
            .child(wgpu_surface(self.surface.clone()).absolute().inset_0())
            .child(
                div()
                    .absolute()
                    .top(gpui::px(4.0))
                    .left(gpui::px(8.0))
                    .text_color(rgb(0x8888bb))
                    .child(format!("{:.0} fps", self.display_fps)),
            )
    }
}

fn main() {
    env_logger::init();
    Application::new().run(|cx: &mut App| {
        _ = cx.open_window(
            WindowOptions::default(),
            |window: &mut Window, cx: &mut App| {
                let surface = window
                    .create_wgpu_surface(800, 600, wgpu::TextureFormat::Rgba8UnormSrgb)
                    .expect("WgpuSurface not supported on this platform");

                cx.new(|_cx| SurfaceExample {
                    surface,
                    state: None,
                    frame_count: 0,
                    last_fps_update: std::time::Instant::now(),
                    display_fps: 0.0,
                })
            },
        );
    });
}
