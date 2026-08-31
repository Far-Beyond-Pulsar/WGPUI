//! A deliberately dense native WGPUI application built against the current
//! direct API.

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;

use wgpu::util::DeviceExt;

use wgpui::{
    ApplicationError, NativeApplication, Styled, SurfaceId, WgpuSurface, WgpuSurfaceHandle,
    WindowOptions, div, rgb,
};

fn main() -> Result<(), ApplicationError> {
    let selected = Rc::new(Cell::new(0_u32));
    let selected_for_button = Rc::clone(&selected);
    let inspected = Rc::new(Cell::new(false));
    let inspected_for_button = Rc::clone(&inspected);
    let hovered_control = Rc::new(Cell::new(0_u8));
    let hovered_control_for_first = Rc::clone(&hovered_control);
    let hovered_control_for_second = Rc::clone(&hovered_control);
    let scroll_offset = Rc::new(Cell::new(0.0_f32));
    let scroll_offset_for_handler = Rc::clone(&scroll_offset);
    let mut surface_demo = None;

    NativeApplication::new(WindowOptions::default(), move |window| {
        if surface_demo.is_none() {
            surface_demo = SurfaceDemo::new(window).ok();
        }
        if let Some(surface_demo) = surface_demo.as_mut() {
            surface_demo.render();
        }
        let surface_description = surface_demo.as_ref().map(SurfaceDemo::description);
        window.performance_debug().set_tile_refresh_flash(
            wgpui::TileRefreshFlash::enabled()
                .with_tile_size(256.0, 256.0)
                .with_color([1.0, 1.0, 0.0, 1.0]),
        );
        let _ = window.interaction();
        let selected = selected.get();
        let inspected = inspected.get();
        let hovered_control = hovered_control.get();
        let scroll_offset = scroll_offset.get();
        let button_color = if selected == 0 {
            rgb(0x2f6fed)
        } else {
            rgb(0x2459bd)
        };
        let first_button_color = if hovered_control == 1 {
            rgb(0x4f8dff)
        } else {
            button_color
        };
        let second_button_color = if hovered_control == 2 {
            rgb(0x273653)
        } else {
            rgb(0x171d29)
        };

        div()
            .id("application")
            .size_full()
            .min_h(0.0)
            .p_6()
            .flex()
            .flex_col()
            .gap_4()
            .bg(rgb(0x10141c))
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap_1()
                            .child(div().text_2xl().text_color(rgb(0xf4f7ff)).child("Command Center"))
                            .child(
                                div()
                                    .text_sm()
                                    .text_color(rgb(0x9ca9c2))
                                    .child("Retained GPU-native application overview"),
                            ),
                    )
                    .child(
                        div()
                            .px_3()
                            .py_1()
                            .rounded_lg()
                            .bg(rgb(0x183b2e))
                            .text_sm()
                            .text_color(rgb(0x71e0ad))
                            .child("ONLINE"),
                    ),
            )
            .child(
                div()
                    .flex()
                    .gap_3()
                    .child(stat_card("Frame time", "1.8 ms", rgb(0x20365f)))
                    .child(stat_card("GPU passes", "12", rgb(0x3d2d5d)))
                    .child(stat_card("Resident tiles", "248", rgb(0x244b43))),
            )
            .child(
                div()
                    .flex()
                    .gap_4()
                    .flex_1()
                    .min_h(0.0)
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap_3()
                            .justify_between()
                            .flex_1()
                            .min_h(0.0)
                            .p_4()
                            .rounded_lg()
                            .border_1()
                            .border_color(rgb(0x293348))
                            .bg(rgb(0x171d29))
                            .id("activity-scroll")
                            .boundary()
                            .overflow_y_scroll()
                            .scroll_offset([0.0, -scroll_offset])
                            .on_scroll({
                                let scroll_offset = Rc::clone(&scroll_offset_for_handler);
                                move |event, _, _| {
                                    scroll_offset.set(
                                        (scroll_offset.get() - event.delta[1])
                                            .clamp(0.0, 520.0),
                                    );
                                }
                            })
                            .child(div().text_lg().text_color(rgb(0xf4f7ff)).child("Recent activity"))
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(rgb(0xff73d9ff))
                                    .child("Scroll this panel to inspect retained activity content"),
                            )
                            .child(activity_row("GPU scene compacted", "just now", rgb(0x54d69b)))
                            .child(activity_row("Atlas page resident", "12 sec ago", rgb(0x5ca8ff)))
                            .child(activity_row("Tile boundary crossed", "48 sec ago", rgb(0xe6b85c)))
                            .child(activity_row("Surface synchronized", "2 min ago", rgb(0xb18cff)))
                            .child(activity_row("Indirect args rebuilt", "3 min ago", rgb(0xf28c68)))
                            .child(activity_row("Atlas upload delta", "4 min ago", rgb(0xd18cff)))
                            .child(activity_row("Occlusion pass complete", "5 min ago", rgb(0x62d4e8)))
                            .child(activity_row("Input region updated", "6 min ago", rgb(0xffc857)))
                            .child(activity_row("Surface presented", "7 min ago", rgb(0x8fd694)))
                            .child(activity_row("Retained node reused", "8 min ago", rgb(0x9ca9ff)))
                            .child(activity_row("Glyph page compacted", "9 min ago", rgb(0xe78ac3))),
                    )
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap_3()
                            .w(250.0)
                            .p_4()
                            .rounded_lg()
                            .border_1()
                            .border_color(rgb(0x293348))
                            .bg(rgb(0x171d29))
                            .child(div().text_lg().text_color(rgb(0xf4f7ff)).child("Controls"))
                            .child(
                                div()
                                    .h(44.0)
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .rounded_lg()
                                    .bg(first_button_color)
                                    .text_color(rgb(0xffffff))
                            .child(format!("Rebuild visible tiles ({selected})"))
                                    .on_click({
                                        let selected = Rc::clone(&selected_for_button);
                                        move |_, _, _| {
                                            selected.set(selected.get().wrapping_add(1));
                                        }
                                    })
                                    .on_hover({
                                        let hovered_control = Rc::clone(&hovered_control_for_first);
                                        move |is_hovered, _, _| {
                                            hovered_control.set(if is_hovered { 1 } else { 0 });
                                        }
                                    }),
                            )
                            .child(
                                div()
                                    .h(44.0)
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .rounded_lg()
                                    .border_1()
                                    .border_color(rgb(0x34415d))
                                    .bg(second_button_color)
                                    .text_color(rgb(0xb8c4dc))
                                    .child(if inspected {
                                        "Scene inspected"
                                    } else {
                                        "Inspect retained scene"
                                    })
                                    .on_click({
                                        let inspected = Rc::clone(&inspected_for_button);
                                        move |_, _, _| {
                                            inspected.set(!inspected.get());
                                        }
                                    })
                                    .on_hover({
                                        let hovered_control = Rc::clone(&hovered_control_for_second);
                                        move |is_hovered, _, _| {
                                            hovered_control.set(if is_hovered { 2 } else { 0 });
                                        }
                                    }),
                            )
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(rgb(0x7f8ba5))
                                    .child("Actions update retained state without rebuilding unchanged content."),
                            )
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(rgb(0x8291ad))
                                    .child("Live 3D surface"),
                            )
                            .when_some(surface_description, |this, description| {
                                this.child(description)
                            }),
                    ),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .p_3()
                    .rounded_lg()
                    .bg(rgb(0x1c2738))
                    .child(div().text_sm().text_color(rgb(0xcbd6ec)).child("All systems nominal"))
                    .child(div().text_xs().text_color(rgb(0x8291ad)).child("WGPUI 2.0 native backend")),
            )
    })
    .run()
}

const SURFACE_SHADER: &str = r#"
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
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.clip_position = uniforms.mvp * vec4<f32>(input.position, 1.0);
    output.color = input.color;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(input.color, 1.0);
}
"#;

#[rustfmt::skip]
const SURFACE_VERTICES: &[[f32; 6]] = &[
    [-0.5, -0.5,  0.5, 0.90, 0.20, 0.20], [ 0.5, -0.5,  0.5, 0.90, 0.20, 0.20],
    [ 0.5,  0.5,  0.5, 1.00, 0.50, 0.50], [-0.5,  0.5,  0.5, 1.00, 0.50, 0.50],
    [ 0.5, -0.5, -0.5, 0.20, 0.80, 0.20], [-0.5, -0.5, -0.5, 0.20, 0.80, 0.20],
    [-0.5,  0.5, -0.5, 0.50, 1.00, 0.50], [ 0.5,  0.5, -0.5, 0.50, 1.00, 0.50],
    [-0.5, -0.5, -0.5, 0.20, 0.20, 0.90], [-0.5, -0.5,  0.5, 0.20, 0.20, 0.90],
    [-0.5,  0.5,  0.5, 0.50, 0.50, 1.00], [-0.5,  0.5, -0.5, 0.50, 0.50, 1.00],
    [ 0.5, -0.5,  0.5, 0.90, 0.90, 0.20], [ 0.5, -0.5, -0.5, 0.90, 0.90, 0.20],
    [ 0.5,  0.5, -0.5, 1.00, 1.00, 0.50], [ 0.5,  0.5,  0.5, 1.00, 1.00, 0.50],
    [-0.5,  0.5,  0.5, 0.20, 0.90, 0.90], [ 0.5,  0.5,  0.5, 0.20, 0.90, 0.90],
    [ 0.5,  0.5, -0.5, 0.50, 1.00, 1.00], [-0.5,  0.5, -0.5, 0.50, 1.00, 1.00],
    [-0.5, -0.5, -0.5, 0.90, 0.20, 0.90], [ 0.5, -0.5, -0.5, 0.90, 0.20, 0.90],
    [ 0.5, -0.5,  0.5, 1.00, 0.50, 1.00], [-0.5, -0.5,  0.5, 1.00, 0.50, 1.00],
];

#[rustfmt::skip]
const SURFACE_INDICES: &[u16] = &[
     0,  1,  2,   0,  2,  3,  4,  5,  6,   4,  6,  7,
     8,  9, 10,   8, 10, 11, 12, 13, 14,  12, 14, 15,
    16, 17, 18,  16, 18, 19, 20, 21, 22,  20, 22, 23,
];

struct CubeRenderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    depth_view: wgpu::TextureView,
    width: u32,
    height: u32,
    started: Instant,
}

impl CubeRenderer {
    fn new(surface: &WgpuSurfaceHandle, width: u32, height: u32) -> Self {
        let device = Arc::new(surface.device().clone());
        let queue = Arc::new(surface.queue().clone());
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("complex app 3d surface shader"),
            source: wgpu::ShaderSource::Wgsl(SURFACE_SHADER.into()),
        });
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("complex app 3d surface uniforms"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("complex app 3d surface bindings"),
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
            label: Some("complex app 3d surface bind group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("complex app 3d surface pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("complex app 3d surface pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: 24,
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
                })],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface.format(),
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
            label: Some("complex app 3d surface vertices"),
            contents: bytemuck::cast_slice(SURFACE_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("complex app 3d surface indices"),
            contents: bytemuck::cast_slice(SURFACE_INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });
        let depth_view = Self::depth_view(&device, width, height);
        Self {
            device,
            queue,
            pipeline,
            vertex_buffer,
            index_buffer,
            uniform_buffer,
            bind_group,
            depth_view,
            width,
            height,
            started: Instant::now(),
        }
    }

    fn depth_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("complex app 3d surface depth"),
                size: wgpu::Extent3d {
                    width: width.max(1),
                    height: height.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.depth_view = Self::depth_view(&self.device, width, height);
    }

    fn render(&mut self, view: &wgpu::TextureView) {
        let elapsed = self.started.elapsed().as_secs_f32();
        let projection = glam::camera::rh::proj::directx::perspective(
            std::f32::consts::FRAC_PI_4,
            self.width as f32 / self.height.max(1) as f32,
            0.1,
            100.0,
        );
        let camera = glam::camera::rh::view::look_at_mat4(
            glam::Vec3::new(0.0, 0.8, 2.4),
            glam::Vec3::ZERO,
            glam::Vec3::Y,
        );
        let model = glam::Mat4::from_rotation_y(elapsed * 1.1)
            * glam::Mat4::from_rotation_x(elapsed * 0.65);
        let mvp = (projection * camera * model).to_cols_array_2d();
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&mvp));
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("complex app 3d surface encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("complex app 3d surface pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.03,
                            g: 0.04,
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
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            pass.draw_indexed(0..SURFACE_INDICES.len() as u32, 0, 0..1);
        }
        self.queue.submit(Some(encoder.finish()));
    }
}

struct SurfaceDemo {
    surface: WgpuSurfaceHandle,
    renderer: Option<CubeRenderer>,
}

impl SurfaceDemo {
    fn new(window: &mut wgpui::Window) -> Result<Self, wgpui::gpu::window::WindowError> {
        let surface =
            window.create_wgpu_surface(218, 140, wgpui::gpu::render::pipelines::TARGET_FORMAT)?;
        Ok(Self {
            surface,
            renderer: None,
        })
    }

    fn render(&mut self) {
        let Some((view, (width, height))) = self.surface.back_view_with_size() else {
            return;
        };
        let renderer = self
            .renderer
            .get_or_insert_with(|| CubeRenderer::new(&self.surface, width, height));
        if renderer.width != width || renderer.height != height {
            renderer.resize(width, height);
        }
        renderer.render(&view);
        self.surface.present();
    }

    fn description(&self) -> wgpui::Description {
        WgpuSurface::new(SurfaceId::from_raw(self.surface.id()))
            .bounds(wgpui::layout::taffy_tree::LayoutRect {
                x: 0.0,
                y: 0.0,
                width: 218.0,
                height: 140.0,
            })
            .style(wgpui::SurfaceStyle {
                corner_radius: 8.0,
                opacity: 1.0,
            })
            .describe()
    }
}

fn stat_card(label: &'static str, value: &'static str, color: wgpui::Rgba) -> wgpui::Div {
    div()
        .flex()
        .flex_col()
        .gap_1()
        .flex_1()
        .p_4()
        .rounded_lg()
        .bg(color)
        .child(div().text_xs().text_color(rgb(0xc0cbe0)).child(label))
        .child(div().text_2xl().text_color(rgb(0xffffff)).child(value))
}

fn activity_row(label: &'static str, time: &'static str, color: wgpui::Rgba) -> wgpui::Div {
    div()
        .flex()
        .flex_shrink_0()
        .items_center()
        .gap_2()
        .p_2()
        .rounded_md()
        .child(div().w(8.0).h(8.0).rounded_lg().bg(color))
        .child(
            div()
                .flex()
                .flex_col()
                .flex_1()
                .child(div().text_sm().text_color(rgb(0xdce5f5)).child(label))
                .child(div().text_xs().text_color(rgb(0x7f8ba5)).child(time)),
        )
}
