//! Phase 6.4 GPU gates for Lyon paths and backdrop filters.

use lyon::math::point;
use lyon::path::Path as LyonPath;
use lyon::tessellation::{BuffersBuilder, FillOptions, FillTessellator, FillVertex, VertexBuffers};
use wgpui_core::geometry::Rect;
use wgpui_core::patch::RecordKey;
use wgpui_core::patch::apply::{ScenePatch, apply};
use wgpui_core::patch::primitive::{BackdropFilter, Path, Quad};
use wgpui_core::scene::Scene;
use wgpui_core::scene::layer::{BoundaryId, LayerKey};
use wgpui_wgpu::render::device::{ComputeContext, context_or_report};
use wgpui_wgpu::render::draw::DrawMode;
use wgpui_wgpu::render::frame::{
    Dirty, FrameError, FrameInput, FrameRenderer, OffscreenTarget, RenderTarget,
};
use wgpui_wgpu::render::pipelines::TARGET_FORMAT;
use wgpui_wgpu::render::readback::read_texture_rgba8;

const WIDTH: u32 = 96;
const HEIGHT: u32 = 64;
const UNCLIPPED: Rect = Rect {
    min_x: -100_000.0,
    min_y: -100_000.0,
    max_x: 100_000.0,
    max_y: 100_000.0,
};
const LEGACY_PATHS_WGSL: &str = include_str!("../../../old/src/platform/cross/shaders/paths.wgsl");
const LEGACY_BACKDROP_WGSL: &str =
    include_str!("../../../old/src/platform/cross/shaders/backdrop_blur.wgsl");

fn input(scene: &Scene, mode: DrawMode) -> FrameInput<'_> {
    FrameInput {
        scene,
        clip: UNCLIPPED,
        poison: &[],
        dirty: Dirty::All,
        uploads: &[],
        composites: &[],
        registry: None,
        atlas: None,
        viewport: [WIDTH as f32, HEIGHT as f32],
        mode,
    }
}

fn layer(scene: &mut Scene) -> wgpui_core::scene::layer::LayerId {
    scene.layer(LayerKey::untiled(BoundaryId::from_raw(1)))
}

fn read_pixel(pixels: &[u8], x: u32, y: u32) -> [u8; 4] {
    let offset = ((y * WIDTH + x) * 4) as usize;
    [
        pixels[offset],
        pixels[offset + 1],
        pixels[offset + 2],
        pixels[offset + 3],
    ]
}

fn assert_pixels_equal(left: &[u8], right: &[u8], label: &str) {
    assert_eq!(left.len(), right.len(), "{label}: readback lengths differ");
    for (index, (left_byte, right_byte)) in left.iter().zip(right).enumerate() {
        assert_eq!(
            left_byte,
            right_byte,
            "{label}: first mismatch at pixel ({}, {}) channel {}",
            (index / 4) as u32 % WIDTH,
            (index / 4) as u32 / WIDTH,
            index % 4,
        );
    }
}

fn lyon_triangle() -> Path {
    let mut builder = LyonPath::builder();
    builder.begin(point(8.0, 8.0));
    builder.line_to(point(40.0, 8.0));
    builder.line_to(point(8.0, 40.0));
    builder.close();
    let shape = builder.build();
    let mut buffers = VertexBuffers::new();
    let mut tessellator = FillTessellator::new();
    let result = tessellator.tessellate_path(
        &shape,
        &FillOptions::default(),
        &mut BuffersBuilder::new(&mut buffers, |vertex: FillVertex| vertex.position()),
    );
    assert!(result.is_ok(), "Lyon must tessellate the test path");
    Path::from_lyon_tessellation(buffers, [1.0, 0.0, 0.0, 1.0])
        .with_clip([0.0, 0.0], [WIDTH as f32, HEIGHT as f32])
}

fn path_scene(path: Path) -> Scene {
    let mut scene = Scene::new();
    let layer = layer(&mut scene);
    let mut patch = ScenePatch::new();
    patch.paths.append(layer, RecordKey::from_raw(1), 0, path);
    apply(&mut scene, &patch).expect("the Lyon path patch must apply");
    scene
}

fn buffer_with(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    usage: wgpu::BufferUsages,
    bytes: &[u8],
) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len() as u64,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytes);
    buffer
}

fn path_bytes(path: &Path) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(path.vertices.len() * Path::SLOT_STRIDE);
    for vertex in &path.vertices {
        for value in vertex.position {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in vertex.st {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let color: [f32; 4] = [0.0, 1.0, 0.5, 1.0];
        for value in color {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in path.clip_origin {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in path.clip_size {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

fn render_legacy_path(context: &ComputeContext, path: &Path) -> Vec<u8> {
    let device = &context.device;
    let queue = &context.queue;
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("legacy paths"),
        source: wgpu::ShaderSource::Wgsl(LEGACY_PATHS_WGSL.into()),
    });
    let mut globals = [0u8; 16];
    globals[0..4].copy_from_slice(&(WIDTH as f32).to_le_bytes());
    globals[4..8].copy_from_slice(&(HEIGHT as f32).to_le_bytes());
    let globals_buffer = buffer_with(
        device,
        queue,
        "legacy path globals",
        wgpu::BufferUsages::UNIFORM,
        &globals,
    );
    let vertex_buffer = buffer_with(
        device,
        queue,
        "legacy path vertices",
        wgpu::BufferUsages::STORAGE,
        &path_bytes(path),
    );
    let globals_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("legacy path globals"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: std::num::NonZeroU64::new(16),
            },
            count: None,
        }],
    });
    let vertices_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("legacy path vertices"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("legacy paths"),
        bind_group_layouts: &[Some(&globals_layout), Some(&vertices_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("legacy paths"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_path"),
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
            entry_point: Some("fs_path"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: TARGET_FORMAT,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: Default::default(),
        cache: None,
    });
    let globals_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("legacy path globals"),
        layout: &globals_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: globals_buffer.as_entire_binding(),
        }],
    });
    let vertices_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("legacy path vertices"),
        layout: &vertices_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: vertex_buffer.as_entire_binding(),
        }],
    });
    let target = OffscreenTarget::new(device, WIDTH, HEIGHT);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("legacy path frame"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("legacy path frame"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target.view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: Default::default(),
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &globals_group, &[]);
        pass.set_bind_group(1, &vertices_group, &[]);
        pass.draw(0..path.vertices.len() as u32, 0..1);
    }
    queue.submit(Some(encoder.finish()));
    target
        .read_pixels(device, queue)
        .expect("reading the legacy path target must succeed")
}

fn render_path(context: &ComputeContext, path: &Path) -> (Vec<u8>, u32) {
    let scene = path_scene(path.clone());
    let mut renderer = FrameRenderer::new(&context.device);
    let target = OffscreenTarget::new(&context.device, WIDTH, HEIGHT);
    let output = renderer
        .render_to(
            &context.device,
            &context.queue,
            &input(&scene, DrawMode::PerSlotIndirect),
            &target.target(),
        )
        .expect("the Lyon path must render");
    let pixels = target
        .read_pixels(&context.device, &context.queue)
        .expect("reading the Lyon path target must succeed");
    (pixels, output.stats.path_vertices_issued)
}

#[test]
fn lyon_path_matches_the_compiled_legacy_shader_on_a_real_adapter() {
    let Some(context) = context_or_report("phase 6.4 lyon path differential") else {
        return;
    };
    let path = lyon_triangle();
    let (ours, vertex_count) = render_path(&context, &path);
    let legacy = render_legacy_path(&context, &path);
    assert!(vertex_count >= 3, "Lyon must provide triangle vertices");
    assert_pixels_equal(&ours, &legacy, "Lyon path versus legacy path shader");
    assert_eq!(read_pixel(&ours, 12, 12), [255, 0, 0, 255]);
    assert_eq!(read_pixel(&ours, 48, 48), [0, 0, 0, 255]);
}

fn backdrop_filter() -> BackdropFilter {
    BackdropFilter {
        origin: [24.0, 16.0],
        size: [40.0, 32.0],
        clip_origin: [0.0, 0.0],
        clip_size: [WIDTH as f32, HEIGHT as f32],
        corner_radii: [4.0; 4],
        blur_radius: 4.0,
        opacity: 1.0,
    }
}

fn scene_with_backdrop(include_filter: bool) -> Scene {
    let mut scene = Scene::new();
    let layer = layer(&mut scene);
    let mut patch = ScenePatch::new();
    patch.quads.append(
        layer,
        RecordKey::from_raw(1),
        0,
        Quad {
            origin: [0.0, 0.0],
            size: [32.0, HEIGHT as f32],
            background: [1.0, 0.0, 0.0, 1.0],
            border_color: [0.0; 4],
            corner_radii: [0.0; 4],
            border_widths: [0.0; 4],
        },
    );
    if include_filter {
        patch
            .backdrop_filters
            .append(layer, RecordKey::from_raw(2), 0, backdrop_filter());
    }
    apply(&mut scene, &patch).expect("the backdrop patch must apply");
    scene
}

struct TestTarget {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl TestTarget {
    fn new(device: &wgpu::Device) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("phase 6.4 comparison target"),
            size: wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TARGET_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self { texture, view }
    }

    fn render_target(&self) -> RenderTarget<'_> {
        RenderTarget {
            view: &self.view,
            width: WIDTH,
            height: HEIGHT,
            clear: wgpu::Color::BLACK,
            source: Some(&self.texture),
        }
    }

    fn read_pixels(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<u8> {
        read_texture_rgba8(device, queue, &self.texture, WIDTH, HEIGHT)
            .expect("reading the comparison target must succeed")
    }
}

fn render_base(context: &ComputeContext, target: &TestTarget) {
    let scene = scene_with_backdrop(false);
    let mut renderer = FrameRenderer::new(&context.device);
    renderer
        .render_to(
            &context.device,
            &context.queue,
            &input(&scene, DrawMode::PerSlotIndirect),
            &target.render_target(),
        )
        .expect("the backdrop base quad must render");
}

fn backdrop_bytes(filter: BackdropFilter) -> [u8; 64] {
    let mut bytes = [0u8; 64];
    let mut offset = 8;
    let put_f32 = |bytes: &mut [u8; 64], offset: &mut usize, value: f32| {
        bytes[*offset..*offset + 4].copy_from_slice(&value.to_le_bytes());
        *offset += 4;
    };
    bytes[4..8].copy_from_slice(&filter.blur_radius.to_le_bytes());
    for value in filter.origin {
        put_f32(&mut bytes, &mut offset, value);
    }
    for value in filter.size {
        put_f32(&mut bytes, &mut offset, value);
    }
    for value in filter.clip_origin {
        put_f32(&mut bytes, &mut offset, value);
    }
    for value in filter.clip_size {
        put_f32(&mut bytes, &mut offset, value);
    }
    for value in filter.corner_radii {
        put_f32(&mut bytes, &mut offset, value);
    }
    put_f32(&mut bytes, &mut offset, filter.opacity);
    bytes
}

fn render_legacy_backdrop(context: &ComputeContext, target: &TestTarget) -> Vec<u8> {
    let device = &context.device;
    let queue = &context.queue;
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("legacy backdrop filter"),
        source: wgpu::ShaderSource::Wgsl(LEGACY_BACKDROP_WGSL.into()),
    });
    let mut globals = [0u8; 16];
    globals[0..4].copy_from_slice(&(WIDTH as f32).to_le_bytes());
    globals[4..8].copy_from_slice(&(HEIGHT as f32).to_le_bytes());
    let globals_buffer = buffer_with(
        device,
        queue,
        "legacy backdrop globals",
        wgpu::BufferUsages::UNIFORM,
        &globals,
    );
    let filter_buffer = buffer_with(
        device,
        queue,
        "legacy backdrop filters",
        wgpu::BufferUsages::STORAGE,
        &backdrop_bytes(backdrop_filter()),
    );
    let snapshot = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("legacy backdrop snapshot"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TARGET_FORMAT,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let snapshot_view = snapshot.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });
    let globals_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("legacy backdrop globals"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: std::num::NonZeroU64::new(16),
            },
            count: None,
        }],
    });
    let filter_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("legacy backdrop filters"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("legacy backdrop texture"),
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
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("legacy backdrop filter"),
        bind_group_layouts: &[
            Some(&globals_layout),
            Some(&filter_layout),
            Some(&texture_layout),
        ],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("legacy backdrop filter"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_backdrop_filter"),
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
            entry_point: Some("fs_backdrop_filter"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: TARGET_FORMAT,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: Default::default(),
        cache: None,
    });
    let globals_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("legacy backdrop globals"),
        layout: &globals_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: globals_buffer.as_entire_binding(),
        }],
    });
    let filter_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("legacy backdrop filters"),
        layout: &filter_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: filter_buffer.as_entire_binding(),
        }],
    });
    let texture_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("legacy backdrop texture"),
        layout: &texture_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&snapshot_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("legacy backdrop frame"),
    });
    encoder.copy_texture_to_texture(
        target.texture.as_image_copy(),
        snapshot.as_image_copy(),
        wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
    );
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("legacy backdrop frame"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target.view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: Default::default(),
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &globals_group, &[]);
        pass.set_bind_group(1, &filter_group, &[]);
        pass.set_bind_group(2, &texture_group, &[]);
        pass.draw(0..4, 0..1);
    }
    queue.submit(Some(encoder.finish()));
    target.read_pixels(device, queue)
}

#[test]
fn backdrop_filter_matches_the_compiled_legacy_shader_after_a_snapshot_copy() {
    let Some(context) = context_or_report("phase 6.4 backdrop differential") else {
        return;
    };
    let scene = scene_with_backdrop(true);
    let ours_target = TestTarget::new(&context.device);
    let mut renderer = FrameRenderer::new(&context.device);
    let output = renderer
        .render_to(
            &context.device,
            &context.queue,
            &input(&scene, DrawMode::PerSlotIndirect),
            &ours_target.render_target(),
        )
        .expect("the backdrop pass must render on a target with a source texture");
    let ours = ours_target.read_pixels(&context.device, &context.queue);

    let legacy_target = TestTarget::new(&context.device);
    render_base(&context, &legacy_target);
    let legacy = render_legacy_backdrop(&context, &legacy_target);

    assert_eq!(output.stats.backdrop_filters_drawn, 1);
    assert_pixels_equal(&ours, &legacy, "backdrop filter versus legacy shader");
    assert_ne!(read_pixel(&ours, 36, 32), [0, 0, 0, 255]);
    assert_eq!(read_pixel(&ours, 80, 32), [0, 0, 0, 255]);
}

#[test]
fn backdrop_gate_fails_without_a_copyable_source() {
    let Some(context) = context_or_report("phase 6.4 backdrop source gate") else {
        return;
    };
    let scene = scene_with_backdrop(true);
    let target = OffscreenTarget::new(&context.device, WIDTH, HEIGHT);
    let mut renderer = FrameRenderer::new(&context.device);
    let result = renderer.render_to(
        &context.device,
        &context.queue,
        &input(&scene, DrawMode::PerSlotIndirect),
        &RenderTarget {
            view: &target.view,
            width: target.width,
            height: target.height,
            clear: wgpu::Color::BLACK,
            source: None,
        },
    );
    assert!(matches!(result, Err(FrameError::BackdropSourceUnavailable)));
}

#[test]
fn legacy_sources_are_compiled_by_the_differential_tests() {
    assert!(LEGACY_PATHS_WGSL.contains("fn vs_path"));
    assert!(LEGACY_PATHS_WGSL.contains("fn fs_path"));
    assert!(LEGACY_BACKDROP_WGSL.contains("fn vs_backdrop_filter"));
    assert!(LEGACY_BACKDROP_WGSL.contains("fn fs_backdrop_filter"));
    assert!(LEGACY_BACKDROP_WGSL.contains("textureSampleLevel"));
    assert!(LEGACY_BACKDROP_WGSL.contains("quad_sdf"));
}
