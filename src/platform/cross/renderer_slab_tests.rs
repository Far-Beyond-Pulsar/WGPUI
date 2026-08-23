//! Shader-validation and pixel-evidence tests for layer slabs (spec #94).
//!
//! The GPU-tier tests build the production `WgpuPipelines` against a headless
//! device and render the SAME layer recipe twice — once composited through
//! the legacy primitive arrays, once spliced from slab buffers through
//! `flush_slab_run`, the exact production draw function — then compare
//! readback bytes. A transform-offset variant proves the 64-byte uniform slot
//! moves a layer without any instance re-upload.

use super::*;
use crate::Bounds as SceneBounds;
use crate::Size as SceneSize;
use crate::{
    ContentMask, Corners, DevicePixels, Edges, Hsla, Point, ScaledPixels, px, point,
};
use crate::platform::cross::render_context::{WgpuContext, WgpuOptions};
use crate::scene::{PolychromeSprite, SceneBatch, SlabRun};
use crate::scene::{Path as ScenePath, Shadow as SceneShadow, Underline as SceneUnderline};

// ---------------------------------------------------------------------
// Naga validation.
// ---------------------------------------------------------------------

/// (label, transform bind-group position, shader body) for every pipeline
/// that can draw spliced slab content.
const SLAB_SHADERS: &[(&str, u32, &str)] = &[
    ("quads", 2, include_str!("shaders/quads.wgsl")),
    ("shadows", 2, include_str!("shaders/shadows.wgsl")),
    ("paths", 2, include_str!("shaders/paths.wgsl")),
    ("underlines", 2, include_str!("shaders/underlines.wgsl")),
    ("mono_sprites", 4, include_str!("shaders/mono_sprites.wgsl")),
    ("poly_sprites", 3, include_str!("shaders/poly_sprites.wgsl")),
];

#[test]
fn every_slab_shader_parses_and_validates_with_naga() {
    for (name, group, body) in SLAB_SHADERS {
        let source = slab_shader_source(name, *group, body).into_owned();
        let module = wgpu::naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|error| panic!("{name} failed to parse: {error:?}"));
        let mut validator = wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .unwrap_or_else(|error| panic!("{name} failed validation: {error:?}"));
    }
}

#[test]
fn vertex_stages_add_the_layer_translate_exactly_once() {
    for (name, group, body) in SLAB_SHADERS {
        let source = slab_shader_source(name, *group, body).into_owned();
        let vs_end = source.find("@fragment").expect("shader has a fragment stage");
        let vs_part = &source[..vs_end];
        let needle = match *name {
            // Paths index raw vertices rather than instances.
            "paths" => "v.xy_position + layer_transform.translate",
            _ => "position + layer_transform.translate",
        };
        assert_eq!(
            vs_part.matches(needle).count(),
            1,
            "{name}: the vertex stage must apply the layer translate exactly once"
        );
    }
}

#[test]
fn fragment_stages_undo_the_translate_exactly_once_where_world_space_is_reread() {
    // Shaders whose fragment stages compare interpolated position against
    // untranslated instance geometry must subtract the translate exactly once
    // via the shared helper. Stages that only consume varyings (mono sprites
    // sample tile_position; paths read clip distances computed in the vertex
    // stage) must not touch it at all — a second undo would double-shift.
    let expectations: &[(&str, usize)] = &[
        ("quads", 2),
        ("shadows", 1),
        ("underlines", 1),
        ("poly_sprites", 1),
        ("mono_sprites", 0),
        ("paths", 0),
    ];
    for (name, expected_calls) in expectations {
        let (_, group, body) = SLAB_SHADERS
            .iter()
            .find(|(label, _, _)| label == name)
            .expect("known shader");
        let source = slab_shader_source(name, *group, body).into_owned();
        let fs_part = &source[source.find("@fragment").expect("fragment stage exists")..];
        assert_eq!(
            fs_part.matches("layer_world_position(").count(),
            *expected_calls,
            "{name}: fragment-stage translate-undo count"
        );
    }
}

#[test]
fn raw_shader_files_stay_pristine_for_the_replay_path() {
    // flamegraph_replay renders these files against its own bind-group
    // layouts; none of them may reference the slab transform directly.
    for (name, _, body) in SLAB_SHADERS {
        assert!(
            !body.contains("layer_transform"),
            "{name}: raw shader must not reference the slab transform"
        );
    }
}

#[test]
fn wgsl_layer_transform_struct_is_64_bytes_like_gpu_layer_transform() {
    let prelude =
        include_str!("shaders/slab_transform.wgsl").replace("{SLAB_TRANSFORM_GROUP}", "2");
    let module = wgpu::naga::front::wgsl::parse_str(&prelude)
        .expect("the shared transform prelude must parse");
    let handle = module
        .types
        .iter()
        .find(|(_, ty)| ty.name.as_deref() == Some("LayerTransform"))
        .map(|(handle, _)| handle)
        .expect("LayerTransform type exists");
    match &module.types[handle].inner {
        wgpu::naga::TypeInner::Struct { span, .. } => assert_eq!(
            *span, 64,
            "WGSL LayerTransform span must equal the Rust slot size"
        ),
        other => panic!("LayerTransform is not a struct: {other:?}"),
    }
    assert_eq!(std::mem::size_of::<GpuLayerTransform>(), 64);
}

// ---------------------------------------------------------------------
// GPU pixel evidence.
// ---------------------------------------------------------------------

const WIDTH: u32 = 256;
const HEIGHT: u32 = 192;

fn headless_harness() -> Option<PixelHarness> {
    let context = Arc::new(WgpuContext::new(&WgpuOptions::default()).ok()?);
    let surface_configuration = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: wgpu::TextureFormat::Rgba8Unorm,
        width: WIDTH,
        height: HEIGHT,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        color_space: wgpu::SurfaceColorSpace::Auto,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    let pipelines = WgpuPipelines::new(context.as_ref(), &surface_configuration, 0);
    let atlas = Arc::new(crate::platform::cross::atlas::WgpuAtlas::new(context.clone()));
    let atlas_sampler = context.device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let buffers = slab_gpu::SlabGpuBuffers::new(
        &context.device,
        context.device.limits().min_uniform_buffer_offset_alignment,
    );
    Some(PixelHarness {
        context,
        pipelines,
        atlas,
        buffers,
        registry: SlabRegistry::new(),
        atlas_sampler,
    })
}

struct PixelHarness {
    context: Arc<WgpuContext>,
    pipelines: WgpuPipelines,
    atlas: Arc<crate::platform::cross::atlas::WgpuAtlas>,
    buffers: slab_gpu::SlabGpuBuffers,
    registry: SlabRegistry,
    atlas_sampler: wgpu::Sampler,
}

impl PixelHarness {
    fn globals(&self) -> GlobalParams {
        GlobalParams {
            viewport_size: [WIDTH as f32, HEIGHT as f32],
            premultimated_alpha: 0,
            pad: 0,
        }
    }

    fn layer_transform_bind_group(&self) -> wgpu::BindGroup {
        self.context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("harness layer transform"),
            layout: &self.pipelines.layer_transform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: self.buffers.transforms_buffer(),
                    offset: 0,
                    size: Some(
                        std::num::NonZeroU64::new(std::mem::size_of::<GpuLayerTransform>() as u64)
                            .expect("non-zero transform size"),
                    ),
                }),
            }],
        })
    }

    /// Sync + upload spans exactly like `resolve_slab_spans`, then build the
    /// bind groups needed to draw them.
    fn prepare_spans(&mut self, scene: &Scene) -> SlabDrawGroups {
        let mut synced_layers: FxHashSet<LayerKey> = FxHashSet::default();
        for span in &scene.layer_slab_spans {
            if !synced_layers.insert(span.key) {
                continue;
            }
            match self.registry.plan_sync(span.key, span.content_token, span.totals) {
                Ok(SyncPlan::Clean) => {}
                Ok(SyncPlan::UploadAllOccupied) => {
                    let slabs = self.registry.entry_slabs(span.key).expect("just synced");
                    let mut scratch: Vec<u8> = Vec::new();
                    for kind in SlabKind::ALL {
                        scratch.clear();
                        for layer_span in scene
                            .layer_slab_spans
                            .iter()
                            .filter(|s| s.key == span.key)
                        {
                            append_packed_kind_bytes(&mut scratch, kind, &layer_span.packed);
                        }
                        let range = slabs.slab(kind);
                        if scratch.is_empty() || range.is_empty() {
                            continue;
                        }
                        self.context.queue.write_buffer(
                            self.buffers.kind_buffer(kind),
                            range.byte_offset(slab_gpu::instance_stride(kind)),
                            &scratch,
                        );
                    }
                }
                Err(error) => panic!("harness sync overflowed: {error:?}"),
            }
            self.registry.set_layer_translate(span.key, span.origin);
            self.registry.note_referenced_pages(span.key, []);
        }
        let transforms_buffer = self.buffers.transforms_buffer().clone();
        let stride = self.buffers.transform_slot_stride;
        for (slot, transform) in self.registry.take_dirty_transforms() {
            self.context.queue.write_buffer(
                &transforms_buffer,
                slot as u64 * stride,
                bytemuck::bytes_of(&transform),
            );
        }
        let transform_bind_group = self.layer_transform_bind_group();
        build_slab_draw_groups(
            &self.context.device,
            &self.pipelines,
            &self.buffers,
            &self.atlas,
            &self.atlas_sampler,
            &transform_bind_group,
            scene,
        )
    }

    /// Upload a scene's flat arrays into the shared fixed buffers, mirroring
    /// draw()'s legacy uploads.
    fn upload_legacy_arrays(&self, scene: &Scene) {
        let uploads: [(&parking_lot::Mutex<wgpu::Buffer>, &[u8]); 5] = [
            (&self.context.quads_buffer, bytemuck::cast_slice(&scene.quads)),
            (&self.context.shadows_buffer, bytemuck::cast_slice(&scene.shadows)),
            (
                &self.context.underlines_buffer,
                bytemuck::cast_slice(&scene.underlines),
            ),
            (
                &self.context.mono_sprites_buffer,
                bytemuck::cast_slice(&scene.monochrome_sprites),
            ),
            (
                &self.context.poly_sprites_buffer,
                bytemuck::cast_slice(&scene.polychrome_sprites),
            ),
        ];
        let usage =
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE;
        for (buffer, data) in uploads {
            if data.is_empty() {
                continue;
            }
            ensure_buffer_size(&self.context.device, buffer, data.len() as u64, "harness", usage);
            self.context.queue.write_buffer(&buffer.lock(), 0, data);
        }
        let mut flat_path_vertices: Vec<GpuPathVertex> = Vec::new();
        for path in &scene.paths {
            let color = path.color.solid;
            let cm = &path.content_mask.bounds;
            let cm_origin = [cm.origin.x.0, cm.origin.y.0];
            let cm_size = [cm.size.width.0, cm.size.height.0];
            for vertex in &path.vertices {
                flat_path_vertices.push(GpuPathVertex {
                    xy_position: [vertex.xy_position.x.0, vertex.xy_position.y.0],
                    st_position: [vertex.st_position.x, vertex.st_position.y],
                    hsla: [color.h, color.s, color.l, color.a],
                    content_mask_origin: cm_origin,
                    content_mask_size: cm_size,
                });
            }
        }
        if !flat_path_vertices.is_empty() {
            let data = bytemuck::cast_slice(&flat_path_vertices);
            ensure_buffer_size(
                &self.context.device,
                &self.context.paths_vertices_buffer,
                data.len() as u64,
                "harness paths",
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            );
            self.context
                .queue
                .write_buffer(&self.context.paths_vertices_buffer.lock(), 0, data);
        }
        let globals = self.globals();
        self.context
            .queue
            .write_buffer(&self.context.globals_buffer, 0, bytemuck::bytes_of(&globals));
    }

    fn buffer_group(
        &self,
        layout: &wgpu::BindGroupLayout,
        buffer: &parking_lot::Mutex<wgpu::Buffer>,
    ) -> wgpu::BindGroup {
        self.context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buffer.lock(),
                    offset: 0,
                    size: None,
                }),
            }],
        })
    }

    fn texture_group(&self, view: &wgpu::TextureView) -> wgpu::BindGroup {
        self.context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipelines.sprites_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.atlas_sampler),
                },
            ],
        })
    }

    /// Record one frame — legacy batches plus spliced spans — into an
    /// offscreen target and read its pixels back.
    fn render_and_read_back(&self, scene: &Scene, groups: Option<&SlabDrawGroups>) -> Vec<u8> {
        let target = self.context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pixel test target"),
            size: wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());

        let quads_bg =
            self.buffer_group(&self.pipelines.quads_bind_group_layout, &self.context.quads_buffer);
        let shadows_bg = self
            .buffer_group(&self.pipelines.shadows_bind_group_layout, &self.context.shadows_buffer);
        let underlines_bg = self.buffer_group(
            &self.pipelines.underlines_bind_group_layout,
            &self.context.underlines_buffer,
        );
        let mono_bg = self.buffer_group(
            &self.pipelines.mono_sprites_bind_group_layout,
            &self.context.mono_sprites_buffer,
        );
        let poly_bg = self.buffer_group(
            &self.pipelines.poly_sprites_bind_group_layout,
            &self.context.poly_sprites_buffer,
        );
        let paths_bg = self.buffer_group(
            &self.pipelines.paths_bind_group_layout,
            &self.context.paths_vertices_buffer,
        );
        let transform_bg = self.layer_transform_bind_group();

        let mut encoder = self
            .context
            .device
            .create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("pixel test pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
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

            let mut quads_first: u32 = 0;
            let mut shadows_first: u32 = 0;
            let mut underlines_first: u32 = 0;
            let mut mono_first: u32 = 0;
            let mut poly_first: u32 = 0;
            let mut paths_offset: u32 = 0;

            let mut emit_legacy = |pass: &mut wgpu::RenderPass<'_>,
                               batch: &crate::scene::PrimitiveBatch<'_>| {
                match batch {
                    PrimitiveBatch::Quads(quads) => {
                        let count = quads.len() as u32;
                        pass.set_pipeline(&self.pipelines.quads_pipeline);
                        pass.set_bind_group(0, &self.pipelines.globals_bind_group, &[]);
                        pass.set_bind_group(1, &quads_bg, &[]);
                        pass.set_bind_group(2, &transform_bg, &[0]);
                        pass.draw(0..4, quads_first..quads_first + count);
                        quads_first += count;
                    }
                    PrimitiveBatch::Shadows(shadows) => {
                        let count = shadows.len() as u32;
                        pass.set_pipeline(&self.pipelines.shadows_pipeline);
                        pass.set_bind_group(0, &self.pipelines.globals_bind_group, &[]);
                        pass.set_bind_group(1, &shadows_bg, &[]);
                        pass.set_bind_group(2, &transform_bg, &[0]);
                        pass.draw(0..4, shadows_first..shadows_first + count);
                        shadows_first += count;
                    }
                    PrimitiveBatch::Underlines(underlines) => {
                        let count = underlines.len() as u32;
                        pass.set_pipeline(&self.pipelines.underlines_pipeline);
                        pass.set_bind_group(0, &self.pipelines.globals_bind_group, &[]);
                        pass.set_bind_group(1, &underlines_bg, &[]);
                        pass.set_bind_group(2, &transform_bg, &[0]);
                        pass.draw(0..4, underlines_first..underlines_first + count);
                        underlines_first += count;
                    }
                    PrimitiveBatch::MonochromeSprites { texture_id, sprites } => {
                        let count = sprites.len() as u32;
                        let tex = self.atlas.get_texture_info(*texture_id);
                        let tex_bg = self.texture_group(&tex.raw_view);
                        pass.set_pipeline(&self.pipelines.mono_sprites_pipeline);
                        pass.set_bind_group(0, &self.pipelines.globals_bind_group, &[]);
                        pass.set_bind_group(1, &self.pipelines.color_adjustments_bind_group, &[]);
                        pass.set_bind_group(2, &tex_bg, &[]);
                        pass.set_bind_group(3, &mono_bg, &[]);
                        pass.set_bind_group(4, &transform_bg, &[0]);
                        pass.draw(0..4, mono_first..mono_first + count);
                        mono_first += count;
                    }
                    PrimitiveBatch::PolychromeSprites { texture_id, sprites } => {
                        let count = sprites.len() as u32;
                        let tex = self.atlas.get_texture_info(*texture_id);
                        let tex_bg = self.texture_group(&tex.raw_view);
                        pass.set_pipeline(&self.pipelines.poly_sprites_pipeline);
                        pass.set_bind_group(0, &self.pipelines.globals_bind_group, &[]);
                        pass.set_bind_group(1, &tex_bg, &[]);
                        pass.set_bind_group(2, &poly_bg, &[]);
                        pass.set_bind_group(3, &transform_bg, &[0]);
                        pass.draw(0..4, poly_first..poly_first + count);
                        poly_first += count;
                    }
                    PrimitiveBatch::Paths(paths) => {
                        let vertex_count: u32 =
                            paths.iter().map(|p| p.vertices.len() as u32).sum();
                        if vertex_count > 0 {
                            pass.set_pipeline(&self.pipelines.paths_pipeline);
                            pass.set_bind_group(0, &self.pipelines.globals_bind_group, &[]);
                            pass.set_bind_group(1, &paths_bg, &[]);
                            pass.set_bind_group(2, &transform_bg, &[0]);
                            pass.draw(paths_offset..paths_offset + vertex_count, 0..1);
                            paths_offset += vertex_count;
                        }
                    }
                    _ => {}
                }
            };

            for frame_batch in scene.frame_batches() {
                match frame_batch {
                    SceneBatch::Primitives(batch) => emit_legacy(&mut pass, &batch),
                    SceneBatch::LayerSlab(index) => {
                        let groups = groups.expect("spans need slab bind groups");
                        let span = &scene.layer_slab_spans[index];
                        let slabs = self.registry.entry_slabs(span.key).unwrap();
                        let slot = self.registry.transform_slot(span.key).unwrap();
                        let stride = self.buffers.transform_slot_stride;
                        let mut pending: Option<SlabPendingRun> = None;
                        for run in &span.runs {
                            match pending.as_mut() {
                                Some(open)
                                    if open.kind == run.kind
                                        && open.texture_id == run.texture_id
                                        && open.start + open.count == run.start =>
                                {
                                    open.count += run.count;
                                    continue;
                                }
                                _ => {}
                            }
                            if let Some(open) = pending.take() {
                                flush_slab_run(
                                    &self.pipelines,
                                    stride,
                                    &mut pass,
                                    &slabs,
                                    groups,
                                    slot,
                                    &open,
                                );
                            }
                            pending = Some(SlabPendingRun {
                                kind: run.kind,
                                texture_id: run.texture_id,
                                start: run.start,
                                count: run.count,
                            });
                        }
                        if let Some(open) = pending.take() {
                            flush_slab_run(
                                &self.pipelines,
                                stride,
                                &mut pass,
                                &slabs,
                                groups,
                                slot,
                                &open,
                            );
                        }
                    }
                }
            }
        }

        // Readback with row padding stripped.
        let bytes_per_pixel = 4u32;
        let unpadded_row = WIDTH * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_row = unpadded_row.div_ceil(align) * align;
        let staging = self.context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pixel test staging"),
            size: (padded_row * HEIGHT) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
        );
        self.context.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.context.device.poll(wgpu::PollType::wait_indefinitely());
        let data = slice.get_mapped_range().expect("staging map succeeded");
        let mut out = Vec::with_capacity((unpadded_row * HEIGHT) as usize);
        for row in 0..HEIGHT {
            let start = (row * padded_row) as usize;
            out.extend_from_slice(&data[start..start + unpadded_row as usize]);
        }
        drop(data);
        staging.unmap();
        out
    }
}

// -- Scene construction helpers --------------------------------------

fn sp(value: f32) -> ScaledPixels {
    ScaledPixels(value)
}

fn rect(x: f32, y: f32, w: f32, h: f32) -> SceneBounds<ScaledPixels> {
    SceneBounds {
        origin: Point { x: sp(x), y: sp(y) },
        size: SceneSize { width: sp(w), height: sp(h) },
    }
}

fn mask() -> ContentMask<ScaledPixels> {
    ContentMask {
        bounds: rect(-1000., -1000., 10_000., 10_000.),
    }
}

fn quad_solid(bounds: SceneBounds<ScaledPixels>, color: Hsla) -> Quad {
    Quad {
        bounds,
        content_mask: mask(),
        background: color.into(),
        corner_radii: Corners {
            top_left: sp(8.),
            top_right: sp(8.),
            bottom_right: sp(8.),
            bottom_left: sp(8.),
        },
        border_widths: Edges::all(sp(2.)),
        border_color: Hsla { h: 0.6, s: 1., l: 0.2, a: 1. },
        ..Default::default()
    }
}

fn shadow_under(bounds: SceneBounds<ScaledPixels>) -> SceneShadow {
    SceneShadow {
        order: 0,
        blur_radius: sp(6.),
        bounds,
        corner_radii: Corners::default(),
        content_mask: mask(),
        color: Hsla { h: 0., s: 0., l: 0., a: 0.9 },
    }
}

fn underline(bounds: SceneBounds<ScaledPixels>) -> SceneUnderline {
    SceneUnderline {
        order: 0,
        pad: 0,
        bounds,
        content_mask: mask(),
        color: Hsla { h: 0.33, s: 1., l: 0.5, a: 1. },
        thickness: sp(3.),
        wavy: 0,
    }
}

fn path_triangle(origin: (f32, f32)) -> ScenePath<ScaledPixels> {
    let (x, y) = origin;
    let mut p = ScenePath::new(point(px(x), px(y)));
    p.move_to(point(px(x), px(y)));
    p.line_to(point(px(x + 60.), px(y)));
    p.line_to(point(px(x + 30.), px(y + 50.)));
    let mut scaled = p.scale(1.0);
    scaled.content_mask = mask();
    scaled.color = crate::Background::from(Hsla { h: 0.12, s: 1., l: 0.55, a: 1. });
    scaled.vertices[0].st_position.x = 0.25;
    scaled
}


fn poly_sprite(
    bounds: SceneBounds<ScaledPixels>,
    tile: AtlasTile,
    opacity: f32,
) -> PolychromeSprite {
    PolychromeSprite {
        order: 0,
        pad: 0,
        grayscale: 0,
        opacity,
        bounds,
        content_mask: mask(),
        corner_radii: Corners::default(),
        tile,
    }
}

/// A real 16x16 polychrome tile through the production atlas entrypoint, so
/// slab and legacy sides bind identical texels.
fn insert_tile(harness: &PixelHarness, image_number: usize) -> anyhow::Result<AtlasTile> {
    use crate::PlatformAtlas as _;
    let key = crate::AtlasKey::Image(crate::RenderImageParams {
        image_id: crate::ImageId(image_number),
        frame_index: 0,
    });
    let tile = harness.atlas.get_or_insert_with(
        &key,
        &mut || {
            let side = 16usize;
            let bytes: Vec<u8> = (0..side * side * 4)
                .map(|i| ((i % 4) * 60 + 30) as u8)
                .collect();
            Ok(Some((
                SceneSize {
                    width: DevicePixels(side as i32),
                    height: DevicePixels(side as i32),
                },
                std::borrow::Cow::Owned(bytes),
            )))
        },
    )?
    .ok_or_else(|| anyhow::anyhow!("atlas refused the test tile"))?;
    Ok(tile)
}

/// The shared layer recipe at `origin`. Every primitive overlaps so orders
/// track paint order identically on both sides of the comparison.
fn paint_layer_recipe(scene: &mut Scene, origin: (f32, f32), tile: AtlasTile) {
    let (dx, dy) = origin;
    scene.insert_primitive(quad_solid(
        rect(dx, dy, 120., 90.),
        Hsla { h: 0.58, s: 0.9, l: 0.45, a: 1. },
    ));
    scene.insert_primitive(shadow_under(rect(dx + 10., dy + 10., 100., 60.)));
    scene.insert_primitive(path_triangle((dx + 20., dy + 15.)));
    scene.insert_primitive(underline(rect(dx + 10., dy + 70., 80., 6.)));
    scene.insert_primitive(poly_sprite(rect(dx + 60., dy + 40., 40., 40.), tile, 1.));
    scene.insert_primitive(quad_solid(
        rect(dx + 30., dy + 25., 50., 50.),
        Hsla { h: 0.02, s: 0.8, l: 0.6, a: 1. },
    ));
}

fn background_quad() -> Quad {
    quad_solid(
        rect(0., 0., WIDTH as f32, HEIGHT as f32),
        Hsla { h: 0.99, s: 0.7, l: 0.18, a: 1. },
    )
}

fn origin_arr(origin: (f32, f32)) -> [f32; 2] {
    [origin.0, origin.1]
}

/// Build the two comparable frames from one recipe: legacy composites through
/// push_retained; spliced emits one span whose packed bytes are relative to
/// `origin` and whose uniform restores it.
fn build_frames(origin: (f32, f32), tile: AtlasTile) -> anyhow::Result<(Scene, Scene)> {
    let panel_bounds = rect(origin.0, origin.1, 200., 160.);
    let mut recording = Scene::default();
    recording.insert_primitive(background_quad());
    recording.begin_layer(LayerKey(101), panel_bounds, true);
    paint_layer_recipe(&mut recording, origin, tile);
    let items = recording.end_layer().unwrap();

    let mut legacy = Scene::default();
    legacy.insert_primitive(background_quad());
    legacy.begin_layer(LayerKey(101), panel_bounds, false);
    for item in &items {
        match item {
            crate::layer::LayerItem::Primitive(primitive) => legacy.push_retained(primitive),
            crate::layer::LayerItem::Nested(_) => unreachable!(),
        }
    }
    legacy.end_layer();
    legacy.finish();

    let mut packed = match crate::scene_pack::pack_layer_items(&items) {
        crate::scene_pack::PackOutcome::Packed(packed) => packed,
        crate::scene_pack::PackOutcome::FellBack(reason) => {
            anyhow::bail!("recipe unexpectedly fell back: {reason:?}")
        }
    };
    crate::platform::cross::slab_gpu::make_packed_relative(&mut packed, origin_arr(origin));
    let mut runs = Vec::new();
    for run in &packed.runs {
        runs.push(SlabRun {
            kind: run.kind,
            start: run.start,
            count: run.count,
            texture_id: run.texture_id,
        });
    }
    let totals = [
        packed.quads.len() as u32,
        packed.shadows.len() as u32,
        packed.total_path_vertices(),
        packed.underlines.len() as u32,
        packed.mono_sprites.len() as u32,
        packed.poly_sprites.len() as u32,
    ];

    let mut spliced = Scene::default();
    spliced.insert_primitive(background_quad());
    spliced.begin_layer(LayerKey(101), panel_bounds, false);
    spliced.push_layer_slab_span(
        panel_bounds,
        LayerKey(101),
        1,
        origin_arr(origin),
        totals,
        runs,
        Arc::from(packed),
    );
    spliced.end_layer();
    spliced.finish();

    Ok((legacy, spliced))
}

#[test]
fn packed_layer_pixels_match_the_legacy_render() -> anyhow::Result<()> {
    let Some(mut harness) = headless_harness() else {
        eprintln!("skipping packed_layer_pixels_match_the_legacy_render: no wgpu adapter");
        return Ok(());
    };
    let tile = insert_tile(&harness, 7)?;

    let (legacy, spliced) = build_frames((24., 16.), tile)?;

    harness.upload_legacy_arrays(&legacy);
    let legacy_bytes = harness.render_and_read_back(&legacy, None);

    let groups = harness.prepare_spans(&spliced);
    harness.upload_legacy_arrays(&spliced);
    let spliced_bytes = harness.render_and_read_back(&spliced, Some(&groups));

    let differing: Vec<usize> = legacy_bytes
        .iter()
        .zip(spliced_bytes.iter())
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(index, _)| index)
        .collect();
    assert!(
        differing.is_empty(),
        "{} differing bytes between the legacy and spliced renders; first at \
         byte {}",
        differing.len(),
        differing.first().copied().unwrap_or(usize::MAX)
    );

    // Guard against a trivially-blank comparison: the layer's blue quad sits
    // near (30, 30); its pixel must not be the clear color.
    let probe = ((28 + 30 * HEIGHT as usize) * 4)..((28 + 30 * HEIGHT as usize) * 4 + 4);
    assert_ne!(&legacy_bytes[probe], &[0, 0, 0, 255][..]);

    Ok(())
}

#[test]
fn a_transform_only_moved_span_renders_shifted_without_reuploading_instances()
-> anyhow::Result<()> {
    let Some(mut harness) = headless_harness() else {
        eprintln!(
            "skipping a_transform_only_moved_span_renders_shifted...: no wgpu adapter"
        );
        return Ok(());
    };
    let tile = insert_tile(&harness, 7)?;

    // Reference: the same recipe painted directly at the shifted location.
    let shifted = (72., 48.);
    let (reference_scene, _) = build_frames(shifted, tile)?;
    harness.upload_legacy_arrays(&reference_scene);
    let reference_bytes = harness.render_and_read_back(&reference_scene, None);

    // Moved layer: content stays packed relative to the ORIGINAL origin; only
    // the uniform slot's translate changes between renders.
    let original = (24., 16.);
    let (_, mut moved_scene) = build_frames(original, tile)?;
    let _groups_initial = harness.prepare_spans(&moved_scene);

    for span in &mut moved_scene.layer_slab_spans {
        span.origin = origin_arr(shifted);
    }
    // Same token: plan_sync must answer Clean for instance data — the move
    // touches exactly one uniform slot and nothing else.
    let token = moved_scene.layer_slab_spans[0].content_token;
    assert_eq!(
        harness
            .registry
            .plan_sync(
                LayerKey(101),
                token,
                moved_scene.layer_slab_spans[0].totals
            )
            .unwrap(),
        SyncPlan::Clean,
        "the move must leave resident instance data untouched"
    );

    let groups = harness.prepare_spans(&moved_scene);
    harness.upload_legacy_arrays(&moved_scene);
    let moved_bytes = harness.render_and_read_back(&moved_scene, Some(&groups));

    let differing: Vec<usize> = reference_bytes
        .iter()
        .zip(moved_bytes.iter())
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(index, _)| index)
        .collect();
    assert!(
        differing.is_empty(),
        "{} differing bytes between the shifted reference and the uniformly \
         translated slab render; first at byte {}",
        differing.len(),
        differing.first().copied().unwrap_or(usize::MAX)
    );
    Ok(())
}
