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

// ---------------------------------------------------------------------
// State-dedup / cross-span merged flushing.
//
// These tests drive every readback through a bounded-wait poll: sibling GPU
// tests share this process's driver, and an indefinite poll can wedge the
// whole suite when they overlap. Time out loudly instead of hanging.
// ---------------------------------------------------------------------

/// A synthetic scene whose one layer is split across TWO adjacent spans with
/// exactly contiguous quad runs (the shape nested layers produce), including
/// an intra-span adjacent run pair, followed by a SECOND layer whose same-kind
/// instances do NOT continue the first layer's stretch. Cross-layer runs sit
/// adjacent in the kind buffer yet must never merge: their transform slots
/// differ, so merging would move pixels.
///
/// Returns `(legacy_composite, spliced)` like [`build_frames`].
fn build_multi_span_frames(origin: (f32, f32)) -> anyhow::Result<(Scene, Scene)> {
    const KEY_A: LayerKey = LayerKey(202);
    const KEY_B: LayerKey = LayerKey(203);
    let panel_a = rect(origin.0, origin.1, 240., 80.);
    let panel_b = rect(origin.0, origin.1 + 90., 120., 70.);

    let mut recording = Scene::default();
    recording.insert_primitive(background_quad());
    recording.begin_layer(KEY_A, panel_a, true);
    for index in 0..8u32 {
        recording.insert_primitive(quad_solid(
            rect(origin.0 + 8. + index as f32 * 26., origin.1 + 12., 34., 56.),
            Hsla { h: index as f32 * 0.125, s: 0.85, l: 0.5, a: if index % 2 == 0 { 1. } else { 0.72 } },
        ));
    }
    let items_a = recording.end_layer().unwrap();
    recording.begin_layer(KEY_B, panel_b, true);
    for index in 0..3u32 {
        recording.insert_primitive(quad_solid(
            rect(origin.0 + 12. + index as f32 * 38., origin.1 + 104., 44., 44.),
            Hsla { h: 0.5 + index as f32 * 0.1, s: 0.7, l: 0.42, a: 0.9 },
        ));
    }
    let items_b = recording.end_layer().unwrap();

    let mut legacy = Scene::default();
    legacy.insert_primitive(background_quad());
    legacy.begin_layer(KEY_A, panel_a, false);
    for item in &items_a {
        match item {
            crate::layer::LayerItem::Primitive(primitive) => legacy.push_retained(primitive),
            crate::layer::LayerItem::Nested(_) => unreachable!(),
        }
    }
    legacy.end_layer();
    legacy.begin_layer(KEY_B, panel_b, false);
    for item in &items_b {
        match item {
            crate::layer::LayerItem::Primitive(primitive) => legacy.push_retained(primitive),
            crate::layer::LayerItem::Nested(_) => unreachable!(),
        }
    }
    legacy.end_layer();
    legacy.finish();

    let pack =
        |items: &[crate::layer::LayerItem]| -> anyhow::Result<Box<crate::scene_pack::PackedLayer>> {
            match crate::scene_pack::pack_layer_items(items) {
                crate::scene_pack::PackOutcome::Packed(packed) => Ok(packed),
                crate::scene_pack::PackOutcome::FellBack(reason) => {
                    anyhow::bail!("recipe unexpectedly fell back: {reason:?}")
                }
            }
        };

    let mut packed_a = pack(&items_a)?;
    let mut packed_b = pack(&items_b)?;
    crate::platform::cross::slab_gpu::make_packed_relative(&mut packed_a, origin_arr(origin));
    crate::platform::cross::slab_gpu::make_packed_relative(&mut packed_b, origin_arr(origin));

    // The synthetic recipes are quads-only; anything else means the split
    // bookkeeping below would silently drop instances.
    for packed in [&packed_a, &packed_b] {
        assert!(packed.shadows.is_empty() && packed.paths.is_empty()
            && packed.underlines.is_empty() && packed.mono_sprites.is_empty()
            && packed.poly_sprites.is_empty());
    }

    // Split layer A's quad stream across two spans: the head carries [0..4),
    // the tail carries [4..8) as TWO adjacent runs. Layer B stays whole. All
    // spans of one layer carry the layer-wide totals.
    let totals_a = [packed_a.quads.len() as u32, 0, 0, 0, 0, 0];
    let totals_b = [packed_b.quads.len() as u32, 0, 0, 0, 0, 0];
    let mid = 4u32;
    let head = Box::new(crate::scene_pack::PackedLayer {
        quads: packed_a.quads[..mid as usize].to_vec(),
        shadows: Vec::new(),
        paths: Vec::new(),
        underlines: Vec::new(),
        mono_sprites: Vec::new(),
        poly_sprites: Vec::new(),
        runs: vec![crate::scene_pack::KindRun {
            kind: SlabKind::Quads,
            start: 0,
            count: mid,
            texture_id: None,
        }],
    });
    let tail = Box::new(crate::scene_pack::PackedLayer {
        quads: packed_a.quads[mid as usize..].to_vec(),
        shadows: Vec::new(),
        paths: Vec::new(),
        underlines: Vec::new(),
        mono_sprites: Vec::new(),
        poly_sprites: Vec::new(),
        runs: vec![
            crate::scene_pack::KindRun { kind: SlabKind::Quads, start: mid, count: 2, texture_id: None },
            crate::scene_pack::KindRun { kind: SlabKind::Quads, start: mid + 2, count: 2, texture_id: None },
        ],
    });

    let mut spliced = Scene::default();
    spliced.insert_primitive(background_quad());
    spliced.begin_layer(KEY_A, panel_a, false);
    spliced.push_layer_slab_span(
        panel_a,
        KEY_A,
        1,
        origin_arr(origin),
        totals_a,
        vec![crate::scene::SlabRun { kind: SlabKind::Quads, start: 0, count: mid, texture_id: None }],
        std::sync::Arc::from(head),
    );
    spliced.push_layer_slab_span(
        panel_a,
        KEY_A,
        1,
        origin_arr(origin),
        totals_a,
        vec![
            crate::scene::SlabRun { kind: SlabKind::Quads, start: mid, count: 2, texture_id: None },
            crate::scene::SlabRun { kind: SlabKind::Quads, start: mid + 2, count: 2, texture_id: None },
        ],
        std::sync::Arc::from(tail),
    );
    spliced.end_layer();
    spliced.begin_layer(KEY_B, panel_b, false);
    spliced.push_layer_slab_span(
        panel_b,
        KEY_B,
        1,
        origin_arr(origin),
        totals_b,
        vec![crate::scene::SlabRun {
            kind: SlabKind::Quads,
            start: 0,
            count: totals_b[0],
            texture_id: None,
        }],
        std::sync::Arc::from(packed_b),
    );
    spliced.end_layer();
    spliced.finish();

    Ok((legacy, spliced))
}

impl PixelHarness {
    /// Render through one of two slab-flushing strategies and read back:
    ///
    /// - `production = true`: the production draw loop — shared bind-state
    ///   tracker, cross-span merged stretches flushed via
    ///   `flush_slab_run_with_state`.
    /// - `production = false`: the naive baseline — per-span pending runs
    ///   flushed through the untracked wrapper, exactly like the pre-existing
    ///   readback helper drives them.
    ///
    /// Both share this method's bounded-wait readback so a wedged driver call
    /// can never hang the suite from here. Returns the pixels plus how many
    /// slab draws were issued.
    fn render_and_read_back_mode(
        &self,
        scene: &Scene,
        groups: Option<&SlabDrawGroups>,
        production: bool,
    ) -> (Vec<u8>, usize) {
        let stride = self.buffers.transform_slot_stride;
        let mut state = super::PassBindState::default();
        let mut open: Option<super::OpenSlabRun> = None;
        let mut slab_draws = 0usize;

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
        let paths_bg = self.buffer_group(
            &self.pipelines.paths_bind_group_layout,
            &self.context.paths_vertices_buffer,
        );

        let mut encoder = self
            .context
            .device
            .create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("pixel test mode pass"),
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
            let mut paths_offset: u32 = 0;

            // Legacy draws read the identity slot; without slab groups this
            // method binds its own transform group, exactly like the original
            // helper.
            let legacy_only_transform = self.layer_transform_bind_group();
            let transform_for_legacy_draws: &wgpu::BindGroup = match groups {
                Some(groups) => &groups.layer_transform,
                None => &legacy_only_transform,
            };

            for frame_batch in scene.frame_batches() {
                match frame_batch {
                    crate::scene::SceneBatch::Primitives(batch) => {
                        if production
                            && primitive_batch_instance_count(&batch) > 0
                            && open.is_some()
                        {
                            slab_draws += 1;
                            super::flush_open_slab_run(
                                &self.pipelines,
                                stride,
                                &mut pass,
                                groups.expect("production mode carries slab groups"),
                                &mut state,
                                &mut open,
                            );
                        }
                        match batch {
                            PrimitiveBatch::Quads(quads) => {
                                let count = quads.len() as u32;
                                state.set_pipeline(
                                    &mut pass,
                                    super::DrawPipelineId::Quads,
                                    &self.pipelines.quads_pipeline,
                                );
                                state.set_bind_group(
                                    &mut pass,
                                    0,
                                    super::BoundGroupId::Globals,
                                    &self.pipelines.globals_bind_group,
                                    &[],
                                );
                                state.set_bind_group(
                                    &mut pass,
                                    1,
                                    super::BoundGroupId::LegacyBuffer(super::LegacyBuffer::Quads),
                                    &quads_bg,
                                    &[],
                                );
                                state.set_bind_group(
                                    &mut pass,
                                    2,
                                    super::BoundGroupId::LayerTransform(0),
                                    transform_for_legacy_draws,
                                    &[0],
                                );
                                pass.draw(0..4, quads_first..quads_first + count);
                                quads_first += count;
                            }
                            PrimitiveBatch::Paths(paths) => {
                                let vertex_count: u32 =
                                    paths.iter().map(|p| p.vertices.len() as u32).sum();
                                if vertex_count > 0 {
                                    state.set_pipeline(
                                        &mut pass,
                                        super::DrawPipelineId::Paths,
                                        &self.pipelines.paths_pipeline,
                                    );
                                    state.set_bind_group(
                                        &mut pass,
                                        0,
                                        super::BoundGroupId::Globals,
                                        &self.pipelines.globals_bind_group,
                                        &[],
                                    );
                                    state.set_bind_group(
                                        &mut pass,
                                        1,
                                        super::BoundGroupId::LegacyBuffer(
                                            super::LegacyBuffer::PathVertices,
                                        ),
                                        &paths_bg,
                                        &[],
                                    );
                                    state.set_bind_group(
                                        &mut pass,
                                        2,
                                        super::BoundGroupId::LayerTransform(0),
                                        transform_for_legacy_draws,
                                        &[0],
                                    );
                                    pass.draw(paths_offset..paths_offset + vertex_count, 0..1);
                                    paths_offset += vertex_count;
                                }
                            }
                            _ => {}
                        }
                    }
                    crate::scene::SceneBatch::LayerSlab(index) => {
                        let groups = match groups {
                            Some(groups) => groups,
                            // The legacy composite carries no spans.
                            None => continue,
                        };
                        if !production {
                            // Naive baseline: per-span pending runs flushed
                            // through the untracked wrapper, exactly like the
                            // pre-existing helper drives them.
                            let span = &scene.layer_slab_spans[index];
                            let slabs = self.registry.entry_slabs(span.key).unwrap();
                            let slot = self.registry.transform_slot(span.key).unwrap();
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
                                    slab_draws += 1;
                                    super::flush_slab_run(
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
                                slab_draws += 1;
                                super::flush_slab_run(
                                    &self.pipelines,
                                    stride,
                                    &mut pass,
                                    &slabs,
                                    groups,
                                    slot,
                                    &open,
                                );
                            }
                            continue;
                        }
                        let span = &scene.layer_slab_spans[index];
                        let slabs = self.registry.entry_slabs(span.key).unwrap();
                        let slot = self.registry.transform_slot(span.key).unwrap();
                        for run in &span.runs {
                            if let Some(open) = open.as_mut()
                                && open.accepts(span.key, &slabs, run)
                            {
                                open.count += run.count;
                                continue;
                            }
                            if open.is_some() {
                                slab_draws += 1;
                                super::flush_open_slab_run(
                                    &self.pipelines,
                                    stride,
                                    &mut pass,
                                    groups,
                                    &mut state,
                                    &mut open,
                                );
                            }
                            open = Some(super::OpenSlabRun {
                                key: span.key,
                                slabs,
                                transform_slot: slot,
                                kind: run.kind,
                                texture_id: run.texture_id,
                                start: run.start,
                                count: run.count,
                            });
                        }
                    }
                }
            }
            if production && open.is_some() {
                slab_draws += 1;
                super::flush_open_slab_run(
                    &self.pipelines,
                    stride,
                    &mut pass,
                    groups.expect("production mode carries slab groups"),
                    &mut state,
                    &mut open,
                );
            }
        }

        // Readback with row padding stripped, under a bounded wait.
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
        let map_result = Arc::new(std::sync::Mutex::new(None::<String>));
        let map_sink = map_result.clone();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            if let Err(error) = result {
                *map_sink.lock().expect("map result mutex") = Some(format!("{error}"));
            }
        });
        let mut mapped = false;
        for _ in 0..60 {
            match self.context.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_millis(500)),
            }) {
                Ok(_) => {}
                Err(wgpu::PollError::Timeout) => continue,
                Err(error) => panic!("device poll failed during readback: {error:?}"),
            }
            if let Some(error) = map_result.lock().expect("map result mutex").as_ref() {
                panic!("readback map failed: {error}");
            }
            mapped = true;
            break;
        }
        assert!(
            mapped,
            "readback never mapped within 30s; the headless device wedged"
        );
        let data = slice.get_mapped_range().expect("staging map succeeded");
        let mut out = Vec::with_capacity((unpadded_row * HEIGHT) as usize);
        for row in 0..HEIGHT {
            let start = (row * padded_row) as usize;
            out.extend_from_slice(&data[start..start + unpadded_row as usize]);
        }
        drop(data);
        staging.unmap();
        (out, slab_draws)
    }
}

#[test]
fn merged_state_tracked_flush_renders_identically_to_naive_flush() -> anyhow::Result<()> {
    let Some(mut harness) = headless_harness() else {
        eprintln!(
            "skipping merged_state_tracked_flush_renders_identically...: no wgpu adapter"
        );
        return Ok(());
    };

    let (legacy, spliced) = build_multi_span_frames((8., 6.))?;

    harness.upload_legacy_arrays(&legacy);
    let legacy_bytes = harness.render_and_read_back_mode(&legacy, None, false).0;

    let groups = harness.prepare_spans(&spliced);
    harness.upload_legacy_arrays(&spliced);
    let naive_bytes = harness.render_and_read_back_mode(&spliced, Some(&groups), false).0;
    let (production_bytes, production_draws) =
        harness.render_and_read_back_mode(&spliced, Some(&groups), true);

    // Pixel evidence: the merged/deduped production loop is byte-identical to
    // BOTH the naive per-run slab loop and the pure legacy composite.
    let naive_vs_production: Vec<usize> = naive_bytes
        .iter()
        .zip(production_bytes.iter())
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(index, _)| index)
        .collect();
    assert!(
        naive_vs_production.is_empty(),
        "{} differing bytes between naive and merged/deduped slab flushing; \
         first at byte {}",
        naive_vs_production.len(),
        naive_vs_production.first().copied().unwrap_or(usize::MAX)
    );
    let legacy_vs_production: Vec<usize> = legacy_bytes
        .iter()
        .zip(production_bytes.iter())
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(index, _)| index)
        .collect();
    assert!(
        legacy_vs_production.is_empty(),
        "{} differing bytes between the legacy composite and the merged slab \
         render; first at byte {}",
        legacy_vs_production.len(),
        legacy_vs_production.first().copied().unwrap_or(usize::MAX)
    );

    // Guard against a trivially-blank comparison: the last quad of layer B
    // sits near (98, 122); its pixel must not be the clear color.
    let probe = ((126 + 130 * HEIGHT as usize) * 4)..((126 + 130 * HEIGHT as usize) * 4 + 4);
    assert_ne!(&production_bytes[probe], &[0, 0, 0, 255][..]);

    // Draw-call evidence: layer A's two spans merge into one stretch (its
    // intra-span pair folds in too), and the tracker skips redundant state
    // for layer B's separate draw. The naive loop issues one draw PER SPAN
    // after intra-span merging (three here); production issues two — the
    // cross-layer boundary cannot merge (different transform slots).
    assert_eq!(
        production_draws, 2,
        "adjacent same-layer spans must merge; cross-layer spans must not"
    );

    Ok(())
}

#[test]
fn cached_slab_groups_survive_clean_only_frames_and_invalidate_per_buffer() -> anyhow::Result<()> {
    let Some(harness) = headless_harness() else {
        eprintln!(
            "skipping cached_slab_groups_survive_clean_only_frames...: no wgpu adapter"
        );
        return Ok(());
    };
    let tile = insert_tile(&harness, 11)?;
    let (_, spliced) = build_frames((24., 16.), tile)?;
    let device = &harness.context.device;

    let mut cache = SlabGroupCache::default();
    let frame_groups = |cache: &mut SlabGroupCache, scene: &Scene| {
        cache.frame_groups(
            device,
            &harness.pipelines,
            &harness.buffers,
            &harness.atlas,
            &harness.atlas_sampler,
            scene,
        )
    };

    let first = frame_groups(&mut cache, &spliced);
    // Six kind buffers + the transform uniform + the scene's one atlas page.
    assert_eq!(
        cache.creation_count(),
        6 + 1 + 1,
        "the first frame builds every group exactly once"
    );
    assert_eq!(first.sprite_textures.len(), 1);

    // A Clean-only frame: identical scene, nothing recreated, handles reused.
    let second = frame_groups(&mut cache, &spliced);
    assert_eq!(
        cache.creation_count(),
        6 + 1 + 1,
        "Clean-only frames must not rebuild any bind groups"
    );
    assert_eq!(second.sprite_textures.len(), 1);

    // An empty-span frame changes the referenced-page set to nothing: the
    // page map drops its entries without rebuilding buffer groups.
    let mut empty_scene = Scene::default();
    empty_scene.finish();
    let third = frame_groups(&mut cache, &empty_scene);
    assert_eq!(cache.creation_count(), 6 + 1 + 1);
    assert!(
        third.sprite_textures.is_empty(),
        "the stale page binding must be released once no span references it"
    );

    // Back to the tiled scene: exactly the missing page group is rebuilt.
    let fourth = frame_groups(&mut cache, &spliced);
    assert_eq!(cache.creation_count(), 6 + 1 + 2);
    assert_eq!(fourth.sprite_textures.len(), 1);

    // A recreated kind buffer invalidates exactly that kind's group.
    cache.invalidate_kind(SlabKind::Shadows);
    let _fifth = frame_groups(&mut cache, &spliced);
    assert_eq!(cache.creation_count(), 6 + 1 + 3);

    // A recreated transform uniform invalidates its group.
    cache.invalidate_transforms();
    let _sixth = frame_groups(&mut cache, &spliced);
    assert_eq!(cache.creation_count(), 6 + 1 + 4);

    Ok(())
}
