use std::sync::Arc;
use wgpu::util::DeviceExt;

use crate::{
    BackdropFilter, FilterBoundary, PrimitiveBatch, Scene,
    platform::cross::{
        atlas::WgpuAtlas,
        render_context::WgpuContext,
        renderer::{
            ColorAdjustments, GpuPathVertex, GlobalParams, MAX_FILTER_DEPTH, RenderingParameters,
            WgpuPipelines, create_filter_group_textures,
        },
    },
};

unsafe fn as_bytes<T>(slice: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            slice.as_ptr() as *const u8,
            slice.len() * std::mem::size_of::<T>(),
        )
    }
}

pub(crate) enum CompositorJob {
    Commit(Scene),
    Resize(u32, u32),
    Shutdown,
}

pub(crate) struct CompositorCompletion {
    pub(crate) frame_ready: bool,
}

pub(crate) struct CompositorHandle {
    pub(crate) job_tx: flume::Sender<CompositorJob>,
    pub(crate) completion_rx: flume::Receiver<CompositorCompletion>,
    pub(crate) pipeline_texture: wgpu::Texture,
    pub(crate) pipeline_texture_view: wgpu::TextureView,
    pub(crate) pipeline_texture_size: wgpu::Extent3d,
}

struct CompositorState {
    context: Arc<WgpuContext>,
    pipelines: WgpuPipelines,
    atlas: Arc<WgpuAtlas>,
    atlas_sampler: wgpu::Sampler,
    surface_sampler: wgpu::Sampler,
    pipeline_texture: wgpu::Texture,
    pipeline_texture_view: wgpu::TextureView,
    backdrop_blur_texture: wgpu::Texture,
    backdrop_blur_texture_view: wgpu::TextureView,
    backdrop_blur_sampler: wgpu::Sampler,
    group_textures: Vec<wgpu::Texture>,
    group_views: Vec<wgpu::TextureView>,
    rendering_parameters: RenderingParameters,
    job_rx: flume::Receiver<CompositorJob>,
    completion_tx: flume::Sender<CompositorCompletion>,
    previous_scene: Option<Scene>,
    first_frame: bool,
}

pub(crate) fn start_compositor(
    context: Arc<WgpuContext>,
    atlas: Arc<WgpuAtlas>,
    format: wgpu::TextureFormat,
    alpha_mode: wgpu::CompositeAlphaMode,
    width: u32,
    height: u32,
    path_sample_count: u32,
) -> (CompositorHandle, std::thread::JoinHandle<()>) {
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };

    let pipeline_texture = context.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("compositor_pipeline_texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let pipeline_texture_view = pipeline_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let backdrop_blur_sampler = context.device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("compositor_backdrop_blur_sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let backdrop_blur_texture = context.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("compositor_backdrop_blur_texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let backdrop_blur_texture_view =
        backdrop_blur_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let pipelines = WgpuPipelines::new(context.as_ref(), format, alpha_mode, path_sample_count);

    let (group_textures, group_views) =
        create_filter_group_textures(&context.device, width, height, format);

    let (job_tx, job_rx) = flume::bounded::<CompositorJob>(1);
    let (completion_tx, completion_rx) = flume::bounded::<CompositorCompletion>(1);

    let atlas_sampler = context.device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("compositor_atlas_sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let surface_sampler = context.device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("compositor_surface_sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let state = CompositorState {
        context: context.clone(),
        pipelines,
        atlas: atlas.clone(),
        atlas_sampler,
        surface_sampler,
        pipeline_texture: pipeline_texture.clone(),
        pipeline_texture_view: pipeline_texture_view.clone(),
        backdrop_blur_texture,
        backdrop_blur_texture_view,
        backdrop_blur_sampler,
        group_textures,
        group_views,
        rendering_parameters: RenderingParameters::from_env(),
        job_rx,
        completion_tx,
        previous_scene: None,
        first_frame: true,
    };

    let join_handle = std::thread::Builder::new()
        .name("wgpui-compositor".to_string())
        .spawn(move || compositor_main(state))
        .expect("failed to spawn compositor thread");

    let handle = CompositorHandle {
        job_tx,
        completion_rx,
        pipeline_texture,
        pipeline_texture_view,
        pipeline_texture_size: size,
    };

    (handle, join_handle)
}

fn compositor_main(mut state: CompositorState) {
    log::info!("Compositor thread started");

    loop {
        let job = match state.job_rx.recv() {
            Ok(job) => job,
            Err(_) => {
                log::info!("Compositor thread: channel closed, shutting down");
                break;
            }
        };

        match job {
            CompositorJob::Commit(scene) => {
                process_commit(&mut state, scene);
            }
            CompositorJob::Resize(width, height) => {
                resize_compositor(&mut state, width, height);
            }
            CompositorJob::Shutdown => {
                log::info!("Compositor thread shutting down");
                break;
            }
        }
    }

    log::info!("Compositor thread exited");
}

fn resize_compositor(state: &mut CompositorState, width: u32, height: u32) {
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let format = state.pipeline_texture.format();

    state.pipeline_texture = state
        .context
        .device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("compositor_pipeline_texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
    state.pipeline_texture_view =
        state
            .pipeline_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

    state.backdrop_blur_texture = state
        .context
        .device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("compositor_backdrop_blur_texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
    state.backdrop_blur_texture_view =
        state
            .backdrop_blur_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

    let (group_textures, group_views) =
        create_filter_group_textures(&state.context.device, width, height, format);
    state.group_textures = group_textures;
    state.group_views = group_views;

    state.first_frame = true;
}

fn process_commit(state: &mut CompositorState, scene: Scene) {
    let mut command_encoder =
        state
            .context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("compositor"),
            });

    state.atlas.before_frame(&mut command_encoder);

    let mut seen_surfaces: Vec<crate::platform::cross::surface_registry::SurfaceId> = Vec::new();
    let mut surface_views: Vec<wgpu::TextureView> = Vec::new();
    let mut surface_param_buffers: Vec<wgpu::Buffer> = Vec::new();

    let color_adjustments = ColorAdjustments {
        gamma_ratios: state.rendering_parameters.gamma_ratios,
        grayscale_enhanced_contrast: state.rendering_parameters.grayscale_enhanced_contrast,
        _padding: [0.0; 3],
    };
    state.context.queue.write_buffer(
        &state.context.color_adjustments_buffer,
        0,
        bytemuck::bytes_of(&color_adjustments),
    );

    let viewport_size = [
        state.pipeline_texture.width() as f32,
        state.pipeline_texture.height() as f32,
    ];

    let globals = GlobalParams {
        viewport_size,
        premultimated_alpha: 0,
        pad: 0,
    };

    state
        .context
        .queue
        .write_buffer(&state.context.globals_buffer, 0, bytemuck::bytes_of(&globals));

    if !scene.quads.is_empty() {
        let data = unsafe { as_bytes(&scene.quads) };
        crate::platform::cross::render_context::ensure_buffer_size(
            &state.context.device,
            &state.context.quads_buffer,
            data.len() as u64,
            "Quads Buffer",
            wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE,
        );
        state
            .context
            .queue
            .write_buffer(&state.context.quads_buffer.lock().unwrap(), 0, data);
    }
    if !scene.shadows.is_empty() {
        let data = unsafe { as_bytes(&scene.shadows) };
        crate::platform::cross::render_context::ensure_buffer_size(
            &state.context.device,
            &state.context.shadows_buffer,
            data.len() as u64,
            "Shadows Buffer",
            wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE,
        );
        state
            .context
            .queue
            .write_buffer(&state.context.shadows_buffer.lock().unwrap(), 0, data);
    }
    if !scene.backdrop_filters.is_empty() {
        let data = unsafe { as_bytes(&scene.backdrop_filters) };
        crate::platform::cross::render_context::ensure_buffer_size(
            &state.context.device,
            &state.context.backdrop_filters_buffer,
            data.len() as u64,
            "Backdrop Filters Buffer",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        state.context.queue.write_buffer(
            &state.context.backdrop_filters_buffer.lock().unwrap(),
            0,
            data,
        );
    }
    if !scene.underlines.is_empty() {
        let data = unsafe { as_bytes(&scene.underlines) };
        crate::platform::cross::render_context::ensure_buffer_size(
            &state.context.device,
            &state.context.underlines_buffer,
            data.len() as u64,
            "Underlines Buffer",
            wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE,
        );
        state.context.queue.write_buffer(
            &state.context.underlines_buffer.lock().unwrap(),
            0,
            data,
        );
    }
    if !scene.monochrome_sprites.is_empty() {
        let data = unsafe { as_bytes(&scene.monochrome_sprites) };
        crate::platform::cross::render_context::ensure_buffer_size(
            &state.context.device,
            &state.context.mono_sprites_buffer,
            data.len() as u64,
            "Monosprites Buffer",
            wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE,
        );
        state.context.queue.write_buffer(
            &state.context.mono_sprites_buffer.lock().unwrap(),
            0,
            data,
        );
    }
    if !scene.polychrome_sprites.is_empty() {
        let data = unsafe { as_bytes(&scene.polychrome_sprites) };
        crate::platform::cross::render_context::ensure_buffer_size(
            &state.context.device,
            &state.context.poly_sprites_buffer,
            data.len() as u64,
            "Poly Sprites Buffer",
            wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE,
        );
        state.context.queue.write_buffer(
            &state.context.poly_sprites_buffer.lock().unwrap(),
            0,
            data,
        );
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
        crate::platform::cross::render_context::ensure_buffer_size(
            &state.context.device,
            &state.context.paths_vertices_buffer,
            data.len() as u64,
            "Path Vertices Buffer",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        state.context.queue.write_buffer(
            &state.context.paths_vertices_buffer.lock().unwrap(),
            0,
            data,
        );
    }

    let quads_buffer_ref = state.context.quads_buffer.lock().unwrap();
    let shadows_buffer_ref = state.context.shadows_buffer.lock().unwrap();
    let backdrop_filters_buffer_ref = state.context.backdrop_filters_buffer.lock().unwrap();
    let underlines_buffer_ref = state.context.underlines_buffer.lock().unwrap();
    let mono_sprites_buffer_ref = state.context.mono_sprites_buffer.lock().unwrap();
    let poly_sprites_buffer_ref = state.context.poly_sprites_buffer.lock().unwrap();
    let paths_vertices_buffer_ref = state.context.paths_vertices_buffer.lock().unwrap();

    let quads_bind_group = state
        .context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compositor_quads_bind_group"),
            layout: &state.pipelines.quads_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &quads_buffer_ref,
                    offset: 0,
                    size: None,
                }),
            }],
        });

    let shadows_bind_group = state
        .context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compositor_shadows_bind_group"),
            layout: &state.pipelines.shadows_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &shadows_buffer_ref,
                    offset: 0,
                    size: None,
                }),
            }],
        });

    let backdrop_filters_bind_group =
        state
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("compositor_backdrop_filters_bind_group"),
                layout: &state.pipelines.backdrop_filters_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &backdrop_filters_buffer_ref,
                        offset: 0,
                        size: None,
                    }),
                }],
            });

    let backdrop_texture_bind_group =
        state
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("compositor_backdrop_texture_bind_group"),
                layout: &state.pipelines.backdrop_texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &state.backdrop_blur_texture_view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&state.backdrop_blur_sampler),
                    },
                ],
            });

    let underlines_bind_group = state
        .context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compositor_underlines_bind_group"),
            layout: &state.pipelines.underlines_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &underlines_buffer_ref,
                    offset: 0,
                    size: None,
                }),
            }],
        });

    let mono_sprites_bind_group = state
        .context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compositor_mono_sprites_bind_group"),
            layout: &state.pipelines.mono_sprites_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &mono_sprites_buffer_ref,
                    offset: 0,
                    size: None,
                }),
            }],
        });

    let poly_sprites_bind_group = state
        .context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compositor_poly_sprites_bind_group"),
            layout: &state.pipelines.poly_sprites_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &poly_sprites_buffer_ref,
                    offset: 0,
                    size: None,
                }),
            }],
        });

    let paths_bind_group = state
        .context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compositor_paths_bind_group"),
            layout: &state.pipelines.paths_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &paths_vertices_buffer_ref,
                    offset: 0,
                    size: None,
                }),
            }],
        });

    {
        let load_op = if state.first_frame {
            wgpu::LoadOp::Clear(wgpu::Color::BLACK)
        } else {
            wgpu::LoadOp::Load
        };
        state.first_frame = false;

        let mut pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("compositor_main"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &state.pipeline_texture_view,
                ops: wgpu::Operations {
                    load: load_op,
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

        let mut quads_first_instance: u32 = 0;
        let mut shadows_first_instance: u32 = 0;
        let mut backdrop_filters_first_instance: u32 = 0;
        let mut underlines_first_instance: u32 = 0;
        let mut mono_sprites_first_instance: u32 = 0;
        let mut poly_sprites_first_instance: u32 = 0;
        let mut paths_vertex_offset: u32 = 0;

        let mut filter_stack: Vec<(FilterBoundary, Option<usize>)> = Vec::new();

        for batch in scene.batches() {
            match batch {
                PrimitiveBatch::Quads(quads) => {
                    let count = quads.len() as u32;
                    pass.set_pipeline(&state.pipelines.quads_pipeline);
                    pass.set_bind_group(0, &state.pipelines.globals_bind_group, &[]);
                    pass.set_bind_group(1, &quads_bind_group, &[]);
                    pass.draw(0..4, quads_first_instance..quads_first_instance + count);
                    quads_first_instance += count;
                }

                PrimitiveBatch::MonochromeSprites {
                    texture_id,
                    sprites,
                } => {
                    let count = sprites.len() as u32;
                    let tex_info = state.atlas.get_texture_info(texture_id);

                    let sprites_texture_bind_group =
                        state
                            .context
                            .device
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("compositor_sprites_bind_group"),
                                layout: &state.pipelines.sprites_bind_group_layout,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: wgpu::BindingResource::TextureView(
                                            &tex_info.raw_view,
                                        ),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: wgpu::BindingResource::Sampler(
                                            &state.atlas_sampler,
                                        ),
                                    },
                                ],
                            });

                    pass.set_pipeline(&state.pipelines.mono_sprites_pipeline);
                    pass.set_bind_group(0, &state.pipelines.globals_bind_group, &[]);
                    pass.set_bind_group(1, &state.pipelines.color_adjustments_bind_group, &[]);
                    pass.set_bind_group(2, &sprites_texture_bind_group, &[]);
                    pass.set_bind_group(3, &mono_sprites_bind_group, &[]);
                    pass.draw(
                        0..4,
                        mono_sprites_first_instance..mono_sprites_first_instance + count,
                    );
                    mono_sprites_first_instance += count;
                }

                PrimitiveBatch::PolychromeSprites {
                    texture_id,
                    sprites,
                } => {
                    let count = sprites.len() as u32;
                    let tex_info = state.atlas.get_texture_info(texture_id);

                    let sprites_texture_bind_group =
                        state
                            .context
                            .device
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("compositor_poly_sprites_texture_bind_group"),
                                layout: &state.pipelines.sprites_bind_group_layout,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: wgpu::BindingResource::TextureView(
                                            &tex_info.raw_view,
                                        ),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: wgpu::BindingResource::Sampler(
                                            &state.atlas_sampler,
                                        ),
                                    },
                                ],
                            });

                    pass.set_pipeline(&state.pipelines.poly_sprites_pipeline);
                    pass.set_bind_group(0, &state.pipelines.globals_bind_group, &[]);
                    pass.set_bind_group(1, &sprites_texture_bind_group, &[]);
                    pass.set_bind_group(2, &poly_sprites_bind_group, &[]);
                    pass.draw(
                        0..4,
                        poly_sprites_first_instance..poly_sprites_first_instance + count,
                    );
                    poly_sprites_first_instance += count;
                }

                PrimitiveBatch::Shadows(shadows) => {
                    let count = shadows.len() as u32;
                    pass.set_pipeline(&state.pipelines.shadows_pipeline);
                    pass.set_bind_group(0, &state.pipelines.globals_bind_group, &[]);
                    pass.set_bind_group(1, &shadows_bind_group, &[]);
                    pass.draw(0..4, shadows_first_instance..shadows_first_instance + count);
                    shadows_first_instance += count;
                }

                PrimitiveBatch::BackdropFilters(backdrop_filters) => {
                    let count = backdrop_filters.len() as u32;

                    drop(pass);

                    command_encoder.copy_texture_to_texture(
                        state.pipeline_texture.as_image_copy(),
                        state.backdrop_blur_texture.as_image_copy(),
                        state.pipeline_texture.size(),
                    );

                    pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("compositor_backdrop_resumed"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &state.pipeline_texture_view,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
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

                    pass.set_pipeline(&state.pipelines.backdrop_filters_pipeline);
                    pass.set_bind_group(0, &state.pipelines.globals_bind_group, &[]);
                    pass.set_bind_group(1, &backdrop_filters_bind_group, &[]);
                    pass.set_bind_group(2, &backdrop_texture_bind_group, &[]);
                    pass.draw(
                        0..4,
                        backdrop_filters_first_instance
                            ..backdrop_filters_first_instance + count,
                    );
                    backdrop_filters_first_instance += count;
                }

                PrimitiveBatch::FilterBoundary(index) => {
                    let boundary = scene.filter_boundaries[index];

                    if boundary.is_start {
                        let depth = filter_stack.len();
                        if depth >= state.group_textures.len() {
                            filter_stack.push((boundary, None));
                        } else {
                            drop(pass);

                            pass = command_encoder.begin_render_pass(
                                &wgpu::RenderPassDescriptor {
                                    label: Some("compositor_filter_group"),
                                    color_attachments: &[Some(
                                        wgpu::RenderPassColorAttachment {
                                            view: &state.group_views[depth],
                                            ops: wgpu::Operations {
                                                load: wgpu::LoadOp::Clear(
                                                    wgpu::Color::TRANSPARENT,
                                                ),
                                                store: wgpu::StoreOp::Store,
                                            },
                                            resolve_target: None,
                                            depth_slice: None,
                                        },
                                    )],
                                    depth_stencil_attachment: None,
                                    timestamp_writes: None,
                                    occlusion_query_set: None,
                                    multiview_mask: None,
                                },
                            );

                            filter_stack.push((boundary, Some(depth)));
                        }
                    } else {
                        let Some((start_boundary, depth)) = filter_stack.pop() else {
                            continue;
                        };

                        let Some(depth) = depth else {
                            continue;
                        };

                        drop(pass);

                        let parent_view: &wgpu::TextureView = match filter_stack.last() {
                            Some((_, Some(parent_depth))) => &state.group_views[*parent_depth],
                            _ => &state.pipeline_texture_view,
                        };

                        pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("compositor_filter_group_resumed"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: parent_view,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
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

                        let composite = BackdropFilter {
                            order: 0,
                            bounds: start_boundary.bounds,
                            content_mask: start_boundary.content_mask,
                            corner_radii: start_boundary.corner_radii,
                            blur_radius: start_boundary.blur_radius,
                            opacity: start_boundary.opacity,
                            _pad: 0,
                        };
                        let composite_buffer = state.context.device.create_buffer_init(
                            &wgpu::util::BufferInitDescriptor {
                                label: Some("compositor_filter_group_composite_buffer"),
                                contents: unsafe { as_bytes(std::slice::from_ref(&composite)) },
                                usage: wgpu::BufferUsages::STORAGE,
                            },
                        );
                        let composite_bind_group = state.context.device.create_bind_group(
                            &wgpu::BindGroupDescriptor {
                                label: Some("compositor_filter_group_composite_bind_group"),
                                layout: &state.pipelines.backdrop_filters_bind_group_layout,
                                entries: &[wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::Buffer(
                                        wgpu::BufferBinding {
                                            buffer: &composite_buffer,
                                            offset: 0,
                                            size: None,
                                        },
                                    ),
                                }],
                            },
                        );
                        let composite_texture_bind_group = state.context.device.create_bind_group(
                            &wgpu::BindGroupDescriptor {
                                label: Some("compositor_filter_group_texture_bind_group"),
                                layout: &state.pipelines.backdrop_texture_bind_group_layout,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: wgpu::BindingResource::TextureView(
                                            &state.group_views[depth],
                                        ),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: wgpu::BindingResource::Sampler(
                                            &state.backdrop_blur_sampler,
                                        ),
                                    },
                                ],
                            },
                        );

                        pass.set_pipeline(&state.pipelines.backdrop_filters_pipeline);
                        pass.set_bind_group(0, &state.pipelines.globals_bind_group, &[]);
                        pass.set_bind_group(1, &composite_bind_group, &[]);
                        pass.set_bind_group(2, &composite_texture_bind_group, &[]);
                        pass.draw(0..4, 0..1);
                    }
                }

                PrimitiveBatch::Underlines(underlines) => {
                    let count = underlines.len() as u32;
                    pass.set_pipeline(&state.pipelines.underlines_pipeline);
                    pass.set_bind_group(0, &state.pipelines.globals_bind_group, &[]);
                    pass.set_bind_group(1, &underlines_bind_group, &[]);
                    pass.draw(
                        0..4,
                        underlines_first_instance..underlines_first_instance + count,
                    );
                    underlines_first_instance += count;
                }

                PrimitiveBatch::Surfaces(surfaces) => {
                    for surface in surfaces {
                        if let crate::SurfaceContent::Wgpu(surface_id) = &surface.content {
                            let _swapped = state
                                .context
                                .surface_registry
                                .swap_ready_display(&state.context.device, *surface_id);

                            if let Some(view) =
                                state.context.surface_registry.front_view(*surface_id)
                            {
                                let params = crate::platform::cross::renderer::SurfaceParams {
                                    bounds: crate::platform::cross::renderer::Bounds {
                                        origin: [
                                            surface.bounds.origin.x.0,
                                            surface.bounds.origin.y.0,
                                        ],
                                        size: [
                                            surface.bounds.size.width.0,
                                            surface.bounds.size.height.0,
                                        ],
                                    },
                                    content_mask: crate::platform::cross::renderer::Bounds {
                                        origin: [
                                            surface.content_mask.bounds.origin.x.0,
                                            surface.content_mask.bounds.origin.y.0,
                                        ],
                                        size: [
                                            surface.content_mask.bounds.size.width.0,
                                            surface.content_mask.bounds.size.height.0,
                                        ],
                                    },
                                };

                                let params_buffer = state.context.device.create_buffer_init(
                                    &wgpu::util::BufferInitDescriptor {
                                        label: Some("compositor_surface_params_buffer"),
                                        contents: bytemuck::bytes_of(&params),
                                        usage: wgpu::BufferUsages::UNIFORM,
                                    },
                                );

                                let surface_bind_group = state.context.device.create_bind_group(
                                    &wgpu::BindGroupDescriptor {
                                        label: Some("compositor_surface_bind_group"),
                                        layout: &state.pipelines.surfaces_bind_group_layout,
                                        entries: &[
                                            wgpu::BindGroupEntry {
                                                binding: 0,
                                                resource: wgpu::BindingResource::Buffer(
                                                    wgpu::BufferBinding {
                                                        buffer: &params_buffer,
                                                        offset: 0,
                                                        size: None,
                                                    },
                                                ),
                                            },
                                            wgpu::BindGroupEntry {
                                                binding: 1,
                                                resource: wgpu::BindingResource::TextureView(
                                                    &view,
                                                ),
                                            },
                                            wgpu::BindGroupEntry {
                                                binding: 2,
                                                resource: wgpu::BindingResource::Sampler(
                                                    &state.surface_sampler,
                                                ),
                                            },
                                        ],
                                    },
                                );

                                pass.set_pipeline(&state.pipelines.surfaces_pipeline);
                                pass.set_bind_group(0, &state.pipelines.globals_bind_group, &[]);
                                pass.set_bind_group(1, &surface_bind_group, &[]);
                                pass.draw(0..4, 0..1);

                                surface_views.push(view);
                                surface_param_buffers.push(params_buffer);

                                state
                                    .context
                                    .surface_registry
                                    .clear_redraw_pending(*surface_id);

                                seen_surfaces.push(*surface_id);
                            }
                        }
                    }
                }

                PrimitiveBatch::Paths(paths) => {
                    let vertex_count: u32 =
                        paths.iter().map(|p| p.vertices.len() as u32).sum();
                    if vertex_count > 0 {
                        pass.set_pipeline(&state.pipelines.paths_pipeline);
                        pass.set_bind_group(0, &state.pipelines.globals_bind_group, &[]);
                        pass.set_bind_group(1, &paths_bind_group, &[]);
                        pass.draw(
                            paths_vertex_offset..paths_vertex_offset + vertex_count,
                            0..1,
                        );
                        paths_vertex_offset += vertex_count;
                    }
                }
            }
        }
    }

    state.context.queue.submit(Some(command_encoder.finish()));
    state.previous_scene = Some(scene);

    let _ = state.completion_tx.try_send(CompositorCompletion {
        frame_ready: true,
    });
}
