//! One frame, assembled: upload, compute, indirect draw.
//! See docs/gpu-native-architecture.md §2's picture, §5.1-§5.3, §8 Phase 4.
//!
//! # Not in §3.5's file map, and why
//!
//! §3.5 lists device/queue creation, pipelines, compute dispatch, atlas,
//! textures, and draw issuance — every *stage*, and no home for the thing that
//! runs them in order. In the legacy backend that thing is `renderer.rs`'s
//! `draw` method, interleaved with the stages it drives across several hundred
//! lines. Phase 4 needs it separate for the same reason Phase 2 needed
//! `patch/emit.rs` separate: §8's gate is a claim about *what a frame did*, and
//! a claim like that is only checkable if a frame is a value something returns
//! rather than an effect something has.
//!
//! This is the third such deviation across four phases (Phase 1's six, Phase 2's
//! two, Phase 3's four), recorded here in the same shape.
//!
//! # What a frame does, in order
//!
//! 1. **Upload** the dirty layers' bytes through §5.0's instructions.
//! 2. **Compute**, per dirty layer only (R-N §8.2's rule, unchanged): ordering
//!    and occlusion, scattered into the arena-shaped buffers Phase 4 reads.
//!    A clean layer's results from a previous frame are still sitting there.
//! 3. **Generate arguments** — one dispatch per kind, over every slot.
//! 4. **Issue** the fixed draw sequence.
//!
//! [`FrameTiming`] measures each separately, and step 4 alone is what §8's
//! Phase 4 gate is about. Its clock covers command *encoding* on the CPU — set
//! pipeline, set bind group, issue draw — and deliberately not the GPU work
//! those commands cause, because the gate's claim is about the CPU's cost.
//!
//! # The clean-frame path, which is the one the gate measures
//!
//! [`FrameInput::dirty`] naming no layers means steps 1 and 2 do nothing at
//! all: no upload, no dispatch, no readback, no walk over any primitive. Step 3
//! still runs (arguments are cheap and a slot's contents may have been culled
//! differently) and step 4 always runs, in full, because §5.3's sequence is
//! fixed. That is the shape of a window sitting still, and it is the shape the
//! benchmark drives at two very different primitive counts.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use wgpui_core::boundary::compositor::CompositeEntry;
use wgpui_core::geometry::Rect;
use wgpui_core::indirect::{DrawSlot, QUAD_VERTEX_COUNT, encode_slots};
use wgpui_core::occlusion::{
    CoverageItem, PoisonRegion, encode_coverage_items, encode_poison_regions, quad_coverage_item,
};
use wgpui_core::ordering::encode_ordering_items;
use wgpui_core::patch::primitive::{PrimitiveKind, Quad};
use wgpui_core::scene::atlas::AtlasKind;
use wgpui_core::scene::layer::LayerId;
use wgpui_core::scene::{Scene, UploadRange};

use crate::render::atlas_upload::AtlasTextures;
use crate::render::buffers::slab_buffers::SlabBuffer;
use crate::render::compute::indirect_args_pass::{
    IndirectArgsBuffers, IndirectArgsError, IndirectArgsPass,
};
use crate::render::compute::occlusion_pass::{OcclusionError, OcclusionPass};
use crate::render::compute::ordering_pass::{OrderingError, OrderingPass};
use crate::render::draw::{
    DrawMode, DrawStats, GlyphDraw, ResolvedArgs, SlotBasePlan, issue_composites, issue_glyphs,
    issue_quads,
};
use crate::render::pipelines::{
    CompositePipeline, Globals, MonoSpritePipeline, QuadPipeline, TARGET_FORMAT,
};
use crate::render::readback::{ReadbackError, StagingReader};
use crate::render::surface_registry::SurfaceRegistry;
use crate::render::textures::external_surface::{
    CompositeConsumer, CompositePlan, plan_composites,
};
use crate::render::textures::layer_texture::LayerTexturePool;

/// Why a frame could not be rendered.
#[derive(Debug)]
pub enum FrameError {
    /// The ordering pass failed.
    Ordering(OrderingError),
    /// The occlusion pass failed.
    Occlusion(OcclusionError),
    /// The indirect-arg pass failed.
    IndirectArgs(IndirectArgsError),
    /// A readback failed.
    Readback(ReadbackError),
}

impl std::fmt::Display for FrameError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FrameError::Ordering(error) => write!(formatter, "{error}"),
            FrameError::Occlusion(error) => write!(formatter, "{error}"),
            FrameError::IndirectArgs(error) => write!(formatter, "{error}"),
            FrameError::Readback(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for FrameError {}

impl From<OrderingError> for FrameError {
    fn from(error: OrderingError) -> Self {
        FrameError::Ordering(error)
    }
}

impl From<OcclusionError> for FrameError {
    fn from(error: OcclusionError) -> Self {
        FrameError::Occlusion(error)
    }
}

impl From<IndirectArgsError> for FrameError {
    fn from(error: IndirectArgsError) -> Self {
        FrameError::IndirectArgs(error)
    }
}

impl From<ReadbackError> for FrameError {
    fn from(error: ReadbackError) -> Self {
        FrameError::Readback(error)
    }
}

/// Which layers changed this frame.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Dirty<'a> {
    /// Every layer — a first frame, or a resize.
    All,
    /// Exactly these, and no others.
    Some(&'a [LayerId]),
}

impl Dirty<'_> {
    fn contains(&self, layer: LayerId) -> bool {
        match self {
            Dirty::All => true,
            Dirty::Some(layers) => layers.contains(&layer),
        }
    }

    fn is_empty(&self) -> bool {
        matches!(self, Dirty::Some(layers) if layers.is_empty())
    }
}

/// What a frame is asked to draw.
pub struct FrameInput<'a> {
    /// The resident scene.
    pub scene: &'a Scene,
    /// The window rectangle every primitive clips to.
    pub clip: Rect,
    /// The frame's filter regions.
    pub poison: &'a [PoisonRegion],
    /// Which layers changed.
    pub dirty: Dirty<'a>,
    /// Upload instructions for the dirty layers' bytes, per §5.0.
    pub uploads: &'a [UploadRange],
    /// The composite entries, in draw order (§5.5).
    pub composites: &'a [CompositeEntry],
    /// The externally-owned surfaces, if any entry names one.
    pub registry: Option<&'a SurfaceRegistry>,
    /// The uploaded atlas pages a glyph draw samples, if this frame has text.
    ///
    /// Borrowed rather than owned by the renderer because the atlas is shared:
    /// the same [`AtlasTextures`] serves every window, and the phase that
    /// rasterises into it (`render/atlas.rs`, `render/atlas_upload.rs`) is not
    /// this one. `None` — or a set with no monochrome page in it — is an
    /// ordinary frame with no rasterised text, not an error.
    pub atlas: Option<&'a AtlasTextures>,
    /// Framebuffer size in pixels.
    pub viewport: [f32; 2],
    /// How the fixed draw sequence reaches the device.
    pub mode: DrawMode,
}

/// What each stage of one frame cost the CPU, in wall clock.
///
/// Every clock here measures CPU work: encoding commands, writing buffers,
/// waiting on a readback. None of them measures how long the GPU then took, so
/// none of them is a frame time — see this module's doc.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct FrameTiming {
    /// Applying §5.0's upload instructions.
    pub upload: Duration,
    /// Encoding and dispatching the dirty layers' ordering and occlusion.
    pub compute: Duration,
    /// Encoding and dispatching indirect-argument generation.
    pub arguments: Duration,
    /// **The gate's own clock**: recording the fixed draw sequence.
    pub draw_issue: Duration,
    /// Reading arguments back, on the fallback path only.
    pub readback: Duration,
}

/// What one frame did.
#[derive(Clone, Debug)]
pub struct FrameOutput {
    /// The counters §8's Phase 4 gate reads.
    pub stats: DrawStats,
    /// What each stage cost.
    pub timing: FrameTiming,
    /// Layers whose ordering and occlusion were recomputed.
    pub layers_recomputed: u32,
    /// Primitives resident in the scene — the gate's independent variable,
    /// carried so a report cannot quote a draw count without the count it is
    /// supposed to be independent of.
    pub primitives_resident: u32,
}

/// An offscreen colour target plus the readback its comparisons need.
pub struct OffscreenTarget {
    texture: wgpu::Texture,
    /// The view render passes attach.
    pub view: wgpu::TextureView,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl OffscreenTarget {
    /// Allocate a target.
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> OffscreenTarget {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("frame target"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
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
        OffscreenTarget {
            texture,
            view,
            width: width.max(1),
            height: height.max(1),
        }
    }

    /// Read the target back as tightly-packed RGBA8 rows.
    ///
    /// Row padding to `COPY_BYTES_PER_ROW_ALIGNMENT` is undone here rather than
    /// left to every caller, because a comparison that forgot to would report
    /// two identical images as differing in their padding.
    pub fn read_pixels(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<u8>, ReadbackError> {
        let unpadded = self.width * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = unpadded.div_ceil(align) * align;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame target readback"),
            size: u64::from(padded) * u64::from(self.height),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("frame target readback"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            if sender.send(result).is_err() {
                log::warn!("wgpui-wgpu: frame readback completed after its receiver was dropped");
            }
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(ReadbackError::Poll)?;
        receiver
            .recv()
            .map_err(|_| ReadbackError::Cancelled)?
            .map_err(ReadbackError::Map)?;

        let pixels = {
            let view = slice.get_mapped_range().map_err(ReadbackError::Range)?;
            let mut pixels = Vec::with_capacity((unpadded * self.height) as usize);
            for row in 0..self.height {
                let start = (row * padded) as usize;
                if let Some(bytes) = view.get(start..start + unpadded as usize) {
                    pixels.extend_from_slice(bytes);
                }
            }
            pixels
        };
        staging.unmap();
        Ok(pixels)
    }
}

/// Everything a frame needs that outlives one: pipelines, arenas, pools.
pub struct FrameRenderer {
    ordering: OrderingPass,
    occlusion: OcclusionPass,
    indirect: IndirectArgsPass,
    /// The instanced quad pipeline.
    pub quads: QuadPipeline,
    /// The instanced monochrome-sprite pipeline (Phase 5.6).
    pub glyphs: MonoSpritePipeline,
    /// The one composite pipeline.
    pub composite: CompositePipeline,
    /// The quad arena.
    pub arena: SlabBuffer,
    /// The glyph arena. A second buffer rather than a second range of the
    /// first: the two kinds have different slot strides, and §5.0's upload
    /// instructions are already addressed per kind.
    pub glyph_arena: SlabBuffer,
    /// Boundary texture retention (§5.5).
    pub textures: LayerTexturePool,
    globals: wgpu::Buffer,
    quad_args: IndirectArgsBuffers,
    glyph_args: IndirectArgsBuffers,
    composite_args: IndirectArgsBuffers,
    /// The per-slot bases and their bind group, kept until the slot table
    /// itself changes — see [`SlotBasePlan`], which is only true of it if
    /// something holds it across frames.
    quad_plan: Option<SlotBasePlan>,
    glyph_plan: Option<SlotBasePlan>,
    quad_plan_builds: u64,
    glyph_plan_builds: u64,
    /// One 16-byte `AtlasPage` uniform per page index ever bound.
    ///
    /// Keyed by page index and never invalidated, because the value is the page
    /// index itself: a page destroyed and recreated is the same number and the
    /// same bytes. The bind group over it is *not* cached, because that names a
    /// texture view which a destroyed page invalidates.
    page_params: HashMap<u32, wgpu::Buffer>,
    reader: StagingReader,
    uploaded_generation: Option<u64>,
}

impl FrameRenderer {
    /// Build every pipeline once.
    pub fn new(device: &wgpu::Device) -> FrameRenderer {
        FrameRenderer {
            ordering: OrderingPass::new(device),
            occlusion: OcclusionPass::new(device),
            indirect: IndirectArgsPass::new(device),
            quads: QuadPipeline::new(device),
            glyphs: MonoSpritePipeline::new(device),
            composite: CompositePipeline::new(device),
            arena: SlabBuffer::new(device, "quad arena"),
            glyph_arena: SlabBuffer::new(device, "glyph arena"),
            textures: LayerTexturePool::default(),
            globals: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("frame globals"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            quad_args: IndirectArgsBuffers::new(device, 1, 1),
            glyph_args: IndirectArgsBuffers::new(device, 1, 1),
            composite_args: IndirectArgsBuffers::new(device, 1, 1),
            quad_plan: None,
            glyph_plan: None,
            quad_plan_builds: 0,
            glyph_plan_builds: 0,
            page_params: HashMap::new(),
            reader: StagingReader::new(),
            uploaded_generation: None,
        }
    }

    /// How many times the readback staging buffer has been allocated.
    pub fn readback_allocations(&self) -> u64 {
        self.reader.allocations()
    }

    /// How many times the quad draw plan has been built.
    ///
    /// A frame loop whose residency does not change must not keep raising this
    /// — that is what [`SlotBasePlan`]'s "per slot-table change rather than per
    /// frame" means, and a counter is what makes it checkable rather than
    /// intended.
    pub fn draw_plan_builds(&self) -> u64 {
        self.quad_plan_builds
    }

    /// The same counter for the glyph pipeline's slot bases.
    pub fn glyph_plan_builds(&self) -> u64 {
        self.glyph_plan_builds
    }

    /// One bind group per live **monochrome** atlas page, in ascending page
    /// order.
    ///
    /// Colour pages are skipped rather than bound: `mono_sprites.wgsl` reads a
    /// single coverage channel, and handing it an `Rgba8Unorm` page would paint
    /// an emoji's red channel as if it were coverage. Colour glyphs need the
    /// `poly_sprites` pipeline, which no phase has built.
    fn atlas_page_bind_groups(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        atlas: Option<&AtlasTextures>,
    ) -> Vec<wgpu::BindGroup> {
        let Some(atlas) = atlas else {
            return Vec::new();
        };
        let mut groups = Vec::new();
        for page in atlas.pages_of_kind(AtlasKind::Monochrome) {
            let Some(view) = atlas.view(page) else {
                continue;
            };
            let params = self.page_params.entry(page).or_insert_with(|| {
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("mono sprites atlas page"),
                    size: MonoSpritePipeline::PAGE_PARAMS_SIZE,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let mut bytes = [0u8; MonoSpritePipeline::PAGE_PARAMS_SIZE as usize];
                bytes[0..4].copy_from_slice(&page.to_le_bytes());
                queue.write_buffer(&buffer, 0, &bytes);
                buffer
            });
            groups.push(self.glyphs.page_bind_group(device, params, view));
        }
        groups
    }

    /// Render one frame into `target`.
    ///
    /// **Both instanced kinds are drawn.** Phase 4 drew only `Quad` and said so
    /// here; Phase 5.6 added the `GlyphRun` half, which takes the identical
    /// route — upload, ordering, occlusion, indirect-argument generation, fixed
    /// draw sequence — over its own arena, with one addition that is genuinely
    /// its own: the pass repeats per bound atlas page, because a glyph's texture
    /// is chosen by its tile and a bind group cannot change inside a draw call.
    /// See [`issue_glyphs`].
    ///
    /// What still has no pipeline: `poly_sprites` (colour glyphs and images),
    /// `shadows`, `paths`, `underlines`, `backdrop_blur`.
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &FrameInput<'_>,
        target: &OffscreenTarget,
    ) -> Result<FrameOutput, FrameError> {
        self.textures.begin_frame();
        let mut timing = FrameTiming::default();

        let table = input.scene.draw_slots();
        let quad_slots: Vec<DrawSlot> = table.kind_slots(PrimitiveKind::Quad).to_vec();
        let glyph_slots: Vec<DrawSlot> = table.kind_slots(PrimitiveKind::GlyphRun).to_vec();
        let arena_slots = input.scene.arena_slots(PrimitiveKind::Quad);
        let glyph_arena_slots = input.scene.arena_slots(PrimitiveKind::GlyphRun);
        let primitives_resident: u32 = quad_slots
            .iter()
            .chain(glyph_slots.iter())
            .map(|slot| slot.count)
            .sum();

        // --- 1. Upload.
        let started = Instant::now();
        let resident = input.scene.quads.resident_bytes();
        let glyph_resident = input.scene.glyph_runs.resident_bytes();
        let grew = self.arena.reserve(device, resident.len() as u64);
        let glyphs_grew = self
            .glyph_arena
            .reserve(device, glyph_resident.len() as u64);
        if grew
            || glyphs_grew
            || self.uploaded_generation.is_none()
            || matches!(input.dirty, Dirty::All)
        {
            self.arena.upload_all(device, queue, resident);
            self.glyph_arena.upload_all(device, queue, glyph_resident);
            self.uploaded_generation = Some(0);
        } else {
            // Filtered by kind rather than handed the whole list: an
            // `UploadRange` is a byte span *within one kind's arena*, so
            // applying a glyph range to the quad buffer would overwrite an
            // unrelated primitive with a glyph's bytes. Before this phase there
            // was one arena and the filter was unnecessary; with two it is the
            // difference between a delta upload and corruption.
            self.arena
                .upload(device, queue, resident, &kind_uploads(input.uploads, PrimitiveKind::Quad));
            self.glyph_arena.upload(
                device,
                queue,
                glyph_resident,
                &kind_uploads(input.uploads, PrimitiveKind::GlyphRun),
            );
        }
        queue.write_buffer(
            &self.globals,
            0,
            &Globals {
                viewport: input.viewport,
            }
            .to_bytes(),
        );
        timing.upload = started.elapsed();

        if !self.quad_args.fits(arena_slots, quad_slots.len() as u32) {
            self.quad_args =
                IndirectArgsBuffers::new(device, arena_slots.max(1), quad_slots.len() as u32 + 1);
        }
        if !self
            .glyph_args
            .fits(glyph_arena_slots, glyph_slots.len() as u32)
        {
            self.glyph_args = IndirectArgsBuffers::new(
                device,
                glyph_arena_slots.max(1),
                glyph_slots.len() as u32 + 1,
            );
        }

        // --- 2. Compute, dirty layers only.
        let started = Instant::now();
        let mut layers_recomputed = 0u32;
        if !input.dirty.is_empty() {
            let mut bounds_bytes = Vec::new();
            let mut item_bytes = Vec::new();
            let mut poison_bytes = Vec::new();
            encode_poison_regions(input.poison, &mut poison_bytes);
            for slot in &quad_slots {
                if slot.count == 0 || !input.dirty.contains(slot.layer) {
                    continue;
                }
                let quads = layer_quads(input.scene, slot.layer);
                let bounds: Vec<Rect> = quads
                    .iter()
                    .map(|quad| Rect::from_origin_size(quad.origin, quad.size))
                    .collect();
                let items: Vec<_> = quads
                    .iter()
                    .map(|quad| quad_coverage_item(quad, input.clip, false))
                    .collect();

                encode_ordering_items(&bounds, &mut bounds_bytes);
                let ordered = self.ordering.run(device, queue, &bounds_bytes)?;
                encode_coverage_items(&items, &mut item_bytes);
                let culled = self
                    .occlusion
                    .run(device, queue, &item_bytes, &poison_bytes)?;

                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("scatter"),
                });
                IndirectArgsPass::scatter(
                    &mut encoder,
                    &ordered.draw_order,
                    &self.quad_args.draw_order,
                    slot.base,
                    slot.count,
                );
                IndirectArgsPass::scatter(
                    &mut encoder,
                    &culled.culled,
                    &self.quad_args.culled,
                    slot.base,
                    slot.count,
                );
                queue.submit(Some(encoder.finish()));
                layers_recomputed += 1;
            }

            // The glyph half, through the identical passes. A glyph is a
            // `CoverageItem::cullee` and never an occluder — a coverage mask is
            // not an opaque rectangle — so this dispatch culls glyphs against
            // the frame's poison regions and against nothing else. Cross-kind
            // occlusion (a glyph behind an opaque quad) is not expressible while
            // the dispatch is per kind; see docs/phase-5.6-results.md.
            for slot in &glyph_slots {
                if slot.count == 0 || !input.dirty.contains(slot.layer) {
                    continue;
                }
                let bounds = layer_glyph_bounds(input.scene, slot.layer);
                let items: Vec<CoverageItem> = bounds
                    .iter()
                    .map(|glyph| CoverageItem::cullee(glyph.intersect(&input.clip)))
                    .collect();

                encode_ordering_items(&bounds, &mut bounds_bytes);
                let ordered = self.ordering.run(device, queue, &bounds_bytes)?;
                encode_coverage_items(&items, &mut item_bytes);
                let culled = self
                    .occlusion
                    .run(device, queue, &item_bytes, &poison_bytes)?;

                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("glyph scatter"),
                });
                IndirectArgsPass::scatter(
                    &mut encoder,
                    &ordered.draw_order,
                    &self.glyph_args.draw_order,
                    slot.base,
                    slot.count,
                );
                IndirectArgsPass::scatter(
                    &mut encoder,
                    &culled.culled,
                    &self.glyph_args.culled,
                    slot.base,
                    slot.count,
                );
                queue.submit(Some(encoder.finish()));
                layers_recomputed += 1;
            }
        }
        timing.compute = started.elapsed();

        // --- 3. Arguments.
        let started = Instant::now();
        let mut slot_bytes = Vec::new();
        encode_slots(&quad_slots, &mut slot_bytes);
        let quad_output = self.indirect.run(
            device,
            queue,
            &self.quad_args,
            &slot_bytes,
            QUAD_VERTEX_COUNT,
            input.mode.first_instance(),
        )?;
        let mut glyph_slot_bytes = Vec::new();
        encode_slots(&glyph_slots, &mut glyph_slot_bytes);
        // The same four-vertex triangle strip: a glyph's sprite is a quad, so
        // §5.3's `QUAD_VERTEX_COUNT` is the vertex count for this kind too.
        let glyph_output = self.indirect.run(
            device,
            queue,
            &self.glyph_args,
            &glyph_slot_bytes,
            QUAD_VERTEX_COUNT,
            input.mode.first_instance(),
        )?;

        let composite_plan = plan_composites(
            device,
            queue,
            &self.composite,
            &CompositeConsumer {
                registry: input.registry,
                textures: Some(&self.textures),
            },
            input.composites,
        );
        let composite_slots = CompositePlan::slots(input.composites.len());
        if !self
            .composite_args
            .fits(composite_slots.len() as u32, composite_slots.len() as u32)
        {
            self.composite_args = IndirectArgsBuffers::new(
                device,
                composite_slots.len().max(1) as u32,
                composite_slots.len() as u32 + 1,
            );
        }
        if !composite_slots.is_empty() {
            queue.write_buffer(
                &self.composite_args.draw_order,
                0,
                &words_to_bytes(&composite_plan.draw_order),
            );
            queue.write_buffer(
                &self.composite_args.culled,
                0,
                &words_to_bytes(&composite_plan.culled_mask),
            );
        }
        let mut composite_slot_bytes = Vec::new();
        encode_slots(&composite_slots, &mut composite_slot_bytes);
        let composite_output = self.indirect.run(
            device,
            queue,
            &self.composite_args,
            &composite_slot_bytes,
            QUAD_VERTEX_COUNT,
            input.mode.first_instance(),
        )?;
        timing.arguments = started.elapsed();

        // --- 3b. Readback, on the fallback path only. Before the pass begins,
        // because it submits its own encoder and blocks.
        let started = Instant::now();
        let quad_resolved = ResolvedArgs::resolve(
            input.mode,
            device,
            queue,
            &self.quad_args,
            quad_output.slot_count,
            &mut self.reader,
        )?;
        let glyph_resolved = ResolvedArgs::resolve(
            input.mode,
            device,
            queue,
            &self.glyph_args,
            glyph_output.slot_count,
            &mut self.reader,
        )?;
        let composite_resolved = ResolvedArgs::resolve(
            input.mode,
            device,
            queue,
            &self.composite_args,
            composite_output.slot_count,
            &mut self.reader,
        )?;
        timing.readback = started.elapsed();

        // --- 4. Issue.
        // Rebuilt only when the slot table changes, which is what makes
        // `QuadDrawPlan`'s "per slot-table change rather than per frame" true:
        // the bases it holds are each layer's `SlabRange`, so a frame that
        // changed contents but not residency reuses the buffer and the bind
        // group untouched. The clean frame the gate measures is exactly that
        // frame.
        let quad_plan = match self.quad_plan.take() {
            Some(plan) if plan.slots() == quad_slots.as_slice() => plan,
            _ => {
                self.quad_plan_builds += 1;
                SlotBasePlan::for_quads(device, queue, &self.quads, &quad_slots)
            }
        };
        let glyph_plan = match self.glyph_plan.take() {
            Some(plan) if plan.slots() == glyph_slots.as_slice() => plan,
            _ => {
                self.glyph_plan_builds += 1;
                SlotBasePlan::for_glyphs(device, queue, &self.glyphs, &glyph_slots)
            }
        };
        let quad_frame_group = self.quads.frame_bind_group(
            device,
            &self.globals,
            self.arena.buffer(),
            &self.quad_args.visible,
        );
        let glyph_frame_group = self.glyphs.frame_bind_group(
            device,
            &self.globals,
            self.glyph_arena.buffer(),
            &self.glyph_args.visible,
        );
        // Rebuilt per frame, not cached: a bind group names a texture view, and
        // an atlas page destroyed between frames leaves a cached one pointing at
        // a texture that no longer exists. The 16-byte uniform behind it *is*
        // cached, because its value is the page index and that never changes.
        let glyph_pages = self.atlas_page_bind_groups(device, queue, input.atlas);
        let composite_frame_group = self.composite.frame_bind_group(device, &self.globals);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("frame"),
        });
        let mut stats;
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("frame"),
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
            let started = Instant::now();
            stats = issue_quads(
                &mut pass,
                &self.quads,
                &quad_plan,
                &quad_frame_group,
                &self.quad_args,
                input.mode,
                &quad_resolved,
            );
            // After the quads and before the composites: text paints over the
            // chrome behind it and under a modal composited on top, which is the
            // painter order §5.3's kind grouping already puts the slots in.
            stats.merge(issue_glyphs(
                &mut pass,
                &self.glyphs,
                GlyphDraw {
                    plan: &glyph_plan,
                    pages: &glyph_pages,
                },
                &glyph_frame_group,
                &self.glyph_args,
                input.mode,
                &glyph_resolved,
            ));
            stats.merge(issue_composites(
                &mut pass,
                &self.composite,
                &composite_frame_group,
                &self.composite_args,
                &composite_plan,
                input.mode,
                &composite_resolved,
            ));
            timing.draw_issue = started.elapsed();
        }
        queue.submit(Some(encoder.finish()));
        self.quad_plan = Some(quad_plan);
        self.glyph_plan = Some(glyph_plan);

        Ok(FrameOutput {
            stats,
            timing,
            layers_recomputed,
            primitives_resident,
        })
    }
}

fn layer_quads(scene: &Scene, layer: LayerId) -> Vec<Quad> {
    scene
        .quads
        .keys(layer)
        .into_iter()
        .filter_map(|key| scene.quads.get(layer, key).copied())
        .collect()
}

/// One layer's glyphs as bounding rectangles, in arena slot order.
///
/// Slot order is what makes this usable at all: the compute passes' outputs are
/// indexed `[0, count)` within a layer and are scattered into the arena at
/// `[base, base + count)`, so entry *i* here has to be arena slot `base + i`.
/// `PrimitiveStore::keys` returns paint order, and `reflow` assigns
/// `slot_offset` by walking exactly that order cumulatively — so flattening the
/// runs in key order, glyph by glyph, reproduces the arena.
///
/// A glyph's rectangle is its raster's, not its advance's: `Glyph::position` is
/// the top-left of the tile (the pen advanced by the raster's bearing) and
/// `atlas_size` is the tile's extent, which is exactly the quad
/// `mono_sprites.wgsl` builds. A blank glyph is a zero-sized rectangle, which
/// intersects nothing and orders under everything.
fn layer_glyph_bounds(scene: &Scene, layer: LayerId) -> Vec<Rect> {
    let mut bounds = Vec::new();
    for key in scene.glyph_runs.keys(layer) {
        let Some(run) = scene.glyph_runs.get(layer, key) else {
            continue;
        };
        bounds.extend(
            run.glyphs
                .iter()
                .map(|glyph| Rect::from_origin_size(glyph.position, glyph.atlas_size)),
        );
    }
    bounds
}

/// The upload instructions addressed to one kind's arena.
fn kind_uploads(uploads: &[UploadRange], kind: PrimitiveKind) -> Vec<UploadRange> {
    uploads
        .iter()
        .filter(|range| range.kind == kind)
        .copied()
        .collect()
}

fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}
