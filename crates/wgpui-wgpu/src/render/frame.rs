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

use crate::debug::DebugTile;
use crate::render::atlas_upload::AtlasTextures;
use crate::render::buffers::slab_buffers::SlabBuffer;
use crate::render::compute::indirect_args_pass::{
    IndirectArgsBuffers, IndirectArgsError, IndirectArgsPass,
};
use crate::render::compute::occlusion_pass::{OcclusionError, OcclusionPass};
use crate::render::compute::ordering_pass::{OrderingError, OrderingPass};
use crate::render::draw::{
    DrawMode, DrawStats, ResolvedArgs, SlotBasePlan, SpriteDraw, issue_backdrop_filters,
    issue_composites, issue_instanced, issue_paths, issue_sprites,
};
use crate::render::pipelines::{
    BackdropPipeline, CompositePipeline, Globals, MonoSpritePipeline, PathPipeline,
    PolySpritePipeline, QuadPipeline, ShadowPipeline, TARGET_FORMAT, UnderlinePipeline,
};
use crate::render::readback::{ReadbackError, StagingReader, read_texture_rgba8};
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
    /// A backdrop filter needs the target texture so the renderer can take the
    /// legacy-style snapshot between render passes.
    BackdropSourceUnavailable,
    /// The optional diagnostic overlay could not be prepared.
    DebugOverlayUnavailable,
}

impl std::fmt::Display for FrameError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FrameError::Ordering(error) => write!(formatter, "{error}"),
            FrameError::Occlusion(error) => write!(formatter, "{error}"),
            FrameError::IndirectArgs(error) => write!(formatter, "{error}"),
            FrameError::Readback(error) => write!(formatter, "{error}"),
            FrameError::BackdropSourceUnavailable => write!(
                formatter,
                "backdrop filtering requires a copyable render-target texture"
            ),
            FrameError::DebugOverlayUnavailable =>
                write!(formatter, "tile refresh diagnostics could not be prepared"),
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
    /// this one. `None` — or a set with no page of a given kind in it — is an
    /// ordinary frame with no rasterised text or no decoded image, not an error.
    /// Both sprite passes read this one field and each filters it to its own
    /// [`AtlasKind`].
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
    /// Scene-arena `write_buffer` calls issued by this frame. A clean retained
    /// frame must report zero here.
    pub scene_upload_calls: u32,
    /// Scene-arena bytes written by this frame.
    pub scene_upload_bytes: u64,
    /// Slot-base plans built by this frame. A steady frame must report zero.
    pub plan_builds: u32,
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
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
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
    /// The body moved to [`read_texture_rgba8`] in Phase 6, unchanged, so that
    /// the swapchain image — which this type does not own and cannot own — is
    /// read by the identical code. A comparison between an offscreen render and
    /// a presented frame is only a comparison if both sides were unpacked the
    /// same way.
    pub fn read_pixels(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<u8>, ReadbackError> {
        read_texture_rgba8(device, queue, &self.texture, self.width, self.height)
    }

    /// Copy the retained presentation buffer into an acquired surface image.
    pub fn copy_to_texture(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        destination: &wgpu::Texture,
    ) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("present retained frame"),
        });
        encoder.copy_texture_to_texture(
            self.texture.as_image_copy(),
            destination.as_image_copy(),
            wgpu::Extent3d {
                width: self.width.min(destination.width()),
                height: self.height.min(destination.height()),
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));
    }

    /// This target as a [`RenderTarget`], cleared to black.
    ///
    /// Black is what every test written before Phase 6 expects, and Phase 5.6's
    /// byte-exact text proof depends on it specifically: white text over black
    /// through a straight-alpha `over` blend is an identity, which is what makes
    /// a rendered pixel *equal* to its atlas texel rather than merely close to
    /// it.
    pub fn target(&self) -> RenderTarget<'_> {
        RenderTarget {
            view: &self.view,
            width: self.width,
            height: self.height,
            clear: wgpu::Color::BLACK,
            source: Some(&self.texture),
        }
    }
}

/// Where one frame's colour goes.
///
/// Phase 4 took an [`OffscreenTarget`] by reference and could, because the only
/// thing a frame had ever been drawn into was a texture this crate allocated.
/// Phase 6 draws into a swapchain image, which is allocated by the presentation
/// engine, handed over one frame at a time, and cannot be owned by anything
/// here. What the renderer actually needs is a view and its extent, so that is
/// what it now asks for — and the fact that both paths pass through this one
/// struct is what makes "the offscreen test and the window draw the same frame"
/// a property of the code rather than a claim about it.
pub struct RenderTarget<'a> {
    /// The view the render pass attaches.
    pub view: &'a wgpu::TextureView,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// What the pass loads with.
    pub clear: wgpu::Color,
    /// The texture behind `view`, when it can be copied for a backdrop filter.
    /// Surface textures and [`OffscreenTarget`] both provide this; callers
    /// supplying only a view get an explicit frame error if a filter is used.
    pub source: Option<&'a wgpu::Texture>,
}

/// Everything a frame needs that outlives one: pipelines, arenas, pools.
pub struct FrameRenderer {
    ordering: OrderingPass,
    occlusion: OcclusionPass,
    indirect: IndirectArgsPass,
    /// The instanced shadow pipeline (Phase 6.3).
    pub shadows: ShadowPipeline,
    /// The instanced quad pipeline.
    pub quads: QuadPipeline,
    /// The instanced underline pipeline (Phase 6.3).
    pub underlines: UnderlinePipeline,
    /// The instanced monochrome-sprite pipeline (Phase 5.6).
    pub glyphs: MonoSpritePipeline,
    /// The instanced polychrome-sprite pipeline (Phase 6.2).
    pub sprites: PolySpritePipeline,
    /// The Lyon-tessellated path pipeline.
    pub paths: PathPipeline,
    /// The framebuffer-sampling backdrop-filter pipeline.
    pub backdrop_filters: BackdropPipeline,
    /// The one composite pipeline.
    pub composite: CompositePipeline,
    /// The shadow arena. Its own buffer, for the reason `sprite_arena` records:
    /// two kinds sharing a slot stride today is a coincidence of two
    /// independent layout decisions, not a licence to share an arena.
    pub shadow_arena: SlabBuffer,
    /// The quad arena.
    pub arena: SlabBuffer,
    /// The underline arena, its own buffer for `shadow_arena`'s reason — the
    /// two kinds' 48-byte strides agreeing is a coincidence of two independent
    /// field sets.
    pub underline_arena: SlabBuffer,
    /// The glyph arena. A second buffer rather than a second range of the
    /// first: the two kinds have different slot strides, and §5.0's upload
    /// instructions are already addressed per kind.
    pub glyph_arena: SlabBuffer,
    /// The image-sprite arena. Its own buffer for the same reason, even though
    /// `PolySprite` and `GlyphRun` happen to share a 48-byte stride: the strides
    /// agreeing today is a coincidence of two independent layout decisions, and
    /// sharing an arena on the strength of it would make one kind's field
    /// changing corrupt the other's.
    pub sprite_arena: SlabBuffer,
    /// Flattened path-vertex arena.
    pub path_arena: SlabBuffer,
    /// Backdrop-filter records.
    pub backdrop_arena: SlabBuffer,
    /// Boundary texture retention (§5.5).
    pub textures: LayerTexturePool,
    globals: wgpu::Buffer,
    shadow_args: IndirectArgsBuffers,
    quad_args: IndirectArgsBuffers,
    underline_args: IndirectArgsBuffers,
    glyph_args: IndirectArgsBuffers,
    sprite_args: IndirectArgsBuffers,
    composite_args: IndirectArgsBuffers,
    /// The per-slot bases and their bind group, kept until the slot table
    /// itself changes — see [`SlotBasePlan`], which is only true of it if
    /// something holds it across frames.
    shadow_plan: Option<SlotBasePlan>,
    quad_plan: Option<SlotBasePlan>,
    underline_plan: Option<SlotBasePlan>,
    glyph_plan: Option<SlotBasePlan>,
    sprite_plan: Option<SlotBasePlan>,
    path_plan: Option<SlotBasePlan>,
    backdrop_plan: Option<SlotBasePlan>,
    shadow_plan_builds: u64,
    quad_plan_builds: u64,
    underline_plan_builds: u64,
    glyph_plan_builds: u64,
    sprite_plan_builds: u64,
    path_plan_builds: u64,
    backdrop_plan_builds: u64,
    /// One 16-byte `AtlasPage` uniform per page index ever bound.
    ///
    /// Keyed by page index and never invalidated, because the value is the page
    /// index itself: a page destroyed and recreated is the same number and the
    /// same bytes. The bind group over it is *not* cached, because that names a
    /// texture view which a destroyed page invalidates.
    page_params: HashMap<u32, wgpu::Buffer>,
    reader: StagingReader,
    uploaded_generation: Option<u64>,
    backdrop_snapshot: Option<wgpu::Texture>,
    backdrop_snapshot_view: Option<wgpu::TextureView>,
    backdrop_sampler: wgpu::Sampler,
    damage_clear_pipeline: wgpu::RenderPipeline,
    damage_clear_bind_group: wgpu::BindGroup,
    damage_clear_color: wgpu::Buffer,
    debug_pipeline: wgpu::RenderPipeline,
    debug_bind_group_layout: wgpu::BindGroupLayout,
    debug_buffer: wgpu::Buffer,
    debug_buffer_capacity: u64,
    debug_bind_group: Option<wgpu::BindGroup>,
    debug_tiles: Vec<DebugTile>,
}

impl FrameRenderer {
    /// Build every pipeline once.
    pub fn new(device: &wgpu::Device) -> FrameRenderer {
        let (damage_clear_pipeline, damage_clear_bind_group, damage_clear_color) =
            create_damage_clear_pipeline(device);
        let (debug_bind_group_layout, debug_pipeline) = create_debug_pipeline(device);
        let debug_buffer_capacity = std::mem::size_of::<DebugTile>() as u64;
        FrameRenderer {
            ordering: OrderingPass::new(device),
            occlusion: OcclusionPass::new(device),
            indirect: IndirectArgsPass::new(device),
            shadows: ShadowPipeline::new(device),
            quads: QuadPipeline::new(device),
            underlines: UnderlinePipeline::new(device),
            glyphs: MonoSpritePipeline::new(device),
            sprites: PolySpritePipeline::new(device),
            paths: PathPipeline::new(device),
            backdrop_filters: BackdropPipeline::new(device),
            composite: CompositePipeline::new(device),
            shadow_arena: SlabBuffer::new(device, "shadow arena"),
            arena: SlabBuffer::new(device, "quad arena"),
            underline_arena: SlabBuffer::new(device, "underline arena"),
            glyph_arena: SlabBuffer::new(device, "glyph arena"),
            sprite_arena: SlabBuffer::new(device, "sprite arena"),
            path_arena: SlabBuffer::new(device, "path arena"),
            backdrop_arena: SlabBuffer::new(device, "backdrop filter arena"),
            textures: LayerTexturePool::default(),
            globals: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("frame globals"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            shadow_args: IndirectArgsBuffers::new(device, 1, 1),
            quad_args: IndirectArgsBuffers::new(device, 1, 1),
            underline_args: IndirectArgsBuffers::new(device, 1, 1),
            glyph_args: IndirectArgsBuffers::new(device, 1, 1),
            sprite_args: IndirectArgsBuffers::new(device, 1, 1),
            composite_args: IndirectArgsBuffers::new(device, 1, 1),
            shadow_plan: None,
            quad_plan: None,
            underline_plan: None,
            glyph_plan: None,
            sprite_plan: None,
            path_plan: None,
            backdrop_plan: None,
            shadow_plan_builds: 0,
            quad_plan_builds: 0,
            underline_plan_builds: 0,
            glyph_plan_builds: 0,
            sprite_plan_builds: 0,
            path_plan_builds: 0,
            backdrop_plan_builds: 0,
            page_params: HashMap::new(),
            reader: StagingReader::new(),
            uploaded_generation: None,
            backdrop_snapshot: None,
            backdrop_snapshot_view: None,
            backdrop_sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("backdrop filter sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                ..Default::default()
            }),
            damage_clear_pipeline,
            damage_clear_bind_group,
            damage_clear_color,
            debug_pipeline,
            debug_bind_group_layout,
            debug_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("tile refresh diagnostics"),
                size: debug_buffer_capacity,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            debug_buffer_capacity,
            debug_bind_group: None,
            debug_tiles: Vec::new(),
        }
    }

    /// Set transient diagnostic rectangles for the next render.
    pub fn set_debug_tiles(&mut self, tiles: Vec<DebugTile>) {
        self.debug_tiles = tiles;
    }

    fn ensure_backdrop_snapshot(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let matches_size = self.backdrop_snapshot.as_ref().is_some_and(|texture| {
            texture.width() == width.max(1) && texture.height() == height.max(1)
        });
        if matches_size {
            return;
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("backdrop filter snapshot"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TARGET_FORMAT,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.backdrop_snapshot = Some(texture);
        self.backdrop_snapshot_view = Some(view);
    }

    fn prepare_debug_tiles(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tiles: &[DebugTile],
    ) -> Option<&wgpu::BindGroup> {
        if tiles.is_empty() {
            return None;
        }
        let bytes = bytemuck::cast_slice(tiles);
        let required = bytes.len() as u64;
        if required > self.debug_buffer_capacity {
            self.debug_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("tile refresh diagnostics"),
                size: required.next_power_of_two(),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.debug_buffer_capacity = required.next_power_of_two();
            self.debug_bind_group = None;
        }
        queue.write_buffer(&self.debug_buffer, 0, bytes);
        if self.debug_bind_group.is_none() {
            self.debug_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tile refresh diagnostics"),
                layout: &self.debug_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.globals.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.debug_buffer.as_entire_binding(),
                    },
                ],
            }));
        }
        self.debug_bind_group.as_ref()
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

    /// The same counter for the shadow pipeline's slot bases.
    pub fn shadow_plan_builds(&self) -> u64 {
        self.shadow_plan_builds
    }

    /// The same counter for the underline pipeline's slot bases.
    pub fn underline_plan_builds(&self) -> u64 {
        self.underline_plan_builds
    }

    /// The same counter for the glyph pipeline's slot bases.
    pub fn glyph_plan_builds(&self) -> u64 {
        self.glyph_plan_builds
    }

    /// The same counter for the image-sprite pipeline's slot bases.
    pub fn sprite_plan_builds(&self) -> u64 {
        self.sprite_plan_builds
    }

    /// How many times the path slot-base plan has been built.
    pub fn path_plan_builds(&self) -> u64 {
        self.path_plan_builds
    }

    /// How many times the backdrop slot-base plan has been built.
    pub fn backdrop_plan_builds(&self) -> u64 {
        self.backdrop_plan_builds
    }

    /// Total scene-arena upload calls since renderer creation.
    pub fn scene_upload_calls(&self) -> u64 {
        self.shadow_arena.upload_calls()
            + self.arena.upload_calls()
            + self.underline_arena.upload_calls()
            + self.glyph_arena.upload_calls()
            + self.sprite_arena.upload_calls()
            + self.path_arena.upload_calls()
            + self.backdrop_arena.upload_calls()
    }

    /// Total scene-arena bytes written since renderer creation.
    pub fn scene_upload_bytes(&self) -> u64 {
        self.shadow_arena.uploaded_bytes()
            + self.arena.uploaded_bytes()
            + self.underline_arena.uploaded_bytes()
            + self.glyph_arena.uploaded_bytes()
            + self.sprite_arena.uploaded_bytes()
            + self.path_arena.uploaded_bytes()
            + self.backdrop_arena.uploaded_bytes()
    }

    fn plan_builds(&self) -> u64 {
        self.shadow_plan_builds
            + self.quad_plan_builds
            + self.underline_plan_builds
            + self.glyph_plan_builds
            + self.sprite_plan_builds
            + self.path_plan_builds
            + self.backdrop_plan_builds
    }

    /// One bind group per live atlas page of `kind`, in ascending page order.
    ///
    /// **Pages are filtered by kind and not merely enumerated.** A coverage page
    /// bound to `poly_sprites.wgsl` would have its single channel read as RGBA,
    /// and a colour page bound to `mono_sprites.wgsl` would have an emoji's red
    /// channel painted as coverage. Neither is an error anything would report —
    /// both just draw the wrong picture — so the filter is what keeps the two
    /// passes apart, and it is one function rather than two so a new kind cannot
    /// acquire a copy that forgets it.
    ///
    /// The 16-byte page uniform is shared between the two pipelines, which is
    /// legitimate rather than a shortcut: both declare the same
    /// `PAGE_PARAMS_SIZE` block holding the same value, the page index, and a
    /// page index is a fact about the atlas rather than about a shader.
    fn atlas_page_bind_groups(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        atlas: Option<&AtlasTextures>,
        kind: AtlasKind,
    ) -> Vec<wgpu::BindGroup> {
        let Some(atlas) = atlas else {
            return Vec::new();
        };
        let mut groups = Vec::new();
        for page in atlas.pages_of_kind(kind) {
            let Some(view) = atlas.view(page) else {
                continue;
            };
            let params = self.page_params.entry(page).or_insert_with(|| {
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("sprite atlas page"),
                    size: MonoSpritePipeline::PAGE_PARAMS_SIZE,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let mut bytes = [0u8; MonoSpritePipeline::PAGE_PARAMS_SIZE as usize];
                bytes[0..4].copy_from_slice(&page.to_le_bytes());
                queue.write_buffer(&buffer, 0, &bytes);
                buffer
            });
            groups.push(match kind {
                AtlasKind::Monochrome => self.glyphs.page_bind_group(device, params, view),
                AtlasKind::Polychrome => self.sprites.page_bind_group(device, params, view),
            });
        }
        groups
    }

    /// Render one frame into `target`.
    ///
    /// **All five instanced kinds are drawn.** Phase 4 drew only `Quad` and
    /// said so here; Phase 5.6 added the `GlyphRun` half, Phase 6.2 the
    /// `PolySprite` half, and Phase 6.3 the `Shadow` and `Underline` halves,
    /// all taking the identical route — upload, ordering, occlusion, indirect-argument
    /// generation, fixed draw sequence — over their own arenas. Two kinds add
    /// something to that route and each addition is one thing:
    ///
    /// - The two **sprite** kinds repeat the pass per bound atlas page, because
    ///   a sprite's texture is chosen by its tile and a bind group cannot change
    ///   inside a draw call. See [`issue_sprites`], one function for both.
    /// - **Shadows** feed the compute passes
    ///   [`wgpui_core::patch::primitive::Shadow::drawn_bounds`] rather than the
    ///   primitive's own rectangle, and take [`CoverageItem::uncullable`] rather
    ///   than [`CoverageItem::cullee`]. Both are noted at the loop that does it.
    ///
    /// `Underline` adds nothing to the route at all, which is worth stating
    /// beside `Shadow`: two kinds landed in the same phase and only one of them
    /// needed a qualification.
    ///
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &FrameInput<'_>,
        target: &OffscreenTarget,
    ) -> Result<FrameOutput, FrameError> {
        let output = self.render_to(device, queue, input, &target.target());
        #[cfg(feature = "devtools")]
        if output.is_ok() {
            static HOOKS: std::sync::OnceLock<wgpui_devtools::hooks::DevtoolsHooks> =
                std::sync::OnceLock::new();
            wgpui_core::hooks::InstrumentationHooks::frame_presented(
                HOOKS.get_or_init(Default::default),
            );
        }
        output
    }

    /// Render one frame into any colour target.
    ///
    /// [`Self::render`]'s body, with the target generalised. The swapchain path
    /// calls this and the offscreen path calls it through `render`, so there is
    /// exactly one implementation of "what a frame does" — see [`RenderTarget`]
    /// for why that mattered enough to change a signature five phases in.
    pub fn render_to(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &FrameInput<'_>,
        target: &RenderTarget<'_>,
    ) -> Result<FrameOutput, FrameError> {
        self.render_to_with_damage(device, queue, input, target, None)
    }

    /// Render only the damaged part of a target whose previous contents are
    /// still valid. The scene and its GPU arenas remain retained scene state;
    /// this rectangle is only a raster/presentation restriction.
    pub fn render_to_with_damage(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &FrameInput<'_>,
        target: &RenderTarget<'_>,
        damage: Option<Rect>,
    ) -> Result<FrameOutput, FrameError> {
        #[cfg(feature = "devtools")]
        let _instrumentation_span = {
            static HOOKS: std::sync::OnceLock<wgpui_devtools::hooks::DevtoolsHooks> =
                std::sync::OnceLock::new();
            let hooks = HOOKS.get_or_init(Default::default);
            wgpui_core::hooks::Span::new(hooks, "frame: render")
        };
        self.textures.begin_frame();
        let mut timing = FrameTiming::default();
        let upload_calls_before = self.scene_upload_calls();
        let upload_bytes_before = self.scene_upload_bytes();
        let plan_builds_before = self.plan_builds();

        let table = input.scene.draw_slots();
        let shadow_slots: Vec<DrawSlot> = table.kind_slots(PrimitiveKind::Shadow).to_vec();
        let quad_slots: Vec<DrawSlot> = table.kind_slots(PrimitiveKind::Quad).to_vec();
        let underline_slots: Vec<DrawSlot> = table.kind_slots(PrimitiveKind::Underline).to_vec();
        let glyph_slots: Vec<DrawSlot> = table.kind_slots(PrimitiveKind::GlyphRun).to_vec();
        let sprite_slots: Vec<DrawSlot> = table.kind_slots(PrimitiveKind::PolySprite).to_vec();
        let path_slots: Vec<DrawSlot> = table.kind_slots(PrimitiveKind::Path).to_vec();
        let backdrop_slots: Vec<DrawSlot> =
            table.kind_slots(PrimitiveKind::BackdropFilter).to_vec();
        if backdrop_slots.iter().any(|slot| slot.count > 0) && target.source.is_none() {
            return Err(FrameError::BackdropSourceUnavailable);
        }
        let shadow_arena_slots = input.scene.arena_slots(PrimitiveKind::Shadow);
        let arena_slots = input.scene.arena_slots(PrimitiveKind::Quad);
        let underline_arena_slots = input.scene.arena_slots(PrimitiveKind::Underline);
        let glyph_arena_slots = input.scene.arena_slots(PrimitiveKind::GlyphRun);
        let sprite_arena_slots = input.scene.arena_slots(PrimitiveKind::PolySprite);
        let primitives_resident: u32 = shadow_slots
            .iter()
            .chain(quad_slots.iter())
            .chain(underline_slots.iter())
            .chain(glyph_slots.iter())
            .chain(sprite_slots.iter())
            .chain(path_slots.iter())
            .chain(backdrop_slots.iter())
            .map(|slot| slot.count)
            .sum();

        // --- 1. Upload.
        let started = Instant::now();
        let shadow_resident = input.scene.shadows.resident_bytes();
        let resident = input.scene.quads.resident_bytes();
        let underline_resident = input.scene.underlines.resident_bytes();
        let glyph_resident = input.scene.glyph_runs.resident_bytes();
        let sprite_resident = input.scene.poly_sprites.resident_bytes();
        let path_resident = input.scene.paths.resident_bytes();
        let backdrop_resident = input.scene.backdrop_filters.resident_bytes();
        let shadows_grew = self
            .shadow_arena
            .reserve(device, shadow_resident.len() as u64);
        let grew = self.arena.reserve(device, resident.len() as u64);
        let underlines_grew = self
            .underline_arena
            .reserve(device, underline_resident.len() as u64);
        let glyphs_grew = self
            .glyph_arena
            .reserve(device, glyph_resident.len() as u64);
        let sprites_grew = self
            .sprite_arena
            .reserve(device, sprite_resident.len() as u64);
        let paths_grew = self.path_arena.reserve(device, path_resident.len() as u64);
        let backdrops_grew = self
            .backdrop_arena
            .reserve(device, backdrop_resident.len() as u64);
        if shadows_grew
            || grew
            || underlines_grew
            || glyphs_grew
            || sprites_grew
            || paths_grew
            || backdrops_grew
            || self.uploaded_generation.is_none()
        {
            self.shadow_arena.upload_all(device, queue, shadow_resident);
            self.arena.upload_all(device, queue, resident);
            self.underline_arena
                .upload_all(device, queue, underline_resident);
            self.glyph_arena.upload_all(device, queue, glyph_resident);
            self.sprite_arena.upload_all(device, queue, sprite_resident);
            self.path_arena.upload_all(device, queue, path_resident);
            self.backdrop_arena
                .upload_all(device, queue, backdrop_resident);
            self.uploaded_generation = Some(0);
        } else {
            // Filtered by kind rather than handed the whole list: an
            // `UploadRange` is a byte span *within one kind's arena*, so
            // applying a glyph range to the quad buffer would overwrite an
            // unrelated primitive with a glyph's bytes. Before this phase there
            // was one arena and the filter was unnecessary; with two it is the
            // difference between a delta upload and corruption.
            self.shadow_arena.upload(
                device,
                queue,
                shadow_resident,
                &kind_uploads(input.uploads, PrimitiveKind::Shadow),
            );
            self.arena.upload(
                device,
                queue,
                resident,
                &kind_uploads(input.uploads, PrimitiveKind::Quad),
            );
            self.underline_arena.upload(
                device,
                queue,
                underline_resident,
                &kind_uploads(input.uploads, PrimitiveKind::Underline),
            );
            self.glyph_arena.upload(
                device,
                queue,
                glyph_resident,
                &kind_uploads(input.uploads, PrimitiveKind::GlyphRun),
            );
            self.sprite_arena.upload(
                device,
                queue,
                sprite_resident,
                &kind_uploads(input.uploads, PrimitiveKind::PolySprite),
            );
            self.path_arena.upload(
                device,
                queue,
                path_resident,
                &kind_uploads(input.uploads, PrimitiveKind::Path),
            );
            self.backdrop_arena.upload(
                device,
                queue,
                backdrop_resident,
                &kind_uploads(input.uploads, PrimitiveKind::BackdropFilter),
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

        if !self
            .shadow_args
            .fits(shadow_arena_slots, shadow_slots.len() as u32)
        {
            self.shadow_args = IndirectArgsBuffers::new(
                device,
                shadow_arena_slots.max(1),
                shadow_slots.len() as u32 + 1,
            );
        }
        if !self.quad_args.fits(arena_slots, quad_slots.len() as u32) {
            self.quad_args =
                IndirectArgsBuffers::new(device, arena_slots.max(1), quad_slots.len() as u32 + 1);
        }
        if !self
            .underline_args
            .fits(underline_arena_slots, underline_slots.len() as u32)
        {
            self.underline_args = IndirectArgsBuffers::new(
                device,
                underline_arena_slots.max(1),
                underline_slots.len() as u32 + 1,
            );
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
        if !self
            .sprite_args
            .fits(sprite_arena_slots, sprite_slots.len() as u32)
        {
            self.sprite_args = IndirectArgsBuffers::new(
                device,
                sprite_arena_slots.max(1),
                sprite_slots.len() as u32 + 1,
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

            // The shadow half, through the identical passes — with the two
            // differences §8's "`QuadPipeline`-shaped" wording does not cover,
            // both of which live here rather than in the pipeline:
            //
            // 1. **The rectangle is the drawn one, not the primitive's.** A
            //    shadow's Gaussian tail rasterises up to
            //    `Shadow::BLUR_MARGIN_SIGMAS * blur_radius` outside its own
            //    bounds, so ordering against `origin`/`size` would sort a large
            //    soft shadow as if it were the small hard rectangle at its
            //    centre.
            // 2. **A shadow is never culled.** `CoverageItem::uncullable`, not
            //    `cullee` — the legacy sweep skips shadows for the same reason
            //    (`src/occlusion.rs:255`), and 2.0's own
            //    `CoverageItem::cullable` doc has named shadows specifically
            //    since Phase 3. Feeding the *drawn* rectangle would in fact make
            //    culling sound, but that would be a deviation from legacy output
            //    on a phase whose gate is byte-exactness against it, so it is
            //    left as an option rather than taken.
            //
            // A shadow is never an occluder either, and for a stronger reason
            // than a sprite's: its own interior is a blurred gradient, so there
            // is no rectangle over which it is opaque at all.
            //
            // **Both adjustments are currently inert, and that is measured
            // rather than assumed.** Reverting either or both leaves every
            // shadow test — the byte-exact gate included — passing, because
            // occlusion dispatches per kind: the quad that would cover a shadow
            // is in a different dispatch, no shadow can occlude another, and
            // `keep_item` keeps an empty-visible item rather than dropping it.
            // So nothing in 2.0 can cull a shadow today whatever this says.
            // They are written this way because they are *right*, and because
            // the day cross-kind occlusion exists (the limit Phase 5.6 recorded
            // for glyphs) a shadow culled against its unblurred rectangle would
            // lose falloff that was never covered. See
            // `tests/legacy_shadow_differential.rs`'s
            // `a_shadow_covered_by_an_opaque_quad_still_paints_its_falloff_outside_it`,
            // which records the experiment.
            for slot in &shadow_slots {
                if slot.count == 0 || !input.dirty.contains(slot.layer) {
                    continue;
                }
                let bounds = layer_shadow_bounds(input.scene, slot.layer);
                let items: Vec<CoverageItem> = bounds
                    .iter()
                    .map(|shadow| CoverageItem::uncullable(shadow.intersect(&input.clip)))
                    .collect();

                encode_ordering_items(&bounds, &mut bounds_bytes);
                let ordered = self.ordering.run(device, queue, &bounds_bytes)?;
                encode_coverage_items(&items, &mut item_bytes);
                let culled = self
                    .occlusion
                    .run(device, queue, &item_bytes, &poison_bytes)?;

                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("shadow scatter"),
                });
                IndirectArgsPass::scatter(
                    &mut encoder,
                    &ordered.draw_order,
                    &self.shadow_args.draw_order,
                    slot.base,
                    slot.count,
                );
                IndirectArgsPass::scatter(
                    &mut encoder,
                    &culled.culled,
                    &self.shadow_args.culled,
                    slot.base,
                    slot.count,
                );
                queue.submit(Some(encoder.finish()));
                layers_recomputed += 1;
            }

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

            // The underline half. Nothing here is qualified the way the shadow
            // loop above is: an underline paints inside its own rectangle and is
            // an ordinary `cullee`, exactly as the legacy sweep classifies it
            // (`src/occlusion.rs:262` lists `Underline` beside `Quad`). It is
            // not an occluder — a wavy rule covers almost none of its own box,
            // and a straight one is thinner than its box by construction.
            for slot in &underline_slots {
                if slot.count == 0 || !input.dirty.contains(slot.layer) {
                    continue;
                }
                let bounds = layer_underline_bounds(input.scene, slot.layer);
                let items: Vec<CoverageItem> = bounds
                    .iter()
                    .map(|underline| CoverageItem::cullee(underline.intersect(&input.clip)))
                    .collect();

                encode_ordering_items(&bounds, &mut bounds_bytes);
                let ordered = self.ordering.run(device, queue, &bounds_bytes)?;
                encode_coverage_items(&items, &mut item_bytes);
                let culled = self
                    .occlusion
                    .run(device, queue, &item_bytes, &poison_bytes)?;

                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("underline scatter"),
                });
                IndirectArgsPass::scatter(
                    &mut encoder,
                    &ordered.draw_order,
                    &self.underline_args.draw_order,
                    slot.base,
                    slot.count,
                );
                IndirectArgsPass::scatter(
                    &mut encoder,
                    &culled.culled,
                    &self.underline_args.culled,
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

            // The image half, through the identical passes and for the identical
            // reason. A sprite is a `CoverageItem::cullee` and never an occluder
            // even when the image behind it is fully opaque: whether a decoded
            // PNG has an alpha channel is not knowable from the primitive, and
            // treating a transparent avatar as an occluder would erase whatever
            // is behind it. The legacy renderer makes the same call — its
            // occlusion pass skips `Primitive::PolychromeSprite` entirely
            // (`src/occlusion.rs`).
            for slot in &sprite_slots {
                if slot.count == 0 || !input.dirty.contains(slot.layer) {
                    continue;
                }
                let bounds = layer_sprite_bounds(input.scene, slot.layer);
                let items: Vec<CoverageItem> = bounds
                    .iter()
                    .map(|sprite| CoverageItem::cullee(sprite.intersect(&input.clip)))
                    .collect();

                encode_ordering_items(&bounds, &mut bounds_bytes);
                let ordered = self.ordering.run(device, queue, &bounds_bytes)?;
                encode_coverage_items(&items, &mut item_bytes);
                let culled = self
                    .occlusion
                    .run(device, queue, &item_bytes, &poison_bytes)?;

                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("sprite scatter"),
                });
                IndirectArgsPass::scatter(
                    &mut encoder,
                    &ordered.draw_order,
                    &self.sprite_args.draw_order,
                    slot.base,
                    slot.count,
                );
                IndirectArgsPass::scatter(
                    &mut encoder,
                    &culled.culled,
                    &self.sprite_args.culled,
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
        let mut shadow_slot_bytes = Vec::new();
        encode_slots(&shadow_slots, &mut shadow_slot_bytes);
        let shadow_output = self.indirect.run(
            device,
            queue,
            &self.shadow_args,
            &shadow_slot_bytes,
            QUAD_VERTEX_COUNT,
            input.mode.first_instance(),
        )?;
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
        let mut underline_slot_bytes = Vec::new();
        encode_slots(&underline_slots, &mut underline_slot_bytes);
        let underline_output = self.indirect.run(
            device,
            queue,
            &self.underline_args,
            &underline_slot_bytes,
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
        let mut sprite_slot_bytes = Vec::new();
        encode_slots(&sprite_slots, &mut sprite_slot_bytes);
        let sprite_output = self.indirect.run(
            device,
            queue,
            &self.sprite_args,
            &sprite_slot_bytes,
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
        let shadow_resolved = ResolvedArgs::resolve(
            input.mode,
            device,
            queue,
            &self.shadow_args,
            shadow_output.slot_count,
            &mut self.reader,
        )?;
        let quad_resolved = ResolvedArgs::resolve(
            input.mode,
            device,
            queue,
            &self.quad_args,
            quad_output.slot_count,
            &mut self.reader,
        )?;
        let underline_resolved = ResolvedArgs::resolve(
            input.mode,
            device,
            queue,
            &self.underline_args,
            underline_output.slot_count,
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
        let sprite_resolved = ResolvedArgs::resolve(
            input.mode,
            device,
            queue,
            &self.sprite_args,
            sprite_output.slot_count,
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
        let mut shadow_plan = match self.shadow_plan.take() {
            Some(plan) if plan.slots() == shadow_slots.as_slice() => plan,
            _ => {
                self.shadow_plan_builds += 1;
                SlotBasePlan::for_shadows(device, queue, &self.shadows, &shadow_slots)
            }
        };
        let mut quad_plan = match self.quad_plan.take() {
            Some(plan) if plan.slots() == quad_slots.as_slice() => plan,
            _ => {
                self.quad_plan_builds += 1;
                SlotBasePlan::for_quads(device, queue, &self.quads, &quad_slots)
            }
        };
        let mut underline_plan = match self.underline_plan.take() {
            Some(plan) if plan.slots() == underline_slots.as_slice() => plan,
            _ => {
                self.underline_plan_builds += 1;
                SlotBasePlan::for_underlines(device, queue, &self.underlines, &underline_slots)
            }
        };
        let mut glyph_plan = match self.glyph_plan.take() {
            Some(plan) if plan.slots() == glyph_slots.as_slice() => plan,
            _ => {
                self.glyph_plan_builds += 1;
                SlotBasePlan::for_glyphs(device, queue, &self.glyphs, &glyph_slots)
            }
        };
        let mut sprite_plan = match self.sprite_plan.take() {
            Some(plan) if plan.slots() == sprite_slots.as_slice() => plan,
            _ => {
                self.sprite_plan_builds += 1;
                SlotBasePlan::for_poly_sprites(device, queue, &self.sprites, &sprite_slots)
            }
        };
        let mut path_plan = match self.path_plan.take() {
            Some(plan) if plan.slots() == path_slots.as_slice() => plan,
            _ => {
                self.path_plan_builds += 1;
                SlotBasePlan::for_paths(device, queue, &self.paths, &path_slots)
            }
        };
        let mut backdrop_plan = match self.backdrop_plan.take() {
            Some(plan) if plan.slots() == backdrop_slots.as_slice() => plan,
            _ => {
                self.backdrop_plan_builds += 1;
                SlotBasePlan::for_backdrop_filters(
                    device,
                    queue,
                    &self.backdrop_filters,
                    &backdrop_slots,
                )
            }
        };
        shadow_plan.sync_transforms(queue, &input.scene.layers);
        quad_plan.sync_transforms(queue, &input.scene.layers);
        underline_plan.sync_transforms(queue, &input.scene.layers);
        glyph_plan.sync_transforms(queue, &input.scene.layers);
        sprite_plan.sync_transforms(queue, &input.scene.layers);
        path_plan.sync_transforms(queue, &input.scene.layers);
        backdrop_plan.sync_transforms(queue, &input.scene.layers);
        let shadow_frame_group = self.shadows.frame_bind_group(
            device,
            &self.globals,
            self.shadow_arena.buffer(),
            &self.shadow_args.visible,
        );
        let quad_frame_group = self.quads.frame_bind_group(
            device,
            &self.globals,
            self.arena.buffer(),
            &self.quad_args.visible,
        );
        let underline_frame_group = self.underlines.frame_bind_group(
            device,
            &self.globals,
            self.underline_arena.buffer(),
            &self.underline_args.visible,
        );
        let glyph_frame_group = self.glyphs.frame_bind_group(
            device,
            &self.globals,
            self.glyph_arena.buffer(),
            &self.glyph_args.visible,
        );
        let sprite_frame_group = self.sprites.frame_bind_group(
            device,
            &self.globals,
            self.sprite_arena.buffer(),
            &self.sprite_args.visible,
        );
        let path_frame_group =
            self.paths
                .frame_bind_group(device, &self.globals, self.path_arena.buffer());
        let backdrop_frame_group = self.backdrop_filters.frame_bind_group(
            device,
            &self.globals,
            self.backdrop_arena.buffer(),
        );
        let has_backdrop_filters = backdrop_slots.iter().any(|slot| slot.count > 0);
        let backdrop_texture_group = if has_backdrop_filters {
            self.ensure_backdrop_snapshot(device, target.width, target.height);
            let Some(view) = self.backdrop_snapshot_view.as_ref() else {
                return Err(FrameError::BackdropSourceUnavailable);
            };
            Some(
                self.backdrop_filters
                    .texture_bind_group(device, view, &self.backdrop_sampler),
            )
        } else {
            None
        };
        // Rebuilt per frame, not cached: a bind group names a texture view, and
        // an atlas page destroyed between frames leaves a cached one pointing at
        // a texture that no longer exists. The 16-byte uniform behind it *is*
        // cached, because its value is the page index and that never changes.
        let glyph_pages =
            self.atlas_page_bind_groups(device, queue, input.atlas, AtlasKind::Monochrome);
        let sprite_pages =
            self.atlas_page_bind_groups(device, queue, input.atlas, AtlasKind::Polychrome);
        let composite_frame_group = self.composite.frame_bind_group(device, &self.globals);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("frame"),
        });
        let debug_tiles = self.debug_tiles.clone();
        let mut stats;
        let scissor = damage.map(|damage| scissor_rect(damage, target.width, target.height));
        if scissor.is_some_and(|[_, _, width, height]| width > 0 && height > 0) {
            let clear_color = [
                target.clear.r as f32,
                target.clear.g as f32,
                target.clear.b as f32,
                target.clear.a as f32,
            ];
            queue.write_buffer(
                &self.damage_clear_color,
                0,
                bytemuck::bytes_of(&clear_color),
            );
        }
        let has_debug_tiles = self
            .prepare_debug_tiles(device, queue, &debug_tiles)
            .is_some();
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("frame"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: if scissor.is_some() {
                            wgpu::LoadOp::Load
                        } else {
                            wgpu::LoadOp::Clear(target.clear)
                        },
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: Default::default(),
            });
            if let Some([x, y, width, height]) = scissor {
                pass.set_scissor_rect(x, y, width, height);
            }
            if scissor.is_some_and(|[_, _, width, height]| width > 0 && height > 0) {
                pass.set_pipeline(&self.damage_clear_pipeline);
                pass.set_bind_group(0, &self.damage_clear_bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
            let started = Instant::now();
            // First, and under everything: `PrimitiveKind::ALL` declares
            // `Shadow` before `Quad` because the legacy sorter's own
            // discriminant does, so a card's drop shadow paints beneath the card.
            stats = issue_instanced(
                &mut pass,
                &self.shadows.pipeline,
                &shadow_plan,
                &shadow_frame_group,
                &self.shadow_args,
                input.mode,
                &shadow_resolved,
            );
            stats.merge(issue_instanced(
                &mut pass,
                &self.quads.pipeline,
                &quad_plan,
                &quad_frame_group,
                &self.quad_args,
                input.mode,
                &quad_resolved,
            ));
            stats.merge(issue_paths(
                &mut pass,
                &self.paths,
                &path_plan,
                &path_frame_group,
            ));
            // Between the quads and the text, which is where the legacy
            // discriminant puts `Underline`: a rule paints over the row
            // background and under the glyphs it belongs to.
            stats.merge(issue_instanced(
                &mut pass,
                &self.underlines.pipeline,
                &underline_plan,
                &underline_frame_group,
                &self.underline_args,
                input.mode,
                &underline_resolved,
            ));
            // After the quads and before the composites: text paints over the
            // chrome behind it and under a modal composited on top, which is the
            // painter order §5.3's kind grouping already puts the slots in.
            stats.merge(issue_sprites(
                &mut pass,
                SpriteDraw {
                    pipeline: &self.glyphs.pipeline,
                    plan: &glyph_plan,
                    pages: &glyph_pages,
                },
                &glyph_frame_group,
                &self.glyph_args,
                input.mode,
                &glyph_resolved,
            ));
            // After the text and before the composites, which is the order
            // `PrimitiveKind::ALL` declares and therefore the order the slot
            // table groups: an avatar or a thumbnail paints over the row label
            // behind it, and under a modal composited on top.
            stats.merge(issue_sprites(
                &mut pass,
                SpriteDraw {
                    pipeline: &self.sprites.pipeline,
                    plan: &sprite_plan,
                    pages: &sprite_pages,
                },
                &sprite_frame_group,
                &self.sprite_args,
                input.mode,
                &sprite_resolved,
            ));
            if has_backdrop_filters {
                drop(pass);
                let Some(source) = target.source else {
                    return Err(FrameError::BackdropSourceUnavailable);
                };
                let Some(snapshot) = self.backdrop_snapshot.as_ref() else {
                    return Err(FrameError::BackdropSourceUnavailable);
                };
                encoder.copy_texture_to_texture(
                    source.as_image_copy(),
                    snapshot.as_image_copy(),
                    wgpu::Extent3d {
                        width: target.width.min(source.width()),
                        height: target.height.min(source.height()),
                        depth_or_array_layers: 1,
                    },
                );
                pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("backdrop filters"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target.view,
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
                if let Some([x, y, width, height]) = scissor {
                    pass.set_scissor_rect(x, y, width, height);
                }
                if let Some(texture_group) = backdrop_texture_group.as_ref() {
                    stats.merge(issue_backdrop_filters(
                        &mut pass,
                        &self.backdrop_filters,
                        &backdrop_plan,
                        &backdrop_frame_group,
                        texture_group,
                    ));
                }
            }
            stats.merge(issue_composites(
                &mut pass,
                &self.composite,
                &composite_frame_group,
                &self.composite_args,
                &composite_plan,
                input.mode,
                &composite_resolved,
            ));
            if has_debug_tiles {
                let Some(bind_group) = self.debug_bind_group.as_ref() else {
                    return Err(FrameError::DebugOverlayUnavailable);
                };
                pass.set_pipeline(&self.debug_pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.draw(0..4, 0..u32::try_from(debug_tiles.len()).unwrap_or(u32::MAX));
            }
            timing.draw_issue = started.elapsed();
        }
        queue.submit(Some(encoder.finish()));
        self.shadow_plan = Some(shadow_plan);
        self.quad_plan = Some(quad_plan);
        self.underline_plan = Some(underline_plan);
        self.glyph_plan = Some(glyph_plan);
        self.sprite_plan = Some(sprite_plan);
        self.path_plan = Some(path_plan);
        self.backdrop_plan = Some(backdrop_plan);

        Ok(FrameOutput {
            stats,
            timing,
            layers_recomputed,
            primitives_resident,
            scene_upload_calls: u32::try_from(
                self.scene_upload_calls()
                    .saturating_sub(upload_calls_before),
            )
            .unwrap_or(u32::MAX),
            scene_upload_bytes: self
                .scene_upload_bytes()
                .saturating_sub(upload_bytes_before),
            plan_builds: u32::try_from(self.plan_builds().saturating_sub(plan_builds_before))
                .unwrap_or(u32::MAX),
        })
    }
}

fn scissor_rect(damage: Rect, target_width: u32, target_height: u32) -> [u32; 4] {
    if damage.is_empty()
        || !damage.min_x.is_finite()
        || !damage.min_y.is_finite()
        || !damage.max_x.is_finite()
        || !damage.max_y.is_finite()
    {
        return [0, 0, 0, 0];
    }
    let min_x = damage.min_x.floor().clamp(0.0, target_width as f32) as u32;
    let min_y = damage.min_y.floor().clamp(0.0, target_height as f32) as u32;
    let max_x = damage.max_x.ceil().clamp(0.0, target_width as f32) as u32;
    let max_y = damage.max_y.ceil().clamp(0.0, target_height as f32) as u32;
    [min_x, min_y, max_x.saturating_sub(min_x), max_y.saturating_sub(min_y)]
}

fn create_damage_clear_pipeline(
    device: &wgpu::Device,
) -> (wgpu::RenderPipeline, wgpu::BindGroup, wgpu::Buffer) {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("damage clear"),
        source: wgpu::ShaderSource::Wgsl(crate::render::shaders::DAMAGE_CLEAR_WGSL.into()),
    });
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("damage clear"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: std::num::NonZeroU64::new(16),
            },
            count: None,
        }],
    });
    let color = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("damage clear color"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("damage clear"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: color.as_entire_binding(),
        }],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("damage clear"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("damage clear"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vertex_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fragment_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: TARGET_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });
    (pipeline, bind_group, color)
}

fn create_debug_pipeline(device: &wgpu::Device) -> (wgpu::BindGroupLayout, wgpu::RenderPipeline) {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("tile refresh diagnostics"),
        source: wgpu::ShaderSource::Wgsl(crate::render::shaders::DEBUG_TILES_WGSL.into()),
    });
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("tile refresh diagnostics"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(8),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(
                        std::mem::size_of::<DebugTile>() as u64,
                    ),
                },
                count: None,
            },
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("tile refresh diagnostics"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("tile refresh diagnostics"),
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
                blend: Some(crate::render::pipelines::ALPHA_OVER),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });
    (bind_group_layout, pipeline)
}

/// One layer's shadows as the rectangles they actually paint into, in arena
/// slot order.
///
/// [`layer_glyph_bounds`]' argument about slot order applies unchanged. The
/// difference here is [`wgpui_core::patch::primitive::Shadow::drawn_bounds`]:
/// unlike every other kind, a
/// shadow's `origin`/`size` is *not* what it covers — the shader grows it by
/// three blur radii on every side and integrates the falloff across that
/// margin. Ordering a shadow by its unblurred rectangle would sort a wide soft
/// shadow as though it were the small hard rectangle at its centre.
fn layer_shadow_bounds(scene: &Scene, layer: LayerId) -> Vec<Rect> {
    let translation = layer_translation(scene, layer);
    scene
        .shadows
        .keys(layer)
        .into_iter()
        .filter_map(|key| scene.shadows.get(layer, key))
        .map(|shadow| {
            let (origin, size) = shadow.drawn_bounds();
            translate_rect(Rect::from_origin_size(origin, size), translation)
        })
        .collect()
}

/// One layer's underlines as bounding rectangles, in arena slot order.
///
/// [`layer_glyph_bounds`]' argument about slot order applies unchanged, and
/// there is nothing else to say: an underline's rectangle *is* its
/// `origin`/`size`. That this function is the boring one is the finding — see
/// [`layer_shadow_bounds`], which is the same function and is not.
fn layer_underline_bounds(scene: &Scene, layer: LayerId) -> Vec<Rect> {
    let translation = layer_translation(scene, layer);
    scene
        .underlines
        .keys(layer)
        .into_iter()
        .filter_map(|key| scene.underlines.get(layer, key))
        .map(|underline| {
            translate_rect(Rect::from_origin_size(underline.origin, underline.size), translation)
        })
        .collect()
}

fn layer_quads(scene: &Scene, layer: LayerId) -> Vec<Quad> {
    let translation = layer_translation(scene, layer);
    scene
        .quads
        .keys(layer)
        .into_iter()
        .filter_map(|key| scene.quads.get(layer, key).copied())
        .map(|mut quad| {
            quad.origin[0] += translation[0];
            quad.origin[1] += translation[1];
            quad
        })
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
    let translation = layer_translation(scene, layer);
    let mut bounds = Vec::new();
    for key in scene.glyph_runs.keys(layer) {
        let Some(run) = scene.glyph_runs.get(layer, key) else {
            continue;
        };
        bounds.extend(
            run.glyphs
                .iter()
                .map(|glyph| {
                    translate_rect(
                        Rect::from_origin_size(glyph.position, glyph.atlas_size),
                        translation,
                    )
                }),
        );
    }
    bounds
}

/// One layer's image sprites as bounding rectangles, in arena slot order.
///
/// [`layer_glyph_bounds`]' argument about slot order applies unchanged and is
/// not repeated. The one difference is which rectangle a sprite contributes: it
/// is the *drawn* rectangle (`origin`/`size`), not the tile's extent, because
/// that is what covers pixels — an image scaled down to a 24px avatar occludes
/// and orders as 24px however large its bitmap is. A sprite whose image has not
/// decoded is still a real rectangle with a real position; it draws nothing
/// because its tile is `NONE`, not because its bounds are empty.
fn layer_sprite_bounds(scene: &Scene, layer: LayerId) -> Vec<Rect> {
    let translation = layer_translation(scene, layer);
    scene
        .poly_sprites
        .keys(layer)
        .into_iter()
        .filter_map(|key| scene.poly_sprites.get(layer, key))
        .map(|sprite| {
            translate_rect(Rect::from_origin_size(sprite.origin, sprite.size), translation)
        })
        .collect()
}

fn layer_translation(scene: &Scene, layer: LayerId) -> [f32; 2] {
    scene
        .layers
        .get(layer)
        .map_or([0.0, 0.0], |layer| layer.transform().translation)
}

fn translate_rect(rectangle: Rect, translation: [f32; 2]) -> Rect {
    Rect {
        min_x: rectangle.min_x + translation[0],
        min_y: rectangle.min_y + translation[1],
        max_x: rectangle.max_x + translation[0],
        max_y: rectangle.max_y + translation[1],
    }
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
