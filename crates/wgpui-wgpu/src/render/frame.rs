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
    DrawMode, DrawStats, ResolvedArgs, SlotBasePlan, SpriteDraw, issue_composites, issue_instanced,
    issue_sprites,
};
use crate::render::pipelines::{
    CompositePipeline, Globals, MonoSpritePipeline, PolySpritePipeline, QuadPipeline,
    ShadowPipeline, TARGET_FORMAT,
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
    /// The instanced monochrome-sprite pipeline (Phase 5.6).
    pub glyphs: MonoSpritePipeline,
    /// The instanced polychrome-sprite pipeline (Phase 6.2).
    pub sprites: PolySpritePipeline,
    /// The one composite pipeline.
    pub composite: CompositePipeline,
    /// The shadow arena. Its own buffer, for the reason `sprite_arena` records:
    /// two kinds sharing a slot stride today is a coincidence of two
    /// independent layout decisions, not a licence to share an arena.
    pub shadow_arena: SlabBuffer,
    /// The quad arena.
    pub arena: SlabBuffer,
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
    /// Boundary texture retention (§5.5).
    pub textures: LayerTexturePool,
    globals: wgpu::Buffer,
    shadow_args: IndirectArgsBuffers,
    quad_args: IndirectArgsBuffers,
    glyph_args: IndirectArgsBuffers,
    sprite_args: IndirectArgsBuffers,
    composite_args: IndirectArgsBuffers,
    /// The per-slot bases and their bind group, kept until the slot table
    /// itself changes — see [`SlotBasePlan`], which is only true of it if
    /// something holds it across frames.
    shadow_plan: Option<SlotBasePlan>,
    quad_plan: Option<SlotBasePlan>,
    glyph_plan: Option<SlotBasePlan>,
    sprite_plan: Option<SlotBasePlan>,
    shadow_plan_builds: u64,
    quad_plan_builds: u64,
    glyph_plan_builds: u64,
    sprite_plan_builds: u64,
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
            shadows: ShadowPipeline::new(device),
            quads: QuadPipeline::new(device),
            glyphs: MonoSpritePipeline::new(device),
            sprites: PolySpritePipeline::new(device),
            composite: CompositePipeline::new(device),
            shadow_arena: SlabBuffer::new(device, "shadow arena"),
            arena: SlabBuffer::new(device, "quad arena"),
            glyph_arena: SlabBuffer::new(device, "glyph arena"),
            sprite_arena: SlabBuffer::new(device, "sprite arena"),
            textures: LayerTexturePool::default(),
            globals: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("frame globals"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            shadow_args: IndirectArgsBuffers::new(device, 1, 1),
            quad_args: IndirectArgsBuffers::new(device, 1, 1),
            glyph_args: IndirectArgsBuffers::new(device, 1, 1),
            sprite_args: IndirectArgsBuffers::new(device, 1, 1),
            composite_args: IndirectArgsBuffers::new(device, 1, 1),
            shadow_plan: None,
            quad_plan: None,
            glyph_plan: None,
            sprite_plan: None,
            shadow_plan_builds: 0,
            quad_plan_builds: 0,
            glyph_plan_builds: 0,
            sprite_plan_builds: 0,
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

    /// The same counter for the shadow pipeline's slot bases.
    pub fn shadow_plan_builds(&self) -> u64 {
        self.shadow_plan_builds
    }

    /// The same counter for the glyph pipeline's slot bases.
    pub fn glyph_plan_builds(&self) -> u64 {
        self.glyph_plan_builds
    }

    /// The same counter for the image-sprite pipeline's slot bases.
    pub fn sprite_plan_builds(&self) -> u64 {
        self.sprite_plan_builds
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
    /// **All four instanced kinds are drawn.** Phase 4 drew only `Quad` and
    /// said so here; Phase 5.6 added the `GlyphRun` half, Phase 6.2 the
    /// `PolySprite` half, and Phase 6.3 the `Shadow` half, all taking the
    /// identical route — upload, ordering, occlusion, indirect-argument
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
    /// What still has no pipeline: `underlines`, `paths`, `backdrop_blur`.
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &FrameInput<'_>,
        target: &OffscreenTarget,
    ) -> Result<FrameOutput, FrameError> {
        self.render_to(device, queue, input, &target.target())
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
        self.textures.begin_frame();
        let mut timing = FrameTiming::default();

        let table = input.scene.draw_slots();
        let shadow_slots: Vec<DrawSlot> = table.kind_slots(PrimitiveKind::Shadow).to_vec();
        let quad_slots: Vec<DrawSlot> = table.kind_slots(PrimitiveKind::Quad).to_vec();
        let glyph_slots: Vec<DrawSlot> = table.kind_slots(PrimitiveKind::GlyphRun).to_vec();
        let sprite_slots: Vec<DrawSlot> = table.kind_slots(PrimitiveKind::PolySprite).to_vec();
        let shadow_arena_slots = input.scene.arena_slots(PrimitiveKind::Shadow);
        let arena_slots = input.scene.arena_slots(PrimitiveKind::Quad);
        let glyph_arena_slots = input.scene.arena_slots(PrimitiveKind::GlyphRun);
        let sprite_arena_slots = input.scene.arena_slots(PrimitiveKind::PolySprite);
        let primitives_resident: u32 = shadow_slots
            .iter()
            .chain(quad_slots.iter())
            .chain(glyph_slots.iter())
            .chain(sprite_slots.iter())
            .map(|slot| slot.count)
            .sum();

        // --- 1. Upload.
        let started = Instant::now();
        let shadow_resident = input.scene.shadows.resident_bytes();
        let resident = input.scene.quads.resident_bytes();
        let glyph_resident = input.scene.glyph_runs.resident_bytes();
        let sprite_resident = input.scene.poly_sprites.resident_bytes();
        let shadows_grew = self
            .shadow_arena
            .reserve(device, shadow_resident.len() as u64);
        let grew = self.arena.reserve(device, resident.len() as u64);
        let glyphs_grew = self
            .glyph_arena
            .reserve(device, glyph_resident.len() as u64);
        let sprites_grew = self
            .sprite_arena
            .reserve(device, sprite_resident.len() as u64);
        if shadows_grew
            || grew
            || glyphs_grew
            || sprites_grew
            || self.uploaded_generation.is_none()
            || matches!(input.dirty, Dirty::All)
        {
            self.shadow_arena
                .upload_all(device, queue, shadow_resident);
            self.arena.upload_all(device, queue, resident);
            self.glyph_arena.upload_all(device, queue, glyph_resident);
            self.sprite_arena.upload_all(device, queue, sprite_resident);
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
            self.arena
                .upload(device, queue, resident, &kind_uploads(input.uploads, PrimitiveKind::Quad));
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
        let shadow_plan = match self.shadow_plan.take() {
            Some(plan) if plan.slots() == shadow_slots.as_slice() => plan,
            _ => {
                self.shadow_plan_builds += 1;
                SlotBasePlan::for_shadows(device, queue, &self.shadows, &shadow_slots)
            }
        };
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
        let sprite_plan = match self.sprite_plan.take() {
            Some(plan) if plan.slots() == sprite_slots.as_slice() => plan,
            _ => {
                self.sprite_plan_builds += 1;
                SlotBasePlan::for_poly_sprites(device, queue, &self.sprites, &sprite_slots)
            }
        };
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
        let mut stats;
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("frame"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(target.clear),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: Default::default(),
            });
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
        self.shadow_plan = Some(shadow_plan);
        self.quad_plan = Some(quad_plan);
        self.glyph_plan = Some(glyph_plan);
        self.sprite_plan = Some(sprite_plan);

        Ok(FrameOutput {
            stats,
            timing,
            layers_recomputed,
            primitives_resident,
        })
    }
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
    scene
        .shadows
        .keys(layer)
        .into_iter()
        .filter_map(|key| scene.shadows.get(layer, key))
        .map(|shadow| {
            let (origin, size) = shadow.drawn_bounds();
            Rect::from_origin_size(origin, size)
        })
        .collect()
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
    scene
        .poly_sprites
        .keys(layer)
        .into_iter()
        .filter_map(|key| scene.poly_sprites.get(layer, key))
        .map(|sprite| Rect::from_origin_size(sprite.origin, sprite.size))
        .collect()
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
