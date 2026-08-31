//! The whole pipeline, driven once per frame, from a `Description`.
//! See docs/gpu-native-architecture.md §2's picture end to end.
//!
//! # Not in §3.5's file map, and why — the fourth such deviation
//!
//! §3.5's `window/` subtree lists four files, all of them OS-integration
//! (`dispatcher`, `keyboard`, `resize_detector`, `app_menu`), and `render/`'s
//! deepest orchestrator is `frame.rs`, which Phase 4 already added for the same
//! kind of reason. Neither names a home for the thing that owns *state across
//! frames*.
//!
//! That thing did not need to exist until now, and the reason is the finding
//! this file exists to record. Every phase through 5.6 tested a single frame or
//! a short fixed sequence, and each built the retained state its own test
//! needed, inline, from nothing: `glyph_sprite_draw.rs` hand-writes a
//! `ScenePatch`; `emit.rs`'s own tests keep a `Window` struct in `#[cfg(test)]`.
//! A window is the first consumer that must hold `Reconciler`, `LayoutTree`,
//! `Emitter`, `Scene` and `FrameRenderer` together and alive for as long as the
//! window is, and hold them **consistently** — see [`FrameLoop`]'s own doc for
//! the specific invariant that turned out to be load-bearing.
//!
//! Phase 1's report recorded six such deviations, Phase 2's two, Phase 3's four,
//! Phase 4's one. This is Phase 6's, in the same shape.
//!
//! # Text materialization
//!
//! The loop also owns the renderer-side text service. Raw string descriptions
//! stay renderer-independent until this boundary, where cached shaping,
//! rasterization, atlas allocation, and atlas texture uploads are performed.
//! The resulting glyph runs still enter the ordinary retained patch protocol;
//! no text-specific draw path bypasses reconciliation.

use wgpui_core::boundary::compositor::{Composite, CompositeEntry, TiledVisit};
use wgpui_core::geometry::Rect;
use wgpui_core::invalidation::axes::Invalidation;
use wgpui_core::invalidation::request::FrameSignals;
use wgpui_core::patch::PatchError;
use wgpui_core::patch::apply::apply;
use wgpui_core::patch::emit::{Emission, Emit, EmitContext, EmitError, Emitter, FrameEmission};
use wgpui_core::patch::primitive::{Glyph, GlyphRun, Material, Quad};
use wgpui_core::reconcile::description::{Description, DescriptionInteraction, RawText};
use wgpui_core::reconcile::diff_key::{ReconcileKey, compare_by_equality};
use wgpui_core::reconcile::plan::FrameStats;
use wgpui_core::reconcile::reconciler::{ReconcileError, Reconciler};
use wgpui_core::reconcile::walk::shared_walk;
use wgpui_core::scene::Scene;
use wgpui_core::scene::layer::LayerId;
use wgpui_core::scene::tile::TileCoord;
use wgpui_layout::taffy_tree::{
    Dimension, Display, FlexDirection, LayoutSize, LayoutStyle, LayoutTree, definite,
};

use crate::debug::{DebugTile, PerformanceDebug};
use crate::render::atlas::{AtlasTileSource, GlyphAtlas};
use crate::render::atlas_upload::AtlasTextures;
use crate::render::draw::DrawMode;
use crate::render::frame::{
    Dirty, FrameError, FrameInput, FrameOutput, FrameRenderer, OffscreenTarget, RenderTarget,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use wgpui_core::patch::primitive::GlyphRun as CoreGlyphRun;
use wgpui_text::patch::{RunPlacement, glyph_runs};
use wgpui_text::raster::GlyphRasterizer;
use wgpui_text::shaping::{FontRun, SharedString, TextShaper};

/// Why a frame could not be produced.
#[derive(Debug)]
pub enum LoopError {
    /// Reconciliation failed.
    Reconcile(ReconcileError),
    /// Emission failed, including any layout error underneath it.
    Emit(EmitError),
    /// The patch could not be applied to the scene.
    Patch(PatchError),
    /// The GPU frame failed.
    Frame(FrameError),
    /// Text shaping failed before a frame could be emitted.
    Text(String),
    /// The plan had no root, so there is no layout node to size the frame
    /// against. Only reachable from a `Description` that produced no nodes at
    /// all.
    NoRoot,
}

impl std::fmt::Display for LoopError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoopError::Reconcile(error) => write!(formatter, "reconcile: {error}"),
            LoopError::Emit(error) => write!(formatter, "emit: {error}"),
            LoopError::Patch(error) => write!(formatter, "patch: {error}"),
            LoopError::Frame(error) => write!(formatter, "frame: {error}"),
            LoopError::Text(error) => write!(formatter, "text: {error}"),
            LoopError::NoRoot => write!(formatter, "the frame plan has no root node"),
        }
    }
}

impl std::error::Error for LoopError {}

impl From<ReconcileError> for LoopError {
    fn from(error: ReconcileError) -> Self {
        LoopError::Reconcile(error)
    }
}

impl From<EmitError> for LoopError {
    fn from(error: EmitError) -> Self {
        LoopError::Emit(error)
    }
}

impl From<PatchError> for LoopError {
    fn from(error: PatchError) -> Self {
        LoopError::Patch(error)
    }
}

impl From<FrameError> for LoopError {
    fn from(error: FrameError) -> Self {
        LoopError::Frame(error)
    }
}

/// What one driven frame did, on both sides of §2's seam.
#[derive(Debug)]
pub struct LoopFrame {
    /// The reconciliation half: what was reused and what was rebuilt.
    pub reconciled: FrameStats,
    /// The emission half: what the patch contained and what each boundary did.
    pub emission: FrameEmission,
    /// The layers the patch touched — exactly what was handed to the GPU as
    /// dirty.
    pub dirty_layers: Vec<LayerId>,
    /// Bytes the patch's application scheduled for upload.
    pub uploaded_bytes: u64,
    /// Whether this frame's viewport differed from the previous frame's, and so
    /// forced a full recompute regardless of what the patch said.
    ///
    /// See [`FrameLoop::draw`]'s note on why the clip is a second, independent
    /// source of dirtiness.
    pub viewport_changed: bool,
    /// The GPU half.
    pub frame: FrameOutput,
    pub interactions: Vec<InteractionRegistration>,
    /// Whether the native handler should schedule another frame for a fading
    /// diagnostic overlay.
    pub needs_redraw: bool,
}

#[derive(Debug)]
pub struct InteractionRegistration {
    pub address: wgpui_core::reconcile::instance::InstanceKey,
    pub bounds: Rect,
    pub order: u64,
    pub interaction: DescriptionInteraction,
}

impl LoopFrame {
    /// Whether this frame changed nothing at all — no records touched, no
    /// bytes uploaded, no layer dirty, and the same viewport as last frame.
    ///
    /// The shape a window sitting still has, and the one `frame.rs`'s
    /// clean-frame path is written for.
    pub fn was_idle(&self) -> bool {
        self.emission.patch.is_empty()
            && self.dirty_layers.is_empty()
            && self.uploaded_bytes == 0
            && !self.viewport_changed
    }
}

/// Everything one frame needs that is not the description itself.
///
/// Grouped rather than passed as eight parameters, which is `frame.rs`'s own
/// [`FrameInput`] idiom one level up and is what a nine-argument `draw` was
/// told to become. The grouping also reads better at the call site: a window's
/// loop varies exactly one of these per frame — the target's view — and naming
/// the rest in a struct makes that obvious.
pub struct LoopInput<'a> {
    /// The uploaded atlas pages a glyph draw samples, if this frame has text.
    pub atlas: Option<&'a AtlasTextures>,
    /// Where the colour goes, and what the pass clears with.
    pub target: &'a RenderTarget<'a>,
    /// How the fixed draw sequence reaches the device.
    pub mode: DrawMode,
    /// This frame's invalidation signals.
    pub signals: &'a FrameSignals,
    /// The composite entries, in draw order (§5.5). Empty for a scene with no
    /// `.boundary()` in it, which is every scene Phase 6 draws.
    pub composites: &'a [CompositeEntry],
}

/// Everything one window holds across frames.
///
/// # The invariant that turned out to be load-bearing
///
/// `FrameRenderer` keeps a `SlotBasePlan` per instanced kind and rebuilds it
/// only when the slot table changes — that is what makes Phase 4's gate ("draw
/// issuing does not grow with the primitive count") true of a *steady* frame
/// rather than only of a first one. It also keeps `uploaded_generation`, which
/// decides between a full arena upload and §5.0's delta upload.
///
/// Both are only correct if the `Scene` they are describing is the same `Scene`
/// every frame. Before this phase nothing enforced that, because nothing ran
/// more than a handful of frames: a test that built a fresh `Scene` and a fresh
/// `FrameRenderer` per frame would have passed every gate written so far and
/// been silently O(n) per frame forever. Owning all five together, in one value
/// with no way to swap one out, is what makes the pairing structural instead of
/// a convention — and [`FrameLoop::draw_plan_builds`] is what lets a test say
/// so out loud rather than assume it.
pub struct FrameLoop {
    reconciler: Reconciler,
    layout: LayoutTree,
    emitter: Emitter,
    scene: Scene,
    renderer: FrameRenderer,
    text_shaper: TextShaper,
    text_rasterizer: GlyphRasterizer,
    text_atlas: GlyphAtlas,
    atlas_textures: AtlasTextures,
    presentation_target: Option<OffscreenTarget>,
    prepared_text: HashMap<TextCacheKey, PreparedText>,
    frames: u64,
    last_viewport: Option<[f32; 2]>,
    viewport_recomputes: u64,
    tile_flash_frames: HashMap<(wgpui_core::scene::layer::BoundaryId, TileCoord), u32>,
    tile_refresh_rates:
        HashMap<(wgpui_core::scene::layer::BoundaryId, TileCoord), TileRefreshRate>,
    damage_refresh_regions: Vec<DamageRefreshRegion>,
    interaction_dirty_regions: Vec<Rect>,
    performance_debug: PerformanceDebug,
    scale_factor: f32,
}

#[derive(Clone)]
struct PreparedText {
    width: f32,
    height: f32,
    baseline: f32,
    runs: Arc<Vec<CoreGlyphRun>>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct TextCacheKey {
    value: Arc<str>,
    font_size_bits: u32,
}

#[derive(Clone, Copy)]
struct TileRefreshRate {
    window_start: Instant,
    samples: u32,
    updates: u32,
    frames_per_second: f32,
    regular: bool,
}

#[derive(Clone, Copy)]
struct DamageRefreshRegion {
    rect: Rect,
    frames_remaining: u32,
    window_start: Instant,
    samples: u32,
    updates: u32,
    frames_per_second: f32,
    regular: bool,
}

#[derive(Clone, Copy)]
struct DebugRefreshRegion {
    rect: Rect,
    frames_remaining: u32,
    updates: u32,
    frames_per_second: f32,
    regular: bool,
}

impl FrameLoop {
    /// Build every pipeline once, and start with an empty scene.
    pub fn new(device: &wgpu::Device) -> FrameLoop {
        FrameLoop {
            reconciler: Reconciler::new(),
            layout: LayoutTree::new(),
            emitter: Emitter::new(),
            scene: Scene::new(),
            renderer: FrameRenderer::new(device),
            text_shaper: TextShaper::new(),
            text_rasterizer: GlyphRasterizer::new(),
            text_atlas: GlyphAtlas::default(),
            atlas_textures: AtlasTextures::new(GlyphAtlas::default().page_size()),
            presentation_target: None,
            prepared_text: HashMap::new(),
            frames: 0,
            last_viewport: None,
            viewport_recomputes: 0,
            tile_flash_frames: HashMap::new(),
            tile_refresh_rates: HashMap::new(),
            damage_refresh_regions: Vec::new(),
            interaction_dirty_regions: Vec::new(),
            performance_debug: PerformanceDebug::default(),
            scale_factor: 1.0,
        }
    }

    /// The resident scene, for inspection.
    pub fn scene(&self) -> &Scene {
        &self.scene
    }

    /// How many frames this loop has drawn.
    pub fn frames(&self) -> u64 {
        self.frames
    }

    /// How many times the quad pipeline's slot bases have been rebuilt.
    ///
    /// A window sitting still must not keep raising this. See this type's doc
    /// for why that is a property of the loop rather than of the renderer.
    pub fn draw_plan_builds(&self) -> u64 {
        self.renderer.draw_plan_builds()
    }

    /// The same counter for the glyph pipeline.
    pub fn glyph_plan_builds(&self) -> u64 {
        self.renderer.glyph_plan_builds()
    }

    /// Number of distinct raw strings retained by the text materializer.
    pub fn prepared_text_count(&self) -> usize {
        self.prepared_text.len()
    }

    /// How many frames were recomputed in full because the viewport changed
    /// rather than because the patch named a layer.
    ///
    /// Exists so the rule in [`Self::draw`]'s "the clip is dirtiness too" note
    /// is observable rather than asserted: a resized window must raise this, and
    /// a still one must not.
    pub fn viewport_recomputes(&self) -> u64 {
        self.viewport_recomputes
    }

    /// Update the opt-in diagnostics used by subsequent frames.
    pub fn set_performance_debug(&mut self, debug: PerformanceDebug) {
        self.performance_debug = debug;
    }

    /// Mark a hitbox region whose hover state changed. This is consumed by the
    /// next frame as a dirty-region hint without invalidating unrelated tiles.
    pub fn mark_interaction_dirty(&mut self, region: Rect) {
        self.interaction_dirty_regions.push(region);
    }

    /// Set the native display scale used when shaping and rasterising text.
    /// Layout and the render target are expressed in physical pixels, so
    /// keeping this conversion at the renderer boundary prevents glyphs from
    /// being laid out at logical size and then squeezed into a physical frame.
    pub fn set_scale_factor(&mut self, scale_factor: f64) {
        if scale_factor.is_finite() && scale_factor > 0.0 {
            let scale_factor = scale_factor as f32;
            if self.scale_factor.to_bits() != scale_factor.to_bits() {
                self.scale_factor = scale_factor;
                self.prepared_text.clear();
            }
        }
    }

    /// Every quad resident in the scene, in paint order, across every layer.
    ///
    /// # Why a checker has to read these rather than compute them
    ///
    /// Phase 6 found this out by getting it wrong first. The reference scene's
    /// quad is described as 320x180, and at 800x500 that is exactly what layout
    /// gives it — so a check written against the constant passes. At 320x200 the
    /// column's two children want 244px of a 200px box, taffy's default
    /// `flex_shrink` of 1 applies, and the quad is legitimately 147px tall. The
    /// renderer was right and the constant was wrong.
    ///
    /// "Matches what was described" therefore means *matches the primitive the
    /// description produced through layout and emission*, which is this — not a
    /// number copied out of the description by hand.
    pub fn resident_quads(&self) -> Vec<Quad> {
        let mut quads = Vec::new();
        for layer in self.scene.layers.ids() {
            for key in self.scene.quads.keys(layer) {
                if let Some(quad) = self.scene.quads.get(layer, key) {
                    quads.push(*quad);
                }
            }
        }
        quads
    }

    /// Every resident glyph, paired with its run's colour, in arena slot order.
    ///
    /// Slot order, not merely paint order: `frame.rs`'s own `layer_glyph_bounds`
    /// documents why the two coincide — `PrimitiveStore::keys` returns paint
    /// order and `reflow` assigns each run's `slot_offset` by walking exactly
    /// that order cumulatively — and a pixel comparison that walked them in any
    /// other order would compare the right texels against the wrong glyph.
    pub fn resident_glyphs(&self) -> Vec<(Glyph, [f32; 4])> {
        let mut glyphs = Vec::new();
        for layer in self.scene.layers.ids() {
            for key in self.scene.glyph_runs.keys(layer) {
                let Some(run) = self.scene.glyph_runs.get(layer, key) else {
                    continue;
                };
                glyphs.extend(run.glyphs.iter().map(|glyph| (*glyph, run.color)));
            }
        }
        glyphs
    }

    /// Reconcile `description`, lay it out at `target`'s size, emit it, apply
    /// the patch, and render the resulting scene into `target`.
    ///
    /// The whole of §2's picture, in the order §2 draws it. Nothing here is new
    /// machinery: every step is a call into a stage some earlier phase built
    /// and proved, and the contribution of this function is that they now run
    /// one after another, on retained state, once per displayed frame.
    ///
    /// # Dirtiness is taken from the patch, not guessed
    ///
    /// [`Dirty`] names the layers whose ordering and occlusion are recomputed,
    /// and `ScenePatch::layers()` already answers exactly that question: a
    /// layer no op in the patch names has the same primitives, in the same
    /// slots, as the frame before, and its previous compute results are still
    /// sitting in the argument buffers. An empty patch therefore yields
    /// `Dirty::Some(&[])` — the clean-frame path — rather than `Dirty::All`,
    /// which would recompute a settled window's whole scene every frame and
    /// make `frame.rs`'s clean-frame path unreachable from a real loop.
    ///
    /// # The clip is dirtiness too, and the patch cannot know it
    ///
    /// That rule is right about *scene content* and incomplete on its own, which
    /// is a thing this loop is the first code in the project able to notice:
    /// occlusion is computed against `FrameInput::clip`, and the clip is the
    /// window's rectangle. Resize the window without changing a single
    /// primitive and the patch is legitimately empty, so `frame.rs` skips step 2
    /// (`//! A clean layer's results from a previous frame are still sitting
    /// there`) and the *indirect arguments still describe the old rectangle*. A
    /// window shrunk and then grown back would keep drawing the shrunk frame's
    /// cull decisions, so content that left the small viewport never comes back.
    ///
    /// The viewport is therefore a second, independent source of dirtiness, and
    /// a frame whose clip moved is `Dirty::All` whatever the patch says.
    ///
    /// Phase 6's own reference scene cannot exhibit the bug — its text element
    /// is `width: 100%`, so every resize re-emits its runs and the patch is
    /// never empty across one — which is exactly why this was found by reading
    /// the rule against `frame.rs`'s contract rather than by a failing test.
    /// [`Self::viewport_recomputes`] makes the fix observable in the direction
    /// that *can* be checked: it must rise on a resized frame and never on a
    /// still one.
    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        description: Description,
        input: &LoopInput<'_>,
    ) -> Result<LoopFrame, LoopError> {
        let mut description = description;
        self.materialize_raw_text(&mut description)?;
        let mut plan = self.reconciler.reconcile(description, &mut self.layout)?;
        let root = plan
            .root()
            .map(|node| node.layout_node)
            .ok_or(LoopError::NoRoot)?;
        let width = input.target.width.max(1) as f32;
        let height = input.target.height.max(1) as f32;
        self.layout
            .compute_layout(root, definite(width, height))
            .map_err(EmitError::from)?;
        let interactions = self.collect_interactions(
            &mut plan,
            input.signals,
            Rect::from_origin_size([0.0, 0.0], [width, height]),
        )?;
        let emission = self
            .emitter
            .emit(&plan, &self.layout, input.signals, &mut self.scene)?;
        let uploads = apply(&mut self.scene, &emission.patch)?;
        let dirty_layers = emission.patch.layers();
        let viewport = [width, height];
        let viewport_changed = self.last_viewport != Some(viewport);
        if viewport_changed {
            self.last_viewport = Some(viewport);
            self.viewport_recomputes += 1;
        }
        let interaction_dirty_regions = self.interaction_dirty_regions.clone();
        let transform_only = emission
            .composites
            .iter()
            .any(|composite| composite.composite == Composite::TransformOnly);
        let debug_tiles = self.refresh_debug_tiles(
            &emission,
            self.performance_debug.tile_refresh_flash(),
            Rect::from_origin_size([0.0, 0.0], viewport),
            viewport_changed,
            &interaction_dirty_regions,
        );
        self.interaction_dirty_regions.clear();
        let debug_damage = debug_tiles.iter().fold(None, |damage, tile| {
            Some(union_damage(
                damage,
                Rect::from_origin_size(
                    [tile.origin_size[0], tile.origin_size[1]],
                    [tile.origin_size[2], tile.origin_size[3]],
                ),
            ))
        });
        let has_debug_damage = debug_damage.is_some();
        let needs_redraw = !debug_tiles.is_empty();
        self.renderer.set_debug_tiles(debug_tiles);
        let can_preserve_presentation = input
            .target
            .source
            .is_some_and(|source| source.usage().contains(wgpu::TextureUsages::COPY_DST));

        let damage = if viewport_changed {
            None
        } else {
            let mut damage = None;
            for region in emission.damage.iter().copied() {
                damage = Some(union_damage(damage, region));
            }
            for region in interaction_dirty_regions.iter().copied() {
                damage = Some(union_damage(damage, region));
            }
            if let Some(debug_damage) = debug_damage {
                damage = Some(union_damage(damage, debug_damage));
            }
            Some(damage.unwrap_or(Rect::EMPTY))
        };

        self.atlas_textures
            .sync(device, queue, &mut self.text_atlas);
        let owned_atlas = Some(&self.atlas_textures);
        let frame_input = FrameInput {
            scene: &self.scene,
            clip: Rect::from_origin_size([0.0, 0.0], [width, height]),
            poison: &[],
            dirty: if viewport_changed
                || transform_only
                || has_debug_damage
                || !can_preserve_presentation
            {
                Dirty::All
            } else {
                Dirty::Some(&dirty_layers)
            },
            uploads: uploads.entries(),
            composites: input.composites,
            registry: None,
            atlas: input.atlas.or(owned_atlas),
            viewport: [width, height],
            mode: input.mode,
        };
        let frame = if can_preserve_presentation {
            let target_is_new = self.presentation_target.as_ref().is_none_or(|target| {
                target.width != input.target.width || target.height != input.target.height
            });
            if target_is_new {
                self.presentation_target = Some(OffscreenTarget::new(
                    device,
                    input.target.width,
                    input.target.height,
                ));
            }
            if let Some(retained_target) = self.presentation_target.as_ref() {
                let retained_render_target = retained_target.target();
                let frame = self.renderer.render_to_with_damage(
                    device,
                    queue,
                    &frame_input,
                    &retained_render_target,
                    if target_is_new { None } else { damage },
                )?;
                if let Some(destination) = input.target.source {
                    retained_target.copy_to_texture(device, queue, destination);
                }
                frame
            } else {
                self.renderer
                    .render_to(device, queue, &frame_input, input.target)?
            }
        } else {
            self.renderer
                .render_to(device, queue, &frame_input, input.target)?
        };
        self.frames += 1;

        Ok(LoopFrame {
            reconciled: plan.stats(),
            emission,
            dirty_layers,
            uploaded_bytes: uploads.byte_count(),
            viewport_changed,
            frame,
            interactions,
            needs_redraw,
        })
    }

    fn refresh_debug_tiles(
        &mut self,
        emission: &FrameEmission,
        flash: crate::debug::TileRefreshFlash,
        viewport: Rect,
        viewport_changed: bool,
        interaction_dirty_regions: &[Rect],
    ) -> Vec<DebugTile> {
        if !flash.enabled {
            self.tile_flash_frames.clear();
            self.tile_refresh_rates.clear();
            self.damage_refresh_regions.clear();
            return Vec::new();
        }
        self.tile_flash_frames.retain(|_, frames| {
            *frames = frames.saturating_sub(1);
            *frames > 0
        });
        self.tile_refresh_rates.retain(|key, _| {
            emission
                .tiled_visits
                .iter()
                .any(|visit| visit.boundary == key.0 && visit.visible.contains(&key.1))
        });
        self.damage_refresh_regions.retain_mut(|region| {
            region.frames_remaining = region.frames_remaining.saturating_sub(1);
            region.frames_remaining > 0
        });
        let now = Instant::now();
        if viewport_changed {
            self.record_damage_refresh(viewport, flash.duration_frames.max(1), now);
        }
        if emission.tiled_visits.is_empty() {
            for region in emission.damage.iter().copied() {
                self.record_damage_refresh(region, flash.duration_frames.max(1), now);
            }
        }
        for visit in &emission.tiled_visits {
            for tile in &visit.visible {
                let key = (visit.boundary, *tile);
                let tile_rect = screen_tile_rect(visit, *tile);
                let presentation_changed = viewport_changed
                    || emission.damage.iter().any(|region| region.intersects(&tile_rect));
                let interaction_changed = interaction_dirty_regions
                    .iter()
                    .any(|region| region.intersects(&tile_rect));
                if presentation_changed || interaction_changed {
                    self.tile_flash_frames
                        .insert(key, flash.duration_frames.max(1));
                    self.record_tile_refresh(key, now);
                }
            }
        }
        if self.tile_flash_frames.is_empty() && self.damage_refresh_regions.is_empty() {
            return Vec::new();
        }
        let mut refresh_regions = Vec::new();
        for refresh_region in &self.damage_refresh_regions {
            let region = refresh_region.rect.intersect(&viewport);
            if region.is_empty() {
                continue;
            }
            refresh_regions.push(DebugRefreshRegion {
                rect: region,
                frames_remaining: refresh_region.frames_remaining,
                updates: refresh_region.updates,
                frames_per_second: refresh_region.frames_per_second,
                regular: refresh_region.regular,
            });
        }
        for visit in &emission.tiled_visits {
            for tile in &visit.visible {
                let key = (visit.boundary, *tile);
                if !self.tile_flash_frames.contains_key(&key) && !flash.viewport_grid {
                    continue;
                }
                let tile_rect = screen_tile_rect(visit, *tile).intersect(&viewport);
                if tile_rect.is_empty() {
                    continue;
                }
                let rate = self.tile_refresh_rates.get(&key);
                refresh_regions.push(DebugRefreshRegion {
                    rect: tile_rect,
                    frames_remaining: self.tile_flash_frames.get(&key).copied().unwrap_or(0),
                    updates: rate.map_or(0, |rate| rate.updates),
                    frames_per_second: rate.map_or(0.0, |rate| rate.frames_per_second),
                    regular: rate.is_some_and(|rate| rate.regular),
                });
            }
        }
        let refresh_regions = if flash.viewport_grid {
            refresh_regions
        } else {
            select_debug_refresh_regions(refresh_regions)
        };
        refresh_regions
            .into_iter()
            .map(|refresh_region| {
                let fade = tile_flash_fade(refresh_region.frames_remaining, flash.duration_frames);
                let mut tile = DebugTile {
                    origin_size: [
                        refresh_region.rect.min_x,
                        refresh_region.rect.min_y,
                        refresh_region.rect.width(),
                        refresh_region.rect.height(),
                    ],
                    color: faded_color(flash.color, fade),
                    border_width: 3.0,
                    _padding: [0.0; 7],
                };
                if refresh_region.regular {
                    tile = tile.with_refresh_rate(refresh_region.frames_per_second);
                } else {
                    tile = tile.with_refresh_count(refresh_region.updates);
                }
                tile
            })
            .collect()
    }

    fn record_tile_refresh(
        &mut self,
        key: (wgpui_core::scene::layer::BoundaryId, TileCoord),
        now: Instant,
    ) {
        let rate = self.tile_refresh_rates.entry(key).or_insert(TileRefreshRate {
            window_start: now,
            samples: 0,
            updates: 0,
            frames_per_second: 0.0,
            regular: false,
        });
        rate.samples = rate.samples.saturating_add(1);
        rate.updates = rate.updates.saturating_add(1);
        let elapsed = now.duration_since(rate.window_start).as_secs_f32();
        if elapsed >= 0.25 {
            rate.frames_per_second = rate.samples as f32 / elapsed;
            rate.regular = rate.samples >= 10;
            rate.window_start = now;
            rate.samples = 0;
        }
    }

    fn record_damage_refresh(&mut self, rect: Rect, duration_frames: u32, now: Instant) {
        if rect.is_empty() {
            return;
        }
        if let Some(region) = self
            .damage_refresh_regions
            .iter_mut()
            .find(|region| same_rect(region.rect, rect))
        {
            region.frames_remaining = duration_frames;
            region.samples = region.samples.saturating_add(1);
            region.updates = region.updates.saturating_add(1);
            let elapsed = now.duration_since(region.window_start).as_secs_f32();
            if elapsed >= 0.25 {
                region.frames_per_second = region.samples as f32 / elapsed;
                region.regular = region.samples >= 10;
                region.window_start = now;
                region.samples = 0;
            }
        } else {
            self.damage_refresh_regions.push(DamageRefreshRegion {
                rect,
                frames_remaining: duration_frames,
                window_start: now,
                samples: 1,
                updates: 1,
                frames_per_second: 0.0,
                regular: false,
            });
        }
    }

    fn collect_interactions(
        &self,
        plan: &mut wgpui_core::reconcile::plan::FramePlan,
        signals: &FrameSignals,
        viewport: Rect,
    ) -> Result<Vec<InteractionRegistration>, LoopError> {
        let walked =
            shared_walk(plan, &self.layout, signals, Some(viewport)).map_err(EmitError::from)?;
        let mut result = Vec::new();
        for index in 0..plan.nodes().len() {
            let node = plan.nodes()[index];
            if let Some(interaction) = plan.take_interaction(index) {
                let geometry = walked.get(index).ok_or(LoopError::NoRoot)?;
                let visible_bounds = geometry.visible_bounds;
                if !visible_bounds.is_empty() {
                    result.push(InteractionRegistration {
                        address: node.address,
                        bounds: visible_bounds,
                        order: index as u64,
                        interaction,
                    });
                }
            }
        }
        Ok(result)
    }

    fn materialize_raw_text(&mut self, description: &mut Description) -> Result<(), LoopError> {
        self.materialize_raw_text_with_metrics(description, None, None)
    }

    fn materialize_raw_text_with_metrics(
        &mut self,
        description: &mut Description,
        inherited_size: Option<f32>,
        inherited_color: Option<[f32; 4]>,
    ) -> Result<(), LoopError> {
        if let Some(raw) = description.take_raw_text() {
            let value = raw.shared_value();
            let (local_size, local_color) = description.text_metrics_value();
            let font_size = local_size.or(inherited_size);
            let color = local_color.or(inherited_color);
            let font_size = font_size
                .filter(|size| size.is_finite() && *size > 0.0)
                .unwrap_or(14.0)
                * self.scale_factor;
            let key = TextCacheKey {
                value: Arc::clone(&value),
                font_size_bits: font_size.to_bits(),
            };
            let prepared = match self.prepared_text.get(&key).cloned() {
                Some(prepared) => prepared,
                None => self.prepare_text(raw, font_size)?,
            };
            let text_color = color.unwrap_or([1.0, 1.0, 1.0, 1.0]);
            let runs = Arc::new(
                prepared
                    .runs
                    .iter()
                    .map(|run| {
                        let mut run = run.clone();
                        run.color = text_color;
                        run
                    })
                    .collect::<Vec<_>>(),
            );
            let baseline = prepared.baseline;
            description.set_intrinsic_size(prepared.width, prepared.height);
            description.set_text_emitter(move |context: &EmitContext, emission: &mut Emission| {
                for run in runs.iter() {
                    let mut positioned = run.clone();
                    for glyph in &mut positioned.glyphs {
                        glyph.position[0] += context.bounds.x;
                        glyph.position[1] += context.bounds.y + baseline;
                    }
                    emission.glyph_run(positioned);
                }
            });
        }
        let (local_size, local_color) = description.text_metrics_value();
        let effective_size = local_size.or(inherited_size);
        let effective_color = local_color.or(inherited_color);
        for child in description.child_descriptions_mut() {
            self.materialize_raw_text_with_metrics(child, effective_size, effective_color)?;
        }
        Ok(())
    }

    fn prepare_text(&mut self, raw: RawText, font_size: f32) -> Result<PreparedText, LoopError> {
        let value = raw.shared_value();
        let shared = SharedString::from(value.as_ref());
        let font = wgpui_text::shaping::font("sans-serif");
        let font_id = self
            .text_shaper
            .resolve_font(&font)
            .map_err(|error| LoopError::Text(error.to_string()))?;
        let font_runs = vec![FontRun::new(shared.len(), font_id)];
        let line = self
            .text_shaper
            .shape_line(&shared, font_size, &font_runs)
            .map_err(|error| LoopError::Text(error.to_string()))?;
        let placement = RunPlacement {
            color: [1.0, 1.0, 1.0, 1.0],
            ..RunPlacement::default()
        };
        let mut source = AtlasTileSource::new(&mut self.text_atlas, |key| {
            self.text_rasterizer
                .rasterize(&mut self.text_shaper, key)
                .ok()
        });
        let (converted, _) = glyph_runs(&line, placement, &mut source);
        let height = 20.0_f32.max(line.ascent + line.descent);
        let prepared = PreparedText {
            width: line.width,
            height,
            baseline: (height - line.ascent - line.descent) * 0.5 + line.ascent,
            runs: Arc::new(converted),
        };
        self.prepared_text.insert(
            TextCacheKey {
                value,
                font_size_bits: font_size.to_bits(),
            },
            prepared.clone(),
        );
        Ok(prepared)
    }
}

fn union_damage(current: Option<Rect>, next: Rect) -> Rect {
    match current {
        Some(current) => current.union(&next),
        None => next,
    }
}

fn same_rect(first: Rect, second: Rect) -> bool {
    const EPSILON: f32 = 0.5;
    (first.min_x - second.min_x).abs() <= EPSILON
        && (first.min_y - second.min_y).abs() <= EPSILON
        && (first.max_x - second.max_x).abs() <= EPSILON
        && (first.max_y - second.max_y).abs() <= EPSILON
}

fn select_debug_refresh_regions(candidates: Vec<DebugRefreshRegion>) -> Vec<DebugRefreshRegion> {
    let mut unique: Vec<DebugRefreshRegion> = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        if let Some(existing) = unique
            .iter_mut()
            .find(|existing| same_rect(existing.rect, candidate.rect))
        {
            existing.frames_remaining = existing.frames_remaining.max(candidate.frames_remaining);
            existing.updates = existing.updates.max(candidate.updates);
            if candidate.regular
                && (!existing.regular || candidate.frames_per_second > existing.frames_per_second)
            {
                existing.frames_per_second = candidate.frames_per_second;
                existing.regular = true;
            }
        } else {
            unique.push(candidate);
        }
    }

    unique
        .iter()
        .enumerate()
        .filter_map(|(candidate_index, candidate)| {
            let contained_by_active_parent =
                unique.iter().enumerate().any(|(parent_index, parent)| {
                    parent_index != candidate_index
                        && !same_rect(parent.rect, candidate.rect)
                        && parent.rect.contains(&candidate.rect)
                        && !debug_child_is_meaningfully_faster(parent, candidate)
                });
            (!contained_by_active_parent).then_some(*candidate)
        })
        .collect()
}

fn debug_child_is_meaningfully_faster(
    parent: &DebugRefreshRegion,
    child: &DebugRefreshRegion,
) -> bool {
    const FASTER_LOOP_RATIO: f32 = 1.25;
    parent.regular
        && child.regular
        && child.frames_per_second > parent.frames_per_second * FASTER_LOOP_RATIO
}

fn tile_flash_fade(frames_remaining: u32, duration_frames: u32) -> f32 {
    if duration_frames == 0 {
        return 0.0;
    }
    (frames_remaining as f32 / duration_frames as f32).clamp(0.0, 1.0)
}

fn faded_color(mut color: [f32; 4], fade: f32) -> [f32; 4] {
    color[3] *= fade;
    color
}

#[cfg(test)]
mod debug_refresh_region_tests {
    use super::*;

    fn region(origin: [f32; 2], size: [f32; 2], frames_per_second: f32) -> DebugRefreshRegion {
        DebugRefreshRegion {
            rect: Rect::from_origin_size(origin, size),
            frames_remaining: 3,
            updates: 12,
            frames_per_second,
            regular: frames_per_second > 0.0,
        }
    }

    #[test]
    fn nested_regions_collapse_to_the_outermost_unless_the_child_is_faster() {
        let selected = select_debug_refresh_regions(vec![
            region([0.0, 0.0], [200.0, 200.0], 30.0),
            region([20.0, 20.0], [80.0, 80.0], 60.0),
        ]);

        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn a_child_with_the_same_update_rate_does_not_add_a_recursive_box() {
        let selected = select_debug_refresh_regions(vec![
            region([0.0, 0.0], [200.0, 200.0], 60.0),
            region([20.0, 20.0], [80.0, 80.0], 60.0),
        ]);

        assert_eq!(selected.len(), 1);
        assert_eq!(
            selected[0].rect,
            Rect::from_origin_size([0.0, 0.0], [200.0, 200.0])
        );
    }

    #[test]
    fn an_unmeasured_parent_keeps_the_outermost_fallback() {
        let selected = select_debug_refresh_regions(vec![
            region([0.0, 0.0], [200.0, 200.0], 0.0),
            region([20.0, 20.0], [80.0, 80.0], 120.0),
        ]);

        assert_eq!(selected.len(), 1);
        assert_eq!(
            selected[0].rect,
            Rect::from_origin_size([0.0, 0.0], [200.0, 200.0])
        );
    }

    #[test]
    fn disjoint_regions_remain_independent() {
        let selected = select_debug_refresh_regions(vec![
            region([0.0, 0.0], [80.0, 80.0], 30.0),
            region([120.0, 0.0], [80.0, 80.0], 30.0),
        ]);

        assert_eq!(selected.len(), 2);
    }
}

fn screen_tile_rect(visit: &TiledVisit, tile: TileCoord) -> Rect {
    let content_rect = visit.grid.tile_bounds(tile);
    let translation = [
        visit.screen_viewport.min_x - visit.content_viewport.min_x,
        visit.screen_viewport.min_y - visit.content_viewport.min_y,
    ];
    Rect::from_origin_size(
        [
            content_rect.min_x + translation[0],
            content_rect.min_y + translation[1],
        ],
        [content_rect.width(), content_rect.height()],
    )
}

/// Phase 6's reference scene: one solid quad and one line of shaped text.
///
/// # Why this lives in the library rather than in a test
///
/// Two dev targets need the *same* scene: the example, which puts it on a
/// screen, and `tests/window_present.rs`, which reads it back off the swapchain
/// and compares it byte for byte. If each built its own tree, the thing checked
/// and the thing looked at would only be the same by inspection. This is the
/// pattern `wgpui-text`'s `test_fonts` and `wgpui-core`'s `test_support` already
/// set for exactly this reason: an integration test cannot see a `#[cfg(test)]`
/// module, so shared scaffolding is a public module in the library.
///
/// # It takes runs rather than text
///
/// Shaping is `wgpui-text`'s, and this crate does not depend on it (see this
/// module's doc). The caller shapes and rasterises; this holds the result.
pub struct ReferenceScene {
    /// The quad's fill, straight alpha.
    ///
    /// Pick components that are exact multiples of 1/255 if the pixels are
    /// going to be compared for equality: the target is `Rgba8Unorm` and an
    /// opaque quad's `rgb` reaches it unmodified, so `k / 255.0` reads back as
    /// exactly `k` while an arbitrary float lands on whichever side of a
    /// rounding boundary the hardware picks.
    pub fill: [f32; 4],
    /// The quad's size in pixels. Its position is layout's answer, not a
    /// constant — it is the first child of a column, so it lands at the origin,
    /// and the tests assert against what was *emitted* rather than against that
    /// assumption.
    pub fill_size: [f32; 2],
    /// The text, already shaped and already holding atlas tiles.
    ///
    /// Positions are relative to the text element's own bounds, which is what
    /// makes the line move when layout moves it.
    pub text: Vec<GlyphRun>,
    /// The height reserved for the text element.
    pub text_height: f32,
    /// Whether each element carries a [`ReferenceKey`] fingerprint.
    ///
    /// **This flag exists because Phase 6 found that it matters, and the
    /// finding was not predicted by any earlier phase.** `Description::new`
    /// attaches no fingerprint, and no fingerprint means R-N §2.3's permissive
    /// default: *assume changed, rebuild*. Reconciliation still reuses the
    /// instance and its layout node — §4.0's ambient guarantee holds — but
    /// `Emitter::emit` only takes its skip path for a node the plan marked
    /// fully reused, so an unfingerprinted element **re-emits every frame**,
    /// and `reconcile_records` pushes an update op per record without
    /// comparing the value it is replacing against the one already there.
    ///
    /// Every phase before this one drew one frame or a short fixed sequence,
    /// where that costs nothing and is invisible. A window redraws forever, and
    /// there it is the difference between a settled frame that uploads nothing
    /// and one that re-uploads the entire scene at display rate.
    ///
    /// Both arms are kept so the difference is measurable rather than asserted:
    /// `false` reproduces what a naive frontend gets today, `true` reproduces
    /// what a fingerprinted one gets. `LoopFrame::was_idle` is the observable.
    pub fingerprinted: bool,
}

/// The reference scene's fingerprint.
///
/// One type across all three elements, with `part` separating them, because
/// [`compare_by_equality`] reports `Invalidation::all` on a failed downcast —
/// which is the type-mismatch rule, and would fire spuriously if the root and
/// its children shared a position and differed only in key type.
#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceKey {
    /// Which element this fingerprints.
    pub part: u8,
    /// The quad's fill.
    pub fill: [f32; 4],
    /// The quad's size.
    pub fill_size: [f32; 2],
    /// The text element's height.
    pub text_height: f32,
    /// How many glyphs the line holds. A cheap stand-in for the line's
    /// content: this scene never changes its text, so anything finer would be
    /// checking a case the scene does not have.
    pub glyphs: usize,
}

impl ReconcileKey for ReferenceKey {
    fn compare(&self, previous: &dyn ReconcileKey) -> Invalidation {
        compare_by_equality(self, previous, Invalidation::all())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// The element type tag for the reference scene's root. Never instantiated —
/// `Description::new::<T>` only reads `T`'s `TypeId`, which is what an instance
/// has to match across frames to be reused.
pub struct ReferenceRoot;

/// The element type tag for the reference scene's quad.
pub struct ReferenceFill;

/// The element type tag for the reference scene's text.
pub struct ReferenceText;

impl ReferenceScene {
    /// This frame's description.
    ///
    /// Built fresh every frame on purpose: a `Description` is the cheap
    /// per-frame value §2 says it is, and reconciliation is ambient (§4.0), so
    /// a loop that rebuilds it is the *default* path rather than a slow one.
    /// Whether that default actually reuses everything is
    /// [`LoopFrame::was_idle`]'s question, and Phase 6's steady-state evidence
    /// is that it does.
    pub fn describe(&self) -> Description {
        let key = |part: u8| ReferenceKey {
            part,
            fill: self.fill,
            fill_size: self.fill_size,
            text_height: self.text_height,
            glyphs: self.text.iter().map(|run| run.glyphs.len()).sum(),
        };
        let fingerprint = |description: Description, part: u8| {
            if self.fingerprinted {
                description.diff_key(key(part))
            } else {
                description
            }
        };
        let fill = fingerprint(
            Description::new::<ReferenceFill>()
                .style(LayoutStyle {
                    size: LayoutSize {
                        width: Dimension::length(self.fill_size[0]),
                        height: Dimension::length(self.fill_size[1]),
                    },
                    ..LayoutStyle::default()
                })
                .emit(SolidFill { color: self.fill }),
            1,
        );
        let text = fingerprint(
            Description::new::<ReferenceText>()
                .style(LayoutStyle {
                    size: LayoutSize {
                        width: Dimension::percent(1.0),
                        height: Dimension::length(self.text_height),
                    },
                    ..LayoutStyle::default()
                })
                .emit(PlacedGlyphs {
                    runs: self.text.clone(),
                }),
            2,
        );
        fingerprint(
            Description::new::<ReferenceRoot>()
                .style(LayoutStyle {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Column,
                    size: LayoutSize {
                        width: Dimension::percent(1.0),
                        height: Dimension::percent(1.0),
                    },
                    ..LayoutStyle::default()
                })
                .child(fill)
                .child(text),
            0,
        )
    }
}

/// An emitter that paints one solid quad filling its element's bounds.
///
/// Phase 6's scene has to be checkable by exact comparison rather than by
/// eyeballing, which means the primitive it draws has to be one whose expected
/// pixels are computable from the description alone. A quad that fills its
/// laid-out rectangle is that: layout says where the rectangle is, and every
/// pixel inside it must be exactly `color` written back to `Rgba8Unorm`.
///
/// Lives here rather than in the example because the swapchain test drives the
/// identical scene — an on-screen claim is only worth something if the thing
/// checked on screen is the same thing checked offscreen.
pub struct SolidFill {
    /// Straight-alpha RGBA the quad is painted with.
    pub color: [f32; 4],
}

impl Emit for SolidFill {
    fn emit(&self, context: &EmitContext, emission: &mut Emission) {
        emission.quad(Quad {
            origin: [context.bounds.x, context.bounds.y],
            size: [context.bounds.width, context.bounds.height],
            background: self.color,
            border_color: [0.0, 0.0, 0.0, 0.0],
            corner_radii: [0.0; 4],
            border_widths: [0.0; 4],
            material: Material::Solid,
        });
    }
}

/// An emitter that places already-shaped runs relative to its element's bounds.
///
/// The offset is what makes the text layout-driven rather than screen-absolute:
/// the runs are shaped once, at an origin of the caller's choosing, and this
/// translates them by wherever layout put the element that owns them. Move the
/// element and the line moves with it, without reshaping.
///
/// The positions are *not* floored here. `wgpui_text::patch::glyph_runs` keeps
/// the fractional pen position while the atlas already carries the fraction as
/// one of four sub-pixel variants, so a glyph at a fractional position blits up
/// to a texel off — Phase 5.6 disclosed this and named `wgpui-text` as where the
/// flooring belongs. Rounding here instead would put the fix in the wrong crate
/// and hide the open item. A caller that needs texel-exact output rounds its
/// runs before handing them over, exactly as Phase 5.6's gate does.
pub struct PlacedGlyphs {
    /// The runs, positioned relative to the element's own bounds.
    pub runs: Vec<GlyphRun>,
}

impl Emit for PlacedGlyphs {
    fn emit(&self, context: &EmitContext, emission: &mut Emission) {
        for run in &self.runs {
            let mut placed = run.clone();
            for glyph in &mut placed.glyphs {
                glyph.position[0] += context.bounds.x;
                glyph.position[1] += context.bounds.y;
            }
            emission.glyph_run(placed);
        }
    }
}
