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
//! # What this is not
//!
//! It does not shape text. `wgpui-text` is a dev-dependency of this crate on
//! purpose (§3.3, and `Cargo.toml`'s note on it): the production edge runs one
//! way, and `AtlasTileSource` takes its rasteriser as a closure precisely so
//! the render crate never names the text crate. A caller shapes and rasterises,
//! and hands this loop `GlyphRun`s through an ordinary [`Emit`]. Phase 6's
//! example does exactly that.

use wgpui_core::boundary::compositor::CompositeEntry;
use wgpui_core::geometry::Rect;
use wgpui_core::invalidation::axes::Invalidation;
use wgpui_core::invalidation::request::FrameSignals;
use wgpui_core::patch::PatchError;
use wgpui_core::patch::apply::apply;
use wgpui_core::patch::emit::{Emission, Emit, EmitContext, EmitError, Emitter, FrameEmission};
use wgpui_core::patch::primitive::{Glyph, GlyphRun, Quad};
use wgpui_core::reconcile::description::Description;
use wgpui_core::reconcile::diff_key::{ReconcileKey, compare_by_equality};
use wgpui_core::reconcile::plan::FrameStats;
use wgpui_core::reconcile::reconciler::{ReconcileError, Reconciler};
use wgpui_core::scene::Scene;
use wgpui_core::scene::layer::LayerId;
use wgpui_layout::taffy_tree::{
    Dimension, Display, FlexDirection, LayoutSize, LayoutStyle, LayoutTree, definite,
};

use crate::render::atlas_upload::AtlasTextures;
use crate::render::draw::DrawMode;
use crate::render::frame::{
    Dirty, FrameError, FrameInput, FrameOutput, FrameRenderer, RenderTarget,
};

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
#[derive(Clone, Debug)]
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
    /// The GPU half.
    pub frame: FrameOutput,
}

impl LoopFrame {
    /// Whether this frame changed nothing at all — no records touched, no
    /// bytes uploaded, no layer dirty.
    ///
    /// The shape a window sitting still has, and the one `frame.rs`'s
    /// clean-frame path is written for.
    pub fn was_idle(&self) -> bool {
        self.emission.patch.is_empty() && self.dirty_layers.is_empty() && self.uploaded_bytes == 0
    }
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
    frames: u64,
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
            frames: 0,
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
    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        description: Description,
        atlas: Option<&AtlasTextures>,
        target: &RenderTarget<'_>,
        mode: DrawMode,
        signals: &FrameSignals,
        composites: &[CompositeEntry],
    ) -> Result<LoopFrame, LoopError> {
        let plan = self.reconciler.reconcile(description, &mut self.layout)?;
        let root = plan.root().map(|node| node.layout_node).ok_or(LoopError::NoRoot)?;
        let width = target.width.max(1) as f32;
        let height = target.height.max(1) as f32;
        self.layout
            .compute_layout(root, definite(width, height))
            .map_err(EmitError::from)?;
        let emission = self
            .emitter
            .emit(&plan, &self.layout, signals, &mut self.scene)?;
        let uploads = apply(&mut self.scene, &emission.patch)?;
        let dirty_layers = emission.patch.layers();

        let input = FrameInput {
            scene: &self.scene,
            clip: Rect::from_origin_size([0.0, 0.0], [width, height]),
            poison: &[],
            dirty: Dirty::Some(&dirty_layers),
            uploads: uploads.entries(),
            composites,
            registry: None,
            atlas,
            viewport: [width, height],
            mode,
        };
        let frame = self.renderer.render_to(device, queue, &input, target)?;
        self.frames += 1;

        Ok(LoopFrame {
            reconciled: plan.stats(),
            emission,
            dirty_layers,
            uploaded_bytes: uploads.byte_count(),
            frame,
        })
    }
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
            corner_radius: 0.0,
            border_width: 0.0,
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
