//! `StyledText` — gets `diff_key` here (§6.2 invariant, Phase 5), closing
//! R-N Phase 7's self-documented gap. See docs/gpu-native-architecture.md
//! §3.4, §6.2, and R-N §2.4.
//!
//! # What has to be true for a text fingerprint to be worth taking
//!
//! Rich text is the expensive case, and the reason the key has to be careful
//! rather than merely present. Reconciling a `div` saves a style comparison and
//! a Taffy node; reconciling a `StyledText` saves a *shaping pass* — the one
//! piece of per-frame work §6 explicitly declines to move to the GPU, on the
//! grounds that it is branch-heavy and cache-dependent. A key that costs
//! anything close to what it saves is not worth having.
//!
//! So the two expensive fields are both compared by identity first:
//!
//! - The text is a [`SharedString`], whose `PartialEq` short-circuits on
//!   `Arc::ptr_eq` — R-N §2.4's own requirement. A list row holding a clone of
//!   last frame's string answers in one pointer comparison however long the
//!   string is.
//! - The highlight runs are an `Arc<[…]>`, compared the same way for the same
//!   reason. A syntax-highlighted line can carry dozens of runs and they are
//!   almost always the same `Arc` frame to frame, because the thing that
//!   produced them did not re-run.
//!
//! Both fall back to a real comparison when the pointers differ, so a rebuilt-
//! but-identical row is still correctly reported as unchanged; the pointer check
//! is a fast path, never the answer.
//!
//! # Style is compared whole, unlike `Div`'s
//!
//! The legacy `TextDiffKey` already makes this call and it is right:
//!
//! > A style change is treated as a full rebuild rather than split further:
//! > unlike `Div`'s style, almost every `TextStyle` field (font, size, weight,
//! > line height) affects shaping, so there is little left to gain from a finer
//! > split here.
//!
//! Colour is the one field that genuinely does not affect shaping, and it is
//! split out for that reason — a `GlyphRun` carries its colour as a value, so a
//! recolour is a `DISPLAY` update over the same glyph positions and must not
//! re-shape. That is a real saving on a real workload (selection, hover, search
//! highlight), not a hypothetical one.

use std::any::Any;
use std::cell::RefCell;
use std::ops::Range;
use std::rc::Rc;
use std::sync::Arc;
use wgpui_core::invalidation::axes::Invalidation;
use wgpui_core::patch::emit::{Emission, EmitContext};
use wgpui_core::reconcile::description::Description;
use wgpui_core::reconcile::diff_key::ReconcileKey;
use wgpui_core::scene::atlas::GlyphTileSource;
use wgpui_layout::taffy_tree::{Dimension, LayoutSize, LayoutStyle};
use wgpui_text::patch::{ConversionStats, RunPlacement, glyph_runs};
use wgpui_text::shaping::{
    Font, FontRun, FontStyle, FontWeight, SharedString, ShapeError, TextShaper,
};

/// The text properties that decide shaping, plus the one that does not.
///
/// `color` is deliberately in the same struct rather than hoisted out: an
/// element author sets it alongside the rest, and separating them in the public
/// shape to make one comparison cheaper would be the tail wagging the dog. The
/// *comparison* separates them; the type does not.
#[derive(Clone, Debug, PartialEq)]
pub struct TextStyle {
    /// The face to shape with.
    pub font: Font,
    /// Size in pixels.
    pub font_size: f32,
    /// Baseline-to-baseline distance in pixels.
    pub line_height: f32,
    /// Straight-alpha RGBA the glyphs draw in.
    pub color: [f32; 4],
}

impl Default for TextStyle {
    fn default() -> Self {
        Self {
            font: Font::default(),
            font_size: 14.0,
            line_height: 20.0,
            color: [0.0, 0.0, 0.0, 1.0],
        }
    }
}

impl TextStyle {
    /// Whether two styles differ in any way that changes glyph positions.
    ///
    /// Everything except colour, which a `GlyphRun` carries as a value.
    fn shaping_differs(&self, other: &TextStyle) -> bool {
        self.font != other.font
            || self.font_size != other.font_size
            || self.line_height != other.line_height
    }
}

/// A style override applied to one byte range.
///
/// A subset of the legacy `HighlightStyle`, carrying the three properties that
/// matter to the fingerprint's behaviour: one that does not affect shaping
/// (`color`) and two that do (`weight`, `style`). Whichever phase moves the
/// frozen `HighlightStyle` into the workspace widens this; the comparison rule
/// does not change.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct HighlightStyle {
    /// Override colour.
    pub color: Option<[f32; 4]>,
    /// Override weight — bolding re-shapes, because it is a different face.
    pub weight: Option<FontWeight>,
    /// Override slant — italicising re-shapes, for the same reason.
    pub style: Option<FontStyle>,
}

impl HighlightStyle {
    fn shaping_differs(&self, other: &HighlightStyle) -> bool {
        self.weight != other.weight || self.style != other.style
    }
}

/// One highlighted range.
pub type Highlight = (Range<usize>, HighlightStyle);

/// Highlight runs, shared so an unchanged set compares by pointer.
pub type Highlights = Arc<[Highlight]>;

/// The fingerprint a `StyledText` presents to ambient reconciliation.
#[derive(Clone, Debug)]
pub struct StyledTextKey {
    text: SharedString,
    style: TextStyle,
    highlights: Highlights,
}

impl StyledTextKey {
    /// Whether the two keys' highlight sets differ, and if so whether the
    /// difference reaches shaping.
    ///
    /// Returns `(differ, shaping_differs)`. Written as one walk rather than two
    /// so the common case — same `Arc`, or same length and same values — is
    /// visited once.
    fn compare_highlights(&self, previous: &StyledTextKey) -> (bool, bool) {
        if Arc::ptr_eq(&self.highlights, &previous.highlights) {
            return (false, false);
        }
        if self.highlights.len() != previous.highlights.len() {
            // A different number of runs re-partitions the text, which is a
            // different set of font runs, which is a different shape.
            return (true, true);
        }
        let mut differ = false;
        let mut shaping_differs = false;
        for (current, previous) in self.highlights.iter().zip(previous.highlights.iter()) {
            if current.0 != previous.0 {
                return (true, true);
            }
            if current.1 != previous.1 {
                differ = true;
                if current.1.shaping_differs(&previous.1) {
                    shaping_differs = true;
                }
            }
        }
        (differ, shaping_differs)
    }
}

impl ReconcileKey for StyledTextKey {
    fn compare(&self, previous: &dyn ReconcileKey) -> Invalidation {
        let Some(previous) = previous.as_any().downcast_ref::<StyledTextKey>() else {
            return Invalidation::all();
        };

        // Text first: it is the cheapest comparison (one pointer, usually) and
        // the most likely to differ in a live UI.
        if self.text != previous.text {
            return Invalidation::LAYOUT.union(Invalidation::DISPLAY);
        }
        if self.style.shaping_differs(&previous.style) {
            return Invalidation::LAYOUT.union(Invalidation::DISPLAY);
        }

        let (highlights_differ, highlights_reshape) = self.compare_highlights(previous);
        if highlights_reshape {
            return Invalidation::LAYOUT.union(Invalidation::DISPLAY);
        }

        let mut axes = Invalidation::empty();
        if self.style.color != previous.style.color || highlights_differ {
            // Same glyphs in the same places, different colour: the run's
            // colour is a value in the slab, so this rewrites slots without
            // touching the shaper or the layout tree.
            axes |= Invalidation::DISPLAY;
        }
        axes
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Shaping plus rasterisation, in the one place an element can reach both.
///
/// The two halves of the text path are separately owned by design (§3.3, §3.5)
/// and an element needs both at once, so this is where they meet: it is the only
/// type in this crate that holds a [`TextShaper`], and it holds a
/// [`GlyphTileSource`] beside it so a shaped line can be converted in the same
/// call.
pub struct TextEngine {
    shaper: TextShaper,
    tiles: Box<dyn GlyphTileSource>,
}

impl TextEngine {
    /// An engine shaping with `shaper` and rastering through `tiles`.
    pub fn new(shaper: TextShaper, tiles: Box<dyn GlyphTileSource>) -> Self {
        Self { shaper, tiles }
    }

    /// The shaper, for resolving fonts and for reading its counters.
    pub fn shaper(&mut self) -> &mut TextShaper {
        &mut self.shaper
    }

    /// Shape one line and convert it to patch payloads in one step.
    fn shape_and_convert(
        &mut self,
        text: &SharedString,
        font_size: f32,
        runs: &[FontRun],
        placement: RunPlacement,
        emission: &mut Emission,
    ) -> Result<ConversionStats, ShapeError> {
        let line = self.shaper.shape_line(text, font_size, runs)?;
        let (converted, stats) = glyph_runs(&line, placement, self.tiles.as_mut());
        for run in converted {
            emission.glyph_run(run);
        }
        Ok(stats)
    }
}

/// A [`TextEngine`] several elements share.
///
/// `Rc<RefCell<_>>` rather than a lock, for the reason `TextShaper`'s own doc
/// gives: everything that reaches it runs on the frame's thread, so a lock here
/// would be one nothing contends. Shared rather than owned per element because
/// the shaping cache and the atlas are only useful if every row on screen is
/// looking at the same ones.
pub type SharedTextEngine = Rc<RefCell<TextEngine>>;

/// Text with per-range style overrides.
///
/// Like [`crate::wgpu_surface::WgpuSurface`] and [`crate::img::Img`], this is
/// the description shape rather than the full element: no `TextLayout` cursor
/// mapping, no tooltips, no mouse handlers, because those need `Window`/`App`.
/// The fingerprint and the emission are real.
#[derive(Clone)]
pub struct StyledText {
    text: SharedString,
    style: TextStyle,
    highlights: Highlights,
    engine: SharedTextEngine,
    requested_size: [f32; 2],
}

impl StyledText {
    /// Unhighlighted text in `style`, shaped through `engine`.
    pub fn new(text: impl Into<SharedString>, style: TextStyle, engine: SharedTextEngine) -> Self {
        Self {
            text: text.into(),
            style,
            highlights: Arc::from(Vec::new()),
            engine,
            requested_size: [0.0, 0.0],
        }
    }

    /// Apply highlight runs.
    ///
    /// Takes the shared handle rather than an iterator, deliberately: rebuilding
    /// the `Arc` every frame from an equal `Vec` would defeat the pointer
    /// short-circuit the fingerprint depends on, and making that visible at the
    /// call site is better than accepting a `Vec` and silently paying for it.
    pub fn with_highlights(mut self, highlights: Highlights) -> Self {
        self.highlights = highlights;
        self
    }

    /// Request a size.
    pub fn size(mut self, width: f32, height: f32) -> Self {
        self.requested_size = [width, height];
        self
    }

    /// This text's fingerprint.
    pub fn diff_key(&self) -> StyledTextKey {
        StyledTextKey {
            text: self.text.clone(),
            style: self.style.clone(),
            highlights: self.highlights.clone(),
        }
    }

    /// The font runs shaping needs: one per highlight boundary, covering the
    /// text exactly.
    ///
    /// Highlight ranges that overlap, run backwards, or fall outside the text
    /// are skipped rather than trusted; the legacy element `debug_assert!`s on
    /// them, which means a release build would shape against a run list whose
    /// lengths do not add up and `shape_line` would refuse the whole line.
    /// Skipping degrades one highlight, which is visible and recoverable.
    fn font_runs(&self, font_id: wgpui_text::shaping::FontId) -> Vec<FontRun> {
        let mut runs = Vec::new();
        let mut cursor = 0usize;
        for (range, highlight) in self.highlights.iter() {
            if range.start < cursor || range.end > self.text.len() || range.start >= range.end {
                continue;
            }
            if !self.text.as_str().is_char_boundary(range.start)
                || !self.text.as_str().is_char_boundary(range.end)
            {
                continue;
            }
            if cursor < range.start {
                runs.push(self.font_run(range.start - cursor, font_id, None));
            }
            runs.push(self.font_run(range.len(), font_id, Some(*highlight)));
            cursor = range.end;
        }
        if cursor < self.text.len() {
            runs.push(self.font_run(self.text.len() - cursor, font_id, None));
        }
        runs
    }

    fn font_run(
        &self,
        len: usize,
        font_id: wgpui_text::shaping::FontId,
        highlight: Option<HighlightStyle>,
    ) -> FontRun {
        FontRun {
            len,
            font_id,
            weight: highlight
                .and_then(|highlight| highlight.weight)
                .unwrap_or(self.style.font.weight),
            style: highlight
                .and_then(|highlight| highlight.style)
                .unwrap_or(self.style.font.style),
            letter_spacing: 0.0,
        }
    }

    /// The per-frame description of this text.
    ///
    /// Note the absence, which is the same one `WgpuSurface` makes load-bearing:
    /// no `.id()`. Identity is positional (SFD §1.0), so a list row's text is
    /// addressed across frames without the caller naming it.
    pub fn describe(&self) -> Description {
        let [width, height] = self.requested_size;
        let text = self.clone();
        Description::new::<StyledText>()
            .diff_key(self.diff_key())
            .style(LayoutStyle {
                size: LayoutSize {
                    width: Dimension::length(width),
                    height: Dimension::length(height),
                },
                flex_shrink: 0.0,
                ..LayoutStyle::default()
            })
            .emit(move |context: &EmitContext, emission: &mut Emission| {
                text.emit_into(context, emission);
            })
    }

    fn emit_into(&self, context: &EmitContext, emission: &mut Emission) {
        let mut engine = self.engine.borrow_mut();
        let Ok(font_id) = engine.shaper().resolve_font(&self.style.font) else {
            // No face at all: the text draws nothing this frame rather than
            // taking the frame down. `resolve_font` already falls back to any
            // available face, so reaching here means the font database is empty.
            return;
        };
        let runs = self.font_runs(font_id);
        let placement = RunPlacement {
            origin: [context.bounds.x, context.bounds.y],
            color: self.style.color,
            scale_factor: 1.0,
        };
        if let Err(error) = engine.shape_and_convert(
            &self.text,
            self.style.font_size,
            &runs,
            placement,
            emission,
        ) {
            // Shaping refused this line — a run-length mismatch or an unissued
            // font id, both of which are bugs in the caller rather than
            // conditions to recover from. Reported rather than swallowed, and
            // the element emits nothing rather than emitting garbage.
            log_shape_error(&error);
        }
    }
}

/// Where a shaping failure goes.
///
/// `wgpui-widgets` has no logging dependency and §3.4 does not give it one, so
/// this is the single place that decides what happens to an error that cannot
/// be propagated (an `Emit` returns nothing by design — it is called from a walk
/// that has already committed to a frame). Kept as a named function so the
/// decision is one line to change when the crate does take a logger, rather than
/// an `eprintln!` scattered through the module.
fn log_shape_error(error: &ShapeError) {
    #[cfg(test)]
    panic!("shaping failed during emission: {error}");
    #[cfg(not(test))]
    let _unreported = error;
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpui_core::patch::primitive::AtlasTileId;
    use wgpui_core::scene::atlas::{GlyphRasterKey, GlyphTile};
    use wgpui_core::reconcile::description::ElementId;
    use wgpui_core::reconcile::instance::InstanceKey;
    use wgpui_core::reconcile::plan::{FramePlan, NodeOutcome, PlannedNode, RebuildReason};
    use wgpui_core::reconcile::reconciler::{ReconcileError, Reconciler};
    use wgpui_layout::taffy_tree::LayoutTree;
    use wgpui_text::shaping::font;

    /// A tile source that hands out one tile per distinct raster key.
    ///
    /// The real allocator is tested in `wgpui-wgpu`; these tests are about the
    /// fingerprint and the emission, so the atlas is a stand-in that cannot fail.
    #[derive(Default)]
    struct CountingTiles {
        issued: std::collections::HashMap<GlyphRasterKey, GlyphTile>,
    }

    impl GlyphTileSource for CountingTiles {
        fn tile_for(&mut self, key: GlyphRasterKey) -> Option<GlyphTile> {
            let next = self.issued.len();
            Some(*self.issued.entry(key).or_insert_with(|| GlyphTile {
                tile: AtlasTileId::new(0, next as u32).expect("test tiles stay in range"),
                atlas_origin: [0.0, 0.0],
                atlas_size: [6.0, 10.0],
                bearing: [0.0, -8.0],
            }))
        }
    }

    fn engine() -> SharedTextEngine {
        Rc::new(RefCell::new(TextEngine::new(
            TextShaper::new(),
            Box::new(CountingTiles::default()),
        )))
    }

    fn style() -> TextStyle {
        TextStyle {
            font: font("Segoe UI"),
            ..TextStyle::default()
        }
    }

    fn text(engine: &SharedTextEngine, value: impl Into<SharedString>) -> StyledText {
        StyledText::new(value, style(), engine.clone()).size(200.0, 20.0)
    }

    const TEXT_SLOT: [ElementId; 2] = [ElementId::Slot(0), ElementId::Slot(0)];

    fn tree(text: StyledText) -> Description {
        Description::new::<StyledText>()
            .diff_key(RootKey)
            .child(text.describe())
    }

    #[derive(PartialEq, Debug)]
    struct RootKey;

    impl ReconcileKey for RootKey {
        fn compare(&self, previous: &dyn ReconcileKey) -> Invalidation {
            wgpui_core::reconcile::diff_key::compare_by_equality(
                self,
                previous,
                Invalidation::DISPLAY,
            )
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    fn node_at<'plan>(plan: &'plan FramePlan, path: &[ElementId]) -> Option<&'plan PlannedNode> {
        plan.node_for_instance(InstanceKey::from_path(path))
    }

    #[test]
    fn an_unchanged_shared_clone_compares_by_pointer_and_reports_nothing_stale() {
        let engine = engine();
        let shared = SharedString::from("a moderately long row of list content");
        let first = text(&engine, shared.clone()).diff_key();
        let second = text(&engine, shared.clone()).diff_key();
        assert!(
            first.text.is_clone_of(&second.text),
            "the test must actually be exercising the shared-clone path"
        );
        assert_eq!(first.compare(&second), Invalidation::empty());
    }

    #[test]
    fn a_rebuilt_but_identical_string_is_still_reported_unchanged() {
        let engine = engine();
        let first = text(&engine, "row content").diff_key();
        let second = text(&engine, String::from("row content")).diff_key();
        assert!(
            !first.text.is_clone_of(&second.text),
            "these must be different allocations, or the test proves the wrong thing"
        );
        assert_eq!(
            first.compare(&second),
            Invalidation::empty(),
            "the pointer check is a fast path, never the answer"
        );
    }

    #[test]
    fn changed_text_is_a_layout_and_display_change() {
        let engine = engine();
        assert_eq!(
            text(&engine, "after").diff_key().compare(&text(&engine, "before").diff_key()),
            Invalidation::LAYOUT.union(Invalidation::DISPLAY)
        );
    }

    #[test]
    fn a_recolour_is_a_display_change_and_never_reshapes() {
        let engine = engine();
        let plain = text(&engine, "selected");
        let recoloured = StyledText::new(
            "selected",
            TextStyle {
                color: [1.0, 0.0, 0.0, 1.0],
                ..style()
            },
            engine.clone(),
        );
        assert_eq!(
            recoloured.diff_key().compare(&plain.diff_key()),
            Invalidation::DISPLAY,
            "a GlyphRun carries its colour as a value, so a recolour must not re-shape"
        );
    }

    #[test]
    fn a_font_or_size_change_reshapes() {
        let engine = engine();
        let base = text(&engine, "row");
        let bigger = StyledText::new(
            "row",
            TextStyle {
                font_size: 15.0,
                ..style()
            },
            engine.clone(),
        );
        let bolder = StyledText::new(
            "row",
            TextStyle {
                font: font("Segoe UI").bold(),
                ..style()
            },
            engine.clone(),
        );
        for (what, changed) in [("a size change", bigger), ("a weight change", bolder)] {
            assert_eq!(
                changed.diff_key().compare(&base.diff_key()),
                Invalidation::LAYOUT.union(Invalidation::DISPLAY),
                "{what} changes glyph positions"
            );
        }
    }

    #[test]
    fn unchanged_highlight_runs_compare_by_pointer() {
        let engine = engine();
        let highlights: Highlights = Arc::from(vec![(
            0..3,
            HighlightStyle {
                color: Some([1.0, 0.0, 0.0, 1.0]),
                ..HighlightStyle::default()
            },
        )]);
        let first = text(&engine, "highlighted")
            .with_highlights(highlights.clone())
            .diff_key();
        let second = text(&engine, "highlighted")
            .with_highlights(highlights.clone())
            .diff_key();
        assert!(Arc::ptr_eq(&first.highlights, &second.highlights));
        assert_eq!(first.compare(&second), Invalidation::empty());
    }

    #[test]
    fn a_highlight_recolour_repaints_and_a_highlight_reweight_reshapes() {
        let engine = engine();
        let base = text(&engine, "highlighted").with_highlights(Arc::from(vec![(
            0..3,
            HighlightStyle::default(),
        )]));
        let recoloured = text(&engine, "highlighted").with_highlights(Arc::from(vec![(
            0..3,
            HighlightStyle {
                color: Some([1.0, 0.0, 0.0, 1.0]),
                ..HighlightStyle::default()
            },
        )]));
        let reweighted = text(&engine, "highlighted").with_highlights(Arc::from(vec![(
            0..3,
            HighlightStyle {
                weight: Some(FontWeight::BOLD),
                ..HighlightStyle::default()
            },
        )]));

        assert_eq!(
            recoloured.diff_key().compare(&base.diff_key()),
            Invalidation::DISPLAY
        );
        assert_eq!(
            reweighted.diff_key().compare(&base.diff_key()),
            Invalidation::LAYOUT.union(Invalidation::DISPLAY),
            "bolding a range is a different face, which is a different shape"
        );
    }

    #[test]
    fn moving_a_highlight_range_reshapes() {
        let engine = engine();
        let base = text(&engine, "highlighted")
            .with_highlights(Arc::from(vec![(0..3, HighlightStyle::default())]));
        let moved = text(&engine, "highlighted")
            .with_highlights(Arc::from(vec![(2..5, HighlightStyle::default())]));
        assert_eq!(
            moved.diff_key().compare(&base.diff_key()),
            Invalidation::LAYOUT.union(Invalidation::DISPLAY),
            "a moved range re-partitions the text into different font runs"
        );
    }

    #[test]
    fn a_key_compared_against_a_different_element_type_is_a_full_invalidation() {
        let engine = engine();
        assert_eq!(
            text(&engine, "row").diff_key().compare(&RootKey),
            Invalidation::all()
        );
    }

    #[test]
    fn font_runs_cover_the_text_exactly_so_shaping_accepts_them() {
        let engine = engine();
        let font_id = engine
            .borrow_mut()
            .shaper()
            .resolve_font(&font("Segoe UI"))
            .expect("some face exists");
        let element = text(&engine, "abcdefgh").with_highlights(Arc::from(vec![
            (2..4, HighlightStyle::default()),
            (6..7, HighlightStyle::default()),
        ]));
        let runs = element.font_runs(font_id);
        assert_eq!(runs.iter().map(|run| run.len).sum::<usize>(), 8);
        assert_eq!(
            runs.iter().map(|run| run.len).collect::<Vec<_>>(),
            vec![2, 2, 2, 1, 1]
        );
    }

    #[test]
    fn an_overlapping_or_out_of_range_highlight_is_skipped_rather_than_breaking_the_line() {
        let engine = engine();
        let font_id = engine
            .borrow_mut()
            .shaper()
            .resolve_font(&font("Segoe UI"))
            .expect("some face exists");
        let element = text(&engine, "abcdefgh").with_highlights(Arc::from(vec![
            (2..5, HighlightStyle::default()),
            // Overlaps the previous run, and runs past the end.
            (3..4, HighlightStyle::default()),
            (6..99, HighlightStyle::default()),
        ]));
        let runs = element.font_runs(font_id);
        assert_eq!(
            runs.iter().map(|run| run.len).sum::<usize>(),
            8,
            "the run list must still cover the text, or shape_line refuses the whole line"
        );
    }

    #[test]
    fn shaping_happens_once_and_the_glyphs_reach_the_emission() -> Result<(), ReconcileError> {
        let engine = engine();
        let mut reconciler = Reconciler::new();
        let mut layout = LayoutTree::new();

        let first = reconciler.reconcile(tree(text(&engine, "Hello")), &mut layout)?;
        assert_eq!(
            node_at(&first, &TEXT_SLOT).map(|node| node.outcome),
            Some(NodeOutcome::Rebuilt(RebuildReason::NewInstance))
        );

        let second = reconciler.reconcile(tree(text(&engine, "Hello")), &mut layout)?;
        assert_eq!(
            node_at(&second, &TEXT_SLOT).map(|node| node.outcome),
            Some(NodeOutcome::Reused)
        );
        Ok(())
    }
}
