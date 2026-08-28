//! **Phase 5's gate**, measured rather than asserted.
//!
//! > Scroll-content-heavy scenes (avatars, multi-run text — SFD §3's stated
//! > motivation) hit the fast path with no per-refill shaping cost for unchanged
//! > rows, under ambient reconciliation (Phase 1), not because they're inside a
//! > `.boundary()`.
//!
//! # What is actually built here
//!
//! Forty list rows, each holding exactly what SFD §3 names as dominating real
//! list rows: an avatar ([`Img`]) and rich text ([`StyledText`] with highlight
//! runs, so it shapes as several font runs rather than one). The tree is driven
//! through the real [`Reconciler`], the real [`Emitter`], and a real [`Scene`] —
//! the same three types every prior phase's gates ran against — with a real
//! `cosmic-text` shaper behind the text.
//!
//! Nothing here is a scroll harness. §8's Phase 5 row is a claim about
//! *reconciliation*, and the scroll framing is about which workload makes the
//! claim matter, not about which mechanism proves it. This is the same shape
//! Phase 1 and Phase 2 used: build the frame twice, measure the second.
//!
//! # The clause that does most of the work
//!
//! "not because they're inside a `.boundary()`." That is a claim about what is
//! *absent*, so [`no_boundary_anywhere`] walks the description tree and asserts
//! it mechanically rather than trusting that nobody typed `.boundary()`. Phase 2
//! learned this the hard way with a test that silently no-op'd itself.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;
use wgpui_core::invalidation::axes::Invalidation;
use wgpui_core::invalidation::request::FrameSignals;
use wgpui_core::patch::apply::apply;
use wgpui_core::patch::emit::Emitter;
use wgpui_core::patch::primitive::AtlasTileId;
use wgpui_core::reconcile::description::{Description, ElementId};
use wgpui_core::reconcile::diff_key::{ReconcileKey, compare_by_equality};
use wgpui_core::reconcile::instance::InstanceKey;
use wgpui_core::reconcile::plan::NodeOutcome;
use wgpui_core::reconcile::reconciler::Reconciler;
use wgpui_core::scene::Scene;
use wgpui_core::scene::atlas::{
    GlyphRasterKey, GlyphTile, GlyphTileSource, ImageRasterKey, ImageTile, ImageTileSource,
    RasterizedImage,
};
use wgpui_layout::taffy_tree::{
    Dimension, FlexDirection, LayoutSize, LayoutStyle, LayoutTree, definite,
};
use wgpui_widgets::image_cache::{DecodedFrame, DecodedImage, ImageCache};
use wgpui_widgets::img::{ImageEngine, ImageSourceId, Img, SharedImageEngine};
use wgpui_widgets::styled_text::{
    HighlightStyle, Highlights, SharedTextEngine, StyledText, TextEngine, TextStyle,
};
use wgpui_text::shaping::{FontWeight, SharedString, TextShaper, font};

/// Rows in the scene. Enough that a per-row cost is visible against measurement
/// noise, and in the range a real list actually keeps resident.
const ROWS: usize = 40;

/// A tile source standing in for the real atlas.
///
/// `wgpui-wgpu`'s `render/atlas.rs` tests the allocator; this gate is about what
/// the shaper is and is not asked to do, so the atlas is a stand-in that always
/// succeeds and never evicts. Substituting it here means a gate failure can only
/// mean the thing the gate is about.
#[derive(Default)]
struct GateTiles {
    issued: std::collections::HashMap<GlyphRasterKey, GlyphTile>,
}

impl GlyphTileSource for GateTiles {
    fn tile_for(&mut self, key: GlyphRasterKey) -> Option<GlyphTile> {
        let next = self.issued.len();
        Some(*self.issued.entry(key).or_insert_with(|| GlyphTile {
            tile: AtlasTileId::new(0, next as u32).expect("the gate stays well inside 24 bits"),
            atlas_origin: [0.0, 0.0],
            atlas_size: [6.0, 10.0],
            bearing: [0.0, -8.0],
        }))
    }
}

struct Row;
struct ListRoot;

#[derive(PartialEq, Debug)]
struct RowKey(usize);

impl ReconcileKey for RowKey {
    fn compare(&self, previous: &dyn ReconcileKey) -> Invalidation {
        compare_by_equality(self, previous, Invalidation::DISPLAY)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// The content of one row, held across frames the way a real list holds it.
///
/// The `SharedString`s live here rather than being rebuilt per frame, which is
/// what makes the pointer short-circuit in `StyledTextKey` the realistic case
/// rather than a contrived one: a list's row data is model state, and a
/// re-render clones handles onto it.
struct RowContent {
    title: SharedString,
    subtitle: SharedString,
    highlights: Highlights,
    avatar: ImageSourceId,
}

fn content() -> Vec<RowContent> {
    (0..ROWS)
        .map(|index| RowContent {
            title: SharedString::from(format!("Row {index} — a reasonably long title line")),
            subtitle: SharedString::from(format!(
                "secondary detail for row {index}, also not short"
            )),
            // Three runs: an unstyled head, a bolded stretch, a coloured
            // stretch. Multi-run is the point — SFD §3 names rich text, not
            // plain text, as what dominates real rows.
            highlights: Arc::from(vec![
                (
                    4..7,
                    HighlightStyle {
                        weight: Some(FontWeight::BOLD),
                        ..HighlightStyle::default()
                    },
                ),
                (
                    12..20,
                    HighlightStyle {
                        color: Some([0.2, 0.4, 0.9, 1.0]),
                        ..HighlightStyle::default()
                    },
                ),
            ]),
            avatar: ImageSourceId::from_raw(index as u64 + 1),
        })
        .collect()
}

fn style() -> TextStyle {
    TextStyle {
        font: font("Segoe UI"),
        ..TextStyle::default()
    }
}

fn row(
    content: &RowContent,
    engine: &SharedTextEngine,
    images: &SharedImageEngine,
) -> Description {
    Description::new::<Row>()
        .diff_key(RowKey(0))
        .style(LayoutStyle {
            flex_direction: FlexDirection::Row,
            size: LayoutSize {
                width: Dimension::length(600.0),
                height: Dimension::length(48.0),
            },
            flex_shrink: 0.0,
            ..LayoutStyle::default()
        })
        .child(
            Img::new(content.avatar, Rc::clone(images))
                .size(40.0, 40.0)
                .describe(),
        )
        .child(
            StyledText::new(content.title.clone(), style(), engine.clone())
                .with_highlights(content.highlights.clone())
                .size(400.0, 20.0)
                .describe(),
        )
        .child(
            StyledText::new(content.subtitle.clone(), style(), engine.clone())
                .size(400.0, 20.0)
                .describe(),
        )
}

fn list(
    content: &[RowContent],
    engine: &SharedTextEngine,
    images: &SharedImageEngine,
) -> Description {
    Description::new::<ListRoot>()
        .diff_key(RowKey(usize::MAX))
        .style(LayoutStyle {
            flex_direction: FlexDirection::Column,
            ..LayoutStyle::default()
        })
        .children(content.iter().map(|row_content| row(row_content, engine, images)))
}

/// Assert no node in the tree is a compositing boundary.
///
/// The gate's "not because they're inside a `.boundary()`" clause, checked
/// mechanically. Recursive over the whole description, so adding one anywhere —
/// including inside an element's own `describe` — fails loudly.
fn no_boundary_anywhere(description: &Description) {
    assert!(
        !description.is_boundary(),
        "{} declared a boundary; the gate is about ambient reconciliation",
        description.type_name()
    );
    for child in description.child_descriptions() {
        no_boundary_anywhere(child);
    }
}

/// What one frame cost.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
struct FrameCost {
    /// Lines `cosmic-text` was actually asked to shape.
    lines_shaped: u64,
    /// Shape requests answered from the shaping cache.
    cache_hits: u64,
    /// Elements whose `Emit` was called.
    nodes_emitted: usize,
    /// Elements the reconciler reused outright.
    nodes_reused: usize,
    /// Bytes the patch uploaded.
    upload_bytes: u64,
}

/// An image tile source that hands out one tile per distinct key.
///
/// [`GateTiles`]' image counterpart, and it exists for the same reason: the gate
/// measures reconciliation, so the atlas is a substitute here and the real one
/// is exercised in `wgpui-wgpu`'s own tests.
#[derive(Default)]
struct GateImageTiles {
    tiles: std::collections::HashMap<ImageRasterKey, ImageTile>,
}

impl ImageTileSource for GateImageTiles {
    fn tile_for(
        &mut self,
        key: ImageRasterKey,
        decode: &mut dyn FnMut(ImageRasterKey) -> Option<RasterizedImage>,
    ) -> Option<ImageTile> {
        if let Some(tile) = self.tiles.get(&key) {
            return Some(*tile);
        }
        let raster = decode(key)?;
        let next = self.tiles.len() as u32;
        let tile = ImageTile {
            tile: AtlasTileId::new(1, next).expect("the gate stays well inside 24 bits"),
            atlas_origin: [next as f32 * 48.0, 0.0],
            atlas_size: [raster.size[0] as f32, raster.size[1] as f32],
        };
        self.tiles.insert(key, tile);
        Some(tile)
    }
}

/// An engine holding one decoded avatar per row.
///
/// Real decoded frames rather than bare ids, so the avatars in this gate go
/// through the whole Phase 6.2 path — decode, tile, sprite — rather than being
/// element-shaped placeholders. The returned ids are asserted against the ones
/// [`content`] fabricates, because the cache mints them and the row data names
/// them and nothing else would notice the two drifting apart.
fn image_engine() -> SharedImageEngine {
    let mut cache = ImageCache::new();
    for index in 0..ROWS {
        let frame = DecodedFrame {
            size: [40, 40],
            texels: vec![index as u8; 40 * 40 * 4],
            delay: std::time::Duration::ZERO,
        };
        let source = cache
            .hold(DecodedImage::from_frames(vec![frame]).expect("one frame is a valid image"))
            .expect("holding a decoded avatar must succeed");
        assert_eq!(
            source,
            ImageSourceId::from_raw(index as u64 + 1),
            "the cache's minted ids must be the ones the row data names, or the \
             avatars in this gate silently stop decoding"
        );
    }
    Rc::new(RefCell::new(ImageEngine::new(
        cache,
        Box::new(GateImageTiles::default()),
    )))
}

struct Harness {
    engine: SharedTextEngine,
    images: SharedImageEngine,
    reconciler: Reconciler,
    layout: LayoutTree,
    emitter: Emitter,
    scene: Scene,
    signals: FrameSignals,
}

impl Harness {
    fn new() -> Self {
        Self {
            engine: Rc::new(RefCell::new(TextEngine::new(
                TextShaper::new(),
                Box::new(GateTiles::default()),
            ))),
            images: image_engine(),
            reconciler: Reconciler::new(),
            layout: LayoutTree::new(),
            emitter: Emitter::new(),
            scene: Scene::new(),
            signals: FrameSignals::new(),
        }
    }

    fn frame(&mut self, description: Description) -> Result<FrameCost, Box<dyn std::error::Error>> {
        no_boundary_anywhere(&description);
        self.engine.borrow_mut().shaper().reset_stats();
        self.engine.borrow_mut().shaper().begin_frame();

        let plan = self.reconciler.reconcile(description, &mut self.layout)?;
        if let Some(root) = plan.nodes().first().map(|node| node.layout_node) {
            self.layout.compute_layout(root, definite(640.0, 2_000.0))?;
        }
        let emission = self
            .emitter
            .emit(&plan, &self.layout, &self.signals, &mut self.scene)?;
        let uploads = apply(&mut self.scene, &emission.patch)?;

        let shaper_stats = self.engine.borrow_mut().shaper().stats();
        Ok(FrameCost {
            lines_shaped: shaper_stats.lines_shaped,
            cache_hits: shaper_stats.cache_hits,
            nodes_emitted: emission.stats.nodes_emitted,
            nodes_reused: plan
                .nodes()
                .iter()
                .filter(|node| node.outcome == NodeOutcome::Reused)
                .count(),
            upload_bytes: uploads.byte_count(),
        })
    }
}

/// **The gate.** An unchanged scroll-content-heavy scene costs zero shaping
/// passes, zero emissions, and zero uploaded bytes on its second frame —
/// with no `.boundary()` anywhere in the tree.
#[test]
fn gate_unchanged_rows_cost_no_shaping_under_ambient_reconciliation()
-> Result<(), Box<dyn std::error::Error>> {
    let content = content();
    let mut harness = Harness::new();

    let first = harness.frame(list(&content, &harness.engine.clone(), &harness.images.clone()))?;
    // Two text elements per row, each shaped once. If this is not what the
    // first frame costs, the second frame's zero means nothing.
    assert_eq!(
        first.lines_shaped,
        (ROWS * 2) as u64,
        "the first frame must actually do the work the second frame skips"
    );
    assert_eq!(first.nodes_emitted, ROWS * 3, "one Img and two texts per row");
    assert!(first.upload_bytes > 0);

    let second = harness.frame(list(&content, &harness.engine.clone(), &harness.images.clone()))?;
    assert_eq!(
        second.lines_shaped, 0,
        "an unchanged row must not reach cosmic-text at all"
    );
    assert_eq!(
        second.cache_hits, 0,
        "and must not reach the shaping cache either — reconciliation, not memoisation, is what skips it"
    );
    assert_eq!(second.nodes_emitted, 0);
    assert_eq!(second.upload_bytes, 0);
    assert_eq!(
        second.nodes_reused,
        1 + ROWS * 4,
        "root, plus each row and its three children"
    );

    // A third identical frame, because "costs nothing once" and "costs nothing
    // every frame" are different claims and only the second one is useful.
    let third = harness.frame(list(&content, &harness.engine.clone(), &harness.images.clone()))?;
    assert_eq!(third, second);
    Ok(())
}

/// The other half of the same claim: the saving is real because the work is
/// real, and a row that *does* change still pays for exactly itself.
#[test]
fn one_changed_row_reshapes_exactly_one_line() -> Result<(), Box<dyn std::error::Error>> {
    let mut content = content();
    let mut harness = Harness::new();
    harness.frame(list(&content, &harness.engine.clone(), &harness.images.clone()))?;
    harness.frame(list(&content, &harness.engine.clone(), &harness.images.clone()))?;

    if let Some(row) = content.get_mut(7) {
        row.title = SharedString::from("Row 7 — edited in place");
    }
    let after = harness.frame(list(&content, &harness.engine.clone(), &harness.images.clone()))?;

    assert_eq!(
        after.lines_shaped, 1,
        "one edited title is one shaping pass, not forty"
    );
    assert_eq!(
        after.nodes_emitted, 1,
        "and one re-emission — the avatar and subtitle beside it are untouched"
    );
    assert!(after.upload_bytes > 0);
    assert_eq!(after.nodes_reused, ROWS * 4);
    Ok(())
}

/// A recolour is the case the `diff_key` split exists for: same glyphs, same
/// places, new colour. It must repaint without re-shaping.
#[test]
fn recolouring_a_row_repaints_without_reshaping() -> Result<(), Box<dyn std::error::Error>> {
    let content = content();
    let mut harness = Harness::new();
    let engine = harness.engine.clone();
    let images = harness.images.clone();

    let recoloured = |index: usize| -> Description {
        Description::new::<ListRoot>()
            .diff_key(RowKey(usize::MAX))
            .style(LayoutStyle {
                flex_direction: FlexDirection::Column,
                ..LayoutStyle::default()
            })
            .children(content.iter().enumerate().map(|(row_index, row_content)| {
                let mut style = style();
                if row_index == index {
                    style.color = [1.0, 0.0, 0.0, 1.0];
                }
                Description::new::<Row>()
                    .diff_key(RowKey(0))
                    .style(LayoutStyle {
                        flex_direction: FlexDirection::Row,
                        size: LayoutSize {
                            width: Dimension::length(600.0),
                            height: Dimension::length(48.0),
                        },
                        flex_shrink: 0.0,
                        ..LayoutStyle::default()
                    })
                    .child(
                        Img::new(row_content.avatar, Rc::clone(&images))
                            .size(40.0, 40.0)
                            .describe(),
                    )
                    .child(
                        StyledText::new(row_content.title.clone(), style.clone(), engine.clone())
                            .with_highlights(row_content.highlights.clone())
                            .size(400.0, 20.0)
                            .describe(),
                    )
                    .child(
                        StyledText::new(row_content.subtitle.clone(), style, engine.clone())
                            .size(400.0, 20.0)
                            .describe(),
                    )
            }))
    };

    harness.frame(recoloured(usize::MAX))?;
    harness.frame(recoloured(usize::MAX))?;
    let after = harness.frame(recoloured(3))?;

    assert_eq!(
        after.nodes_emitted, 2,
        "both of row 3's text elements repaint"
    );
    assert_eq!(
        after.lines_shaped, 0,
        "a recolour must not re-shape — this is the whole reason colour is split out of the style comparison"
    );
    assert_eq!(
        after.cache_hits, 2,
        "the re-emission is answered by the shaping cache, which is what makes a recolour cheap rather than free"
    );
    Ok(())
}

/// The shaping cache is a backstop, not the gate's mechanism — a distinction
/// worth checking rather than asserting, because if the cache were doing the
/// work the gate would still pass while claiming the wrong thing.
///
/// Driving the same content through a *fresh* reconciler each frame removes
/// reconciliation and leaves only the cache. The row count of shaping passes
/// then goes from zero to zero-with-cache-hits, which is a different and much
/// weaker result: the cache avoids `cosmic-text` but not the per-row emission,
/// conversion, and upload that reconciliation avoids outright.
#[test]
fn without_reconciliation_the_cache_alone_is_measurably_weaker()
-> Result<(), Box<dyn std::error::Error>> {
    let content = content();
    let mut reconciled = Harness::new();
    reconciled.frame(list(&content, &reconciled.engine.clone(), &reconciled.images.clone()))?;
    let with_reconciliation = reconciled.frame(list(&content, &reconciled.engine.clone(), &reconciled.images.clone()))?;

    let mut cache_only = Harness::new();
    cache_only.frame(list(&content, &cache_only.engine.clone(), &cache_only.images.clone()))?;
    // A fresh reconciler and scene: every element is a new instance, so nothing
    // is reused and the shaping cache is the only thing left standing.
    cache_only.reconciler = Reconciler::new();
    cache_only.layout = LayoutTree::new();
    cache_only.emitter = Emitter::new();
    cache_only.scene = Scene::new();
    let without_reconciliation = cache_only.frame(list(&content, &cache_only.engine.clone(), &cache_only.images.clone()))?;

    assert_eq!(with_reconciliation.lines_shaped, 0);
    assert_eq!(without_reconciliation.lines_shaped, 0, "the cache holds");
    assert_eq!(
        without_reconciliation.cache_hits,
        (ROWS * 2) as u64,
        "the cache is what saved it, and says so"
    );
    assert_eq!(with_reconciliation.nodes_emitted, 0);
    assert_eq!(
        without_reconciliation.nodes_emitted,
        ROWS * 3,
        "the cache cannot skip emission, conversion, or upload; reconciliation can"
    );
    assert_eq!(with_reconciliation.upload_bytes, 0);
    assert!(without_reconciliation.upload_bytes > 0);
    Ok(())
}

/// What the avoided work actually costs, so the gate's zero has a magnitude
/// beside it rather than only a comparison.
///
/// Wall-clock, on whatever machine runs it, and therefore asserted only as
/// "shaping is not free" rather than against a threshold — a timing threshold in
/// a test is a flake waiting for a slower CI box. The number itself is reported
/// in `docs/phase-5-results.md` with the machine it came from.
#[test]
fn the_avoided_shaping_is_not_free() -> Result<(), Box<dyn std::error::Error>> {
    let content = content();
    let mut shaper = TextShaper::new();
    let font_id = shaper.resolve_font(&font("Segoe UI"))?;

    let started = Instant::now();
    for row in &content {
        for text in [&row.title, &row.subtitle] {
            shaper.shape_line_uncached(
                text.as_str(),
                14.0,
                &[wgpui_text::shaping::FontRun::new(text.len(), font_id)],
            )?;
        }
    }
    let elapsed = started.elapsed();

    assert_eq!(shaper.stats().lines_shaped, (ROWS * 2) as u64);
    assert!(
        elapsed.as_nanos() > 0,
        "shaping {} lines took no measurable time, which means this measured nothing",
        ROWS * 2
    );
    println!(
        "shaping {} lines from cold: {:?} ({:?} per line)",
        ROWS * 2,
        elapsed,
        elapsed / (ROWS * 2) as u32
    );
    Ok(())
}

/// Identity is positional throughout — no element in this tree is ever named,
/// which is the `.id()`-free property SFD §1.0 makes possible and every gate
/// above quietly depends on.
#[test]
fn no_row_is_ever_named() {
    let content = content();
    let engine: SharedTextEngine = Rc::new(RefCell::new(TextEngine::new(
        TextShaper::new(),
        Box::new(GateTiles::default()),
    )));
    let description = list(&content, &engine, &image_engine());

    fn unnamed(description: &Description) {
        assert_eq!(
            description.element_id(),
            None,
            "{} was named; the gate is about positional identity",
            description.type_name()
        );
        for child in description.child_descriptions() {
            unnamed(child);
        }
    }
    unnamed(&description);

    // And positional identity is stable: the same slot path addresses the same
    // instance across frames without anything opting in.
    let path = [ElementId::Slot(0), ElementId::Slot(7), ElementId::Slot(1)];
    assert_eq!(
        InstanceKey::from_path(&path),
        InstanceKey::from_path(&path)
    );
}
