//! Shaped-run → patch conversion: the arrow §3.3 names as this file's whole
//! job, and Phase 5's headline deliverable. See
//! docs/gpu-native-architecture.md §2, §3.3, §6.
//!
//! # What changes here relative to the legacy backend
//!
//! Almost nothing about *shaping* — that is [`crate::shaping`]'s business and
//! §6 freezes it. What changes is what happens after. The legacy paint path
//! walks a shaped line and calls `Window::paint_glyph` per glyph, which pushes a
//! `MonochromeSprite` straight onto a `Scene` that is rebuilt from scratch every
//! frame. 2.0 turns the same shaped line into
//! [`GlyphRun`]/[`wgpui_core::patch::primitive::Glyph`] patch payloads, which
//! are addressed, diffed, and delta-uploaded like every other primitive — so a
//! run that did not change is not re-uploaded, and, more importantly, a run
//! whose *element* did not change is never converted in the first place.
//!
//! # One `GlyphRun` per shaped run, not per line
//!
//! A [`crate::shaping::ShapedLine`] is already segmented by face, because
//! fallback can substitute one mid-line. That segmentation is kept rather than
//! flattened, for a reason that outlives this phase: a face decides which atlas
//! a glyph's raster comes out of (a colour emoji is not a coverage mask), and a
//! draw call cannot mix texture formats. Flattening now would mean re-deriving
//! the same split in the sprite pipeline later, from data that no longer records
//! it.
//!
//! # Every shaped glyph produces a slot, including the ones that draw nothing
//!
//! A space shapes to a positioned glyph with a real advance and no coverage.
//! It could be dropped — it costs a slab slot and draws nothing — and it is
//! not, because `line_layout`'s index-to-position mapping walks glyphs to answer
//! "where is byte 12", and a run with holes in it answers wrong. The slot is
//! marked with [`AtlasTileId::NONE`], which
//! [`wgpui_core::patch::primitive::GlyphRun::atlas_tiles`] filters out, so a
//! blank glyph is never a tile reference and can never make its layer subscribe
//! to an eviction it does not care about.

use crate::shaping::{ShapedLine, ShapedRun};
use wgpui_core::patch::primitive::{AtlasTileId, Glyph, GlyphRun};
use wgpui_core::scene::atlas::{AtlasKind, GlyphRasterKey, GlyphTileSource};

/// How many horizontal sub-pixel positions a glyph is rasterised at.
///
/// Four, matching the legacy `SUBPIXEL_VARIANTS_X`. Text advances by fractional
/// pixels, and rounding every glyph to a whole pixel visibly bunches and spaces
/// letters; rasterising four variants and picking the nearest is the standard
/// trade of four times the atlas footprint for correct spacing.
pub const SUBPIXEL_VARIANTS_X: u8 = 4;

/// How many vertical sub-pixel positions a glyph is rasterised at.
///
/// One, matching the legacy `SUBPIXEL_VARIANTS_Y`: horizontal text sits on whole
/// baselines, so vertical variants would multiply the atlas footprint for no
/// visible difference.
pub const SUBPIXEL_VARIANTS_Y: u8 = 1;

/// The sub-pixel variant a glyph at `position` (in device pixels) rasterises at.
///
/// Same expression as the legacy `Window::paint_glyph`, kept identical rather
/// than tidied, so a glyph rasterised by either backend lands in the same
/// variant bucket while both exist.
pub fn subpixel_variant(position: [f32; 2]) -> [u8; 2] {
    [
        (position[0].fract() * f32::from(SUBPIXEL_VARIANTS_X)).floor() as u8,
        (position[1].fract() * f32::from(SUBPIXEL_VARIANTS_Y)).floor() as u8,
    ]
}

/// Where a shaped line is placed and how it is coloured.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RunPlacement {
    /// The line's origin in the owning layer's coordinate space — the pen
    /// position of its first glyph's baseline.
    pub origin: [f32; 2],
    /// Straight-alpha RGBA the run draws in.
    pub color: [f32; 4],
    /// Device-pixel ratio the glyphs are rasterised at.
    ///
    /// Part of the raster identity, not of the layout: the same line at 1× and
    /// 2× is the same glyph positions and two different sets of bitmaps.
    pub scale_factor: f32,
}

impl Default for RunPlacement {
    fn default() -> Self {
        Self {
            origin: [0.0, 0.0],
            color: [0.0, 0.0, 0.0, 1.0],
            scale_factor: 1.0,
        }
    }
}

/// What one conversion actually did.
///
/// Reported because Phase 5's gate is a claim about work not happening, and the
/// only honest way to check that is to count it when it does.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ConversionStats {
    /// Runs produced.
    pub runs: usize,
    /// Glyph slots written.
    pub glyphs: usize,
    /// Slots that reference a real atlas tile.
    pub tiles_referenced: usize,
    /// Slots that draw nothing — whitespace and refused rasters.
    pub blank_glyphs: usize,
}

/// Turn one shaped line into the patch payloads the scene accepts.
///
/// `tiles` is asked for a raster per glyph; see
/// [`GlyphTileSource::tile_for`] for what `None` means and why it is ordinary.
pub fn glyph_runs(
    line: &ShapedLine,
    placement: RunPlacement,
    tiles: &mut dyn GlyphTileSource,
) -> (Vec<GlyphRun>, ConversionStats) {
    let mut stats = ConversionStats::default();
    let mut runs = Vec::with_capacity(line.runs.len());
    for run in &line.runs {
        let converted = glyph_run(run, line.font_size, placement, tiles, &mut stats);
        // An empty run cannot happen from shaping, but a caller can construct
        // one; emitting it would cost a record that draws nothing and would
        // make the run count disagree with what is on screen.
        if !converted.glyphs.is_empty() {
            runs.push(converted);
        }
    }
    stats.runs = runs.len();
    (runs, stats)
}

fn glyph_run(
    run: &ShapedRun,
    font_size: f32,
    placement: RunPlacement,
    tiles: &mut dyn GlyphTileSource,
    stats: &mut ConversionStats,
) -> GlyphRun {
    let mut glyphs = Vec::with_capacity(run.glyphs.len());
    for shaped in &run.glyphs {
        // The pen position in the layer's own space, which is what the glyph
        // slot carries, and in device pixels, which is what decides the raster.
        let pen = [
            placement.origin[0] + shaped.position[0],
            placement.origin[1] + shaped.position[1],
        ];
        let device = [
            pen[0] * placement.scale_factor,
            pen[1] * placement.scale_factor,
        ];

        let key = GlyphRasterKey {
            font: u32::try_from(run.font_id.0).unwrap_or(u32::MAX),
            glyph: shaped.id.0,
            font_size_bits: (font_size * placement.scale_factor).to_bits(),
            subpixel: subpixel_variant(device),
            scale_factor_bits: placement.scale_factor.to_bits(),
            kind: if shaped.is_emoji {
                AtlasKind::Polychrome
            } else {
                AtlasKind::Monochrome
            },
        };

        let glyph = match tiles.tile_for(key) {
            Some(tile) => {
                stats.tiles_referenced += 1;
                Glyph {
                    position: [pen[0] + tile.bearing[0], pen[1] + tile.bearing[1]],
                    atlas_origin: tile.atlas_origin,
                    atlas_size: tile.atlas_size,
                    glyph_id: shaped.id.0,
                    atlas_tile: tile.tile,
                }
            }
            None => {
                stats.blank_glyphs += 1;
                Glyph {
                    position: pen,
                    glyph_id: shaped.id.0,
                    atlas_tile: AtlasTileId::NONE,
                    ..Glyph::ZERO
                }
            }
        };
        glyphs.push(glyph);
        stats.glyphs += 1;
    }

    GlyphRun {
        color: placement.color,
        glyphs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shaping::{FontId, FontRun, GlyphId, ShapedGlyph, SharedString, TextShaper};
    use std::collections::HashMap;
    use wgpui_core::scene::atlas::GlyphTile;

    /// A tile source that allocates a fresh tile per distinct key and reports
    /// nothing for the glyphs a caller marks blank.
    ///
    /// Not a mock of the real atlas: a *substitute* for it, so this module's
    /// tests measure conversion and only conversion. The real allocator's
    /// packing is tested where it lives, in `wgpui-wgpu`'s `render/atlas.rs`,
    /// and the two are wired together in this file's last test.
    #[derive(Default)]
    struct FakeAtlas {
        tiles: HashMap<GlyphRasterKey, GlyphTile>,
        blank_glyphs: Vec<u32>,
        requests: usize,
    }

    impl GlyphTileSource for FakeAtlas {
        fn tile_for(&mut self, key: GlyphRasterKey) -> Option<GlyphTile> {
            self.requests += 1;
            if self.blank_glyphs.contains(&key.glyph) {
                return None;
            }
            let next = self.tiles.len();
            Some(*self.tiles.entry(key).or_insert_with(|| GlyphTile {
                tile: AtlasTileId::new(0, next as u32).expect("test tiles stay in range"),
                atlas_origin: [next as f32 * 8.0, 0.0],
                atlas_size: [8.0, 12.0],
                bearing: [0.5, -9.0],
            }))
        }
    }

    fn shaped(glyphs: &[(u32, f32, bool)]) -> ShapedRun {
        ShapedRun {
            font_id: FontId(0),
            glyphs: glyphs
                .iter()
                .map(|(id, x, is_emoji)| ShapedGlyph {
                    id: GlyphId(*id),
                    position: [*x, 0.0],
                    index: 0,
                    is_emoji: *is_emoji,
                })
                .collect(),
        }
    }

    fn line(runs: Vec<ShapedRun>) -> ShapedLine {
        ShapedLine {
            font_size: 16.0,
            width: 40.0,
            ascent: 12.0,
            descent: 4.0,
            len: 4,
            runs,
        }
    }

    #[test]
    fn each_shaped_glyph_becomes_one_slab_slot() {
        let mut atlas = FakeAtlas::default();
        let (runs, stats) = glyph_runs(
            &line(vec![shaped(&[
                (1, 0.0, false),
                (2, 8.0, false),
                (3, 16.0, false),
            ])]),
            RunPlacement::default(),
            &mut atlas,
        );
        assert_eq!(runs.len(), 1);
        assert_eq!(stats.glyphs, 3);
        assert_eq!(stats.tiles_referenced, 3);
        assert_eq!(stats.blank_glyphs, 0);

        let run = runs.first().expect("one run");
        assert_eq!(run.glyphs.len(), 3);
        // `slot_count` is what the slab allocator reserves against.
        use wgpui_core::patch::primitive::Primitive;
        assert_eq!(run.slot_count(), 3);
    }

    #[test]
    fn a_glyph_with_no_raster_keeps_its_slot_and_references_no_tile() {
        let mut atlas = FakeAtlas {
            blank_glyphs: vec![2],
            ..FakeAtlas::default()
        };
        let (runs, stats) = glyph_runs(
            &line(vec![shaped(&[
                (1, 0.0, false),
                (2, 8.0, false),
                (3, 16.0, false),
            ])]),
            RunPlacement::default(),
            &mut atlas,
        );
        let run = runs.first().expect("one run");
        assert_eq!(
            run.glyphs.len(),
            3,
            "dropping the space would break index-to-position mapping"
        );
        assert_eq!(stats.blank_glyphs, 1);
        assert_eq!(
            run.glyphs.get(1).map(|glyph| glyph.atlas_tile),
            Some(AtlasTileId::NONE)
        );
        assert_eq!(
            run.atlas_tiles().count(),
            2,
            "a blank glyph must never subscribe its layer to an eviction"
        );
    }

    #[test]
    fn a_runs_position_is_the_lines_origin_plus_the_shaped_offset_plus_the_bearing() {
        let mut atlas = FakeAtlas::default();
        let placement = RunPlacement {
            origin: [100.0, 200.0],
            ..RunPlacement::default()
        };
        let (runs, _) = glyph_runs(
            &line(vec![shaped(&[(1, 8.0, false)])]),
            placement,
            &mut atlas,
        );
        assert_eq!(
            runs.first()
                .and_then(|run| run.glyphs.first())
                .map(|g| g.position),
            // origin + shaped offset + the fake atlas's bearing of [0.5, -9.0]
            Some([108.5, 191.0])
        );
    }

    #[test]
    fn the_run_colour_comes_from_the_placement_and_reaches_every_glyph_slot() {
        let mut atlas = FakeAtlas::default();
        let placement = RunPlacement {
            color: [0.2, 0.4, 0.6, 1.0],
            ..RunPlacement::default()
        };
        let (runs, _) = glyph_runs(
            &line(vec![shaped(&[(1, 0.0, false), (2, 8.0, false)])]),
            placement,
            &mut atlas,
        );
        assert_eq!(
            runs.first().map(|run| run.color),
            Some([0.2, 0.4, 0.6, 1.0])
        );
    }

    #[test]
    fn shaped_runs_stay_separate_so_a_face_change_stays_visible() {
        let mut atlas = FakeAtlas::default();
        let text_run = shaped(&[(1, 0.0, false)]);
        let emoji_run = ShapedRun {
            font_id: FontId(1),
            ..shaped(&[(2, 8.0, true)])
        };
        let (runs, stats) = glyph_runs(
            &line(vec![text_run, emoji_run]),
            RunPlacement::default(),
            &mut atlas,
        );
        assert_eq!(runs.len(), 2, "flattening would lose the atlas-kind split");
        assert_eq!(stats.runs, 2);
    }

    #[test]
    fn an_emoji_glyph_is_requested_from_the_colour_atlas_and_text_from_the_coverage_one() {
        let mut atlas = FakeAtlas::default();
        glyph_runs(
            &line(vec![shaped(&[(1, 0.0, false)]), shaped(&[(2, 8.0, true)])]),
            RunPlacement::default(),
            &mut atlas,
        );
        let kinds: Vec<AtlasKind> = atlas.tiles.keys().map(|key| key.kind).collect();
        assert!(kinds.contains(&AtlasKind::Monochrome));
        assert!(kinds.contains(&AtlasKind::Polychrome));
    }

    #[test]
    fn the_same_glyph_at_the_same_subpixel_offset_shares_one_tile() {
        let mut atlas = FakeAtlas::default();
        // Two 'l's exactly 8px apart: same glyph, same fractional offset.
        let (runs, _) = glyph_runs(
            &line(vec![shaped(&[(7, 0.0, false), (7, 8.0, false)])]),
            RunPlacement::default(),
            &mut atlas,
        );
        let tiles: Vec<AtlasTileId> = runs
            .first()
            .map(|run| run.glyphs.iter().map(|g| g.atlas_tile).collect())
            .unwrap_or_default();
        assert_eq!(tiles.first(), tiles.get(1));
        assert_eq!(atlas.tiles.len(), 1);
        assert_eq!(atlas.requests, 2, "the source deduplicates, not the caller");
    }

    #[test]
    fn the_same_glyph_at_a_different_subpixel_offset_gets_its_own_raster() {
        let mut atlas = FakeAtlas::default();
        glyph_runs(
            &line(vec![shaped(&[(7, 0.0, false), (7, 8.5, false)])]),
            RunPlacement::default(),
            &mut atlas,
        );
        assert_eq!(
            atlas.tiles.len(),
            2,
            "rounding both to one raster is what makes text bunch up"
        );
    }

    #[test]
    fn scale_factor_is_part_of_the_raster_identity_and_not_of_the_position() {
        let mut atlas = FakeAtlas::default();
        let one_x = RunPlacement::default();
        let two_x = RunPlacement {
            scale_factor: 2.0,
            ..RunPlacement::default()
        };
        let (at_one, _) = glyph_runs(&line(vec![shaped(&[(7, 3.0, false)])]), one_x, &mut atlas);
        let (at_two, _) = glyph_runs(&line(vec![shaped(&[(7, 3.0, false)])]), two_x, &mut atlas);
        assert_eq!(atlas.tiles.len(), 2, "two scales are two sets of bitmaps");
        assert_eq!(
            at_one
                .first()
                .and_then(|r| r.glyphs.first())
                .map(|g| g.position),
            at_two
                .first()
                .and_then(|r| r.glyphs.first())
                .map(|g| g.position),
            "layout positions are in logical pixels and do not move with the scale"
        );
    }

    #[test]
    fn subpixel_variants_partition_the_unit_interval() {
        assert_eq!(subpixel_variant([0.0, 0.0]), [0, 0]);
        assert_eq!(subpixel_variant([0.24, 0.0]), [0, 0]);
        assert_eq!(subpixel_variant([0.25, 0.0]), [1, 0]);
        assert_eq!(subpixel_variant([0.75, 0.0]), [3, 0]);
        assert_eq!(subpixel_variant([12.99, 0.9]), [3, 0]);
    }

    /// Real shaping through the real conversion: the two halves of this crate
    /// meeting, with nothing synthetic between them except the tile source.
    #[test]
    fn a_really_shaped_line_converts_to_runs_covering_every_glyph() {
        let mut shaper = TextShaper::new();
        let font_id = shaper
            .resolve_font(&crate::shaping::font("Segoe UI"))
            .expect("some face exists");
        let text = SharedString::from("Hello, world");
        let shaped_line = shaper
            .shape_line(&text, 16.0, &[FontRun::new(text.len(), font_id)])
            .expect("shaping must succeed");

        let mut atlas = FakeAtlas::default();
        let (runs, stats) = glyph_runs(&shaped_line, RunPlacement::default(), &mut atlas);

        assert_eq!(
            stats.glyphs,
            shaped_line.glyph_count(),
            "every shaped glyph must reach a slab slot"
        );
        assert_eq!(runs.len(), shaped_line.runs.len());
        use wgpui_core::patch::primitive::Primitive;
        let slots: u32 = runs.iter().map(|run| run.slot_count()).sum();
        assert_eq!(slots as usize, shaped_line.glyph_count());
    }
}
