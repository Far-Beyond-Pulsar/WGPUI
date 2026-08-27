//! Glyph rasterisation: a font outline becomes the pixels an allocated atlas
//! tile is supposed to hold. See docs/gpu-native-architecture.md §3.3, §6, and
//! §9's "Load-bearing, disclosed by Phase 5" row, which is the gap this file
//! closes.
//!
//! # This is a port, not a design
//!
//! The reference is `src/platform/cross/text_system.rs` —
//! `CosmicTextSystemState::raster_bounds` and
//! `CosmicTextSystemState::rasterize_glyph`, plus `src/text_system.rs`'s
//! `TextSystem::rasterize_glyph` wrapper — and it is followed line for line:
//! build a `cosmic_text::CacheKey` from the face, glyph index, device font size,
//! sub-pixel shift and weight; ask `SwashCache::get_image`; take the bitmap's
//! placement as the size and bearing; convert `swash`'s content type into the
//! one byte-per-texel coverage mask or four byte-per-texel colour bitmap the
//! atlas wants. Nothing about hinting, gamma, or colour-emoji handling is
//! decided here that the legacy backend has not already decided.
//!
//! # What is fused relative to the legacy, and why that is not a change
//!
//! The legacy path calls `get_image` twice per glyph: once through
//! `glyph_raster_bounds` to learn the size, and again through `rasterize_glyph`
//! to get the pixels, with a `HashMap<RenderGlyphParams, Bounds<DevicePixels>>`
//! in `TextSystem` in front of the first. Both calls land on the same
//! `SwashCache::image_cache` entry, so the second is already a lookup — the
//! two-step exists because the legacy `PlatformTextSystem` trait is shaped that
//! way (the atlas needs a size before it will call the build closure), not
//! because two rasterisations happen. [`GlyphRasterizer::rasterize`] returns
//! size, bearing and pixels together, which is what
//! `wgpui-wgpu`'s `GlyphAtlas` actually wants, and the legacy `raster_bounds`
//! cache has nothing left to cache: the atlas already answers a resident glyph
//! without reaching this file at all.
//!
//! # Caching
//!
//! Two layers, neither of them new:
//!
//! - `SwashCache::image_cache`, inside `cosmic-text`, memoises outline →
//!   bitmap per `CacheKey`. That is the expensive half and it is the legacy
//!   crate's own cache, kept.
//! - `wgpui-wgpu`'s `GlyphAtlas` maps [`wgpui_core::scene::atlas::GlyphRasterKey`]
//!   → tile, and only calls a rasteriser on a miss. So the `swash` → atlas
//!   format conversion below runs exactly once per distinct raster, not once per
//!   glyph occurrence — a paragraph's forty `e`s cost one conversion.
//!
//! This file adds no third cache. A `HashMap` here would sit between two caches
//! that already cover the same key, and would have to be invalidated when the
//! font database changes, which is a maintenance cost against no measurement.

use crate::shaping::{FontId, ShapeError, TextShaper};
use cosmic_text::{CacheKey, CacheKeyFlags, SwashCache, SwashContent};
use wgpui_core::scene::atlas::{AtlasKind, GlyphRasterKey, RasterizedGlyph};

/// How many horizontal sub-pixel positions a glyph is rasterised at.
///
/// Re-exported from [`crate::patch`] rather than redeclared: the variant the
/// converter picks and the shift the rasteriser applies have to be the same
/// quantisation or the bitmap does not match the position it was requested for.
pub use crate::patch::{SUBPIXEL_VARIANTS_X, SUBPIXEL_VARIANTS_Y};

/// Why a glyph produced no bitmap.
///
/// Every variant is ordinary rather than exceptional — whitespace and
/// unmapped codepoints reach all of them in normal text — which is why
/// [`GlyphRasterizer::rasterize`] returns a `Result` a caller is expected to
/// turn into a blank glyph rather than a frame failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RasterError {
    /// The key named a [`FontId`] the shaper never issued.
    UnknownFont(ShapeError),
    /// The glyph index does not fit the `u16` `cosmic-text` indexes glyphs by.
    GlyphIdOutOfRange(u32),
    /// `swash` produced no image for this outline at this size.
    NoOutline,
    /// The outline is real but covers no pixels — a space, a zero-width joiner.
    ///
    /// The legacy `rasterize_glyph` bails on exactly this
    /// (`anyhow::bail!("glyph bounds are empty")`), and the legacy paint path
    /// checks `raster_bounds.is_zero()` before it ever asks, so a blank glyph
    /// never reaches the atlas under either backend.
    EmptyRaster,
    /// The converted bitmap's length disagrees with its declared size.
    ///
    /// Reported rather than trusted because the atlas blits it row by row: a
    /// short bitmap would take texels from beyond its own end, and a `swash`
    /// content type this code does not expect is a better error than a wrong
    /// picture.
    MalformedBitmap {
        /// Bytes the size and kind imply.
        expected: usize,
        /// Bytes the conversion produced.
        actual: usize,
    },
}

impl std::fmt::Display for RasterError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RasterError::UnknownFont(error) => write!(formatter, "{error}"),
            RasterError::GlyphIdOutOfRange(glyph) => {
                write!(formatter, "glyph index {glyph} does not fit a u16")
            }
            RasterError::NoOutline => {
                formatter.write_str("the font has no outline for this glyph at this size")
            }
            RasterError::EmptyRaster => formatter.write_str("the glyph covers no pixels"),
            RasterError::MalformedBitmap { expected, actual } => write!(
                formatter,
                "a rasterised glyph declared {expected} bytes and produced {actual}"
            ),
        }
    }
}

impl std::error::Error for RasterError {}

impl From<ShapeError> for RasterError {
    fn from(error: ShapeError) -> Self {
        RasterError::UnknownFont(error)
    }
}

/// What a rasteriser has been made to do.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct RasterStats {
    /// Requests that produced a bitmap.
    pub rasterized: u64,
    /// Requests that produced nothing — blanks, missing outlines, bad keys.
    pub declined: u64,
}

/// Turns a [`GlyphRasterKey`] into the pixels its atlas tile holds.
///
/// Separate from [`TextShaper`] rather than a field on it, deliberately: shaping
/// and rasterising are asked for at different times by different callers (a
/// layout pass shapes; a paint pass rasterises whatever is not already resident)
/// and §6 only freezes the first. Keeping them apart means a caller that never
/// draws — a measurement pass, `line_wrapper`, the reconciliation gate — never
/// allocates a `SwashCache` at all. The shaper is passed in per call because the
/// two share one `cosmic_text::FontSystem`, which is the one thing they cannot
/// each own a copy of.
pub struct GlyphRasterizer {
    cache: SwashCache,
    stats: RasterStats,
}

impl Default for GlyphRasterizer {
    fn default() -> Self {
        Self::new()
    }
}

impl GlyphRasterizer {
    /// A rasteriser with an empty bitmap cache.
    pub fn new() -> Self {
        Self {
            cache: SwashCache::new(),
            stats: RasterStats::default(),
        }
    }

    /// What this rasteriser has been made to do since it was created.
    pub fn stats(&self) -> RasterStats {
        self.stats
    }

    /// Reset the counters, so a test can measure one frame rather than a run.
    pub fn reset_stats(&mut self) {
        self.stats = RasterStats::default();
    }

    /// Bitmaps currently held by `cosmic-text`'s own cache.
    ///
    /// The legacy crate reports the same number under its `flamegraph` feature
    /// (`CosmicTextSystem::glyph_cache_memory_usage` walks
    /// `SwashCache::image_cache`); this is the count rather than the byte total,
    /// because nothing in 2.0 has a memory-report surface to put bytes into yet.
    pub fn cached_bitmap_count(&self) -> usize {
        self.cache.image_cache.len()
    }

    /// Rasterise one glyph.
    ///
    /// `shaper` must be the shaper that issued `key.font`; it owns both the face
    /// table the key indexes and the `FontSystem` `swash` scales against.
    pub fn rasterize(
        &mut self,
        shaper: &mut TextShaper,
        key: GlyphRasterKey,
    ) -> Result<RasterizedGlyph, RasterError> {
        match self.rasterize_inner(shaper, key) {
            Ok(glyph) => {
                self.stats.rasterized += 1;
                Ok(glyph)
            }
            Err(error) => {
                self.stats.declined += 1;
                Err(error)
            }
        }
    }

    fn rasterize_inner(
        &mut self,
        shaper: &mut TextShaper,
        key: GlyphRasterKey,
    ) -> Result<RasterizedGlyph, RasterError> {
        let (database_id, weight) = shaper.raster_face(FontId(key.font as usize))?;
        let glyph_id =
            u16::try_from(key.glyph).map_err(|_| RasterError::GlyphIdOutOfRange(key.glyph))?;

        // The legacy expression, kept verbatim rather than simplified. The
        // division by the scale factor is the reason `GlyphRasterKey` carries
        // one: it makes the shift a *logical*-pixel quantity, which means the
        // four sub-pixel variants do not map one-to-one onto `swash`'s four
        // bins at any scale but 1×. That is a legacy quirk, not an improvement,
        // and reproducing it is the point — both backends have to agree about
        // which bitmap a variant means while both exist.
        let scale_factor = f32::from_bits(key.scale_factor_bits);
        let subpixel_shift = [
            f32::from(key.subpixel[0]) / f32::from(SUBPIXEL_VARIANTS_X) / scale_factor,
            f32::from(key.subpixel[1]) / f32::from(SUBPIXEL_VARIANTS_Y) / scale_factor,
        ];

        let cache_key = CacheKey::new(
            database_id,
            glyph_id,
            // Already device pixels: `crate::patch` multiplies by the scale
            // factor before it builds the key, exactly as the legacy
            // `params.font_size * params.scale_factor` does at the call.
            f32::from_bits(key.font_size_bits),
            (subpixel_shift[0], subpixel_shift[1].trunc()),
            weight,
            CacheKeyFlags::empty(),
        )
        .0;

        let image = self
            .cache
            .get_image(shaper.font_system_mut(), cache_key)
            .as_ref()
            .ok_or(RasterError::NoOutline)?;

        let size = [image.placement.width, image.placement.height];
        if size[0] == 0 || size[1] == 0 {
            return Err(RasterError::EmptyRaster);
        }
        // `glyph_raster_bounds`'s origin, unchanged: `point(placement.left,
        // -placement.top)`. `top` is measured up from the baseline and the
        // bearing is measured down from the pen, so the sign flips.
        let bearing = [image.placement.left as f32, -(image.placement.top as f32)];
        let texels = convert(key.kind, image.content, &image.data);

        let rasterized = RasterizedGlyph {
            size,
            kind: key.kind,
            bearing,
            texels,
        };
        if !rasterized.is_well_formed() {
            return Err(RasterError::MalformedBitmap {
                expected: rasterized.expected_texel_bytes(),
                actual: rasterized.texels.len(),
            });
        }
        Ok(rasterized)
    }
}

/// `swash`'s content types mapped onto the atlas's two texel formats.
///
/// Transcribed from the legacy `rasterize_glyph`'s two `match` arms, including
/// the Rec. 709 luminance weights it uses to flatten a sub-pixel mask. The two
/// cross cases are not hypothetical: a colour-emoji face can produce a plain
/// mask for a glyph with no colour form, and a monochrome run can pick up a
/// bitmap-only face whose glyphs are colour.
fn convert(kind: AtlasKind, content: SwashContent, data: &[u8]) -> Vec<u8> {
    match kind {
        AtlasKind::Polychrome => match content {
            SwashContent::Color => data.to_vec(),
            SwashContent::Mask => data
                .iter()
                .flat_map(|alpha| [255, 255, 255, *alpha])
                .collect(),
            SwashContent::SubpixelMask => data
                .chunks_exact(4)
                .flat_map(|pixel| [255, 255, 255, luminance(pixel)])
                .collect(),
        },
        AtlasKind::Monochrome => match content {
            SwashContent::Mask => data.to_vec(),
            SwashContent::SubpixelMask => data.chunks_exact(4).map(luminance).collect(),
            SwashContent::Color => data.chunks_exact(4).filter_map(|pixel| pixel.get(3).copied()).collect(),
        },
    }
}

/// Rec. 709 luminance of an RGB(A) texel, as the legacy `rasterize_glyph`
/// computes it.
fn luminance(pixel: &[u8]) -> u8 {
    let channel = |index: usize| f32::from(pixel.get(index).copied().unwrap_or(0));
    (channel(0) * 0.2126 + channel(1) * 0.7152 + channel(2) * 0.0722) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shaping::{FontRun, SharedString, font};
    use crate::test_fonts;

    fn key_for(font_id: FontId, glyph: u32) -> GlyphRasterKey {
        GlyphRasterKey {
            font: font_id.0 as u32,
            glyph,
            font_size_bits: 32.0f32.to_bits(),
            subpixel: [0, 0],
            scale_factor_bits: 1.0f32.to_bits(),
            kind: AtlasKind::Monochrome,
        }
    }

    /// The glyph index of a character in the embedded test face, via real
    /// shaping — the only honest way to get one, since glyph indices are
    /// font-local and nothing outside the face knows them.
    fn glyph_of(shaper: &mut TextShaper, font_id: FontId, character: char) -> u32 {
        let text = SharedString::from(character.to_string());
        let line = shaper
            .shape_line(&text, 32.0, &[FontRun::new(text.len(), font_id)])
            .expect("the embedded face shapes one character");
        line.runs
            .first()
            .and_then(|run| run.glyphs.first())
            .map(|glyph| glyph.id.0)
            .expect("one character shapes to at least one glyph")
    }

    #[test]
    fn a_letter_rasterises_to_a_coverage_mask_with_ink_in_it() {
        let mut shaper = test_fonts::shaper();
        let font_id = shaper
            .resolve_font(&font(test_fonts::FAMILY))
            .expect("the embedded face resolves");
        let glyph = glyph_of(&mut shaper, font_id, 'H');

        let mut rasterizer = GlyphRasterizer::new();
        let raster = rasterizer
            .rasterize(&mut shaper, key_for(font_id, glyph))
            .expect("'H' at 32px has an outline");

        assert_eq!(raster.kind, AtlasKind::Monochrome);
        assert!(raster.is_well_formed());
        assert_eq!(
            raster.texels.len(),
            raster.size[0] as usize * raster.size[1] as usize,
            "a coverage mask is one byte per texel"
        );
        assert!(
            raster.texels.iter().any(|coverage| *coverage > 0),
            "a capital H that rasterises to nothing is not a rasterised H"
        );
        assert!(
            raster.size[0] > 0 && raster.size[1] > 0 && raster.size[1] <= 64,
            "'H' at 32px should be a few tens of texels tall, got {:?}",
            raster.size
        );
        assert_eq!(rasterizer.stats().rasterized, 1);
        assert_eq!(rasterizer.stats().declined, 0);
    }

    #[test]
    fn a_capital_sits_above_the_baseline_and_a_comma_hangs_below_it() {
        let mut shaper = test_fonts::shaper();
        let font_id = shaper
            .resolve_font(&font(test_fonts::FAMILY))
            .expect("the embedded face resolves");
        let capital = glyph_of(&mut shaper, font_id, 'H');
        let comma = glyph_of(&mut shaper, font_id, ',');

        let mut rasterizer = GlyphRasterizer::new();
        let capital = rasterizer
            .rasterize(&mut shaper, key_for(font_id, capital))
            .expect("'H' has an outline");
        let comma = rasterizer
            .rasterize(&mut shaper, key_for(font_id, comma))
            .expect("',' has an outline");

        assert!(
            capital.bearing[1] < 0.0,
            "a capital's ink starts above the pen, so its y bearing is negative: {:?}",
            capital.bearing
        );
        assert!(
            comma.bearing[1] + comma.size[1] as f32 > 0.0,
            "a comma's ink reaches below the baseline: bearing {:?} size {:?}",
            comma.bearing,
            comma.size
        );
    }

    #[test]
    fn a_space_declines_rather_than_producing_an_empty_bitmap() {
        let mut shaper = test_fonts::shaper();
        let font_id = shaper
            .resolve_font(&font(test_fonts::FAMILY))
            .expect("the embedded face resolves");
        let space = glyph_of(&mut shaper, font_id, ' ');

        let mut rasterizer = GlyphRasterizer::new();
        assert_eq!(
            rasterizer.rasterize(&mut shaper, key_for(font_id, space)),
            Err(RasterError::EmptyRaster)
        );
        assert_eq!(rasterizer.stats().declined, 1);
    }

    #[test]
    fn a_font_id_the_shaper_never_issued_is_reported_not_indexed() {
        let mut shaper = test_fonts::shaper();
        let mut rasterizer = GlyphRasterizer::new();
        assert!(matches!(
            rasterizer.rasterize(&mut shaper, key_for(FontId(7), 1)),
            Err(RasterError::UnknownFont(ShapeError::UnknownFont(FontId(7))))
        ));
    }

    #[test]
    fn a_glyph_index_beyond_a_u16_is_reported_rather_than_truncated() {
        let mut shaper = test_fonts::shaper();
        let font_id = shaper
            .resolve_font(&font(test_fonts::FAMILY))
            .expect("the embedded face resolves");
        let mut rasterizer = GlyphRasterizer::new();
        assert_eq!(
            rasterizer.rasterize(&mut shaper, key_for(font_id, 70_000)),
            Err(RasterError::GlyphIdOutOfRange(70_000))
        );
    }

    #[test]
    fn different_sub_pixel_variants_of_one_glyph_are_different_bitmaps() {
        let mut shaper = test_fonts::shaper();
        let font_id = shaper
            .resolve_font(&font(test_fonts::FAMILY))
            .expect("the embedded face resolves");
        let glyph = glyph_of(&mut shaper, font_id, 'n');

        let mut rasterizer = GlyphRasterizer::new();
        let at_zero = rasterizer
            .rasterize(&mut shaper, key_for(font_id, glyph))
            .expect("'n' has an outline");
        let at_half = rasterizer
            .rasterize(
                &mut shaper,
                GlyphRasterKey {
                    subpixel: [2, 0],
                    ..key_for(font_id, glyph)
                },
            )
            .expect("'n' has an outline at every variant");
        assert_ne!(
            (at_zero.texels.as_slice(), at_zero.bearing),
            (at_half.texels.as_slice(), at_half.bearing),
            "if a half-pixel shift changed nothing, sub-pixel positioning would be free and pointless"
        );
    }

    #[test]
    fn a_bigger_size_is_a_bigger_bitmap() {
        let mut shaper = test_fonts::shaper();
        let font_id = shaper
            .resolve_font(&font(test_fonts::FAMILY))
            .expect("the embedded face resolves");
        let glyph = glyph_of(&mut shaper, font_id, 'H');

        let mut rasterizer = GlyphRasterizer::new();
        let small = rasterizer
            .rasterize(
                &mut shaper,
                GlyphRasterKey {
                    font_size_bits: 12.0f32.to_bits(),
                    ..key_for(font_id, glyph)
                },
            )
            .expect("'H' at 12px");
        let large = rasterizer
            .rasterize(
                &mut shaper,
                GlyphRasterKey {
                    font_size_bits: 48.0f32.to_bits(),
                    ..key_for(font_id, glyph)
                },
            )
            .expect("'H' at 48px");
        assert!(
            large.size[0] > small.size[0] && large.size[1] > small.size[1],
            "48px {:?} must be larger than 12px {:?}",
            large.size,
            small.size
        );
    }

    #[test]
    fn the_same_key_twice_reaches_cosmic_texts_own_bitmap_cache() {
        let mut shaper = test_fonts::shaper();
        let font_id = shaper
            .resolve_font(&font(test_fonts::FAMILY))
            .expect("the embedded face resolves");
        let glyph = glyph_of(&mut shaper, font_id, 'H');

        let mut rasterizer = GlyphRasterizer::new();
        let first = rasterizer
            .rasterize(&mut shaper, key_for(font_id, glyph))
            .expect("'H' has an outline");
        assert_eq!(rasterizer.cached_bitmap_count(), 1);
        let second = rasterizer
            .rasterize(&mut shaper, key_for(font_id, glyph))
            .expect("'H' has an outline");
        assert_eq!(first, second);
        assert_eq!(
            rasterizer.cached_bitmap_count(),
            1,
            "a repeat request must not add a second bitmap"
        );
    }

    #[test]
    fn a_colour_request_for_a_mask_glyph_is_widened_to_rgba() {
        let mut shaper = test_fonts::shaper();
        let font_id = shaper
            .resolve_font(&font(test_fonts::FAMILY))
            .expect("the embedded face resolves");
        let glyph = glyph_of(&mut shaper, font_id, 'H');

        let mut rasterizer = GlyphRasterizer::new();
        let mask = rasterizer
            .rasterize(&mut shaper, key_for(font_id, glyph))
            .expect("'H' as a coverage mask");
        let colour = rasterizer
            .rasterize(
                &mut shaper,
                GlyphRasterKey {
                    kind: AtlasKind::Polychrome,
                    ..key_for(font_id, glyph)
                },
            )
            .expect("'H' widened into the colour atlas");

        assert_eq!(mask.size, colour.size);
        assert_eq!(colour.texels.len(), mask.texels.len() * 4);
        assert!(colour.is_well_formed());
        // The legacy widening is `[255, 255, 255, alpha]` — white ink whose
        // coverage lives in the alpha channel, so the sprite pipeline can tint
        // it the same way it tints a mask.
        assert_eq!(
            colour.texels.chunks_exact(4).map(|texel| texel[3]).collect::<Vec<u8>>(),
            mask.texels,
            "widening must move coverage into alpha and nowhere else"
        );
        assert!(colour.texels.chunks_exact(4).all(|texel| texel[..3] == [255, 255, 255]));
    }

    #[test]
    fn the_two_content_conversions_are_the_legacy_ones() {
        // Not a round trip through swash: the arms that matter here are the
        // *cross* cases, which need a font that produces them to reach
        // organically. Checked directly against the legacy expressions instead,
        // so a future tidy-up of `convert` that changes a weight fails here.
        assert_eq!(
            convert(AtlasKind::Monochrome, SwashContent::SubpixelMask, &[10, 20, 30, 255]),
            vec![(10.0 * 0.2126 + 20.0 * 0.7152 + 30.0 * 0.0722) as u8]
        );
        assert_eq!(
            convert(AtlasKind::Monochrome, SwashContent::Color, &[1, 2, 3, 4]),
            vec![4],
            "a colour bitmap flattened into a mask keeps its alpha, not its luminance"
        );
        assert_eq!(
            convert(AtlasKind::Polychrome, SwashContent::Mask, &[7]),
            vec![255, 255, 255, 7]
        );
        assert_eq!(
            convert(AtlasKind::Polychrome, SwashContent::Color, &[1, 2, 3, 4]),
            vec![1, 2, 3, 4],
            "a colour bitmap in the colour atlas is copied, never reinterpreted"
        );
    }
}
