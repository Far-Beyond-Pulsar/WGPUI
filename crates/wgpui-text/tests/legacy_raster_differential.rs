//! Phase 5.5's gate: `wgpui-text`'s rasteriser produces the same pixels as the
//! legacy one, for a representative set of glyphs, sizes, sub-pixel variants and
//! scale factors.
//!
//! Same discipline as Phase 3's occlusion differential — prove the new path
//! agrees with the reference rather than prove it is self-consistent — and the
//! same shape of oracle: an independent implementation, compared record for
//! record, that can fail.
//!
//! # Why the oracle is a transcription and not a call into `gpui-ce`
//!
//! It would be better to call the legacy code. It is not reachable:
//! `RenderGlyphParams` is `pub(crate)` in the root crate, `PlatformTextSystem`
//! is a private trait, and `TextSystem::rasterize_glyph` is `pub(crate)` too, so
//! no code outside `gpui-ce` can invoke the legacy rasteriser at all. Making it
//! reachable means changing `src/`, which every phase from 1 onward is forbidden
//! and which §9 freezes ("legacy backend is frozen (bugfixes only)").
//!
//! So [`legacy`] below transcribes `src/platform/cross/text_system.rs`'s
//! `CosmicTextSystemState::{raster_bounds, rasterize_glyph}` and the face
//! loading in `load_family` that feeds them, expression for expression, against
//! its *own* `FontSystem` and `SwashCache`. That is weaker than calling the real
//! thing in exactly one way, stated plainly rather than buried: it cannot catch
//! the legacy file changing underneath it. It is not weaker in the way that
//! matters — the two sides share no state, no cache, and no code path, so an
//! agreement is a real agreement and a disagreement is a real bug in one of
//! them. §7.4 below deliberately breaks the new side and watches the gate fail,
//! which is what makes the agreement worth believing.
//!
//! Both sides shape and rasterise against the same single embedded face
//! (`wgpui_text::test_fonts`), loaded into two separate databases containing
//! nothing else, so "they resolved different faces" is not a way for this test
//! to pass or fail by accident.

use cosmic_text::{CacheKey, CacheKeyFlags, FontSystem, SwashCache, SwashContent, fontdb};
use std::sync::Arc;
use wgpui_core::scene::atlas::{AtlasKind, GlyphRasterKey};
use wgpui_text::raster::{GlyphRasterizer, RasterError};
use wgpui_text::shaping::{FontRun, SharedString, TextShaper, font};
use wgpui_text::test_fonts;

/// What either side produced for one glyph request.
///
/// Compared as a whole rather than field by field so that "one declined and the
/// other did not" is a failure with the same weight as "the bytes differ" —
/// which it is: a glyph the new path skips is a glyph missing from the screen.
#[derive(Clone, Debug, PartialEq)]
enum Raster {
    /// Size, bearing, and the texels themselves.
    Bitmap {
        size: [u32; 2],
        bearing: [f32; 2],
        texels: Vec<u8>,
    },
    /// A real outline covering no pixels — whitespace.
    Empty,
    /// `swash` produced no image at all.
    NoOutline,
}

/// The legacy rasteriser, transcribed. See this file's module doc for why it is
/// a transcription.
mod legacy {
    use super::*;

    /// `src/text_system.rs`'s `SUBPIXEL_VARIANTS_X` / `_Y`.
    const SUBPIXEL_VARIANTS_X: u8 = 4;
    const SUBPIXEL_VARIANTS_Y: u8 = 1;

    /// `CosmicTextSystemState`, reduced to the fields the raster path reads.
    pub struct TextSystem {
        pub font_system: FontSystem,
        swash_cache: SwashCache,
        loaded_fonts: Vec<LoadedFont>,
    }

    pub struct LoadedFont {
        font: Arc<cosmic_text::Font>,
        weight: fontdb::Weight,
    }

    /// `src/text_system.rs`'s `RenderGlyphParams`.
    #[derive(Copy, Clone, Debug)]
    pub struct RenderGlyphParams {
        pub font_index: usize,
        pub glyph_id: u32,
        pub font_size: f32,
        pub subpixel_variant: [u8; 2],
        pub scale_factor: f32,
        pub is_emoji: bool,
    }

    impl TextSystem {
        pub fn new() -> Self {
            Self {
                font_system: test_fonts::font_system(),
                swash_cache: SwashCache::new(),
                loaded_fonts: Vec::new(),
            }
        }

        /// `load_family`, minus the platform system-font substitution and the
        /// Windows postscript-name allowlist — neither reaches a database
        /// holding one non-Windows face. The `charmap().map('m')` reject is
        /// kept, because it is what decides whether a face loads at all.
        pub fn load_family(&mut self, name: &str) -> Vec<usize> {
            let families: Vec<(fontdb::ID, fontdb::Weight)> = self
                .font_system
                .db()
                .faces()
                .filter(|face| face.families.iter().any(|family| family.0 == name))
                .map(|face| (face.id, face.weight))
                .collect();

            let mut loaded = Vec::new();
            for (database_id, weight) in families {
                let Some(font) = self.font_system.get_font(database_id, weight) else {
                    continue;
                };
                if font.as_swash().charmap().map('m') == 0 {
                    self.font_system.db_mut().remove_face(font.id());
                    continue;
                }
                loaded.push(self.loaded_fonts.len());
                self.loaded_fonts.push(LoadedFont { font, weight });
            }
            loaded
        }

        fn image(&mut self, params: &RenderGlyphParams) -> Option<cosmic_text::SwashImage> {
            let loaded_font = self.loaded_fonts.get(params.font_index)?;
            let font = loaded_font.font.clone();
            let weight = loaded_font.weight;
            let subpixel_shift = [
                params.subpixel_variant[0] as f32 / SUBPIXEL_VARIANTS_X as f32
                    / params.scale_factor,
                params.subpixel_variant[1] as f32 / SUBPIXEL_VARIANTS_Y as f32
                    / params.scale_factor,
            ];
            self.swash_cache
                .get_image(
                    &mut self.font_system,
                    CacheKey::new(
                        font.id(),
                        params.glyph_id as u16,
                        params.font_size * params.scale_factor,
                        (subpixel_shift[0], subpixel_shift[1].trunc()),
                        weight,
                        CacheKeyFlags::empty(),
                    )
                    .0,
                )
                .clone()
        }

        /// `raster_bounds` and `rasterize_glyph`, in the order
        /// `TextSystem::rasterize_glyph` calls them: bounds first, then the
        /// bytes, with the empty-bounds bail in between.
        pub fn rasterize_glyph(&mut self, params: &RenderGlyphParams) -> Raster {
            let Some(image) = self.image(params) else {
                return Raster::NoOutline;
            };
            // `raster_bounds`.
            let bounds_origin = [image.placement.left as f32, -image.placement.top as f32];
            let bounds_size = [image.placement.width, image.placement.height];

            // `rasterize_glyph`'s first line.
            if bounds_size[0] == 0 || bounds_size[1] == 0 {
                return Raster::Empty;
            }

            let data = if params.is_emoji {
                match image.content {
                    SwashContent::Color => image.data,
                    SwashContent::Mask => image
                        .data
                        .into_iter()
                        .flat_map(|alpha| [255, 255, 255, alpha])
                        .collect(),
                    SwashContent::SubpixelMask => image
                        .data
                        .chunks_exact(4)
                        .flat_map(|pixel| {
                            let alpha = (pixel[0] as f32 * 0.2126
                                + pixel[1] as f32 * 0.7152
                                + pixel[2] as f32 * 0.0722)
                                as u8;
                            [255, 255, 255, alpha]
                        })
                        .collect(),
                }
            } else {
                match image.content {
                    SwashContent::Mask => image.data,
                    SwashContent::SubpixelMask => image
                        .data
                        .chunks_exact(4)
                        .map(|pixel| {
                            (pixel[0] as f32 * 0.2126
                                + pixel[1] as f32 * 0.7152
                                + pixel[2] as f32 * 0.0722) as u8
                        })
                        .collect(),
                    SwashContent::Color => {
                        image.data.chunks_exact(4).map(|pixel| pixel[3]).collect()
                    }
                }
            };

            Raster::Bitmap {
                size: bounds_size,
                bearing: bounds_origin,
                texels: data,
            }
        }
    }
}

/// The glyphs the differential runs over: everything a Latin UI actually draws.
///
/// Taken through real shaping rather than written down, because glyph indices
/// are font-local and nothing outside the face knows them — and because a hand-
/// written list would silently stop covering the face if the face changed.
const SAMPLE_TEXT: &str =
    "Hamburgefonstiv HAMBURGEFONSTIV 0123456789 ,.;:!?'\"()[]{}-_+=@#$%&*/\\<>|~^`";

fn shaped_glyph_ids(shaper: &mut TextShaper, font_id: wgpui_text::shaping::FontId) -> Vec<u32> {
    let text = SharedString::from(SAMPLE_TEXT);
    let line = shaper
        .shape_line(&text, 16.0, &[FontRun::new(text.len(), font_id)])
        .expect("the embedded face shapes ASCII");
    let mut ids: Vec<u32> = line
        .runs
        .iter()
        .flat_map(|run| run.glyphs.iter().map(|glyph| glyph.id.0))
        .collect();
    ids.sort_unstable();
    ids.dedup();
    ids
}

fn ours(rasterizer: &mut GlyphRasterizer, shaper: &mut TextShaper, key: GlyphRasterKey) -> Raster {
    match rasterizer.rasterize(shaper, key) {
        Ok(glyph) => Raster::Bitmap {
            size: glyph.size,
            bearing: glyph.bearing,
            texels: glyph.texels,
        },
        Err(RasterError::EmptyRaster) => Raster::Empty,
        Err(RasterError::NoOutline) => Raster::NoOutline,
        Err(other) => panic!("the differential's own inputs must be valid keys: {other}"),
    }
}

/// One row of the differential: both sides, for one request.
fn compare(
    rasterizer: &mut GlyphRasterizer,
    shaper: &mut TextShaper,
    reference: &mut legacy::TextSystem,
    reference_index: usize,
    glyph: u32,
    font_size: f32,
    scale_factor: f32,
    variant: u8,
    kind: AtlasKind,
) -> (Raster, Raster) {
    let key = GlyphRasterKey {
        font: 0,
        glyph,
        // `crate::patch` scales before it builds the key; the legacy scales at
        // the call. Same product, and this is where the two conventions meet.
        font_size_bits: (font_size * scale_factor).to_bits(),
        subpixel: [variant, 0],
        scale_factor_bits: scale_factor.to_bits(),
        kind,
    };
    let mine = ours(rasterizer, shaper, key);
    let theirs = reference.rasterize_glyph(&legacy::RenderGlyphParams {
        font_index: reference_index,
        glyph_id: glyph,
        font_size,
        subpixel_variant: [variant, 0],
        scale_factor,
        is_emoji: kind == AtlasKind::Polychrome,
    });
    (mine, theirs)
}

struct Sides {
    shaper: TextShaper,
    rasterizer: GlyphRasterizer,
    reference: legacy::TextSystem,
    reference_index: usize,
    glyphs: Vec<u32>,
}

fn both_sides() -> Sides {
    let mut shaper = test_fonts::shaper();
    let font_id = shaper
        .resolve_font(&font(test_fonts::FAMILY))
        .expect("the embedded face resolves");
    // The new side numbers its faces from zero and this is its first, so
    // `GlyphRasterKey::font` is 0 below. Asserted rather than assumed, because a
    // silent mismatch here would rasterise from a face the test never named.
    assert_eq!(font_id.0, 0);

    let mut reference = legacy::TextSystem::new();
    let loaded = reference.load_family(test_fonts::FAMILY);
    assert_eq!(
        loaded,
        vec![0],
        "both sides must resolve the one embedded face, or the differential compares two fonts"
    );

    let glyphs = shaped_glyph_ids(&mut shaper, font_id);
    assert!(
        glyphs.len() > 30,
        "a differential over a handful of glyphs is not a differential: got {}",
        glyphs.len()
    );

    Sides {
        shaper,
        rasterizer: GlyphRasterizer::new(),
        reference,
        reference_index: 0,
        glyphs,
    }
}

const FONT_SIZES: [f32; 3] = [12.0, 16.0, 24.0];
const SCALE_FACTORS: [f32; 3] = [1.0, 1.5, 2.0];

/// The gate.
#[test]
fn every_rasterised_glyph_matches_the_legacy_path_byte_for_byte() {
    let mut sides = both_sides();

    let mut compared = 0usize;
    let mut with_ink = 0usize;
    let mut blank = 0usize;
    let mut disagreements = Vec::new();

    for &glyph in &sides.glyphs {
        for &font_size in &FONT_SIZES {
            for &scale_factor in &SCALE_FACTORS {
                for variant in 0..4u8 {
                    let (mine, theirs) = compare(
                        &mut sides.rasterizer,
                        &mut sides.shaper,
                        &mut sides.reference,
                        sides.reference_index,
                        glyph,
                        font_size,
                        scale_factor,
                        variant,
                        AtlasKind::Monochrome,
                    );
                    compared += 1;
                    match &theirs {
                        Raster::Bitmap { .. } => with_ink += 1,
                        _ => blank += 1,
                    }
                    if mine != theirs {
                        disagreements.push(format!(
                            "glyph {glyph} at {font_size}px x{scale_factor} variant {variant}: \
                             ours {} vs legacy {}",
                            describe(&mine),
                            describe(&theirs)
                        ));
                    }
                }
            }
        }
    }

    println!(
        "differential: {compared} requests compared, {with_ink} produced a bitmap, \
         {blank} declined"
    );
    assert!(
        disagreements.is_empty(),
        "{} of {compared} requests disagreed:\n{}",
        disagreements.len(),
        disagreements
            .iter()
            .take(10)
            .cloned()
            .collect::<Vec<String>>()
            .join("\n")
    );
    // A differential where nothing had ink would agree perfectly and prove
    // nothing. Both halves are asserted so neither can quietly become the whole.
    assert!(
        with_ink > compared / 2,
        "most of the sample must actually rasterise: {with_ink} of {compared}"
    );
    assert!(blank > 0, "the sample must include the space glyph's decline");
}

/// The colour arm: the same glyphs requested out of the polychrome atlas, which
/// is the legacy `is_emoji: true` path.
///
/// The embedded face has no colour glyphs, so what this exercises is the
/// `Mask` → RGBA widening — which is the arm a real emoji font reaches for any
/// codepoint it has no colour form for, and the one a monochrome-only test
/// machine can still check.
#[test]
fn the_colour_arm_matches_the_legacy_emoji_path() {
    let mut sides = both_sides();
    let mut compared = 0usize;
    let mut disagreements = 0usize;

    for &glyph in &sides.glyphs {
        for &font_size in &FONT_SIZES {
            let (mine, theirs) = compare(
                &mut sides.rasterizer,
                &mut sides.shaper,
                &mut sides.reference,
                sides.reference_index,
                glyph,
                font_size,
                1.0,
                0,
                AtlasKind::Polychrome,
            );
            compared += 1;
            if mine != theirs {
                disagreements += 1;
            }
            if let Raster::Bitmap { size, texels, .. } = &theirs {
                assert_eq!(
                    texels.len(),
                    size[0] as usize * size[1] as usize * 4,
                    "the legacy emoji path is four bytes per texel"
                );
            }
        }
    }
    assert_eq!(disagreements, 0, "{disagreements} of {compared} disagreed");
    assert!(compared > 30);
}

/// The gate can fail.
///
/// Phase 5's own report makes the point: a gate that passes on the first run has
/// proved a number is what you expected; a gate that also fails when the
/// mechanism is broken has proved the number is *about* the mechanism. The
/// break used here is the one real decision this port makes — the scale factor
/// in the sub-pixel shift — because it is exactly the thing that would have been
/// wrong had `GlyphRasterKey` not gained a `scale_factor_bits` field, and it
/// would have been invisible at 1x.
#[test]
fn dropping_the_scale_factor_from_the_sub_pixel_shift_breaks_the_agreement() {
    let mut sides = both_sides();
    let mut differed = 0usize;
    let mut checked = 0usize;

    for &glyph in &sides.glyphs {
        for variant in 1..4u8 {
            // The correct request: 16px logical content at 2x.
            let (correct, reference) = compare(
                &mut sides.rasterizer,
                &mut sides.shaper,
                &mut sides.reference,
                sides.reference_index,
                glyph,
                16.0,
                2.0,
                variant,
                AtlasKind::Monochrome,
            );
            assert_eq!(correct, reference, "the correct arm must still agree");

            // The mistake: same device size, same variant, scale factor claimed
            // to be 1 — which is what a key without `scale_factor_bits` would
            // have forced, since 16px at 2x and 32px at 1x are one key.
            let mistaken = ours(
                &mut sides.rasterizer,
                &mut sides.shaper,
                GlyphRasterKey {
                    font: 0,
                    glyph,
                    font_size_bits: 32.0f32.to_bits(),
                    subpixel: [variant, 0],
                    scale_factor_bits: 1.0f32.to_bits(),
                    kind: AtlasKind::Monochrome,
                },
            );
            checked += 1;
            if mistaken != reference {
                differed += 1;
            }
        }
    }

    println!("falsification: {differed} of {checked} requests differ once the scale is dropped");
    assert!(
        differed > 0,
        "if collapsing the scale factor changed nothing, the field would be dead weight and \
         this gate would be measuring nothing"
    );
}

/// The other direction of the same claim: the differential is sensitive to the
/// bytes, not only to the size.
#[test]
fn perturbing_a_single_texel_is_caught() {
    let mut sides = both_sides();
    let glyph = *sides.glyphs.last().expect("the sample has glyphs");
    let (mine, theirs) = compare(
        &mut sides.rasterizer,
        &mut sides.shaper,
        &mut sides.reference,
        sides.reference_index,
        glyph,
        24.0,
        1.0,
        0,
        AtlasKind::Monochrome,
    );
    assert_eq!(mine, theirs);
    let Raster::Bitmap {
        size,
        bearing,
        mut texels,
    } = mine
    else {
        panic!("the sample's last glyph must have ink for this check to mean anything");
    };
    let first = texels.first().copied().unwrap_or(0);
    texels[0] = first.wrapping_add(1);
    assert_ne!(
        Raster::Bitmap {
            size,
            bearing,
            texels
        },
        theirs,
        "comparison must reach the texels, not stop at the dimensions"
    );
}

fn describe(raster: &Raster) -> String {
    match raster {
        Raster::Bitmap {
            size,
            bearing,
            texels,
        } => format!(
            "{}x{} bearing {:?} ({} bytes, checksum {})",
            size[0],
            size[1],
            bearing,
            texels.len(),
            texels.iter().map(|byte| u64::from(*byte)).sum::<u64>()
        ),
        Raster::Empty => "empty".to_owned(),
        Raster::NoOutline => "no outline".to_owned(),
    }
}
