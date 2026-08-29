//! Phase 5.6's gate: text that is shaped, reconciled, rasterised and uploaded
//! is now also **drawn**, and the pixels it draws are the atlas texels it
//! claimed.
//!
//! `docs/gpu-native-architecture.md` §9's risk row this closes ends: "nothing
//! yet draws it: there is no sprite render pipeline and no sprite primitive kind
//! consuming those atlas tiles." `tests/glyph_atlas_upload.rs` proved one level
//! down — "every glyph claiming a tile finds ink in it", in the page. This file
//! is the level up: **every glyph drawn on screen has that tile's own texels at
//! its own screen position**, read back off a real framebuffer.
//!
//! # Why the comparison can be exact, and is
//!
//! White text on black through the pipeline's straight-alpha `over` blend
//! reduces to an identity. The shader emits `rgb = 1` and
//! `alpha = colour.a * coverage`; the blend computes
//! `src.rgb * srcAlpha + dst.rgb * (1 - srcAlpha)`, which with `dst = 0` and
//! `src.rgb = 1` is just `srcAlpha`, which is `coverage`, which is the atlas
//! texel divided by 255. Written back to `Rgba8Unorm` it is the texel byte
//! again. So a rendered pixel is not merely *similar* to its atlas texel — it is
//! numerically the same byte, and the test asserts equality rather than a
//! threshold. Anything else is a bug, in the same spirit as
//! `tests/indirect_draw.rs`' bit-exact mode comparison.
//!
//! This works because `mono_sprites.wgsl` blits one texel to one pixel — see
//! that shader's header for why a glyph quad is a 1:1 blit and not a filtered
//! sample.
//!
//! # What the test has to do to the shaped runs, and why that is disclosed
//!
//! A 1:1 blit is only texel-exact when the quad's corners are whole pixels, and
//! `wgpui_text::patch::glyph_runs` does not floor the pen position — the legacy
//! `Window::paint_glyph` does, because the sub-pixel variant already carries the
//! fraction. So the exact test rounds each glyph's position, and
//! [`text_draws_without_being_rounded_first`] then renders the *unmodified*
//! conversion output to show the ordinary path paints too. The flooring belongs
//! in `wgpui-text`, not here; see `docs/phase-5.6-results.md`.
//!
//! # If there is no adapter
//!
//! Reports and returns, per Phase 0's standard.

use wgpui_core::geometry::Rect;
use wgpui_core::patch::RecordKey;
use wgpui_core::patch::apply::{ScenePatch, apply};
use wgpui_core::patch::primitive::{Glyph, GlyphRun};
use wgpui_core::scene::atlas::{AtlasKind, GlyphRasterKey, RasterizedGlyph};
use wgpui_core::scene::layer::{BoundaryId, LayerKey};
use wgpui_core::scene::Scene;
use wgpui_wgpu::render::atlas::{AtlasTileSource, GlyphAtlas, TilePlacement};
use wgpui_wgpu::render::atlas_upload::AtlasTextures;
use wgpui_wgpu::render::device::{ComputeContext, context_or_report};
use wgpui_wgpu::render::draw::DrawMode;
use wgpui_wgpu::render::frame::{Dirty, FrameInput, FrameOutput, FrameRenderer, OffscreenTarget};
use wgpui_text::patch::{RunPlacement, glyph_runs};
use wgpui_text::raster::GlyphRasterizer;
use wgpui_text::shaping::{FontRun, SharedString, font};
use wgpui_text::test_fonts;

const WIDTH: u32 = 512;
const HEIGHT: u32 = 128;

/// A scene holding exactly the given runs in one layer.
fn scene_with(runs: &[GlyphRun]) -> Scene {
    let mut scene = Scene::new();
    let layer = scene.layer(LayerKey::untiled(BoundaryId::from_raw(1)));
    let mut patch = ScenePatch::new();
    for (index, run) in runs.iter().enumerate() {
        patch.glyph_runs.append(
            layer,
            RecordKey::from_raw(index as u64 + 1),
            u32::try_from(index).unwrap_or(u32::MAX),
            run.clone(),
        );
    }
    apply(&mut scene, &patch).expect("seeding one layer with glyph runs must apply");
    scene
}

fn input<'a>(
    scene: &'a Scene,
    atlas: &'a AtlasTextures,
    mode: DrawMode,
) -> FrameInput<'a> {
    FrameInput {
        scene,
        clip: Rect::from_origin_size([0.0, 0.0], [WIDTH as f32, HEIGHT as f32]),
        poison: &[],
        dirty: Dirty::All,
        uploads: &[],
        composites: &[],
        registry: None,
        atlas: Some(atlas),
        viewport: [WIDTH as f32, HEIGHT as f32],
        mode,
    }
}

fn render(
    context: &ComputeContext,
    renderer: &mut FrameRenderer,
    target: &OffscreenTarget,
    input: &FrameInput<'_>,
) -> (FrameOutput, Vec<u8>) {
    let output = renderer
        .render(&context.device, &context.queue, input, target)
        .expect("a frame must render");
    let pixels = target
        .read_pixels(&context.device, &context.queue)
        .expect("reading the target back must succeed");
    (output, pixels)
}

fn pixel(pixels: &[u8], x: u32, y: u32) -> [u8; 4] {
    let index = ((y * WIDTH + x) * 4) as usize;
    match pixels.get(index..index + 4) {
        Some(bytes) => [bytes[0], bytes[1], bytes[2], bytes[3]],
        None => [0, 0, 0, 0],
    }
}

fn painted_pixels(pixels: &[u8]) -> usize {
    pixels
        .chunks_exact(4)
        .filter(|pixel| pixel[0] != 0 || pixel[1] != 0 || pixel[2] != 0)
        .count()
}

/// Shape a real line, rasterise it through the real path, and hand back the
/// runs beside the atlas that now holds their texels.
fn shaped_runs(page_size: u32, text: &str, origin: [f32; 2]) -> (Vec<GlyphRun>, GlyphAtlas) {
    let mut shaper = test_fonts::shaper();
    let font_id = shaper
        .resolve_font(&font(test_fonts::FAMILY))
        .expect("the embedded face resolves");
    let text = SharedString::from(text);
    let line = shaper
        .shape_line(&text, 24.0, &[FontRun::new(text.len(), font_id)])
        .expect("shaping must succeed");

    let mut atlas = GlyphAtlas::new(page_size);
    let mut rasterizer = GlyphRasterizer::new();
    let runs = {
        let mut source = AtlasTileSource::new(&mut atlas, |key| {
            rasterizer.rasterize(&mut shaper, key).ok()
        });
        glyph_runs(
            &line,
            RunPlacement {
                origin,
                // White, so the blend reduces to the identity this file's doc
                // sets out and the comparison can be exact.
                color: [1.0, 1.0, 1.0, 1.0],
                scale_factor: 1.0,
            },
            &mut source,
        )
        .0
    };
    (runs, atlas)
}

/// How many glyph rectangles cover each pixel of the framebuffer.
///
/// Adjacent glyphs' rasters genuinely overlap — a real line of 24px text in this
/// face has pairs whose boxes share a column, because a raster is wider than its
/// advance wherever a letter leans. Where two overlap the framebuffer holds the
/// *blend* of both, which is correct and is not what one tile's texels say, so
/// the exact comparisons below run only over the pixels exactly one glyph
/// claims. That is a restriction on what can be checked, not a tolerance: on
/// those pixels equality still has to be exact.
fn glyph_coverage(runs: &[GlyphRun]) -> Vec<u8> {
    let mut coverage = vec![0u8; (WIDTH * HEIGHT) as usize];
    for run in runs {
        for glyph in &run.glyphs {
            if glyph.atlas_tile.is_none() {
                continue;
            }
            for row in 0..glyph.atlas_size[1] as u32 {
                for column in 0..glyph.atlas_size[0] as u32 {
                    let x = glyph.position[0] as u32 + column;
                    let y = glyph.position[1] as u32 + row;
                    if x >= WIDTH || y >= HEIGHT {
                        continue;
                    }
                    if let Some(count) = coverage.get_mut((y * WIDTH + x) as usize) {
                        *count = count.saturating_add(1);
                    }
                }
            }
        }
    }
    coverage
}

/// What one exact comparison of every glyph against its own tile found.
#[derive(Default, Debug)]
struct TexelComparison {
    compared: usize,
    inked: usize,
    shared: usize,
}

/// Compare every glyph's tile texels against the pixels it should have become.
///
/// Panics with the offending glyph and pixel on the first disagreement, which is
/// the whole of the phase's gate — see this file's doc for why equality rather
/// than a threshold is the right assertion.
fn compare_glyph_texels(
    runs: &[GlyphRun],
    atlas: &GlyphAtlas,
    pixels: &[u8],
) -> TexelComparison {
    let coverage = glyph_coverage(runs);
    let mut result = TexelComparison::default();
    for run in runs {
        for glyph in &run.glyphs {
            if glyph.atlas_tile.is_none() {
                continue;
            }
            let texels = atlas
                .tile_texels(TilePlacement {
                    tile: glyph.atlas_tile,
                    kind: AtlasKind::Monochrome,
                    origin: glyph.atlas_origin,
                    size: glyph.atlas_size,
                    bearing: [0.0, 0.0],
                })
                .expect("a referenced tile is resident");
            let width = glyph.atlas_size[0] as u32;
            for (index, expected) in texels.iter().enumerate() {
                let x = glyph.position[0] as u32 + index as u32 % width;
                let y = glyph.position[1] as u32 + index as u32 / width;
                if x >= WIDTH || y >= HEIGHT {
                    continue;
                }
                if coverage.get((y * WIDTH + x) as usize).copied().unwrap_or(0) != 1 {
                    result.shared += 1;
                    continue;
                }
                let found = pixel(pixels, x, y);
                assert_eq!(
                    [found[0], found[1], found[2]],
                    [*expected; 3],
                    "glyph {} on page {:?} should have put texel {index} on screen \
                     at ({x}, {y}) as {expected}, found {found:?}",
                    glyph.glyph_id,
                    glyph.atlas_tile.page()
                );
                // The target's alpha is *not* the coverage, and asserting that
                // it were would be asserting the wrong thing: the pass clears to
                // opaque black and the alpha blend is `One`/`OneMinusSrcAlpha`,
                // so `a = srcAlpha + 1 - srcAlpha` is 1 whatever the coverage
                // was. A framebuffer that keeps its opacity under text is the
                // correct outcome; it is stated because "alpha equals coverage"
                // is the plausible wrong expectation.
                assert_eq!(found[3], 255, "the target must stay opaque");
                result.compared += 1;
                if *expected > 0 {
                    result.inked += 1;
                }
            }
        }
    }
    result
}

/// Move every glyph onto a whole pixel, so the 1:1 blit is texel-exact.
fn rounded(runs: &[GlyphRun]) -> Vec<GlyphRun> {
    runs.iter()
        .map(|run| GlyphRun {
            color: run.color,
            glyphs: run
                .glyphs
                .iter()
                .map(|glyph| Glyph {
                    position: [glyph.position[0].round(), glyph.position[1].round()],
                    ..*glyph
                })
                .collect(),
        })
        .collect()
}

/// **The phase's gate.** Every glyph's own atlas texels land at the glyph's own
/// screen position, byte for byte.
#[test]
fn every_glyph_draws_its_own_tile_texels_at_its_own_position() {
    let Some(context) = context_or_report("glyph_sprite_texels") else {
        return;
    };

    let (runs, mut atlas) = shaped_runs(256, "Hamburgefonstiv 0123", [16.0, 80.0]);
    let runs = rounded(&runs);
    let mut textures = AtlasTextures::for_atlas(&atlas);
    let upload = textures.sync(&context.device, &context.queue, &mut atlas);
    assert_eq!(upload.skipped, 0);
    assert!(upload.rectangles > 10, "a real line of text is many rasters");

    let scene = scene_with(&runs);
    let mut renderer = FrameRenderer::new(&context.device);
    let target = OffscreenTarget::new(&context.device, WIDTH, HEIGHT);
    let mode = DrawMode::best_available(context.indirect);
    let (output, pixels) = render(
        &context,
        &mut renderer,
        &target,
        &input(&scene, &textures, mode),
    );

    println!(
        "glyph_sprite_texels [{}]: {} slots, {} glyph draws over {} atlas page(s), \
         {} painted pixels, instances known to CPU: {:?}",
        mode.name(),
        output.stats.slots_visited,
        output.stats.sprite_draws_issued,
        output.stats.atlas_pages_bound,
        painted_pixels(&pixels),
        output.stats.instances_known_to_cpu,
    );
    assert!(output.stats.sprite_draws_issued > 0, "no glyph draw was issued");
    assert!(
        painted_pixels(&pixels) > 500,
        "twenty glyphs of 24px text painted {} pixels, which is not text",
        painted_pixels(&pixels)
    );

    // The comparison itself: every glyph's tile, texel by texel, against the
    // pixels it should have become.
    let comparison = compare_glyph_texels(&runs, &atlas, &pixels);
    println!("  {comparison:?}");
    let (compared, inked) = (comparison.compared, comparison.inked);
    assert!(
        compared > 2_000,
        "only {compared} texels were compared, which does not prove much"
    );
    assert!(
        inked * 4 > compared,
        "a comparison that is mostly blank texels would pass on an empty \
         framebuffer: {inked} inked of {compared}"
    );
}

/// The same line, drawn exactly as `wgpui_text::patch::glyph_runs` produced it,
/// with nothing rounded.
///
/// The gate above doctors the positions to make an exact comparison possible;
/// this one shows the undoctored path paints in the right places, to within the
/// one pixel that flooring would remove.
#[test]
fn text_draws_without_being_rounded_first() {
    let Some(context) = context_or_report("glyph_sprite_unrounded") else {
        return;
    };

    let (runs, mut atlas) = shaped_runs(256, "Hamburgefonstiv 0123", [16.0, 80.0]);
    let mut textures = AtlasTextures::for_atlas(&atlas);
    textures.sync(&context.device, &context.queue, &mut atlas);
    let scene = scene_with(&runs);
    let mut renderer = FrameRenderer::new(&context.device);
    let target = OffscreenTarget::new(&context.device, WIDTH, HEIGHT);
    let (_, pixels) = render(
        &context,
        &mut renderer,
        &target,
        &input(&scene, &textures, DrawMode::best_available(context.indirect)),
    );

    let mut found_ink = 0usize;
    let mut expected_ink = 0usize;
    for run in &runs {
        for glyph in &run.glyphs {
            if glyph.atlas_tile.is_none() {
                continue;
            }
            expected_ink += 1;
            // The glyph's box, grown by one pixel on every side: a quad whose
            // corners are fractional covers at most one more pixel than one
            // whose corners are whole.
            let left = (glyph.position[0].floor() as i64 - 1).max(0) as u32;
            let top = (glyph.position[1].floor() as i64 - 1).max(0) as u32;
            let right = ((glyph.position[0] + glyph.atlas_size[0]).ceil() as u32 + 1).min(WIDTH);
            let bottom = ((glyph.position[1] + glyph.atlas_size[1]).ceil() as u32 + 1).min(HEIGHT);
            let mut painted = false;
            for y in top..bottom {
                for x in left..right {
                    if pixel(&pixels, x, y)[0] > 0 {
                        painted = true;
                    }
                }
            }
            if painted {
                found_ink += 1;
            }
        }
    }
    println!(
        "glyph_sprite_unrounded: {found_ink} of {expected_ink} inked glyphs painted \
         inside their own box"
    );
    assert!(expected_ink > 15, "the line must have real glyphs in it");
    assert_eq!(
        found_ink, expected_ink,
        "every glyph that claims a tile must put ink on the screen inside its \
         own bounding box — the sentence §9's risk row says was not true"
    );
}

/// Glyphs spread over several atlas pages all draw, each exactly once.
///
/// The page loop's whole reason to exist. A 64-texel page holds only a handful
/// of 24px glyphs, so this line genuinely spans pages rather than hoping to.
#[test]
fn glyphs_on_several_atlas_pages_all_draw_and_none_draws_twice() {
    let Some(context) = context_or_report("glyph_sprite_pages") else {
        return;
    };

    let (runs, mut atlas) = shaped_runs(64, "multipage glyph atlas spanning", [8.0, 40.0]);
    let runs = rounded(&runs);
    let mut pages: Vec<u32> = runs
        .iter()
        .flat_map(|run| run.atlas_tiles())
        .filter_map(|tile| tile.page())
        .collect();
    pages.sort_unstable();
    pages.dedup();
    assert!(
        pages.len() > 1,
        "a 64-texel page must not have held the whole line: {pages:?}"
    );

    let mut textures = AtlasTextures::for_atlas(&atlas);
    textures.sync(&context.device, &context.queue, &mut atlas);
    assert_eq!(
        textures.pages_of_kind(AtlasKind::Monochrome).len(),
        pages.len(),
        "every page the glyphs reference must have a texture"
    );

    let scene = scene_with(&runs);
    let mut renderer = FrameRenderer::new(&context.device);
    let target = OffscreenTarget::new(&context.device, WIDTH, HEIGHT);
    let (output, pixels) = render(
        &context,
        &mut renderer,
        &target,
        &input(&scene, &textures, DrawMode::best_available(context.indirect)),
    );
    assert_eq!(
        output.stats.atlas_pages_bound as usize,
        pages.len(),
        "one bind group per live monochrome page"
    );

    // Each glyph is checked against its own tile's texels, exactly as the gate
    // does. A glyph drawn by the wrong page's pass would read another glyph's
    // texels and disagree; a glyph drawn twice would blend over itself and come
    // out brighter than its own coverage.
    let comparison = compare_glyph_texels(&runs, &atlas, &pixels);
    println!(
        "glyph_sprite_pages: {} pages bound, {comparison:?}",
        output.stats.atlas_pages_bound
    );
    assert!(comparison.compared > 500);
    assert!(comparison.inked * 4 > comparison.compared);
}

/// Every available draw mode draws the same text.
///
/// Phase 4's discipline applied to the new pipeline: a gate about how cheaply
/// the CPU issues draws is worth nothing if the modes disagree about the
/// picture, and the glyph pass repeats its sequence per page in every one of
/// them.
#[test]
fn every_draw_mode_draws_the_same_text() {
    let Some(context) = context_or_report("glyph_sprite_modes") else {
        return;
    };

    let (runs, mut atlas) = shaped_runs(128, "every mode agrees", [12.0, 60.0]);
    let runs = rounded(&runs);
    let mut textures = AtlasTextures::for_atlas(&atlas);
    textures.sync(&context.device, &context.queue, &mut atlas);
    let scene = scene_with(&runs);
    let mut renderer = FrameRenderer::new(&context.device);
    let target = OffscreenTarget::new(&context.device, WIDTH, HEIGHT);

    let mut reference: Option<Vec<u8>> = None;
    for mode in DrawMode::ALL
        .into_iter()
        .filter(|mode| mode.is_available(context.indirect))
    {
        let (output, pixels) = render(
            &context,
            &mut renderer,
            &target,
            &input(&scene, &textures, mode),
        );
        assert!(
            painted_pixels(&pixels) > 200,
            "{} painted almost nothing, so comparing it proves nothing",
            mode.name()
        );
        match &reference {
            None => {
                println!(
                    "  reference {}: {} painted pixels, {} glyph draws",
                    mode.name(),
                    painted_pixels(&pixels),
                    output.stats.sprite_draws_issued
                );
                reference = Some(pixels);
            }
            Some(expected) => {
                assert!(
                    *expected == pixels,
                    "{} drew different text from the reference mode",
                    mode.name()
                );
            }
        }
    }
}

/// A glyph with no raster holds its slot and paints nothing, and a frame with no
/// atlas at all renders rather than failing.
///
/// Both are ordinary states, not errors: whitespace shapes to a positioned
/// glyph with no coverage (`patch/primitive.rs`), and a window whose text has
/// not been rasterised yet is a window that has not drawn its text yet.
#[test]
fn a_blank_glyph_and_a_missing_atlas_are_both_ordinary() {
    let Some(context) = context_or_report("glyph_sprite_blanks") else {
        return;
    };

    let (runs, mut atlas) = shaped_runs(256, "spaced  out  text", [16.0, 80.0]);
    let runs = rounded(&runs);
    let blanks = runs
        .iter()
        .flat_map(|run| run.glyphs.iter())
        .filter(|glyph| glyph.atlas_tile.is_none())
        .count();
    assert!(blanks > 0, "the spaces must still hold their slots");

    let mut textures = AtlasTextures::for_atlas(&atlas);
    textures.sync(&context.device, &context.queue, &mut atlas);
    let scene = scene_with(&runs);
    let mut renderer = FrameRenderer::new(&context.device);
    let target = OffscreenTarget::new(&context.device, WIDTH, HEIGHT);
    let mode = DrawMode::best_available(context.indirect);
    let (with_atlas, painted) = render(
        &context,
        &mut renderer,
        &target,
        &input(&scene, &textures, mode),
    );
    assert!(painted_pixels(&painted) > 200);

    // Each blank glyph's own box: a blank glyph has zero size, so "its box" is
    // the pen position, and nothing may have been written there by the blank
    // itself. Checking the whole framebuffer is empty would be wrong — its
    // neighbours legitimately paint.
    for run in &runs {
        for glyph in &run.glyphs {
            if !glyph.atlas_tile.is_none() {
                continue;
            }
            assert_eq!(
                glyph.atlas_size,
                [0.0, 0.0],
                "a glyph with no tile must have no raster extent either"
            );
        }
    }

    // The same scene with no atlas: no page to bind, so no glyph pass, and no
    // error.
    let empty = AtlasTextures::new(256);
    let (without_atlas, blank) = render(
        &context,
        &mut renderer,
        &target,
        &input(&scene, &empty, mode),
    );
    assert_eq!(
        without_atlas.stats.atlas_pages_bound, 0,
        "no page means nothing to bind"
    );
    assert_eq!(without_atlas.stats.sprite_draws_issued, 0);
    assert_eq!(
        without_atlas.stats.sprite_slots_unavailable, 2,
        "the scene's one layer contributes one slot to each of the two sprite \
         passes, and with no atlas at all both are walked and found to have no \
         texture to bind — which is `DrawStats::sprite_slots_unavailable`'s \
         whole purpose"
    );
    assert_eq!(
        with_atlas.stats.sprite_slots_unavailable, 1,
        "with a monochrome page the glyph slot becomes available and the image \
         slot does not, so the counter is measuring the difference rather than \
         always reporting the slot count. Phase 5.6 asserted 0 here because \
         there was one sprite pass; the number changed and the claim did not"
    );
    assert_eq!(
        painted_pixels(&blank),
        0,
        "with no atlas there is nothing to sample, so nothing paints"
    );
    assert_eq!(
        with_atlas.stats.slots_visited, without_atlas.stats.slots_visited,
        "§5.3's sequence is fixed: a frame that drew no text still walked it"
    );
}

/// A synthetic raster whose every texel is distinct, drawn at a known position.
///
/// The gate above proves the real path; this proves the *addressing* in the one
/// way real text cannot, because a real glyph's raster is smooth and a shifted
/// row still looks like a glyph. Here a one-texel error is a visible byte
/// mismatch.
#[test]
fn a_ramp_raster_lands_texel_for_texel_with_no_row_shift() {
    let Some(context) = context_or_report("glyph_sprite_ramp") else {
        return;
    };

    let mut atlas = GlyphAtlas::new(128);
    let key = GlyphRasterKey {
        font: 0,
        glyph: 7,
        font_size_bits: 16.0f32.to_bits(),
        subpixel: [0, 0],
        scale_factor_bits: 1.0f32.to_bits(),
        kind: AtlasKind::Monochrome,
    };
    // 13 × 11 rather than a square: a transposed index reads as a mismatch
    // instead of as a plausible picture.
    let (width, height) = (13u32, 11u32);
    let tile = atlas
        .get_or_insert_raster(
            key,
            &RasterizedGlyph {
                size: [width, height],
                kind: AtlasKind::Monochrome,
                bearing: [0.0, 0.0],
                texels: (0..width * height)
                    .map(|index| 1 + (index % 250) as u8)
                    .collect(),
            },
        )
        .expect("a 128px page holds one small raster");

    let run = GlyphRun {
        color: [1.0, 1.0, 1.0, 1.0],
        glyphs: vec![Glyph {
            position: [40.0, 24.0],
            atlas_origin: tile.origin,
            atlas_size: tile.size,
            glyph_id: 7,
            atlas_tile: tile.tile,
        }],
    };
    assert_eq!(tile.size, [width as f32, height as f32]);

    let mut textures = AtlasTextures::for_atlas(&atlas);
    textures.sync(&context.device, &context.queue, &mut atlas);
    let scene = scene_with(&[run]);
    let mut renderer = FrameRenderer::new(&context.device);
    let target = OffscreenTarget::new(&context.device, WIDTH, HEIGHT);
    let (_, pixels) = render(
        &context,
        &mut renderer,
        &target,
        &input(&scene, &textures, DrawMode::best_available(context.indirect)),
    );

    for row in 0..height {
        for column in 0..width {
            let expected = 1 + ((row * width + column) % 250) as u8;
            assert_eq!(
                pixel(&pixels, 40 + column, 24 + row)[0],
                expected,
                "texel ({column}, {row}) did not land at its own pixel"
            );
        }
    }
    // And nothing was painted outside the tile: a quad sized from the wrong
    // field would spill.
    assert_eq!(
        painted_pixels(&pixels),
        (width * height) as usize,
        "the sprite must occupy exactly its raster's extent and no more"
    );
}
