//! Phase 6.2's layer-3 gate: the polychrome pipeline draws, and the pixels it
//! draws are the atlas texels it claimed.
//!
//! `tests/glyph_atlas_upload.rs` proved one level down — a decoded frame's bytes
//! reach an `Rgba8Unorm` page. This file is the level up: **every image drawn on
//! screen has that tile's own texels at its own screen position**, read back off
//! a real framebuffer, compared as equality rather than as a tolerance.
//!
//! # Why the comparison can be exact, and what it costs to keep it exact
//!
//! An opaque sprite over black through the pipeline's straight-alpha `over`
//! blend reduces to an identity. The shader emits `rgb = texel.rgb` and
//! `alpha = texel.a * opacity * coverage`; with `texel.a = 1`, `opacity = 1` and
//! no corner radius the coverage term is exactly 1 at every pixel centre, so the
//! blend computes `src.rgb * 1 + dst.rgb * 0`, which is the texel. Written back
//! to `Rgba8Unorm` it is the texel byte again.
//!
//! Two of those three conditions are the test's, not the pipeline's, and are
//! stated rather than hidden:
//!
//! - **Opaque.** A translucent texel over black composites to `rgb * a`, which
//!   is correct rendering and not equality. [`a_translucent_sprite_composites_over_the_target`]
//!   covers that case separately, against the arithmetic rather than against the
//!   texel.
//! - **Natural size.** `poly_sprites.wgsl` maps the quad's local coordinate
//!   through `size / atlas_size`; at 1:1 that is the identity and the load is a
//!   blit. [`a_scaled_sprite_samples_the_nearest_texel`] covers the other case
//!   and records what it does *not* claim — legacy interpolates there and this
//!   does not.
//!
//! # If there is no adapter
//!
//! Reports and returns, per Phase 0's standard.

use wgpui_core::geometry::Rect;
use wgpui_core::patch::RecordKey;
use wgpui_core::patch::apply::{ScenePatch, apply};
use wgpui_core::patch::primitive::PolySprite;
use wgpui_core::scene::Scene;
use wgpui_core::scene::atlas::{ImageRasterKey, RasterizedImage};
use wgpui_core::scene::layer::{BoundaryId, LayerKey};
use wgpui_wgpu::render::atlas::{GlyphAtlas, TilePlacement};
use wgpui_wgpu::render::atlas_upload::AtlasTextures;
use wgpui_wgpu::render::device::{ComputeContext, context_or_report};
use wgpui_wgpu::render::draw::DrawMode;
use wgpui_wgpu::render::frame::{Dirty, FrameInput, FrameOutput, FrameRenderer, OffscreenTarget};

const WIDTH: u32 = 256;
const HEIGHT: u32 = 128;

/// A bitmap whose every texel is distinct, so a transposed row, a swapped
/// channel or an off-by-one origin shows up as a mismatch rather than as an
/// identical-looking block.
///
/// Fully opaque, which is what makes the framebuffer comparison an equality —
/// see this file's header.
fn opaque_ramp(width: u32, height: u32) -> RasterizedImage {
    let mut texels = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            texels.extend_from_slice(&[
                (x * 7 + 3) as u8,
                (y * 11 + 5) as u8,
                (x * 3 + y * 5) as u8,
                0xFF,
            ]);
        }
    }
    RasterizedImage {
        size: [width, height],
        texels,
    }
}

fn key(source: u64) -> ImageRasterKey {
    ImageRasterKey {
        source,
        frame_index: 0,
        scale_factor_bits: 1.0f32.to_bits(),
    }
}

/// A scene holding exactly the given sprites in one layer.
fn scene_with(sprites: &[PolySprite]) -> Scene {
    let mut scene = Scene::new();
    let layer = scene.layer(LayerKey::untiled(BoundaryId::from_raw(1)));
    let mut patch = ScenePatch::new();
    for (index, sprite) in sprites.iter().enumerate() {
        patch.poly_sprites.append(
            layer,
            RecordKey::from_raw(index as u64 + 1),
            u32::try_from(index).unwrap_or(u32::MAX),
            *sprite,
        );
    }
    apply(&mut scene, &patch).expect("seeding one layer with sprites must apply");
    scene
}

/// A sprite drawn at `origin`, at the tile's own size.
fn natural(placement: TilePlacement, origin: [f32; 2]) -> PolySprite {
    PolySprite {
        origin,
        size: placement.size,
        atlas_origin: placement.origin,
        atlas_size: placement.size,
        corner_radius: 0.0,
        opacity: 1.0,
        grayscale: false,
        atlas_tile: placement.tile,
    }
}

fn input<'a>(scene: &'a Scene, atlas: &'a AtlasTextures, mode: DrawMode) -> FrameInput<'a> {
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

/// Upload one bitmap into a fresh atlas and hand back everything a frame needs.
fn atlas_with(
    context: &ComputeContext,
    image: &RasterizedImage,
    source: u64,
) -> (GlyphAtlas, AtlasTextures, TilePlacement) {
    let mut atlas = GlyphAtlas::new(256);
    let placement = atlas
        .get_or_insert_image(key(source), image)
        .expect("the bitmap fits a 256px page");
    let mut textures = AtlasTextures::for_atlas(&atlas);
    let report = textures.sync(&context.device, &context.queue, &mut atlas);
    assert_eq!(report.rectangles, 1, "the frame must actually be uploaded");
    (atlas, textures, placement)
}

/// **The gate.** Every pixel of a sprite drawn at its natural size is byte-for-
/// byte the atlas texel behind it, and nothing outside the sprite is painted.
#[test]
fn an_image_drawn_at_its_natural_size_is_its_atlas_texels_exactly() {
    let Some(context) = context_or_report("image_sprite_exact") else {
        return;
    };
    let image = opaque_ramp(37, 23);
    let (_atlas, textures, placement) = atlas_with(&context, &image, 1);

    let origin = [16.0, 9.0];
    let scene = scene_with(&[natural(placement, origin)]);
    let mut renderer = FrameRenderer::new(&context.device);
    let target = OffscreenTarget::new(&context.device, WIDTH, HEIGHT);
    let (output, pixels) = render(
        &context,
        &mut renderer,
        &target,
        &input(
            &scene,
            &textures,
            DrawMode::best_available(context.indirect),
        ),
    );

    assert_eq!(
        output.stats.atlas_pages_bound, 1,
        "one colour page is bound"
    );
    assert!(
        output.stats.sprite_draws_issued > 0,
        "no sprite draw was issued, so the comparison below would be vacuous"
    );

    let mut compared = 0usize;
    for y in 0..23u32 {
        for x in 0..37u32 {
            let source = ((y * 37 + x) * 4) as usize;
            let expected = image
                .texels
                .get(source..source + 4)
                .expect("the ramp covers every texel");
            let drawn = pixel(&pixels, origin[0] as u32 + x, origin[1] as u32 + y);
            assert_eq!(
                drawn.as_slice(),
                expected,
                "the pixel at ({x}, {y}) inside the sprite is not its atlas texel"
            );
            compared += 1;
        }
    }
    assert_eq!(compared, 37 * 23);
    assert_eq!(
        painted_pixels(&pixels),
        // The ramp's first texel is (3, 5, 0) and every texel has at least one
        // non-zero channel, so every texel of it paints and nothing else does.
        compared,
        "the sprite must paint its own rectangle and not one pixel more"
    );
    assert_eq!(
        pixel(&pixels, origin[0] as u32 - 1, origin[1] as u32),
        [0, 0, 0, 255],
        "the pixel immediately left of the sprite must be untouched"
    );
    assert_eq!(
        pixel(&pixels, origin[0] as u32 + 37, origin[1] as u32),
        [0, 0, 0, 255],
        "and the one immediately right of it"
    );
}

/// Every draw mode draws the same image.
///
/// The same claim `tests/indirect_draw.rs` makes for quads and
/// `tests/glyph_sprite_draw.rs` for text, made once more for the kind that
/// arrived after both — because a gate about one mode says nothing about the
/// three the renderer will actually pick on other hardware.
#[test]
fn every_draw_mode_draws_the_same_image() {
    let Some(context) = context_or_report("image_sprite_modes") else {
        return;
    };
    let image = opaque_ramp(31, 17);
    let (_atlas, textures, placement) = atlas_with(&context, &image, 1);
    let scene = scene_with(&[
        natural(placement, [8.0, 8.0]),
        natural(placement, [80.0, 40.0]),
    ]);

    let mut renderer = FrameRenderer::new(&context.device);
    let target = OffscreenTarget::new(&context.device, WIDTH, HEIGHT);
    let available: Vec<DrawMode> = DrawMode::ALL
        .into_iter()
        .filter(|mode| mode.is_available(context.indirect))
        .collect();
    assert!(
        available.len() >= 2,
        "at least two modes are always available"
    );

    let mut reference: Option<Vec<u8>> = None;
    for mode in available {
        let (output, pixels) = render(
            &context,
            &mut renderer,
            &target,
            &input(&scene, &textures, mode),
        );
        assert_eq!(
            painted_pixels(&pixels),
            2 * 31 * 17,
            "[{}] both sprites must paint",
            mode.name()
        );
        match &reference {
            None => reference = Some(pixels),
            Some(expected) => assert!(
                *expected == pixels,
                "[{}] drew a different picture from the reference mode ({} draws)",
                mode.name(),
                output.stats.sprite_draws_issued
            ),
        }
    }
}

/// A translucent texel composites rather than overwrites — checked against the
/// arithmetic, not against the texel.
#[test]
fn a_translucent_sprite_composites_over_the_target() {
    let Some(context) = context_or_report("image_sprite_alpha") else {
        return;
    };
    // One row of four texels: opaque white, half-alpha white, transparent, and
    // an opaque mid grey. Over black, straight-alpha `over` gives
    // `rgb * a` in each case.
    let image = RasterizedImage {
        size: [4, 1],
        texels: vec![
            255, 255, 255, 255, //
            255, 255, 255, 128, //
            255, 255, 255, 0, //
            128, 64, 32, 255,
        ],
    };
    let (_atlas, textures, placement) = atlas_with(&context, &image, 1);
    let scene = scene_with(&[natural(placement, [10.0, 10.0])]);
    let mut renderer = FrameRenderer::new(&context.device);
    let target = OffscreenTarget::new(&context.device, WIDTH, HEIGHT);
    let (_, pixels) = render(
        &context,
        &mut renderer,
        &target,
        &input(
            &scene,
            &textures,
            DrawMode::best_available(context.indirect),
        ),
    );

    assert_eq!(pixel(&pixels, 10, 10), [255, 255, 255, 255]);
    assert_eq!(
        pixel(&pixels, 12, 10),
        [0, 0, 0, 255],
        "a fully transparent texel must leave the target alone — the shader \
         discards rather than writing a zero-alpha black"
    );
    let half = pixel(&pixels, 11, 10);
    // `255 * (128/255)` is 128 exactly; allowing one unit of slack covers the
    // rounding a driver is permitted to choose at a `Rgba8Unorm` write.
    for channel in 0..3 {
        let value = i32::from(half[channel]);
        assert!(
            (value - 128).abs() <= 1,
            "a half-alpha white texel over black must composite to about 128, got {half:?}"
        );
    }
    assert_eq!(
        pixel(&pixels, 13, 10),
        [128, 64, 32, 255],
        "an opaque coloured texel is still exact"
    );
}

/// A scaled sprite uses linear filtering, while natural-size sprites retain
/// their exact integer-load path.
#[test]
fn a_scaled_sprite_interpolates_between_texels() {
    let Some(context) = context_or_report("image_sprite_scaled") else {
        return;
    };
    let image = RasterizedImage {
        size: [2, 1],
        texels: vec![200, 0, 0, 255, 0, 200, 0, 255],
    };
    let (_atlas, textures, placement) = atlas_with(&context, &image, 1);
    let scene = scene_with(&[PolySprite {
        origin: [20.0, 20.0],
        size: [4.0, 2.0],
        ..natural(placement, [20.0, 20.0])
    }]);
    let mut renderer = FrameRenderer::new(&context.device);
    let target = OffscreenTarget::new(&context.device, WIDTH, HEIGHT);
    let (_, pixels) = render(
        &context,
        &mut renderer,
        &target,
        &input(
            &scene,
            &textures,
            DrawMode::best_available(context.indirect),
        ),
    );

    let mut saw_interpolated = false;
    for x in 20..22u32 {
        for y in 20..22u32 {
            let sampled = pixel(&pixels, x, y);
            saw_interpolated |= sampled == [150, 50, 0, 255] || sampled == [50, 150, 0, 255];
            assert!(
                sampled == [200, 0, 0, 255]
                    || sampled == [0, 200, 0, 255]
                    || sampled == [150, 50, 0, 255]
                    || sampled == [50, 150, 0, 255],
                "scaled sampling must use the source texels, got {sampled:?}"
            );
        }
    }
    for x in 22..24u32 {
        for y in 20..22u32 {
            let sampled = pixel(&pixels, x, y);
            assert!(
                sampled == [200, 0, 0, 255]
                    || sampled == [0, 200, 0, 255]
                    || sampled == [150, 50, 0, 255]
                    || sampled == [50, 150, 0, 255],
                "scaled sampling must use the source texels, got {sampled:?}"
            );
        }
    }
    assert!(
        saw_interpolated,
        "scaled sampling must interpolate at an edge"
    );
}

/// Linear filtering also interpolates coverage, and a fully transparent edge
/// remains transparent instead of writing a zero-alpha colour to the target.
#[test]
fn a_scaled_sprite_interpolates_translucent_and_transparent_texels() {
    let Some(context) = context_or_report("image_sprite_scaled_alpha") else {
        return;
    };
    let image = RasterizedImage {
        size: [2, 1],
        texels: vec![255, 255, 255, 255, 255, 255, 255, 0],
    };
    let (_atlas, textures, placement) = atlas_with(&context, &image, 1);
    let scene = scene_with(&[PolySprite {
        origin: [20.0, 20.0],
        size: [4.0, 2.0],
        ..natural(placement, [20.0, 20.0])
    }]);
    let mut renderer = FrameRenderer::new(&context.device);
    let target = OffscreenTarget::new(&context.device, WIDTH, HEIGHT);
    let (_, pixels) = render(
        &context,
        &mut renderer,
        &target,
        &input(
            &scene,
            &textures,
            DrawMode::best_available(context.indirect),
        ),
    );

    assert_eq!(pixel(&pixels, 20, 20), [191, 191, 191, 255]);
    assert_eq!(pixel(&pixels, 21, 20), [64, 64, 64, 255]);
    assert_eq!(
        pixel(&pixels, 22, 20),
        [0, 0, 0, 255],
        "the transparent source edge must not paint the target"
    );
}

/// Grayscale and opacity are applied, and a corner radius rounds the rectangle.
///
/// Not a pixel-exact claim about the antialiased rim — that is a transcription
/// of the legacy signed-distance expression and the honest check on it is that
/// the corner is *cut* while the centre is not.
#[test]
fn style_reaches_the_shader() {
    let Some(context) = context_or_report("image_sprite_style") else {
        return;
    };
    let image = RasterizedImage {
        size: [16, 16],
        texels: std::iter::repeat_n([200u8, 100, 50, 255], 16 * 16)
            .flatten()
            .collect(),
    };
    let (_atlas, textures, placement) = atlas_with(&context, &image, 1);
    let mut renderer = FrameRenderer::new(&context.device);
    let target = OffscreenTarget::new(&context.device, WIDTH, HEIGHT);
    let mode = DrawMode::best_available(context.indirect);

    let plain = scene_with(&[natural(placement, [10.0, 10.0])]);
    let (_, plain_pixels) = render(
        &context,
        &mut renderer,
        &target,
        &input(&plain, &textures, mode),
    );
    assert_eq!(pixel(&plain_pixels, 18, 18), [200, 100, 50, 255]);

    let gray = scene_with(&[PolySprite {
        grayscale: true,
        ..natural(placement, [10.0, 10.0])
    }]);
    let (_, gray_pixels) = render(
        &context,
        &mut renderer,
        &target,
        &input(&gray, &textures, mode),
    );
    let luma = pixel(&gray_pixels, 18, 18);
    assert_eq!(
        luma[0], luma[1],
        "grayscale must equalise the channels, got {luma:?}"
    );
    assert_eq!(luma[1], luma[2]);
    // Rec. 709 over (200, 100, 50): 0.2126*200 + 0.7152*100 + 0.0722*50 ≈ 117.7.
    assert!(
        (i32::from(luma[0]) - 118).abs() <= 1,
        "the luma weights must be the legacy ones, got {luma:?}"
    );

    let faded = scene_with(&[PolySprite {
        opacity: 0.5,
        ..natural(placement, [10.0, 10.0])
    }]);
    let (_, faded_pixels) = render(
        &context,
        &mut renderer,
        &target,
        &input(&faded, &textures, mode),
    );
    let half = pixel(&faded_pixels, 18, 18);
    assert!(
        (i32::from(half[0]) - 100).abs() <= 1,
        "half opacity over black must halve the channel, got {half:?}"
    );

    let rounded = scene_with(&[PolySprite {
        corner_radius: 6.0,
        ..natural(placement, [10.0, 10.0])
    }]);
    let (_, rounded_pixels) = render(
        &context,
        &mut renderer,
        &target,
        &input(&rounded, &textures, mode),
    );
    assert_eq!(
        pixel(&rounded_pixels, 18, 18),
        [200, 100, 50, 255],
        "the centre of a rounded sprite is untouched"
    );
    assert_eq!(
        pixel(&rounded_pixels, 10, 10),
        [0, 0, 0, 255],
        "and its corner is cut away"
    );
    assert!(
        painted_pixels(&rounded_pixels) < painted_pixels(&plain_pixels),
        "a rounded sprite must paint strictly fewer pixels than a square one"
    );
}

/// A sprite whose image has not decoded holds its slot and draws nothing.
///
/// The `PolySprite` half of the rule a whitespace glyph follows, and the reason
/// `AtlasTileId::NONE` is in the payload at all: an image that is still loading
/// occupies its layout box and its slab slot exactly as it will once it arrives,
/// so the frame it arrives in is a value update and not an insert.
#[test]
fn a_sprite_with_no_tile_draws_nothing_and_does_not_disturb_its_neighbour() {
    let Some(context) = context_or_report("image_sprite_undecoded") else {
        return;
    };
    let image = opaque_ramp(20, 20);
    let (_atlas, textures, placement) = atlas_with(&context, &image, 1);
    let scene = scene_with(&[
        PolySprite {
            origin: [10.0, 10.0],
            size: [20.0, 20.0],
            ..PolySprite::ZERO
        },
        natural(placement, [60.0, 10.0]),
    ]);
    let mut renderer = FrameRenderer::new(&context.device);
    let target = OffscreenTarget::new(&context.device, WIDTH, HEIGHT);
    let (output, pixels) = render(
        &context,
        &mut renderer,
        &target,
        &input(
            &scene,
            &textures,
            DrawMode::best_available(context.indirect),
        ),
    );

    assert_eq!(
        pixel(&pixels, 15, 15),
        [0, 0, 0, 255],
        "the blank paints nothing"
    );
    assert_eq!(
        painted_pixels(&pixels),
        20 * 20,
        "exactly the decoded sprite's own rectangle is painted"
    );
    assert_eq!(
        output.stats.sprite_slots_unavailable, 1,
        "the *image* slot was available and the sprite declined to draw — the \
         pass did not decline to issue it. The one unavailable slot is the \
         glyph pass's: this scene has a colour page and no monochrome one, and \
         `sprite_slots_unavailable` is one counter across both passes"
    );
    assert!(
        output.stats.sprite_draws_issued > 0,
        "and the image pass did issue its draws"
    );
}
