//! **Phase 6.2's gate.**
//!
//! > A real image file loads, decodes, uploads, and renders byte-exact against
//! > the legacy renderer's output for the same source.
//!
//! Nothing here is synthetic. The files are the ones the legacy examples ship
//! (`examples/legacy/image/`), read off disk; the decode is
//! `wgpui_widgets::image_cache`'s; the element is a real
//! [`wgpui_widgets::img::Img`] driven through the real `Reconciler`, `Emitter`
//! and `Scene`; the upload is the real atlas; and the pixels are read back off a
//! real framebuffer.
//!
//! # What "byte-exact against the legacy renderer" is actually checked against
//!
//! Two links, each compared against a side that shares no code with it:
//!
//! 1. **Decode.** Our frame is compared against the legacy call sequence itself
//!    — `image::guess_format` then
//!    `load_from_memory_with_format(..).into_rgba8()`, which is
//!    `src/elements/img.rs`'s `ImageAssetLoader::load` verbatim. Unlike Phase
//!    5.5's rasteriser oracle this is not a transcription: that half of the
//!    legacy path is `image`'s own public API at the version the root crate
//!    pins, so this compares against the real thing.
//! 2. **Draw.** Every pixel of the rendered sprite is compared against what the
//!    legacy `poly_sprites.wgsl` would emit for that texel, computed from the
//!    oracle's own bytes: `rgb` straight through, `alpha = texel.a`, composited
//!    `over` black.
//!
//! Chaining the two is what makes the sentence "the pixels on screen are the
//! ones the legacy decoder produced" a checked claim rather than two separate
//! half-claims.
//!
//! # The bar, and the tolerance that was designed in and then not needed
//!
//! An **opaque** texel is exact by construction: `src.rgb * 1 + dst.rgb * 0` is
//! the texel and `Rgba8Unorm` writes it back unchanged. A **translucent** texel
//! composites to `round(rgb * a / 255)`, which the GPU computes in `f32` and the
//! CPU here computes in `f64`, so the two *could* disagree by one unit at a
//! rounding boundary — a real difference from Phase 5.6's pure-coverage case,
//! where no such multiply existed.
//!
//! That tolerance is computed and reported, but it is **not** what the gate
//! passes on. Measured on the reference adapter (RTX 4060, Vulkan, 561.03),
//! every pixel of both assets agreed byte-for-byte, translucent ones included,
//! and both tests assert exactly that. The ±1 classification survives as a
//! diagnostic: a one-unit divergence dies with "this is the byte-exact bar"
//! rather than with a raw mismatch, so it can be read and judged instead of
//! being silently absorbed by a looser assertion.
//!
//! Because "byte-exact on an asset that is entirely opaque" would be a much
//! weaker sentence than it sounds, the source texels are classified as opaque /
//! translucent / transparent and printed, and the PNG gate asserts its asset
//! actually contains translucent texels — otherwise the blend this is all about
//! is never exercised at all.
//!
//! # If there is no adapter
//!
//! Reports and returns, per Phase 0's standard.

use std::path::PathBuf;

use wgpui_core::geometry::Rect;
use wgpui_core::invalidation::request::FrameSignals;
use wgpui_core::patch::apply::apply;
use wgpui_core::patch::emit::Emitter;
use wgpui_core::reconcile::description::Description;
use wgpui_core::reconcile::reconciler::Reconciler;
use wgpui_core::scene::Scene;
use wgpui_core::scene::atlas::AtlasKind;
use wgpui_layout::taffy_tree::{Dimension, LayoutSize, LayoutStyle, LayoutTree, definite};
use wgpui_wgpu::render::atlas::{GlyphAtlas, SharedImageAtlas};
use wgpui_wgpu::render::atlas_upload::AtlasTextures;
use wgpui_wgpu::render::device::{ComputeContext, context_or_report};
use wgpui_wgpu::render::draw::DrawMode;
use wgpui_wgpu::render::frame::{Dirty, FrameInput, FrameRenderer, OffscreenTarget};
use wgpui_widgets::image_cache::{DecodedFrame, ImageCache, decode};
use wgpui_widgets::img::{ImageEngine, ImageStyle, Img, ObjectFit, SharedImageEngine};

/// One of the legacy examples' own image files.
///
/// Read from the repository rather than embedded, because "a real image file"
/// is the gate's own wording and a byte array pasted into a test is neither
/// real nor a file.
fn asset(name: &str) -> Vec<u8> {
    let path: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "..",
        "wgpui-examples-2",
        "examples",
        "legacy",
        "image",
        name,
    ]
    .iter()
    .collect();
    std::fs::read(&path).unwrap_or_else(|error| {
        panic!(
            "the gate's own asset {} must exist: {error}",
            path.display()
        )
    })
}

/// What the legacy loader produces for these bytes: `src/elements/img.rs`'s
/// `ImageAssetLoader::load`, its still-image arm, called rather than
/// transcribed.
fn legacy_decode(bytes: &[u8]) -> ([u32; 2], Vec<u8>) {
    let format = image::guess_format(bytes).expect("a real file has a recognisable format");
    let decoded = image::load_from_memory_with_format(bytes, format)
        .expect("the legacy decoder must handle its own example asset")
        .into_rgba8();
    ([decoded.width(), decoded.height()], decoded.into_raw())
}

/// What the legacy loader produces for a GIF: every frame, first one taken.
fn legacy_decode_gif_first_frame(bytes: &[u8]) -> ([u32; 2], Vec<u8>) {
    use image::AnimationDecoder;
    let decoder = image::codecs::gif::GifDecoder::new(std::io::Cursor::new(bytes))
        .expect("the legacy decoder must handle its own example asset");
    let frame = decoder
        .into_frames()
        .next()
        .expect("an animated GIF has at least one frame")
        .expect("the first frame must decode");
    let buffer = frame.into_buffer();
    ([buffer.width(), buffer.height()], buffer.into_raw())
}

struct Panel;

/// An engine over `atlas`, holding `frame` as its only source.
fn engine_holding(
    atlas: &std::rc::Rc<std::cell::RefCell<GlyphAtlas>>,
    frame: DecodedFrame,
) -> (SharedImageEngine, wgpui_widgets::img::ImageSourceId) {
    use wgpui_widgets::image_cache::DecodedImage;

    let mut cache = ImageCache::new();
    let source = cache
        .hold(DecodedImage::from_frames(vec![frame]).expect("one frame is a valid image"))
        .expect("holding a decoded image must succeed");
    let engine = std::rc::Rc::new(std::cell::RefCell::new(ImageEngine::new(
        cache,
        Box::new(SharedImageAtlas::new(std::rc::Rc::clone(atlas))),
    )));
    (engine, source)
}

/// What the whole path produced: the rendered framebuffer, and the scene it
/// came from.
struct Rendered {
    pixels: Vec<u8>,
    width: u32,
    sprites_resident: u32,
    atlas_pages: usize,
}

/// Drive an `Img` from description to framebuffer.
///
/// Reconcile, lay out, emit, apply, upload the atlas, render, read back — the
/// same sequence a window runs, with no step stubbed. That is the point of the
/// gate: any of these being wired up wrong is a failure here, not a discovery
/// later.
fn render_element(
    context: &ComputeContext,
    atlas: &std::rc::Rc<std::cell::RefCell<GlyphAtlas>>,
    image: Img,
    size: [u32; 2],
) -> Rendered {
    let mut reconciler = Reconciler::new();
    let mut layout = LayoutTree::new();
    let mut emitter = Emitter::new();
    let mut scene = Scene::new();

    let description = Description::new::<Panel>()
        .style(LayoutStyle {
            size: LayoutSize {
                width: Dimension::length(size[0] as f32),
                height: Dimension::length(size[1] as f32),
            },
            ..LayoutStyle::default()
        })
        .child(image.describe());

    let plan = reconciler
        .reconcile(description, &mut layout)
        .expect("reconciling one element must succeed");
    let root = plan
        .nodes()
        .first()
        .map(|node| node.layout_node)
        .expect("the plan has a root");
    layout
        .compute_layout(root, definite(size[0] as f32, size[1] as f32))
        .expect("laying out one element must succeed");
    let emission = emitter
        .emit(&plan, &layout, &FrameSignals::new(), &mut scene)
        .expect("emitting must succeed");
    apply(&mut scene, &emission.patch).expect("applying the patch must succeed");

    // The atlas was filled during emission, by the element asking for its tile.
    let mut textures = AtlasTextures::for_atlas(&atlas.borrow());
    let report = {
        let mut borrowed = atlas.borrow_mut();
        textures.sync(&context.device, &context.queue, &mut borrowed)
    };
    assert_eq!(
        report.skipped, 0,
        "every rectangle the atlas queued must reach a texture"
    );

    let mut renderer = FrameRenderer::new(&context.device);
    let target = OffscreenTarget::new(&context.device, size[0], size[1]);
    let input = FrameInput {
        scene: &scene,
        clip: Rect::from_origin_size([0.0, 0.0], [size[0] as f32, size[1] as f32]),
        poison: &[],
        dirty: Dirty::All,
        uploads: &[],
        composites: &[],
        registry: None,
        atlas: Some(&textures),
        viewport: [size[0] as f32, size[1] as f32],
        mode: DrawMode::best_available(context.indirect),
    };
    let output = renderer
        .render(&context.device, &context.queue, &input, &target)
        .expect("a frame must render");
    let pixels = target
        .read_pixels(&context.device, &context.queue)
        .expect("reading the target back must succeed");

    Rendered {
        pixels,
        width: size[0],
        sprites_resident: output.primitives_resident,
        atlas_pages: textures.pages_of_kind(AtlasKind::Polychrome).len(),
    }
}

/// What the legacy fragment shader emits for one texel, composited over the
/// target's own opaque black.
///
/// `rgb` straight through and `alpha = texel.a * opacity * coverage`, with
/// opacity 1 and coverage 1 at every pixel centre of a square sprite. The
/// straight-alpha `over` blend against `dst = (0, 0, 0, 1)` then gives
/// `rgb * a` in the colour channels and `a + 1 * (1 - a)` — which is 1, always —
/// in the alpha one.
///
/// **The alpha channel is 255 whatever the texel's alpha was**, and getting that
/// wrong is what this gate caught on its first run: an oracle that expected the
/// texel's own alpha to survive the blend disagreed with the framebuffer on
/// every transparent pixel of a real icon. `OffscreenTarget::target` clears to
/// `wgpu::Color::BLACK`, which is opaque, so there is no transparency in the
/// destination for a transparent source to preserve. A fully transparent texel
/// takes the shader's `discard` and leaves that clear colour untouched, which
/// this expression reproduces rather than special-cases.
///
/// Computed in `f64` so the CPU side is not the source of the rounding the
/// comparison is about.
fn expected_over_black(texel: [u8; 4]) -> [u8; 4] {
    let alpha = f64::from(texel[3]) / 255.0;
    let channel = |value: u8| (f64::from(value) * alpha).round() as u8;
    [
        channel(texel[0]),
        channel(texel[1]),
        channel(texel[2]),
        0xFF,
    ]
}

/// What one comparison found.
///
/// The alpha classes are carried, not just the agreement counts, because
/// "byte-exact" over an asset that happens to be entirely opaque is a weaker
/// claim than the same sentence over one that is not, and a report that cannot
/// tell the two apart is not a report. `translucent` is the population the ±1
/// tolerance is even *allowed* to apply to; if it is zero, the tolerance was
/// never under test and this says so rather than letting a clean number imply
/// otherwise.
struct Agreement {
    exact: usize,
    within_one: usize,
    opaque: usize,
    translucent: usize,
    transparent: usize,
}

/// Compare a rendered frame against an oracle bitmap, pixel by pixel.
///
/// Panics with the first real divergence rather than with a count, because
/// "1,400 pixels differ" is not a bug report.
fn compare(rendered: &Rendered, oracle_size: [u32; 2], oracle: &[u8], label: &str) -> Agreement {
    let mut exact = 0usize;
    let mut within_one = 0usize;
    let mut opaque = 0usize;
    let mut translucent = 0usize;
    let mut transparent = 0usize;
    for y in 0..oracle_size[1] {
        for x in 0..oracle_size[0] {
            let source = ((y * oracle_size[0] + x) * 4) as usize;
            let texel: [u8; 4] = oracle
                .get(source..source + 4)
                .expect("the oracle covers every texel")
                .try_into()
                .expect("four bytes");
            let expected = expected_over_black(texel);
            match texel[3] {
                0x00 => transparent += 1,
                0xFF => opaque += 1,
                _ => translucent += 1,
            }

            let index = ((y * rendered.width + x) * 4) as usize;
            let drawn: [u8; 4] = rendered
                .pixels
                .get(index..index + 4)
                .expect("the framebuffer covers the sprite")
                .try_into()
                .expect("four bytes");

            if drawn == expected {
                exact += 1;
                continue;
            }
            let close = (0..4).all(|channel| {
                (i32::from(drawn[channel]) - i32::from(expected[channel])).abs() <= 1
            });
            assert!(
                close,
                "[{label}] pixel ({x}, {y}) is {drawn:?}, and the legacy decoder's \
                 texel {texel:?} composites to {expected:?}"
            );
            assert!(
                texel[3] != 0xFF,
                "[{label}] pixel ({x}, {y}) is opaque and must be exact, not close: \
                 {drawn:?} against {expected:?}"
            );
            within_one += 1;
        }
    }
    Agreement {
        exact,
        within_one,
        opaque,
        translucent,
        transparent,
    }
}

/// **The gate**, on a real PNG.
#[test]
fn a_real_png_loads_decodes_uploads_and_renders_as_the_legacy_decoder_produced_it() {
    let Some(context) = context_or_report("phase_6_2_png_gate") else {
        return;
    };
    let bytes = asset("app-icon.png");

    // --- Link 1: the decode, against the legacy call sequence itself.
    let (legacy_size, legacy_texels) = legacy_decode(&bytes);
    let ours = decode(&bytes).expect("2.0 must decode the legacy example's own asset");
    let frame = ours.frame(0).expect("one frame").clone();
    assert_eq!(frame.size, legacy_size, "decoded size");
    assert_eq!(
        frame.texels, legacy_texels,
        "the decoded bytes must be the legacy decoder's own, not merely similar"
    );
    assert_eq!(ours.frame_count(), 1, "a PNG is a still image");

    // --- Link 2: the draw, against what those bytes composite to.
    let atlas = std::rc::Rc::new(std::cell::RefCell::new(GlyphAtlas::new(1024)));
    let (engine, source) = engine_holding(&atlas, frame);
    let image = Img::new(source, engine)
        .size(legacy_size[0] as f32, legacy_size[1] as f32)
        .style(ImageStyle {
            // Natural size, no rounding, fully opaque: the conditions under
            // which the comparison is an equality rather than a tolerance.
            object_fit: ObjectFit::Fill,
            ..ImageStyle::default()
        });
    let rendered = render_element(&context, &atlas, image, legacy_size);

    assert_eq!(rendered.sprites_resident, 1, "one sprite, from one element");
    assert_eq!(rendered.atlas_pages, 1, "one colour page");

    let found = compare(&rendered, legacy_size, &legacy_texels, "app-icon.png");
    let total = (legacy_size[0] * legacy_size[1]) as usize;
    println!(
        "phase_6_2_png_gate: {} of {total} pixels byte-exact, {} within one unit; \
         source texels: {} opaque, {} translucent, {} transparent",
        found.exact, found.within_one, found.opaque, found.translucent, found.transparent
    );
    assert!(
        found.translucent > 0,
        "this asset must contain translucent texels, or the blend the gate is \
         about is never exercised and 'byte-exact' means only 'exact where alpha \
         does nothing'"
    );
    assert_eq!(
        found.exact, total,
        "every pixel of this asset agreed byte-for-byte on the run that set this \
         bar, translucent ones included; a run that needs the ±1 tolerance is a \
         change in behaviour and must be read, not absorbed"
    );
}

/// The same gate, on a real animated GIF's first frame.
///
/// Included because a GIF is the one format whose decode is a *different* legacy
/// arm — `GifDecoder::into_frames()` rather than `load_from_memory_with_format`
/// — and a differential that only ever exercises one arm proves one arm.
#[test]
fn a_real_gifs_first_frame_renders_as_the_legacy_decoder_produced_it() {
    let Some(context) = context_or_report("phase_6_2_gif_gate") else {
        return;
    };
    let bytes = asset("black-cat-typing.gif");

    let (legacy_size, legacy_texels) = legacy_decode_gif_first_frame(&bytes);
    let ours = decode(&bytes).expect("2.0 must decode the legacy example's own GIF");
    assert!(
        ours.is_animated(),
        "this asset is animated, or the arm under test is not the one being taken"
    );
    let frame = ours.frame(0).expect("frame 0").clone();
    assert_eq!(frame.size, legacy_size);
    assert_eq!(
        frame.texels, legacy_texels,
        "the GIF arm's bytes must be the legacy GIF arm's own"
    );

    let atlas = std::rc::Rc::new(std::cell::RefCell::new(GlyphAtlas::new(1024)));
    let (engine, source) = engine_holding(&atlas, frame);
    let image = Img::new(source, engine)
        .size(legacy_size[0] as f32, legacy_size[1] as f32)
        .style(ImageStyle {
            object_fit: ObjectFit::Fill,
            ..ImageStyle::default()
        });
    let rendered = render_element(&context, &atlas, image, legacy_size);

    let found = compare(
        &rendered,
        legacy_size,
        &legacy_texels,
        "black-cat-typing.gif",
    );
    let total = (legacy_size[0] * legacy_size[1]) as usize;
    println!(
        "phase_6_2_gif_gate: {} of {total} pixels byte-exact, {} within one unit; \
         source texels: {} opaque, {} translucent, {} transparent; \
         {} frames decoded, frame 0 drawn",
        found.exact,
        found.within_one,
        found.opaque,
        found.translucent,
        found.transparent,
        ours.frame_count()
    );
    assert_eq!(
        found.exact, total,
        "the GIF arm is held to the same bar as the PNG one; a differential that \
         asserts only 'something matched' proves only that something matched"
    );
}

/// The differential can fail.
///
/// A gate nobody has watched fail is a gate nobody knows works. This corrupts
/// one texel of the oracle and confirms the comparison notices — which also
/// pins down that the comparison is reading the pixels it claims to.
#[test]
fn the_comparison_actually_detects_a_wrong_pixel() {
    let Some(context) = context_or_report("phase_6_2_gate_falsification") else {
        return;
    };
    let bytes = asset("app-icon.png");
    let (size, mut texels) = legacy_decode(&bytes);
    let frame = decode(&bytes)
        .expect("decode")
        .frame(0)
        .expect("one frame")
        .clone();

    let atlas = std::rc::Rc::new(std::cell::RefCell::new(GlyphAtlas::new(1024)));
    let (engine, source) = engine_holding(&atlas, frame);
    let image = Img::new(source, engine)
        .size(size[0] as f32, size[1] as f32)
        .style(ImageStyle {
            object_fit: ObjectFit::Fill,
            ..ImageStyle::default()
        });
    let rendered = render_element(&context, &atlas, image, size);

    // Find an opaque texel and change it beyond the tolerance.
    let opaque = texels
        .chunks_exact(4)
        .position(|texel| texel[3] == 0xFF)
        .expect("a real icon has an opaque texel");
    let index = opaque * 4;
    texels[index] = texels[index].wrapping_add(64);

    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = compare(&rendered, size, &texels, "corrupted");
    }));
    assert!(
        outcome.is_err(),
        "the comparison must reject a corrupted oracle, or agreement with an \
         intact one means nothing"
    );
}
