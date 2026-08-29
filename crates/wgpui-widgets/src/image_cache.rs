//! Image loading/decoding cache. See docs/gpu-native-architecture.md §3.4, §8
//! (Phase 6.2).
//!
//! # What this is a port of, and what it deliberately is not
//!
//! The decode half of `src/elements/img.rs`'s `ImageAssetLoader::load` — the
//! `image::guess_format` dispatch, the GIF and animated-WebP frame paths, the
//! `into_rgba8()` fallback for every still format — plus `src/svg_renderer.rs`'s
//! `render_single_frame_inner`, which is the same decode one layer over.
//! Expression for expression where it matters, because a differential against
//! the legacy output is this phase's gate and two decoders that "do the same
//! thing" are not two decoders that produce the same bytes.
//!
//! What is *not* ported is everything around it: `Resource::Path`/`Uri`/
//! `Embedded` resolution, the HTTP client, the asset source, and the `Asset`
//! future. Those need `App` (§3.4 puts them elsewhere) and none of them is about
//! turning bytes into pixels. [`decode`] takes a `&[u8]` and that is the whole
//! of its input, which is also what makes it testable against the legacy
//! expressions directly.
//!
//! # Straight alpha, and the one place legacy and this disagree
//!
//! Every [`DecodedFrame`] holds **straight** (non-premultiplied) RGBA8. For
//! `image`'s formats that is free: `into_rgba8()` already produces straight
//! alpha. For SVG it is not — `resvg` renders into a `tiny_skia::Pixmap`, whose
//! texels are premultiplied — and [`decode`] un-premultiplies before handing the
//! frame on.
//!
//! Legacy does not. `render_single_frame_inner` puts the pixmap's bytes into a
//! `RenderImage` unchanged and leaves the correction to the shader's
//! `premultiplied_alpha` global, which `src/platform/cross/renderer.rs` sets
//! only when the *surface's* composite alpha mode is `PreMultiplied` — a
//! property of the window, not of the image. On a surface that is `Opaque`
//! (which is the ordinary case on Windows/Vulkan) a translucent SVG is therefore
//! composited as though its already-multiplied colour were straight, and comes
//! out darker than it should. 2.0 un-premultiplies at the decode instead, so a
//! translucent SVG composites the same way on every surface. That is a
//! *deliberate divergence* from legacy output for one input class, recorded here
//! and in docs/phase-6.2-results.md rather than discovered later as a
//! differential failure.
//!
//! # Animation: frames are decoded, and nothing advances them
//!
//! [`decode`] returns every frame of an animated GIF or WebP, with each frame's
//! delay, because that is what the decoder already hands over and dropping it
//! would mean re-decoding to get it back. What does *not* exist is anything that
//! advances [`crate::img::Img::frame_index`] over time: 2.0 has no animation
//! driver at all (`wgpui_core::window::animation` and `crate::animation` are
//! both still stubs). So an animated source renders the frame it is asked for,
//! forever, and a caller that ticks the index gets animation. Stated plainly
//! because "GIF decoding works" and "GIFs animate" are different claims and only
//! the first is true here.

use std::collections::HashMap;
use std::io::Cursor;
use std::time::Duration;

use wgpui_core::scene::atlas::{ImageRasterKey, RasterizedImage};

use crate::img::ImageSourceId;

/// One decoded frame: straight-alpha RGBA8 at its natural size.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedFrame {
    /// The frame's size in pixels, `[width, height]`.
    pub size: [u32; 2],
    /// Straight-alpha RGBA8, row-major, tightly packed.
    pub texels: Vec<u8>,
    /// How long this frame is shown before the next.
    ///
    /// Zero for a still image. Carried for animated sources because the decoder
    /// already produces it and recovering it later would mean decoding again —
    /// not because anything in 2.0 yet reads it. See this module's doc.
    pub delay: Duration,
}

impl DecodedFrame {
    /// This frame as the atlas's upload vocabulary.
    ///
    /// A copy, because [`RasterizedImage`] owns its texels and the cache keeps
    /// its own. That copy happens once per frame per tile — on the miss path
    /// only, since [`wgpui_core::scene::atlas::ImageTileSource`]'s whole
    /// contract is that a resident key never reaches the decoder — so it costs
    /// one memcpy per image per atlas residency, not one per frame drawn.
    pub fn to_raster(&self) -> RasterizedImage {
        RasterizedImage {
            size: self.size,
            texels: self.texels.clone(),
        }
    }
}

/// A decoded image: one frame for a still, many for an animation.
///
/// Never empty — [`decode`] reports [`ImageDecodeError::NoFrames`] rather than
/// returning one of these with nothing in it, so every consumer can treat frame
/// 0 as present.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedImage {
    frames: Vec<DecodedFrame>,
}

impl DecodedImage {
    /// An image from frames produced somewhere other than [`decode`] — a
    /// background task, a re-scaled SVG, or a test.
    ///
    /// Fallible rather than infallible so the "never empty, never zero-area"
    /// invariant every consumer relies on has exactly one place it is
    /// established, whichever door the frames came in through.
    pub fn from_frames(frames: Vec<DecodedFrame>) -> Result<Self, ImageDecodeError> {
        finish(frames)
    }

    /// Every frame, in playback order.
    pub fn frames(&self) -> &[DecodedFrame] {
        &self.frames
    }

    /// How many frames this source has. At least one.
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Whether this source has more than one frame.
    pub fn is_animated(&self) -> bool {
        self.frames.len() > 1
    }

    /// Select the frame visible after `elapsed` in a looping animation.
    ///
    /// Delays are accumulated rather than treated as a fixed frame rate. A
    /// zero-delay animation still advances safely by using its first frame;
    /// malformed timing data must not create a hot loop in a window driver.
    pub fn frame_index_at(&self, elapsed: Duration) -> u32 {
        if self.frames.len() <= 1 {
            return 0;
        }
        let cycle = self
            .frames
            .iter()
            .map(|frame| frame.delay)
            .fold(Duration::ZERO, |total, delay| total.saturating_add(delay));
        if cycle.is_zero() {
            return 0;
        }
        let remaining_seconds = elapsed.as_secs_f64() % cycle.as_secs_f64();
        let mut remaining = Duration::from_secs_f64(remaining_seconds);
        for (index, frame) in self.frames.iter().enumerate() {
            if remaining < frame.delay {
                return index as u32;
            }
            remaining = remaining.saturating_sub(frame.delay);
        }
        (self.frames.len() - 1) as u32
    }

    /// One frame, wrapping the index into range.
    ///
    /// Wrapping rather than clamping or failing, because that is what a looping
    /// animation wants and because a caller ticking an index has no reason to
    /// know the frame count. A still image therefore answers frame 7 with its
    /// only frame rather than with nothing.
    pub fn frame(&self, frame_index: u32) -> Option<&DecodedFrame> {
        if self.frames.is_empty() {
            return None;
        }
        self.frames.get(frame_index as usize % self.frames.len())
    }

    /// The natural size of frame 0, which is the size an unsized `Img` takes.
    pub fn natural_size(&self) -> [u32; 2] {
        self.frames
            .first()
            .map(|frame| frame.size)
            .unwrap_or([0, 0])
    }
}

/// Why a byte buffer could not be turned into pixels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImageDecodeError {
    /// The bytes match no format `image` recognises and are not valid SVG
    /// either.
    ///
    /// Both halves are reported together on purpose: legacy's dispatch is
    /// "whatever `guess_format` does not claim, hand to the SVG renderer", so
    /// the SVG parse failure *is* the format failure for anything that is not a
    /// raster image, and reporting only one of them would name the wrong cause
    /// for half the inputs.
    UnrecognisedFormat {
        /// What `usvg` said when it was handed the same bytes.
        svg: String,
    },
    /// The format was recognised and the pixels could not be produced.
    Decode(String),
    /// The decoder produced no frames at all.
    ///
    /// Not reachable from `image`'s still decoders, which always produce one.
    /// Reachable from a zero-frame GIF, which is malformed and which would
    /// otherwise become a `DecodedImage` whose `frame(0)` is `None` — an
    /// invariant every consumer would then have to re-check.
    NoFrames,
    /// The image decoded to a zero-width or zero-height bitmap.
    ///
    /// Refused here rather than at the atlas, which would report
    /// `AtlasError::EmptyRaster` from three layers down with no idea which
    /// source it came from.
    EmptySize,
}

impl std::fmt::Display for ImageDecodeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImageDecodeError::UnrecognisedFormat { svg } => write!(
                formatter,
                "the bytes are not an image format this build decodes, and not SVG either: {svg}"
            ),
            ImageDecodeError::Decode(message) => {
                write!(formatter, "image decode failed: {message}")
            }
            ImageDecodeError::NoFrames => formatter.write_str("the decoder produced no frames"),
            ImageDecodeError::EmptySize => {
                formatter.write_str("the image decoded to a zero-area bitmap")
            }
        }
    }
}

impl std::error::Error for ImageDecodeError {}

/// The scale `resvg` rasterises at, over and above the requested one.
///
/// The legacy `SMOOTH_SVG_SCALE_FACTOR`, kept at the same value and for the
/// stated reason — "we render them at twice the size to get a higher-quality
/// result." It matters more here than there, because this phase's sprite
/// pipeline samples nearest rather than filtering: a vector that was rasterised
/// at 2× and drawn at 1× would need a downscale it does not get. See
/// docs/phase-6.2-results.md, which records that consequence rather than leaving
/// it implied.
pub const SMOOTH_SVG_SCALE_FACTOR: f32 = 2.0;

/// Turn an encoded image into frames of straight-alpha RGBA8.
///
/// The dispatch is `src/elements/img.rs`'s: whatever `image::guess_format`
/// claims goes to `image`, and everything else goes to the SVG rasteriser.
pub fn decode(bytes: &[u8]) -> Result<DecodedImage, ImageDecodeError> {
    let frames = match image::guess_format(bytes) {
        Ok(format) => decode_raster(bytes, format)?,
        Err(_) => decode_svg(bytes, 1.0)?,
    };
    finish(frames)
}

/// Rasterise SVG bytes at `scale_factor`, ignoring `image`'s format detection.
///
/// Exposed beside [`decode`] because an SVG is the one source whose pixels are a
/// function of the size it will be drawn at, so a caller that already knows that
/// size can ask for it. [`decode`] uses 1.0, which is what legacy's
/// `ImageAssetLoader` passes.
pub fn decode_svg_at(bytes: &[u8], scale_factor: f32) -> Result<DecodedImage, ImageDecodeError> {
    finish(decode_svg(bytes, scale_factor)?)
}

fn finish(frames: Vec<DecodedFrame>) -> Result<DecodedImage, ImageDecodeError> {
    if frames.is_empty() {
        return Err(ImageDecodeError::NoFrames);
    }
    if frames
        .iter()
        .any(|frame| frame.size[0] == 0 || frame.size[1] == 0)
    {
        return Err(ImageDecodeError::EmptySize);
    }
    Ok(DecodedImage { frames })
}

fn decode_raster(
    bytes: &[u8],
    format: image::ImageFormat,
) -> Result<Vec<DecodedFrame>, ImageDecodeError> {
    use image::AnimationDecoder;

    match format {
        image::ImageFormat::Gif => {
            let decoder = image::codecs::gif::GifDecoder::new(Cursor::new(bytes))
                .map_err(|error| ImageDecodeError::Decode(error.to_string()))?;
            let mut frames = Vec::new();
            for frame in decoder.into_frames() {
                let frame = frame.map_err(|error| ImageDecodeError::Decode(error.to_string()))?;
                frames.push(animation_frame(frame));
            }
            Ok(frames)
        }
        image::ImageFormat::WebP => {
            let mut decoder = image::codecs::webp::WebPDecoder::new(Cursor::new(bytes))
                .map_err(|error| ImageDecodeError::Decode(error.to_string()))?;
            if decoder.has_animation() {
                // The legacy call, kept: a transparent background rather than
                // the file's declared one, so a frame that only paints its
                // changed region composites over nothing instead of over a
                // colour the caller never asked for.
                decoder
                    .set_background_color(image::Rgba([0, 0, 0, 0]))
                    .map_err(|error| ImageDecodeError::Decode(error.to_string()))?;
                let mut frames = Vec::new();
                for frame in decoder.into_frames() {
                    let frame =
                        frame.map_err(|error| ImageDecodeError::Decode(error.to_string()))?;
                    frames.push(animation_frame(frame));
                }
                Ok(frames)
            } else {
                let image = image::DynamicImage::from_decoder(decoder)
                    .map_err(|error| ImageDecodeError::Decode(error.to_string()))?;
                Ok(vec![still_frame(image.into_rgba8())])
            }
        }
        _ => {
            let image = image::load_from_memory_with_format(bytes, format)
                .map_err(|error| ImageDecodeError::Decode(error.to_string()))?;
            Ok(vec![still_frame(image.into_rgba8())])
        }
    }
}

fn still_frame(buffer: image::RgbaImage) -> DecodedFrame {
    let size = [buffer.width(), buffer.height()];
    DecodedFrame {
        size,
        texels: buffer.into_raw(),
        delay: Duration::ZERO,
    }
}

fn animation_frame(frame: image::Frame) -> DecodedFrame {
    let delay = Duration::from(frame.delay());
    let mut decoded = still_frame(frame.into_buffer());
    decoded.delay = delay;
    decoded
}

fn decode_svg(bytes: &[u8], scale_factor: f32) -> Result<Vec<DecodedFrame>, ImageDecodeError> {
    use resvg::tiny_skia;

    let options = usvg::Options::default();
    let tree = usvg::Tree::from_data(bytes, &options).map_err(|error| {
        ImageDecodeError::UnrecognisedFormat {
            svg: error.to_string(),
        }
    })?;
    let svg_size = tree.size();
    let scale = scale_factor * SMOOTH_SVG_SCALE_FACTOR;
    let mut pixmap = tiny_skia::Pixmap::new(
        (svg_size.width() * scale) as u32,
        (svg_size.height() * scale) as u32,
    )
    .ok_or(ImageDecodeError::EmptySize)?;
    resvg::render(
        &tree,
        tiny_skia::Transform::from_scale(scale, scale),
        &mut pixmap.as_mut(),
    );

    let size = [pixmap.width(), pixmap.height()];
    // Un-premultiplied here rather than left to a shader flag. See this module's
    // doc for why that is a deliberate divergence from legacy and not an
    // oversight in either direction.
    let mut texels = Vec::with_capacity((size[0] * size[1] * 4) as usize);
    for texel in pixmap.pixels() {
        let straight = texel.demultiply();
        texels.extend_from_slice(&[
            straight.red(),
            straight.green(),
            straight.blue(),
            straight.alpha(),
        ]);
    }
    Ok(vec![DecodedFrame {
        size,
        texels,
        delay: Duration::ZERO,
    }])
}

/// Decoded images, addressed by the identity `Img` reconciles against.
///
/// # Why the cache issues the id rather than being handed one
///
/// [`ImageSourceId`]'s own doc sets the rule this depends on: "a source that is
/// reloaded — re-fetched, re-decoded, replaced on disk — is issued a new id
/// rather than mutating in place, which is what makes comparing identity rather
/// than content sound." A cache that let a caller choose the id could not
/// enforce that; a cache that mints one on every [`Self::insert`] enforces it by
/// construction, and an `Img`'s `diff_key` therefore cannot report "unchanged"
/// across a reload.
///
/// # What it deliberately does not do
///
/// No eviction, no memory budget, and no background decoding. Eviction wants a
/// policy and a measurement nobody has taken, and the atlas — which is where the
/// bytes actually cost GPU memory — already has one (R-N §4.3's subscription,
/// closed for sprites in this phase). Background decoding wants `App`'s
/// executor, which §3.4 puts outside this crate. Both are named in
/// docs/phase-6.2-results.md as open rather than left to look finished.
#[derive(Debug, Default)]
pub struct ImageCache {
    images: HashMap<ImageSourceId, DecodedImage>,
    next_source: u64,
}

impl ImageCache {
    /// An empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Decode `bytes` and hold the result under a fresh source id.
    pub fn insert(&mut self, bytes: &[u8]) -> Result<ImageSourceId, ImageDecodeError> {
        self.hold(decode(bytes)?)
    }

    /// Hold an already-decoded image under a fresh source id.
    ///
    /// For a caller that decoded elsewhere — a background task, a test, or
    /// [`decode_svg_at`] with a scale this cache does not choose.
    pub fn hold(&mut self, image: DecodedImage) -> Result<ImageSourceId, ImageDecodeError> {
        self.next_source += 1;
        let source = ImageSourceId::from_raw(self.next_source);
        self.images.insert(source, image);
        Ok(source)
    }

    /// The decoded image for `source`, if it is held.
    pub fn get(&self, source: ImageSourceId) -> Option<&DecodedImage> {
        self.images.get(&source)
    }

    /// One frame of one source.
    pub fn frame(&self, source: ImageSourceId, frame_index: u32) -> Option<&DecodedFrame> {
        self.get(source)?.frame(frame_index)
    }

    /// The bitmap an [`ImageRasterKey`] names, in the atlas's vocabulary.
    ///
    /// The function an
    /// [`wgpui_core::scene::atlas::ImageTileSource`] implementation's closure
    /// is: a key in, texels out, `None` when the source is not held. It is here
    /// rather than in the tile source because the cache is what knows how a key
    /// maps onto a frame, and the tile source is what knows about atlas pages.
    pub fn raster(&self, key: ImageRasterKey) -> Option<RasterizedImage> {
        Some(
            self.frame(ImageSourceId::from_raw(key.source), key.frame_index)?
                .to_raster(),
        )
    }

    /// How many sources are held.
    pub fn len(&self) -> usize {
        self.images.len()
    }

    /// Whether the cache holds nothing.
    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }

    /// Drop one source's pixels.
    ///
    /// Returns whether it was held. The atlas tiles that came from it are not
    /// touched: they are evicted through the atlas's own path
    /// (R-N §4.3), which is what makes the layers referencing them repaint. A
    /// cache that reached into the atlas would be a second eviction mechanism
    /// with its own chance of missing a subscriber.
    pub fn remove(&mut self, source: ImageSourceId) -> bool {
        self.images.remove(&source).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny PNG, encoded by `image` itself so the test's input is a real file
    /// and not a hand-written header.
    fn png(width: u32, height: u32) -> Vec<u8> {
        let mut buffer = image::RgbaImage::new(width, height);
        for (x, y, pixel) in buffer.enumerate_pixels_mut() {
            *pixel = image::Rgba([(x * 9) as u8, (y * 5) as u8, (x ^ y) as u8, 0xFF]);
        }
        let mut bytes = Vec::new();
        image::DynamicImage::ImageRgba8(buffer)
            .write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
            .expect("encoding a PNG in memory must succeed");
        bytes
    }

    // `r##`, not `r#`: a colour literal contains `"#`, which would close a
    // single-hash raw string in the middle of the document.
    const SQUARE_SVG: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="6">
        <rect width="10" height="6" fill="#3366cc"/>
    </svg>"##;

    #[test]
    fn a_png_decodes_to_one_straight_alpha_frame_at_its_natural_size() {
        let image = decode(&png(7, 5)).expect("a real PNG decodes");
        assert_eq!(image.frame_count(), 1);
        assert!(!image.is_animated());
        assert_eq!(image.natural_size(), [7, 5]);
        let frame = image.frame(0).expect("frame 0 is always present");
        assert_eq!(frame.texels.len(), 7 * 5 * 4);
        assert_eq!(frame.delay, Duration::ZERO);
        assert_eq!(
            frame.texels.first(),
            Some(&0u8),
            "texel (0,0)'s red channel is `0 * 9`"
        );
    }

    /// **The decode differential.** Our output is what the legacy expressions
    /// produce, byte for byte, from the same bytes.
    ///
    /// The oracle is the legacy call sequence itself —
    /// `image::guess_format` then `load_from_memory_with_format(..).into_rgba8()`
    /// — rather than a transcription of it, because unlike Phase 5.5's
    /// rasteriser oracle this half of the legacy path *is* reachable: it is
    /// `image`'s own public API, at the version the root crate pins. So this
    /// compares against the real thing rather than against a re-derivation of
    /// it.
    #[test]
    fn a_still_image_decodes_to_exactly_what_the_legacy_expressions_produce() {
        for (label, bytes) in [("png", png(9, 4)), ("small png", png(1, 1))] {
            // `src/elements/img.rs`, `ImageAssetLoader::load`, verbatim:
            let format = image::guess_format(&bytes).expect("a real file has a format");
            let legacy = image::load_from_memory_with_format(&bytes, format)
                .expect("legacy decode")
                .into_rgba8();
            let legacy_size = [legacy.width(), legacy.height()];
            let legacy_texels = legacy.into_raw();

            let ours = decode(&bytes).expect("our decode");
            let frame = ours.frame(0).expect("one frame");
            assert_eq!(frame.size, legacy_size, "[{label}] size");
            assert_eq!(
                frame.texels, legacy_texels,
                "[{label}] the decoded bytes must be the legacy decoder's own"
            );
        }
    }

    #[test]
    fn an_svg_rasterises_at_the_legacy_smoothing_scale() {
        let image = decode(SQUARE_SVG.as_bytes()).expect("an SVG rasterises");
        assert_eq!(
            image.natural_size(),
            [20, 12],
            "a 10x6 document at the legacy 2x smoothing scale"
        );
        let frame = image.frame(0).expect("one frame");
        assert_eq!(frame.texels.len(), 20 * 12 * 4);
        // #3366cc, opaque, and — the point — *not* premultiplied into something
        // darker, which is what the pixmap holds before `demultiply`.
        assert_eq!(
            frame.texels.get(0..4),
            Some([0x33, 0x66, 0xCC, 0xFF].as_slice())
        );
    }

    /// The divergence from legacy, asserted rather than described.
    ///
    /// A half-transparent fill is premultiplied in the pixmap `resvg` produces;
    /// legacy uploads that and this un-premultiplies it. The assertion is on the
    /// straight value, so a future change that stops un-premultiplying fails
    /// here instead of producing quietly darker SVGs.
    #[test]
    fn a_translucent_svg_is_un_premultiplied_at_the_decode() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4">
            <rect width="4" height="4" fill="#ffffff" fill-opacity="0.5"/>
        </svg>"##;
        let image = decode(svg.as_bytes()).expect("an SVG rasterises");
        let frame = image.frame(0).expect("one frame");
        let first = frame.texels.get(0..4).expect("at least one texel");
        assert!(
            first[3] > 100 && first[3] < 160,
            "the fill is about half transparent, got alpha {}",
            first[3]
        );
        assert!(
            first[0] > 250,
            "white at half alpha is still white in straight alpha; premultiplied \
             it would be about 128, and this is {}",
            first[0]
        );
    }

    #[test]
    fn bytes_that_are_neither_an_image_nor_svg_are_reported_with_both_causes() {
        let error = decode(b"not a picture").expect_err("garbage must not decode");
        assert!(matches!(error, ImageDecodeError::UnrecognisedFormat { .. }));
        // The message names the SVG parse failure, which is the actual cause for
        // anything that is not a raster image.
        assert!(!error.to_string().is_empty());
    }

    #[test]
    fn a_truncated_file_of_a_known_format_reports_a_decode_failure_not_a_format_one() {
        let bytes = png(8, 8);
        let truncated = &bytes[..bytes.len() / 2];
        // The PNG signature is intact, so `guess_format` claims it and the
        // failure is a decode failure rather than a format one — which is the
        // distinction the two variants exist to draw.
        assert!(matches!(
            decode(truncated),
            Err(ImageDecodeError::Decode(_))
        ));
    }

    #[test]
    fn the_cache_issues_a_new_id_per_insert_so_a_reload_can_never_compare_equal() {
        let mut cache = ImageCache::new();
        let bytes = png(4, 4);
        let first = cache.insert(&bytes).expect("decode");
        let second = cache.insert(&bytes).expect("decode");
        assert_ne!(
            first, second,
            "identical bytes inserted twice are two sources — which is what makes \
             `Img`'s key sound, per `ImageSourceId`'s own contract"
        );
        assert_eq!(cache.len(), 2);
        assert!(cache.get(first).is_some());
        assert!(cache.remove(first));
        assert!(!cache.remove(first));
        assert!(cache.get(second).is_some());
    }

    #[test]
    fn the_cache_answers_a_raster_key_with_the_frame_it_names() {
        let mut cache = ImageCache::new();
        let source = cache.insert(&png(6, 3)).expect("decode");
        let key = ImageRasterKey {
            source: source.as_raw(),
            frame_index: 0,
            scale_factor_bits: 1.0f32.to_bits(),
        };
        let raster = cache.raster(key).expect("a held source rasters");
        assert_eq!(raster.size, [6, 3]);
        assert!(raster.is_well_formed());
        assert_eq!(
            cache.raster(ImageRasterKey { source: 999, ..key }),
            None,
            "a source the cache does not hold is `None`, not a panic"
        );
    }

    #[test]
    fn a_still_image_answers_any_frame_index_with_its_only_frame() {
        let image = decode(&png(2, 2)).expect("decode");
        assert_eq!(image.frame(0), image.frame(7));
        assert_eq!(image.frame(0), image.frame(u32::MAX));
    }

    #[test]
    fn animation_uses_each_decoded_frame_delay_and_loops() {
        let frames = (0..3)
            .map(|_| DecodedFrame {
                size: [1, 1],
                texels: vec![0; 4],
                delay: Duration::from_millis(100),
            })
            .collect();
        let image = DecodedImage::from_frames(frames).expect("frames");
        assert_eq!(image.frame_index_at(Duration::from_millis(0)), 0);
        assert_eq!(image.frame_index_at(Duration::from_millis(100)), 1);
        assert_eq!(image.frame_index_at(Duration::from_millis(299)), 2);
        assert_eq!(image.frame_index_at(Duration::from_millis(300)), 0);
    }

    /// A real animated GIF, encoded here so the input is a file and not a
    /// fixture path.
    #[test]
    fn an_animated_gif_decodes_every_frame_with_its_delay() {
        let mut bytes = Vec::new();
        {
            let mut encoder = image::codecs::gif::GifEncoder::new(Cursor::new(&mut bytes));
            encoder
                .set_repeat(image::codecs::gif::Repeat::Infinite)
                .expect("setting the repeat count must succeed");
            for index in 0..3u8 {
                let mut buffer = image::RgbaImage::new(4, 4);
                for pixel in buffer.pixels_mut() {
                    *pixel = image::Rgba([index * 80, 0, 0, 0xFF]);
                }
                encoder
                    .encode_frame(image::Frame::from_parts(
                        buffer,
                        0,
                        0,
                        image::Delay::from_numer_denom_ms(100, 1),
                    ))
                    .expect("encoding a GIF frame must succeed");
            }
        }

        let image = decode(&bytes).expect("a real GIF decodes");
        assert_eq!(image.frame_count(), 3, "every frame, not just the first");
        assert!(image.is_animated());
        assert_eq!(image.natural_size(), [4, 4]);
        for (index, frame) in image.frames().iter().enumerate() {
            assert_eq!(frame.size, [4, 4]);
            assert_eq!(
                frame.delay,
                Duration::from_millis(100),
                "frame {index}'s delay must survive the decode"
            );
        }
        // The frames are distinguishable, which is what makes "every frame" a
        // real claim rather than three copies of one.
        assert_ne!(image.frame(0), image.frame(1));
        assert_eq!(
            image.frame(3),
            image.frame(0),
            "the index wraps, so a caller ticking it loops"
        );
    }
}
