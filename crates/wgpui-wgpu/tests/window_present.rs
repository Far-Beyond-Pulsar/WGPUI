//! **Phase 6's gate**: the frame a display would scan out is byte-exact.
//!
//! Every phase through 5.6 proved its work by reading back an *offscreen*
//! texture. This file does the same comparison one level up, against the actual
//! swapchain image of an actual window, read back immediately before it is
//! presented. That is possible because `WindowSurface` configures the surface
//! with `TextureUsages::COPY_SRC` and because the swapchain's format is
//! `TARGET_FORMAT` itself — both checked rather than assumed, see
//! `src/window.rs`'s module doc.
//!
//! It is therefore **not** the offscreen-parallel-render fallback: nothing here
//! renders the scene twice and compares the copies. The bytes compared are the
//! bytes in the image handed to `Queue::present`.
//!
//! # Why this is one test function and not eight
//!
//! A process may create one `winit::EventLoop`, and a created one cannot be
//! recreated after it exits. Cargo runs tests on spawned threads in one
//! process, so eight `#[test]`s that each wanted a window would give seven
//! failures and no information. The checks below are therefore sequenced inside
//! one event loop, each collecting its own failures rather than panicking, so a
//! run reports every check's verdict instead of the first one that tripped.
//!
//! # What this covers that the example does not, and the reverse
//!
//! The resize checks here call [`WindowSurface::resize`] directly — one layer
//! below a `WindowEvent::Resized`, and deterministic. Driving a *real* window
//! manager resize needs the event loop to keep pumping and depends on what the
//! WM does with the request, which is right for a harness a human runs and
//! wrong for a test. `examples/phase6_window.rs --resize` is that harness, and
//! Phase 6's report cites a genuine mouse-drag run of it (156 real resize
//! events) alongside these.
//!
//! # If there is no adapter, or no window
//!
//! Reports and returns, per Phase 0's standard. A headless CI container has
//! neither and must not be told it has coverage it does not have.

use std::sync::Arc;

use wgpui_core::invalidation::request::FrameSignals;
use wgpui_core::patch::primitive::{Glyph, GlyphRun};
use wgpui_core::scene::atlas::AtlasKind;
use wgpui_wgpu::render::atlas::{AtlasTileSource, GlyphAtlas, TilePlacement};
use wgpui_wgpu::render::atlas_upload::AtlasTextures;
use wgpui_wgpu::render::device::ComputeContext;
use wgpui_wgpu::render::draw::DrawMode;
use wgpui_wgpu::render::frame::RenderTarget;
use wgpui_wgpu::render::pipelines::TARGET_FORMAT;
use wgpui_wgpu::render::readback::read_texture_rgba8;
use wgpui_wgpu::window::frame_loop::{FrameLoop, LoopInput, ReferenceScene};
use wgpui_wgpu::window::{
    Acquired, PROOF_MAGENTA, PROOF_MAGENTA_BYTES, SurfaceFormatChoice, WindowSurface, clear_frame,
};

/// The size the byte-exact glyph comparison runs at.
///
/// Chosen so nothing flex-shrinks: the column's two children want 244px of 500,
/// so both get their described size and both land on whole pixels. A shrunk
/// layout puts glyphs at fractional positions, and a 1:1 texel blit is only
/// texel-exact on whole-pixel corners — see `docs/phase-5.6-results.md`, which
/// disclosed that and named `wgpui-text` as where the flooring belongs.
const EXACT_WIDTH: u32 = 800;
const EXACT_HEIGHT: u32 = 500;

const FILL: [f32; 4] = [64.0 / 255.0, 160.0 / 255.0, 240.0 / 255.0, 1.0];
const FILL_BYTES: [u8; 4] = [64, 160, 240, 255];
const FILL_SIZE: [f32; 2] = [320.0, 180.0];
const TEXT_HEIGHT: f32 = 64.0;
const TEXT: &str = "WGPUI 2.0 through the new stack";
const TEXT_ORIGIN: [f32; 2] = [16.0, 40.0];

/// Sizes the resize check drives, in order.
///
/// Down-then-up, twice, with a very small size in the middle. Shrinking is the
/// direction that frees swapchain images and is the documented sharp edge; a
/// script that only grew would test the easy half.
const RESIZE_SIZES: &[(u32, u32)] = &[
    (900, 560),
    (400, 260),
    (160, 100),
    (64, 48),
    (1100, 700),
    (200, 140),
    (1280, 800),
    (EXACT_WIDTH, EXACT_HEIGHT),
];

#[derive(Default)]
struct Report {
    lines: Vec<String>,
    failures: Vec<String>,
}

impl Report {
    fn note(&mut self, line: impl Into<String>) {
        let line = line.into();
        println!("  {line}");
        self.lines.push(line);
    }

    fn fail(&mut self, line: impl Into<String>) {
        let line = line.into();
        println!("  FAILED: {line}");
        self.failures.push(line);
    }

    fn check(&mut self, condition: bool, line: impl Into<String>) {
        if condition {
            self.note(line);
        } else {
            self.fail(line);
        }
    }
}

struct Harness {
    report: Report,
    done: bool,
}

impl winit::application::ApplicationHandler for Harness {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.done {
            return;
        }
        self.done = true;
        self.run(event_loop);
        event_loop.exit();
    }

    fn window_event(
        &mut self,
        _event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        _event: winit::event::WindowEvent,
    ) {
    }
}

impl Harness {
    fn run(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let attributes = winit::window::Window::default_attributes()
            .with_title("WGPUI 2.0 — Phase 6 gate")
            .with_inner_size(winit::dpi::PhysicalSize::new(EXACT_WIDTH, EXACT_HEIGHT));
        let window = match event_loop.create_window(attributes) {
            Ok(window) => Arc::new(window),
            Err(error) => {
                println!(
                    "window_present: SKIPPED — this session cannot create a window ({error}). \
                     Every offscreen gate in this crate still ran; the on-screen half did not, \
                     and a human must re-run this on a desktop session."
                );
                return;
            }
        };
        let (mut surface, context) = match WindowSurface::new(Arc::clone(&window)) {
            Ok(pair) => pair,
            Err(error) => {
                println!(
                    "window_present: SKIPPED — no presentable device for this window ({error}). \
                     The offscreen gates still ran; the on-screen half did not."
                );
                return;
            }
        };
        println!("window_present: running on {}", context.describe());

        self.check(
            surface.format() == TARGET_FORMAT,
            format!(
                "the swapchain is {:?}, the format every pipeline is compiled against",
                surface.format()
            ),
        );
        self.check(
            surface.format_choice() == SurfaceFormatChoice::Target,
            "the pipelines draw straight into the swapchain, with no blit in between",
        );

        self.clear_frames_are_exact(&mut surface, &context);
        self.resizing_keeps_presenting_correctly(&mut surface, &context);
        self.the_pipeline_reaches_the_swapchain(&mut surface, &context);

        let stats = surface.stats();
        self.check(
            stats.lost == 0,
            format!("no acquire was lost across the whole run ({stats:?})"),
        );
        self.check(
            stats.presents == stats.acquires,
            format!(
                "every acquired image was presented: {} acquired, {} presented",
                stats.acquires, stats.presents
            ),
        );
    }

    fn check(&mut self, condition: bool, line: impl Into<String>) {
        self.report.check(condition, line);
    }

    /// **Milestone A's gate.** Every pixel of every presented image is exactly
    /// the clear colour, read off the swapchain rather than off a copy.
    fn clear_frames_are_exact(&mut self, surface: &mut WindowSurface, context: &ComputeContext) {
        const FRAMES: u32 = 8;
        let mut verified = 0u32;
        let mut pixels_compared = 0usize;
        for _ in 0..FRAMES {
            let Acquired::Frame(texture) = surface.acquire(&context.device) else {
                continue;
            };
            let view = texture
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            clear_frame(&context.device, &context.queue, &view, PROOF_MAGENTA);
            let (width, height) = surface.size();
            match read_texture_rgba8(
                &context.device,
                &context.queue,
                &texture.texture,
                width,
                height,
            ) {
                Ok(pixels) => match first_pixel_other_than(&pixels, width, height, |_, _| {
                    PROOF_MAGENTA_BYTES
                }) {
                    Some(message) => self.report.fail(format!("clear frame: {message}")),
                    None => {
                        verified += 1;
                        pixels_compared += (width * height) as usize;
                    }
                },
                Err(error) => self.report.fail(format!("clear readback failed: {error}")),
            }
            surface.present(&context.queue, texture);
        }
        self.check(
            verified == FRAMES,
            format!(
                "{verified}/{FRAMES} presented images were exactly the clear colour \
                 ({pixels_compared} pixels compared)"
            ),
        );
    }

    /// **Milestone B's gate.** The surface reconfigures down and back up, and
    /// every size still presents correctly.
    ///
    /// Driven through [`WindowSurface::resize`] rather than a real
    /// `WindowEvent::Resized` — see this file's doc for why, and for where the
    /// real-event evidence lives.
    fn resizing_keeps_presenting_correctly(
        &mut self,
        surface: &mut WindowSurface,
        context: &ComputeContext,
    ) {
        let before = surface.stats();
        let mut sizes_verified = Vec::new();
        for &(width, height) in RESIZE_SIZES {
            if !surface.resize(&context.device, width, height) {
                self.report
                    .fail(format!("resize to {width}x{height} did not reconfigure"));
                continue;
            }
            // Two frames per size: the first exercises the freshly configured
            // swapchain, the second exercises a steady one at the new size.
            let mut good = 0;
            for _ in 0..2 {
                let Acquired::Frame(texture) = surface.acquire(&context.device) else {
                    continue;
                };
                let view = texture
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                clear_frame(&context.device, &context.queue, &view, PROOF_MAGENTA);
                let (actual_width, actual_height) = surface.size();
                match read_texture_rgba8(
                    &context.device,
                    &context.queue,
                    &texture.texture,
                    actual_width,
                    actual_height,
                ) {
                    Ok(pixels) => {
                        match first_pixel_other_than(
                            &pixels,
                            actual_width,
                            actual_height,
                            |_, _| PROOF_MAGENTA_BYTES,
                        ) {
                            Some(message) => self
                                .report
                                .fail(format!("after resize to {width}x{height}: {message}")),
                            None => good += 1,
                        }
                    }
                    Err(error) => self.report.fail(format!("resize readback failed: {error}")),
                }
                surface.present(&context.queue, texture);
            }
            if good == 2 {
                sizes_verified.push((width, height));
            }
        }
        let after = surface.stats();
        self.check(
            sizes_verified.len() == RESIZE_SIZES.len(),
            format!(
                "every size presented exactly the clear colour: {:?}",
                sizes_verified
            ),
        );
        self.check(
            after.configures - before.configures == RESIZE_SIZES.len() as u64,
            format!(
                "one `Surface::configure` per size change, no more: {} for {} sizes",
                after.configures - before.configures,
                RESIZE_SIZES.len()
            ),
        );
        self.check(
            after.lost == before.lost,
            "no acquire was lost across the resize sequence",
        );
    }

    /// **Milestones C and D's gate.** The whole pipeline runs into the
    /// swapchain, and the image that would reach the display is byte-exact.
    fn the_pipeline_reaches_the_swapchain(
        &mut self,
        surface: &mut WindowSurface,
        context: &ComputeContext,
    ) {
        let (runs, mut atlas) = match shape_reference_text() {
            Some(pair) => pair,
            None => {
                self.report.fail("the reference text could not be shaped");
                return;
            }
        };
        let glyph_count: usize = runs.iter().map(|run| run.glyphs.len()).sum();
        let mut textures = AtlasTextures::for_atlas(&atlas);
        let upload = textures.sync(&context.device, &context.queue, &mut atlas);
        self.check(
            upload.skipped == 0 && upload.rectangles > 10,
            format!(
                "the atlas holds real rasters: {} rectangles uploaded, {} skipped, {} glyphs shaped",
                upload.rectangles, upload.skipped, glyph_count
            ),
        );

        if !surface.resize(&context.device, EXACT_WIDTH, EXACT_HEIGHT) {
            self.report.note("already at the exact-comparison size");
        }
        let mut frame_loop = FrameLoop::new(&context.device);
        let reference = ReferenceScene {
            fill: FILL,
            fill_size: FILL_SIZE,
            text: runs,
            text_height: TEXT_HEIGHT,
            fingerprinted: true,
        };
        let mode = DrawMode::best_available(context.indirect);

        const FRAMES: u32 = 6;
        let mut idle = 0u32;
        let mut last_pixels = Vec::new();
        let mut resident = 0u32;
        let mut draw_calls = 0u32;
        for _ in 0..FRAMES {
            let Acquired::Frame(texture) = surface.acquire(&context.device) else {
                continue;
            };
            let view = texture
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            let (width, height) = surface.size();
            let target = RenderTarget {
                view: &view,
                width,
                height,
                // Black: Phase 5.6's white-on-black identity is what makes the
                // glyph comparison below an equality rather than a threshold.
                clear: wgpu::Color::BLACK,
            };
            match frame_loop.draw(
                &context.device,
                &context.queue,
                reference.describe(),
                &LoopInput {
                    atlas: Some(&textures),
                    target: &target,
                    mode,
                    signals: &FrameSignals::new(),
                    composites: &[],
                },
            ) {
                Ok(frame) => {
                    if frame.was_idle() {
                        idle += 1;
                    }
                    resident = frame.frame.primitives_resident;
                    draw_calls = frame.frame.stats.draw_calls_issued;
                }
                Err(error) => {
                    self.report.fail(format!("FrameLoop::draw failed: {error}"));
                    surface.present(&context.queue, texture);
                    return;
                }
            }
            match read_texture_rgba8(
                &context.device,
                &context.queue,
                &texture.texture,
                width,
                height,
            ) {
                Ok(pixels) => last_pixels = pixels,
                Err(error) => self.report.fail(format!("scene readback failed: {error}")),
            }
            surface.present(&context.queue, texture);
        }

        self.check(
            resident as usize == glyph_count + 1 && draw_calls >= 2,
            format!(
                "the scene is resident and both kinds drew: {resident} primitives \
                 ({glyph_count} glyphs + 1 quad), {draw_calls} draw calls"
            ),
        );
        self.check(
            idle == FRAMES - 1,
            format!(
                "a fingerprinted scene settles: {idle} of {FRAMES} frames changed nothing \
                 (the first builds it)"
            ),
        );
        self.check(
            frame_loop.draw_plan_builds() == 1 && frame_loop.glyph_plan_builds() == 1,
            format!(
                "the slot bases were built once each across {FRAMES} frames, not per frame: \
                 {}/{}",
                frame_loop.draw_plan_builds(),
                frame_loop.glyph_plan_builds()
            ),
        );

        if last_pixels.is_empty() {
            self.report.fail("no presented image was read back");
            return;
        }

        // --- The quad: exactly its emitted rectangle, exactly its emitted
        // colour. Read out of the scene rather than compared against
        // `FILL_SIZE`, because at a size where the column flex-shrinks those
        // two differ and the emitted one is the right answer.
        let quads = frame_loop.resident_quads();
        match quads.as_slice() {
            [quad] => {
                let left = quad.origin[0] as u32;
                let top = quad.origin[1] as u32;
                let right = ((quad.origin[0] + quad.size[0]) as u32).min(EXACT_WIDTH);
                let bottom = ((quad.origin[1] + quad.size[1]) as u32).min(EXACT_HEIGHT);
                let mut wrong = None;
                let mut compared = 0usize;
                for y in top..bottom {
                    for x in left..right {
                        let found = pixel(&last_pixels, EXACT_WIDTH, x, y);
                        if found != FILL_BYTES {
                            wrong = Some(format!("({x}, {y}) is {found:?}"));
                            break;
                        }
                        compared += 1;
                    }
                    if wrong.is_some() {
                        break;
                    }
                }
                match wrong {
                    Some(where_) => self.report.fail(format!(
                        "the quad's pixels are not its colour: {where_}, expected {FILL_BYTES:?}"
                    )),
                    None => self.report.note(format!(
                        "every one of the quad's {compared} pixels is exactly {FILL_BYTES:?}, \
                         over its emitted rectangle {:?} at {:?}",
                        quad.size, quad.origin
                    )),
                }
                // Bounds, both edges, exactly: one pixel further out is not the
                // fill. This is what separates "the right colour somewhere"
                // from "the right colour in the right place".
                let spills_right = right < EXACT_WIDTH
                    && pixel(&last_pixels, EXACT_WIDTH, right, (top + bottom) / 2) == FILL_BYTES;
                let spills_down = bottom < EXACT_HEIGHT
                    && pixel(&last_pixels, EXACT_WIDTH, (left + right) / 2, bottom) == FILL_BYTES;
                self.check(
                    !spills_right && !spills_down,
                    format!("the quad ends exactly at x={right}, y={bottom} and no further"),
                );
                self.check(
                    quad.background == FILL && quad.size == FILL_SIZE,
                    format!(
                        "the emitted quad is what the description asked for: {:?} at {:?}",
                        quad.size, quad.origin
                    ),
                );
            }
            other => self
                .report
                .fail(format!("the scene holds {} quads, expected 1", other.len())),
        }

        // --- The text: every glyph's own atlas texels, at its own position,
        // byte for byte. Phase 5.6's comparison, against the presented image.
        let glyphs: Vec<Glyph> = frame_loop
            .resident_glyphs()
            .into_iter()
            .map(|(glyph, _)| glyph)
            .collect();
        let comparison = compare_glyph_texels(&glyphs, &atlas, &last_pixels);
        self.report.note(format!(
            "glyph texel comparison: {} compared, {} inked, {} skipped as shared between \
             overlapping rasters",
            comparison.compared, comparison.inked, comparison.shared
        ));
        match comparison.mismatch {
            Some(message) => self.report.fail(message),
            None => self.check(
                comparison.compared > 1_500 && comparison.inked * 4 > comparison.compared,
                format!(
                    "every one of {} presented texels is its own atlas tile's byte, {} of them \
                     inked — a comparison that were mostly blank would pass on an empty frame",
                    comparison.compared, comparison.inked
                ),
            ),
        }
    }
}

fn pixel(pixels: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
    let offset = ((y * width + x) * 4) as usize;
    match pixels.get(offset..offset + 4) {
        Some(bytes) => [bytes[0], bytes[1], bytes[2], bytes[3]],
        None => [0, 0, 0, 0],
    }
}

fn first_pixel_other_than(
    pixels: &[u8],
    width: u32,
    height: u32,
    expected: impl Fn(u32, u32) -> [u8; 4],
) -> Option<String> {
    let wanted = (width as usize) * (height as usize) * 4;
    if pixels.len() != wanted {
        return Some(format!(
            "the readback is {} bytes, expected {wanted} for {width}x{height}",
            pixels.len()
        ));
    }
    for y in 0..height {
        for x in 0..width {
            let found = pixel(pixels, width, x, y);
            let wanted = expected(x, y);
            if found != wanted {
                return Some(format!(
                    "pixel ({x}, {y}) of the presented {width}x{height} image is {found:?}, \
                     expected {wanted:?}"
                ));
            }
        }
    }
    None
}

/// What one exact comparison of every glyph against its own tile found.
#[derive(Default, Debug)]
struct TexelComparison {
    compared: usize,
    inked: usize,
    shared: usize,
    mismatch: Option<String>,
}

/// How many glyph rectangles cover each pixel.
///
/// Adjacent rasters genuinely overlap — a raster is wider than its advance
/// wherever a letter leans — and where two do, the framebuffer holds the blend
/// of both, which no single tile's texels describe. Those pixels are skipped.
/// **That is a restriction on what can be checked, not a tolerance**: on the
/// pixels it does check, equality is still exact. Phase 5.6 found this the same
/// way, by a comparison failing at one shared column.
fn glyph_coverage(glyphs: &[Glyph]) -> Vec<u8> {
    let mut coverage = vec![0u8; (EXACT_WIDTH * EXACT_HEIGHT) as usize];
    for glyph in glyphs {
        if glyph.atlas_tile.is_none() {
            continue;
        }
        for row in 0..glyph.atlas_size[1] as u32 {
            for column in 0..glyph.atlas_size[0] as u32 {
                let x = glyph.position[0] as u32 + column;
                let y = glyph.position[1] as u32 + row;
                if x >= EXACT_WIDTH || y >= EXACT_HEIGHT {
                    continue;
                }
                if let Some(count) = coverage.get_mut((y * EXACT_WIDTH + x) as usize) {
                    *count = count.saturating_add(1);
                }
            }
        }
    }
    coverage
}

fn compare_glyph_texels(
    glyphs: &[Glyph],
    atlas: &GlyphAtlas,
    pixels: &[u8],
) -> TexelComparison {
    let coverage = glyph_coverage(glyphs);
    let mut result = TexelComparison::default();
    for glyph in glyphs {
        if glyph.atlas_tile.is_none() {
            continue;
        }
        let Some(texels) = atlas.tile_texels(TilePlacement {
            tile: glyph.atlas_tile,
            kind: AtlasKind::Monochrome,
            origin: glyph.atlas_origin,
            size: glyph.atlas_size,
            bearing: [0.0, 0.0],
        }) else {
            result.mismatch = Some(format!(
                "glyph {} names tile {:?}, which is not resident in the atlas",
                glyph.glyph_id, glyph.atlas_tile
            ));
            return result;
        };
        let width = glyph.atlas_size[0] as u32;
        if width == 0 {
            continue;
        }
        for (index, expected) in texels.iter().enumerate() {
            let x = glyph.position[0] as u32 + index as u32 % width;
            let y = glyph.position[1] as u32 + index as u32 / width;
            if x >= EXACT_WIDTH || y >= EXACT_HEIGHT {
                continue;
            }
            if coverage
                .get((y * EXACT_WIDTH + x) as usize)
                .copied()
                .unwrap_or(0)
                != 1
            {
                result.shared += 1;
                continue;
            }
            let found = pixel(pixels, EXACT_WIDTH, x, y);
            if [found[0], found[1], found[2]] != [*expected; 3] {
                result.mismatch = Some(format!(
                    "glyph {} should have put texel {index} on the presented image at \
                     ({x}, {y}) as {expected}, found {found:?}",
                    glyph.glyph_id
                ));
                return result;
            }
            // The target's alpha is not the coverage, and asserting that it
            // were would assert the wrong thing: the pass clears to opaque
            // black and the alpha blend is `One`/`OneMinusSrcAlpha`, so
            // `a = srcAlpha + 1 - srcAlpha` is 1 whatever the coverage was.
            // Phase 5.6 records this as one of the wrong beliefs its own gate
            // corrected.
            if found[3] != 255 {
                result.mismatch = Some(format!(
                    "the presented image lost its opacity at ({x}, {y}): alpha {}",
                    found[3]
                ));
                return result;
            }
            result.compared += 1;
            if *expected > 0 {
                result.inked += 1;
            }
        }
    }
    result
}

/// Shape and rasterise the reference line through the real text path.
///
/// Positions are rounded for the reason Phase 5.6's gate rounds them: a glyph
/// sprite is a 1:1 texel blit, exact only on whole-pixel corners, and
/// `wgpui_text::patch::glyph_runs` keeps the fractional pen position because the
/// atlas already carries the fraction as a sub-pixel variant.
fn shape_reference_text() -> Option<(Vec<GlyphRun>, GlyphAtlas)> {
    use wgpui_text::patch::{RunPlacement, glyph_runs};
    use wgpui_text::raster::GlyphRasterizer;
    use wgpui_text::shaping::{FontRun, SharedString, font};

    let mut shaper = wgpui_text::test_fonts::shaper();
    let font_id = shaper
        .resolve_font(&font(wgpui_text::test_fonts::FAMILY))
        .ok()?;
    let text = SharedString::from(TEXT);
    let line = shaper
        .shape_line(&text, 24.0, &[FontRun::new(text.len(), font_id)])
        .ok()?;

    let mut atlas = GlyphAtlas::new(512);
    let mut rasterizer = GlyphRasterizer::new();
    let runs = {
        let mut source = AtlasTileSource::new(&mut atlas, |key| {
            rasterizer.rasterize(&mut shaper, key).ok()
        });
        glyph_runs(
            &line,
            RunPlacement {
                origin: TEXT_ORIGIN,
                color: [1.0, 1.0, 1.0, 1.0],
                scale_factor: 1.0,
            },
            &mut source,
        )
        .0
    };
    let runs = runs
        .into_iter()
        .map(|mut run| {
            for glyph in &mut run.glyphs {
                glyph.position = [glyph.position[0].round(), glyph.position[1].round()];
            }
            run
        })
        .collect();
    Some((runs, atlas))
}

/// An event loop that can be built from a test thread, where the platform
/// allows it.
///
/// **`winit::EventLoop::new()` does not return an error off the main thread —
/// it panics**, with a message calling the situation "a significant
/// cross-platform compatibility hazard". That is a correct default and it is
/// also a real constraint on this gate: `cargo test` runs every test on a
/// spawned thread, so the escape hatch is the only way a window test exists at
/// all.
///
/// Windows and the two Linux backends provide one. **macOS does not, and cannot
/// — AppKit genuinely requires the main thread** — so on macOS this returns
/// `None` and the test reports itself skipped rather than pretending. The
/// on-screen gate there is `examples/phase6_window.rs`, which runs on the main
/// thread like any binary. Stating that plainly is the point: §11's action 2
/// already carries "every number so far is one machine", and this is one more
/// axis on which that is true.
fn event_loop_for_test() -> Option<winit::event_loop::EventLoop<()>> {
    #[cfg(target_os = "windows")]
    {
        use winit::platform::windows::EventLoopBuilderExtWindows;
        winit::event_loop::EventLoop::builder()
            .with_any_thread(true)
            .build()
            .ok()
    }
    #[cfg(target_os = "linux")]
    {
        use winit::platform::wayland::EventLoopBuilderExtWayland;
        use winit::platform::x11::EventLoopBuilderExtX11;
        let mut builder = winit::event_loop::EventLoop::builder();
        EventLoopBuilderExtX11::with_any_thread(&mut builder, true);
        EventLoopBuilderExtWayland::with_any_thread(&mut builder, true);
        builder.build().ok()
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        None
    }
}

#[test]
fn the_presented_frame_is_byte_exact() {
    let event_loop = match event_loop_for_test() {
        Some(event_loop) => event_loop,
        None => {
            println!(
                "window_present: SKIPPED — no winit event loop is available from a test thread \
                 on this platform. Every offscreen gate in this crate still ran; the on-screen \
                 half did not. Run `cargo run -p wgpui-wgpu --example phase6_window -- \
                 --scene --verify --frames 8`, which runs on the main thread and gates the \
                 same claims."
            );
            return;
        }
    };
    let mut harness = Harness {
        report: Report::default(),
        done: false,
    };
    if let Err(error) = event_loop.run_app(&mut harness) {
        panic!("the event loop failed: {error}");
    }
    assert!(
        harness.report.failures.is_empty(),
        "{} of {} checks failed:\n  {}",
        harness.report.failures.len(),
        harness.report.failures.len() + harness.report.lines.len(),
        harness.report.failures.join("\n  ")
    );
    assert!(
        harness.report.lines.len() >= 10,
        "only {} checks ran, which is not the gate",
        harness.report.lines.len()
    );
}
