//! Phase 6's runnable entry point: a real window, a real swapchain, a real
//! present loop. See docs/gpu-native-architecture.md §11's action 1.
//!
//! Run it with no arguments and a magenta window appears and stays up until it
//! is closed. Every other mode exists because a phase report has to be able to
//! cite something other than "it looked fine":
//!
//! ```text
//! cargo run -p wgpui-wgpu --example phase6_window
//! cargo run -p wgpui-wgpu --example phase6_window -- --frames 240
//! cargo run -p wgpui-wgpu --example phase6_window -- --hold-ms 6000
//! cargo run -p wgpui-wgpu --example phase6_window -- --resize --frames 400
//! cargo run -p wgpui-wgpu --example phase6_window -- --verify --frames 8
//! ```
//!
//! `--verify` reads the *presented swapchain image itself* back and compares it
//! byte for byte, which this surface supports because it is configured with
//! `COPY_SRC`. It is checked before every present rather than after, since a
//! presented image is the compositor's again.
//!
//! The process exits non-zero if any assertion fails, so a run of this file is
//! a gate rather than a demonstration.

use std::sync::Arc;
use std::time::{Duration, Instant};

use wgpui_core::invalidation::request::FrameSignals;
use wgpui_core::patch::primitive::GlyphRun;
use wgpui_wgpu::render::atlas::{AtlasTileSource, GlyphAtlas};
use wgpui_wgpu::render::atlas_upload::AtlasTextures;
use wgpui_wgpu::render::device::ComputeContext;
use wgpui_wgpu::render::draw::DrawMode;
use wgpui_wgpu::render::frame::RenderTarget;
use wgpui_wgpu::render::readback::read_texture_rgba8;
use wgpui_wgpu::window::frame_loop::{FrameLoop, LoopInput, ReferenceScene};
use wgpui_wgpu::window::resize_detector::ResizeDetector;
use wgpui_wgpu::window::{
    Acquired, PROOF_MAGENTA, PROOF_MAGENTA_BYTES, SurfaceStats, WindowSurface, clear_frame,
};

const INITIAL_WIDTH: u32 = 800;
const INITIAL_HEIGHT: u32 = 500;

/// The reference quad's fill.
///
/// Each component is an exact multiple of 1/255, so an opaque quad written to
/// `Rgba8Unorm` reads back as exactly `[64, 160, 240, 255]` — see
/// [`ReferenceScene::fill`] for why an arbitrary float would not.
const FILL: [f32; 4] = [64.0 / 255.0, 160.0 / 255.0, 240.0 / 255.0, 1.0];
const FILL_BYTES: [u8; 4] = [64, 160, 240, 255];
const FILL_SIZE: [f32; 2] = [320.0, 180.0];
const TEXT_HEIGHT: f32 = 64.0;
const TEXT: &str = "WGPUI 2.0 through the new stack";
const TEXT_SIZE: f32 = 24.0;
/// Where the shaped line's pen starts, relative to the text element's bounds.
const TEXT_ORIGIN: [f32; 2] = [16.0, 40.0];

/// A scripted resize sequence, in physical pixels.
///
/// Down-then-up on purpose, and repeated: shrinking is the direction that frees
/// swapchain images and is the documented sharp edge in surface reconfiguration
/// on several backends, so a script that only grew would be testing the easy
/// half. The last entry restores the starting size so a run leaves the window
/// where it found it.
const RESIZE_SCRIPT: &[(u32, u32)] = &[
    (900, 560),
    (640, 400),
    (320, 200),
    (160, 100),
    (1100, 700),
    (200, 140),
    (1280, 800),
    (240, 160),
    (INITIAL_WIDTH, INITIAL_HEIGHT),
];

/// How many frames to present at each scripted size before moving on.
const FRAMES_PER_RESIZE_STEP: u32 = 6;

#[derive(Clone, Debug)]
struct Options {
    frames: Option<u32>,
    hold: Option<Duration>,
    resize: bool,
    verify: bool,
    scene: bool,
    no_fingerprint: bool,
}

impl Options {
    fn parse() -> Result<Options, String> {
        let mut options = Options {
            frames: None,
            hold: None,
            resize: false,
            verify: false,
            scene: false,
            no_fingerprint: false,
        };
        let mut arguments = std::env::args().skip(1);
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--frames" => {
                    let value = arguments.next().ok_or("--frames needs a count")?;
                    options.frames = Some(value.parse::<u32>().map_err(|error| error.to_string())?);
                }
                "--hold-ms" => {
                    let value = arguments.next().ok_or("--hold-ms needs a duration")?;
                    options.hold = Some(Duration::from_millis(
                        value.parse::<u64>().map_err(|error| error.to_string())?,
                    ));
                }
                "--resize" => options.resize = true,
                "--verify" => options.verify = true,
                "--scene" => options.scene = true,
                "--no-fingerprint" => options.no_fingerprint = true,
                other => return Err(format!("unrecognised argument {other:?}")),
            }
        }
        if options.resize && options.frames.is_none() {
            options.frames = Some(RESIZE_SCRIPT.len() as u32 * FRAMES_PER_RESIZE_STEP + 12);
        }
        Ok(options)
    }
}

/// What the run observed, checked at exit rather than asserted mid-loop, so a
/// failure reports every number rather than the first one that tripped.
#[derive(Default)]
struct Observed {
    frames_drawn: u32,
    frames_verified: u32,
    verify_failures: Vec<String>,
    resize_steps_dispatched: usize,
    resizes_answered_synchronously: usize,
    /// `--scene` only: frames where the patch was empty, no layer was dirty,
    /// and nothing was uploaded. A steady window's every frame after the first.
    idle_frames: u32,
    /// `--scene` only: bytes the patches scheduled for GPU upload across the
    /// whole run. The measurable half of the fingerprint finding — a settled
    /// window should add nothing to this after its first frame.
    uploaded_bytes: u64,
    /// `--scene` only: frames recomputed in full because the viewport moved
    /// rather than because the patch named a layer. One per distinct size
    /// presented, and never more — see `FrameLoop::draw`'s note on why the clip
    /// is dirtiness the patch cannot report.
    viewport_recomputes: u64,
    last_frame: Option<String>,
    sizes_presented: Vec<(u32, u32)>,
    surface: SurfaceStats,
    resize_events_seen: u64,
    resize_reconfigurations: u64,
    failure: Option<String>,
}

struct App {
    options: Options,
    started: Instant,
    live: Option<Live>,
    observed: Observed,
}

struct Live {
    surface: WindowSurface,
    context: ComputeContext,
    resizes: ResizeDetector,
    /// Index of the next `RESIZE_SCRIPT` entry to request.
    next_resize_step: usize,
    /// Frames drawn since the last scripted resize was requested.
    frames_since_resize: u32,
    /// `--scene` only: the whole pipeline, and the atlas its text samples.
    scene: Option<SceneState>,
}

struct SceneState {
    frame_loop: FrameLoop,
    atlas: AtlasTextures,
    reference: ReferenceScene,
    mode: DrawMode,
}

impl winit::application::ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.live.is_some() {
            return;
        }
        let attributes = winit::window::Window::default_attributes()
            .with_title("WGPUI 2.0 — Phase 6")
            .with_inner_size(winit::dpi::PhysicalSize::new(INITIAL_WIDTH, INITIAL_HEIGHT));
        let window = match event_loop.create_window(attributes) {
            Ok(window) => Arc::new(window),
            Err(error) => {
                self.observed.failure = Some(format!("create_window failed: {error}"));
                event_loop.exit();
                return;
            }
        };
        let (surface, context) = match WindowSurface::new(Arc::clone(&window)) {
            Ok(pair) => pair,
            Err(error) => {
                self.observed.failure = Some(format!("WindowSurface::new failed: {error}"));
                event_loop.exit();
                return;
            }
        };
        println!("device:  {}", context.describe());
        println!(
            "surface: {:?} {:?} {:?} at {}x{}",
            surface.format(),
            surface.format_choice(),
            surface.present_mode(),
            surface.size().0,
            surface.size().1,
        );

        let mut resizes = ResizeDetector::new();
        let (width, height) = surface.size();
        resizes.seed(width, height);

        let scene = if self.options.scene {
            let (runs, mut atlas) = shape_reference_text();
            let mut textures = AtlasTextures::for_atlas(&atlas);
            let upload = textures.sync(&context.device, &context.queue, &mut atlas);
            println!(
                "atlas: {} rectangle(s) uploaded, {} skipped, {} page(s)",
                upload.rectangles,
                upload.skipped,
                textures.page_count()
            );
            Some(SceneState {
                frame_loop: FrameLoop::new(&context.device),
                atlas: textures,
                reference: ReferenceScene {
                    fill: FILL,
                    fingerprinted: !self.options.no_fingerprint,
                    fill_size: FILL_SIZE,
                    text: runs,
                    text_height: TEXT_HEIGHT,
                },
                mode: DrawMode::best_available(context.indirect),
            })
        } else {
            None
        };

        self.live = Some(Live {
            surface,
            context,
            resizes,
            next_resize_step: 0,
            frames_since_resize: 0,
            scene,
        });
        window.request_redraw();
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let Some(live) = self.live.as_mut() else {
            return;
        };
        match event {
            winit::event::WindowEvent::CloseRequested => event_loop.exit(),
            winit::event::WindowEvent::Resized(size) => {
                // The real event, straight off winit, with no synthetic size
                // invented anywhere. `--resize` drives `request_inner_size`;
                // this is where the window manager's answer comes back.
                live.resizes.on_resize_event(size.width, size.height);
                live.surface.window().request_redraw();
            }
            winit::event::WindowEvent::RedrawRequested => self.draw(event_loop),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Some(live) = self.live.as_ref() {
            live.surface.window().request_redraw();
        }
    }
}

impl App {
    fn draw(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let Some(live) = self.live.as_mut() else {
            return;
        };

        if let Some((width, height)) = live.resizes.take_pending() {
            live.surface.resize(&live.context.device, width, height);
        }

        let texture = match live.surface.acquire(&live.context.device) {
            Acquired::Frame(texture) => texture,
            Acquired::Skipped(_) => {
                self.finish_if_done(event_loop);
                return;
            }
            Acquired::Lost => {
                self.observed.failure =
                    Some("swapchain acquire failed even after a reconfigure".to_string());
                event_loop.exit();
                return;
            }
        };
        let view = texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let (width, height) = live.surface.size();

        // The one place the two modes differ. `--scene` runs the real pipeline
        // — reconcile, lay out, emit, apply, compute, indirect draw — straight
        // into the swapchain view; the default clears it. Both write into the
        // same acquired image and both are presented by the same call below.
        let mut idle_frames = 0u32;
        match live.scene.as_mut() {
            Some(scene) => {
                let target = RenderTarget {
                    view: &view,
                    width,
                    height,
                    // Black, so Phase 5.6's white-on-black identity holds and
                    // the text's pixels are its atlas texels exactly.
                    clear: wgpu::Color::BLACK,
                    source: Some(&texture.texture),
                };
                match scene.frame_loop.draw(
                    &live.context.device,
                    &live.context.queue,
                    scene.reference.describe(),
                    &LoopInput {
                        atlas: Some(&scene.atlas),
                        target: &target,
                        mode: scene.mode,
                        signals: &FrameSignals::new(),
                        composites: &[],
                    },
                ) {
                    Ok(frame) => {
                        self.observed.uploaded_bytes += frame.uploaded_bytes;
                        if frame.was_idle() {
                            idle_frames = 1;
                        }
                        self.observed.viewport_recomputes = scene.frame_loop.viewport_recomputes();
                        self.observed.last_frame = Some(format!(
                            "resident={} draws={} slots={} plan_builds={}/{}",
                            frame.frame.primitives_resident,
                            frame.frame.stats.draw_calls_issued,
                            frame.frame.stats.slots_visited,
                            scene.frame_loop.draw_plan_builds(),
                            scene.frame_loop.glyph_plan_builds(),
                        ));
                    }
                    Err(error) => {
                        self.observed.failure = Some(format!("FrameLoop::draw failed: {error}"));
                        event_loop.exit();
                        return;
                    }
                }
            }
            None => clear_frame(
                &live.context.device,
                &live.context.queue,
                &view,
                PROOF_MAGENTA,
            ),
        }
        self.observed.idle_frames += idle_frames;

        if self.options.verify {
            match read_texture_rgba8(
                &live.context.device,
                &live.context.queue,
                &texture.texture,
                width,
                height,
            ) {
                Ok(pixels) => {
                    self.observed.frames_verified += 1;
                    let wrong = match live.scene.as_ref() {
                        Some(scene) => wrong_scene_pixel(
                            &pixels,
                            width,
                            height,
                            &scene.frame_loop.resident_quads(),
                            &scene.frame_loop.resident_glyphs(),
                        ),
                        None => wrong_pixel(&pixels, width, height),
                    };
                    if let Some(message) = wrong {
                        self.observed.verify_failures.push(format!(
                            "frame {} at {width}x{height}: {message}",
                            self.observed.frames_drawn
                        ));
                    }
                }
                Err(error) => self
                    .observed
                    .verify_failures
                    .push(format!("readback failed: {error}")),
            }
        }

        live.surface.present(&live.context.queue, texture);
        self.observed.frames_drawn += 1;
        if self.observed.sizes_presented.last() != Some(&(width, height)) {
            self.observed.sizes_presented.push((width, height));
        }

        if self.options.resize {
            live.frames_since_resize += 1;
            if live.frames_since_resize >= FRAMES_PER_RESIZE_STEP
                && let Some(&(width, height)) = RESIZE_SCRIPT.get(live.next_resize_step)
            {
                // `request_inner_size` asks the window manager; the answer
                // arrives as a real `WindowEvent::Resized`, which is what the
                // surface actually reconfigures on. Some platforms also answer
                // synchronously, and winit's return says which happened —
                // recorded rather than discarded, because "the resize was
                // applied immediately" and "the resize came back as an event"
                // are different claims and the report makes one of them.
                let answered = live
                    .surface
                    .window()
                    .request_inner_size(winit::dpi::PhysicalSize::new(width, height));
                if answered.is_some() {
                    self.observed.resizes_answered_synchronously += 1;
                }
                live.next_resize_step += 1;
                live.frames_since_resize = 0;
                self.observed.resize_steps_dispatched += 1;
            }
        }

        self.finish_if_done(event_loop);
    }

    fn finish_if_done(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let done = match (self.options.frames, self.options.hold) {
            (Some(frames), _) => self.observed.frames_drawn >= frames,
            (None, Some(hold)) => self.started.elapsed() >= hold,
            (None, None) => false,
        };
        if done {
            if let Some(live) = self.live.as_ref() {
                self.observed.surface = live.surface.stats();
                self.observed.resize_events_seen = live.resizes.events_seen();
                self.observed.resize_reconfigurations = live.resizes.reconfigurations();
            }
            event_loop.exit();
        }
    }
}

/// Shape and rasterise the reference line through the real text path.
///
/// Positions are rounded to whole pixels for the same reason Phase 5.6's gate
/// rounds them: a glyph sprite is a 1:1 texel blit, which is only texel-exact
/// when the quad's corners are whole pixels, and
/// `wgpui_text::patch::glyph_runs` keeps the fractional pen position because
/// the atlas already carries the fraction as a sub-pixel variant. The flooring
/// belongs in `wgpui-text`; §11's action 4 still carries it as open.
fn shape_reference_text() -> (Vec<GlyphRun>, GlyphAtlas) {
    use wgpui_text::patch::{RunPlacement, glyph_runs};
    use wgpui_text::raster::GlyphRasterizer;
    use wgpui_text::shaping::{FontRun, SharedString, font};

    let mut shaper = wgpui_text::test_fonts::shaper();
    let font_id = match shaper.resolve_font(&font(wgpui_text::test_fonts::FAMILY)) {
        Ok(font_id) => font_id,
        Err(error) => {
            eprintln!("phase6_window: the embedded face did not resolve: {error:?}");
            return (Vec::new(), GlyphAtlas::new(512));
        }
    };
    let text = SharedString::from(TEXT);
    let line = match shaper.shape_line(&text, TEXT_SIZE, &[FontRun::new(text.len(), font_id)]) {
        Ok(line) => line,
        Err(error) => {
            eprintln!("phase6_window: shaping failed: {error:?}");
            return (Vec::new(), GlyphAtlas::new(512));
        }
    };

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
                // White, so the straight-alpha `over` blend over a black clear
                // reduces to the identity Phase 5.6's proof rests on.
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
    (runs, atlas)
}

/// The reference scene's expected framebuffer, checked against the presented
/// image.
///
/// **Checked against the primitives the scene actually holds, not against the
/// constants they were described with.** Those two are the same thing at
/// 800x500 and are *not* the same thing at 320x200: the column's two children
/// want 244px of a 200px box, taffy's default `flex_shrink` applies, and the
/// quad is legitimately 147px tall. The first version of this function compared
/// against `FILL_SIZE` and failed on exactly those frames — with the renderer
/// right and the check wrong. Reading the emitted quad back is what makes
/// "matches what was described" mean the whole chain rather than one constant.
///
/// Three claims, and each fails differently:
///
/// 1. **The quad is exactly its emitted rectangle, in exactly its emitted
///    colour** — equality, not a tolerance.
/// 2. **The quad does not spill**: one column past its right edge and one row
///    past its bottom edge are not the fill.
/// 3. **The text is present, and only inside the rows its glyphs claim.**
///
/// The byte-exact glyph-texel comparison — every glyph's own tile, at its own
/// position — lives in `tests/window_present.rs`, which carries the atlas's
/// texels alongside. This is the cheap check that runs every presented frame.
fn wrong_scene_pixel(
    pixels: &[u8],
    width: u32,
    height: u32,
    quads: &[wgpui_core::patch::primitive::Quad],
    glyphs: &[(wgpui_core::patch::primitive::Glyph, [f32; 4])],
) -> Option<String> {
    let expected = (width as usize) * (height as usize) * 4;
    if pixels.len() != expected {
        return Some(format!(
            "readback is {} bytes, expected {expected}",
            pixels.len()
        ));
    }
    let at = |x: u32, y: u32| -> [u8; 4] {
        let offset = ((y * width + x) * 4) as usize;
        match pixels.get(offset..offset + 4) {
            Some(bytes) => [bytes[0], bytes[1], bytes[2], bytes[3]],
            None => [0, 0, 0, 0],
        }
    };

    let [quad] = quads else {
        return Some(format!(
            "the scene holds {} quads, expected exactly the reference fill",
            quads.len()
        ));
    };
    if quad.background != FILL {
        return Some(format!(
            "the emitted quad's background is {:?}, expected {FILL:?}",
            quad.background
        ));
    }
    let left = quad.origin[0].round().max(0.0) as u32;
    let top = quad.origin[1].round().max(0.0) as u32;
    let right = ((quad.origin[0] + quad.size[0]).round().max(0.0) as u32).min(width);
    let bottom = ((quad.origin[1] + quad.size[1]).round().max(0.0) as u32).min(height);
    for y in top..bottom {
        for x in left..right {
            if at(x, y) != FILL_BYTES {
                return Some(format!(
                    "quad pixel ({x}, {y}) is {:?}, expected {FILL_BYTES:?}; the emitted quad is \
                     {:?} at {:?}",
                    at(x, y),
                    quad.size,
                    quad.origin
                ));
            }
        }
    }
    if right < width && bottom > top && at(right, (top + bottom) / 2) == FILL_BYTES {
        return Some(format!(
            "the quad painted x={right}, one past the right edge it emitted"
        ));
    }
    if bottom < height && right > left && at((left + right) / 2, bottom) == FILL_BYTES {
        return Some(format!(
            "the quad painted y={bottom}, one past the bottom edge it emitted"
        ));
    }

    // The rows the glyphs themselves claim, taken from the emitted glyphs
    // rather than from `TEXT_HEIGHT` — at a shrunk size the text element is
    // shorter than it asked for, exactly as the quad is.
    let inked: Vec<&wgpui_core::patch::primitive::Glyph> = glyphs
        .iter()
        .map(|(glyph, _)| glyph)
        .filter(|glyph| glyph.atlas_size[0] > 0.0 && glyph.atlas_size[1] > 0.0)
        .collect();
    let Some(first) = inked.first() else {
        return Some("the scene holds no inked glyphs at all".to_string());
    };
    let mut text_top = first.position[1];
    let mut text_bottom = first.position[1] + first.atlas_size[1];
    for glyph in &inked {
        text_top = text_top.min(glyph.position[1]);
        text_bottom = text_bottom.max(glyph.position[1] + glyph.atlas_size[1]);
    }
    let text_top = (text_top.max(0.0) as u32).min(height);
    let text_bottom = ((text_bottom.ceil().max(0.0) as u32) + 1).min(height);

    let mut ink = 0usize;
    for y in text_top..text_bottom {
        for x in 0..width {
            if at(x, y) != [0, 0, 0, 255] && !(x >= left && x < right && y >= top && y < bottom) {
                ink += 1;
            }
        }
    }
    if ink < 100 {
        return Some(format!(
            "rows {text_top}..{text_bottom}, which the emitted glyphs claim, hold only {ink} \
             non-clear pixels outside the quad, so the line did not draw"
        ));
    }
    if text_bottom < height {
        for x in 0..width {
            if at(x, text_bottom) != [0, 0, 0, 255]
                && !(x >= left && x < right && text_bottom >= top && text_bottom < bottom)
            {
                return Some(format!(
                    "row {text_bottom}, below every emitted glyph's extent, is painted"
                ));
            }
        }
    }
    None
}

/// The first pixel that is not [`PROOF_MAGENTA_BYTES`], described.
fn wrong_pixel(pixels: &[u8], width: u32, height: u32) -> Option<String> {
    let expected = (width as usize) * (height as usize) * 4;
    if pixels.len() != expected {
        return Some(format!(
            "readback is {} bytes, expected {expected}",
            pixels.len()
        ));
    }
    for y in 0..height as usize {
        for x in 0..width as usize {
            let offset = (y * width as usize + x) * 4;
            let Some(pixel) = pixels.get(offset..offset + 4) else {
                return Some(format!("pixel ({x}, {y}) is past the end of the readback"));
            };
            if pixel != PROOF_MAGENTA_BYTES {
                return Some(format!(
                    "pixel ({x}, {y}) is {pixel:?}, expected {PROOF_MAGENTA_BYTES:?}"
                ));
            }
        }
    }
    None
}

fn main() -> std::process::ExitCode {
    let options = match Options::parse() {
        Ok(options) => options,
        Err(message) => {
            eprintln!("phase6_window: {message}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let event_loop = match winit::event_loop::EventLoop::new() {
        Ok(event_loop) => event_loop,
        Err(error) => {
            eprintln!("phase6_window: no event loop on this platform/session: {error}");
            return std::process::ExitCode::FAILURE;
        }
    };
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let mut app = App {
        options: options.clone(),
        started: Instant::now(),
        live: None,
        observed: Observed::default(),
    };
    if let Err(error) = event_loop.run_app(&mut app) {
        eprintln!("phase6_window: run_app failed: {error}");
        return std::process::ExitCode::FAILURE;
    }
    // The exit path may be a close request rather than the frame/hold budget,
    // in which case `finish_if_done` never ran and the counters are still on
    // the live surface.
    if let Some(live) = app.live.as_ref() {
        app.observed.surface = live.surface.stats();
        app.observed.resize_events_seen = live.resizes.events_seen();
        app.observed.resize_reconfigurations = live.resizes.reconfigurations();
    }

    let observed = &app.observed;
    println!("frames drawn:            {}", observed.frames_drawn);
    println!("frames verified:         {}", observed.frames_verified);
    println!("surface stats:           {:?}", observed.surface);
    println!("resize events seen:      {}", observed.resize_events_seen);
    println!(
        "reconfigurations:        {}",
        observed.resize_reconfigurations
    );
    println!(
        "resize steps dispatched: {}",
        observed.resize_steps_dispatched
    );
    println!(
        "resizes answered sync:   {}",
        observed.resizes_answered_synchronously
    );
    println!("sizes presented:         {:?}", observed.sizes_presented);
    if options.scene {
        println!("idle frames:             {}", observed.idle_frames);
        println!("bytes uploaded:          {}", observed.uploaded_bytes);
        println!("viewport recomputes:     {}", observed.viewport_recomputes);
        println!(
            "last frame:              {}",
            observed.last_frame.as_deref().unwrap_or("none")
        );
    }
    println!("elapsed:                 {:?}", app.started.elapsed());

    let mut failed = false;
    if let Some(failure) = &observed.failure {
        eprintln!("FAILED: {failure}");
        failed = true;
    }
    for failure in observed.verify_failures.iter().take(8) {
        eprintln!("FAILED: {failure}");
        failed = true;
    }
    if observed.surface.lost > 0 {
        eprintln!("FAILED: {} acquires were lost", observed.surface.lost);
        failed = true;
    }
    if options.verify && observed.frames_verified == 0 {
        eprintln!("FAILED: --verify was asked for and no frame was verified");
        failed = true;
    }
    // One full recompute per distinct size presented, and not one more: the
    // first claim is the resize fix doing its job, the second is that it did not
    // turn every settled frame into a rebuild.
    if options.scene && observed.viewport_recomputes != observed.sizes_presented.len() as u64 {
        eprintln!(
            "FAILED: {} viewport recomputes for {} sizes presented",
            observed.viewport_recomputes,
            observed.sizes_presented.len()
        );
        failed = true;
    }
    if options.resize && observed.sizes_presented.len() < 2 {
        eprintln!(
            "FAILED: --resize presented only {:?}, so no reconfiguration was exercised",
            observed.sizes_presented
        );
        failed = true;
    }

    if failed {
        std::process::ExitCode::FAILURE
    } else {
        println!("OK");
        std::process::ExitCode::SUCCESS
    }
}
