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

use wgpui_wgpu::render::device::ComputeContext;
use wgpui_wgpu::render::readback::read_texture_rgba8;
use wgpui_wgpu::window::resize_detector::ResizeDetector;
use wgpui_wgpu::window::{
    Acquired, PROOF_MAGENTA, PROOF_MAGENTA_BYTES, SurfaceStats, WindowSurface, clear_frame,
};

const INITIAL_WIDTH: u32 = 800;
const INITIAL_HEIGHT: u32 = 500;

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
}

impl Options {
    fn parse() -> Result<Options, String> {
        let mut options = Options {
            frames: None,
            hold: None,
            resize: false,
            verify: false,
        };
        let mut arguments = std::env::args().skip(1);
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--frames" => {
                    let value = arguments.next().ok_or("--frames needs a count")?;
                    options.frames =
                        Some(value.parse::<u32>().map_err(|error| error.to_string())?);
                }
                "--hold-ms" => {
                    let value = arguments.next().ok_or("--hold-ms needs a duration")?;
                    options.hold = Some(Duration::from_millis(
                        value.parse::<u64>().map_err(|error| error.to_string())?,
                    ));
                }
                "--resize" => options.resize = true,
                "--verify" => options.verify = true,
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
            "surface: {:?} {:?} at {}x{}",
            surface.format(),
            surface.format_choice(),
            surface.size().0,
            surface.size().1,
        );

        let mut resizes = ResizeDetector::new();
        let (width, height) = surface.size();
        resizes.seed(width, height);

        self.live = Some(Live {
            surface,
            context,
            resizes,
            next_resize_step: 0,
            frames_since_resize: 0,
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
        clear_frame(
            &live.context.device,
            &live.context.queue,
            &view,
            PROOF_MAGENTA,
        );

        let (width, height) = live.surface.size();
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
                    if let Some(message) = wrong_pixel(&pixels, width, height) {
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
    println!("reconfigurations:        {}", observed.resize_reconfigurations);
    println!("resize steps dispatched: {}", observed.resize_steps_dispatched);
    println!(
        "resizes answered sync:   {}",
        observed.resizes_answered_synchronously
    );
    println!("sizes presented:         {:?}", observed.sizes_presented);
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
