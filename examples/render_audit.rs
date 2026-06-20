//! On-display pixel-correctness audit for the real renderer.
//!
//! Opens a real window, paints a known scene through the production render path —
//! a solid background (built-in quads), a fixed green box (more built-in quads),
//! and a row of magenta `SolidQuad` *plugin* primitives — then, after a few
//! frames, reads the GPU framebuffer back via [`gpui::Window::read_back_pixels`]
//! and verifies the expected colors are actually present. It writes the captured
//! frame to `/tmp/render_audit_capture.png` for visual inspection and, if a
//! golden PNG is supplied via `RENDER_AUDIT_GOLDEN`, compares against it.
//!
//! Exits 0 on success and 1 on any mismatch, so it can gate a display-backed
//! check:
//!     cargo run --example render_audit
//!
//! Unlike the headless smoke tests, this exercises real pixels on the real GPU,
//! including the custom render-primitive draw integrated into the renderer's
//! main pass.

use std::process::ExitCode;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Duration;

use gpui::{
    AppContext as _,
    Application,
    Bounds,
    Context,
    IntoElement,
    ParentElement,
    PixelBuffer,
    Render,
    Styled,
    Timer,
    Window,
    WindowBounds,
    WindowOptions,
    canvas,
    div,
    px,
    rgb,
    size,
};
use gpui_prim_solid::{SolidQuad, SolidQuads};

/// Distinctive colors so each subsystem is identifiable in the readback.
const BACKGROUND: u32 = 0x113355; // built-in quad (window background)
const GREEN_BOX: u32 = 0x22cc44; // built-in quad (a child div)
const PLUGIN_RGBA: [f32; 4] = [1.0, 0.0, 1.0, 1.0]; // magenta plugin SolidQuad

/// 0 = success, 1 = mismatch. Set by the audit task before it quits the app.
static EXIT_CODE: AtomicU8 = AtomicU8::new(1);

struct AuditView {
    registered: bool,
}

impl Render for AuditView {
    fn render(&mut self, window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        // Register the plugin once, on first render.
        if !self.registered {
            window.register_render_primitive(Box::new(SolidQuads));
            self.registered = true;
        }

        div()
            .size_full()
            .bg(rgb(BACKGROUND))
            .flex()
            .child(div().w(px(140.)).h(px(140.)).bg(rgb(GREEN_BOX)))
            .child(
                canvas(
                    |_bounds, _window, _cx| (),
                    move |bounds, _, window, _cx| {
                        // A row of magenta plugin quads in device pixels.
                        for index in 0..16 {
                            let x = 220.0 + index as f32 * 16.0;
                            let quad = SolidQuad::new([x, 220.0], [12.0, 12.0], PLUGIN_RGBA);
                            window.paint_custom_primitive(
                                SolidQuads::TYPE,
                                quad.as_bytes(),
                                bounds,
                            );
                        }
                    },
                )
                .size_full(),
            )
    }
}

// Structural color classifiers. We assert on the *relationship* between channels
// rather than exact sRGB bytes, so the audit is robust to the compositor's gamma
// and premultiplied-alpha handling (which shift absolute values) while still
// catching real regressions: a blank frame, a missing built-in or plugin draw,
// or grossly wrong colors.

/// Green-dominant: the built-in green box (G clearly the largest channel).
fn is_greenish(texel: &[u8]) -> bool {
    texel[1] > 120 && texel[1] > texel[0] + 20 && texel[1] > texel[2] + 20
}

/// Magenta-like: the plugin SolidQuads (R and B both high, G low).
fn is_magenta(texel: &[u8]) -> bool {
    texel[0] > 150 && texel[2] > 150 && texel[1] < 100
}

fn count<F: Fn(&[u8]) -> bool>(image: &PixelBuffer, predicate: F) -> usize {
    image
        .pixels
        .chunks_exact(4)
        .filter(|t| predicate(t))
        .count()
}

/// Verify the captured frame contains each subsystem's contribution, write it to
/// disk, and (optionally) compare against a golden. Returns true on success.
fn validate(image: &PixelBuffer) -> bool {
    let total = (image.width * image.height) as usize;
    let green = count(image, is_greenish);
    let magenta = count(image, is_magenta);
    // Whatever is neither the green box nor the plugin quads is background.
    let background = total - green - magenta;

    println!(
        "captured {}x{} ({total} px): background={background} green={green} magenta={magenta}",
        image.width, image.height
    );

    let capture_path = "/tmp/render_audit_capture.png";
    match image.save_png(capture_path) {
        Ok(()) => println!("wrote capture to {capture_path}"),
        Err(error) => eprintln!("failed to write capture: {error:#}"),
    }

    let mut ok = true;
    if background < total / 2 {
        eprintln!("FAIL: background does not dominate the frame ({background}/{total})");
        ok = false;
    }
    if green < 2000 {
        eprintln!("FAIL: built-in green box not rendered (got {green} green px)");
        ok = false;
    }
    if magenta < 500 {
        eprintln!("FAIL: plugin magenta SolidQuads not rendered (got {magenta} magenta px)");
        ok = false;
    }

    // Optional golden comparison.
    if let Ok(golden_path) = std::env::var("RENDER_AUDIT_GOLDEN") {
        match PixelBuffer::load_png(&golden_path) {
            Ok(golden) => match image.pixels_exceeding(&golden, 12) {
                Some(diff) if diff <= total / 100 => {
                    println!("golden match: {diff} px differ (<= 1%)");
                }
                Some(diff) => {
                    eprintln!("FAIL: golden mismatch, {diff} px differ (> 1%)");
                    ok = false;
                }
                None => {
                    eprintln!("FAIL: golden has different dimensions");
                    ok = false;
                }
            },
            Err(error) => eprintln!("could not load golden {golden_path}: {error:#}"),
        }
    }

    ok
}

fn main() -> ExitCode {
    Application::new().run(|cx| {
        let bounds = Bounds::centered(None, size(px(640.), px(480.)), cx);
        let window = cx
            .open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    ..Default::default()
                },
                |_window, cx| cx.new(|_cx| AuditView { registered: false }),
            )
            .expect("failed to open window");
        let window = window.into();

        cx.spawn(async move |cx| {
            // Let a few frames render so the persistent framebuffer is populated.
            Timer::after(Duration::from_millis(600)).await;

            // Print the window's own logical size + scale factor so the capture
            // dimensions can be confirmed as (window size x scale), i.e. this is
            // the app window's framebuffer, not the screen.
            if let Ok((viewport, scale, bounds)) = cx.update_window(window, |_view, window, _cx| {
                (
                    window.viewport_size(),
                    window.scale_factor(),
                    window.bounds(),
                )
            }) {
                println!(
                    "window: logical viewport {:?}, scale {scale}, bounds {:?}",
                    viewport, bounds
                );
                println!(
                    "expected framebuffer = {} x {} device px",
                    f32::from(viewport.width) * scale,
                    f32::from(viewport.height) * scale,
                );
            }

            let captured = cx
                .update_window(window, |_view, window, _cx| window.read_back_pixels())
                .ok()
                .flatten();

            let ok = match captured {
                Some(image) => validate(&image),
                None => {
                    eprintln!("FAIL: read_back_pixels returned None (no GPU frame captured)");
                    false
                }
            };
            println!("render_audit: {}", if ok { "PASS" } else { "FAIL" });

            // Record the result and quit cleanly. Quitting (rather than
            // process::exit) deliberately exercises the real window/wgpu teardown
            // path, so this tool also guards against teardown regressions.
            EXIT_CODE.store(u8::from(!ok), Ordering::SeqCst);
            let _ = cx.update(|cx| cx.quit());
        })
        .detach();
    });

    // Set by the audit task before it quit; defaults to failure if it never ran.
    ExitCode::from(EXIT_CODE.load(Ordering::SeqCst))
}
