//! Color-calibration probe: render known built-in solid-color rows, read the GPU
//! framebuffer back, and print requested-vs-captured RGBA for each. Reveals
//! linear/sRGB mismatches, channel swaps, and alpha issues in the built-in quad
//! path. Diagnostic tool, not a gate.
//!     cargo run --example color_probe

use std::time::Duration;

use gpui::{
    AppContext as _,
    Application,
    Bounds,
    Context,
    IntoElement,
    ParentElement,
    Render,
    Styled,
    Timer,
    Window,
    WindowBounds,
    WindowOptions,
    div,
    px,
    rgb,
    size,
};

const ROW_H: f32 = 40.0;
const SWATCHES: &[(&str, u32)] = &[
    ("red", 0xff0000),
    ("green", 0x00ff00),
    ("blue", 0x0000ff),
    ("white", 0xffffff),
    ("gray50", 0x808080),
    ("gray25", 0x404040),
    ("bg-blue", 0x113355),
    ("box-green", 0x22cc44),
];

struct ProbeView;

impl Render for ProbeView {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        div().size_full().flex().flex_col().children(
            SWATCHES
                .iter()
                .map(|(_name, hex)| div().h(px(ROW_H)).bg(rgb(*hex))),
        )
    }
}

fn rgb_bytes(color: u32) -> [u8; 3] {
    [
        ((color >> 16) & 0xff) as u8,
        ((color >> 8) & 0xff) as u8,
        (color & 0xff) as u8,
    ]
}

fn main() {
    Application::new().run(|cx| {
        let bounds = Bounds::centered(None, size(px(400.), px(400.)), cx);
        let window = cx
            .open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    ..Default::default()
                },
                |_window, cx| cx.new(|_cx| ProbeView),
            )
            .expect("failed to open window");
        let window = window.into();

        cx.spawn(async move |cx| {
            Timer::after(Duration::from_millis(600)).await;

            let info = cx.update_window(window, |_view, window, _cx| {
                (
                    window.viewport_size(),
                    window.scale_factor(),
                    window.read_back_pixels(),
                )
            });

            if let Ok((viewport, scale, Some(image))) = info {
                println!(
                    "viewport {:?}, scale {scale}, framebuffer {}x{}",
                    viewport, image.width, image.height
                );
                let x = image.width / 2;
                for (index, (name, hex)) in SWATCHES.iter().enumerate() {
                    let logical_y = index as f32 * ROW_H + ROW_H / 2.0;
                    let y = (logical_y * scale) as u32;
                    if y >= image.height {
                        continue;
                    }
                    let got = image.pixel_at(x, y);
                    let expected = rgb_bytes(*hex);
                    println!(
                        "{name:10} req ({:3},{:3},{:3})  got ({:3},{:3},{:3}) a={:3}",
                        expected[0], expected[1], expected[2], got[0], got[1], got[2], got[3]
                    );
                }
            } else {
                eprintln!("probe: failed to read back pixels: {info:?}");
            }

            let _ = cx.update(|cx| cx.quit());
        })
        .detach();
    });
}
