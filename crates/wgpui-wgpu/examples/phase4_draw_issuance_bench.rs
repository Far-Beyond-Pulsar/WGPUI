//! §8's Phase 4 gate, measured on real hardware: a clean window's CPU-side
//! draw-issuing work is O(layer slots), independent of resident primitive
//! count.
//! See docs/gpu-native-architecture.md §5.3, §8 Phase 4.
//!
//! Run with `cargo run -p wgpui-wgpu --release --example
//! phase4_draw_issuance_bench`.
//!
//! # What is being timed, precisely
//!
//! The wall clock around `render/draw.rs`'s issuing loop, and nothing else: set
//! pipeline, set bind group, issue draw, per slot. That is *command encoding on
//! the CPU*. It is deliberately not a frame time and deliberately not GPU work,
//! because the gate's claim is about what the CPU spends, and §5.3's own
//! sentence is "a clean window's CPU-side draw-issuing cost becomes O(layer
//! slots), not O(resident primitives)".
//!
//! Excluded, and named rather than quietly omitted: pipeline construction
//! (once, at startup, as Phase 3's benchmark also excludes it and for the same
//! reason Spike A's write-up gives), buffer uploads, the compute passes, and the
//! argument-generation dispatch. Every one of those runs only on a frame where
//! something changed; the gate is about the frame where nothing did.
//!
//! # Methodology, carried over from Phase 3 unchanged
//!
//! - Two warm-up runs, discarded and disclosed, rather than argued away
//!   afterwards.
//! - Median and best both reported. For an even count the median is the lower
//!   middle, so every printed number is a duration actually observed.
//! - The adapter is named before any number is printed, and a software
//!   rasterizer is called out loudly.
//!
//! # Two sweeps, because either alone answers half the question
//!
//! **Sweep A** holds the layer count fixed and raises the primitive count by
//! more than two orders of magnitude. If draw-issuing cost is O(layer slots),
//! this line is flat.
//!
//! **Sweep B** holds the primitive count roughly fixed and raises the layer
//! count. If draw-issuing cost is O(layer slots), this line rises — and it
//! should, because "O(layer slots)" is a claim about what the cost *is*, not
//! only about what it is not. A benchmark that only showed the flat line would
//! be consistent with the cost being zero, which it is not.

use std::time::Duration;

use wgpui_core::geometry::Rect;
use wgpui_core::patch::primitive::Quad;
use wgpui_core::test_support::ui_walk::MultiLayerSceneDriver;
use wgpui_wgpu::render::device::{ComputeContext, headless_compute_context};
use wgpui_wgpu::render::draw::DrawMode;
use wgpui_wgpu::render::frame::{Dirty, FrameInput, FrameRenderer, OffscreenTarget};

const WIDTH: f32 = 1920.0;
const HEIGHT: f32 = 1080.0;
const WARM_UP: usize = 2;
const RUNS: usize = 12;

fn main() {
    let context = match headless_compute_context() {
        Ok(context) => context,
        Err(error) => {
            eprintln!(
                "NO ADAPTER: {error}\n\
                 No number below would mean anything without one, so none is printed."
            );
            return;
        }
    };
    println!("== Adapter ==");
    println!("  {}", context.describe());
    if context.is_software() {
        println!(
            "  WARNING: this is a software rasterizer. Correctness claims still \
             hold; every timing below is meaningless as hardware evidence."
        );
    }
    println!(
        "  draw modes available: {:?}",
        DrawMode::ALL
            .iter()
            .filter(|mode| mode.is_available(context.indirect))
            .map(|mode| mode.name())
            .collect::<Vec<_>>()
    );
    println!(
        "\n  Timed: CPU command encoding of the fixed draw sequence, on a clean \
         frame.\n  Excluded: pipeline construction, uploads, compute, \
         argument generation.\n  {WARM_UP} warm-up runs discarded, {RUNS} timed."
    );

    println!("\n== Sweep A: 8 layers, primitive count rising ==");
    println!(
        "{:>12} {:>8} {:>7} {:>8} {:>12} {:>12} {:>12}  mode",
        "primitives", "layers", "slots", "calls", "median", "best", "readback"
    );
    for per_layer in [32u32, 320, 3_200, 12_800] {
        for mode in available(&context) {
            run_case(&context, 8, per_layer, mode);
        }
    }

    println!("\n== Sweep B: ~25,600 primitives, layer count rising ==");
    println!(
        "{:>12} {:>8} {:>7} {:>8} {:>12} {:>12} {:>12}  mode",
        "primitives", "layers", "slots", "calls", "median", "best", "readback"
    );
    for layers in [2usize, 8, 32, 128] {
        let per_layer = 25_600 / layers as u32;
        for mode in available(&context) {
            run_case(&context, layers, per_layer, mode);
        }
    }

    println!(
        "\nRead these against §8's Phase 4 gate: Sweep A flat is the gate; Sweep \
         B rising is what makes the gate a claim about a cost rather than about \
         its absence."
    );
}

fn available(context: &ComputeContext) -> Vec<DrawMode> {
    DrawMode::ALL
        .into_iter()
        .filter(|mode| mode.is_available(context.indirect))
        .collect()
}

fn run_case(context: &ComputeContext, layers: usize, per_layer: u32, mode: DrawMode) {
    let mut scene = MultiLayerSceneDriver::new(layers);
    scene.clip = Rect::from_origin_size([0.0, 0.0], [WIDTH, HEIGHT]);
    for index in 0..layers {
        let quads = tile_grid(index, per_layer);
        if scene.set_layer(index, &quads).is_err() {
            eprintln!("  seeding layer {index} failed; skipping this case");
            return;
        }
    }

    let mut renderer = FrameRenderer::new(&context.device);
    let target = OffscreenTarget::new(&context.device, WIDTH as u32, HEIGHT as u32);

    let dirty = FrameInput {
        scene: &scene.scene,
        clip: scene.clip,
        poison: &scene.poison,
        dirty: Dirty::All,
        uploads: &[],
        composites: &[],
        registry: None,
        atlas: None,
        viewport: [WIDTH, HEIGHT],
        mode,
    };
    if renderer
        .render(&context.device, &context.queue, &dirty, &target)
        .is_err()
    {
        eprintln!("  the seeding frame failed to render; skipping this case");
        return;
    }
    let clean = FrameInput {
        dirty: Dirty::Some(&[]),
        ..dirty
    };

    let mut samples = Vec::with_capacity(RUNS);
    let mut readbacks = Vec::with_capacity(RUNS);
    let mut slots = 0u32;
    let mut calls = 0u32;
    let mut primitives = 0u32;
    for run in 0..WARM_UP + RUNS {
        let output = match renderer.render(&context.device, &context.queue, &clean, &target) {
            Ok(output) => output,
            Err(error) => {
                eprintln!("  frame failed: {error}");
                return;
            }
        };
        if run >= WARM_UP {
            samples.push(output.timing.draw_issue);
            readbacks.push(output.timing.readback);
        }
        slots = output.stats.slots_visited;
        calls = output.stats.draw_calls_issued;
        primitives = output.primitives_resident;
    }

    println!(
        "{primitives:>12} {layers:>8} {slots:>7} {calls:>8} {:>12} {:>12} {:>12}  {}",
        format!("{:.2?}", median(&mut samples)),
        format!("{:.2?}", best(&samples)),
        format!("{:.2?}", median(&mut readbacks)),
        mode.name()
    );
}

/// The lower middle for an even count, so every printed number is a duration
/// that was actually observed rather than an interpolation between two.
fn median(samples: &mut [Duration]) -> Duration {
    samples.sort_unstable();
    samples
        .get(samples.len() / 2)
        .copied()
        .unwrap_or(Duration::ZERO)
}

fn best(samples: &[Duration]) -> Duration {
    samples.iter().copied().min().unwrap_or(Duration::ZERO)
}

/// A grid of quads filling the window, so the scene is realistic enough that
/// ordering and occlusion have work to do on the seeding frame.
fn tile_grid(layer: usize, count: u32) -> Vec<Quad> {
    let columns = (count as f32).sqrt().ceil().max(1.0) as u32;
    let cell_width = WIDTH / columns as f32;
    let cell_height = HEIGHT / columns as f32;
    let offset = layer as f32 * 3.0;
    (0..count)
        .map(|index| {
            let column = index % columns;
            let row = index / columns;
            Quad {
                origin: [
                    column as f32 * cell_width + offset,
                    row as f32 * cell_height + offset,
                ],
                size: [cell_width * 0.9, cell_height * 0.9],
                background: [
                    0.2 + (index % 7) as f32 * 0.1,
                    0.3,
                    0.4 + (layer % 5) as f32 * 0.1,
                    1.0,
                ],
                border_color: [0.0, 0.0, 0.0, 1.0],
                corner_radius: 2.0,
                border_width: 1.0,
            }
        })
        .collect()
}
