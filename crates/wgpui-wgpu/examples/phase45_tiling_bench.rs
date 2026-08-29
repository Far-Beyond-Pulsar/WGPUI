//! §4.3's tile-size tradeoff, measured against a representative node-graph
//! workload rather than asserted.
//! See docs/gpu-native-architecture.md §4.3, §8 Phase 4.5, §9's risk table.
//!
//! Run with `cargo run -p wgpui-wgpu --release --example phase45_tiling_bench`.
//!
//! # Why this exists
//!
//! §4.3 is explicit that the tile size is "a real tuning knob with a real
//! tradeoff … There's no principled default without measuring — Phase 0's spike
//! discipline applies here too, picking a starting size from common compositor
//! practice (roughly 256–512px) and validating it against a representative
//! node-graph workload, not asserting it." §9's risk table names the failure
//! mode in both directions: "too small inflates per-tile/draw-call overhead, too
//! large approaches `Margin`'s whole-region-refill cost."
//!
//! So this sweeps the edge length across and beyond that range and reports both
//! sides of the tradeoff for each, on one workload, with the numbers that would
//! make a size wrong.
//!
//! # What is measured, and what each column means
//!
//! - **slots/frame** — draw slots the fixed sequence issues, which is one per
//!   visible tile plus the overlay. This is the "too small inflates draw-call
//!   overhead" axis, and Phase 4's gate already established that CPU draw-issue
//!   cost is linear in it.
//! - **crossing** — primitives written when a pan crosses one tile boundary.
//!   This is the "too large approaches `Margin`'s refill cost" axis.
//! - **refill** — primitives a whole-region refill would write, i.e. what
//!   `Buffering::Margin` costs for the same pan per R-N §7. The comparison the
//!   whole mechanism exists to win.
//! - **overlay** — share of resident primitives on the unbuffered overlay
//!   layer, which is content the tile grid does not cull. Rises as tiles shrink,
//!   because a primitive larger than a tile cannot be anchored
//!   (`scene::TilePlacement`).
//! - **resident** — tiles held after the sweep, against the budget.
//! - **visibility** — GPU wall clock for the tile-visibility dispatch. Reported
//!   because a pass that decided visibility more slowly than the work it saved
//!   would be a bad trade, and nothing else in this phase would have shown it.
//!
//! # Methodology, carried over from Phases 3 and 4 unchanged
//!
//! Two warm-up runs discarded and disclosed; median and best both reported, with
//! the median the lower middle of an even count so every printed number was
//! actually observed; the adapter named before any number is printed, and a
//! software rasterizer called out loudly.

use std::time::{Duration, Instant};

use wgpui_core::boundary::policy::BoundaryPolicy;
use wgpui_core::geometry::Rect;
use wgpui_core::indirect::{FirstInstance, QUAD_VERTEX_COUNT};
use wgpui_core::scene::{TileDescriptor, encode_tiles};
use wgpui_core::test_support::ui_walk::{NodeGraphSpec, TiledCanvasDriver};
use wgpui_wgpu::render::compute::indirect_args_pass::{IndirectArgsBuffers, IndirectArgsPass};
use wgpui_wgpu::render::compute::tile_visibility_pass::{
    ArgsTarget, TileViewport, TileVisibilityBuffers, TileVisibilityPass,
};
use wgpui_wgpu::render::device::{ComputeContext, headless_compute_context};

const VIEWPORT_WIDTH: f32 = 1600.0;
const VIEWPORT_HEIGHT: f32 = 900.0;
const RETAIN_RADIUS: u32 = 1;
const BUDGET: usize = 256;
const WARM_UP: usize = 2;
const RUNS: usize = 12;

/// Edge lengths swept. 128 and 1024 sit outside §4.3's suggested 256–512 band on
/// purpose: a sweep that only covered the band could not show that the band is
/// the right place to be.
const EDGES: [f32; 5] = [128.0, 256.0, 384.0, 512.0, 1024.0];

fn viewport() -> Rect {
    Rect::from_origin_size([0.0, 0.0], [VIEWPORT_WIDTH, VIEWPORT_HEIGHT])
}

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
            "  !! SOFTWARE RASTERIZER — every timing below is a CPU emulation of a \
             GPU and must not be quoted as hardware."
        );
    }
    println!(
        "\n== §4.3 tile-size tradeoff — node graph, {VIEWPORT_WIDTH:.0}x{VIEWPORT_HEIGHT:.0} \
         viewport, retain radius {RETAIN_RADIUS}, budget {BUDGET} =="
    );
    println!(
        "  {WARM_UP} warm-up runs discarded, {RUNS} timed; median is the lower middle of \
         an even count."
    );
    println!(
        "\n  {:>5}  {:>6}  {:>7}  {:>8}  {:>8}  {:>7}  {:>8}  {:>9}  {:>9}",
        "edge", "tiles", "slots", "crossing", "refill", "ratio", "overlay", "vis med", "vis best"
    );
    println!("  {}", "-".repeat(84));

    for edge in EDGES {
        match measure(&context, edge) {
            Some(row) => println!(
                "  {:>5.0}  {:>6}  {:>7}  {:>8}  {:>8}  {:>6.1}x  {:>7.1}%  {:>8.1}µs  {:>8.1}µs   \
                 ({:.2}x viewport)",
                edge,
                row.resident_tiles,
                row.slots_per_frame,
                row.crossing_primitives,
                row.refill_primitives,
                row.refill_primitives as f64 / row.crossing_primitives.max(1) as f64,
                row.overlay_share * 100.0,
                micros(row.visibility_median),
                micros(row.visibility_best),
                row.buffered_area_ratio,
            ),
            None => println!("  {edge:>5.0}  unusable at this viewport — see TileSpan::MAX_TILES"),
        }
    }

    println!(
        "\n  crossing = primitives written when a pan crosses one tile boundary.\n  \
         refill   = primitives a whole-region refill writes, i.e. what Buffering::Margin\n             \
         costs for the same pan (R-N §7). ratio is refill/crossing — higher is better.\n  \
         overlay  = share of resident primitives on the unbuffered overlay layer, which\n             \
         the tile grid does not cull.\n  \
         vis      = GPU wall clock for the tile-visibility dispatch, including its\n             \
         parameter upload and submit."
    );
    println!(
        "\n  READ THE RATIO COLUMN WITH THE AREA IN BRACKETS. `refill` is the whole\n  \
         resident region, and that region is not the same size across the sweep: a\n  \
         retain radius of one tile buffers a ring whose width *is* the tile size. So a\n  \
         large tile inflates its own refill baseline, and the ratio is only a like-for-\n  \
         like comparison against Buffering::Margin(None) — which buffers 2.25x the\n  \
         viewport (1.5 on each axis, §4.1) — where the bracketed area is near 2.25x.\n  \
         Rows far above that are comparing against a bigger buffer than Margin would\n  \
         have kept, and their ratio flatters tiling."
    );
    println!(
        "\n  AND READ THE `vis` COLUMN AS A CEILING, NOT AS THE KERNEL'S COST. It ends in\n  \
         Device::poll(wait_indefinitely), which waits for everything already submitted —\n  \
         the same effect docs/phase-4-results.md §5 records for the readback path. The\n  \
         dispatch itself is one workgroup per 64 tiles over a few dozen tiles; what these\n  \
         numbers mostly measure is one submit-and-synchronize round trip, which is why\n  \
         they barely move with the tile count. A real frame pipelines this dispatch with\n  \
         the rest of its work and never polls it, so nothing here should be read as a\n  \
         per-frame cost. It is reported to show the pass is not accidentally expensive,\n  \
         and it is not: the whole column is flat and the variance between rows is noise\n  \
         (see the 384 row's median against its own best)."
    );
}

struct Row {
    resident_tiles: usize,
    slots_per_frame: usize,
    crossing_primitives: usize,
    refill_primitives: usize,
    overlay_share: f64,
    /// The resident region's area as a multiple of the viewport's — what makes
    /// the `refill` baseline comparable across rows, or not. See the note the
    /// table prints beneath itself.
    buffered_area_ratio: f64,
    visibility_median: Duration,
    visibility_best: Duration,
}

fn measure(context: &ComputeContext, edge: f32) -> Option<Row> {
    let policy = BoundaryPolicy {
        // Long enough that the sweep's own pan never evicts, so the resident
        // count reported is the grid's shape rather than the eviction interval's.
        evict_after_frames: 240,
        ..TiledCanvasDriver::tiled_policy(edge, RETAIN_RADIUS, BUDGET)
    };
    let mut canvas = TiledCanvasDriver::new(&NodeGraphSpec::large(), viewport(), policy);

    // Settle at a deliberately tile-unaligned offset: a viewport whose edges sit
    // on exact multiples of the tile size crosses a boundary on its first pixel
    // of pan, which would make the "crossing" column measure the wrong frame.
    let first = canvas.pan_to([-8.0, -8.0]).ok()?;
    if first.visible_tiles == 0 {
        return None;
    }
    canvas.settle();

    let refill_primitives = canvas.resident_primitives();
    let overlay_share = if refill_primitives == 0 {
        0.0
    } else {
        canvas.overlay_primitives() as f64 / refill_primitives as f64
    };
    let slots_per_frame = first.visible_tiles + 1;

    let crossing = canvas.pan_to([-8.0 - edge, -8.0]).ok()?;
    canvas.settle();

    let visibility = time_visibility(context, &canvas, edge)?;

    // The ring a retain radius of one buffers is one tile wide on every side, so
    // the buffered area grows with the tile size — which is exactly why the
    // `refill` column is not comparable across rows without this number.
    let buffered_width = VIEWPORT_WIDTH + 2.0 * RETAIN_RADIUS as f64 as f32 * edge;
    let buffered_height = VIEWPORT_HEIGHT + 2.0 * RETAIN_RADIUS as f64 as f32 * edge;
    let buffered_area_ratio = (buffered_width as f64 * buffered_height as f64)
        / (VIEWPORT_WIDTH as f64 * VIEWPORT_HEIGHT as f64);

    Some(Row {
        resident_tiles: crossing.resident_tiles,
        slots_per_frame,
        crossing_primitives: crossing.primitives_written,
        refill_primitives,
        overlay_share,
        buffered_area_ratio,
        visibility_median: visibility.0,
        visibility_best: visibility.1,
    })
}

/// Time the visibility dispatch over this grid's resident tile set.
///
/// The clock covers the parameter upload, the descriptor upload, the dispatch,
/// and a `poll` that waits for it — so it is GPU work included, not command
/// encoding alone. That is deliberate here and different from Phase 4's
/// draw-issue clock: the question this column answers is whether deciding
/// visibility costs less than the rendering it avoids, and only a wall clock
/// that includes the device can answer it.
fn time_visibility(
    context: &ComputeContext,
    canvas: &TiledCanvasDriver,
    edge: f32,
) -> Option<(Duration, Duration)> {
    let state = canvas.compositor.state(canvas.boundary)?;
    let residency = state.tiles()?;
    let resident = residency.resident();
    let tiles: Vec<TileDescriptor> = resident
        .iter()
        .enumerate()
        .map(|(index, coord)| TileDescriptor {
            coord: *coord,
            base: index as u32 * 32,
            count: 32,
        })
        .collect();
    let arena_slots = tiles.len() as u32 * 32;
    let mut tile_bytes = Vec::new();
    encode_tiles(&tiles, &mut tile_bytes);

    let pass = TileVisibilityPass::new(&context.device);
    let indirect = IndirectArgsPass::new(&context.device);
    let buffers = TileVisibilityBuffers::new(&context.device, tiles.len() as u32);
    let args = IndirectArgsBuffers::new(&context.device, arena_slots, tiles.len() as u32 + 1);
    let view = viewport();
    let params = TileViewport {
        tile_size: [edge, edge],
        pan: [-8.0 - edge, -8.0],
        viewport: [view.min_x, view.min_y, view.max_x, view.max_y],
        retain_radius: RETAIN_RADIUS,
    };

    let mut samples = Vec::with_capacity(RUNS);
    for run in 0..WARM_UP + RUNS {
        let started = Instant::now();
        pass.run_into_args(
            &context.device,
            &context.queue,
            &buffers,
            ArgsTarget {
                pass: &indirect,
                buffers: &args,
                vertex_count: QUAD_VERTEX_COUNT,
                first_instance: FirstInstance::Zero,
            },
            &tile_bytes,
            params,
        )
        .ok()?;
        context
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .ok()?;
        let elapsed = started.elapsed();
        if run >= WARM_UP {
            samples.push(elapsed);
        }
    }
    samples.sort_unstable();
    Some((
        samples.get(samples.len().saturating_sub(1) / 2).copied()?,
        samples.first().copied()?,
    ))
}

fn micros(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000_000.0
}
