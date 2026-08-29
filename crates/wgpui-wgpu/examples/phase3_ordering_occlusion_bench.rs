//! Phase 3 gate #2: Phase 0's Spike A numbers, reproduced on the real pipeline.
//! See docs/gpu-native-architecture.md §5.1, §5.2, §8 Phase 3.
//!
//! Run with:
//!
//!     cargo run -p wgpui-wgpu --example phase3_ordering_occlusion_bench --release --offline
//!
//! # What is measured, and against what
//!
//! §8's Phase 3 gate: "Spike numbers from Phase 0 reproduced on the real
//! pipeline, not just the synthetic case." So the scene is not Spike A's
//! 100,000-quad cluster grid. It is `wgpui_core::test_support::ui_walk`'s
//! editor-shaped scene — window chrome, a scrolling tree panel, a node-graph
//! viewport, a docked inspector, a modal — built through the real
//! `Scene`/`Layer`/`PrimitiveStore` APIs by real `ScenePatch` appends, and read
//! back out of the resident store rather than out of the generator's `Vec`.
//!
//! **Two shapes of that scene, because either alone answers only half.** At
//! normal zoom a hundred-thousand-primitive editor has most of its content
//! *below* the window — a retained layer holds its whole list and its whole
//! graph — so the occlusion pass early-outs on primitives that clip to nothing,
//! and the measurement is really about ordering. Zoomed out, the same primitive
//! count is packed into the visible area, most of it overlapping, and occlusion
//! has real work to do. Both are run.
//!
//! **CPU path**: `ordering::painter_orders_via_tree` (a faithful port of
//! `src/bounds_tree.rs`, the algorithm the renderer runs today) plus
//! `ordering::draw_order` (standing in for `Scene::finish`'s `sort_by_key`),
//! plus `occlusion::keep_mask`.
//!
//! **A note on fairness that runs against this benchmark's own interest.**
//! `occlusion::keep_mask` walks the *same* two-level AABB hierarchy the shader
//! does. The renderer's actual occlusion sweep (`src/occlusion.rs`) has no such
//! structure — it accumulates an unbounded occluder list and tests every cullee
//! against all of it, which is quadratic at this scale. Measuring against that
//! would flatter the GPU enormously. So the CPU side is given the accelerated
//! algorithm, and the occlusion comparison below is CPU-versus-GPU execution of
//! one algorithm rather than one algorithm versus another. The ordering
//! comparison is *not* levelled this way and does not need to be:
//! `BoundsTree` is already the fast structure, and it is what ships.
//!
//! **GPU path**: `OrderingPass::run` and `OcclusionPass::run`. The timed window
//! covers input encoding, buffer creation, uploads, every dispatch, the submit,
//! the poll, and the ordering pass's convergence readbacks — an end-to-end
//! wall-clock number from the CPU's point of view, on Phase 0's own terms.
//! Pipeline construction is *outside* it and is reported separately, because a
//! real frame builds pipelines once at startup and Spike A's own write-up
//! blames its 2–3× run-to-run variance on having built them inside the window.
//! Reading the *results* back is also outside it, for Spike A's other stated
//! reason: a real consumer would not round-trip them. The occlusion pass's
//! completion is forced inside the window, since a submitted-but-unwaited
//! dispatch is not work done.
//!
//! Correctness is checked, not assumed: every run compares the GPU's orders,
//! draw permutation, and keep mask against the CPU reference element for
//! element and reports the mismatch count.

use std::time::{Duration, Instant};

use wgpui_core::geometry::Rect;
use wgpui_core::occlusion::{
    CoverageItem, encode_coverage_items, encode_poison_regions, keep_mask, quad_coverage_item,
};
use wgpui_core::ordering::{draw_order, encode_ordering_items, painter_orders_via_tree};
use wgpui_core::patch::primitive::Quad;
use wgpui_core::test_support::ui_walk::{SceneDriver, UiFrame, UiSceneSpec, build_frame};
use wgpui_wgpu::render::compute::occlusion_pass::OcclusionPass;
use wgpui_wgpu::render::compute::ordering_pass::OrderingPass;
use wgpui_wgpu::render::device::{ComputeContext, headless_compute_context};

/// Sized so each scene lands near Spike A's 100,000 primitives, for a
/// like-for-like reading against `docs/phase-0-results.md`.
const LIST_ROWS: u32 = 9_000;
const NODES: u32 = 12_000;
/// Zoom for the dense scene: nodes about a sixth of their normal size, which
/// packs the same primitive count into the visible viewport.
const DENSE_SCALE: f32 = 0.16;
const DENSE_LIST_ROWS: u32 = 400;
const DENSE_NODES: u32 = 16_000;
const RUNS: u32 = 6;
/// Runs discarded before any timing is recorded, on both sides equally.
///
/// Phase 0's Spike A reported a 14.7× first run against a 5.5–6.9× cluster and
/// concluded, correctly, that the outlier was warm-up rather than signal — but
/// it had no way to separate them, because every run was a fresh process. Here
/// the runs share one, so the warm-up can simply be excluded and *named*
/// instead of being averaged in and then argued away. The CPU side discards the
/// same count for the same reason: its first pass is the one that faults in the
/// scene's pages.
const WARMUP_RUNS: u32 = 2;

struct CpuTiming {
    ordering: Duration,
    sort: Duration,
    occlusion: Duration,
}

impl CpuTiming {
    fn total(&self) -> Duration {
        self.ordering + self.sort + self.occlusion
    }
}

fn coverage_items(quads: &[Quad], frame: &UiFrame) -> Vec<CoverageItem> {
    quads
        .iter()
        .map(|quad| quad_coverage_item(quad, frame.clip, false))
        .collect()
}

fn bounds_of(quads: &[Quad]) -> Vec<Rect> {
    quads
        .iter()
        .map(|quad| Rect::from_origin_size(quad.origin, quad.size))
        .collect()
}

fn main() {
    println!("=== Phase 3 gate #2: ordering + occlusion, GPU compute vs. the CPU path ===");

    let context = match headless_compute_context() {
        Ok(context) => context,
        Err(error) => {
            println!("NO USABLE GPU ADAPTER: {error}");
            println!(
                "The performance half of Phase 3's gate cannot be reported from this machine. \
                 The correctness half does not need one — see `cargo test -p wgpui-core` and \
                 `cargo test -p wgpui-wgpu --test compute_differential`."
            );
            return;
        }
    };
    println!("Adapter: {}", context.describe());
    if context.is_software() {
        println!(
            "WARNING: this is a software rasterizer. Any timing below says nothing about real \
             hardware and must not be quoted as if it did."
        );
    }

    // Pipelines are built once, outside every measured window — see the module
    // doc for why that is the honest place for them.
    let pipeline_start = Instant::now();
    let ordering = OrderingPass::new(&context.device);
    let occlusion = OcclusionPass::new(&context.device);
    println!(
        "Compute pipelines built once in {:.3?} (excluded from every timing below)",
        pipeline_start.elapsed()
    );

    measure(
        &context,
        &ordering,
        &occlusion,
        "A. normal zoom — most content scrolled below the window",
        &UiSceneSpec::large(LIST_ROWS, NODES),
    );
    measure(
        &context,
        &ordering,
        &occlusion,
        "B. zoomed out — the same primitive count packed into the visible area",
        &UiSceneSpec::large_dense(DENSE_LIST_ROWS, DENSE_NODES, DENSE_SCALE),
    );
}

fn measure(
    context: &ComputeContext,
    ordering: &OrderingPass,
    occlusion: &OcclusionPass,
    title: &str,
    spec: &UiSceneSpec,
) {
    println!("\n================ Scene {title} ================");
    let frame = build_frame("benchmark", spec);

    // --- Scene construction, through the real APIs. Not part of either
    // measurement: both paths consume the same already-resident scene.
    let build_start = Instant::now();
    let mut driver = SceneDriver::new();
    let plan = match driver.apply_frame(&frame.quads) {
        Ok(plan) => plan,
        Err(error) => {
            println!("scene construction failed: {error:?}");
            return;
        }
    };
    let build_time = build_start.elapsed();
    let quads = driver.resident_quads();

    let bounds = bounds_of(&quads);
    let items = coverage_items(&quads, &frame);
    let opaque = items.iter().filter(|item| item.opaque.is_some()).count();
    let visible = items.iter().filter(|item| !item.visible.is_empty()).count();
    println!(
        "  {}x{} window, {} list rows, {} nodes, content scale {}",
        spec.width, spec.height, spec.list_rows, spec.nodes, spec.content_scale
    );
    println!(
        "  built through Scene/PrimitiveStore in {build_time:.3?}: {} primitives resident, \
         {} upload entries, {} bytes",
        quads.len(),
        plan.len(),
        plan.byte_count()
    );
    println!(
        "  {visible} of {} primitives clip to a non-empty visible region; {opaque} qualify as \
         occluders under R-N §8.3",
        items.len()
    );

    let mut cpu_reference_orders = Vec::new();
    let mut cpu_reference_draw = Vec::new();
    let mut cpu_reference_keep = Vec::new();

    println!(
        "\n  --- CPU path (BoundsTree + stable sort + accelerated coverage sweep) ---\n      \
         the first {WARMUP_RUNS} runs are warm-up and are excluded from the summary"
    );
    let mut cpu_timings = Vec::new();
    for run in 1..=RUNS {
        let ordering_start = Instant::now();
        let orders = painter_orders_via_tree(&bounds);
        let ordering_time = ordering_start.elapsed();

        let sort_start = Instant::now();
        let draw = draw_order(&orders);
        let sort_time = sort_start.elapsed();

        let occlusion_start = Instant::now();
        let keep = keep_mask(&items, &frame.poison);
        let occlusion_time = occlusion_start.elapsed();

        let timing = CpuTiming {
            ordering: ordering_time,
            sort: sort_time,
            occlusion: occlusion_time,
        };
        println!(
            "    run {run}{}: BoundsTree {:>10.3?}  sort {:>10.3?}  occlusion {:>10.3?}  \
             total {:>10.3?}",
            if run <= WARMUP_RUNS { " (warm-up)" } else { "         " },
            timing.ordering,
            timing.sort,
            timing.occlusion,
            timing.total()
        );
        if run > WARMUP_RUNS {
            cpu_timings.push(timing);
        }
        cpu_reference_orders = orders;
        cpu_reference_draw = draw;
        cpu_reference_keep = keep;
    }

    let culled = cpu_reference_keep.iter().filter(|kept| !**kept).count();
    let max_order = cpu_reference_orders.iter().copied().max().unwrap_or(0);
    println!(
        "    {culled} of {} primitives culled ({:.1}% overall, {:.1}% of the visible ones); \
         deepest painter order {max_order}",
        quads.len(),
        100.0 * culled as f64 / quads.len().max(1) as f64,
        100.0 * culled as f64 / visible.max(1) as f64
    );

    // --- GPU path.
    println!(
        "\n  --- GPU compute path (end-to-end: encode, upload, dispatch, submit, poll) ---\n      \
         the first {WARMUP_RUNS} runs are warm-up and are excluded from the summary"
    );
    let mut bounds_bytes = Vec::new();
    let mut item_bytes = Vec::new();
    let mut poison_bytes = Vec::new();
    let mut gpu_totals = Vec::new();
    let mut gpu_ordering_totals = Vec::new();
    let mut gpu_occlusion_totals = Vec::new();

    for run in 1..=RUNS {
        let ordering_start = Instant::now();
        encode_ordering_items(&bounds, &mut bounds_bytes);
        let ordered = match ordering.run(&context.device, &context.queue, &bounds_bytes) {
            Ok(output) => output,
            Err(error) => {
                println!("    run {run}: ordering pass failed: {error}");
                return;
            }
        };
        let ordering_time = ordering_start.elapsed();

        let occlusion_start = Instant::now();
        encode_coverage_items(&items, &mut item_bytes);
        encode_poison_regions(&frame.poison, &mut poison_bytes);
        let culled_output =
            match occlusion.run(&context.device, &context.queue, &item_bytes, &poison_bytes) {
                Ok(output) => output,
                Err(error) => {
                    println!("    run {run}: occlusion pass failed: {error}");
                    return;
                }
            };
        // `run` submits without waiting; reading the mask is what forces
        // completion, and the gate's number has to include the work rather than
        // only its submission.
        let gpu_keep =
            match occlusion.read_keep_mask(&context.device, &context.queue, &culled_output) {
                Ok(mask) => mask,
                Err(error) => {
                    println!("    run {run}: reading the keep mask failed: {error}");
                    return;
                }
            };
        let occlusion_time = occlusion_start.elapsed();
        let total = ordering_time + occlusion_time;

        println!(
            "    run {run}{}: ordering {:>10.3?} ({} relax iterations, {} submissions)  \
             occlusion {:>10.3?}  total {:>10.3?}",
            if run <= WARMUP_RUNS { " (warm-up)" } else { "         " },
            ordering_time,
            ordered.iterations,
            ordered.submissions,
            occlusion_time,
            total
        );
        if run > WARMUP_RUNS {
            gpu_totals.push(total);
            gpu_ordering_totals.push(ordering_time);
            gpu_occlusion_totals.push(occlusion_time);
        }

        // --- Correctness, outside the measured window.
        let gpu_orders = match ordering.read_orders(&context.device, &context.queue, &ordered) {
            Ok(values) => values,
            Err(error) => {
                println!("    run {run}: reading painter orders failed: {error}");
                return;
            }
        };
        let gpu_draw = match ordering.read_draw_order(&context.device, &context.queue, &ordered) {
            Ok(values) => values,
            Err(error) => {
                println!("    run {run}: reading the draw permutation failed: {error}");
                return;
            }
        };
        println!(
            "      exact match vs. CPU reference — orders: {} mismatches, draw order: {} \
             mismatches, keep mask: {} mismatches",
            mismatches(&gpu_orders, &cpu_reference_orders),
            mismatches(&gpu_draw, &cpu_reference_draw),
            mismatches(&gpu_keep, &cpu_reference_keep),
        );
    }

    // --- Summary. Median and best are both reported: one statistic alone
    // either hides the variance or lets an outlier stand in for the number.
    let cpu_total: Vec<Duration> = cpu_timings.iter().map(CpuTiming::total).collect();
    let cpu_order: Vec<Duration> = cpu_timings.iter().map(|t| t.ordering + t.sort).collect();
    let cpu_occlude: Vec<Duration> = cpu_timings.iter().map(|t| t.occlusion).collect();

    println!(
        "\n  --- Summary ({} primitives, {} timed runs after {WARMUP_RUNS} warm-up) ---",
        quads.len(),
        cpu_timings.len()
    );
    report("ordering (tree+sort vs. relax+bitonic)", &cpu_order, &gpu_ordering_totals);
    report("occlusion (coverage sweep)", &cpu_occlude, &gpu_occlusion_totals);
    report("both", &cpu_total, &gpu_totals);
}

fn report(what: &str, cpu: &[Duration], gpu: &[Duration]) {
    println!("    {what}");
    line("median", median(cpu), median(gpu));
    line("best", best(cpu), best(gpu));
}

fn line(statistic: &str, cpu: Duration, gpu: Duration) {
    let verdict = if gpu.is_zero() || cpu.is_zero() {
        "unmeasurable".to_string()
    } else if gpu < cpu {
        format!("GPU {:.2}x faster", cpu.as_secs_f64() / gpu.as_secs_f64())
    } else {
        format!("GPU {:.2}x SLOWER", gpu.as_secs_f64() / cpu.as_secs_f64())
    };
    println!("      {statistic:<8} CPU {cpu:>10.3?}   GPU {gpu:>10.3?}   {verdict}");
}

fn best(values: &[Duration]) -> Duration {
    values.iter().copied().min().unwrap_or(Duration::ZERO)
}

fn mismatches<T: PartialEq>(left: &[T], right: &[T]) -> usize {
    if left.len() != right.len() {
        return left.len().max(right.len());
    }
    left.iter().zip(right).filter(|(a, b)| a != b).count()
}

fn median(values: &[Duration]) -> Duration {
    let mut values = values.to_vec();
    values.sort_unstable();
    if values.is_empty() {
        return Duration::ZERO;
    }
    // For an even count, the lower of the two middles rather than their mean:
    // every number this prints is then a duration that was actually observed.
    values
        .get((values.len() - 1) / 2)
        .copied()
        .unwrap_or(Duration::ZERO)
}
