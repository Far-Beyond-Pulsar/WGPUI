//! Phase 3 gate #1, the compute arm: `WGPUI_OCCLUSION=validate`'s discipline
//! (R-N §8.5) run against the GPU passes.
//! See docs/gpu-native-architecture.md §5.1, §5.2, §8 Phase 3.
//!
//! Three claims, checked per frame of a scripted UI walk over a scene resident
//! in a real `wgpui_core::scene::Scene`:
//!
//! 1. The ordering pass computes the same painter orders and the same draw
//!    permutation as the CPU `BoundsTree`, exactly — not approximately.
//! 2. The occlusion pass computes the same keep mask as the CPU reference,
//!    exactly.
//! 3. Rasterizing the scene with culling on and with culling off, both through
//!    the *GPU-computed* draw order, produces bit-identical framebuffers. This
//!    is the gate's own wording — "culled/unculled scenes match exactly" —
//!    checked as pixels rather than as a restatement of the culling rule.
//!
//! Claims 1 and 2 are what make the compute path a *port*; claim 3 is what
//! makes culling provably an optimization. Losing either one of the first two
//! without the third would mean the shader and the reference agree on being
//! wrong, so all three are asserted rather than the strongest one alone.
//!
//! # If there is no adapter
//!
//! This test reports and returns rather than failing. That is not a hole in the
//! gate: `wgpui-core`'s own `gate_1_culled_and_unculled_scenes_are_pixel_identical_over_a_scripted_walk`
//! runs the identical walk through the CPU reference with no device at all and
//! is not skippable. What is lost without an adapter is the *port* half —
//! whether the WGSL agrees with the Rust — and Phase 0's report sets the
//! standard for saying so plainly rather than implying coverage that did not
//! run.

use wgpui_core::geometry::Rect;
use wgpui_core::occlusion::{
    CoverageItem, encode_coverage_items, encode_poison_regions, keep_mask, quad_coverage_item,
};
use wgpui_core::ordering::{draw_order, encode_ordering_items, painter_orders_via_tree};
use wgpui_core::patch::primitive::Quad;
use wgpui_core::test_support::raster::rasterize;
use wgpui_core::test_support::ui_walk::{SceneDriver, UiFrame, UiSceneSpec, scripted_walk};
use wgpui_wgpu::render::compute::occlusion_pass::OcclusionPass;
use wgpui_wgpu::render::compute::ordering_pass::OrderingPass;
use wgpui_wgpu::render::device::{ComputeContext, headless_compute_context};

/// Open a device, or explain why the compute arm did not run.
fn context_or_skip(test: &str) -> Option<ComputeContext> {
    match headless_compute_context() {
        Ok(context) => {
            println!("{test}: running on {}", context.describe());
            Some(context)
        }
        Err(error) => {
            println!(
                "{test}: SKIPPED — {error}. The CPU arm of this gate \
                 (wgpui_core::occlusion::tests) still ran; the WGSL/Rust agreement \
                 half did not, and a human must re-run this on hardware."
            );
            None
        }
    }
}

fn bounds_of(quads: &[Quad]) -> Vec<Rect> {
    quads
        .iter()
        .map(|quad| Rect::from_origin_size(quad.origin, quad.size))
        .collect()
}

fn coverage_items(quads: &[Quad], frame: &UiFrame) -> Vec<CoverageItem> {
    quads
        .iter()
        .map(|quad| quad_coverage_item(quad, frame.clip, false))
        .collect()
}

#[test]
fn gate_1_the_compute_path_matches_the_cpu_reference_over_a_scripted_walk() {
    let Some(context) = context_or_skip("gate_1_compute_arm") else {
        return;
    };
    let ordering = OrderingPass::new(&context.device);
    let occlusion = OcclusionPass::new(&context.device);

    let spec = UiSceneSpec::small();
    let width = spec.width as u32;
    let height = spec.height as u32;
    let mut driver = SceneDriver::new();

    let mut item_bytes = Vec::new();
    let mut poison_bytes = Vec::new();
    let mut bounds_bytes = Vec::new();

    let mut total_culled = 0usize;
    let mut total_primitives = 0usize;
    let mut deepest_relaxation = 0u32;

    for frame in scripted_walk(&spec) {
        driver
            .apply_frame(&frame.quads)
            .expect("the walk's frames must apply to a real scene");
        let quads = driver.resident_quads();
        let bounds = bounds_of(&quads);
        let items = coverage_items(&quads, &frame);

        // --- Claim 1: ordering.
        encode_ordering_items(&bounds, &mut bounds_bytes);
        let ordered = ordering
            .run(&context.device, &context.queue, &bounds_bytes)
            .expect("the ordering pass must converge");
        let gpu_orders = ordering
            .read_orders(&context.device, &context.queue, &ordered)
            .expect("reading painter orders back must succeed");
        let gpu_draw = ordering
            .read_draw_order(&context.device, &context.queue, &ordered)
            .expect("reading the draw permutation back must succeed");
        deepest_relaxation = deepest_relaxation.max(ordered.iterations);

        let cpu_orders = painter_orders_via_tree(&bounds);
        assert_eq!(
            gpu_orders, cpu_orders,
            "frame {}: the compute ordering pass disagrees with the CPU BoundsTree",
            frame.label
        );
        assert_eq!(
            gpu_draw,
            draw_order(&cpu_orders),
            "frame {}: the GPU draw permutation is not the stable painter order",
            frame.label
        );

        // --- Claim 2: occlusion.
        encode_coverage_items(&items, &mut item_bytes);
        encode_poison_regions(&frame.poison, &mut poison_bytes);
        let culled = occlusion
            .run(&context.device, &context.queue, &item_bytes, &poison_bytes)
            .expect("the occlusion pass must dispatch");
        let gpu_keep = occlusion
            .read_keep_mask(&context.device, &context.queue, &culled)
            .expect("reading the keep mask back must succeed");
        let cpu_keep = keep_mask(&items, &frame.poison);
        assert_eq!(
            gpu_keep.len(),
            cpu_keep.len(),
            "frame {}: the compute pass decided a different number of primitives",
            frame.label
        );
        let first_disagreement = gpu_keep
            .iter()
            .zip(&cpu_keep)
            .position(|(gpu, cpu)| gpu != cpu);
        assert_eq!(
            first_disagreement, None,
            "frame {}: the compute occlusion pass disagrees with the CPU reference \
             at primitive {first_disagreement:?}",
            frame.label
        );

        // --- Claim 3: the gate's own wording, over the GPU's own outputs.
        let unculled = rasterize(&quads, &gpu_draw, None, width, height);
        let culled_image = rasterize(&quads, &gpu_draw, Some(&gpu_keep), width, height);
        assert_eq!(
            unculled.first_difference(&culled_image),
            None,
            "frame {}: GPU culling changed the rendered output",
            frame.label
        );
        assert!(
            unculled.painted_pixel_count() > (width * height / 2) as usize,
            "frame {} painted almost nothing, so comparing it proves nothing",
            frame.label
        );

        let culled_count = gpu_keep.iter().filter(|kept| !**kept).count();
        assert!(
            culled_count > 0,
            "frame {}: the compute pass culled nothing, so this frame tested nothing",
            frame.label
        );
        total_culled += culled_count;
        total_primitives += quads.len();
    }

    println!(
        "gate_1_compute_arm: {total_culled} of {total_primitives} primitives culled \
         across the walk; deepest relaxation {deepest_relaxation} iterations"
    );
    assert!(
        total_culled * 20 > total_primitives,
        "the walk culled only {total_culled} of {total_primitives} primitives — \
         too few for the comparison to be meaningful"
    );
}

/// The differential above compares two implementations of the same rules. This
/// checks that the *shader* is the thing being compared, by feeding it a case
/// whose right answer is known independently of either implementation: a stack
/// of concentric opaque squares each *larger* than the last, where every square
/// but the topmost is covered and the painter orders are exactly 1, 2, 3, ...
#[test]
fn the_compute_passes_agree_with_a_hand_computed_answer() {
    let Some(context) = context_or_skip("hand_computed") else {
        return;
    };
    let ordering = OrderingPass::new(&context.device);
    let occlusion = OcclusionPass::new(&context.device);

    let count = 12u32;
    let quads: Vec<Quad> = (0..count)
        .map(|index| Quad {
            // Each square strictly contains every earlier one, so each overlaps
            // all of them (the recurrence must step by exactly one) and each is
            // completely covered by the next (every cull but the last).
            origin: [16.0 - index as f32, 16.0 - index as f32],
            size: [100.0 + 2.0 * index as f32, 100.0 + 2.0 * index as f32],
            background: [0.2, 0.3, 0.4, 1.0],
            border_color: [0.0, 0.0, 0.0, 1.0],
            corner_radii: [0.0; 4],
            border_widths: [0.0; 4],
            material: wgpui_core::patch::primitive::Material::Solid,
        })
        .collect();

    let mut bounds_bytes = Vec::new();
    encode_ordering_items(&bounds_of(&quads), &mut bounds_bytes);
    let ordered = ordering
        .run(&context.device, &context.queue, &bounds_bytes)
        .expect("the ordering pass must converge");
    let orders = ordering
        .read_orders(&context.device, &context.queue, &ordered)
        .expect("reading painter orders back must succeed");
    assert_eq!(
        orders,
        (1..=count).collect::<Vec<u32>>(),
        "a strictly nested stack must produce orders 1..n"
    );
    assert!(
        ordered.iterations >= count,
        "a chain of {count} needs at least that many relaxation iterations, ran {}",
        ordered.iterations
    );

    let clip = Rect::from_origin_size([0.0, 0.0], [256.0, 256.0]);
    let items: Vec<CoverageItem> = quads
        .iter()
        .map(|quad| quad_coverage_item(quad, clip, false))
        .collect();
    let mut item_bytes = Vec::new();
    let mut poison_bytes = Vec::new();
    encode_coverage_items(&items, &mut item_bytes);
    encode_poison_regions(&[], &mut poison_bytes);
    let culled = occlusion
        .run(&context.device, &context.queue, &item_bytes, &poison_bytes)
        .expect("the occlusion pass must dispatch");
    let keep = occlusion
        .read_keep_mask(&context.device, &context.queue, &culled)
        .expect("reading the keep mask back must succeed");

    let mut expected = vec![false; count as usize];
    if let Some(last) = expected.last_mut() {
        *last = true;
    }
    assert_eq!(
        keep, expected,
        "every square but the topmost is completely covered by the one above it"
    );
}

/// The occlusion shader's `FLAG_CULLABLE` bit, differentially — the one flag
/// this file has never exercised.
///
/// `CoverageItem::cullable` has existed since Phase 3 and every item every test
/// here builds comes from `quad_coverage_item`, which always sets it. So the
/// shader's `(item.flags & FLAG_CULLABLE) == 0u` branch was reachable in
/// principle and unreached in practice for three phases. Phase 6.3's `Shadow`
/// is the first primitive kind that takes it, on every frame, which makes the
/// gap worth closing here rather than trusting the branch on inspection.
///
/// The scene is the strongest available shape: an item completely covered by an
/// opaque square above it, built twice — once cullable, once not — so the
/// comparison is against the *flag* and not against the geometry.
#[test]
fn the_shader_honours_the_uncullable_flag_exactly_as_the_cpu_does() {
    let Some(context) = context_or_skip("uncullable_flag") else {
        return;
    };
    let occlusion = OcclusionPass::new(&context.device);
    let covered = Rect::from_origin_size([20.0, 20.0], [40.0, 40.0]);
    let cover = Rect::from_origin_size([0.0, 0.0], [200.0, 200.0]);

    for (label, hidden, expected_first) in [
        ("cullable", CoverageItem::cullee(covered), false),
        ("uncullable", CoverageItem::uncullable(covered), true),
    ] {
        let items = [hidden, CoverageItem::occluder(cover, cover)];
        let mut item_bytes = Vec::new();
        let mut poison_bytes = Vec::new();
        encode_coverage_items(&items, &mut item_bytes);
        encode_poison_regions(&[], &mut poison_bytes);
        let culled = occlusion
            .run(&context.device, &context.queue, &item_bytes, &poison_bytes)
            .expect("the occlusion pass must dispatch");
        let gpu = occlusion
            .read_keep_mask(&context.device, &context.queue, &culled)
            .expect("reading the keep mask back must succeed");
        let cpu = keep_mask(&items, &[]);

        assert_eq!(gpu, cpu, "[{label}] the two paths must agree");
        assert_eq!(
            gpu,
            vec![expected_first, true],
            "[{label}] and they must agree on the right answer, not merely with \
             each other"
        );
    }
}

/// A single overlap chain, at two depths, is the sharpest test of the
/// relaxation's iteration count as well as its exactness — the painter order is
/// exactly `1..count` and nothing shallower reaches it.
fn ordered_chain(context: &ComputeContext, ordering: &OrderingPass, count: u32) -> (Vec<u32>, u32) {
    // Each square overlaps its predecessor by one pixel and nothing else.
    let bounds: Vec<Rect> = (0..count)
        .map(|index| Rect::from_origin_size([index as f32 * 4.0, 0.0], [5.0, 10.0]))
        .collect();

    let mut bounds_bytes = Vec::new();
    encode_ordering_items(&bounds, &mut bounds_bytes);
    let ordered = ordering
        .run(&context.device, &context.queue, &bounds_bytes)
        .expect("the ordering pass must converge");
    let orders = ordering
        .read_orders(&context.device, &context.queue, &ordered)
        .expect("reading painter orders back must succeed");

    assert_eq!(orders, painter_orders_via_tree(&bounds));
    assert_eq!(orders.last().copied(), Some(count));
    (orders, ordered.submissions)
}

/// A layer whose overlap chain is deeper than the first relaxation batch must
/// still come out exact — the difference between "runs to a fixed point" and
/// "runs a budget and hopes," which is the thing Phase 0's spike did not have
/// to answer.
///
/// The depth at which that second submission becomes necessary is itself the
/// measurement this test carries. A per-primitive kernel advanced one primitive
/// per iteration, so 300 already needed several batches; the block-collapsing
/// kernel advances up to 64, so 300 now settles inside the first batch and it
/// takes a chain spanning more than `RELAX_FIRST_BATCH` blocks to leave it.
/// Both are asserted, so a regression in either direction — losing exactness,
/// or quietly losing the collapse — fails here rather than only showing up as a
/// slower benchmark.
#[test]
fn a_chain_deeper_than_one_relaxation_batch_still_converges() {
    let Some(context) = context_or_skip("deep_chain") else {
        return;
    };
    let ordering = OrderingPass::new(&context.device);

    let (_, submissions) = ordered_chain(&context, &ordering, 300);
    assert_eq!(
        submissions, 1,
        "a 300-deep chain spans 5 blocks and must settle inside the first batch"
    );

    // 2,048 primitives is 32 blocks, twice `RELAX_FIRST_BATCH`, so the
    // convergence loop genuinely has to run a second time.
    let (_, submissions) = ordered_chain(&context, &ordering, 2_048);
    assert!(
        submissions > 1,
        "a 2048-deep chain spans more blocks than the first batch has iterations, \
         so it should have needed more than one submission, took {submissions}"
    );
}

/// An empty layer must not dispatch anything, allocate a zero-sized binding, or
/// fail. Real frames hit this constantly — a boundary whose content was just
/// swept holds nothing.
#[test]
fn an_empty_layer_costs_the_passes_nothing() {
    let Some(context) = context_or_skip("empty_layer") else {
        return;
    };
    let ordering = OrderingPass::new(&context.device);
    let occlusion = OcclusionPass::new(&context.device);

    let ordered = ordering
        .run(&context.device, &context.queue, &[])
        .expect("an empty layer must not fail");
    assert_eq!(ordered.count, 0);
    assert_eq!(ordered.submissions, 0);
    assert!(
        ordering
            .read_orders(&context.device, &context.queue, &ordered)
            .expect("reading nothing back must succeed")
            .is_empty()
    );

    let culled = occlusion
        .run(&context.device, &context.queue, &[], &[])
        .expect("an empty layer must not fail");
    assert_eq!(culled.count, 0);
}

/// Malformed input is reported, not read past the end of. Both passes take raw
/// bytes from an encoder they do not control.
#[test]
fn a_mis_sized_input_is_rejected_rather_than_misread() {
    let Some(context) = context_or_skip("malformed_input") else {
        return;
    };
    let ordering = OrderingPass::new(&context.device);
    let occlusion = OcclusionPass::new(&context.device);

    assert!(
        ordering
            .run(&context.device, &context.queue, &[0u8; 17])
            .is_err()
    );
    assert!(
        occlusion
            .run(&context.device, &context.queue, &[0u8; 7], &[])
            .is_err()
    );
    assert!(
        occlusion
            .run(&context.device, &context.queue, &[], &[0u8; 5])
            .is_err()
    );
}
