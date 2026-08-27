//! §5.3's indirect-arg generation, checked against its CPU reference the way
//! Phase 3 checked ordering and occlusion against theirs.
//! See docs/gpu-native-architecture.md §5.3, §8 Phase 4.
//!
//! The claim is narrow and it is the only one a compute shader supports:
//! `shaders/indirect_args.wgsl` is a faithful transcription of
//! `wgpui_core::indirect::indirect_args`, which is itself tested headlessly
//! against hand-computed cases. Both the argument records and the indirection
//! buffer are compared, not just the records — a compaction that produced the
//! right *counts* from the wrong *instances* would draw the wrong primitives in
//! the right quantity, which is exactly the bug a count-only comparison misses.
//!
//! The inputs are Phase 3's real outputs, moved into place by
//! `IndirectArgsPass::scatter` with no readback in between, which is the seam
//! `docs/phase-3-results.md` §2 named as open ("nothing yet consumes them into
//! a draw call").
//!
//! # If there is no adapter
//!
//! Reports and returns, per Phase 0's standard. `wgpui-core`'s own
//! `indirect::tests` run the same computation with no device at all and are not
//! skippable; what is lost here is the WGSL/Rust agreement half.

use wgpui_core::geometry::Rect;
use wgpui_core::indirect::{
    DrawSlot, FirstInstance, UNUSED_INSTANCE, encode_slots, indirect_args,
};
use wgpui_core::occlusion::{encode_coverage_items, encode_poison_regions};
use wgpui_core::ordering::encode_ordering_items;
use wgpui_core::patch::primitive::{PrimitiveKind, Quad};
use wgpui_core::scene::layer::{BoundaryId, LayerId, LayerKey};
use wgpui_core::test_support::ui_walk::{
    MultiLayerSceneDriver, UiFrame, UiSceneSpec, build_frame, scripted_walk,
};
use wgpui_wgpu::render::compute::indirect_args_pass::{IndirectArgsBuffers, IndirectArgsPass};
use wgpui_wgpu::render::compute::occlusion_pass::OcclusionPass;
use wgpui_wgpu::render::compute::ordering_pass::OrderingPass;
use wgpui_wgpu::render::device::{ComputeContext, context_or_report, headless_compute_context};

/// How many layers the harness splits a frame across. Enough that "one draw per
/// (layer, kind) slot" is a claim about a sequence rather than about one entry,
/// and few enough that every frame runs Phase 3 six times without the
/// differential taking minutes.
const HARNESS_LAYERS: usize = 6;

/// One frame's Phase 3 + Phase 4 run over a multi-layer scene, checked against
/// the CPU reference at every step.
struct Harness {
    ordering: OrderingPass,
    occlusion: OcclusionPass,
    indirect: IndirectArgsPass,
}

impl Harness {
    fn new(device: &wgpu::Device) -> Harness {
        Harness {
            ordering: OrderingPass::new(device),
            occlusion: OcclusionPass::new(device),
            indirect: IndirectArgsPass::new(device),
        }
    }

    /// Run Phase 3 per layer, scatter into the arena, run Phase 4 over every
    /// slot, and return the GPU's answer beside the CPU's.
    fn run(
        &self,
        context: &ComputeContext,
        scene: &MultiLayerSceneDriver,
        first_instance: FirstInstance,
    ) -> (
        Vec<wgpui_core::indirect::DrawIndirectArgs>,
        Vec<u32>,
        wgpui_core::indirect::IndirectArgs,
        u32,
    ) {
        self.run_kind(context, scene, PrimitiveKind::Quad, first_instance)
    }

    fn run_kind(
        &self,
        context: &ComputeContext,
        scene: &MultiLayerSceneDriver,
        kind: PrimitiveKind,
        first_instance: FirstInstance,
    ) -> (
        Vec<wgpui_core::indirect::DrawIndirectArgs>,
        Vec<u32>,
        wgpui_core::indirect::IndirectArgs,
        u32,
    ) {
        let device = &context.device;
        let queue = &context.queue;
        let table = scene.scene.draw_slots();
        let slots: Vec<DrawSlot> = table.kind_slots(kind).to_vec();
        let arena_slots = scene.scene.arena_slots(kind);

        let buffers = IndirectArgsBuffers::new(device, arena_slots, slots.len() as u32);

        // Phase 3, per layer, then scattered into arena position — the whole of
        // the wiring, and no readback anywhere in it.
        let mut cpu_draw_order = vec![0u32; arena_slots as usize];
        let mut cpu_culled = vec![0u32; arena_slots as usize];
        let mut bounds_bytes = Vec::new();
        let mut item_bytes = Vec::new();
        let mut poison_bytes = Vec::new();
        for slot in &slots {
            let quads = scene.layer_quads(slot.layer);
            let bounds: Vec<Rect> = quads
                .iter()
                .map(|quad| Rect::from_origin_size(quad.origin, quad.size))
                .collect();
            let items = scene.coverage_items(slot.layer);

            encode_ordering_items(&bounds, &mut bounds_bytes);
            let ordered = self
                .ordering
                .run(device, queue, &bounds_bytes)
                .expect("the ordering pass must converge");
            encode_coverage_items(&items, &mut item_bytes);
            encode_poison_regions(&scene.poison, &mut poison_bytes);
            let culled = self
                .occlusion
                .run(device, queue, &item_bytes, &poison_bytes)
                .expect("the occlusion pass must dispatch");

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("scatter phase 3 output"),
            });
            IndirectArgsPass::scatter(
                &mut encoder,
                &ordered.draw_order,
                &buffers.draw_order,
                slot.base,
                slot.count,
            );
            IndirectArgsPass::scatter(
                &mut encoder,
                &culled.culled,
                &buffers.culled,
                slot.base,
                slot.count,
            );
            queue.submit(Some(encoder.finish()));

            // The CPU reference reads the same numbers, so read Phase 3's back
            // here (outside anything measured) rather than recomputing them:
            // the claim under test is Phase 4's transcription, not Phase 3's,
            // which has its own gate.
            let orders = self
                .ordering
                .read_draw_order(device, queue, &ordered)
                .expect("reading the draw permutation back must succeed");
            let keep = self
                .occlusion
                .read_keep_mask(device, queue, &culled)
                .expect("reading the keep mask back must succeed");
            for (local, value) in orders.iter().enumerate() {
                if let Some(entry) = cpu_draw_order.get_mut(slot.base as usize + local) {
                    *entry = *value;
                }
            }
            for (local, kept) in keep.iter().enumerate() {
                if let Some(entry) = cpu_culled.get_mut(slot.base as usize + local) {
                    *entry = u32::from(!*kept);
                }
            }
        }

        let mut slot_bytes = Vec::new();
        encode_slots(&slots, &mut slot_bytes);
        let output = self
            .indirect
            .run(
                device,
                queue,
                &buffers,
                &slot_bytes,
                wgpui_core::indirect::QUAD_VERTEX_COUNT,
                first_instance,
            )
            .expect("the indirect-arg pass must dispatch");

        let gpu_args = self
            .indirect
            .read_args(device, queue, &buffers, output.slot_count)
            .expect("reading the argument records back must succeed");
        let gpu_visible = self
            .indirect
            .read_visible(device, queue, &buffers)
            .expect("reading the indirection buffer back must succeed");
        let gpu_packed_count = self
            .indirect
            .read_draw_count(device, queue, &buffers)
            .expect("reading the packed count back must succeed");

        let reference = indirect_args(
            &slots,
            &cpu_draw_order,
            &cpu_culled,
            arena_slots as usize,
            wgpui_core::indirect::QUAD_VERTEX_COUNT,
            first_instance,
        );
        (gpu_args, gpu_visible, reference, gpu_packed_count)
    }
}

/// **Phase 4's differential**: over a scripted UI walk in a scene with one layer
/// per panel, the compute pass's argument records and indirection buffer equal
/// the CPU reference's exactly, in both `first_instance` encodings.
#[test]
fn the_indirect_args_pass_matches_the_cpu_reference_over_a_scripted_walk() {
    let Some(context) = context_or_report("indirect_args_differential") else {
        return;
    };
    let harness = Harness::new(&context.device);
    let spec = UiSceneSpec::small();
    let mut scene = MultiLayerSceneDriver::new(HARNESS_LAYERS);

    let mut total_slots = 0usize;
    let mut total_empty_slots = 0usize;
    let mut total_instances = 0u32;
    let mut total_culled = 0u32;

    for frame in scripted_walk(&spec) {
        scene
            .apply_frame(&frame)
            .expect("the walk's frames must apply to a real scene");

        // §5.3's "regardless of how many are actually zero", exercised rather
        // than asserted: none of these layers holds a glyph run, so every
        // GlyphRun slot is a live entry in the fixed sequence that expands to
        // no work at all.
        let (empty_kind_args, _, empty_kind_reference, empty_kind_packed) = harness.run_kind(
            &context,
            &scene,
            PrimitiveKind::GlyphRun,
            FirstInstance::Zero,
        );
        assert_eq!(empty_kind_args, empty_kind_reference.args);
        assert_eq!(empty_kind_packed, 0);
        assert!(
            !empty_kind_args.is_empty() && empty_kind_args.iter().all(|record| record.is_empty()),
            "frame {}: every layer should contribute an unpopulated GlyphRun slot",
            frame.label
        );
        total_slots += empty_kind_args.len();
        total_empty_slots += empty_kind_args.len();

        for encoding in [FirstInstance::Zero, FirstInstance::SlotBase] {
            let (gpu_args, gpu_visible, reference, packed_count) =
                harness.run(&context, &scene, encoding);

            assert_eq!(
                gpu_args, reference.args,
                "frame {} ({encoding:?}): the compute pass's argument records \
                 disagree with the CPU reference",
                frame.label
            );
            assert_eq!(
                gpu_visible.len(),
                reference.visible.len(),
                "frame {}: the indirection buffer is the wrong length",
                frame.label
            );
            let first_disagreement = gpu_visible
                .iter()
                .zip(&reference.visible)
                .position(|(gpu, cpu)| gpu != cpu);
            assert_eq!(
                first_disagreement, None,
                "frame {} ({encoding:?}): the indirection buffer disagrees at \
                 entry {first_disagreement:?} — a compaction that gets the counts \
                 right and the instances wrong draws the wrong primitives",
                frame.label
            );
            assert_eq!(
                packed_count as usize,
                reference.packed.len(),
                "frame {} ({encoding:?}): the packed draw count disagrees",
                frame.label
            );

            if encoding == FirstInstance::Zero {
                assert!(
                    gpu_args.iter().all(|record| record.first_instance == 0),
                    "frame {}: README's Custom Device Gotcha — the default \
                     encoding must never emit a nonzero firstInstance",
                    frame.label
                );
                total_slots += gpu_args.len();
                total_empty_slots += gpu_args.iter().filter(|a| a.is_empty()).count();
                let drawn: u32 = gpu_args.iter().map(|a| a.instance_count).sum();
                total_instances += drawn;
                total_culled += scene
                    .scene
                    .draw_slots()
                    .kind_slots(PrimitiveKind::Quad)
                    .iter()
                    .map(|slot| slot.count)
                    .sum::<u32>()
                    - drawn;
            }
        }
    }

    println!(
        "indirect_args_differential: {total_slots} slot records across the walk, \
         {total_empty_slots} of them empty, {total_instances} instances drawn, \
         {total_culled} compacted away"
    );
    assert!(
        total_empty_slots > 0,
        "no slot came out empty across the whole walk, so the \"a fixed \
         sequence includes the zeroes\" half of §5.3 was never exercised"
    );
    assert!(
        total_culled > 0,
        "the compaction never dropped an instance across the whole walk, so \
         `instance_count < count` was never exercised and the comparison is \
         only checking that nothing was culled"
    );
    assert!(total_instances > 0);
}

/// The same walk driven into **one** layer, where culling actually bites.
///
/// This is not redundant with the multi-layer differential above, and the
/// reason is a real property of the design rather than a quirk of the harness:
/// instance-tier occlusion is scoped per layer (R-N §8.2, so an occluder
/// animating above a static layer can never churn that layer's slab), so
/// splitting a scene across more layers *reduces* how much of it is culled. The
/// six-layer harness above therefore exercises the slot sequence well and the
/// compaction barely; this exercises the compaction against a mask that drops
/// hundreds of primitives, which is where an off-by-one in the running offset
/// between chunks would show up.
#[test]
fn the_indirect_args_pass_matches_the_cpu_reference_when_most_of_a_layer_is_culled() {
    let Some(context) = context_or_report("indirect_args_single_layer") else {
        return;
    };
    let harness = Harness::new(&context.device);
    let mut scene = MultiLayerSceneDriver::new(1);
    let mut total_culled = 0u32;

    for frame in scripted_walk(&UiSceneSpec::small()) {
        scene
            .apply_frame(&frame)
            .expect("the walk's frames must apply to a real scene");
        let (gpu_args, gpu_visible, reference, packed) =
            harness.run(&context, &scene, FirstInstance::Zero);
        assert_eq!(gpu_args, reference.args, "frame {}", frame.label);
        assert_eq!(
            gpu_visible
                .iter()
                .zip(&reference.visible)
                .position(|(gpu, cpu)| gpu != cpu),
            None,
            "frame {}: the indirection buffer disagrees with the reference",
            frame.label
        );
        assert_eq!(packed as usize, reference.packed.len());

        let reserved: u32 = scene
            .scene
            .draw_slots()
            .kind_slots(PrimitiveKind::Quad)
            .iter()
            .map(|slot| slot.count)
            .sum();
        let drawn: u32 = gpu_args.iter().map(|record| record.instance_count).sum();
        assert!(drawn <= reserved);
        total_culled += reserved - drawn;
    }

    println!("indirect_args_single_layer: {total_culled} instances compacted away");
    assert!(
        total_culled > 100,
        "only {total_culled} instances were compacted away across the walk — \
         too few for a chunked compaction's running offset to have been tested"
    );
}

/// The compaction's own rule, on a case whose answer is known without either
/// implementation: a stack of concentric squares each larger than the last, so
/// every one but the topmost is covered, drawn bottom-to-top.
#[test]
fn a_fully_covered_layer_produces_a_zero_instance_record() {
    let Some(context) = context_or_report("fully_covered_layer") else {
        return;
    };
    let harness = Harness::new(&context.device);

    // One layer only: slots are ordered by `LayerId`, which is a hash of the
    // layer key, so with several layers "the first record" would name whichever
    // layer hashed lowest rather than the one seeded here.
    let mut scene = MultiLayerSceneDriver::new(1);
    let covered: Vec<Quad> = (0..12u32)
        .map(|index| Quad {
            origin: [16.0 - index as f32, 16.0 - index as f32],
            size: [100.0 + 2.0 * index as f32, 100.0 + 2.0 * index as f32],
            background: [0.2, 0.3, 0.4, 1.0],
            border_color: [0.0, 0.0, 0.0, 1.0],
            corner_radius: 0.0,
            border_width: 0.0,
        })
        .collect();
    scene.clip = Rect::from_origin_size([0.0, 0.0], [512.0, 512.0]);
    scene
        .set_layer(0, &covered)
        .expect("seeding a layer must succeed");

    let (args, visible, reference, packed) = harness.run(&context, &scene, FirstInstance::Zero);
    assert_eq!(args, reference.args);
    assert_eq!(args.len(), 1);
    assert_eq!(
        args.first().map(|record| record.instance_count),
        Some(1),
        "eleven of twelve concentric squares are covered; only the top draws"
    );
    assert_eq!(packed, 1);
    let base = scene
        .scene
        .draw_slots()
        .kind_slots(PrimitiveKind::Quad)
        .first()
        .map(|slot| slot.base as usize)
        .expect("the seeded layer has a quad slot");
    assert_eq!(
        visible.get(base).copied(),
        Some(base as u32 + 11),
        "the surviving instance is the last one in paint order"
    );
    assert!(
        visible
            .iter()
            .skip(base + 1)
            .take(11)
            .all(|entry| *entry == UNUSED_INSTANCE),
        "entries past the live prefix must be the sentinel, not stale indices"
    );
}

/// A slot table that disagrees with its arena is reported rather than
/// dispatched with — a `copy_buffer_to_buffer` or a shader write past the end
/// is an uncaptured device error, which aborts the process by default.
#[test]
fn a_slot_outside_its_arena_is_rejected() {
    let Some(context) = context_or_report("slot_outside_arena") else {
        return;
    };
    let pass = IndirectArgsPass::new(&context.device);
    let buffers = IndirectArgsBuffers::new(&context.device, 64, 4);
    let mut bytes = Vec::new();
    encode_slots(
        &[DrawSlot {
            layer: LayerId::from_raw(1),
            kind: PrimitiveKind::Quad,
            base: 32,
            count: 64,
        }],
        &mut bytes,
    );
    assert!(
        pass.run(
            &context.device,
            &context.queue,
            &buffers,
            &bytes,
            4,
            FirstInstance::Zero
        )
        .is_err()
    );
    assert!(
        pass.run(
            &context.device,
            &context.queue,
            &buffers,
            &[0u8; 17],
            4,
            FirstInstance::Zero
        )
        .is_err(),
        "a mis-sized slot table must be reported, not read past the end of"
    );
}

/// An empty scene has no slots, dispatches nothing, and does not fail. Real
/// frames hit this: a window whose only boundary was just swept holds nothing.
#[test]
fn an_empty_slot_table_costs_the_pass_nothing() {
    let Some(context) = context_or_report("empty_slot_table") else {
        return;
    };
    let pass = IndirectArgsPass::new(&context.device);
    let buffers = IndirectArgsBuffers::new(&context.device, 0, 0);
    let output = pass
        .run(
            &context.device,
            &context.queue,
            &buffers,
            &[],
            4,
            FirstInstance::Zero,
        )
        .expect("an empty slot table must not fail");
    assert_eq!(output.slot_count, 0);
    assert_eq!(
        pass.read_draw_count(&context.device, &context.queue, &buffers)
            .expect("reading the packed count back must succeed"),
        0
    );
}

/// The slot table is what §8's Phase 4 gate is about, stated where a real
/// `Scene` can be asked: two scenes with the same layers and a thousandfold
/// difference in primitive count produce the same number of slots.
#[test]
fn the_slot_table_does_not_grow_with_the_primitive_count() {
    let spec = UiSceneSpec::small();
    let small = build_frame("small", &spec);
    let large = build_frame(
        "large",
        &UiSceneSpec {
            list_rows: 4_000,
            nodes: 2_000,
            ..spec
        },
    );

    let slots_of = |frame: &UiFrame| -> usize {
        let mut scene = MultiLayerSceneDriver::new(HARNESS_LAYERS);
        scene.apply_frame(frame).expect("frame applies");
        scene.scene.draw_slots().len()
    };
    assert_eq!(slots_of(&small), slots_of(&large));
    assert!(large.quads.len() > small.quads.len() * 10);
}

/// A guard on the harness above rather than on the code: the multi-layer scene
/// must actually be multi-layer, or every claim about "one per (layer, kind)
/// slot" is being made about a single slot.
#[test]
fn the_harness_scene_holds_more_than_one_layer() {
    let mut scene = MultiLayerSceneDriver::new(HARNESS_LAYERS);
    scene
        .apply_frame(&build_frame("initial", &UiSceneSpec::small()))
        .expect("frame applies");
    assert!(scene.scene.layers.len() >= 3);
    assert_eq!(
        scene
            .scene
            .draw_slots()
            .kind_slots(PrimitiveKind::Quad)
            .len(),
        scene.scene.layers.len(),
        "every layer holds quads, so every layer is one quad slot"
    );
    assert_ne!(
        LayerId::from_key(LayerKey::untiled(BoundaryId::from_raw(1))),
        LayerId::from_key(LayerKey::untiled(BoundaryId::from_raw(2)))
    );
}

/// Guard against the differential above passing vacuously if the device
/// disappears: it must be an explicit skip, never a silent one.
#[test]
fn a_missing_adapter_is_reported_rather_than_passing_silently() {
    match headless_compute_context() {
        Ok(context) => println!("indirect draw tests ran on {}", context.describe()),
        Err(error) => println!(
            "NO ADAPTER — every GPU test in this file skipped: {error}. \
             wgpui-core's own indirect::tests still ran."
        ),
    }
}
