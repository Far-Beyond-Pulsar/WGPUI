//! §8's Phase 4 gates, end to end on a real device.
//! See docs/gpu-native-architecture.md §5.3, §5.5, §8 Phase 4.
//!
//! > A clean window's CPU-side draw-issuing work is O(layer slots), independent
//! > of resident primitive count, measured directly.
//!
//! > **`WgpuSurface` check**: a viewport panel fully covered by a modal
//! > (occlusion-culled per §5.2/Phase 3) issues zero draws for its embedded 3D
//! > content, and `WgpuSurfaceHandle`'s existing concurrency tests pass
//! > unmodified against the unified consumer path.
//!
//! Plus the correctness claim the gates rest on and which neither of them
//! states: every draw mode renders the *same picture*. Four paths reach the
//! framebuffer — per-slot `draw_indirect`, `multi_draw_indirect`,
//! `multi_draw_indirect_count`, and the CPU-readback fallback — and a gate
//! about how cheaply the CPU issues draws is worth nothing if the draws differ.
//! The comparison is bit-exact rather than tolerant: all four write the same
//! format through the same shader, so anything but equality is a bug.
//!
//! # If there is no adapter
//!
//! Reports and returns, per Phase 0's standard.

use std::time::Duration;

use wgpui_core::boundary::compositor::{CompositeEntry, CompositeSource, ExternalSurfaceId};
use wgpui_core::geometry::Rect;
use wgpui_core::patch::primitive::{PrimitiveKind, Quad};
use wgpui_core::scene::layer::BoundaryId;
use wgpui_core::test_support::ui_walk::{MultiLayerSceneDriver, UiSceneSpec, build_frame};
use wgpui_wgpu::render::device::{ComputeContext, context_or_report};
use wgpui_wgpu::render::draw::DrawMode;
use wgpui_wgpu::render::frame::{Dirty, FrameInput, FrameOutput, FrameRenderer, OffscreenTarget};
use wgpui_wgpu::render::pipelines::TARGET_FORMAT;
use wgpui_wgpu::render::surface_registry::SurfaceRegistry;

const LAYERS: usize = 6;

fn window(spec: &UiSceneSpec) -> Rect {
    Rect::from_origin_size([0.0, 0.0], [spec.width, spec.height])
}

fn modes(context: &ComputeContext) -> Vec<DrawMode> {
    DrawMode::ALL
        .into_iter()
        .filter(|mode| mode.is_available(context.indirect))
        .collect()
}

/// Render one frame in `mode` and hand back the pixels beside the stats.
fn render(
    context: &ComputeContext,
    renderer: &mut FrameRenderer,
    target: &OffscreenTarget,
    input: &FrameInput<'_>,
) -> (FrameOutput, Vec<u8>) {
    let output = renderer
        .render(&context.device, &context.queue, input, target)
        .expect("a frame must render");
    let pixels = target
        .read_pixels(&context.device, &context.queue)
        .expect("reading the target back must succeed");
    (output, pixels)
}

/// Where two framebuffers first disagree, as a pixel index.
fn first_difference(left: &[u8], right: &[u8]) -> Option<usize> {
    if left.len() != right.len() {
        return Some(0);
    }
    left.chunks_exact(4)
        .zip(right.chunks_exact(4))
        .position(|(a, b)| a != b)
}

fn scene_slot_table_len(frame: &wgpui_core::test_support::ui_walk::UiFrame) -> usize {
    let mut scene = MultiLayerSceneDriver::new(LAYERS);
    scene.apply_frame(frame).expect("the frame applies");
    scene.scene.draw_slots().len()
}

fn painted_pixels(pixels: &[u8]) -> usize {
    pixels
        .chunks_exact(4)
        .filter(|pixel| pixel[0] != 0 || pixel[1] != 0 || pixel[2] != 0)
        .count()
}

/// Every available draw mode renders the same picture.
///
/// The claim both gates rest on. Without it, "the CPU issued one call instead
/// of forty" would be a statement about a different image.
#[test]
fn every_draw_mode_renders_the_same_picture() {
    let Some(context) = context_or_report("indirect_draw_modes") else {
        return;
    };
    let spec = UiSceneSpec::small();
    let frame = build_frame("initial", &spec);
    let mut scene = MultiLayerSceneDriver::new(LAYERS);
    scene.apply_frame(&frame).expect("the frame applies");

    let mut renderer = FrameRenderer::new(&context.device);
    let target = OffscreenTarget::new(&context.device, spec.width as u32, spec.height as u32);

    let available = modes(&context);
    println!(
        "indirect_draw_modes: {} of {} modes available on this device: {:?}",
        available.len(),
        DrawMode::ALL.len(),
        available.iter().map(|mode| mode.name()).collect::<Vec<_>>()
    );
    assert!(
        available.contains(&DrawMode::PerSlotIndirect)
            && available.contains(&DrawMode::CpuReadback),
        "the two featureless modes must always be available"
    );

    let mut reference: Option<Vec<u8>> = None;
    for mode in available {
        let input = FrameInput {
            scene: &scene.scene,
            clip: window(&spec),
            poison: &scene.poison,
            dirty: Dirty::All,
            uploads: &[],
            composites: &[],
            registry: None,
            viewport: [spec.width, spec.height],
            mode,
        };
        let (output, pixels) = render(&context, &mut renderer, &target, &input);

        assert!(
            painted_pixels(&pixels) > (spec.width * spec.height / 8.0) as usize,
            "{} painted almost nothing, so comparing it proves nothing",
            mode.name()
        );
        match &reference {
            None => {
                println!(
                    "  reference {}: {} slots, {} draw calls, {} painted pixels",
                    mode.name(),
                    output.stats.slots_visited,
                    output.stats.draw_calls_issued,
                    painted_pixels(&pixels)
                );
                reference = Some(pixels);
            }
            Some(expected) => {
                assert_eq!(
                    first_difference(expected, &pixels),
                    None,
                    "{} rendered a different picture from the reference mode",
                    mode.name()
                );
                println!(
                    "  {}: identical, {} draw calls, instances known to CPU: {:?}",
                    mode.name(),
                    output.stats.draw_calls_issued,
                    output.stats.instances_known_to_cpu
                );
            }
        }
    }
}

/// **Gate 1**: a clean window's CPU-side draw-issuing work is O(layer slots),
/// independent of resident primitive count.
///
/// Two scenes, the same six layers, a ~40× difference in resident primitives.
/// Every draw-issuing counter is asserted *equal*, not merely similar, and the
/// clock is reported beside them rather than asserted on — a wall clock on a
/// shared machine is evidence, and a counter is proof.
#[test]
fn gate_1_a_clean_windows_draw_issuing_work_is_independent_of_primitive_count() {
    let Some(context) = context_or_report("gate_1_draw_issuance") else {
        return;
    };
    let spec = UiSceneSpec::small();
    let small = build_frame("small", &spec);
    let large = build_frame(
        "large",
        &UiSceneSpec {
            list_rows: 3_000,
            nodes: 1_500,
            ..spec
        },
    );

    let measure = |frame: &wgpui_core::test_support::ui_walk::UiFrame,
                   mode: DrawMode|
     -> (FrameOutput, Duration) {
        let mut scene = MultiLayerSceneDriver::new(LAYERS);
        scene.apply_frame(frame).expect("the frame applies");
        let mut renderer = FrameRenderer::new(&context.device);
        let target = OffscreenTarget::new(&context.device, spec.width as u32, spec.height as u32);

        // Frame 1 is dirty: it does the uploads and the compute. Every later
        // frame is the clean window the gate is about.
        let dirty = FrameInput {
            scene: &scene.scene,
            clip: window(&spec),
            poison: &scene.poison,
            dirty: Dirty::All,
            uploads: &[],
            composites: &[],
            registry: None,
            viewport: [spec.width, spec.height],
            mode,
        };
        renderer
            .render(&context.device, &context.queue, &dirty, &target)
            .expect("the first frame must render");

        let clean = FrameInput {
            dirty: Dirty::Some(&[]),
            ..dirty
        };
        // A handful of clean frames, taking the best draw-issue time: the
        // machine is shared and the first is warm-up, exactly as Phase 3's
        // benchmark methodology records.
        let mut best = Duration::MAX;
        let mut output = renderer
            .render(&context.device, &context.queue, &clean, &target)
            .expect("a clean frame must render");
        for _ in 0..16 {
            output = renderer
                .render(&context.device, &context.queue, &clean, &target)
                .expect("a clean frame must render");
            best = best.min(output.timing.draw_issue);
        }
        (output, best)
    };

    // Every available mode, not just the best one: the gate's claim is about
    // the fixed sequence, and a mode that collapses it to one call would satisfy
    // "the same at both counts" trivially while telling us nothing about the
    // per-slot path a featureless device takes.
    let mut small_output = None;
    let mut large_output = None;
    for mode in modes(&context) {
        let (small_result, small_time) = measure(&small, mode);
        let (large_result, large_time) = measure(&large, mode);
        println!(
            "gate_1_draw_issuance [{}]: {} primitives -> {} slots, {} draw calls, \
             {} binds, best draw-issue {:?}",
            mode.name(),
            small_result.primitives_resident,
            small_result.stats.slots_visited,
            small_result.stats.draw_calls_issued,
            small_result.stats.bind_group_binds,
            small_time
        );
        println!(
            "gate_1_draw_issuance [{}]: {} primitives -> {} slots, {} draw calls, \
             {} binds, best draw-issue {:?}",
            mode.name(),
            large_result.primitives_resident,
            large_result.stats.slots_visited,
            large_result.stats.draw_calls_issued,
            large_result.stats.bind_group_binds,
            large_time
        );
        assert_eq!(
            large_result.stats.draw_calls_issued, small_result.stats.draw_calls_issued,
            "[{}] draw-issuing work must not grow with the primitive count",
            mode.name()
        );
        assert_eq!(
            large_result.stats.bind_group_binds, small_result.stats.bind_group_binds,
            "[{}] bind-group work must not grow with the primitive count",
            mode.name()
        );
        assert_eq!(
            large_result.stats.slots_visited, small_result.stats.slots_visited,
            "[{}] the fixed sequence must be the same length at both counts",
            mode.name()
        );
        if mode == DrawMode::PerSlotIndirect {
            assert_eq!(
                large_result.stats.draw_calls_issued as usize, LAYERS,
                "the per-slot path must issue exactly one call per slot — the \
                 form of the claim that is not satisfiable by collapsing"
            );
            small_output = Some(small_result);
            large_output = Some(large_result);
        }
    }

    let small_output = small_output.expect("the per-slot path is always available");
    let large_output = large_output.expect("the per-slot path is always available");

    assert!(
        large_output.primitives_resident > small_output.primitives_resident * 20,
        "the two scenes differ by too little for the comparison to mean anything: \
         {} against {}",
        large_output.primitives_resident,
        small_output.primitives_resident
    );
    assert_eq!(
        large_output.stats.slots_visited, small_output.stats.slots_visited,
        "the fixed sequence must be the same length at both primitive counts"
    );
    assert_eq!(
        large_output.stats.draw_calls_issued, small_output.stats.draw_calls_issued,
        "§8's Phase 4 gate: draw-issuing work is O(layer slots), not O(primitives)"
    );
    assert_eq!(
        large_output.stats.bind_group_binds,
        small_output.stats.bind_group_binds
    );
    assert_eq!(
        large_output.stats.slots_visited as usize,
        LAYERS,
        "one entry per layer for the one kind that has a pipeline — the slot \
         table also names every layer's GlyphRun slot, and nothing draws those \
         because Phase 4 built one instanced pipeline (see `render/frame.rs`)"
    );
    assert_eq!(
        scene_slot_table_len(&large),
        LAYERS * PrimitiveKind::COUNT,
        "the table itself does name every (layer, kind) pair, including the \
         kinds a layer holds nothing of"
    );
    assert_eq!(
        large_output.layers_recomputed, 0,
        "a clean frame must recompute no layer's ordering or occlusion"
    );

    // The counter that says the CPU never learned what it drew.
    assert_eq!(
        large_output.stats.instances_known_to_cpu, None,
        "§5.3: the GPU decides how much work each call expands to \"without the \
         CPU ever finding out the count\""
    );
    assert_eq!(large_output.stats.readback_words, 0);
}

/// The same gate, taken the other way: the CPU-readback fallback *does* learn
/// the counts, and pays for it in a way the counters make visible.
///
/// Worth asserting because it is the difference the gate is claiming. If the
/// fallback also reported `None`, the counter would be measuring nothing.
#[test]
fn the_fallback_path_is_the_one_that_learns_the_counts() {
    let Some(context) = context_or_report("fallback_learns_counts") else {
        return;
    };
    let spec = UiSceneSpec::small();
    let mut scene = MultiLayerSceneDriver::new(LAYERS);
    scene
        .apply_frame(&build_frame("initial", &spec))
        .expect("the frame applies");
    // One layer emptied on purpose: its slot stays in the fixed sequence with a
    // zero reservation, which is the case §5.3's "regardless of how many are
    // actually zero" is about and the one the fallback can decline to issue.
    scene.set_layer(0, &[]).expect("emptying a layer is fine");
    let mut renderer = FrameRenderer::new(&context.device);
    let target = OffscreenTarget::new(&context.device, spec.width as u32, spec.height as u32);

    let input = FrameInput {
        scene: &scene.scene,
        clip: window(&spec),
        poison: &scene.poison,
        dirty: Dirty::All,
        uploads: &[],
        composites: &[],
        registry: None,
        viewport: [spec.width, spec.height],
        mode: DrawMode::CpuReadback,
    };
    let (first, _) = render(&context, &mut renderer, &target, &input);
    let known = first
        .stats
        .instances_known_to_cpu
        .expect("the fallback reads the records back, so it knows");
    assert!(known > 0);
    assert!(first.stats.readback_words > 0);
    assert_eq!(
        first.stats.slots_skipped + first.stats.draw_calls_issued,
        first.stats.slots_visited,
        "every slot is either drawn or knowingly skipped"
    );
    assert_eq!(
        first.stats.slots_skipped, 1,
        "the emptied layer's slot is the one the fallback declines to issue — \
         the one thing an indirect path cannot do, because it does not know"
    );

    // The staging buffer must be reused across frames, or the fallback
    // allocates GPU memory every frame in the one path that exists because the
    // device is already the weak one.
    let allocations_after_first = renderer.readback_allocations();
    for _ in 0..20 {
        renderer
            .render(&context.device, &context.queue, &input, &target)
            .expect("a frame must render");
    }
    assert_eq!(
        renderer.readback_allocations(),
        allocations_after_first,
        "the readback staging buffer must not be reallocated in steady state"
    );
}

/// **Gate 2**: a viewport panel fully covered by a modal issues zero draws for
/// its embedded 3D content — and the producer's frame is not consumed, which is
/// the observable form of "stops being drawn at all".
#[test]
fn gate_2_a_covered_viewport_issues_no_draws_and_consumes_no_produced_frame() {
    let Some(context) = context_or_report("gate_2_covered_viewport") else {
        return;
    };
    let spec = UiSceneSpec::small();
    let registry = SurfaceRegistry::new();
    let surface = registry.create(&context.device, 256, 256, TARGET_FORMAT);

    // The external render thread presents a frame. Nothing about how it does so
    // changed in this phase.
    registry.swap_rendering_ready_no_sync(surface);
    assert!(
        registry.has_unconsumed_frame(surface),
        "the producer just presented, so its frame is unconsumed"
    );

    let mut scene = MultiLayerSceneDriver::new(1);
    scene.clip = window(&spec);
    scene
        .set_layer(0, &[chrome_quad(&spec)])
        .expect("seeding a layer must succeed");
    let mut renderer = FrameRenderer::new(&context.device);
    let target = OffscreenTarget::new(&context.device, spec.width as u32, spec.height as u32);

    // The modal is a texture-retained boundary with a real baked texture, so
    // both composite entries are real and the difference between the two cases
    // is the layer tier's decision and nothing else.
    let modal_boundary = BoundaryId::from_raw(42);
    renderer.textures.begin_frame();
    renderer.textures.acquire(
        &context.device,
        modal_boundary,
        spec.width as u32,
        spec.height as u32,
        1,
    );

    let viewport_entry = CompositeEntry::sampled(
        CompositeSource::External(ExternalSurfaceId::from_raw(surface.as_raw())),
        Rect::from_origin_size([80.0, 60.0], [240.0, 180.0]),
        window(&spec),
    );
    let modal = |bounds: Rect| CompositeEntry {
        source_is_opaque: true,
        ..CompositeEntry::sampled(
            CompositeSource::BoundaryTexture(modal_boundary),
            bounds,
            window(&spec),
        )
    };

    fn frame_input<'a>(
        scene: &'a MultiLayerSceneDriver,
        spec: &UiSceneSpec,
        registry: &'a SurfaceRegistry,
        composites: &'a [CompositeEntry],
        mode: DrawMode,
    ) -> FrameInput<'a> {
        FrameInput {
            scene: &scene.scene,
            clip: Rect::from_origin_size([0.0, 0.0], [spec.width, spec.height]),
            poison: &scene.poison,
            dirty: Dirty::All,
            uploads: &[],
            composites,
            registry: Some(registry),
            viewport: [spec.width, spec.height],
            mode,
        }
    }
    let mode = DrawMode::best_available(context.indirect);

    // --- Covered: a modal over the whole window.
    let covered = [viewport_entry, modal(window(&spec))];
    let (output, _) = render(
        &context,
        &mut renderer,
        &target,
        &frame_input(&scene, &spec, &registry, &covered, mode),
    );
    assert_eq!(
        output.stats.composite_entries_visited, 2,
        "both entries were considered"
    );
    assert_eq!(
        output.stats.composite_entries_culled, 1,
        "the layer tier must drop the covered viewport"
    );
    assert_eq!(
        output.stats.composite_entries_unavailable, 0,
        "both producers have something ready, so neither case is an accident"
    );
    assert_eq!(
        output.stats.composite_draws_issued, 1,
        "§5.5: only the modal draws — the viewport fully covered by it issues \
         zero draws for its embedded 3D content"
    );
    assert!(
        registry.has_unconsumed_frame(surface),
        "the covered viewport must not have consumed the producer's frame — \
         this is what \"stops being drawn at all\" means on the consumer side"
    );

    // --- Uncovered: the same modal moved clear of the viewport.
    let uncovered = [viewport_entry, modal(Rect::from_origin_size(
        [400.0, 0.0],
        [80.0, 320.0],
    ))];
    let (output, _) = render(
        &context,
        &mut renderer,
        &target,
        &frame_input(&scene, &spec, &registry, &uncovered, mode),
    );
    assert_eq!(
        output.stats.composite_entries_culled, 0,
        "with the modal moved clear, nothing is covered"
    );
    assert_eq!(
        output.stats.composite_draws_issued, 2,
        "the viewport and the modal both draw"
    );
    assert!(
        !registry.has_unconsumed_frame(surface),
        "a viewport that is actually composited consumes the frame its producer \
         presented, exactly as the legacy surfaces batch does"
    );
}

/// The control for gate 2's culling claim: with no cover at all, the viewport's
/// composite entry actually paints, so the culled case is measuring an absence
/// rather than a path that never worked.
#[test]
fn an_uncovered_composite_entry_actually_paints() {
    let Some(context) = context_or_report("composite_paints") else {
        return;
    };
    let spec = UiSceneSpec::small();
    let mut renderer = FrameRenderer::new(&context.device);
    let target = OffscreenTarget::new(&context.device, spec.width as u32, spec.height as u32);
    let panel = BoundaryId::from_raw(3);

    // Bake a solid colour into a boundary texture, exactly as a texture-retained
    // boundary's rasterize step would.
    renderer.textures.begin_frame();
    renderer
        .textures
        .acquire(&context.device, panel, 128, 128, 1);
    let view = renderer
        .textures
        .view(panel)
        .expect("the boundary has a texture")
        .clone();
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bake"),
        });
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("bake"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color {
                    r: 1.0,
                    g: 0.0,
                    b: 0.0,
                    a: 1.0,
                }),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: Default::default(),
    });
    context.queue.submit(Some(encoder.finish()));

    let mut scene = MultiLayerSceneDriver::new(1);
    scene.clip = window(&spec);
    scene.set_layer(0, &[]).expect("an empty layer is fine");

    let entry = CompositeEntry::sampled(
        CompositeSource::BoundaryTexture(panel),
        Rect::from_origin_size([40.0, 40.0], [128.0, 128.0]),
        window(&spec),
    );
    let input = FrameInput {
        scene: &scene.scene,
        clip: window(&spec),
        poison: &scene.poison,
        dirty: Dirty::All,
        uploads: &[],
        composites: std::slice::from_ref(&entry),
        registry: None,
        viewport: [spec.width, spec.height],
        mode: DrawMode::best_available(context.indirect),
    };
    let (output, pixels) = render(&context, &mut renderer, &target, &input);
    assert_eq!(output.stats.composite_draws_issued, 1);
    assert_eq!(
        painted_pixels(&pixels),
        128 * 128,
        "the baked texture must land as exactly its own rectangle"
    );

    // And covering it removes exactly those pixels.
    let cover = CompositeEntry {
        source_is_opaque: true,
        ..CompositeEntry::sampled(
            CompositeSource::BoundaryTexture(BoundaryId::from_raw(99)),
            window(&spec),
            window(&spec),
        )
    };
    let covered = [entry, cover];
    let (output, pixels) = render(
        &context,
        &mut renderer,
        &target,
        &FrameInput {
            composites: &covered,
            ..input
        },
    );
    assert_eq!(output.stats.composite_entries_culled, 1);
    assert_eq!(output.stats.composite_draws_issued, 0);
    assert_eq!(
        painted_pixels(&pixels),
        0,
        "nothing painted, because the cover has no texture and the covered \
         entry was dropped"
    );
}

fn chrome_quad(spec: &UiSceneSpec) -> Quad {
    Quad {
        origin: [0.0, 0.0],
        size: [spec.width, 24.0],
        background: [0.15, 0.16, 0.2, 1.0],
        border_color: [0.0, 0.0, 0.0, 0.0],
        corner_radius: 0.0,
        border_width: 0.0,
    }
}
