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
use wgpui_core::scene::layer::LayerTransform;
use wgpui_core::test_support::ui_walk::{MultiLayerSceneDriver, UiSceneSpec, build_frame};
use wgpui_wgpu::render::device::{ComputeContext, context_or_report};
use wgpui_wgpu::render::draw::DrawMode;
use wgpui_wgpu::render::frame::{Dirty, FrameInput, FrameOutput, FrameRenderer, OffscreenTarget};
use wgpui_wgpu::render::pipelines::TARGET_FORMAT;
use wgpui_wgpu::render::surface_registry::SurfaceRegistry;
use wgpui_wgpu::debug::DebugTile;

const LAYERS: usize = 6;

/// Primitive kinds with a render path. Backdrop filters are conditional: their
/// pass is omitted when the scene has no filter records.
const DRAWN_KINDS: usize = PrimitiveKind::COUNT;

const VISITED_KINDS: usize = DRAWN_KINDS - 1;

/// Of those, the ones that sample an atlas page and therefore report through
/// [`wgpui_wgpu::render::draw::DrawStats::sprite_slots_unavailable`]:
/// `GlyphRun` and `PolySprite`.
const SPRITE_KINDS: usize = 2;

/// The rest: kinds with a pipeline that samples no texture, so their slots are
/// always issuable. `Quad` since Phase 4 and `Shadow` since Phase 6.3 — both go
/// through [`wgpui_wgpu::render::draw::issue_instanced`], which is one function
/// for both.
///
/// This is the multiplier the per-slot path's draw count carries, and it was
/// implicitly `1` before Phase 6.3 rather than absent: two assertions below read
/// `LAYERS` where they meant `LAYERS * NON_ATLAS_KINDS`, and failed the moment a
/// second texture-free kind existed. Naming it is what stops that recurring.
const NON_ATLAS_KINDS: usize = DRAWN_KINDS - SPRITE_KINDS;

const INSTANCED_NON_ATLAS_KINDS: usize = NON_ATLAS_KINDS - 2;

const VISITED_NON_ATLAS_KINDS: usize = NON_ATLAS_KINDS - 1;

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
            atlas: None,
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

#[test]
fn a_layer_transform_moves_native_pixels_without_scene_uploads() {
    let Some(context) = context_or_report("layer_transform") else {
        return;
    };
    let spec = UiSceneSpec::small();
    let frame = build_frame("layer_transform", &spec);
    let mut scene = MultiLayerSceneDriver::new(1);
    scene.apply_frame(&frame).expect("the frame applies");
    let layer = scene
        .scene
        .layers
        .ids()
        .first()
        .copied()
        .expect("the test scene has one layer");
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
        atlas: None,
        viewport: [spec.width, spec.height],
        mode: DrawMode::best_available(context.indirect),
    };
    let (_, initial) = render(&context, &mut renderer, &target, &input);

    assert!(scene
        .scene
        .layers
        .set_transform(layer, LayerTransform::translated(32.0, 0.0)));
    let translated_input = FrameInput {
        scene: &scene.scene,
        clip: window(&spec),
        poison: &scene.poison,
        dirty: Dirty::All,
        uploads: &[],
        composites: &[],
        registry: None,
        atlas: None,
        viewport: [spec.width, spec.height],
        mode: DrawMode::best_available(context.indirect),
    };
    let (output, translated) = render(&context, &mut renderer, &target, &translated_input);

    assert!(first_difference(&initial, &translated).is_some());
    assert_eq!(output.scene_upload_bytes, 0);
}

#[test]
fn a_layer_clip_is_enforced_after_the_layer_transform() {
    let Some(context) = context_or_report("layer_clip") else {
        return;
    };
    let spec = UiSceneSpec {
        width: 128.0,
        height: 128.0,
        ..UiSceneSpec::small()
    };
    let quad = Quad {
        origin: [0.0, 0.0],
        size: [128.0, 128.0],
        background: [0.1, 0.8, 0.2, 1.0],
        ..Quad::ZERO
    };
    let mut scene = MultiLayerSceneDriver::new(1);
    scene.set_layer(0, &[quad]).expect("the clip scene applies");
    let layer = scene
        .scene
        .layers
        .ids()
        .first()
        .copied()
        .expect("the test scene has one layer");
    let clip = Rect::from_origin_size([16.0, 20.0], [48.0, 40.0]);
    assert!(scene.scene.layers.set_clip(layer, Some(clip)));
    assert!(scene
        .scene
        .layers
        .set_transform(layer, LayerTransform::translated(8.0, 10.0)));
    assert_eq!(
        scene.scene.draw_slots().kind_slots(PrimitiveKind::Quad)[0].layer,
        layer
    );

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
        atlas: None,
        viewport: [spec.width, spec.height],
        mode: DrawMode::best_available(context.indirect),
    };
    let (_, pixels) = render(&context, &mut renderer, &target, &input);

    let mut painted_inside = 0;
    for (index, pixel) in pixels.chunks_exact(4).enumerate() {
        let x = (index % spec.width as usize) as f32;
        let y = (index / spec.width as usize) as f32;
        let painted = pixel[0] != 0 || pixel[1] != 0 || pixel[2] != 0;
        let inside = x + 0.5 >= clip.min_x
            && x + 0.5 < clip.max_x
            && y + 0.5 >= clip.min_y
            && y + 0.5 < clip.max_y;
        if inside {
            painted_inside += usize::from(painted);
        } else {
            assert!(!painted, "layer clip leaked at ({x}, {y})");
        }
    }
    assert_eq!(painted_inside, (clip.width() * clip.height()) as usize);
}

#[test]
fn a_layer_transform_keeps_rounded_quad_edges_attached_to_the_quad() {
    let Some(context) = context_or_report("rounded_layer_transform") else {
        return;
    };
    let spec = UiSceneSpec {
        width: 128.0,
        height: 128.0,
        ..UiSceneSpec::small()
    };
    let quad = Quad {
        origin: [24.0, 28.0],
        size: [56.0, 44.0],
        background: [0.15, 0.35, 0.75, 1.0],
        border_color: [0.95, 0.9, 0.7, 1.0],
        corner_radii: [12.0; 4],
        border_widths: [2.0; 4],
        ..Quad::ZERO
    };
    let translation = [17.0, 11.0];
    let shifted = Quad {
        origin: [quad.origin[0] + translation[0], quad.origin[1] + translation[1]],
        ..quad
    };

    let mut transformed_scene = MultiLayerSceneDriver::new(1);
    transformed_scene
        .set_layer(0, &[quad])
        .expect("the rounded quad scene applies");
    let layer = transformed_scene
        .scene
        .layers
        .ids()
        .first()
        .copied()
        .expect("the test scene has one layer");
    let mut expected_scene = MultiLayerSceneDriver::new(1);
    expected_scene
        .set_layer(0, &[shifted])
        .expect("the shifted rounded quad scene applies");

    let mut renderer = FrameRenderer::new(&context.device);
    let target = OffscreenTarget::new(&context.device, spec.width as u32, spec.height as u32);
    let initial_input = FrameInput {
        scene: &transformed_scene.scene,
        clip: window(&spec),
        poison: &transformed_scene.poison,
        dirty: Dirty::All,
        uploads: &[],
        composites: &[],
        registry: None,
        atlas: None,
        viewport: [spec.width, spec.height],
        mode: DrawMode::best_available(context.indirect),
    };
    render(&context, &mut renderer, &target, &initial_input);
    assert!(transformed_scene
        .scene
        .layers
        .set_transform(layer, LayerTransform::translated(translation[0], translation[1])));
    let transformed_input = FrameInput {
        scene: &transformed_scene.scene,
        clip: window(&spec),
        poison: &transformed_scene.poison,
        dirty: Dirty::All,
        uploads: &[],
        composites: &[],
        registry: None,
        atlas: None,
        viewport: [spec.width, spec.height],
        mode: DrawMode::best_available(context.indirect),
    };
    let (_, transformed) = render(&context, &mut renderer, &target, &transformed_input);

    let expected_input = FrameInput {
        scene: &expected_scene.scene,
        clip: window(&spec),
        poison: &expected_scene.poison,
        dirty: Dirty::All,
        uploads: &[],
        composites: &[],
        registry: None,
        atlas: None,
        viewport: [spec.width, spec.height],
        mode: DrawMode::best_available(context.indirect),
    };
    let mut expected_renderer = FrameRenderer::new(&context.device);
    let (_, expected) = render(&context, &mut expected_renderer, &target, &expected_input);

    let differing_pixels = expected
        .chunks_exact(4)
        .zip(transformed.chunks_exact(4))
        .filter(|(expected, transformed)| expected != transformed)
        .count();
    assert!(
        differing_pixels <= 8,
        "layer translation changed {} rounded-edge pixels",
        differing_pixels
    );
}

#[test]
fn damaged_presentation_preserves_untouched_pixels() {
    let Some(context) = context_or_report("damaged_presentation") else {
        return;
    };
    let spec = UiSceneSpec {
        width: 128.0,
        height: 64.0,
        ..UiSceneSpec::small()
    };
    let quad = |origin: [f32; 2], background: [f32; 4]| Quad {
        origin,
        size: [48.0, 48.0],
        background,
        ..Quad::ZERO
    };
    let initial = [
        quad([8.0, 8.0], [0.8, 0.1, 0.1, 1.0]),
        quad([72.0, 8.0], [0.1, 0.2, 0.8, 1.0]),
    ];
    let updated = [
        quad([8.0, 8.0], [0.1, 0.8, 0.2, 1.0]),
        initial[1],
    ];

    let mut scene = MultiLayerSceneDriver::new(1);
    scene.set_layer(0, &initial).expect("the initial scene applies");
    let target = OffscreenTarget::new(&context.device, spec.width as u32, spec.height as u32);
    let mut renderer = FrameRenderer::new(&context.device);
    let initial_input = FrameInput {
        scene: &scene.scene,
        clip: window(&spec),
        poison: &scene.poison,
        dirty: Dirty::All,
        uploads: &[],
        composites: &[],
        registry: None,
        atlas: None,
        viewport: [spec.width, spec.height],
        mode: DrawMode::best_available(context.indirect),
    };
    render(&context, &mut renderer, &target, &initial_input);
    scene.set_layer(0, &updated).expect("the update applies");
    let mut renderer = FrameRenderer::new(&context.device);
    let updated_input = FrameInput {
        scene: &scene.scene,
        clip: window(&spec),
        poison: &scene.poison,
        dirty: Dirty::All,
        uploads: &[],
        composites: &[],
        registry: None,
        atlas: None,
        viewport: [spec.width, spec.height],
        mode: DrawMode::best_available(context.indirect),
    };
    let retained_target = target.target();
    renderer
        .render_to_with_damage(
            &context.device,
            &context.queue,
            &updated_input,
            &retained_target,
            Some(Rect::from_origin_size([0.0, 0.0], [64.0, 64.0])),
        )
        .expect("the damaged frame renders");
    let damaged = target
        .read_pixels(&context.device, &context.queue)
        .expect("the damaged frame reads back");

    let mut expected_scene = MultiLayerSceneDriver::new(1);
    expected_scene
        .set_layer(0, &updated)
        .expect("the expected scene applies");
    let expected_target = OffscreenTarget::new(&context.device, spec.width as u32, spec.height as u32);
    let mut expected_renderer = FrameRenderer::new(&context.device);
    let expected_input = FrameInput {
        scene: &expected_scene.scene,
        clip: window(&spec),
        poison: &expected_scene.poison,
        dirty: Dirty::All,
        uploads: &[],
        composites: &[],
        registry: None,
        atlas: None,
        viewport: [spec.width, spec.height],
        mode: DrawMode::best_available(context.indirect),
    };
    render(
        &context,
        &mut expected_renderer,
        &expected_target,
        &expected_input,
    );
    let expected = expected_target
        .read_pixels(&context.device, &context.queue)
        .expect("the expected frame reads back");
    assert_eq!(damaged, expected);
}

#[test]
fn damaged_presentation_clears_removed_content() {
    let Some(context) = context_or_report("damaged_presentation_removal") else {
        return;
    };
    let spec = UiSceneSpec {
        width: 64.0,
        height: 64.0,
        ..UiSceneSpec::small()
    };
    let quad = Quad {
        origin: [8.0, 8.0],
        size: [40.0, 40.0],
        background: [0.8, 0.2, 0.1, 1.0],
        ..Quad::ZERO
    };
    let mut scene = MultiLayerSceneDriver::new(1);
    scene.set_layer(0, &[quad]).expect("the initial scene applies");
    let target = OffscreenTarget::new(&context.device, spec.width as u32, spec.height as u32);
    let mut renderer = FrameRenderer::new(&context.device);
    let initial_input = FrameInput {
        scene: &scene.scene,
        clip: window(&spec),
        poison: &scene.poison,
        dirty: Dirty::All,
        uploads: &[],
        composites: &[],
        registry: None,
        atlas: None,
        viewport: [spec.width, spec.height],
        mode: DrawMode::best_available(context.indirect),
    };
    render(&context, &mut renderer, &target, &initial_input);

    scene.set_layer(0, &[]).expect("the removal applies");
    let updated_input = FrameInput {
        scene: &scene.scene,
        clip: window(&spec),
        poison: &scene.poison,
        dirty: Dirty::All,
        uploads: &[],
        composites: &[],
        registry: None,
        atlas: None,
        viewport: [spec.width, spec.height],
        mode: DrawMode::best_available(context.indirect),
    };
    let retained_target = target.target();
    FrameRenderer::new(&context.device)
        .render_to_with_damage(
            &context.device,
            &context.queue,
            &updated_input,
            &retained_target,
            Some(Rect::from_origin_size([0.0, 0.0], [64.0, 64.0])),
        )
        .expect("the damaged removal frame renders");
    let pixels = target
        .read_pixels(&context.device, &context.queue)
        .expect("the damaged removal frame reads back");
    assert!(pixels.chunks_exact(4).all(|pixel| pixel == [0, 0, 0, 255]));
}

#[test]
fn tile_refresh_diagnostics_draw_the_measured_rate_label() {
    let Some(context) = context_or_report("tile_refresh_rate_label") else {
        return;
    };
    let spec = UiSceneSpec {
        width: 64.0,
        height: 64.0,
        ..UiSceneSpec::small()
    };
    let scene = MultiLayerSceneDriver::new(1);
    let target = OffscreenTarget::new(&context.device, spec.width as u32, spec.height as u32);
    let mut renderer = FrameRenderer::new(&context.device);
    renderer.set_debug_tiles(vec![
        DebugTile {
            origin_size: [0.0, 0.0, 64.0, 64.0],
            color: [1.0, 0.0, 1.0, 0.25],
            border_width: 3.0,
            _padding: [0.0; 7],
        }
        .with_refresh_rate(60.0),
    ]);
    let input = FrameInput {
        scene: &scene.scene,
        clip: window(&spec),
        poison: &scene.poison,
        dirty: Dirty::All,
        uploads: &[],
        composites: &[],
        registry: None,
        atlas: None,
        viewport: [spec.width, spec.height],
        mode: DrawMode::best_available(context.indirect),
    };
    let (_, pixels) = render(&context, &mut renderer, &target, &input);
    assert!(
        pixels
            .chunks_exact(4)
            .any(|pixel| pixel == [255, 255, 255, 255]),
        "the active tile must contain visible rate glyphs"
    );
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
            atlas: None,
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
            large_result.stats.draw_calls_issued,
            small_result.stats.draw_calls_issued,
            "[{}] draw-issuing work must not grow with the primitive count",
            mode.name()
        );
        assert_eq!(
            large_result.stats.bind_group_binds,
            small_result.stats.bind_group_binds,
            "[{}] bind-group work must not grow with the primitive count",
            mode.name()
        );
        assert_eq!(
            large_result.stats.slots_visited,
            small_result.stats.slots_visited,
            "[{}] the fixed sequence must be the same length at both counts",
            mode.name()
        );
        if mode == DrawMode::PerSlotIndirect {
            assert_eq!(
                large_result.stats.draw_calls_issued as usize,
                LAYERS * INSTANCED_NON_ATLAS_KINDS,
                "the per-slot path must issue exactly one call per slot of every \
                 kind that has a texture-free pipeline — the form of the claim \
                 that is not satisfiable by collapsing. The sprite passes issue \
                 nothing here because this scene has no atlas at all."
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
        LAYERS * VISITED_KINDS,
        "one entry per (layer, drawn-kind) slot. Phase 4 asserted `LAYERS` here \
         and explained that nothing drew the GlyphRun half because only one \
         instanced pipeline existed; Phase 5.6 built the second and Phase 6.2 \
         the third, so every half of the table is now walked. The gate itself is \
         untouched — the equalities above still say the work does not grow with \
         the primitive count — and what changed is the premise, not the claim"
    );
    assert_eq!(
        large_output.stats.sprite_slots_unavailable as usize,
        LAYERS * SPRITE_KINDS,
        "this scene has no atlas, so both sprite passes' slots are walked and \
         found to have no texture to bind — see \
         `DrawStats::sprite_slots_unavailable`, which is one counter across both \
         passes"
    );
    assert_eq!(large_output.stats.sprite_draws_issued, 0);
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
        atlas: None,
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
        first.stats.slots_skipped
            + first.stats.draw_calls_issued
            + first.stats.sprite_slots_unavailable,
        first.stats.slots_visited,
        "every slot is either drawn, knowingly skipped, or — since Phase 5.6 — \
         found to have no atlas page to bind at all, which is a third outcome \
         and not a fourth name for skipping"
    );
    assert_eq!(
        first.stats.slots_skipped as usize,
        LAYERS * (VISITED_NON_ATLAS_KINDS - 1) + 1,
        "the fallback declines to issue exactly the slots it has read and found \
         empty — the one thing an indirect path cannot do, because it does not \
         know. That is every visited empty kind this scene holds none of (shadow, \
         underline, and path), plus the one quad layer emptied above; backdrop \
         slots are not visited when the scene has no backdrop filter."
    );

    // The staging buffer must be reused across frames, or the fallback
    // allocates GPU memory every frame in the one path that exists because the
    // device is already the weak one.
    let allocations_after_first = renderer.readback_allocations();
    let plan_builds_after_first = renderer.draw_plan_builds();
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
    assert_eq!(
        renderer.draw_plan_builds(),
        plan_builds_after_first,
        "the per-slot bases and their bind group are a function of residency, \
         so a frame loop that changes no layer's reservation must not rebuild \
         them — `QuadDrawPlan`'s own doc says \"per slot-table change rather \
         than per frame\", and this is what makes that checkable"
    );
    assert_eq!(plan_builds_after_first, 1);
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
            atlas: None,
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
    let uncovered = [
        viewport_entry,
        modal(Rect::from_origin_size([400.0, 0.0], [80.0, 320.0])),
    ];
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
        atlas: None,
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
        corner_radii: [0.0; 4],
        border_widths: [0.0; 4],
        material: wgpui_core::patch::primitive::Material::Solid,
    }
}
