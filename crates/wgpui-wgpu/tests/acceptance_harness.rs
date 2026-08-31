//! Cross-system acceptance gates for the retained renderer.
//!
//! This is intentionally one small harness rather than a second renderer. It
//! drives the production patch protocol, scene, compositor, input tree, and
//! WGPU frame renderer together, then compares the observable contracts at the
//! seams. The device-backed tests skip only when no adapter can be opened; the
//! core portions of the acceptance matrix remain deterministic and runnable in
//! that environment.

use std::error::Error;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use wgpui_core::boundary::compositor::{Composite, Compositor};
use wgpui_core::boundary::policy::{BoundaryPolicy, Buffering, Size};
use wgpui_core::geometry::Rect;
use wgpui_core::invalidation::axes::Invalidation;
use wgpui_core::invalidation::request::{FrameSignals, InvalidationRequest};
use wgpui_core::patch::apply::{ScenePatch, apply};
use wgpui_core::patch::primitive::{Material, Primitive, PrimitiveKind, Quad};
use wgpui_core::patch::{PatchOp, RecordKey};
use wgpui_core::scene::{
    BoundaryId, LayerId, LayerKey, Scene, TileCoord, TileGrid, TileResidency, TileSpan,
};
use wgpui_core::window::{
    DispatchTree, EventResult, HitTestIndex, InputEvent, KeyUpEvent, Modifiers,
};

use wgpui_wgpu::debug::DebugTile;
use wgpui_wgpu::render::device::{ComputeContext, context_or_report};
use wgpui_wgpu::render::draw::{DrawMode, DrawStats};
use wgpui_wgpu::render::frame::{Dirty, FrameInput, FrameOutput, FrameRenderer, OffscreenTarget};

const WIDTH: u32 = 96;
const HEIGHT: u32 = 64;
const TILE_EDGE: f32 = 32.0;
const STRESS_ITEMS: usize = 2_048;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DiagnosticsMode {
    Disabled,
    Enabled,
    Capture,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CapturedFrame {
    slots: Vec<(LayerId, PrimitiveKind, u32, u32)>,
    dirty_layers: Vec<LayerId>,
    upload_bytes: u64,
    draw_stats: DrawStats,
}

#[derive(Clone, Copy, Debug)]
struct OverheadBudget {
    max_debug_tiles: usize,
    max_capture_slots: usize,
}

impl DiagnosticsMode {
    fn debug_tiles(self) -> Vec<DebugTile> {
        match self {
            Self::Enabled => vec![DebugTile {
                origin_size: [0.0, 0.0, TILE_EDGE, TILE_EDGE],
                color: [1.0, 0.0, 1.0, 0.35],
                border_width: 2.0,
                _padding: [0.0; 7],
            }],
            Self::Disabled | Self::Capture => Vec::new(),
        }
    }

    fn budget(self) -> OverheadBudget {
        match self {
            Self::Disabled => OverheadBudget {
                max_debug_tiles: 0,
                max_capture_slots: 0,
            },
            Self::Enabled => OverheadBudget {
                max_debug_tiles: 1,
                max_capture_slots: 0,
            },
            Self::Capture => OverheadBudget {
                max_debug_tiles: 0,
                max_capture_slots: 2 * PrimitiveKind::COUNT,
            },
        }
    }
}

fn quad(origin: [f32; 2], size: [f32; 2], color: [f32; 4]) -> Quad {
    Quad {
        origin,
        size,
        background: color,
        border_color: [0.0; 4],
        corner_radii: [0.0; 4],
        border_widths: [0.0; 4],
        material: Material::Solid,
    }
}

fn two_layer_scene() -> Result<(Scene, Vec<LayerId>), Box<dyn Error>> {
    let mut scene = Scene::new();
    let bottom = scene.layer(LayerKey::untiled(BoundaryId::from_raw(1)));
    let top = scene.layer(LayerKey::untiled(BoundaryId::from_raw(2)));
    let mut patch = ScenePatch::new();
    patch.quads.append(
        bottom,
        RecordKey::from_raw(1),
        0,
        quad(
            [0.0, 0.0],
            [WIDTH as f32, HEIGHT as f32],
            [0.1, 0.2, 0.3, 1.0],
        ),
    );
    patch.quads.append(
        top,
        RecordKey::from_raw(2),
        0,
        quad([16.0, 12.0], [48.0, 32.0], [0.8, 0.3, 0.1, 1.0]),
    );
    apply(&mut scene, &patch)?;
    Ok((scene, vec![bottom, top]))
}

fn capture_frame(
    scene: &Scene,
    mode: DiagnosticsMode,
    dirty_layers: &[LayerId],
    upload_bytes: u64,
    output: &FrameOutput,
) -> CapturedFrame {
    let slots = if mode == DiagnosticsMode::Capture {
        scene
            .draw_slots()
            .slots()
            .iter()
            .map(|slot| (slot.layer, slot.kind, slot.base, slot.count))
            .collect()
    } else {
        Vec::new()
    };
    CapturedFrame {
        slots,
        dirty_layers: dirty_layers.to_vec(),
        upload_bytes,
        draw_stats: output.stats,
    }
}

fn assert_budget(mode: DiagnosticsMode, capture: &CapturedFrame, debug_tile_count: usize) {
    let budget = mode.budget();
    assert!(debug_tile_count <= budget.max_debug_tiles);
    assert!(capture.slots.len() <= budget.max_capture_slots);
}

fn render_once(
    context: &ComputeContext,
    scene: &Scene,
    mode: DiagnosticsMode,
    draw_mode: DrawMode,
    dirty_layers: &[LayerId],
    uploads: &[wgpui_core::scene::UploadRange],
) -> Result<(FrameOutput, Vec<u8>, CapturedFrame), Box<dyn Error>> {
    let mut renderer = FrameRenderer::new(&context.device);
    renderer.set_debug_tiles(mode.debug_tiles());
    let target = OffscreenTarget::new(&context.device, WIDTH, HEIGHT);
    let input = FrameInput {
        scene,
        clip: Rect::from_origin_size([0.0, 0.0], [WIDTH as f32, HEIGHT as f32]),
        poison: &[],
        dirty: Dirty::Some(dirty_layers),
        uploads,
        composites: &[],
        registry: None,
        atlas: None,
        viewport: [WIDTH as f32, HEIGHT as f32],
        mode: draw_mode,
    };
    let output = renderer.render(&context.device, &context.queue, &input, &target)?;
    let pixels = target.read_pixels(&context.device, &context.queue)?;
    let capture = capture_frame(
        scene,
        mode,
        dirty_layers,
        output.scene_upload_bytes,
        &output,
    );
    assert_budget(mode, &capture, mode.debug_tiles().len());
    Ok((output, pixels, capture))
}

fn assert_same_frame_work(left: &FrameOutput, right: &FrameOutput) {
    assert_eq!(left.stats, right.stats);
    assert_eq!(left.layers_recomputed, right.layers_recomputed);
    assert_eq!(left.primitives_resident, right.primitives_resident);
    assert_eq!(left.scene_upload_calls, right.scene_upload_calls);
    assert_eq!(left.scene_upload_bytes, right.scene_upload_bytes);
    assert_eq!(left.plan_builds, right.plan_builds);
}

#[test]
fn diagnostics_off_on_and_capture_preserve_frame_work_and_bound_overhead()
-> Result<(), Box<dyn Error>> {
    let Some(context) = context_or_report("acceptance diagnostics differential") else {
        return Ok(());
    };
    let (scene, layers) = two_layer_scene()?;
    let dirty = layers.clone();
    let mut uploads = Vec::new();
    for layer in &dirty {
        let Some(record) = scene.quads.keys(*layer).first().copied() else {
            return Err(std::io::Error::other("scene fixture lost a quad key").into());
        };
        let Some(range) = scene.quads.record_byte_range(*layer, record) else {
            return Err(std::io::Error::other("scene fixture lost a quad slot").into());
        };
        uploads.push(wgpui_core::scene::UploadRange {
            kind: PrimitiveKind::Quad,
            byte_offset: range.start,
            byte_length: range.end - range.start,
        });
    }
    let (disabled, disabled_pixels, disabled_capture) = render_once(
        &context,
        &scene,
        DiagnosticsMode::Disabled,
        DrawMode::best_available(context.indirect),
        &dirty,
        &uploads,
    )?;
    let (enabled, enabled_pixels, enabled_capture) = render_once(
        &context,
        &scene,
        DiagnosticsMode::Enabled,
        DrawMode::best_available(context.indirect),
        &dirty,
        &uploads,
    )?;
    let (capture, capture_pixels, captured) = render_once(
        &context,
        &scene,
        DiagnosticsMode::Capture,
        DrawMode::best_available(context.indirect),
        &dirty,
        &uploads,
    )?;

    assert_same_frame_work(&disabled, &enabled);
    assert_same_frame_work(&disabled, &capture);
    assert_eq!(disabled_capture.dirty_layers, enabled_capture.dirty_layers);
    assert_eq!(disabled_capture.upload_bytes, enabled_capture.upload_bytes);
    assert_eq!(disabled_capture.draw_stats, enabled_capture.draw_stats);
    assert_eq!(disabled_pixels, capture_pixels);
    assert_ne!(
        disabled_pixels, enabled_pixels,
        "enabled diagnostics must produce visible evidence"
    );
    assert_eq!(enabled_capture.slots.len(), 0);
    assert_eq!(captured.slots.len(), 2 * PrimitiveKind::COUNT);
    assert!(
        captured
            .slots
            .windows(2)
            .all(|pair| (pair[0].1, pair[0].0, pair[0].2) <= (pair[1].1, pair[1].0, pair[1].2))
    );
    Ok(())
}

#[test]
fn dirty_uploads_are_delta_only_and_primitive_order_is_stable_across_draw_modes()
-> Result<(), Box<dyn Error>> {
    let Some(context) = context_or_report("acceptance upload and ordering differential") else {
        return Ok(());
    };
    let (mut scene, layers) = two_layer_scene()?;
    let first_slots = scene.draw_slots();
    assert_eq!(first_slots.kind_slots(PrimitiveKind::Quad).len(), 2);
    assert!(
        first_slots
            .slots()
            .windows(2)
            .all(|pair| { (pair[0].kind, pair[0].layer) <= (pair[1].kind, pair[1].layer) })
    );

    let top = *layers.get(1).ok_or("missing top layer")?;
    let top_key = *scene.quads.keys(top).first().ok_or("missing top quad")?;
    let mut update = ScenePatch::new();
    update.quads.update(
        top,
        top_key,
        quad([20.0, 12.0], [44.0, 32.0], [0.8, 0.3, 0.1, 1.0]),
    );
    let plan = apply(&mut scene, &update)?;
    assert_eq!(plan.len(), 1);
    assert_eq!(plan.byte_count(), Quad::SLOT_STRIDE as u64);
    assert_eq!(
        plan.entries().first().map(|entry| entry.kind),
        Some(PrimitiveKind::Quad)
    );

    let dirty = vec![top];
    let mut pixels_by_mode = Vec::new();
    for mode in DrawMode::ALL {
        if !mode.is_available(context.indirect) {
            continue;
        }
        let (output, pixels, _) = render_once(
            &context,
            &scene,
            DiagnosticsMode::Disabled,
            mode,
            &dirty,
            plan.entries(),
        )?;
        assert!(output.scene_upload_bytes >= Quad::SLOT_STRIDE as u64);
        pixels_by_mode.push((mode, pixels));
    }
    let Some((reference_mode, reference_pixels)) = pixels_by_mode.first() else {
        return Err("the device exposed no draw mode".into());
    };
    for (mode, pixels) in pixels_by_mode.iter().skip(1) {
        assert_eq!(
            pixels,
            reference_pixels,
            "{} differs from {}",
            mode.name(),
            reference_mode.name()
        );
    }

    Ok(())
}

#[test]
fn tiled_scroll_damage_and_unsupported_configuration_have_explicit_results() {
    let boundary = BoundaryId::from_raw(77);
    let sibling = BoundaryId::from_raw(78);
    let policy = BoundaryPolicy {
        buffering: Buffering::Tiled {
            tile_size: Size::pixels(TILE_EDGE, TILE_EDGE),
            retain_radius: 1,
        },
        resident_tile_budget: 128,
        ..BoundaryPolicy::default()
    };
    let viewport = Rect::from_origin_size([0.0, 0.0], [64.0, 64.0]);
    let mut compositor = Compositor::new();
    let first = compositor.visit_tiled(boundary, policy, 1, viewport);
    assert!(first.is_some());
    let Some(first) = first else { return };
    assert_eq!(first.revealed.len(), first.visible.len());
    assert!(first.visible.contains(&TileCoord::new(-1, -1)));

    assert!(compositor.set_transform(
        boundary,
        wgpui_core::scene::layer::LayerTransform::translated(-TILE_EDGE, 0.0)
    ));
    let second = compositor.visit_tiled(boundary, policy, 2, viewport);
    let Some(second) = second else { return };
    assert_eq!(second.revealed.len(), 4);
    assert!(second.revealed.iter().all(|tile| tile.x == 3));
    let resolved = compositor.resolve(
        boundary,
        wgpui_core::invalidation::reason::Reason::Scroll,
        false,
        4,
        true,
    );
    assert_eq!(
        resolved.map(|value| value.composite),
        Some(Composite::TransformOnly)
    );

    let sibling_visit = compositor.visit_tiled(sibling, policy, 2, viewport);
    let Some(sibling_visit) = sibling_visit else {
        return;
    };
    assert_ne!(
        first.tile_layer(TileCoord::ORIGIN),
        sibling_visit.tile_layer(TileCoord::ORIGIN)
    );

    let invalid = Buffering::Tiled {
        tile_size: Size::pixels(0.0, TILE_EDGE),
        retain_radius: 1,
    };
    assert!(invalid.tile_grid().is_none());
    assert!(
        Compositor::new()
            .visit_tiled(
                boundary,
                BoundaryPolicy {
                    buffering: invalid,
                    ..policy
                },
                1,
                viewport
            )
            .is_none()
    );

    let huge = TileSpan {
        min: TileCoord::new(i32::MIN, i32::MIN),
        max: TileCoord::new(i32::MAX, i32::MAX),
    };
    assert_eq!(huge.tiles().len(), TileSpan::MAX_TILES as usize);
}

#[test]
fn capability_fallbacks_and_thousand_item_stress_remain_bounded() {
    assert_eq!(
        DrawMode::best_available(wgpui_wgpu::render::device::IndirectSupport::NONE),
        DrawMode::PerSlotIndirect
    );
    assert!(
        !DrawMode::MultiDrawIndirect
            .is_available(wgpui_wgpu::render::device::IndirectSupport::NONE)
    );
    assert!(DrawMode::CpuReadback.is_available(wgpui_wgpu::render::device::IndirectSupport::NONE));
    assert!(
        !wgpui_wgpu::render::device::IndirectSupport {
            first_instance: false,
            multi_draw_count: true
        }
        .supports_native_multi_draw()
    );

    let mut residency = TileResidency::new(STRESS_ITEMS);
    let grid = TileGrid::square(TILE_EDGE);
    assert!(grid.is_some());
    let Some(grid) = grid else { return };
    let span = grid.visible_span(
        Rect::from_origin_size([0.0, 0.0], [TILE_EDGE * 32.0, TILE_EDGE * 64.0]),
        0,
    );
    assert!(span.is_some());
    let Some(span) = span else { return };
    let revealed = residency.mark(span, 1);
    assert_eq!(revealed.len(), STRESS_ITEMS);
    assert_eq!(residency.sweep(1, 0).len(), 0);
    assert_eq!(residency.len(), STRESS_ITEMS);
    assert_eq!(residency.over_budget(), 0);

    let mut hit_test = HitTestIndex::default();
    let mut dispatch = DispatchTree::new();
    let root = dispatch.root();
    let calls = Arc::new(AtomicUsize::new(0));
    let mut hitboxes = Vec::with_capacity(STRESS_ITEMS);
    for index in 0..STRESS_ITEMS {
        let x = (index % 64) as f32;
        let y = (index / 64) as f32;
        let hitbox = hit_test.insert(Rect::from_origin_size([x, y], [1.0, 1.0]), 0);
        let node = dispatch.new_node(Some(root));
        let count = Arc::clone(&calls);
        assert!(dispatch.on_input(node, move |_event| {
            count.fetch_add(1, Ordering::Relaxed);
            EventResult::HANDLED
        }));
        assert!(dispatch.bind_hitbox(hitbox, node));
        hitboxes.push(hitbox);
    }
    for hitbox in hitboxes.iter().copied() {
        let Some(entry) = hit_test.get(hitbox) else {
            continue;
        };
        assert!(dispatch.dispatch_input(
            hitbox,
            &InputEvent::KeyUp(KeyUpEvent {
                key: "stress".into(),
                modifiers: Modifiers::none(),
            }),
        ));
        assert!(entry.contains([entry.bounds.min_x, entry.bounds.min_y]));
    }
    assert_eq!(calls.load(Ordering::Relaxed), STRESS_ITEMS);
}

#[test]
fn invalidation_damage_does_not_confuse_input_with_content() {
    let layer = LayerId::from_raw(301);
    let mut signals = FrameSignals::new();
    signals.scrolled(layer);
    signals.raise(InvalidationRequest::data_changed(
        wgpui_core::invalidation::request::InvalidationScope::Instance(
            wgpui_core::reconcile::instance::InstanceKey::from_raw(9),
        ),
    ));
    assert_eq!(
        signals.reason_for_layer(layer),
        wgpui_core::invalidation::reason::Reason::Scroll
    );

    let mut scene = Scene::new();
    let declared = scene.layer(LayerKey::untiled(BoundaryId::from_raw(302)));
    assert!(scene.layers.mark_clean(declared));
    let mut patch = ScenePatch::new();
    patch.hitboxes.insert(
        declared,
        RecordKey::from_raw(303),
        0,
        wgpui_core::scene::record::Hitbox {
            instance: wgpui_core::reconcile::instance::InstanceKey::from_raw(303),
            bounds: [0.0, 0.0, 10.0, 10.0],
            opaque: true,
        },
    );
    let plan = apply(&mut scene, &patch);
    assert!(plan.is_ok());
    assert!(plan.as_ref().is_ok_and(|value| value.is_empty()));
    assert_eq!(patch.content_layers(), Vec::<LayerId>::new());
    assert_eq!(
        scene.layers.get(declared).map(|value| value.invalidation()),
        Some(Invalidation::HIT)
    );

    let kinds: Vec<_> = patch
        .hitboxes
        .patches()
        .iter()
        .map(|entry| match &entry.op {
            PatchOp::Insert { .. } => "insert",
            PatchOp::Update { .. } => "update",
            PatchOp::Remove { .. } => "remove",
        })
        .collect();
    assert_eq!(kinds, vec!["insert"]);
}
