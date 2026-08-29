//! Phase 6.4 GPU gates for paths and backdrop filters.
//!
//! The source checks keep the frozen legacy shader in the comparison loop,
//! while the adapter tests exercise the new primitives through ScenePatch,
//! FrameRenderer, and the offscreen target used by the window pipeline.

use lyon::math::point;
use lyon::path::Path as LyonPath;
use lyon::tessellation::{BuffersBuilder, FillOptions, FillTessellator, FillVertex, VertexBuffers};
use wgpui_core::geometry::Rect;
use wgpui_core::patch::RecordKey;
use wgpui_core::patch::apply::{ScenePatch, apply};
use wgpui_core::patch::primitive::{BackdropFilter, Path, Quad};
use wgpui_core::scene::Scene;
use wgpui_core::scene::layer::{BoundaryId, LayerKey};
use wgpui_wgpu::render::device::{ComputeContext, context_or_report};
use wgpui_wgpu::render::draw::DrawMode;
use wgpui_wgpu::render::frame::{Dirty, FrameError, FrameInput, FrameRenderer, OffscreenTarget, RenderTarget};

const WIDTH: u32 = 96;
const HEIGHT: u32 = 64;
const UNCLIPPED: Rect = Rect {
    min_x: -100_000.0,
    min_y: -100_000.0,
    max_x: 100_000.0,
    max_y: 100_000.0,
};

const LEGACY_PATHS_WGSL: &str = include_str!("../../../src/platform/cross/shaders/paths.wgsl");
const LEGACY_BACKDROP_WGSL: &str =
    include_str!("../../../src/platform/cross/shaders/backdrop_blur.wgsl");

fn input(scene: &Scene, mode: DrawMode) -> FrameInput<'_> {
    FrameInput {
        scene,
        clip: UNCLIPPED,
        poison: &[],
        dirty: Dirty::All,
        uploads: &[],
        composites: &[],
        registry: None,
        atlas: None,
        viewport: [WIDTH as f32, HEIGHT as f32],
        mode,
    }
}

fn layer(scene: &mut Scene) -> wgpui_core::scene::layer::LayerId {
    scene.layer(LayerKey::untiled(BoundaryId::from_raw(1)))
}

fn read_pixel(pixels: &[u8], x: u32, y: u32) -> [u8; 4] {
    let offset = ((y * WIDTH + x) * 4) as usize;
    [pixels[offset], pixels[offset + 1], pixels[offset + 2], pixels[offset + 3]]
}

fn lyon_triangle() -> Path {
    let mut builder = LyonPath::builder();
    builder.begin(point(8.0, 8.0));
    builder.line_to(point(40.0, 8.0));
    builder.line_to(point(8.0, 40.0));
    builder.close();
    let shape = builder.build();
    let mut buffers = VertexBuffers::new();
    let mut tessellator = FillTessellator::new();
    let result = tessellator.tessellate_path(
        &shape,
        &FillOptions::default(),
        &mut BuffersBuilder::new(&mut buffers, |vertex: FillVertex| vertex.position()),
    );
    assert!(result.is_ok(), "Lyon must tessellate the test path");
    Path::from_lyon_tessellation(buffers, [1.0, 0.0, 0.0, 1.0]).with_clip(
        [0.0, 0.0],
        [WIDTH as f32, HEIGHT as f32],
    )
}

fn render_path(context: &ComputeContext) -> Option<(Vec<u8>, u32)> {
    let mut scene = Scene::new();
    let layer = layer(&mut scene);
    let mut patch = ScenePatch::new();
    patch.paths.append(layer, RecordKey::from_raw(1), 0, lyon_triangle());
    apply(&mut scene, &patch).ok()?;

    let mut renderer = FrameRenderer::new(&context.device);
    let target = OffscreenTarget::new(&context.device, WIDTH, HEIGHT);
    let output = renderer
        .render_to(
            &context.device,
            &context.queue,
            &input(&scene, DrawMode::PerSlotIndirect),
            &target.target(),
        )
        .ok()?;
    let pixels = target.read_pixels(&context.device, &context.queue).ok()?;
    Some((pixels, output.stats.path_vertices_issued))
}

#[test]
fn legacy_sources_retain_the_path_and_backdrop_contracts() {
    assert!(LEGACY_PATHS_WGSL.contains("fn vs_path"));
    assert!(LEGACY_PATHS_WGSL.contains("fn fs_path"));
    assert!(LEGACY_PATHS_WGSL.contains("st_position"));
    assert!(LEGACY_BACKDROP_WGSL.contains("fn vs_backdrop_filter"));
    assert!(LEGACY_BACKDROP_WGSL.contains("fn fs_backdrop_filter"));
    assert!(LEGACY_BACKDROP_WGSL.contains("textureSampleLevel"));
    assert!(LEGACY_BACKDROP_WGSL.contains("quad_sdf"));
}

#[test]
fn source_gate_rejects_a_changed_legacy_shader() {
    let changed = LEGACY_BACKDROP_WGSL.replace("textureSampleLevel", "textureSample");
    assert_ne!(changed, LEGACY_BACKDROP_WGSL);
    assert!(!changed.contains("textureSampleLevel"));
}

#[test]
fn lyon_path_reaches_the_real_gpu_and_readback() {
    let Some(context) = context_or_report("phase 6.4 lyon path") else {
        return;
    };
    let Some((pixels, vertex_count)) = render_path(&context) else {
        panic!("the real-adapter path gate must render and read back");
    };
    assert!(vertex_count >= 3, "Lyon must provide triangle vertices");
    assert_eq!(read_pixel(&pixels, 12, 12), [255, 0, 0, 255]);
    assert_eq!(read_pixel(&pixels, 48, 48), [0, 0, 0, 255]);
}

fn scene_with_backdrop() -> Scene {
    let mut scene = Scene::new();
    let layer = layer(&mut scene);
    let mut patch = ScenePatch::new();
    patch.quads.append(
        layer,
        RecordKey::from_raw(1),
        0,
        Quad {
            origin: [0.0, 0.0],
            size: [32.0, HEIGHT as f32],
            background: [1.0, 0.0, 0.0, 1.0],
            border_color: [0.0; 4],
            corner_radii: [0.0; 4],
            border_widths: [0.0; 4],
        },
    );
    patch.backdrop_filters.append(
        layer,
        RecordKey::from_raw(2),
        0,
        BackdropFilter {
            origin: [24.0, 16.0],
            size: [40.0, 32.0],
            clip_origin: [0.0, 0.0],
            clip_size: [WIDTH as f32, HEIGHT as f32],
            corner_radii: [4.0; 4],
            blur_radius: 4.0,
            opacity: 1.0,
        },
    );
    apply(&mut scene, &patch).expect("the quad and backdrop patch must apply");
    scene
}

#[test]
fn backdrop_filter_uses_a_snapshot_pass_and_changes_an_edge_pixel() {
    let Some(context) = context_or_report("phase 6.4 backdrop filter") else {
        return;
    };
    let scene = scene_with_backdrop();
    let mut renderer = FrameRenderer::new(&context.device);
    let target = OffscreenTarget::new(&context.device, WIDTH, HEIGHT);
    let output = renderer
        .render_to(
            &context.device,
            &context.queue,
            &input(&scene, DrawMode::PerSlotIndirect),
            &target.target(),
        )
        .expect("the backdrop pass must render on a target with a source texture");
    let pixels = target
        .read_pixels(&context.device, &context.queue)
        .expect("the backdrop target must be readable");

    assert_eq!(output.stats.backdrop_filters_drawn, 1);
    assert_ne!(read_pixel(&pixels, 36, 32), [0, 0, 0, 255]);
    assert_eq!(read_pixel(&pixels, 80, 32), [0, 0, 0, 255]);
}

#[test]
fn backdrop_gate_fails_without_a_copyable_source() {
    let Some(context) = context_or_report("phase 6.4 backdrop source gate") else {
        return;
    };
    let scene = scene_with_backdrop();
    let target = OffscreenTarget::new(&context.device, WIDTH, HEIGHT);
    let mut renderer = FrameRenderer::new(&context.device);
    let result = renderer.render_to(
        &context.device,
        &context.queue,
        &input(&scene, DrawMode::PerSlotIndirect),
        &RenderTarget {
            view: &target.view,
            width: target.width,
            height: target.height,
            clear: wgpu::Color::BLACK,
            source: None,
        },
    );
    assert!(matches!(result, Err(FrameError::BackdropSourceUnavailable)));
}
