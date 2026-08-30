//! Phase 6.3's gate for shadows: **byte-exact against legacy output**.
//!
//! > Both pipelines byte-exact against legacy output, same discipline, reusing
//! > the now-three-times-proven pattern rather than re-deriving it.
//!
//! # What "against legacy" means here, precisely
//!
//! Not a transcription of the legacy renderer, and not a CPU model of it. This
//! test compiles **the legacy shader file itself** —
//! `src/platform/cross/shaders/shadows.wgsl`, `include_str!`-ed so that moving
//! or deleting it fails the build — feeds it a buffer in the legacy `Shadow`
//! struct's own 72-byte layout, and renders it into the same `Rgba8Unorm`
//! target through the same blend state the legacy renderer selects. The other
//! arm is 2.0's real `FrameRenderer`: patch, apply, upload, ordering, occlusion,
//! indirect-argument generation, indirect draw. Then every pixel of the two
//! framebuffers is compared for equality.
//!
//! That is a stronger oracle than Phase 5.5's rasteriser differential, which had
//! to transcribe because `TextSystem::rasterize_glyph` is `pub(crate)`. Nothing
//! is transcribed here except the *inputs*, and each transcription is itself
//! asserted (§ "the colour, and why it is not simply passed through").
//!
//! # The three ways the legacy arm is not literally the legacy renderer
//!
//! Stated up front rather than found later. All three are argued to be no-ops
//! for this comparison, and two of them are mechanically checked:
//!
//! 1. **No `slab_shader_source` wrapper.** The legacy renderer does not hand
//!    this file to `create_shader_module` directly: `renderer.rs:99` prepends
//!    `slab_transform.wgsl` and rewrites two expressions to thread a per-layer
//!    translate through. Both rewrites are the identity at a zero translate
//!    (`position + vec2(0.0)`, `layer_world_position(p)`), which is the only
//!    transform this test uses. The file is *designed* to be rendered
//!    unwrapped — `slab_shader_source`'s own doc says "the `.wgsl` files
//!    themselves stay byte-pristine: `flamegraph_replay` renders them against
//!    its own bind-group layouts" — so this arm is a second such consumer, not
//!    a novel use. [`the_legacy_source_still_has_the_shape_this_test_relies_on`]
//!    asserts both rewrite patterns are still present, which is the same
//!    assertion `slab_shader_source` makes, so a drift in the legacy file breaks
//!    this test loudly instead of weakening it silently.
//! 2. **`premultiplied_alpha` is 0.** The legacy globals carry a flag 2.0 has no
//!    equivalent of; it is set from the *surface's* composite alpha mode, and
//!    `flamegraph_replay.rs:596` sets it to 0 for the same offscreen situation
//!    this test is in. The blend state matches: `wgpu::BlendState::ALPHA_BLENDING`
//!    is what `renderer.rs:890` selects for a non-premultiplied surface, and it
//!    is field-for-field `render/pipelines.rs`'s `ALPHA_OVER`.
//! 3. **`content_mask` covers everything.** 2.0 has no per-primitive clip: the
//!    frame's clip reaches the occlusion pass instead (§5.2). So the legacy arm
//!    is given a mask far larger than the viewport and its clip-distance path
//!    never rejects a fragment, which is what makes the two arms comparable at
//!    all. **Per-fragment clipping is therefore outside this proof.**
//!
//! # The colour, and why it is not simply passed through
//!
//! The legacy `Shadow` carries HSLA and converts in its vertex shader; 2.0's
//! carries straight RGBA. So the two arms are only comparable if the conversion
//! is exact for the colours under test. Every case here uses a colour whose
//! conversion involves nothing but 0, 0.5, 1, 2, 3 and 6 — values every IEEE-754
//! implementation represents and combines exactly — and
//! [`the_colour_transcription_is_checked_rather_than_assumed`] runs the legacy
//! `hsla_to_rgba` transcribed into Rust and asserts it produces exactly the
//! bytes handed to 2.0. **Colour-space conversion is not what this gate proves;
//! it is what the gate holds fixed so it can prove something else.**

use wgpui_core::geometry::Rect;
use wgpui_core::patch::RecordKey;
use wgpui_core::patch::apply::{ScenePatch, apply};
use wgpui_core::patch::primitive::{Quad, Shadow};
use wgpui_core::scene::Scene;
use wgpui_core::scene::layer::{BoundaryId, LayerKey};
use wgpui_wgpu::render::device::{ComputeContext, context_or_report};
use wgpui_wgpu::render::draw::DrawMode;
use wgpui_wgpu::render::frame::{Dirty, FrameInput, FrameRenderer, OffscreenTarget, RenderTarget};
use wgpui_wgpu::render::pipelines::TARGET_FORMAT;

/// The legacy shader, byte for byte off disk. A build error here means the
/// frozen tree moved and this differential no longer has a subject.
const LEGACY_SHADOWS_WGSL: &str =
    include_str!("../../../old/src/platform/cross/shaders/shadows.wgsl");

const WIDTH: u32 = 224;
const HEIGHT: u32 = 176;

/// Bytes one legacy `Shadow` occupies. `src/scene.rs:1488` asserts the same
/// number on the Rust struct; restated here because this file builds those
/// bytes by hand and a silent disagreement would misalign every instance past
/// the first.
const LEGACY_SHADOW_STRIDE: usize = 72;

/// `wgpu::BlendState::ALPHA_BLENDING`, which `renderer.rs:890` selects for a
/// non-premultiplied surface and which `render/pipelines.rs`'s `ALPHA_OVER` is
/// field-for-field. Spelled out rather than referenced so this arm's blend is
/// visibly the legacy one rather than visibly 2.0's.
const LEGACY_BLEND: wgpu::BlendState = wgpu::BlendState::ALPHA_BLENDING;

/// What both arms clear to, and deliberately **not** black.
///
/// The first version of this file used `OffscreenTarget::target()`'s black —
/// the clear every test before this one wants, because Phase 5.6's white-text
/// proof depends on it. It made the very first case agree on 39,424 of 39,424
/// pixels while painting *nothing*: a 50%-alpha black shadow composited over
/// opaque black is `rgb = 0`, `alpha = 1`, which is the clear colour again. The
/// gate's own vacuity guard caught it, which is the argument for having one.
///
/// So both arms clear to an asymmetric mid colour instead. Asymmetric on
/// purpose: three equal channels would let a red/blue swap pass unnoticed on
/// any grey-scaled shadow, and every dark *and* light shadow is now visible
/// against it.
const CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: 0.25,
    g: 0.5,
    b: 0.75,
    a: 1.0,
};

/// One comparison: the same shadow expressed both ways.
struct Case {
    name: &'static str,
    shadow: Shadow,
    /// The legacy struct's HSLA for `shadow.color`. See this module's doc.
    hsla: [f32; 4],
}

/// The legacy `hsla_to_rgba`, transcribed expression for expression from
/// `src/platform/cross/shaders/shadows.wgsl:57`.
///
/// Exists only so the test can assert its own colour inputs agree; it is never
/// used to produce a comparison value.
fn hsla_to_rgba(hsla: [f32; 4]) -> [f32; 4] {
    let h = hsla[0] * 6.0;
    let s = hsla[1];
    let l = hsla[2];
    let a = hsla[3];

    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h % 2.0) - 1.0).abs());
    let m = l - c / 2.0;
    let mut color = [m, m, m];

    if (0.0..1.0).contains(&h) {
        color[0] += c;
        color[1] += x;
    } else if (1.0..2.0).contains(&h) {
        color[0] += x;
        color[1] += c;
    } else if (2.0..3.0).contains(&h) {
        color[1] += c;
        color[2] += x;
    } else if (3.0..4.0).contains(&h) {
        color[1] += x;
        color[2] += c;
    } else if (4.0..5.0).contains(&h) {
        color[0] += x;
        color[2] += c;
    } else {
        color[0] += c;
        color[2] += x;
    }

    [color[0], color[1], color[2], a]
}

/// Every case, chosen to move each field of the shadow independently.
///
/// Colours are restricted to the exactly-representable set this module's doc
/// explains. Geometry is not restricted: positions, sizes, radii and blur radii
/// are deliberately fractional, because a shadow's falloff is a smooth function
/// and a comparison that only ever landed on integers would be testing far less
/// than it looks like it is.
fn cases() -> Vec<Case> {
    let black_half = ([0.0, 0.0, 0.0, 0.5], [0.0, 0.0, 0.0, 0.5]);
    let grey = ([0.0, 0.0, 0.5, 1.0], [0.5, 0.5, 0.5, 1.0]);
    let red = ([0.0, 1.0, 0.5, 1.0], [1.0, 0.0, 0.0, 1.0]);
    let cyan = ([0.5, 1.0, 0.5, 0.75], [0.0, 1.0, 1.0, 0.75]);
    let white = ([0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]);

    vec![
        // Phase 6.6's addition: four different radii, so `pick_corner_radius`
        // has to select a different one in each quadrant. This case is the whole
        // reason `Shadow` grew from one radius to four, and it would have been
        // unrepresentable before that.
        Case {
            name: "four different corner radii, one per quadrant",
            shadow: Shadow {
                origin: [48.0, 40.0],
                size: [128.0, 96.0],
                color: grey.1,
                corner_radii: [24.0, 4.0, 16.0, 0.0],
                blur_radius: 9.0,
            },
            hsla: grey.0,
        },
        Case {
            name: "the ordinary card shadow",
            shadow: Shadow {
                origin: [48.0, 40.0],
                size: [128.0, 96.0],
                color: black_half.1,
                corner_radii: [8.0; 4],
                blur_radius: 12.0,
            },
            hsla: black_half.0,
        },
        Case {
            name: "a heavy blur, wider than the rectangle it comes from",
            shadow: Shadow {
                origin: [80.0, 64.0],
                size: [24.0, 20.0],
                color: grey.1,
                corner_radii: [4.0; 4],
                blur_radius: 28.0,
            },
            hsla: grey.0,
        },
        Case {
            name: "square corners",
            shadow: Shadow {
                origin: [40.5, 32.25],
                size: [100.75, 80.5],
                color: red.1,
                corner_radii: [0.0; 4],
                blur_radius: 6.5,
            },
            hsla: red.0,
        },
        Case {
            name: "a fully rounded pill",
            shadow: Shadow {
                origin: [32.0, 60.0],
                size: [160.0, 48.0],
                color: cyan.1,
                corner_radii: [24.0; 4],
                blur_radius: 9.25,
            },
            hsla: cyan.0,
        },
        Case {
            name: "a radius larger than half the rectangle, which neither side clamps",
            shadow: Shadow {
                origin: [72.0, 56.0],
                size: [60.0, 40.0],
                color: white.1,
                corner_radii: [45.0; 4],
                blur_radius: 7.0,
            },
            hsla: white.0,
        },
        Case {
            name: "a sub-pixel blur radius",
            shadow: Shadow {
                origin: [56.25, 48.75],
                size: [110.5, 70.0],
                color: grey.1,
                corner_radii: [3.5; 4],
                blur_radius: 0.75,
            },
            hsla: grey.0,
        },
        Case {
            name: "reaching past the top-left corner of the viewport",
            shadow: Shadow {
                origin: [-30.0, -24.0],
                size: [90.0, 70.0],
                color: red.1,
                corner_radii: [10.0; 4],
                blur_radius: 14.0,
            },
            hsla: red.0,
        },
        Case {
            name: "a zero blur radius, which both sides make NaN",
            shadow: Shadow {
                origin: [64.0, 52.0],
                size: [96.0, 72.0],
                color: white.1,
                corner_radii: [6.0; 4],
                blur_radius: 0.0,
            },
            hsla: white.0,
        },
    ]
}

/// The 72 bytes the legacy `Shadow` occupies, in its own field order.
///
/// Built by hand rather than by `bytemuck`-casting the legacy struct, because
/// that struct is `pub(crate)` in a frozen crate this test must not edit. The
/// offsets are asserted against WGSL's own layout rules in
/// [`the_legacy_struct_layout_is_the_one_wgsl_derives`].
fn encode_legacy_shadow(shadow: &Shadow, hsla: [f32; 4]) -> [u8; LEGACY_SHADOW_STRIDE] {
    let mut bytes = [0u8; LEGACY_SHADOW_STRIDE];
    let mut put = |offset: usize, value: f32| {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    };
    // `order: u32` at 0 stays zero: the legacy shader reads the field's bytes
    // for nothing, and 2.0 computes draw order on the GPU (§5.1).
    put(4, shadow.blur_radius);
    put(8, shadow.origin[0]);
    put(12, shadow.origin[1]);
    put(16, shadow.size[0]);
    put(20, shadow.size[1]);
    // Four per-corner radii, in the legacy `Corners` field order.
    //
    // Phase 6.3 wrote one value four times here and disclosed "per-corner radii
    // are outside this proof," because 2.0's `Shadow` carried one uniform
    // radius. Phase 6.6 widened it (a `rounded_t_md()` box's shadow has two
    // round corners and two square ones), so this passes the four through and
    // [`a_per_corner_shadow_agrees_with_the_legacy_pick_corner_radius`] moves
    // all four independently. **That limitation is closed, not restated.**
    for (corner, radius) in shadow.corner_radii.iter().enumerate() {
        put(24 + corner * 4, *radius);
    }
    // A content mask far larger than the viewport, per this module's doc.
    put(40, -100_000.0);
    put(44, -100_000.0);
    put(48, 200_000.0);
    put(52, 200_000.0);
    for (index, channel) in hsla.iter().enumerate() {
        put(56 + index * 4, *channel);
    }
    bytes
}

/// Render one shadow through the legacy shader, unwrapped, and read it back.
fn render_legacy(context: &ComputeContext, case: &Case) -> Vec<u8> {
    let device = &context.device;
    let queue = &context.queue;

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("legacy shadows"),
        source: wgpu::ShaderSource::Wgsl(LEGACY_SHADOWS_WGSL.into()),
    });

    // `Globals { viewport_size: vec2<f32>, premultiplied_alpha: u32, pad: u32 }`
    let mut globals = [0u8; 16];
    globals[0..4].copy_from_slice(&(WIDTH as f32).to_le_bytes());
    globals[4..8].copy_from_slice(&(HEIGHT as f32).to_le_bytes());
    // premultiplied_alpha = 0 and pad = 0 — see this module's doc.
    let globals_buffer = buffer_with(
        device,
        queue,
        "legacy globals",
        wgpu::BufferUsages::UNIFORM,
        &globals,
    );

    let shadow_bytes = encode_legacy_shadow(&case.shadow, case.hsla);
    let shadow_buffer = buffer_with(
        device,
        queue,
        "legacy shadows",
        wgpu::BufferUsages::STORAGE,
        &shadow_bytes,
    );

    let globals_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("legacy globals"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: std::num::NonZeroU64::new(16),
            },
            count: None,
        }],
    });
    let shadow_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("legacy shadow storage"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("legacy shadows"),
        bind_group_layouts: &[Some(&globals_layout), Some(&shadow_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("legacy shadows"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_shadow"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs_shadow"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: TARGET_FORMAT,
                blend: Some(LEGACY_BLEND),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: Default::default(),
        cache: None,
    });

    let globals_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("legacy globals"),
        layout: &globals_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: globals_buffer.as_entire_binding(),
        }],
    });
    let shadow_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("legacy shadow storage"),
        layout: &shadow_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: shadow_buffer.as_entire_binding(),
        }],
    });

    let target = OffscreenTarget::new(device, WIDTH, HEIGHT);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("legacy shadow frame"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("legacy shadow frame"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target.view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(CLEAR_COLOR),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: Default::default(),
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &globals_group, &[]);
        pass.set_bind_group(1, &shadow_group, &[]);
        pass.draw(0..4, 0..1);
    }
    queue.submit(Some(encoder.finish()));

    target
        .read_pixels(device, queue)
        .expect("reading the legacy target back must succeed")
}

fn buffer_with(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    usage: wgpu::BufferUsages,
    bytes: &[u8],
) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len() as u64,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytes);
    buffer
}

/// Render the same shadow through 2.0's whole frame path and read it back.
///
/// `shadow` is `None` for the empty-scene render the clear colour is measured
/// from — see [`measured_clear_pixel`].
fn render_2_0(context: &ComputeContext, shadow: Option<Shadow>, mode: DrawMode) -> Vec<u8> {
    let mut scene = Scene::new();
    let layer = scene.layer(LayerKey::untiled(BoundaryId::from_raw(1)));
    let mut patch = ScenePatch::new();
    if let Some(shadow) = shadow {
        patch
            .shadows
            .append(layer, RecordKey::from_raw(1), 0, shadow);
    }
    apply(&mut scene, &patch).expect("seeding one layer with a shadow must apply");

    let mut renderer = FrameRenderer::new(&context.device);
    let target = OffscreenTarget::new(&context.device, WIDTH, HEIGHT);
    let input = FrameInput {
        scene: &scene,
        // Deliberately larger than the viewport: the legacy arm's content mask
        // is effectively infinite, so 2.0's clip has to be too or the two arms
        // are answering different questions at the frame edges. The case that
        // reaches past the top-left corner is what makes this matter.
        clip: Rect::from_origin_size([-100_000.0, -100_000.0], [200_000.0, 200_000.0]),
        poison: &[],
        dirty: Dirty::All,
        uploads: &[],
        composites: &[],
        registry: None,
        atlas: None,
        viewport: [WIDTH as f32, HEIGHT as f32],
        mode,
    };
    // `render_to` rather than `render`, only so the clear colour can be chosen:
    // `OffscreenTarget::target()` hard-codes black, and see [`CLEAR_COLOR`] for
    // why black is the one colour this comparison cannot use. Everything else
    // about the path is identical — `render` is `render_to` over
    // `target.target()`.
    renderer
        .render_to(
            &context.device,
            &context.queue,
            &input,
            &RenderTarget {
                view: &target.view,
                width: target.width,
                height: target.height,
                clear: CLEAR_COLOR,
                source: None,
            },
        )
        .expect("a frame must render");
    target
        .read_pixels(&context.device, &context.queue)
        .expect("reading the 2.0 target back must succeed")
}

/// The clear colour as the device actually wrote it, measured rather than
/// derived.
///
/// `CLEAR_COLOR`'s components are not exact multiples of `1/255`, so predicting
/// the byte would mean predicting the driver's rounding. Rendering an empty
/// scene and reading pixel zero is exact and needs no such prediction — and it
/// doubles as a check that both arms clear identically, which is a precondition
/// of the whole comparison and would otherwise be assumed.
fn measured_clear_pixel(context: &ComputeContext) -> [u8; 4] {
    let ours = render_2_0(context, None, DrawMode::best_available(context.indirect));
    let legacy = render_legacy(
        context,
        &Case {
            name: "empty",
            // Fully transparent and zero-sized: the legacy arm still issues its
            // draw, and must still leave the clear untouched.
            shadow: Shadow::ZERO,
            hsla: [0.0, 0.0, 0.0, 0.0],
        },
    );
    let ours: [u8; 4] = [ours[0], ours[1], ours[2], ours[3]];
    let legacy: [u8; 4] = [legacy[0], legacy[1], legacy[2], legacy[3]];
    assert_eq!(
        ours, legacy,
        "the two arms must clear to the same bytes, or every later comparison \
         is between two different backgrounds"
    );
    ours
}

/// What comparing two framebuffers found.
#[derive(Default, Debug)]
struct Comparison {
    total: usize,
    exact: usize,
    /// Pixels where at least one arm painted something over the clear colour.
    /// Without this the "byte-exact" count is dominated by agreeing background.
    painted: usize,
    /// Painted pixels that agree.
    painted_exact: usize,
    /// The first disagreement, as (x, y, legacy, ours).
    first_difference: Option<(usize, usize, [u8; 4], [u8; 4])>,
}

fn compare(legacy: &[u8], ours: &[u8], clear: [u8; 4]) -> Comparison {
    let mut result = Comparison::default();
    assert_eq!(
        legacy.len(),
        ours.len(),
        "both arms read back the same extent"
    );
    for (index, (left, right)) in legacy.chunks_exact(4).zip(ours.chunks_exact(4)).enumerate() {
        let left: [u8; 4] = [left[0], left[1], left[2], left[3]];
        let right: [u8; 4] = [right[0], right[1], right[2], right[3]];
        result.total += 1;
        let painted = left != clear || right != clear;
        if painted {
            result.painted += 1;
        }
        if left == right {
            result.exact += 1;
            if painted {
                result.painted_exact += 1;
            }
        } else if result.first_difference.is_none() {
            result.first_difference =
                Some((index % WIDTH as usize, index / WIDTH as usize, left, right));
        }
    }
    result
}

/// **The gate.** Every case, byte-exact.
#[test]
fn phase_6_3_shadow_gate() {
    let Some(context) = context_or_report("phase_6_3_shadow_gate") else {
        return;
    };
    println!(
        "adapter: {:?} {:?} driver {:?}",
        context.adapter_info.name, context.adapter_info.backend, context.adapter_info.driver_info
    );

    let clear = measured_clear_pixel(&context);
    println!("both arms clear to {clear:?}");

    let mut total = 0usize;
    let mut exact = 0usize;
    let mut painted = 0usize;
    for case in cases() {
        assert_eq!(
            hsla_to_rgba(case.hsla),
            case.shadow.color,
            "[{}] the two arms must be given the same colour",
            case.name
        );
        let legacy = render_legacy(&context, &case);
        let ours = render_2_0(
            &context,
            Some(case.shadow),
            DrawMode::best_available(context.indirect),
        );
        let result = compare(&legacy, &ours, clear);
        println!(
            "  {}: {} of {} pixels byte-exact ({} of {} painted)",
            case.name, result.exact, result.total, result.painted_exact, result.painted
        );
        assert_eq!(
            result.exact, result.total,
            "[{}] byte-exact means byte-exact; first difference at {:?}",
            case.name, result.first_difference
        );
        // A case that painted nothing would agree trivially. The zero-blur case
        // is the one where that is *expected* — both sides make NaN and draw
        // nothing — so it is excluded by name rather than by lowering the bar
        // for everything.
        if case.shadow.blur_radius > 0.0 {
            assert!(
                result.painted > 500,
                "[{}] painted only {} pixels, which is too few for the \
                 agreement to mean anything",
                case.name,
                result.painted
            );
        }
        total += result.total;
        exact += result.exact;
        painted += result.painted;
    }
    println!(
        "phase_6_3_shadow_gate: {exact} of {total} pixels byte-exact, {painted} of them painted \
         by at least one arm"
    );
    assert_eq!(exact, total);
}

/// The gate has been watched failing.
///
/// A differential nobody has seen reject anything is a differential nobody
/// knows works. The perturbation is the smallest one that is still definitely
/// wrong: one ULP-scale change to the blur radius, which moves the falloff
/// everywhere and nothing else.
#[test]
fn the_comparison_actually_detects_a_wrong_shadow() {
    let Some(context) = context_or_report("shadow_differential_detects_a_difference") else {
        return;
    };
    let case = Case {
        name: "control",
        shadow: Shadow {
            origin: [48.0, 40.0],
            size: [128.0, 96.0],
            color: [0.5, 0.5, 0.5, 1.0],
            corner_radii: [8.0; 4],
            blur_radius: 12.0,
        },
        hsla: [0.0, 0.0, 0.5, 1.0],
    };
    let clear = measured_clear_pixel(&context);
    let mode = DrawMode::best_available(context.indirect);
    let legacy = render_legacy(&context, &case);
    let control = compare(
        &legacy,
        &render_2_0(&context, Some(case.shadow), mode),
        clear,
    );
    assert_eq!(control.exact, control.total, "the control must agree");
    assert!(
        control.painted > 500,
        "the control painted only {} pixels, so agreeing proves little and \
         disagreeing would prove less",
        control.painted
    );

    let perturbed = Shadow {
        blur_radius: 12.05,
        ..case.shadow
    };
    let wrong = compare(&legacy, &render_2_0(&context, Some(perturbed), mode), clear);
    assert!(
        wrong.exact < wrong.total,
        "a 0.4% change in blur radius must be visible to this comparison; it \
         reported {} of {} exact",
        wrong.exact,
        wrong.total
    );
    println!(
        "the perturbed shadow disagrees at {} of {} pixels, first at {:?}",
        wrong.total - wrong.exact,
        wrong.total,
        wrong.first_difference
    );
}

/// Every draw mode this adapter offers reaches the same pixels.
///
/// `tests/indirect_draw.rs` makes this claim for quads; making it again for a
/// kind whose fragment shader is expensive is worth the seconds, because the
/// mode is what decides how many instances the GPU thinks it is drawing and a
/// shadow drawn twice would composite twice and look like a *shader* bug.
#[test]
fn every_draw_mode_produces_the_same_shadow() {
    let Some(context) = context_or_report("shadow_draw_modes") else {
        return;
    };
    let case = Case {
        name: "mode sweep",
        shadow: Shadow {
            origin: [40.0, 36.0],
            size: [120.0, 88.0],
            color: [1.0, 0.0, 0.0, 1.0],
            corner_radii: [10.0; 4],
            blur_radius: 11.0,
        },
        hsla: [0.0, 1.0, 0.5, 1.0],
    };
    let clear = measured_clear_pixel(&context);
    let legacy = render_legacy(&context, &case);
    let mut modes = 0;
    for mode in DrawMode::ALL {
        if !mode.is_available(context.indirect) {
            continue;
        }
        modes += 1;
        let result = compare(
            &legacy,
            &render_2_0(&context, Some(case.shadow), mode),
            clear,
        );
        assert_eq!(
            result.exact,
            result.total,
            "[{}] first difference at {:?}",
            mode.name(),
            result.first_difference
        );
        assert!(result.painted > 500, "[{}] painted nothing", mode.name());
    }
    assert!(modes >= 2, "only {modes} draw modes were exercised");
    println!("every shadow drawn identically across {modes} draw modes");
}

/// A card's drop shadow still shows around the card — the behaviour
/// `window_shadow` and the `shadow` bench are actually about.
///
/// # What this test proves, and the larger thing it does not
///
/// It was written to prove the two decisions `Shadow` makes that no earlier
/// primitive kind did — [`Shadow::drawn_bounds`] reaching the ordering pass, and
/// [`wgpui_core::occlusion::CoverageItem::uncullable`] instead of `cullee`. It
/// does **not** prove either, and that was established by removing them rather
/// than by reading the code:
///
/// - Replacing `uncullable` with `cullee`: this test still passes, and so does
///   the whole gate above.
/// - Feeding the ordering pass `origin`/`size` instead of `drawn_bounds`: same.
/// - Both at once: same.
///
/// The reason is a limitation `frame.rs` already documents for glyphs and that
/// applies with more force here: **occlusion dispatches per kind**, so the quad
/// below is in a different dispatch and can never cull the shadow whatever its
/// flag says. Nor can a shadow cull a shadow — a shadow is never an occluder.
/// And `keep_item` keeps an item whose visible rectangle is empty rather than
/// dropping it, so even a shadow entirely outside the clip survives.
///
/// So both adjustments are **correct, matching legacy, and currently inert**.
/// They are kept because they are right — and because the moment cross-kind
/// occlusion exists, a shadow culled against its unblurred rectangle would lose
/// falloff that was never covered. Recorded here rather than left for a future
/// phase to discover that a mechanism it depended on had never been exercised.
///
/// What the test *does* prove is worth keeping on its own terms: the composite
/// a real drop shadow produces — shadow under card, falloff visible around it —
/// comes out of the real frame path, with `Shadow` sorting below `Quad`.
#[test]
fn a_shadow_covered_by_an_opaque_quad_still_paints_its_falloff_outside_it() {
    let Some(context) = context_or_report("shadow_survives_coverage") else {
        return;
    };
    let clear = measured_clear_pixel(&context);
    let mode = DrawMode::best_available(context.indirect);

    let shadow = Shadow {
        origin: [80.0, 60.0],
        size: [64.0, 48.0],
        color: [1.0, 1.0, 1.0, 1.0],
        corner_radii: [4.0; 4],
        blur_radius: 16.0,
    };
    // Strictly larger than the shadow's own rectangle on every side. Written
    // that way so a coverage test against that rectangle *would* succeed
    // completely — see this test's doc for why no such test runs today.
    let cover = Quad {
        origin: [70.0, 50.0],
        size: [84.0, 68.0],
        background: [0.0, 0.0, 0.0, 1.0],
        border_color: [0.0, 0.0, 0.0, 1.0],
        corner_radii: [0.0; 4],
        border_widths: [0.0; 4],
    };

    let render = |with_shadow: bool| -> Vec<u8> {
        let mut scene = Scene::new();
        let layer = scene.layer(LayerKey::untiled(BoundaryId::from_raw(1)));
        let mut patch = ScenePatch::new();
        if with_shadow {
            patch
                .shadows
                .append(layer, RecordKey::from_raw(1), 0, shadow);
        }
        patch.quads.append(layer, RecordKey::from_raw(2), 0, cover);
        apply(&mut scene, &patch).expect("the patch must apply");

        let mut renderer = FrameRenderer::new(&context.device);
        let target = OffscreenTarget::new(&context.device, WIDTH, HEIGHT);
        let input = FrameInput {
            scene: &scene,
            clip: Rect::from_origin_size([0.0, 0.0], [WIDTH as f32, HEIGHT as f32]),
            poison: &[],
            dirty: Dirty::All,
            uploads: &[],
            composites: &[],
            registry: None,
            atlas: None,
            viewport: [WIDTH as f32, HEIGHT as f32],
            mode,
        };
        renderer
            .render_to(
                &context.device,
                &context.queue,
                &input,
                &RenderTarget {
                    view: &target.view,
                    width: target.width,
                    height: target.height,
                    clear: CLEAR_COLOR,
                    source: None,
                },
            )
            .expect("a frame must render");
        target
            .read_pixels(&context.device, &context.queue)
            .expect("reading back must succeed")
    };

    let with_shadow = render(true);
    let without_shadow = render(false);
    let pixel = |bytes: &[u8], x: usize, y: usize| -> [u8; 4] {
        let index = (y * WIDTH as usize + x) * 4;
        [
            bytes[index],
            bytes[index + 1],
            bytes[index + 2],
            bytes[index + 3],
        ]
    };

    // Four samples outside the quad on each side, inside the 3σ margin.
    let samples = [(60usize, 84usize), (164, 84), (112, 40), (112, 128)];
    let mut lit = 0;
    for (x, y) in samples {
        let covered = pixel(&without_shadow, x, y);
        assert_eq!(
            covered, clear,
            "sample ({x}, {y}) must be outside the quad, or it proves nothing"
        );
        let shadowed = pixel(&with_shadow, x, y);
        assert_ne!(
            shadowed, clear,
            "sample ({x}, {y}) is inside the shadow's blur margin and outside \
             the covering quad, and it is blank — the shadow did not reach the \
             framebuffer at all"
        );
        lit += 1;
    }
    assert_eq!(lit, samples.len());

    // And the quad really does cover the shadow's own rectangle, so the
    // assertion above is about the falloff rather than about a shadow that was
    // never covered in the first place.
    let centre = pixel(&with_shadow, 112, 84);
    assert_eq!(
        centre,
        pixel(&without_shadow, 112, 84),
        "the quad paints over the shadow's core in both renders — `Shadow` \
         sorts below `Quad`"
    );
    println!("all {lit} out-of-quad samples carry shadow falloff; the core is covered");
}

/// The legacy file still has the shape this test's "no wrapper" argument rests
/// on.
///
/// These are `slab_shader_source`'s own assertions (`renderer.rs:108`, `:134`),
/// restated here. If the frozen shader is ever edited so that either pattern
/// stops matching exactly once, the legacy renderer's own load would fail — and
/// so should this test, rather than quietly comparing against a shader whose
/// wrapper is no longer a no-op.
#[test]
fn the_legacy_source_still_has_the_shape_this_test_relies_on() {
    assert_eq!(
        LEGACY_SHADOWS_WGSL
            .matches("let device_position = position / globals.viewport_size")
            .count(),
        1,
        "the vertex-position pattern the slab wrapper rewrites has drifted"
    );
    assert_eq!(
        LEGACY_SHADOWS_WGSL
            .matches("let center_to_point = input.position.xy - center;")
            .count(),
        1,
        "the fragment pattern the slab wrapper rewrites has drifted"
    );
    // And the entry points this test names.
    assert!(LEGACY_SHADOWS_WGSL.contains("fn vs_shadow("));
    assert!(LEGACY_SHADOWS_WGSL.contains("fn fs_shadow("));
}

/// The hand-built legacy struct bytes land where WGSL says they do.
///
/// WGSL derives `Shadow`'s layout from its members: `order: u32` at 0,
/// `blur_radius: f32` at 4, then `Bounds` (two `vec2<f32>`, so 8-byte aligned)
/// at 8, `Corners` (four `f32`) at 24, another `Bounds` at 40, and `Hsla` (four
/// `f32`) at 56 — 72 bytes, which is exactly what `src/scene.rs:1488` asserts of
/// the Rust struct the renderer actually uploads.
#[test]
fn the_legacy_struct_layout_is_the_one_wgsl_derives() {
    let shadow = Shadow {
        origin: [1.0, 2.0],
        size: [3.0, 4.0],
        color: [0.0, 0.0, 0.0, 0.25],
        corner_radii: [5.0; 4],
        blur_radius: 6.0,
    };
    let bytes = encode_legacy_shadow(&shadow, [0.125, 0.25, 0.375, 0.25]);
    assert_eq!(&bytes[0..4], &0u32.to_le_bytes());
    assert_eq!(&bytes[4..8], &6.0f32.to_le_bytes());
    assert_eq!(&bytes[8..12], &1.0f32.to_le_bytes());
    assert_eq!(&bytes[20..24], &4.0f32.to_le_bytes());
    for corner in 0..4 {
        assert_eq!(
            &bytes[24 + corner * 4..28 + corner * 4],
            &5.0f32.to_le_bytes()
        );
    }
    assert_eq!(&bytes[56..60], &0.125f32.to_le_bytes());
    assert_eq!(&bytes[68..72], &0.25f32.to_le_bytes());
}

/// The colour handed to each arm is the same colour, checked rather than
/// assumed.
#[test]
fn the_colour_transcription_is_checked_rather_than_assumed() {
    for case in cases() {
        assert_eq!(
            hsla_to_rgba(case.hsla),
            case.shadow.color,
            "[{}] HSLA {:?} does not convert to RGBA {:?}",
            case.name,
            case.hsla,
            case.shadow.color
        );
    }
    // And the transcription is not vacuous — it distinguishes colours.
    assert_ne!(
        hsla_to_rgba([0.0, 1.0, 0.5, 1.0]),
        hsla_to_rgba([0.5, 1.0, 0.5, 1.0])
    );
}
