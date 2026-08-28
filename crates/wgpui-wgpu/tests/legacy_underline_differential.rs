//! Phase 6.3's gate for underlines: **byte-exact against legacy output**.
//!
//! `tests/legacy_shadow_differential.rs`'s method, one shader over, and its
//! header carries the full argument for why compiling the frozen legacy `.wgsl`
//! file itself is a stronger oracle than a transcription and why rendering it
//! without the `slab_shader_source` wrapper is a no-op at the identity
//! transform. Only what differs for this kind is repeated here.
//!
//! # The legacy alpha is squared, and the differential holds both sides to it
//!
//! `fs_underline` ends `blend_color(input.color, input.color.a)`, and
//! `blend_color`'s body is `alpha = color.a * alpha_factor` — so a translucent
//! underline composites at `a²`. The wavy branch does the same one step later.
//! 2.0's `underlines.wgsl` reproduces this deliberately (see its header), so the
//! two arms agree — and [`the_legacy_alpha_really_is_squared`] pins the
//! behaviour down as a *measured fact about both* rather than leaving it as a
//! claim in a comment. It is a legacy bug; parity is the goal while both
//! backends coexist, and `docs/phase-6.3-results.md` flags it for revisit.
//!
//! # What this proof does not cover
//!
//! Per-fragment clipping (2.0 clips in the occlusion pass instead) and
//! colour-space conversion (both arms are given colours whose HSLA→RGBA is
//! exact — see the shadow differential's header).

use wgpui_core::geometry::Rect;
use wgpui_core::patch::RecordKey;
use wgpui_core::patch::apply::{ScenePatch, apply};
use wgpui_core::patch::primitive::Underline;
use wgpui_core::scene::Scene;
use wgpui_core::scene::layer::{BoundaryId, LayerKey};
use wgpui_wgpu::render::device::{ComputeContext, context_or_report};
use wgpui_wgpu::render::draw::DrawMode;
use wgpui_wgpu::render::frame::{
    Dirty, FrameInput, FrameRenderer, OffscreenTarget, RenderTarget,
};
use wgpui_wgpu::render::pipelines::TARGET_FORMAT;

/// The legacy shader, byte for byte off disk.
const LEGACY_UNDERLINES_WGSL: &str =
    include_str!("../../../src/platform/cross/shaders/underlines.wgsl");

const WIDTH: u32 = 256;
const HEIGHT: u32 = 96;

/// Bytes one legacy `Underline` occupies — `src/scene.rs:1468` asserts the same
/// number of the Rust struct.
const LEGACY_UNDERLINE_STRIDE: usize = 64;

/// See the shadow differential's `LEGACY_BLEND` and `CLEAR_COLOR`.
const LEGACY_BLEND: wgpu::BlendState = wgpu::BlendState::ALPHA_BLENDING;
const CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: 0.25,
    g: 0.5,
    b: 0.75,
    a: 1.0,
};

struct Case {
    name: &'static str,
    underline: Underline,
    /// The legacy struct's HSLA for `underline.color`.
    hsla: [f32; 4],
}

/// The legacy `hsla_to_rgba`, transcribed from
/// `src/platform/cross/shaders/underlines.wgsl:51` — the same function the
/// shadow shader carries, and asserted the same way.
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

fn cases() -> Vec<Case> {
    let red = ([0.0, 1.0, 0.5, 1.0], [1.0, 0.0, 0.0, 1.0]);
    let cyan = ([0.5, 1.0, 0.5, 1.0], [0.0, 1.0, 1.0, 1.0]);
    let white = ([0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
    // Half alpha, which is the case the squared-alpha quirk actually bites on.
    let translucent_white = ([0.0, 0.0, 1.0, 0.5], [1.0, 1.0, 1.0, 0.5]);
    let black = ([0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]);

    vec![
        Case {
            name: "a plain straight rule",
            underline: Underline {
                origin: [24.0, 40.0],
                size: [200.0, 4.0],
                color: red.1,
                thickness: 1.0,
                wavy: false,
            },
            hsla: red.0,
        },
        Case {
            name: "a straight rule at half alpha, where the legacy square shows",
            underline: Underline {
                origin: [24.0, 40.0],
                size: [200.0, 4.0],
                color: translucent_white.1,
                thickness: 1.0,
                wavy: false,
            },
            hsla: translucent_white.0,
        },
        Case {
            name: "the wavy spelling-error squiggle",
            underline: Underline {
                origin: [16.0, 32.0],
                size: [220.0, 12.0],
                color: red.1,
                thickness: 2.0,
                wavy: true,
            },
            hsla: red.0,
        },
        Case {
            name: "a thick wave in a tall box",
            underline: Underline {
                origin: [12.5, 20.25],
                size: [230.0, 40.0],
                color: cyan.1,
                thickness: 6.5,
                wavy: true,
            },
            hsla: cyan.0,
        },
        Case {
            name: "a hairline wave, where the antialiased edge is most of the ink",
            underline: Underline {
                origin: [20.0, 44.0],
                size: [210.0, 9.0],
                color: white.1,
                thickness: 0.75,
                wavy: true,
            },
            hsla: white.0,
        },
        Case {
            name: "a wave at half alpha",
            underline: Underline {
                origin: [20.0, 30.0],
                size: [210.0, 16.0],
                color: translucent_white.1,
                thickness: 3.0,
                wavy: true,
            },
            hsla: translucent_white.0,
        },
        Case {
            name: "a strikethrough-shaped rule spanning the whole viewport",
            underline: Underline {
                origin: [-20.0, 46.0],
                size: [300.0, 3.0],
                color: black.1,
                thickness: 3.0,
                wavy: false,
            },
            hsla: black.0,
        },
        Case {
            name: "a wave whose thickness exceeds its box, which neither side clamps",
            underline: Underline {
                origin: [30.0, 38.0],
                size: [190.0, 6.0],
                color: cyan.1,
                thickness: 14.0,
                wavy: true,
            },
            hsla: cyan.0,
        },
    ]
}

/// The 64 bytes the legacy `Underline` occupies, in its own field order:
/// `order: u32` at 0, `pad: u32` at 4, `bounds` at 8, `content_mask` at 24,
/// `color` at 40, `thickness` at 56, `wavy` at 60.
fn encode_legacy_underline(
    underline: &Underline,
    hsla: [f32; 4],
) -> [u8; LEGACY_UNDERLINE_STRIDE] {
    let mut bytes = [0u8; LEGACY_UNDERLINE_STRIDE];
    {
        let mut put = |offset: usize, value: f32| {
            bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        };
        put(8, underline.origin[0]);
        put(12, underline.origin[1]);
        put(16, underline.size[0]);
        put(20, underline.size[1]);
        // A content mask far larger than the viewport — see the shadow
        // differential's header for why, and for what that puts outside the
        // proof.
        put(24, -100_000.0);
        put(28, -100_000.0);
        put(32, 200_000.0);
        put(36, 200_000.0);
        for (index, channel) in hsla.iter().enumerate() {
            put(40 + index * 4, *channel);
        }
        put(56, underline.thickness);
    }
    bytes[60..64].copy_from_slice(&u32::from(underline.wavy).to_le_bytes());
    bytes
}

fn render_legacy(context: &ComputeContext, case: &Case) -> Vec<u8> {
    let device = &context.device;
    let queue = &context.queue;

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("legacy underlines"),
        source: wgpu::ShaderSource::Wgsl(LEGACY_UNDERLINES_WGSL.into()),
    });

    let mut globals = [0u8; 16];
    globals[0..4].copy_from_slice(&(WIDTH as f32).to_le_bytes());
    globals[4..8].copy_from_slice(&(HEIGHT as f32).to_le_bytes());
    // premultiplied_alpha = 0, matching `flamegraph_replay.rs:596`'s offscreen
    // use of these same shaders.
    let globals_buffer =
        buffer_with(device, queue, "legacy globals", wgpu::BufferUsages::UNIFORM, &globals);
    let underline_bytes = encode_legacy_underline(&case.underline, case.hsla);
    let underline_buffer = buffer_with(
        device,
        queue,
        "legacy underlines",
        wgpu::BufferUsages::STORAGE,
        &underline_bytes,
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
    let underline_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("legacy underline storage"),
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
        label: Some("legacy underlines"),
        bind_group_layouts: &[Some(&globals_layout), Some(&underline_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("legacy underlines"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_underline"),
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
            entry_point: Some("fs_underline"),
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
    let underline_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("legacy underline storage"),
        layout: &underline_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: underline_buffer.as_entire_binding(),
        }],
    });

    let target = OffscreenTarget::new(device, WIDTH, HEIGHT);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("legacy underline frame"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("legacy underline frame"),
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
        pass.set_bind_group(1, &underline_group, &[]);
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

/// Render the same underline through 2.0's whole frame path.
fn render_2_0(context: &ComputeContext, underline: Option<Underline>, mode: DrawMode) -> Vec<u8> {
    let mut scene = Scene::new();
    let layer = scene.layer(LayerKey::untiled(BoundaryId::from_raw(1)));
    let mut patch = ScenePatch::new();
    if let Some(underline) = underline {
        patch
            .underlines
            .append(layer, RecordKey::from_raw(1), 0, underline);
    }
    apply(&mut scene, &patch).expect("seeding one layer with an underline must apply");

    let mut renderer = FrameRenderer::new(&context.device);
    let target = OffscreenTarget::new(&context.device, WIDTH, HEIGHT);
    let input = FrameInput {
        scene: &scene,
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
            },
        )
        .expect("a frame must render");
    target
        .read_pixels(&context.device, &context.queue)
        .expect("reading the 2.0 target back must succeed")
}

#[derive(Default, Debug)]
struct Comparison {
    total: usize,
    exact: usize,
    painted: usize,
    painted_exact: usize,
    first_difference: Option<(usize, usize, [u8; 4], [u8; 4])>,
}

fn compare(legacy: &[u8], ours: &[u8], clear: [u8; 4]) -> Comparison {
    let mut result = Comparison::default();
    assert_eq!(legacy.len(), ours.len(), "both arms read back the same extent");
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

fn measured_clear_pixel(context: &ComputeContext) -> [u8; 4] {
    let ours = render_2_0(context, None, DrawMode::best_available(context.indirect));
    let legacy = render_legacy(
        context,
        &Case {
            name: "empty",
            underline: Underline::ZERO,
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

/// **The gate.** Every case, byte-exact.
#[test]
fn phase_6_3_underline_gate() {
    let Some(context) = context_or_report("phase_6_3_underline_gate") else {
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
    let mut wavy_cases = 0usize;
    for case in cases() {
        assert_eq!(
            hsla_to_rgba(case.hsla),
            case.underline.color,
            "[{}] the two arms must be given the same colour",
            case.name
        );
        let legacy = render_legacy(&context, &case);
        let ours = render_2_0(
            &context,
            Some(case.underline),
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
        assert!(
            result.painted > 200,
            "[{}] painted only {} pixels, which is too few for the agreement to \
             mean anything",
            case.name,
            result.painted
        );
        if case.underline.wavy {
            wavy_cases += 1;
        }
        total += result.total;
        exact += result.exact;
        painted += result.painted;
    }
    assert!(
        wavy_cases >= 4,
        "the wavy branch is the whole of the interesting fragment maths and only \
         {wavy_cases} cases exercise it"
    );
    println!(
        "phase_6_3_underline_gate: {exact} of {total} pixels byte-exact, {painted} of them \
         painted by at least one arm, across {wavy_cases} wavy cases"
    );
    assert_eq!(exact, total);
}

/// The gate has been watched failing.
#[test]
fn the_comparison_actually_detects_a_wrong_underline() {
    let Some(context) = context_or_report("underline_differential_detects_a_difference") else {
        return;
    };
    let clear = measured_clear_pixel(&context);
    let mode = DrawMode::best_available(context.indirect);
    let case = Case {
        name: "control",
        underline: Underline {
            origin: [16.0, 32.0],
            size: [220.0, 12.0],
            color: [1.0, 0.0, 0.0, 1.0],
            thickness: 2.0,
            wavy: true,
        },
        hsla: [0.0, 1.0, 0.5, 1.0],
    };
    let legacy = render_legacy(&context, &case);
    let control = compare(&legacy, &render_2_0(&context, Some(case.underline), mode), clear);
    assert_eq!(control.exact, control.total, "the control must agree");
    assert!(control.painted > 200);

    // Two perturbations, because this shader has two branches and a single one
    // would only ever prove the comparison discriminates on the branch it hit.
    let thicker = Underline {
        thickness: 2.05,
        ..case.underline
    };
    let straightened = Underline {
        wavy: false,
        ..case.underline
    };
    for (label, wrong) in [("a 2.5% thicker wave", thicker), ("wavy turned off", straightened)] {
        let result = compare(&legacy, &render_2_0(&context, Some(wrong), mode), clear);
        assert!(
            result.exact < result.total,
            "[{label}] must be visible to this comparison; it reported {} of {} \
             exact",
            result.exact,
            result.total
        );
        println!(
            "{label}: disagrees at {} of {} pixels, first at {:?}",
            result.total - result.exact,
            result.total,
            result.first_difference
        );
    }
}

/// Every draw mode reaches the same pixels.
#[test]
fn every_draw_mode_produces_the_same_underline() {
    let Some(context) = context_or_report("underline_draw_modes") else {
        return;
    };
    let clear = measured_clear_pixel(&context);
    let case = Case {
        name: "mode sweep",
        underline: Underline {
            origin: [16.0, 32.0],
            size: [220.0, 12.0],
            color: [0.0, 1.0, 1.0, 1.0],
            thickness: 2.5,
            wavy: true,
        },
        hsla: [0.5, 1.0, 0.5, 1.0],
    };
    let legacy = render_legacy(&context, &case);
    let mut modes = 0;
    for mode in DrawMode::ALL {
        if !mode.is_available(context.indirect) {
            continue;
        }
        modes += 1;
        let result = compare(&legacy, &render_2_0(&context, Some(case.underline), mode), clear);
        assert_eq!(
            result.exact, result.total,
            "[{}] first difference at {:?}",
            mode.name(),
            result.first_difference
        );
        assert!(result.painted > 200, "[{}] painted nothing", mode.name());
    }
    assert!(modes >= 2, "only {modes} draw modes were exercised");
    println!("every underline drawn identically across {modes} draw modes");
}

/// The squared alpha, measured on both arms rather than asserted in a comment.
///
/// A straight underline of solid white at alpha `a` over an opaque background
/// composites to `background * (1 - a²) + white * a²`. At `a = 0.5` that is
/// `a² = 0.25`, not `0.5` — so the rendered pixel sits a quarter of the way to
/// white, not halfway. This test reads the actual byte and checks it against
/// the squared prediction *and* against the unsquared one, so it fails whether
/// the behaviour is corrected in 2.0 or drifts in some other direction.
#[test]
fn the_legacy_alpha_really_is_squared() {
    let Some(context) = context_or_report("underline_alpha_is_squared") else {
        return;
    };
    let clear = measured_clear_pixel(&context);
    let case = Case {
        name: "half-alpha white rule",
        underline: Underline {
            origin: [0.0, 40.0],
            size: [WIDTH as f32, 8.0],
            color: [1.0, 1.0, 1.0, 0.5],
            thickness: 8.0,
            wavy: false,
        },
        hsla: [0.0, 0.0, 1.0, 0.5],
    };
    let legacy = render_legacy(&context, &case);
    let ours = render_2_0(
        &context,
        Some(case.underline),
        DrawMode::best_available(context.indirect),
    );
    assert_eq!(
        compare(&legacy, &ours, clear).exact,
        WIDTH as usize * HEIGHT as usize,
        "the two arms must agree before the byte means anything"
    );

    // A pixel inside the rule: row 43 of a rule spanning rows 40..48.
    let index = (43 * WIDTH as usize + WIDTH as usize / 2) * 4;
    let painted = [ours[index], ours[index + 1], ours[index + 2]];
    assert_ne!(
        painted,
        [clear[0], clear[1], clear[2]],
        "the sample must be inside the rule"
    );

    let over = |background: u8, alpha: f32| -> f32 {
        let source = 1.0f32;
        (source * alpha + f32::from(background) / 255.0 * (1.0 - alpha)) * 255.0
    };
    let squared = over(clear[0], 0.25);
    let unsquared = over(clear[0], 0.5);
    assert!(
        (f32::from(painted[0]) - squared).abs() <= 1.0,
        "the red channel read {} and the squared-alpha prediction is {squared:.1}",
        painted[0]
    );
    assert!(
        (f32::from(painted[0]) - unsquared).abs() > 1.0,
        "the squared and unsquared predictions must actually differ here, or \
         this test proves nothing: read {}, unsquared prediction {unsquared:.1}",
        painted[0]
    );
    println!(
        "a 50%-alpha underline composites at 25%: channel read {}, squared \
         prediction {squared:.1}, unsquared {unsquared:.1}",
        painted[0]
    );
}

/// The legacy file still has the shape this test's "no wrapper" argument rests
/// on — `slab_shader_source`'s own two assertions, restated.
#[test]
fn the_legacy_source_still_has_the_shape_this_test_relies_on() {
    assert_eq!(
        LEGACY_UNDERLINES_WGSL
            .matches("let device_position = position / globals.viewport_size")
            .count(),
        1,
        "the vertex-position pattern the slab wrapper rewrites has drifted"
    );
    assert_eq!(
        LEGACY_UNDERLINES_WGSL
            .matches("let st = (input.position.xy - underline.bounds.origin)")
            .count(),
        1,
        "the fragment pattern the slab wrapper rewrites has drifted"
    );
    assert!(LEGACY_UNDERLINES_WGSL.contains("fn vs_underline("));
    assert!(LEGACY_UNDERLINES_WGSL.contains("fn fs_underline("));
    // And the two constants 2.0's port transcribes.
    assert!(LEGACY_UNDERLINES_WGSL.contains("WAVE_FREQUENCY: f32 = 2.0"));
    assert!(LEGACY_UNDERLINES_WGSL.contains("WAVE_HEIGHT_RATIO: f32 = 0.8"));
}

/// The hand-built legacy struct bytes land where WGSL says they do.
#[test]
fn the_legacy_struct_layout_is_the_one_wgsl_derives() {
    let underline = Underline {
        origin: [1.0, 2.0],
        size: [3.0, 4.0],
        color: [0.0, 0.0, 0.0, 0.25],
        thickness: 7.0,
        wavy: true,
    };
    let bytes = encode_legacy_underline(&underline, [0.125, 0.25, 0.375, 0.25]);
    assert_eq!(&bytes[0..8], &[0u8; 8], "order and pad are both zero");
    assert_eq!(&bytes[8..12], &1.0f32.to_le_bytes());
    assert_eq!(&bytes[20..24], &4.0f32.to_le_bytes());
    assert_eq!(&bytes[40..44], &0.125f32.to_le_bytes());
    assert_eq!(&bytes[52..56], &0.25f32.to_le_bytes());
    assert_eq!(&bytes[56..60], &7.0f32.to_le_bytes());
    assert_eq!(&bytes[60..64], &1u32.to_le_bytes());
}
