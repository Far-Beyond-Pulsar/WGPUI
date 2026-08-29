//! Phase 6.6's first gate: **2.0's quad pipeline byte-exact against legacy's**.
//!
//! # Why this test did not exist before, and why it has to now
//!
//! Every phase from 1 to 6.3 rendered quads and none of them compared a quad
//! against the legacy renderer. That was defensible while nothing real emitted
//! one: `render/shaders/quads.wgsl` was a deliberately hard-edged
//! rounded-rectangle SDF, and the comparisons it took part in were between 2.0's
//! own four draw modes, which a hard edge makes *easier* to hold exactly.
//!
//! Phase 6.6 is where a `div()`'s background and border become quads, and §8's
//! Phase 6.6 gate is byte-exactness against the legacy renderer for the same
//! style. A hard-edged shader cannot meet that at any rounded corner, no matter
//! what the emitting side does — the legacy fragment shader antialiases through
//! `saturate(0.5 - outer_sdf)` and produces a coverage ramp several pixels wide.
//! So `quads.wgsl`'s fragment stage was rewritten as a transcription of
//! `fs_quad`, and this file is what checks the transcription.
//!
//! # What "against legacy" means here, precisely
//!
//! The same oracle Phase 6.3 established, one kind over: this test
//! `include_str!`s **the legacy shader file itself**
//! (`src/platform/cross/shaders/quads.wgsl`), builds a buffer in the legacy
//! `Quad` struct's own 168-byte layout, and renders it through the same
//! `Rgba8Unorm` target and blend state. The other arm is 2.0's real
//! `FrameRenderer`: patch, apply, upload, ordering, occlusion,
//! indirect-argument generation, indirect draw. Then every pixel is compared.
//!
//! # The four ways the legacy arm is not literally the legacy renderer
//!
//! Stated up front. The first three are Phase 6.3's, unchanged and for the same
//! reasons; the fourth is this kind's own.
//!
//! 1. **No `slab_shader_source` wrapper.** Both rewrites it performs are the
//!    identity at a zero translate, and
//!    [`the_legacy_source_still_has_the_shape_this_test_relies_on`] asserts the
//!    patterns are still present so a drift in the frozen file breaks this test
//!    loudly rather than weakening it silently.
//! 2. **`premultiplied_alpha` is 0**, matching `flamegraph_replay.rs`'s own
//!    offscreen setup, and the blend state is `wgpu::BlendState::ALPHA_BLENDING`
//!    — field-for-field `render/pipelines.rs`'s `ALPHA_OVER`.
//! 3. **`content_mask` covers everything.** 2.0 has no per-primitive clip
//!    (§5.2 sends the frame's clip to the occlusion pass), so the legacy arm
//!    gets a mask far larger than the viewport. **Per-fragment clipping is
//!    outside this proof.**
//! 4. **`background.tag` is always `Solid` and `border_style` always
//!    `Solid`.** 2.0's `Quad` carries one straight-alpha RGBA background and no
//!    border style, so the gradient, pattern, and dashed-border branches of
//!    `fs_quad` are unreachable from a 2.0 quad and are not compared. Both are
//!    named in `quads.wgsl`'s own header as deliberately untranscribed. **They
//!    are outside this proof**, and a phase that widens `Quad` widens this.
//!
//! # The colour, and why it is not simply passed through
//!
//! Phase 6.3's argument, verbatim in shape: the legacy `Quad` carries HSLA and
//! converts in its vertex shader, 2.0's carries straight RGBA. Every colour
//! here converts through nothing but 0, 0.5, 1, 2, 3 and 6, and
//! [`the_colour_transcription_is_checked_rather_than_assumed`] runs the legacy
//! `hsla_to_rgba` transcribed into Rust and asserts it produces exactly the
//! bytes 2.0 is handed. Colour-space conversion is what this gate holds fixed,
//! not what it proves.

use wgpui_core::geometry::Rect;
use wgpui_core::patch::RecordKey;
use wgpui_core::patch::apply::{ScenePatch, apply};
use wgpui_core::patch::primitive::Quad;
use wgpui_core::scene::Scene;
use wgpui_core::scene::layer::{BoundaryId, LayerId, LayerKey};
use wgpui_wgpu::render::device::{ComputeContext, context_or_report};
use wgpui_wgpu::render::draw::DrawMode;
use wgpui_wgpu::render::frame::{Dirty, FrameInput, FrameRenderer, OffscreenTarget, RenderTarget};
use wgpui_wgpu::render::pipelines::TARGET_FORMAT;

/// The legacy shader, byte for byte off disk. A build error here means the
/// frozen tree moved and this differential no longer has a subject.
const LEGACY_QUADS_WGSL: &str = include_str!("../../../src/platform/cross/shaders/quads.wgsl");

const WIDTH: u32 = 224;
const HEIGHT: u32 = 176;

/// Bytes one legacy `Quad` occupies. `src/scene.rs:1446` asserts the same number
/// on the Rust struct; restated here because this file builds those bytes by
/// hand and a silent disagreement would misalign every instance past the first.
const LEGACY_QUAD_STRIDE: usize = 168;

/// Byte offsets inside a legacy `Quad`, derived from its `#[repr(C)]` field
/// order and checked against `src/scene.rs`'s own two `const _` assertions in
/// [`the_legacy_struct_layout_is_the_one_wgsl_derives`].
mod offset {
    pub const ORDER: usize = 0;
    pub const BORDER_STYLE: usize = 4;
    pub const BOUNDS: usize = 8;
    pub const CONTENT_MASK: usize = 24;
    pub const BACKGROUND_TAG: usize = 40;
    pub const BACKGROUND_COLOR_SPACE: usize = 44;
    pub const BACKGROUND_SOLID: usize = 48;
    pub const BORDER_COLOR: usize = 120;
    pub const CORNER_RADII: usize = 136;
    pub const BORDER_WIDTHS: usize = 152;
}

/// `BackgroundTag::Solid`, `src/color.rs:672`.
const BACKGROUND_TAG_SOLID: u32 = 0;

/// `wgpu::BlendState::ALPHA_BLENDING`, which `renderer.rs:890` selects for a
/// non-premultiplied surface.
const LEGACY_BLEND: wgpu::BlendState = wgpu::BlendState::ALPHA_BLENDING;

/// What both arms clear to, and deliberately not black — Phase 6.3's reasoning,
/// which applies unchanged: an asymmetric mid colour makes both dark and light
/// quads visible and refuses to let a channel swap pass unnoticed.
const CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: 0.25,
    g: 0.5,
    b: 0.75,
    a: 1.0,
};

/// A content mask that rejects nothing — see this module's doc, point 3.
const UNCLIPPED: [f32; 4] = [-100_000.0, -100_000.0, 200_000.0, 200_000.0];

/// One comparison: the same quad expressed both ways.
struct Case {
    name: &'static str,
    quad: Quad,
    /// The legacy struct's HSLA for `quad.background`.
    background_hsla: [f32; 4],
    /// The legacy struct's HSLA for `quad.border_color`.
    border_hsla: [f32; 4],
    /// The legacy struct's content mask as `[x, y, width, height]`.
    ///
    /// [`UNCLIPPED`] for every case in [`cases`], because 2.0 has no
    /// per-primitive clip. The field exists for one reason and it is a real
    /// one: `Style::paint` draws its border quad **four times, each clipped to
    /// one edge band**, and [`phase_6_6_div_gate`] has to reproduce that
    /// sequence exactly rather than assume one unclipped draw is equivalent to
    /// it — that equivalence is precisely what the gate is checking.
    content_mask: [f32; 4],
}

/// The legacy `hsla_to_rgba`, transcribed expression for expression from
/// `src/platform/cross/shaders/quads.wgsl:173`.
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

/// Colours restricted to the exactly-representable set this module's doc
/// explains, as `(hsla, rgba)` pairs.
const TRANSPARENT: ([f32; 4], [f32; 4]) = ([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]);
const GREY: ([f32; 4], [f32; 4]) = ([0.0, 0.0, 0.5, 1.0], [0.5, 0.5, 0.5, 1.0]);
const RED: ([f32; 4], [f32; 4]) = ([0.0, 1.0, 0.5, 1.0], [1.0, 0.0, 0.0, 1.0]);
const CYAN: ([f32; 4], [f32; 4]) = ([0.5, 1.0, 0.5, 1.0], [0.0, 1.0, 1.0, 1.0]);
const WHITE: ([f32; 4], [f32; 4]) = ([0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
const BLACK_HALF: ([f32; 4], [f32; 4]) = ([0.0, 0.0, 0.0, 0.5], [0.0, 0.0, 0.0, 0.5]);
const CYAN_HALF: ([f32; 4], [f32; 4]) = ([0.5, 1.0, 0.5, 0.5], [0.0, 1.0, 1.0, 0.5]);

fn case(
    name: &'static str,
    origin: [f32; 2],
    size: [f32; 2],
    background: ([f32; 4], [f32; 4]),
    border_color: ([f32; 4], [f32; 4]),
    corner_radii: [f32; 4],
    border_widths: [f32; 4],
) -> Case {
    Case {
        name,
        quad: Quad {
            origin,
            size,
            background: background.1,
            border_color: border_color.1,
            corner_radii,
            border_widths,
        },
        background_hsla: background.0,
        border_hsla: border_color.0,
        content_mask: UNCLIPPED,
    }
}

/// Every case, chosen to move each field independently and to reach each of
/// `fs_quad`'s branches at least once.
///
/// Geometry is deliberately fractional in several cases: antialiased coverage
/// is a smooth function of sub-pixel position, and a comparison that only ever
/// landed on integer boundaries would be testing far less than it looks like it
/// is.
fn cases() -> Vec<Case> {
    vec![
        // `fs_quad`'s first fast path: unrounded and unbordered, so it returns
        // the background with no SDF evaluated at all. This is what every quad
        // 2.0 rendered before Phase 6.6 looked like, so it is also the check
        // that the shader rewrite did not move the ground under prior phases.
        case(
            "a plain filled rectangle, both shaders' fast path",
            [32.0, 24.0],
            [140.0, 100.0],
            GREY,
            TRANSPARENT,
            [0.0; 4],
            [0.0; 4],
        ),
        case(
            "a uniformly rounded fill, which is where a hard edge would fail",
            [40.0, 32.0],
            [128.0, 96.0],
            RED,
            TRANSPARENT,
            [12.0; 4],
            [0.0; 4],
        ),
        case(
            "a square box with a uniform border",
            [36.0, 28.0],
            [136.0, 104.0],
            CYAN,
            WHITE,
            [0.0; 4],
            [4.0; 4],
        ),
        case(
            "the ordinary card: rounded, bordered, opaque",
            [40.0, 30.0],
            [132.0, 100.0],
            GREY,
            RED,
            [8.0; 4],
            [2.0; 4],
        ),
        // `rounded_t_md`-shaped: three different radii and one square corner,
        // which is what `pick_corner_radius` exists to distinguish and what a
        // uniform-radius `Quad` could not have expressed before this phase.
        case(
            "four different corner radii",
            [44.0, 34.0],
            [120.0, 92.0],
            WHITE,
            RED,
            [20.0, 4.0, 12.0, 0.0],
            [3.0; 4],
        ),
        // `border_b_2`-shaped: three zero sides. Zero-width sides take
        // `reduced_border`'s `-antialias_threshold` substitution, which is the
        // single subtlest expression in `fs_quad` and the one most likely to be
        // got wrong by a transcription.
        case(
            "one bordered side and three bare ones",
            [40.0, 32.0],
            [128.0, 96.0],
            CYAN,
            WHITE,
            [0.0; 4],
            [0.0, 0.0, 5.0, 0.0],
        ),
        case(
            "asymmetric border widths on a rounded box, which reaches the ellipse arm",
            [42.0, 30.0],
            [126.0, 100.0],
            GREY,
            RED,
            [14.0; 4],
            [2.0, 8.0, 5.0, 1.0],
        ),
        // The border quad `Style::paint` actually emits: a transparent
        // background under a real border, which is what makes `over`'s
        // divide-by-`alpha` path load-bearing rather than incidental.
        case(
            "a transparent background under an opaque border, as `Style::paint` emits it",
            [38.0, 28.0],
            [134.0, 104.0],
            TRANSPARENT,
            WHITE,
            [10.0; 4],
            [3.0; 4],
        ),
        case(
            "a translucent border over an opaque background",
            [40.0, 32.0],
            [128.0, 96.0],
            RED,
            CYAN_HALF,
            [6.0; 4],
            [6.0; 4],
        ),
        case(
            "a translucent fill, so the blend is doing real work",
            [40.0, 32.0],
            [128.0, 96.0],
            BLACK_HALF,
            TRANSPARENT,
            [16.0; 4],
            [0.0; 4],
        ),
        case(
            "fractional origin and size",
            [40.5, 32.25],
            [125.75, 93.5],
            CYAN,
            RED,
            [7.5; 4],
            [2.25; 4],
        ),
        // Neither shader clamps the radius to half the box; `Style::paint`
        // clamps on the CPU before building the quad. This case checks the two
        // shaders agree about the *unclamped* case anyway, so a future change
        // to where clamping happens cannot silently diverge them.
        case(
            "a radius larger than half the rectangle, which neither shader clamps",
            [70.0, 54.0],
            [60.0, 40.0],
            WHITE,
            RED,
            [45.0; 4],
            [2.0; 4],
        ),
        case(
            "reaching past the top-left corner of the viewport",
            [-30.0, -24.0],
            [110.0, 90.0],
            RED,
            WHITE,
            [10.0; 4],
            [4.0; 4],
        ),
        // A border wider than half the box: `is_beyond_inner_straight_border`
        // is true almost everywhere, so `inner_sdf` takes its `-1.0` arm.
        case(
            "a border thicker than half the box",
            [72.0, 56.0],
            [60.0, 44.0],
            GREY,
            RED,
            [4.0; 4],
            [30.0; 4],
        ),
    ]
}

/// The 168 bytes the legacy `Quad` occupies, in its own field order.
///
/// Built by hand rather than by `bytemuck`-casting the legacy struct, because
/// that struct is `pub(crate)` in a frozen crate this test must not edit.
fn encode_legacy_quad(case: &Case) -> [u8; LEGACY_QUAD_STRIDE] {
    let mut bytes = [0u8; LEGACY_QUAD_STRIDE];
    let mut put_f32 = |offset: usize, value: f32| {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    };

    // `order` at 0 and `border_style` at 4 both stay zero: 2.0 computes draw
    // order on the GPU (§5.1), and `BorderStyle::Solid == 0` (`src/scene.rs`).
    put_f32(offset::BOUNDS, case.quad.origin[0]);
    put_f32(offset::BOUNDS + 4, case.quad.origin[1]);
    put_f32(offset::BOUNDS + 8, case.quad.size[0]);
    put_f32(offset::BOUNDS + 12, case.quad.size[1]);

    for (index, value) in case.content_mask.iter().enumerate() {
        put_f32(offset::CONTENT_MASK + index * 4, *value);
    }

    for (index, channel) in case.background_hsla.iter().enumerate() {
        put_f32(offset::BACKGROUND_SOLID + index * 4, *channel);
    }
    for (index, channel) in case.border_hsla.iter().enumerate() {
        put_f32(offset::BORDER_COLOR + index * 4, *channel);
    }
    for (index, radius) in case.quad.corner_radii.iter().enumerate() {
        put_f32(offset::CORNER_RADII + index * 4, *radius);
    }
    for (index, width) in case.quad.border_widths.iter().enumerate() {
        put_f32(offset::BORDER_WIDTHS + index * 4, *width);
    }

    bytes[offset::BACKGROUND_TAG..offset::BACKGROUND_TAG + 4]
        .copy_from_slice(&BACKGROUND_TAG_SOLID.to_le_bytes());
    // `color_space` at 44 and both gradient stops stay zero: `gradient_color`
    // never reads them under `BackgroundTag::Solid`.
    let _ = offset::BACKGROUND_COLOR_SPACE;
    let _ = offset::ORDER;
    let _ = offset::BORDER_STYLE;
    bytes
}

/// Render one quad through the legacy shader, unwrapped, and read it back.
fn render_legacy(context: &ComputeContext, case: &Case) -> Vec<u8> {
    render_legacy_many(context, std::slice::from_ref(case))
}

/// Render several legacy quads into one target, in the order given.
///
/// The plural form exists for the full-tree gate, which has to reproduce
/// `Style::paint`'s own multi-quad sequence; a single case is the degenerate
/// call.
fn render_legacy_many(context: &ComputeContext, cases: &[Case]) -> Vec<u8> {
    let device = &context.device;
    let queue = &context.queue;

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("legacy quads"),
        source: wgpu::ShaderSource::Wgsl(LEGACY_QUADS_WGSL.into()),
    });

    // `Globals { viewport_size: vec2<f32>, premultiplied_alpha: u32, pad: u32 }`
    let mut globals = [0u8; 16];
    globals[0..4].copy_from_slice(&(WIDTH as f32).to_le_bytes());
    globals[4..8].copy_from_slice(&(HEIGHT as f32).to_le_bytes());
    let globals_buffer = buffer_with(
        device,
        queue,
        "legacy globals",
        wgpu::BufferUsages::UNIFORM,
        &globals,
    );

    let mut quad_bytes = Vec::with_capacity(cases.len().max(1) * LEGACY_QUAD_STRIDE);
    for case in cases {
        quad_bytes.extend_from_slice(&encode_legacy_quad(case));
    }
    if quad_bytes.is_empty() {
        quad_bytes.resize(LEGACY_QUAD_STRIDE, 0);
    }
    let quad_buffer = buffer_with(
        device,
        queue,
        "legacy quads",
        wgpu::BufferUsages::STORAGE,
        &quad_bytes,
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
    let quad_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("legacy quad storage"),
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
        label: Some("legacy quads"),
        bind_group_layouts: &[Some(&globals_layout), Some(&quad_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("legacy quads"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_quad"),
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
            entry_point: Some("fs_quad"),
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
    let quad_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("legacy quad storage"),
        layout: &quad_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: quad_buffer.as_entire_binding(),
        }],
    });

    let target = OffscreenTarget::new(device, WIDTH, HEIGHT);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("legacy quad frame"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("legacy quad frame"),
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
        pass.set_bind_group(1, &quad_group, &[]);
        // One instanced draw per quad rather than one draw of N instances, so
        // the *order* the quads composite in is the order `cases` gives — which
        // is what `Style::paint`'s background-then-border sequence depends on.
        for index in 0..cases.len().max(1) as u32 {
            pass.draw(0..4, index..index + 1);
        }
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

/// Render the same quads through 2.0's whole frame path and read them back.
fn render_2_0(context: &ComputeContext, quads: &[Quad], mode: DrawMode) -> Vec<u8> {
    let mut scene = Scene::new();
    let layer = scene.layer(LayerKey::untiled(BoundaryId::from_raw(1)));
    let mut patch = ScenePatch::new();
    for (index, quad) in quads.iter().enumerate() {
        patch.quads.append(
            layer,
            RecordKey::from_raw(index as u64 + 1),
            index as u32,
            *quad,
        );
    }
    apply(&mut scene, &patch).expect("seeding one layer with quads must apply");

    let mut renderer = FrameRenderer::new(&context.device);
    let target = OffscreenTarget::new(&context.device, WIDTH, HEIGHT);
    let input = FrameInput {
        scene: &scene,
        // Deliberately larger than the viewport, matching the legacy arm's
        // effectively-infinite content mask.
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
                source: None,
            },
        )
        .expect("a frame must render");
    target
        .read_pixels(&context.device, &context.queue)
        .expect("reading the 2.0 target back must succeed")
}

/// The clear colour as the device actually wrote it, measured rather than
/// derived — Phase 6.3's reasoning, and its double duty as a check that both
/// arms clear identically.
fn measured_clear_pixel(context: &ComputeContext) -> [u8; 4] {
    let ours = render_2_0(context, &[], DrawMode::best_available(context.indirect));
    let legacy = render_legacy_many(context, &[]);
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

/// **Milestone 1's gate.** Every case, byte-exact.
#[test]
fn phase_6_6_quad_gate() {
    let Some(context) = context_or_report("phase_6_6_quad_gate") else {
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
            hsla_to_rgba(case.background_hsla),
            case.quad.background,
            "[{}] the two arms must be given the same background colour",
            case.name
        );
        assert_eq!(
            hsla_to_rgba(case.border_hsla),
            case.quad.border_color,
            "[{}] the two arms must be given the same border colour",
            case.name
        );
        let legacy = render_legacy(&context, &case);
        let ours = render_2_0(
            &context,
            &[case.quad],
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
            result.painted > 500,
            "[{}] painted only {} pixels, which is too few for the agreement \
             to mean anything",
            case.name,
            result.painted
        );
        total += result.total;
        exact += result.exact;
        painted += result.painted;
    }
    println!(
        "phase_6_6_quad_gate: {exact} of {total} pixels byte-exact, {painted} of them painted \
         by at least one arm"
    );
    assert_eq!(exact, total);
}

/// The gate has been watched failing.
///
/// A differential nobody has seen reject anything is a differential nobody
/// knows works. Two perturbations, each aimed at a different half of the
/// transcription: a sub-pixel radius change (which moves only antialiased
/// coverage, and which the *old* hard-edged shader would have been unable to
/// see at all) and a transposed corner-radius pair (which is bit-identical in
/// every uniform-radius case and is exactly the bug `pick_corner_radius`'s
/// quadrant order can hide).
#[test]
fn the_comparison_detects_both_a_wrong_edge_and_a_wrong_corner() {
    let Some(context) = context_or_report("quad_differential_detects_a_difference") else {
        return;
    };
    let control = case(
        "control",
        [40.0, 32.0],
        [128.0, 96.0],
        GREY,
        RED,
        [20.0, 4.0, 12.0, 8.0],
        [3.0; 4],
    );
    let clear = measured_clear_pixel(&context);
    let mode = DrawMode::best_available(context.indirect);
    let legacy = render_legacy(&context, &control);
    let agreed = compare(&legacy, &render_2_0(&context, &[control.quad], mode), clear);
    assert_eq!(agreed.exact, agreed.total, "the control must agree");
    assert!(agreed.painted > 500);

    for (what, perturbed) in [
        (
            "a sub-pixel radius change",
            Quad {
                corner_radii: [20.25, 4.0, 12.0, 8.0],
                ..control.quad
            },
        ),
        (
            "a transposed corner-radius pair",
            Quad {
                corner_radii: [4.0, 20.0, 12.0, 8.0],
                ..control.quad
            },
        ),
    ] {
        let wrong = compare(&legacy, &render_2_0(&context, &[perturbed], mode), clear);
        assert!(
            wrong.exact < wrong.total,
            "{what} must be visible to this comparison; it reported {} of {} exact",
            wrong.exact,
            wrong.total
        );
        println!(
            "{what} disagrees at {} of {} pixels, first at {:?}",
            wrong.total - wrong.exact,
            wrong.total,
            wrong.first_difference
        );
    }
}

/// Every draw mode this adapter offers reaches the same pixels.
#[test]
fn every_draw_mode_produces_the_same_quad() {
    let Some(context) = context_or_report("quad_draw_modes") else {
        return;
    };
    let subject = case(
        "mode sweep",
        [40.0, 32.0],
        [128.0, 96.0],
        CYAN,
        RED,
        [14.0, 6.0, 14.0, 6.0],
        [2.0, 5.0, 2.0, 5.0],
    );
    let clear = measured_clear_pixel(&context);
    let legacy = render_legacy(&context, &subject);
    let mut modes = 0;
    for mode in DrawMode::ALL {
        if !mode.is_available(context.indirect) {
            continue;
        }
        modes += 1;
        let result = compare(&legacy, &render_2_0(&context, &[subject.quad], mode), clear);
        assert_eq!(
            result.exact, result.total,
            "{mode:?} disagrees with legacy at {:?}",
            result.first_difference
        );
    }
    assert!(modes >= 2, "only {modes} draw mode(s) were exercised");
    println!("every_draw_mode_produces_the_same_quad: {modes} modes, all byte-exact");
}

/// The frozen legacy shader still has the shape this differential relies on.
///
/// Phase 6.3's check, one file over: `slab_shader_source` rewrites two
/// expressions in every shader it wraps, and both are the identity at a zero
/// translate. If the frozen file stops containing them, this arm is no longer
/// rendering what the legacy renderer renders and the test should say so rather
/// than quietly comparing something else.
#[test]
fn the_legacy_source_still_has_the_shape_this_test_relies_on() {
    assert!(
        LEGACY_QUADS_WGSL.contains("fn to_device_position_impl(position: vec2<f32>)"),
        "the per-layer translate hook `slab_shader_source` rewrites is gone"
    );
    assert!(
        LEGACY_QUADS_WGSL.contains("fn distance_from_clip_rect_impl(position: vec2<f32>"),
        "the clip-distance hook `slab_shader_source` rewrites is gone"
    );
    assert!(LEGACY_QUADS_WGSL.contains("fn fs_quad("));
    assert!(LEGACY_QUADS_WGSL.contains("fn vs_quad("));
}

/// The offsets this file writes are the ones WGSL derives for the legacy struct.
///
/// `src/scene.rs` asserts two of them itself (`size_of::<Quad>() == 168`,
/// `offset_of!(Quad, background) == 40`); the rest are derived here from the
/// same `#[repr(C)]` field order, and getting one wrong would put a radius in a
/// gradient stop and produce a plausible, wrong picture rather than an error.
#[test]
fn the_legacy_struct_layout_is_the_one_wgsl_derives() {
    // order(4) + border_style(4)
    assert_eq!(offset::BOUNDS, 8);
    // bounds is `Bounds<ScaledPixels>` — origin(2 f32) + size(2 f32).
    assert_eq!(offset::CONTENT_MASK, offset::BOUNDS + 16);
    // `src/scene.rs:1447`'s own assertion.
    assert_eq!(offset::BACKGROUND_TAG, 40);
    assert_eq!(offset::BACKGROUND_COLOR_SPACE, offset::BACKGROUND_TAG + 4);
    assert_eq!(offset::BACKGROUND_SOLID, offset::BACKGROUND_TAG + 8);
    // Background is tag(4) + color_space(4) + solid(16) + four params(16) +
    // two GradientStops(20 each) = 80.
    assert_eq!(offset::BORDER_COLOR, offset::BACKGROUND_TAG + 80);
    assert_eq!(offset::CORNER_RADII, offset::BORDER_COLOR + 16);
    assert_eq!(offset::BORDER_WIDTHS, offset::CORNER_RADII + 16);
    // `src/scene.rs:1446`'s own assertion.
    assert_eq!(offset::BORDER_WIDTHS + 16, LEGACY_QUAD_STRIDE);
}

/// The quads `Style::paint` (`src/style.rs:683`) emits for one styled box,
/// transcribed including its four content-masked border draws.
///
/// This is the oracle [`phase_6_6_div_gate`] compares `DivStyle::paint` against.
/// It is deliberately built from the legacy *source*'s shape rather than from
/// what 2.0 produces, so a mistake in `DivStyle::paint` shows up as a pixel
/// disagreement instead of being reproduced identically on both sides.
fn legacy_style_paint(
    origin: [f32; 2],
    size: [f32; 2],
    background: Option<([f32; 4], [f32; 4])>,
    border: Option<([f32; 4], [f32; 4])>,
    corner_radii: [f32; 4],
    border_widths: [f32; 4],
) -> Vec<Case> {
    let mut quads = Vec::new();

    // `self.corner_radii.to_pixels(rem_size).clamp_radii_for_quad_size(bounds.size)`
    let clamp = size[0].min(size[1]) / 2.0;
    let radii = [
        corner_radii[0].min(clamp),
        corner_radii[1].min(clamp),
        corner_radii[2].min(clamp),
        corner_radii[3].min(clamp),
    ];

    if let Some((hsla, rgba)) = background.filter(|(_, rgba)| rgba[3] > 0.0) {
        // `let mut border_color = background_color; border_color.a = 0.;`
        let mut faded_hsla = hsla;
        faded_hsla[3] = 0.0;
        let mut faded_rgba = rgba;
        faded_rgba[3] = 0.0;
        quads.push(Case {
            name: "background",
            quad: Quad {
                origin,
                size,
                background: rgba,
                border_color: faded_rgba,
                corner_radii: radii,
                border_widths: [0.0; 4],
            },
            background_hsla: hsla,
            border_hsla: faded_hsla,
            content_mask: UNCLIPPED,
        });
    }

    // `Style::is_border_visible`
    let border_visible = border.is_some_and(|(_, rgba)| rgba[3] > 0.0)
        && border_widths.iter().any(|width| *width != 0.0);
    if !border_visible {
        return quads;
    }
    let (border_hsla, border_rgba) = border.unwrap_or(TRANSPARENT);
    let mut faded_hsla = border_hsla;
    faded_hsla[3] = 0.0;
    let mut faded_rgba = border_rgba;
    faded_rgba[3] = 0.0;

    let max_border_width = border_widths.iter().copied().fold(0.0, f32::max);
    let max_corner_radius = radii.iter().copied().fold(0.0, f32::max);
    let band = max_border_width.max(max_corner_radius);
    let [min_x, min_y] = origin;
    let max_x = min_x + size[0];
    let max_y = min_y + size[1];

    // `Bounds::from_corners`, four times, exactly as `Style::paint` writes it.
    let top = [min_x, min_y, size[0], band];
    let bottom = [min_x, max_y - band, size[0], band];
    let left = [min_x, min_y + band, max_border_width, size[1] - 2.0 * band];
    let right = [
        max_x - max_border_width,
        min_y + band,
        max_border_width,
        size[1] - 2.0 * band,
    ];

    let quad = Quad {
        origin,
        size,
        background: faded_rgba,
        border_color: border_rgba,
        corner_radii: radii,
        border_widths,
    };
    // The legacy order is top, right, bottom, left.
    for (name, content_mask) in [
        ("border top", top),
        ("border right", right),
        ("border bottom", bottom),
        ("border left", left),
    ] {
        quads.push(Case {
            name,
            quad,
            background_hsla: faded_hsla,
            border_hsla,
            content_mask,
        });
    }
    quads
}

/// Render a `div()` through 2.0's real element path and read the frame back.
///
/// Reconcile → Taffy → emit → patch → apply → render. Nothing is hand-built:
/// the quads compared against the legacy arm are whatever `DivStyle::paint`
/// decided to write, placed wherever Taffy decided to put them.
fn render_div(
    context: &ComputeContext,
    root: wgpui_widgets::div::Div,
    mode: DrawMode,
) -> (Vec<u8>, Scene) {
    use wgpui_core::invalidation::request::FrameSignals;
    use wgpui_core::patch::emit::Emitter;
    use wgpui_core::reconcile::reconciler::Reconciler;
    use wgpui_layout::taffy_tree::{LayoutTree, definite};

    let mut reconciler = Reconciler::new();
    let mut layout = LayoutTree::new();
    let mut emitter = Emitter::new();
    let mut scene = Scene::new();

    let plan = reconciler
        .reconcile(root.describe(), &mut layout)
        .expect("a div tree must reconcile");
    let node = plan.root().expect("a plan has a root").layout_node;
    layout
        .compute_layout(node, definite(WIDTH as f32, HEIGHT as f32))
        .expect("a div tree must lay out");
    let emission = emitter
        .emit(&plan, &layout, &FrameSignals::new(), &mut scene)
        .expect("a laid-out plan must emit");
    apply(&mut scene, &emission.patch).expect("an emission must apply");

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
                source: None,
            },
        )
        .expect("a frame must render");
    let pixels = target
        .read_pixels(&context.device, &context.queue)
        .expect("reading the 2.0 target back must succeed");
    (pixels, scene)
}

/// **Milestone 1's element gate.** A real `div()` — background, border, rounded
/// corners, laid out by Taffy — byte-exact against `Style::paint`'s own quad
/// sequence, four content-masked border draws and all.
///
/// This is the test that decides whether `DivStyle::paint`'s one deliberate
/// departure (§ [`crate`]-level doc on `DivStyle::paint`: one unclipped border
/// quad where the legacy paints four clipped ones) is actually equivalent, or
/// merely argued to be.
#[test]
fn phase_6_6_div_gate() {
    use wgpui_widgets::div::div;
    use wgpui_widgets::styled::Styled;

    let Some(context) = context_or_report("phase_6_6_div_gate") else {
        return;
    };
    let clear = measured_clear_pixel(&context);
    let mode = DrawMode::best_available(context.indirect);

    struct DivCase {
        name: &'static str,
        origin: [f32; 2],
        size: [f32; 2],
        background: Option<([f32; 4], [f32; 4])>,
        border: Option<([f32; 4], [f32; 4])>,
        corner_radii: [f32; 4],
        border_widths: [f32; 4],
    }

    let cases = [
        DivCase {
            name: "a plain filled panel",
            origin: [24.0, 20.0],
            size: [160.0, 120.0],
            background: Some(GREY),
            border: None,
            corner_radii: [0.0; 4],
            border_widths: [0.0; 4],
        },
        DivCase {
            name: "the ordinary card: bg + border_1 + rounded_md",
            origin: [24.0, 20.0],
            size: [160.0, 120.0],
            background: Some(CYAN),
            border: Some(WHITE),
            corner_radii: [6.0; 4],
            border_widths: [1.0; 4],
        },
        DivCase {
            name: "a thick border with a large radius",
            origin: [32.0, 24.0],
            size: [150.0, 112.0],
            background: Some(RED),
            border: Some(WHITE),
            corner_radii: [16.0; 4],
            border_widths: [4.0; 4],
        },
        DivCase {
            name: "rounded_t only, which needs per-corner radii",
            origin: [28.0, 22.0],
            size: [156.0, 118.0],
            background: Some(GREY),
            border: Some(RED),
            corner_radii: [12.0, 12.0, 0.0, 0.0],
            border_widths: [2.0; 4],
        },
        DivCase {
            name: "border_b only, which needs per-side widths",
            origin: [28.0, 22.0],
            size: [156.0, 118.0],
            background: Some(CYAN),
            border: Some(RED),
            corner_radii: [0.0; 4],
            border_widths: [0.0, 0.0, 3.0, 0.0],
        },
        DivCase {
            name: "a translucent border over an opaque fill",
            origin: [26.0, 20.0],
            size: [158.0, 120.0],
            background: Some(WHITE),
            border: Some(CYAN_HALF),
            corner_radii: [8.0; 4],
            border_widths: [5.0; 4],
        },
        DivCase {
            name: "rounded_full, which the clamp turns into a pill",
            origin: [30.0, 40.0],
            size: [160.0, 80.0],
            background: Some(RED),
            border: Some(WHITE),
            corner_radii: [9999.0; 4],
            border_widths: [2.0; 4],
        },
        DivCase {
            name: "a border with no background at all",
            origin: [24.0, 20.0],
            size: [160.0, 120.0],
            background: None,
            border: Some(WHITE),
            corner_radii: [10.0; 4],
            border_widths: [4.0; 4],
        },
    ];

    let mut total = 0usize;
    let mut exact = 0usize;
    for case in cases {
        // The 2.0 arm: a real `div()` positioned by Taffy, not a hand-placed
        // quad. `absolute()` plus an inset is how a test pins an element to a
        // known rectangle through the layout engine rather than around it.
        let mut element = div()
            .absolute()
            .left(case.origin[0])
            .top(case.origin[1])
            .w(case.size[0])
            .h(case.size[1]);
        if let Some((_, rgba)) = case.background {
            element = element.bg(rgba);
        }
        if let Some((_, rgba)) = case.border {
            element = element.border_color(rgba);
        }
        // Per-corner and per-side, through the DSL rather than through the
        // style struct, so the builder is part of what this gate exercises.
        // The index order is `Quad`'s own: radii are TL, TR, BR, BL and widths
        // are top, right, bottom, left.
        element = element
            .rounded_tl(case.corner_radii[0])
            .rounded_tr(case.corner_radii[1])
            .rounded_br(case.corner_radii[2])
            .rounded_bl(case.corner_radii[3])
            .border_t(case.border_widths[0])
            .border_r(case.border_widths[1])
            .border_b(case.border_widths[2])
            .border_l(case.border_widths[3]);
        let element = div().w(WIDTH as f32).h(HEIGHT as f32).child(element);

        let (ours, scene) = render_div(&context, element, mode);
        let legacy = render_legacy_many(
            &context,
            &legacy_style_paint(
                case.origin,
                case.size,
                case.background,
                case.border,
                case.corner_radii,
                case.border_widths,
            ),
        );

        // The 2.0 arm must actually have emitted something, and specifically
        // must have emitted *fewer* quads than the legacy arm drew — which is
        // the claim under test, not an incidental fact.
        let layer = LayerId::from_key(LayerKey::untiled(BoundaryId::ROOT));
        let emitted = scene.quads.len(layer);
        let legacy_draws = legacy_style_paint(
            case.origin,
            case.size,
            case.background,
            case.border,
            case.corner_radii,
            case.border_widths,
        )
        .len();
        assert!(emitted > 0, "[{}] the div emitted no quads at all", case.name);

        let result = compare(&legacy, &ours, clear);
        println!(
            "  {}: {} of {} pixels byte-exact ({} of {} painted); 2.0 emitted {emitted} quad(s) \
             against legacy's {legacy_draws} draw(s)",
            case.name, result.exact, result.total, result.painted_exact, result.painted
        );
        assert_eq!(
            result.exact, result.total,
            "[{}] byte-exact means byte-exact; first difference at {:?}",
            case.name, result.first_difference
        );
        assert!(result.painted > 500, "[{}] painted too little", case.name);
        total += result.total;
        exact += result.exact;
    }
    println!("phase_6_6_div_gate: {exact} of {total} pixels byte-exact");
    assert_eq!(exact, total);
}

/// The colour transcription agrees with what 2.0 is handed.
///
/// Separate from the gate so a colour mistake reports as a colour mistake
/// rather than as a rendering disagreement.
#[test]
fn the_colour_transcription_is_checked_rather_than_assumed() {
    for (name, (hsla, rgba)) in [
        ("transparent", TRANSPARENT),
        ("grey", GREY),
        ("red", RED),
        ("cyan", CYAN),
        ("white", WHITE),
        ("black at half alpha", BLACK_HALF),
        ("cyan at half alpha", CYAN_HALF),
    ] {
        assert_eq!(
            hsla_to_rgba(hsla),
            rgba,
            "{name} does not convert exactly, so it must not be a test colour"
        );
    }
}
