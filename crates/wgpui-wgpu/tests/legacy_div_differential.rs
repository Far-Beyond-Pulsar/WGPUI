//! **Phase 6.6's headline gate**: a real, non-trivial styled `div()` tree —
//! background, border, rounded corners, box shadow, nested children laid out by
//! Taffy — reconciles, emits, and renders byte-exact against the legacy
//! renderer for the same content.
//!
//! `legacy_quad_differential.rs` proves one quad and one childless `div()`.
//! This file is the whole thing: several elements, several kinds, real layout.
//!
//! # The oracle, and the one thing it must not be
//!
//! The legacy arm compiles **both frozen shader files themselves**
//! (`src/platform/cross/shaders/quads.wgsl` and `shadows.wgsl`) and replays
//! `Style::paint`'s own per-element sequence against them: every `box-shadow`
//! layer, then the background quad, then the four content-masked border draws.
//!
//! What it must not do, and does not: **derive its geometry from 2.0's output.**
//! Every rectangle the legacy arm draws is computed in this file from the flex
//! arithmetic the tree implies — border box, padding, gap, stretch — and
//! [`the_layout_oracle_matches_what_taffy_actually_computed`] then asserts that
//! Taffy agrees with it. If the oracle read positions out of the 2.0 scene, the
//! gate would prove the shaders agree and prove nothing at all about layout,
//! which is half of what this phase built.
//!
//! # The one divergence this file measures rather than argues
//!
//! `Style::paint` paints a parent's border **after** its children (the border is
//! painted by the continuation's caller, so the children have already gone in).
//! In 2.0 a parent's whole emission is appended before any child's, so a
//! parent's border lands *under* its children instead of over them.
//!
//! For any child that stays inside its parent's padding — every child of a
//! laid-out flex box that does not overflow — the two orders touch disjoint
//! pixels and produce identical output. The gate's trees are of that shape, and
//! [`an_overflowing_child_is_where_the_paint_order_difference_becomes_visible`]
//! builds one that is not, and measures the disagreement, so the limitation is a
//! number in a report rather than a sentence in a comment.

use wgpui_core::boundary::Pixels;
use wgpui_core::color::Hsla;
use wgpui_core::geometry::{point, Rect};
use wgpui_core::patch::apply::apply;
use wgpui_core::patch::primitive::{Quad, Shadow};
use wgpui_core::scene::Scene;
use wgpui_core::scene::layer::{BoundaryId, LayerId, LayerKey};
use wgpui_wgpu::render::device::{ComputeContext, context_or_report};
use wgpui_wgpu::render::draw::DrawMode;
use wgpui_wgpu::render::frame::{Dirty, FrameInput, FrameRenderer, OffscreenTarget, RenderTarget};
use wgpui_wgpu::render::pipelines::TARGET_FORMAT;
use wgpui_widgets::div::{Div, div};
use wgpui_widgets::styled::Styled;

const LEGACY_QUADS_WGSL: &str = include_str!("../../../old/src/platform/cross/shaders/quads.wgsl");
const LEGACY_SHADOWS_WGSL: &str = include_str!("../../../old/src/platform/cross/shaders/shadows.wgsl");

const WIDTH: u32 = 256;
const HEIGHT: u32 = 208;

/// `src/scene.rs:1446`.
const LEGACY_QUAD_STRIDE: usize = 168;
/// `src/scene.rs:1488`.
const LEGACY_SHADOW_STRIDE: usize = 72;

const LEGACY_BLEND: wgpu::BlendState = wgpu::BlendState::ALPHA_BLENDING;

const CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: 0.25,
    g: 0.5,
    b: 0.75,
    a: 1.0,
};

const UNCLIPPED: [f32; 4] = [-100_000.0, -100_000.0, 200_000.0, 200_000.0];

/// Colours restricted to the exactly-representable set the sibling
/// differentials explain, as `(hsla, rgba)` pairs.
const TRANSPARENT: ([f32; 4], [f32; 4]) = ([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]);
const GREY: ([f32; 4], [f32; 4]) = ([0.0, 0.0, 0.5, 1.0], [0.5, 0.5, 0.5, 1.0]);
const RED: ([f32; 4], [f32; 4]) = ([0.0, 1.0, 0.5, 1.0], [1.0, 0.0, 0.0, 1.0]);
const CYAN: ([f32; 4], [f32; 4]) = ([0.5, 1.0, 0.5, 1.0], [0.0, 1.0, 1.0, 1.0]);
const WHITE: ([f32; 4], [f32; 4]) = ([0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
const BLACK_QUARTER: ([f32; 4], [f32; 4]) = ([0.0, 0.0, 0.0, 0.25], [0.0, 0.0, 0.0, 0.25]);

/// The legacy `hsla_to_rgba`, shared verbatim by both frozen shaders.
fn hsla_to_rgba(hsla: [f32; 4]) -> [f32; 4] {
    let h = hsla[0] * 6.0;
    let (s, l, a) = (hsla[1], hsla[2], hsla[3]);
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

/// One quad the legacy arm draws, in both colour spaces plus its clip.
#[derive(Copy, Clone, Debug)]
struct LegacyQuad {
    quad: Quad,
    background_hsla: [f32; 4],
    border_hsla: [f32; 4],
    content_mask: [f32; 4],
}

/// One shadow the legacy arm draws.
#[derive(Copy, Clone, Debug)]
struct LegacyShadow {
    shadow: Shadow,
    hsla: [f32; 4],
}

/// One element's resolved geometry and style, as this file's own layout oracle
/// computes it.
///
/// Deliberately a flat description rather than a tree: the legacy arm needs the
/// *painted* sequence, and reconstructing that from a tree here would mean
/// reimplementing the walk 2.0 does, which is the thing under test.
#[derive(Clone, Debug)]
struct Painted {
    name: &'static str,
    origin: [f32; 2],
    size: [f32; 2],
    background: Option<([f32; 4], [f32; 4])>,
    border: Option<([f32; 4], [f32; 4])>,
    corner_radii: [f32; 4],
    border_widths: [f32; 4],
    shadows: Vec<ShadowLayer>,
}

/// One `box-shadow` layer as this file's oracle states it: the colour in both
/// spaces, plus the three geometry numbers `Window::paint_shadows` reads.
#[derive(Copy, Clone, Debug)]
struct ShadowLayer {
    color: ([f32; 4], [f32; 4]),
    offset: [f32; 2],
    blur_radius: f32,
    spread_radius: f32,
}

impl Painted {
    fn plain(name: &'static str, origin: [f32; 2], size: [f32; 2]) -> Painted {
        Painted {
            name,
            origin,
            size,
            background: None,
            border: None,
            corner_radii: [0.0; 4],
            border_widths: [0.0; 4],
            shadows: Vec::new(),
        }
    }

    /// The clamped radii `Style::paint` computes once and reuses for both the
    /// quads and every shadow layer.
    fn clamped_radii(&self) -> [f32; 4] {
        let max = self.size[0].min(self.size[1]) / 2.0;
        [
            self.corner_radii[0].min(max),
            self.corner_radii[1].min(max),
            self.corner_radii[2].min(max),
            self.corner_radii[3].min(max),
        ]
    }

    /// Every `Shadow` `Window::paint_shadows` would insert for this element.
    fn legacy_shadows(&self) -> Vec<LegacyShadow> {
        let radii = self.clamped_radii();
        self.shadows
            .iter()
            .map(|layer| LegacyShadow {
                shadow: Shadow {
                    origin: [
                        self.origin[0] + layer.offset[0] - layer.spread_radius,
                        self.origin[1] + layer.offset[1] - layer.spread_radius,
                    ],
                    size: [
                        self.size[0] + 2.0 * layer.spread_radius,
                        self.size[1] + 2.0 * layer.spread_radius,
                    ],
                    color: layer.color.1,
                    corner_radii: radii,
                    blur_radius: layer.blur_radius,
                },
                hsla: layer.color.0,
            })
            .collect()
    }

    /// Every `Quad` `Style::paint` would insert for this element — background
    /// first, then the border quad four times, each clipped to one edge band.
    fn legacy_quads(&self) -> Vec<LegacyQuad> {
        let radii = self.clamped_radii();
        let mut quads = Vec::new();

        if let Some((hsla, rgba)) = self.background.filter(|(_, rgba)| rgba[3] > 0.0) {
            let mut faded_hsla = hsla;
            faded_hsla[3] = 0.0;
            let mut faded_rgba = rgba;
            faded_rgba[3] = 0.0;
            quads.push(LegacyQuad {
                quad: Quad {
                    origin: self.origin,
                    size: self.size,
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

        let visible = self.border.is_some_and(|(_, rgba)| rgba[3] > 0.0)
            && self.border_widths.iter().any(|width| *width != 0.0);
        if !visible {
            return quads;
        }
        let (border_hsla, border_rgba) = self.border.unwrap_or(TRANSPARENT);
        let mut faded_hsla = border_hsla;
        faded_hsla[3] = 0.0;
        let mut faded_rgba = border_rgba;
        faded_rgba[3] = 0.0;

        let max_border_width = self.border_widths.iter().copied().fold(0.0, f32::max);
        let band = max_border_width.max(radii.iter().copied().fold(0.0, f32::max));
        let [min_x, min_y] = self.origin;
        let (max_x, max_y) = (min_x + self.size[0], min_y + self.size[1]);
        let masks = [
            [min_x, min_y, self.size[0], band],
            [
                max_x - max_border_width,
                min_y + band,
                max_border_width,
                self.size[1] - 2.0 * band,
            ],
            [min_x, max_y - band, self.size[0], band],
            [
                min_x,
                min_y + band,
                max_border_width,
                self.size[1] - 2.0 * band,
            ],
        ];
        for content_mask in masks {
            quads.push(LegacyQuad {
                quad: Quad {
                    origin: self.origin,
                    size: self.size,
                    background: faded_rgba,
                    border_color: border_rgba,
                    corner_radii: radii,
                    border_widths: self.border_widths,
                },
                background_hsla: faded_hsla,
                border_hsla,
                content_mask,
            });
        }
        quads
    }
}

fn encode_legacy_quad(entry: &LegacyQuad) -> [u8; LEGACY_QUAD_STRIDE] {
    let mut bytes = [0u8; LEGACY_QUAD_STRIDE];
    let mut put = |offset: usize, value: f32| {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    };
    // `order` at 0 and `border_style` at 4 stay zero (`BorderStyle::Solid`).
    put(8, entry.quad.origin[0]);
    put(12, entry.quad.origin[1]);
    put(16, entry.quad.size[0]);
    put(20, entry.quad.size[1]);
    for (index, value) in entry.content_mask.iter().enumerate() {
        put(24 + index * 4, *value);
    }
    // `background.tag` at 40 stays zero (`BackgroundTag::Solid`), as does
    // `color_space` at 44 and both gradient stops at 80..120.
    for (index, channel) in entry.background_hsla.iter().enumerate() {
        put(48 + index * 4, *channel);
    }
    for (index, channel) in entry.border_hsla.iter().enumerate() {
        put(120 + index * 4, *channel);
    }
    for (index, radius) in entry.quad.corner_radii.iter().enumerate() {
        put(136 + index * 4, *radius);
    }
    for (index, width) in entry.quad.border_widths.iter().enumerate() {
        put(152 + index * 4, *width);
    }
    bytes
}

fn encode_legacy_shadow(entry: &LegacyShadow) -> [u8; LEGACY_SHADOW_STRIDE] {
    let mut bytes = [0u8; LEGACY_SHADOW_STRIDE];
    let mut put = |offset: usize, value: f32| {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    };
    put(4, entry.shadow.blur_radius);
    put(8, entry.shadow.origin[0]);
    put(12, entry.shadow.origin[1]);
    put(16, entry.shadow.size[0]);
    put(20, entry.shadow.size[1]);
    for (index, radius) in entry.shadow.corner_radii.iter().enumerate() {
        put(24 + index * 4, *radius);
    }
    for (index, value) in UNCLIPPED.iter().enumerate() {
        put(40 + index * 4, *value);
    }
    for (index, channel) in entry.hsla.iter().enumerate() {
        put(56 + index * 4, *channel);
    }
    bytes
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
        size: bytes.len().max(16) as u64,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytes);
    buffer
}

fn frame_layouts(device: &wgpu::Device) -> (wgpu::BindGroupLayout, wgpu::BindGroupLayout) {
    let globals = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
    let storage = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("legacy primitive storage"),
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
    (globals, storage)
}

fn legacy_pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    vertex_entry: &str,
    fragment_entry: &str,
    layout: &wgpu::PipelineLayout,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(vertex_entry),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some(vertex_entry),
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
            module,
            entry_point: Some(fragment_entry),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: TARGET_FORMAT,
                blend: Some(LEGACY_BLEND),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: Default::default(),
        cache: None,
    })
}

/// Render one frame through the two frozen legacy shaders: every shadow first,
/// then every quad, which is the order both renderers batch in.
fn render_legacy(context: &ComputeContext, elements: &[Painted]) -> Vec<u8> {
    let device = &context.device;
    let queue = &context.queue;

    let shadows: Vec<LegacyShadow> = elements.iter().flat_map(Painted::legacy_shadows).collect();
    let quads: Vec<LegacyQuad> = elements.iter().flat_map(Painted::legacy_quads).collect();

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

    let mut shadow_bytes = Vec::new();
    for entry in &shadows {
        shadow_bytes.extend_from_slice(&encode_legacy_shadow(entry));
    }
    let mut quad_bytes = Vec::new();
    for entry in &quads {
        quad_bytes.extend_from_slice(&encode_legacy_quad(entry));
    }
    let shadow_buffer = buffer_with(
        device,
        queue,
        "legacy shadows",
        wgpu::BufferUsages::STORAGE,
        &shadow_bytes,
    );
    let quad_buffer = buffer_with(
        device,
        queue,
        "legacy quads",
        wgpu::BufferUsages::STORAGE,
        &quad_bytes,
    );

    let (globals_layout, storage_layout) = frame_layouts(device);
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("legacy"),
        bind_group_layouts: &[Some(&globals_layout), Some(&storage_layout)],
        immediate_size: 0,
    });
    let quad_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("legacy quads"),
        source: wgpu::ShaderSource::Wgsl(LEGACY_QUADS_WGSL.into()),
    });
    let shadow_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("legacy shadows"),
        source: wgpu::ShaderSource::Wgsl(LEGACY_SHADOWS_WGSL.into()),
    });
    let quad_pipeline =
        legacy_pipeline(device, &quad_module, "vs_quad", "fs_quad", &pipeline_layout);
    let shadow_pipeline = legacy_pipeline(
        device,
        &shadow_module,
        "vs_shadow",
        "fs_shadow",
        &pipeline_layout,
    );

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
        layout: &storage_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: shadow_buffer.as_entire_binding(),
        }],
    });
    let quad_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("legacy quad storage"),
        layout: &storage_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: quad_buffer.as_entire_binding(),
        }],
    });

    let target = OffscreenTarget::new(device, WIDTH, HEIGHT);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("legacy div frame"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("legacy div frame"),
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
        pass.set_bind_group(0, &globals_group, &[]);
        if !shadows.is_empty() {
            pass.set_pipeline(&shadow_pipeline);
            pass.set_bind_group(1, &shadow_group, &[]);
            for index in 0..shadows.len() as u32 {
                pass.draw(0..4, index..index + 1);
            }
        }
        if !quads.is_empty() {
            pass.set_pipeline(&quad_pipeline);
            pass.set_bind_group(1, &quad_group, &[]);
            for index in 0..quads.len() as u32 {
                pass.draw(0..4, index..index + 1);
            }
        }
    }
    queue.submit(Some(encoder.finish()));
    target
        .read_pixels(device, queue)
        .expect("reading the legacy target back must succeed")
}

/// What 2.0's element path produced for one frame.
struct Rendered {
    pixels: Vec<u8>,
    quads: Vec<Quad>,
    shadows: Vec<Shadow>,
}

/// Render a `div()` tree through 2.0's real element path.
///
/// Reconcile → Taffy → emit → patch → apply → compute ordering/occlusion →
/// indirect draw. Nothing about the geometry is supplied by this function.
fn render_div(context: &ComputeContext, root: Div, mode: DrawMode) -> Rendered {
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

    let layer = LayerId::from_key(LayerKey::untiled(BoundaryId::ROOT));
    let quads = scene
        .quads
        .keys(layer)
        .into_iter()
        .filter_map(|key| scene.quads.get(layer, key).copied())
        .collect();
    let shadows = scene
        .shadows
        .keys(layer)
        .into_iter()
        .filter_map(|key| scene.shadows.get(layer, key).copied())
        .collect();

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
    Rendered {
        pixels: target
            .read_pixels(&context.device, &context.queue)
            .expect("reading the 2.0 target back must succeed"),
        quads,
        shadows,
    }
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
    assert_eq!(legacy.len(), ours.len());
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
    let ours = render_div(
        context,
        div().w(WIDTH as f32).h(HEIGHT as f32),
        DrawMode::best_available(context.indirect),
    );
    let legacy = render_legacy(context, &[]);
    let ours: [u8; 4] = [
        ours.pixels[0],
        ours.pixels[1],
        ours.pixels[2],
        ours.pixels[3],
    ];
    let legacy: [u8; 4] = [legacy[0], legacy[1], legacy[2], legacy[3]];
    assert_eq!(
        ours, legacy,
        "the two arms must clear to the same bytes, or every later comparison \
         is between two different backgrounds"
    );
    ours
}

// ---------------------------------------------------------------------------
// The tree under test, and this file's independent layout oracle for it.
// ---------------------------------------------------------------------------

/// The card's own geometry, chosen so nothing lands on a viewport edge.
const CARD_ORIGIN: [f32; 2] = [24.0, 20.0];
const CARD_SIZE: [f32; 2] = [208.0, 168.0];
const CARD_BORDER: f32 = 2.0;
const CARD_PADDING: f32 = 12.0;
const CARD_RADIUS: f32 = 10.0;
const ROW_GAP: f32 = 8.0;
const ROW_HEIGHT: f32 = 28.0;
const ROW_RADIUS: f32 = 4.0;
const ROW_COUNT: usize = 3;
/// A pill sitting inside each row, so the tree is three levels deep rather than
/// two — §8's Phase 1 gate uses "a plain three-level-deep div" as its own
/// standard for a non-trivial tree, and this gate should not be shallower.
const PILL_INSET: f32 = 4.0;
const PILL_SIZE: [f32; 2] = [40.0, 12.0];

/// The tree: a shadowed, bordered, rounded card holding three rows, each
/// holding a pill.
fn card_tree() -> Div {
    let rows = (0..ROW_COUNT).map(|index| {
        div()
            .h(ROW_HEIGHT)
            .flex_shrink_0()
            .p(PILL_INSET)
            .bg(row_color(index).1)
            .rounded(ROW_RADIUS)
            .child(
                div()
                    .w(PILL_SIZE[0])
                    .h(PILL_SIZE[1])
                    .flex_shrink_0()
                    .bg(WHITE.1)
                    .rounded_full(),
            )
    });
    div().w(WIDTH as f32).h(HEIGHT as f32).child(
        div()
            .absolute()
            .left(CARD_ORIGIN[0])
            .top(CARD_ORIGIN[1])
            .w(CARD_SIZE[0])
            .h(CARD_SIZE[1])
            .flex_col()
            .p(CARD_PADDING)
            .gap(ROW_GAP)
            .bg(GREY.1)
            .border_color(WHITE.1)
            .border(CARD_BORDER)
            .rounded(CARD_RADIUS)
            .shadow(vec![wgpui_widgets::div::interactivity::style::BoxShadow {
                color: Hsla { h: 0.0, s: 0.0, l: 0.0, a: 0.25 },
                offset: point(Pixels(0.0), Pixels(6.0)),
                blur_radius: Pixels(10.0),
                spread_radius: Pixels(-2.0),
            }])
            .children(rows),
    )
}

fn row_color(index: usize) -> ([f32; 4], [f32; 4]) {
    match index % 3 {
        0 => RED,
        1 => CYAN,
        _ => GREY,
    }
}

/// This file's own layout arithmetic for [`card_tree`].
///
/// Flexbox, done by hand: the card's content box is inset by its border and its
/// padding; rows stack down that box separated by the gap and stretched to its
/// width; each pill sits at its row's own content origin, which is the row's
/// padding in from its corner. Nothing here reads anything Taffy produced —
/// [`the_layout_oracle_matches_what_taffy_actually_computed`] is what confronts
/// the two.
fn card_tree_oracle() -> Vec<Painted> {
    let inset = CARD_BORDER + CARD_PADDING;
    let content_x = CARD_ORIGIN[0] + inset;
    let content_y = CARD_ORIGIN[1] + inset;
    let content_width = CARD_SIZE[0] - 2.0 * inset;

    let mut painted = vec![Painted {
        name: "card",
        origin: CARD_ORIGIN,
        size: CARD_SIZE,
        background: Some(GREY),
        border: Some(WHITE),
        corner_radii: [CARD_RADIUS; 4],
        border_widths: [CARD_BORDER; 4],
        shadows: vec![ShadowLayer {
            color: BLACK_QUARTER,
            offset: [0.0, 6.0],
            blur_radius: 10.0,
            spread_radius: -2.0,
        }],
    }];

    for index in 0..ROW_COUNT {
        let row_y = content_y + index as f32 * (ROW_HEIGHT + ROW_GAP);
        painted.push(Painted {
            background: Some(row_color(index)),
            corner_radii: [ROW_RADIUS; 4],
            ..Painted::plain("row", [content_x, row_y], [content_width, ROW_HEIGHT])
        });
        painted.push(Painted {
            background: Some(WHITE),
            // `rounded_full()` is 9999px, clamped to half the shorter side.
            corner_radii: [PILL_SIZE[1] / 2.0; 4],
            ..Painted::plain(
                "pill",
                [content_x + PILL_INSET, row_y + PILL_INSET],
                PILL_SIZE,
            )
        });
    }
    painted
}

/// **The Phase 6.6 gate.** The whole tree, byte-exact.
#[test]
fn phase_6_6_div_tree_gate() {
    let Some(context) = context_or_report("phase_6_6_div_tree_gate") else {
        return;
    };
    println!(
        "adapter: {:?} {:?} driver {:?}",
        context.adapter_info.name, context.adapter_info.backend, context.adapter_info.driver_info
    );
    let clear = measured_clear_pixel(&context);
    let mode = DrawMode::best_available(context.indirect);

    let oracle = card_tree_oracle();
    for element in &oracle {
        if let Some((hsla, rgba)) = element.background {
            assert_eq!(
                hsla_to_rgba(hsla),
                rgba,
                "[{}] the two arms must be given the same background colour",
                element.name
            );
        }
        for layer in &element.shadows {
            assert_eq!(hsla_to_rgba(layer.color.0), layer.color.1);
        }
    }

    let legacy = render_legacy(&context, &oracle);
    let ours = render_div(&context, card_tree(), mode);

    // Vacuity guards, before the comparison: this has to be a real tree with
    // real content on both sides, not two blank frames agreeing.
    assert_eq!(
        ours.quads.len(),
        1 + 1 + ROW_COUNT * 2,
        "card background + card border + one background per row and per pill"
    );
    assert_eq!(ours.shadows.len(), 1);
    let legacy_draws: usize = oracle
        .iter()
        .map(|element| element.legacy_quads().len() + element.legacy_shadows().len())
        .sum();

    let result = compare(&legacy, &ours.pixels, clear);
    println!(
        "phase_6_6_div_tree_gate: {} of {} pixels byte-exact ({} of {} painted); \
         2.0 emitted {} quads and {} shadows against legacy's {legacy_draws} draws",
        result.exact,
        result.total,
        result.painted_exact,
        result.painted,
        ours.quads.len(),
        ours.shadows.len()
    );
    assert!(
        result.painted > 20_000,
        "only {} pixels were painted, which is too few for a whole-tree \
         agreement to mean anything",
        result.painted
    );
    assert_eq!(
        result.exact, result.total,
        "byte-exact means byte-exact; first difference at {:?}",
        result.first_difference
    );
}

/// The layout oracle above is confronted with what Taffy actually computed.
///
/// Without this the gate could pass while both arms drew the same wrong
/// rectangles, because the 2.0 arm is the only one that runs a layout engine.
/// Rectangles are read out of the emitted primitives rather than out of the
/// layout tree, so the check covers the emit walk's ancestor-origin
/// accumulation too, not only Taffy's answer.
#[test]
fn the_layout_oracle_matches_what_taffy_actually_computed() {
    let Some(context) = context_or_report("div_tree_layout_oracle") else {
        return;
    };
    let ours = render_div(
        &context,
        card_tree(),
        DrawMode::best_available(context.indirect),
    );

    // Emission order within a layer is append order, and the walk is
    // depth-first, so: card background, card border, then per row its own
    // background followed by its pill's.
    let mut expected: Vec<([f32; 2], [f32; 2])> = Vec::new();
    let oracle = card_tree_oracle();
    for element in &oracle {
        for _ in 0..element.legacy_quads().len().min(2) {
            expected.push((element.origin, element.size));
        }
    }
    // The card contributes two quads (background and border) at the same
    // rectangle; every other element contributes one.
    let actual: Vec<([f32; 2], [f32; 2])> = ours
        .quads
        .iter()
        .map(|quad| (quad.origin, quad.size))
        .collect();
    assert_eq!(
        actual, expected,
        "Taffy's placement and this file's flex arithmetic must agree, or the \
         gate is comparing two arms that are both wrong in the same way"
    );

    let shadow = ours.shadows.first().copied().expect("the card casts one");
    assert_eq!(
        (shadow.origin, shadow.size),
        (
            [CARD_ORIGIN[0] + 2.0, CARD_ORIGIN[1] + 6.0 + 2.0],
            [CARD_SIZE[0] - 4.0, CARD_SIZE[1] - 4.0]
        ),
        "offset by (0,6) then shrunk by the -2 spread on every side"
    );
}

/// The gate has been watched failing, in the two ways a *tree* can fail that a
/// single primitive cannot.
///
/// A shader-level perturbation is already covered by
/// `legacy_quad_differential.rs`. What this file has to show it can catch is a
/// wrong *tree*: a child in the wrong place, and a missing element.
#[test]
fn the_comparison_detects_a_misplaced_child_and_a_missing_one() {
    let Some(context) = context_or_report("div_tree_detects_a_difference") else {
        return;
    };
    let clear = measured_clear_pixel(&context);
    let mode = DrawMode::best_available(context.indirect);
    let legacy = render_legacy(&context, &card_tree_oracle());

    let control = compare(
        &legacy,
        &render_div(&context, card_tree(), mode).pixels,
        clear,
    );
    assert_eq!(control.exact, control.total, "the control must agree");

    // 1. One pixel of extra gap: every row below the first moves down by one,
    //    which no amount of shader correctness can absorb.
    let nudged = {
        let rows = (0..ROW_COUNT).map(|index| {
            div()
                .h(ROW_HEIGHT)
                .flex_shrink_0()
                .p(PILL_INSET)
                .bg(row_color(index).1)
                .rounded(ROW_RADIUS)
                .child(
                    div()
                        .w(PILL_SIZE[0])
                        .h(PILL_SIZE[1])
                        .flex_shrink_0()
                        .bg(WHITE.1)
                        .rounded_full(),
                )
        });
        div().w(WIDTH as f32).h(HEIGHT as f32).child(
            div()
                .absolute()
                .left(CARD_ORIGIN[0])
                .top(CARD_ORIGIN[1])
                .w(CARD_SIZE[0])
                .h(CARD_SIZE[1])
                .flex_col()
                .p(CARD_PADDING)
                .gap(ROW_GAP + 1.0)
                .bg(GREY.1)
                .border_color(WHITE.1)
                .border(CARD_BORDER)
                .rounded(CARD_RADIUS)
                .shadow(vec![wgpui_widgets::div::interactivity::style::BoxShadow {
                    color: Hsla { h: 0.0, s: 0.0, l: 0.0, a: 0.25 },
                    offset: point(Pixels(0.0), Pixels(6.0)),
                    blur_radius: Pixels(10.0),
                    spread_radius: Pixels(-2.0),
                }])
                .children(rows),
        )
    };
    let moved = compare(&legacy, &render_div(&context, nudged, mode).pixels, clear);
    assert!(
        moved.exact < moved.total,
        "a one-pixel gap change must be visible; it reported {} of {} exact",
        moved.exact,
        moved.total
    );
    println!(
        "a one-pixel gap change disagrees at {} of {} pixels",
        moved.total - moved.exact,
        moved.total
    );

    // 2. The shadow removed entirely.
    let unshadowed = {
        let tree = card_tree();
        // Rebuilt rather than mutated, because `Div` is a builder and its style
        // is not reachable after `describe`. `shadow_none()` is the DSL's own
        // way to say this, so the perturbation goes through the public surface.
        drop(tree);
        let rows = (0..ROW_COUNT).map(|index| {
            div()
                .h(ROW_HEIGHT)
                .flex_shrink_0()
                .p(PILL_INSET)
                .bg(row_color(index).1)
                .rounded(ROW_RADIUS)
                .child(
                    div()
                        .w(PILL_SIZE[0])
                        .h(PILL_SIZE[1])
                        .flex_shrink_0()
                        .bg(WHITE.1)
                        .rounded_full(),
                )
        });
        div().w(WIDTH as f32).h(HEIGHT as f32).child(
            div()
                .absolute()
                .left(CARD_ORIGIN[0])
                .top(CARD_ORIGIN[1])
                .w(CARD_SIZE[0])
                .h(CARD_SIZE[1])
                .flex_col()
                .p(CARD_PADDING)
                .gap(ROW_GAP)
                .bg(GREY.1)
                .border_color(WHITE.1)
                .border(CARD_BORDER)
                .rounded(CARD_RADIUS)
                .shadow_none()
                .children(rows),
        )
    };
    let flat = compare(
        &legacy,
        &render_div(&context, unshadowed, mode).pixels,
        clear,
    );
    assert!(
        flat.exact < flat.total,
        "removing the box-shadow must be visible; it reported {} of {} exact",
        flat.exact,
        flat.total
    );
    println!(
        "removing the shadow disagrees at {} of {} pixels",
        flat.total - flat.exact,
        flat.total
    );
}

/// Every draw mode this adapter offers renders the whole tree identically.
#[test]
fn every_draw_mode_renders_the_same_tree() {
    let Some(context) = context_or_report("div_tree_draw_modes") else {
        return;
    };
    let clear = measured_clear_pixel(&context);
    let legacy = render_legacy(&context, &card_tree_oracle());
    let mut modes = 0;
    for mode in DrawMode::ALL {
        if !mode.is_available(context.indirect) {
            continue;
        }
        modes += 1;
        let result = compare(
            &legacy,
            &render_div(&context, card_tree(), mode).pixels,
            clear,
        );
        assert_eq!(
            result.exact, result.total,
            "{mode:?} disagrees with legacy at {:?}",
            result.first_difference
        );
    }
    assert!(modes >= 2, "only {modes} draw mode(s) were exercised");
    println!("every_draw_mode_renders_the_same_tree: {modes} modes, all byte-exact");
}

/// **The disclosed divergence, measured.**
///
/// `Style::paint` paints a parent's border after its children; 2.0 appends a
/// parent's whole emission before any child's. This test builds the one shape
/// where that is visible — an absolutely-positioned child overlapping its
/// parent's border band — and reports the size of the disagreement, so
/// `docs/phase-6.6-results.md` carries a number rather than a caveat.
///
/// It asserts the disagreement exists rather than asserting it does not. A
/// future phase that fixes the ordering should watch this test fail and then
/// invert it.
#[test]
fn an_overflowing_child_is_where_the_paint_order_difference_becomes_visible() {
    let Some(context) = context_or_report("div_tree_paint_order") else {
        return;
    };
    let clear = measured_clear_pixel(&context);
    let mode = DrawMode::best_available(context.indirect);

    let parent_origin = [40.0, 32.0];
    let parent_size = [160.0, 120.0];
    let border = 12.0;
    // Absolutely positioned at the parent's own origin, so it sits squarely on
    // top of the whole left and top border band.
    let child_origin = parent_origin;
    let child_size = [60.0, 60.0];

    let tree = div().w(WIDTH as f32).h(HEIGHT as f32).child(
        div()
            .absolute()
            .left(parent_origin[0])
            .top(parent_origin[1])
            .w(parent_size[0])
            .h(parent_size[1])
            .bg(GREY.1)
            .border_color(RED.1)
            .border(border)
            .child(
                div()
                    .absolute()
                    .left(-border)
                    .top(-border)
                    .w(child_size[0])
                    .h(child_size[1])
                    .flex_shrink_0()
                    .bg(CYAN.1),
            ),
    );

    // The legacy order: parent background, child background, parent border.
    let legacy_order = [
        Painted {
            background: Some(GREY),
            ..Painted::plain("parent background", parent_origin, parent_size)
        },
        Painted {
            background: Some(CYAN),
            ..Painted::plain("child", child_origin, child_size)
        },
        Painted {
            border: Some(RED),
            border_widths: [border; 4],
            ..Painted::plain("parent border", parent_origin, parent_size)
        },
    ];
    let legacy = render_legacy(&context, &legacy_order);
    let ours = render_div(&context, tree, mode);
    let result = compare(&legacy, &ours.pixels, clear);

    println!(
        "an overflowing child disagrees at {} of {} pixels ({} painted); \
         2.0 draws the parent's border under its children, legacy draws it over",
        result.total - result.exact,
        result.total,
        result.painted
    );
    assert!(
        result.exact < result.total,
        "this test exists to measure a known divergence; if it now agrees, the \
         paint-order gap has been closed and this test should be inverted"
    );

    // And the same tree with the child moved fully inside the padding agrees
    // exactly — which is what makes the divergence a *scoped* one rather than a
    // general disagreement about children.
    let inside_origin = [
        parent_origin[0] + border + 4.0,
        parent_origin[1] + border + 4.0,
    ];
    let inside_tree = div().w(WIDTH as f32).h(HEIGHT as f32).child(
        div()
            .absolute()
            .left(parent_origin[0])
            .top(parent_origin[1])
            .w(parent_size[0])
            .h(parent_size[1])
            .bg(GREY.1)
            .border_color(RED.1)
            .border(border)
            .child(
                div()
                    .absolute()
                    .left(4.0)
                    .top(4.0)
                    .w(child_size[0])
                    .h(child_size[1])
                    .flex_shrink_0()
                    .bg(CYAN.1),
            ),
    );
    let inside_legacy = render_legacy(
        &context,
        &[
            Painted {
                background: Some(GREY),
                ..Painted::plain("parent background", parent_origin, parent_size)
            },
            Painted {
                background: Some(CYAN),
                ..Painted::plain("child", inside_origin, child_size)
            },
            Painted {
                border: Some(RED),
                border_widths: [border; 4],
                ..Painted::plain("parent border", parent_origin, parent_size)
            },
        ],
    );
    let inside = compare(
        &inside_legacy,
        &render_div(&context, inside_tree, mode).pixels,
        clear,
    );
    assert_eq!(
        inside.exact, inside.total,
        "a child that stays inside its parent's border is unaffected by the \
         ordering difference; first difference at {:?}",
        inside.first_difference
    );
}

/// The frozen legacy shaders still have the shape this differential relies on.
#[test]
fn the_legacy_sources_still_have_the_shape_this_test_relies_on() {
    for (name, source) in [
        ("quads", LEGACY_QUADS_WGSL),
        ("shadows", LEGACY_SHADOWS_WGSL),
    ] {
        assert!(
            source.contains("fn to_device_position_impl(position: vec2<f32>)"),
            "{name}: the per-layer translate hook `slab_shader_source` rewrites is gone"
        );
        assert!(
            source.contains("fn distance_from_clip_rect_impl(position: vec2<f32>"),
            "{name}: the clip-distance hook `slab_shader_source` rewrites is gone"
        );
    }
    assert!(LEGACY_QUADS_WGSL.contains("fn fs_quad("));
    assert!(LEGACY_SHADOWS_WGSL.contains("fn fs_shadow("));
}
