// The instanced quad pipeline, pulling per-instance data out of a storage
// buffer through §5.3's indirection buffer.
//
// §1 of docs/gpu-native-architecture.md records that the legacy renderer's
// per-instance data is *already* storage-buffer vertex pulling rather than
// vertex-attribute instancing, and that this "is already the right shape for
// GPU-computed content." This shader is that shape with one addition: it reads
// `visible[slot_base + instance_index]` to find its arena slot instead of using
// `instance_index` directly, because culling removes an arbitrary subset and
// ordering permutes what is left, and neither is a contiguous range.
//
// # Where `slot_base` comes from, and why it is a uniform
//
// `wgpui_core::indirect`'s module doc has the full argument. In short: under
// `FirstInstance::Zero` the CPU supplies the slot's base through this uniform
// with a dynamic offset it already knows, and every argument record carries
// `first_instance: 0` — which is what makes the default path immune to
// README's "Custom Device Gotcha" and legal on WebGPU. Under
// `FirstInstance::SlotBase` the uniform is zero and `first_instance` carries
// the base instead, which is what a `multi_draw_indirect` needs. The expression
// below is the same either way, which is the point.
//
// # Phase 6.6: the fragment shader is now a port, not a simplification
//
// Phases 1–6.3 shipped a deliberately hard-edged rounded-rectangle SDF here,
// with a comment saying antialiasing was declined on purpose because "every
// comparison this shader takes part in is a bit-exact one between two draw
// paths." That reasoning was sound for those comparisons and wrong for this
// one: §8's Phase 6.6 gate is byte-exactness against the *legacy renderer*, and
// the legacy renderer antialiases. A hard-edged shader can never match it at a
// rounded corner, so the discard-based version could not have passed the gate
// under any amount of care on the emitting side.
//
// So `fragment_main` below is a transcription of `fs_quad` from
// `src/platform/cross/shaders/quads.wgsl`, expression for expression, for the
// case 2.0's `Quad` can express. `tests/legacy_quad_differential.rs` compiles
// that legacy file itself and compares pixel for pixel, so the transcription is
// checked rather than asserted.
//
// **What is transcribed and what is deliberately not**, named here rather than
// discovered later. Not transcribed, because `wgpui_core::patch::primitive::Quad`
// has no field for any of it:
//
// - Dashed borders (`border_style == 1`). The port takes the solid arm.
// - The per-fragment content mask. §5.2 sends the frame's clip to the occlusion
//   pass instead, so there is no `clip_distances` to test.
// - `hsla_to_rgba`. The legacy struct carries HSLA and converts in its vertex
//   shader; 2.0 carries straight RGBA and the conversion happens on the CPU,
//   before a colour ever reaches a slot.
// - `premultiplied_alpha`. 2.0 has no equivalent flag; `blend_color` here is
//   the legacy expression with `multiplier` fixed at 1.0, which is what the
//   legacy shader itself computes when the flag is 0.
//
// Transcribed in full: `pick_corner_radius`, `quad_sdf_impl`,
// `quarter_ellipse_sdf`, `over`, the `reduced_border` trick for zero-width
// sides, both background fast paths, and the final
// `mix`/`saturate(antialias_threshold - …)` composite. Those are the parts a
// rounded, bordered, antialiased box actually depends on.

struct Globals {
    // Framebuffer size in pixels.
    viewport: vec2<f32>,
    padding: vec2<f32>,
};

// Four scalars rather than `base: u32` plus a `vec3<u32>`: WGSL aligns a
// `vec3<u32>` to 16 bytes, so the obvious spelling is a 32-byte struct and the
// binding's `min_binding_size` then has to be 32 for a block that carries one
// useful word. Caught by wgpu's own pipeline validation, not by reading.
struct SlotBase {
    base: u32,
    padding_0: u32,
    translation: vec2<f32>,
};

// 144 bytes, matching `wgpui_core::patch::primitive::Quad::SLOT_STRIDE` and the
// field order `Quad::encode` writes.
struct QuadSlot {
    origin_size: vec4<f32>,
    background: vec4<f32>,
    border_color: vec4<f32>,
    // Corner radii in the legacy `Corners` order: top-left, top-right,
    // bottom-right, bottom-left.
    corner_radii: vec4<f32>,
    // Border widths in the legacy `Edges` order: top, right, bottom, left.
    border_widths: vec4<f32>,
    // kind, padding, padding, padding; first color; second color; parameters.
    material_kind: vec4<u32>,
    material_first: vec4<f32>,
    material_second: vec4<f32>,
    material_parameters: vec4<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> quads: array<QuadSlot>;
@group(0) @binding(2) var<storage, read> visible: array<u32>;
@group(1) @binding(0) var<uniform> slot: SlotBase;

// `wgpui_core::indirect::UNUSED_INSTANCE`.
const UNUSED_INSTANCE: u32 = 0xffffffffu;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) @interpolate(flat) arena_index: u32,
};

@vertex
fn vertex_main(
    @builtin(vertex_index) corner: u32,
    @builtin(instance_index) instance: u32,
) -> VertexOutput {
    var out: VertexOutput;
    let arena_index = visible[slot.base + instance];
    if (arena_index == UNUSED_INSTANCE) {
        // Every corner at one clip-space point, so the two triangles have zero
        // area and produce no fragments. Reachable only if an argument record
        // and the indirection buffer disagree, which is a bug — but a bug that
        // must not draw primitive 0 many times over.
        out.position = vec4<f32>(-2.0, -2.0, 0.0, 1.0);
        out.local = vec2<f32>(0.0, 0.0);
        out.arena_index = 0u;
        return out;
    }

    let quad = quads[arena_index];
    // Four corners of a triangle strip: (0,0) (1,0) (0,1) (1,1).
    let unit = vec2<f32>(f32(corner & 1u), f32((corner >> 1u) & 1u));
    let point = quad.origin_size.xy + unit * quad.origin_size.zw + slot.translation;
    out.position = vec4<f32>(
        point.x / globals.viewport.x * 2.0 - 1.0,
        1.0 - point.y / globals.viewport.y * 2.0,
        0.0,
        1.0,
    );
    out.local = unit * quad.origin_size.zw;
    out.arena_index = arena_index;
    return out;
}

// `pick_corner_radius`, transcribed. The quadrant test is `< 0.0` on both axes
// and the order of the four radii is the legacy `Corners` field order, so a
// point above-left of the centre takes `corner_radii.x`.
fn pick_corner_radius(center_to_point: vec2<f32>, radii: vec4<f32>) -> f32 {
    if center_to_point.x < 0.0 {
        if center_to_point.y < 0.0 {
            return radii.x;
        } else {
            return radii.w;
        }
    } else {
        if center_to_point.y < 0.0 {
            return radii.y;
        } else {
            return radii.z;
        }
    }
}

// `quad_sdf_impl`, transcribed.
fn quad_sdf_impl(corner_center_to_point: vec2<f32>, corner_radius: f32) -> f32 {
    if corner_radius == 0.0 {
        return max(corner_center_to_point.x, corner_center_to_point.y);
    } else {
        let signed_distance_to_inset_quad =
            length(max(vec2<f32>(0.0), corner_center_to_point)) +
            min(0.0, max(corner_center_to_point.x, corner_center_to_point.y));

        return signed_distance_to_inset_quad - corner_radius;
    }
}

// `quarter_ellipse_sdf`, transcribed.
fn quarter_ellipse_sdf(point: vec2<f32>, radii: vec2<f32>) -> f32 {
    let circle_vec = point / radii;
    let unit_circle_sdf = length(circle_vec) - 1.0;
    return unit_circle_sdf * (radii.x + radii.y) * -0.5;
}

// `over`, transcribed.
fn over(below: vec4<f32>, above: vec4<f32>) -> vec4<f32> {
    let alpha = above.a + below.a * (1.0 - above.a);
    let color = (above.rgb * above.a + below.rgb * below.a * (1.0 - above.a)) / alpha;
    return vec4<f32>(color, alpha);
}

// `blend_color`, transcribed with `premultiplied_alpha` fixed at 0 — see this
// file's header for why 2.0 has no such flag and why 0 is the right constant.
fn blend_color(color: vec4<f32>, alpha_factor: f32) -> vec4<f32> {
    let alpha = color.a * alpha_factor;
    return vec4<f32>(color.rgb, alpha);
}

fn material_color(quad: QuadSlot, local: vec2<f32>) -> vec4<f32> {
    let kind = quad.material_kind.x;
    if kind == 0u {
        return quad.background;
    }
    let unit = local / max(quad.origin_size.zw, vec2<f32>(0.000001));
    var amount = 0.0;
    if kind == 1u {
        amount = dot(unit - vec2<f32>(0.5), quad.material_parameters.xy) + 0.5;
    } else if kind == 2u {
        let delta = (unit - quad.material_parameters.xy) / max(quad.material_parameters.zw, vec2<f32>(0.000001));
        amount = length(delta);
    } else if kind == 3u {
        let interval = max(quad.material_parameters.y, 0.000001);
        let diagonal = (local.x + local.y) / interval;
        amount = select(0.0, 1.0, fract(diagonal) < quad.material_parameters.x / interval);
        return mix(quad.material_second, quad.material_first, amount);
    } else if kind == 4u {
        let cell = max(quad.material_parameters.x, 0.000001);
        let parity = (floor(local.x / cell) + floor(local.y / cell)) % 2.0;
        return mix(quad.material_second, quad.material_first, parity);
    } else if kind == 5u {
        let width = max(quad.material_parameters.x, 0.000001);
        return mix(quad.material_second, quad.material_first,
            select(0.0, 1.0, fract(local.y / width) < 0.5));
    }
    return mix(quad.material_first, quad.material_second, saturate(amount));
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let quad = quads[in.arena_index];

    let background_color = material_color(quad, in.local);

    let unrounded = quad.corner_radii.x == 0.0 &&
        quad.corner_radii.y == 0.0 &&
        quad.corner_radii.z == 0.0 &&
        quad.corner_radii.w == 0.0;

    // Fast path when the quad is not rounded and doesn't have any border.
    if quad.border_widths.x == 0.0 &&
            quad.border_widths.y == 0.0 &&
            quad.border_widths.z == 0.0 &&
            quad.border_widths.w == 0.0 &&
            unrounded {
        return blend_color(background_color, 1.0);
    }

    let size = quad.origin_size.zw;
    let half_size = size / 2.0;
    // `input.position.xy - quad.bounds.origin`, the legacy expression, and
    // deliberately **not** the interpolated `local` varying this shader still
    // carries for the unrounded fast path. The two are equal in exact
    // arithmetic and are not obliged to be equal in floating point: one is a
    // subtraction of two window coordinates, the other is a barycentric
    // interpolation of a per-vertex value. Byte-exactness against the legacy
    // renderer is decided at the antialiased corner, where a one-ULP difference
    // in `point` is a different coverage value and a different byte.
    let point = in.position.xy - quad.origin_size.xy;
    let center_to_point = point - half_size;

    let antialias_threshold = 0.5;

    let corner_radius = pick_corner_radius(center_to_point, quad.corner_radii);

    // Width of the nearest borders. `Edges` order is top, right, bottom, left.
    let border = vec2<f32>(
        select(
            quad.border_widths.y,
            quad.border_widths.w,
            center_to_point.x < 0.0
        ),
        select(
            quad.border_widths.z,
            quad.border_widths.x,
            center_to_point.y < 0.0
        )
    );

    // 0-width borders are reduced so that `inner_sdf >= antialias_threshold`.
    let reduced_border = vec2<f32>(select(border.x, -antialias_threshold, border.x == 0.0),
        select(border.y, -antialias_threshold, border.y == 0.0));

    let corner_to_point = abs(center_to_point) - half_size;
    let corner_center_to_point = corner_to_point + corner_radius;

    let is_near_rounded_corner = corner_center_to_point.x >= 0.0 &&
            corner_center_to_point.y >= 0.0;

    let straight_border_inner_corner_to_point = corner_to_point + reduced_border;

    let is_beyond_inner_straight_border = straight_border_inner_corner_to_point.x > 0.0 ||
            straight_border_inner_corner_to_point.y > 0.0;

    let is_within_inner_straight_border = straight_border_inner_corner_to_point.x < -antialias_threshold &&
        straight_border_inner_corner_to_point.y < -antialias_threshold;

    // Fast path for points that must be part of the background.
    if is_within_inner_straight_border && !is_near_rounded_corner {
        return blend_color(background_color, 1.0);
    }

    let outer_sdf = quad_sdf_impl(corner_center_to_point, corner_radius);

    var inner_sdf = 0.0;
    if corner_center_to_point.x <= 0.0 || corner_center_to_point.y <= 0.0 {
        inner_sdf = -max(straight_border_inner_corner_to_point.x,
            straight_border_inner_corner_to_point.y);
    } else if is_beyond_inner_straight_border {
        inner_sdf = -1.0;
    } else if reduced_border.x == reduced_border.y {
        inner_sdf = -(outer_sdf + reduced_border.x);
    } else {
        let ellipse_radii = max(vec2<f32>(0.0), corner_radius - reduced_border);
        inner_sdf = quarter_ellipse_sdf(corner_center_to_point, ellipse_radii);
    }

    let border_sdf = max(inner_sdf, outer_sdf);

    var color = background_color;
    if border_sdf < antialias_threshold {
        let border_color = quad.border_color;
        // Blend the border on top of the background and then linearly
        // interpolate between the two as we slide inside the background.
        let blended_border = over(background_color, border_color);
        color = mix(background_color, blended_border,
            saturate(antialias_threshold - inner_sdf));
    }

    return blend_color(color, saturate(antialias_threshold - outer_sdf));
}
