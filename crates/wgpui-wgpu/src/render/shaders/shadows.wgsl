// The instanced shadow pipeline: a blurred, rounded rectangle per instance,
// pulled out of a storage buffer through §5.3's indirection buffer.
//
// # What this file is, precisely
//
// The two-line placeholder that stood here claimed the real shader would be
// "moved as-is from src/platform/cross/shaders/shadows.wgsl in a later phase".
// It was a placeholder, not a port — the identical claim Phase 5.6 found false
// for `mono_sprites.wgsl`. So this file is written now, and it is a genuine
// port of the legacy shader's *mathematics* wrapped in `quads.wgsl`'s
// vertex-pulling and indirection shape. The two halves are worth telling apart:
//
// * **Transcribed, expression for expression, from the legacy file**:
//   `to_device_position_impl`, `gaussian`, `erf`, `blur_along_x`, the four-step
//   integration in the fragment entry point, and the `3.0 * blur_radius` vertex
//   margin. These are what "byte-exact against legacy output" is a claim about,
//   and they are deliberately not simplified, refactored, or "improved" —
//   `tests/legacy_shadow_differential.rs` compiles the legacy file itself and
//   compares, so any divergence here is a measured failure rather than a
//   stylistic difference.
// * **Replaced, because 2.0's protocol carries different data**: the per-instance
//   lookup goes through `visible[slot.base + instance]` rather than
//   `@builtin(instance_index)` (culling and ordering permute the arena, exactly
//   as for quads); the colour arrives as straight-alpha RGBA rather than as HSLA
//   converted in the vertex shader; and there is no `content_mask`/clip-distance
//   path at all, because 2.0 clips through the occlusion pass (§5.2) instead of
//   per fragment.
//
// # Why the vertex position keeps the legacy spelling
//
// `quads.wgsl` writes the same projection as `p / v * 2.0 - 1.0` /
// `1.0 - p / v * 2.0`. That is algebraically identical to the legacy
// `p / v * vec2(2.0, -2.0) + vec2(-1.0, 1.0)` and bit-identical in exact IEEE
// arithmetic — but the two spellings offer a compiler different fused
// multiply-add opportunities, and a single-ULP difference in a clip coordinate
// can move which side of a pixel centre an edge lands on. Since this shader
// exists to be compared against that exact expression, it keeps that exact
// expression.
//
// # Why the fragment re-reads the *unexpanded* rectangle
//
// The legacy vertex shader mutates a local copy of the shadow to grow its
// bounds by the blur margin; the fragment shader then reads `b_shadows[id]`
// again and gets the original, unexpanded rectangle back. Every distance the
// falloff is computed from is relative to that original rectangle's centre, and
// the margin exists only to give the tail somewhere to rasterise. Reproduced
// here, and stated because the alternative reading — that the fragment sees the
// grown rectangle — produces a plausible-looking shadow of the wrong size.

struct Globals {
    // Framebuffer size in pixels.
    viewport: vec2<f32>,
    padding: vec2<f32>,
};

// `quads.wgsl`'s `SlotBase`, unchanged and for the reason recorded there: a
// `vec3<u32>` tail would force a 32-byte `min_binding_size` for one useful word.
struct SlotBase {
    base: u32,
    padding_0: u32,
    translation: vec2<f32>,
    clip_origin: vec2<f32>,
    clip_size: vec2<f32>,
};

// 64 bytes, matching `wgpui_core::patch::primitive::Shadow::SLOT_STRIDE` and the
// field order `Shadow::encode` writes.
struct ShadowSlot {
    origin_size: vec4<f32>,
    color: vec4<f32>,
    // Corner radii in the legacy `Corners` order: top-left, top-right,
    // bottom-right, bottom-left.
    corner_radii: vec4<f32>,
    // blur_radius, and three words of padding.
    blur: vec4<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> shadows: array<ShadowSlot>;
@group(0) @binding(2) var<storage, read> visible: array<u32>;
@group(1) @binding(0) var<uniform> slot: SlotBase;

// `wgpui_core::indirect::UNUSED_INSTANCE`.
const UNUSED_INSTANCE: u32 = 0xffffffffu;
// The legacy file's own constant, to the same digits.
const M_PI_F: f32 = 3.1415926;
// `wgpui_core::patch::primitive::Shadow::BLUR_MARGIN_SIGMAS`. The Rust side
// grows the ordering rectangle by this; here it grows the drawn triangle strip.
// If the two disagree the falloff is clipped by a triangle edge.
const BLUR_MARGIN_SIGMAS: f32 = 3.0;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) color: vec4<f32>,
    @location(1) @interpolate(flat) arena_index: u32,
};

fn to_device_position_impl(position: vec2<f32>) -> vec4<f32> {
    let device_position = position / globals.viewport * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0);
    return vec4<f32>(device_position, 0.0, 1.0);
}

// `pick_corner_radius`, transcribed from the legacy shadow shader, which shares
// it verbatim with the legacy quad shader. The quadrant test is `< 0.0` on both
// axes and the four radii are in the legacy `Corners` field order.
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

fn gaussian(x: f32, sigma: f32) -> f32 {
    return exp(-(x * x) / (2.0 * sigma * sigma)) / (sqrt(2.0 * M_PI_F) * sigma);
}

fn erf(v: vec2<f32>) -> vec2<f32> {
    let s = sign(v);
    let a = abs(v);
    let r1 = 1.0 + (0.278393 + (0.230389 + (0.000972 + 0.078108 * a) * a) * a) * a;
    let r2 = r1 * r1;
    return s - s / (r2 * r2);
}

fn blur_along_x(x: f32, y: f32, sigma: f32, corner: f32, half_size: vec2<f32>) -> f32 {
    let delta = min(half_size.y - corner - abs(y), 0.0);
    let curved = half_size.x - corner + sqrt(max(0.0, corner * corner - delta * delta));
    let integral = 0.5 + 0.5 * erf((x + vec2<f32>(-curved, curved)) * (sqrt(0.5) / sigma));
    return integral.y - integral.x;
}

@vertex
fn vertex_main(
    @builtin(vertex_index) corner: u32,
    @builtin(instance_index) instance: u32,
) -> VertexOutput {
    var out: VertexOutput;
    let arena_index = visible[slot.base + instance];
    if (arena_index == UNUSED_INSTANCE) {
        // `quads.wgsl`'s degenerate strip, for the same reason: an argument
        // record and the indirection buffer disagreeing is a bug, but it must
        // not draw primitive 0 many times over.
        out.position = vec4<f32>(-2.0, -2.0, 0.0, 1.0);
        out.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.arena_index = 0u;
        return out;
    }

    let shadow = shadows[arena_index];
    // The legacy unit vertex is `vec2(f32(id & 1u), 0.5 * f32(id & 2u))`, which
    // produces (0,0) (1,0) (0,1) (1,1) — the same four corners as the spelling
    // below, which `quads.wgsl` already uses. Kept in `quads.wgsl`'s form
    // because the values are exactly equal (both are 0.0 or 1.0, with no
    // rounding to differ over) and one spelling across the crate is worth more
    // than matching the legacy file where it makes no difference.
    let unit = vec2<f32>(f32(corner & 1u), f32((corner >> 1u) & 1u));

    let margin = BLUR_MARGIN_SIGMAS * shadow.blur.x;
    let expanded_origin = shadow.origin_size.xy - vec2<f32>(margin);
    let expanded_size = shadow.origin_size.zw + 2.0 * vec2<f32>(margin);

    out.position = to_device_position_impl(unit * expanded_size + expanded_origin + slot.translation);
    out.color = shadow.color;
    out.arena_index = arena_index;
    return out;
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if slot.clip_size.x >= 0.0 {
        let clip_max = slot.clip_origin + slot.clip_size;
        if in.position.x < slot.clip_origin.x || in.position.y < slot.clip_origin.y
            || in.position.x >= clip_max.x || in.position.y >= clip_max.y {
            discard;
        }
    }
    // The unexpanded rectangle — see this file's header.
    let shadow = shadows[in.arena_index];
    let half_size = shadow.origin_size.zw / 2.0;
    let center = shadow.origin_size.xy + half_size + slot.translation;
    let center_to_point = in.position.xy - center;

    // `pick_corner_radius`, transcribed. Phase 6.3 collapsed this branch because
    // `Shadow` carried one uniform radius; Phase 6.6 widened the primitive to
    // four (a `rounded_t_md()` box casts a shadow with two round corners and two
    // square ones), so the branch is now the legacy one.
    //
    // Note what is still *not* here: unlike `quads.wgsl`, no
    // `min(radius, half_size)` clamp — the legacy shadow shader does not clamp
    // either, and adding one would change output for an over-large radius.
    let corner_radius = pick_corner_radius(center_to_point, shadow.corner_radii);
    let blur_radius = shadow.blur.x;

    let low = center_to_point.y - half_size.y;
    let high = center_to_point.y + half_size.y;
    let start = clamp(-BLUR_MARGIN_SIGMAS * blur_radius, low, high);
    let end = clamp(BLUR_MARGIN_SIGMAS * blur_radius, low, high);

    // Four samples, as the legacy shader takes. A zero `blur_radius` collapses
    // `step` to zero and makes `gaussian` divide by zero, so `alpha` comes out
    // NaN and the shadow draws nothing — a legacy behaviour, reproduced rather
    // than fixed, because parity is this phase's goal while both backends
    // coexist. `tests/legacy_shadow_differential.rs` pins it as a case both
    // sides agree on; docs/phase-6.3-results.md flags it for revisit.
    let step = (end - start) / 4.0;
    var y = start + step * 0.5;
    var alpha = 0.0;
    for (var i = 0; i < 4; i += 1) {
        let blur = blur_along_x(
            center_to_point.x,
            center_to_point.y - y,
            blur_radius,
            corner_radius,
            half_size,
        );
        alpha += blur * gaussian(y, blur_radius) * step;
        y += step;
    }

    // The legacy `blend_color` with `premultiplied_alpha == 0`, which is the
    // only mode 2.0 has: `TARGET_FORMAT`'s blend state is straight-alpha `over`
    // (`render/pipelines.rs`'s `ALPHA_OVER`), so the fragment emits straight
    // alpha and the multiplier is 1.0.
    return vec4<f32>(in.color.rgb, in.color.a * alpha);
}
