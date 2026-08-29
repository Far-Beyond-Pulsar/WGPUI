// The instanced underline pipeline: a straight or wavy rule per instance,
// pulled out of a storage buffer through §5.3's indirection buffer.
//
// # What this file is, precisely
//
// The two-line placeholder that stood here claimed the real shader would be
// "moved as-is from src/platform/cross/shaders/underlines.wgsl in a later
// phase". It was a placeholder, not a port — the same false claim Phase 5.6
// found on `mono_sprites.wgsl` and Phase 6.3 found on `shadows.wgsl`. This is
// `shadows.wgsl`'s structure exactly, one shader over: the legacy file's
// mathematics transcribed, wrapped in `quads.wgsl`'s vertex-pulling shape,
// with the same three substitutions (indirection lookup, straight RGBA instead
// of HSLA, no per-fragment clip).
//
// # Two legacy behaviours reproduced rather than corrected
//
// 1. **The alpha is squared.** `fs_underline` computes
//    `blend_color(input.color, input.color.a)`, and `blend_color`'s own body is
//    `alpha = color.a * alpha_factor` — so the factor *is* the colour's alpha
//    and the result is `a²`. A 50%-alpha underline therefore paints at 25%. The
//    wavy branch does the same thing one step further along
//    (`alpha * input.color.a`, again multiplied by `color.a`). Fully opaque
//    underlines are unaffected, which is presumably why this has survived, and
//    fully opaque is what almost every real underline is.
// 2. **The wavy flag is masked to its low byte** (`wavy & 0xFFu`). The legacy
//    struct's field is a `u32` fed from a `bool`, so the mask never changes an
//    answer; it is kept because removing it would be a change to the file this
//    shader is compared against, and this phase's gate is exactness rather than
//    tidiness.
//
// Both are ported deliberately, on Phase 5.5's stated precedent for the 2×
// sub-pixel aliasing quirk: reproducing legacy output exactly is the goal while
// both backends coexist. docs/phase-6.3-results.md flags the first for revisit
// once that reason stops applying.
//
// # Why the vertex position keeps the legacy spelling
//
// `shadows.wgsl`'s header has the argument in full: the two spellings are
// algebraically identical and offer a compiler different fused multiply-add
// opportunities, and a single-ULP difference in a clip coordinate can move
// which side of a pixel centre an edge lands on.

struct Globals {
    // Framebuffer size in pixels.
    viewport: vec2<f32>,
    padding: vec2<f32>,
};

struct SlotBase {
    base: u32,
    padding_0: u32,
    padding_1: u32,
    padding_2: u32,
};

// 48 bytes, matching `wgpui_core::patch::primitive::Underline::SLOT_STRIDE` and
// the field order `Underline::encode` writes.
struct UnderlineSlot {
    origin_size: vec4<f32>,
    color: vec4<f32>,
    thickness: f32,
    // Declared `u32` and read as one, rather than packed into a trailing
    // `vec4<f32>` and bit-cast. `Underline::encode` writes the boolean as the
    // word `1`, whose bit pattern read as `f32` is a denormal — and a GPU is
    // free to flush denormals to zero on load, which would turn every wavy
    // underline straight on exactly the hardware that does. `mono_sprites.wgsl`
    // declares `atlas_tile` the same way for the same class of reason.
    wavy: u32,
    padding_0: u32,
    padding_1: u32,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> underlines: array<UnderlineSlot>;
@group(0) @binding(2) var<storage, read> visible: array<u32>;
@group(1) @binding(0) var<uniform> slot: SlotBase;

// `wgpui_core::indirect::UNUSED_INSTANCE`.
const UNUSED_INSTANCE: u32 = 0xffffffffu;
// The legacy file's own constants, to the same digits.
const M_PI_F: f32 = 3.1415926;
const WAVE_FREQUENCY: f32 = 2.0;
const WAVE_HEIGHT_RATIO: f32 = 0.8;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) color: vec4<f32>,
    @location(1) @interpolate(flat) arena_index: u32,
};

fn to_device_position_impl(position: vec2<f32>) -> vec4<f32> {
    let device_position = position / globals.viewport * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0);
    return vec4<f32>(device_position, 0.0, 1.0);
}

@vertex
fn vertex_main(
    @builtin(vertex_index) corner: u32,
    @builtin(instance_index) instance: u32,
) -> VertexOutput {
    var out: VertexOutput;
    let arena_index = visible[slot.base + instance];
    if (arena_index == UNUSED_INSTANCE) {
        out.position = vec4<f32>(-2.0, -2.0, 0.0, 1.0);
        out.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.arena_index = 0u;
        return out;
    }

    let underline = underlines[arena_index];
    let unit = vec2<f32>(f32(corner & 1u), f32((corner >> 1u) & 1u));
    // No margin, unlike `shadows.wgsl`: an underline paints inside its own
    // rectangle, which is why this kind is `Quad`-shaped in the compute passes
    // too and shadows are not.
    out.position = to_device_position_impl(
        unit * underline.origin_size.zw + underline.origin_size.xy,
    );
    out.color = underline.color;
    out.arena_index = arena_index;
    return out;
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let underline = underlines[in.arena_index];
    let thickness = underline.thickness;
    let wavy = underline.wavy;

    // See this file's header: `input.color.a` is the alpha *factor*, and
    // `blend_color` multiplies it by `color.a` again.
    if ((wavy & 0xFFu) == 0u) {
        return vec4<f32>(in.color.rgb, in.color.a * in.color.a);
    }

    let bounds_origin = underline.origin_size.xy;
    let bounds_size = underline.origin_size.zw;
    let half_thickness = thickness * 0.5;

    let st = (in.position.xy - bounds_origin) / bounds_size.y - vec2<f32>(0.0, 0.5);
    let frequency = M_PI_F * WAVE_FREQUENCY * thickness / bounds_size.y;
    let amplitude = (thickness * WAVE_HEIGHT_RATIO) / bounds_size.y;

    let sine = sin(st.x * frequency) * amplitude;
    let d_sine = cos(st.x * frequency) * amplitude * frequency;
    let distance = (st.y - sine) / sqrt(1.0 + d_sine * d_sine);
    let distance_in_pixels = distance * bounds_size.y;
    let distance_from_top_border = distance_in_pixels - half_thickness;
    let distance_from_bottom_border = distance_in_pixels + half_thickness;
    let alpha = saturate(0.5 - max(-distance_from_bottom_border, distance_from_top_border));
    return vec4<f32>(in.color.rgb, in.color.a * (alpha * in.color.a));
}
