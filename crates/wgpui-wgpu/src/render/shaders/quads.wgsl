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

struct Globals {
    // Framebuffer size in pixels.
    viewport: vec2<f32>,
    padding: vec2<f32>,
};

struct SlotBase {
    base: u32,
    padding: vec3<u32>,
};

// 64 bytes, matching `wgpui_core::patch::primitive::Quad::SLOT_STRIDE` and the
// field order `Quad::encode` writes.
struct QuadSlot {
    origin_size: vec4<f32>,
    background: vec4<f32>,
    border_color: vec4<f32>,
    // corner_radius, border_width, and two words of padding.
    radius_width: vec4<f32>,
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
    let point = quad.origin_size.xy + unit * quad.origin_size.zw;
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

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let quad = quads[in.arena_index];
    let size = quad.origin_size.zw;
    let half_size = size * 0.5;
    let radius = min(quad.radius_width.x, min(half_size.x, half_size.y));
    let border = quad.radius_width.y;

    // Rounded-rectangle signed distance, positive outside. Hard-edged rather
    // than antialiased on purpose: every comparison this shader takes part in
    // is a bit-exact one between two draw paths, and a coverage ramp would make
    // "identical" depend on rasterization order.
    let centred = abs(in.local - half_size) - (half_size - vec2<f32>(radius, radius));
    let distance = length(max(centred, vec2<f32>(0.0, 0.0))) - radius;
    if (distance > 0.0) {
        discard;
    }
    if (border > 0.0 && distance > -border) {
        return quad.border_color;
    }
    return quad.background;
}
