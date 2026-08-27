// The one composite pipeline (§5.5, Gap 2): an already-rendered texture placed
// into the ordered scene, clipped, and drawn.
//
// Two producers reach this shader — a boundary's own baked texture
// (`render/textures/layer_texture.rs`) and an externally-rendered surface
// (`render/textures/external_surface.rs`, backed by `SurfaceRegistry`'s
// triple buffer) — and nothing here can tell them apart. That is the whole of
// §5.5's Gap 2: "a `WgpuSurface` becomes the degenerate case of a compositing
// boundary." In the legacy backend these are two separate ~180-line arms of one
// `match`, each building its own bind group from its own texture source; here
// the difference ends at which `wgpu::TextureView` the caller binds.
//
// The draw is issued indirectly, from the same argument buffer and the same
// fixed sequence the quad pipeline uses, so an entry whose instance count the
// compute pass wrote as zero expands to nothing without the CPU asking.

struct Globals {
    viewport: vec2<f32>,
    padding: vec2<f32>,
};

struct CompositeParams {
    // Destination rectangle: origin.xy, size.xy.
    bounds: vec4<f32>,
    // Clip rectangle: origin.xy, size.xy.
    content_mask: vec4<f32>,
    // opacity, corner_radius, and two words of padding.
    style: vec4<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(1) @binding(0) var<uniform> params: CompositeParams;
@group(1) @binding(1) var source: texture_2d<f32>;
@group(1) @binding(2) var source_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) window: vec2<f32>,
    @location(2) local: vec2<f32>,
};

@vertex
fn vertex_main(@builtin(vertex_index) corner: u32) -> VertexOutput {
    var out: VertexOutput;
    let unit = vec2<f32>(f32(corner & 1u), f32((corner >> 1u) & 1u));
    let point = params.bounds.xy + unit * params.bounds.zw;
    out.position = vec4<f32>(
        point.x / globals.viewport.x * 2.0 - 1.0,
        1.0 - point.y / globals.viewport.y * 2.0,
        0.0,
        1.0,
    );
    out.uv = unit;
    out.window = point;
    out.local = unit * params.bounds.zw;
    return out;
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // The content mask clips; it never scales. A boundary's buffered margin
    // content lives inside `bounds` and outside `content_mask`, and must not
    // paint — R-N §8.3's overdraw exemption seen from the drawing side.
    let mask_min = params.content_mask.xy;
    let mask_max = params.content_mask.xy + params.content_mask.zw;
    if (in.window.x < mask_min.x || in.window.y < mask_min.y
        || in.window.x > mask_max.x || in.window.y > mask_max.y) {
        discard;
    }

    let size = params.bounds.zw;
    let half_size = size * 0.5;
    let radius = min(params.style.y, min(half_size.x, half_size.y));
    let centred = abs(in.local - half_size) - (half_size - vec2<f32>(radius, radius));
    if (length(max(centred, vec2<f32>(0.0, 0.0))) - radius > 0.0) {
        discard;
    }

    let sampled = textureSample(source, source_sampler, in.uv);
    return vec4<f32>(sampled.rgb, sampled.a * params.style.x);
}
