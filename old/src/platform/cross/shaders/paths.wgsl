// Paths shader
//
// Renders pre-tessellated path geometry produced by lyon/PathBuilder.
// Each vertex in the storage buffer carries its own position, ST curve coords,
// HSLA color, and content-mask bounds so no per-draw uniform changes are needed.
//
// Curve-edge triangles (from Path::curve_to) use the quadratic Bézier SDF:
//   discard if s*s > t
// Fill triangles (from lyon tessellation) always have st = (0, 1), so
//   0*0 = 0 < 1  →  always kept.

struct Globals {
    viewport_size: vec2<f32>,
    premultiplied_alpha: u32,
    pad: u32,
}

// Mirror of GpuPathVertex in renderer.rs (must match repr(C) layout exactly)
struct GpuPathVertex {
    xy_position:         vec2<f32>,   // offset  0
    st_position:         vec2<f32>,   // offset  8
    hsla:                vec4<f32>,   // offset 16  (h, s, l, a)
    content_mask_origin: vec2<f32>,   // offset 32
    content_mask_size:   vec2<f32>,   // offset 40
}                                     // stride  48

struct PathVarying {
    @builtin(position)              position:       vec4<f32>,
    @location(0)                    st:             vec2<f32>,
    @location(1) @interpolate(flat) color:          vec4<f32>,
    @location(2)                    clip_distances: vec4<f32>,
}

@group(0) @binding(0) var<uniform>      globals:          Globals;
@group(1) @binding(0) var<storage,read> b_path_vertices:  array<GpuPathVertex>;

// HSLA → linear RGBA (same formula used by all other shaders in this crate)
fn hsla_to_rgba(h: f32, s: f32, l: f32, a: f32) -> vec4<f32> {
    let hh = h * 6.0;
    let c  = (1.0 - abs(2.0 * l - 1.0)) * s;
    let x  = c * (1.0 - abs(hh % 2.0 - 1.0));
    let m  = l - c / 2.0;
    var rgb = vec3<f32>(m, m, m);
    if      hh >= 0.0 && hh < 1.0 { rgb += vec3<f32>(c, x, 0.0); }
    else if hh >= 1.0 && hh < 2.0 { rgb += vec3<f32>(x, c, 0.0); }
    else if hh >= 2.0 && hh < 3.0 { rgb += vec3<f32>(0.0, c, x); }
    else if hh >= 3.0 && hh < 4.0 { rgb += vec3<f32>(0.0, x, c); }
    else if hh >= 4.0 && hh < 5.0 { rgb += vec3<f32>(x, 0.0, c); }
    else                            { rgb += vec3<f32>(c, 0.0, x); }
    return vec4<f32>(rgb, a);
}

@vertex
fn vs_path(@builtin(vertex_index) vid: u32) -> PathVarying {
    let v = b_path_vertices[vid];

    // Pixel-space → NDC
    let device_pos = v.xy_position / globals.viewport_size
                   * vec2<f32>(2.0, -2.0)
                   + vec2<f32>(-1.0, 1.0);

    // Signed distances to the four edges of the content mask (positive = inside)
    let tl   = v.xy_position - v.content_mask_origin;
    let br   = v.content_mask_origin + v.content_mask_size - v.xy_position;
    let clip = vec4<f32>(tl.x, br.x, tl.y, br.y);

    let color = hsla_to_rgba(v.hsla.x, v.hsla.y, v.hsla.z, v.hsla.w);

    return PathVarying(
        vec4<f32>(device_pos, 0.0, 1.0),
        v.st_position,
        color,
        clip,
    );
}

@fragment
fn fs_path(v: PathVarying) -> @location(0) vec4<f32> {
    // Content-mask clipping
    if any(v.clip_distances < vec4<f32>(0.0)) {
        discard;
    }

    // Quadratic Bézier SDF (curve-edge triangles only; fill triangles pass always)
    let s = v.st.x;
    let t = v.st.y;
    if s * s > t {
        discard;
    }

    let a           = v.color.a;
    let multiplier  = select(1.0, a, globals.premultiplied_alpha != 0u);
    return vec4<f32>(v.color.rgb * multiplier, a);
}
