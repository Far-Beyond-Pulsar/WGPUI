struct BlitParams {
    uv_origin: vec2f,
    uv_size: vec2f,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@group(0) @binding(0) var pipeline_texture: texture_2d<f32>;
@group(0) @binding(1) var sampler_: sampler;
@group(0) @binding(2) var<uniform> blit_params: BlitParams;

@vertex
fn vs_blit(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let positions = array<vec4f, 4>(
        vec4f(-1.0, -1.0, 0.0, 1.0),
        vec4f( 1.0, -1.0, 0.0, 1.0),
        vec4f(-1.0,  1.0, 0.0, 1.0),
        vec4f( 1.0,  1.0, 0.0, 1.0),
    );
    let uvs = array<vec2f, 4>(
        vec2f(0.0, 1.0),
        vec2f(1.0, 1.0),
        vec2f(0.0, 0.0),
        vec2f(1.0, 0.0),
    );
    var out: VertexOutput;
    out.position = positions[vertex_index];
    out.uv = blit_params.uv_origin + uvs[vertex_index] * blit_params.uv_size;
    return out;
}

@fragment
fn fs_blit(@location(0) uv: vec2f) -> @location(0) vec4f {
    return textureSample(pipeline_texture, sampler_, uv);
}
