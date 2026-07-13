struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

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
    return VertexOutput(positions[vertex_index], uvs[vertex_index]);
}

@group(0) @binding(0) var pipeline_texture: texture_2d<f32>;
@group(0) @binding(1) var sampler_: sampler;

@fragment
fn fs_blit(@location(0) uv: vec2f) -> @location(0) vec4f {
    return textureSample(pipeline_texture, sampler_, uv);
}
