struct Globals { viewport: vec2<f32>, };
struct Tile { origin_size: vec4<f32>, color: vec4<f32>, };
@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> tiles: array<Tile>;
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};
@vertex
fn vertex_main(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let tile = tiles[instance_index];
    let corners = array<vec2<f32>, 4>(vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0));
    let pixel = tile.origin_size.xy + corners[vertex_index] * tile.origin_size.zw;
    let clip = pixel / globals.viewport * 2.0 - vec2<f32>(1.0, 1.0);
    var output: VertexOutput;
    output.position = vec4<f32>(clip.x, -clip.y, 0.0, 1.0);
    output.color = tile.color;
    return output;
}
@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> { return input.color; }
