struct Globals { viewport: vec2<f32>, };
struct Tile { origin_size: vec4<f32>, color: vec4<f32>, border_width: f32, _padding: vec3<f32>, };
@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> tiles: array<Tile>;
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) local_position: vec2<f32>,
    @location(2) tile_size: vec2<f32>,
    @location(3) border_width: f32,
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
    output.local_position = corners[vertex_index] * tile.origin_size.zw;
    output.tile_size = tile.origin_size.zw;
    output.border_width = tile.border_width;
    return output;
}
@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let border = input.border_width;
    let inside = input.local_position.x >= border
        && input.local_position.y >= border
        && input.local_position.x < input.tile_size.x - border
        && input.local_position.y < input.tile_size.y - border;
    var color = input.color;
    if (inside) {
        color.a *= 0.12;
    }
    return color;
}
