struct Globals {
    viewport: vec2<f32>,
    padding: vec2<f32>,
}

struct SlotBase {
    base: u32,
    padding_0: u32,
    padding_1: u32,
    padding_2: u32,
}

struct PathVertex {
    position: vec2<f32>,
    st: vec2<f32>,
    color: vec4<f32>,
    clip_origin: vec2<f32>,
    clip_size: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) color: vec4<f32>,
    @location(1) st: vec2<f32>,
    @location(2) clip_distances: vec4<f32>,
}

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> paths: array<PathVertex>;
@group(1) @binding(0) var<uniform> slot: SlotBase;

@vertex
fn vertex_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let vertex = paths[slot.base + vertex_index];
    let device_position = vertex.position / globals.viewport
        * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0);
    let top_left = vertex.position - vertex.clip_origin;
    let bottom_right = vertex.clip_origin + vertex.clip_size - vertex.position;
    return VertexOutput(
        vec4<f32>(device_position, 0.0, 1.0), vertex.color, vertex.st,
        vec4<f32>(top_left.x, bottom_right.x, top_left.y, bottom_right.y));
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    if any(input.clip_distances < vec4<f32>(0.0)) { discard; }
    if input.st.x * input.st.x > input.st.y { discard; }
    return input.color;
}
// See docs/gpu-native-architecture.md §3.5.
