struct Globals {
    viewport: vec2<f32>,
    padding: vec2<f32>,
}

struct SlotBase {
    base: u32,
    padding_0: u32,
    translation: vec2<f32>,
    clip_origin: vec2<f32>,
    clip_size: vec2<f32>,
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
    let screen_position = vertex.position + slot.translation;
    let device_position = screen_position / globals.viewport
        * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0);
    let clip_origin = vertex.clip_origin + slot.translation;
    let top_left = screen_position - clip_origin;
    let bottom_right = clip_origin + vertex.clip_size - screen_position;
    var clip_distances = vec4<f32>(top_left.x, bottom_right.x, top_left.y, bottom_right.y);
    if slot.clip_size.x >= 0.0 {
        let layer_clip_max = slot.clip_origin + slot.clip_size;
        clip_distances = min(
            clip_distances,
            vec4<f32>(
                screen_position.x - slot.clip_origin.x,
                layer_clip_max.x - screen_position.x,
                screen_position.y - slot.clip_origin.y,
                layer_clip_max.y - screen_position.y,
            ),
        );
    }
    return VertexOutput(
        vec4<f32>(device_position, 0.0, 1.0), vertex.color, vertex.st,
        clip_distances);
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    if any(input.clip_distances < vec4<f32>(0.0)) { discard; }
    if input.st.x * input.st.x > input.st.y { discard; }
    return input.color;
}
// See docs/gpu-native-architecture.md §3.5.
