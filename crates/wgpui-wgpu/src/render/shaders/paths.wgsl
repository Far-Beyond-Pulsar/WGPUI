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
    bounds_origin: vec2<f32>,
    bounds_size: vec2<f32>,
    // kind, padding, padding, padding; first color; second color; parameters.
    material_kind: vec4<u32>,
    material_first: vec4<f32>,
    material_second: vec4<f32>,
    material_parameters: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) color: vec4<f32>,
    @location(1) st: vec2<f32>,
    @location(2) clip_distances: vec4<f32>,
    @location(3) @interpolate(flat) material_kind: u32,
    @location(4) @interpolate(flat) material_first: vec4<f32>,
    @location(5) @interpolate(flat) material_second: vec4<f32>,
    @location(6) @interpolate(flat) material_parameters: vec4<f32>,
    @location(7) local: vec2<f32>,
    @location(8) local_pixels: vec2<f32>,
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
    let local_pixels = vertex.position - vertex.bounds_origin;
    let local = local_pixels / max(vertex.bounds_size, vec2<f32>(0.000001));
    return VertexOutput(
        vec4<f32>(device_position, 0.0, 1.0), vertex.color, vertex.st,
        clip_distances, vertex.material_kind.x, vertex.material_first,
        vertex.material_second, vertex.material_parameters, local, local_pixels);
}

fn material_color(input: VertexOutput) -> vec4<f32> {
    if input.material_kind == 0u {
        return input.color;
    }
    var amount = 0.0;
    if input.material_kind == 1u {
        amount = dot(input.local - vec2<f32>(0.5), input.material_parameters.xy) + 0.5;
    } else if input.material_kind == 2u {
        let delta = (input.local - input.material_parameters.xy)
            / max(input.material_parameters.zw, vec2<f32>(0.000001));
        amount = length(delta);
    } else if input.material_kind == 3u {
        let interval = max(input.material_parameters.y, 0.000001);
        let diagonal = (input.local_pixels.x + input.local_pixels.y) / interval;
        let on = select(0.0, 1.0,
            fract(diagonal) < input.material_parameters.x / interval);
        return mix(input.material_second, input.material_first, on);
    } else if input.material_kind == 4u {
        let cell = max(input.material_parameters.x, 0.000001);
        let parity = (floor(input.local_pixels.x / cell)
            + floor(input.local_pixels.y / cell)) % 2.0;
        return mix(input.material_second, input.material_first, parity);
    } else if input.material_kind == 5u {
        let width = max(input.material_parameters.x, 0.000001);
        return mix(input.material_second, input.material_first,
            select(0.0, 1.0, fract(input.local_pixels.y / width) < 0.5));
    }
    return mix(input.material_first, input.material_second, clamp(amount, 0.0, 1.0));
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    if any(input.clip_distances < vec4<f32>(0.0)) { discard; }
    if input.st.x * input.st.x > input.st.y { discard; }
    return material_color(input);
}
// See docs/gpu-native-architecture.md §3.5.
