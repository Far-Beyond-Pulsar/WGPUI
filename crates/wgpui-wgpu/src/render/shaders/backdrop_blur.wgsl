const M_PI_F: f32 = 3.1415926;

struct Globals { viewport: vec2<f32>, padding: vec2<f32>, }
struct SlotBase { base: u32, padding_0: u32, padding_1: u32, padding_2: u32, }
struct BackdropFilter {
    origin: vec2<f32>, size: vec2<f32>,
    clip_origin: vec2<f32>, clip_size: vec2<f32>,
    corner_radii: vec4<f32>, blur_radius: f32, opacity: f32, padding: vec2<f32>,
}
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) clip_distances: vec4<f32>,
}

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> filters: array<BackdropFilter>;
@group(1) @binding(0) var<uniform> slot: SlotBase;
@group(2) @binding(0) var backdrop_texture: texture_2d<f32>;
@group(2) @binding(1) var backdrop_sampler: sampler;

fn current_filter() -> BackdropFilter { return filters[slot.base]; }
fn gaussian(x: f32, sigma: f32) -> f32 {
    return exp(-(x * x) / (2.0 * sigma * sigma)) / (sqrt(2.0 * M_PI_F) * sigma);
}
fn rounded_rectangle_sdf(point: vec2<f32>, origin: vec2<f32>, size: vec2<f32>, radii: vec4<f32>) -> f32 {
    let center = origin + size / 2.0;
    let half_size = size / 2.0;
    var radius = radii.x;
    if point.x >= center.x && point.y < center.y { radius = radii.y; }
    if point.x >= center.x && point.y >= center.y { radius = radii.z; }
    if point.x < center.x && point.y >= center.y { radius = radii.w; }
    let q = abs(point - center) - half_size + vec2<f32>(radius);
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0))) - radius;
}

@vertex
fn vertex_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let unit_vertex = vec2<f32>(f32(vertex_index & 1u), 0.5 * f32(vertex_index & 2u));
    let current = current_filter();
    let pixel_position = unit_vertex * current.size + current.origin;
    let device_position = pixel_position / globals.viewport
        * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0);
    let top_left = pixel_position - current.clip_origin;
    let bottom_right = current.clip_origin + current.clip_size - pixel_position;
    return VertexOutput(vec4<f32>(device_position, 0.0, 1.0),
        vec4<f32>(top_left.x, bottom_right.x, top_left.y, bottom_right.y));
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    if any(input.clip_distances < vec4<f32>(0.0)) { discard; }
    let current = current_filter();
    let pixel_position = input.position.xy;
    var blurred_color = vec4<f32>(0.0);
    var total_weight = 0.0;
    if current.blur_radius < 0.5 {
        blurred_color = textureSampleLevel(backdrop_texture, backdrop_sampler,
            pixel_position / globals.viewport, 0.0);
    } else {
        let effective_radius = min(current.blur_radius, 32.0);
        let kernel_size = i32(ceil(effective_radius * 2.0));
        let sigma = max(effective_radius / 2.0, 0.0001);
        var dy = -kernel_size;
        loop {
            if dy > kernel_size { break; }
            var dx = -kernel_size;
            loop {
                if dx > kernel_size { break; }
                let offset = vec2<f32>(f32(dx), f32(dy));
                let sample_position = pixel_position + offset;
                let sample_uv = sample_position / globals.viewport;
                if all(sample_uv >= vec2<f32>(0.0)) && all(sample_uv <= vec2<f32>(1.0)) {
                    let weight = gaussian(length(offset), sigma);
                    blurred_color += textureSampleLevel(backdrop_texture, backdrop_sampler,
                        sample_uv, 0.0) * weight;
                    total_weight += weight;
                }
                dx += 1;
            }
            dy += 1;
        }
        if total_weight > 0.0 { blurred_color /= total_weight; }
    }
    let mask = clamp(0.5 - rounded_rectangle_sdf(pixel_position, current.origin,
        current.size, current.corner_radii), 0.0, 1.0);
    let factor = mask * current.opacity;
    return vec4<f32>(blurred_color.rgb * factor, blurred_color.a * factor);
}
// See docs/gpu-native-architecture.md §3.5.
