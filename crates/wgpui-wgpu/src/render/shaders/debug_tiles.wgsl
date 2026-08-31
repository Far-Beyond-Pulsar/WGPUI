struct Globals { viewport: vec2<f32>, };
struct Tile { origin_size: vec4<f32>, color: vec4<f32>, border_width: f32, refresh_rate: f32, display_mode: f32, _padding: array<f32, 5>, };
@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> tiles: array<Tile>;
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) local_position: vec2<f32>,
    @location(2) tile_size: vec2<f32>,
    @location(3) border_width: f32,
    @location(4) refresh_rate: f32,
    @location(5) display_mode: f32,
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
    output.refresh_rate = tile.refresh_rate;
    output.display_mode = tile.display_mode;
    return output;
}

fn segment_on(digit: i32, segment: i32) -> bool {
    switch digit {
        case 0: { return segment != 6; }
        case 1: { return segment == 1 || segment == 2; }
        case 2: { return segment == 0 || segment == 1 || segment == 6 || segment == 4 || segment == 3; }
        case 3: { return segment == 0 || segment == 1 || segment == 6 || segment == 2 || segment == 3; }
        case 4: { return segment == 5 || segment == 6 || segment == 1 || segment == 2; }
        case 5: { return segment == 0 || segment == 5 || segment == 6 || segment == 2 || segment == 3; }
        case 6: { return segment != 1; }
        case 7: { return segment == 0 || segment == 1 || segment == 2; }
        case 8: { return true; }
        case 9: { return segment != 4; }
        default: { return false; }
    }
}

fn refresh_label(input: VertexOutput) -> bool {
    if input.refresh_rate < 0.5 {
        return false;
    }
    let point = input.local_position - vec2<f32>(5.0, 5.0);
    if point.x < 0.0 || point.y < 0.0 || point.x >= 20.0 || point.y >= 9.0 {
        return false;
    }
    let digit_index = i32(floor(point.x / 7.0));
    let digit_power = pow(10.0, f32(2 - digit_index));
    let digit = i32(floor(input.refresh_rate / digit_power)) % 10;
    let local = vec2<f32>(point.x - f32(digit_index) * 7.0, point.y);
    let segment = select(
        select(
            select(6, 0, local.y < 2.0 && local.x >= 1.0 && local.x < 6.0),
            5,
            local.x < 2.0 && local.y >= 1.0 && local.y < 5.0,
        ),
        1,
        local.x >= 5.0 && local.y >= 1.0 && local.y < 5.0,
    );
    if local.y >= 4.0 && local.x < 2.0 {
        return segment_on(digit, 4);
    }
    if local.y >= 4.0 && local.x >= 5.0 {
        return segment_on(digit, 2);
    }
    if local.y >= 7.0 && local.x >= 1.0 && local.x < 6.0 {
        return segment_on(digit, 3);
    }
    if local.y >= 3.0 && local.y < 5.0 && local.x >= 1.0 && local.x < 6.0 {
        return segment_on(digit, 6);
    }
    return segment_on(digit, segment);
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let border = input.border_width;
    let inside_border = input.local_position.x >= border
        && input.local_position.y >= border
        && input.local_position.x < input.tile_size.x - border
        && input.local_position.y < input.tile_size.y - border;
    if refresh_label(input) {
        return vec4<f32>(1.0, 1.0, 0.0, input.color.a);
    }
    if inside_border {
        discard;
    }
    return input.color;
}
