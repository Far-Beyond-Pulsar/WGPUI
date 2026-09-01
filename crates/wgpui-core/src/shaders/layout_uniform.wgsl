struct Params {
    item_count: u32,
    axis: u32,
    justify: u32,
    align: u32,
    origin: vec2<f32>,
    container_size: vec2<f32>,
    padding: vec4<f32>,
    gap: f32,
    rounding_scale: f32,
    _padding: vec2<f32>,
};

// Five vec4 values are 80 bytes, matching REGULAR_LAYOUT_ITEM_STRIDE. The
// first record contains the already-resolved flex size plus min bounds; the
// second contains max bounds plus grow/shrink factors.
struct Item {
    size_min: vec4<f32>,
    max_flex: vec4<f32>,
    margin: vec4<f32>,
    transform_linear: vec4<f32>,
    transform_translation: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> items: array<Item>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;

fn item_size(item: Item) -> vec2<f32> {
    return clamp(item.size_min.xy, item.size_min.zw, item.max_flex.xy);
}

fn main_size(size: vec2<f32>, row: bool) -> f32 {
    if (row) { return size.x; }
    return size.y;
}

fn cross_size(size: vec2<f32>, row: bool) -> f32 {
    if (row) { return size.y; }
    return size.x;
}

fn round_edge(value: f32) -> f32 {
    let scaled = value * params.rounding_scale;
    if (scaled >= 0.0) {
        return floor(scaled + 0.5) / params.rounding_scale;
    }
    return ceil(scaled - 0.5) / params.rounding_scale;
}

@compute @workgroup_size(64)
fn compute_layout(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= params.item_count) { return; }

    let row = params.axis == 0u || params.axis == 2u;
    let reverse = params.axis == 2u || params.axis == 3u;
    let main_start = select(params.padding.y, params.padding.x, row);
    let main_end = select(params.padding.w, params.padding.z, row);
    let cross_start = select(params.padding.x, params.padding.y, row);
    let container_main = select(params.container_size.y, params.container_size.x, row);
    let container_cross = select(params.container_size.x, params.container_size.y, row);
    let available_main = max(container_main - main_start - main_end, 0.0);
    let cross_end = select(params.padding.z, params.padding.w, row);
    let available_cross = max(container_cross - cross_start - cross_end, 0.0);

    var occupied = params.gap * f32(max(params.item_count, 1u) - 1u);
    var item_index = 0u;
    loop {
        if (item_index >= params.item_count) { break; }
        let item = items[item_index];
        let size = item_size(item);
        let margin = select(item.margin.y + item.margin.w, item.margin.x + item.margin.z, row);
        occupied += main_size(size, row) + margin;
        item_index += 1u;
    }
    let free = max(available_main - occupied, 0.0);
    var leading = 0.0;
    var extra_gap = 0.0;
    if (params.justify == 1u) { leading = free; }
    if (params.justify == 2u) { leading = free * 0.5; }
    if (params.justify == 3u && params.item_count > 1u) { extra_gap = free / f32(params.item_count - 1u); }
    if (params.justify == 4u && params.item_count > 0u) { leading = free / f32(params.item_count) * 0.5; extra_gap = free / f32(params.item_count); }
    if (params.justify == 5u && params.item_count > 0u) { leading = free / f32(params.item_count + 1u); extra_gap = leading; }

    var cursor = main_start + leading;
    var prior = 0u;
    loop {
        if (prior >= index) { break; }
        let prior_item = items[prior];
        let prior_size = item_size(prior_item);
        let prior_margin = select(prior_item.margin.y, prior_item.margin.x, row);
        let prior_end_margin = select(prior_item.margin.w, prior_item.margin.z, row);
        cursor += prior_margin + main_size(prior_size, row) + prior_end_margin + params.gap + extra_gap;
        prior += 1u;
    }

    let item = items[index];
    let size = item_size(item);
    let item_main = main_size(size, row);
    let item_cross = min(cross_size(size, row), available_cross);
    let cross_margin = select(item.margin.y, item.margin.x, row);
    let cross_end_margin = select(item.margin.w, item.margin.z, row);
    let cross_free = max(available_cross - item_cross - cross_margin - cross_end_margin, 0.0);
    var cross_offset = cross_margin;
    if (params.align == 1u) { cross_offset += cross_free; }
    if (params.align == 2u) { cross_offset += cross_free * 0.5; }
    let item_margin = select(item.margin.y, item.margin.x, row);
    cursor += item_margin;
    let main_position = select(cursor, available_main + main_start - (cursor - main_start) - item_main, reverse);
    let cross_position = cross_start + cross_offset;
    var rectangle = vec4<f32>(main_position, cross_position, item_main, item_cross);
    if (!row) { rectangle = vec4<f32>(cross_position, main_position, item_cross, item_main); }

    let linear = item.transform_linear;
    let translation = item.transform_translation.xy;
    let left = round_edge((params.origin.x + rectangle.x) * linear.x + translation.x);
    let top = round_edge((params.origin.y + rectangle.y) * linear.w + translation.y);
    let right = round_edge((params.origin.x + rectangle.x + rectangle.z) * linear.x + translation.x);
    let bottom = round_edge((params.origin.y + rectangle.y + rectangle.w) * linear.w + translation.y);
    output[index] = vec4<f32>(left, top, max(right - left, 0.0), max(bottom - top, 0.0));
}
