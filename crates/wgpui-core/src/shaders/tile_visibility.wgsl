// Tile visibility (§4.3): which tiles of a `Buffering::Tiled` boundary are in
// range at the current pan offset, and — the part that matters — the draw slots
// the indirect-arg pass then generates arguments from.
//
// This is a transcription of `wgpui_core::scene::tile::tile_visibility`, checked
// against it for exact equality by `wgpui-wgpu/tests/tile_visibility.rs`. Read
// that function's doc for what the computation is and why the rectangle
// dilation is exact rather than approximate; this file is only how it is spelled
// for the GPU.
//
// # This is not a drawing pipeline, and that is the design
//
// §4.3 asks for "a compute pass [that] computes tile visibility directly from a
// pan-offset uniform and writes indirect draw args only for in-range tiles; the
// CPU never enumerates tile candidates." What that turns into here is one
// invocation per resident tile writing one `vec4<u32>` — because the record it
// writes, `[base, count, 0, 0]`, is *exactly* the slot record
// `indirect_args.wgsl` already consumes (`wgpui_core::indirect::encode_slots`'
// layout, asserted byte-identical in `scene/tile.rs`'s own tests).
//
// So an out-of-range tile is a slot with a zero count, and Phase 4's existing
// `compact` turns that into a zero-instance argument record while `pack` drops
// it from the multi-draw entirely. Nothing new draws, no second mechanism
// decides what draws, and the entire tile-visibility feature is this file plus
// the buffer it writes into. That is §4.3's "tiling needs almost no new
// machinery" being true rather than being claimed.
//
// # What the CPU still enumerates, honestly
//
// The draw path is as §4.3 describes: the CPU hands over resident tile
// descriptors and learns nothing about which of them draw. But *residency* is
// still CPU-side, and has to be — a newly-revealed tile needs its content
// rendered into it, which is CPU work no visibility kernel can do. So
// `TileResidency` runs the same predicate on the CPU for the tiles it keeps,
// while this kernel decides what draws. The two are the same rule and are
// checked against each other; see `docs/phase-4.5-results.md`.

struct Params {
    // Tile edge lengths in logical pixels.
    tile_size: vec2<f32>,
    // The boundary's layer transform: where its content composites. The plane
    // slides under a fixed window by the negative of this.
    pan: vec2<f32>,
    // The boundary's visible rectangle in window space: min_x, min_y, max_x,
    // max_y.
    viewport: vec4<f32>,
    // Resident tiles in this dispatch.
    tile_count: u32,
    // How many tiles beyond the viewport stay in range.
    retain_radius: u32,
    pad_0: u32,
    pad_1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
// [coord.x, coord.y, base, count] per resident tile. The two coordinates are
// `i32`s bit-cast in place — a plane has no origin corner, so they are signed.
@group(0) @binding(1) var<storage, read> tiles: array<vec4<u32>>;
// [base, count, 0, 0] per tile, in `indirect_args.wgsl`'s slot layout.
@group(0) @binding(2) var<storage, read_write> slots: array<vec4<u32>>;
// 1 in range, 0 out. Not read by the draw path — it exists so a differential can
// name the tile that disagreed rather than only the slot.
@group(0) @binding(3) var<storage, read_write> in_range: array<u32>;

@compute @workgroup_size(64)
fn tile_visibility(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= params.tile_count) {
        return;
    }
    let tile = tiles[index];
    let coord = vec2<f32>(f32(bitcast<i32>(tile.x)), f32(bitcast<i32>(tile.y)));
    let tile_min = coord * params.tile_size;
    let tile_max = tile_min + params.tile_size;

    // The content-plane rectangle under the viewport, dilated by the retain
    // radius. Dilating the rectangle by `radius * tile_size` selects exactly the
    // tiles that dilating the coordinate span by `radius` selects, because tile
    // edges sit at exact multiples of the tile size — which is what lets this
    // kernel test rectangles while residency works in coordinates.
    let margin = f32(params.retain_radius) * params.tile_size;
    let range_min = params.viewport.xy - params.pan - margin;
    let range_max = params.viewport.zw - params.pan + margin;

    // `Rect::intersects`, edge for edge: strict on every side, so a tile the
    // range merely touches along an edge is out of range. The Rust reference
    // reaches the same predicate through `Rect::intersects` itself, and the
    // differential compares the results for exact equality rather than for
    // approximate agreement.
    let overlaps = range_min.x < tile_max.x
        && tile_min.x < range_max.x
        && range_min.y < tile_max.y
        && tile_min.y < range_max.y;

    in_range[index] = select(0u, 1u, overlaps);
    slots[index] = vec4<u32>(tile.z, select(0u, tile.w, overlaps), 0u, 0u);
}
