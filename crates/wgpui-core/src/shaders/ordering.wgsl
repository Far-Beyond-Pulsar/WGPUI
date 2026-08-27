// Painter ordering for one layer, as a compute pass.
// docs/gpu-native-architecture.md §5.1; the recurrence and the CPU reference
// this transcribes live in `ordering.rs` / `ordering/bounds_tree.rs`.
//
// The computation, restated from `ordering.rs`:
//
//     order[i] = 1 + max{ order[j] : j < i and bounds[j] intersects bounds[i] }
//
// solved by Jacobi relaxation over ping-ponged `order_in`/`order_out` buffers.
// `changed` counts invocations whose value moved this iteration, so the host
// can tell a converged fixed point from a truncated budget rather than assume
// one — Phase 0's Spike A ran a fixed 128 iterations and checked afterwards;
// this runs until the counter reads zero.
//
// The inner query is bounded by a two-level AABB hierarchy over paint order:
// `blocks[b]` unions primitives [b*64, b*64+64) and `superblocks[s]` unions
// blocks [s*64, s*64+64). Rejecting a node whose union does not intersect the
// query is exact — a union containing a strictly-intersecting member
// necessarily intersects too — so this prunes without changing the answer. It
// is the same idea `BoundsTree::find_max_ordering` prunes with, built over
// paint order (which a real layer emits spatially coherently) instead of over
// an incrementally balanced tree, because a compute pass cannot rebalance.
//
// Degradation, stated rather than discovered: a layer whose primitives are
// spatially scattered relative to paint order gets loose block unions and
// approaches the O(n^2) scan. It stays correct; it stops being fast.

const BLOCK_SIZE: u32 = 64u;
const SUPERBLOCK_SIZE: u32 = 64u;
const SORT_SENTINEL: u32 = 4294967295u;

struct Params {
    // Primitives in this layer.
    count: u32,
    // ceil(count / BLOCK_SIZE).
    block_count: u32,
    // ceil(block_count / SUPERBLOCK_SIZE).
    superblock_count: u32,
    // count rounded up to a power of two — the bitonic sort's array length.
    padded_count: u32,
};

struct BitonicStage {
    // Partner distance for this stage.
    span: u32,
    // Direction-selecting stage width.
    width: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> bounds: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> blocks: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> superblocks: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> order_in: array<u32>;
@group(0) @binding(5) var<storage, read_write> order_out: array<u32>;
@group(0) @binding(6) var<storage, read_write> changed: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> sort_key: array<u32>;
@group(0) @binding(8) var<storage, read_write> sort_value: array<u32>;
@group(1) @binding(0) var<uniform> stage: BitonicStage;

// Strict on every edge, matching `Rect::intersects` and, through it,
// `Bounds::intersects` in the legacy backend. Two rectangles that merely touch
// do not step each other's order.
fn overlaps(left: vec4<f32>, right: vec4<f32>) -> bool {
    return left.x < right.z && right.x < left.z && left.y < right.w && right.y < left.w;
}

fn union_of(left: vec4<f32>, right: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        min(left.x, right.x),
        min(left.y, right.y),
        max(left.z, right.z),
        max(left.w, right.w),
    );
}

// A rectangle that intersects nothing, used for the tail of a partly-filled
// hierarchy node. Its min edges sit above its max edges, so `overlaps` is false
// against every finite query.
fn never_overlaps() -> vec4<f32> {
    return vec4<f32>(1.0, 1.0, -1.0, -1.0);
}

@compute @workgroup_size(64)
fn build_blocks(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let block = global_id.x;
    if (block >= params.block_count) {
        return;
    }
    let first = block * BLOCK_SIZE;
    let last = min(first + BLOCK_SIZE, params.count);
    if (first >= last) {
        blocks[block] = never_overlaps();
        return;
    }
    var accumulated = bounds[first];
    var index = first + 1u;
    loop {
        if (index >= last) {
            break;
        }
        accumulated = union_of(accumulated, bounds[index]);
        index = index + 1u;
    }
    blocks[block] = accumulated;
}

@compute @workgroup_size(64)
fn build_superblocks(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let super_index = global_id.x;
    if (super_index >= params.superblock_count) {
        return;
    }
    let first = super_index * SUPERBLOCK_SIZE;
    let last = min(first + SUPERBLOCK_SIZE, params.block_count);
    if (first >= last) {
        superblocks[super_index] = never_overlaps();
        return;
    }
    var accumulated = blocks[first];
    var index = first + 1u;
    loop {
        if (index >= last) {
            break;
        }
        accumulated = union_of(accumulated, blocks[index]);
        index = index + 1u;
    }
    superblocks[super_index] = accumulated;
}

@compute @workgroup_size(64)
fn relax(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let target_index = global_id.x;
    if (target_index >= params.count) {
        return;
    }
    let query = bounds[target_index];
    var best: u32 = 0u;

    var super_index: u32 = 0u;
    loop {
        if (super_index >= params.superblock_count) {
            break;
        }
        // Every primitive under this superblock is painted at or after
        // `target_index`, so no earlier primitive remains anywhere above it.
        if (super_index * SUPERBLOCK_SIZE * BLOCK_SIZE >= target_index) {
            break;
        }
        if (overlaps(superblocks[super_index], query)) {
            var block = super_index * SUPERBLOCK_SIZE;
            let block_end = min(block + SUPERBLOCK_SIZE, params.block_count);
            loop {
                if (block >= block_end) {
                    break;
                }
                let first = block * BLOCK_SIZE;
                if (first >= target_index) {
                    break;
                }
                if (overlaps(blocks[block], query)) {
                    var probe = first;
                    let probe_end = min(first + BLOCK_SIZE, target_index);
                    loop {
                        if (probe >= probe_end) {
                            break;
                        }
                        if (overlaps(bounds[probe], query)) {
                            let candidate = order_in[probe];
                            if (candidate > best) {
                                best = candidate;
                            }
                        }
                        probe = probe + 1u;
                    }
                }
                block = block + 1u;
            }
        }
        super_index = super_index + 1u;
    }

    let resolved = best + 1u;
    if (resolved != order_in[target_index]) {
        atomicAdd(&changed[0], 1u);
    }
    order_out[target_index] = resolved;
}

// Fill the sort arrays from whichever order buffer the host bound as
// `order_in` after the last relaxation iteration. Padding entries take a
// sentinel key so they sort to the end and never appear in the draw
// permutation.
@compute @workgroup_size(64)
fn pack(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= params.padded_count) {
        return;
    }
    if (index >= params.count) {
        sort_key[index] = SORT_SENTINEL;
        sort_value[index] = SORT_SENTINEL;
        return;
    }
    sort_key[index] = order_in[index];
    sort_value[index] = index;
}

// Ascending by (key, value). The value tie-break is what makes an unstable
// bitonic network reproduce `Scene::finish`'s stable `sort_by_key`: two
// non-overlapping primitives sharing an order keep their emission order.
fn sorts_before(left_key: u32, left_value: u32, right_key: u32, right_value: u32) -> bool {
    if (left_key != right_key) {
        return left_key < right_key;
    }
    return left_value < right_value;
}

@compute @workgroup_size(64)
fn bitonic(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= params.padded_count) {
        return;
    }
    let partner = index ^ stage.span;
    if (partner <= index) {
        return;
    }
    let ascending = (index & stage.width) == 0u;
    let key_here = sort_key[index];
    let key_there = sort_key[partner];
    let value_here = sort_value[index];
    let value_there = sort_value[partner];
    let there_first = sorts_before(key_there, value_there, key_here, value_here);
    // Swap when the pair is out of order for this stage's direction.
    let swap = (ascending && there_first) || (!ascending && !there_first);
    if (swap) {
        sort_key[index] = key_there;
        sort_key[partner] = key_here;
        sort_value[index] = value_there;
        sort_value[partner] = value_here;
    }
}
