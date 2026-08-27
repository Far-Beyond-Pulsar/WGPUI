// Painter ordering for one layer, as a compute pass.
// docs/gpu-native-architecture.md §5.1; the recurrence and the CPU reference
// this transcribes live in `ordering.rs` / `ordering/bounds_tree.rs`.
//
// The computation, restated from `ordering.rs`:
//
//     order[i] = 1 + max{ order[j] : j < i and bounds[j] intersects bounds[i] }
//
// solved by relaxation over ping-ponged `order_in`/`order_out` buffers.
// `changed[0]` counts invocations whose value moved this iteration, so the host
// can tell a converged fixed point from a truncated budget rather than assume
// one — Phase 0's Spike A ran a fixed 128 iterations and checked afterwards;
// this runs until the counter reads zero.
//
// # Two prunings, and why the second one is load-bearing
//
// **Spatial.** The inner query is bounded by a two-level AABB hierarchy over
// paint order: `hierarchy[b]` unions primitives [b*64, b*64+64) and
// `hierarchy[block_count + s]` unions blocks [s*64, s*64+64). Rejecting a node
// whose union does not intersect the query is exact — a union containing a
// strictly-intersecting member necessarily intersects too — so this prunes
// without changing the answer. It is the same idea
// `BoundsTree::find_max_ordering` prunes with, built over paint order (which a
// real layer emits spatially coherently) instead of over an incrementally
// balanced tree, because a compute pass cannot rebalance.
//
// **Temporal, and this is what makes the pass viable at all.** Plain Jacobi
// rescans every neighbour every iteration, so its cost is
// `O(primitives × neighbours × iterations)` — and the iteration count is the
// *deepest painter order in the layer*, since each pass propagates one step
// along the overlap chain. Measured on a zoomed-out node graph whose chain runs
// 577 deep, that lost to the CPU `BoundsTree` by ~35×. The fix is to exploit
// that the relaxation is monotone: values only ever rise. So an iteration
// computes
//
//     order_out[i] = max(order_in[i], 1 + max{ order_in[j] : j in a block that
//                                              changed last iteration })
//
// and skips every block that settled, because a settled block's contribution is
// already folded into `order_in[i]`. `changed_in`/`changed_out` carry one flag
// per block and per superblock so a settled *region* is rejected with a single
// load. The fixed point is unchanged — the induction is written out in
// `ordering.rs` — and the total work collapses from
// `primitives × iterations` to roughly `primitives + edges`.
//
// Degradation, stated rather than discovered: a layer whose primitives are
// spatially scattered relative to paint order gets loose block unions and
// approaches the O(n^2) scan per iteration. It stays correct; it stops being
// fast.

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

// Blocks and superblocks share one buffer — blocks at [0, block_count),
// superblocks at [block_count, block_count + superblock_count) — and so do the
// change flags, whose word 0 is the global counter. Both consolidations exist
// to keep this shader at eight storage bindings, which is WebGPU's own
// downlevel ceiling: a pass that needed ten would work on this desktop and stop
// working on the platforms §0's constraint 2 keeps as a best-effort target.
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> bounds: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> hierarchy: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> order_in: array<u32>;
@group(0) @binding(4) var<storage, read_write> order_out: array<u32>;
@group(0) @binding(5) var<storage, read> changed_in: array<u32>;
@group(0) @binding(6) var<storage, read_write> changed_out: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> sort_key: array<u32>;
@group(0) @binding(8) var<storage, read_write> sort_value: array<u32>;
@group(1) @binding(0) var<uniform> stage: BitonicStage;

// Word 0 of a change buffer is the global counter; block `b`'s flag is at
// `1 + b` and superblock `s`'s at `1 + block_count + s`.
fn block_flag_index(block: u32) -> u32 {
    return 1u + block;
}

fn superblock_flag_index(superblock: u32) -> u32 {
    return 1u + params.block_count + superblock;
}

fn superblock_node_index(superblock: u32) -> u32 {
    return params.block_count + superblock;
}

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
        hierarchy[block] = never_overlaps();
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
    hierarchy[block] = accumulated;
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
        hierarchy[superblock_node_index(super_index)] = never_overlaps();
        return;
    }
    var accumulated = hierarchy[first];
    var index = first + 1u;
    loop {
        if (index >= last) {
            break;
        }
        accumulated = union_of(accumulated, hierarchy[index]);
        index = index + 1u;
    }
    hierarchy[superblock_node_index(super_index)] = accumulated;
}

// The highest order among primitives in blocks strictly before `block` that
// overlap `query`, never below `best`. Prunes spatially (AABB) and temporally
// (settled blocks contribute nothing new).
fn scan_earlier_blocks(block: u32, query: vec4<f32>, best: u32) -> u32 {
    var best = best;
    var super_index: u32 = 0u;
    loop {
        if (super_index >= params.superblock_count) {
            break;
        }
        // Every block under this superblock is at or after `block`, so no
        // earlier primitive remains anywhere above it.
        if (super_index * SUPERBLOCK_SIZE >= block) {
            break;
        }
        // A region that settled last iteration contributes nothing new: its
        // effect is already inside each primitive's current value. One load
        // rejects up to 4,096 primitives.
        if (changed_in[superblock_flag_index(super_index)] != 0u
            && overlaps(hierarchy[superblock_node_index(super_index)], query)) {
            var probe_block = super_index * SUPERBLOCK_SIZE;
            let block_end = min(probe_block + SUPERBLOCK_SIZE, block);
            loop {
                if (probe_block >= block_end) {
                    break;
                }
                if (changed_in[block_flag_index(probe_block)] != 0u
                    && overlaps(hierarchy[probe_block], query)) {
                    var probe = probe_block * BLOCK_SIZE;
                    let probe_end = min(probe + BLOCK_SIZE, params.count);
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
                probe_block = probe_block + 1u;
            }
        }
        super_index = super_index + 1u;
    }
    return best;
}

// One invocation per *block*, walking its 64 primitives in paint order and
// using the values it has just computed for its own earlier members.
//
// This is the second lever on the iteration count, and the larger one. A
// per-primitive kernel advances the overlap chain exactly one primitive per
// iteration, so a layer whose deepest painter order is 577 needs 577 passes.
// Collapsing a block's internal chain inside one invocation advances it by up
// to a whole block instead, which on the measured zoomed-out node graph cut the
// iteration count by more than an order of magnitude. The cost is parallelism:
// this dispatches `primitives / 64` invocations rather than `primitives`, so
// each one does 64 times the work. That trade is only worth making because the
// iteration count was what dominated.
//
// Reading a freshly computed value for a same-block predecessor makes this
// Gauss-Seidel rather than Jacobi. The fixed point is unchanged: the iteration
// is monotone and starts below the fixed point, so any fair update order
// converges to the same least fixed point — the argument is written out in
// `ordering.rs`.
@compute @workgroup_size(64)
fn relax(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let block = global_id.x;
    if (block >= params.block_count) {
        return;
    }
    let first_item = block * BLOCK_SIZE;
    let last_item = min(first_item + BLOCK_SIZE, params.count);

    var resolved_here: array<u32, 64>;
    var moved = false;
    var slot: u32 = 0u;
    loop {
        let target_index = first_item + slot;
        if (target_index >= last_item) {
            break;
        }
        let query = bounds[target_index];
        let current = order_in[target_index];
        var best = scan_earlier_blocks(block, query, 0u);

        var earlier: u32 = 0u;
        loop {
            if (earlier >= slot) {
                break;
            }
            if (overlaps(bounds[first_item + earlier], query)
                && resolved_here[earlier] > best) {
                best = resolved_here[earlier];
            }
            earlier = earlier + 1u;
        }

        // `max` with the current value, not a plain assignment: the scan above
        // saw only the blocks that moved, and everything else's contribution is
        // already in `current`. This is what makes the temporal pruning sound —
        // see this file's header and `ordering.rs`.
        let resolved = max(current, best + 1u);
        resolved_here[slot] = resolved;
        order_out[target_index] = resolved;
        if (resolved != current) {
            moved = true;
        }
        slot = slot + 1u;
    }

    if (moved) {
        atomicAdd(&changed_out[0], 1u);
        atomicStore(&changed_out[block_flag_index(block)], 1u);
        atomicStore(&changed_out[superblock_flag_index(block / SUPERBLOCK_SIZE)], 1u);
    }
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
