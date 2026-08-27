// Instance-tier occlusion culling for one layer, as a compute pass.
// docs/gpu-native-architecture.md §5.2; R-N §8.1's fine tier and §8.3's
// conservative opaque-region rules.
//
// This is a transcription of `occlusion.rs`'s `keep_item` and
// `occlusion/coverage.rs`'s `fully_covered`, statement for statement — not a
// GPU-flavoured reimplementation. §5.2's claim is that the compute path is "the
// same computation, restated as data-parallel," and the differential harness
// asserts that by comparing this shader's output to those functions' output for
// exact equality, so any divergence here is a bug rather than a tolerance.
//
// The two structural differences from the CPU reference are both forced by the
// address space and are mirrored back into the CPU reference so the two still
// agree exactly:
//
//   * The occluder set is capped at MAX_OCCLUDERS (a shader has no growable
//     per-invocation storage). Dropping occluders can only miss a cull.
//   * Candidates are gathered by walking forward from the target rather than by
//     accumulating a set backwards, because every invocation decides its own
//     item with no state carried from any other.
//
// R-N §8.4's constraint is structural here rather than enforced: this pass is
// handed geometry and returns a mask. It has no access to hitboxes, dispatch
// nodes, or layout, so culling cannot suppress anything but `DISPLAY` work.

const BLOCK_SIZE: u32 = 64u;
const SUPERBLOCK_SIZE: u32 = 64u;
const MAX_OCCLUDERS: u32 = 32u;
const MAX_EDGES: u32 = 66u;

const FLAG_CULLABLE: u32 = 1u;
const FLAG_PROTECTED: u32 = 2u;
const FLAG_HAS_OPAQUE: u32 = 4u;

struct Params {
    // Primitives in this layer.
    count: u32,
    // Backdrop-filter / filter-group regions, already dilated by blur radius.
    poison_count: u32,
    // ceil(count / BLOCK_SIZE).
    block_count: u32,
    // ceil(block_count / SUPERBLOCK_SIZE).
    superblock_count: u32,
};

struct Item {
    // Bounds intersected with the content mask.
    visible: vec4<f32>,
    // Conservative opaque region; meaningful only with FLAG_HAS_OPAQUE.
    opaque: vec4<f32>,
    flags: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

struct Poison {
    region: vec4<f32>,
    // Poisons every item strictly below this index.
    above_index: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> items: array<Item>;
@group(0) @binding(2) var<storage, read> poison: array<Poison>;
@group(0) @binding(3) var<storage, read_write> blocks: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> superblocks: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> culled: array<u32>;

// Per-invocation scratch. `array<vec4<f32>, 32>` plus two float arrays is about
// a kilobyte per invocation; it lives in each thread's private space, which is
// why MAX_OCCLUDERS is a tuning knob rather than an arbitrary bound.
var<private> occluders: array<vec4<f32>, 32>;
var<private> edges: array<f32, 66>;
var<private> tops: array<f32, 32>;
var<private> bottoms: array<f32, 32>;

fn is_empty(region: vec4<f32>) -> bool {
    return !(region.z > region.x) || !(region.w > region.y);
}

fn intersect(left: vec4<f32>, right: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        max(left.x, right.x),
        max(left.y, right.y),
        min(left.z, right.z),
        min(left.w, right.w),
    );
}

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
    var accumulated = items[first].visible;
    var index = first + 1u;
    loop {
        if (index >= last) {
            break;
        }
        accumulated = union_of(accumulated, items[index].visible);
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

// Whether a filter declared above `index` reads through `region`.
fn is_poisoned(index: u32, region: vec4<f32>) -> bool {
    var zone: u32 = 0u;
    loop {
        if (zone >= params.poison_count) {
            break;
        }
        let entry = poison[zone];
        if (entry.above_index > index && !is_empty(intersect(entry.region, region))) {
            return true;
        }
        zone = zone + 1u;
    }
    return false;
}

// `coverage::fully_covered`, over the first `gathered` entries of `occluders`.
// Clips them to the target in place first, exactly as the CPU reference clips
// into its own array before building the edge list.
fn fully_covered(target: vec4<f32>, gathered: u32) -> bool {
    if (is_empty(target)) {
        return true;
    }

    var clipped_count: u32 = 0u;
    var index: u32 = 0u;
    loop {
        if (index >= gathered) {
            break;
        }
        let overlap = intersect(occluders[index], target);
        if (!is_empty(overlap)) {
            // Compaction is safe in place: clipped_count never exceeds index.
            occluders[clipped_count] = overlap;
            clipped_count = clipped_count + 1u;
        }
        index = index + 1u;
    }
    if (clipped_count == 0u) {
        return false;
    }

    var edge_count: u32 = 0u;
    edges[edge_count] = target.x;
    edge_count = edge_count + 1u;
    edges[edge_count] = target.z;
    edge_count = edge_count + 1u;
    index = 0u;
    loop {
        if (index >= clipped_count) {
            break;
        }
        edges[edge_count] = occluders[index].x;
        edge_count = edge_count + 1u;
        edges[edge_count] = occluders[index].z;
        edge_count = edge_count + 1u;
        index = index + 1u;
    }

    var sorted: u32 = 1u;
    loop {
        if (sorted >= edge_count) {
            break;
        }
        let value = edges[sorted];
        var position: u32 = sorted;
        loop {
            if (position == 0u) {
                break;
            }
            if (edges[position - 1u] <= value) {
                break;
            }
            edges[position] = edges[position - 1u];
            position = position - 1u;
        }
        edges[position] = value;
        sorted = sorted + 1u;
    }

    var covered = true;
    var slice: u32 = 0u;
    loop {
        if (slice + 1u >= edge_count) {
            break;
        }
        let left = edges[slice];
        let right = edges[slice + 1u];
        slice = slice + 1u;
        if (right <= left) {
            continue;
        }
        // Spelled as the CPU reference spells it, not as (left + right) / 2:
        // the two differ in the last bit for large coordinates.
        let midpoint = left + (right - left) / 2.0;

        var interval_count: u32 = 0u;
        var probe: u32 = 0u;
        loop {
            if (probe >= clipped_count) {
                break;
            }
            let region = occluders[probe];
            if (region.x <= midpoint && region.z >= midpoint) {
                tops[interval_count] = region.y;
                bottoms[interval_count] = region.w;
                interval_count = interval_count + 1u;
            }
            probe = probe + 1u;
        }

        var ordered: u32 = 1u;
        loop {
            if (ordered >= interval_count) {
                break;
            }
            let top = tops[ordered];
            let bottom = bottoms[ordered];
            var position: u32 = ordered;
            loop {
                if (position == 0u) {
                    break;
                }
                if (tops[position - 1u] <= top) {
                    break;
                }
                tops[position] = tops[position - 1u];
                bottoms[position] = bottoms[position - 1u];
                position = position - 1u;
            }
            tops[position] = top;
            bottoms[position] = bottom;
            ordered = ordered + 1u;
        }

        var covered_to = target.y;
        var walked: u32 = 0u;
        loop {
            if (walked >= interval_count) {
                break;
            }
            let top = tops[walked];
            let bottom = bottoms[walked];
            walked = walked + 1u;
            if (top > covered_to) {
                break;
            }
            if (bottom > covered_to) {
                covered_to = bottom;
            }
            if (covered_to >= target.w) {
                break;
            }
        }
        if (covered_to < target.w) {
            covered = false;
            break;
        }
    }
    return covered;
}

// Collect at most MAX_OCCLUDERS qualifying occluders painted above
// `target_index`, in ascending paint order, and return how many. The ascending
// walk is what makes the cap deterministic and therefore comparable against the
// CPU reference, which caps the same way.
fn gather_occluders(target_index: u32, target: vec4<f32>) -> u32 {
    var gathered: u32 = 0u;
    var done = false;
    var super_index = target_index / (SUPERBLOCK_SIZE * BLOCK_SIZE);
    loop {
        if (done || super_index >= params.superblock_count) {
            break;
        }
        if (overlaps(superblocks[super_index], target)) {
            var block = super_index * SUPERBLOCK_SIZE;
            let block_end = min(block + SUPERBLOCK_SIZE, params.block_count);
            loop {
                if (done || block >= block_end) {
                    break;
                }
                if (overlaps(blocks[block], target)) {
                    let first = block * BLOCK_SIZE;
                    var probe = max(first, target_index + 1u);
                    let probe_end = min(first + BLOCK_SIZE, params.count);
                    loop {
                        if (probe >= probe_end) {
                            break;
                        }
                        if (gathered >= MAX_OCCLUDERS) {
                            done = true;
                            break;
                        }
                        let candidate = items[probe];
                        let usable = (candidate.flags & FLAG_HAS_OPAQUE) != 0u
                            && (candidate.flags & FLAG_PROTECTED) == 0u
                            && !is_poisoned(probe, candidate.visible)
                            && !is_empty(intersect(candidate.opaque, target));
                        if (usable) {
                            occluders[gathered] = candidate.opaque;
                            gathered = gathered + 1u;
                        }
                        probe = probe + 1u;
                    }
                }
                block = block + 1u;
            }
        }
        super_index = super_index + 1u;
    }
    return gathered;
}

@compute @workgroup_size(64)
fn cull(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let target_index = global_id.x;
    if (target_index >= params.count) {
        return;
    }
    let item = items[target_index];
    if ((item.flags & FLAG_CULLABLE) == 0u
        || (item.flags & FLAG_PROTECTED) != 0u
        || is_empty(item.visible)
        || is_poisoned(target_index, item.visible))
    {
        culled[target_index] = 0u;
        return;
    }
    let gathered = gather_occluders(target_index, item.visible);
    if (fully_covered(item.visible, gathered)) {
        culled[target_index] = 1u;
    } else {
        culled[target_index] = 0u;
    }
}
