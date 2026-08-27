// Indirect draw-arg generation (§5.3): turn Phase 3's per-layer draw
// permutation and cull mask into the arguments a `draw_indirect` reads, plus
// the indirection buffer the instanced pipelines pull through.
//
// This is a transcription of `wgpui_core::indirect::indirect_args`, checked
// against it for exact equality by `wgpui-wgpu/tests/indirect_draw.rs`. Read
// that function's doc for what the computation is and why the indirection
// buffer is arena-shaped; this file is only how it is spelled for the GPU.
//
// Two entry points:
//
//   compact — one workgroup per (layer, kind) slot. Walks the slot's primitives
//             in draw order and packs the survivors into the front of the
//             slot's own indirection range, then writes that slot's argument
//             record. Order-preserving: an unordered atomic append would be
//             shorter and would destroy the painter order the ordering pass
//             just spent a relaxation computing.
//
//   pack    — one workgroup total. Compacts the populated slots' records into
//             `packed_args` and writes how many there are, which is what a
//             `multi_draw_indirect_count` reads. Order-preserving for the same
//             reason one level up: painter order *across* layers is layer
//             order.
//
// Both compactions are the same chunked Hillis-Steele scan over 64 lanes, with
// a running offset carried between chunks. That is what makes them
// order-preserving and still parallel; the alternative that fits in one
// invocation is a serial scan over the whole slot, which on a hundred-thousand-
// primitive layer is a hundred thousand serial steps in one thread.

struct Params {
    // Slots in this dispatch's kind.
    slot_count: u32,
    // 0: every record carries first_instance = 0 (README's "Custom Device
    // Gotcha" — the default). 1: the record carries the slot's arena base,
    // which is what a multi_draw_indirect needs and what
    // INDIRECT_FIRST_INSTANCE permits.
    first_instance_mode: u32,
    // Vertices one instance draws.
    vertex_count: u32,
    // Slots in the kind's arena, i.e. the length of `visible`.
    arena_slots: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
// [base, count, 0, 0] per slot.
@group(0) @binding(1) var<storage, read> slots: array<vec4<u32>>;
// Arena-shaped. `draw_order[base + position]` is a layer-local index.
@group(0) @binding(2) var<storage, read> draw_order: array<u32>;
// Arena-shaped. 1 = the occlusion pass dropped this primitive.
@group(0) @binding(3) var<storage, read> culled: array<u32>;
// Arena-shaped indirection buffer.
@group(0) @binding(4) var<storage, read_write> visible: array<u32>;
// One record per slot.
@group(0) @binding(5) var<storage, read_write> args: array<vec4<u32>>;
// The populated records, packed.
@group(0) @binding(6) var<storage, read_write> packed_args: array<vec4<u32>>;
// One word: how many entries `packed_args` holds.
@group(0) @binding(7) var<storage, read_write> draw_count: array<u32>;

const BLOCK: u32 = 64u;

// `wgpui_core::indirect::UNUSED_INSTANCE`. Not zero — zero is a real arena
// slot, so a shader that read past a slot's instance_count would silently draw
// primitive 0 rather than produce something obviously wrong.
const UNUSED_INSTANCE: u32 = 0xffffffffu;

var<workgroup> scan: array<u32, 64>;

// Inclusive prefix sum of `value` across the workgroup, leaving the total in
// `scan[BLOCK - 1u]`. Every barrier below is in uniform control flow: `lane` is
// the only non-uniform value and it never gates a barrier.
fn inclusive_scan(lane: u32, value: u32) -> u32 {
    scan[lane] = value;
    workgroupBarrier();
    for (var offset: u32 = 1u; offset < BLOCK; offset = offset * 2u) {
        var addend: u32 = 0u;
        if (lane >= offset) {
            addend = scan[lane - offset];
        }
        workgroupBarrier();
        scan[lane] = scan[lane] + addend;
        workgroupBarrier();
    }
    return scan[lane];
}

@compute @workgroup_size(64)
fn compact(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let slot_index = group.x;
    // Uniform across the workgroup, so returning here does not strand a barrier.
    if (slot_index >= params.slot_count) {
        return;
    }
    let slot = slots[slot_index];
    let base = slot.x;
    let count = slot.y;
    let lane = local.x;

    var written: u32 = 0u;
    var chunk: u32 = 0u;
    loop {
        if (chunk >= count) {
            break;
        }
        let position = chunk + lane;
        var keep: u32 = 0u;
        var arena_index: u32 = 0u;
        if (position < count) {
            let local_index = draw_order[base + position];
            // The ordering pass pads its sort network past the primitive count
            // and the padding reads back as u32::MAX; a slot whose range was
            // copied in wholesale can carry one, and it must not become an
            // instance.
            if (local_index < count) {
                arena_index = base + local_index;
                if (culled[arena_index] == 0u) {
                    keep = 1u;
                }
            }
        }
        let inclusive = inclusive_scan(lane, keep);
        let total = scan[BLOCK - 1u];
        if (keep == 1u) {
            visible[base + written + inclusive - 1u] = arena_index;
        }
        // Before the next chunk overwrites `scan`, every lane must have read
        // its own `inclusive` and the shared `total` out of it.
        workgroupBarrier();
        written = written + total;
        chunk = chunk + BLOCK;
    }

    if (lane == 0u) {
        var first_instance: u32 = 0u;
        if (params.first_instance_mode == 1u) {
            first_instance = base;
        }
        args[slot_index] = vec4<u32>(params.vertex_count, written, 0u, first_instance);
    }
}

// Fill a slot's unused indirection entries with UNUSED_INSTANCE, so a stale
// value from a previous frame can never be mistaken for a live instance.
// Separate from `compact` because it covers the whole arena rather than one
// slot's live prefix, and because it must run *before* compaction rather than
// after it.
@compute @workgroup_size(64)
fn clear_visible(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.arena_slots) {
        return;
    }
    visible[id.x] = UNUSED_INSTANCE;
}

@compute @workgroup_size(64)
fn pack(@builtin(local_invocation_id) local: vec3<u32>) {
    let lane = local.x;
    var written: u32 = 0u;
    var chunk: u32 = 0u;
    loop {
        if (chunk >= params.slot_count) {
            break;
        }
        let index = chunk + lane;
        var keep: u32 = 0u;
        var record: vec4<u32> = vec4<u32>(0u, 0u, 0u, 0u);
        if (index < params.slot_count) {
            record = args[index];
            if (record.x != 0u && record.y != 0u) {
                keep = 1u;
            }
        }
        let inclusive = inclusive_scan(lane, keep);
        let total = scan[BLOCK - 1u];
        if (keep == 1u) {
            packed_args[written + inclusive - 1u] = record;
        }
        workgroupBarrier();
        written = written + total;
        chunk = chunk + BLOCK;
    }
    if (lane == 0u) {
        draw_count[0] = written;
    }
}
