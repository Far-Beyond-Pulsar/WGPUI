//! §4.3's tile-visibility pass, checked against its CPU reference and driven
//! through Phase 4's indirect-arg mechanism end to end.
//! See docs/gpu-native-architecture.md §4.3, §8 Phase 4.5.
//!
//! Three claims, in the order they build on each other:
//!
//! 1. **The transcription is exact.** `tile_visibility.wgsl` and
//!    `wgpui_core::scene::tile_visibility` agree slot-for-slot and flag-for-flag
//!    over a scripted pan — Phase 3's discipline, unchanged.
//! 2. **The GPU-written slot table drives Phase 4's mechanism unmodified.** The
//!    argument records generated from it equal the ones
//!    `wgpui_core::indirect::indirect_args` computes for the same tiles, with no
//!    slot table ever reaching the CPU in between.
//! 3. **An out-of-range tile issues a draw for zero instances**, which is how a
//!    tile stops being drawn without anything new deciding that it should.

use wgpui_core::geometry::Rect;
use wgpui_core::indirect::{
    DrawSlot, FirstInstance, QUAD_VERTEX_COUNT, UNUSED_INSTANCE, indirect_args,
};
use wgpui_core::patch::primitive::PrimitiveKind;
use wgpui_core::scene::layer::LayerId;
use wgpui_core::scene::{TileCoord, TileDescriptor, TileGrid, encode_tiles, tile_visibility};

use wgpui_wgpu::render::compute::indirect_args_pass::{IndirectArgsBuffers, IndirectArgsPass};
use wgpui_wgpu::render::compute::tile_visibility_pass::{
    ArgsTarget, TileViewport, TileVisibilityBuffers, TileVisibilityPass,
};
use wgpui_wgpu::render::device::context_or_report;

const TILE_EDGE: f32 = 256.0;
const RETAIN_RADIUS: u32 = 1;
/// Slots one tile reserves in the arena. Uniform across tiles here so the
/// arena's shape is obvious in a failing assertion; nothing in the pass depends
/// on it being uniform.
const PER_TILE: u32 = 8;

fn grid() -> TileGrid {
    TileGrid::square(TILE_EDGE).expect("256px is a usable tile edge")
}

/// A resident tile set spanning both signs, so negative coordinates are
/// exercised rather than assumed to work — they are bit-cast through a `u32` on
/// the way to the shader, which is precisely where a sign would be lost.
fn resident_tiles() -> Vec<TileDescriptor> {
    let mut tiles = Vec::new();
    let mut base = 0u32;
    for y in -3..=3i32 {
        for x in -3..=3i32 {
            tiles.push(TileDescriptor {
                coord: TileCoord::new(x, y),
                base,
                count: PER_TILE,
            });
            base += PER_TILE;
        }
    }
    tiles
}

fn arena_slots(tiles: &[TileDescriptor]) -> u32 {
    tiles
        .iter()
        .map(|tile| tile.base + tile.count)
        .max()
        .unwrap_or(0)
}

/// A pan script that crosses tile boundaries in both axes and both directions.
fn pan_script() -> Vec<[f32; 2]> {
    let mut script = vec![[0.0, 0.0]];
    for step in 1..=10i32 {
        script.push([-(step as f32) * 97.0, 0.0]);
    }
    for step in 1..=10i32 {
        script.push([-970.0, -(step as f32) * 61.0]);
    }
    for step in 1..=10i32 {
        script.push([-970.0 + step as f32 * 143.0, -610.0 + step as f32 * 89.0]);
    }
    script
}

fn viewport() -> Rect {
    Rect::from_origin_size([0.0, 0.0], [900.0, 620.0])
}

fn tile_viewport(pan: [f32; 2]) -> TileViewport {
    let view = viewport();
    TileViewport {
        tile_size: [TILE_EDGE, TILE_EDGE],
        pan,
        viewport: [view.min_x, view.min_y, view.max_x, view.max_y],
        retain_radius: RETAIN_RADIUS,
    }
}

#[test]
fn the_tile_visibility_transcription_matches_its_cpu_reference_over_a_pan() {
    let Some(context) = context_or_report("tile visibility differential") else {
        return;
    };
    let pass = TileVisibilityPass::new(&context.device);
    let tiles = resident_tiles();
    let mut tile_bytes = Vec::new();
    encode_tiles(&tiles, &mut tile_bytes);
    let buffers = TileVisibilityBuffers::new(&context.device, tiles.len() as u32);
    let grid = grid();
    let slots = arena_slots(&tiles);

    let mut visible_seen = 0usize;
    let mut hidden_seen = 0usize;
    for pan in pan_script() {
        let tile_count = pass
            .run(
                &context.device,
                &context.queue,
                &buffers,
                &tile_bytes,
                tile_viewport(pan),
                slots,
            )
            .expect("the dispatch must succeed");
        assert_eq!(tile_count, tiles.len() as u32);

        let content_viewport = TileGrid::content_viewport(
            viewport(),
            wgpui_core::scene::layer::LayerTransform::translated(pan[0], pan[1]),
        );
        let reference = tile_visibility(&grid, &tiles, content_viewport, RETAIN_RADIUS);

        let gpu_mask = pass
            .read_in_range(&context.device, &context.queue, &buffers, tile_count)
            .expect("the mask reads back");
        let gpu_slots = pass
            .read_slots(&context.device, &context.queue, &buffers, tile_count)
            .expect("the slots read back");

        for (index, tile) in tiles.iter().enumerate() {
            assert_eq!(
                gpu_mask.get(index).copied(),
                reference.in_range.get(index).copied(),
                "pan {pan:?} disagreed about tile {:?}",
                tile.coord
            );
        }
        assert_eq!(
            gpu_slots, reference.slots,
            "pan {pan:?} produced a different slot table"
        );

        visible_seen += reference.visible_count();
        hidden_seen += tiles.len() - reference.visible_count();
    }

    // A differential where every tile was always in range, or never was, would
    // agree perfectly and prove nothing.
    assert!(
        visible_seen > 0 && hidden_seen > 0,
        "the pan script never exercised both answers: {visible_seen} visible, \
         {hidden_seen} hidden"
    );
}

/// The claim that makes this a *reuse* of Phase 4 rather than a lookalike: the
/// arguments generated from the GPU-written slot table are the ones the CPU
/// reference computes for the same tiles — and the CPU never saw the table.
#[test]
fn arguments_generated_from_the_gpu_written_slot_table_match_the_cpu_reference() {
    let Some(context) = context_or_report("tile visibility into indirect args") else {
        return;
    };
    let visibility = TileVisibilityPass::new(&context.device);
    let indirect = IndirectArgsPass::new(&context.device);
    let tiles = resident_tiles();
    let slots = arena_slots(&tiles);
    let mut tile_bytes = Vec::new();
    encode_tiles(&tiles, &mut tile_bytes);

    let visibility_buffers = TileVisibilityBuffers::new(&context.device, tiles.len() as u32);
    let args_buffers = IndirectArgsBuffers::new(&context.device, slots, tiles.len() as u32 + 1);

    // Identity draw order over the arena, nothing culled: this test is about the
    // slot table's route to the arguments, and Phase 3's own results are already
    // differentiated in `indirect_args_differential.rs`.
    let draw_order: Vec<u8> = tiles
        .iter()
        .flat_map(|tile| 0..tile.count)
        .flat_map(|local| local.to_le_bytes())
        .collect();
    context
        .queue
        .write_buffer(&args_buffers.draw_order, 0, &draw_order);
    context
        .queue
        .write_buffer(&args_buffers.culled, 0, &vec![0u8; slots as usize * 4][..]);

    let grid = grid();
    let mut any_empty = false;
    let mut any_populated = false;
    for pan in pan_script() {
        let output = visibility
            .run_into_args(
                &context.device,
                &context.queue,
                &visibility_buffers,
                ArgsTarget {
                    pass: &indirect,
                    buffers: &args_buffers,
                    vertex_count: QUAD_VERTEX_COUNT,
                    first_instance: FirstInstance::Zero,
                },
                &tile_bytes,
                tile_viewport(pan),
            )
            .expect("the two dispatches must succeed");
        assert_eq!(output.slot_count, tiles.len() as u32);

        let content_viewport = TileGrid::content_viewport(
            viewport(),
            wgpui_core::scene::layer::LayerTransform::translated(pan[0], pan[1]),
        );
        let reference_visibility = tile_visibility(&grid, &tiles, content_viewport, RETAIN_RADIUS);
        let reference_slots: Vec<DrawSlot> = reference_visibility
            .slots
            .iter()
            .enumerate()
            .map(|(index, slot)| DrawSlot {
                layer: LayerId::from_raw(index as u64 + 1),
                kind: PrimitiveKind::Quad,
                base: slot[0],
                count: slot[1],
            })
            .collect();
        let order: Vec<u32> = tiles.iter().flat_map(|tile| 0..tile.count).collect();
        let reference = indirect_args(
            &reference_slots,
            &order,
            &vec![0u32; slots as usize],
            slots as usize,
            QUAD_VERTEX_COUNT,
            FirstInstance::Zero,
        );

        let gpu_args = indirect
            .read_args(
                &context.device,
                &context.queue,
                &args_buffers,
                output.slot_count,
            )
            .expect("the arguments read back");
        assert_eq!(
            gpu_args, reference.args,
            "pan {pan:?} generated different draw arguments"
        );

        let gpu_visible = indirect
            .read_visible(&context.device, &context.queue, &args_buffers)
            .expect("the indirection buffer reads back");
        assert_eq!(
            gpu_visible, reference.visible,
            "pan {pan:?} produced a different indirection buffer"
        );

        // §4.3's actual promise, in the form a draw call sees it: an
        // out-of-range tile's record draws nothing, and its indirection range
        // holds no instances at all.
        for (index, tile) in tiles.iter().enumerate() {
            let in_range = reference_visibility.in_range.get(index).copied() == Some(1);
            let record = gpu_args.get(index).copied().unwrap_or_default();
            if in_range {
                any_populated = true;
                assert_eq!(
                    record.instance_count, tile.count,
                    "an in-range tile must draw its whole reservation"
                );
            } else {
                any_empty = true;
                assert_eq!(
                    record.instance_count, 0,
                    "tile {:?} is out of range and must draw zero instances",
                    tile.coord
                );
                let start = tile.base as usize;
                let end = start + tile.count as usize;
                assert!(
                    gpu_visible
                        .get(start..end)
                        .unwrap_or(&[])
                        .iter()
                        .all(|entry| *entry == UNUSED_INSTANCE),
                    "tile {:?} is out of range but left live instances in its \
                     indirection range",
                    tile.coord
                );
            }
        }
    }
    assert!(
        any_empty && any_populated,
        "the script never produced both a drawn and an undrawn tile"
    );
}

/// A bookkeeping bug must be reported rather than dispatched with — the same
/// rule `IndirectArgsPass::run` holds to, moved onto the input the CPU still
/// owns once the slot table is written on the device.
#[test]
fn a_tile_reserving_past_the_arena_is_refused_rather_than_dispatched() {
    let Some(context) = context_or_report("tile visibility validation") else {
        return;
    };
    let pass = TileVisibilityPass::new(&context.device);
    let buffers = TileVisibilityBuffers::new(&context.device, 4);
    let mut bytes = Vec::new();
    encode_tiles(
        &[TileDescriptor {
            coord: TileCoord::ORIGIN,
            base: 60,
            count: 8,
        }],
        &mut bytes,
    );
    assert!(
        pass.run(
            &context.device,
            &context.queue,
            &buffers,
            &bytes,
            tile_viewport([0.0, 0.0]),
            64,
        )
        .is_err(),
        "a tile ending at slot 68 in a 64-slot arena must be refused"
    );

    // And a malformed table, for the same reason.
    assert!(
        pass.run(
            &context.device,
            &context.queue,
            &buffers,
            &[0u8; 7],
            tile_viewport([0.0, 0.0]),
            64,
        )
        .is_err()
    );

    // More tiles than the buffers hold is a caller bug too, not a silent
    // truncation of the visible set.
    let mut many = Vec::new();
    encode_tiles(
        &(0..16i32)
            .map(|x| TileDescriptor {
                coord: TileCoord::new(x, 0),
                base: 0,
                count: 1,
            })
            .collect::<Vec<_>>(),
        &mut many,
    );
    assert!(
        pass.run(
            &context.device,
            &context.queue,
            &buffers,
            &many,
            tile_viewport([0.0, 0.0]),
            64,
        )
        .is_err()
    );
}

/// A boundary with no resident tiles must dispatch nothing and not fail — the
/// first frame of any tiled boundary, before anything has been revealed.
#[test]
fn a_boundary_with_no_resident_tiles_is_inert() {
    let Some(context) = context_or_report("tile visibility empty") else {
        return;
    };
    let pass = TileVisibilityPass::new(&context.device);
    let buffers = TileVisibilityBuffers::new(&context.device, 1);
    let count = pass
        .run(
            &context.device,
            &context.queue,
            &buffers,
            &[],
            tile_viewport([0.0, 0.0]),
            0,
        )
        .expect("an empty tile set is not an error");
    assert_eq!(count, 0);
}
