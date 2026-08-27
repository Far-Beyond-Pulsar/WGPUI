//! Dispatches `wgpui_core::shaders::TILE_VISIBILITY_WGSL` (§4.3).
//! See docs/gpu-native-architecture.md §4.3, §8 Phase 4.5.
//!
//! # What this pass produces, and what it deliberately is not
//!
//! One `[base, count, 0, 0]` slot record per resident tile, with the count
//! zeroed for tiles outside (viewport ∪ retain radius). That record layout is
//! not a coincidence and not a convention this file invented — it is exactly
//! [`wgpui_core::indirect::encode_slots`]' layout, the input
//! `indirect_args.wgsl`'s `compact` already reads.
//!
//! So this pass has no draw path, no pipeline, and no output buffer of its own
//! beyond the one it hands straight to
//! [`IndirectArgsPass::run_with_slots`]. An out-of-range tile becomes a
//! zero-instance argument record through Phase 4's machinery untouched, and
//! `pack` drops it from a `multi_draw_indirect_count` entirely. §4.3 claims
//! tiling "needs almost no new machinery"; this file is how much that turned out
//! to be.
//!
//! # The computation is written once, in Rust
//!
//! [`wgpui_core::scene::tile::tile_visibility`] is the reference and
//! `tile_visibility.wgsl` is its transcription, compared for exact equality by
//! `tests/tile_visibility.rs`. Phase 3's discipline unchanged, and for the same
//! reason: there is no CPU-side result to eyeball.
//!
//! # Where the CPU is still involved, stated rather than glossed
//!
//! §4.3's wording is that "the CPU never enumerates tile candidates," and on the
//! draw path that is exactly true — the CPU hands over resident tile descriptors
//! and never learns which of them drew. It is not true of *residency*, and
//! cannot be: a newly-revealed tile has to have content rendered into it, which
//! is CPU work no visibility kernel can do. `wgpui_core`'s `TileResidency` runs
//! the same predicate on the CPU for the tiles it keeps resident. The two are
//! the same rule, checked against each other, and `docs/phase-4.5-results.md`
//! says so rather than letting the shader's existence imply otherwise.

use std::num::NonZeroU64;

use wgpui_core::indirect::{DRAW_SLOT_STRIDE, FirstInstance};
use wgpui_core::ordering::BLOCK_SIZE;
use wgpui_core::scene::TILE_DESCRIPTOR_STRIDE;

use crate::render::compute::indirect_args_pass::{
    GeneratedSlots, IndirectArgsBuffers, IndirectArgsOutput, IndirectArgsPass,
};
use crate::render::readback::{ReadbackError, read_u32_buffer};

/// Bytes the shader's `Params` uniform occupies.
const PARAMS_SIZE: u64 = 48;

/// Why a tile-visibility dispatch failed.
#[derive(Debug)]
pub enum TileVisibilityError {
    /// The encoded descriptor table's length is not a whole number of tiles.
    MalformedTiles {
        /// Bytes supplied.
        bytes: usize,
        /// Bytes one descriptor occupies.
        stride: usize,
    },
    /// A tile's reservation runs past the arena it claims to be in.
    ///
    /// [`IndirectArgsPass::run`] makes this check against a CPU-uploaded slot
    /// table; the whole point of this pass is that the slot table is written on
    /// the device, so the check moves here, onto the descriptors the caller
    /// *did* write. Dropping it rather than moving it would hand
    /// `copy_buffer_to_buffer` and the compaction an out-of-range base, which is
    /// an uncaptured device error and by default aborts the process.
    TileOutsideArena {
        /// Which descriptor.
        index: usize,
        /// Its last slot index.
        end: u64,
        /// Slots the arena holds.
        arena_slots: u32,
    },
    /// More tiles than the argument buffers are sized for.
    TooManyTiles {
        /// Descriptors supplied.
        tiles: u32,
        /// Slots the buffers hold.
        capacity: u32,
    },
    /// A readback failed.
    Readback(ReadbackError),
}

impl std::fmt::Display for TileVisibilityError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TileVisibilityError::MalformedTiles { bytes, stride } => write!(
                formatter,
                "tile descriptor table is {bytes} bytes, not a multiple of the \
                 {stride}-byte stride"
            ),
            TileVisibilityError::TileOutsideArena {
                index,
                end,
                arena_slots,
            } => write!(
                formatter,
                "tile {index} ends at arena slot {end}, past the arena's {arena_slots}"
            ),
            TileVisibilityError::TooManyTiles { tiles, capacity } => write!(
                formatter,
                "{tiles} resident tiles against argument buffers sized for {capacity} slots"
            ),
            TileVisibilityError::Readback(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for TileVisibilityError {}

impl From<ReadbackError> for TileVisibilityError {
    fn from(error: ReadbackError) -> Self {
        TileVisibilityError::Readback(error)
    }
}

/// Where a tiled boundary's content sits relative to its window, this frame.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TileViewport {
    /// Tile edge lengths in logical pixels.
    pub tile_size: [f32; 2],
    /// The boundary's layer transform — where its content composites.
    pub pan: [f32; 2],
    /// The boundary's visible rectangle in window space: min, then max.
    pub viewport: [f32; 4],
    /// How many tiles beyond the viewport stay in range.
    pub retain_radius: u32,
}

/// The buffers one tile-visibility dispatch writes.
///
/// Held across frames by the caller for the same reason
/// [`IndirectArgsBuffers`] is: the resident tile set changes slowly, and
/// reallocating per frame would put an allocation in front of a pass whose whole
/// purpose is that the CPU stops doing per-frame work.
pub struct TileVisibilityBuffers {
    /// `[base, count, 0, 0]` per tile — the slot table
    /// [`IndirectArgsPass::run_with_slots`] consumes.
    pub slots: wgpu::Buffer,
    /// `1` in range, `0` out, per tile.
    pub in_range: wgpu::Buffer,
    /// Tiles these buffers are sized for.
    pub tile_capacity: u32,
}

impl TileVisibilityBuffers {
    /// Allocate for at most `tile_capacity` resident tiles.
    pub fn new(device: &wgpu::Device, tile_capacity: u32) -> TileVisibilityBuffers {
        let capacity = tile_capacity.max(1);
        TileVisibilityBuffers {
            slots: storage_buffer(
                device,
                "tile visibility slots",
                u64::from(capacity) * DRAW_SLOT_STRIDE as u64,
            ),
            in_range: storage_buffer(device, "tile visibility mask", u64::from(capacity) * 4),
            tile_capacity: capacity,
        }
    }

    /// Whether these buffers can serve `tiles` tiles without being reallocated.
    pub fn fits(&self, tiles: u32) -> bool {
        self.tile_capacity >= tiles
    }
}

/// Where a tile-visibility result turns into draw arguments: the pass that does
/// it, the buffers it writes, and the encoding it writes them in.
///
/// One type rather than four parameters because all four are properties of the
/// same argument-generation stage, and the two that are not obviously paired
/// still are: `vertex_count` and `first_instance` together *are* the record
/// encoding, and splitting them across a call site is how a record ends up
/// carrying a base the shader is not expecting (`indirect.rs`'s
/// `FirstInstance` doc has the full story). Also what keeps
/// [`TileVisibilityPass::run_into_args`] inside clippy's argument limit without
/// a suppression, which is the same finding and the same resolution Phases 3
/// and 4 both recorded.
#[derive(Copy, Clone)]
pub struct ArgsTarget<'a> {
    /// The argument-generation pass.
    pub pass: &'a IndirectArgsPass,
    /// Its arena-shaped buffers.
    pub buffers: &'a IndirectArgsBuffers,
    /// Vertices one instance draws.
    pub vertex_count: u32,
    /// Where each record's base index is carried.
    pub first_instance: FirstInstance,
}

/// The compiled tile-visibility pipeline. Build once, dispatch once per tiled
/// boundary per frame.
pub struct TileVisibilityPass {
    layout: wgpu::BindGroupLayout,
    visibility: wgpu::ComputePipeline,
}

impl TileVisibilityPass {
    /// Compile `tile_visibility.wgsl`.
    pub fn new(device: &wgpu::Device) -> TileVisibilityPass {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wgpui tile visibility"),
            source: wgpu::ShaderSource::Wgsl(wgpui_core::shaders::TILE_VISIBILITY_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tile visibility"),
            entries: &[
                uniform_entry(0, PARAMS_SIZE),
                storage_entry(1, true),
                storage_entry(2, false),
                storage_entry(3, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tile visibility"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        TileVisibilityPass {
            visibility: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("tile_visibility"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("tile_visibility"),
                compilation_options: Default::default(),
                cache: None,
            }),
            layout,
        }
    }

    /// Decide which tiles are in range and write their draw slots.
    ///
    /// `tiles` is [`wgpui_core::scene::encode_tiles`]' output for the boundary's
    /// resident tile set. Returns nothing about which tiles drew — that is the
    /// point, and it is the same shape as
    /// [`crate::render::draw::DrawStats::instances_known_to_cpu`] being `None`
    /// on every indirect path.
    pub fn run(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffers: &TileVisibilityBuffers,
        tiles: &[u8],
        viewport: TileViewport,
        arena_slots: u32,
    ) -> Result<u32, TileVisibilityError> {
        if !tiles.len().is_multiple_of(TILE_DESCRIPTOR_STRIDE) {
            return Err(TileVisibilityError::MalformedTiles {
                bytes: tiles.len(),
                stride: TILE_DESCRIPTOR_STRIDE,
            });
        }
        let tile_count = u32::try_from(tiles.len() / TILE_DESCRIPTOR_STRIDE).unwrap_or(u32::MAX);
        if !buffers.fits(tile_count) {
            return Err(TileVisibilityError::TooManyTiles {
                tiles: tile_count,
                capacity: buffers.tile_capacity,
            });
        }
        // The validation `IndirectArgsPass::run` does on a CPU-uploaded slot
        // table, moved onto the input the CPU still owns — see
        // `TileVisibilityError::TileOutsideArena`.
        for (index, tile) in tiles.chunks_exact(TILE_DESCRIPTOR_STRIDE).enumerate() {
            let base = u64::from(read_u32(tile, 8));
            let count = u64::from(read_u32(tile, 12));
            if base + count > u64::from(arena_slots) {
                return Err(TileVisibilityError::TileOutsideArena {
                    index,
                    end: base + count,
                    arena_slots,
                });
            }
        }

        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile visibility params"),
            size: PARAMS_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut params_bytes = Vec::with_capacity(PARAMS_SIZE as usize);
        for value in [
            viewport.tile_size[0],
            viewport.tile_size[1],
            viewport.pan[0],
            viewport.pan[1],
        ] {
            params_bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in viewport.viewport {
            params_bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in [tile_count, viewport.retain_radius, 0, 0] {
            params_bytes.extend_from_slice(&value.to_le_bytes());
        }
        queue.write_buffer(&params, 0, &params_bytes);

        let tile_buffer = storage_buffer(
            device,
            "tile descriptors",
            (tiles.len() as u64).max(TILE_DESCRIPTOR_STRIDE as u64),
        );
        if !tiles.is_empty() {
            queue.write_buffer(&tile_buffer, 0, tiles);
        }

        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tile visibility"),
            layout: &self.layout,
            entries: &[
                binding(0, &params),
                binding(1, &tile_buffer),
                binding(2, &buffers.slots),
                binding(3, &buffers.in_range),
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tile visibility"),
        });
        if tile_count > 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.visibility);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(tile_count.div_ceil(BLOCK_SIZE), 1, 1);
        }
        queue.submit(Some(encoder.finish()));
        Ok(tile_count)
    }

    /// Decide visibility and generate this frame's indirect draw arguments from
    /// the result, without the slot table ever reaching the CPU.
    ///
    /// The whole of §4.3's draw path in one call, and it is two dispatches: this
    /// pass writes the slots, [`IndirectArgsPass::run_with_slots`] turns them
    /// into arguments. Nothing between them is read back.
    pub fn run_into_args(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffers: &TileVisibilityBuffers,
        target: ArgsTarget<'_>,
        tiles: &[u8],
        viewport: TileViewport,
    ) -> Result<IndirectArgsOutput, TileVisibilityError> {
        let tile_count = self.run(
            device,
            queue,
            buffers,
            tiles,
            viewport,
            target.buffers.arena_slots,
        )?;
        Ok(target.pass.run_with_slots(
            device,
            queue,
            target.buffers,
            GeneratedSlots {
                buffer: &buffers.slots,
                count: tile_count,
            },
            target.vertex_count,
            target.first_instance,
        ))
    }

    /// Read the visibility mask back — the differential's use, never the draw
    /// path's.
    pub fn read_in_range(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffers: &TileVisibilityBuffers,
        tile_count: u32,
    ) -> Result<Vec<u32>, TileVisibilityError> {
        Ok(read_u32_buffer(
            device,
            queue,
            &buffers.in_range,
            tile_count as usize,
        )?)
    }

    /// Read the generated slot table back — the differential's use, never the
    /// draw path's.
    pub fn read_slots(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffers: &TileVisibilityBuffers,
        tile_count: u32,
    ) -> Result<Vec<[u32; 4]>, TileVisibilityError> {
        let words = read_u32_buffer(device, queue, &buffers.slots, tile_count as usize * 4)?;
        Ok(words
            .chunks_exact(4)
            .map(|chunk| [chunk[0], chunk[1], chunk[2], chunk[3]])
            .collect())
    }
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    match bytes.get(offset..offset + 4) {
        Some(slice) => u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]),
        None => 0,
    }
}

fn storage_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        // Never zero: a zero-sized storage binding is invalid, and a boundary
        // with no resident tiles still has to bind something.
        size: size.max(16),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn binding<'a>(index: u32, buffer: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding: index,
        resource: buffer.as_entire_binding(),
    }
}

fn uniform_entry(binding: u32, size: u64) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(size),
        },
        count: None,
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
