//! Dispatches `wgpui_core::shaders::INDIRECT_ARGS_WGSL` (§5.3).
//! See docs/gpu-native-architecture.md §5.3, §8 Phase 4.
//!
//! # What this pass produces
//!
//! For one primitive kind, across every (layer, kind) slot at once: the
//! indirection buffer the instanced pipeline pulls through, one
//! `draw_indirect` argument record per slot, and — for a device that has
//! `MULTI_DRAW_INDIRECT_COUNT` — the populated records packed with their count.
//!
//! The computation is [`wgpui_core::indirect::indirect_args`], written once in
//! Rust and transcribed into WGSL, checked against it for exact equality by
//! `tests/indirect_draw.rs`. That is Phase 3's discipline unchanged, and it is
//! the only discipline a compute shader supports: there is no CPU-side result
//! to eyeball.
//!
//! # This is the pass that consumes Phase 3
//!
//! `docs/phase-3-results.md` §2 states the seam this closes in as many words:
//! "The compute passes write orders, a draw permutation, and a keep mask;
//! nothing yet consumes them into a draw call." [`IndirectArgsPass::scatter`]
//! is the whole of the wiring — one `copy_buffer_to_buffer` per layer, moving
//! that layer's `OrderingOutput::draw_order` and `OcclusionOutput::culled` into
//! its own range of an arena-shaped buffer. No readback, no re-encode, no CPU
//! walk over primitives: the ordering pass's output is already `u32`s in draw
//! order, and the arena base it lands at is the `SlabRange` the CPU already
//! holds.
//!
//! Phase 3's per-layer outputs are indexed `[0, count)` and this pass reads them
//! at `[base, base + count)`, which is why `scatter` exists at all rather than
//! the pass taking the buffers directly. Keeping the layer-local convention
//! upstream is deliberate: it is what lets an `OrderingOutput` be copied into
//! place rather than rewritten.
//!
//! # Why one workgroup per slot rather than one invocation
//!
//! Compaction has to preserve draw order, so it is a prefix sum, not an atomic
//! append. A single invocation per slot would make that prefix sum a serial
//! scan over the slot's whole primitive count — a hundred thousand serial steps
//! in one thread for a large layer, which is the exact mistake Phase 3's own
//! results doc records finding by measurement in the relaxation kernel. A
//! workgroup does it 64 lanes at a time with a running offset between chunks.
//!
//! The cost of that choice, stated rather than discovered later: the dispatch
//! is one workgroup per *slot*, so a scene with four layers occupies four
//! workgroups no matter how many primitives they hold. That is fine for the
//! work being done (a compaction is memory-bound and the arena is read once)
//! and it is the wrong shape if a future phase ever needs this pass to be the
//! frame's dominant cost. It is not, and §8's Phase 4 gate is about CPU work.

use std::num::NonZeroU64;

use wgpui_core::indirect::{
    DRAW_INDIRECT_ARGS_STRIDE, DRAW_SLOT_STRIDE, DrawIndirectArgs, FirstInstance, decode_args,
};
use wgpui_core::ordering::BLOCK_SIZE;

use crate::render::readback::{ReadbackError, read_u32_buffer};

/// Why an indirect-arg dispatch failed.
#[derive(Debug)]
pub enum IndirectArgsError {
    /// The encoded slot table's length is not a whole number of slots.
    MalformedSlots {
        /// Bytes supplied.
        bytes: usize,
        /// Bytes one slot occupies.
        stride: usize,
    },
    /// A slot's reservation runs past the arena it claims to be in — a
    /// bookkeeping bug in the caller, reported rather than dispatched with.
    SlotOutsideArena {
        /// Which slot.
        index: usize,
        /// Its last slot index.
        end: u64,
        /// Slots the arena holds.
        arena_slots: u32,
    },
    /// A readback failed.
    Readback(ReadbackError),
}

impl std::fmt::Display for IndirectArgsError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndirectArgsError::MalformedSlots { bytes, stride } => write!(
                formatter,
                "indirect slot table is {bytes} bytes, not a multiple of the \
                 {stride}-byte stride"
            ),
            IndirectArgsError::SlotOutsideArena {
                index,
                end,
                arena_slots,
            } => write!(
                formatter,
                "slot {index} ends at arena slot {end}, past the arena's {arena_slots}"
            ),
            IndirectArgsError::Readback(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for IndirectArgsError {}

impl From<ReadbackError> for IndirectArgsError {
    fn from(error: ReadbackError) -> Self {
        IndirectArgsError::Readback(error)
    }
}

/// The arena-shaped inputs one kind's dispatch reads, and the buffers it
/// writes.
///
/// Held across frames by the caller: the arena is what it is, and recreating
/// these per frame would put an allocation in front of the one pass whose whole
/// purpose is that the CPU stops doing per-frame work.
pub struct IndirectArgsBuffers {
    /// Arena-shaped. `draw_order[base + position]` is a layer-local index, as
    /// Phase 3's ordering pass writes it.
    pub draw_order: wgpu::Buffer,
    /// Arena-shaped. `1` where the occlusion pass dropped the primitive.
    pub culled: wgpu::Buffer,
    /// Arena-shaped indirection buffer — what the instanced pipeline pulls
    /// through, and the pass's real product.
    pub visible: wgpu::Buffer,
    /// One argument record per slot, in slot-table order.
    pub args: wgpu::Buffer,
    /// The populated records, packed. What a `multi_draw_indirect_count` reads.
    pub packed_args: wgpu::Buffer,
    /// One word: how many entries `packed_args` holds.
    pub draw_count: wgpu::Buffer,
    /// Slots the arena holds.
    pub arena_slots: u32,
    /// Slots the argument buffers are sized for.
    pub slot_capacity: u32,
}

impl IndirectArgsBuffers {
    /// Allocate for an arena of `arena_slots` and a slot table of at most
    /// `slot_capacity` entries.
    pub fn new(device: &wgpu::Device, arena_slots: u32, slot_capacity: u32) -> IndirectArgsBuffers {
        let arena_bytes = u64::from(arena_slots) * 4;
        let args_bytes = u64::from(slot_capacity) * DRAW_INDIRECT_ARGS_STRIDE as u64;
        IndirectArgsBuffers {
            draw_order: storage_buffer(device, "indirect draw order", arena_bytes),
            culled: storage_buffer(device, "indirect culled", arena_bytes),
            visible: storage_buffer(device, "indirect visible", arena_bytes),
            args: indirect_buffer(device, "indirect args", args_bytes),
            packed_args: indirect_buffer(device, "indirect args packed", args_bytes),
            draw_count: indirect_buffer(device, "indirect draw count", 4),
            arena_slots,
            slot_capacity,
        }
    }

    /// Whether these buffers can serve an arena of `arena_slots` and `slots`
    /// entries without being reallocated.
    pub fn fits(&self, arena_slots: u32, slots: u32) -> bool {
        self.arena_slots >= arena_slots && self.slot_capacity >= slots
    }
}

/// What one dispatch decided, as far as the CPU is allowed to know it without
/// asking.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct IndirectArgsOutput {
    /// Slots the dispatch covered — the length of the fixed draw sequence.
    pub slot_count: u32,
    /// Where each record's base index lives.
    pub first_instance: FirstInstance,
    /// Vertices one instance draws.
    pub vertex_count: u32,
}

/// Compiled pipelines for §5.3's indirect-arg pass. Build once, dispatch once
/// per kind per frame.
pub struct IndirectArgsPass {
    layout: wgpu::BindGroupLayout,
    clear_visible: wgpu::ComputePipeline,
    compact: wgpu::ComputePipeline,
    pack: wgpu::ComputePipeline,
}

impl IndirectArgsPass {
    /// Compile every entry point in `indirect_args.wgsl`.
    pub fn new(device: &wgpu::Device) -> IndirectArgsPass {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wgpui indirect args"),
            source: wgpu::ShaderSource::Wgsl(wgpui_core::shaders::INDIRECT_ARGS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("indirect args"),
            entries: &[
                uniform_entry(0, 16),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, false),
                storage_entry(5, false),
                storage_entry(6, false),
                storage_entry(7, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("indirect args"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let build = |entry_point: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        IndirectArgsPass {
            clear_visible: build("clear_visible"),
            compact: build("compact"),
            pack: build("pack"),
            layout,
        }
    }

    /// Move one layer's Phase 3 results into its own range of an arena-shaped
    /// buffer.
    ///
    /// `count` words from `source`'s start to `base` in `destination`. No
    /// readback and no re-encode — see this module's doc for why that is the
    /// whole of the wiring between Phase 3's output and Phase 4's input.
    ///
    /// A copy that would run past `destination` is skipped rather than
    /// submitted: `copy_buffer_to_buffer` validation failure is an uncaptured
    /// device error, which by default aborts the process, and a slot table that
    /// disagrees with its arena is a bookkeeping bug the caller should see as a
    /// wrong picture rather than a crash.
    pub fn scatter(
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::Buffer,
        destination: &wgpu::Buffer,
        base: u32,
        count: u32,
    ) {
        if count == 0 {
            return;
        }
        let bytes = u64::from(count) * 4;
        let offset = u64::from(base) * 4;
        if source.size() < bytes || destination.size() < offset + bytes {
            return;
        }
        encoder.copy_buffer_to_buffer(source, 0, destination, offset, bytes);
    }

    /// Generate one kind's indirect draw arguments.
    ///
    /// `slots` is `wgpui_core::indirect::encode_slots`'s output for that kind,
    /// in draw order. `buffers.draw_order` and `buffers.culled` must already
    /// hold this frame's scattered Phase 3 results.
    pub fn run(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffers: &IndirectArgsBuffers,
        slots: &[u8],
        vertex_count: u32,
        first_instance: FirstInstance,
    ) -> Result<IndirectArgsOutput, IndirectArgsError> {
        if !slots.len().is_multiple_of(DRAW_SLOT_STRIDE) {
            return Err(IndirectArgsError::MalformedSlots {
                bytes: slots.len(),
                stride: DRAW_SLOT_STRIDE,
            });
        }
        let slot_count = u32::try_from(slots.len() / DRAW_SLOT_STRIDE).unwrap_or(u32::MAX);
        for (index, slot) in slots.chunks_exact(DRAW_SLOT_STRIDE).enumerate() {
            let base = u64::from(read_u32(slot, 0));
            let count = u64::from(read_u32(slot, 4));
            if base + count > u64::from(buffers.arena_slots) {
                return Err(IndirectArgsError::SlotOutsideArena {
                    index,
                    end: base + count,
                    arena_slots: buffers.arena_slots,
                });
            }
        }

        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("indirect args params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut params_bytes = Vec::with_capacity(16);
        for value in [
            slot_count,
            first_instance.as_u32(),
            vertex_count,
            buffers.arena_slots,
        ] {
            params_bytes.extend_from_slice(&value.to_le_bytes());
        }
        queue.write_buffer(&params, 0, &params_bytes);

        let slot_buffer = storage_buffer(
            device,
            "indirect slots",
            (slots.len() as u64).max(DRAW_SLOT_STRIDE as u64),
        );
        if !slots.is_empty() {
            queue.write_buffer(&slot_buffer, 0, slots);
        }

        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("indirect args"),
            layout: &self.layout,
            entries: &[
                binding(0, &params),
                binding(1, &slot_buffer),
                binding(2, &buffers.draw_order),
                binding(3, &buffers.culled),
                binding(4, &buffers.visible),
                binding(5, &buffers.args),
                binding(6, &buffers.packed_args),
                binding(7, &buffers.draw_count),
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("indirect args"),
        });
        dispatch(
            &mut encoder,
            &self.clear_visible,
            &group,
            buffers.arena_slots.div_ceil(BLOCK_SIZE),
        );
        // One workgroup per slot — see this module's doc.
        dispatch(&mut encoder, &self.compact, &group, slot_count);
        // One workgroup total: the slot count is tens, not thousands, and the
        // compaction has to stay order-preserving across the whole table.
        dispatch(&mut encoder, &self.pack, &group, 1);
        queue.submit(Some(encoder.finish()));

        Ok(IndirectArgsOutput {
            slot_count,
            first_instance,
            vertex_count,
        })
    }

    /// Read the argument records back — the differential harness's use, and the
    /// CPU-readback fallback's.
    pub fn read_args(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffers: &IndirectArgsBuffers,
        slot_count: u32,
    ) -> Result<Vec<DrawIndirectArgs>, IndirectArgsError> {
        let words = read_u32_buffer(device, queue, &buffers.args, slot_count as usize * 4)?;
        Ok(decode_args(&words))
    }

    /// Read the packed records back.
    pub fn read_packed_args(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffers: &IndirectArgsBuffers,
    ) -> Result<Vec<DrawIndirectArgs>, IndirectArgsError> {
        let count = self.read_draw_count(device, queue, buffers)?;
        let words = read_u32_buffer(device, queue, &buffers.packed_args, count as usize * 4)?;
        Ok(decode_args(&words))
    }

    /// Read how many packed records there are.
    pub fn read_draw_count(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffers: &IndirectArgsBuffers,
    ) -> Result<u32, IndirectArgsError> {
        let words = read_u32_buffer(device, queue, &buffers.draw_count, 1)?;
        Ok(words.first().copied().unwrap_or(0))
    }

    /// Read the indirection buffer back.
    pub fn read_visible(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffers: &IndirectArgsBuffers,
    ) -> Result<Vec<u32>, IndirectArgsError> {
        Ok(read_u32_buffer(
            device,
            queue,
            &buffers.visible,
            buffers.arena_slots as usize,
        )?)
    }
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    match bytes.get(offset..offset + 4) {
        Some(slice) => u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]),
        None => 0,
    }
}

fn dispatch(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    group: &wgpu::BindGroup,
    workgroups: u32,
) {
    if workgroups == 0 {
        return;
    }
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, group, &[]);
    pass.dispatch_workgroups(workgroups, 1, 1);
}

fn storage_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        // Never zero: a zero-sized storage binding is invalid, and an empty
        // arena still has to bind something.
        size: size.max(16),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// A buffer a `draw_indirect` reads its arguments out of, which a compute pass
/// also writes and a readback also copies from.
fn indirect_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(DRAW_INDIRECT_ARGS_STRIDE as u64),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::INDIRECT
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

#[cfg(test)]
mod tests {
    use wgpui_core::indirect::DrawIndirectArgs;

    /// The one layout claim `wgpui-core` cannot check for itself: that its
    /// device-free restatement of the `draw_indirect` argument record matches
    /// the one `wgpu` will actually read.
    #[test]
    fn the_argument_record_matches_wgpus_own_layout() {
        assert_eq!(
            std::mem::size_of::<wgpu::util::DrawIndirectArgs>(),
            wgpui_core::indirect::DRAW_INDIRECT_ARGS_STRIDE
        );
        let ours = DrawIndirectArgs {
            vertex_count: 4,
            instance_count: 7,
            first_vertex: 0,
            first_instance: 64,
        };
        let theirs = wgpu::util::DrawIndirectArgs {
            vertex_count: 4,
            instance_count: 7,
            first_vertex: 0,
            first_instance: 64,
        };
        let ours_bytes: Vec<u8> = ours
            .to_array()
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        assert_eq!(
            ours_bytes,
            theirs.as_bytes(),
            "field order or width drifted from the specification's layout"
        );
    }
}
