//! Dispatches `wgpui_core::shaders::ORDERING_WGSL` (§5.1). Phase 0's Spike A
//! prototype (`examples/spike_a_ordering_occlusion.rs`) is the throwaway
//! precursor to this pass's real implementation.
//! See docs/gpu-native-architecture.md §5.1, §8 Phase 0/Phase 3.
//!
//! # What this pass produces
//!
//! For one layer's primitives, in paint order: each primitive's painter order
//! (`BoundsTree`'s number, computed by relaxation) and the draw permutation
//! sorting them by `(order, index)` — the same two things `Scene::finish`
//! produces today with a tree walk and nine `sort_by_key` calls.
//!
//! It does **not** move any bytes. Phase 2 established that a record keeps the
//! slot it was inserted at, and the permutation is a view over that residency,
//! not a reshuffle of it. Phase 4 turns the permutation into indirect draw args
//! (§5.3); until then it feeds the same CPU-side draw-range decision
//! `Scene::draw_ranges` already makes.
//!
//! # Three differences from Spike A, all of them the point
//!
//! 1. **The neighbour search is bounded by structure, not by the benchmark.**
//!    Spike A's synthetic scene was 200 disjoint clusters, so its relax kernel
//!    could scan "the 500 quads in my own cluster" for free — its own write-up
//!    flags that as the thing Phase 3 still had to earn. This builds a
//!    two-level AABB hierarchy over paint order instead, which needs nothing of
//!    the scene.
//! 2. **Convergence is checked, not budgeted.** Spike A ran a fixed 128
//!    iterations and reported the residual afterwards. This relaxes until the
//!    change counter reads zero, so a scene with a deeper overlap chain than
//!    the budget gets more iterations rather than a wrong answer.
//! 3. **Pipelines are built once.** Spike A built its pipelines inside its
//!    measured window, which is most of why its own numbers varied 2–3× between
//!    runs. [`OrderingPass::new`] is called once and [`OrderingPass::run`] many
//!    times.

use std::num::NonZeroU64;

use wgpui_core::ordering::{BLOCK_SIZE, block_count, superblock_count};

use crate::render::readback::{ReadbackError, read_u32_buffer};

/// Relaxation iterations in the first submission, which also carries the sort.
///
/// Chosen so a UI layer — whose overlap chains are nesting-depth deep, a
/// handful of levels — converges in one submission and pays exactly one
/// four-byte readback for the proof. Deeper scenes fall through to
/// [`RELAX_BATCH`] and cost extra round trips rather than correctness.
const RELAX_FIRST_BATCH: u32 = 16;

/// Relaxation iterations per follow-up submission, once the first did not
/// converge. Larger than the first batch because a scene that needed more than
/// sixteen is likely to need many more, and each batch costs a round trip.
const RELAX_BATCH: u32 = 48;

/// Hard ceiling on total iterations.
///
/// The fixed point is reached in at most `max_order` iterations and `max_order`
/// is at most the primitive count, so this is a guard against a bug rather than
/// against a legitimate scene: a layer whose overlap chain is four thousand
/// deep is not something the renderer should silently spend minutes on.
const RELAX_ITERATION_LIMIT: u32 = 4096;

/// Why an ordering dispatch failed.
#[derive(Debug)]
pub enum OrderingError {
    /// The bounds slice's length is not a whole number of primitives.
    MalformedInput {
        /// Bytes supplied.
        bytes: usize,
        /// Bytes one primitive occupies.
        stride: usize,
    },
    /// A readback failed.
    Readback(ReadbackError),
    /// The relaxation did not reach a fixed point within
    /// [`RELAX_ITERATION_LIMIT`].
    NotConverged {
        /// Iterations run before giving up.
        iterations: u32,
        /// Primitives still moving on the last iteration.
        still_changing: u32,
    },
}

impl std::fmt::Display for OrderingError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrderingError::MalformedInput { bytes, stride } => write!(
                formatter,
                "ordering input is {bytes} bytes, not a multiple of the {stride}-byte stride"
            ),
            OrderingError::Readback(error) => write!(formatter, "{error}"),
            OrderingError::NotConverged {
                iterations,
                still_changing,
            } => write!(
                formatter,
                "painter-order relaxation did not converge in {iterations} iterations \
                 ({still_changing} primitives still moving)"
            ),
        }
    }
}

impl std::error::Error for OrderingError {}

impl From<ReadbackError> for OrderingError {
    fn from(error: ReadbackError) -> Self {
        OrderingError::Readback(error)
    }
}

/// GPU-resident results of one ordering dispatch.
///
/// The buffers stay on the device. Reading them back is the caller's separate,
/// explicitly-priced step ([`OrderingPass::read_orders`],
/// [`OrderingPass::read_draw_order`]) rather than part of the pass, because a
/// real frame would not round-trip them.
pub struct OrderingOutput {
    /// `count` painter orders, in paint order.
    pub orders: wgpu::Buffer,
    /// `padded_count` primitive indices in draw order; entries past `count`
    /// hold `u32::MAX` padding.
    pub draw_order: wgpu::Buffer,
    /// Primitives this layer holds.
    pub count: u32,
    /// `count` rounded up to a power of two — the sort network's width.
    pub padded_count: u32,
    /// Relaxation iterations actually run.
    pub iterations: u32,
    /// Submissions the convergence loop cost. One means the scene converged
    /// inside [`RELAX_FIRST_BATCH`].
    pub submissions: u32,
}

/// Compiled pipelines for §5.1's ordering pass. Build once, dispatch per layer.
pub struct OrderingPass {
    data_layout: wgpu::BindGroupLayout,
    stage_layout: wgpu::BindGroupLayout,
    build_blocks: wgpu::ComputePipeline,
    build_superblocks: wgpu::ComputePipeline,
    relax: wgpu::ComputePipeline,
    pack: wgpu::ComputePipeline,
    bitonic: wgpu::ComputePipeline,
    stage_alignment: u32,
}

impl OrderingPass {
    /// Compile every entry point in `ordering.wgsl`.
    pub fn new(device: &wgpu::Device) -> OrderingPass {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wgpui ordering"),
            source: wgpu::ShaderSource::Wgsl(wgpui_core::shaders::ORDERING_WGSL.into()),
        });

        // One explicit layout shared by every entry point, rather than five
        // derived ones: the entry points read and write overlapping subsets of
        // the same buffers, and a derived layout per pipeline would mean a
        // bind group per pipeline for no benefit.
        let data_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ordering data"),
            entries: &[
                uniform_entry(0, 16),
                storage_entry(1, true),
                storage_entry(2, false),
                storage_entry(3, false),
                storage_entry(4, true),
                storage_entry(5, false),
                storage_entry(6, false),
                storage_entry(7, false),
                storage_entry(8, false),
            ],
        });
        let stage_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ordering bitonic stage"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: NonZeroU64::new(16),
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ordering"),
            bind_group_layouts: &[Some(&data_layout), Some(&stage_layout)],
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

        OrderingPass {
            build_blocks: build("build_blocks"),
            build_superblocks: build("build_superblocks"),
            relax: build("relax"),
            pack: build("pack"),
            bitonic: build("bitonic"),
            data_layout,
            stage_layout,
            stage_alignment: device.limits().min_uniform_buffer_offset_alignment.max(16),
        }
    }

    /// Compute painter orders and the draw permutation for one layer.
    ///
    /// `bounds` is `wgpui_core::ordering::encode_ordering_items`'s output: one
    /// `vec4<f32>` per primitive, in paint order.
    pub fn run(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bounds: &[u8],
    ) -> Result<OrderingOutput, OrderingError> {
        let stride = wgpui_core::ordering::ORDERING_ITEM_STRIDE;
        if !bounds.len().is_multiple_of(stride) {
            return Err(OrderingError::MalformedInput {
                bytes: bounds.len(),
                stride,
            });
        }
        let count = u32::try_from(bounds.len() / stride).unwrap_or(u32::MAX);
        if count == 0 {
            return Ok(OrderingOutput {
                orders: empty_storage_buffer(device, "ordering orders"),
                draw_order: empty_storage_buffer(device, "ordering draw order"),
                count: 0,
                padded_count: 0,
                iterations: 0,
                submissions: 0,
            });
        }

        let padded_count = count.next_power_of_two();
        let blocks = block_count(count);
        let superblocks = superblock_count(count);

        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ordering params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut params_bytes = Vec::with_capacity(16);
        for value in [count, blocks, superblocks, padded_count] {
            params_bytes.extend_from_slice(&value.to_le_bytes());
        }
        queue.write_buffer(&params, 0, &params_bytes);

        let bounds_buffer = storage_buffer(device, "ordering bounds", bounds.len() as u64);
        queue.write_buffer(&bounds_buffer, 0, bounds);
        let block_buffer = storage_buffer(device, "ordering blocks", blocks as u64 * 16);
        let superblock_buffer =
            storage_buffer(device, "ordering superblocks", superblocks as u64 * 16);

        let order_bytes = count as u64 * 4;
        let order_a = storage_buffer(device, "ordering order a", order_bytes);
        let order_b = storage_buffer(device, "ordering order b", order_bytes);
        // Both sides start at the recurrence's floor so the first iteration
        // reads a defined value whichever way the ping-pong points.
        let initial = vec![1u32; count as usize];
        let initial_bytes: Vec<u8> = initial.iter().flat_map(|v| v.to_le_bytes()).collect();
        queue.write_buffer(&order_a, 0, &initial_bytes);
        queue.write_buffer(&order_b, 0, &initial_bytes);

        let changed = storage_buffer(device, "ordering changed", 4);
        let sort_key = storage_buffer(device, "ordering sort keys", padded_count as u64 * 4);
        let sort_value = storage_buffer(device, "ordering sort values", padded_count as u64 * 4);

        // Two bind groups differing only in which order buffer is the input.
        // `data_groups[k]` reads `[order_a, order_b][k]`, so after using group
        // `k` the freshly written buffer is the other one.
        let data_groups = [
            self.data_group(device, &params, &bounds_buffer, &block_buffer, &superblock_buffer, &order_a, &order_b, &changed, &sort_key, &sort_value),
            self.data_group(device, &params, &bounds_buffer, &block_buffer, &superblock_buffer, &order_b, &order_a, &changed, &sort_key, &sort_value),
        ];

        let stages = bitonic_stages(padded_count);
        let (stage_buffer, stage_group) = self.stage_resources(device, queue, &stages);

        let block_groups = blocks.div_ceil(BLOCK_SIZE).max(1);
        let superblock_groups = superblocks.div_ceil(BLOCK_SIZE).max(1);
        let item_groups = count.div_ceil(BLOCK_SIZE);
        let sort_groups = padded_count.div_ceil(BLOCK_SIZE);

        let mut parity = 0usize;
        let mut iterations = 0u32;
        let mut submissions = 0u32;

        // First submission: hierarchy, an optimistic relaxation batch, and the
        // sort. A layer that converges inside the batch is finished here.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ordering"),
        });
        dispatch(&mut encoder, &self.build_blocks, &data_groups[parity], &stage_group, 0, block_groups);
        dispatch(&mut encoder, &self.build_superblocks, &data_groups[parity], &stage_group, 0, superblock_groups);
        parity = self.encode_relaxation(
            &mut encoder,
            &data_groups,
            &stage_group,
            &changed,
            parity,
            RELAX_FIRST_BATCH,
            item_groups,
        );
        iterations += RELAX_FIRST_BATCH;
        self.encode_sort(
            &mut encoder,
            &data_groups[parity],
            &stage_group,
            &stages,
            sort_groups,
        );
        queue.submit(Some(encoder.finish()));
        submissions += 1;

        let mut still_changing = self.read_changed(device, queue, &changed)?;
        while still_changing != 0 {
            if iterations >= RELAX_ITERATION_LIMIT {
                return Err(OrderingError::NotConverged {
                    iterations,
                    still_changing,
                });
            }
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ordering relaxation"),
            });
            parity = self.encode_relaxation(
                &mut encoder,
                &data_groups,
                &stage_group,
                &changed,
                parity,
                RELAX_BATCH,
                item_groups,
            );
            iterations += RELAX_BATCH;
            queue.submit(Some(encoder.finish()));
            submissions += 1;
            still_changing = self.read_changed(device, queue, &changed)?;
        }

        // The sort issued in the first submission is only valid if that
        // submission also converged; otherwise redo it over the settled orders.
        if submissions > 1 {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ordering sort"),
            });
            self.encode_sort(
                &mut encoder,
                &data_groups[parity],
                &stage_group,
                &stages,
                sort_groups,
            );
            queue.submit(Some(encoder.finish()));
            submissions += 1;
        }

        // `stage_buffer` is bound by `stage_group` and must outlive the
        // submissions above; naming it here is what keeps it alive that long.
        drop(stage_buffer);

        let orders = if parity == 0 { order_a } else { order_b };
        Ok(OrderingOutput {
            orders,
            draw_order: sort_value,
            count,
            padded_count,
            iterations,
            submissions,
        })
    }

    /// Read the painter orders back. Deliberately not part of [`Self::run`] —
    /// see this module's doc.
    pub fn read_orders(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output: &OrderingOutput,
    ) -> Result<Vec<u32>, OrderingError> {
        Ok(read_u32_buffer(
            device,
            queue,
            &output.orders,
            output.count as usize,
        )?)
    }

    /// Read the draw permutation back, trimmed to the real primitives.
    pub fn read_draw_order(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output: &OrderingOutput,
    ) -> Result<Vec<u32>, OrderingError> {
        let mut values = read_u32_buffer(
            device,
            queue,
            &output.draw_order,
            output.padded_count as usize,
        )?;
        values.truncate(output.count as usize);
        Ok(values)
    }

    #[allow(clippy::too_many_arguments)]
    fn data_group(
        &self,
        device: &wgpu::Device,
        params: &wgpu::Buffer,
        bounds: &wgpu::Buffer,
        blocks: &wgpu::Buffer,
        superblocks: &wgpu::Buffer,
        order_in: &wgpu::Buffer,
        order_out: &wgpu::Buffer,
        changed: &wgpu::Buffer,
        sort_key: &wgpu::Buffer,
        sort_value: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ordering data"),
            layout: &self.data_layout,
            entries: &[
                binding(0, params),
                binding(1, bounds),
                binding(2, blocks),
                binding(3, superblocks),
                binding(4, order_in),
                binding(5, order_out),
                binding(6, changed),
                binding(7, sort_key),
                binding(8, sort_value),
            ],
        })
    }

    fn stage_resources(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        stages: &[(u32, u32)],
    ) -> (wgpu::Buffer, wgpu::BindGroup) {
        let stride = self.stage_alignment as u64;
        let size = stride * stages.len().max(1) as u64;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ordering bitonic stages"),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut bytes = vec![0u8; size as usize];
        for (index, (span, width)) in stages.iter().enumerate() {
            let offset = index * stride as usize;
            if let Some(slot) = bytes.get_mut(offset..offset + 8) {
                slot[0..4].copy_from_slice(&span.to_le_bytes());
                slot[4..8].copy_from_slice(&width.to_le_bytes());
            }
        }
        queue.write_buffer(&buffer, 0, &bytes);
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ordering bitonic stage"),
            layout: &self.stage_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buffer,
                    offset: 0,
                    size: NonZeroU64::new(16),
                }),
            }],
        });
        (buffer, group)
    }

    /// Encode `count` relaxation iterations, returning the new parity.
    ///
    /// The change counter is cleared immediately before the *last* iteration of
    /// the batch, inside the same encoder, so the value read afterwards reports
    /// only that iteration's residual. A `queue.write_buffer` here would be
    /// ordered before the whole encoder rather than between two of its passes —
    /// the same trap Spike A's own comment records.
    fn encode_relaxation(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        data_groups: &[wgpu::BindGroup; 2],
        stage_group: &wgpu::BindGroup,
        changed: &wgpu::Buffer,
        parity: usize,
        count: u32,
        workgroups: u32,
    ) -> usize {
        let mut parity = parity;
        for iteration in 0..count {
            if iteration + 1 == count {
                encoder.clear_buffer(changed, 0, None);
            }
            dispatch(
                encoder,
                &self.relax,
                &data_groups[parity],
                stage_group,
                0,
                workgroups,
            );
            parity ^= 1;
        }
        parity
    }

    fn encode_sort(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        data_group: &wgpu::BindGroup,
        stage_group: &wgpu::BindGroup,
        stages: &[(u32, u32)],
        sort_groups: u32,
    ) {
        // Over the padded width, not the primitive count: `pack` is what writes
        // the sentinel keys that keep padding entries out of the result.
        dispatch(encoder, &self.pack, data_group, stage_group, 0, sort_groups);
        for index in 0..stages.len() {
            let offset = index as u32 * self.stage_alignment;
            dispatch(
                encoder,
                &self.bitonic,
                data_group,
                stage_group,
                offset,
                sort_groups,
            );
        }
    }

    fn read_changed(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        changed: &wgpu::Buffer,
    ) -> Result<u32, OrderingError> {
        let values = read_u32_buffer(device, queue, changed, 1)?;
        Ok(values.first().copied().unwrap_or(0))
    }
}

/// The `(span, width)` pairs of a bitonic network over `length` elements,
/// which must be a power of two.
fn bitonic_stages(length: u32) -> Vec<(u32, u32)> {
    let mut stages = Vec::new();
    let mut width = 2u32;
    while width <= length {
        let mut span = width / 2;
        while span >= 1 {
            stages.push((span, width));
            span /= 2;
        }
        width *= 2;
    }
    stages
}

fn dispatch(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    data_group: &wgpu::BindGroup,
    stage_group: &wgpu::BindGroup,
    stage_offset: u32,
    workgroups: u32,
) {
    if workgroups == 0 {
        return;
    }
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, data_group, &[]);
    pass.set_bind_group(1, stage_group, &[stage_offset]);
    pass.dispatch_workgroups(workgroups, 1, 1);
}

fn storage_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        // Never zero: a zero-sized storage binding is invalid, and a layer with
        // no blocks (an empty layer) still has to bind something.
        size: size.max(16),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn empty_storage_buffer(device: &wgpu::Device, label: &str) -> wgpu::Buffer {
    storage_buffer(device, label, 0)
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
    use super::*;

    #[test]
    fn a_bitonic_network_over_one_element_has_no_stages() {
        assert!(bitonic_stages(1).is_empty());
    }

    #[test]
    fn a_bitonic_network_has_the_triangular_number_of_stages() {
        // A network over 2^m has m(m+1)/2 compare-exchange stages.
        for exponent in 1..=8u32 {
            let length = 1u32 << exponent;
            assert_eq!(
                bitonic_stages(length).len(),
                (exponent * (exponent + 1) / 2) as usize,
                "network over {length} elements"
            );
        }
    }

    #[test]
    fn bitonic_stage_spans_halve_within_each_width() {
        assert_eq!(
            bitonic_stages(8),
            vec![(1, 2), (2, 4), (1, 4), (4, 8), (2, 8), (1, 8)]
        );
    }
}
