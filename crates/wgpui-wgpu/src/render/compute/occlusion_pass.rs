//! Dispatches `wgpui_core::shaders::OCCLUSION_WGSL` (§5.2). Phase 0's Spike A
//! prototype (`examples/spike_a_ordering_occlusion.rs`) is the throwaway
//! precursor to this pass's real implementation.
//! See docs/gpu-native-architecture.md §5.2, §8 Phase 0/Phase 3.
//!
//! # What this pass produces, and what it must never change
//!
//! One flag per primitive of one dirty layer: whether that primitive's emission
//! can be skipped because opaque content painted above it covers every pixel it
//! could contribute. R-N §8.2 puts this tier at emission time and only for
//! layers that are dirty anyway, so a clean layer never passes through here and
//! an occluder animating above a static layer can never churn that layer's
//! slab. Nothing in this file touches residency, hitboxes, or dispatch nodes —
//! it is handed geometry and returns a mask, which is R-N §8.4's constraint
//! made structural rather than remembered.
//!
//! # Why Spike A's cull kernel is not what this is
//!
//! Spike A culled against a hand-built per-cluster occluder list with a single
//! fully-containing-rectangle test, and its own write-up says so: "NOT R-N
//! §8.3's full conservative test... this synthetic scene has no such properties
//! to test." This runs the real test — solid background, opacity, corner-radius
//! inset, border-opacity inset, backdrop-filter poisoning, blur margin — over
//! the layer's own primitives, found through an AABB hierarchy rather than
//! handed to it by the benchmark's structure.

use std::num::NonZeroU64;

use wgpui_core::occlusion::{COVERAGE_ITEM_STRIDE, POISON_REGION_STRIDE};
use wgpui_core::ordering::{BLOCK_SIZE, block_count, superblock_count};

use crate::render::readback::{ReadbackError, read_u32_buffer};

/// Why an occlusion dispatch failed.
#[derive(Debug)]
pub enum OcclusionError {
    /// An input slice's length is not a whole number of records.
    MalformedInput {
        /// Which input.
        what: &'static str,
        /// Bytes supplied.
        bytes: usize,
        /// Bytes one record occupies.
        stride: usize,
    },
    /// A readback failed.
    Readback(ReadbackError),
}

impl std::fmt::Display for OcclusionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OcclusionError::MalformedInput {
                what,
                bytes,
                stride,
            } => write!(
                formatter,
                "occlusion {what} input is {bytes} bytes, not a multiple of the \
                 {stride}-byte stride"
            ),
            OcclusionError::Readback(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for OcclusionError {}

impl From<ReadbackError> for OcclusionError {
    fn from(error: ReadbackError) -> Self {
        OcclusionError::Readback(error)
    }
}

/// GPU-resident result of one occlusion dispatch.
pub struct OcclusionOutput {
    /// One `u32` per primitive: `1` culled, `0` kept.
    pub culled: wgpu::Buffer,
    /// Primitives this layer holds.
    pub count: u32,
}

/// Compiled pipelines for §5.2's occlusion pass. Build once, dispatch per dirty
/// layer.
pub struct OcclusionPass {
    layout: wgpu::BindGroupLayout,
    build_blocks: wgpu::ComputePipeline,
    build_superblocks: wgpu::ComputePipeline,
    cull: wgpu::ComputePipeline,
}

impl OcclusionPass {
    /// Compile every entry point in `occlusion.wgsl`.
    pub fn new(device: &wgpu::Device) -> OcclusionPass {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wgpui occlusion"),
            source: wgpu::ShaderSource::Wgsl(wgpui_core::shaders::OCCLUSION_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("occlusion"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(16),
                    },
                    count: None,
                },
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(5, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("occlusion"),
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
        OcclusionPass {
            build_blocks: build("build_blocks"),
            build_superblocks: build("build_superblocks"),
            cull: build("cull"),
            layout,
        }
    }

    /// Decide which of one layer's primitives must still be emitted.
    ///
    /// `items` is `wgpui_core::occlusion::encode_coverage_items`'s output and
    /// `poison` is `encode_poison_regions`'s, both in paint order.
    pub fn run(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        items: &[u8],
        poison: &[u8],
    ) -> Result<OcclusionOutput, OcclusionError> {
        if !items.len().is_multiple_of(COVERAGE_ITEM_STRIDE) {
            return Err(OcclusionError::MalformedInput {
                what: "item",
                bytes: items.len(),
                stride: COVERAGE_ITEM_STRIDE,
            });
        }
        if !poison.len().is_multiple_of(POISON_REGION_STRIDE) {
            return Err(OcclusionError::MalformedInput {
                what: "poison region",
                bytes: poison.len(),
                stride: POISON_REGION_STRIDE,
            });
        }
        let count = u32::try_from(items.len() / COVERAGE_ITEM_STRIDE).unwrap_or(u32::MAX);
        let poison_count =
            u32::try_from(poison.len() / POISON_REGION_STRIDE).unwrap_or(u32::MAX);
        if count == 0 {
            return Ok(OcclusionOutput {
                culled: storage_buffer(device, "occlusion culled", 0),
                count: 0,
            });
        }

        let blocks = block_count(count);
        let superblocks = superblock_count(count);

        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("occlusion params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut params_bytes = Vec::with_capacity(16);
        for value in [count, poison_count, blocks, superblocks] {
            params_bytes.extend_from_slice(&value.to_le_bytes());
        }
        queue.write_buffer(&params, 0, &params_bytes);

        let item_buffer = storage_buffer(device, "occlusion items", items.len() as u64);
        queue.write_buffer(&item_buffer, 0, items);
        let poison_buffer = storage_buffer(device, "occlusion poison", poison.len() as u64);
        if !poison.is_empty() {
            queue.write_buffer(&poison_buffer, 0, poison);
        }
        let block_buffer = storage_buffer(device, "occlusion blocks", blocks as u64 * 16);
        let superblock_buffer =
            storage_buffer(device, "occlusion superblocks", superblocks as u64 * 16);
        let culled = storage_buffer(device, "occlusion culled", count as u64 * 4);

        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("occlusion"),
            layout: &self.layout,
            entries: &[
                binding(0, &params),
                binding(1, &item_buffer),
                binding(2, &poison_buffer),
                binding(3, &block_buffer),
                binding(4, &superblock_buffer),
                binding(5, &culled),
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("occlusion"),
        });
        dispatch(
            &mut encoder,
            &self.build_blocks,
            &group,
            blocks.div_ceil(BLOCK_SIZE).max(1),
        );
        dispatch(
            &mut encoder,
            &self.build_superblocks,
            &group,
            superblocks.div_ceil(BLOCK_SIZE).max(1),
        );
        dispatch(&mut encoder, &self.cull, &group, count.div_ceil(BLOCK_SIZE));
        queue.submit(Some(encoder.finish()));

        Ok(OcclusionOutput { culled, count })
    }

    /// Read the mask back as `keep` flags, matching
    /// `wgpui_core::occlusion::keep_mask`'s polarity so the two can be compared
    /// directly rather than through an inversion the caller has to remember.
    pub fn read_keep_mask(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output: &OcclusionOutput,
    ) -> Result<Vec<bool>, OcclusionError> {
        let values = read_u32_buffer(device, queue, &output.culled, output.count as usize)?;
        Ok(values.into_iter().map(|value| value == 0).collect())
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
        // A zero-sized storage binding is invalid, and a layer with no poison
        // regions still has to bind something.
        size: size.max(POISON_REGION_STRIDE as u64),
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
