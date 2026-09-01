//! GPU lowering and dispatch for regular retained layout.
//!
//! `wgpui-layout` owns eligibility and the backend-neutral byte contract. This
//! module owns the live `wgpu` resources. The result remains on the device
//! until a caller explicitly requests readback, so a render command encoder
//! can consume it without a CPU round trip.

use std::{cell::RefCell, num::NonZeroU64};

use wgpui_core::shaders::LAYOUT_UNIFORM_WGSL;
use wgpui_layout::regular::{
    RegularLayoutFallback, RegularLayoutInput, DEFAULT_GPU_MIN_ITEMS, REGULAR_LAYOUT_ITEM_STRIDE,
    REGULAR_LAYOUT_PARAMS_SIZE,
};

use crate::render::readback::{read_u32_buffer, ReadbackError};

const WORKGROUP_SIZE: u32 = 64;

#[derive(Debug)]
pub enum LayoutPassError {
    Fallback(RegularLayoutFallback),
    Readback(ReadbackError),
}

impl std::fmt::Display for LayoutPassError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fallback(reason) => write!(formatter, "regular layout fallback: {reason}"),
            Self::Readback(error) => write!(formatter, "regular layout readback: {error}"),
        }
    }
}

impl std::error::Error for LayoutPassError {}

impl From<ReadbackError> for LayoutPassError {
    fn from(error: ReadbackError) -> Self {
        Self::Readback(error)
    }
}

/// GPU-resident output of a regular layout dispatch.
pub struct LayoutOutput {
    pub rectangles: wgpu::Buffer,
    pub count: u32,
}

/// Compiled regular-layout pipeline. Construct once and reuse across frames.
pub struct LayoutPass {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    minimum_items: usize,
    resources: RefCell<Option<CachedLayout>>,
}

struct CachedLayout {
    params: Vec<u8>,
    items: Vec<u8>,
    params_buffer: wgpu::Buffer,
    items_buffer: wgpu::Buffer,
    rectangles: wgpu::Buffer,
    rectangle_capacity: u64,
}

impl LayoutPass {
    /// Compile the production regular-content shader with its default workload
    /// gate. Small lines stay on the exact CPU/Taffy path because a standalone
    /// dispatch cannot amortize its submission cost for them.
    pub fn new(device: &wgpu::Device) -> Self {
        Self::with_minimum_items(device, DEFAULT_GPU_MIN_ITEMS)
    }

    /// Construct a pass with an explicit threshold, useful for profiling and
    /// differential tests without changing the shipping default.
    pub fn with_minimum_items(device: &wgpu::Device, minimum_items: usize) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wgpui regular layout"),
            source: wgpu::ShaderSource::Wgsl(LAYOUT_UNIFORM_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("regular layout"),
            entries: &[
                uniform_entry(0, REGULAR_LAYOUT_PARAMS_SIZE as u64),
                storage_entry(1, true),
                storage_entry(2, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("regular layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("regular layout"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("compute_layout"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            layout,
            pipeline,
            minimum_items,
            resources: RefCell::new(None),
        }
    }

    /// Dispatch an eligible regular line. No CPU result is returned here: the
    /// output buffer can be bound by the next GPU pass. A fallback is explicit
    /// and leaves the caller's retained CPU/Taffy result authoritative.
    pub fn run(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &RegularLayoutInput,
    ) -> Result<LayoutOutput, LayoutPassError> {
        input.validate().map_err(LayoutPassError::Fallback)?;
        if input.items.len() < self.minimum_items {
            return Err(LayoutPassError::Fallback(
                RegularLayoutFallback::WorkloadTooSmall,
            ));
        }
        if device.limits().max_storage_buffers_per_shader_stage < 2
            || device.limits().max_compute_workgroups_per_dimension == 0
        {
            return Err(LayoutPassError::Fallback(
                RegularLayoutFallback::DeviceUnsupported,
            ));
        }
        let packed = input.pack().map_err(LayoutPassError::Fallback)?;
        if packed.params.len() != REGULAR_LAYOUT_PARAMS_SIZE
            || packed.items.len() != input.items.len() * REGULAR_LAYOUT_ITEM_STRIDE
        {
            return Err(LayoutPassError::Fallback(
                RegularLayoutFallback::InvalidNumber,
            ));
        }
        let rectangle_count = input.items.len() as u64;
        let rectangle_size = rectangle_count * 16;
        let mut resources = self.resources.borrow_mut();
        let resources_were_empty = resources.is_none();
        let cached = resources.get_or_insert_with(|| {
            let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("regular layout params"),
                size: REGULAR_LAYOUT_PARAMS_SIZE as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&params_buffer, 0, &packed.params);
            let items_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("regular layout items"),
                size: packed.items.len() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&items_buffer, 0, &packed.items);
            let rectangles = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("regular layout rectangles"),
                size: rectangle_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            CachedLayout {
                params: packed.params.clone(),
                items: packed.items.clone(),
                params_buffer,
                items_buffer,
                rectangles,
                rectangle_capacity: rectangle_size,
            }
        });
        let params_changed = cached.params != packed.params;
        let items_changed = cached.items != packed.items;
        if params_changed {
            write_changed_ranges(queue, &cached.params_buffer, &cached.params, &packed.params);
            cached.params.clone_from(&packed.params);
        }
        if items_changed {
            if cached.items_buffer.size() < packed.items.len() as u64 {
                cached.items_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("regular layout items"),
                    size: packed.items.len() as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&cached.items_buffer, 0, &packed.items);
            } else {
                write_changed_ranges(queue, &cached.items_buffer, &cached.items, &packed.items);
            }
            cached.items.clone_from(&packed.items);
        }
        if cached.rectangle_capacity < rectangle_size || params_changed || items_changed {
            cached.rectangles = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("regular layout rectangles"),
                size: rectangle_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            cached.rectangle_capacity = rectangle_size;
        }
        let params = &cached.params_buffer;
        let items = &cached.items_buffer;
        let rectangles = &cached.rectangles;
        if !resources_were_empty && !params_changed && !items_changed {
            return Ok(LayoutOutput {
                rectangles: rectangles.clone(),
                count: input.items.len() as u32,
            });
        }
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("regular layout"),
            layout: &self.layout,
            entries: &[
                binding(0, &params),
                binding(1, &items),
                binding(2, &rectangles),
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("regular layout"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups((input.items.len() as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        queue.submit(Some(encoder.finish()));
        Ok(LayoutOutput {
            rectangles: rectangles.clone(),
            count: input.items.len() as u32,
        })
    }

    /// Dispatch and synchronously consume the output. Production rendering
    /// should use [`Self::run`] and bind the returned buffer; this method is
    /// the explicit readback path for CPU consumers and differential tests.
    pub fn run_and_read(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &RegularLayoutInput,
    ) -> Result<Vec<[f32; 4]>, LayoutPassError> {
        let output = self.run(device, queue, input)?;
        self.read(device, queue, &output)
    }

    /// Read layout rectangles after GPU completion, preserving the shader's
    /// `[x, y, width, height]` record order.
    pub fn read(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output: &LayoutOutput,
    ) -> Result<Vec<[f32; 4]>, LayoutPassError> {
        let values = read_u32_buffer(device, queue, &output.rectangles, output.count as usize * 4)?;
        Ok(values
            .chunks_exact(4)
            .map(|chunk| {
                [
                    f32::from_bits(chunk[0]),
                    f32::from_bits(chunk[1]),
                    f32::from_bits(chunk[2]),
                    f32::from_bits(chunk[3]),
                ]
            })
            .collect())
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

fn binding<'a>(index: u32, buffer: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding: index,
        resource: buffer.as_entire_binding(),
    }
}

fn write_changed_ranges(queue: &wgpu::Queue, buffer: &wgpu::Buffer, old: &[u8], new: &[u8]) {
    let common_length = old.len().min(new.len());
    let mut range_start = None;
    for index in 0..common_length {
        if old[index] != new[index] {
            if range_start.is_none() {
                range_start = Some(index);
            }
        } else if let Some(start) = range_start.take() {
            write_aligned_range(queue, buffer, new, start, index);
        }
    }
    if let Some(start) = range_start {
        write_aligned_range(queue, buffer, new, start, common_length);
    }
    if new.len() > common_length {
        write_aligned_range(queue, buffer, new, common_length, new.len());
    }
}

fn write_aligned_range(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    data: &[u8],
    start: usize,
    end: usize,
) {
    let aligned_start = start / 4 * 4;
    let aligned_end = end.saturating_add(3).min(data.len()) / 4 * 4;
    if aligned_start < aligned_end {
        queue.write_buffer(
            buffer,
            aligned_start as u64,
            &data[aligned_start..aligned_end],
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::device::context_or_report;
    use wgpui_layout::regular::{
        RegularAlignItems, RegularAxis, RegularJustifyContent, RegularLayoutItem,
    };

    fn input(count: usize) -> RegularLayoutInput {
        RegularLayoutInput {
            origin: [0.5, 1.0],
            container_size: [640.0, 96.0],
            padding: [7.0, 5.0, 9.0, 3.0],
            gap: 2.5,
            axis: RegularAxis::Row,
            justify_content: RegularJustifyContent::SpaceAround,
            align_items: RegularAlignItems::Center,
            rounding_scale: 1.0,
            items: (0..count)
                .map(|index| {
                    let mut item = RegularLayoutItem::fixed(12.0 + (index % 3) as f32, 22.0);
                    if index % 7 == 0 {
                        item.min_size[0] = 13.0;
                        item.max_size[0] = 14.0;
                    }
                    item.transform[4] = if index % 2 == 0 { 0.25 } else { 0.0 };
                    item
                })
                .collect(),
        }
    }

    #[test]
    fn the_default_workload_gate_is_explicit() {
        let Some(context) = context_or_report("regular_layout_workload_gate") else {
            return;
        };
        let pass = LayoutPass::new(&context.device);
        let result = pass.run(&context.device, &context.queue, &input(1));
        assert!(matches!(
            result,
            Err(LayoutPassError::Fallback(
                RegularLayoutFallback::WorkloadTooSmall
            ))
        ));
    }

    #[test]
    fn gpu_output_matches_the_cpu_reference() {
        let Some(context) = context_or_report("regular_layout_differential") else {
            return;
        };
        let input = input(65);
        let expected = input
            .compute_cpu()
            .expect("CPU reference should accept input");
        let pass = LayoutPass::with_minimum_items(&context.device, 1);
        let actual = match pass.run_and_read(&context.device, &context.queue, &input) {
            Ok(rectangles) => rectangles,
            Err(LayoutPassError::Fallback(RegularLayoutFallback::DeviceUnsupported)) => return,
            Err(error) => panic!("regular layout dispatch failed: {error}"),
        };
        let cached = pass
            .run_and_read(&context.device, &context.queue, &input)
            .expect("cached regular layout dispatch should remain readable");
        assert_eq!(cached, actual);
        assert_eq!(actual.len(), expected.len());
        for (index, (gpu_rectangle, expected)) in actual.iter().zip(expected).enumerate() {
            for (component, (gpu, cpu)) in gpu_rectangle
                .iter()
                .zip([expected.x, expected.y, expected.width, expected.height])
                .enumerate()
            {
                assert!(
                    (gpu - cpu).abs() <= 0.0001,
                    "item {index} component {component}: GPU {gpu} != CPU {cpu}; gpu={:?} cpu=({},{},{},{})",
                    gpu_rectangle,
                    expected.x,
                    expected.y,
                    expected.width,
                    expected.height
                );
            }
        }

        for axis in [
            RegularAxis::Column,
            RegularAxis::RowReverse,
            RegularAxis::ColumnReverse,
        ] {
            let mut directional_input = input.clone();
            directional_input.axis = axis;
            let expected = directional_input
                .compute_cpu()
                .expect("CPU reference should accept direction");
            let actual = pass
                .run_and_read(&context.device, &context.queue, &directional_input)
                .expect("directional regular layout dispatch should succeed");
            assert_eq!(actual.len(), expected.len());
            for (index, (gpu_rectangle, expected)) in actual.iter().zip(expected).enumerate() {
                for (gpu, cpu) in gpu_rectangle.iter().zip([
                    expected.x,
                    expected.y,
                    expected.width,
                    expected.height,
                ]) {
                    assert!(
                        (gpu - cpu).abs() <= 0.0001,
                        "axis {axis:?}, item {index}: GPU {gpu_rectangle:?}, CPU ({},{},{},{})",
                        expected.x,
                        expected.y,
                        expected.width,
                        expected.height
                    );
                }
            }
        }
    }
}
