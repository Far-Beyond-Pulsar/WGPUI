//! Phase 0, Spike B (docs/gpu-native-architecture.md §8, §11, §6.1):
//! a 10,000-row uniform grid's layout as a compute kernel vs. today's
//! `uniform_list`'s CPU loop (`src/elements/uniform_list.rs`).
//!
//! Run with:
//!
//!     cargo run -p wgpui-wgpu --example spike_b_uniform_layout --release --offline
//!
//! # Methodology
//!
//! **CPU path**: the exact per-item position formula `uniform_list`'s
//! `prepaint` uses today (`src/elements/uniform_list.rs:551-553`):
//!
//! ```ignore
//! let item_origin = padded_bounds.origin
//!     + visual_scroll_offset
//!     + point(Pixels::ZERO, item_height * ix);
//! ```
//!
//! computed for all 10,000 rows (not just the scrolled-into-view subset —
//! `uniform_list` only ever computes the visible range in production, which
//! makes the real CPU cost lower than what's measured here; this spike
//! intentionally computes the full item count on both sides for a fair,
//! apples-to-apples comparison of "compute N positions", not a comparison
//! that also credits the CPU side for a virtualization optimization the GPU
//! side isn't being asked to replicate).
//!
//! **GPU path**: one compute pass, `item_count` invocations, each computing
//! the identical formula from `(item_count, item_height, container origin,
//! scroll offset)` — the exact inputs §6.1 names. Timed end-to-end (buffer
//! creation, upload of the tiny parameter uniform, the compute dispatch,
//! and the final `poll(Wait)`), the same methodology as Spike A.
//!
//! This is the spike most likely to go the *other* way from Spike A: the
//! per-item computation is a single multiply-add, so the fixed cost of a
//! GPU dispatch (buffer/pipeline setup, driver submission, `poll(Wait)`
//! round-trip) is a much larger fraction of the total than in Spike A's
//! much heavier per-item workload. Phase 0's gate is explicit that a spike
//! not winning is a real, useful answer, not a failure of the harness — see
//! docs/phase-0-results.md for the measured result and what it implies for
//! Phase 6.1's scope.

use std::time::Instant;

const ITEM_COUNT: u32 = 10_000;
const ITEM_HEIGHT: f32 = 24.0;
const CONTAINER_ORIGIN: (f32, f32) = (0.0, 0.0);
const SCROLL_OFFSET: (f32, f32) = (0.0, -1234.0);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LayoutParams {
    item_count: u32,
    item_height: f32,
    origin_x: f32,
    origin_y: f32,
    scroll_x: f32,
    scroll_y: f32,
    _pad0: u32,
    _pad1: u32,
}

fn run_cpu_path() -> (Vec<[f32; 2]>, std::time::Duration) {
    let start = Instant::now();
    let mut positions = Vec::with_capacity(ITEM_COUNT as usize);
    for ix in 0..ITEM_COUNT {
        let x = CONTAINER_ORIGIN.0 + SCROLL_OFFSET.0;
        let y = CONTAINER_ORIGIN.1 + SCROLL_OFFSET.1 + ITEM_HEIGHT * ix as f32;
        positions.push([x, y]);
    }
    let elapsed = start.elapsed();
    (positions, elapsed)
}

const LAYOUT_SHADER: &str = r#"
struct Params {
    item_count: u32,
    item_height: f32,
    origin_x: f32,
    origin_y: f32,
    scroll_x: f32,
    scroll_y: f32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> positions: array<vec2<f32>>;

@compute @workgroup_size(64)
fn compute_layout(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ix = gid.x;
    if (ix >= params.item_count) {
        return;
    }
    let x = params.origin_x + params.scroll_x;
    let y = params.origin_y + params.scroll_y + params.item_height * f32(ix);
    positions[ix] = vec2<f32>(x, y);
}
"#;

fn read_storage_buffer_vec2(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    count: usize,
) -> Vec<[f32; 2]> {
    let size = (count * std::mem::size_of::<[f32; 2]>()) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device.poll failed");
    rx.recv()
        .expect("map_async channel closed")
        .expect("buffer map failed");

    let data = slice.get_mapped_range().expect("get_mapped_range failed");
    let values: Vec<[f32; 2]> = bytemuck::cast_slice(&data[..]).to_vec();
    drop(data);
    staging.unmap();
    values
}

fn main() {
    println!("=== Phase 0 Spike B: uniform-list layout, GPU compute kernel vs. CPU loop ===");
    println!("Rows: {ITEM_COUNT}, item_height: {ITEM_HEIGHT}px");

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::DX12,
        flags: wgpu::InstanceFlags::default(),
        backend_options: wgpu::BackendOptions::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        display: None,
    });
    let adapters = pollster::block_on(
        instance.enumerate_adapters(wgpu::Backends::VULKAN | wgpu::Backends::DX12),
    );
    let Some(adapter) = adapters.into_iter().next() else {
        println!(
            "NO GPU ADAPTER AVAILABLE (real or software) — cannot run the GPU half of this spike."
        );
        println!("See examples/adapter_probe.rs for the full honesty report.");
        return;
    };
    let info = adapter.get_info();
    let name_lower = info.name.to_lowercase();
    let is_software = matches!(info.device_type, wgpu::DeviceType::Cpu)
        || name_lower.contains("llvmpipe")
        || name_lower.contains("warp")
        || name_lower.contains("software")
        || name_lower.contains("microsoft basic render");
    println!(
        "Adapter: name={:?} backend={:?} device_type={:?} driver_info={:?} software_fallback={}",
        info.name, info.backend, info.device_type, info.driver_info, is_software
    );

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("spike_b device"),
        ..Default::default()
    }))
    .expect("request_device failed");

    // --- CPU reference path.
    let (cpu_positions, cpu_time) = run_cpu_path();
    println!();
    println!("--- CPU path (src/elements/uniform_list.rs's per-item formula) ---");
    println!(
        "  total: {cpu_time:>10.3?}  ({:.1} ns/row)",
        cpu_time.as_nanos() as f64 / ITEM_COUNT as f64
    );

    // --- GPU path, single dispatch, timed end-to-end.
    let gpu_start = Instant::now();

    let params = LayoutParams {
        item_count: ITEM_COUNT,
        item_height: ITEM_HEIGHT,
        origin_x: CONTAINER_ORIGIN.0,
        origin_y: CONTAINER_ORIGIN.1,
        scroll_x: SCROLL_OFFSET.0,
        scroll_y: SCROLL_OFFSET.1,
        _pad0: 0,
        _pad1: 0,
    };
    let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("layout params"),
        size: std::mem::size_of::<LayoutParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

    let positions_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("positions"),
        size: (ITEM_COUNT as usize * std::mem::size_of::<[f32; 2]>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("layout"),
        source: wgpu::ShaderSource::Wgsl(LAYOUT_SHADER.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("layout pipeline"),
        layout: None,
        module: &module,
        entry_point: Some("compute_layout"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("layout bind group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: positions_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("spike_b"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(ITEM_COUNT.div_ceil(64), 1, 1);
    }
    queue.submit(Some(encoder.finish()));
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device.poll failed");
    let gpu_total = gpu_start.elapsed();

    let gpu_positions =
        read_storage_buffer_vec2(&device, &queue, &positions_buffer, ITEM_COUNT as usize);

    let mut mismatches = 0usize;
    for i in 0..ITEM_COUNT as usize {
        if gpu_positions[i] != cpu_positions[i] {
            mismatches += 1;
        }
    }

    println!();
    println!("--- GPU path (1 compute dispatch, end-to-end) ---");
    println!("  total (buffer create+upload, dispatch, submit, poll): {gpu_total:>10.3?}");
    println!(
        "  position[] exact match vs. CPU: {} / {ITEM_COUNT} ({} mismatches)",
        ITEM_COUNT as usize - mismatches,
        mismatches
    );

    println!();
    println!("--- Summary ---");
    println!("  CPU total: {cpu_time:>10.3?}");
    println!(
        "  GPU total: {gpu_total:>10.3?}  (adapter: {:?}, software_fallback={is_software})",
        info.name
    );
    if gpu_total < cpu_time {
        println!(
            "  GPU path is {:.2}x faster end-to-end on this hardware.",
            cpu_time.as_secs_f64() / gpu_total.as_secs_f64()
        );
    } else {
        println!(
            "  GPU path is {:.2}x SLOWER end-to-end on this hardware -- at this problem size and this \
             per-item workload, dispatch/submit/poll overhead dominates the actual (trivial) per-item \
             math. See docs/phase-0-results.md for discussion of what this implies for Phase 6.1's scope.",
            gpu_total.as_secs_f64() / cpu_time.as_secs_f64()
        );
    }
}
