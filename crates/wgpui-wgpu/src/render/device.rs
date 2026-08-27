//! Device/queue creation and feature negotiation.
//! See docs/gpu-native-architecture.md §3.5 (this file's eventual role is
//! today's `src/platform/cross/render_context.rs`).
//!
//! Phase 3 needs one thing from this file and does not build the rest: a
//! **headless** device, with no surface and no window, for the compute passes
//! §5.1/§5.2 add. Porting `render_context.rs`'s surface configuration, present
//! modes, and swapchain handling is windowing work that belongs with the rest
//! of `window/`, and nothing in this phase would exercise it.
//!
//! # Features, deliberately none
//!
//! §8's Phase 3 row asks for compute ordering and occlusion, and neither needs
//! a device feature: compute shaders, storage buffers, and atomics are core
//! WebGPU. `INDIRECT_FIRST_INSTANCE` and `MULTI_DRAW_INDIRECT_COUNT` — which
//! `render_context.rs:104-176` already negotiates and which §1's table notes
//! nothing has ever used — stay unrequested here, because Phase 4 is what
//! issues an indirect draw and Phase 4 is where requesting them earns its
//! keep. Requesting a feature this phase cannot exercise would make a device
//! that works today fail to open for no benefit.

/// Why a headless compute device could not be opened.
///
/// Reported rather than panicked so a caller on a machine with no usable
/// adapter — a CI container, a remote session with no GPU — can say so plainly
/// instead of aborting. Phase 0's own report makes this distinction load
/// bearing: a missing adapter invalidates a performance claim, not a
/// correctness one.
#[derive(Debug)]
pub enum ContextError {
    /// No adapter at all, software included.
    NoAdapter(wgpu::RequestAdapterError),
    /// An adapter exists but would not open a device.
    NoDevice(wgpu::RequestDeviceError),
}

impl std::fmt::Display for ContextError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContextError::NoAdapter(error) => {
                write!(formatter, "no wgpu adapter available: {error}")
            }
            ContextError::NoDevice(error) => {
                write!(formatter, "adapter would not open a device: {error}")
            }
        }
    }
}

impl std::error::Error for ContextError {}

/// A live device and queue, plus what the adapter behind them actually is.
pub struct ComputeContext {
    /// The open device.
    pub device: wgpu::Device,
    /// Its queue.
    pub queue: wgpu::Queue,
    /// The adapter's self-report, carried so every measurement can name the
    /// hardware it ran on — Phase 0's honesty standard, kept.
    pub adapter_info: wgpu::AdapterInfo,
}

impl ComputeContext {
    /// Whether the adapter is a CPU/software rasterizer rather than real
    /// hardware.
    ///
    /// A software adapter is perfectly good for the correctness half of §8's
    /// Phase 3 gate and worthless for the performance half, so every caller
    /// that reports a timing has to be able to ask.
    pub fn is_software(&self) -> bool {
        is_software_adapter(&self.adapter_info)
    }

    /// A one-line description for a report or a test's output.
    pub fn describe(&self) -> String {
        format!(
            "{} ({:?}, {:?}, driver {:?}{})",
            self.adapter_info.name,
            self.adapter_info.backend,
            self.adapter_info.device_type,
            self.adapter_info.driver_info,
            if self.is_software() {
                ", SOFTWARE FALLBACK"
            } else {
                ""
            }
        )
    }
}

/// Whether an adapter is a software rasterizer.
///
/// `device_type` alone is not enough: WARP and llvmpipe do report `Cpu`, but
/// the name check is what `examples/adapter_probe.rs` already uses and is kept
/// identical so the two agree about the same machine.
pub fn is_software_adapter(info: &wgpu::AdapterInfo) -> bool {
    let name = info.name.to_lowercase();
    matches!(info.device_type, wgpu::DeviceType::Cpu)
        || name.contains("llvmpipe")
        || name.contains("warp")
        || name.contains("software")
        || name.contains("microsoft basic render")
}

/// Open a headless device for compute work, preferring real hardware.
///
/// Uses `request_adapter` with `HighPerformance` rather than
/// `enumerate_adapters().next()`: Phase 0's spikes took the first enumerated
/// adapter because they were probing what the machine had, but a pass that
/// ships wants the driver's own preference, and `request_adapter` is what
/// declines a software fallback when hardware is present.
pub fn headless_compute_context() -> Result<ComputeContext, ContextError> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::DX12,
        flags: wgpu::InstanceFlags::default(),
        backend_options: wgpu::BackendOptions::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        display: None,
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        ..Default::default()
    }))
    .map_err(ContextError::NoAdapter)?;
    let adapter_info = adapter.get_info();

    // The adapter's own limits rather than `Limits::default()`: the ordering
    // pass binds eight storage buffers in one group, which is exactly the
    // downlevel default's ceiling, and leaving no headroom would make an
    // unrelated later binding fail on hardware that has room for it.
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("wgpui compute"),
        required_features: wgpu::Features::empty(),
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .map_err(ContextError::NoDevice)?;

    Ok(ComputeContext {
        device,
        queue,
        adapter_info,
    })
}
