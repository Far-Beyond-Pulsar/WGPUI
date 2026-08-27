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
//! # Features
//!
//! §8's Phase 3 row asks for compute ordering and occlusion, and neither needs
//! a device feature: compute shaders, storage buffers, and atomics are core
//! WebGPU. Phase 3 therefore requested none, on the stated principle that
//! "requesting a feature this phase cannot exercise would make a device that
//! works today fail to open for no benefit."
//!
//! **Phase 4 is where the indirect-draw features earn their keep**, so this
//! file now negotiates the two indirect-draw features
//! `render_context.rs:104-176` already asks for —
//! `INDIRECT_FIRST_INSTANCE` and `MULTI_DRAW_INDIRECT_COUNT` — and reports what
//! it got as [`IndirectSupport`]. (There is no third: `MULTI_DRAW_INDIRECT` is
//! not a feature in wgpu 30 at all, which [`IndirectSupport`]'s own doc
//! explains.) It negotiates them **best-effort on every platform**,
//! which is deliberately weaker than what the legacy path does (hard-required
//! on native outside macOS) and is not an oversight:
//!
//! - Phase 3's stated principle still holds. A CI container, a remote session,
//!   or a WARP fallback would stop opening a device at all if these were hard
//!   requirements, and every Phase 3 test and benchmark opens its device
//!   through this function.
//! - §5.3's CPU-readback fallback exists exactly so a missing feature is a
//!   slower path rather than a failure. Making the feature mandatory here would
//!   make the fallback unreachable, and an unreachable fallback is an untested
//!   one.
//! - The per-slot draw path (`render/draw.rs`) needs none of the three. It is
//!   the default and it is what runs on a device that reports nothing, which is
//!   also what WebGPU reports.
//!
//! The legacy backend's hard requirement is right for *its* job — it opens one
//! device for a real window and would rather fail loudly at startup than
//! silently lose the fast path — and nothing here changes it.

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

/// Which of §5.3's indirect-draw features the open device actually has.
///
/// Reported rather than assumed, because both are optional here (see this
/// module's doc) and because §8's Phase 4 gate is only meaningful if a report
/// can say which path it measured.
///
/// # Two things `wgpu = "30"` turns out to say that §5.3 does not
///
/// Both were found by trying to write `wgpu::Features::MULTI_DRAW_INDIRECT` and
/// having it not exist, and both are recorded here rather than only in the
/// phase report, because they change what the code below can honestly claim.
///
/// 1. **There is no `MULTI_DRAW_INDIRECT` feature in wgpu 30.**
///    `RenderPass::multi_draw_indirect` is always callable, and where the
///    backend cannot do it natively wgpu *emulates it as a series of
///    `draw_indirect` calls* — on the CPU, inside `wgpu-core`. So calling it is
///    never wrong, and it only stops being a per-slot CPU loop when
///    `MULTI_DRAW_INDIRECT_COUNT` is present, whose own documentation says so:
///    "This feature being present also implies that all calls to
///    `multi_draw_indirect` … are not being emulated." That single feature is
///    therefore what [`Self::multi_draw_count`] tracks, and what
///    [`Self::supports_native_multi_draw`] answers.
///
/// 2. **`README.md`'s "Custom Device Gotcha" is now wgpu's own rule, not just a
///    driver's habit.** `InstanceFlags::VALIDATION_INDIRECT_CALL` — on by
///    default — states that "if `Features::INDIRECT_FIRST_INSTANCE` is not
///    enabled on the device, the `first_instance` indirect argument must be 0",
///    and that violating calls are *transformed into no-ops*. The failure the
///    README describes as drivers silently dropping a draw is the same failure,
///    reproducible on purpose, at the API layer. It is the direct justification
///    for `wgpui_core::indirect::FirstInstance::Zero` being the default rather
///    than an accommodation.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct IndirectSupport {
    /// `INDIRECT_FIRST_INSTANCE`: an indirect draw may carry a nonzero
    /// `firstInstance`. Without it the draw is dropped — by the driver
    /// (`README.md`'s "Custom Device Gotcha", already hit in production by this
    /// crate for externally embedded content) or, in wgpu 30, by wgpu itself.
    pub first_instance: bool,
    /// `MULTI_DRAW_INDIRECT_COUNT`: the record count may be read from a GPU
    /// buffer — and, per its own documentation, `multi_draw_indirect` is native
    /// rather than emulated as a CPU-side loop of `draw_indirect`.
    pub multi_draw_count: bool,
}

impl IndirectSupport {
    /// A device with neither — WebGPU's own position, and what the per-slot
    /// draw path is written to run on.
    pub const NONE: IndirectSupport = IndirectSupport {
        first_instance: false,
        multi_draw_count: false,
    };

    /// What an adapter or device reports.
    pub fn from_features(features: wgpu::Features) -> IndirectSupport {
        IndirectSupport {
            first_instance: features.contains(wgpu::Features::INDIRECT_FIRST_INSTANCE),
            multi_draw_count: features.contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT),
        }
    }

    /// The features this crate asks for, best-effort.
    pub fn wanted() -> wgpu::Features {
        wgpu::Features::INDIRECT_FIRST_INSTANCE | wgpu::Features::MULTI_DRAW_INDIRECT_COUNT
    }

    /// Whether a `multi_draw_indirect` genuinely collapses the CPU's work.
    ///
    /// Both halves are required and each rules out a different failure. Without
    /// `MULTI_DRAW_INDIRECT_COUNT` the call is emulated as the same per-slot
    /// loop with an extra layer, so it buys the CPU nothing. Without
    /// `INDIRECT_FIRST_INSTANCE` it is *wrong*: one call covers many records
    /// and no bind group can change between them, so each record addresses its
    /// own instance range through `firstInstance`, which is precisely the
    /// argument a device lacking the feature refuses to honour.
    pub const fn supports_native_multi_draw(self) -> bool {
        self.multi_draw_count && self.first_instance
    }

    /// A one-line description for a report or a test's output.
    pub fn describe(self) -> String {
        format!(
            "INDIRECT_FIRST_INSTANCE={} MULTI_DRAW_INDIRECT_COUNT={}",
            self.first_instance, self.multi_draw_count
        )
    }
}

/// A live device and queue, plus what the adapter behind them actually is.
pub struct ComputeContext {
    /// The open device.
    pub device: wgpu::Device,
    /// Its queue.
    pub queue: wgpu::Queue,
    /// The adapter's self-report, carried so every measurement can name the
    /// hardware it ran on — Phase 0's honesty standard, kept.
    pub adapter_info: wgpu::AdapterInfo,
    /// Which of §5.3's features the device actually opened with.
    pub indirect: IndirectSupport,
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
            "{} ({:?}, {:?}, driver {:?}{}) [{}]",
            self.adapter_info.name,
            self.adapter_info.backend,
            self.adapter_info.device_type,
            self.adapter_info.driver_info,
            if self.is_software() {
                ", SOFTWARE FALLBACK"
            } else {
                ""
            },
            self.indirect.describe(),
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

    // Best-effort, per this module's doc: whatever of §5.3's three features
    // this adapter has, and nothing required. `request_device` rejects a
    // feature the adapter does not advertise, so the intersection is the
    // request rather than a retry loop.
    let requested = IndirectSupport::wanted() & adapter.features();

    // The adapter's own limits rather than `Limits::default()`: the ordering
    // pass binds eight storage buffers in one group, which is exactly the
    // downlevel default's ceiling, and leaving no headroom would make an
    // unrelated later binding fail on hardware that has room for it.
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("wgpui compute"),
        required_features: requested,
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .map_err(ContextError::NoDevice)?;

    // Read back off the *device*, not off the adapter: the request above is
    // what was asked for and this is what was granted, and a report that
    // conflates the two would be claiming a path it never took.
    let indirect = IndirectSupport::from_features(device.features());

    Ok(ComputeContext {
        device,
        queue,
        adapter_info,
        indirect,
    })
}

/// Open a device for `test`, or print why it did not and return `None`.
///
/// Phase 3 spelled this out inline in its one integration test; Phase 4 has
/// three test files and a benchmark that need the identical behaviour, and
/// duplicating it four times is how the four drift into three different notions
/// of "skipped". The rule it encodes is Phase 0's: a missing adapter is
/// reported plainly, never allowed to look like coverage that ran.
pub fn context_or_report(test: &str) -> Option<ComputeContext> {
    match headless_compute_context() {
        Ok(context) => {
            println!("{test}: running on {}", context.describe());
            Some(context)
        }
        Err(error) => {
            println!(
                "{test}: SKIPPED — {error}. The headless half of this gate (in \
                 wgpui-core) still ran; the GPU half did not, and a human must \
                 re-run this on hardware."
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multi_draw_is_refused_unless_both_halves_are_present() {
        // README's "Custom Device Gotcha": a multi-draw addresses each record's
        // instance range through `firstInstance`, so a device with the count
        // feature but not the first-instance feature would drop every record
        // whose base is nonzero — which is all but the first.
        assert!(
            !IndirectSupport {
                first_instance: false,
                multi_draw_count: true,
            }
            .supports_native_multi_draw()
        );
        // And the other way round is not wrong, only pointless: wgpu 30
        // emulates `multi_draw_indirect` as a CPU-side loop of `draw_indirect`
        // without the count feature, so it saves the CPU nothing.
        assert!(
            !IndirectSupport {
                first_instance: true,
                multi_draw_count: false,
            }
            .supports_native_multi_draw()
        );
        assert!(
            IndirectSupport {
                first_instance: true,
                multi_draw_count: true,
            }
            .supports_native_multi_draw()
        );
        assert!(!IndirectSupport::NONE.supports_native_multi_draw());
    }

    #[test]
    fn support_is_read_out_of_the_feature_set_rather_than_assumed() {
        assert_eq!(
            IndirectSupport::from_features(wgpu::Features::empty()),
            IndirectSupport::NONE
        );
        assert_eq!(
            IndirectSupport::from_features(IndirectSupport::wanted()),
            IndirectSupport {
                first_instance: true,
                multi_draw_count: true,
            }
        );
    }
}
