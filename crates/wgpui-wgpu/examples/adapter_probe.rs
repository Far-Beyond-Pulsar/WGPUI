//! Phase 0 (docs/gpu-native-architecture.md §8, §11): before trusting any
//! spike number, find out what GPU adapter (if any) this environment
//! actually hands back. Run with:
//!
//!     cargo run -p wgpui-wgpu --example adapter_probe --offline
//!
//! Mirrors the legacy backend's own established pattern for headless device
//! creation (`src/flamegraph_gpu.rs`'s `create_headless_device_with_features`,
//! `src/platform/cross/render_context.rs`'s `WgpuContext::new`):
//! `enumerate_adapters` + pick, not `request_adapter` — there is no window
//! here to supply a `compatible_surface`, and this is the pattern the crate
//! already uses for exactly that case. Prints every adapter `wgpu`
//! enumerates on the native backends (Vulkan, DX12) and reports, honestly,
//! whether the one this probe would use for a device is real hardware or a
//! software/CPU fallback (llvmpipe, WARP, etc.) — see the module doc on
//! `create_headless_device_with_features` for why a CI/sandbox environment
//! with neither is an expected, not exceptional, outcome.

fn main() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::DX12,
        flags: wgpu::InstanceFlags::default(),
        backend_options: wgpu::BackendOptions::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        display: None,
    });

    println!("== Adapters enumerated on VULKAN | DX12 ==");
    let adapters = pollster::block_on(
        instance.enumerate_adapters(wgpu::Backends::VULKAN | wgpu::Backends::DX12),
    );
    if adapters.is_empty() {
        println!(
            "(none enumerated -- no usable GPU adapter, real or software, on this backend set)"
        );
    }
    for adapter in &adapters {
        let info = adapter.get_info();
        println!(
            "  name={:?} backend={:?} device_type={:?} driver={:?} driver_info={:?} vendor={:#x} device_id={:#x}",
            info.name,
            info.backend,
            info.device_type,
            info.driver,
            info.driver_info,
            info.vendor,
            info.device
        );
    }

    println!();
    println!(
        "== Picking the first adapter (this crate's established pattern) and requesting a device =="
    );
    let Some(adapter) = adapters.into_iter().next() else {
        println!("  No adapter available at all -- spikes cannot run a real compute pass here.");
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
        "  Selected adapter: name={:?} backend={:?} device_type={:?} driver_info={:?}",
        info.name, info.backend, info.device_type, info.driver_info
    );
    println!(
        "  software/CPU-fallback adapter: {}",
        if is_software { "YES" } else { "no" }
    );

    match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("adapter_probe device"),
        ..Default::default()
    })) {
        Ok(_) => println!("  request_device: OK"),
        Err(error) => println!("  request_device: FAILED: {error}"),
    }
}
