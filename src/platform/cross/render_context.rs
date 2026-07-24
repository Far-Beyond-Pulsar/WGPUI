use std::sync::{Arc, Mutex};

use super::surface_registry::SurfaceRegistry;

/// Options for configuring the WGPU device used by the application.
#[derive(Clone, Debug)]
pub struct WgpuOptions {
    /// Additional WGPU features to request when creating the device.
    /// These are OR'd with the features WGPUI itself requires.
    pub additional_features: wgpu::Features,
}

impl Default for WgpuOptions {
    fn default() -> Self {
        Self {
            additional_features: wgpu::Features::empty(),
        }
    }
}

pub struct WgpuContext {
    pub(super) adapter: wgpu::Adapter,
    pub(super) device: wgpu::Device,
    pub(super) queue: wgpu::Queue,
    pub(super) instance: wgpu::Instance,

    pub(super) globals_buffer: wgpu::Buffer,
    pub(super) quads_buffer: Mutex<wgpu::Buffer>,
    pub(super) shadows_buffer: Mutex<wgpu::Buffer>,
    pub(super) backdrop_filters_buffer: Mutex<wgpu::Buffer>,
    pub(super) underlines_buffer: Mutex<wgpu::Buffer>,
    pub(super) mono_sprites_buffer: Mutex<wgpu::Buffer>,
    pub(super) poly_sprites_buffer: Mutex<wgpu::Buffer>,
    pub(super) color_adjustments_buffer: wgpu::Buffer,
    pub(super) paths_vertices_buffer: Mutex<wgpu::Buffer>,

    pub(crate) surface_registry: Arc<SurfaceRegistry>,

    /// Guards swapchain reconfiguration against concurrent queue submission.
    ///
    /// `Surface::configure` waits for the device to go idle and fails with
    /// `GpuWaitTimeout` if anything submits while it waits — wgpu-core treats a
    /// non-empty queue after the wait as proof that "another thread is submitting
    /// at the same time". Because `WgpuSurfaceHandle` hands out clones of this
    /// device and queue, every external render thread is such a submitter.
    ///
    /// External render threads hold a **read** guard across render + submit;
    /// `Surface::configure` call sites hold the **write** guard. Read guards do
    /// not block each other, so any number of surfaces keep rendering at
    /// independent rates; only reconfiguration (resize) serializes against them.
    ///
    /// The renderer's own submits deliberately do NOT take a read guard: they run
    /// on the same thread as `configure`, so guarding them would self-deadlock.
    pub(crate) gpu_submit_lock: Arc<parking_lot::RwLock<()>>,
}

impl WgpuContext {
    pub fn new(options: &WgpuOptions) -> anyhow::Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: None,
        });

        // Features WGPUI itself needs for its rendering pipeline.
        let wgpui_features = wgpu::Features::TEXTURE_BINDING_ARRAY
            | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
            | wgpu::Features::PRIMITIVE_INDEX
            | wgpu::Features::INDIRECT_FIRST_INSTANCE;

        // Combine with any additional features requested by the application.
        let required_features = wgpui_features | options.additional_features;

        let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));

        // On macOS, some features are optional — prefer adapters that expose them
        // but do not require them, since Metal may not advertise them on all hardware.
        // On all other platforms, require them outright.
        #[cfg(target_os = "macos")]
        let (adapter, device_features) = {
            let optional_features = wgpu::Features::MULTI_DRAW_INDIRECT_COUNT
                | wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
            let adapter = adapters
                .into_iter()
                .filter(|adapter| adapter.features().contains(required_features))
                .max_by_key(|adapter| adapter.features().contains(optional_features))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "No adapter available with required features: {:?}",
                        required_features
                    )
                })?;
            let device_features = if adapter.features().contains(optional_features) {
                required_features | optional_features
            } else {
                required_features
            };
            (adapter, device_features)
        };

        #[cfg(not(target_os = "macos"))]
        let (adapter, device_features) = {
            let required_features = required_features
                | wgpu::Features::MULTI_DRAW_INDIRECT_COUNT
                | wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
            let adapter = adapters
                .into_iter()
                .find(|adapter| adapter.features().contains(required_features))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "No adapter available with required features: {:?}",
                        required_features
                    )
                })?;
            (adapter, required_features)
        };

        // PIPELINE_STATISTICS_QUERY is best-effort on every platform (unlike
        // TIMESTAMP_QUERY/TIMESTAMP_QUERY_INSIDE_ENCODERS above, which are
        // already required outright on non-macOS): cross-backend wgpu 30
        // support for it is thin, so it's runtime-checked rather than folded
        // into `required_features` even on non-macOS.
        #[cfg(feature = "flamegraph")]
        let device_features = {
            let pipeline_statistics_query = wgpu::Features::PIPELINE_STATISTICS_QUERY;
            if adapter.features().contains(pipeline_statistics_query) {
                device_features | pipeline_statistics_query
            } else {
                device_features
            }
        };

        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: device_features,
                required_limits: wgpu::Limits {
                    max_binding_array_elements_per_shader_stage: 512,
                    ..adapter.limits()
                },
                ..Default::default()
            }))?;

        let globals_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Globals Buffer"),
            // FIXME(mdeand): Hack
            size: 16 as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let quads_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Quads Buffer"),
            // TODO(mdeand): Determine appropriate size
            size: 8 * 1024 * 1024, // 1 MB buffer for quads, for now. (:
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let mono_sprites_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Monosprites Buffer"),
            // TODO(mdeand): Determine appropriate size, or make resizable.
            size: 8 * 1024 * 1024,
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let shadows_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadows Buffer"),
            size: 8 * 1024 * 1024,
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let backdrop_filters_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Backdrop Filters Buffer"),
            size: 8 * 1024 * 1024,
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let underlines_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Underlines Buffer"),
            size: 8 * 1024 * 1024,
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let poly_sprites_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Poly Sprites Buffer"),
            size: 8 * 1024 * 1024,
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let paths_vertices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Path Vertices Buffer"),
            size: 8 * 1024 * 1024, // 8 MB – ~174 k vertices @ 48 bytes each
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let color_adjustments_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Color Adjustments Buffer"),
            size: 1024 * 16, // TODO(mdeand): 16 KB buffer for color adjustments, for now. (:
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        Ok(Self {
            adapter,
            device,
            queue,
            instance,

            globals_buffer,
            quads_buffer: Mutex::new(quads_buffer),
            shadows_buffer: Mutex::new(shadows_buffer),
            backdrop_filters_buffer: Mutex::new(backdrop_filters_buffer),
            underlines_buffer: Mutex::new(underlines_buffer),
            mono_sprites_buffer: Mutex::new(mono_sprites_buffer),
            poly_sprites_buffer: Mutex::new(poly_sprites_buffer),
            color_adjustments_buffer,

            paths_vertices_buffer: Mutex::new(paths_vertices_buffer),
            surface_registry: Arc::new(SurfaceRegistry::new()),
            gpu_submit_lock: Arc::new(parking_lot::RwLock::new(())),
        })
    }
}

/// Ensures a buffer is large enough to hold the required size.
/// If the buffer is too small, it will be recreated with the new size.
pub(super) fn ensure_buffer_size(
    device: &wgpu::Device,
    buffer: &Mutex<wgpu::Buffer>,
    required_size: u64,
    label: &str,
    usage: wgpu::BufferUsages,
) {
    let mut buffer_guard = buffer.lock().unwrap();
    let current_size = buffer_guard.size();
    if current_size < required_size {
        // Recreate buffer with new size (add some headroom to avoid frequent reallocations)
        let new_size = (required_size * 3 / 2).max(required_size + 1024 * 1024);
        *buffer_guard = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: new_size,
            usage,
            mapped_at_creation: false,
        });
    }
}
