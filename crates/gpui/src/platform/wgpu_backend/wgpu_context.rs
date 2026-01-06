pub struct WgpuContext {
    pub(super) adapter: wgpu::Adapter,
    pub(super) device: wgpu::Device,
    pub(super) queue: wgpu::Queue,
    pub(super) instance: wgpu::Instance,
    pub(super) globals_buffer: wgpu::Buffer,
    pub(super) quads_buffer: wgpu::Buffer,
    // pub(super) mono_sprites_buffer: wgpu::Buffer,
    // pub(super) poly_sprites_buffer: wgpu::Buffer,
}

impl WgpuContext {
    pub fn new() -> anyhow::Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;

        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::BUFFER_BINDING_ARRAY
                    | wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY,
                required_limits: wgpu::Limits {
                    // TODO(mdeand): Tune these limits as needed
                    max_binding_array_elements_per_shader_stage: 1,
                    max_vertex_attributes: 32,
                    ..Default::default()
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

        // Create a buffer for quads
        let quads_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Quads Buffer"),
            // TODO(mdeand): Determine appropriate size
            size: 1024 * 1024, // 1 MB buffer for quads, for now. (:
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        Ok(Self {
            adapter,
            device,
            queue,
            instance,
            globals_buffer,
            quads_buffer,
        })
    }
}
