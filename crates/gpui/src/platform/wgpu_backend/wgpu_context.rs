pub struct WgpuContext {
    pub(super) device: wgpu::Device,
    pub(super) queue: wgpu::Queue,
    pub(super) instance: wgpu::Instance,
}
