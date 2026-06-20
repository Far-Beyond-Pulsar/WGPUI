//! Example WGPUI render-primitive plugin: solid-color rectangles.
//!
//! This crate demonstrates the multi-crate primitive architecture. It depends
//! **only** on `gpui-render-primitive` (not on the `gpui` core), so editing it
//! recompiles only this crate. The core registers a [`SolidQuads`] once and then
//! draws every queued [`SolidQuad`] instance for a frame in a single batched call.
//!
//! Build a [`SolidQuad`] with [`SolidQuad::new`], get its raw bytes with
//! [`SolidQuad::as_bytes`], and submit it through the core's
//! `Window::paint_custom_primitive(SolidQuads::TYPE, bytes, bounds)`.

use std::any::Any;

use bytemuck::{Pod, Zeroable};
use gpui_render_primitive::{
    BatchDraw,
    PipelineContext,
    PrimBounds,
    PrimitiveTypeId,
    RenderPrimitive,
    wgpu,
};

/// One solid-color rectangle, in scaled (device) pixels. `repr(C)` + `Pod` so the
/// core can append its bytes into a batch and the GPU can read them as instances.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SolidQuad {
    /// Top-left corner, scaled pixels.
    pub origin: [f32; 2],
    /// Width/height, scaled pixels.
    pub size: [f32; 2],
    /// Premultiplied RGBA, 0..1.
    pub color: [f32; 4],
}

impl SolidQuad {
    pub fn new(origin: [f32; 2], size: [f32; 2], color: [f32; 4]) -> Self {
        Self {
            origin,
            size,
            color,
        }
    }

    /// Raw instance bytes to hand to the core's `paint_custom_primitive`.
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }

    /// Screen bounds for damage/ordering.
    pub fn bounds(&self) -> PrimBounds {
        PrimBounds::new(self.origin[0], self.origin[1], self.size[0], self.size[1])
    }
}

/// The registered primitive-type plugin for [`SolidQuad`].
pub struct SolidQuads;

impl SolidQuads {
    pub const TYPE: PrimitiveTypeId = PrimitiveTypeId("gpui_prim_solid::SolidQuad");

    const SHADER: &'static str = r#"
struct Globals { viewport_size: vec2<f32> };
@group(0) @binding(0) var<uniform> globals: Globals;

struct Instance {
    @location(0) origin: vec2<f32>,
    @location(1) size: vec2<f32>,
    @location(2) color: vec4<f32>,
};

struct VsOut { @builtin(position) pos: vec4<f32>, @location(0) color: vec4<f32> };

@vertex
fn vs_main(@builtin(vertex_index) vid: u32, inst: Instance) -> VsOut {
    // Two-triangle quad in [0,1]^2.
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
    );
    let local = corners[vid];
    let px = inst.origin + local * inst.size;
    // Pixel -> normalized device coords (y down -> y up).
    let ndc = vec2<f32>(px.x / globals.viewport_size.x * 2.0 - 1.0,
                        1.0 - px.y / globals.viewport_size.y * 2.0);
    var out: VsOut;
    out.pos = vec4<f32>(ndc, 0.0, 1.0);
    out.color = inst.color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> { return in.color; }
"#;
}

/// Opaque GPU state cached by the core between frames.
struct SolidPipeline {
    pipeline: wgpu::RenderPipeline,
    instances: wgpu::Buffer,
    capacity: u64,
}

impl RenderPrimitive for SolidQuads {
    fn type_key(&self) -> PrimitiveTypeId {
        Self::TYPE
    }

    fn build(&self, ctx: &PipelineContext<'_>) -> Box<dyn Any + Send + Sync> {
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gpui_prim_solid"),
                source: wgpu::ShaderSource::Wgsl(Self::SHADER.into()),
            });
        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gpui_prim_solid.layout"),
                bind_group_layouts: &[Some(ctx.globals_layout)],
                immediate_size: 0,
            });
        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SolidQuad>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32x4],
        };
        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("gpui_prim_solid.pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[instance_layout],
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: ctx.surface_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                multiview_mask: None,
                cache: None,
            });
        let instances = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpui_prim_solid.instances"),
            size: 0,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Box::new(SolidPipeline {
            pipeline,
            instances,
            capacity: 0,
        })
    }

    fn draw(&self, state: &mut (dyn Any + Send + Sync), batch: BatchDraw<'_, '_>) {
        let Some(state) = state.downcast_mut::<SolidPipeline>() else {
            return;
        };
        let needed = batch.instances.len() as u64;
        if needed == 0 {
            return;
        }
        if needed > state.capacity {
            state.instances = batch.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpui_prim_solid.instances"),
                size: needed,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            state.capacity = needed;
        }
        batch
            .queue
            .write_buffer(&state.instances, 0, batch.instances);
        batch.pass.set_pipeline(&state.pipeline);
        batch.pass.set_bind_group(0, batch.globals, &[]);
        batch
            .pass
            .set_vertex_buffer(0, state.instances.slice(0..needed));
        batch.pass.draw(0..6, 0..batch.instance_count);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solid_quad_is_pod_and_round_trips_bytes() {
        let q = SolidQuad::new([10.0, 20.0], [30.0, 40.0], [1.0, 0.5, 0.25, 1.0]);
        let bytes = q.as_bytes();
        assert_eq!(bytes.len(), std::mem::size_of::<SolidQuad>());
        let back: SolidQuad = *bytemuck::from_bytes(bytes);
        assert_eq!(back.origin, [10.0, 20.0]);
        assert_eq!(back.color[1], 0.5);
        assert_eq!(q.bounds(), PrimBounds::new(10.0, 20.0, 30.0, 40.0));
    }

    #[test]
    fn type_key_is_stable() {
        assert_eq!(SolidQuads.type_key(), SolidQuads::TYPE);
    }
}
