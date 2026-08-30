//! Phase 6, before any design: can this machine open a real window at all, and
//! what does a `wgpu::Surface` bound to it actually report?
//!
//! Phase 0 checked adapter availability directly rather than assuming it, and
//! every result since has cited that probe. This is the same act one level up:
//! the window/present path's entire premise is that an OS window exists and a
//! swapchain can be configured on it, and neither of those is knowable from a
//! headless test. It answers four questions and exits:
//!
//! 1. Does `EventLoop::new()` succeed on this platform/session?
//! 2. Does `create_window` return a window, and does it report itself visible?
//! 3. Which formats, present modes and alpha modes does the surface support —
//!    specifically, is `render::pipelines::TARGET_FORMAT` among them, since
//!    every pipeline Phases 4–5.6 built is compiled against exactly that one?
//! 4. Can `get_current_texture` / `present` run repeatedly without error?
//!
//! Deliberately not a library type: this exists to be read once, and the answers
//! it produced are recorded in `docs/phase-6-results.md`.

use std::sync::Arc;

use wgpui_wgpu::render::device;
use wgpui_wgpu::render::pipelines::TARGET_FORMAT;

struct Probe {
    frames_presented: u32,
    reported: bool,
}

impl winit::application::ApplicationHandler for Probe {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.reported {
            return;
        }
        self.reported = true;

        let attributes = winit::window::Window::default_attributes()
            .with_title("wgpui 2.0 — Phase 6 window probe")
            .with_inner_size(winit::dpi::PhysicalSize::new(640u32, 360u32));
        let window = match event_loop.create_window(attributes) {
            Ok(window) => Arc::new(window),
            Err(error) => {
                println!("create_window FAILED: {error}");
                event_loop.exit();
                return;
            }
        };
        println!("create_window: OK");
        println!("  inner_size    = {:?}", window.inner_size());
        println!("  scale_factor  = {}", window.scale_factor());
        println!("  is_visible    = {:?}", window.is_visible());
        println!("  is_minimized  = {:?}", window.is_minimized());

        let instance = device::instance();
        let surface = match instance.create_surface(Arc::clone(&window)) {
            Ok(surface) => surface,
            Err(error) => {
                println!("create_surface FAILED: {error}");
                event_loop.exit();
                return;
            }
        };
        println!("create_surface: OK");

        let context = match device::context_for(&instance, Some(&surface)) {
            Ok(context) => context,
            Err(error) => {
                println!("context_for(surface) FAILED: {error}");
                event_loop.exit();
                return;
            }
        };
        println!("device: {}", context.describe());

        let capabilities = surface.get_capabilities(&context.adapter);
        println!("surface capabilities:");
        println!("  formats       = {:?}", capabilities.formats);
        println!("  present_modes = {:?}", capabilities.present_modes);
        println!("  alpha_modes   = {:?}", capabilities.alpha_modes);
        println!("  usages        = {:?}", capabilities.usages);
        println!(
            "  TARGET_FORMAT ({TARGET_FORMAT:?}) supported = {}",
            capabilities.formats.contains(&TARGET_FORMAT)
        );

        let size = window.inner_size();
        let configuration = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            format: if capabilities.formats.contains(&TARGET_FORMAT) {
                TARGET_FORMAT
            } else {
                capabilities.formats[0]
            },
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: capabilities.alpha_modes[0],
            color_space: wgpu::SurfaceColorSpace::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&context.device, &configuration);
        println!(
            "configure: OK at {}x{}",
            configuration.width, configuration.height
        );

        for frame in 0..8u32 {
            let acquired = surface.get_current_texture();
            let texture = match acquired {
                wgpu::CurrentSurfaceTexture::Success(texture)
                | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
                other => {
                    println!("frame {frame}: acquire returned {other:?}");
                    break;
                }
            };
            let view = texture
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            let mut encoder =
                context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("probe clear"),
                    });
            {
                let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("probe clear"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 1.0,
                                g: 0.0,
                                b: 1.0,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: Default::default(),
                });
            }
            context.queue.submit(Some(encoder.finish()));
            window.pre_present_notify();
            context.queue.present(texture);
            self.frames_presented += 1;
        }
        println!("frames presented: {}", self.frames_presented);
        println!("  is_visible after present = {:?}", window.is_visible());
        event_loop.exit();
    }

    fn window_event(
        &mut self,
        _event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        _event: winit::event::WindowEvent,
    ) {
    }
}

fn main() {
    let event_loop = match winit::event_loop::EventLoop::new() {
        Ok(event_loop) => {
            println!("EventLoop::new: OK");
            event_loop
        }
        Err(error) => {
            println!("EventLoop::new FAILED: {error}");
            println!("This environment cannot host a winit event loop at all.");
            return;
        }
    };
    let mut probe = Probe {
        frames_presented: 0,
        reported: false,
    };
    match event_loop.run_app(&mut probe) {
        Ok(()) => println!("run_app returned cleanly"),
        Err(error) => println!("run_app FAILED: {error}"),
    }
}
