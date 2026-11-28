pub mod atlas;
pub mod dispatcher;
pub mod keyboard;
#[cfg(target_os = "macos")]
mod macos_open_urls;
pub mod platform;
pub mod render_context;
pub mod renderer;
pub mod resize_detector;
pub mod slab;
pub mod slab_gpu;
pub mod surface_registry;
pub mod text_system;
pub mod window;
#[cfg(target_os = "windows")]
pub(crate) mod windows_touchpad;

/// Re-export so the `PlatformWindow::with_winit_window` trait method can name this type
/// without pulling winit into every file that uses `platform.rs`.
pub use winit::window::Window as WinitWindow;
