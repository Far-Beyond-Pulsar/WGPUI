pub mod atlas;
pub mod dispatcher;
pub mod keyboard;
pub mod platform;
pub mod render_context;
pub mod renderer;
pub mod resize_detector;
pub mod surface_registry;
pub mod text_system;
pub mod window;

/// Re-export so the `PlatformWindow::with_winit_window` trait method can name this type
/// without pulling winit into every file that uses `platform.rs`.
pub use winit::window::Window as WinitWindow;
