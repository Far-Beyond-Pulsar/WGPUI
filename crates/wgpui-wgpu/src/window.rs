//! A real OS window, the `wgpu::Surface` bound to it, and the present loop.
//! See docs/gpu-native-architecture.md §3.5, §9's window/present risk row,
//! and §11's action 1.
//!
//! # What this closes
//!
//! Phases 0–5.6 proved every claim headlessly or by GPU readback into an
//! offscreen texture, and §9 recorded the consequence plainly: nothing in the
//! rearchitecture had ever been on a screen. This file is the missing half —
//! `winit` creates a window, `wgpu` binds a swapchain to it, and
//! [`WindowSurface::acquire`] hands the frame renderer a `TextureView` that the
//! display will actually scan out.
//!
//! # The one thing that made this small, and it was checked rather than assumed
//!
//! Every render pipeline Phases 4–5.6 built is compiled against
//! [`crate::render::pipelines::TARGET_FORMAT`] (`Rgba8Unorm`), and a
//! `RenderPipeline`'s colour target format is fixed at creation. Had the
//! swapchain been unable to offer that format, this phase would have needed
//! either format-parametric pipelines or an offscreen-plus-blit path — both
//! real work, and both changing what the offscreen tests actually prove about
//! the on-screen frame.
//!
//! It does not need either. `Surface::get_capabilities` on this machine reports
//! `[Bgra8UnormSrgb, Rgba8UnormSrgb, Bgra8Unorm, Rgba8Unorm, Rgba16Float,
//! Rgb10a2Unorm]`, so [`SurfaceFormatChoice`] takes `TARGET_FORMAT` directly and
//! the pixels a test reads back offscreen and the pixels the display scans out
//! come from the same pipeline writing the same format. Where a surface cannot
//! offer it, [`WindowSurface::new`] fails with [`WindowError::NoTargetFormat`]
//! and says so, rather than quietly configuring a format the pipelines will
//! refuse at draw time — see that variant's doc for why refusing is the honest
//! choice here.
//!
//! # sRGB, and why the non-sRGB format is the right one
//!
//! The legacy renderer picks the first *non*-sRGB surface format on purpose
//! (`src/platform/cross/renderer.rs`: "the shaders output sRGB values directly,
//! so we need a non-sRGB surface format to avoid a double linear-to-sRGB
//! conversion"). 2.0's shaders have the same property — `quads.wgsl` writes the
//! colour it was given — and `TARGET_FORMAT` is already the non-sRGB one, so
//! preferring it agrees with the legacy choice rather than merely happening to.
//!
//! # What this file is not
//!
//! There is no input plumbing here. `window/keyboard.rs`, `window/dispatcher.rs`
//! and `window/app_menu.rs` are still the Phase 0 stubs they have always been;
//! §11's action 1 named "winit event loop, surface/swapchain configuration,
//! resize handling, an actual runnable entry point" and this file is those four
//! and nothing else. Claiming otherwise would be the exact failure mode this
//! phase exists to end.

pub mod app_menu;
pub mod dispatcher;
pub mod frame_loop;
pub mod keyboard;
pub mod resize_detector;

use std::sync::Arc;

use crate::render::device::{ComputeContext, ContextError};
use crate::render::pipelines::TARGET_FORMAT;

/// Why a window's swapchain could not be brought up.
#[derive(Debug)]
pub enum WindowError {
    /// `wgpu` could not create a surface for this window handle.
    Surface(wgpu::CreateSurfaceError),
    /// No adapter would open a device able to present to this surface.
    Context(ContextError),
    /// The surface cannot present [`TARGET_FORMAT`].
    ///
    /// Reported rather than worked around. Every pipeline in
    /// `render/pipelines.rs` fixes its colour target format at creation, so a
    /// swapchain in another format is not a degraded path — it is a device
    /// error at the first draw call. The two real fixes (format-parametric
    /// pipelines, or an offscreen target blitted to the swapchain) are both
    /// larger than this phase and neither is guessed at here: a machine that
    /// hits this deserves the message rather than a silent second render pass
    /// nothing has ever measured.
    NoTargetFormat(Vec<wgpu::TextureFormat>),
    /// The surface reports no formats at all — typically an adapter that cannot
    /// present to this window.
    NoFormats,
}

impl std::fmt::Display for WindowError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WindowError::Surface(error) => write!(formatter, "could not create surface: {error}"),
            WindowError::Context(error) => write!(formatter, "{error}"),
            WindowError::NoTargetFormat(formats) => write!(
                formatter,
                "surface cannot present {TARGET_FORMAT:?}, which every render pipeline is \
                 compiled against; it offers {formats:?}"
            ),
            WindowError::NoFormats => {
                write!(formatter, "surface reports no supported formats at all")
            }
        }
    }
}

impl std::error::Error for WindowError {}

impl From<ContextError> for WindowError {
    fn from(error: ContextError) -> Self {
        WindowError::Context(error)
    }
}

/// Which format a surface was configured with, and whether it is the one the
/// pipelines want.
///
/// Only one variant is reachable today, because [`WindowError::NoTargetFormat`]
/// rejects the other case. It exists as a named value rather than a `bool` so a
/// report can state which format the frames it is describing were presented in
/// without re-deriving it from capability lists.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SurfaceFormatChoice {
    /// The swapchain is [`TARGET_FORMAT`]: pipelines draw straight into it and
    /// a readback of the presented image is byte-comparable with an offscreen
    /// render of the same scene.
    Target,
}

/// How a frame acquisition ended.
///
/// Three outcomes rather than two, for the same reason
/// `DrawStats::glyph_slots_unavailable` exists in Phase 5.6: "the window could
/// not be drawn to" is genuinely not "the draw was skipped" and genuinely not
/// "an error occurred", and collapsing it into either would make the loop's own
/// counters lie about what happened.
pub enum Acquired {
    /// The swapchain image to draw into. Hand it to
    /// [`WindowSurface::present`] when the work targeting it is submitted.
    Frame(wgpu::SurfaceTexture),
    /// Nothing to present to this frame, and nothing is wrong.
    ///
    /// A minimized or fully occluded window, or an acquire that timed out
    /// because the compositor has not released an image yet.
    Skipped(SkipReason),
    /// Acquire failed, was reconfigured, and failed again.
    Lost,
}

/// Why a frame had no swapchain image.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SkipReason {
    /// The window is minimized or entirely behind another.
    Occluded,
    /// The compositor did not hand an image back in time.
    Timeout,
    /// The window has a zero-sized client area, so there is no swapchain.
    ZeroSized,
}

/// What a window's swapchain has done since it was created.
///
/// Every field is a count of a real call, not of an intention. Phase 6's resize
/// evidence is these numbers before and after a scripted resize sequence: a
/// reconfiguration that did not happen, a retry that was needed, or an acquire
/// that was lost all show up here rather than in a narrative.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct SurfaceStats {
    /// `Surface::configure` calls, including the one at creation.
    pub configures: u64,
    /// `get_current_texture` calls that returned an image.
    pub acquires: u64,
    /// Acquires that returned `Suboptimal` — an image, but one whose properties
    /// no longer match the surface.
    pub suboptimal: u64,
    /// Acquires that needed a reconfigure-and-retry to succeed.
    pub retries: u64,
    /// Frames with no image, by [`SkipReason`].
    pub skipped: u64,
    /// Acquires that failed even after a reconfigure.
    pub lost: u64,
    /// `Queue::present` calls.
    pub presents: u64,
}

/// A live OS window and the swapchain configured on it.
///
/// Holds the `wgpu::Instance` the surface was created from. `wgpu`'s own
/// handles are internally reference counted so this is not strictly required
/// for soundness, but a surface only ever belongs to one instance and keeping
/// them in one value is what stops a second `device::instance()` call from
/// producing a surface and an adapter that cannot see each other.
pub struct WindowSurface {
    instance: wgpu::Instance,
    window: Arc<winit::window::Window>,
    surface: wgpu::Surface<'static>,
    configuration: wgpu::SurfaceConfiguration,
    format_choice: SurfaceFormatChoice,
    stats: SurfaceStats,
}

impl WindowSurface {
    /// Create a surface for `window` and open a device that can present to it.
    ///
    /// The order matters and is the legacy backend's: surface first, then
    /// `request_adapter` with it as `compatible_surface`. This machine
    /// enumerates five adapters, and an adapter chosen without reference to the
    /// window can be one that cannot present to it — a failure that would
    /// otherwise surface as an empty capability list at `configure` time rather
    /// than at selection time.
    pub fn new(
        window: Arc<winit::window::Window>,
    ) -> Result<(WindowSurface, ComputeContext), WindowError> {
        let instance = crate::render::device::instance();
        let surface = instance
            .create_surface(Arc::clone(&window))
            .map_err(WindowError::Surface)?;
        let context = crate::render::device::context_for(&instance, Some(&surface))?;

        let capabilities = surface.get_capabilities(&context.adapter);
        if capabilities.formats.is_empty() {
            return Err(WindowError::NoFormats);
        }
        if !capabilities.formats.contains(&TARGET_FORMAT) {
            return Err(WindowError::NoTargetFormat(capabilities.formats.to_vec()));
        }

        let size = window.inner_size();
        let configuration = wgpu::SurfaceConfiguration {
            // `COPY_SRC` is what makes Phase 6's Milestone D the strong proof
            // rather than the fallback: it is what allows the *presented image
            // itself* to be copied to a staging buffer and compared, instead of
            // an offscreen render of the same scene standing in for it. It is
            // requested unconditionally because a surface that cannot do it
            // would fail here loudly rather than silently make the proof
            // weaker; every desktop backend this crate targets reports it.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            format: TARGET_FORMAT,
            width: size.width.max(1),
            height: size.height.max(1),
            // `Immediate` presents without waiting for vsync, so a window can be
            // resized at the full rate the OS delivers WM_SIZE events instead of
            // being paced to the display refresh.
            present_mode: wgpu::PresentMode::Immediate,
            alpha_mode: capabilities
                .alpha_modes
                .first()
                .copied()
                .unwrap_or(wgpu::CompositeAlphaMode::Auto),
            color_space: wgpu::SurfaceColorSpace::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&context.device, &configuration);

        Ok((
            WindowSurface {
                instance,
                window,
                surface,
                configuration,
                format_choice: SurfaceFormatChoice::Target,
                stats: SurfaceStats {
                    configures: 1,
                    ..SurfaceStats::default()
                },
            },
            context,
        ))
    }

    /// The winit window this presents to.
    pub fn window(&self) -> &Arc<winit::window::Window> {
        &self.window
    }

    /// The instance the surface belongs to.
    pub fn instance(&self) -> &wgpu::Instance {
        &self.instance
    }

    /// The swapchain's format.
    pub fn format(&self) -> wgpu::TextureFormat {
        self.configuration.format
    }

    /// Whether the swapchain is the format the pipelines draw.
    pub fn format_choice(&self) -> SurfaceFormatChoice {
        self.format_choice
    }

    /// The configured size in physical pixels.
    pub fn size(&self) -> (u32, u32) {
        (self.configuration.width, self.configuration.height)
    }

    /// The counters this surface has accumulated.
    pub fn stats(&self) -> SurfaceStats {
        self.stats
    }

    /// Reconfigure to `width` × `height`, or report that nothing changed.
    ///
    /// Returns whether a `Surface::configure` actually happened, so a caller
    /// can assert that a resize event it dispatched reached the swapchain
    /// rather than assuming it. A zero extent is refused rather than clamped:
    /// `configure` rejects it, and a minimized window has nothing to present
    /// to — see [`resize_detector::ResizeDetector::on_resize_event`], which
    /// drops the event before it ever reaches here.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) -> bool {
        if width == 0 || height == 0 {
            return false;
        }
        if self.configuration.width == width && self.configuration.height == height {
            return false;
        }
        self.configuration.width = width;
        self.configuration.height = height;
        self.reconfigure(device);
        true
    }

    /// Apply the current configuration to the surface.
    ///
    /// The legacy backend routes every `Surface::configure` through one method
    /// and takes an exclusive lock around it, because `configure` waits for the
    /// device to go idle and fails fatally if anything submits during that wait
    /// (`src/platform/cross/renderer.rs`'s `reconfigure_surface`). 2.0 has no
    /// external render threads sharing this device yet — `SurfaceRegistry`'s
    /// producer side is not wired to a window — so there is nothing to lock
    /// out, and adding a lock with no second party would be a guess about a
    /// mechanism that does not exist here. The single-entry-point half of the
    /// legacy discipline is kept: this is the only `configure` call site.
    fn reconfigure(&mut self, device: &wgpu::Device) {
        self.surface.configure(device, &self.configuration);
        self.stats.configures += 1;
    }

    /// Acquire the next swapchain image, reconfiguring once if the surface went
    /// stale.
    ///
    /// The retry is the legacy backend's exact structure: `Outdated`, `Lost` and
    /// `Validation` all mean "the swapchain no longer matches the window", and
    /// all three are recoverable by reconfiguring and asking again. This is not
    /// belt-and-braces — a window resized between the last present and this
    /// acquire produces `Outdated` on the very next frame, so a loop without
    /// this retry drops a frame on every resize even when the resize itself was
    /// handled correctly.
    pub fn acquire(&mut self, device: &wgpu::Device) -> Acquired {
        if self.configuration.width == 0 || self.configuration.height == 0 {
            self.stats.skipped += 1;
            return Acquired::Skipped(SkipReason::ZeroSized);
        }
        match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture) => {
                self.stats.acquires += 1;
                Acquired::Frame(texture)
            }
            wgpu::CurrentSurfaceTexture::Suboptimal(texture) => {
                self.stats.acquires += 1;
                self.stats.suboptimal += 1;
                Acquired::Frame(texture)
            }
            wgpu::CurrentSurfaceTexture::Timeout => {
                self.stats.skipped += 1;
                Acquired::Skipped(SkipReason::Timeout)
            }
            wgpu::CurrentSurfaceTexture::Occluded => {
                self.stats.skipped += 1;
                Acquired::Skipped(SkipReason::Occluded)
            }
            wgpu::CurrentSurfaceTexture::Outdated
            | wgpu::CurrentSurfaceTexture::Lost
            | wgpu::CurrentSurfaceTexture::Validation => {
                self.reconfigure(device);
                self.stats.retries += 1;
                match self.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(texture) => {
                        self.stats.acquires += 1;
                        Acquired::Frame(texture)
                    }
                    wgpu::CurrentSurfaceTexture::Suboptimal(texture) => {
                        self.stats.acquires += 1;
                        self.stats.suboptimal += 1;
                        Acquired::Frame(texture)
                    }
                    wgpu::CurrentSurfaceTexture::Timeout => {
                        self.stats.skipped += 1;
                        Acquired::Skipped(SkipReason::Timeout)
                    }
                    wgpu::CurrentSurfaceTexture::Occluded => {
                        self.stats.skipped += 1;
                        Acquired::Skipped(SkipReason::Occluded)
                    }
                    _ => {
                        self.stats.lost += 1;
                        Acquired::Lost
                    }
                }
            }
        }
    }

    /// Present an acquired image.
    ///
    /// `pre_present_notify` before `Queue::present` is winit's documented
    /// contract, not decoration: on Wayland it is what lets the compositor
    /// associate the buffer attach with the frame callback, and calling it is
    /// the portable habit even where the platform ignores it.
    pub fn present(&mut self, queue: &wgpu::Queue, texture: wgpu::SurfaceTexture) {
        self.window.pre_present_notify();
        queue.present(texture);
        self.stats.presents += 1;
    }
}

/// A solid colour that could not be an accidental default.
///
/// Phase 6's Milestone A needs its clear to be unambiguous evidence: black is
/// what an uncleared attachment, a failed draw, and a device-lost swapchain all
/// look like, so black on screen proves nothing. Fully saturated magenta is
/// produced by no default path in this crate — `TARGET_FORMAT` is `Rgba8Unorm`,
/// so this reads back as exactly `[255, 0, 255, 255]` with no colour-space
/// conversion to argue about.
pub const PROOF_MAGENTA: wgpu::Color = wgpu::Color {
    r: 1.0,
    g: 0.0,
    b: 1.0,
    a: 1.0,
};

/// The bytes [`PROOF_MAGENTA`] must read back as, in `TARGET_FORMAT`.
pub const PROOF_MAGENTA_BYTES: [u8; 4] = [255, 0, 255, 255];

/// Clear an acquired swapchain image to `color` and nothing else.
///
/// Milestone A's whole frame: no scene, no pipeline, no compute. Kept as a
/// function rather than inlined into the example because the swapchain-readback
/// test drives the identical code path — a proof of "what the display scans out"
/// is only worth something if it is the same code that produced what was
/// scanned out.
pub fn clear_frame(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    view: &wgpu::TextureView,
    color: wgpu::Color,
) {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("window clear"),
    });
    {
        let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("window clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(color),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: Default::default(),
        });
    }
    queue.submit(Some(encoder.finish()));
}
