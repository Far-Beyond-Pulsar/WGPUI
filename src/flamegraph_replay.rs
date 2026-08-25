//! Replay engine (Phase 6 of the profiling epic, see issue #62): reconstructs
//! and re-drives a previously triggered frame outside the code path that
//! originally produced it, building on Phase 4's [`crate::DeepCapture`] (GPU
//! command stream + fixed-buffer resource contents) and Phase 5's
//! [`crate::UiTreeCapture`] (element tree/layout/style/Scene snapshot).
//!
//! # Scope decisions (see the progress comments on issue #62 for the full
//! reasoning)
//!
//! 1. **In-process replay only, not from-disk trace deserialization.** Phase
//!    1's trace types (`FrameCapture` and friends, `flamegraph.rs`) are
//!    `Serialize`-only -- several fields hold `&'static str`, which blocks a
//!    clean `'static`-bound `Deserialize` impl (see that module's own
//!    doc comment). Building real owned-mirror `Deserialize` support for the
//!    binary trace format is a separate, harder problem than replay itself,
//!    so this module replays directly against a live [`crate::DeepCapture`]/
//!    [`crate::UiTreeCapture`] value already sitting in the calling process
//!    (typically just taken from [`crate::take_completed_deep_capture`]/
//!    [`crate::take_completed_ui_tree_capture`]), not a file loaded from
//!    disk days later. That is also what an in-app viewer (phase 7,
//!    `WGPUI-Component`) primarily wants: scrub what was just captured.
//! 2. **In-crate module behind `feature = "flamegraph"`, not a separate
//!    crate.** The viewer lives in a different repo but already depends on
//!    `gpui` directly, so a gated module costs it nothing extra to consume,
//!    versus standing up and validating a new workspace member.
//!
//! # GPU replay fidelity
//!
//! [`render_deep_capture_step`] re-submits a captured draw call's raw,
//! fixed-buffer resource bytes to a real [`wgpu::Device`]/[`wgpu::Queue`],
//! through the *actual* production shader for that draw call's kind
//! (`platform/cross/shaders/*.wgsl`, `include_str!`'d verbatim). Every
//! buffer-backed kind -- `Quads`, `Shadows`, `Underlines`,
//! `BackdropFilters`, `Paths` -- is wired up to a real pipeline this round.
//! Each of these shaders reads its instance/vertex data directly out of a
//! raw storage buffer (`Quads`/`Shadows`/`Underlines`/`BackdropFilters`
//! indexed by `@builtin(instance_index)`, `Paths` indexed by
//! `@builtin(vertex_index)` -- see `render_paths_step`'s doc comment for why
//! that one differs), so the exact bytes Phase 4 read back from the live
//! `WgpuContext` buffer (`platform/cross/render_context.rs`) can be
//! uploaded to a fresh buffer and drawn with unchanged pipeline state -- no
//! host-side decoding of the shader's struct fields is needed to reproduce
//! the GPU's fragment output, only to know *which* range within the buffer
//! a given draw call covers (already recorded on
//! [`crate::DeepCaptureDrawCall`]). `render_quads_step`/`render_shadows_step`/
//! `render_underlines_step`/`render_paths_step` share this recipe via
//! `create_globals_bind_group`/`create_storage_bind_group`/
//! `render_pipeline_offscreen`; only the shader module, entry points,
//! primitive topology, and bind group visibility actually differ per kind,
//! matched against `WgpuPipelines::new` in `renderer.rs` rather than
//! guessed. `render_backdrop_filters_step` follows the same recipe for its
//! own `@group(1)` instance buffer, but also needs a `@group(2)` texture +
//! sampler (the content being blurred) that Phase 4 never captures at all --
//! see that function's doc comment for how it degrades just that one input.
//!
//! `MonoSprites`/`PolySprites` (atlas-textured) now replay with real atlas
//! texture content too, once Phase 4b (issue #72) landed atlas/surface
//! texture readback (`DeepCapture::texture_contents`,
//! `DeepCaptureTextureContents`): `render_mono_sprites_step`/
//! `render_poly_sprites_step` follow the same buffer-backed recipe for their
//! own instance buffer (`@group(3)`/`@group(2)` respectively), plus a real
//! `@group(2)`/`@group(1)` atlas texture + sampler built from the captured
//! bytes (`create_atlas_texture_bind_group`) instead of a placeholder.
//! `MonoSprites` also needs a `ColorAdjustments` uniform Phase 4 never
//! captures; `create_color_adjustments_bind_group`'s doc comment explains
//! why an all-zero default is a safe, documented simplification there.
//!
//! `Surfaces` still degrades to a generated checkerboard placeholder
//! ([`placeholder_checkerboard_rgba`]) via [`render_deep_capture_step`]
//! even though its texture *content* is now capturable -- [`DeepCaptureDrawCall`]
//! has no per-call geometry (a `Surfaces` draw call's `SurfaceParams`:
//! `bounds`/`content_mask`) to position/mask a replayed quad with, and that
//! is a distinct gap from texture-content readback itself, out of scope for
//! issue #72 as filed. [`DeepCaptureReplay::resource_status`] reports
//! [`DrawCallResourceStatus::TextureContentUnavailable`] for every
//! `Surfaces` draw call unconditionally for this same reason, regardless of
//! whether that specific surface's bytes were actually captured -- "nothing
//! to replay faithfully" remains true either way until that follow-up
//! geometry capture exists.

use crate::flamegraph::{
    DeepCapture, DeepCaptureBufferKind, DeepCaptureDrawCall, DeepCaptureTextureContents, DeepCaptureTextureId,
    DrawCallKind,
};
use crate::flamegraph_ui_capture::{SceneSnapshot, UiElementNode, UiTreeCapture};

// ---------------------------------------------------------------------------
// CPU/UI-tree replay.

/// Reconstructs parent/child relationships from a [`UiTreeCapture`]'s flat,
/// depth-annotated DFS node list (`UiElementNode::depth`), so a caller can
/// walk the tree shape without re-deriving it from scratch every time. Pure
/// data transform, no live app state and no GPU involved -- can be built,
/// queried, and torn down entirely independent of the app that produced the
/// capture, and independent of GPU replay's device requirements.
#[derive(Debug, Clone)]
pub struct UiTreeReplay {
    capture: UiTreeCapture,
    parents: Vec<Option<usize>>,
    children: Vec<Vec<usize>>,
    roots: Vec<usize>,
}

impl UiTreeReplay {
    /// Build a replay view over `capture`. `O(n)` in the number of captured
    /// nodes: one pass with a depth-ordered stack, the same "keep popping
    /// while the top of the stack is not shallower than the current node"
    /// trick used to reconstruct a tree from a preorder depth list.
    pub fn new(capture: UiTreeCapture) -> Self {
        let node_count = capture.nodes.len();
        let mut parents: Vec<Option<usize>> = vec![None; node_count];
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); node_count];
        let mut roots: Vec<usize> = Vec::new();
        let mut stack: Vec<usize> = Vec::new();

        for (index, node) in capture.nodes.iter().enumerate() {
            while let Some(&top) = stack.last() {
                if capture.nodes[top].depth < node.depth {
                    break;
                }
                stack.pop();
            }
            match stack.last() {
                Some(&parent_index) => {
                    parents[index] = Some(parent_index);
                    children[parent_index].push(index);
                }
                None => roots.push(index),
            }
            stack.push(index);
        }

        Self {
            capture,
            parents,
            children,
            roots,
        }
    }

    /// The underlying capture this replay was built from.
    pub fn capture(&self) -> &UiTreeCapture {
        &self.capture
    }

    /// Number of nodes in the captured tree.
    pub fn node_count(&self) -> usize {
        self.capture.nodes.len()
    }

    /// The node at `index`, if `index` is in range.
    pub fn node(&self, index: usize) -> Option<&UiElementNode> {
        self.capture.nodes.get(index)
    }

    /// `index`'s parent, if it has one (root nodes return `None`).
    pub fn parent(&self, index: usize) -> Option<usize> {
        self.parents.get(index).copied().flatten()
    }

    /// `index`'s direct children, in the original DFS (paint) order.
    pub fn children(&self, index: usize) -> &[usize] {
        self.children.get(index).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Indices of every node with no captured parent (usually just one, the
    /// window's root element, but a capture that started mid-tree for any
    /// reason could have more).
    pub fn roots(&self) -> &[usize] {
        &self.roots
    }

    /// Every node index, in the same DFS order [`UiTreeCapture::nodes`]
    /// already stores them in -- a named accessor so callers don't need to
    /// know that detail of the underlying representation.
    pub fn depth_first_indices(&self) -> impl Iterator<Item = usize> + '_ {
        0..self.capture.nodes.len()
    }

    /// The frame's captured paint primitive list.
    pub fn scene(&self) -> &SceneSnapshot {
        &self.capture.scene
    }
}

// ---------------------------------------------------------------------------
// GPU replay: stepping over a `DeepCapture`'s command stream.

/// Whether the resource(s) a given draw call in a [`DeepCaptureReplay`]
/// depends on are actually available to replay against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrawCallResourceStatus {
    /// Every resource this draw call needs was touched by the capture and
    /// its readback completed -- [`render_deep_capture_step`] can use real
    /// captured bytes for it. For `MonoSprites`/`PolySprites` this means
    /// both the instance buffer *and* the atlas texture it references;
    /// `Surfaces` can never report this (see [`Self::TextureContentUnavailable`]).
    Available,
    /// This draw call references a fixed buffer
    /// ([`DeepCaptureDrawCall::buffer_kind`]) that the capture touched but
    /// whose readback did not complete (`DeepCapture::resources_finalized`
    /// was false, or this specific buffer's map failed).
    BufferReadbackMissing,
    /// Either this is a `Surfaces` draw call (which always reports this --
    /// see this module's doc comment for why even a captured surface
    /// texture isn't enough to replay one faithfully), or it's a
    /// `MonoSprites`/`PolySprites` call whose referenced atlas texture
    /// wasn't captured (no `atlas_texture_id`, or that texture's readback
    /// didn't complete). Either way there is nothing to replay faithfully;
    /// [`render_deep_capture_step`] substitutes a placeholder.
    TextureContentUnavailable,
    /// This draw call has no associated fixed-buffer resource at all (should
    /// not occur for a well-formed capture, but reported rather than
    /// panicking if the step index is out of range or otherwise
    /// unrecognized).
    NoResource,
}

/// Stepping cursor over one [`DeepCapture`]'s command stream --
/// draw-call-level granularity within a replayed frame, mirroring
/// RenderDoc's "step to next draw call" model. Pure bookkeeping over
/// already-captured data; does not itself touch a GPU device (see
/// [`render_deep_capture_step`] for the part that does).
#[derive(Debug, Clone)]
pub struct DeepCaptureReplay {
    capture: DeepCapture,
    cursor: usize,
}

impl DeepCaptureReplay {
    /// Build a replay session positioned at the first draw call (index `0`),
    /// mirroring RenderDoc opening a capture already stopped on a draw call
    /// rather than before/after the whole stream. Call
    /// [`Self::step_to_next_draw_call`]/[`Self::step_to_previous_draw_call`]/
    /// [`Self::seek`] to move the cursor.
    pub fn new(capture: DeepCapture) -> Self {
        Self { capture, cursor: 0 }
    }

    /// The underlying capture this replay was built from.
    pub fn capture(&self) -> &DeepCapture {
        &self.capture
    }

    /// Number of draw calls in the captured command stream.
    pub fn draw_call_count(&self) -> usize {
        self.capture.draw_calls.len()
    }

    /// The cursor's current position -- the index [`Self::current_draw_call`]
    /// reads from. Always `0` when the capture has no draw calls.
    pub fn current_step(&self) -> usize {
        self.cursor
    }

    /// The draw call at the cursor's current position, if any (`None` only
    /// when the capture has no draw calls at all).
    pub fn current_draw_call(&self) -> Option<&DeepCaptureDrawCall> {
        self.capture.draw_calls.get(self.cursor)
    }

    /// Move the cursor back to the first draw call.
    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    /// Move the cursor to the next draw call and return it. Returns `None`
    /// without moving the cursor when already at the last draw call (or when
    /// the capture is empty) -- the cursor never runs off the end, so it is
    /// always safe to read [`Self::current_draw_call`] afterward.
    pub fn step_to_next_draw_call(&mut self) -> Option<&DeepCaptureDrawCall> {
        if self.cursor + 1 >= self.capture.draw_calls.len() {
            return None;
        }
        self.cursor += 1;
        self.capture.draw_calls.get(self.cursor)
    }

    /// Move the cursor to the previous draw call and return it. Returns
    /// `None` without moving the cursor when already at the first draw call
    /// (or when the capture is empty).
    pub fn step_to_previous_draw_call(&mut self) -> Option<&DeepCaptureDrawCall> {
        if self.cursor == 0 {
            return None;
        }
        self.cursor -= 1;
        self.capture.draw_calls.get(self.cursor)
    }

    /// Move the cursor directly to `step` and return the draw call there.
    /// Does nothing and returns `None` if `step` is out of range, leaving
    /// the cursor at its previous position.
    pub fn seek(&mut self, step: usize) -> Option<&DeepCaptureDrawCall> {
        if step >= self.capture.draw_calls.len() {
            return None;
        }
        self.cursor = step;
        self.capture.draw_calls.get(step)
    }

    /// Resource availability for the draw call at `step`, independent of the
    /// cursor's current position -- see [`DrawCallResourceStatus`].
    pub fn resource_status(&self, step: usize) -> DrawCallResourceStatus {
        let Some(call) = self.capture.draw_calls.get(step) else {
            return DrawCallResourceStatus::NoResource;
        };

        match call.kind {
            DrawCallKind::MonoSprites | DrawCallKind::PolySprites => {
                let texture_available = call
                    .atlas_texture_id
                    .is_some_and(|id| self.capture.texture_contents(DeepCaptureTextureId::Atlas(id)).is_some());
                if !texture_available {
                    return DrawCallResourceStatus::TextureContentUnavailable;
                }
                match call.buffer_kind {
                    Some(kind) => {
                        if self.capture.buffer_contents(kind).is_some() {
                            DrawCallResourceStatus::Available
                        } else {
                            DrawCallResourceStatus::BufferReadbackMissing
                        }
                    }
                    None => DrawCallResourceStatus::NoResource,
                }
            }
            // Always unavailable regardless of whether this specific
            // surface's texture bytes were captured -- see this module's
            // doc comment for why texture content alone isn't enough to
            // replay a `Surfaces` call faithfully.
            DrawCallKind::Surfaces => DrawCallResourceStatus::TextureContentUnavailable,
            _ => match call.buffer_kind {
                Some(kind) => {
                    if self.capture.buffer_contents(kind).is_some() {
                        DrawCallResourceStatus::Available
                    } else {
                        DrawCallResourceStatus::BufferReadbackMissing
                    }
                }
                None => DrawCallResourceStatus::NoResource,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// GPU replay: real re-submission against a `wgpu::Device`/`wgpu::Queue`.

/// Errors from [`render_deep_capture_step`].
#[derive(Debug, thiserror::Error)]
pub enum ReplayError {
    /// `step` is not a valid index into the capture's command stream.
    #[error("draw call step {0} is out of range")]
    StepOutOfRange(usize),
    /// This draw call's `kind` has no wired-up replay pipeline this round.
    /// See this module's doc comment for which kinds are covered.
    #[error("no replay pipeline is wired up for draw call kind {0:?} yet")]
    UnsupportedDrawCallKind(DrawCallKind),
    /// The draw call references a fixed buffer whose readback never
    /// completed (see [`DrawCallResourceStatus::BufferReadbackMissing`]).
    #[error("buffer contents for {0:?} were not available in this capture")]
    MissingBufferContents(DeepCaptureBufferKind),
    /// `viewport_width`/`viewport_height` passed to
    /// [`render_deep_capture_step`] was zero.
    #[error("replay viewport must be non-zero in both dimensions")]
    EmptyViewport,
    /// A `wgpu` device-side operation failed (buffer map, adapter request,
    /// etc.). Carries a message rather than the original error type, since
    /// `wgpu`'s own error types are not uniformly `Clone`/`'static`-friendly
    /// across the operations this module performs.
    #[error("GPU replay failed: {0}")]
    Device(String),
}

/// The result of replaying one draw call's GPU work: the rendered contents
/// of an offscreen `viewport_width` x `viewport_height` target, as tightly
/// packed (no row padding) 8-bit RGBA.
#[derive(Debug, Clone)]
pub struct ReplayRenderOutput {
    /// Rendered target width, in pixels.
    pub width: u32,
    /// Rendered target height, in pixels.
    pub height: u32,
    /// Tightly packed 8-bit RGBA pixel data, `width * height * 4` bytes,
    /// row-major from the top-left.
    pub rgba8: Vec<u8>,
    /// `true` when this output is a generated placeholder (see
    /// [`placeholder_checkerboard_rgba`]) because the draw call's real
    /// texture content is unavailable
    /// ([`DrawCallResourceStatus::TextureContentUnavailable`]), rather than
    /// an actual GPU render of captured resource bytes.
    pub texture_unavailable: bool,
}

impl ReplayRenderOutput {
    /// The RGBA8 pixel at `(x, y)`, if in bounds.
    pub fn pixel(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let offset = ((y * self.width + x) * 4) as usize;
        let slice = self.rgba8.get(offset..offset + 4)?;
        Some([slice[0], slice[1], slice[2], slice[3]])
    }
}

/// Generate a placeholder checkerboard image, used by
/// [`render_deep_capture_step`] in place of real pixels for draw calls whose
/// texture content is unavailable (see this module's doc comment). `cell` is
/// the checkerboard square size in pixels (clamped to at least `1`).
pub fn placeholder_checkerboard_rgba(width: u32, height: u32, cell: u32) -> Vec<u8> {
    let cell = cell.max(1);
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    const LIGHT: [u8; 4] = [200, 200, 200, 255];
    const DARK: [u8; 4] = [120, 120, 120, 255];
    for y in 0..height {
        for x in 0..width {
            let checker = ((x / cell) + (y / cell)).is_multiple_of(2);
            pixels.extend_from_slice(if checker { &LIGHT } else { &DARK });
        }
    }
    pixels
}

/// Matches `quads.wgsl`'s `Globals` uniform layout exactly (`vec2<f32>` +
/// two `u32`s, 16 bytes, no implicit padding).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ReplayGlobals {
    viewport_size: [f32; 2],
    premultiplied_alpha: u32,
    pad: u32,
}

const REPLAY_TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Re-submit the draw call at `step` in `replay`'s command stream against a
/// real `device`/`queue`, reproducing its GPU output outside the live app
/// that originally recorded it. Renders into a fresh, offscreen
/// `viewport_width` x `viewport_height` target and reads the result back to
/// host memory -- this is a diagnostic/replay operation, not a hot-path one,
/// so it blocks on the GPU (`PollType::wait_indefinitely`) rather than
/// integrating with a caller's own per-frame poll loop, and builds its
/// pipeline fresh on every call rather than caching it across steps.
///
/// See this module's doc comment for exactly which [`DrawCallKind`]s have a
/// real pipeline wired up this round (every buffer-backed kind -- `Quads`,
/// `Shadows`, `Underlines`, `BackdropFilters`, `Paths` -- plus
/// `MonoSprites`/`PolySprites` when their referenced atlas texture was
/// captured) versus which degrade to a placeholder (`Surfaces` always;
/// `MonoSprites`/`PolySprites` when their atlas texture wasn't captured),
/// via [`placeholder_checkerboard_rgba`].
pub fn render_deep_capture_step(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    replay: &DeepCaptureReplay,
    step: usize,
    viewport_width: u32,
    viewport_height: u32,
) -> Result<ReplayRenderOutput, ReplayError> {
    if viewport_width == 0 || viewport_height == 0 {
        return Err(ReplayError::EmptyViewport);
    }
    let call = replay
        .capture()
        .draw_calls
        .get(step)
        .ok_or(ReplayError::StepOutOfRange(step))?;

    match call.kind {
        DrawCallKind::MonoSprites | DrawCallKind::PolySprites => {
            let texture_contents = call
                .atlas_texture_id
                .and_then(|id| replay.capture().texture_contents(DeepCaptureTextureId::Atlas(id)));
            let Some(texture_contents) = texture_contents else {
                return Ok(placeholder_output(viewport_width, viewport_height));
            };
            let buffer_kind = call.buffer_kind.ok_or(ReplayError::UnsupportedDrawCallKind(call.kind))?;
            let instance_contents = replay
                .capture()
                .buffer_contents(buffer_kind)
                .ok_or(ReplayError::MissingBufferContents(buffer_kind))?;
            match call.kind {
                DrawCallKind::MonoSprites => render_mono_sprites_step(
                    device,
                    queue,
                    &instance_contents.bytes,
                    texture_contents,
                    call,
                    viewport_width,
                    viewport_height,
                ),
                DrawCallKind::PolySprites => render_poly_sprites_step(
                    device,
                    queue,
                    &instance_contents.bytes,
                    texture_contents,
                    call,
                    viewport_width,
                    viewport_height,
                ),
                // Unreachable in practice (this arm only matches
                // MonoSprites/PolySprites), kept as a real error rather than
                // `unreachable!()` for the same reason every other
                // "shouldn't happen" arm in this module is.
                _ => Err(ReplayError::UnsupportedDrawCallKind(call.kind)),
            }
        }
        DrawCallKind::Surfaces => {
            // No per-draw-call geometry (`SurfaceParams`: `bounds`/
            // `content_mask`) is captured anywhere in `DeepCapture` -- issue
            // #72 completed texture *content* readback, but faithfully
            // positioning and clip-masking a `Surfaces` quad needs that
            // geometry too, a distinct, still-open gap (see this module's
            // doc comment). Even when this call's surface texture bytes are
            // available, there is nothing to draw them *at* yet, so this
            // still degrades to the placeholder unconditionally.
            Ok(placeholder_output(viewport_width, viewport_height))
        }
        DrawCallKind::Quads
        | DrawCallKind::Shadows
        | DrawCallKind::Underlines
        | DrawCallKind::BackdropFilters
        | DrawCallKind::Paths => {
            let buffer_kind = call.buffer_kind.ok_or(ReplayError::UnsupportedDrawCallKind(call.kind))?;
            let contents = replay
                .capture()
                .buffer_contents(buffer_kind)
                .ok_or(ReplayError::MissingBufferContents(buffer_kind))?;

            match call.kind {
                DrawCallKind::Quads => {
                    render_quads_step(device, queue, &contents.bytes, call, viewport_width, viewport_height)
                }
                DrawCallKind::Shadows => {
                    render_shadows_step(device, queue, &contents.bytes, call, viewport_width, viewport_height)
                }
                DrawCallKind::Underlines => {
                    render_underlines_step(device, queue, &contents.bytes, call, viewport_width, viewport_height)
                }
                DrawCallKind::BackdropFilters => {
                    render_backdrop_filters_step(device, queue, &contents.bytes, call, viewport_width, viewport_height)
                }
                DrawCallKind::Paths => {
                    render_paths_step(device, queue, &contents.bytes, call, viewport_width, viewport_height)
                }
                // Unreachable in practice (this arm only matches the five
                // buffer-backed kinds listed above it), same non-panicking
                // rationale as every other such arm in this module.
                _ => Err(ReplayError::UnsupportedDrawCallKind(call.kind)),
            }
        }
    }
}

/// Builds the generated checkerboard placeholder [`ReplayRenderOutput`]
/// [`render_deep_capture_step`] substitutes whenever a draw call's real
/// texture content isn't available (or, for `Surfaces`, never can be
/// faithfully positioned regardless -- see this module's doc comment).
fn placeholder_output(viewport_width: u32, viewport_height: u32) -> ReplayRenderOutput {
    ReplayRenderOutput {
        width: viewport_width,
        height: viewport_height,
        rgba8: placeholder_checkerboard_rgba(viewport_width, viewport_height, 16),
        texture_unavailable: true,
    }
}

/// Builds the `@group(0)` globals uniform bind group every production
/// shader this module replays shares verbatim (`quads.wgsl`, `shadows.wgsl`,
/// `underlines.wgsl`, `backdrop_blur.wgsl`, and `paths.wgsl` all declare the
/// identical `Globals` struct at `@group(0) @binding(0)`), so it is built
/// once here rather than duplicated in every `render_*_step`.
fn create_globals_bind_group(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    viewport_width: u32,
    viewport_height: u32,
) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
    let globals_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("flamegraph_replay_globals_bind_group_layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let globals = ReplayGlobals {
        viewport_size: [viewport_width as f32, viewport_height as f32],
        premultiplied_alpha: 0,
        pad: 0,
    };
    let globals_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flamegraph_replay_globals_buffer"),
        size: core::mem::size_of::<ReplayGlobals>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&globals_buffer, 0, bytemuck::bytes_of(&globals));

    let globals_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("flamegraph_replay_globals_bind_group"),
        layout: &globals_bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: globals_buffer.as_entire_binding(),
        }],
    });

    (globals_bind_group_layout, globals_bind_group)
}

/// Builds a single-binding read-only storage buffer bind group from raw
/// captured bytes -- the "same recipe" every buffer-backed [`DrawCallKind`]
/// follows: the captured bytes are the exact ones the live `WgpuContext`
/// buffer held, reinterpreted by the shader's own storage-buffer struct
/// layout, so no host-side decoding is needed here, only re-upload.
/// `visibility` must match the real pipeline's bind group layout for this
/// kind (`WgpuPipelines::new`, `renderer.rs`) -- most kinds read their
/// storage buffer from both stages (`VERTEX_FRAGMENT`), but `Paths` only
/// reads it from the vertex stage.
fn create_storage_bind_group(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    visibility: wgpu::ShaderStages,
    bytes: &[u8],
) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(&format!("{label}_bind_group_layout")),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let buffer_size = bytes.len().max(1) as u64;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("{label}_buffer")),
        size: buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !bytes.is_empty() {
        queue.write_buffer(&buffer, 0, bytes);
    }

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("{label}_bind_group")),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    (bind_group_layout, bind_group)
}

/// Runs one draw call's already-built `pipeline`/`bind_groups` against a
/// fresh, offscreen `viewport_width` x `viewport_height` target and reads
/// the result back to host memory as tightly packed (no row padding) 8-bit
/// RGBA. Shared tail end of every `render_*_step` helper below -- only the
/// shader/pipeline/bind-group setup differs per [`DrawCallKind`] (handled by
/// each caller); the target texture, render pass, and readback mechanics
/// (including `wgpu::COPY_BYTES_PER_ROW_ALIGNMENT` row-padding, which this
/// crate's own fixed-buffer readback in `flamegraph_gpu.rs` never has to
/// deal with since that path copies linear buffer bytes, not a texture) are
/// identical regardless of which primitive kind produced them.
fn render_pipeline_offscreen(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::RenderPipeline,
    bind_groups: &[&wgpu::BindGroup],
    vertex_range: (u32, u32),
    instance_range: (u32, u32),
    viewport_width: u32,
    viewport_height: u32,
) -> Result<ReplayRenderOutput, ReplayError> {
    let target_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("flamegraph_replay_target_texture"),
        size: wgpu::Extent3d {
            width: viewport_width,
            height: viewport_height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: REPLAY_TARGET_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flamegraph_replay_encoder"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("flamegraph_replay_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        for (index, bind_group) in bind_groups.iter().enumerate() {
            pass.set_bind_group(index as u32, *bind_group, &[]);
        }
        pass.draw(vertex_range.0..vertex_range.1, instance_range.0..instance_range.1);
    }

    let bytes_per_pixel = 4u32;
    let unpadded_bytes_per_row = viewport_width * bytes_per_pixel;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
    let staging_size = (padded_bytes_per_row as u64) * (viewport_height as u64);
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flamegraph_replay_staging_buffer"),
        size: staging_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(viewport_height),
            },
        },
        wgpu::Extent3d {
            width: viewport_width,
            height: viewport_height,
            depth_or_array_layers: 1,
        },
    );

    queue.submit(Some(encoder.finish()));

    let map_ready = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let map_ok = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    {
        let map_ready = map_ready.clone();
        let map_ok = map_ok.clone();
        staging_buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            map_ok.store(result.is_ok(), std::sync::atomic::Ordering::Release);
            map_ready.store(true, std::sync::atomic::Ordering::Release);
        });
    }

    // Blocking wait: this is a diagnostic/replay call, not a per-frame hot
    // path -- see `render_deep_capture_step`'s doc comment.
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|error| ReplayError::Device(error.to_string()))?;
    if !map_ready.load(std::sync::atomic::Ordering::Acquire) || !map_ok.load(std::sync::atomic::Ordering::Acquire) {
        return Err(ReplayError::Device("staging buffer map_async did not complete successfully".to_string()));
    }

    let rgba8 = {
        let slice = staging_buffer.slice(..);
        let view = slice
            .get_mapped_range()
            .map_err(|error| ReplayError::Device(error.to_string()))?;
        let mut packed = Vec::with_capacity((unpadded_bytes_per_row as usize) * (viewport_height as usize));
        for row in 0..viewport_height as usize {
            let start = row * padded_bytes_per_row as usize;
            let end = start + unpadded_bytes_per_row as usize;
            packed.extend_from_slice(&view[start..end]);
        }
        packed
    };
    staging_buffer.unmap();

    Ok(ReplayRenderOutput {
        width: viewport_width,
        height: viewport_height,
        rgba8,
        texture_unavailable: false,
    })
}

fn render_quads_step(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    quads_bytes: &[u8],
    call: &DeepCaptureDrawCall,
    viewport_width: u32,
    viewport_height: u32,
) -> Result<ReplayRenderOutput, ReplayError> {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("flamegraph_replay_quads_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("platform/cross/shaders/quads.wgsl").into()),
    });

    let (globals_bind_group_layout, globals_bind_group) =
        create_globals_bind_group(device, queue, viewport_width, viewport_height);
    let (quads_bind_group_layout, quads_bind_group) = create_storage_bind_group(
        device,
        queue,
        "flamegraph_replay_quads",
        wgpu::ShaderStages::VERTEX_FRAGMENT,
        quads_bytes,
    );

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("flamegraph_replay_quads_pipeline_layout"),
        bind_group_layouts: &[Some(&globals_bind_group_layout), Some(&quads_bind_group_layout)],
        immediate_size: 0,
    });

    let color_targets = [Some(wgpu::ColorTargetState {
        format: REPLAY_TARGET_FORMAT,
        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
        write_mask: wgpu::ColorWrites::ALL,
    })];

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("flamegraph_replay_quads_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_quad"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_quad"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &color_targets,
        }),
        multiview_mask: None,
        cache: None,
    });

    render_pipeline_offscreen(
        device,
        queue,
        &pipeline,
        &[&globals_bind_group, &quads_bind_group],
        call.vertex_range,
        call.instance_range,
        viewport_width,
        viewport_height,
    )
}

/// Mirrors [`render_quads_step`] exactly, against the real production
/// `shadows.wgsl` shader (`platform/cross/shaders/shadows.wgsl`) and its own
/// bind group layout (`shadows_bind_group_layout` in `renderer.rs`: a single
/// `VERTEX_FRAGMENT`-visible read-only storage buffer at `@group(1)
/// @binding(0)`, same shape as `Quads`'s).
fn render_shadows_step(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    shadows_bytes: &[u8],
    call: &DeepCaptureDrawCall,
    viewport_width: u32,
    viewport_height: u32,
) -> Result<ReplayRenderOutput, ReplayError> {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("flamegraph_replay_shadows_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("platform/cross/shaders/shadows.wgsl").into()),
    });

    let (globals_bind_group_layout, globals_bind_group) =
        create_globals_bind_group(device, queue, viewport_width, viewport_height);
    let (shadows_bind_group_layout, shadows_bind_group) = create_storage_bind_group(
        device,
        queue,
        "flamegraph_replay_shadows",
        wgpu::ShaderStages::VERTEX_FRAGMENT,
        shadows_bytes,
    );

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("flamegraph_replay_shadows_pipeline_layout"),
        bind_group_layouts: &[Some(&globals_bind_group_layout), Some(&shadows_bind_group_layout)],
        immediate_size: 0,
    });

    let color_targets = [Some(wgpu::ColorTargetState {
        format: REPLAY_TARGET_FORMAT,
        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
        write_mask: wgpu::ColorWrites::ALL,
    })];

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("flamegraph_replay_shadows_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_shadow"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_shadow"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &color_targets,
        }),
        multiview_mask: None,
        cache: None,
    });

    render_pipeline_offscreen(
        device,
        queue,
        &pipeline,
        &[&globals_bind_group, &shadows_bind_group],
        call.vertex_range,
        call.instance_range,
        viewport_width,
        viewport_height,
    )
}

/// Mirrors [`render_quads_step`] exactly, against the real production
/// `underlines.wgsl` shader and its own bind group layout
/// (`underlines_bind_group_layout` in `renderer.rs`: a single
/// `VERTEX_FRAGMENT`-visible read-only storage buffer at `@group(1)
/// @binding(0)`, same shape as `Quads`'s/`Shadows`'s).
fn render_underlines_step(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    underlines_bytes: &[u8],
    call: &DeepCaptureDrawCall,
    viewport_width: u32,
    viewport_height: u32,
) -> Result<ReplayRenderOutput, ReplayError> {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("flamegraph_replay_underlines_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("platform/cross/shaders/underlines.wgsl").into()),
    });

    let (globals_bind_group_layout, globals_bind_group) =
        create_globals_bind_group(device, queue, viewport_width, viewport_height);
    let (underlines_bind_group_layout, underlines_bind_group) = create_storage_bind_group(
        device,
        queue,
        "flamegraph_replay_underlines",
        wgpu::ShaderStages::VERTEX_FRAGMENT,
        underlines_bytes,
    );

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("flamegraph_replay_underlines_pipeline_layout"),
        bind_group_layouts: &[Some(&globals_bind_group_layout), Some(&underlines_bind_group_layout)],
        immediate_size: 0,
    });

    let color_targets = [Some(wgpu::ColorTargetState {
        format: REPLAY_TARGET_FORMAT,
        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
        write_mask: wgpu::ColorWrites::ALL,
    })];

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("flamegraph_replay_underlines_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_underline"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_underline"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &color_targets,
        }),
        multiview_mask: None,
        cache: None,
    });

    render_pipeline_offscreen(
        device,
        queue,
        &pipeline,
        &[&globals_bind_group, &underlines_bind_group],
        call.vertex_range,
        call.instance_range,
        viewport_width,
        viewport_height,
    )
}

/// Builds a placeholder `@group(2)` texture + sampler bind group for
/// `render_backdrop_filters_step` -- see that function's doc comment for
/// why a real one is not available.
fn create_placeholder_backdrop_texture_bind_group(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
    let width = width.max(1);
    let height = height.max(1);

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("flamegraph_replay_backdrop_texture_bind_group_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("flamegraph_replay_backdrop_placeholder_texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: REPLAY_TARGET_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let placeholder_bytes = placeholder_checkerboard_rgba(width, height, 16);
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &placeholder_bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 4),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("flamegraph_replay_backdrop_placeholder_sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("flamegraph_replay_backdrop_texture_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });

    (bind_group_layout, bind_group)
}

/// Mirrors [`render_quads_step`]'s recipe for `backdrop_blur.wgsl`'s own
/// `@group(1)` instance buffer (`backdrop_filters_bind_group_layout` in
/// `renderer.rs`: a single `VERTEX_FRAGMENT`-visible read-only storage
/// buffer, same shape as `Quads`'s/`Shadows`'s/`Underlines`'s) -- but that
/// shader also samples a `@group(2)` texture + sampler
/// (`backdrop_texture_bind_group_layout`), the *already-rendered content
/// behind the element* being blurred (either a full backdrop snapshot for
/// CSS `backdrop-filter`, or an offscreen group texture for a content
/// `filter` group). Phase 4's [`crate::DeepCapture`] never reads that back
/// at all -- unlike the atlas/surface texture readback issue #72 tracks,
/// there is no single `SurfaceId`/atlas tile to key it by, since it is
/// whatever happened to be painted behind this element, a strictly harder
/// capture problem than either of #72's two cases. So this replays the
/// *real* captured instance data (bounds, blur radius, corner radii,
/// opacity) through the *real* production shader and pipeline, faithfully
/// reproducing geometry/masking/opacity, but samples a generated
/// placeholder image ([`placeholder_checkerboard_rgba`]) where the live
/// backdrop content would have been -- the blurred picture itself is not
/// the original frame's pixels.
fn render_backdrop_filters_step(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    backdrop_filters_bytes: &[u8],
    call: &DeepCaptureDrawCall,
    viewport_width: u32,
    viewport_height: u32,
) -> Result<ReplayRenderOutput, ReplayError> {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("flamegraph_replay_backdrop_filters_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("platform/cross/shaders/backdrop_blur.wgsl").into()),
    });

    let (globals_bind_group_layout, globals_bind_group) =
        create_globals_bind_group(device, queue, viewport_width, viewport_height);
    let (backdrop_filters_bind_group_layout, backdrop_filters_bind_group) = create_storage_bind_group(
        device,
        queue,
        "flamegraph_replay_backdrop_filters",
        wgpu::ShaderStages::VERTEX_FRAGMENT,
        backdrop_filters_bytes,
    );
    let (backdrop_texture_bind_group_layout, backdrop_texture_bind_group) =
        create_placeholder_backdrop_texture_bind_group(device, queue, viewport_width, viewport_height);

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("flamegraph_replay_backdrop_filters_pipeline_layout"),
        bind_group_layouts: &[
            Some(&globals_bind_group_layout),
            Some(&backdrop_filters_bind_group_layout),
            Some(&backdrop_texture_bind_group_layout),
        ],
        immediate_size: 0,
    });

    let color_targets = [Some(wgpu::ColorTargetState {
        format: REPLAY_TARGET_FORMAT,
        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
        write_mask: wgpu::ColorWrites::ALL,
    })];

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("flamegraph_replay_backdrop_filters_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_backdrop_filter"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_backdrop_filter"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &color_targets,
        }),
        multiview_mask: None,
        cache: None,
    });

    render_pipeline_offscreen(
        device,
        queue,
        &pipeline,
        &[&globals_bind_group, &backdrop_filters_bind_group, &backdrop_texture_bind_group],
        call.vertex_range,
        call.instance_range,
        viewport_width,
        viewport_height,
    )
}

/// Mirrors [`render_quads_step`]'s recipe for `paths.wgsl`, with two real
/// differences from every other buffer-backed kind, both confirmed against
/// `renderer.rs` rather than assumed:
///
/// 1. **Different vertex layout.** `paths.wgsl`'s `b_path_vertices` holds
///    flat `GpuPathVertex` records (position, ST curve coordinates, color,
///    content-mask bounds -- see `renderer.rs`'s `GpuPathVertex` and
///    `render_context.rs`'s `paths_vertices_buffer`), not `Quad`'s
///    bounds/background/corner-radii shape. This function does not need to
///    know that layout itself, though -- exactly like every other kind here,
///    the captured bytes are re-uploaded verbatim and reinterpreted by the
///    shader, so only the *test* building a synthetic capture (see this
///    module's test module) needs to construct bytes matching it.
/// 2. **Indexed by `@builtin(vertex_index)`, not `@builtin(instance_index)`.**
///    `paths.wgsl` has no per-instance unit-quad expansion -- each vertex in
///    `call.vertex_range` is one already-tessellated triangle vertex, drawn
///    with `wgpu::PrimitiveTopology::TriangleList` (`paths_pipeline` in
///    `renderer.rs`, not `TriangleStrip` like the other kinds) and a fixed
///    `instance_range` of `0..1`. `paths_bind_group_layout` in `renderer.rs`
///    is also `ShaderStages::VERTEX` only (not `VERTEX_FRAGMENT`), since
///    `fs_path` reads varyings, not the storage buffer directly.
fn render_paths_step(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    path_vertices_bytes: &[u8],
    call: &DeepCaptureDrawCall,
    viewport_width: u32,
    viewport_height: u32,
) -> Result<ReplayRenderOutput, ReplayError> {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("flamegraph_replay_paths_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("platform/cross/shaders/paths.wgsl").into()),
    });

    let (globals_bind_group_layout, globals_bind_group) =
        create_globals_bind_group(device, queue, viewport_width, viewport_height);
    let (paths_bind_group_layout, paths_bind_group) = create_storage_bind_group(
        device,
        queue,
        "flamegraph_replay_paths",
        wgpu::ShaderStages::VERTEX,
        path_vertices_bytes,
    );

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("flamegraph_replay_paths_pipeline_layout"),
        bind_group_layouts: &[Some(&globals_bind_group_layout), Some(&paths_bind_group_layout)],
        immediate_size: 0,
    });

    let color_targets = [Some(wgpu::ColorTargetState {
        format: REPLAY_TARGET_FORMAT,
        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
        write_mask: wgpu::ColorWrites::ALL,
    })];

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("flamegraph_replay_paths_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_path"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_path"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &color_targets,
        }),
        multiview_mask: None,
        cache: None,
    });

    render_pipeline_offscreen(
        device,
        queue,
        &pipeline,
        &[&globals_bind_group, &paths_bind_group],
        call.vertex_range,
        call.instance_range,
        viewport_width,
        viewport_height,
    )
}

/// Matches `renderer.rs`'s private `ColorAdjustments` struct layout exactly
/// (`vec4<f32>` gamma ratios + `f32` grayscale contrast + 12 bytes of
/// padding, giving the 32-byte, 16-byte-aligned stride WGSL's uniform-buffer
/// layout rules require for a struct whose largest member is `vec4<f32>`).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ReplayColorAdjustments {
    gamma_ratios: [f32; 4],
    grayscale_enhanced_contrast: f32,
    _padding: [f32; 3],
}

/// Builds `mono_sprites.wgsl`'s `@group(1)` `ColorAdjustments` uniform bind
/// group with an all-zero value. Phase 4/4b never captures this uniform (it
/// isn't one of `WgpuContext`'s per-primitive fixed buffers, just a single
/// small global one -- `WgpuContext::color_adjustments_buffer`), so replay
/// has no real captured value to re-upload here, unlike every other bind
/// group this module builds. An all-zero value is not an arbitrary
/// placeholder, though: working through `mono_sprites.wgsl`'s own math,
/// `light_on_dark_contrast` returns `0` when `grayscale_enhanced_contrast`
/// is `0` (its `enhancedContrast` multiplier), which makes `enhance_contrast`
/// an identity (`sample * 1 / (sample * 0 + 1) == sample`), and
/// `apply_alpha_correction` with an all-zero `gamma_ratios` reduces to
/// `a + a * (1 - a) * 0 == a`. So this doesn't reproduce production's
/// contrast/gamma correction, but it also doesn't corrupt the replayed
/// glyph's color -- the raw atlas sample comes through unmodified.
fn create_color_adjustments_bind_group(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("flamegraph_replay_color_adjustments_bind_group_layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let adjustments = ReplayColorAdjustments {
        gamma_ratios: [0.0; 4],
        grayscale_enhanced_contrast: 0.0,
        _padding: [0.0; 3],
    };
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flamegraph_replay_color_adjustments_buffer"),
        size: core::mem::size_of::<ReplayColorAdjustments>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytemuck::bytes_of(&adjustments));

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("flamegraph_replay_color_adjustments_bind_group"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    (bind_group_layout, bind_group)
}

/// Builds a `texture_2d<f32>` + `sampler` bind group (matching
/// `sprites_bind_group_layout` in `renderer.rs`: a `VERTEX_FRAGMENT`-visible
/// filterable texture at binding 0, a `FRAGMENT`-visible filtering sampler
/// at binding 1) from a real captured [`DeepCaptureTextureContents`] -- the
/// atlas-texture counterpart to [`create_storage_bind_group`]'s "re-upload
/// captured bytes verbatim" recipe. `bytes_per_pixel` picks the pixel
/// format: `1` is the atlas's monochrome pages (`R8Unorm`), anything else
/// (in practice always `4`) its polychrome ones (`Rgba8Unorm`) -- the only
/// two formats `WgpuAtlas::push_texture` (`platform/cross/atlas.rs`) ever
/// creates, so this doesn't need to carry a full `wgpu::TextureFormat`
/// mirror through the capture's data model just for this.
fn create_atlas_texture_bind_group(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    contents: &DeepCaptureTextureContents,
) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
    let format = if contents.bytes_per_pixel == 1 {
        wgpu::TextureFormat::R8Unorm
    } else {
        wgpu::TextureFormat::Rgba8Unorm
    };

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("flamegraph_replay_sprite_texture_bind_group_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });

    let width = contents.width.max(1);
    let height = contents.height.max(1);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("flamegraph_replay_sprite_texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    if !contents.bytes.is_empty() {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &contents.bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * contents.bytes_per_pixel),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("flamegraph_replay_sprite_sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("flamegraph_replay_sprite_texture_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });

    (bind_group_layout, bind_group)
}

/// Mirrors [`render_quads_step`]'s recipe for `mono_sprites.wgsl`'s own
/// `@group(3)` instance buffer (`mono_sprites_bind_group_layout` in
/// `renderer.rs`: a single `VERTEX`-visible read-only storage buffer), plus
/// two more real inputs `Quads` never needed: `@group(2)`'s atlas texture +
/// sampler, built from `texture_contents` (real captured bytes, issue #72 --
/// the completion of item 1's placeholder for this kind), and `@group(1)`'s
/// `ColorAdjustments` uniform, for which replay has no captured value (see
/// [`create_color_adjustments_bind_group`]'s doc comment for why an all-zero
/// default is a safe, documented simplification here).
fn render_mono_sprites_step(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mono_sprites_bytes: &[u8],
    texture_contents: &DeepCaptureTextureContents,
    call: &DeepCaptureDrawCall,
    viewport_width: u32,
    viewport_height: u32,
) -> Result<ReplayRenderOutput, ReplayError> {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("flamegraph_replay_mono_sprites_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("platform/cross/shaders/mono_sprites.wgsl").into()),
    });

    let (globals_bind_group_layout, globals_bind_group) =
        create_globals_bind_group(device, queue, viewport_width, viewport_height);
    let (color_adjustments_bind_group_layout, color_adjustments_bind_group) =
        create_color_adjustments_bind_group(device, queue);
    let (sprite_texture_bind_group_layout, sprite_texture_bind_group) =
        create_atlas_texture_bind_group(device, queue, texture_contents);
    let (mono_sprites_bind_group_layout, mono_sprites_bind_group) = create_storage_bind_group(
        device,
        queue,
        "flamegraph_replay_mono_sprites",
        wgpu::ShaderStages::VERTEX,
        mono_sprites_bytes,
    );

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("flamegraph_replay_mono_sprites_pipeline_layout"),
        bind_group_layouts: &[
            Some(&globals_bind_group_layout),
            Some(&color_adjustments_bind_group_layout),
            Some(&sprite_texture_bind_group_layout),
            Some(&mono_sprites_bind_group_layout),
        ],
        immediate_size: 0,
    });

    let color_targets = [Some(wgpu::ColorTargetState {
        format: REPLAY_TARGET_FORMAT,
        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
        write_mask: wgpu::ColorWrites::ALL,
    })];

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("flamegraph_replay_mono_sprites_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_mono_sprite"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_mono_sprite"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &color_targets,
        }),
        multiview_mask: None,
        cache: None,
    });

    render_pipeline_offscreen(
        device,
        queue,
        &pipeline,
        &[
            &globals_bind_group,
            &color_adjustments_bind_group,
            &sprite_texture_bind_group,
            &mono_sprites_bind_group,
        ],
        call.vertex_range,
        call.instance_range,
        viewport_width,
        viewport_height,
    )
}

/// Mirrors [`render_quads_step`]'s recipe for `poly_sprites.wgsl`'s own
/// `@group(2)` instance buffer (`poly_sprites_bind_group_layout` in
/// `renderer.rs`: a single `VERTEX_FRAGMENT`-visible read-only storage
/// buffer), plus a real `@group(1)` atlas texture + sampler built from
/// `texture_contents` (real captured bytes, issue #72). Simpler than
/// [`render_mono_sprites_step`]: `poly_sprites.wgsl` has no
/// `ColorAdjustments`-style uniform dependency at all.
fn render_poly_sprites_step(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    poly_sprites_bytes: &[u8],
    texture_contents: &DeepCaptureTextureContents,
    call: &DeepCaptureDrawCall,
    viewport_width: u32,
    viewport_height: u32,
) -> Result<ReplayRenderOutput, ReplayError> {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("flamegraph_replay_poly_sprites_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("platform/cross/shaders/poly_sprites.wgsl").into()),
    });

    let (globals_bind_group_layout, globals_bind_group) =
        create_globals_bind_group(device, queue, viewport_width, viewport_height);
    let (sprite_texture_bind_group_layout, sprite_texture_bind_group) =
        create_atlas_texture_bind_group(device, queue, texture_contents);
    let (poly_sprites_bind_group_layout, poly_sprites_bind_group) = create_storage_bind_group(
        device,
        queue,
        "flamegraph_replay_poly_sprites",
        wgpu::ShaderStages::VERTEX_FRAGMENT,
        poly_sprites_bytes,
    );

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("flamegraph_replay_poly_sprites_pipeline_layout"),
        bind_group_layouts: &[
            Some(&globals_bind_group_layout),
            Some(&sprite_texture_bind_group_layout),
            Some(&poly_sprites_bind_group_layout),
        ],
        immediate_size: 0,
    });

    let color_targets = [Some(wgpu::ColorTargetState {
        format: REPLAY_TARGET_FORMAT,
        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
        write_mask: wgpu::ColorWrites::ALL,
    })];

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("flamegraph_replay_poly_sprites_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_poly_sprite"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_poly_sprite"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &color_targets,
        }),
        multiview_mask: None,
        cache: None,
    });

    render_pipeline_offscreen(
        device,
        queue,
        &pipeline,
        &[&globals_bind_group, &sprite_texture_bind_group, &poly_sprites_bind_group],
        call.vertex_range,
        call.instance_range,
        viewport_width,
        viewport_height,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flamegraph::{DeepCaptureBufferContents, DeepCaptureDrawCall};
    use crate::flamegraph_ui_capture::{SceneSnapshot, UiBounds, UiElementNode, UiTreeCapture};

    fn sample_ui_tree_capture() -> UiTreeCapture {
        // Depth-DFS shape:
        //   0: root (depth 0)
        //     1: child-a (depth 1)
        //       2: grandchild (depth 2)
        //     3: child-b (depth 1)
        UiTreeCapture {
            window_id: 7,
            nodes: vec![
                UiElementNode {
                    type_name: "Root",
                    global_id_hash: 1,
                    depth: 0,
                    bounds: UiBounds::default(),
                    style: None,
                },
                UiElementNode {
                    type_name: "ChildA",
                    global_id_hash: 2,
                    depth: 1,
                    bounds: UiBounds::default(),
                    style: None,
                },
                UiElementNode {
                    type_name: "Grandchild",
                    global_id_hash: 3,
                    depth: 2,
                    bounds: UiBounds::default(),
                    style: None,
                },
                UiElementNode {
                    type_name: "ChildB",
                    global_id_hash: 4,
                    depth: 1,
                    bounds: UiBounds::default(),
                    style: None,
                },
            ],
            scene: SceneSnapshot::default(),
        }
    }

    #[test]
    fn ui_tree_replay_reconstructs_parent_child_relationships() {
        let replay = UiTreeReplay::new(sample_ui_tree_capture());

        assert_eq!(replay.node_count(), 4);
        assert_eq!(replay.roots(), &[0]);
        assert_eq!(replay.parent(0), None);
        assert_eq!(replay.parent(1), Some(0));
        assert_eq!(replay.parent(2), Some(1));
        assert_eq!(replay.parent(3), Some(0), "child-b should reattach to root, not the grandchild");
        assert_eq!(replay.children(0), &[1, 3]);
        assert_eq!(replay.children(1), &[2]);
        assert!(replay.children(2).is_empty());
        assert!(replay.children(3).is_empty());

        assert_eq!(replay.node(1).map(|node| node.type_name), Some("ChildA"));
        assert_eq!(replay.depth_first_indices().collect::<Vec<_>>(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn ui_tree_replay_handles_empty_capture() {
        let replay = UiTreeReplay::new(UiTreeCapture::default());
        assert_eq!(replay.node_count(), 0);
        assert!(replay.roots().is_empty());
    }

    fn sample_deep_capture() -> DeepCapture {
        DeepCapture {
            draw_calls: vec![
                DeepCaptureDrawCall {
                    sequence: 0,
                    kind: DrawCallKind::Quads,
                    pipeline_label: "quads",
                    pass_label: "main",
                    vertex_range: (0, 4),
                    instance_range: (0, 1),
                    bind_group_count: 2,
                    buffer_kind: Some(DeepCaptureBufferKind::Quads),
                    atlas_texture_id: None,
                    surface_id: None,
                },
                DeepCaptureDrawCall {
                    sequence: 1,
                    kind: DrawCallKind::MonoSprites,
                    pipeline_label: "mono_sprites",
                    pass_label: "main",
                    vertex_range: (0, 4),
                    instance_range: (0, 2),
                    bind_group_count: 4,
                    buffer_kind: Some(DeepCaptureBufferKind::MonoSprites),
                    atlas_texture_id: Some(9),
                    surface_id: None,
                },
                DeepCaptureDrawCall {
                    sequence: 2,
                    kind: DrawCallKind::Shadows,
                    pipeline_label: "shadows",
                    pass_label: "main",
                    vertex_range: (0, 4),
                    instance_range: (0, 1),
                    bind_group_count: 2,
                    buffer_kind: Some(DeepCaptureBufferKind::Shadows),
                    atlas_texture_id: None,
                    surface_id: None,
                },
            ],
            buffer_contents: vec![DeepCaptureBufferContents {
                kind: DeepCaptureBufferKind::Quads,
                bytes: vec![0; 64],
            }],
            texture_contents: Vec::new(),
            resources_finalized: false,
        }
    }

    #[test]
    fn deep_capture_replay_steps_through_command_stream() {
        let mut replay = DeepCaptureReplay::new(sample_deep_capture());

        assert_eq!(replay.draw_call_count(), 3);
        assert_eq!(replay.current_step(), 0);
        assert_eq!(replay.current_draw_call().map(|call| call.sequence), Some(0));

        let next = replay.step_to_next_draw_call().expect("should advance to the second draw call");
        assert_eq!(next.sequence, 1);
        assert_eq!(replay.current_step(), 1);

        let next = replay.step_to_next_draw_call().expect("should advance to the third draw call");
        assert_eq!(next.sequence, 2);
        assert_eq!(replay.current_step(), 2);

        assert!(
            replay.step_to_next_draw_call().is_none(),
            "already at the last draw call, stepping forward should report None"
        );
        assert_eq!(replay.current_step(), 2, "cursor should stay parked at the last valid draw call");

        let back = replay.step_to_previous_draw_call().expect("should step back to the second draw call");
        assert_eq!(back.sequence, 1);
        assert_eq!(replay.current_step(), 1);

        let seeked = replay.seek(0).expect("seeking within range should succeed");
        assert_eq!(seeked.sequence, 0);
        assert_eq!(replay.current_step(), 0);

        assert!(replay.seek(99).is_none(), "seeking out of range should report None");
        assert_eq!(replay.current_step(), 0, "cursor should not move on an out-of-range seek");

        replay.step_to_previous_draw_call();
        assert_eq!(replay.current_step(), 0, "stepping back from the first draw call should be a no-op");

        replay.seek(2);
        replay.reset();
        assert_eq!(replay.current_step(), 0);
    }

    #[test]
    fn deep_capture_replay_reports_resource_status_per_kind() {
        let replay = DeepCaptureReplay::new(sample_deep_capture());

        assert_eq!(replay.resource_status(0), DrawCallResourceStatus::Available);
        assert_eq!(replay.resource_status(1), DrawCallResourceStatus::TextureContentUnavailable);
        assert_eq!(
            replay.resource_status(2),
            DrawCallResourceStatus::BufferReadbackMissing,
            "Shadows buffer was never added to buffer_contents in this fixture"
        );
        assert_eq!(replay.resource_status(99), DrawCallResourceStatus::NoResource);
    }

    #[test]
    fn placeholder_checkerboard_has_expected_size_and_alternates() {
        let pixels = placeholder_checkerboard_rgba(4, 2, 1);
        assert_eq!(pixels.len(), 4 * 2 * 4);
        // Adjacent cells (cell size 1) should differ.
        let pixel_at = |x: usize, y: usize| -> [u8; 4] {
            let offset = (y * 4 + x) * 4;
            [pixels[offset], pixels[offset + 1], pixels[offset + 2], pixels[offset + 3]]
        };
        assert_ne!(pixel_at(0, 0), pixel_at(1, 0));
    }

    /// Creates a headless (surface-less) `wgpu::Device`/`Queue`, mirroring
    /// `flamegraph_gpu.rs`'s own test helper of the same shape (see that
    /// module's doc comment on why `enumerate_adapters` + pick-first is used
    /// instead of `request_adapter`, and why a missing adapter skips rather
    /// than fails the test in this sandbox).
    fn create_headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: None,
        });
        let adapter = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
            .into_iter()
            .next()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
    }

    /// Builds the raw bytes for one `Quad` the exact same way
    /// `WgpuRenderer::draw` does when uploading `scene.quads` to the GPU
    /// (`bytemuck::cast_slice(&scene.quads)`, `platform/cross/renderer.rs`),
    /// mirroring the production write path exactly so this test's captured
    /// bytes are genuinely representative of what Phase 4 would have read
    /// back from a live frame.
    fn quad_bytes(quad: &crate::scene::Quad) -> Vec<u8> {
        bytemuck::cast_slice(std::slice::from_ref(quad)).to_vec()
    }

    /// End-to-end GPU replay test: builds one opaque red `Quad` covering the
    /// whole replay viewport, captures its raw bytes the way a real
    /// `DeepCapture` would hold them, replays that single draw call against
    /// a real headless device, and asserts the rendered output's center
    /// pixel is red -- the concrete "reproduce the frame outside the live
    /// app" claim from issue #62, for the one primitive kind this phase
    /// wires up a real pipeline for.
    #[test]
    fn render_deep_capture_step_reproduces_a_captured_quad() {
        let Some((device, queue)) = create_headless_device() else {
            eprintln!("skipping render_deep_capture_step_reproduces_a_captured_quad: no wgpu adapter available");
            return;
        };

        let width = 32u32;
        let height = 32u32;
        let bounds = crate::Bounds {
            origin: crate::Point {
                x: crate::ScaledPixels::from(0.0),
                y: crate::ScaledPixels::from(0.0),
            },
            size: crate::Size {
                width: crate::ScaledPixels::from(width as f32),
                height: crate::ScaledPixels::from(height as f32),
            },
        };
        let red: crate::Hsla = crate::rgb(0xff0000).into();
        let quad = crate::scene::Quad {
            order: 0,
            border_style: crate::BorderStyle::default(),
            bounds,
            content_mask: crate::ContentMask { bounds },
            background: crate::solid_background(red),
            border_color: crate::Hsla { h: 0.0, s: 0.0, l: 0.0, a: 0.0 },
            corner_radii: Default::default(),
            border_widths: Default::default(),
        };

        let capture = DeepCapture {
            draw_calls: vec![DeepCaptureDrawCall {
                sequence: 0,
                kind: DrawCallKind::Quads,
                pipeline_label: "quads",
                pass_label: "main",
                vertex_range: (0, 4),
                instance_range: (0, 1),
                bind_group_count: 2,
                buffer_kind: Some(DeepCaptureBufferKind::Quads),
                atlas_texture_id: None,
                surface_id: None,
            }],
            buffer_contents: vec![DeepCaptureBufferContents {
                kind: DeepCaptureBufferKind::Quads,
                bytes: quad_bytes(&quad),
            }],
            texture_contents: Vec::new(),
            resources_finalized: true,
        };

        let replay = DeepCaptureReplay::new(capture);
        assert_eq!(replay.resource_status(0), DrawCallResourceStatus::Available);

        let output = render_deep_capture_step(&device, &queue, &replay, 0, width, height)
            .expect("replaying the captured quad draw call should succeed");

        assert!(!output.texture_unavailable);
        assert_eq!(output.width, width);
        assert_eq!(output.height, height);

        let pixel = output.pixel(width / 2, height / 2).expect("center pixel should be in bounds");
        assert!(pixel[0] > 200, "expected a strongly red center pixel, got {pixel:?}");
        assert!(pixel[1] < 60, "expected little green in the center pixel, got {pixel:?}");
        assert!(pixel[2] < 60, "expected little blue in the center pixel, got {pixel:?}");
        assert!(pixel[3] > 200, "expected an opaque center pixel, got {pixel:?}");
    }

    #[test]
    fn render_deep_capture_step_reports_texture_unavailable_placeholder_for_sprites() {
        let Some((device, queue)) = create_headless_device() else {
            eprintln!(
                "skipping render_deep_capture_step_reports_texture_unavailable_placeholder_for_sprites: no wgpu adapter available"
            );
            return;
        };

        let replay = DeepCaptureReplay::new(sample_deep_capture());
        let output = render_deep_capture_step(&device, &queue, &replay, 1, 8, 8)
            .expect("sprite draw calls should degrade gracefully rather than error");
        assert!(output.texture_unavailable);
        assert_eq!(output.rgba8.len(), 8 * 8 * 4);
    }

    /// End-to-end GPU replay test for `MonoSprites` (issue #72's completion
    /// of item 1's placeholder for this kind): builds a small, fully-opaque
    /// monochrome atlas page and one `MonochromeSprite` instance covering
    /// the whole replay viewport with a solid green text color, captures
    /// both the instance buffer bytes and the atlas texture bytes the way a
    /// real `DeepCapture` would hold them, replays that single draw call
    /// against a real headless device, and asserts the rendered output's
    /// center pixel is green and opaque -- proving `render_mono_sprites_step`
    /// actually samples the real captured atlas texture (not a placeholder)
    /// and that the all-zero `ColorAdjustments` default
    /// (`create_color_adjustments_bind_group`) doesn't corrupt the color.
    #[test]
    fn render_deep_capture_step_reproduces_a_captured_mono_sprite() {
        let Some((device, queue)) = create_headless_device() else {
            eprintln!("skipping render_deep_capture_step_reproduces_a_captured_mono_sprite: no wgpu adapter available");
            return;
        };

        let width = 32u32;
        let height = 32u32;

        // A 4x4 monochrome atlas page, fully opaque coverage everywhere
        // (255 = full intensity in R8Unorm), so the sampled alpha is 1.0
        // across the whole tile.
        let atlas_width = 4u32;
        let atlas_height = 4u32;
        let atlas_bytes = vec![255u8; (atlas_width * atlas_height) as usize];

        let bounds = crate::Bounds {
            origin: crate::Point {
                x: crate::ScaledPixels::from(0.0),
                y: crate::ScaledPixels::from(0.0),
            },
            size: crate::Size {
                width: crate::ScaledPixels::from(width as f32),
                height: crate::ScaledPixels::from(height as f32),
            },
        };
        let green: crate::Hsla = crate::rgb(0x00ff00).into();
        let sprite = crate::scene::MonochromeSprite {
            order: 0,
            pad: 0,
            bounds,
            content_mask: crate::ContentMask { bounds },
            text_color: crate::solid_text_color(green),
            tile: crate::AtlasTile {
                texture_id: crate::AtlasTextureId {
                    index: 0,
                    kind: crate::AtlasTextureKind::Monochrome,
                },
                tile_id: crate::TileId(0),
                padding: 0,
                bounds: crate::Bounds {
                    origin: crate::Point {
                        x: crate::DevicePixels(0),
                        y: crate::DevicePixels(0),
                    },
                    size: crate::Size {
                        width: crate::DevicePixels(atlas_width as i32),
                        height: crate::DevicePixels(atlas_height as i32),
                    },
                },
            },
            transformation: crate::scene::TransformationMatrix::unit(),
        };
        let sprite_bytes = bytemuck::cast_slice(std::slice::from_ref(&sprite)).to_vec();

        let encoded_atlas_id = 0u64; // (Monochrome as u64) << 32 | index 0
        let capture = DeepCapture {
            draw_calls: vec![DeepCaptureDrawCall {
                sequence: 0,
                kind: DrawCallKind::MonoSprites,
                pipeline_label: "mono_sprites",
                pass_label: "main",
                vertex_range: (0, 4),
                instance_range: (0, 1),
                bind_group_count: 4,
                buffer_kind: Some(DeepCaptureBufferKind::MonoSprites),
                atlas_texture_id: Some(encoded_atlas_id),
                surface_id: None,
            }],
            buffer_contents: vec![DeepCaptureBufferContents {
                kind: DeepCaptureBufferKind::MonoSprites,
                bytes: sprite_bytes,
            }],
            texture_contents: vec![DeepCaptureTextureContents {
                id: DeepCaptureTextureId::Atlas(encoded_atlas_id),
                width: atlas_width,
                height: atlas_height,
                bytes_per_pixel: 1,
                bytes: atlas_bytes,
            }],
            resources_finalized: true,
        };

        let replay = DeepCaptureReplay::new(capture);
        assert_eq!(replay.resource_status(0), DrawCallResourceStatus::Available);

        let output = render_deep_capture_step(&device, &queue, &replay, 0, width, height)
            .expect("replaying the captured mono sprite draw call should succeed");

        assert!(!output.texture_unavailable);
        let pixel = output.pixel(width / 2, height / 2).expect("center pixel should be in bounds");
        assert!(pixel[0] < 60, "expected little red in the center pixel, got {pixel:?}");
        assert!(pixel[1] > 200, "expected a strongly green center pixel, got {pixel:?}");
        assert!(pixel[2] < 60, "expected little blue in the center pixel, got {pixel:?}");
        assert!(pixel[3] > 200, "expected an opaque center pixel, got {pixel:?}");
    }

    /// End-to-end GPU replay test for `PolySprites`, mirroring the
    /// `MonoSprites` test above but for a polychrome (RGBA) atlas page and a
    /// `PolychromeSprite` instance, without `MonoSprites`'s extra
    /// `ColorAdjustments` dependency.
    #[test]
    fn render_deep_capture_step_reproduces_a_captured_poly_sprite() {
        let Some((device, queue)) = create_headless_device() else {
            eprintln!("skipping render_deep_capture_step_reproduces_a_captured_poly_sprite: no wgpu adapter available");
            return;
        };

        let width = 32u32;
        let height = 32u32;

        // A 2x2 polychrome atlas page, every pixel opaque blue.
        let atlas_width = 2u32;
        let atlas_height = 2u32;
        let atlas_bytes: Vec<u8> = std::iter::repeat_n([0u8, 0, 255, 255], (atlas_width * atlas_height) as usize)
            .flatten()
            .collect();

        let bounds = crate::Bounds {
            origin: crate::Point {
                x: crate::ScaledPixels::from(0.0),
                y: crate::ScaledPixels::from(0.0),
            },
            size: crate::Size {
                width: crate::ScaledPixels::from(width as f32),
                height: crate::ScaledPixels::from(height as f32),
            },
        };
        let sprite = crate::scene::PolychromeSprite {
            order: 0,
            pad: 0,
            grayscale: 0,
            opacity: 1.0,
            bounds,
            content_mask: crate::ContentMask { bounds },
            corner_radii: Default::default(),
            tile: crate::AtlasTile {
                texture_id: crate::AtlasTextureId {
                    index: 0,
                    kind: crate::AtlasTextureKind::Polychrome,
                },
                tile_id: crate::TileId(0),
                padding: 0,
                bounds: crate::Bounds {
                    origin: crate::Point {
                        x: crate::DevicePixels(0),
                        y: crate::DevicePixels(0),
                    },
                    size: crate::Size {
                        width: crate::DevicePixels(atlas_width as i32),
                        height: crate::DevicePixels(atlas_height as i32),
                    },
                },
            },
        };
        let sprite_bytes = bytemuck::cast_slice(std::slice::from_ref(&sprite)).to_vec();

        let encoded_atlas_id = 1u64 << 32; // (Polychrome as u64) << 32 | index 0
        let capture = DeepCapture {
            draw_calls: vec![DeepCaptureDrawCall {
                sequence: 0,
                kind: DrawCallKind::PolySprites,
                pipeline_label: "poly_sprites",
                pass_label: "main",
                vertex_range: (0, 4),
                instance_range: (0, 1),
                bind_group_count: 3,
                buffer_kind: Some(DeepCaptureBufferKind::PolySprites),
                atlas_texture_id: Some(encoded_atlas_id),
                surface_id: None,
            }],
            buffer_contents: vec![DeepCaptureBufferContents {
                kind: DeepCaptureBufferKind::PolySprites,
                bytes: sprite_bytes,
            }],
            texture_contents: vec![DeepCaptureTextureContents {
                id: DeepCaptureTextureId::Atlas(encoded_atlas_id),
                width: atlas_width,
                height: atlas_height,
                bytes_per_pixel: 4,
                bytes: atlas_bytes,
            }],
            resources_finalized: true,
        };

        let replay = DeepCaptureReplay::new(capture);
        assert_eq!(replay.resource_status(0), DrawCallResourceStatus::Available);

        let output = render_deep_capture_step(&device, &queue, &replay, 0, width, height)
            .expect("replaying the captured poly sprite draw call should succeed");

        assert!(!output.texture_unavailable);
        let pixel = output.pixel(width / 2, height / 2).expect("center pixel should be in bounds");
        assert!(pixel[0] < 60, "expected little red in the center pixel, got {pixel:?}");
        assert!(pixel[1] < 60, "expected little green in the center pixel, got {pixel:?}");
        assert!(pixel[2] > 200, "expected a strongly blue center pixel, got {pixel:?}");
        assert!(pixel[3] > 200, "expected an opaque center pixel, got {pixel:?}");
    }

    /// `Surfaces` always degrades to the checkerboard placeholder, even once
    /// its texture content has actually been captured -- see this module's
    /// doc comment for why (no captured `SurfaceParams` geometry to
    /// position/mask it with). This is the regression case for that
    /// specific claim: unlike
    /// `render_deep_capture_step_reports_texture_unavailable_placeholder_for_sprites`
    /// above (which has no captured texture at all for its `MonoSprites`
    /// call), this fixture's `Surfaces` call has real `texture_contents`
    /// keyed by its `surface_id`, and the replay still must not error or
    /// attempt to draw it.
    #[test]
    fn render_deep_capture_step_still_placeholders_surfaces_with_captured_texture() {
        let Some((device, queue)) = create_headless_device() else {
            eprintln!(
                "skipping render_deep_capture_step_still_placeholders_surfaces_with_captured_texture: no wgpu adapter available"
            );
            return;
        };

        let capture = DeepCapture {
            draw_calls: vec![DeepCaptureDrawCall {
                sequence: 0,
                kind: DrawCallKind::Surfaces,
                pipeline_label: "surfaces",
                pass_label: "main",
                vertex_range: (0, 4),
                instance_range: (0, 1),
                bind_group_count: 2,
                buffer_kind: None,
                atlas_texture_id: None,
                surface_id: Some(7),
            }],
            buffer_contents: Vec::new(),
            texture_contents: vec![DeepCaptureTextureContents {
                id: DeepCaptureTextureId::Surface(7),
                width: 4,
                height: 4,
                bytes_per_pixel: 4,
                bytes: vec![0; 4 * 4 * 4],
            }],
            resources_finalized: true,
        };

        let replay = DeepCaptureReplay::new(capture);
        assert_eq!(
            replay.resource_status(0),
            DrawCallResourceStatus::TextureContentUnavailable,
            "Surfaces should report unavailable even with a captured texture present"
        );

        let output = render_deep_capture_step(&device, &queue, &replay, 0, 8, 8)
            .expect("Surfaces draw calls should degrade gracefully rather than error");
        assert!(output.texture_unavailable);
        assert_eq!(output.rgba8.len(), 8 * 8 * 4);
    }

    /// `Shadows` now has a wired-up replay pipeline (see `render_shadows_step`),
    /// so the fixture's `Shadows` draw call (index 2 in `sample_deep_capture`)
    /// should fail with `MissingBufferContents`, not
    /// `UnsupportedDrawCallKind` -- `sample_deep_capture` only populates
    /// `buffer_contents` for `Quads`, deliberately leaving `Shadows`'s
    /// readback "missing" to exercise that error path.
    #[test]
    fn render_deep_capture_step_reports_missing_buffer_contents_for_unread_buffer() {
        let Some((device, queue)) = create_headless_device() else {
            eprintln!(
                "skipping render_deep_capture_step_reports_missing_buffer_contents_for_unread_buffer: no wgpu adapter available"
            );
            return;
        };

        let replay = DeepCaptureReplay::new(sample_deep_capture());
        let error = render_deep_capture_step(&device, &queue, &replay, 2, 8, 8)
            .expect_err("Shadows buffer contents were never populated in this fixture");
        assert!(matches!(error, ReplayError::MissingBufferContents(DeepCaptureBufferKind::Shadows)));
    }

    /// Builds the raw bytes for one `Shadow` the exact same way
    /// `WgpuRenderer::draw` uploads `scene.shadows` to the GPU -- a
    /// `bytemuck::cast_slice` reinterpret, mirroring [`quad_bytes`] above but
    /// for `crate::scene::Shadow`, proving `render_shadows_step`'s "same
    /// recipe" claim generalizes to a shader with a different (if structurally
    /// similar) storage-buffer struct.
    fn shadow_bytes(shadow: &crate::scene::Shadow) -> Vec<u8> {
        bytemuck::cast_slice(std::slice::from_ref(shadow)).to_vec()
    }

    /// End-to-end GPU replay test for `Shadows`, mirroring
    /// `render_deep_capture_step_reproduces_a_captured_quad` exactly but for
    /// a `Shadow` primitive: a large solid shape with a small blur radius
    /// (so the shadow's fragment-shader blur integral saturates to ~opaque
    /// well before its edges) and a content mask far larger than the shape
    /// itself (so nothing clips), asserting the rendered center pixel is a
    /// strongly opaque red -- proving `render_shadows_step` actually
    /// reproduces a captured `Shadows` draw call's real GPU output, not just
    /// that it fails to error.
    #[test]
    fn render_deep_capture_step_reproduces_a_captured_shadow() {
        let Some((device, queue)) = create_headless_device() else {
            eprintln!("skipping render_deep_capture_step_reproduces_a_captured_shadow: no wgpu adapter available");
            return;
        };

        let width = 32u32;
        let height = 32u32;
        let bounds = crate::Bounds {
            origin: crate::Point {
                x: crate::ScaledPixels::from(0.0),
                y: crate::ScaledPixels::from(0.0),
            },
            size: crate::Size {
                width: crate::ScaledPixels::from(width as f32),
                height: crate::ScaledPixels::from(height as f32),
            },
        };
        // Far larger than `bounds` (and than `bounds` expanded by the
        // vertex shader's `3 * blur_radius` margin), so the shadow's own
        // clip test never discards it.
        let content_mask_bounds = crate::Bounds {
            origin: crate::Point {
                x: crate::ScaledPixels::from(-1000.0),
                y: crate::ScaledPixels::from(-1000.0),
            },
            size: crate::Size {
                width: crate::ScaledPixels::from(2000.0),
                height: crate::ScaledPixels::from(2000.0),
            },
        };
        let red: crate::Hsla = crate::rgb(0xff0000).into();
        let shadow = crate::scene::Shadow {
            order: 0,
            blur_radius: crate::ScaledPixels::from(2.0),
            bounds,
            corner_radii: Default::default(),
            content_mask: crate::ContentMask { bounds: content_mask_bounds },
            color: red,
        };

        let capture = DeepCapture {
            draw_calls: vec![DeepCaptureDrawCall {
                sequence: 0,
                kind: DrawCallKind::Shadows,
                pipeline_label: "shadows",
                pass_label: "main",
                vertex_range: (0, 4),
                instance_range: (0, 1),
                bind_group_count: 2,
                buffer_kind: Some(DeepCaptureBufferKind::Shadows),
                atlas_texture_id: None,
                surface_id: None,
            }],
            buffer_contents: vec![DeepCaptureBufferContents {
                kind: DeepCaptureBufferKind::Shadows,
                bytes: shadow_bytes(&shadow),
            }],
            texture_contents: Vec::new(),
            resources_finalized: true,
        };

        let replay = DeepCaptureReplay::new(capture);
        assert_eq!(replay.resource_status(0), DrawCallResourceStatus::Available);

        let output = render_deep_capture_step(&device, &queue, &replay, 0, width, height)
            .expect("replaying the captured shadow draw call should succeed");

        assert!(!output.texture_unavailable);
        assert_eq!(output.width, width);
        assert_eq!(output.height, height);

        let pixel = output.pixel(width / 2, height / 2).expect("center pixel should be in bounds");
        assert!(pixel[0] > 200, "expected a strongly red center pixel, got {pixel:?}");
        assert!(pixel[1] < 60, "expected little green in the center pixel, got {pixel:?}");
        assert!(pixel[2] < 60, "expected little blue in the center pixel, got {pixel:?}");
        assert!(pixel[3] > 200, "expected a near-opaque center pixel, got {pixel:?}");
    }

    /// Mirrors `renderer.rs`'s private `GpuPathVertex` layout exactly
    /// (`xy_position`, `st_position`, `hsla`, `content_mask_origin`,
    /// `content_mask_size`, 48-byte stride) -- that struct is `pub(crate)`-
    /// invisible outside `renderer.rs`, so this test defines its own
    /// byte-identical copy rather than reusing it, the same thing a real
    /// `DeepCapture`'s raw bytes would look like without this test needing
    /// visibility into the renderer's private type.
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct TestPathVertex {
        xy_position: [f32; 2],
        st_position: [f32; 2],
        hsla: [f32; 4],
        content_mask_origin: [f32; 2],
        content_mask_size: [f32; 2],
    }

    /// End-to-end GPU replay test for `Paths`, the one buffer-backed kind
    /// with a genuinely different vertex layout and indexing scheme from
    /// every other kind here (see `render_paths_step`'s doc comment): builds
    /// one opaque red fill triangle (`st = (0, 1)`, always kept per
    /// `paths.wgsl`'s own doc comment) large enough to fully cover the
    /// replay viewport's center pixel, replays it as a real `Paths` draw
    /// call (`TriangleList` topology, `vertex_index`-addressed, no
    /// per-instance expansion), and asserts that pixel is red -- proving
    /// `render_paths_step` handles the different-vertex-layout case
    /// correctly, not just that it compiles.
    #[test]
    fn render_deep_capture_step_reproduces_a_captured_path_triangle() {
        let Some((device, queue)) = create_headless_device() else {
            eprintln!(
                "skipping render_deep_capture_step_reproduces_a_captured_path_triangle: no wgpu adapter available"
            );
            return;
        };

        let width = 32u32;
        let height = 32u32;

        // A right triangle with legs at x = -10 / y = -10 and hypotenuse
        // `x + y = 40`, comfortably covering the viewport's center pixel
        // (16, 16), where `x + y == 32 < 40`.
        let fill_vertex = |xy_position: [f32; 2]| TestPathVertex {
            xy_position,
            st_position: [0.0, 1.0],
            hsla: [0.0, 1.0, 0.5, 1.0],
            content_mask_origin: [-1000.0, -1000.0],
            content_mask_size: [2000.0, 2000.0],
        };
        let vertices = [fill_vertex([-10.0, -10.0]), fill_vertex([50.0, -10.0]), fill_vertex([-10.0, 50.0])];
        let path_bytes = bytemuck::cast_slice(&vertices).to_vec();

        let capture = DeepCapture {
            draw_calls: vec![DeepCaptureDrawCall {
                sequence: 0,
                kind: DrawCallKind::Paths,
                pipeline_label: "paths",
                pass_label: "main",
                vertex_range: (0, 3),
                instance_range: (0, 1),
                bind_group_count: 2,
                buffer_kind: Some(DeepCaptureBufferKind::Paths),
                atlas_texture_id: None,
                surface_id: None,
            }],
            buffer_contents: vec![DeepCaptureBufferContents {
                kind: DeepCaptureBufferKind::Paths,
                bytes: path_bytes,
            }],
            texture_contents: Vec::new(),
            resources_finalized: true,
        };

        let replay = DeepCaptureReplay::new(capture);
        assert_eq!(replay.resource_status(0), DrawCallResourceStatus::Available);

        let output = render_deep_capture_step(&device, &queue, &replay, 0, width, height)
            .expect("replaying the captured path draw call should succeed");

        assert!(!output.texture_unavailable);
        assert_eq!(output.width, width);
        assert_eq!(output.height, height);

        let pixel = output.pixel(width / 2, height / 2).expect("center pixel should be in bounds");
        assert!(pixel[0] > 200, "expected a strongly red center pixel, got {pixel:?}");
        assert!(pixel[1] < 60, "expected little green in the center pixel, got {pixel:?}");
        assert!(pixel[2] < 60, "expected little blue in the center pixel, got {pixel:?}");
        assert!(pixel[3] > 200, "expected an opaque center pixel, got {pixel:?}");
    }
}
