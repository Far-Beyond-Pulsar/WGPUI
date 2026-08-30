//! Producer-side triple-buffer for externally-rendered `WgpuSurface`
//! content — UNCHANGED (§5.5, §9's explicit "don't touch this"). Moved,
//! not rebuilt, from today's `src/surface_registry.rs` (772 lines).
//! See docs/gpu-native-architecture.md §5.5, §9.
//!
//! # What "moved, not rebuilt" meant in practice
//!
//! §9's risk table names the specific way Phase 4 could go wrong: "unifying
//! `WgpuSurface`'s composite path with `.boundary()`'s accidentally touches
//! `SurfaceRegistry`'s cross-thread producer-side synchronization … hard-won,
//! carefully-documented concurrency code that has nothing to do with the bug
//! being fixed." So this file is `src/platform/cross/surface_registry.rs`
//! copied, with exactly four differences, every one of them listed here so a
//! reviewer can diff the two and account for each:
//!
//! 1. **This module doc**, which the legacy file does not have.
//! 2. **The `#[cfg(feature = "flamegraph")]` members are not here** —
//!    `front_texture_snapshot`, `memory_usage`, `SurfaceTextureSnapshot`, and
//!    the `flamegraph_tests` module. They call
//!    `super::render_context::texel_size`/`texture_memory_bytes`, which live in
//!    the legacy crate, and §3.6/§8's Phase 7 is what moves devtools. Nothing
//!    else references them.
//! 3. **`SurfaceId` gains `from_raw`/`as_raw`.** The legacy field is
//!    `pub(crate)` and its one consumer reaches in as `surface_id.0` from a
//!    sibling module; a consumer in another crate needs a name for the same
//!    thing. Additive, and the field is unchanged.
//! 4. **`impl Default for SurfaceRegistry`**, because `new()` without it is a
//!    clippy finding under this workspace's `--deny warnings`. It calls `new`.
//!
//! Not among the differences, deliberately: the atomic state packing, the
//! compare-exchange swap loops, the generation gating, the backpressure rule,
//! the resize skip-while-compositing guard, the `Ordering` on every load and
//! store, and the six model tests at the bottom. Those are the thing §9 is
//! protecting.
//!
//! # And what the *consumer* side does instead
//!
//! Phase 4's actual change is one crate module over, in
//! `render/textures/external_surface.rs`: how an already-produced texture
//! enters the ordered scene and gets drawn. It calls
//! [`SurfaceRegistry::swap_ready_display_if_new`] and
//! [`SurfaceRegistry::front_view`] — the same two calls the legacy renderer's
//! surfaces batch makes, in the same order — and nothing else.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::Mutex;
use std::sync::MutexGuard;

/// An opaque identifier for a registered WGPU surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SurfaceId(pub(crate) u64);

impl SurfaceId {
    /// Wrap a raw handle.
    pub const fn from_raw(raw: u64) -> SurfaceId {
        SurfaceId(raw)
    }

    /// The raw handle.
    pub const fn as_raw(self) -> u64 {
        self.0
    }
}

/// Triple-buffered surface for lock-free rendering.
///
/// Uses three buffers with atomic index swaps:
/// - `rendering`: Currently being rendered by external thread
/// - `ready`: Latest complete frame, ready to display
/// - `display`: Currently being composited by GPUI
///
/// This allows external thread and compositor to run independently without blocking.
struct TripleBuffer {
    // The textures must stay alive for as long as their views are registered.
    _textures: [wgpu::Texture; 3],
    views: [wgpu::TextureView; 3],

    // Packed state: 2 bits each for rendering/ready/display indices.
    // layout: [display(2-bit) | ready(2-bit) | rendering(2-bit)]
    state: AtomicU8,

    // GPU synchronization: Track submission indices for each buffer to ensure
    // GPU work is complete before swapping buffers
    submission_indices: Mutex<[Option<wgpu::SubmissionIndex>; 3]>,

    // Redraw coalescing: prevents flooding compositor with thousands of requests/sec
    redraw_pending: std::sync::atomic::AtomicBool,

    // Monotonic count of producer swaps (rendering → ready): one increment per
    // frame the external renderer presents.
    frame_generation: AtomicU64,
    // The `frame_generation` value the compositor last swapped into `display`.
    // The compositor swaps `ready → display` only when these differ, so a paint
    // with no newly produced frame holds the current display buffer instead of
    // rotating to a stale one.
    last_composited_generation: AtomicU64,

    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
}

impl TripleBuffer {
    #[inline]
    fn pack_state(rendering: u8, ready: u8, display: u8) -> u8 {
        debug_assert!(rendering < 3 && ready < 3 && display < 3);
        debug_assert!(rendering != ready && ready != display && display != rendering);
        (display << 4) | (ready << 2) | rendering
    }

    #[inline]
    fn unpack_state(state: u8) -> (u8, u8, u8) {
        let rendering = state & 0x03;
        let ready = (state >> 2) & 0x03;
        let display = (state >> 4) & 0x03;
        (rendering, ready, display)
    }
}

/// Thread-safe registry of all active WGPU surfaces.
/// Maps `SurfaceId` to triple-buffered texture sets.
pub struct SurfaceRegistry {
    surfaces: Mutex<HashMap<SurfaceId, TripleBuffer>>,
    next_id: AtomicU64,
}

impl SurfaceRegistry {
    fn lock_surfaces(&self) -> MutexGuard<'_, HashMap<SurfaceId, TripleBuffer>> {
        self.surfaces
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub fn new() -> Self {
        Self {
            surfaces: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
        }
    }

    /// Create a new triple-buffered surface. Returns its `SurfaceId`.
    pub fn create(
        &self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> SurfaceId {
        let id = SurfaceId(self.next_id.fetch_add(1, Ordering::Relaxed));
        let tb = Self::create_triple_buffer(device, width, height, format);
        self.lock_surfaces().insert(id, tb);
        id
    }

    /// Atomically swap rendering and ready buffers (called by external thread after rendering).
    ///
    /// This is the "present" operation - it makes the newly rendered frame available
    /// to the compositor and gives the external thread a recycled buffer to render into.
    ///
    /// The `submission_idx` is stored to track GPU work completion, allowing the compositor
    /// to poll before sampling to prevent reading incomplete frames.
    ///
    /// Returns immediately without blocking.
    pub fn swap_rendering_ready(&self, id: SurfaceId, submission_idx: wgpu::SubmissionIndex) {
        if let Some(tb) = self.lock_surfaces().get(&id) {
            let current = tb.state.load(Ordering::Acquire);
            let (rendering, ready, display) = TripleBuffer::unpack_state(current);

            log::trace!(
                "[surface_id={:?}] swap_rendering_ready called - state before: rendering={}, ready={}, display={}",
                id,
                rendering,
                ready,
                display
            );

            // Store submission index for the buffer we just rendered to
            let mut submissions = tb
                .submission_indices
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(submission) = submissions.get_mut(rendering as usize) {
                *submission = Some(submission_idx);
            }

            // Atomic swap: rendering ↔ ready
            let mut current = tb.state.load(Ordering::Acquire);
            loop {
                let (rendering, ready, display) = TripleBuffer::unpack_state(current);
                let next = TripleBuffer::pack_state(ready, rendering, display);
                match tb
                    .state
                    .compare_exchange(current, next, Ordering::AcqRel, Ordering::Acquire)
                {
                    Ok(_) => break,
                    Err(updated) => current = updated,
                }
            }

            // A newly rendered frame now sits in `ready`; advance the generation
            // so the compositor swaps it to `display` exactly once.
            tb.frame_generation.fetch_add(1, Ordering::Release);
        }
    }

    /// Atomically swap rendering and ready buffers without GPU synchronization.
    ///
    /// DEPRECATED: Use swap_rendering_ready() with SubmissionIndex for proper GPU sync.
    /// This method exists for backward compatibility only.
    pub fn swap_rendering_ready_no_sync(&self, id: SurfaceId) {
        if let Some(tb) = self.lock_surfaces().get(&id) {
            let mut current = tb.state.load(Ordering::Acquire);
            loop {
                let (rendering, ready, display) = TripleBuffer::unpack_state(current);
                let next = TripleBuffer::pack_state(ready, rendering, display);
                match tb
                    .state
                    .compare_exchange(current, next, Ordering::AcqRel, Ordering::Acquire)
                {
                    Ok(_) => break,
                    Err(updated) => current = updated,
                }
            }

            // A newly rendered frame now sits in `ready`; advance the generation
            // so the compositor swaps it to `display` exactly once.
            tb.frame_generation.fetch_add(1, Ordering::Release);
        }
    }

    /// Atomically swap ready and display buffers with GPU synchronization.
    ///
    /// Polls the GPU to check if the ready buffer's work is complete before swapping.
    /// This ensures the compositor never samples incomplete frames.
    ///
    /// Returns `true` if a swap occurred, `false` if GPU work is incomplete (compositor
    /// should reuse the current display buffer).
    pub fn swap_ready_display(&self, _device: &wgpu::Device, id: SurfaceId) -> bool {
        if let Some(tb) = self.lock_surfaces().get(&id) {
            // Atomic swap: ready ↔ display
            // NOTE: We do NOT call device.poll() here because:
            // 1. The render thread owns the device and is actively using it
            // 2. Calling poll from multiple threads causes driver contention ("device lost")
            // 3. WGPU internally handles synchronization when textures are accessed
            // 4. The triple-buffer lock-free swaps are already safe
            let mut current = tb.state.load(Ordering::Acquire);
            loop {
                let (rendering, ready, display) = TripleBuffer::unpack_state(current);
                let next = TripleBuffer::pack_state(rendering, display, ready);
                match tb
                    .state
                    .compare_exchange(current, next, Ordering::AcqRel, Ordering::Acquire)
                {
                    Ok(_) => {
                        // Record consumption here too, not just in
                        // `swap_ready_display_if_new`. Producers use
                        // `has_unconsumed_frame` for backpressure, and if this
                        // path (the fast blit) left the composited generation
                        // behind, it would look to them like their frames were
                        // never being consumed.
                        tb.last_composited_generation.store(
                            tb.frame_generation.load(Ordering::Acquire),
                            Ordering::Release,
                        );
                        return true;
                    }
                    Err(updated) => current = updated,
                }
            }
        }
        false
    }

    /// Get the rendering buffer's `TextureView` (what external code renders into).
    pub fn back_view(&self, id: SurfaceId) -> Option<wgpu::TextureView> {
        let surfaces = self.lock_surfaces();
        surfaces.get(&id).map(|tb| {
            let (rendering, _, _) = TripleBuffer::unpack_state(tb.state.load(Ordering::Acquire));
            tb.views[rendering as usize].clone()
        })
    }

    /// Get the display buffer's `TextureView` (what the compositor reads from).
    pub fn front_view(&self, id: SurfaceId) -> Option<wgpu::TextureView> {
        let surfaces = self.lock_surfaces();
        surfaces.get(&id).map(|tb| {
            let (_, _, display) = TripleBuffer::unpack_state(tb.state.load(Ordering::Acquire));
            tb.views[display as usize].clone()
        })
    }

    /// Atomically retrieve both the rendering view and the corresponding texture
    /// dimensions. This is useful when a caller needs to create auxiliary
    /// resources (e.g. a depth buffer) that must exactly match the view's size.
    pub fn lock_and_get_back_with_size(
        &self,
        id: SurfaceId,
    ) -> Option<(wgpu::TextureView, (u32, u32))> {
        let surfaces = self.lock_surfaces();
        surfaces.get(&id).map(|tb| {
            let (rendering, _, _) = TripleBuffer::unpack_state(tb.state.load(Ordering::Acquire));
            (tb.views[rendering as usize].clone(), (tb.width, tb.height))
        })
    }

    /// Resize all three buffers, creating new textures with GPU synchronization.
    ///
    /// SAFETY: Waits for all pending GPU work to complete before destroying textures.
    /// This prevents use-after-free and ensures all GPU commands finish before
    /// texture resources are released.
    ///
    /// Also skips resize if compositor is actively using the buffers (redraw_pending).
    /// Returns `true` if the resize completed, `false` if it was skipped due to active composition.
    pub fn resize(&self, device: &wgpu::Device, id: SurfaceId, width: u32, height: u32) -> bool {
        let mut surfaces = self.lock_surfaces();
        if let Some(tb) = surfaces.get_mut(&id) {
            if tb.width == width && tb.height == height {
                return true;
            }

            // CRITICAL: Don't resize while compositor is rendering this surface!
            // If redraw_pending is true, compositor is using the buffers.
            // Skip resize - the element will retry on next frame.
            if tb.redraw_pending.load(Ordering::Relaxed) {
                return false;
            }

            // NOTE: We do NOT call device.poll() here because:
            // 1. The render thread owns the device and may be actively using it
            // 2. Calling poll from compositor thread causes device corruption
            // 3. WGPU internally ref-counts textures, so old views remain valid until dropped
            // 4. The skip-if-redraw-pending check above prevents resize during active composition

            // Now safe to recreate textures
            let new_tb = Self::create_triple_buffer(device, width, height, tb.format);
            *tb = new_tb;
            return true;
        }
        false
    }

    /// Recreate every surface texture after the device has been recovered.
    /// Surface IDs and dimensions remain stable, so producers can continue to
    /// use their handles and publish a fresh frame without re-registering.
    pub fn recover(&self, device: &wgpu::Device) {
        let mut surfaces = self.lock_surfaces();
        for surface in surfaces.values_mut() {
            let width = surface.width;
            let height = surface.height;
            let format = surface.format;
            *surface = Self::create_triple_buffer(device, width, height, format);
        }
    }

    /// Get the current size of a surface.
    pub fn size(&self, id: SurfaceId) -> Option<(u32, u32)> {
        let surfaces = self.lock_surfaces();
        surfaces.get(&id).map(|tb| (tb.width, tb.height))
    }

    /// Get the texture format for a surface.
    pub fn format(&self, id: SurfaceId) -> Option<wgpu::TextureFormat> {
        let surfaces = self.lock_surfaces();
        surfaces.get(&id).map(|tb| tb.format)
    }

    /// Remove a surface from the registry.
    pub fn remove(&self, id: SurfaceId) {
        self.lock_surfaces().remove(&id);
    }

    /// Set the redraw pending flag, returning the previous value.
    /// Used by present() to coalesce multiple redraw requests.
    pub fn set_redraw_pending(&self, id: SurfaceId) -> bool {
        if let Some(tb) = self.lock_surfaces().get(&id) {
            tb.redraw_pending.swap(true, Ordering::Relaxed)
        } else {
            false
        }
    }

    /// Clear the redraw pending flag.
    /// Called by the compositor after consuming a frame.
    pub fn clear_redraw_pending(&self, id: SurfaceId) {
        if let Some(tb) = self.lock_surfaces().get(&id) {
            tb.redraw_pending.store(false, Ordering::Relaxed);
        }
    }

    /// Get all surfaces that have pending redraws.
    /// Used by the fast blit path to check which surfaces need updating.
    pub fn get_pending_surfaces(&self) -> Vec<SurfaceId> {
        let surfaces = self.lock_surfaces();
        surfaces
            .iter()
            .filter(|(_, tb)| tb.redraw_pending.load(Ordering::Relaxed))
            .map(|(id, _)| *id)
            .collect()
    }

    /// Swap `ready → display` only if the external renderer has presented a new
    /// frame since the last successful compositor swap. Returns `true` if a swap
    /// occurred; when it returns `false`, the caller should keep compositing the
    /// current `display` buffer (via [`front_view`](Self::front_view)) unchanged.
    ///
    /// This is the gated counterpart to [`swap_ready_display`](Self::swap_ready_display).
    /// The GPUI paint path composites a surface every frame regardless of whether
    /// the producer rendered anything, so an *ungated* swap there rotates `display`
    /// to a stale buffer whenever the producer skipped a frame (engine-lock
    /// contention, a pending resize, …), making the canvas strobe. Gating on the
    /// producer generation keeps `display` steady until a genuinely new frame is
    /// ready.
    ///
    /// Runs entirely under the surfaces mutex, so the generation compare, the
    /// buffer swap, and the generation store are atomic with respect to the
    /// producer's `swap_rendering_ready*`.
    pub fn swap_ready_display_if_new(&self, id: SurfaceId) -> bool {
        if let Some(tb) = self.lock_surfaces().get(&id) {
            let current_gen = tb.frame_generation.load(Ordering::Acquire);
            let last = tb.last_composited_generation.load(Ordering::Acquire);
            if !Self::should_composite_swap(current_gen, last) {
                return false;
            }

            // Atomic swap: ready ↔ display
            let mut current = tb.state.load(Ordering::Acquire);
            loop {
                let (rendering, ready, display) = TripleBuffer::unpack_state(current);
                let next = TripleBuffer::pack_state(rendering, display, ready);
                match tb
                    .state
                    .compare_exchange(current, next, Ordering::AcqRel, Ordering::Acquire)
                {
                    Ok(_) => break,
                    Err(updated) => current = updated,
                }
            }

            tb.last_composited_generation
                .store(current_gen, Ordering::Release);
            return true;
        }
        false
    }

    /// True if the producer has published a frame that the compositor has not
    /// promoted to `display` yet.
    ///
    /// The triple buffer holds exactly **one** `ready` frame: publishing a
    /// second before the first is consumed makes `swap_rendering_ready` recycle
    /// the unconsumed buffer as the next render target, discarding that frame's
    /// pixels. The GPU work, command buffers and per-frame allocations behind it
    /// are not free, though — so an external render thread that ignores this is
    /// an unbounded producer feeding a display-rate consumer.
    ///
    /// Render threads should use this for backpressure: skip producing while it
    /// returns true. Prefer a *bounded* wait — the fast-blit consumer
    /// ([`swap_ready_display`](Self::swap_ready_display)) does not advance the
    /// composited generation, so this can stay true indefinitely on that path.
    pub fn has_unconsumed_frame(&self, id: SurfaceId) -> bool {
        self.lock_surfaces().get(&id).is_some_and(|tb| {
            Self::should_composite_swap(
                tb.frame_generation.load(Ordering::Acquire),
                tb.last_composited_generation.load(Ordering::Acquire),
            )
        })
    }

    /// The current producer-swap generation for a surface (increments once per
    /// presented frame). Returns `None` if the surface is not registered.
    pub fn frame_generation(&self, id: SurfaceId) -> Option<u64> {
        self.surfaces
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&id)
            .map(|tb| tb.frame_generation.load(Ordering::Acquire))
    }

    /// Pure decision function used by [`swap_ready_display_if_new`](Self::swap_ready_display_if_new):
    /// the compositor should swap `ready → display` iff the producer has advanced
    /// the generation since the compositor last presented. Both start at `0`, so
    /// the first compositor paint before any frame is produced is a no-op (keeps
    /// the initial buffer). Split out so the gating logic is unit-testable without
    /// a GPU device.
    #[inline]
    pub fn should_composite_swap(current_generation: u64, last_composited: u64) -> bool {
        current_generation != last_composited
    }

    fn create_triple_buffer(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> TripleBuffer {
        let w = width.max(1);
        let h = height.max(1);

        // Phase 4b of the profiling epic (issue #72) reads a surface's
        // currently-displayed triple-buffer texture back via
        // `copy_texture_to_buffer` during a triggered GPU deep capture,
        // which requires `COPY_SRC` on the source texture or wgpu's
        // validator rejects the encoder outright -- the exact same class of
        // hard, process-wide panic `render_context.rs`'s fixed buffers hit
        // before `COPY_SRC` was added to them (see that fix's commit
        // message for the full incident). Add it only when the capture code
        // that actually needs it is compiled in, so a non-`flamegraph`
        // build's surface textures are byte-for-byte the same as before
        // this change.
        #[cfg(feature = "flamegraph")]
        let deep_capture_readback = wgpu::TextureUsages::COPY_SRC;
        #[cfg(not(feature = "flamegraph"))]
        let deep_capture_readback = wgpu::TextureUsages::empty();

        let create_texture = |label: &str| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | deep_capture_readback,
                view_formats: &[],
            })
        };

        let tex0 = create_texture("surface_buffer_0");
        let tex1 = create_texture("surface_buffer_1");
        let tex2 = create_texture("surface_buffer_2");

        let view0 = tex0.create_view(&wgpu::TextureViewDescriptor::default());
        let view1 = tex1.create_view(&wgpu::TextureViewDescriptor::default());
        let view2 = tex2.create_view(&wgpu::TextureViewDescriptor::default());

        TripleBuffer {
            _textures: [tex0, tex1, tex2],
            views: [view0, view1, view2],
            state: AtomicU8::new(TripleBuffer::pack_state(0, 1, 2)),
            submission_indices: Mutex::new([None, None, None]),
            redraw_pending: std::sync::atomic::AtomicBool::new(false),
            frame_generation: AtomicU64::new(0),
            last_composited_generation: AtomicU64::new(0),
            width: w,
            height: h,
            format,
        }
    }
}

/// Difference 4 of the four this module's doc lists: additive, calls `new`.
impl Default for SurfaceRegistry {
    fn default() -> Self {
        SurfaceRegistry::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{SurfaceRegistry, TripleBuffer};

    // Minimal, GPU-free model of the three-buffer roles. We track which "frame"
    // (a monotonically increasing id) currently lives in each physical buffer,
    // so we can assert what the compositor would actually display after a
    // sequence of producer/consumer swaps — the real textures are irrelevant to
    // the swap/gating logic under test.
    struct Model {
        state: u8,
        /// Frame id stored in each physical buffer (0 = never rendered).
        contents: [u32; 3],
        /// Producer generation (count of rendering→ready swaps).
        generation: u64,
        /// Generation the compositor last swapped into display.
        last_composited: u64,
    }

    impl Model {
        fn new() -> Self {
            Self {
                state: TripleBuffer::pack_state(0, 1, 2),
                contents: [0; 3],
                generation: 0,
                last_composited: 0,
            }
        }

        /// External renderer draws `frame` into the rendering buffer, then swaps
        /// rendering ↔ ready (mirrors `swap_rendering_ready*`).
        fn produce(&mut self, frame: u32) {
            let (rendering, ready, display) = TripleBuffer::unpack_state(self.state);
            self.contents[rendering as usize] = frame;
            self.state = TripleBuffer::pack_state(ready, rendering, display);
            self.generation += 1;
        }

        /// Old, ungated compositor: always swaps ready ↔ display.
        fn composite_ungated(&mut self) {
            let (rendering, ready, display) = TripleBuffer::unpack_state(self.state);
            self.state = TripleBuffer::pack_state(rendering, display, ready);
        }

        /// New, gated compositor: swaps only when a new frame was produced
        /// (mirrors `swap_ready_display_if_new`).
        fn composite_gated(&mut self) {
            if !SurfaceRegistry::should_composite_swap(self.generation, self.last_composited) {
                return;
            }
            let (rendering, ready, display) = TripleBuffer::unpack_state(self.state);
            self.state = TripleBuffer::pack_state(rendering, display, ready);
            self.last_composited = self.generation;
        }

        /// The frame the compositor would currently display.
        fn displayed_frame(&self) -> u32 {
            let (_, _, display) = TripleBuffer::unpack_state(self.state);
            self.contents[display as usize]
        }
    }

    #[test]
    fn should_composite_swap_only_on_new_generation() {
        assert!(!SurfaceRegistry::should_composite_swap(0, 0));
        assert!(!SurfaceRegistry::should_composite_swap(5, 5));
        assert!(SurfaceRegistry::should_composite_swap(1, 0));
        assert!(SurfaceRegistry::should_composite_swap(6, 5));
    }

    #[test]
    fn indices_stay_a_permutation_across_swaps() {
        // Any sequence of transpositions must keep the three roles distinct,
        // otherwise `pack_state`'s debug asserts would fire and buffers alias.
        let mut m = Model::new();
        for frame in 1..=20u32 {
            m.produce(frame);
            m.composite_gated();
            let (r, ready, d) = TripleBuffer::unpack_state(m.state);
            assert!(
                r != ready && ready != d && d != r,
                "roles collided: {:?}",
                (r, ready, d)
            );
        }
    }

    #[test]
    fn ungated_compositor_regresses_to_stale_frame() {
        // Reproduces the bug: one produced frame, then the compositor paints
        // twice (as the GPUI path does every frame). The second, unpaired swap
        // rotates `display` to a buffer holding an older frame.
        let mut m = Model::new();
        m.produce(1);
        m.composite_ungated();
        assert_eq!(
            m.displayed_frame(),
            1,
            "first composite shows the new frame"
        );

        m.composite_ungated(); // unpaired paint, no new frame produced
        assert_ne!(
            m.displayed_frame(),
            1,
            "BUG: unpaired ungated swap regressed display off the latest frame"
        );
    }

    #[test]
    fn gated_compositor_holds_latest_frame_on_unpaired_paints() {
        // The fix: without a new produced frame, repeated compositor paints keep
        // showing the latest frame instead of strobing to a stale buffer.
        let mut m = Model::new();
        m.produce(1);
        m.composite_gated();
        assert_eq!(m.displayed_frame(), 1);

        for _ in 0..10 {
            m.composite_gated(); // unpaired paints (viewport skipped a frame)
            assert_eq!(
                m.displayed_frame(),
                1,
                "gated compositor must hold the last frame with no new production"
            );
        }
    }

    #[test]
    fn gated_compositor_tracks_new_frames() {
        // Normal 1:1 pairing advances the displayed frame each time.
        let mut m = Model::new();
        for frame in 1..=8u32 {
            m.produce(frame);
            m.composite_gated();
            assert_eq!(m.displayed_frame(), frame);
        }
    }

    #[test]
    fn gated_compositor_shows_latest_when_producer_outruns_compositor() {
        // Producer renders several frames before one composite; the compositor
        // should jump straight to the newest completed frame, never a stale one.
        let mut m = Model::new();
        m.produce(1);
        m.produce(2);
        m.produce(3);
        m.composite_gated();
        assert_eq!(m.displayed_frame(), 3);
    }
}
