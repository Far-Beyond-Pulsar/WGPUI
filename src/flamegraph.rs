//! CPU-side flamegraph capture engine (Phase 1 of the profiling epic, see issue #57).
//!
//! This module owns the data model, per-thread span recorder, capture session
//! lifecycle, and binary trace export. It absorbs and replaces the old
//! `profiler.rs`, whose flat merge-by-location buffer could not represent
//! call-stack nesting; `CpuSpan::depth` fixes that by recording stack position
//! at push time.
//!
//! GPU timestamp capture lives in `flamegraph_gpu.rs` so that this module never
//! needs to depend on `wgpu`. The two modules communicate through the plain
//! (non-wgpu) types defined here: `GpuSpan`, `GpuPassKind`, `GpuClockCalibration`.

use std::{
    cell::{LazyCell, RefCell},
    collections::{HashMap, VecDeque},
    hash::{DefaultHasher, Hash, Hasher},
    io::Write,
    sync::{
        Arc, Weak,
        atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering},
    },
    thread::ThreadId,
    time::SystemTime,
};
use crate::time_ext::Instant;

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

// The data model below derives `Serialize` only, not `Deserialize`, for most
// types. Several fields hold `&'static str` (`SpanName::Static`,
// `ElementAttribution`'s fields), and a genuinely `'static`-bound
// `Deserialize` impl can't be derived for those without leaking memory.
// `export_trace` is a write-only format this round (see module docs); the
// inline round-trip test in `tests` below decodes into separate owned mirror
// types instead of these live-process types, which is also what a future
// out-of-process viewer would do.
//
// The Phase 2 counter types below (`FrameCounters` and everything it's built
// from) are plain numeric data with no `&'static str` fields, so they derive
// `Deserialize` directly and are reused as-is by the round-trip test instead
// of needing their own mirror types.

/// Coarse-grained category for a [`CpuSpan`] or [`GpuSpan`], used by a future
/// viewer to color/group spans without needing to parse span names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum SpanCategory {
    /// A whole-window `Window::draw`/`Window::draw_roots` span.
    WindowFrame,
    /// `Drawable::request_layout`.
    ElementRequestLayout,
    /// `Drawable::prepaint`.
    ElementPrepaint,
    /// `Drawable::paint`.
    ElementPaint,
    /// A background-executor task.
    BackgroundTask,
    /// A GPU render pass bracketed by timestamp writes.
    GpuRenderPass,
    /// The whole-encoder submit-to-present bracket.
    GpuSubmitPresent,
    /// A span created through the public `flamegraph_span!` macro.
    UserDefined,
}

/// The name of a span. This round only produces `Static`; `Interned` exists so
/// that a future dynamic-name mechanism (e.g. per-entity type names built at
/// runtime) does not require a trace-format break.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum SpanName {
    /// A `&'static str`, the common case for instrumentation call sites.
    Static(&'static str),
    /// An index into a future string-interning table. Unused in Phase 1.
    Interned(u32),
}

/// Attribution of a [`CpuSpan`] to the element that produced it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ElementAttribution {
    /// `type_name::<E>()` of the element.
    pub type_name: &'static str,
    /// A `seahash` hash of the element's `GlobalElementId`, computed only when
    /// capture is active (never unconditionally, per the zero-overhead rule).
    /// Zero when the element has no stable id.
    pub global_id_hash: u64,
    /// Source file and line, when available (reuses the existing
    /// `Component::source_location()` / `InspectorElementId` plumbing).
    pub source_location: Option<(&'static str, u32)>,
}

/// Hash a `GlobalElementId` for element attribution, shared by Phase 1's
/// `ElementAttribution::global_id_hash` (`element.rs`) and Phase 5's UI-tree
/// capture (`flamegraph_ui_capture.rs`), so both hash element identity the
/// same way and their `global_id_hash` values are directly comparable/
/// joinable across the two capture mechanisms.
pub(crate) fn hash_global_element_id(id: &crate::GlobalElementId) -> u64 {
    let mut hasher = seahash::SeaHasher::new();
    id.hash(&mut hasher);
    hasher.finish()
}

/// A hashed, serde-friendly stand-in for `std::thread::ThreadId`, which does
/// not implement `Serialize`/`Deserialize` itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct ThreadKey(u64);

impl ThreadKey {
    fn current() -> Self {
        let mut hasher = DefaultHasher::new();
        std::thread::current().id().hash(&mut hasher);
        ThreadKey(hasher.finish())
    }
}

/// A single completed CPU-side span.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct CpuSpan {
    /// Span name.
    pub name: SpanName,
    /// Span category.
    pub category: SpanCategory,
    /// Nesting depth, taken from the span stack's length at push time. This is
    /// what lets a viewer reconstruct call-stack nesting in O(1) per span,
    /// without walking a parent-pointer tree.
    pub depth: u16,
    /// Start time in nanoseconds relative to the owning `Capture`'s anchor.
    pub start_ns: u64,
    /// Duration in nanoseconds, saturating at `u32::MAX` (~4.29s) rather than
    /// wrapping, since a span that long is already pathological for a frame
    /// profiler and silent wraparound would be worse than a clamped value.
    pub duration_ns: u32,
    /// Which thread recorded this span.
    pub thread_id: ThreadKey,
    /// Element attribution, when this span was produced by element drawing.
    pub element: Option<ElementAttribution>,
}

/// Which of the renderer's timestamp-bracketed GPU passes a [`GpuSpan`] covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum GpuPassKind {
    /// The main render pass.
    Main,
    /// The main render pass, resumed after a filter-group interruption.
    MainResumed,
    /// A `with_filter_layer` offscreen group render pass.
    FilterGroup,
    /// A filter-group pass, resumed after nested filter groups.
    FilterGroupResumed,
    /// The fast (no-compositor) surface blit pass.
    FastSurfaceBlit,
    /// The whole-encoder submit-to-present bracket.
    SubmitPresent,
}

/// A single completed GPU-side span, resolved from a wgpu timestamp query pair.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct GpuSpan {
    /// Span name.
    pub name: SpanName,
    /// Start time in nanoseconds relative to the owning `Capture`'s anchor,
    /// converted from GPU ticks using the session's [`GpuClockCalibration`].
    pub start_ns: u64,
    /// Duration in nanoseconds.
    pub duration_ns: u32,
    /// Which pass this span covers.
    pub pass_kind: GpuPassKind,
    /// The `GpuQueryManager` generation this span's query pair came from, for
    /// diagnosing readback ordering issues in a future viewer.
    pub query_set_generation: u64,
}

/// One-time CPU/GPU clock calibration for a capture session, computed from a
/// bracketing pair of `write_timestamp` calls and `queue.get_timestamp_period()`.
/// Stored as plain data (no wgpu types) so it can live in this module.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize)]
pub struct GpuClockCalibration {
    /// CPU-side anchor, in nanoseconds since the `Capture`'s anchor `Instant`.
    pub cpu_anchor_ns: u64,
    /// GPU-side anchor, in raw timestamp ticks.
    pub gpu_anchor_ticks: u64,
    /// Nanoseconds per GPU timestamp tick (`queue.get_timestamp_period()`).
    pub ns_per_tick: f32,
    /// Whether calibration actually ran (false when no capture with
    /// `capture_gpu: true` has ever started).
    pub calibrated: bool,
}

/// All spans captured for one `Window::draw` cycle.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FrameCapture {
    /// Monotonically increasing index, unique within a capture session.
    pub frame_index: u64,
    /// Hashed `WindowId` this frame belongs to (multi-window captures filter
    /// on this field rather than needing separate per-window captures).
    pub window_id: u64,
    /// Spans recorded on the thread that drew this frame (the foreground
    /// thread), in completion order.
    pub cpu_spans: Vec<CpuSpan>,
    /// Spans recorded on other threads whose `start_ns` fell within this
    /// frame's window.
    pub background_spans: Vec<CpuSpan>,
    /// GPU spans, attached asynchronously after query readback. May be empty
    /// even for an otherwise-complete frame; check `gpu_spans_finalized`.
    pub gpu_spans: Vec<GpuSpan>,
    /// Whether `gpu_spans` is done being populated. GPU spans typically arrive
    /// 1-2 frames after the CPU side closes, because of the resolve/map_async
    /// readback latency.
    pub gpu_spans_finalized: bool,
    /// Set when this frame recorded more GPU passes than
    /// `MAX_GPU_SPANS_PER_FRAME` could hold, so `gpu_spans` is incomplete.
    pub gpu_spans_truncated: bool,
    /// Frame start, in nanoseconds relative to the capture's anchor.
    pub frame_start_ns: u64,
    /// Frame end, in nanoseconds relative to the capture's anchor.
    pub frame_end_ns: u64,
    /// CPU wall-clock instant `queue.submit()` returned for this frame's GPU
    /// work, anchor-relative. This is a **CPU-side observation of when the
    /// CPU asked the GPU to run something** -- not when the GPU actually
    /// started running it (that's `gpu_spans`' calibrated `start_ns`, a value
    /// *inferred* from a GPU-clock timestamp via the session's one-time
    /// calibration, not directly observed on the CPU timeline) and not when
    /// drawing finished on the CPU (`frame_end_ns`, which happens before
    /// submission is even asked for). Plotting this alongside the calibrated
    /// GPU span start is what surfaces GPU queue backlog: a growing gap here
    /// means the GPU is falling behind CPU-side submission, not that
    /// individual passes are slow. `None` until a submission has been
    /// correlated to this frame (or if this frame never requested GPU
    /// capture at all).
    pub cpu_gpu_submit_ns: Option<u64>,
    /// CPU wall-clock instant the render thread's non-blocking poll first
    /// observed this frame's GPU readback as complete, anchor-relative. This
    /// is when the **CPU found out** the GPU had fenced/finished -- not when
    /// the GPU actually finished (that's the end of `gpu_spans`' calibrated
    /// timeline) and not a blocking wait's completion (readback is polled
    /// non-blockingly, so this always lags true GPU completion by however
    /// long it took the render thread to next call `poll_readback`, on the
    /// order of 1-2 frames by design -- see `GpuSpan`'s doc comment). The gap
    /// between the calibrated GPU end time and this value is CPU-side
    /// readback/polling latency, distinct from GPU execution time itself.
    /// `None` until observed.
    pub cpu_gpu_fence_observed_ns: Option<u64>,
    /// Aggregate "how much work" counters for this frame (Phase 2, issue
    /// #58), gathered alongside the timing spans above. See
    /// [`FrameCounters`].
    pub counters: FrameCounters,
}

/// Number of `RenderPass::draw` calls and total primitive count issued for
/// one [`PrimitiveBatch`](crate::platform::cross::renderer) kind during a
/// frame. "Primitives" means instance count for quads/shadows/sprites/
/// underlines/backdrop_filters/surfaces, and path count for paths (matching
/// how each kind is submitted in `WgpuRenderer::draw`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct PassCounter {
    /// Number of `RenderPass::draw` calls issued for this primitive kind.
    pub draw_calls: u32,
    /// Total primitive count submitted across those draw calls.
    pub primitives: u32,
}

impl PassCounter {
    fn record(&mut self, primitives: u32) {
        self.draw_calls = self.draw_calls.saturating_add(1);
        self.primitives = self.primitives.saturating_add(primitives);
    }
}

/// Per-frame draw-call/primitive tallies, one [`PassCounter`] per
/// `PrimitiveBatch` kind. Tallied directly in `WgpuRenderer::draw`'s
/// `PrimitiveBatch` match arms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct DrawCallCounters {
    /// `PrimitiveBatch::Quads`.
    pub quads: PassCounter,
    /// `PrimitiveBatch::Shadows`.
    pub shadows: PassCounter,
    /// `PrimitiveBatch::MonochromeSprites`.
    pub mono_sprites: PassCounter,
    /// `PrimitiveBatch::PolychromeSprites`.
    pub poly_sprites: PassCounter,
    /// `PrimitiveBatch::Paths`.
    pub paths: PassCounter,
    /// `PrimitiveBatch::Underlines`.
    pub underlines: PassCounter,
    /// `PrimitiveBatch::BackdropFilters`.
    pub backdrop_filters: PassCounter,
    /// `PrimitiveBatch::Surfaces`.
    pub surfaces: PassCounter,
}

impl DrawCallCounters {
    fn get_mut(&mut self, kind: DrawCallKind) -> &mut PassCounter {
        match kind {
            DrawCallKind::Quads => &mut self.quads,
            DrawCallKind::Shadows => &mut self.shadows,
            DrawCallKind::MonoSprites => &mut self.mono_sprites,
            DrawCallKind::PolySprites => &mut self.poly_sprites,
            DrawCallKind::Paths => &mut self.paths,
            DrawCallKind::Underlines => &mut self.underlines,
            DrawCallKind::BackdropFilters => &mut self.backdrop_filters,
            DrawCallKind::Surfaces => &mut self.surfaces,
        }
    }
}

/// Which `PrimitiveBatch` kind a [`record_draw_call`] call is tallying. `pub`
/// (not `pub(crate)`) since Phase 4's [`DeepCaptureDrawCall::kind`] field also
/// uses this type and is itself part of the public `flamegraph` API surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DrawCallKind {
    /// `PrimitiveBatch::Quads`.
    Quads,
    /// `PrimitiveBatch::Shadows`.
    Shadows,
    /// `PrimitiveBatch::MonochromeSprites`.
    MonoSprites,
    /// `PrimitiveBatch::PolychromeSprites`.
    PolySprites,
    /// `PrimitiveBatch::Paths`.
    Paths,
    /// `PrimitiveBatch::Underlines`.
    Underlines,
    /// `PrimitiveBatch::BackdropFilters`.
    BackdropFilters,
    /// `PrimitiveBatch::Surfaces`.
    Surfaces,
}

/// Per-frame atlas tile allocator activity, tallied in `WgpuAtlas`
/// (`src/platform/cross/atlas.rs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct AtlasCounters {
    /// New tiles allocated (a `get_or_insert_with` cache miss that produced a
    /// tile).
    pub tiles_allocated: u32,
    /// Tiles removed from the atlas (`PlatformAtlas::remove`).
    pub tiles_evicted: u32,
    /// `get_or_insert_with` calls that found an existing tile.
    pub cache_hits: u32,
    /// `get_or_insert_with` calls that did not find an existing tile (whether
    /// or not building the replacement actually produced a tile).
    pub cache_misses: u32,
}

/// Per-frame input/notification activity, tallied in `Window::dispatch_event`
/// and `App::notify`/`WindowInvalidator::invalidate`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct EventCounters {
    /// `Window::dispatch_event` calls (mouse/keyboard/etc. input).
    pub input_events_dispatched: u32,
    /// `AppContext::notify`/`Context::notify` calls.
    pub notify_calls: u32,
    /// Entities marked dirty via `WindowInvalidator::invalidate`.
    pub entities_invalidated: u32,
}

/// Aggregate "how much work" counters for one frame (Phase 2 of the
/// profiling epic, issue #58), the direct successor to the reverted
/// `render_stats` module's stderr counters — same data, but attached to the
/// [`FrameCapture`] it describes instead of being dumped to stderr on a
/// timer, so it can be queried through [`Capture::counter_summary`].
///
/// Attribution note: draw-call/atlas/event work is tallied into a
/// thread-local accumulator (see `FRAME_COUNTERS` below) that is drained into
/// whichever `FrameCapture` closes next, mirroring the single-foreground-
/// thread assumption `CaptureState::last_opened_frame_index` already
/// documents. `EventCounters` in particular can include work that happened
/// between the previous frame's close and this frame's open (e.g. input
/// dispatched while idle, or entities invalidated by a background task
/// completion) — that is intentional: it is exactly the work that
/// contributed to this frame being drawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct FrameCounters {
    /// Per-`PrimitiveBatch`-kind draw-call/primitive tallies.
    pub draw_calls: DrawCallCounters,
    /// Atlas tile allocator activity.
    pub atlas: AtlasCounters,
    /// Input/notification activity.
    pub events: EventCounters,
}

/// A stopped capture session: a ring-buffered (by frame count) sequence of
/// [`FrameCapture`]s, ready to be inspected or exported.
#[derive(Debug)]
pub struct Capture {
    anchor: Instant,
    frames: VecDeque<FrameCapture>,
    /// Frame-count ring-buffer bound this capture was configured with.
    pub max_frames: usize,
    // Always `false` on a `Capture` returned by `CaptureHandle::stop` today
    // (this type only represents already-stopped sessions this round); kept
    // as part of the locked-in data model since later phases may return a
    // `Capture` view onto a still-running session.
    #[allow(dead_code)]
    enabled: AtomicBool,
    /// Total `Window::on_request_frame` invocations across the whole session
    /// that took the full compositor draw path (`Window::draw` +
    /// `Window::present`). Session-wide rather than ring-buffer-windowed,
    /// unlike everything in `counter_summary`'s per-frame aggregates: a
    /// fast-path present (`Window::present_framebuffer_only`) never opens a
    /// `FrameCapture` at all (there is no compositor work to attribute spans
    /// or counters to), so this ratio can't be recovered from the ring
    /// buffer after the fact the way the other stats can. See
    /// `record_frame_pacing`.
    pub full_draw_frame_count: u64,
    /// Total `Window::on_request_frame` invocations across the whole session
    /// that took the fast, no-compositor present-only path.
    pub fast_path_frame_count: u64,
}

impl Capture {
    /// Iterate over captured frames, oldest first.
    pub fn frames(&self) -> impl Iterator<Item = &FrameCapture> {
        self.frames.iter()
    }

    /// Number of frames currently held.
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Approximate heap bytes retained by this stopped capture's own frames
    /// (Phase 3 of the profiling epic, issue #59): the CPU/GPU span vectors
    /// held inside every [`FrameCapture`] in `self.frames`, plus a
    /// per-`FrameCapture` fixed-field allowance. A profiler that doesn't
    /// account for its own footprint is misleading, so this is meant to sit
    /// alongside [`MemorySnapshot::capture_engine_bytes`] (which reports the
    /// *live*, still-recording engine's footprint) as the equivalent number
    /// for a trace that has already been stopped and handed to the caller.
    pub fn retained_trace_bytes(&self) -> u64 {
        self.frames.iter().map(frame_capture_memory_usage).sum()
    }

    /// Aggregate mean/max statistics over the frames currently held in this
    /// capture's ring buffer (see [`CounterSummary`]'s doc comment). This is
    /// the Phase 2 (issue #58) replacement for the reverted `render_stats`
    /// module's periodic stderr dump — a queryable API instead of a timer.
    pub fn counter_summary(&self) -> CounterSummary {
        let frame_count = self.frames.len();

        let (mean_frame_duration_ms, max_frame_duration_ms, fps) = if frame_count == 0 {
            (0.0, 0.0, 0.0)
        } else {
            let mut total_duration_ns: u128 = 0;
            let mut max_duration_ns: u64 = 0;
            for frame in &self.frames {
                let duration_ns = frame.frame_end_ns.saturating_sub(frame.frame_start_ns);
                total_duration_ns += duration_ns as u128;
                max_duration_ns = max_duration_ns.max(duration_ns);
            }
            let mean_ms = (total_duration_ns as f64 / frame_count as f64) / 1.0e6;
            let max_ms = max_duration_ns as f64 / 1.0e6;

            let fps = if let (Some(first), Some(last)) = (self.frames.front(), self.frames.back())
                && frame_count >= 2
            {
                let span_ns = last.frame_end_ns.saturating_sub(first.frame_start_ns);
                if span_ns > 0 {
                    ((frame_count - 1) as f64) / (span_ns as f64 / 1.0e9)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            (mean_ms, max_ms, fps)
        };

        let draw_calls = DrawCallSummary {
            quads: pass_counter_summary(&self.frames, |counters| counters.quads),
            shadows: pass_counter_summary(&self.frames, |counters| counters.shadows),
            mono_sprites: pass_counter_summary(&self.frames, |counters| counters.mono_sprites),
            poly_sprites: pass_counter_summary(&self.frames, |counters| counters.poly_sprites),
            paths: pass_counter_summary(&self.frames, |counters| counters.paths),
            underlines: pass_counter_summary(&self.frames, |counters| counters.underlines),
            backdrop_filters: pass_counter_summary(&self.frames, |counters| counters.backdrop_filters),
            surfaces: pass_counter_summary(&self.frames, |counters| counters.surfaces),
        };

        let total_hits: u64 = self.frames.iter().map(|frame| frame.counters.atlas.cache_hits as u64).sum();
        let total_misses: u64 = self.frames.iter().map(|frame| frame.counters.atlas.cache_misses as u64).sum();
        let cache_hit_rate = if total_hits + total_misses > 0 {
            total_hits as f64 / (total_hits + total_misses) as f64
        } else {
            0.0
        };
        let atlas = AtlasSummary {
            tiles_allocated: mean_max(self.frames.iter().map(|frame| frame.counters.atlas.tiles_allocated)),
            tiles_evicted: mean_max(self.frames.iter().map(|frame| frame.counters.atlas.tiles_evicted)),
            cache_hits: mean_max(self.frames.iter().map(|frame| frame.counters.atlas.cache_hits)),
            cache_misses: mean_max(self.frames.iter().map(|frame| frame.counters.atlas.cache_misses)),
            cache_hit_rate,
        };

        let events = EventSummary {
            input_events_dispatched: mean_max(
                self.frames.iter().map(|frame| frame.counters.events.input_events_dispatched),
            ),
            notify_calls: mean_max(self.frames.iter().map(|frame| frame.counters.events.notify_calls)),
            entities_invalidated: mean_max(
                self.frames.iter().map(|frame| frame.counters.events.entities_invalidated),
            ),
        };

        CounterSummary {
            frame_count,
            fps,
            mean_frame_duration_ms,
            max_frame_duration_ms,
            draw_calls,
            atlas,
            events,
            gpu_timeline: gpu_timeline_summary(&self.frames),
            present_mode: current_present_mode(),
            full_draw_frame_count: self.full_draw_frame_count,
            fast_path_frame_count: self.fast_path_frame_count,
        }
    }

    /// Write this capture out in WGPUI's versioned binary trace format:
    /// an 8-byte magic, a `u32` format version, a fixed `bytemuck::Pod` header,
    /// then a sequence of independently length-prefixed, bincode-encoded
    /// [`FrameCapture`] chunks (not one big `Vec<FrameCapture>` blob), so the
    /// format is streaming-friendly from day one. There is no public reader
    /// this round; a future viewer defines its own.
    pub fn export_trace(&self, writer: &mut impl Write) -> anyhow::Result<()> {
        let anchor_system_time = SystemTime::now()
            .checked_sub(self.anchor.elapsed())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        let anchor_unix_nanos = anchor_system_time
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
            .min(u64::MAX as u128) as u64;

        let calibration = gpu_calibration();
        let header = TraceHeader {
            anchor_unix_nanos,
            gpu_cpu_anchor_ns: calibration.cpu_anchor_ns,
            gpu_anchor_ticks: calibration.gpu_anchor_ticks,
            frame_count: self.frames.len() as u32,
            cpu_clock_source: CPU_CLOCK_SOURCE_STD_INSTANT,
            gpu_ns_per_tick: calibration.ns_per_tick,
            gpu_calibrated: calibration.calibrated as u32,
            reserved: [0; 16],
        };

        writer.write_all(&TRACE_MAGIC)?;
        writer.write_all(&TRACE_FORMAT_VERSION.to_le_bytes())?;
        writer.write_all(bytemuck::bytes_of(&header))?;

        for frame in &self.frames {
            let encoded = bincode::serde::encode_to_vec(frame, bincode::config::standard())
                .map_err(|error| anyhow::anyhow!("failed to encode frame capture: {error}"))?;
            let length = u32::try_from(encoded.len())
                .map_err(|_| anyhow::anyhow!("frame capture too large to encode"))?;
            writer.write_all(&length.to_le_bytes())?;
            writer.write_all(&encoded)?;
        }

        Ok(())
    }
}

const TRACE_MAGIC: [u8; 8] = *b"WGPUIFLG";
const TRACE_FORMAT_VERSION: u32 = 1;
const CPU_CLOCK_SOURCE_STD_INSTANT: u32 = 0;

/// Fixed-size POD trace header. Field order is chosen so that no implicit
/// padding is inserted (three `u64`s, then four `u32`-sized fields, then a
/// byte array), which `bytemuck::Pod`'s derive requires.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TraceHeader {
    anchor_unix_nanos: u64,
    gpu_cpu_anchor_ns: u64,
    gpu_anchor_ticks: u64,
    frame_count: u32,
    cpu_clock_source: u32,
    gpu_ns_per_tick: f32,
    gpu_calibrated: u32,
    reserved: [u8; 16],
}

/// Error returned by [`start_capture`] when a capture session is already active.
#[derive(Debug, thiserror::Error)]
#[error("a flamegraph capture is already in progress")]
pub struct AlreadyCapturingError;

/// Options for [`start_capture`].
#[derive(Debug, Clone, Copy)]
pub struct CaptureOptions {
    /// Ring-buffer bound, in frames. Older frames are evicted once exceeded.
    pub max_frames: usize,
    /// Whether to also record GPU timestamp spans. When false, `flamegraph_gpu`
    /// never allocates a `QuerySet`, so GPU capture is also zero-cost when
    /// unused.
    pub capture_gpu: bool,
}

impl Default for CaptureOptions {
    fn default() -> Self {
        Self {
            max_frames: 600,
            capture_gpu: true,
        }
    }
}

/// A handle to an in-progress capture session. Dropping this without calling
/// [`CaptureHandle::stop`] leaves the capture running; there is intentionally
/// no `Drop`-based auto-stop, so that capture lifetime is explicit.
pub struct CaptureHandle {
    state: Arc<CaptureState>,
}

impl CaptureHandle {
    /// Stop the capture session and return the accumulated frames.
    pub fn stop(self) -> Capture {
        CAPTURE_ENABLED.store(false, Ordering::Release);
        *ACTIVE_CAPTURE.lock() = None;
        self.state.finalize_open_frames();

        Capture {
            anchor: self.state.anchor,
            frames: self.state.finished_frames.lock().clone(),
            max_frames: self.state.max_frames,
            enabled: AtomicBool::new(false),
            full_draw_frame_count: self.state.full_draw_frames.load(Ordering::Relaxed),
            fast_path_frame_count: self.state.fast_path_frames.load(Ordering::Relaxed),
        }
    }

    /// Whether a capture session is still active. Always true until `stop` is
    /// called (single-session invariant enforced by [`start_capture`]).
    pub fn is_recording(&self) -> bool {
        CAPTURE_ENABLED.load(Ordering::Relaxed)
    }
}

/// Whether a capture is currently active. Checked with `Ordering::Relaxed` by
/// every `enter_span` call; this is the entire cost of instrumentation when a
/// flamegraph-enabled build is not actively capturing.
static CAPTURE_ENABLED: AtomicBool = AtomicBool::new(false);

static ACTIVE_CAPTURE: parking_lot::Mutex<Option<Arc<CaptureState>>> = parking_lot::Mutex::new(None);

static GPU_CALIBRATION: parking_lot::Mutex<GpuClockCalibration> = parking_lot::Mutex::new(GpuClockCalibration {
    cpu_anchor_ns: 0,
    gpu_anchor_ticks: 0,
    ns_per_tick: 0.0,
    calibrated: false,
});

/// Start a new capture session. Only one session may be active at a time;
/// starting a second one while one is already running is a explicit error,
/// never a silent implicit restart of the previous session.
pub fn start_capture(options: CaptureOptions) -> Result<CaptureHandle, AlreadyCapturingError> {
    if CAPTURE_ENABLED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return Err(AlreadyCapturingError);
    }

    let state = Arc::new(CaptureState::new(options));
    *ACTIVE_CAPTURE.lock() = Some(state.clone());
    Ok(CaptureHandle { state })
}

/// Whether a capture is currently recording. Cheap (`Ordering::Relaxed` atomic
/// load); call sites that need to do non-trivial attribution work (e.g.
/// hashing a `GlobalElementId`) should check this before doing that work,
/// rather than doing it unconditionally inside `enter_span`.
pub fn capture_enabled() -> bool {
    CAPTURE_ENABLED.load(Ordering::Relaxed)
}

/// Current session's GPU clock calibration, or a zeroed/`calibrated: false`
/// value if no GPU-capturing session has ever run. Used by
/// `present_synced_traced` to let embedding apps put their own wgpu
/// submissions on the same unified timeline.
pub fn gpu_calibration() -> GpuClockCalibration {
    *GPU_CALIBRATION.lock()
}

pub(crate) fn set_gpu_calibration(calibration: GpuClockCalibration) {
    *GPU_CALIBRATION.lock() = calibration;
}

/// Whether the active capture (if any) wants GPU spans. `flamegraph_gpu` uses
/// this to decide whether to lazily allocate its `QuerySet`s at all.
pub(crate) fn active_capture_wants_gpu() -> bool {
    ACTIVE_CAPTURE
        .lock()
        .as_ref()
        .map(|state| state.capture_gpu)
        .unwrap_or(false)
}

/// The most recently opened CPU-side frame index, used by `flamegraph_gpu` to
/// tag a `GpuQueryManager` generation with the frame it belongs to. See the
/// doc comment on `CaptureState::last_opened_frame_index` for why this is
/// safe to read this way instead of threading a frame index through
/// `PlatformWindow::draw`.
pub(crate) fn current_gpu_correlation_frame_index() -> Option<u64> {
    ACTIVE_CAPTURE.lock().as_ref().and_then(|state| {
        let index = state.last_opened_frame_index.load(Ordering::Acquire);
        (index != u64::MAX).then_some(index)
    })
}

/// CPU wall-clock "now," anchor-relative in the same nanosecond timeline as
/// every other timestamp in a `Capture` (`frame_start_ns`, `frame_end_ns`,
/// and -- after calibration -- `GpuSpan::start_ns`). `flamegraph_gpu` uses
/// this to stamp `queue.submit()`/readback-observed instants so they're
/// directly comparable to calibrated GPU timestamps on one shared timeline,
/// the same way `current_gpu_correlation_frame_index` lets it tag spans by
/// frame without a parameter threaded through `PlatformWindow::draw`. `None`
/// when no capture is active.
pub(crate) fn anchor_relative_now_ns() -> Option<u64> {
    ACTIVE_CAPTURE.lock().as_ref().map(|state| state.now_ns())
}

/// Attach resolved GPU spans to the frame they belong to, if that frame is
/// still held by the active capture's ring buffer (it may have been evicted
/// already, given GPU readback latency of 1-2 frames). `submit_cpu_ns` is the
/// CPU-observed instant `queue.submit()` returned for this frame's GPU work;
/// `fence_observed_cpu_ns` is the CPU-observed instant the render thread's
/// non-blocking poll first saw that work as complete. Both are `None` if
/// unavailable (e.g. this generation was reset before submission), and both
/// are CPU-side *observations*, not the GPU's own execution timing --
/// see the doc comments on `FrameCapture`'s matching fields for why that
/// distinction matters and how it differs from `gpu_spans`' calibrated
/// (inferred, not directly observed) start/end times.
pub(crate) fn attach_gpu_spans(
    frame_index: u64,
    spans: Vec<GpuSpan>,
    truncated: bool,
    submit_cpu_ns: Option<u64>,
    fence_observed_cpu_ns: Option<u64>,
) {
    if let Some(state) = ACTIVE_CAPTURE.lock().as_ref() {
        let mut finished = state.finished_frames.lock();
        if let Some(frame) = finished.iter_mut().find(|frame| frame.frame_index == frame_index) {
            frame.cpu_gpu_submit_ns = submit_cpu_ns;
            frame.cpu_gpu_fence_observed_ns = fence_observed_cpu_ns;
            frame.gpu_spans = spans;
            frame.gpu_spans_finalized = true;
            frame.gpu_spans_truncated = truncated;
        }
    }
}

/// Open a new frame for CPU-side span bucketing. Returns `None` when no
/// capture is active, so callers can thread an `Option<u64>` through and skip
/// the matching `close_frame_cpu_side` call for free.
pub(crate) fn open_frame_cpu_side(window_id: u64) -> Option<u64> {
    if !capture_enabled() {
        return None;
    }
    ACTIVE_CAPTURE
        .lock()
        .as_ref()
        .map(|state| state.open_frame(window_id))
}

/// Close a frame opened with `open_frame_cpu_side`, bucketing completed spans
/// from this thread (as `cpu_spans`) and from other threads whose `start_ns`
/// falls in this frame's window (as `background_spans`), then finalizing it
/// into the capture's ring buffer. `gpu_spans` is left empty/unfinalized;
/// `attach_gpu_spans` fills it in later.
pub(crate) fn close_frame_cpu_side(frame_index: Option<u64>) {
    let Some(frame_index) = frame_index else {
        return;
    };
    if let Some(state) = ACTIVE_CAPTURE.lock().as_ref() {
        state.close_frame(frame_index);
    }
}

struct OpenFrame {
    window_id: u64,
    frame_start_ns: u64,
}

struct CaptureState {
    anchor: Instant,
    max_frames: usize,
    capture_gpu: bool,
    next_frame_index: AtomicU64,
    /// The most recently opened frame's index, or `u64::MAX` if none has been
    /// opened yet. `flamegraph_gpu`'s `GpuQueryManager` reads this (via
    /// `current_gpu_correlation_frame_index`) to tag its own timestamp-query
    /// generation with the CPU-side frame it belongs to, rather than needing
    /// a frame index threaded through the `PlatformWindow::draw` trait. This
    /// relies on GPUI's single-foreground-thread draw model (AGENTS.md: "All
    /// use of entities and UI rendering occurs on a single foreground
    /// thread."): `Window::draw` opens a frame and, still on that thread,
    /// synchronously drives rendering down into `WgpuRenderer::draw` before
    /// any other frame can be opened.
    last_opened_frame_index: AtomicU64,
    open_frames: parking_lot::Mutex<HashMap<u64, OpenFrame>>,
    finished_frames: parking_lot::Mutex<VecDeque<FrameCapture>>,
    /// Background-thread spans drained but not yet claimed by a frame window.
    /// Partitioned into a frame's `background_spans` at `close_frame` time.
    pending_background_spans: parking_lot::Mutex<Vec<CpuSpan>>,
    /// Session-wide (not ring-buffer-windowed) frame-pacing counters. See
    /// `record_frame_pacing` and `Capture::full_draw_frame_count`.
    full_draw_frames: AtomicU64,
    fast_path_frames: AtomicU64,
}

impl CaptureState {
    fn new(options: CaptureOptions) -> Self {
        Self {
            anchor: Instant::now(),
            max_frames: options.max_frames.max(1),
            capture_gpu: options.capture_gpu,
            next_frame_index: AtomicU64::new(0),
            last_opened_frame_index: AtomicU64::new(u64::MAX),
            open_frames: parking_lot::Mutex::new(HashMap::new()),
            finished_frames: parking_lot::Mutex::new(VecDeque::new()),
            pending_background_spans: parking_lot::Mutex::new(Vec::new()),
            full_draw_frames: AtomicU64::new(0),
            fast_path_frames: AtomicU64::new(0),
        }
    }

    fn now_ns(&self) -> u64 {
        Instant::now().duration_since(self.anchor).as_nanos().min(u64::MAX as u128) as u64
    }

    fn open_frame(&self, window_id: u64) -> u64 {
        let frame_index = self.next_frame_index.fetch_add(1, Ordering::Relaxed);
        self.last_opened_frame_index.store(frame_index, Ordering::Release);
        self.open_frames.lock().insert(
            frame_index,
            OpenFrame {
                window_id,
                frame_start_ns: self.now_ns(),
            },
        );
        frame_index
    }

    fn close_frame(&self, frame_index: u64) {
        let Some(open_frame) = self.open_frames.lock().remove(&frame_index) else {
            return;
        };
        let frame_end_ns = self.now_ns();

        let cpu_spans = drain_current_thread_spans();
        collect_other_thread_spans_into(&self.pending_background_spans);
        let background_spans = {
            let mut pending = self.pending_background_spans.lock();
            let taken = std::mem::take(&mut *pending);
            let (in_window, remaining): (Vec<_>, Vec<_>) = taken.into_iter().partition(|span| {
                span.start_ns >= open_frame.frame_start_ns && span.start_ns < frame_end_ns
            });
            *pending = remaining;
            in_window
        };

        let frame = FrameCapture {
            frame_index,
            window_id: open_frame.window_id,
            cpu_spans,
            background_spans,
            gpu_spans: Vec::new(),
            gpu_spans_finalized: !self.capture_gpu,
            gpu_spans_truncated: false,
            frame_start_ns: open_frame.frame_start_ns,
            frame_end_ns,
            cpu_gpu_submit_ns: None,
            cpu_gpu_fence_observed_ns: None,
            counters: take_frame_counters(),
        };

        let mut finished = self.finished_frames.lock();
        finished.push_back(frame);
        while finished.len() > self.max_frames {
            finished.pop_front();
        }
    }

    fn finalize_open_frames(&self) {
        let remaining: Vec<u64> = self.open_frames.lock().keys().copied().collect();
        for frame_index in remaining {
            self.close_frame(frame_index);
        }
    }
}

struct PendingSpan {
    name: SpanName,
    category: SpanCategory,
    element: Option<ElementAttribution>,
    start: Instant,
    depth: u16,
}

// Per-thread completed-span budget, mirroring `profiler.rs`'s flat 20MB
// budget but scoped per-thread and smaller, since frame-count ring-buffering
// on `Capture` is now the primary bound and this buffer is just a bridge
// until the next `close_frame_cpu_side` drains it.
const THREAD_SPAN_BUDGET_BYTES: usize = 2 * 1024 * 1024;
const MAX_THREAD_SPANS: usize = THREAD_SPAN_BUDGET_BYTES / core::mem::size_of::<CpuSpan>();

type ThreadSpans = circular_buffer::CircularBuffer<MAX_THREAD_SPANS, CpuSpan>;
type GuardedThreadRecorder = spin::Mutex<ThreadRecorder>;

struct ThreadRecorder {
    thread_id: ThreadId,
    completed: Box<ThreadSpans>,
}

struct GlobalThreadRecorderEntry {
    thread_id: ThreadId,
    recorder: Weak<GuardedThreadRecorder>,
}

static GLOBAL_THREAD_RECORDERS: spin::Mutex<Vec<GlobalThreadRecorderEntry>> = spin::Mutex::new(Vec::new());

impl Drop for ThreadRecorder {
    fn drop(&mut self) {
        let mut recorders = GLOBAL_THREAD_RECORDERS.lock();
        if let Some(index) = recorders.iter().position(|entry| entry.thread_id == self.thread_id) {
            recorders.swap_remove(index);
        }
    }
}

thread_local! {
    static SPAN_STACK: RefCell<SmallVec<[PendingSpan; 32]>> = RefCell::new(SmallVec::new());
    static THREAD_RECORDER: LazyCell<Arc<GuardedThreadRecorder>> = LazyCell::new(register_thread_recorder);
}

fn register_thread_recorder() -> Arc<GuardedThreadRecorder> {
    let thread_id = std::thread::current().id();
    let recorder = Arc::new(spin::Mutex::new(ThreadRecorder {
        thread_id,
        completed: ThreadSpans::boxed(),
    }));
    GLOBAL_THREAD_RECORDERS.lock().push(GlobalThreadRecorderEntry {
        thread_id,
        recorder: Arc::downgrade(&recorder),
    });
    recorder
}

fn drain_current_thread_spans() -> Vec<CpuSpan> {
    THREAD_RECORDER.with(|recorder| {
        let mut recorder = recorder.lock();
        let (first, second) = recorder.completed.as_slices();
        let mut spans = Vec::with_capacity(first.len() + second.len());
        spans.extend_from_slice(first);
        spans.extend_from_slice(second);
        recorder.completed.clear();
        spans
    })
}

fn collect_other_thread_spans_into(pending: &parking_lot::Mutex<Vec<CpuSpan>>) {
    let current_thread_id = std::thread::current().id();
    let recorders = GLOBAL_THREAD_RECORDERS.lock();
    let mut pending = pending.lock();
    for entry in recorders.iter() {
        if entry.thread_id == current_thread_id {
            continue;
        }
        let Some(recorder) = entry.recorder.upgrade() else {
            continue;
        };
        let mut recorder = recorder.lock();
        let (first, second) = recorder.completed.as_slices();
        pending.extend_from_slice(first);
        pending.extend_from_slice(second);
        recorder.completed.clear();
    }
}

/// RAII guard returned by [`enter_span`]. Dropping it closes the span. When
/// capture is disabled at the time `enter_span` was called, this is a no-op
/// guard that costs nothing to construct or drop.
pub struct SpanGuard {
    handle: Option<()>,
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        if self.handle.take().is_none() {
            return;
        }
        let completed = SPAN_STACK.with(|stack| stack.borrow_mut().pop());
        let Some(pending) = completed else {
            return;
        };
        let Some(anchor) = active_capture_anchor() else {
            return;
        };
        let now = Instant::now();
        let start_ns = pending.start.duration_since(anchor).as_nanos().min(u64::MAX as u128) as u64;
        let duration_ns = now
            .duration_since(pending.start)
            .as_nanos()
            .min(u32::MAX as u128) as u32;

        let span = CpuSpan {
            name: pending.name,
            category: pending.category,
            depth: pending.depth,
            start_ns,
            duration_ns,
            thread_id: ThreadKey::current(),
            element: pending.element,
        };

        THREAD_RECORDER.with(|recorder| {
            recorder.lock().completed.push_back(span);
        });
    }
}

fn active_capture_anchor() -> Option<Instant> {
    ACTIVE_CAPTURE.lock().as_ref().map(|state| state.anchor)
}

/// The active capture's CPU clock anchor, in the same `Instant` used to
/// compute `CpuSpan::start_ns`. `flamegraph_gpu` uses this during calibration
/// so `GpuSpan::start_ns` lands on the same timeline as CPU spans.
pub(crate) fn capture_anchor() -> Option<Instant> {
    active_capture_anchor()
}

/// Open a new CPU span. Returns a [`SpanGuard`] whose `Drop` closes the span.
///
/// When capture is disabled, this does a single `Ordering::Relaxed` atomic
/// load and returns immediately without touching `Instant::now()`, the
/// thread-local span stack, or allocating — the entire cost of instrumenting
/// a call site in a flamegraph-enabled-but-idle build.
pub fn enter_span(name: SpanName, category: SpanCategory, element: Option<ElementAttribution>) -> SpanGuard {
    if !capture_enabled() {
        return SpanGuard { handle: None };
    }

    SPAN_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        let depth = stack.len().min(u16::MAX as usize) as u16;
        stack.push(PendingSpan {
            name,
            category,
            element,
            start: Instant::now(),
            depth,
        });
    });

    SpanGuard { handle: Some(()) }
}

/// Open a CPU span with a category of [`SpanCategory::UserDefined`] and no
/// element attribution. Intended for use both by internal call sites and by
/// embedding applications.
#[macro_export]
macro_rules! flamegraph_span {
    ($name:expr) => {
        $crate::flamegraph_span!($name, $crate::SpanCategory::UserDefined)
    };
    ($name:expr, $category:expr) => {
        let _flamegraph_span_guard = $crate::enter_span($crate::SpanName::Static($name), $category, None);
    };
}

// ---------------------------------------------------------------------------
// Phase 2: aggregate frame counters (issue #58).
//
// Draw-call/atlas/event counts are tallied into a thread-local accumulator
// rather than threaded through call sites as return values, for the same
// reason `flamegraph_gpu` correlates GPU spans to a frame index via
// `current_gpu_correlation_frame_index` instead of a parameter threaded
// through `PlatformWindow::draw`: the call sites (`WgpuRenderer::draw`'s
// `PrimitiveBatch` match arms, `WgpuAtlas::get_or_insert_with`,
// `Window::dispatch_event`, `App::notify`) have no natural way to reach the
// currently-open `FrameCapture`, and all of them run on GPUI's single
// foreground thread (AGENTS.md), so a thread-local is sufficient and avoids
// plumbing a capture handle through every one of them.
thread_local! {
    static FRAME_COUNTERS: RefCell<FrameCounters> = RefCell::new(FrameCounters::default());
}

/// Take and reset the calling thread's accumulated counters. Called once per
/// `close_frame`, from the same foreground thread that opened the frame.
fn take_frame_counters() -> FrameCounters {
    FRAME_COUNTERS.with(|counters| counters.take())
}

/// Tally one `RenderPass::draw` call for `kind`, contributing `primitives`
/// primitives. Called unconditionally from `WgpuRenderer::draw`'s
/// `PrimitiveBatch` match arms (behind `#[cfg(feature = "flamegraph")]` at
/// the call site, since this function only exists in this module); a single
/// `Ordering::Relaxed` atomic load is the entire cost when capture is
/// disabled.
pub(crate) fn record_draw_call(kind: DrawCallKind, primitives: u32) {
    if !capture_enabled() {
        return;
    }
    FRAME_COUNTERS.with(|counters| counters.borrow_mut().draw_calls.get_mut(kind).record(primitives));
}

/// Tally a new atlas tile allocation (`get_or_insert_with` cache miss that
/// produced a tile).
pub(crate) fn record_atlas_tile_allocated() {
    if !capture_enabled() {
        return;
    }
    FRAME_COUNTERS.with(|counters| counters.borrow_mut().atlas.tiles_allocated += 1);
}

/// Tally an atlas tile eviction (`PlatformAtlas::remove`).
pub(crate) fn record_atlas_tile_evicted() {
    if !capture_enabled() {
        return;
    }
    FRAME_COUNTERS.with(|counters| counters.borrow_mut().atlas.tiles_evicted += 1);
}

/// Tally an atlas `get_or_insert_with` call that found an existing tile.
pub(crate) fn record_atlas_cache_hit() {
    if !capture_enabled() {
        return;
    }
    FRAME_COUNTERS.with(|counters| counters.borrow_mut().atlas.cache_hits += 1);
}

/// Tally an atlas `get_or_insert_with` call that did not find an existing
/// tile.
pub(crate) fn record_atlas_cache_miss() {
    if !capture_enabled() {
        return;
    }
    FRAME_COUNTERS.with(|counters| counters.borrow_mut().atlas.cache_misses += 1);
}

/// Tally a `Window::dispatch_event` call.
pub(crate) fn record_input_event_dispatched() {
    if !capture_enabled() {
        return;
    }
    FRAME_COUNTERS.with(|counters| counters.borrow_mut().events.input_events_dispatched += 1);
}

/// Tally an `App::notify` call.
pub(crate) fn record_notify_call() {
    if !capture_enabled() {
        return;
    }
    FRAME_COUNTERS.with(|counters| counters.borrow_mut().events.notify_calls += 1);
}

/// Tally an entity marked dirty via `WindowInvalidator::invalidate`.
pub(crate) fn record_entity_invalidated() {
    if !capture_enabled() {
        return;
    }
    FRAME_COUNTERS.with(|counters| counters.borrow_mut().events.entities_invalidated += 1);
}

/// Tally one `Window::on_request_frame` invocation as either a full
/// compositor draw or a fast, no-compositor present-only frame. Session-wide
/// (see `Capture::full_draw_frame_count`'s doc comment for why), so this
/// updates `CaptureState` directly rather than the thread-local per-frame
/// accumulator.
pub(crate) fn record_frame_pacing(is_full_draw: bool) {
    if !capture_enabled() {
        return;
    }
    if let Some(state) = ACTIVE_CAPTURE.lock().as_ref() {
        let counter = if is_full_draw {
            &state.full_draw_frames
        } else {
            &state.fast_path_frames
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }
}

/// The wgpu surface present mode currently configured for the renderer.
/// Session-global rather than per-frame: WGPUI only reads
/// `GPUI_PRESENT_MODE`/`GPUI_DISABLE_VSYNC` once, at surface creation, so it
/// does not vary frame-to-frame today. See `set_present_mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum PresentMode {
    /// Vsync enabled (`wgpu::PresentMode::Fifo`), the default.
    #[default]
    Fifo,
    /// `wgpu::PresentMode::Mailbox`.
    Mailbox,
    /// `wgpu::PresentMode::Immediate` (`GPUI_DISABLE_VSYNC=1`).
    Immediate,
    /// Any other `wgpu::PresentMode` (e.g. `FifoRelaxed`), reported as-is
    /// without needing this module to depend on `wgpu`.
    Other,
}

impl std::fmt::Display for PresentMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            PresentMode::Fifo => "fifo",
            PresentMode::Mailbox => "mailbox",
            PresentMode::Immediate => "immediate",
            PresentMode::Other => "other",
        })
    }
}

static CURRENT_PRESENT_MODE: AtomicU8 = AtomicU8::new(0);

/// Record the renderer's current present mode. Called once from
/// `WgpuRenderer::new`, unconditionally (not gated on `capture_enabled`):
/// it's a single relaxed atomic store, rare (once per renderer/surface
/// creation), and needs to be visible to a capture that starts later in the
/// session.
pub(crate) fn set_present_mode(mode: PresentMode) {
    CURRENT_PRESENT_MODE.store(mode as u8, Ordering::Relaxed);
}

/// The most recently recorded present mode, or [`PresentMode::Fifo`] (wgpu's
/// own default) if `set_present_mode` has never been called.
pub fn current_present_mode() -> PresentMode {
    match CURRENT_PRESENT_MODE.load(Ordering::Relaxed) {
        1 => PresentMode::Mailbox,
        2 => PresentMode::Immediate,
        3 => PresentMode::Other,
        _ => PresentMode::Fifo,
    }
}

/// Mean and max of a `u32` metric over a window of frames.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct MeanMax {
    /// Arithmetic mean over the window.
    pub mean: f64,
    /// Maximum value seen in the window.
    pub max: u32,
}

fn mean_max(values: impl Iterator<Item = u32> + Clone) -> MeanMax {
    let count = values.clone().count();
    if count == 0 {
        return MeanMax::default();
    }
    let sum: u64 = values.clone().map(u64::from).sum();
    let max = values.max().unwrap_or(0);
    MeanMax {
        mean: sum as f64 / count as f64,
        max,
    }
}

/// Mean/max of a [`PassCounter`]'s two fields over a window of frames.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct PassCounterSummary {
    /// Mean/max `RenderPass::draw` call count.
    pub draw_calls: MeanMax,
    /// Mean/max primitive count.
    pub primitives: MeanMax,
}

fn pass_counter_summary(
    frames: &VecDeque<FrameCapture>,
    select: impl Fn(&DrawCallCounters) -> PassCounter,
) -> PassCounterSummary {
    PassCounterSummary {
        draw_calls: mean_max(frames.iter().map(|frame| select(&frame.counters.draw_calls).draw_calls)),
        primitives: mean_max(frames.iter().map(|frame| select(&frame.counters.draw_calls).primitives)),
    }
}

/// Mean/max draw-call/primitive counts per `PrimitiveBatch` kind, over a
/// window of frames.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct DrawCallSummary {
    /// `PrimitiveBatch::Quads`.
    pub quads: PassCounterSummary,
    /// `PrimitiveBatch::Shadows`.
    pub shadows: PassCounterSummary,
    /// `PrimitiveBatch::MonochromeSprites`.
    pub mono_sprites: PassCounterSummary,
    /// `PrimitiveBatch::PolychromeSprites`.
    pub poly_sprites: PassCounterSummary,
    /// `PrimitiveBatch::Paths`.
    pub paths: PassCounterSummary,
    /// `PrimitiveBatch::Underlines`.
    pub underlines: PassCounterSummary,
    /// `PrimitiveBatch::BackdropFilters`.
    pub backdrop_filters: PassCounterSummary,
    /// `PrimitiveBatch::Surfaces`.
    pub surfaces: PassCounterSummary,
}

/// Mean/max atlas activity, plus a session-window cache hit rate, over a
/// window of frames.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct AtlasSummary {
    /// Mean/max new tile allocations per frame.
    pub tiles_allocated: MeanMax,
    /// Mean/max tile evictions per frame.
    pub tiles_evicted: MeanMax,
    /// Mean/max atlas cache hits per frame.
    pub cache_hits: MeanMax,
    /// Mean/max atlas cache misses per frame.
    pub cache_misses: MeanMax,
    /// `sum(cache_hits) / sum(cache_hits + cache_misses)` over the window, or
    /// `0.0` if there were no lookups at all. Computed from totals rather
    /// than as a mean of per-frame ratios, since most frames have zero
    /// lookups and would otherwise dilute the average toward zero.
    pub cache_hit_rate: f64,
}

/// Mean/max input/notification activity, over a window of frames.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct EventSummary {
    /// Mean/max `Window::dispatch_event` calls per frame.
    pub input_events_dispatched: MeanMax,
    /// Mean/max `App::notify` calls per frame.
    pub notify_calls: MeanMax,
    /// Mean/max entities invalidated per frame.
    pub entities_invalidated: MeanMax,
}

/// Mean/max CPU-to-GPU timeline correlation over a window of frames -- when
/// the CPU asked the GPU to run a frame's work vs. when the GPU actually
/// started (backlog), and when the GPU actually finished vs. when the CPU
/// found out (readback/polling latency). See `FrameCapture::cpu_gpu_submit_ns`
/// and `cpu_gpu_fence_observed_ns`'s doc comments for the underlying
/// distinction this reports on: a CPU-side *observation* of submit/fence,
/// not the GPU's own (calibrated, inferred) execution timing, which is what
/// `gpu_spans` already gives you.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct GpuTimelineSummary {
    /// Mean/max nanoseconds from `cpu_gpu_submit_ns` (CPU asked the GPU to
    /// run this frame) to the calibrated start of the frame's first GPU
    /// span (GPU actually began running it). A growing value here means the
    /// GPU queue is backlogged relative to CPU submission -- not that
    /// individual passes got slower.
    pub submit_to_gpu_start: MeanMax,
    /// Mean/max nanoseconds from the calibrated end of a frame's last GPU
    /// span (GPU actually finished) to `cpu_gpu_fence_observed_ns` (CPU found
    /// out). This is readback/polling latency, not GPU execution time --
    /// expect it to sit around 1-2 frame durations by design (see
    /// `FrameCapture::gpu_spans_finalized`'s doc comment).
    pub gpu_end_to_fence_observed: MeanMax,
    /// How many frames in the window had everything needed for the two
    /// stats above: both CPU-observed instants present, and `gpu_spans`
    /// finalized and non-empty. GPU data lags CPU data by design (calibration
    /// only runs once GPU capture is requested, and readback itself lags
    /// 1-2 frames), so this is very often smaller than `frame_count`.
    pub samples: usize,
}

fn gpu_timeline_summary(frames: &VecDeque<FrameCapture>) -> GpuTimelineSummary {
    let mut submit_to_start = Vec::new();
    let mut end_to_observed = Vec::new();

    for frame in frames {
        if !frame.gpu_spans_finalized || frame.gpu_spans.is_empty() {
            continue;
        }
        let Some(submit_ns) = frame.cpu_gpu_submit_ns else {
            continue;
        };
        let Some(observed_ns) = frame.cpu_gpu_fence_observed_ns else {
            continue;
        };

        let gpu_start_ns = frame.gpu_spans.iter().map(|span| span.start_ns).min();
        let gpu_end_ns = frame
            .gpu_spans
            .iter()
            .map(|span| span.start_ns.saturating_add(span.duration_ns as u64))
            .max();
        let (Some(gpu_start_ns), Some(gpu_end_ns)) = (gpu_start_ns, gpu_end_ns) else {
            continue;
        };

        submit_to_start.push(gpu_start_ns.saturating_sub(submit_ns).min(u32::MAX as u64) as u32);
        end_to_observed.push(observed_ns.saturating_sub(gpu_end_ns).min(u32::MAX as u64) as u32);
    }

    let samples = submit_to_start.len();
    GpuTimelineSummary {
        submit_to_gpu_start: mean_max(submit_to_start.into_iter()),
        gpu_end_to_fence_observed: mean_max(end_to_observed.into_iter()),
        samples,
    }
}

/// Aggregated statistics over the frames currently held in a [`Capture`]'s
/// ring buffer, from [`Capture::counter_summary`]. Replaces what the
/// reverted `render_stats` module reported via a periodic stderr dump with a
/// queryable API: a future viewer (phase 7) can read it directly, or an
/// embedding app can print it itself (see the `Display` impl below) as a
/// thin convenience wrapper.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct CounterSummary {
    /// Frames currently held in the ring buffer this summary was computed
    /// over (i.e. `Capture::frame_count()` at the time of the call).
    pub frame_count: usize,
    /// Frames per second, computed from the time span between the first and
    /// last frame in the window (not from `1.0 / mean_frame_duration_ms`,
    /// which would ignore idle time between draws).
    pub fps: f64,
    /// Mean frame (draw + present) duration in milliseconds, over the window.
    pub mean_frame_duration_ms: f64,
    /// Max frame (draw + present) duration in milliseconds, over the window.
    pub max_frame_duration_ms: f64,
    /// Per-`PrimitiveBatch`-kind draw-call/primitive statistics.
    pub draw_calls: DrawCallSummary,
    /// Atlas tile allocator statistics.
    pub atlas: AtlasSummary,
    /// Input/notification statistics.
    pub events: EventSummary,
    /// CPU-observed GPU submit/fence timeline correlation statistics.
    pub gpu_timeline: GpuTimelineSummary,
    /// The renderer's current present mode.
    pub present_mode: PresentMode,
    /// See `Capture::full_draw_frame_count`'s doc comment: these two are
    /// session-wide totals, not bounded to the ring-buffer window like
    /// everything else in this struct.
    pub full_draw_frame_count: u64,
    /// See `Capture::fast_path_frame_count`'s doc comment.
    pub fast_path_frame_count: u64,
}

impl std::fmt::Display for CounterSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "=== WGPUI frame counters ({} frames, {:.1} fps, present={}) ===",
            self.frame_count, self.fps, self.present_mode
        )?;
        writeln!(
            f,
            "frame duration: mean {:.3}ms max {:.3}ms",
            self.mean_frame_duration_ms, self.max_frame_duration_ms
        )?;
        let total_paced = self.full_draw_frame_count + self.fast_path_frame_count;
        if total_paced > 0 {
            writeln!(
                f,
                "frame pacing: {} full-draw, {} fast-path ({:.1}% full-draw)",
                self.full_draw_frame_count,
                self.fast_path_frame_count,
                100.0 * self.full_draw_frame_count as f64 / total_paced as f64
            )?;
        }
        writeln!(f, "--- draw calls (mean/max primitives, mean/max draw calls) ---")?;
        for (name, pass) in [
            ("quads", &self.draw_calls.quads),
            ("shadows", &self.draw_calls.shadows),
            ("mono_sprites", &self.draw_calls.mono_sprites),
            ("poly_sprites", &self.draw_calls.poly_sprites),
            ("paths", &self.draw_calls.paths),
            ("underlines", &self.draw_calls.underlines),
            ("backdrop_filters", &self.draw_calls.backdrop_filters),
            ("surfaces", &self.draw_calls.surfaces),
        ] {
            writeln!(
                f,
                "{:<18} primitives {:>8.1}/{:<6} draw calls {:>6.1}/{:<6}",
                name, pass.primitives.mean, pass.primitives.max, pass.draw_calls.mean, pass.draw_calls.max
            )?;
        }
        writeln!(
            f,
            "--- atlas: allocated {:.1}/{} evicted {:.1}/{} hit-rate {:.1}% ---",
            self.atlas.tiles_allocated.mean,
            self.atlas.tiles_allocated.max,
            self.atlas.tiles_evicted.mean,
            self.atlas.tiles_evicted.max,
            self.atlas.cache_hit_rate * 100.0
        )?;
        writeln!(
            f,
            "--- events: input {:.1}/{} notify {:.1}/{} invalidated {:.1}/{} ---",
            self.events.input_events_dispatched.mean,
            self.events.input_events_dispatched.max,
            self.events.notify_calls.mean,
            self.events.notify_calls.max,
            self.events.entities_invalidated.mean,
            self.events.entities_invalidated.max,
        )?;
        if self.gpu_timeline.samples > 0 {
            writeln!(
                f,
                "--- gpu timeline ({} samples): submit->gpu-start {:.3}/{:.3}ms  gpu-end->cpu-observed {:.3}/{:.3}ms ---",
                self.gpu_timeline.samples,
                self.gpu_timeline.submit_to_gpu_start.mean / 1.0e6,
                self.gpu_timeline.submit_to_gpu_start.max as f64 / 1.0e6,
                self.gpu_timeline.gpu_end_to_fence_observed.mean / 1.0e6,
                self.gpu_timeline.gpu_end_to_fence_observed.max as f64 / 1.0e6,
            )
        } else {
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 3: on-demand memory snapshots (issue #59).
//
// Unlike the spans/counters above, memory sizes are not tracked continuously
// per-frame -- they're cheap to recompute from the subsystems that already
// own the data (a wgpu buffer already knows its own `size()`, an `Arena`
// already knows its own `capacity()`), so there's nothing to accumulate.
// `MemorySnapshot`/`GpuMemorySnapshot` are plain data, computed and returned
// fresh on every call.
//
// #59 suggested exposing this as `Capture::memory_snapshot()`. That doesn't
// fit the real merged shape: a stopped `Capture` (this module's only
// `Capture` type so far, see its doc comment) has no reference back to the
// live `App`/`Window`/`WgpuRenderer` state the CPU/GPU subsystems actually
// live in -- `element_arena` is on `App`, the glyph cache is on the
// process-wide `TextSystem`, the shaped-line cache and GPU renderer are
// per-`Window`. Forcing the query through `Capture` would mean threading all
// of that state into a type whose only job today is holding already-finished
// frames. Instead, each snapshot is computed where its data actually lives:
// `Window::memory_snapshot` (CPU) and `Window::gpu_memory_snapshot` (GPU),
// both `#[cfg(feature = "flamegraph")]`-gated same as everything else here.
// This mirrors, rather than fights, the same lesson phase 1/2 already
// learned about call sites with no natural path to a specific type --
// `current_gpu_correlation_frame_index` and the `FRAME_COUNTERS`
// thread-local accumulator both exist because the natural owner of the data
// (a `GpuQueryManager` generation, a draw-call counter) isn't the same place
// that has a handle to the `Capture`/`FrameCapture` it needs to reach.
// `MemorySnapshot`/`GpuMemorySnapshot` face the mirror-image version of that
// problem -- the natural owner of a `Capture` doesn't have a handle to the
// subsystems -- so they get the mirror-image fix: compute at the source
// instead of routing through `Capture`.

/// On-demand snapshot of memory held by WGPUI's own CPU-side subsystems for
/// one `Window` (Phase 3 of the profiling epic, issue #59). This is
/// explicitly not a general-purpose heap profiler -- it reports sizes of
/// known allocating subsystems (element arena, text caches, image cache, and
/// the flamegraph engine's own buffers) rather than intercepting every
/// allocation. See [`Window::memory_snapshot`](crate::Window::memory_snapshot).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct MemorySnapshot {
    /// Reserved capacity of the per-frame element arena (`Window::draw`'s
    /// `ElementArenaScope`). See `App::element_arena_capacity_bytes`.
    pub element_arena_bytes: u64,
    /// Text system cache sizes. See [`TextSystemMemory`].
    pub text_system: TextSystemMemory,
    /// Sum of every registered `RetainAllImageCache`'s currently-loaded image
    /// bytes. Custom `ImageCache` implementations are not registered and are
    /// therefore invisible here -- this profiler only has visibility into
    /// WGPUI's own built-in cache, not arbitrary embedder-provided ones.
    pub image_cache_bytes: u64,
    /// The flamegraph capture engine's own live footprint: per-thread
    /// completed-span ring buffers, plus (if a capture is currently running)
    /// the frames and pending background spans it's holding. Zero until the
    /// first span is ever recorded on any thread, consistent with this
    /// module's zero-overhead-when-idle rule. See
    /// `capture_engine_memory_usage`.
    pub capture_engine_bytes: u64,
}

impl MemorySnapshot {
    /// Sum of every field above: one headline "how much is WGPUI holding
    /// onto for this window" number.
    pub fn total_bytes(&self) -> u64 {
        self.element_arena_bytes
            + self.text_system.total_bytes()
            + self.image_cache_bytes
            + self.capture_engine_bytes
    }
}

/// Text system cache sizes, part of [`MemorySnapshot`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct TextSystemMemory {
    /// The rasterized-glyph bitmap cache (cosmic-text's `SwashCache`),
    /// shared by every window in the `App` (one `PlatformTextSystem` per
    /// `App`).
    pub glyph_cache_bytes: u64,
    /// This window's shaped-line layout cache (`LineLayoutCache`), covering
    /// both the current and the previous frame's retained entries -- the
    /// cache deliberately keeps one frame of history so lines that didn't
    /// change survive a redraw without being re-shaped.
    pub shaped_line_cache_bytes: u64,
}

impl TextSystemMemory {
    /// Sum of both fields above.
    pub fn total_bytes(&self) -> u64 {
        self.glyph_cache_bytes + self.shaped_line_cache_bytes
    }
}

/// On-demand snapshot of GPU memory held by one window's renderer (Phase 3 of
/// the profiling epic, issue #59). Mostly summing sizes that already exist on
/// wgpu resources (`wgpu::Buffer::size`, texture dimensions) rather than any
/// new tracking. See
/// [`Window::gpu_memory_snapshot`](crate::Window::gpu_memory_snapshot).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct GpuMemorySnapshot {
    /// Fixed-size buffers in `render_context.rs` (quads/shadows/sprites/
    /// paths/underlines/backdrop_filters/globals/color_adjustments), summed
    /// from each buffer's own `wgpu::Buffer::size()`.
    pub fixed_buffer_bytes: u64,
    /// Atlas texture memory (`atlas.rs`), monochrome and polychrome
    /// textures combined.
    pub atlas_bytes: u64,
    /// `SurfaceRegistry`'s triple-buffered offscreen textures, across every
    /// registered surface.
    pub surface_registry_bytes: u64,
    /// The swapchain's own backing textures, computed from
    /// `surface_configuration`'s dimensions/format -- wgpu doesn't expose
    /// the presentation engine's actual image count, so this uses
    /// `desired_maximum_frame_latency` as a best-effort estimate of how many
    /// images it's holding.
    pub swapchain_bytes: u64,
}

impl GpuMemorySnapshot {
    /// Sum of every field above.
    pub fn total_bytes(&self) -> u64 {
        self.fixed_buffer_bytes + self.atlas_bytes + self.surface_registry_bytes + self.swapchain_bytes
    }
}

/// The flamegraph capture engine's own live CPU memory footprint: every
/// thread's completed-span ring buffer capacity (`THREAD_SPAN_BUDGET_BYTES`
/// each, for every thread that has ever recorded a span), plus -- if a
/// capture is currently running -- the frames its ring buffer is holding and
/// any background spans awaiting the next frame close. Cheap: no allocation,
/// just reading `Vec`/`VecDeque` lengths already held behind locks this
/// module takes elsewhere. Used by [`MemorySnapshot::capture_engine_bytes`].
pub(crate) fn capture_engine_memory_usage() -> u64 {
    let thread_recorder_bytes =
        (GLOBAL_THREAD_RECORDERS.lock().len() as u64) * (THREAD_SPAN_BUDGET_BYTES as u64);

    let active_capture_bytes = ACTIVE_CAPTURE.lock().as_ref().map_or(0, |state| {
        let pending_bytes = (state.pending_background_spans.lock().len() as u64)
            * (core::mem::size_of::<CpuSpan>() as u64);
        let frames_bytes: u64 = state.finished_frames.lock().iter().map(frame_capture_memory_usage).sum();
        pending_bytes + frames_bytes
    });

    thread_recorder_bytes + active_capture_bytes
}

/// Approximate heap bytes held by one [`FrameCapture`]'s span vectors, plus a
/// per-frame fixed-field allowance. Shared by `capture_engine_memory_usage`
/// (a still-running capture's currently-buffered frames) and
/// `Capture::retained_trace_bytes` (a stopped capture's frames).
fn frame_capture_memory_usage(frame: &FrameCapture) -> u64 {
    let cpu_span_bytes = ((frame.cpu_spans.len() + frame.background_spans.len()) as u64)
        * (core::mem::size_of::<CpuSpan>() as u64);
    let gpu_span_bytes = (frame.gpu_spans.len() as u64) * (core::mem::size_of::<GpuSpan>() as u64);
    core::mem::size_of::<FrameCapture>() as u64 + cpu_span_bytes + gpu_span_bytes
}

// ---------------------------------------------------------------------------
// Phase 4: on-demand GPU deep capture (issue #60).
//
// Everything above this point is the phase 1-3 always-on-capable path: cheap
// enough (a handful of relaxed atomic loads plus, while a session is
// recording, some bookkeeping) to leave instrumented in a shipping build.
// `DeepCapture` is deliberately not part of that path. Reading back full
// buffer contents is orders of magnitude more expensive than the timestamp
// pairs `flamegraph_gpu`'s `GpuQueryManager` resolves every frame, so this
// models a completely separate, single-shot mode: armed explicitly with
// `request_deep_capture()`, fires once on the very next `WgpuRenderer::draw()`
// call, and is torn down (its wgpu staging buffers dropped) the moment
// readback completes. There is no persistent state here beyond one
// `AtomicBool` (armed?) and one `Mutex<Option<DeepCapture>>` (last completed
// result) -- the expensive state (staging buffers, in-flight command stream)
// lives entirely in `flamegraph_gpu::DeepCaptureRecorder` /
// `DeepCapturePendingReadback`, scoped to `WgpuRenderer` and only allocated
// while a capture is actually in flight.
//
// Resource contents are read back once per touched fixed buffer for the
// whole captured frame, not once per draw call boundary as a literal reading
// of "resource contents at time of use" might suggest. This is intentional:
// every fixed buffer in `WgpuContext` (`quads_buffer`, `shadows_buffer`, ...)
// is written exactly once per frame, with the *entire* frame's data for that
// primitive kind, before any draw call reads from it (see
// `WgpuRenderer::draw`'s `ensure_buffer_size`/`write_buffer` calls, which all
// happen before the render-pass loop). So a buffer's contents are already
// identical at every draw call boundary that reads it within one frame --
// reading it back per call would mean N redundant `map_async` round trips of
// the same bytes for N draw calls sharing that buffer, for zero additional
// information. Recording each draw call's `vertex_range`/`instance_range`
// alongside the once-per-buffer snapshot is exactly as informative (a viewer
// slices the snapshot using those ranges) at a fraction of the readback cost.
//
// Same "no natural path to the data" constraint phases 1-3 kept hitting
// applies here too: `WgpuContext`'s fixed buffers are `pub(super)` to
// `platform::cross`, invisible to this module and to `flamegraph_gpu.rs`
// (neither lives inside that module). Rather than widening that visibility,
// the buffer-copy commands are recorded from `WgpuRenderer::draw` itself
// (which already holds live `&wgpu::Buffer` references for its own bind-group
// setup), and only the resulting `wgpu::Buffer` staging handles/state machine
// live in `flamegraph_gpu.rs`.

/// Which of `WgpuContext`'s fixed-size resource buffers (`src/platform/cross/
/// render_context.rs`) fed a [`DeepCaptureDrawCall`]. `Surfaces` batches build
/// a fresh per-surface uniform buffer on every call rather than reading one of
/// these shared buffers, so `DeepCaptureDrawCall::buffer_kind` is `None` for
/// them -- there is deliberately no `Surfaces` variant here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeepCaptureBufferKind {
    /// `WgpuContext::quads_buffer`.
    Quads,
    /// `WgpuContext::shadows_buffer`.
    Shadows,
    /// `WgpuContext::underlines_buffer`.
    Underlines,
    /// `WgpuContext::mono_sprites_buffer`.
    MonoSprites,
    /// `WgpuContext::poly_sprites_buffer`.
    PolySprites,
    /// `WgpuContext::backdrop_filters_buffer`.
    BackdropFilters,
    /// `WgpuContext::paths_vertices_buffer`.
    Paths,
}

/// One entry in a [`DeepCapture`]'s command stream: full identifying detail
/// for a single `RenderPass::draw` call recorded during the one triggered
/// frame, in submission order. Walks the same `PrimitiveBatch` match arms in
/// `WgpuRenderer::draw` that phase 2's `record_draw_call` already tallies
/// counts from (see `DrawCallCounters`) -- this is the same call sites,
/// recording identifying detail instead of just a running total.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeepCaptureDrawCall {
    /// Position in the command stream, 0-based, in submission order.
    pub sequence: u32,
    /// Which `PrimitiveBatch` kind this call issued.
    pub kind: DrawCallKind,
    /// The `wgpu::RenderPipeline` label bound for this call (see
    /// `WgpuPipelines` in `renderer.rs`, e.g. `"quads"`, `"mono_sprites"`).
    /// Doubles as shader identity per the issue's ask: each `DrawCallKind`
    /// maps 1:1 to exactly one pipeline/shader module in `WgpuRenderer::draw`,
    /// so the pipeline label already uniquely identifies the shader.
    pub pipeline_label: &'static str,
    /// The render pass this call was recorded into (`"main"`,
    /// `"main_resumed"`, `"filter_group"`, or `"filter_group_resumed"` --
    /// see `GpuPassKind`, which this mirrors).
    pub pass_label: &'static str,
    /// Vertex range passed to `RenderPass::draw` (start, end).
    pub vertex_range: (u32, u32),
    /// Instance range passed to `RenderPass::draw` (start, end).
    pub instance_range: (u32, u32),
    /// Number of bind groups set for this call (2-4 depending on kind).
    pub bind_group_count: u32,
    /// Which fixed resource buffer (if any) this call's instance/vertex data
    /// came from; see [`DeepCaptureBufferKind`]'s doc comment for why
    /// `Surfaces` calls are always `None` here.
    pub buffer_kind: Option<DeepCaptureBufferKind>,
    /// `(AtlasTextureKind as u64) << 32 | AtlasTextureId::index`, for sprite
    /// calls (`MonoSprites`/`PolySprites`); `None` for every other kind.
    pub atlas_texture_id: Option<u64>,
    /// `SurfaceId`'s inner id (`platform/cross/surface_registry.rs`), for
    /// `Surfaces` calls; `None` for every other kind (Phase 4b, issue #72 --
    /// the surface-identity counterpart to `atlas_texture_id` above, added
    /// alongside real texture-content readback since without it a replay
    /// has no way to know *which* touched surface's content a given
    /// `Surfaces` draw call actually composited).
    pub surface_id: Option<u64>,
}

/// One fixed buffer's full contents at the time of the triggered frame, part
/// of a [`DeepCapture`]. Read back once per buffer touched by that frame's
/// command stream -- see the module-level comment above this section for why
/// once-per-buffer (not once-per-draw-call) is both sufficient and far
/// cheaper.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeepCaptureBufferContents {
    /// Which buffer this is.
    pub kind: DeepCaptureBufferKind,
    /// Raw bytes copied back from the GPU buffer, from offset 0 through the
    /// buffer's full `wgpu::Buffer::size()` (which may include unwritten
    /// tail bytes past what this frame actually wrote -- a viewer should
    /// slice using the `vertex_range`/`instance_range` recorded on the
    /// [`DeepCaptureDrawCall`]s that reference this buffer, not assume the
    /// whole slice is meaningful).
    pub bytes: Vec<u8>,
}

/// Identifies one texture a [`DeepCapture`] read back the pixel contents of
/// (Phase 4b of the profiling epic, issue #72) -- either one atlas texture
/// page or one composited surface's currently displayed triple-buffer
/// texture. The texture-content counterpart to [`DeepCaptureBufferKind`],
/// but unlike the seven fixed buffers (a small, statically-known set),
/// atlas pages and surfaces are both allocated dynamically at runtime
/// (`WgpuAtlas`/`SurfaceRegistry`, `platform/cross/atlas.rs`/
/// `surface_registry.rs`), so each variant carries the underlying dynamic
/// id rather than this being itself a fixed-size enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeepCaptureTextureId {
    /// One atlas texture page, encoded exactly the same way
    /// [`DeepCaptureDrawCall::atlas_texture_id`] already is:
    /// `(AtlasTextureKind as u64) << 32 | AtlasTextureId::index`.
    Atlas(u64),
    /// One composited surface -- `SurfaceId`'s inner id
    /// (`platform/cross/surface_registry.rs`).
    Surface(u64),
}

/// One texture's full pixel contents at the time of the triggered frame,
/// part of a [`DeepCapture`] -- the texture-content counterpart to
/// [`DeepCaptureBufferContents`] (see that type's doc comment for the
/// "recorded once per touched resource, not once per draw call" rationale,
/// which applies here identically). `bytes` is tightly packed with no
/// `wgpu::COPY_BYTES_PER_ROW_ALIGNMENT` row padding -- that padding exists
/// only as a GPU-side copy constraint and is stripped during readback
/// before it ever reaches this type, the same way
/// [`crate::flamegraph_replay`]'s GPU replay already strips it off a render
/// target before handing pixels back to a caller.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeepCaptureTextureContents {
    /// Which texture this is.
    pub id: DeepCaptureTextureId,
    /// Texture width, in pixels.
    pub width: u32,
    /// Texture height, in pixels.
    pub height: u32,
    /// Bytes per pixel -- `1` for the atlas's monochrome (`R8Unorm`) pages,
    /// `4` for its polychrome (`Rgba8Unorm`) pages, and whatever a given
    /// surface's own `wgpu::TextureFormat` requires. Computed from the real
    /// format via `wgpu::TextureFormat::block_copy_size` at readback time,
    /// never assumed, since surfaces may be created with any format a
    /// caller of `create_wgpu_surface` chooses.
    pub bytes_per_pixel: u32,
    /// Tightly packed pixel bytes, `width * height * bytes_per_pixel`
    /// bytes, row-major from the top-left.
    pub bytes: Vec<u8>,
}

/// One on-demand, single-frame "deep capture" (Phase 4 of the profiling
/// epic, issue #60): the full command stream and fixed-buffer resource
/// contents for exactly one triggered `WgpuRenderer::draw()` call. See the
/// module-level comment above this section for the full design rationale and
/// how this differs from phase 1's always-on [`Capture`].
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct DeepCapture {
    /// Every `RenderPass::draw` call issued during the captured frame, in
    /// submission order.
    pub draw_calls: Vec<DeepCaptureDrawCall>,
    /// Full contents of each fixed buffer touched by at least one entry in
    /// `draw_calls`, one entry per distinct [`DeepCaptureBufferKind`] seen.
    pub buffer_contents: Vec<DeepCaptureBufferContents>,
    /// Full pixel contents of each atlas texture page/surface touched by at
    /// least one entry in `draw_calls`, one entry per distinct
    /// [`DeepCaptureTextureId`] seen (Phase 4b, issue #72). May be empty
    /// even when `draw_calls` references atlas/surface content, if that
    /// texture's readback did not complete -- see `resources_finalized`.
    pub texture_contents: Vec<DeepCaptureTextureContents>,
    /// Whether every buffer/texture readback that was attempted actually
    /// completed successfully. `true` for an empty `draw_calls`/
    /// `buffer_contents`/`texture_contents` (there was nothing to read
    /// back). `false` if any individual buffer's or texture's `map_async`
    /// reported an error -- that resource is simply missing from
    /// `buffer_contents`/`texture_contents` rather than the whole capture
    /// being discarded, so a partial result is still usable.
    pub resources_finalized: bool,
}

impl DeepCapture {
    /// The buffer contents recorded for `kind`, if that buffer was touched by
    /// this capture's command stream and its readback completed.
    pub fn buffer_contents(&self, kind: DeepCaptureBufferKind) -> Option<&DeepCaptureBufferContents> {
        self.buffer_contents.iter().find(|contents| contents.kind == kind)
    }

    /// The texture contents recorded for `id`, if that atlas page/surface
    /// was touched by this capture's command stream and its readback
    /// completed.
    pub fn texture_contents(&self, id: DeepCaptureTextureId) -> Option<&DeepCaptureTextureContents> {
        self.texture_contents.iter().find(|contents| contents.id == id)
    }
}

/// Whether a deep capture has been armed (via [`request_deep_capture`]) but
/// not yet consumed by a `WgpuRenderer::draw()` call.
static DEEP_CAPTURE_REQUESTED: AtomicBool = AtomicBool::new(false);

/// The most recently completed deep capture, if any and if it has not
/// already been taken by [`take_completed_deep_capture`].
static COMPLETED_DEEP_CAPTURE: parking_lot::Mutex<Option<DeepCapture>> = parking_lot::Mutex::new(None);

/// Arm a one-shot deep GPU capture: full command stream plus fixed-buffer
/// resource contents for the very next `WgpuRenderer::draw()` call. Distinct
/// from [`start_capture`]'s always-on session -- see this module's Phase 4
/// section doc comment for why full resource readback needs its own,
/// far-more-expensive, non-persistent path. Independent of whether a phase
/// 1-3 capture session is currently running: a deep capture can be requested
/// with or without one active.
///
/// Calling this while a deep capture is already armed or actively being read
/// back is a no-op; the in-flight one is left to finish. Check
/// [`deep_capture_requested`]/[`take_completed_deep_capture`] if that
/// distinction matters to the caller.
pub fn request_deep_capture() {
    DEEP_CAPTURE_REQUESTED.store(true, Ordering::Release);
}

/// Whether a deep capture has been requested but not yet started recording.
/// Does not reflect a capture that has started recording but not finished
/// readback -- there is no public "in progress" signal this round, only
/// "requested" and "the last completed result" (see
/// [`take_completed_deep_capture`]).
pub fn deep_capture_requested() -> bool {
    DEEP_CAPTURE_REQUESTED.load(Ordering::Acquire)
}

/// Consume the pending request, if any, so at most one `WgpuRenderer::draw()`
/// call starts recording per `request_deep_capture()` call.
pub(crate) fn take_deep_capture_request() -> bool {
    DEEP_CAPTURE_REQUESTED.swap(false, Ordering::AcqRel)
}

/// Publish a finished deep capture, overwriting whatever the previous one
/// left behind if it was never collected. Called by
/// `flamegraph_gpu::DeepCapturePendingReadback::poll` once every touched
/// buffer's readback has resolved (successfully or not).
pub(crate) fn complete_deep_capture(capture: DeepCapture) {
    *COMPLETED_DEEP_CAPTURE.lock() = Some(capture);
}

/// Take the most recently completed deep capture, if any, clearing it so a
/// second call returns `None` until another capture finishes.
pub fn take_completed_deep_capture() -> Option<DeepCapture> {
    COMPLETED_DEEP_CAPTURE.lock().take()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    // Owned mirror types used only to verify the wire format round-trips.
    // The live-process types above are intentionally `Serialize`-only (see
    // the module-level comment near their definitions): a real reader, in or
    // out of process, would define its own owned types like these rather
    // than deserialize directly into structs holding `&'static str`.

    #[derive(Debug, PartialEq, Deserialize)]
    enum DecodedSpanCategory {
        WindowFrame,
        ElementRequestLayout,
        ElementPrepaint,
        ElementPaint,
        BackgroundTask,
        GpuRenderPass,
        GpuSubmitPresent,
        UserDefined,
    }

    #[derive(Debug, PartialEq, Deserialize)]
    enum DecodedSpanName {
        Static(String),
        Interned(u32),
    }

    #[derive(Debug, PartialEq, Deserialize)]
    struct DecodedElementAttribution {
        type_name: String,
        global_id_hash: u64,
        source_location: Option<(String, u32)>,
    }

    #[derive(Debug, PartialEq, Deserialize)]
    struct DecodedCpuSpan {
        name: DecodedSpanName,
        category: DecodedSpanCategory,
        depth: u16,
        start_ns: u64,
        duration_ns: u32,
        thread_id: u64,
        element: Option<DecodedElementAttribution>,
    }

    #[derive(Debug, PartialEq, Deserialize)]
    enum DecodedGpuPassKind {
        Main,
        MainResumed,
        FilterGroup,
        FilterGroupResumed,
        FastSurfaceBlit,
        SubmitPresent,
    }

    #[derive(Debug, PartialEq, Deserialize)]
    struct DecodedGpuSpan {
        name: DecodedSpanName,
        start_ns: u64,
        duration_ns: u32,
        pass_kind: DecodedGpuPassKind,
        query_set_generation: u64,
    }

    #[derive(Debug, PartialEq, Deserialize)]
    struct DecodedFrameCapture {
        frame_index: u64,
        window_id: u64,
        cpu_spans: Vec<DecodedCpuSpan>,
        background_spans: Vec<DecodedCpuSpan>,
        gpu_spans: Vec<DecodedGpuSpan>,
        gpu_spans_finalized: bool,
        gpu_spans_truncated: bool,
        frame_start_ns: u64,
        frame_end_ns: u64,
        cpu_gpu_submit_ns: Option<u64>,
        cpu_gpu_fence_observed_ns: Option<u64>,
        // `FrameCounters` derives `Deserialize` directly (plain numeric
        // data, no `&'static str` fields), so it's reused as-is here rather
        // than needing its own decoded mirror type.
        counters: FrameCounters,
    }

    fn read_trace(bytes: &[u8]) -> anyhow::Result<(TraceHeader, Vec<DecodedFrameCapture>)> {
        anyhow::ensure!(
            bytes.len() >= TRACE_MAGIC.len() + 4 + core::mem::size_of::<TraceHeader>(),
            "trace too short"
        );
        anyhow::ensure!(bytes[..8] == TRACE_MAGIC, "bad magic");
        let version = u32::from_le_bytes(bytes[8..12].try_into()?);
        anyhow::ensure!(version == TRACE_FORMAT_VERSION, "unsupported trace version");

        let header_size = core::mem::size_of::<TraceHeader>();
        let header_bytes = &bytes[12..12 + header_size];
        // `header_bytes` is a slice into a `Vec<u8>` starting at a fixed but
        // arbitrary byte offset, so it isn't guaranteed to satisfy
        // `TraceHeader`'s 8-byte alignment; read it unaligned instead of
        // reinterpreting the slice in place.
        let header: TraceHeader = bytemuck::pod_read_unaligned(header_bytes);

        let mut offset = 12 + header_size;
        let mut frames = Vec::with_capacity(header.frame_count as usize);
        for _ in 0..header.frame_count {
            anyhow::ensure!(bytes.len() >= offset + 4, "truncated frame length prefix");
            let length = u32::from_le_bytes(bytes[offset..offset + 4].try_into()?) as usize;
            offset += 4;
            anyhow::ensure!(bytes.len() >= offset + length, "truncated frame body");
            let (frame, _): (DecodedFrameCapture, usize) =
                bincode::serde::decode_from_slice(&bytes[offset..offset + length], bincode::config::standard())
                    .map_err(|error| anyhow::anyhow!("failed to decode frame capture: {error}"))?;
            offset += length;
            frames.push(frame);
        }

        Ok((header, frames))
    }

    fn decode(span: &CpuSpan) -> DecodedCpuSpan {
        DecodedCpuSpan {
            name: match span.name {
                SpanName::Static(name) => DecodedSpanName::Static(name.to_string()),
                SpanName::Interned(index) => DecodedSpanName::Interned(index),
            },
            category: decode_category(span.category),
            depth: span.depth,
            start_ns: span.start_ns,
            duration_ns: span.duration_ns,
            thread_id: span.thread_id.0,
            element: span.element.map(|element| DecodedElementAttribution {
                type_name: element.type_name.to_string(),
                global_id_hash: element.global_id_hash,
                source_location: element
                    .source_location
                    .map(|(file, line)| (file.to_string(), line)),
            }),
        }
    }

    fn decode_category(category: SpanCategory) -> DecodedSpanCategory {
        match category {
            SpanCategory::WindowFrame => DecodedSpanCategory::WindowFrame,
            SpanCategory::ElementRequestLayout => DecodedSpanCategory::ElementRequestLayout,
            SpanCategory::ElementPrepaint => DecodedSpanCategory::ElementPrepaint,
            SpanCategory::ElementPaint => DecodedSpanCategory::ElementPaint,
            SpanCategory::BackgroundTask => DecodedSpanCategory::BackgroundTask,
            SpanCategory::GpuRenderPass => DecodedSpanCategory::GpuRenderPass,
            SpanCategory::GpuSubmitPresent => DecodedSpanCategory::GpuSubmitPresent,
            SpanCategory::UserDefined => DecodedSpanCategory::UserDefined,
        }
    }

    fn decode_frame(frame: &FrameCapture) -> DecodedFrameCapture {
        DecodedFrameCapture {
            frame_index: frame.frame_index,
            window_id: frame.window_id,
            cpu_spans: frame.cpu_spans.iter().map(decode).collect(),
            background_spans: frame.background_spans.iter().map(decode).collect(),
            gpu_spans: Vec::new(),
            gpu_spans_finalized: frame.gpu_spans_finalized,
            gpu_spans_truncated: frame.gpu_spans_truncated,
            frame_start_ns: frame.frame_start_ns,
            frame_end_ns: frame.frame_end_ns,
            cpu_gpu_submit_ns: frame.cpu_gpu_submit_ns,
            cpu_gpu_fence_observed_ns: frame.cpu_gpu_fence_observed_ns,
            counters: frame.counters,
        }
    }

    #[test]
    fn nesting_and_trace_round_trip() {
        let handle = start_capture(CaptureOptions {
            max_frames: 8,
            capture_gpu: false,
        })
        .expect("no other capture should be active in this test process at this point");

        for i in 0..3u32 {
            let frame_index = open_frame_cpu_side(1);
            {
                let _draw = enter_span(SpanName::Static("Window::draw"), SpanCategory::WindowFrame, None);
                {
                    let _draw_roots =
                        enter_span(SpanName::Static("Window::draw_roots"), SpanCategory::WindowFrame, None);
                }
            }

            // Phase 2 (issue #58): exercise the FrameCounters call sites the
            // same way the real instrumentation in renderer.rs/atlas.rs/
            // window.rs/app.rs does, with values that vary per frame so
            // mean/max in `counter_summary` are distinguishable from a
            // constant.
            record_draw_call(DrawCallKind::Quads, 10 + i * 5); // 10, 15, 20
            record_draw_call(DrawCallKind::Surfaces, 1);
            record_atlas_cache_hit();
            if i == 0 {
                record_atlas_cache_miss();
                record_atlas_tile_allocated();
            }
            for _ in 0..=i {
                record_input_event_dispatched();
            }
            record_notify_call();
            record_entity_invalidated();
            record_frame_pacing(true);

            close_frame_cpu_side(frame_index);
        }

        // Two fast-path (present-only) frames: these never open a
        // FrameCapture (no compositor work to attribute spans/counters to),
        // so they only show up in the session-wide pacing totals, not in any
        // individual frame's counters.
        record_frame_pacing(false);
        record_frame_pacing(false);

        let capture = handle.stop();
        assert_eq!(capture.frame_count(), 3);
        assert_eq!(capture.full_draw_frame_count, 3);
        assert_eq!(capture.fast_path_frame_count, 2);

        let frames: Vec<&FrameCapture> = capture.frames().collect();
        assert_eq!(
            frames[0].counters.draw_calls.quads,
            PassCounter { draw_calls: 1, primitives: 10 }
        );
        assert_eq!(
            frames[2].counters.draw_calls.quads,
            PassCounter { draw_calls: 1, primitives: 20 }
        );
        assert_eq!(
            frames[0].counters.draw_calls.surfaces,
            PassCounter { draw_calls: 1, primitives: 1 }
        );
        assert_eq!(frames[0].counters.atlas.tiles_allocated, 1, "frame 0 allocated a tile");
        assert_eq!(frames[1].counters.atlas.tiles_allocated, 0, "frame 1 did not allocate a tile");
        assert_eq!(frames[0].counters.atlas.cache_misses, 1);
        assert_eq!(frames[1].counters.atlas.cache_misses, 0);
        assert_eq!(frames[0].counters.atlas.cache_hits, 1);
        assert_eq!(frames[0].counters.events.input_events_dispatched, 1);
        assert_eq!(frames[2].counters.events.input_events_dispatched, 3);
        assert_eq!(frames[0].counters.events.notify_calls, 1);
        assert_eq!(frames[0].counters.events.entities_invalidated, 1);

        let frame = frames[0];
        let draw = frame
            .cpu_spans
            .iter()
            .find(|span| span.name == SpanName::Static("Window::draw"))
            .expect("Window::draw span recorded");
        let draw_roots = frame
            .cpu_spans
            .iter()
            .find(|span| span.name == SpanName::Static("Window::draw_roots"))
            .expect("Window::draw_roots span recorded");
        assert_eq!(draw.depth, 0, "Window::draw should be the outermost span");
        assert_eq!(draw_roots.depth, 1, "Window::draw_roots should nest one level under Window::draw");

        let summary = capture.counter_summary();
        assert_eq!(summary.frame_count, 3);
        assert_eq!(summary.draw_calls.quads.primitives.mean, 15.0, "(10 + 15 + 20) / 3");
        assert_eq!(summary.draw_calls.quads.primitives.max, 20);
        assert_eq!(summary.draw_calls.quads.draw_calls, MeanMax { mean: 1.0, max: 1 });
        assert_eq!(summary.atlas.tiles_allocated, MeanMax { mean: 1.0 / 3.0, max: 1 });
        assert!(
            (summary.atlas.cache_hit_rate - 0.75).abs() < 1e-9,
            "3 hits, 1 miss across the window => 0.75, got {}",
            summary.atlas.cache_hit_rate
        );
        assert_eq!(summary.events.input_events_dispatched, MeanMax { mean: 2.0, max: 3 });
        assert_eq!(summary.full_draw_frame_count, 3);
        assert_eq!(summary.fast_path_frame_count, 2);
        // Not asserted against a precise value since it depends on wall-clock
        // timing between the frames recorded above, but it should always be
        // finite and non-negative.
        assert!(summary.fps.is_finite() && summary.fps >= 0.0, "fps was {}", summary.fps);

        let mut buffer = Vec::new();
        capture.export_trace(&mut buffer).expect("export_trace should succeed");

        let (header, frames) = read_trace(&buffer).expect("round trip decode should succeed");
        assert_eq!(header.frame_count, 3);
        assert_eq!(frames.len(), 3);
        let expected: Vec<DecodedFrameCapture> = capture.frames().map(decode_frame).collect();
        assert_eq!(frames, expected);

        // Phase 3 (issue #59): the capture engine's own footprint should be
        // visible while a capture is running -- three finished frames, each
        // with the "Window::draw"/"Window::draw_roots" CPU spans recorded
        // above, are still held by `ACTIVE_CAPTURE` at this point (the
        // capture hasn't been stopped yet). `capture_engine_memory_usage` is
        // recomputed from scratch on every call rather than tracked
        // continuously, so it's safe to call here mid-capture and again
        // after `stop()` below.
        assert!(
            capture_engine_memory_usage() > 0,
            "engine footprint should be nonzero with frames still buffered and a thread-local span recorder live"
        );

        // The capture's own retained bytes should equal the sum of each
        // buffered frame's span vectors (each of the 3 frames here has 2 CPU
        // spans -- "Window::draw" and "Window::draw_roots" -- and no
        // background/GPU spans), computed independently of
        // `frame_capture_memory_usage` to actually exercise the formula
        // rather than just calling it a second time.
        let expected_retained_bytes: u64 = capture
            .frames()
            .map(|frame| {
                core::mem::size_of::<FrameCapture>() as u64
                    + ((frame.cpu_spans.len() + frame.background_spans.len()) as u64)
                        * (core::mem::size_of::<CpuSpan>() as u64)
                    + (frame.gpu_spans.len() as u64) * (core::mem::size_of::<GpuSpan>() as u64)
            })
            .sum();
        assert_eq!(capture.retained_trace_bytes(), expected_retained_bytes);
        assert!(capture.retained_trace_bytes() > 0);

        // A freshly stopped, empty capture should report zero retained
        // bytes -- there's nothing buffered to hold onto. Sequential with
        // (not concurrent with) the capture above: only one flamegraph
        // capture session may be active process-wide, so this test is the
        // sole owner of that global state for its whole duration rather than
        // splitting into a second `#[test]` that could race with this one
        // under the default parallel test runner.
        let empty_handle = start_capture(CaptureOptions {
            max_frames: 8,
            capture_gpu: false,
        })
        .expect("the capture above was already stopped, so starting a new one here should succeed");
        let empty_capture = empty_handle.stop();
        assert_eq!(empty_capture.frame_count(), 0);
        assert_eq!(empty_capture.retained_trace_bytes(), 0);

        // Now that every capture from this test is stopped, `ACTIVE_CAPTURE`
        // is empty, so the engine's live footprint should be nothing but
        // this thread's own completed-span recorder -- still allocated,
        // since a thread keeps its ring buffer for its whole lifetime once
        // first used, which is exactly why this is cheap and safe to call
        // with no capture running at all.
        let recorder_count = GLOBAL_THREAD_RECORDERS.lock().len() as u64;
        assert_eq!(
            capture_engine_memory_usage(),
            recorder_count * (THREAD_SPAN_BUDGET_BYTES as u64)
        );
    }

    #[test]
    fn memory_snapshot_total_bytes_sums_every_field() {
        let snapshot = MemorySnapshot {
            element_arena_bytes: 1_048_576,
            text_system: TextSystemMemory {
                glyph_cache_bytes: 2048,
                shaped_line_cache_bytes: 4096,
            },
            image_cache_bytes: 8192,
            capture_engine_bytes: 16384,
        };
        assert_eq!(snapshot.text_system.total_bytes(), 2048 + 4096);
        assert_eq!(snapshot.total_bytes(), 1_048_576 + 2048 + 4096 + 8192 + 16384);
    }

    #[test]
    fn gpu_memory_snapshot_total_bytes_sums_every_field() {
        let snapshot = GpuMemorySnapshot {
            fixed_buffer_bytes: 64 * 1024 * 1024,
            atlas_bytes: 4 * 1024 * 1024,
            surface_registry_bytes: 12 * 1024 * 1024,
            swapchain_bytes: 3 * 1920 * 1080 * 4,
        };
        assert_eq!(
            snapshot.total_bytes(),
            64 * 1024 * 1024 + 4 * 1024 * 1024 + 12 * 1024 * 1024 + 3 * 1920 * 1080 * 4
        );
    }

    // Phase 4 (issue #60): the public arm/retrieve half of the deep-capture
    // lifecycle. `WgpuRenderer::draw`'s actual recording/readback is covered
    // in `flamegraph_gpu`'s test module (it needs a real `wgpu::Device`);
    // this test covers everything reachable without one -- the request flag,
    // the completed-capture mailbox, and `DeepCapture::buffer_contents`'s
    // lookup helper. `DEEP_CAPTURE_REQUESTED`/`COMPLETED_DEEP_CAPTURE` are
    // process-wide statics that no other test in this suite touches (nothing
    // outside a real `WgpuRenderer::draw` call exercises them), so unlike
    // `ACTIVE_CAPTURE` above there's no cross-test ownership concern here.
    #[test]
    fn deep_capture_request_and_retrieval_round_trip() {
        assert!(!deep_capture_requested(), "should start unarmed");
        assert!(!take_deep_capture_request(), "taking with nothing armed should report false");

        request_deep_capture();
        assert!(deep_capture_requested(), "should be armed immediately after requesting");
        assert!(take_deep_capture_request(), "the first take after a request should report true");
        assert!(
            !take_deep_capture_request(),
            "a second take with nothing newly requested should report false"
        );
        assert!(!deep_capture_requested(), "taking the request should clear the armed flag");

        assert!(
            take_completed_deep_capture().is_none(),
            "nothing has completed yet, so retrieval should be empty"
        );

        let quads_bytes = vec![1u8, 2, 3, 4];
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
                bytes: quads_bytes.clone(),
            }],
            texture_contents: Vec::new(),
            resources_finalized: true,
        };
        complete_deep_capture(capture);

        let taken = take_completed_deep_capture().expect("the capture published above should be retrievable");
        assert_eq!(taken.draw_calls.len(), 1);
        assert!(taken.resources_finalized);
        assert_eq!(
            taken.buffer_contents(DeepCaptureBufferKind::Quads).map(|contents| &contents.bytes),
            Some(&quads_bytes)
        );
        assert!(
            taken.buffer_contents(DeepCaptureBufferKind::Shadows).is_none(),
            "a buffer kind that was never touched should not be present"
        );

        assert!(
            take_completed_deep_capture().is_none(),
            "taking the completed capture should clear the mailbox, so a second take is empty"
        );
    }
}
