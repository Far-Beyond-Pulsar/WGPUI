//! Mesa/MangoHud-style on-screen performance overlay.
//!
//! Three independent runtime switches live here, all off by default and all
//! flippable from the Inspector's Utilities tab — see
//! `crates/ui/wgpui-component/crates/ui/src/inspector.rs`'s
//! `render_utilities_tab`:
//!
//! - [`set_hud_enabled`] — the overlay itself: a corner panel with current
//!   FPS, a scrolling frame-time graph colour-coded like Mesa's own HUD
//!   (green/yellow/red against 60fps/30fps), and a live primitive-count
//!   readout read directly off the just-finished [`crate::Scene`].
//! - [`set_layer_fps_enabled`] — per-layer re-render-rate labels painted next
//!   to the [`crate::layer`] debug tint. Meaningful only alongside that tint,
//!   so the Inspector nests its switch under the layer-debug one; nothing
//!   here enforces that, it would just have no tint to sit next to without it.
//! - [`set_slow_frame_flash_enabled`] — a border that flashes red for one
//!   frame whenever a frame exceeds [`SLOW_FRAME_MS`], so a stall shows up
//!   even when you weren't looking at the graph at that instant.
//!
//! [`record_frame`] is the one piece of actual state: a rolling history of
//! frame durations, recorded once per [`crate::Window::draw`] regardless of
//! whether the HUD is currently visible — so turning it on shows an
//! already-populated graph instead of one that fills in over the next few
//! seconds.

use crate::time_ext::Instant;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};

/// How many frame-time samples the graph keeps. At 60fps this is four
/// seconds of history — enough to see a stutter's shape, short enough that
/// the graph stays legible at HUD width.
pub(crate) const HISTORY_LEN: usize = 240;

/// Frame time below which the HUD graph draws green (60fps or better).
pub(crate) const OK_FRAME_MS: f32 = 1000.0 / 60.0;
/// Frame time above which the HUD graph draws red (worse than 30fps); the
/// green-to-red band between this and [`OK_FRAME_MS`] draws yellow.
pub(crate) const SLOW_FRAME_MS: f32 = 1000.0 / 30.0;

static HUD_ENABLED: AtomicBool = AtomicBool::new(false);
static LAYER_FPS_ENABLED: AtomicBool = AtomicBool::new(false);
static SLOW_FRAME_FLASH_ENABLED: AtomicBool = AtomicBool::new(false);

/// Turn the Mesa-style HUD overlay on or off at runtime.
pub fn set_hud_enabled(enabled: bool) {
    HUD_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Whether the HUD overlay is currently on.
pub fn is_hud_enabled() -> bool {
    HUD_ENABLED.load(Ordering::Relaxed)
}

/// Turn per-layer re-render-rate labels on or off. See the module doc
/// comment for why this is meaningful only alongside
/// [`crate::is_layer_debug_enabled`].
pub fn set_layer_fps_enabled(enabled: bool) {
    LAYER_FPS_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Whether per-layer re-render-rate labels are currently on.
pub fn is_layer_fps_enabled() -> bool {
    LAYER_FPS_ENABLED.load(Ordering::Relaxed)
}

/// Turn the slow-frame flash border on or off.
pub fn set_slow_frame_flash_enabled(enabled: bool) {
    SLOW_FRAME_FLASH_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Whether the slow-frame flash border is currently on.
pub fn is_slow_frame_flash_enabled() -> bool {
    SLOW_FRAME_FLASH_ENABLED.load(Ordering::Relaxed)
}

/// Rolling frame-time history, in milliseconds, oldest first.
struct History {
    last_frame_start: Option<Instant>,
    samples: VecDeque<f32>,
}

static HISTORY: Mutex<Option<History>> = Mutex::new(None);

/// A snapshot of the frame-time history, cheap to compute from and safe to
/// hold past the lock: current/avg/max in ms, plus the sample list itself
/// for the graph.
pub(crate) struct Snapshot {
    pub(crate) samples: Vec<f32>,
    pub(crate) current_ms: f32,
    pub(crate) avg_ms: f32,
    pub(crate) max_ms: f32,
}

impl Snapshot {
    pub(crate) fn fps(&self) -> f32 {
        if self.avg_ms <= 0.0 {
            0.0
        } else {
            1000.0 / self.avg_ms
        }
    }
}

/// Record that a frame just finished, returning this frame's duration in
/// milliseconds (`0.0` for the very first frame, which has nothing to
/// measure against). Called once per [`crate::Window::draw`] — see that
/// method's own call site — unconditionally, so the history is warm the
/// instant the HUD is switched on.
pub(crate) fn record_frame() -> f32 {
    let mut guard = HISTORY.lock();
    let history = guard.get_or_insert_with(|| History {
        last_frame_start: None,
        samples: VecDeque::with_capacity(HISTORY_LEN),
    });
    let now = Instant::now();
    let ms = match history.last_frame_start.replace(now) {
        Some(previous) => now.saturating_duration_since(previous).as_secs_f32() * 1000.0,
        None => 0.0,
    };
    history.samples.push_back(ms);
    while history.samples.len() > HISTORY_LEN {
        history.samples.pop_front();
    }
    ms
}

/// The current frame-time history, for [`crate::Window`]'s HUD paint code.
pub(crate) fn snapshot() -> Snapshot {
    let guard = HISTORY.lock();
    let Some(history) = guard.as_ref() else {
        return Snapshot {
            samples: Vec::new(),
            current_ms: 0.0,
            avg_ms: 0.0,
            max_ms: 0.0,
        };
    };
    let samples: Vec<f32> = history.samples.iter().copied().collect();
    let current_ms = samples.last().copied().unwrap_or(0.0);
    let max_ms = samples.iter().copied().fold(0.0f32, f32::max);
    let avg_ms = if samples.is_empty() {
        0.0
    } else {
        samples.iter().sum::<f32>() / samples.len() as f32
    };
    Snapshot {
        samples,
        current_ms,
        avg_ms,
        max_ms,
    }
}

/// Whether the just-recorded frame was slow enough for
/// [`is_slow_frame_flash_enabled`] to flash the border, per [`SLOW_FRAME_MS`].
pub(crate) fn is_slow_frame(frame_ms: f32) -> bool {
    frame_ms > SLOW_FRAME_MS
}

/// Colour a frame-time sample by the same green/yellow/red bands the graph
/// uses, as `(h, s, l)` — callers apply their own alpha.
pub(crate) fn frame_time_color(ms: f32) -> (f32, f32, f32) {
    if ms <= OK_FRAME_MS {
        (0.33, 0.65, 0.45) // green
    } else if ms <= SLOW_FRAME_MS {
        (0.14, 0.75, 0.5) // yellow
    } else {
        (0.0, 0.75, 0.55) // red
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_time_color_bands_match_the_documented_thresholds() {
        assert_eq!(frame_time_color(OK_FRAME_MS - 1.0).0, 0.33, "at 60fps: green");
        assert_eq!(
            frame_time_color((OK_FRAME_MS + SLOW_FRAME_MS) / 2.0).0,
            0.14,
            "between 60fps and 30fps: yellow"
        );
        assert_eq!(
            frame_time_color(SLOW_FRAME_MS + 1.0).0,
            0.0,
            "worse than 30fps: red"
        );
    }

    #[test]
    fn is_slow_frame_matches_the_thirty_fps_threshold() {
        assert!(!is_slow_frame(SLOW_FRAME_MS - 0.01));
        assert!(is_slow_frame(SLOW_FRAME_MS + 0.01));
    }

    #[test]
    fn snapshot_of_empty_history_reads_as_zero_not_nan() {
        // A fresh `Snapshot` should never divide by zero into a NaN FPS —
        // regression guard for the empty-history path specifically, since
        // `record_frame` is what would normally populate this and this test
        // deliberately doesn't call it (that static is shared process-wide
        // and other tests may have already touched it).
        let empty = Snapshot {
            samples: Vec::new(),
            current_ms: 0.0,
            avg_ms: 0.0,
            max_ms: 0.0,
        };
        assert_eq!(empty.fps(), 0.0);
        assert!(empty.fps().is_finite());
    }

    #[test]
    fn fps_is_the_reciprocal_of_average_frame_time() {
        let snapshot = Snapshot {
            samples: vec![16.0, 16.0, 16.0],
            current_ms: 16.0,
            avg_ms: 16.0,
            max_ms: 16.0,
        };
        assert!((snapshot.fps() - 62.5).abs() < 0.01);
    }
}
