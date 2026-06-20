//! UI refresh-heat profiling: a 2D analogue of Unreal's shader-complexity view.
//!
//! Each rendered view's refresh cost (the time its layout/prepaint took this
//! frame, ~0 when its cached prepaint is reused) is mapped to a [`RefreshHeat`]
//! level. [`Window`](crate::Window) records these per frame and, when the
//! refresh overlay is enabled, paints a translucent heat-colored quad over each
//! view plus a gradient legend so hot (frequently/expensively re-rendering)
//! regions stand out.

use std::time::Duration;

use collections::{FxHashMap, FxHashSet};

use crate::{Bounds, EntityId, Hsla, Pixels, hsla};

/// Refresh-cost heat level for a view, mapped from how long its most recent
/// refresh took. Surfaced through the refresh overlay and `FrameMetrics`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
pub enum RefreshHeat {
    /// Reused or trivially cheap this frame.
    #[default]
    Cold,
    /// Inexpensive refresh.
    Warm,
    /// Noticeable refresh cost.
    Hot,
    /// Dominant refresh cost; the first place to look when optimizing.
    Critical,
}

impl RefreshHeat {
    fn from_elapsed(elapsed: Duration) -> Self {
        match elapsed.as_micros() {
            0..=249 => RefreshHeat::Cold,
            250..=999 => RefreshHeat::Warm,
            1_000..=3_999 => RefreshHeat::Hot,
            _ => RefreshHeat::Critical,
        }
    }

    /// Translucent overlay color for this heat level, green (cold) -> red
    /// (critical), matching the shader-complexity gradient.
    pub fn overlay_color(self) -> Hsla {
        let hue = match self {
            RefreshHeat::Cold => 0.34,
            RefreshHeat::Warm => 0.16,
            RefreshHeat::Hot => 0.07,
            RefreshHeat::Critical => 0.0,
        };
        hsla(hue, 0.9, 0.5, 0.35)
    }

    /// Opaque swatch color for the legend bar.
    pub fn legend_color(self) -> Hsla {
        let mut color = self.overlay_color();
        color.a = 1.0;
        color
    }

    /// The heat levels in ascending order, for rendering the legend gradient.
    pub fn gradient() -> [RefreshHeat; 4] {
        [
            RefreshHeat::Cold,
            RefreshHeat::Warm,
            RefreshHeat::Hot,
            RefreshHeat::Critical,
        ]
    }
}

#[derive(Clone, Copy, Debug)]
struct RefreshProfileSample {
    last_time: Duration,
    last_bounds: Bounds<Pixels>,
}

/// Accumulates per-view refresh cost across frames and exposes the current
/// frame's heat regions for the overlay.
#[derive(Default)]
pub(crate) struct RefreshProfiler {
    samples: FxHashMap<EntityId, RefreshProfileSample>,
    seen_this_frame: FxHashSet<EntityId>,
}

impl RefreshProfiler {
    /// Begin a new frame: forget which views were drawn last frame so the
    /// overlay only reflects what is on screen now (historical timing is kept).
    pub(crate) fn begin_frame(&mut self) {
        self.seen_this_frame.clear();
    }

    /// Record one view's refresh for this frame and return its heat level.
    pub(crate) fn record(
        &mut self,
        view_id: EntityId,
        bounds: Bounds<Pixels>,
        elapsed: Duration,
    ) -> RefreshHeat {
        self.seen_this_frame.insert(view_id);
        self.samples.insert(
            view_id,
            RefreshProfileSample {
                last_time: elapsed,
                last_bounds: bounds,
            },
        );
        RefreshHeat::from_elapsed(elapsed)
    }

    /// The heat regions drawn this frame, for painting the overlay.
    pub(crate) fn regions(&self) -> Vec<(Bounds<Pixels>, RefreshHeat)> {
        self.seen_this_frame
            .iter()
            .filter_map(|view_id| self.samples.get(view_id))
            .map(|sample| {
                (
                    sample.last_bounds,
                    RefreshHeat::from_elapsed(sample.last_time),
                )
            })
            .collect()
    }

    /// Hottest heat level recorded this frame, or `Cold` when nothing was drawn.
    pub(crate) fn frame_heat(&self) -> RefreshHeat {
        self.seen_this_frame
            .iter()
            .filter_map(|view_id| self.samples.get(view_id))
            .map(|sample| RefreshHeat::from_elapsed(sample.last_time))
            .max()
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{point, px, size};

    fn bounds() -> Bounds<Pixels> {
        Bounds {
            origin: point(px(0.), px(0.)),
            size: size(px(10.), px(10.)),
        }
    }

    #[test]
    fn refresh_heat_maps_elapsed_time_to_levels() {
        assert_eq!(
            RefreshHeat::from_elapsed(Duration::from_micros(100)),
            RefreshHeat::Cold
        );
        assert_eq!(
            RefreshHeat::from_elapsed(Duration::from_micros(500)),
            RefreshHeat::Warm
        );
        assert_eq!(
            RefreshHeat::from_elapsed(Duration::from_micros(2_000)),
            RefreshHeat::Hot
        );
        assert_eq!(
            RefreshHeat::from_elapsed(Duration::from_micros(8_000)),
            RefreshHeat::Critical
        );
    }

    #[test]
    fn profiler_reports_only_current_frame_regions() {
        let mut profiler = RefreshProfiler::default();
        let cold_view = EntityId::from(1);
        let hot_view = EntityId::from(2);

        profiler.begin_frame();
        assert_eq!(
            profiler.record(cold_view, bounds(), Duration::from_micros(10)),
            RefreshHeat::Cold
        );
        assert_eq!(
            profiler.record(hot_view, bounds(), Duration::from_micros(5_000)),
            RefreshHeat::Critical
        );
        assert_eq!(profiler.regions().len(), 2);
        assert_eq!(profiler.frame_heat(), RefreshHeat::Critical);

        // Next frame only the cold view draws; the hot view drops out of the
        // overlay even though its history is retained.
        profiler.begin_frame();
        profiler.record(cold_view, bounds(), Duration::from_micros(10));
        let regions = profiler.regions();
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].1, RefreshHeat::Cold);
        assert_eq!(profiler.frame_heat(), RefreshHeat::Cold);
    }

    #[test]
    fn empty_frame_is_cold() {
        let mut profiler = RefreshProfiler::default();
        profiler.begin_frame();
        assert!(profiler.regions().is_empty());
        assert_eq!(profiler.frame_heat(), RefreshHeat::Cold);
    }

    #[test]
    fn heat_levels_have_distinct_overlay_colors() {
        let colors: Vec<_> = RefreshHeat::gradient()
            .into_iter()
            .map(|heat| heat.overlay_color())
            .collect();
        for (index, color) in colors.iter().enumerate() {
            assert!((color.a - 0.35).abs() < f32::EPSILON);
            for other in &colors[index + 1..] {
                assert_ne!(color.h, other.h, "each heat level needs a distinct hue");
            }
        }
        // Legend swatches are opaque.
        assert!((RefreshHeat::Cold.legend_color().a - 1.0).abs() < f32::EPSILON);
    }
}
