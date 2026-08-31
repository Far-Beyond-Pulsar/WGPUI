//! Opt-in visual diagnostics for the retained renderer.
//!
//! These controls are deliberately separate from the production scene. A
//! diagnostic must not allocate scene slots, dirty retained primitives, or
//! change the result when it is disabled.

use std::time::Duration;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DebugTile {
    pub origin_size: [f32; 4],
    pub color: [f32; 4],
    /// Width of the outline in logical pixels. The inset between neighboring
    /// tiles is intentional so adjacent refreshes remain individually legible.
    pub border_width: f32,
    /// Storage buffers use a 16-byte WGSL struct alignment. Keep the host
    /// stride at 64 bytes even though the meaningful fields end at byte 36.
    pub _padding: [f32; 7],
}

unsafe impl bytemuck::Zeroable for DebugTile {}
unsafe impl bytemuck::Pod for DebugTile {}

/// A tile-refresh visualizer configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileRefreshFlash {
    /// Whether refreshed tiles are shaded.
    pub enabled: bool,
    /// Tile edge in logical pixels. This must match the tiled boundary policy.
    pub tile_size: [f32; 2],
    /// Number of displayed frames a refreshed tile remains shaded.
    pub duration_frames: u32,
    /// RGBA overlay colour. Alpha controls how strongly content is tinted.
    pub color: [f32; 4],
    /// Also flash the visible diagnostic grid when the scene has not yet
    /// opted a boundary into tiled residency.
    pub viewport_grid: bool,
}

/// Categories rendered by the opt-in retained tile visualizer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DebugVisualization {
    /// Whether the visualizer contributes an overlay.
    pub enabled: bool,
    /// Draw the root/tile ownership boundaries.
    pub show_ownership: bool,
    /// Draw tiles eligible to draw after visibility and clip tests.
    pub show_visibility: bool,
    /// Draw tiles with retained content, including tiles outside the viewport.
    pub show_residency: bool,
    /// Draw effective root clips.
    pub show_clips: bool,
    /// Draw root transforms and transformed tile bounds.
    pub show_transforms: bool,
    /// Draw tiles exposed by the latest visit.
    pub show_newly_exposed: bool,
    /// Draw raster and compositing damage regions.
    pub show_damage: bool,
    /// Ownership color.
    pub ownership_color: [f32; 4],
    /// Visibility color.
    pub visibility_color: [f32; 4],
    /// Residency color.
    pub residency_color: [f32; 4],
    /// Clip color.
    pub clip_color: [f32; 4],
    /// Transform color.
    pub transform_color: [f32; 4],
    /// Newly exposed color.
    pub newly_exposed_color: [f32; 4],
    /// Damage color.
    pub damage_color: [f32; 4],
}

impl Default for DebugVisualization {
    fn default() -> Self {
        Self {
            enabled: false,
            show_ownership: true,
            show_visibility: true,
            show_residency: true,
            show_clips: true,
            show_transforms: true,
            show_newly_exposed: true,
            show_damage: true,
            ownership_color: [0.1, 0.7, 1.0, 0.55],
            visibility_color: [0.2, 1.0, 0.2, 0.45],
            residency_color: [1.0, 0.7, 0.1, 0.35],
            clip_color: [0.9, 0.2, 1.0, 0.7],
            transform_color: [0.2, 0.9, 0.9, 0.7],
            newly_exposed_color: [1.0, 0.2, 0.1, 0.75],
            damage_color: [1.0, 0.0, 0.6, 0.65],
        }
    }
}

impl DebugVisualization {
    /// Enable all categories with the default diagnostic colors.
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            ..Self::default()
        }
    }

    /// Disable the overlay while preserving category and color choices.
    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }
}

impl Default for TileRefreshFlash {
    fn default() -> Self {
        Self {
            enabled: false,
            tile_size: [256.0, 256.0],
            duration_frames: 2,
            color: [1.0, 0.0, 1.0, 0.35],
            viewport_grid: false,
        }
    }
}

impl TileRefreshFlash {
    /// Enable the visualizer using the default magenta two-frame flash.
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            ..Self::default()
        }
    }

    /// Configure a flash that lasts for approximately `duration`.
    pub fn for_duration(duration: Duration) -> Self {
        let frames = (duration.as_secs_f64() * 60.0).ceil() as u32;
        Self {
            enabled: true,
            duration_frames: frames.max(1),
            ..Self::default()
        }
    }

    /// Set the logical tile dimensions used to locate the overlay.
    pub fn with_tile_size(mut self, width: f32, height: f32) -> Self {
        if width.is_finite() && width > 0.0 && height.is_finite() && height > 0.0 {
            self.tile_size = [width, height];
        }
        self
    }

    /// Set the overlay colour.
    pub fn with_color(mut self, color: [f32; 4]) -> Self {
        self.color = color;
        self
    }

    /// Also flash the visible tile grid when an application has not yet
    /// opted a boundary into tiled residency.
    pub fn with_viewport_grid(mut self, enabled: bool) -> Self {
        self.viewport_grid = enabled;
        self
    }
}

/// Performance diagnostics exposed by a native WGPUI window.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PerformanceDebug {
    tile_refresh: TileRefreshFlash,
    visualization: DebugVisualization,
}

impl PerformanceDebug {
    /// Enable the tile refresh flash.
    pub fn flash_tile_refreshes(&mut self) {
        self.tile_refresh = TileRefreshFlash::enabled();
    }

    /// Enable the tile refresh flash with a custom duration.
    pub fn flash_tile_refreshes_for(&mut self, duration: Duration) {
        self.tile_refresh = TileRefreshFlash::for_duration(duration);
    }

    /// Disable all visual diagnostics.
    pub fn disable(&mut self) {
        self.tile_refresh.enabled = false;
        self.visualization.enabled = false;
    }

    /// Configure the tile refresh visualizer.
    pub fn set_tile_refresh_flash(&mut self, flash: TileRefreshFlash) {
        self.tile_refresh = flash;
    }

    /// Current tile refresh configuration.
    pub const fn tile_refresh_flash(&self) -> TileRefreshFlash {
        self.tile_refresh
    }

    /// Configure the retained tile/root/damage visualizer.
    pub fn set_visualization(&mut self, visualization: DebugVisualization) {
        self.visualization = visualization;
    }

    /// Current retained tile/root/damage visualization configuration.
    pub const fn visualization(&self) -> DebugVisualization {
        self.visualization
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostics_are_disabled_by_default() {
        assert!(!PerformanceDebug::default().tile_refresh_flash().enabled);
    }

    #[test]
    fn default_flash_is_magenta_and_two_frames() {
        let mut debug = PerformanceDebug::default();
        debug.flash_tile_refreshes();
        let flash = debug.tile_refresh_flash();
        assert_eq!(flash.duration_frames, 2);
        assert_eq!(flash.color, [1.0, 0.0, 1.0, 0.35]);
    }

    #[test]
    fn invalid_tile_dimensions_do_not_replace_a_valid_configuration() {
        let flash = TileRefreshFlash::enabled().with_tile_size(64.0, 32.0);
        assert_eq!(flash.with_tile_size(f32::NAN, 0.0).tile_size, [64.0, 32.0]);
    }

    #[test]
    fn visualization_is_opt_in_and_disable_turns_off_both_overlays() {
        let mut debug = PerformanceDebug::default();
        assert!(!debug.visualization().enabled);
        debug.set_visualization(DebugVisualization::enabled());
        debug.flash_tile_refreshes();
        debug.disable();
        assert!(!debug.visualization().enabled);
        assert!(!debug.tile_refresh_flash().enabled);
    }
}
