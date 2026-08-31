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

impl DebugTile {
    /// Attach the measured refresh rate shown in this tile's diagnostic label.
    ///
    /// The value occupies the first word of the existing padding so the
    /// diagnostic storage stride and public layout remain unchanged.
    pub fn with_refresh_rate(mut self, frames_per_second: f32) -> Self {
        self._padding[0] = frames_per_second.max(0.0);
        self._padding[1] = 1.0;
        self
    }

    /// Attach a refresh count shown in this region's diagnostic label.
    pub fn with_refresh_count(mut self, count: u32) -> Self {
        self._padding[0] = count as f32;
        self._padding[1] = 0.0;
        self
    }
}

/// A tile-refresh visualizer configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileRefreshFlash {
    /// Whether refreshed tiles are outlined.
    pub enabled: bool,
    /// Tile edge in logical pixels. This must match the tiled boundary policy.
    pub tile_size: [f32; 2],
    /// Number of displayed frames a refreshed tile remains outlined.
    pub duration_frames: u32,
    /// RGBA outline colour. The default is opaque yellow; alpha controls the
    /// outline itself and never tints the content inside it.
    pub color: [f32; 4],
    /// Also draw the visible tile outlines for tiled boundaries while their
    /// diagnostics are active.
    pub viewport_grid: bool,
}

impl Default for TileRefreshFlash {
    fn default() -> Self {
        Self {
            enabled: false,
            tile_size: [256.0, 256.0],
            duration_frames: 2,
            color: [1.0, 1.0, 0.0, 1.0],
            viewport_grid: false,
        }
    }
}

impl TileRefreshFlash {
    /// Enable the visualizer using the default yellow two-frame outline.
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

    /// Also draw visible tile outlines for tiled boundaries.
    pub fn with_viewport_grid(mut self, enabled: bool) -> Self {
        self.viewport_grid = enabled;
        self
    }
}

/// Performance diagnostics exposed by a native WGPUI window.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PerformanceDebug {
    tile_refresh: TileRefreshFlash,
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
    }

    /// Configure the tile refresh visualizer.
    pub fn set_tile_refresh_flash(&mut self, flash: TileRefreshFlash) {
        self.tile_refresh = flash;
    }

    /// Current tile refresh configuration.
    pub const fn tile_refresh_flash(&self) -> TileRefreshFlash {
        self.tile_refresh
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
    fn default_flash_is_yellow_and_two_frames() {
        let mut debug = PerformanceDebug::default();
        debug.flash_tile_refreshes();
        let flash = debug.tile_refresh_flash();
        assert_eq!(flash.duration_frames, 2);
        assert_eq!(flash.color, [1.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn invalid_tile_dimensions_do_not_replace_a_valid_configuration() {
        let flash = TileRefreshFlash::enabled().with_tile_size(64.0, 32.0);
        assert_eq!(flash.with_tile_size(f32::NAN, 0.0).tile_size, [64.0, 32.0]);
    }

    #[test]
    fn a_refresh_rate_is_stored_without_changing_the_debug_tile_stride() {
        let tile = DebugTile {
            origin_size: [0.0; 4],
            color: [1.0; 4],
            border_width: 1.0,
            _padding: [0.0; 7],
        }
        .with_refresh_rate(60.0);
        assert_eq!(tile._padding[0], 60.0);
        assert_eq!(tile._padding[1], 1.0);
        assert_eq!(std::mem::size_of::<DebugTile>(), 64);
    }

    #[test]
    fn a_refresh_count_uses_the_same_storage_as_a_rate() {
        let tile = DebugTile {
            origin_size: [0.0; 4],
            color: [1.0; 4],
            border_width: 1.0,
            _padding: [0.0; 7],
        }
        .with_refresh_count(4);
        assert_eq!(tile._padding[0], 4.0);
        assert_eq!(tile._padding[1], 0.0);
        assert_eq!(std::mem::size_of::<DebugTile>(), 64);
    }
}
