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
}

impl Default for TileRefreshFlash {
    fn default() -> Self {
        Self {
            enabled: false,
            tile_size: [256.0, 256.0],
            duration_frames: 2,
            color: [1.0, 0.0, 1.0, 0.35],
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
}
