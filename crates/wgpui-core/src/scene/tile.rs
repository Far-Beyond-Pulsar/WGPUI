//! `TileCoord` and the `(boundary, TileCoord)` half of a layer address.
//! See docs/gpu-native-architecture.md §4.3.
//!
//! **Phase 1 defines the address, not the mechanism.** §4.3's tile grid — the
//! visibility compute pass, spatial mark-and-sweep eviction, the resident-tile
//! LRU budget — is Phase 4.5 work and none of it exists here. What exists is
//! the observation §4.3 opens with: "a tile is just a `Layer`, addressed one
//! dimension further." Making [`crate::scene::layer::LayerKey`] carry an
//! optional tile coordinate from the start costs one field and means Phase 4.5
//! extends a mechanism rather than reshaping the identity of every layer in a
//! shipped scene.

/// A tile's integer coordinate on its boundary's content plane.
///
/// Coordinates are signed because a freely-pannable plane has no origin corner
/// — panning up and left from the initial view produces negative coordinates,
/// and clamping them at zero would silently alias two distinct tiles onto one
/// address.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileCoord {
    /// Column index along the horizontal axis.
    pub x: i32,
    /// Row index along the vertical axis.
    pub y: i32,
}

impl TileCoord {
    /// The tile containing the plane's origin.
    pub const ORIGIN: TileCoord = TileCoord { x: 0, y: 0 };

    /// A coordinate at `(x, y)`.
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negative_coordinates_are_distinct_addresses() {
        assert_ne!(TileCoord::new(-1, 0), TileCoord::new(1, 0));
        assert_ne!(TileCoord::new(0, -1), TileCoord::ORIGIN);
    }
}
