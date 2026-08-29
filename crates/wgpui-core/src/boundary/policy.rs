//! `BoundaryPolicy`, `Buffering` enum (`None`/`Margin`/`Tiled`).
//! See docs/gpu-native-architecture.md §4.1, §4.3.
//!
//! # Tuning only, never correctness
//!
//! §4.1 states the rule this file exists to hold to: a `.boundary()` with no
//! policy at all is already correct. Every field below changes *how* a boundary
//! that is known to be dirty gets rasterized and buffered — never *whether* it
//! is considered dirty, which the reconciler alone decides (§4.0) and
//! [`crate::boundary::compositor`] alone consumes. Nothing in this file can
//! make a boundary skip work it needed to do.
//!
//! # Why `Pixels`/`Size` are defined here
//!
//! §4.1 spells `Buffering::Margin` as `Option<Size<Pixels>>`, and both types
//! belong to the frontend geometry surface §7 freezes — which still lives in
//! the legacy crate and has no home in the workspace yet (§3's file map gives
//! `wgpui-core` no geometry module). Rather than either widen the signature to
//! a bare `[f32; 2]` or pull the legacy crate across the boundary §3 draws,
//! the two types are declared minimally here, carrying exactly the meaning
//! their frozen counterparts have. Whichever phase moves geometry into the
//! workspace replaces them and nothing else in this file changes.

use crate::scene::tile::TileGrid;
use std::fmt;
use std::ops::{Add, AddAssign, Neg, Sub};

/// A length in logical pixels.
#[derive(Copy, Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct Pixels(pub f32);

impl Pixels {
    /// Zero length.
    pub const ZERO: Pixels = Pixels(0.0);

    /// This length scaled by `factor`.
    pub fn scaled(self, factor: f32) -> Self {
        Pixels(self.0 * factor)
    }

    /// The underlying value.
    pub const fn value(self) -> f32 {
        self.0
    }

    pub const fn to_f64(self) -> f64 { self.0 as f64 }

    pub fn max(self, other: Self) -> Self { Self(self.0.max(other.0)) }
}

impl Add for Pixels {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

impl Sub for Pixels {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }
}

impl AddAssign for Pixels {
    fn add_assign(&mut self, other: Self) { self.0 += other.0; }
}

impl Neg for Pixels {
    type Output = Self;

    fn neg(self) -> Self { Self(-self.0) }
}

impl fmt::Display for Pixels {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

/// A width/height pair.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Size<T> {
    /// Extent along the x axis.
    pub width: T,
    /// Extent along the y axis.
    pub height: T,
}

impl Size<Pixels> {
    /// A zero-by-zero size.
    pub const ZERO: Size<Pixels> = Size {
        width: Pixels::ZERO,
        height: Pixels::ZERO,
    };

    /// A size in logical pixels.
    pub const fn pixels(width: f32, height: f32) -> Self {
        Size {
            width: Pixels(width),
            height: Pixels(height),
        }
    }

    /// Both extents scaled by `factor`.
    pub fn scaled(self, factor: f32) -> Self {
        Size {
            width: self.width.scaled(factor),
            height: self.height.scaled(factor),
        }
    }
}

/// How a boundary buffers ahead of scroll or pan.
///
/// `Margin` is R-N §7/SFD's existing overscroll buffer, generalized only in
/// name. `Tiled` is §4.3's grid, whose mechanism (visibility pass, spatial
/// eviction, resident-tile budget) is Phase 4.5 — the variant exists here from
/// the start so `LayerKey` and this enum are shaped for it rather than reshaped
/// around it later, which is the same reason `scene/tile.rs` already defines a
/// `TileCoord` nothing yet produces.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Buffering {
    /// No buffer beyond the visible bounds.
    None,
    /// One rectangular region sized to viewport + margin, refilled wholesale
    /// when scrolled past it. Right for linear content: lists, columns,
    /// anything scrolling along one or two bounded axes with a defined content
    /// extent. Auto-sized from the viewport when unset (SFD §7).
    Margin(Option<Size<Pixels>>),
    /// A grid of independently cached tiles (§4.3). Right for freeform,
    /// arbitrarily-positioned, pannable-in-any-direction content, where
    /// `Margin` would have to grow multiplicatively in both axes.
    ///
    /// Built in Phase 4.5: [`crate::scene::tile`] is the grid, the visibility
    /// predicate, and the residency budget, and
    /// [`crate::boundary::compositor::Compositor::visit_tiled`] is what a frame
    /// calls to get this boundary's live tile set.
    Tiled {
        /// Edge length of one tile.
        ///
        /// [`crate::scene::TileGrid::DEFAULT_EDGE`] is the measured starting
        /// point; see `docs/phase-4.5-results.md` for the sweep it came from.
        tile_size: Size<Pixels>,
        /// How many tiles beyond the viewport stay resident.
        retain_radius: u32,
    },
}

impl Default for Buffering {
    fn default() -> Self {
        Buffering::Margin(None)
    }
}

impl Buffering {
    /// The fraction of the viewport an unset [`Buffering::Margin`] buffers on
    /// each axis, per R-N §7's own suggestion (`overdraw_margin: viewport *
    /// 0.5`), which SFD §1.1 adopts as the implicit policy for a scroll
    /// container that names no margin.
    pub const AUTO_MARGIN_FRACTION: f32 = 0.5;

    /// The overdraw margin this buffering asks for, given the boundary's
    /// viewport.
    ///
    /// **[`Buffering::Tiled`] now reports its real margin**, which is its retain
    /// radius measured in pixels — `retain_radius × tile_size` on each axis.
    /// Until Phase 4.5 it reported the auto margin instead, as a placeholder
    /// with a stated reason (reporting zero would have made a boundary asking
    /// for *more* buffering receive none). That placeholder is closed: the
    /// number below is what the mechanism actually keeps resident, so a caller
    /// reading it gets an answer about this boundary rather than about the
    /// variant it fell back to. A tile size this crate cannot build a grid from
    /// still reports the auto margin, because such a boundary really does fall
    /// back to untiled buffering — see
    /// [`crate::boundary::compositor::Compositor::visit_tiled`].
    pub fn margin(self, viewport: Size<Pixels>) -> Size<Pixels> {
        match self {
            Buffering::None => Size::ZERO,
            Buffering::Margin(Some(margin)) => margin,
            Buffering::Margin(None) => viewport.scaled(Self::AUTO_MARGIN_FRACTION),
            Buffering::Tiled {
                tile_size,
                retain_radius,
            } => match TileGrid::new(tile_size) {
                Some(_) => tile_size.scaled(retain_radius as f32),
                None => viewport.scaled(Self::AUTO_MARGIN_FRACTION),
            },
        }
    }

    /// The tile grid this buffering describes, or `None` for a variant that is
    /// not tiled — or a tile size no grid can be built from.
    ///
    /// The two `None` cases are deliberately one: a caller's response to both is
    /// the same, which is to buffer this boundary the untiled way.
    pub fn tile_grid(self) -> Option<TileGrid> {
        match self {
            Buffering::Tiled { tile_size, .. } => TileGrid::new(tile_size),
            _ => None,
        }
    }

    /// How many tiles beyond the viewport this buffering keeps resident, or `0`
    /// for a variant that has no tiles.
    pub const fn retain_radius(self) -> u32 {
        match self {
            Buffering::Tiled { retain_radius, .. } => retain_radius,
            _ => 0,
        }
    }
}

/// Whether a boundary is cheaper to keep as primitives or to composite through
/// its own texture.
///
/// **Phase 2 produces this decision and stops there.** No texture is created,
/// pooled, or drawn anywhere in this crate — §3.1 puts every live device in
/// `wgpui-wgpu`, and §8 puts the compositing entry itself in Phase 4. What is
/// real here is that the decision is made per boundary, from that boundary's
/// own primitive count, and is observable.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Retention {
    /// The boundary keeps its slab and re-draws it with the layer transform
    /// folded in — R-N §3.3's "a layer holding twelve quads is cheaper to
    /// re-emit than to composite through a texture."
    Primitives,
    /// The boundary rasterizes into a texture of its own and composites that.
    Texture,
}

/// Tuning for one compositing boundary. Every field is optional in the sense
/// that [`BoundaryPolicy::default`] is always a correct answer.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BoundaryPolicy {
    /// Below this primitive count the boundary stays primitive-retained (no
    /// texture). Same role as R-N's `rasterize_above`, same default.
    pub rasterize_above: usize,
    /// How this boundary buffers ahead of scroll or pan.
    pub buffering: Buffering,
    /// Frames a boundary may go unvisited before its retained resources are
    /// returned to the pool (R-N §3.4's mark-and-sweep interval). Under
    /// [`Buffering::Tiled`] the same interval applies per tile, with "unvisited"
    /// meaning "out of range" (§4.3).
    pub evict_after_frames: u32,
    /// The total resident-tile cap for a [`Buffering::Tiled`] boundary, beyond
    /// which the least recently visited tiles are evicted.
    ///
    /// §4.3 and §9's risk table both call this out as a first-class part of the
    /// mechanism rather than a follow-up: `evict_after_frames` is a per-tile
    /// timer, and an erratic pan can hold far more tiles inside that interval at
    /// once than R-N's one-buffer-per-layer design ever had to bound. Ignored by
    /// every other [`Buffering`] variant, which has one buffer and needs no cap.
    pub resident_tile_budget: usize,
}

impl BoundaryPolicy {
    /// R-N §3.3's default threshold, unchanged.
    pub const DEFAULT_RASTERIZE_ABOVE: usize = 256;

    /// R-N §3.4's default eviction interval, unchanged.
    pub const DEFAULT_EVICT_AFTER_FRAMES: u32 = 60;

    /// The default resident-tile cap.
    ///
    /// Sized against what the default grid actually needs: a 2560×1440 viewport
    /// at [`crate::scene::TileGrid::DEFAULT_EDGE`] with a retain radius of 1 is
    /// 12×8 = 96 tiles in range, so a budget below that would be permanently
    /// over-budget on an ordinary window. 256 leaves room for roughly two and a
    /// half viewports' worth of recently-visited tiles beyond the live set,
    /// which is what bounds a direction reversal without discarding the tiles it
    /// is about to reverse back onto.
    pub const DEFAULT_RESIDENT_TILE_BUDGET: usize = 256;

    /// Which retention a boundary holding `primitive_count` primitives gets.
    pub const fn retention_for(&self, primitive_count: usize) -> Retention {
        if primitive_count > self.rasterize_above {
            Retention::Texture
        } else {
            Retention::Primitives
        }
    }
}

impl Default for BoundaryPolicy {
    fn default() -> Self {
        Self {
            rasterize_above: Self::DEFAULT_RASTERIZE_ABOVE,
            buffering: Buffering::default(),
            evict_after_frames: Self::DEFAULT_EVICT_AFTER_FRAMES,
            resident_tile_budget: Self::DEFAULT_RESIDENT_TILE_BUDGET,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bare_boundary_buffers_half_a_viewport_on_each_axis() {
        let policy = BoundaryPolicy::default();
        let viewport = Size::pixels(800.0, 600.0);
        assert_eq!(
            policy.buffering.margin(viewport),
            Size::pixels(400.0, 300.0)
        );
    }

    #[test]
    fn an_explicit_margin_is_used_verbatim() {
        let buffering = Buffering::Margin(Some(Size::pixels(32.0, 64.0)));
        assert_eq!(
            buffering.margin(Size::pixels(800.0, 600.0)),
            Size::pixels(32.0, 64.0)
        );
    }

    #[test]
    fn opting_out_of_buffering_asks_for_no_margin() {
        assert_eq!(
            Buffering::None.margin(Size::pixels(800.0, 600.0)),
            Size::ZERO
        );
    }

    /// Phase 2 left this variant reporting `Margin(None)`'s auto margin with an
    /// `is_implemented()` flag saying so. Phase 4.5 closes both: the margin is
    /// now the retain radius in pixels, which is what the mechanism keeps.
    #[test]
    fn a_tiled_boundarys_margin_is_its_retain_radius_in_pixels() {
        let tiled = Buffering::Tiled {
            tile_size: Size::pixels(256.0, 256.0),
            retain_radius: 2,
        };
        assert_eq!(
            tiled.margin(Size::pixels(800.0, 600.0)),
            Size::pixels(512.0, 512.0)
        );
        assert_eq!(tiled.retain_radius(), 2);
        assert!(tiled.tile_grid().is_some());

        let unbuffered = Buffering::Tiled {
            tile_size: Size::pixels(256.0, 256.0),
            retain_radius: 0,
        };
        assert_eq!(unbuffered.margin(Size::pixels(800.0, 600.0)), Size::ZERO);
    }

    #[test]
    fn a_tile_size_no_grid_can_be_built_from_falls_back_to_the_auto_margin() {
        // The fallback the Phase 2 placeholder's reasoning was right about, kept
        // for the one case that still needs it: such a boundary really is
        // buffered untiled, so reporting zero would still hand a boundary that
        // asked for more buffering none at all.
        let broken = Buffering::Tiled {
            tile_size: Size::pixels(0.0, 0.0),
            retain_radius: 4,
        };
        assert_eq!(
            broken.margin(Size::pixels(800.0, 600.0)),
            Buffering::Margin(None).margin(Size::pixels(800.0, 600.0))
        );
        assert!(broken.tile_grid().is_none());
        assert!(Buffering::default().tile_grid().is_none());
    }

    #[test]
    fn retention_follows_the_primitive_count_across_the_threshold() {
        let policy = BoundaryPolicy::default();
        assert_eq!(policy.retention_for(12), Retention::Primitives);
        assert_eq!(
            policy.retention_for(BoundaryPolicy::DEFAULT_RASTERIZE_ABOVE),
            Retention::Primitives
        );
        assert_eq!(
            policy.retention_for(BoundaryPolicy::DEFAULT_RASTERIZE_ABOVE + 1),
            Retention::Texture
        );
    }
}
