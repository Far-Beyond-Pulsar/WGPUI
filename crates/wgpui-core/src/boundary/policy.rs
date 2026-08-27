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
    /// **Not implemented in Phase 2.** A boundary declaring this is buffered as
    /// `Margin(None)` today; see [`Buffering::is_implemented`].
    Tiled {
        /// Edge length of one tile.
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
    /// [`Buffering::Tiled`] reports the auto margin rather than a tile-derived
    /// one: its own mechanism does not exist yet (Phase 4.5), and reporting
    /// zero would silently make a boundary that asked for *more* buffering
    /// receive *none*.
    pub fn margin(self, viewport: Size<Pixels>) -> Size<Pixels> {
        match self {
            Buffering::None => Size::ZERO,
            Buffering::Margin(Some(margin)) => margin,
            Buffering::Margin(None) | Buffering::Tiled { .. } => {
                viewport.scaled(Self::AUTO_MARGIN_FRACTION)
            }
        }
    }

    /// Whether this variant's own mechanism is built. `false` for
    /// [`Buffering::Tiled`] until Phase 4.5, which is recorded here rather than
    /// only in prose so a caller can assert on it.
    pub const fn is_implemented(self) -> bool {
        !matches!(self, Buffering::Tiled { .. })
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
    /// returned to the pool (R-N §3.4's mark-and-sweep interval).
    pub evict_after_frames: u32,
}

impl BoundaryPolicy {
    /// R-N §3.3's default threshold, unchanged.
    pub const DEFAULT_RASTERIZE_ABOVE: usize = 256;

    /// R-N §3.4's default eviction interval, unchanged.
    pub const DEFAULT_EVICT_AFTER_FRAMES: u32 = 60;

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

    #[test]
    fn tiled_is_declared_but_not_implemented_in_this_phase() {
        let tiled = Buffering::Tiled {
            tile_size: Size::pixels(256.0, 256.0),
            retain_radius: 1,
        };
        assert!(!tiled.is_implemented());
        assert!(Buffering::default().is_implemented());
        assert_eq!(
            tiled.margin(Size::pixels(800.0, 600.0)),
            Buffering::Margin(None).margin(Size::pixels(800.0, 600.0)),
            "an unbuilt variant must fall back to more buffering, never to none"
        );
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
