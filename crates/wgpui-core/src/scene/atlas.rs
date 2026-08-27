//! Atlas-eviction subscription for GPU-resident glyph tiles — R-N §4.3's
//! last unaddressed hazard, closed here. See docs/gpu-native-architecture.md
//! §8 (Phase 5) and docs/retained-layers.md §4.3.
//!
//! # The hazard, quoted
//!
//! > **Atlas tile references.** Sprites carry `tile.tile_id` into the sort key.
//! > A retained slab holds tile references that the atlas may evict. Layers must
//! > subscribe to atlas eviction and take `DISPLAY` when a tile they reference
//! > is dropped — this is the same hazard `force_render` handles today after
//! > device recovery.
//!
//! Under an immediate-mode renderer this cannot bite: every frame re-records
//! every sprite, so a tile freed between frames is simply re-requested. The
//! whole point of a persistent slab is that it does *not* do that — a layer
//! that reconciled clean keeps last frame's bytes, tile coordinates included,
//! and if the allocator has since handed those texels to a different glyph the
//! layer draws the wrong picture with no error anywhere. Nothing about the
//! layer changed, so nothing invalidates it. That is why the subscription has
//! to exist and why it has to run from the *atlas* side.
//!
//! Phase 5 is the first phase where this is a real hazard rather than a
//! theoretical one, because it is the first phase in which anything in 2.0
//! actually references an atlas tile.
//!
//! # Why this scans rather than maintaining an index
//!
//! The obvious implementation is a `HashMap<AtlasTileId, HashSet<LayerId>>`
//! updated on every patch. This does not do that, on purpose:
//!
//! - **An index can drift; a scan cannot.** The index would have to be updated
//!   on insert, on update, on remove, on layer removal, and on slab relocation,
//!   and a missed update is silent — it produces exactly the stale-texels bug
//!   the mechanism exists to prevent, only now with a mechanism in place that
//!   claims to prevent it. Deriving the answer from the resident primitives
//!   makes "what does this layer reference" true by construction.
//! - **The cost lands where there is room for it.** A scan is `O(resident
//!   glyphs)` and runs once per eviction event; an index costs a little on
//!   every patch, which is the per-frame path §5.0 spends its whole design
//!   budget keeping cheap. Evictions happen when an atlas page fills or a
//!   device is lost — rare, and already expensive on the GPU side.
//!
//! If a workload ever shows this scan on a profile, the index is a
//! self-contained change behind [`Scene::layers_referencing`]. It is not one
//! today, and building it now would be paying a real maintenance cost against a
//! measurement nobody has taken.

use crate::invalidation::axes::Invalidation;
use crate::patch::primitive::AtlasTileId;
use crate::scene::layer::LayerId;
use crate::scene::Scene;

/// Which atlas a raster belongs in.
///
/// Not a rendering detail that could live in `wgpui-wgpu`: it decides which
/// texture format a tile is allocated out of, and the crate that *shapes* the
/// text is the only one that knows whether a glyph came from a colour emoji
/// face. So the vocabulary is shared here, where both sides can name it, and
/// neither side has to depend on the other (§3.3, §3.5).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AtlasKind {
    /// Single-channel coverage: ordinary text.
    #[default]
    Monochrome,
    /// Full colour: emoji, and image sprites.
    Polychrome,
}

/// The exact identity of one rasterised glyph.
///
/// # Why the fields rather than a hash
///
/// An atlas keyed by a `u64` hash is one collision away from drawing the wrong
/// glyph, silently, in a way that reproduces only for one user with one font at
/// one zoom level. The field set is small enough to compare directly, so it is
/// compared directly. Every field is part of the identity for a reason:
///
/// - `font` and `glyph` name the outline. Both are needed: glyph indices are
///   font-local, so glyph 42 means different things in different faces — and
///   fallback means one run can span faces.
/// - `font_size_bits` because a raster is resolution-specific; the same outline
///   at 12px and 13px are two different bitmaps.
/// - `subpixel` because a glyph drawn at a fractional pixel offset is a
///   different bitmap again, and quantising the offset into a small number of
///   variants is what makes text look right without a raster per position.
/// - `kind` because a colour emoji and a coverage mask are not interchangeable
///   even when everything else matches.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct GlyphRasterKey {
    /// The face the outline comes from, as the shaper numbers its faces.
    pub font: u32,
    /// The font-local glyph index.
    pub glyph: u32,
    /// Bit pattern of the pixel size the glyph is rasterised at.
    pub font_size_bits: u32,
    /// Quantised sub-pixel position, `[x, y]`.
    pub subpixel: [u8; 2],
    /// Which atlas the raster belongs in.
    pub kind: AtlasKind,
}

/// What the atlas allocator dropped.
///
/// Two granularities because the allocator has two: a whole page is destroyed
/// when the atlas shrinks or the device is recovered, and individual tiles are
/// freed out of pages that stay live when a glyph ages out. The legacy atlas
/// reports both through one channel for the same reason — see
/// `WgpuAtlas::drain_destroyed_pages`, whose doc names "fully-destroyed textures
/// and tiles freed out of still-live pages alike."
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AtlasEviction {
    /// One tile was freed; its page is still live and its other tiles are still
    /// valid.
    Tile(AtlasTileId),
    /// A whole page was destroyed; every tile in it is invalid.
    Page(u32),
}

impl AtlasEviction {
    /// Whether this eviction invalidates `tile`.
    pub fn covers(self, tile: AtlasTileId) -> bool {
        match self {
            // `AtlasTileId::NONE` is not a tile reference and can never be
            // evicted: a glyph with no raster (whitespace) has nothing to lose.
            AtlasEviction::Tile(evicted) => !evicted.is_none() && evicted == tile,
            AtlasEviction::Page(page) => tile.page() == Some(page),
        }
    }
}

impl Scene {
    /// Every live layer holding a glyph whose raster is in `evicted`, in
    /// ascending layer order.
    ///
    /// A pure query: it reports what would be affected without changing
    /// anything, so a caller can log or assert against it before acting.
    pub fn layers_referencing(&self, evicted: AtlasEviction) -> Vec<LayerId> {
        let mut affected = Vec::new();
        for layer in self.layers.ids() {
            let references = self
                .glyph_runs
                .keys(layer)
                .into_iter()
                .filter_map(|key| self.glyph_runs.get(layer, key))
                .flat_map(|run| run.atlas_tiles())
                .any(|tile| evicted.covers(tile));
            if references {
                affected.push(layer);
            }
        }
        affected
    }

    /// Take `DISPLAY` on every layer referencing an evicted tile, and report
    /// which ones.
    ///
    /// `DISPLAY` and not [`Invalidation::all`]: nothing about the layer's
    /// layout, hit geometry, or composite position changed — its glyphs are in
    /// the same places, at the same sizes, and only the texels those glyphs
    /// point at are gone. Re-emitting the run re-requests a tile and rewrites
    /// the same slots. Over-invalidating here would turn a rare atlas event into
    /// a full relayout of every text-bearing layer on screen, which is exactly
    /// the sledgehammer `force_render` is in the legacy backend and exactly what
    /// R-N's axis vocabulary exists to avoid.
    ///
    /// Returns the affected layers rather than a count so a caller can act on
    /// them — and so a test can assert on identity rather than on arithmetic.
    pub fn evict_atlas(&mut self, evicted: AtlasEviction) -> Vec<LayerId> {
        let affected = self.layers_referencing(evicted);
        for layer in &affected {
            self.layers.invalidate(*layer, Invalidation::DISPLAY);
        }
        affected
    }

    /// Apply several evictions at once, reporting the union of affected layers
    /// in ascending order with no duplicates.
    ///
    /// The allocator drains a batch of events per frame, so this is the shape
    /// callers actually want; doing it in one pass also means a layer touched by
    /// two evictions is invalidated once.
    pub fn evict_atlas_batch(
        &mut self,
        evictions: impl IntoIterator<Item = AtlasEviction>,
    ) -> Vec<LayerId> {
        let mut affected: Vec<LayerId> = Vec::new();
        for eviction in evictions {
            for layer in self.evict_atlas(eviction) {
                if !affected.contains(&layer) {
                    affected.push(layer);
                }
            }
        }
        affected.sort_unstable();
        affected
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patch::apply::{ScenePatch, apply};
    use crate::patch::primitive::{Glyph, GlyphRun, Quad};
    use crate::patch::{PatchError, RecordKey};
    use crate::scene::layer::{BoundaryId, LayerKey};

    fn tile(page: u32, slot: u32) -> AtlasTileId {
        AtlasTileId::new(page, slot).expect("test tiles are in range")
    }

    fn glyph(tile: AtlasTileId) -> Glyph {
        Glyph {
            atlas_tile: tile,
            ..Glyph::ZERO
        }
    }

    /// Three layers: one referencing page 0, one referencing page 1, one
    /// holding only quads. Returns their handles in that order.
    fn scene_with_text() -> Result<(Scene, [LayerId; 3]), PatchError> {
        let mut scene = Scene::new();
        let page_zero = scene.layer(LayerKey::untiled(BoundaryId::from_raw(1)));
        let page_one = scene.layer(LayerKey::untiled(BoundaryId::from_raw(2)));
        let quads_only = scene.layer(LayerKey::untiled(BoundaryId::from_raw(3)));

        let mut patch = ScenePatch::new();
        patch.glyph_runs.append(
            page_zero,
            RecordKey::from_raw(1),
            0,
            GlyphRun {
                color: [1.0; 4],
                glyphs: vec![glyph(tile(0, 1)), Glyph::ZERO, glyph(tile(0, 2))],
            },
        );
        patch.glyph_runs.append(
            page_one,
            RecordKey::from_raw(2),
            0,
            GlyphRun {
                color: [1.0; 4],
                glyphs: vec![glyph(tile(1, 1))],
            },
        );
        patch
            .quads
            .append(quads_only, RecordKey::from_raw(3), 0, Quad::ZERO);
        apply(&mut scene, &patch)?;
        for layer in [page_zero, page_one, quads_only] {
            scene.layers.mark_clean(layer);
        }
        Ok((scene, [page_zero, page_one, quads_only]))
    }

    /// **Phase 5's atlas-eviction regression test.** Evicting a page takes
    /// `DISPLAY` on exactly the layers that reference it — not on the layer
    /// referencing a different page, and not on the layer holding no text at
    /// all.
    #[test]
    fn evicting_a_page_invalidates_only_the_layers_referencing_it() -> Result<(), PatchError> {
        let (mut scene, [page_zero, page_one, quads_only]) = scene_with_text()?;
        for layer in [page_zero, page_one, quads_only] {
            assert!(
                scene.layers.get(layer).is_some_and(|l| l.is_clean()),
                "every layer starts the frame clean, or this test proves nothing"
            );
        }

        let affected = scene.evict_atlas(AtlasEviction::Page(0));
        assert_eq!(affected, vec![page_zero]);
        assert_eq!(
            scene.layers.get(page_zero).map(|l| l.invalidation()),
            Some(Invalidation::DISPLAY),
            "an evicted tile is a repaint, never a relayout"
        );
        assert!(scene.layers.get(page_one).is_some_and(|l| l.is_clean()));
        assert!(scene.layers.get(quads_only).is_some_and(|l| l.is_clean()));
        Ok(())
    }

    #[test]
    fn evicting_one_tile_spares_the_rest_of_its_page() -> Result<(), PatchError> {
        let (mut scene, [page_zero, page_one, _]) = scene_with_text()?;

        // Tile (0, 9) is in the same page as the layer's glyphs but is not one
        // of them, so nothing is affected.
        assert_eq!(scene.evict_atlas(AtlasEviction::Tile(tile(0, 9))), vec![]);
        assert!(scene.layers.get(page_zero).is_some_and(|l| l.is_clean()));

        assert_eq!(
            scene.evict_atlas(AtlasEviction::Tile(tile(0, 2))),
            vec![page_zero]
        );
        assert!(scene.layers.get(page_one).is_some_and(|l| l.is_clean()));
        Ok(())
    }

    #[test]
    fn a_glyph_with_no_raster_is_not_a_tile_reference() -> Result<(), PatchError> {
        let mut scene = Scene::new();
        let layer = scene.layer(LayerKey::untiled(BoundaryId::from_raw(1)));
        let mut patch = ScenePatch::new();
        patch.glyph_runs.append(
            layer,
            RecordKey::from_raw(1),
            0,
            GlyphRun {
                color: [1.0; 4],
                glyphs: vec![Glyph::ZERO, Glyph::ZERO],
            },
        );
        apply(&mut scene, &patch)?;
        scene.layers.mark_clean(layer);

        // `AtlasTileId::NONE` reports page `None`, so no page eviction may
        // match it and no tile eviction may either.
        assert_eq!(scene.evict_atlas(AtlasEviction::Page(0)), vec![]);
        assert_eq!(
            scene.evict_atlas(AtlasEviction::Tile(AtlasTileId::NONE)),
            vec![]
        );
        assert!(scene.layers.get(layer).is_some_and(|l| l.is_clean()));
        Ok(())
    }

    #[test]
    fn a_batch_invalidates_a_doubly_affected_layer_once_and_reports_it_once()
    -> Result<(), PatchError> {
        let (mut scene, [page_zero, page_one, _]) = scene_with_text()?;
        let affected = scene.evict_atlas_batch([
            AtlasEviction::Page(0),
            AtlasEviction::Tile(tile(0, 1)),
            AtlasEviction::Page(1),
        ]);
        let mut expected = vec![page_zero, page_one];
        expected.sort_unstable();
        assert_eq!(affected, expected);
        assert_eq!(
            scene.layers.get(page_zero).map(|l| l.invalidation()),
            Some(Invalidation::DISPLAY)
        );
        Ok(())
    }

    #[test]
    fn a_layer_removed_from_the_scene_stops_subscribing() -> Result<(), PatchError> {
        let (mut scene, [page_zero, _, _]) = scene_with_text()?;
        assert!(scene.remove_layer(page_zero));
        // Nothing to scan and nothing to invalidate: the residency answer comes
        // from the resident primitives, so removing them removes the
        // subscription with no separate teardown step to forget.
        assert_eq!(scene.evict_atlas(AtlasEviction::Page(0)), vec![]);
        Ok(())
    }

    #[test]
    fn re_emitting_a_run_with_new_tiles_moves_the_subscription() -> Result<(), PatchError> {
        let (mut scene, [page_zero, _, _]) = scene_with_text()?;

        let mut patch = ScenePatch::new();
        patch.glyph_runs.update(
            page_zero,
            RecordKey::from_raw(1),
            GlyphRun {
                color: [1.0; 4],
                glyphs: vec![glyph(tile(2, 5)), Glyph::ZERO, glyph(tile(2, 6))],
            },
        );
        apply(&mut scene, &patch)?;
        scene.layers.mark_clean(page_zero);

        assert_eq!(
            scene.evict_atlas(AtlasEviction::Page(0)),
            vec![],
            "the layer no longer references page 0"
        );
        assert_eq!(scene.evict_atlas(AtlasEviction::Page(2)), vec![page_zero]);
        Ok(())
    }
}
