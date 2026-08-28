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

impl AtlasKind {
    /// Bytes one texel of this kind occupies, in a CPU-side page buffer and in
    /// the GPU texture it is uploaded to.
    ///
    /// One for a coverage mask (`R8Unorm`) and four for colour (`Rgba8Unorm`) —
    /// the same two formats the legacy `WgpuAtlas::push_texture` maps its two
    /// `AtlasTextureKind`s onto. It lives here rather than in `wgpui-wgpu`
    /// because the crate that *rasterises* has to produce bytes of this width
    /// and cannot see a `wgpu::TextureFormat` (§3.3, §3.5).
    pub const fn bytes_per_pixel(self) -> u32 {
        match self {
            AtlasKind::Monochrome => 1,
            AtlasKind::Polychrome => 4,
        }
    }
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
/// - `scale_factor_bits` because the rasteriser turns a sub-pixel *variant* back
///   into a sub-pixel *offset* by dividing by the device-pixel ratio (the legacy
///   `CosmicTextSystemState::rasterize_glyph` does exactly this), so two
///   requests agreeing on every other field but not on the scale factor produce
///   two different bitmaps. Added in Phase 5.5, when building the rasteriser
///   made the dependency real: Phase 5 folded the scale into `font_size_bits`,
///   which is right for the *size* and silently wrong for the *offset* — a
///   16px glyph at 2× and a 32px glyph at 1× are the same device size and
///   different rasters, and the legacy `RenderGlyphParams` hashes the two
///   fields separately for the same reason.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct GlyphRasterKey {
    /// The face the outline comes from, as the shaper numbers its faces.
    pub font: u32,
    /// The font-local glyph index.
    pub glyph: u32,
    /// Bit pattern of the pixel size the glyph is rasterised at — already
    /// multiplied by the scale factor, so this is a device-pixel size.
    pub font_size_bits: u32,
    /// Quantised sub-pixel position, `[x, y]`.
    pub subpixel: [u8; 2],
    /// Bit pattern of the device-pixel ratio the raster was requested at.
    pub scale_factor_bits: u32,
    /// Which atlas the raster belongs in.
    pub kind: AtlasKind,
}

/// The exact identity of one decoded image frame, at one scale.
///
/// The image counterpart of [`GlyphRasterKey`], and the same reasoning about
/// fields rather than a hash applies: an atlas keyed by a digest is one
/// collision away from drawing the wrong picture. Three fields, each part of the
/// identity for a reason:
///
/// - `source` names the resource. It is [`crate::patch::primitive::AtlasTileId`]-
///   opaque on purpose — a path, a URI, an embedded asset, an in-memory
///   registration — because how a source is named is the image cache's business
///   and not the atlas's. A source that is reloaded is issued a new id rather
///   than mutating in place, which is what makes comparing identity rather than
///   content sound (the same argument `wgpui_widgets::img::ImageSourceId` makes
///   for the reconciliation key, one level up).
/// - `frame_index` because an animated source's frames are separate bitmaps that
///   are legitimately resident at the same time — a GIF that has looped once has
///   every frame in the atlas and cycles between tiles rather than re-uploading.
/// - `scale_factor_bits` because a bitmap decoded for a 2× display is a
///   different bitmap from the same source at 1×, exactly as a glyph raster is.
///   Phase 6.2 decodes at 1× only and this field is what makes adding the second
///   scale a decode-side change rather than an atlas-side one.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ImageRasterKey {
    /// Which resource the bitmap came from.
    pub source: u64,
    /// Which frame of that resource. `0` for a still image.
    pub frame_index: u32,
    /// Bit pattern of the device-pixel ratio the bitmap was produced at.
    pub scale_factor_bits: u32,
}

/// What a tile in the atlas holds: a glyph's raster, or an image's bitmap.
///
/// # Why the two share one map and one page numbering
///
/// They already share the pages. [`AtlasKind::Polychrome`] is the format a
/// colour emoji and a PNG both need, and Phase 5.5 made the allocator kind-aware
/// rather than glyph-aware precisely so a second producer would not need a
/// second allocator. What was missing was a *name* a non-glyph tile could be
/// looked up by, and this is it — the legacy atlas draws the same line, with an
/// `AtlasKey::Image` variant beside its glyph one (`src/platform.rs`).
///
/// Sharing the numbering matters for eviction specifically: a page index has to
/// identify a page globally, or an [`AtlasEviction::Page`] would be ambiguous
/// and every subscriber would have to be told which producer it meant.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AtlasKey {
    /// One rasterised glyph.
    Glyph(GlyphRasterKey),
    /// One decoded image frame.
    Image(ImageRasterKey),
}

impl AtlasKey {
    /// Which atlas this key's texels belong in.
    ///
    /// An image is always [`AtlasKind::Polychrome`]: a decoded frame is RGBA
    /// whether or not the source had colour in it, because the alpha channel is
    /// not optional and a coverage page has no room for it.
    pub const fn kind(self) -> AtlasKind {
        match self {
            AtlasKey::Glyph(glyph) => glyph.kind,
            AtlasKey::Image(_) => AtlasKind::Polychrome,
        }
    }
}

impl From<GlyphRasterKey> for AtlasKey {
    fn from(key: GlyphRasterKey) -> Self {
        AtlasKey::Glyph(key)
    }
}

impl From<ImageRasterKey> for AtlasKey {
    fn from(key: ImageRasterKey) -> Self {
        AtlasKey::Image(key)
    }
}

/// A resident image bitmap: where its texels are and how big they are.
///
/// [`GlyphTile`] without the bearing, which is the one field that would be a
/// lie: a glyph's ink sits at an offset from its pen position, and an image has
/// no pen. Carrying a field that is always `[0.0, 0.0]` would invite a caller to
/// add it to a position and get the right answer for the wrong reason.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ImageTile {
    /// The tile's identity, which a resident
    /// [`crate::patch::primitive::PolySprite`] carries and an eviction names.
    pub tile: AtlasTileId,
    /// Top-left of the bitmap within its atlas page, in texels.
    pub atlas_origin: [f32; 2],
    /// Size of the bitmap, in texels.
    pub atlas_size: [f32; 2],
}

/// One decoded image frame's pixels: the bitmap an allocated tile is supposed to
/// hold.
///
/// The image half of [`ImageTileSource`]'s vocabulary, and it lives here for the
/// same reason [`RasterizedGlyph`] does: `wgpui-widgets` produces one (it is the
/// crate that owns the decoder) and `wgpui-wgpu` consumes one (it is the crate
/// that owns the atlas pages), and neither names the other.
///
/// # Straight alpha, stated because it is the field that gets this wrong
///
/// [`Self::texels`] is **straight** (non-premultiplied) RGBA8, which is what
/// `image`'s `into_rgba8()` produces for every still format and what the sprite
/// pipeline's `over` blend expects. A producer whose decoder emits premultiplied
/// texels — `resvg`, via `tiny_skia::Pixmap` — has to un-premultiply before
/// building one of these. That is a real difference from the legacy path, which
/// uploads the pixmap as-is and relies on a `premultiplied_alpha` shader flag it
/// only sets on surfaces whose composite alpha mode is `PreMultiplied`; see
/// `wgpui_widgets::image_cache` and docs/phase-6.2-results.md.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RasterizedImage {
    /// The bitmap's size in texels, `[width, height]`.
    pub size: [u32; 2],
    /// The texels, row-major, four bytes each, straight alpha, no row padding.
    pub texels: Vec<u8>,
}

impl RasterizedImage {
    /// How many bytes [`Self::texels`] must hold for [`Self::size`].
    pub fn expected_texel_bytes(&self) -> usize {
        self.size[0] as usize
            * self.size[1] as usize
            * AtlasKind::Polychrome.bytes_per_pixel() as usize
    }

    /// Whether the bitmap's length agrees with its declared size.
    pub fn is_well_formed(&self) -> bool {
        self.texels.len() == self.expected_texel_bytes()
    }
}

/// Where a decoded image frame's tile comes from.
///
/// The image counterpart of [`GlyphTileSource`], separate rather than a second
/// method on it because the two seams have different producers and neither crate
/// implements both: `wgpui-text` calls the glyph one, `wgpui-widgets` calls this,
/// and `wgpui-wgpu` implements both over one allocator.
///
/// An implementation is expected to allocate on demand and cache, for the same
/// reason the glyph one is: a list of forty rows showing one avatar asks for one
/// key forty times and takes no steps to deduplicate, because the source is the
/// thing that already holds the key-to-tile map.
///
/// # Why the pixels arrive as a call-time closure and not as a constructor
/// argument
///
/// This is the one place the two seams genuinely differ, and it falls out of
/// §3.3/§3.4/§3.5's own crate split rather than from a preference.
///
/// A glyph's raster is produced by a rasteriser that can live *inside* the tile
/// source: `wgpui-wgpu` constructs one with a closure over `swash` and never
/// needs anything else. An image's pixels cannot work that way — they come from
/// the decode cache, which §3.4 puts in `wgpui-widgets`, while the tile source
/// is in the crate that owns the device (§3.5). Neither crate may name the
/// other, and neither can own the other's half, so the pull has to cross at the
/// call. Passing the decoder per call is what lets `wgpui-wgpu` allocate tiles
/// for images it has no way to decode and `wgpui-widgets` decode images it has
/// no way to upload.
pub trait ImageTileSource {
    /// The tile holding `key`'s bitmap, allocating and uploading it if needed.
    ///
    /// `decode` is called **only on a miss**, which is the whole reason it is a
    /// closure rather than a value: decoding an image frame is megabytes of
    /// work, and a resident key must never pay it. An implementation that calls
    /// `decode` unconditionally is correct and unusably slow.
    ///
    /// `None` means "this sprite draws nothing" — the source has not decoded
    /// yet, the decode failed, or the atlas refused the bitmap. All three are
    /// ordinary and produce a positioned sprite carrying
    /// [`AtlasTileId::NONE`], never a dropped sprite: an image that is still
    /// loading occupies its layout box and its slab slot exactly as it will once
    /// it arrives.
    fn tile_for(
        &mut self,
        key: ImageRasterKey,
        decode: &mut dyn FnMut(ImageRasterKey) -> Option<RasterizedImage>,
    ) -> Option<ImageTile>;
}

/// A resident glyph raster: where its texels are, and where they go relative to
/// the pen.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GlyphTile {
    /// The tile's identity, which a resident [`crate::patch::primitive::Glyph`]
    /// carries and an eviction names.
    pub tile: AtlasTileId,
    /// Top-left of the raster within its atlas page, in texels.
    pub atlas_origin: [f32; 2],
    /// Size of the raster, in texels.
    pub atlas_size: [f32; 2],
    /// Offset from the pen position to the raster's top-left, in pixels.
    ///
    /// Shaping gives a glyph's *pen* position; a raster's ink sits at some
    /// offset from it (a comma's ink hangs below the baseline, a capital's sits
    /// well above). That offset is a property of the rasterised bitmap, which is
    /// why it arrives from the tile source rather than from shaping.
    pub bearing: [f32; 2],
}

/// One glyph's rasterised bitmap: the pixels an allocated tile is supposed to
/// hold, plus where they sit relative to the pen.
///
/// The other half of [`GlyphTileSource`]'s vocabulary, and it lives here for the
/// same reason the trait does. `wgpui-text` produces one of these (it is the
/// crate that owns `cosmic-text`, and therefore `swash`); `wgpui-wgpu` consumes
/// one (it is the crate that owns the atlas pages the texels are copied into).
/// Neither names the other, so the type they agree on is here.
///
/// Phase 5 left this half of the seam as a closure parameter with no type behind
/// it, because nothing rasterised. Phase 5.5 is that closure's implementation.
#[derive(Clone, Debug, PartialEq)]
pub struct RasterizedGlyph {
    /// The bitmap's size in texels, `[width, height]`.
    pub size: [u32; 2],
    /// Which atlas the texels belong in, and therefore how wide a texel is.
    pub kind: AtlasKind,
    /// Offset from the pen position to the bitmap's top-left, in pixels.
    ///
    /// The legacy `glyph_raster_bounds` returns exactly this as its bounds
    /// origin: `(placement.left, -placement.top)`.
    pub bearing: [f32; 2],
    /// The texels, row-major, tightly packed at [`AtlasKind::bytes_per_pixel`]
    /// bytes each. No row padding: alignment is a GPU-upload concern and is
    /// applied at the copy, exactly as the legacy `WgpuAtlasState::upload_texture`
    /// does it.
    pub texels: Vec<u8>,
}

impl RasterizedGlyph {
    /// How many bytes [`Self::texels`] must hold for [`Self::size`] and
    /// [`Self::kind`].
    pub fn expected_texel_bytes(&self) -> usize {
        self.size[0] as usize * self.size[1] as usize * self.kind.bytes_per_pixel() as usize
    }

    /// Whether the bitmap's length agrees with its declared size and kind.
    ///
    /// Checked rather than assumed at every boundary this crosses: a bitmap
    /// whose length disagrees with its size would be blitted into an atlas page
    /// row by row, and a short one would silently take texels from the next
    /// glyph's row.
    pub fn is_well_formed(&self) -> bool {
        self.texels.len() == self.expected_texel_bytes()
    }
}

/// Where a shaped glyph's raster comes from.
///
/// §6's accounting, made into a trait: "`wgpui-text` produces glyph positions
/// and atlas tile *requests*; `wgpui-wgpu`'s atlas allocator turns requests into
/// actual tile coordinates; neither owns the other's job." This is the seam
/// between those two sentences, and it lives in `wgpui-core` so that it costs no
/// dependency edge in either direction — `wgpui-text` calls it, `wgpui-wgpu`
/// implements it, and neither crate names the other.
///
/// An implementation is expected to rasterise on demand and cache: the caller
/// asks for the same key many times per frame (every `e` in a paragraph is one
/// key) and takes no steps to deduplicate, because the source is the thing that
/// already has to hold a map from key to tile.
pub trait GlyphTileSource {
    /// The tile holding `key`'s raster, rasterising and allocating if needed.
    ///
    /// `None` means "this glyph draws nothing" — whitespace, a zero-coverage
    /// control character, or a raster the atlas refused. All three are ordinary
    /// and produce a positioned glyph carrying [`AtlasTileId::NONE`], never a
    /// dropped glyph: `line_layout`'s index-to-position mapping counts on every
    /// shaped glyph being present.
    fn tile_for(&mut self, key: GlyphRasterKey) -> Option<GlyphTile>;
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
    /// Every live layer holding a glyph or sprite whose raster is in `evicted`,
    /// in ascending layer order.
    ///
    /// A pure query: it reports what would be affected without changing
    /// anything, so a caller can log or assert against it before acting.
    ///
    /// **Both tile-referencing kinds are scanned.** Phase 6.2 added
    /// [`crate::patch::primitive::PolySprite`], which references a colour tile
    /// exactly as a [`crate::patch::primitive::Glyph`] references a coverage
    /// one, and an eviction it did not see is the same stale-texels bug this
    /// module's doc is about — an image sprite left pointing at a freed
    /// rectangle draws whatever was allocated over it, with nothing to notice.
    /// The scan is what makes that hard to get wrong: adding a kind that holds
    /// tiles and forgetting this function is a missing clause here, not a
    /// missing update site scattered across the patch path.
    pub fn layers_referencing(&self, evicted: AtlasEviction) -> Vec<LayerId> {
        let mut affected = Vec::new();
        for layer in self.layers.ids() {
            let glyphs = self
                .glyph_runs
                .keys(layer)
                .into_iter()
                .filter_map(|key| self.glyph_runs.get(layer, key))
                .flat_map(|run| run.atlas_tiles())
                .any(|tile| evicted.covers(tile));
            let sprites = self
                .poly_sprites
                .keys(layer)
                .into_iter()
                .filter_map(|key| self.poly_sprites.get(layer, key))
                .filter_map(|sprite| sprite.atlas_tile())
                .any(|tile| evicted.covers(tile));
            if glyphs || sprites {
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

    /// The Phase 6.2 half of the same subscription: an image sprite is a tile
    /// reference too, and an eviction it did not see is the same bug.
    #[test]
    fn evicting_a_page_invalidates_the_layers_whose_sprites_reference_it()
    -> Result<(), PatchError> {
        use crate::patch::primitive::PolySprite;

        let mut scene = Scene::new();
        let with_image = scene.layer(LayerKey::untiled(BoundaryId::from_raw(1)));
        let other_image = scene.layer(LayerKey::untiled(BoundaryId::from_raw(2)));
        let no_tile = scene.layer(LayerKey::untiled(BoundaryId::from_raw(3)));

        let mut patch = ScenePatch::new();
        patch.poly_sprites.append(
            with_image,
            RecordKey::from_raw(1),
            0,
            PolySprite {
                atlas_tile: tile(5, 1),
                ..PolySprite::ZERO
            },
        );
        patch.poly_sprites.append(
            other_image,
            RecordKey::from_raw(2),
            0,
            PolySprite {
                atlas_tile: tile(6, 1),
                ..PolySprite::ZERO
            },
        );
        // A sprite whose image has not decoded yet: a slot, no tile reference.
        patch
            .poly_sprites
            .append(no_tile, RecordKey::from_raw(3), 0, PolySprite::ZERO);
        apply(&mut scene, &patch)?;
        for layer in [with_image, other_image, no_tile] {
            scene.layers.mark_clean(layer);
        }

        assert_eq!(scene.evict_atlas(AtlasEviction::Page(5)), vec![with_image]);
        assert_eq!(
            scene.layers.get(with_image).map(|l| l.invalidation()),
            Some(Invalidation::DISPLAY)
        );
        assert!(scene.layers.get(other_image).is_some_and(|l| l.is_clean()));
        assert!(scene.layers.get(no_tile).is_some_and(|l| l.is_clean()));

        assert_eq!(
            scene.evict_atlas(AtlasEviction::Tile(tile(6, 1))),
            vec![other_image]
        );
        assert_eq!(
            scene.evict_atlas(AtlasEviction::Tile(AtlasTileId::NONE)),
            vec![],
            "a sprite with no tile must never subscribe to an eviction"
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
