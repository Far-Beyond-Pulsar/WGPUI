//! The glyph/sprite atlas allocator — today's `src/platform/cross/atlas.rs`
//! bin-packing half, ported. See docs/gpu-native-architecture.md §3.5, §8
//! (Phase 5), and docs/retained-layers.md §4.3.
//!
//! # The split this file draws, and why it is where it is
//!
//! §6's accounting says `wgpui-text` produces glyph positions and atlas tile
//! *requests*, and this crate turns requests into tile coordinates. That is the
//! whole contract, and it means the allocator is pure bin-packing over integer
//! rectangles: [`GlyphAtlas`] opens no device, creates no texture, and uploads
//! no texels. Those are real work and they belong in this crate too — the
//! legacy file does texture creation, `write_texture` upload batching, and
//! reference-counted page destruction in the same 539 lines — but they are
//! *separable* work, and separating them buys something specific: every
//! assertion in this module's tests runs headlessly, on any machine, with no
//! adapter. An atlas whose packing decisions can only be checked on hardware is
//! an atlas whose packing decisions do not get checked.
//!
//! `etagere`'s `BucketedAtlasAllocator` is the same allocator the legacy file
//! uses, at the same version the root crate pins, so packing behaviour is
//! shared rather than re-derived.
//!
//! # What is deliberately not here
//!
//! Texture creation, upload batching, and the `Monochrome`/`Polychrome`
//! `wgpu::TextureFormat` mapping. The legacy file has all three and they are a
//! mechanical move once something in 2.0 actually draws a glyph — which nothing
//! does yet, because there is no sprite pipeline (`render/pipelines.rs` names
//! its own unbuilt work). Writing the upload path now would mean writing it
//! against an imagined consumer. Named here rather than left as a silent gap.

use std::collections::HashMap;
use wgpui_core::patch::primitive::AtlasTileId;
use wgpui_core::scene::atlas::{AtlasEviction, AtlasKind, GlyphRasterKey};

/// Side length of a page, in texels, when the atlas opens one.
///
/// The legacy `DEFAULT_ATLAS_SIZE` is 1024×1024 and this matches it. A page
/// holds on the order of 65,000 typical text glyph rasters at that size, so a
/// text-heavy window lives comfortably in one page and never exercises the
/// multi-page path — which is exactly why the multi-page path needs tests that
/// force it rather than tests that hope to reach it.
pub const DEFAULT_PAGE_SIZE: u32 = 1024;

/// Where a raster ended up.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TilePlacement {
    /// The tile's identity, as a resident [`wgpui_core::patch::primitive::Glyph`]
    /// carries it.
    pub tile: AtlasTileId,
    /// Which atlas it was allocated out of.
    pub kind: AtlasKind,
    /// Top-left of the raster within its page, in texels.
    pub origin: [f32; 2],
    /// Size of the raster, in texels.
    pub size: [f32; 2],
    /// Offset from the pen position to the raster's top-left, in pixels.
    ///
    /// Held here rather than recomputed by the caller because it is a property
    /// of the rasterised bitmap, and the atlas is already what remembers that
    /// the bitmap exists — so a cache hit answers with it too, rather than
    /// forcing a re-rasterise to recover a number nothing else knows.
    pub bearing: [f32; 2],
}

/// An allocation could not be made.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AtlasError {
    /// The raster is larger than a whole page, so no page can ever hold it.
    ///
    /// The legacy allocator grows the page to fit in this case
    /// (`min_size.max(&DEFAULT_ATLAS_SIZE)`). This reports instead, because
    /// growing means a page whose size is a function of the largest thing ever
    /// put in it, and a caller that hands the atlas a 4000px raster by accident
    /// gets a 4000px page rather than an error. A caller that genuinely wants a
    /// bigger page asks for one at construction.
    TooLargeForAPage {
        /// The size that was requested.
        requested: [u32; 2],
        /// The page side length this atlas uses.
        page_size: u32,
    },
    /// A zero-area raster was requested.
    ///
    /// Not an allocation: a glyph with no coverage carries
    /// [`AtlasTileId::NONE`] and never reaches the atlas at all. Reaching it
    /// with one means a caller confused "no raster" with "an empty raster".
    EmptyRaster,
    /// More than 255 pages, or more than 16,777,215 live tiles in one page —
    /// the limits [`AtlasTileId`]'s packing imposes.
    ///
    /// Unreachable in practice (255 pages of 1024² is 267 megatexels), and
    /// reported rather than wrapped because a wrapped tile id aliases a live
    /// one, which is the corruption R-N §4.3's hazard is about.
    OutOfTileIds,
}

impl std::fmt::Display for AtlasError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AtlasError::TooLargeForAPage {
                requested: [width, height],
                page_size,
            } => write!(
                formatter,
                "a {width}x{height} raster does not fit a {page_size}x{page_size} atlas page"
            ),
            AtlasError::EmptyRaster => {
                formatter.write_str("a zero-area raster has no tile to allocate")
            }
            AtlasError::OutOfTileIds => {
                formatter.write_str("the atlas has no tile ids left to issue")
            }
        }
    }
}

impl std::error::Error for AtlasError {}

/// What an atlas currently holds.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct AtlasStats {
    /// Pages currently open.
    pub pages: usize,
    /// Rasters currently resident.
    pub tiles: usize,
    /// Requests answered from an already-resident tile.
    pub cache_hits: u64,
    /// Requests that had to allocate.
    pub allocations: u64,
    /// Tiles freed, individually or by page destruction.
    pub evictions: u64,
}

struct Page {
    index: u32,
    kind: AtlasKind,
    allocator: etagere::BucketedAtlasAllocator,
    /// `etagere`'s handle for each of our slots, so a slot can be deallocated.
    ///
    /// Indexed by slot rather than mapped, because slots are dense: they are
    /// handed out from `free_slots` first and only then from the tail.
    allocations: Vec<Option<etagere::AllocId>>,
    free_slots: Vec<u32>,
    live: usize,
}

impl Page {
    /// Take a slot number, reusing a freed one if there is one.
    ///
    /// Reuse rather than a monotonic counter, so the 24-bit slot field bounds
    /// *peak live tiles* rather than total allocations over the process's
    /// lifetime — the difference between a limit no atlas can reach and one a
    /// long-running editor reaches in an afternoon.
    fn take_slot(&mut self) -> Option<u32> {
        match self.free_slots.pop() {
            Some(slot) => Some(slot),
            None => u32::try_from(self.allocations.len()).ok(),
        }
    }
}

/// The CPU-side atlas: rasters in, tile coordinates out, evictions reported.
///
/// Two independent atlases in one type, one per [`AtlasKind`], because a
/// coverage mask and a colour bitmap cannot share a texture format. They share
/// the [`AtlasTileId`] page numbering, so a page index identifies a page
/// globally and an eviction never has to say which kind it meant.
pub struct GlyphAtlas {
    page_size: u32,
    pages: Vec<Page>,
    /// The next page index to issue.
    ///
    /// Monotonic, never `pages.len()`. A destroyed page is removed from
    /// `pages`, so numbering from the length would hand a *new* page the
    /// destroyed one's index — and a retained slab still holding a tile in the
    /// old page would silently start sampling the new page's real texels
    /// instead of being caught by its eviction. That is the same
    /// stale-reference failure R-N §4.3 is about, reintroduced one level down,
    /// and a test caught this counter being missing rather than a review
    /// finding it.
    next_page_index: u32,
    tiles_by_key: HashMap<GlyphRasterKey, TilePlacement>,
    keys_by_tile: HashMap<AtlasTileId, GlyphRasterKey>,
    pending_evictions: Vec<AtlasEviction>,
    stats: AtlasStats,
}

impl Default for GlyphAtlas {
    fn default() -> Self {
        Self::new(DEFAULT_PAGE_SIZE)
    }
}

impl GlyphAtlas {
    /// An atlas whose pages are `page_size` texels on a side.
    pub fn new(page_size: u32) -> Self {
        Self {
            page_size,
            pages: Vec::new(),
            next_page_index: 0,
            tiles_by_key: HashMap::new(),
            keys_by_tile: HashMap::new(),
            pending_evictions: Vec::new(),
            stats: AtlasStats::default(),
        }
    }

    /// What this atlas currently holds.
    pub fn stats(&self) -> AtlasStats {
        AtlasStats {
            pages: self.pages.len(),
            tiles: self.tiles_by_key.len(),
            ..self.stats
        }
    }

    /// Where `key`'s raster lives, if it is resident.
    ///
    /// The read-only half of [`Self::get_or_insert`], for a caller that wants to
    /// know whether a request would allocate without making it do so.
    pub fn get(&self, key: &GlyphRasterKey) -> Option<TilePlacement> {
        self.tiles_by_key.get(key).copied()
    }

    /// Where `key`'s raster lives, allocating a tile for it if it is not
    /// resident.
    ///
    /// `size` is the raster's dimensions in texels; the caller has already
    /// rasterised, or is about to, and this decides where the texels go.
    pub fn get_or_insert(
        &mut self,
        key: GlyphRasterKey,
        metrics: RasterMetrics,
    ) -> Result<TilePlacement, AtlasError> {
        if let Some(placement) = self.tiles_by_key.get(&key) {
            self.stats.cache_hits += 1;
            return Ok(*placement);
        }

        let size = metrics.size;
        let [width, height] = size;
        if width == 0 || height == 0 {
            return Err(AtlasError::EmptyRaster);
        }
        if width > self.page_size || height > self.page_size {
            return Err(AtlasError::TooLargeForAPage {
                requested: size,
                page_size: self.page_size,
            });
        }

        let placement = self.allocate(key, metrics)?;
        self.tiles_by_key.insert(key, placement);
        self.keys_by_tile.insert(placement.tile, key);
        self.stats.allocations += 1;
        Ok(placement)
    }

    fn allocate(
        &mut self,
        key: GlyphRasterKey,
        metrics: RasterMetrics,
    ) -> Result<TilePlacement, AtlasError> {
        let size = metrics.size;
        let requested = etagere::size2(
            i32::try_from(size[0]).map_err(|_| AtlasError::TooLargeForAPage {
                requested: size,
                page_size: self.page_size,
            })?,
            i32::try_from(size[1]).map_err(|_| AtlasError::TooLargeForAPage {
                requested: size,
                page_size: self.page_size,
            })?,
        );

        // Newest page first, matching the legacy allocator's `.rev()`: the most
        // recently opened page is the one with room, so trying it first avoids
        // walking every full page on every allocation once an atlas has grown.
        for index in (0..self.pages.len()).rev() {
            let kind = self.pages[index].kind;
            if kind != key.kind {
                continue;
            }
            if let Some(placement) = self.allocate_in(index, key, metrics, requested)? {
                return Ok(placement);
            }
        }

        let index = self.open_page(key.kind)?;
        self.allocate_in(index, key, metrics, requested)?
            .ok_or(AtlasError::TooLargeForAPage {
                requested: size,
                page_size: self.page_size,
            })
    }

    fn allocate_in(
        &mut self,
        index: usize,
        key: GlyphRasterKey,
        metrics: RasterMetrics,
        requested: etagere::Size,
    ) -> Result<Option<TilePlacement>, AtlasError> {
        let size = metrics.size;
        let page_index = match self.pages.get(index) {
            Some(page) => page.index,
            None => return Ok(None),
        };
        // Reserve the slot before allocating so a page that has run out of tile
        // ids does not leak an `etagere` allocation nothing can address.
        let Some(slot) = self.pages.get_mut(index).and_then(Page::take_slot) else {
            return Err(AtlasError::OutOfTileIds);
        };
        let Some(tile) = AtlasTileId::new(page_index, slot) else {
            if let Some(page) = self.pages.get_mut(index) {
                page.free_slots.push(slot);
            }
            return Err(AtlasError::OutOfTileIds);
        };

        let Some(page) = self.pages.get_mut(index) else {
            return Ok(None);
        };
        let Some(allocation) = page.allocator.allocate(requested) else {
            page.free_slots.push(slot);
            return Ok(None);
        };

        let slot_index = slot as usize;
        if slot_index >= page.allocations.len() {
            page.allocations.resize(slot_index + 1, None);
        }
        if let Some(entry) = page.allocations.get_mut(slot_index) {
            *entry = Some(allocation.id);
        }
        page.live += 1;

        Ok(Some(TilePlacement {
            tile,
            kind: key.kind,
            origin: [
                allocation.rectangle.min.x as f32,
                allocation.rectangle.min.y as f32,
            ],
            size: [size[0] as f32, size[1] as f32],
            bearing: metrics.bearing,
        }))
    }

    fn open_page(&mut self, kind: AtlasKind) -> Result<usize, AtlasError> {
        let index = self.next_page_index;
        // Probe the packing: a page index no tile id can name is unusable, so
        // it is refused before the page is opened rather than after.
        if AtlasTileId::new(index, 0).is_none() {
            return Err(AtlasError::OutOfTileIds);
        }
        self.next_page_index = index.saturating_add(1);
        let side = i32::try_from(self.page_size).map_err(|_| AtlasError::TooLargeForAPage {
            requested: [self.page_size, self.page_size],
            page_size: self.page_size,
        })?;
        self.pages.push(Page {
            index,
            kind,
            allocator: etagere::BucketedAtlasAllocator::new(etagere::size2(side, side)),
            allocations: Vec::new(),
            free_slots: Vec::new(),
            live: 0,
        });
        Ok(self.pages.len() - 1)
    }

    /// Free one raster's tile, queueing the eviction its layers must see.
    ///
    /// Returns whether the raster was resident. The tile's texels stay in the
    /// page until something else is allocated over them, which is exactly why
    /// the eviction has to be reported: a retained slab pointing at them draws
    /// a stale glyph and then, later, a *wrong* one, with nothing in between to
    /// notice.
    pub fn evict(&mut self, key: &GlyphRasterKey) -> bool {
        let Some(placement) = self.tiles_by_key.remove(key) else {
            return false;
        };
        self.keys_by_tile.remove(&placement.tile);
        self.free_tile(placement.tile);
        self.pending_evictions
            .push(AtlasEviction::Tile(placement.tile));
        self.stats.evictions += 1;
        true
    }

    /// Destroy a whole page, freeing every raster in it.
    ///
    /// Reported as one [`AtlasEviction::Page`] rather than as one event per
    /// tile: a page can hold tens of thousands of tiles, and a layer
    /// referencing any of them needs `DISPLAY` exactly once.
    pub fn destroy_page(&mut self, page_index: u32) -> bool {
        let Some(position) = self.pages.iter().position(|page| page.index == page_index) else {
            return false;
        };
        let dropped: Vec<GlyphRasterKey> = self
            .tiles_by_key
            .iter()
            .filter(|(_, placement)| placement.tile.page() == Some(page_index))
            .map(|(key, _)| *key)
            .collect();
        let dropped_count = dropped.len();
        for key in dropped {
            if let Some(placement) = self.tiles_by_key.remove(&key) {
                self.keys_by_tile.remove(&placement.tile);
            }
        }
        // The page is removed from the list; its index is never reissued, which
        // is what `next_page_index` is for.
        self.pages.remove(position);
        self.pending_evictions
            .push(AtlasEviction::Page(page_index));
        self.stats.evictions += dropped_count as u64;
        true
    }

    fn free_tile(&mut self, tile: AtlasTileId) {
        let (Some(page_index), Some(slot)) = (tile.page(), tile.slot()) else {
            return;
        };
        let Some(page) = self.pages.iter_mut().find(|page| page.index == page_index) else {
            return;
        };
        let Some(entry) = page.allocations.get_mut(slot as usize) else {
            return;
        };
        if let Some(allocation) = entry.take() {
            page.allocator.deallocate(allocation);
            page.live = page.live.saturating_sub(1);
            page.free_slots.push(slot);
        }
    }

    /// Take the evictions that have accumulated since the last drain.
    ///
    /// Destructive, like the legacy `drain_destroyed_pages`: each event is
    /// reported once, to one subscriber. The caller hands them to
    /// [`wgpui_core::scene::Scene::evict_atlas_batch`], which is the other half
    /// of R-N §4.3's subscription.
    pub fn drain_evictions(&mut self) -> Vec<AtlasEviction> {
        std::mem::take(&mut self.pending_evictions)
    }

    /// Whether any eviction is waiting to be drained.
    pub fn has_pending_evictions(&self) -> bool {
        !self.pending_evictions.is_empty()
    }

    /// Live tiles in a page, for tests and diagnostics.
    pub fn live_tiles_in_page(&self, page_index: u32) -> Option<usize> {
        self.pages
            .iter()
            .find(|page| page.index == page_index)
            .map(|page| page.live)
    }
}

/// The dimensions and placement of one rasterised glyph bitmap.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RasterMetrics {
    /// The bitmap's size in texels.
    pub size: [u32; 2],
    /// Offset from the pen position to the bitmap's top-left, in pixels.
    pub bearing: [f32; 2],
}

/// A [`GlyphTileSource`] built from an atlas and a rasteriser.
///
/// # Why the rasteriser is a closure and not a type in this crate
///
/// Rasterising a glyph means `swash` (which `cosmic-text` already carries) plus
/// a decision about hinting, gamma, and colour-emoji handling, and the shape of
/// that decision is set by what draws the result. Nothing draws glyphs in 2.0
/// yet — `render/pipelines.rs` names the missing sprite pipeline itself — so
/// writing a rasteriser now would mean writing it against an imagined consumer
/// and then rewriting it. This closes the seam without guessing: the atlas half
/// is real and tested, the rasteriser half is a parameter, and the phase that
/// builds the sprite pipeline supplies it without touching anything here.
pub struct AtlasTileSource<'atlas, Rasterize> {
    atlas: &'atlas mut GlyphAtlas,
    rasterize: Rasterize,
}

impl<'atlas, Rasterize> AtlasTileSource<'atlas, Rasterize>
where
    Rasterize: FnMut(GlyphRasterKey) -> Option<RasterMetrics>,
{
    /// A tile source over `atlas`, rasterising with `rasterize`.
    pub fn new(atlas: &'atlas mut GlyphAtlas, rasterize: Rasterize) -> Self {
        Self { atlas, rasterize }
    }
}

impl<Rasterize> wgpui_core::scene::atlas::GlyphTileSource for AtlasTileSource<'_, Rasterize>
where
    Rasterize: FnMut(GlyphRasterKey) -> Option<RasterMetrics>,
{
    fn tile_for(&mut self, key: GlyphRasterKey) -> Option<wgpui_core::scene::atlas::GlyphTile> {
        // A resident tile answers without rasterising, which is the point of the
        // atlas holding a key map at all: a paragraph asks for the same 'e'
        // dozens of times per frame, and the caller deliberately does not
        // deduplicate (`wgpui-text`'s `patch` module says so).
        let placement = match self.atlas.get(&key) {
            Some(placement) => {
                self.atlas.stats.cache_hits += 1;
                placement
            }
            None => {
                let metrics = (self.rasterize)(key)?;
                // A refused allocation is `None`, not an error the caller has to
                // handle: one glyph failing to find atlas space degrades to a
                // blank glyph rather than failing the frame. A caller that wants
                // to know why asks `get_or_insert` directly.
                self.atlas.get_or_insert(key, metrics).ok()?
            }
        };
        Some(wgpui_core::scene::atlas::GlyphTile {
            tile: placement.tile,
            atlas_origin: placement.origin,
            atlas_size: placement.size,
            bearing: placement.bearing,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raster(width: u32, height: u32) -> RasterMetrics {
        RasterMetrics {
            size: [width, height],
            bearing: [0.0, 0.0],
        }
    }

    fn key(glyph: u32) -> GlyphRasterKey {
        GlyphRasterKey {
            font: 0,
            glyph,
            font_size_bits: 16.0f32.to_bits(),
            subpixel: [0, 0],
            scale_factor_bits: 1.0f32.to_bits(),
            kind: AtlasKind::Monochrome,
        }
    }

    #[test]
    fn a_raster_is_allocated_once_and_returned_thereafter() {
        let mut atlas = GlyphAtlas::default();
        let first = atlas.get_or_insert(key(1), raster(8, 12)).expect("allocate");
        let second = atlas.get_or_insert(key(1), raster(8, 12)).expect("hit");
        assert_eq!(first, second);
        assert_eq!(atlas.stats().allocations, 1);
        assert_eq!(atlas.stats().cache_hits, 1);
        assert_eq!(first.size, [8.0, 12.0]);
        assert_eq!(first.tile.page(), Some(0));
    }

    #[test]
    fn different_rasters_get_different_tiles_that_do_not_overlap() {
        let mut atlas = GlyphAtlas::default();
        let a = atlas.get_or_insert(key(1), raster(16, 16)).expect("allocate");
        let b = atlas.get_or_insert(key(2), raster(16, 16)).expect("allocate");
        assert_ne!(a.tile, b.tile);

        let overlaps = a.origin[0] < b.origin[0] + b.size[0]
            && b.origin[0] < a.origin[0] + a.size[0]
            && a.origin[1] < b.origin[1] + b.size[1]
            && b.origin[1] < a.origin[1] + a.size[1];
        assert!(
            !overlaps,
            "two live tiles must never share texels: {a:?} vs {b:?}"
        );
    }

    #[test]
    fn every_field_of_the_raster_key_makes_a_distinct_tile() {
        let base = key(1);
        let variants = [
            GlyphRasterKey { font: 1, ..base },
            GlyphRasterKey { glyph: 2, ..base },
            GlyphRasterKey {
                font_size_bits: 17.0f32.to_bits(),
                ..base
            },
            GlyphRasterKey {
                subpixel: [1, 0],
                ..base
            },
            GlyphRasterKey {
                scale_factor_bits: 2.0f32.to_bits(),
                ..base
            },
            GlyphRasterKey {
                kind: AtlasKind::Polychrome,
                ..base
            },
        ];
        let mut atlas = GlyphAtlas::default();
        let original = atlas.get_or_insert(base, raster(8, 8)).expect("allocate");
        for variant in variants {
            let placement = atlas.get_or_insert(variant, raster(8, 8)).expect("allocate");
            assert_ne!(
                placement.tile, original.tile,
                "{variant:?} must not share a tile with {base:?}"
            );
        }
    }

    #[test]
    fn a_colour_raster_never_lands_in_a_monochrome_page() {
        let mut atlas = GlyphAtlas::default();
        let mono = atlas.get_or_insert(key(1), raster(8, 8)).expect("allocate");
        let colour = atlas
            .get_or_insert(
                GlyphRasterKey {
                    kind: AtlasKind::Polychrome,
                    ..key(1)
                },
                raster(8, 8),
            )
            .expect("allocate");
        assert_ne!(
            mono.tile.page(),
            colour.tile.page(),
            "a coverage mask and a colour bitmap cannot share a texture format"
        );
        assert_eq!(atlas.stats().pages, 2);
    }

    #[test]
    fn filling_a_page_opens_another_rather_than_failing() {
        // A 64px page holds exactly four 32x32 rasters.
        let mut atlas = GlyphAtlas::new(64);
        let placements: Vec<TilePlacement> = (0..8)
            .map(|glyph| {
                atlas
                    .get_or_insert(key(glyph), raster(32, 32))
                    .expect("a full page must spill to a new one")
            })
            .collect();
        assert!(
            atlas.stats().pages > 1,
            "eight 32x32 rasters cannot fit one 64x64 page"
        );
        let pages: Vec<Option<u32>> = placements.iter().map(|p| p.tile.page()).collect();
        assert!(pages.iter().any(|page| *page != pages[0]));
    }

    #[test]
    fn a_raster_larger_than_a_page_is_reported_rather_than_growing_the_page() {
        let mut atlas = GlyphAtlas::new(64);
        assert_eq!(
            atlas.get_or_insert(key(1), raster(65, 8)),
            Err(AtlasError::TooLargeForAPage {
                requested: [65, 8],
                page_size: 64
            })
        );
        assert_eq!(atlas.stats().pages, 0, "a refused request opens no page");
    }

    #[test]
    fn a_zero_area_raster_is_refused() {
        let mut atlas = GlyphAtlas::default();
        assert_eq!(atlas.get_or_insert(key(1), raster(0, 8)), Err(AtlasError::EmptyRaster));
        assert_eq!(atlas.get_or_insert(key(1), raster(8, 0)), Err(AtlasError::EmptyRaster));
    }

    #[test]
    fn evicting_a_raster_reports_its_tile_and_frees_the_space() {
        let mut atlas = GlyphAtlas::new(64);
        let placement = atlas.get_or_insert(key(1), raster(32, 32)).expect("allocate");
        assert_eq!(atlas.live_tiles_in_page(0), Some(1));

        assert!(atlas.evict(&key(1)));
        assert!(!atlas.evict(&key(1)), "a second eviction is not an event");
        assert_eq!(
            atlas.drain_evictions(),
            vec![AtlasEviction::Tile(placement.tile)]
        );
        assert!(atlas.drain_evictions().is_empty(), "draining is destructive");
        assert_eq!(atlas.live_tiles_in_page(0), Some(0));
        assert_eq!(atlas.get(&key(1)), None);
    }

    #[test]
    fn a_freed_slot_is_reused_so_the_slot_field_bounds_live_tiles_not_lifetime_allocations() {
        let mut atlas = GlyphAtlas::new(64);
        let first = atlas.get_or_insert(key(1), raster(8, 8)).expect("allocate");
        atlas.evict(&key(1));
        let second = atlas.get_or_insert(key(2), raster(8, 8)).expect("allocate");
        assert_eq!(
            first.tile.slot(),
            second.tile.slot(),
            "the freed slot must be reused, or a long-running process exhausts 24 bits"
        );
    }

    #[test]
    fn destroying_a_page_reports_one_event_however_many_tiles_it_held() {
        let mut atlas = GlyphAtlas::new(64);
        for glyph in 0..4 {
            atlas.get_or_insert(key(glyph), raster(32, 32)).expect("allocate");
        }
        assert_eq!(atlas.stats().tiles, 4);

        assert!(atlas.destroy_page(0));
        assert!(!atlas.destroy_page(0), "a destroyed page is gone");
        assert_eq!(atlas.drain_evictions(), vec![AtlasEviction::Page(0)]);
        assert_eq!(atlas.stats().tiles, 0);
        assert_eq!(atlas.stats().pages, 0);
        for glyph in 0..4 {
            assert_eq!(atlas.get(&key(glyph)), None);
        }
    }

    #[test]
    fn a_destroyed_pages_index_is_never_reissued() {
        let mut atlas = GlyphAtlas::new(64);
        atlas.get_or_insert(key(0), raster(8, 8)).expect("allocate");
        assert!(atlas.destroy_page(0));
        let reopened = atlas.get_or_insert(key(1), raster(8, 8)).expect("allocate");
        assert_ne!(
            reopened.tile.page(),
            Some(0),
            "reissuing a destroyed page's index would let a stale slab start sampling real texels again"
        );
    }

    #[test]
    fn a_tile_source_rasterises_once_and_answers_from_the_atlas_thereafter() {
        use wgpui_core::scene::atlas::GlyphTileSource;

        let mut atlas = GlyphAtlas::default();
        let mut rasterised = 0usize;
        {
            let mut source = AtlasTileSource::new(&mut atlas, |_key| {
                rasterised += 1;
                Some(RasterMetrics {
                    size: [8, 12],
                    bearing: [0.5, -9.0],
                })
            });
            let first = source.tile_for(key(1)).expect("a raster with metrics");
            let second = source.tile_for(key(1)).expect("resident");
            assert_eq!(first, second);
            assert_eq!(first.bearing, [0.5, -9.0]);
            assert_eq!(first.atlas_size, [8.0, 12.0]);
        }
        assert_eq!(rasterised, 1, "a resident glyph must not be rasterised again");
        assert_eq!(atlas.stats().allocations, 1);
        assert_eq!(atlas.stats().cache_hits, 1);
    }

    #[test]
    fn a_glyph_the_rasteriser_declines_becomes_a_blank_rather_than_an_error() {
        use wgpui_core::scene::atlas::GlyphTileSource;

        let mut atlas = GlyphAtlas::default();
        let mut source = AtlasTileSource::new(&mut atlas, |_key| None);
        assert_eq!(source.tile_for(key(1)), None);
    }

    #[test]
    fn a_glyph_the_atlas_cannot_fit_becomes_a_blank_rather_than_failing_the_frame() {
        use wgpui_core::scene::atlas::GlyphTileSource;

        let mut atlas = GlyphAtlas::new(16);
        let mut source = AtlasTileSource::new(&mut atlas, |_key| {
            Some(RasterMetrics {
                size: [64, 64],
                bearing: [0.0, 0.0],
            })
        });
        assert_eq!(
            source.tile_for(key(1)),
            None,
            "one oversized glyph must not take the frame down with it"
        );
    }

    /// The whole subscription, end to end: the atlas frees a tile, the scene is
    /// told, and exactly the layer holding that tile takes `DISPLAY`.
    #[test]
    fn an_eviction_reaches_the_scene_and_invalidates_the_layer_that_referenced_it()
    -> Result<(), Box<dyn std::error::Error>> {
        use wgpui_core::invalidation::axes::Invalidation;
        use wgpui_core::patch::apply::{ScenePatch, apply};
        use wgpui_core::patch::primitive::{Glyph, GlyphRun};
        use wgpui_core::patch::RecordKey;
        use wgpui_core::scene::layer::{BoundaryId, LayerKey};
        use wgpui_core::scene::Scene;

        let mut atlas = GlyphAtlas::new(64);
        let placement = atlas.get_or_insert(key(1), raster(8, 8))?;
        let untouched = atlas.get_or_insert(key(2), raster(8, 8))?;

        let mut scene = Scene::new();
        let with_text = scene.layer(LayerKey::untiled(BoundaryId::from_raw(1)));
        let other_text = scene.layer(LayerKey::untiled(BoundaryId::from_raw(2)));
        let mut patch = ScenePatch::new();
        patch.glyph_runs.append(
            with_text,
            RecordKey::from_raw(1),
            0,
            GlyphRun {
                color: [1.0; 4],
                glyphs: vec![Glyph {
                    atlas_tile: placement.tile,
                    ..Glyph::ZERO
                }],
            },
        );
        patch.glyph_runs.append(
            other_text,
            RecordKey::from_raw(2),
            0,
            GlyphRun {
                color: [1.0; 4],
                glyphs: vec![Glyph {
                    atlas_tile: untouched.tile,
                    ..Glyph::ZERO
                }],
            },
        );
        apply(&mut scene, &patch)?;
        scene.layers.mark_clean(with_text);
        scene.layers.mark_clean(other_text);

        atlas.evict(&key(1));
        let affected = scene.evict_atlas_batch(atlas.drain_evictions());

        assert_eq!(affected, vec![with_text]);
        assert_eq!(
            scene.layers.get(with_text).map(|layer| layer.invalidation()),
            Some(Invalidation::DISPLAY)
        );
        assert!(
            scene
                .layers
                .get(other_text)
                .is_some_and(|layer| layer.is_clean()),
            "a layer referencing a different tile must not be disturbed"
        );
        Ok(())
    }
}
