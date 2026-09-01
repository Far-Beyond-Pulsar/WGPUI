//! The glyph/sprite atlas allocator — today's `src/platform/cross/atlas.rs`
//! bin-packing half, ported. See docs/gpu-native-architecture.md §3.5, §8
//! (Phase 5), and docs/retained-layers.md §4.3.
//!
//! # The split this file draws, and why it is where it is
//!
//! §6's accounting says `wgpui-text` produces glyph positions and atlas tile
//! *requests*, and this crate turns requests into tile coordinates. The
//! allocator is pure bin-packing over integer rectangles, and — since Phase 5.5
//! — the CPU-side page buffers those rectangles address: [`GlyphAtlas`] opens no
//! device, creates no texture, and issues no `write_texture`. It holds the
//! texels; `render/atlas_upload.rs` is what copies them onto a GPU it owns a
//! handle to.
//!
//! That split is the same one Phase 5 drew and for the same stated reason, moved
//! one step further along: every assertion in this module's tests runs
//! headlessly, on any machine, with no adapter — including, now, assertions
//! about the actual pixels landing at the actual coordinates. An atlas whose
//! blitting can only be checked on hardware is an atlas whose blitting does not
//! get checked.
//!
//! `etagere`'s `BucketedAtlasAllocator` is the same allocator the legacy file
//! uses, at the same version the root crate pins, so packing behaviour is
//! shared rather than re-derived.
//!
//! # Page buffers, and what they cost
//!
//! A page is `page_size²` texels at [`AtlasKind::bytes_per_pixel`] bytes each —
//! 1 MiB for a monochrome 1024² page, 4 MiB for a colour one — held on the CPU
//! for the lifetime of the page. The legacy atlas does not do this: it writes
//! straight through to the texture and keeps no copy. The copy is kept here
//! because it is what makes the whole path testable without a device, and
//! because it is what a `write_texture` reads from anyway; if a real workload
//! ever shows the resident cost mattering, dropping a page's buffer after its
//! last upload is a self-contained change behind [`GlyphAtlas::page_texels`].
//!
//! # What is deliberately not here
//!
//! `wgpu::Texture` creation, the `Monochrome`/`Polychrome` `TextureFormat`
//! mapping, and the row-alignment padding a copy needs. Those live in
//! `render/atlas_upload.rs`, which is the only part of the glyph path that
//! needs a device.

use std::collections::HashMap;
use std::cell::RefCell;
use std::rc::Rc;
use wgpui_core::patch::primitive::AtlasTileId;
use wgpui_core::scene::atlas::{
    AtlasEviction, AtlasKey, AtlasKind, GlyphRasterKey, ImageRasterKey, ImageTile, ImageTileSource,
    RasterizedGlyph, RasterizedImage,
};
use wgpui_text::engine::SharedTextShaper;
use wgpui_text::raster::GlyphRasterizer;
#[cfg(feature = "devtools")]
use wgpui_devtools::{AtlasPackingSnapshot, AtlasPageRecord, AtlasPlacementRecord, SnapshotLimits};

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
    /// A bitmap's byte count disagrees with the size and kind it declares.
    ///
    /// Refused rather than blitted as far as it goes: the copy walks rows, so a
    /// bitmap one row short would either read past its own end or leave a row of
    /// whatever the page held before — and the page held another glyph.
    MalformedBitmap {
        /// Bytes the declared size and kind imply.
        expected: usize,
        /// Bytes the bitmap actually carries.
        actual: usize,
    },
    /// A raster was requested out of one atlas and its bitmap belongs in the
    /// other.
    ///
    /// A coverage mask written into a colour page is not a rendering artefact,
    /// it is three quarters of a row of the next glyph — so the two are checked
    /// against each other rather than assumed consistent because one caller
    /// happens to derive both from the same key.
    KindMismatch {
        /// What the key asked for.
        requested: AtlasKind,
        /// What the bitmap says it is.
        rasterized: AtlasKind,
    },
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
            AtlasError::MalformedBitmap { expected, actual } => write!(
                formatter,
                "a raster declared {expected} bytes of texels and carries {actual}"
            ),
            AtlasError::KindMismatch {
                requested,
                rasterized,
            } => write!(
                formatter,
                "a {rasterized:?} bitmap cannot satisfy a {requested:?} request"
            ),
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

/// A rectangle of one page whose texels have changed since the last upload.
///
/// Recorded per written tile rather than coalesced into a per-page dirty
/// rectangle: a page fills from many small glyphs scattered by the bin packer,
/// so the bounding box of a frame's writes is very nearly the whole page, and
/// uploading it would cost megabytes to move a few kilobytes. The legacy atlas
/// makes the same choice — one `write_texture` per tile, in
/// `WgpuAtlasState::upload_texture`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PendingUpload {
    /// Which page the rectangle is in.
    pub page: u32,
    /// Which atlas that page belongs to, and therefore its texture format.
    pub kind: AtlasKind,
    /// Top-left of the rectangle within the page, in texels.
    pub origin: [u32; 2],
    /// Size of the rectangle, in texels.
    pub size: [u32; 2],
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
    /// The page's texels, row-major, `page_size * bytes_per_pixel` per row.
    ///
    /// A freed tile's texels are left as they are rather than cleared, exactly
    /// as the legacy atlas leaves them: what makes a stale reference safe is the
    /// eviction event, not the texels being blanked, and clearing would cost a
    /// write per eviction to change nothing observable.
    texels: Vec<u8>,
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

    /// Copy `texels` into the page at `origin`, row by row.
    ///
    /// Bounds are re-derived from the page's own dimensions rather than trusted
    /// from the placement, so a caller that hands over a rectangle the page
    /// cannot hold writes nothing instead of writing into the next row.
    fn blit(&mut self, page_size: u32, origin: [u32; 2], size: [u32; 2], texels: &[u8]) -> bool {
        let bytes_per_pixel = self.kind.bytes_per_pixel() as usize;
        let page_stride = page_size as usize * bytes_per_pixel;
        let row_bytes = size[0] as usize * bytes_per_pixel;
        if origin[0].saturating_add(size[0]) > page_size
            || origin[1].saturating_add(size[1]) > page_size
            || texels.len() != row_bytes * size[1] as usize
        {
            return false;
        }
        for row in 0..size[1] as usize {
            let source = row * row_bytes;
            let destination =
                (origin[1] as usize + row) * page_stride + origin[0] as usize * bytes_per_pixel;
            let (Some(from), Some(into)) = (
                texels.get(source..source + row_bytes),
                self.texels.get_mut(destination..destination + row_bytes),
            ) else {
                return false;
            };
            into.copy_from_slice(from);
        }
        true
    }

    /// The texels of one rectangle of this page, tightly packed.
    fn read(&self, page_size: u32, origin: [u32; 2], size: [u32; 2]) -> Option<Vec<u8>> {
        let bytes_per_pixel = self.kind.bytes_per_pixel() as usize;
        let page_stride = page_size as usize * bytes_per_pixel;
        let row_bytes = size[0] as usize * bytes_per_pixel;
        let mut out = Vec::with_capacity(row_bytes * size[1] as usize);
        for row in 0..size[1] as usize {
            let start =
                (origin[1] as usize + row) * page_stride + origin[0] as usize * bytes_per_pixel;
            out.extend_from_slice(self.texels.get(start..start + row_bytes)?);
        }
        Some(out)
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
    tiles_by_key: HashMap<AtlasKey, TilePlacement>,
    keys_by_tile: HashMap<AtlasTileId, AtlasKey>,
    pending_evictions: Vec<AtlasEviction>,
    pending_uploads: Vec<PendingUpload>,
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
            pending_uploads: Vec::new(),
            stats: AtlasStats::default(),
        }
    }

    /// The side length, in texels, of every page in this atlas.
    pub fn page_size(&self) -> u32 {
        self.page_size
    }

    /// What this atlas currently holds.
    pub fn stats(&self) -> AtlasStats {
        AtlasStats {
            pages: self.pages.len(),
            tiles: self.tiles_by_key.len(),
            ..self.stats
        }
    }

    #[cfg(feature = "devtools")]
    pub fn resource_snapshot(&self, limits: SnapshotLimits) -> AtlasPackingSnapshot {
        let pages = self
            .pages
            .iter()
            .map(|page| AtlasPageRecord {
                page_id: page.index,
                kind: atlas_kind_tag(page.kind),
                width: self.page_size,
                height: self.page_size,
                live_tiles: match self.live_tiles_in_page(page.index) {
                    Some(count) => u32::try_from(count).unwrap_or(u32::MAX),
                    None => 0,
                },
            })
            .collect();
        let mut placements = self
            .tiles_by_key
            .values()
            .filter_map(|placement| {
                let page_id = placement.tile.page()?;
                let slot = placement.tile.slot()?;
                let tile_id = page_id.checked_shl(24)?.checked_add(slot)?;
                Some(AtlasPlacementRecord {
                    tile_id,
                    page_id,
                    kind: atlas_kind_tag(placement.kind),
                    x: placement.origin[0] as u32,
                    y: placement.origin[1] as u32,
                    width: placement.size[0] as u32,
                    height: placement.size[1] as u32,
                })
            })
            .collect::<Vec<_>>();
        placements.sort_unstable_by_key(|placement| placement.tile_id);
        AtlasPackingSnapshot::new(pages, placements, limits)
    }

    /// Where `key`'s raster lives, if it is resident.
    ///
    /// The read-only half of [`Self::get_or_insert`], for a caller that wants to
    /// know whether a request would allocate without making it do so.
    pub fn get(&self, key: impl Into<AtlasKey>) -> Option<TilePlacement> {
        self.tiles_by_key.get(&key.into()).copied()
    }

    /// Reserve space for `key`'s raster without supplying its texels.
    ///
    /// The space-only half, kept from Phase 5 for the callers that genuinely
    /// only decide placement — the packing tests, and anything measuring how a
    /// glyph set packs without paying to rasterise it. A tile reserved this way
    /// holds whatever the page held before, so a caller that draws from it is
    /// drawing another glyph's texels; [`Self::get_or_insert_raster`] is the one
    /// to use when the pixels exist.
    pub fn get_or_insert(
        &mut self,
        key: impl Into<AtlasKey>,
        metrics: RasterMetrics,
    ) -> Result<TilePlacement, AtlasError> {
        let key = key.into();
        if let Some(placement) = self.tiles_by_key.get(&key) {
            self.stats.cache_hits += 1;
            return Ok(*placement);
        }
        self.insert_new(key, metrics)
    }

    /// Where `key`'s raster lives, allocating a tile and writing its texels into
    /// the page if it is not already resident.
    ///
    /// This is the whole point of the phase: the bitmap `wgpui-text`'s
    /// rasteriser produced ends up in the space the allocator reserved for it,
    /// and [`Self::drain_uploads`] then names the rectangle a device has to be
    /// told about.
    ///
    /// A resident key is answered without looking at `raster` at all — the
    /// caller is expected to have checked [`Self::get`] first if rasterising is
    /// what it wants to avoid, which is exactly what [`AtlasTileSource`] does.
    pub fn get_or_insert_raster(
        &mut self,
        key: impl Into<AtlasKey>,
        raster: &RasterizedGlyph,
    ) -> Result<TilePlacement, AtlasError> {
        let key = key.into();
        if raster.kind != key.kind() {
            return Err(AtlasError::KindMismatch {
                requested: key.kind(),
                rasterized: raster.kind,
            });
        }
        if !raster.is_well_formed() {
            return Err(AtlasError::MalformedBitmap {
                expected: raster.expected_texel_bytes(),
                actual: raster.texels.len(),
            });
        }
        self.insert_texels(
            key,
            RasterMetrics {
                size: raster.size,
                bearing: raster.bearing,
            },
            &raster.texels,
        )
    }

    /// Where `key`'s decoded frame lives, uploading its texels into a fresh tile
    /// if it is not already resident.
    ///
    /// The image half of [`Self::get_or_insert_raster`], over the same pages and
    /// the same tile numbering. It is a separate entry point rather than a
    /// second `Into<AtlasKey>` call because the two producers hand over
    /// different value types — a glyph carries a bearing and a kind, an image
    /// carries neither — and collapsing them would mean an image caller
    /// constructing a `RasterizedGlyph` whose `bearing` is a lie and whose
    /// `kind` is redundant with the key.
    ///
    /// A resident key is answered without looking at `image` at all, so a caller
    /// that wants to avoid *decoding* checks [`Self::get`] first — which is
    /// exactly what [`ImageAtlasSource`] does.
    pub fn get_or_insert_image(
        &mut self,
        key: ImageRasterKey,
        image: &RasterizedImage,
    ) -> Result<TilePlacement, AtlasError> {
        if !image.is_well_formed() {
            return Err(AtlasError::MalformedBitmap {
                expected: image.expected_texel_bytes(),
                actual: image.texels.len(),
            });
        }
        self.insert_texels(
            AtlasKey::Image(key),
            RasterMetrics {
                size: image.size,
                // An image has no pen and therefore no bearing. Zero here is not
                // a placeholder: `ImageTile` deliberately does not carry the
                // field, so nothing downstream can add it to a position.
                bearing: [0.0, 0.0],
            },
            &image.texels,
        )
    }

    /// Allocate `key` if it is not resident and write `texels` into its tile.
    ///
    /// The shared body of the two `get_or_insert_*` entry points, so that "a
    /// resident key is a cache hit and is never rewritten" is one decision
    /// rather than two that can drift.
    fn insert_texels(
        &mut self,
        key: AtlasKey,
        metrics: RasterMetrics,
        texels: &[u8],
    ) -> Result<TilePlacement, AtlasError> {
        if let Some(placement) = self.tiles_by_key.get(&key) {
            self.stats.cache_hits += 1;
            return Ok(*placement);
        }
        let placement = self.insert_new(key, metrics)?;
        self.write_texels(placement, texels);
        Ok(placement)
    }

    fn insert_new(
        &mut self,
        key: AtlasKey,
        metrics: RasterMetrics,
    ) -> Result<TilePlacement, AtlasError> {
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

    /// Copy a tile's texels into its page and queue the upload that carries them
    /// to a texture.
    ///
    /// A blit that does not fit is dropped rather than partially applied and the
    /// upload is not queued, so a page never reports a rectangle whose texels it
    /// did not actually write. `placement` comes from this atlas, so a rejection
    /// here is a bug in this file rather than a caller error, which is why it
    /// reports nothing back: there is no caller decision to make.
    fn write_texels(&mut self, placement: TilePlacement, texels: &[u8]) {
        let page_size = self.page_size;
        let origin = [placement.origin[0] as u32, placement.origin[1] as u32];
        let size = [placement.size[0] as u32, placement.size[1] as u32];
        let Some(index) = placement.tile.page() else {
            return;
        };
        let Some(page) = self.pages.iter_mut().find(|page| page.index == index) else {
            return;
        };
        if !page.blit(page_size, origin, size, texels) {
            return;
        }
        self.pending_uploads.push(PendingUpload {
            page: index,
            kind: placement.kind,
            origin,
            size,
        });
    }

    /// The texels a resident tile currently holds, tightly packed.
    ///
    /// The read half of [`Self::get_or_insert_raster`], and what lets a test
    /// assert that a glyph's pixels are where the allocator said they would be
    /// without opening a device.
    pub fn tile_texels(&self, placement: TilePlacement) -> Option<Vec<u8>> {
        let index = placement.tile.page()?;
        let page = self.pages.iter().find(|page| page.index == index)?;
        page.read(
            self.page_size,
            [placement.origin[0] as u32, placement.origin[1] as u32],
            [placement.size[0] as u32, placement.size[1] as u32],
        )
    }

    /// A whole page's texels, row-major.
    pub fn page_texels(&self, page_index: u32) -> Option<&[u8]> {
        self.pages
            .iter()
            .find(|page| page.index == page_index)
            .map(|page| page.texels.as_slice())
    }

    /// Which atlas a live page belongs to.
    pub fn page_kind(&self, page_index: u32) -> Option<AtlasKind> {
        self.pages
            .iter()
            .find(|page| page.index == page_index)
            .map(|page| page.kind)
    }

    /// Every live page, in the order they were opened.
    pub fn page_indices(&self) -> Vec<u32> {
        self.pages.iter().map(|page| page.index).collect()
    }

    /// Take the page rectangles written since the last drain.
    ///
    /// Destructive, like [`Self::drain_evictions`], and for the same reason:
    /// one upload, one uploader. A caller that drops these without uploading
    /// them leaves the texture behind the page buffer, which is why nothing
    /// but the uploader should call it.
    pub fn drain_uploads(&mut self) -> Vec<PendingUpload> {
        std::mem::take(&mut self.pending_uploads)
    }

    /// Whether any texels are waiting to reach a texture.
    pub fn has_pending_uploads(&self) -> bool {
        !self.pending_uploads.is_empty()
    }

    fn allocate(
        &mut self,
        key: AtlasKey,
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
            if kind != key.kind() {
                continue;
            }
            if let Some(placement) = self.allocate_in(index, key, metrics, requested)? {
                return Ok(placement);
            }
        }

        let index = self.open_page(key.kind())?;
        self.allocate_in(index, key, metrics, requested)?
            .ok_or(AtlasError::TooLargeForAPage {
                requested: size,
                page_size: self.page_size,
            })
    }

    fn allocate_in(
        &mut self,
        index: usize,
        key: AtlasKey,
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
            kind: key.kind(),
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
        let texel_bytes =
            self.page_size as usize * self.page_size as usize * kind.bytes_per_pixel() as usize;
        self.pages.push(Page {
            index,
            kind,
            allocator: etagere::BucketedAtlasAllocator::new(etagere::size2(side, side)),
            allocations: Vec::new(),
            free_slots: Vec::new(),
            live: 0,
            // Zeroed, so an unwritten region of a page is transparent rather
            // than whatever the allocator happened to be handed — which matters
            // because `get_or_insert` reserves space without texels, and a
            // sprite sampling that space should draw nothing, not noise.
            texels: vec![0; texel_bytes],
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
    pub fn evict(&mut self, key: impl Into<AtlasKey>) -> bool {
        let Some(placement) = self.tiles_by_key.remove(&key.into()) else {
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
        let dropped: Vec<AtlasKey> = self
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
        // Its queued uploads go with it. An upload names a page and a rectangle
        // and the uploader reads the texels back out of the page, so a queued
        // upload for a page that no longer exists is either a silent no-op or a
        // read of the wrong page, depending on how carefully the uploader is
        // written — and neither is a thing to leave to the uploader.
        self.pending_uploads
            .retain(|upload| upload.page != page_index);
        self.pending_evictions.push(AtlasEviction::Page(page_index));
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

#[cfg(feature = "devtools")]
const fn atlas_kind_tag(kind: AtlasKind) -> u8 {
    match kind {
        AtlasKind::Monochrome => 0,
        AtlasKind::Polychrome => 1,
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
/// # Why the rasteriser is still a closure
///
/// Phase 5 made it a parameter because nothing rasterised and writing one would
/// have meant writing it against an imagined consumer. Phase 5.5 wrote one — in
/// `wgpui-text`, where `cosmic-text` and therefore `swash` live — and the
/// parameter stays, because the dependency edge §3.3/§3.5 draw runs the other
/// way: `wgpui-wgpu` cannot name `wgpui_text::raster::GlyphRasterizer` without
/// depending on the crate that shapes text, which would put a font database
/// inside the crate whose job is a device. The closure is how the two meet, and
/// [`wgpui_core::scene::atlas::RasterizedGlyph`] is the vocabulary they meet in.
pub struct AtlasTileSource<'atlas, Rasterize> {
    atlas: &'atlas mut GlyphAtlas,
    rasterize: Rasterize,
}

impl<'atlas, Rasterize> AtlasTileSource<'atlas, Rasterize>
where
    Rasterize: FnMut(GlyphRasterKey) -> Option<RasterizedGlyph>,
{
    /// A tile source over `atlas`, rasterising with `rasterize`.
    pub fn new(atlas: &'atlas mut GlyphAtlas, rasterize: Rasterize) -> Self {
        Self { atlas, rasterize }
    }
}

impl<Rasterize> wgpui_core::scene::atlas::GlyphTileSource for AtlasTileSource<'_, Rasterize>
where
    Rasterize: FnMut(GlyphRasterKey) -> Option<RasterizedGlyph>,
{
    fn tile_for(&mut self, key: GlyphRasterKey) -> Option<wgpui_core::scene::atlas::GlyphTile> {
        // A resident tile answers without rasterising, which is the point of the
        // atlas holding a key map at all: a paragraph asks for the same 'e'
        // dozens of times per frame, and the caller deliberately does not
        // deduplicate (`wgpui-text`'s `patch` module says so).
        let placement = match self.atlas.get(key) {
            Some(placement) => {
                self.atlas.stats.cache_hits += 1;
                placement
            }
            None => {
                let raster = (self.rasterize)(key)?;
                // A refused allocation is `None`, not an error the caller has to
                // handle: one glyph failing to find atlas space degrades to a
                // blank glyph rather than failing the frame. A caller that wants
                // to know why asks `get_or_insert_raster` directly.
                self.atlas.get_or_insert_raster(key, &raster).ok()?
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

/// A real glyph source shared by rich text elements and the frame loop.
pub struct SharedAtlasTileSource {
    atlas: Rc<RefCell<GlyphAtlas>>,
    shaper: SharedTextShaper,
    rasterizer: Rc<RefCell<GlyphRasterizer>>,
}

impl SharedAtlasTileSource {
    pub fn new(
        atlas: Rc<RefCell<GlyphAtlas>>,
        shaper: SharedTextShaper,
        rasterizer: Rc<RefCell<GlyphRasterizer>>,
    ) -> Self {
        Self { atlas, shaper, rasterizer }
    }
}

impl wgpui_core::scene::atlas::GlyphTileSource for SharedAtlasTileSource {
    fn tile_for(&mut self, key: GlyphRasterKey) -> Option<wgpui_core::scene::atlas::GlyphTile> {
        if let Some(placement) = self.atlas.borrow().get(key) {
            self.atlas.borrow_mut().stats.cache_hits += 1;
            return Some(wgpui_core::scene::atlas::GlyphTile {
                tile: placement.tile,
                atlas_origin: placement.origin,
                atlas_size: placement.size,
                bearing: placement.bearing,
            });
        }
        let raster = {
            let mut rasterizer = self.rasterizer.borrow_mut();
            let mut shaper = self.shaper.borrow_mut();
            rasterizer.rasterize(&mut shaper, key).ok()?
        };
        let placement = self
            .atlas
            .borrow_mut()
            .get_or_insert_raster(key, &raster)
            .ok()?;
        Some(wgpui_core::scene::atlas::GlyphTile {
            tile: placement.tile,
            atlas_origin: placement.origin,
            atlas_size: placement.size,
            bearing: placement.bearing,
        })
    }
}

/// Place one decoded frame in `atlas`, decoding it only if it is not resident.
///
/// The body of [`ImageTileSource`] for this crate, factored out so the borrowing
/// and the shared adapters below cannot drift: they differ in how they reach a
/// [`GlyphAtlas`] and in nothing else.
fn image_tile_in(
    atlas: &mut GlyphAtlas,
    key: ImageRasterKey,
    decode: &mut dyn FnMut(ImageRasterKey) -> Option<RasterizedImage>,
) -> Option<ImageTile> {
    let placement = match atlas.get(key) {
        Some(placement) => {
            atlas.stats.cache_hits += 1;
            placement
        }
        None => {
            let image = decode(key)?;
            // A refused allocation is `None`, not an error the caller has to
            // handle: one image failing to find atlas space degrades to a sprite
            // that draws nothing rather than failing the frame — the same rule a
            // glyph follows, and the one that keeps a 4000px photograph from
            // taking a window down with it.
            atlas.get_or_insert_image(key, &image).ok()?
        }
    };
    Some(ImageTile {
        tile: placement.tile,
        atlas_origin: placement.origin,
        atlas_size: placement.size,
    })
}

/// An [`ImageTileSource`] over a borrowed atlas.
///
/// [`AtlasTileSource`]'s image counterpart for a caller that already has the
/// atlas in hand — a test, or a frame that borrows it for one call. It cannot be
/// boxed into anything outliving the borrow, which is what [`SharedImageAtlas`]
/// is for.
///
/// Unlike [`AtlasTileSource`] it holds no closure: the decoder arrives per call,
/// because §3.4 puts the decode cache in `wgpui-widgets` and this crate may not
/// name it. See [`ImageTileSource`]'s own doc, which records why the two seams
/// differ in exactly this way.
pub struct ImageAtlasSource<'atlas> {
    atlas: &'atlas mut GlyphAtlas,
}

impl<'atlas> ImageAtlasSource<'atlas> {
    /// A tile source over `atlas`.
    pub fn new(atlas: &'atlas mut GlyphAtlas) -> Self {
        Self { atlas }
    }
}

impl ImageTileSource for ImageAtlasSource<'_> {
    fn tile_for(
        &mut self,
        key: ImageRasterKey,
        decode: &mut dyn FnMut(ImageRasterKey) -> Option<RasterizedImage>,
    ) -> Option<ImageTile> {
        image_tile_in(self.atlas, key, decode)
    }
}

/// An [`ImageTileSource`] over an atlas shared with the frame renderer.
///
/// The form an element can actually hold: `wgpui_widgets::img::ImageEngine`
/// keeps a `Box<dyn ImageTileSource>` for the lifetime of a window, and a
/// borrowing source cannot be boxed that long. The renderer takes
/// `atlas.borrow_mut()` when it uploads pages and this takes it when it
/// allocates a tile; the two never overlap, because uploading happens between
/// frames and allocating happens during emission.
///
/// `Rc<RefCell<_>>` and not a lock, for the reason
/// `wgpui_widgets::styled_text::SharedTextEngine` gives: everything that reaches
/// it runs on the frame's thread.
#[derive(Clone)]
pub struct SharedImageAtlas {
    atlas: std::rc::Rc<std::cell::RefCell<GlyphAtlas>>,
}

impl SharedImageAtlas {
    /// A tile source over `atlas`.
    pub fn new(atlas: std::rc::Rc<std::cell::RefCell<GlyphAtlas>>) -> Self {
        Self { atlas }
    }

    /// The atlas itself, for uploading its pages or draining its evictions.
    pub fn atlas(&self) -> &std::rc::Rc<std::cell::RefCell<GlyphAtlas>> {
        &self.atlas
    }
}

impl ImageTileSource for SharedImageAtlas {
    fn tile_for(
        &mut self,
        key: ImageRasterKey,
        decode: &mut dyn FnMut(ImageRasterKey) -> Option<RasterizedImage>,
    ) -> Option<ImageTile> {
        // `try_borrow_mut` rather than `borrow_mut`: a re-entrant borrow would
        // be a bug in the caller's frame structure, and a sprite that draws
        // nothing is a better report of it than a panic inside a paint walk.
        let mut atlas = self.atlas.try_borrow_mut().ok()?;
        image_tile_in(&mut atlas, key, decode)
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

    /// A bitmap whose every texel is `fill`, so a blit can be checked for
    /// landing in the right rectangle *and* for not spilling out of it.
    fn bitmap(width: u32, height: u32, kind: AtlasKind, fill: u8) -> RasterizedGlyph {
        RasterizedGlyph {
            size: [width, height],
            kind,
            bearing: [0.0, 0.0],
            texels: vec![fill; (width * height * kind.bytes_per_pixel()) as usize],
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
        let first = atlas
            .get_or_insert(key(1), raster(8, 12))
            .expect("allocate");
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
        let a = atlas
            .get_or_insert(key(1), raster(16, 16))
            .expect("allocate");
        let b = atlas
            .get_or_insert(key(2), raster(16, 16))
            .expect("allocate");
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
            let placement = atlas
                .get_or_insert(variant, raster(8, 8))
                .expect("allocate");
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
    fn page_id_exhaustion_is_reported_without_opening_or_aliasing_a_page() {
        let mut atlas = GlyphAtlas::new(64);
        atlas.next_page_index = 255;
        let result = atlas.get_or_insert(key(1), raster(8, 8));
        assert_eq!(result, Err(AtlasError::OutOfTileIds));
        assert!(atlas.page_indices().is_empty());
        assert!(atlas.get(key(1)).is_none());
    }

    #[test]
    fn a_zero_area_raster_is_refused() {
        let mut atlas = GlyphAtlas::default();
        assert_eq!(
            atlas.get_or_insert(key(1), raster(0, 8)),
            Err(AtlasError::EmptyRaster)
        );
        assert_eq!(
            atlas.get_or_insert(key(1), raster(8, 0)),
            Err(AtlasError::EmptyRaster)
        );
    }

    #[test]
    fn evicting_a_raster_reports_its_tile_and_frees_the_space() {
        let mut atlas = GlyphAtlas::new(64);
        let placement = atlas
            .get_or_insert(key(1), raster(32, 32))
            .expect("allocate");
        assert_eq!(atlas.live_tiles_in_page(0), Some(1));

        assert!(atlas.evict(key(1)));
        assert!(!atlas.evict(key(1)), "a second eviction is not an event");
        assert_eq!(
            atlas.drain_evictions(),
            vec![AtlasEviction::Tile(placement.tile)]
        );
        assert!(
            atlas.drain_evictions().is_empty(),
            "draining is destructive"
        );
        assert_eq!(atlas.live_tiles_in_page(0), Some(0));
        assert_eq!(atlas.get(key(1)), None);
    }

    #[test]
    fn a_freed_slot_is_reused_so_the_slot_field_bounds_live_tiles_not_lifetime_allocations() {
        let mut atlas = GlyphAtlas::new(64);
        let first = atlas.get_or_insert(key(1), raster(8, 8)).expect("allocate");
        atlas.evict(key(1));
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
            atlas
                .get_or_insert(key(glyph), raster(32, 32))
                .expect("allocate");
        }
        assert_eq!(atlas.stats().tiles, 4);

        assert!(atlas.destroy_page(0));
        assert!(!atlas.destroy_page(0), "a destroyed page is gone");
        assert_eq!(atlas.drain_evictions(), vec![AtlasEviction::Page(0)]);
        assert_eq!(atlas.stats().tiles, 0);
        assert_eq!(atlas.stats().pages, 0);
        for glyph in 0..4 {
            assert_eq!(atlas.get(key(glyph)), None);
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
                Some(RasterizedGlyph {
                    bearing: [0.5, -9.0],
                    ..bitmap(8, 12, AtlasKind::Monochrome, 0xAB)
                })
            });
            let first = source.tile_for(key(1)).expect("a raster with metrics");
            let second = source.tile_for(key(1)).expect("resident");
            assert_eq!(first, second);
            assert_eq!(first.bearing, [0.5, -9.0]);
            assert_eq!(first.atlas_size, [8.0, 12.0]);
        }
        assert_eq!(
            rasterised, 1,
            "a resident glyph must not be rasterised again"
        );
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
            Some(bitmap(64, 64, AtlasKind::Monochrome, 1))
        });
        assert_eq!(
            source.tile_for(key(1)),
            None,
            "one oversized glyph must not take the frame down with it"
        );
    }

    #[test]
    fn a_rasters_texels_land_in_the_rectangle_the_allocator_reserved_for_it() {
        let mut atlas = GlyphAtlas::new(64);
        let placement = atlas
            .get_or_insert_raster(key(1), &bitmap(8, 4, AtlasKind::Monochrome, 0x7F))
            .expect("allocate");
        assert_eq!(
            atlas.tile_texels(placement),
            Some(vec![0x7F; 32]),
            "the tile must read back exactly what was written into it"
        );

        // And nowhere else. Everything outside the tile is still the zero the
        // page opened with, which is what proves the blit respected its stride
        // rather than writing 32 contiguous bytes.
        let page = atlas.page_texels(0).expect("page 0 is live");
        assert_eq!(page.len(), 64 * 64);
        let written: usize = page.iter().filter(|texel| **texel == 0x7F).count();
        assert_eq!(written, 32, "a 8x4 blit must touch 32 texels and no more");
        let origin = [placement.origin[0] as usize, placement.origin[1] as usize];
        for row in 0..4usize {
            let start = (origin[1] + row) * 64 + origin[0];
            assert_eq!(
                page.get(start..start + 8),
                Some([0x7F; 8].as_slice()),
                "row {row} of the tile must sit at its own stride offset"
            );
        }
    }

    #[test]
    fn a_colour_raster_is_four_bytes_per_texel_in_its_page() {
        let mut atlas = GlyphAtlas::new(16);
        let colour_key = GlyphRasterKey {
            kind: AtlasKind::Polychrome,
            ..key(1)
        };
        let mut raster = bitmap(2, 2, AtlasKind::Polychrome, 0);
        raster.texels = vec![
            1, 2, 3, 4, 5, 6, 7, 8, // row 0
            9, 10, 11, 12, 13, 14, 15, 16, // row 1
        ];
        let placement = atlas
            .get_or_insert_raster(colour_key, &raster)
            .expect("allocate");
        assert_eq!(atlas.tile_texels(placement), Some(raster.texels));
        assert_eq!(
            atlas
                .page_texels(placement.tile.page().expect("a page"))
                .map(<[u8]>::len),
            Some(16 * 16 * 4)
        );
    }

    #[test]
    fn neighbouring_tiles_do_not_write_over_each_other() {
        let mut atlas = GlyphAtlas::new(64);
        let first = atlas
            .get_or_insert_raster(key(1), &bitmap(16, 16, AtlasKind::Monochrome, 0x11))
            .expect("allocate");
        let second = atlas
            .get_or_insert_raster(key(2), &bitmap(16, 16, AtlasKind::Monochrome, 0x22))
            .expect("allocate");
        assert_ne!(first.origin, second.origin);
        assert_eq!(atlas.tile_texels(first), Some(vec![0x11; 256]));
        assert_eq!(atlas.tile_texels(second), Some(vec![0x22; 256]));
    }

    #[test]
    fn a_resident_raster_is_not_written_a_second_time() {
        let mut atlas = GlyphAtlas::new(64);
        atlas
            .get_or_insert_raster(key(1), &bitmap(8, 8, AtlasKind::Monochrome, 0x11))
            .expect("allocate");
        assert_eq!(atlas.drain_uploads().len(), 1);

        // A second request with *different* texels must not rewrite the tile:
        // the key is the raster's identity, and a caller handing over different
        // pixels for the same key has already disagreed with the atlas about
        // what the key means.
        let placement = atlas
            .get_or_insert_raster(key(1), &bitmap(8, 8, AtlasKind::Monochrome, 0x99))
            .expect("resident");
        assert_eq!(atlas.tile_texels(placement), Some(vec![0x11; 64]));
        assert!(!atlas.has_pending_uploads());
    }

    #[test]
    fn every_write_queues_exactly_one_upload_naming_its_own_rectangle() {
        let mut atlas = GlyphAtlas::new(64);
        let first = atlas
            .get_or_insert_raster(key(1), &bitmap(8, 4, AtlasKind::Monochrome, 1))
            .expect("allocate");
        let second = atlas
            .get_or_insert_raster(key(2), &bitmap(2, 6, AtlasKind::Monochrome, 2))
            .expect("allocate");
        let uploads = atlas.drain_uploads();
        assert_eq!(
            uploads,
            vec![
                PendingUpload {
                    page: 0,
                    kind: AtlasKind::Monochrome,
                    origin: [first.origin[0] as u32, first.origin[1] as u32],
                    size: [8, 4],
                },
                PendingUpload {
                    page: 0,
                    kind: AtlasKind::Monochrome,
                    origin: [second.origin[0] as u32, second.origin[1] as u32],
                    size: [2, 6],
                },
            ]
        );
        assert!(atlas.drain_uploads().is_empty(), "draining is destructive");
    }

    #[test]
    fn destroying_a_page_takes_its_queued_uploads_with_it() {
        let mut atlas = GlyphAtlas::new(64);
        atlas
            .get_or_insert_raster(key(1), &bitmap(8, 8, AtlasKind::Monochrome, 1))
            .expect("allocate");
        assert!(atlas.has_pending_uploads());
        assert!(atlas.destroy_page(0));
        assert!(
            !atlas.has_pending_uploads(),
            "an upload naming a page that no longer exists has nothing to read from"
        );
    }

    #[test]
    fn a_bitmap_whose_length_disagrees_with_its_size_is_refused_rather_than_blitted() {
        let mut atlas = GlyphAtlas::new(64);
        let mut short = bitmap(8, 8, AtlasKind::Monochrome, 1);
        short.texels.pop();
        assert_eq!(
            atlas.get_or_insert_raster(key(1), &short),
            Err(AtlasError::MalformedBitmap {
                expected: 64,
                actual: 63
            })
        );
        assert_eq!(atlas.stats().pages, 0, "a refused raster opens no page");
        assert!(!atlas.has_pending_uploads());
    }

    #[test]
    fn a_bitmap_of_the_wrong_kind_is_refused_rather_than_reinterpreted() {
        let mut atlas = GlyphAtlas::new(64);
        assert_eq!(
            atlas.get_or_insert_raster(key(1), &bitmap(8, 8, AtlasKind::Polychrome, 1)),
            Err(AtlasError::KindMismatch {
                requested: AtlasKind::Monochrome,
                rasterized: AtlasKind::Polychrome
            })
        );
    }

    /// A fresh page is transparent, and a *reused* rectangle is not.
    ///
    /// Both halves are the same claim from two sides, and the second is the one
    /// worth pinning down: freeing a tile does not blank its texels (the legacy
    /// atlas does not either), so a tile whose space was reserved without
    /// supplying pixels can read back as the glyph that used to live there.
    /// That is safe only because the eviction event — not a blanked page — is
    /// what makes a stale reference visible, and because nothing samples a tile
    /// it did not write. Asserted rather than left as a comment, so a future
    /// change that starts clearing on eviction has to notice this is the reason
    /// it did not before.
    #[test]
    fn a_fresh_page_is_transparent_and_a_reused_rectangle_keeps_its_old_texels() {
        let mut atlas = GlyphAtlas::new(64);
        let untouched = atlas.get_or_insert(key(9), raster(4, 4)).expect("allocate");
        assert_eq!(
            atlas.tile_texels(untouched),
            Some(vec![0; 16]),
            "a page opens zeroed, so unwritten space samples as nothing"
        );
        assert!(
            !atlas.has_pending_uploads(),
            "reserving space uploads nothing"
        );

        atlas
            .get_or_insert_raster(key(1), &bitmap(16, 16, AtlasKind::Monochrome, 0xFF))
            .expect("allocate");
        assert!(atlas.evict(key(1)));
        let reused = atlas
            .get_or_insert(key(2), raster(16, 16))
            .expect("allocate");
        assert_eq!(
            atlas.tile_texels(reused),
            Some(vec![0xFF; 256]),
            "the freed rectangle still holds the evicted glyph's texels"
        );
    }

    // ---- Phase 6.2: the polychrome tile producer -------------------------

    fn image_key(source: u64, frame_index: u32) -> ImageRasterKey {
        ImageRasterKey {
            source,
            frame_index,
            scale_factor_bits: 1.0f32.to_bits(),
        }
    }

    /// A `width x height` RGBA bitmap whose every texel is distinct, so a blit
    /// that transposes rows or drops a channel cannot pass by accident.
    fn image(width: u32, height: u32) -> RasterizedImage {
        let mut texels = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                texels.extend_from_slice(&[x as u8, y as u8, (x ^ y) as u8, 0xFF]);
            }
        }
        RasterizedImage {
            size: [width, height],
            texels,
        }
    }

    #[test]
    fn an_image_frame_lands_in_a_colour_page_and_reads_back_exactly() {
        let mut atlas = GlyphAtlas::new(64);
        let bitmap = image(8, 4);
        let placement = atlas
            .get_or_insert_image(image_key(1, 0), &bitmap)
            .expect("allocate");

        assert_eq!(placement.kind, AtlasKind::Polychrome);
        assert_eq!(placement.size, [8.0, 4.0]);
        assert_eq!(placement.bearing, [0.0, 0.0], "an image has no pen");
        assert_eq!(
            atlas.tile_texels(placement),
            Some(bitmap.texels.clone()),
            "the tile must read back exactly the decoded bytes"
        );
        assert_eq!(
            atlas.page_kind(placement.tile.page().expect("a page")),
            Some(AtlasKind::Polychrome)
        );
        assert_eq!(atlas.drain_uploads().len(), 1);
    }

    #[test]
    fn each_frame_of_an_animated_source_gets_its_own_tile() {
        // A GIF that has looped once holds every frame at the same time and
        // cycles between tiles; if the frame index were not part of the key it
        // would instead re-upload over frame 0 on every tick.
        let mut atlas = GlyphAtlas::new(64);
        let first = atlas
            .get_or_insert_image(image_key(1, 0), &image(8, 8))
            .expect("allocate");
        let second = atlas
            .get_or_insert_image(image_key(1, 1), &image(8, 8))
            .expect("allocate");
        let other_source = atlas
            .get_or_insert_image(image_key(2, 0), &image(8, 8))
            .expect("allocate");
        assert_ne!(first.tile, second.tile);
        assert_ne!(first.tile, other_source.tile);
        assert_eq!(atlas.stats().tiles, 3);
    }

    #[test]
    fn a_glyph_and_an_image_can_share_a_colour_page_without_sharing_a_tile() {
        let mut atlas = GlyphAtlas::new(64);
        let emoji = atlas
            .get_or_insert_raster(
                GlyphRasterKey {
                    kind: AtlasKind::Polychrome,
                    ..key(1)
                },
                &bitmap(8, 8, AtlasKind::Polychrome, 0x11),
            )
            .expect("allocate");
        let picture = atlas
            .get_or_insert_image(image_key(1, 0), &image(8, 8))
            .expect("allocate");

        assert_eq!(
            emoji.tile.page(),
            picture.tile.page(),
            "one colour format, one set of pages — this is why the allocator was \
             made kind-aware rather than glyph-aware"
        );
        assert_ne!(emoji.tile, picture.tile);
        assert_eq!(atlas.stats().pages, 1);
        assert_eq!(atlas.tile_texels(emoji), Some(vec![0x11; 8 * 8 * 4]));
    }

    #[test]
    fn a_glyph_key_and_an_image_key_never_collide() {
        // The two key spaces are disjoint by construction — a `u64` source id
        // and a `u32` font id live in different variants — and this asserts the
        // construction rather than trusting it, because a collision here draws
        // one resource's pixels for another's request.
        let mut atlas = GlyphAtlas::new(64);
        let glyph = atlas.get_or_insert(key(1), raster(8, 8)).expect("allocate");
        let picture = atlas
            .get_or_insert_image(image_key(1, 0), &image(8, 8))
            .expect("allocate");
        assert_ne!(glyph.tile, picture.tile);
        assert_eq!(
            atlas.get(image_key(1, 0)).map(|p| p.tile),
            Some(picture.tile)
        );
        assert_eq!(atlas.get(key(1)).map(|p| p.tile), Some(glyph.tile));
    }

    #[test]
    fn an_image_whose_length_disagrees_with_its_size_is_refused_rather_than_blitted() {
        let mut atlas = GlyphAtlas::new(64);
        let mut short = image(8, 8);
        short.texels.pop();
        assert_eq!(
            atlas.get_or_insert_image(image_key(1, 0), &short),
            Err(AtlasError::MalformedBitmap {
                expected: 8 * 8 * 4,
                actual: 8 * 8 * 4 - 1,
            })
        );
        assert_eq!(atlas.stats().pages, 0, "a refused image opens no page");
        assert!(!atlas.has_pending_uploads());
    }

    #[test]
    fn an_image_larger_than_a_page_is_reported_rather_than_growing_the_page() {
        let mut atlas = GlyphAtlas::new(64);
        assert_eq!(
            atlas.get_or_insert_image(image_key(1, 0), &image(65, 8)),
            Err(AtlasError::TooLargeForAPage {
                requested: [65, 8],
                page_size: 64,
            })
        );
    }

    #[test]
    fn an_image_source_decodes_once_and_answers_from_the_atlas_thereafter() {
        use wgpui_core::scene::atlas::ImageTileSource;

        let mut atlas = GlyphAtlas::new(64);
        let mut decodes = 0usize;
        {
            let mut decode = |_key| {
                decodes += 1;
                Some(image(8, 12))
            };
            let mut source = ImageAtlasSource::new(&mut atlas);
            let first = source
                .tile_for(image_key(1, 0), &mut decode)
                .expect("a decoded frame");
            let second = source
                .tile_for(image_key(1, 0), &mut decode)
                .expect("resident");
            assert_eq!(first, second);
            assert_eq!(first.atlas_size, [8.0, 12.0]);
        }
        assert_eq!(
            decodes, 1,
            "a resident frame must not be decoded again — an image frame is \
             megabytes, and decoding one twice inside a frame loop is a stall"
        );
        assert_eq!(atlas.stats().allocations, 1);
        assert_eq!(atlas.stats().cache_hits, 1);
    }

    #[test]
    fn an_image_the_decoder_declines_or_the_atlas_refuses_becomes_a_blank_sprite() {
        use wgpui_core::scene::atlas::ImageTileSource;

        let mut atlas = GlyphAtlas::new(16);
        assert_eq!(
            ImageAtlasSource::new(&mut atlas).tile_for(image_key(1, 0), &mut |_key| None),
            None,
            "an image that has not loaded yet is ordinary, not an error"
        );
        assert_eq!(
            ImageAtlasSource::new(&mut atlas)
                .tile_for(image_key(2, 0), &mut |_key| Some(image(64, 64))),
            None,
            "one oversized photograph must not take the frame down with it"
        );
    }

    #[test]
    fn a_shared_atlas_source_places_tiles_in_the_atlas_the_renderer_uploads() {
        use wgpui_core::scene::atlas::ImageTileSource;

        let shared = std::rc::Rc::new(std::cell::RefCell::new(GlyphAtlas::new(64)));
        let mut source = SharedImageAtlas::new(std::rc::Rc::clone(&shared));
        let tile = source
            .tile_for(image_key(1, 0), &mut |_key| Some(image(8, 8)))
            .expect("a decoded frame");

        // The point of the shared form: the renderer's own handle sees the tile
        // and the upload it queued, without the element having handed anything
        // over.
        let atlas = shared.borrow();
        assert_eq!(atlas.get(image_key(1, 0)).map(|p| p.tile), Some(tile.tile));
        assert!(atlas.has_pending_uploads());
    }

    #[test]
    fn a_shared_atlas_already_borrowed_yields_a_blank_rather_than_panicking() {
        use wgpui_core::scene::atlas::ImageTileSource;

        let shared = std::rc::Rc::new(std::cell::RefCell::new(GlyphAtlas::new(64)));
        let mut source = SharedImageAtlas::new(std::rc::Rc::clone(&shared));
        let _held = shared.borrow_mut();
        assert_eq!(
            source.tile_for(image_key(1, 0), &mut |_key| Some(image(8, 8))),
            None,
            "a re-entrant borrow is a bug in the caller's frame structure, and a \
             blank sprite reports it better than a panic inside a paint walk"
        );
    }

    #[test]
    fn evicting_an_image_frees_its_space_and_reports_its_tile() {
        let mut atlas = GlyphAtlas::new(64);
        let placement = atlas
            .get_or_insert_image(image_key(1, 0), &image(32, 32))
            .expect("allocate");
        assert!(atlas.evict(image_key(1, 0)));
        assert!(!atlas.evict(image_key(1, 0)));
        assert_eq!(
            atlas.drain_evictions(),
            vec![AtlasEviction::Tile(placement.tile)]
        );
        assert_eq!(atlas.get(image_key(1, 0)), None);
    }

    /// The whole subscription, end to end: the atlas frees a tile, the scene is
    /// told, and exactly the layer holding that tile takes `DISPLAY`.
    #[test]
    fn an_eviction_reaches_the_scene_and_invalidates_the_layer_that_referenced_it()
    -> Result<(), Box<dyn std::error::Error>> {
        use wgpui_core::invalidation::axes::Invalidation;
        use wgpui_core::patch::RecordKey;
        use wgpui_core::patch::apply::{ScenePatch, apply};
        use wgpui_core::patch::primitive::{Glyph, GlyphRun};
        use wgpui_core::scene::Scene;
        use wgpui_core::scene::layer::{BoundaryId, LayerKey};

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

        atlas.evict(key(1));
        let affected = scene.evict_atlas_batch(atlas.drain_evictions());

        assert_eq!(affected, vec![with_text]);
        assert_eq!(
            scene
                .layers
                .get(with_text)
                .map(|layer| layer.invalidation()),
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
