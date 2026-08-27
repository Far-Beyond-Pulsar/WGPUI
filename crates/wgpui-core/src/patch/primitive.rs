//! Per-kind patch payloads: what a primitive kind must tell the scene so it
//! can be slab-allocated, encoded, and delta-uploaded.
//! See docs/gpu-native-architecture.md §2, §5.0.
//!
//! # Why two kinds, and only two, in Phase 1
//!
//! The legacy renderer has seven instanced primitive kinds (quads, shadows,
//! paths, underlines, mono/poly sprites, surfaces). Porting all seven now
//! would be seven repetitions of the same twenty lines and would not test the
//! architecture any further once the pattern holds. What *does* test it is
//! covering the two structurally different shapes the protocol has to serve:
//!
//! - [`Quad`] — **fixed size**. Exactly one slab slot, always. The shape every
//!   quad/shadow/underline/sprite kind shares.
//! - [`GlyphRun`] — **variable size**. One slab slot per glyph, so a run's slot
//!   count changes with its content and can cross a size class between frames.
//!   The shape text runs and paths share, and the one that actually exercises
//!   the allocator's fall-up/relocate path and §5.0's "insert/remove that
//!   forces the allocator to relocate" disclosure.
//!
//! Adding a third kind is: implement [`Primitive`], add a [`PrimitiveKind`]
//! variant, add one `PrimitiveStore` field to
//! [`crate::scene::Scene`]. Nothing in `patch`, `scene::slab`, or the upload
//! machinery is written per-kind.
//!
//! # Why a trait plus a small tag enum, rather than a trait object
//!
//! Payloads are plain data of differing shape and size, produced once per
//! changed primitive per frame — the hottest allocation path in the system. A
//! `Box<dyn AnyPrimitive>` per patch would heap-allocate every payload and
//! force a downcast in every store, buying dynamic extensibility the framework
//! does not want: the kind set is closed and known at compile time, because
//! each kind also needs its own render pipeline in `wgpui-wgpu` (§3.5). So the
//! protocol is generic ([`crate::patch::PatchList<P>`],
//! `PrimitiveStore<P>`) and monomorphised per kind, while
//! [`PrimitiveKind`] exists only as a runtime *tag* — enough to address a kind
//! in a flat upload-instruction list without making the payloads dynamic.

/// Which primitive kind a slab, patch list, or upload instruction refers to.
///
/// A runtime tag only: payload types stay statically known via [`Primitive`].
/// Ordering is declaration order and is used as a dense array index by
/// [`PrimitiveKind::index`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum PrimitiveKind {
    /// Fixed-size, one slot per primitive. See [`Quad`].
    Quad,
    /// Variable-size, one slot per glyph. See [`GlyphRun`].
    GlyphRun,
}

impl PrimitiveKind {
    /// Every kind, in declaration order.
    pub const ALL: [PrimitiveKind; PrimitiveKind::COUNT] =
        [PrimitiveKind::Quad, PrimitiveKind::GlyphRun];

    /// Number of kinds; the width of every per-kind array in this crate.
    pub const COUNT: usize = 2;

    /// Dense index into a per-kind array.
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Bytes one slab slot of this kind occupies in the resident buffer.
    pub const fn slot_stride(self) -> usize {
        match self {
            PrimitiveKind::Quad => Quad::SLOT_STRIDE,
            PrimitiveKind::GlyphRun => GlyphRun::SLOT_STRIDE,
        }
    }
}

/// A primitive kind's contract with the scene: how many slab slots one value
/// occupies, and how to write it into the resident buffer.
///
/// Encoding is deliberately byte-oriented rather than `bytemuck`-cast: it
/// keeps `wgpui-core` dependency-free (§3.1), it makes a kind's GPU layout an
/// explicit, reviewable decision rather than a consequence of Rust field
/// order, and it lets a headless test compare resident bytes for exact
/// equality — which is precisely Phase 1's round-trip gate.
pub trait Primitive: Clone + PartialEq + 'static {
    /// The runtime tag for this kind.
    const KIND: PrimitiveKind;

    /// Bytes one slab slot of this kind occupies.
    const SLOT_STRIDE: usize;

    /// How many slab slots this value occupies. Fixed-size kinds return `1`;
    /// variable-size kinds return a content-dependent count, which may be `0`
    /// (an empty text run is representable and costs no slots).
    fn slot_count(&self) -> u32;

    /// Write this value into `destination`, which is exactly
    /// `slot_count() * SLOT_STRIDE` bytes.
    ///
    /// Returns [`EncodeError`] rather than panicking on a length mismatch: a
    /// mis-sized destination is a bookkeeping bug in the scene, and the scene
    /// surfaces it as an error to its caller rather than aborting the process
    /// — the same "a miss is a rebuild, never a crash" discipline R-N §2.2
    /// sets for reconciliation.
    fn encode(&self, destination: &mut [u8]) -> Result<(), EncodeError>;
}

/// A primitive was handed a destination buffer of the wrong size.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct EncodeError {
    /// Bytes the value needs: `slot_count() * SLOT_STRIDE`.
    pub expected: usize,
    /// Bytes the caller actually provided.
    pub actual: usize,
}

impl std::fmt::Display for EncodeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "primitive encode destination is {} bytes, expected {}",
            self.actual, self.expected
        )
    }
}

impl std::error::Error for EncodeError {}

/// Cursor over an encode destination that refuses to write past its end.
///
/// Every `write_*` is checked, so a kind whose `SLOT_STRIDE` disagrees with
/// what its `encode` actually writes fails loudly at the first test that runs
/// it rather than silently corrupting the neighbouring primitive's slot.
struct SlotWriter<'a> {
    destination: &'a mut [u8],
    offset: usize,
}

impl<'a> SlotWriter<'a> {
    fn new(destination: &'a mut [u8]) -> Self {
        Self {
            destination,
            offset: 0,
        }
    }

    fn write_f32(&mut self, value: f32) -> Result<(), EncodeError> {
        self.write_bytes(&value.to_le_bytes())
    }

    fn write_f32_array<const N: usize>(&mut self, values: [f32; N]) -> Result<(), EncodeError> {
        for value in values {
            self.write_f32(value)?;
        }
        Ok(())
    }

    fn write_u32(&mut self, value: u32) -> Result<(), EncodeError> {
        self.write_bytes(&value.to_le_bytes())
    }

    /// Advance over `count` bytes of explicit padding, zeroing them so two
    /// encodings of equal values always produce equal bytes.
    fn write_padding(&mut self, count: usize) -> Result<(), EncodeError> {
        let end = self.offset.checked_add(count).ok_or(EncodeError {
            expected: usize::MAX,
            actual: self.destination.len(),
        })?;
        let available = self.destination.len();
        let slice = self
            .destination
            .get_mut(self.offset..end)
            .ok_or(EncodeError {
                expected: end,
                actual: available,
            })?;
        slice.fill(0);
        self.offset = end;
        Ok(())
    }

    fn write_bytes(&mut self, bytes: &[u8]) -> Result<(), EncodeError> {
        let end = self.offset.checked_add(bytes.len()).ok_or(EncodeError {
            expected: usize::MAX,
            actual: self.destination.len(),
        })?;
        let available = self.destination.len();
        let slice = self
            .destination
            .get_mut(self.offset..end)
            .ok_or(EncodeError {
                expected: end,
                actual: available,
            })?;
        slice.copy_from_slice(bytes);
        self.offset = end;
        Ok(())
    }

    /// Confirm the writer consumed exactly the destination it was given.
    fn finish(self) -> Result<(), EncodeError> {
        if self.offset == self.destination.len() {
            Ok(())
        } else {
            Err(EncodeError {
                expected: self.offset,
                actual: self.destination.len(),
            })
        }
    }
}

/// A fixed-size, rounded, bordered rectangle — the representative
/// **fixed-size** primitive kind (one slab slot, always).
///
/// Field set is the subset of the legacy renderer's quad that matters for
/// exercising the protocol; it is not a port of that struct, and Phase 1 does
/// not draw it anywhere.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quad {
    /// Top-left corner in the owning layer's coordinate space.
    pub origin: [f32; 2],
    /// Width and height.
    pub size: [f32; 2],
    /// Straight-alpha RGBA fill.
    pub background: [f32; 4],
    /// Straight-alpha RGBA border.
    pub border_color: [f32; 4],
    /// Uniform corner radius.
    pub corner_radius: f32,
    /// Uniform border width.
    pub border_width: f32,
}

impl Quad {
    /// A zero-sized, fully transparent quad — a convenient starting point for
    /// tests and for callers building a quad field by field.
    pub const ZERO: Quad = Quad {
        origin: [0.0, 0.0],
        size: [0.0, 0.0],
        background: [0.0, 0.0, 0.0, 0.0],
        border_color: [0.0, 0.0, 0.0, 0.0],
        corner_radius: 0.0,
        border_width: 0.0,
    };
}

impl Primitive for Quad {
    const KIND: PrimitiveKind = PrimitiveKind::Quad;

    // 56 bytes of payload, padded to 64 so a slot boundary is also a 16-byte
    // std430 boundary — the layout a storage-buffer vertex-pulling shader
    // (§1's finding: the renderer already pulls per-instance data this way)
    // reads without a per-field alignment fixup.
    const SLOT_STRIDE: usize = 64;

    fn slot_count(&self) -> u32 {
        1
    }

    fn encode(&self, destination: &mut [u8]) -> Result<(), EncodeError> {
        let mut writer = SlotWriter::new(destination);
        writer.write_f32_array(self.origin)?;
        writer.write_f32_array(self.size)?;
        writer.write_f32_array(self.background)?;
        writer.write_f32_array(self.border_color)?;
        writer.write_f32(self.corner_radius)?;
        writer.write_f32(self.border_width)?;
        writer.write_padding(8)?;
        writer.finish()
    }
}

/// Which atlas allocation a glyph's raster lives in.
///
/// # Why one packed `u32` rather than a `(page, slot)` pair
///
/// Phase 1 left [`Glyph`] with exactly four bytes of tail padding (44 bytes of
/// payload in a 48-byte slot), and this is what Phase 5 puts in it — so giving
/// glyphs a real atlas identity costs zero extra bytes per glyph and leaves
/// [`GlyphRun::SLOT_STRIDE`] unchanged. A `(u32, u32)` pair would have pushed
/// the slot to 64 bytes for information that comfortably fits in 32 bits.
///
/// The split is 8 bits of page and 24 bits of slot: 255 live atlas pages
/// (`u8::MAX`, since `0xFF` is reserved by [`AtlasTileId::NONE`]) against
/// 16,777,215 tiles per page. A 4096×4096 page holds ~65,000 typical glyph
/// rasters, so the slot field has three orders of magnitude of headroom and the
/// page field has more pages than any atlas this framework will open.
///
/// Both halves matter to different consumers, which is why neither can be
/// dropped: the page tells the sprite pipeline which texture to sample, and the
/// slot is what an eviction names when the allocator frees one tile out of a
/// page that is otherwise still live (R-N §4.3).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AtlasTileId(u32);

impl AtlasTileId {
    /// Bits of [`AtlasTileId`] given to the slot; the rest are the page.
    const SLOT_BITS: u32 = 24;
    const SLOT_MASK: u32 = (1 << Self::SLOT_BITS) - 1;
    /// The page index [`AtlasTileId::NONE`] occupies, and so the one no real
    /// page may use.
    const RESERVED_PAGE: u32 = 0xFF;

    /// No tile: a glyph with no raster at all.
    ///
    /// Whitespace shapes to a positioned glyph with a real advance and no
    /// coverage, and a run that carries it is more useful than one that silently
    /// drops it — index-to-position mapping in `line_layout` depends on the
    /// glyph being there. Such a glyph draws nothing and, importantly, is not a
    /// tile reference: it must never make its layer subscribe to an eviction.
    pub const NONE: AtlasTileId = AtlasTileId(u32::MAX);

    /// A tile in `page` at `slot`, or `None` if either exceeds its field.
    ///
    /// Fallible rather than masking, because a silently truncated page index
    /// would make one page's eviction poison a different page's layers — the
    /// exact failure R-N §4.3's hazard is about.
    pub const fn new(page: u32, slot: u32) -> Option<AtlasTileId> {
        if page >= Self::RESERVED_PAGE || slot > Self::SLOT_MASK {
            return None;
        }
        Some(AtlasTileId((page << Self::SLOT_BITS) | slot))
    }

    /// The atlas page this tile lives in, or `None` for [`AtlasTileId::NONE`].
    pub const fn page(self) -> Option<u32> {
        if self.is_none() {
            None
        } else {
            Some(self.0 >> Self::SLOT_BITS)
        }
    }

    /// The tile's slot within its page, or `None` for [`AtlasTileId::NONE`].
    pub const fn slot(self) -> Option<u32> {
        if self.is_none() {
            None
        } else {
            Some(self.0 & Self::SLOT_MASK)
        }
    }

    /// Whether this is [`AtlasTileId::NONE`].
    pub const fn is_none(self) -> bool {
        self.0 == u32::MAX
    }

    /// The packed representation, which is what reaches the GPU.
    pub const fn as_raw(self) -> u32 {
        self.0
    }
}

/// One positioned glyph inside a [`GlyphRun`]. Occupies exactly one slab slot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Glyph {
    /// Baseline-relative position in the owning layer's coordinate space.
    pub position: [f32; 2],
    /// Top-left of this glyph's raster in the atlas, in texels.
    pub atlas_origin: [f32; 2],
    /// Size of this glyph's raster in the atlas, in texels.
    pub atlas_size: [f32; 2],
    /// Font-specific glyph index, carried through for debugging and for
    /// re-rastering this glyph after its tile is evicted.
    pub glyph_id: u32,
    /// Which atlas allocation holds this glyph's raster.
    ///
    /// This is the reference R-N §4.3 warns about — "a retained slab holds tile
    /// references that the atlas may evict" — made addressable. Before Phase 5
    /// a resident glyph named its texels by coordinate only, so nothing could
    /// answer "which layers reference the page I am about to drop."
    pub atlas_tile: AtlasTileId,
}

impl Glyph {
    /// A glyph with every field zeroed and no atlas tile.
    pub const ZERO: Glyph = Glyph {
        position: [0.0, 0.0],
        atlas_origin: [0.0, 0.0],
        atlas_size: [0.0, 0.0],
        glyph_id: 0,
        atlas_tile: AtlasTileId::NONE,
    };
}

/// A run of already-shaped glyphs sharing one colour — the representative
/// **variable-size** primitive kind (one slab slot per glyph).
///
/// Shaping stays on the CPU and is not this type's concern (§6); this is the
/// post-shaping, GPU-bound form, which is what the patch protocol carries.
#[derive(Clone, Debug, PartialEq)]
pub struct GlyphRun {
    /// Straight-alpha RGBA colour, replicated into every glyph's slot because
    /// each glyph is drawn as its own instance.
    pub color: [f32; 4],
    /// The run's glyphs, in visual order.
    pub glyphs: Vec<Glyph>,
}

impl GlyphRun {
    /// A run with no glyphs, which legitimately occupies zero slab slots.
    pub fn empty(color: [f32; 4]) -> Self {
        Self {
            color,
            glyphs: Vec::new(),
        }
    }

    /// Every real atlas tile this run references, [`AtlasTileId::NONE`]
    /// excluded.
    ///
    /// Duplicates are not removed: a run of repeated characters legitimately
    /// references one tile many times, and the callers that matter
    /// ([`crate::scene::atlas`]) are asking a membership question, not counting.
    pub fn atlas_tiles(&self) -> impl Iterator<Item = AtlasTileId> + '_ {
        self.glyphs
            .iter()
            .map(|glyph| glyph.atlas_tile)
            .filter(|tile| !tile.is_none())
    }
}

impl Primitive for GlyphRun {
    const KIND: PrimitiveKind = PrimitiveKind::GlyphRun;

    // 48 bytes of payload per glyph, which is already a 16-byte std430
    // boundary. Phase 1 wrote 44 and padded 4; Phase 5 spent exactly that
    // padding on `Glyph::atlas_tile`, so the stride is unchanged and no
    // resident layout moved.
    const SLOT_STRIDE: usize = 48;

    fn slot_count(&self) -> u32 {
        // A run longer than u32::MAX glyphs is not representable in the slab's
        // slot addressing; saturating here makes the scene reject it as an
        // overflow at allocation time rather than wrapping to a small count
        // and silently truncating the run.
        u32::try_from(self.glyphs.len()).unwrap_or(u32::MAX)
    }

    fn encode(&self, destination: &mut [u8]) -> Result<(), EncodeError> {
        let mut writer = SlotWriter::new(destination);
        for glyph in &self.glyphs {
            writer.write_f32_array(glyph.position)?;
            writer.write_f32_array(glyph.atlas_origin)?;
            writer.write_f32_array(glyph.atlas_size)?;
            writer.write_u32(glyph.glyph_id)?;
            writer.write_f32_array(self.color)?;
            writer.write_u32(glyph.atlas_tile.as_raw())?;
        }
        writer.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quad_encodes_exactly_one_slot() {
        let mut bytes = vec![0xAA; Quad::SLOT_STRIDE];
        let quad = Quad {
            origin: [1.0, 2.0],
            size: [3.0, 4.0],
            background: [0.1, 0.2, 0.3, 1.0],
            border_color: [0.0, 0.0, 0.0, 1.0],
            corner_radius: 5.0,
            border_width: 1.5,
        };
        assert_eq!(quad.slot_count(), 1);
        assert!(quad.encode(&mut bytes).is_ok());
        assert_eq!(&bytes[0..4], &1.0f32.to_le_bytes());
        // Padding must be zeroed, not left as the caller's fill pattern, or
        // two encodings of equal values would not produce equal bytes.
        assert_eq!(&bytes[56..64], &[0u8; 8]);
    }

    #[test]
    fn quad_rejects_a_mis_sized_destination_instead_of_panicking() {
        let mut too_small = vec![0u8; Quad::SLOT_STRIDE - 1];
        assert!(Quad::ZERO.encode(&mut too_small).is_err());
        let mut too_large = vec![0u8; Quad::SLOT_STRIDE + 1];
        assert!(Quad::ZERO.encode(&mut too_large).is_err());
    }

    #[test]
    fn glyph_run_slot_count_tracks_glyph_count() {
        let run = GlyphRun {
            color: [1.0, 1.0, 1.0, 1.0],
            glyphs: vec![Glyph::ZERO; 7],
        };
        assert_eq!(run.slot_count(), 7);
        let mut bytes = vec![0u8; 7 * GlyphRun::SLOT_STRIDE];
        assert!(run.encode(&mut bytes).is_ok());
    }

    #[test]
    fn empty_glyph_run_occupies_no_slots() {
        let run = GlyphRun::empty([1.0, 0.0, 0.0, 1.0]);
        assert_eq!(run.slot_count(), 0);
        let mut bytes: Vec<u8> = Vec::new();
        assert!(run.encode(&mut bytes).is_ok());
    }

    #[test]
    fn glyph_run_replicates_run_colour_into_every_glyph_slot() {
        let run = GlyphRun {
            color: [0.25, 0.5, 0.75, 1.0],
            glyphs: vec![Glyph::ZERO; 2],
        };
        let mut bytes = vec![0u8; 2 * GlyphRun::SLOT_STRIDE];
        assert!(run.encode(&mut bytes).is_ok());
        let first = &bytes[28..44];
        let second = &bytes[GlyphRun::SLOT_STRIDE + 28..GlyphRun::SLOT_STRIDE + 44];
        assert_eq!(first, second);
        assert_eq!(&first[0..4], &0.25f32.to_le_bytes());
    }

    #[test]
    fn an_atlas_tile_round_trips_its_page_and_slot() {
        let tile = AtlasTileId::new(3, 1_234).expect("3 and 1234 are both in range");
        assert_eq!(tile.page(), Some(3));
        assert_eq!(tile.slot(), Some(1_234));
        assert!(!tile.is_none());

        let widest = AtlasTileId::new(0xFE, 0x00FF_FFFF).expect("the widest legal tile");
        assert_eq!(widest.page(), Some(0xFE));
        assert_eq!(widest.slot(), Some(0x00FF_FFFF));
    }

    #[test]
    fn an_out_of_range_page_or_slot_is_refused_rather_than_truncated() {
        // Truncation here would make one page's eviction poison another page's
        // layers, which is exactly the hazard the tile id exists to close.
        assert_eq!(AtlasTileId::new(0xFF, 0), None);
        assert_eq!(AtlasTileId::new(0x100, 0), None);
        assert_eq!(AtlasTileId::new(0, 0x0100_0000), None);
    }

    #[test]
    fn no_real_tile_can_collide_with_none() {
        assert!(AtlasTileId::NONE.is_none());
        assert_eq!(AtlasTileId::NONE.page(), None);
        assert_eq!(AtlasTileId::NONE.slot(), None);
        for page in 0..0xFF {
            let tile = AtlasTileId::new(page, 0x00FF_FFFF).expect("in range");
            assert_ne!(tile, AtlasTileId::NONE);
        }
    }

    #[test]
    fn a_runs_tile_references_skip_glyphs_that_have_no_raster() {
        let tile = AtlasTileId::new(1, 7).expect("in range");
        let run = GlyphRun {
            color: [1.0; 4],
            glyphs: vec![
                Glyph {
                    atlas_tile: tile,
                    ..Glyph::ZERO
                },
                // A space: positioned, advancing, and not a tile reference.
                Glyph::ZERO,
                Glyph {
                    atlas_tile: tile,
                    ..Glyph::ZERO
                },
            ],
        };
        assert_eq!(run.atlas_tiles().collect::<Vec<_>>(), vec![tile, tile]);
    }

    #[test]
    fn the_atlas_tile_lands_in_the_padding_phase_1_left_and_moves_nothing() {
        let tile = AtlasTileId::new(2, 9).expect("in range");
        let run = GlyphRun {
            color: [0.25, 0.5, 0.75, 1.0],
            glyphs: vec![Glyph {
                glyph_id: 42,
                atlas_tile: tile,
                ..Glyph::ZERO
            }],
        };
        let mut bytes = vec![0u8; GlyphRun::SLOT_STRIDE];
        assert!(run.encode(&mut bytes).is_ok());
        assert_eq!(GlyphRun::SLOT_STRIDE, 48, "the stride must not have grown");
        // Everything Phase 1 wrote is where Phase 1 wrote it.
        assert_eq!(&bytes[24..28], &42u32.to_le_bytes());
        assert_eq!(&bytes[28..32], &0.25f32.to_le_bytes());
        // And the tile occupies exactly the four bytes that used to be padding.
        assert_eq!(&bytes[44..48], &tile.as_raw().to_le_bytes());
    }

    #[test]
    fn kind_tags_agree_with_their_payload_types() {
        assert_eq!(PrimitiveKind::Quad.slot_stride(), Quad::SLOT_STRIDE);
        assert_eq!(PrimitiveKind::GlyphRun.slot_stride(), GlyphRun::SLOT_STRIDE);
        for (index, kind) in PrimitiveKind::ALL.iter().enumerate() {
            assert_eq!(kind.index(), index);
        }
    }
}
