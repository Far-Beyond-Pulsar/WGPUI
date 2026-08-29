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
//! Phase 6.2 is the first phase to actually *do* that, with [`PolySprite`] —
//! and the claim above held: the addition is a variant, a payload type, a
//! `Scene` field, and the three `match` arms the compiler pointed at. Nothing
//! in the slab allocator, the patch protocol, the upload planner, or the
//! indirect-draw slot table needed a line of per-kind code.
//!
//! [`PolySprite`] is structurally [`Quad`]'s shape (fixed size, one slot) with
//! [`Glyph`]'s atlas reference, which is why it is a third kind rather than a
//! third *shape*: it needed no new protocol mechanism, only a new payload and
//! a pipeline that samples a colour page.
//!
//! Phase 6.3 adds [`Shadow`] and [`Underline`], both [`Quad`]'s shape with no
//! atlas reference at all — the cheapest additions so far, and the claim held a
//! second and third time. [`Shadow`] is nonetheless the first kind whose
//! *drawn* extent is larger than its own rectangle: the shader expands the
//! bounds by [`Shadow::BLUR_MARGIN_SIGMAS`] times the blur radius so the
//! Gaussian falloff has somewhere to land. That is a fact about geometry rather
//! than about the protocol — nothing here changes — but it is why
//! [`Shadow::drawn_bounds`] exists and why `wgpui-wgpu` feeds *that* rectangle
//! to the ordering and occlusion passes.
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
    /// Fixed-size, one slot per primitive, drawn under everything else in its
    /// layer. See [`Shadow`].
    Shadow,
    /// Fixed-size, one slot per primitive. See [`Quad`].
    Quad,
    /// Variable-size, pre-tessellated vector geometry. See [`Path`].
    Path,
    /// Fixed-size, one slot per primitive, drawn under its layer's text.
    /// See [`Underline`].
    Underline,
    /// Variable-size, one slot per glyph. See [`GlyphRun`].
    GlyphRun,
    /// Fixed-size, one slot per sprite, referencing a colour atlas tile.
    /// See [`PolySprite`].
    PolySprite,
    /// A rounded rectangle that samples the framebuffer behind it. See
    /// [`BackdropFilter`].
    BackdropFilter,
}

impl PrimitiveKind {
    /// Every kind, in declaration order.
    ///
    /// Declaration order is also *paint* order within a frame, because
    /// [`crate::indirect::SlotTable`] groups the fixed draw sequence by kind and
    /// the render pass issues the groups in this order. `PolySprite` is declared
    /// after `GlyphRun` for that reason and not alphabetically: an image drawn
    /// over a label is the ordinary case (an avatar over a row background, a
    /// thumbnail over a card), and cross-kind z-order within one layer is not
    /// expressible while the ordering dispatch is per kind — see
    /// docs/phase-5.6-results.md, which discloses the same limit for text.
    ///
    /// Phase 6.3's two additions are placed by the legacy renderer's own
    /// tie-break order rather than by preference: `src/scene.rs`'s
    /// `PrimitiveKind` reads `Shadow, Quad, Path, Underline, MonochromeSprite,
    /// PolychromeSprite`, and at equal draw order that discriminant is what
    /// decides which of two primitives paints on top. Dropping `Path` (Phase
    /// 6.4) leaves exactly the sequence below, so 2.0's kind grouping and the
    /// legacy sorter agree about relative paint order by construction.
    pub const ALL: [PrimitiveKind; PrimitiveKind::COUNT] = [
        PrimitiveKind::Shadow,
        PrimitiveKind::Quad,
        PrimitiveKind::Path,
        PrimitiveKind::Underline,
        PrimitiveKind::GlyphRun,
        PrimitiveKind::PolySprite,
        PrimitiveKind::BackdropFilter,
    ];

    /// Number of kinds; the width of every per-kind array in this crate.
    pub const COUNT: usize = 7;

    /// Dense index into a per-kind array.
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Bytes one slab slot of this kind occupies in the resident buffer.
    pub const fn slot_stride(self) -> usize {
        match self {
            PrimitiveKind::Shadow => Shadow::SLOT_STRIDE,
            PrimitiveKind::Quad => Quad::SLOT_STRIDE,
            PrimitiveKind::Path => Path::SLOT_STRIDE,
            PrimitiveKind::Underline => Underline::SLOT_STRIDE,
            PrimitiveKind::GlyphRun => GlyphRun::SLOT_STRIDE,
            PrimitiveKind::PolySprite => PolySprite::SLOT_STRIDE,
            PrimitiveKind::BackdropFilter => BackdropFilter::SLOT_STRIDE,
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
/// # What Phase 6.6 widened, and why
///
/// Phase 1 gave this type one uniform corner radius and one uniform border
/// width, on the stated grounds that the field set was "a subset ... that
/// matters for exercising the protocol." Phase 6.6 is the phase where something
/// real emits it, and the real thing is [`crate::patch::primitive::Quad`]'s
/// legacy counterpart (`src/scene.rs`'s `Quad`), which carries four radii and
/// four widths because the Tailwind surface `wgpui-widgets` presents
/// (`rounded_t_md`, `border_b_1`) is per-corner and per-side. A uniform radius
/// cannot express those, and the alternative — emitting extra plain quads to
/// fake one rounded side — is not what the legacy renderer draws, so it could
/// never be byte-exact against it.
///
/// What is still deliberately absent, and named rather than implied: a gradient
/// or pattern background (2.0's `background` is one solid straight-alpha RGBA),
/// a border style (the legacy dashed-border branch), a content mask (§5.2 sends
/// the clip to the occlusion pass instead), and an element-opacity field
/// (folded into the colours by whoever builds the quad, exactly as
/// `quad_coverage_item` already documents).
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
    /// Corner radii, in the legacy `Corners` order: top-left, top-right,
    /// bottom-right, bottom-left.
    ///
    /// The order matters and is not arbitrary — it is the order the legacy
    /// `Corners<T>` struct declares its fields in, which is the order its WGSL
    /// counterpart reads them in, so a quad built here and a quad built there
    /// select the same radius in the same quadrant.
    pub corner_radii: [f32; 4],
    /// Border widths, in the legacy `Edges` order: top, right, bottom, left.
    pub border_widths: [f32; 4],
}

impl Quad {
    /// Index of the top-left radius in [`Quad::corner_radii`].
    pub const TOP_LEFT: usize = 0;
    /// Index of the top-right radius in [`Quad::corner_radii`].
    pub const TOP_RIGHT: usize = 1;
    /// Index of the bottom-right radius in [`Quad::corner_radii`].
    pub const BOTTOM_RIGHT: usize = 2;
    /// Index of the bottom-left radius in [`Quad::corner_radii`].
    pub const BOTTOM_LEFT: usize = 3;

    /// Index of the top width in [`Quad::border_widths`].
    pub const TOP: usize = 0;
    /// Index of the right width in [`Quad::border_widths`].
    pub const RIGHT: usize = 1;
    /// Index of the bottom width in [`Quad::border_widths`].
    pub const BOTTOM: usize = 2;
    /// Index of the left width in [`Quad::border_widths`].
    pub const LEFT: usize = 3;

    /// A zero-sized, fully transparent quad — a convenient starting point for
    /// tests and for callers building a quad field by field.
    pub const ZERO: Quad = Quad {
        origin: [0.0, 0.0],
        size: [0.0, 0.0],
        background: [0.0, 0.0, 0.0, 0.0],
        border_color: [0.0, 0.0, 0.0, 0.0],
        corner_radii: [0.0; 4],
        border_widths: [0.0; 4],
    };

    /// The largest of the four corner radii.
    ///
    /// What occlusion insets by (§5.2): a rounded corner is the one part of a
    /// quad's rectangle that is not covered, so the inset has to assume the
    /// worst corner rather than an average of them.
    pub fn max_corner_radius(&self) -> f32 {
        self.corner_radii.iter().copied().fold(0.0, f32::max)
    }

    /// The largest of the four border widths.
    pub fn max_border_width(&self) -> f32 {
        self.border_widths.iter().copied().fold(0.0, f32::max)
    }
}

impl Primitive for Quad {
    const KIND: PrimitiveKind = PrimitiveKind::Quad;

    // 80 bytes of payload, which is already a multiple of 16 and so needs no
    // tail padding — the reason Phase 1's 56-byte payload was padded to 64.
    // Phase 6.6 grew it from 64 by replacing two scalars with two `vec4<f32>`s;
    // the shader reads the same five 16-byte rows it always did, plus one more.
    const SLOT_STRIDE: usize = 80;

    fn slot_count(&self) -> u32 {
        1
    }

    fn encode(&self, destination: &mut [u8]) -> Result<(), EncodeError> {
        let mut writer = SlotWriter::new(destination);
        writer.write_f32_array(self.origin)?;
        writer.write_f32_array(self.size)?;
        writer.write_f32_array(self.background)?;
        writer.write_f32_array(self.border_color)?;
        writer.write_f32_array(self.corner_radii)?;
        writer.write_f32_array(self.border_widths)?;
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

/// One image sprite: a rectangle of screen filled from one colour atlas tile.
///
/// The legacy `PolychromeSprite` (`src/scene.rs`), reduced to what 2.0's
/// protocol carries and no further. What is deliberately absent, with the
/// reason in each case:
///
/// - `order` — §5.1's ordering pass computes draw order on the GPU from the
///   primitive's bounds; a CPU-assigned order field is exactly what Phase 3
///   replaced.
/// - `content_mask` — the frame's clip rectangle reaches the occlusion pass as
///   a [`crate::occlusion::CoverageItem`], not as a per-primitive field. This is
///   the same choice [`Quad`] made in Phase 1 and it is not new here.
/// - `AtlasTextureId { index, kind }` — [`AtlasTileId`] already packs page and
///   slot into one word, and the kind is implied: a sprite of this kind is
///   always in a [`crate::scene::atlas::AtlasKind::Polychrome`] page.
/// - per-corner radii — [`Quad`] carries one uniform radius in 2.0 and this
///   matches it rather than inventing a second convention. Four radii is a
///   16-byte addition to both kinds, together, when something needs it.
///
/// # Why the tile's extent is carried as well as the screen rectangle
///
/// The two are not the same number and the difference is the whole of
/// object-fit. `size` is where layout put the image; `atlas_size` is how big the
/// decoded bitmap actually is. A sprite drawn at its natural size has them equal
/// — which is the case the byte-exact proof uses, because a 1:1 blit is the only
/// mapping under which "the pixel on screen *is* the texel in the atlas" is a
/// statement about equality rather than about interpolation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolySprite {
    /// Top-left of the drawn rectangle in the owning layer's coordinate space.
    pub origin: [f32; 2],
    /// Width and height of the drawn rectangle, in the same space.
    pub size: [f32; 2],
    /// Top-left of this sprite's bitmap in the atlas, in texels.
    pub atlas_origin: [f32; 2],
    /// Size of this sprite's bitmap in the atlas, in texels.
    pub atlas_size: [f32; 2],
    /// Uniform corner radius the sprite is clipped to.
    pub corner_radius: f32,
    /// Straight alpha the sprite composites at.
    pub opacity: f32,
    /// Whether to desaturate — the legacy `PolychromeSprite::grayscale`, which
    /// is a `u8` there and a flag here, encoded as a whole word because a
    /// storage-buffer shader reads words.
    pub grayscale: bool,
    /// Which atlas allocation holds this sprite's bitmap.
    ///
    /// [`AtlasTileId::NONE`] for a sprite whose image has not been decoded or
    /// whose tile the atlas refused. Such a sprite keeps its slab slot and draws
    /// nothing, exactly as a whitespace [`Glyph`] does, and — importantly — is
    /// not a tile reference, so it never subscribes its layer to an eviction.
    pub atlas_tile: AtlasTileId,
}

impl PolySprite {
    /// A zero-sized, fully transparent sprite referencing no tile.
    pub const ZERO: PolySprite = PolySprite {
        origin: [0.0, 0.0],
        size: [0.0, 0.0],
        atlas_origin: [0.0, 0.0],
        atlas_size: [0.0, 0.0],
        corner_radius: 0.0,
        opacity: 0.0,
        grayscale: false,
        atlas_tile: AtlasTileId::NONE,
    };

    /// The atlas tile this sprite references, or `None` when it draws nothing.
    ///
    /// The [`GlyphRun::atlas_tiles`] of this kind, singular because a sprite has
    /// exactly one tile. Both exist so [`crate::scene::Scene::layers_referencing`]
    /// can ask the same question of every kind without knowing how many tiles a
    /// kind holds.
    pub fn atlas_tile(&self) -> Option<AtlasTileId> {
        if self.atlas_tile.is_none() {
            None
        } else {
            Some(self.atlas_tile)
        }
    }
}

impl Primitive for PolySprite {
    const KIND: PrimitiveKind = PrimitiveKind::PolySprite;

    // 48 bytes of payload, exactly filled: four `vec2<f32>` (32) plus two `f32`
    // and two `u32` (16). Already a 16-byte std430 boundary with no padding, so
    // unlike `Quad` this kind spends nothing on alignment.
    const SLOT_STRIDE: usize = 48;

    fn slot_count(&self) -> u32 {
        1
    }

    fn encode(&self, destination: &mut [u8]) -> Result<(), EncodeError> {
        let mut writer = SlotWriter::new(destination);
        writer.write_f32_array(self.origin)?;
        writer.write_f32_array(self.size)?;
        writer.write_f32_array(self.atlas_origin)?;
        writer.write_f32_array(self.atlas_size)?;
        writer.write_f32(self.corner_radius)?;
        writer.write_f32(self.opacity)?;
        writer.write_u32(u32::from(self.grayscale))?;
        writer.write_u32(self.atlas_tile.as_raw())?;
        writer.finish()
    }
}

/// One drop shadow: a blurred, rounded rectangle painted under its layer's
/// other content.
///
/// The legacy `Shadow` (`src/scene.rs:1478`), reduced to what 2.0's protocol
/// carries. What is deliberately absent is exactly what [`PolySprite`]'s doc
/// lists and for the same reasons — `order` (§5.1 computes it on the GPU),
/// `content_mask` (the clip reaches the occlusion pass as a
/// [`crate::occlusion::CoverageItem`]), and per-corner radii ([`Quad`] carries
/// one uniform radius in 2.0 and this matches it rather than inventing a second
/// convention).
///
/// # The one thing that is genuinely new about this kind
///
/// A shadow paints *outside* its own rectangle. The legacy vertex shader
/// (`src/platform/cross/shaders/shadows.wgsl:148`) expands the bounds by
/// `3.0 * blur_radius` on every side before projecting them, because the
/// Gaussian's tail has to have somewhere to land; the fragment shader then
/// integrates the falloff over exactly that margin. Every other primitive kind
/// in 2.0 paints within its own `origin`/`size`, and two consumers care about
/// the difference:
///
/// - **Ordering** (§5.1) sorts by bounds, so a shadow's ordering rectangle is
///   the expanded one — that is the area it actually covers.
/// - **Occlusion** (§5.2) must not cull a shadow at all. That is not this
///   phase's invention: [`crate::occlusion::CoverageItem::cullable`] has said
///   so since Phase 3, naming shadows specifically, and the legacy sweep does
///   the same (`src/occlusion.rs:266`).
///
/// [`Shadow::drawn_bounds`] is the one place that arithmetic lives, so the two
/// consumers and the shader cannot drift apart silently.
///
/// **Both of those are latent today, and Phase 6.3 measured that rather than
/// assuming otherwise**: 2.0's occlusion dispatches per primitive kind, so
/// nothing can cull a shadow whatever flag it carries, and reverting either
/// choice leaves every shadow test passing. They are written this way because
/// they are correct and because the day cross-kind occlusion exists a shadow
/// culled against its unblurred rectangle would lose falloff that was never
/// covered. `docs/phase-6.3-results.md` has the experiment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shadow {
    /// Top-left of the *unblurred* rectangle in the owning layer's coordinate
    /// space. The drawn extent is larger — see [`Shadow::drawn_bounds`].
    pub origin: [f32; 2],
    /// Width and height of the unblurred rectangle.
    pub size: [f32; 2],
    /// Straight-alpha RGBA the shadow composites at full coverage.
    pub color: [f32; 4],
    /// Corner radii of the unblurred rectangle, in [`Quad::corner_radii`]'s
    /// order: top-left, top-right, bottom-right, bottom-left.
    ///
    /// Widened from one uniform radius by Phase 6.6, for `Quad`'s reason and at
    /// the same time: a `div()` with `rounded_t_md()` and a `box-shadow` casts a
    /// shadow whose top corners are round and whose bottom corners are square,
    /// and one radius cannot say that. Before the widening a `DivStyle` had to
    /// pick the widest corner and over-round the other three.
    pub corner_radii: [f32; 4],
    /// Gaussian sigma, in the same units as `size`. Zero is a legitimate value
    /// and draws a hard-edged rounded rectangle.
    pub blur_radius: f32,
}

impl Shadow {
    /// How many blur radii the drawn rectangle extends past the shadow's own on
    /// each side.
    ///
    /// Three, transcribed from the legacy vertex shader's `3.0 * blur_radius`
    /// margin, which is also the integration range its fragment shader clamps
    /// to (`start`/`end` at `±3.0 * blur_radius`). A Gaussian carries 99.7% of
    /// its mass inside three sigma, so the two agree that nothing meaningful is
    /// being clipped — but the number is a transcription, not a derivation, and
    /// it must match the shader's or the outermost band of the falloff is cut
    /// off by a triangle edge.
    pub const BLUR_MARGIN_SIGMAS: f32 = 3.0;

    /// A zero-sized, fully transparent, unblurred shadow.
    pub const ZERO: Shadow = Shadow {
        origin: [0.0, 0.0],
        size: [0.0, 0.0],
        color: [0.0, 0.0, 0.0, 0.0],
        corner_radii: [0.0; 4],
        blur_radius: 0.0,
    };

    /// The rectangle this shadow's fragments can actually land in: its own,
    /// grown by [`Shadow::BLUR_MARGIN_SIGMAS`] blur radii on every side.
    ///
    /// Returned as `(origin, size)` rather than a `Rect` so `patch` keeps its
    /// module boundary — `geometry` does not depend on `patch` and this does not
    /// reverse that.
    pub fn drawn_bounds(&self) -> ([f32; 2], [f32; 2]) {
        let margin = Self::BLUR_MARGIN_SIGMAS * self.blur_radius;
        (
            [self.origin[0] - margin, self.origin[1] - margin],
            [self.size[0] + 2.0 * margin, self.size[1] + 2.0 * margin],
        )
    }
}

impl Primitive for Shadow {
    const KIND: PrimitiveKind = PrimitiveKind::Shadow;

    // 52 bytes of payload, padded to 64 so a slot boundary is also a 16-byte
    // std430 boundary — `Quad`'s reasoning, one field set over. Phase 6.6 grew
    // this from 48 by replacing one radius scalar with four.
    const SLOT_STRIDE: usize = 64;

    fn slot_count(&self) -> u32 {
        1
    }

    fn encode(&self, destination: &mut [u8]) -> Result<(), EncodeError> {
        let mut writer = SlotWriter::new(destination);
        writer.write_f32_array(self.origin)?;
        writer.write_f32_array(self.size)?;
        writer.write_f32_array(self.color)?;
        writer.write_f32_array(self.corner_radii)?;
        writer.write_f32(self.blur_radius)?;
        writer.write_padding(12)?;
        writer.finish()
    }
}

/// One underline or strikethrough rule: a straight or wavy band under a run of
/// text.
///
/// The legacy `Underline` (`src/scene.rs:1457`), reduced the same way [`Shadow`]
/// and [`PolySprite`] are — no `order`, no `content_mask`, and no `pad` word,
/// since 2.0's encoder writes an explicit byte layout rather than inheriting
/// Rust's field alignment.
///
/// Unlike [`Shadow`], this kind is [`Quad`]-shaped in *every* respect and not
/// just at the pipeline: it paints inside its own rectangle, and it is an
/// ordinary [`crate::occlusion::CoverageItem::cullee`] — which is the legacy
/// sweep's own classification (`src/occlusion.rs:262` lists `Underline`
/// alongside `Quad`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Underline {
    /// Top-left of the band in the owning layer's coordinate space.
    pub origin: [f32; 2],
    /// Width and height of the band. `size[1]` is the band's *box*, not its
    /// stroke: a wavy underline needs vertical room for the wave, and the
    /// legacy shader derives the wave's frequency and amplitude from the ratio
    /// of `thickness` to this height.
    pub size: [f32; 2],
    /// Straight-alpha RGBA.
    pub color: [f32; 4],
    /// Stroke thickness, in the same units as `size`.
    pub thickness: f32,
    /// Whether to draw a sine wave rather than a straight rule — the
    /// squiggly-underline spelling-error decoration.
    pub wavy: bool,
}

impl Underline {
    /// A zero-sized, fully transparent, straight underline.
    pub const ZERO: Underline = Underline {
        origin: [0.0, 0.0],
        size: [0.0, 0.0],
        color: [0.0, 0.0, 0.0, 0.0],
        thickness: 0.0,
        wavy: false,
    };
}

impl Primitive for Underline {
    const KIND: PrimitiveKind = PrimitiveKind::Underline;

    // 40 bytes of payload, padded to 48 — [`Shadow`]'s layout exactly, which is
    // [`Quad`]'s reasoning. The two kinds sharing a stride is a coincidence of
    // two independent field sets, not a shared decision, and `wgpui-wgpu` gives
    // each its own arena for that reason.
    const SLOT_STRIDE: usize = 48;

    fn slot_count(&self) -> u32 {
        1
    }

    fn encode(&self, destination: &mut [u8]) -> Result<(), EncodeError> {
        let mut writer = SlotWriter::new(destination);
        writer.write_f32_array(self.origin)?;
        writer.write_f32_array(self.size)?;
        writer.write_f32_array(self.color)?;
        writer.write_f32(self.thickness)?;
        // A whole word for one bit, for [`PolySprite::grayscale`]'s reason: a
        // storage-buffer shader reads words.
        writer.write_u32(u32::from(self.wavy))?;
        writer.write_padding(8)?;
        writer.finish()
    }
}

/// One vertex in a Lyon-tessellated path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PathVertex {
    /// Position in the owning layer's pixel space.
    pub position: [f32; 2],
    /// Quadratic-curve coordinates. Lyon-produced fill and stroke triangles
    /// use `[0, 1]`; the fields remain available for legacy curve triangles.
    pub st: [f32; 2],
}

/// A pre-tessellated vector path.
///
/// The CPU geometry producer is intentionally Lyon-compatible: callers can
/// pass the `VertexBuffers` produced by the legacy `PathBuilder` tessellators
/// to [`Path::from_lyon_tessellation`]. The GPU stores the resulting flat
/// vertex stream, so no path interpretation or tessellation occurs in a
/// render pass.
#[derive(Clone, Debug, PartialEq)]
pub struct Path {
    /// Flat triangle-list vertices in draw order.
    pub vertices: Vec<PathVertex>,
    /// Straight-alpha RGBA colour.
    pub color: [f32; 4],
    /// Top-left of the per-path content mask.
    pub clip_origin: [f32; 2],
    /// Size of the per-path content mask.
    pub clip_size: [f32; 2],
    /// Axis-aligned bounds of the tessellated vertices.
    pub bounds_origin: [f32; 2],
    /// Size of the axis-aligned bounds.
    pub bounds_size: [f32; 2],
}

impl Path {
    /// Bytes one tessellated vertex occupies in the GPU arena.
    pub const SLOT_STRIDE: usize = 48;

    /// Build a path from a flat vertex stream.
    pub fn new(vertices: Vec<PathVertex>, color: [f32; 4]) -> Self {
        let (bounds_origin, bounds_size) = bounds_of(&vertices);
        Self {
            vertices,
            color,
            clip_origin: bounds_origin,
            clip_size: bounds_size,
            bounds_origin,
            bounds_size,
        }
    }

    /// Convert the exact triangle output of Lyon's fill/stroke tessellators.
    ///
    /// Malformed index buffers are tolerated by dropping incomplete triangles;
    /// Lyon itself never produces them, but this keeps a bad producer from
    /// turning a render request into an indexing panic.
    pub fn from_lyon_tessellation(
        buffers: lyon::tessellation::VertexBuffers<lyon::math::Point, u16>,
        color: [f32; 4],
    ) -> Self {
        let mut vertices = Vec::with_capacity(buffers.indices.len());
        for triangle in buffers.indices.chunks_exact(3) {
            let Some(first) = buffers.vertices.get(usize::from(triangle[0])) else {
                continue;
            };
            let Some(second) = buffers.vertices.get(usize::from(triangle[1])) else {
                continue;
            };
            let Some(third) = buffers.vertices.get(usize::from(triangle[2])) else {
                continue;
            };
            for point in [first, second, third] {
                vertices.push(PathVertex {
                    position: [point.x, point.y],
                    st: [0.0, 1.0],
                });
            }
        }
        Self::new(vertices, color)
    }

    /// Replace the default bounds mask with the caller's content mask.
    pub fn with_clip(mut self, origin: [f32; 2], size: [f32; 2]) -> Self {
        self.clip_origin = origin;
        self.clip_size = size;
        self
    }
}

impl Primitive for Path {
    const KIND: PrimitiveKind = PrimitiveKind::Path;
    const SLOT_STRIDE: usize = Self::SLOT_STRIDE;

    fn slot_count(&self) -> u32 {
        u32::try_from(self.vertices.len()).unwrap_or(u32::MAX)
    }

    fn encode(&self, destination: &mut [u8]) -> Result<(), EncodeError> {
        let mut writer = SlotWriter::new(destination);
        for vertex in &self.vertices {
            writer.write_f32_array(vertex.position)?;
            writer.write_f32_array(vertex.st)?;
            writer.write_f32_array(self.color)?;
            writer.write_f32_array(self.clip_origin)?;
            writer.write_f32_array(self.clip_size)?;
        }
        writer.finish()
    }
}

/// A rounded rectangle that blurs the already-rendered framebuffer behind it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BackdropFilter {
    /// Top-left of the filter rectangle in the owning layer's pixel space.
    pub origin: [f32; 2],
    /// Width and height of the filter rectangle.
    pub size: [f32; 2],
    /// Top-left of the content mask.
    pub clip_origin: [f32; 2],
    /// Size of the content mask.
    pub clip_size: [f32; 2],
    /// Rounded-corner radii in top-left, top-right, bottom-right, bottom-left
    /// order.
    pub corner_radii: [f32; 4],
    /// Gaussian radius in pixels. The legacy shader caps sampling at 32px.
    pub blur_radius: f32,
    /// Straight-alpha multiplier applied to the sampled result.
    pub opacity: f32,
}

impl BackdropFilter {
    /// A transparent, empty filter.
    pub const ZERO: Self = Self {
        origin: [0.0; 2],
        size: [0.0; 2],
        clip_origin: [0.0; 2],
        clip_size: [0.0; 2],
        corner_radii: [0.0; 4],
        blur_radius: 0.0,
        opacity: 0.0,
    };
}

impl Primitive for BackdropFilter {
    const KIND: PrimitiveKind = PrimitiveKind::BackdropFilter;
    const SLOT_STRIDE: usize = 64;

    fn slot_count(&self) -> u32 {
        1
    }

    fn encode(&self, destination: &mut [u8]) -> Result<(), EncodeError> {
        let mut writer = SlotWriter::new(destination);
        writer.write_f32_array(self.origin)?;
        writer.write_f32_array(self.size)?;
        writer.write_f32_array(self.clip_origin)?;
        writer.write_f32_array(self.clip_size)?;
        writer.write_f32_array(self.corner_radii)?;
        writer.write_f32(self.blur_radius)?;
        writer.write_f32(self.opacity)?;
        writer.write_padding(8)?;
        writer.finish()
    }
}

fn bounds_of(vertices: &[PathVertex]) -> ([f32; 2], [f32; 2]) {
    let Some(first) = vertices.first() else {
        return ([0.0; 2], [0.0; 2]);
    };
    let mut minimum = first.position;
    let mut maximum = first.position;
    for vertex in vertices.iter().skip(1) {
        minimum[0] = minimum[0].min(vertex.position[0]);
        minimum[1] = minimum[1].min(vertex.position[1]);
        maximum[0] = maximum[0].max(vertex.position[0]);
        maximum[1] = maximum[1].max(vertex.position[1]);
    }
    (minimum, [maximum[0] - minimum[0], maximum[1] - minimum[1]])
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
            corner_radii: [5.0, 6.0, 7.0, 8.0],
            border_widths: [1.5, 2.5, 3.5, 4.5],
        };
        assert_eq!(quad.slot_count(), 1);
        assert!(quad.encode(&mut bytes).is_ok());
        assert_eq!(&bytes[0..4], &1.0f32.to_le_bytes());
        // Each of the four radii and four widths must reach its own word, in
        // declaration order: the shader's `pick_corner_radius` selects by
        // quadrant, so a transposed pair rounds the wrong corner and nothing
        // about the total byte count would notice.
        assert_eq!(&bytes[48..52], &5.0f32.to_le_bytes());
        assert_eq!(&bytes[52..56], &6.0f32.to_le_bytes());
        assert_eq!(&bytes[56..60], &7.0f32.to_le_bytes());
        assert_eq!(&bytes[60..64], &8.0f32.to_le_bytes());
        assert_eq!(&bytes[64..68], &1.5f32.to_le_bytes());
        assert_eq!(&bytes[76..80], &4.5f32.to_le_bytes());
        assert_eq!(quad.max_corner_radius(), 8.0);
        assert_eq!(quad.max_border_width(), 4.5);
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
    fn a_shadow_encodes_exactly_one_slot_at_the_offsets_the_shader_reads() {
        let mut bytes = vec![0xAAu8; Shadow::SLOT_STRIDE];
        let shadow = Shadow {
            origin: [10.0, 20.0],
            size: [64.0, 48.0],
            color: [0.25, 0.5, 0.75, 1.0],
            corner_radii: [6.0, 7.0, 8.0, 9.0],
            blur_radius: 4.0,
        };
        assert_eq!(shadow.slot_count(), 1);
        assert!(shadow.encode(&mut bytes).is_ok());
        // Asserted by offset rather than by round-trip, for `PolySprite`'s
        // reason: the only reader is WGSL, where nothing checks the layout.
        assert_eq!(&bytes[0..4], &10.0f32.to_le_bytes());
        assert_eq!(&bytes[4..8], &20.0f32.to_le_bytes());
        assert_eq!(&bytes[8..12], &64.0f32.to_le_bytes());
        assert_eq!(&bytes[12..16], &48.0f32.to_le_bytes());
        assert_eq!(&bytes[16..20], &0.25f32.to_le_bytes());
        assert_eq!(&bytes[28..32], &1.0f32.to_le_bytes());
        // Each radius in its own word, in `pick_corner_radius`'s quadrant order.
        assert_eq!(&bytes[32..36], &6.0f32.to_le_bytes());
        assert_eq!(&bytes[36..40], &7.0f32.to_le_bytes());
        assert_eq!(&bytes[40..44], &8.0f32.to_le_bytes());
        assert_eq!(&bytes[44..48], &9.0f32.to_le_bytes());
        assert_eq!(&bytes[48..52], &4.0f32.to_le_bytes());
        assert_eq!(&bytes[52..64], &[0u8; 12]);
    }

    #[test]
    fn a_shadow_rejects_a_mis_sized_destination_instead_of_panicking() {
        let mut too_small = vec![0u8; Shadow::SLOT_STRIDE - 1];
        assert!(Shadow::ZERO.encode(&mut too_small).is_err());
        let mut too_large = vec![0u8; Shadow::SLOT_STRIDE + 1];
        assert!(Shadow::ZERO.encode(&mut too_large).is_err());
    }

    #[test]
    fn a_shadows_drawn_rectangle_grows_by_three_sigma_on_every_side() {
        // The number the vertex shader hard-codes. If these disagree the
        // outermost band of the falloff is clipped by a triangle edge, which
        // looks like a subtly wrong shadow rather than like an error.
        let shadow = Shadow {
            origin: [100.0, 200.0],
            size: [40.0, 30.0],
            blur_radius: 5.0,
            ..Shadow::ZERO
        };
        assert_eq!(shadow.drawn_bounds(), ([85.0, 185.0], [70.0, 60.0]));

        let unblurred = Shadow {
            blur_radius: 0.0,
            ..shadow
        };
        assert_eq!(
            unblurred.drawn_bounds(),
            (shadow.origin, shadow.size),
            "a zero blur radius must draw exactly its own rectangle, not a \
             degenerate one"
        );
    }

    #[test]
    fn an_underline_encodes_exactly_one_slot_at_the_offsets_the_shader_reads() {
        let mut bytes = vec![0xAAu8; Underline::SLOT_STRIDE];
        let underline = Underline {
            origin: [12.0, 34.0],
            size: [200.0, 5.0],
            color: [1.0, 0.0, 0.0, 1.0],
            thickness: 1.5,
            wavy: true,
        };
        assert_eq!(underline.slot_count(), 1);
        assert!(underline.encode(&mut bytes).is_ok());
        assert_eq!(&bytes[0..4], &12.0f32.to_le_bytes());
        assert_eq!(&bytes[4..8], &34.0f32.to_le_bytes());
        assert_eq!(&bytes[8..12], &200.0f32.to_le_bytes());
        assert_eq!(&bytes[12..16], &5.0f32.to_le_bytes());
        assert_eq!(&bytes[16..20], &1.0f32.to_le_bytes());
        assert_eq!(&bytes[28..32], &1.0f32.to_le_bytes());
        assert_eq!(&bytes[32..36], &1.5f32.to_le_bytes());
        assert_eq!(&bytes[36..40], &1u32.to_le_bytes());
        assert_eq!(&bytes[40..48], &[0u8; 8]);

        // The flag is a whole word and takes exactly two values.
        let mut straight = vec![0xFFu8; Underline::SLOT_STRIDE];
        assert!(
            Underline {
                wavy: false,
                ..underline
            }
            .encode(&mut straight)
            .is_ok()
        );
        assert_eq!(&straight[36..40], &0u32.to_le_bytes());
    }

    #[test]
    fn an_underline_rejects_a_mis_sized_destination_instead_of_panicking() {
        let mut too_small = vec![0u8; Underline::SLOT_STRIDE - 1];
        assert!(Underline::ZERO.encode(&mut too_small).is_err());
        let mut too_large = vec![0u8; Underline::SLOT_STRIDE + 1];
        assert!(Underline::ZERO.encode(&mut too_large).is_err());
    }

    #[test]
    fn the_kind_order_is_the_legacy_renderers_own_tie_break() {
        // Legacy `src/scene.rs:1015`'s `PrimitiveKind` breaks an equal draw
        // order by discriminant: Shadow, Quad, Path, Underline,
        // MonochromeSprite, PolychromeSprite. 2.0 groups its fixed draw
        // sequence by kind in `ALL`'s order, so the two agree about relative
        // paint order only if this holds. `Path` is Phase 6.4 and its absence
        // does not disturb the rest of the sequence.
        assert_eq!(
            PrimitiveKind::ALL,
            [
                PrimitiveKind::Shadow,
                PrimitiveKind::Quad,
                PrimitiveKind::Path,
                PrimitiveKind::Underline,
                PrimitiveKind::GlyphRun,
                PrimitiveKind::PolySprite,
                PrimitiveKind::BackdropFilter,
            ]
        );
        // Spelled again as the four relations that actually matter, so a
        // failure names which one moved rather than printing two arrays.
        assert!(PrimitiveKind::Shadow < PrimitiveKind::Quad);
        assert!(PrimitiveKind::Quad < PrimitiveKind::Path);
        assert!(PrimitiveKind::Path < PrimitiveKind::Underline);
        assert!(PrimitiveKind::Underline < PrimitiveKind::GlyphRun);
        assert!(PrimitiveKind::GlyphRun < PrimitiveKind::PolySprite);
    }

    #[test]
    fn kind_tags_agree_with_their_payload_types() {
        assert_eq!(PrimitiveKind::Shadow.slot_stride(), Shadow::SLOT_STRIDE);
        assert_eq!(
            PrimitiveKind::Underline.slot_stride(),
            Underline::SLOT_STRIDE
        );
        assert_eq!(PrimitiveKind::Quad.slot_stride(), Quad::SLOT_STRIDE);
        assert_eq!(PrimitiveKind::Path.slot_stride(), Path::SLOT_STRIDE);
        assert_eq!(PrimitiveKind::GlyphRun.slot_stride(), GlyphRun::SLOT_STRIDE);
        assert_eq!(
            PrimitiveKind::PolySprite.slot_stride(),
            PolySprite::SLOT_STRIDE
        );
        for (index, kind) in PrimitiveKind::ALL.iter().enumerate() {
            assert_eq!(kind.index(), index);
        }
        assert_eq!(PrimitiveKind::ALL.len(), PrimitiveKind::COUNT);
    }

    fn sprite(tile: AtlasTileId) -> PolySprite {
        PolySprite {
            origin: [10.0, 20.0],
            size: [64.0, 48.0],
            atlas_origin: [128.0, 256.0],
            atlas_size: [64.0, 48.0],
            corner_radius: 4.0,
            opacity: 0.5,
            grayscale: true,
            atlas_tile: tile,
        }
    }

    #[test]
    fn a_poly_sprite_encodes_exactly_one_slot_with_no_padding() {
        let tile = AtlasTileId::new(2, 9).expect("in range");
        let mut bytes = vec![0xAAu8; PolySprite::SLOT_STRIDE];
        let sprite = sprite(tile);
        assert_eq!(sprite.slot_count(), 1);
        assert!(sprite.encode(&mut bytes).is_ok());

        // Every field, at the offset the shader's `SpriteSlot` reads it from.
        // Asserted by offset rather than by round-trip because there is no
        // decoder: the only reader is WGSL, where nothing checks the layout.
        assert_eq!(&bytes[0..4], &10.0f32.to_le_bytes());
        assert_eq!(&bytes[4..8], &20.0f32.to_le_bytes());
        assert_eq!(&bytes[8..12], &64.0f32.to_le_bytes());
        assert_eq!(&bytes[12..16], &48.0f32.to_le_bytes());
        assert_eq!(&bytes[16..20], &128.0f32.to_le_bytes());
        assert_eq!(&bytes[20..24], &256.0f32.to_le_bytes());
        assert_eq!(&bytes[24..28], &64.0f32.to_le_bytes());
        assert_eq!(&bytes[28..32], &48.0f32.to_le_bytes());
        assert_eq!(&bytes[32..36], &4.0f32.to_le_bytes());
        assert_eq!(&bytes[36..40], &0.5f32.to_le_bytes());
        assert_eq!(&bytes[40..44], &1u32.to_le_bytes());
        assert_eq!(&bytes[44..48], &tile.as_raw().to_le_bytes());
    }

    #[test]
    fn a_poly_sprite_rejects_a_mis_sized_destination_instead_of_panicking() {
        let mut too_small = vec![0u8; PolySprite::SLOT_STRIDE - 1];
        assert!(PolySprite::ZERO.encode(&mut too_small).is_err());
        let mut too_large = vec![0u8; PolySprite::SLOT_STRIDE + 1];
        assert!(PolySprite::ZERO.encode(&mut too_large).is_err());
    }

    #[test]
    fn a_sprite_with_no_tile_is_not_a_tile_reference() {
        // The same rule a whitespace glyph follows: it holds its slot, draws
        // nothing, and must never make its layer subscribe to an eviction.
        assert_eq!(PolySprite::ZERO.atlas_tile(), None);
        let tile = AtlasTileId::new(1, 3).expect("in range");
        assert_eq!(sprite(tile).atlas_tile(), Some(tile));
    }

    #[test]
    fn the_grayscale_flag_encodes_as_zero_or_one_and_nothing_else() {
        let tile = AtlasTileId::new(0, 0).expect("in range");
        let mut bytes = vec![0xFFu8; PolySprite::SLOT_STRIDE];
        assert!(
            PolySprite {
                grayscale: false,
                ..sprite(tile)
            }
            .encode(&mut bytes)
            .is_ok()
        );
        assert_eq!(&bytes[40..44], &0u32.to_le_bytes());
    }
}
