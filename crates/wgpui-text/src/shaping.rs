//! `cosmic-text` integration — today's `src/platform/cross/text_system.rs`
//! core, isolated behind one type. See docs/gpu-native-architecture.md §3.3,
//! §6.
//!
//! # What this crate does and does not decide
//!
//! §6 draws the boundary this module respects, and it is worth restating
//! because it is the whole reason `wgpui-text` is a separate crate rather than
//! a folder in `wgpui-wgpu`:
//!
//! > Text shaping stays on the CPU, via `cosmic-text`, unchanged. […] What *is*
//! > data-parallel and already GPU-appropriate — placing already-shaped glyphs
//! > as instanced sprites — stays exactly that.
//!
//! So this module turns `(text, font size, font runs)` into positioned glyph
//! ids, on the CPU, and stops. It never opens a device, never rasterises a
//! glyph, and never allocates an atlas tile. [`crate::patch`] turns what comes
//! out of here into `GlyphRun`/`Glyph` patch payloads plus atlas tile
//! *requests*; `wgpui-wgpu`'s allocator turns requests into coordinates.
//!
//! # Geometry is bare `f32`, following `wgpui-core::geometry`'s precedent
//!
//! No `Pixels`, `Point<T>`, or `Size<T>`. Those are part of the frontend
//! contract §7 freezes, they still live in the legacy crate, and
//! `wgpui-core::geometry` already set the convention for this workspace: declare
//! the small amount of arithmetic actually needed rather than pull the legacy
//! crate across the boundary §3 draws. `Glyph::position` in
//! `wgpui-core::patch::primitive` is `[f32; 2]` for the same reason, so the
//! conversion in [`crate::patch`] is a move, not a unit change.
//!
//! # The shaping cache is not the fast path this phase is gated on
//!
//! There is a cache here ([`TextShaper::shape_line`] returns an `Arc` and
//! remembers it), and it is worth having — but Phase 5's gate is about
//! reconciliation, not memoisation. An unchanged row costs nothing because the
//! reconciler marks it reused and `patch::emit` therefore never calls its
//! emitter, so the shaper is not reached *at all*; the cache only bounds the
//! cost of the cases where it is. [`TextShaper::stats`] reports both numbers
//! separately so a test can tell which mechanism did the work — and
//! `docs/phase-5-results.md` reports them separately for the same reason.

use crate::fonts::fallbacks::FontFallbacks;
use crate::fonts::features::FontFeatures;
use cosmic_text::{
    Attrs, AttrsList, Ellipsize, Family, Font as CosmicFont, FontSystem, Hinting, ShapeBuffer,
    ShapeLine, Shaping, Wrap, fontdb,
};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// An immutable string that is cheap to clone and cheap to compare when it is
/// a clone.
///
/// The legacy `SharedString` is a `SmolStr`, which additionally stores short
/// strings inline; this is the reference-counted half of that, which is the
/// half reconciliation depends on. R-N §2.4 asks a key comparison to
/// short-circuit on unchanged shared clones, and [`PartialEq`] here does
/// exactly that: two handles onto the same allocation answer in one pointer
/// comparison, regardless of how long the text is. That is what makes
/// `StyledText`'s fingerprint affordable for a list of long rows.
#[derive(Clone, Debug, Default, Eq)]
pub struct SharedString(Arc<str>);

impl SharedString {
    /// The text.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Byte length.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the text is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Whether these two handles are clones of one another, rather than merely
    /// equal.
    ///
    /// Exposed because a test that means to prove the short-circuit fires has
    /// to be able to say which case it built.
    pub fn is_clone_of(&self, other: &SharedString) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl PartialEq for SharedString {
    fn eq(&self, other: &Self) -> bool {
        // The clause R-N §2.4 asks for: an unchanged shared clone answers
        // without touching a byte of the text.
        Arc::ptr_eq(&self.0, &other.0) || self.0 == other.0
    }
}

impl Hash for SharedString {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        // By content, never by pointer: two equal strings from different
        // allocations must land in the same cache bucket, or the shaping cache
        // would miss on every freshly-built row.
        self.0.hash(hasher);
    }
}

impl std::fmt::Display for SharedString {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl AsRef<str> for SharedString {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl From<&str> for SharedString {
    fn from(value: &str) -> Self {
        SharedString(Arc::from(value))
    }
}

impl From<String> for SharedString {
    fn from(value: String) -> Self {
        SharedString(Arc::from(value))
    }
}

/// A font face loaded into a [`TextShaper`], addressed by index.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FontId(pub usize);

/// A font-specific glyph index, as produced by shaping.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GlyphId(pub u32);

/// A font's weight on the usual 100–900 scale.
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct FontWeight(pub f32);

impl Default for FontWeight {
    fn default() -> Self {
        FontWeight::NORMAL
    }
}

impl Eq for FontWeight {}

impl Hash for FontWeight {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        // Bit-pattern hashing, so `Hash` and `Eq` agree. A `FontWeight` is
        // always one of a handful of authored constants, never the result of
        // arithmetic, so no NaN or signed-zero case is reachable here — and if
        // one ever were, hashing the bits makes it a cache miss rather than a
        // `HashMap` invariant violation.
        self.0.to_bits().hash(hasher);
    }
}

impl FontWeight {
    /// Thin, 100.
    pub const THIN: FontWeight = FontWeight(100.0);
    /// Extra light, 200.
    pub const EXTRA_LIGHT: FontWeight = FontWeight(200.0);
    /// Light, 300.
    pub const LIGHT: FontWeight = FontWeight(300.0);
    /// Normal, 400.
    pub const NORMAL: FontWeight = FontWeight(400.0);
    /// Medium, 500.
    pub const MEDIUM: FontWeight = FontWeight(500.0);
    /// Semibold, 600.
    pub const SEMIBOLD: FontWeight = FontWeight(600.0);
    /// Bold, 700.
    pub const BOLD: FontWeight = FontWeight(700.0);
    /// Extra bold, 800.
    pub const EXTRA_BOLD: FontWeight = FontWeight(800.0);
    /// Black, 900.
    pub const BLACK: FontWeight = FontWeight(900.0);
}

impl From<FontWeight> for cosmic_text::Weight {
    fn from(value: FontWeight) -> Self {
        cosmic_text::Weight(value.0 as u16)
    }
}

/// Upright, italic, or oblique.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum FontStyle {
    /// Upright.
    #[default]
    Normal,
    /// A dedicated italic face.
    Italic,
    /// A slanted upright face.
    Oblique,
}

impl From<FontStyle> for cosmic_text::Style {
    fn from(value: FontStyle) -> Self {
        match value {
            FontStyle::Normal => cosmic_text::Style::Normal,
            FontStyle::Italic => cosmic_text::Style::Italic,
            FontStyle::Oblique => cosmic_text::Style::Oblique,
        }
    }
}

/// A request for a font face: what a caller asks for, before resolution.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Font {
    /// The family name.
    pub family: SharedString,
    /// The weight.
    pub weight: FontWeight,
    /// Upright, italic, or oblique.
    pub style: FontStyle,
    /// OpenType features to apply.
    pub features: FontFeatures,
    /// Families to try when the primary one has no glyph.
    pub fallbacks: Option<FontFallbacks>,
}

/// A font by family name, normal weight, upright, no features.
pub fn font(family: impl Into<SharedString>) -> Font {
    Font {
        family: family.into(),
        ..Font::default()
    }
}

impl Font {
    /// This font at bold weight.
    pub fn bold(mut self) -> Self {
        self.weight = FontWeight::BOLD;
        self
    }

    /// This font italicised.
    pub fn italic(mut self) -> Self {
        self.style = FontStyle::Italic;
        self
    }
}

/// One contiguous stretch of a line shaped with the same face and spacing.
///
/// `len` is a byte count, not a character count, matching the legacy `FontRun`
/// and matching what `cosmic-text`'s span ranges want.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FontRun {
    /// Byte length of the stretch this run covers.
    pub len: usize,
    /// The resolved face.
    pub font_id: FontId,
    /// Weight to shape at.
    pub weight: FontWeight,
    /// Style to shape at.
    pub style: FontStyle,
    /// Extra advance added after each glyph, in pixels.
    pub letter_spacing: f32,
}

impl FontRun {
    /// A run of `len` bytes in `font_id`, normal weight, upright, no extra
    /// letter spacing.
    pub fn new(len: usize, font_id: FontId) -> Self {
        Self {
            len,
            font_id,
            weight: FontWeight::NORMAL,
            style: FontStyle::Normal,
            letter_spacing: 0.0,
        }
    }
}

/// One shaped glyph, positioned relative to the line's origin.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ShapedGlyph {
    /// The font-specific glyph index.
    pub id: GlyphId,
    /// Offset from the line origin, in pixels.
    pub position: [f32; 2],
    /// Byte offset in the source text of the cluster this glyph came from.
    pub index: usize,
    /// Whether this glyph came from a colour emoji face, which decides whether
    /// its raster is monochrome coverage or full colour — and therefore which
    /// atlas it is allocated out of (`crate::patch`).
    pub is_emoji: bool,
}

/// A stretch of shaped glyphs sharing one face.
#[derive(Clone, Debug, PartialEq)]
pub struct ShapedRun {
    /// The face every glyph in this run came from — which may not be the face
    /// the caller asked for, if fallback substituted one.
    pub font_id: FontId,
    /// The glyphs, in visual order.
    pub glyphs: Vec<ShapedGlyph>,
}

/// One shaped line: the output of the only CPU-bound step §6 keeps.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ShapedLine {
    /// The size it was shaped at, in pixels.
    pub font_size: f32,
    /// Advance width of the whole line.
    pub width: f32,
    /// Distance from the baseline to the highest glyph top.
    pub ascent: f32,
    /// Distance from the baseline to the lowest glyph bottom.
    pub descent: f32,
    /// The runs, in visual order.
    pub runs: Vec<ShapedRun>,
    /// Byte length of the text that was shaped.
    pub len: usize,
}

impl ShapedLine {
    /// Total glyph count across every run.
    pub fn glyph_count(&self) -> usize {
        self.runs.iter().map(|run| run.glyphs.len()).sum()
    }
}

/// What a [`TextShaper`] has actually been made to do.
///
/// Reported rather than inferred, because Phase 5's gate is a claim about work
/// *not* happening and the only honest way to check that is to count.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaperStats {
    /// Times `cosmic-text` was actually asked to shape a line. This is the
    /// number the gate holds flat.
    pub lines_shaped: u64,
    /// Times a shape request was answered from the cache instead.
    pub cache_hits: u64,
    /// Font faces resolved out of the font database.
    pub fonts_resolved: u64,
}

/// A font face could not be resolved.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShapeError {
    /// No face in the font database matched the request, and the database has
    /// no usable face at all to fall back on.
    NoFontMatched {
        /// The family that was asked for.
        family: String,
    },
    /// A [`FontRun`] named a [`FontId`] this shaper never issued.
    UnknownFont(FontId),
    /// The runs' byte lengths do not add up to the text's byte length.
    RunLengthMismatch {
        /// Bytes the runs cover.
        runs: usize,
        /// Bytes the text actually has.
        text: usize,
    },
    /// A font's OpenType feature list could not be converted.
    InvalidFeatures(crate::fonts::features::InvalidFeatureTag),
}

impl std::fmt::Display for ShapeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShapeError::NoFontMatched { family } => {
                write!(formatter, "no font face matched family {family:?}")
            }
            ShapeError::UnknownFont(FontId(id)) => {
                write!(formatter, "font id {id} was never issued by this shaper")
            }
            ShapeError::RunLengthMismatch { runs, text } => write!(
                formatter,
                "font runs cover {runs} bytes but the text is {text} bytes"
            ),
            ShapeError::InvalidFeatures(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for ShapeError {}

impl From<crate::fonts::features::InvalidFeatureTag> for ShapeError {
    fn from(error: crate::fonts::features::InvalidFeatureTag) -> Self {
        ShapeError::InvalidFeatures(error)
    }
}

struct LoadedFont {
    face: Arc<CosmicFont>,
    features: cosmic_text::FontFeatures,
    is_known_emoji_font: bool,
    /// The weight this face was loaded at.
    ///
    /// Held because `cosmic_text::CacheKey` takes it, and the rasteriser
    /// ([`crate::raster`]) has to reproduce the legacy cache key exactly or it
    /// gets a different bitmap for the same glyph. Loading is what decides it,
    /// so loading is what records it — recovering it later from the database
    /// would be re-deriving something already known.
    weight: fontdb::Weight,
}

/// Cache identity of a shape request.
///
/// Floats are keyed by bit pattern so `Hash` and `Eq` agree — the same reason
/// [`FontWeight`] hashes its bits.
#[derive(Clone, PartialEq, Eq, Hash)]
struct ShapeCacheKey {
    text: SharedString,
    font_size_bits: u32,
    runs: Vec<FontRunKey>,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
struct FontRunKey {
    len: usize,
    font_id: FontId,
    weight_bits: u32,
    style: FontStyle,
    letter_spacing_bits: u32,
}

impl From<&FontRun> for FontRunKey {
    fn from(run: &FontRun) -> Self {
        Self {
            len: run.len,
            font_id: run.font_id,
            weight_bits: run.weight.0.to_bits(),
            style: run.style,
            letter_spacing_bits: run.letter_spacing.to_bits(),
        }
    }
}

struct CachedLine {
    line: Arc<ShapedLine>,
    last_used: u64,
}

/// Family names whose faces are treated as colour emoji.
///
/// The legacy system carries the same list (`is_known_emoji_font`), and it
/// matters here for the same reason: an emoji glyph's raster is full colour, so
/// it comes out of a different atlas than a monochrome coverage mask, and
/// `crate::patch` has to know which before it can request a tile.
const KNOWN_EMOJI_FAMILIES: [&str; 4] = [
    "Noto Color Emoji",
    "Apple Color Emoji",
    "Segoe UI Emoji",
    "Twemoji Mozilla",
];

/// The CPU text-shaping engine: `cosmic-text` plus the caches that keep it off
/// the per-frame path.
///
/// Not `Sync`, and deliberately not wrapped in a lock the way the legacy
/// `CosmicTextSystem` is. Everything in 2.0 that reaches this runs on the
/// frame's own thread (§6: "user code stays exactly where it is"), and a lock
/// here would be a lock nothing contends. A caller that wants to share one
/// shaper between elements wraps it in `Rc<RefCell<_>>`, which is what
/// `wgpui-widgets` does.
pub struct TextShaper {
    font_system: FontSystem,
    scratch: ShapeBuffer,
    loaded_fonts: Vec<LoadedFont>,
    font_ids_by_request: HashMap<Font, FontId>,
    font_ids_by_database_id: HashMap<fontdb::ID, FontId>,
    cache: HashMap<ShapeCacheKey, CachedLine>,
    frame: u64,
    stats: ShaperStats,
}

/// Renderer-independent metrics returned by text measurement.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct TextMeasurement {
    /// Logical advance width.
    pub width: f32,
    /// Maximum ascent above the baseline.
    pub ascent: f32,
    /// Maximum descent below the baseline.
    pub descent: f32,
    /// UTF-8 length of the measured text.
    pub byte_length: usize,
}

impl Default for TextShaper {
    fn default() -> Self {
        Self::new()
    }
}

impl TextShaper {
    /// A shaper over the system font database.
    pub fn new() -> Self {
        Self::with_font_system(FontSystem::new())
    }

    /// A shaper over a caller-supplied font database.
    ///
    /// Exists so a test can shape against exactly the faces it embedded rather
    /// than against whatever the machine running it happens to have installed
    /// — the difference between a test that means something on CI and one that
    /// means something only on the author's laptop.
    pub fn with_font_system(font_system: FontSystem) -> Self {
        Self {
            font_system,
            scratch: ShapeBuffer::default(),
            loaded_fonts: Vec::new(),
            font_ids_by_request: HashMap::new(),
            font_ids_by_database_id: HashMap::new(),
            cache: HashMap::new(),
            frame: 0,
            stats: ShaperStats::default(),
        }
    }

    /// What this shaper has been made to do since it was created.
    pub fn stats(&self) -> ShaperStats {
        self.stats
    }

    /// Reset the counters, so a test can measure one frame rather than a run.
    pub fn reset_stats(&mut self) {
        self.stats = ShaperStats::default();
    }

    /// Shaped lines currently held in the cache.
    pub fn cached_line_count(&self) -> usize {
        self.cache.len()
    }

    /// Begin a frame, for cache aging.
    pub fn begin_frame(&mut self) {
        self.frame = self.frame.saturating_add(1);
    }

    /// Drop cached lines untouched for more than `max_age` frames.
    ///
    /// Age rather than a size cap, because the working set of a text UI is
    /// "whatever is on screen", which is a property of time and not of count: a
    /// window showing 40 rows wants 40 entries whether the document has 100
    /// lines or 100,000. A caller that wants a hard ceiling calls this with a
    /// smaller `max_age`.
    pub fn sweep(&mut self, max_age: u64) -> usize {
        let cutoff = self.frame.saturating_sub(max_age);
        let before = self.cache.len();
        self.cache.retain(|_, cached| cached.last_used >= cutoff);
        before - self.cache.len()
    }

    /// Resolve a font request to a loaded face, loading it if needed.
    ///
    /// Resolution is memoised on the request, so the common case — every row in
    /// a list asking for the same font — costs one hash lookup.
    pub fn resolve_font(&mut self, request: &Font) -> Result<FontId, ShapeError> {
        if let Some(font_id) = self.font_ids_by_request.get(request) {
            return Ok(*font_id);
        }

        let families = [fontdb::Family::Name(request.family.as_str())];
        let query = fontdb::Query {
            families: &families,
            weight: request.weight.into(),
            stretch: fontdb::Stretch::Normal,
            style: request.style.into(),
        };
        // A miss on the requested family is not an error while the database has
        // anything at all: falling back to *some* face is what lets a UI render
        // on a machine missing the font it asked for, which is the same
        // decision the legacy system makes.
        let database_id = self
            .font_system
            .db()
            .query(&query)
            .or_else(|| self.font_system.db().faces().next().map(|face| face.id))
            .ok_or_else(|| ShapeError::NoFontMatched {
                family: request.family.as_str().to_owned(),
            })?;

        let features = cosmic_text::FontFeatures::try_from(&request.features)?;
        let font_id = self.load_face(database_id, request.weight.into(), features)?;
        self.font_ids_by_request.insert(request.clone(), font_id);
        Ok(font_id)
    }

    /// The families a resolved face belongs to, for diagnostics and for the
    /// attribute list shaping builds.
    fn family_name(&self, database_id: fontdb::ID) -> Option<String> {
        self.font_system
            .db()
            .face(database_id)
            .and_then(|face| face.families.first().map(|(name, _)| name.clone()))
    }

    fn load_face(
        &mut self,
        database_id: fontdb::ID,
        weight: fontdb::Weight,
        features: cosmic_text::FontFeatures,
    ) -> Result<FontId, ShapeError> {
        if let Some(font_id) = self.font_ids_by_database_id.get(&database_id) {
            return Ok(*font_id);
        }

        let is_known_emoji_font = self
            .family_name(database_id)
            .is_some_and(|family| KNOWN_EMOJI_FAMILIES.contains(&family.as_str()));

        let face = self
            .font_system
            .get_font(database_id, weight)
            .ok_or_else(|| ShapeError::NoFontMatched {
                family: self.family_name(database_id).unwrap_or_default(),
            })?;

        let font_id = FontId(self.loaded_fonts.len());
        self.loaded_fonts.push(LoadedFont {
            face,
            features,
            is_known_emoji_font,
            weight,
        });
        self.font_ids_by_database_id.insert(database_id, font_id);
        self.stats.fonts_resolved += 1;
        Ok(font_id)
    }

    fn loaded(&self, font_id: FontId) -> Result<&LoadedFont, ShapeError> {
        self.loaded_fonts
            .get(font_id.0)
            .ok_or(ShapeError::UnknownFont(font_id))
    }

    /// Whether a resolved face is a colour emoji face.
    pub fn is_emoji_font(&self, font_id: FontId) -> Result<bool, ShapeError> {
        Ok(self.loaded(font_id)?.is_known_emoji_font)
    }

    /// The database identity and weight of a resolved face — everything
    /// `cosmic_text::CacheKey` needs to name an outline.
    ///
    /// Returned by value rather than by reference so a caller can hold it while
    /// borrowing [`Self::font_system_mut`], which rasterising requires: the same
    /// split the legacy `CosmicTextSystemState` gets for free by keeping the
    /// swash cache, the font system, and the loaded-face table in one struct.
    pub(crate) fn raster_face(
        &self,
        font_id: FontId,
    ) -> Result<(fontdb::ID, fontdb::Weight), ShapeError> {
        let loaded = self.loaded(font_id)?;
        Ok((loaded.face.id(), loaded.weight))
    }

    /// The font database this shaper shapes against.
    ///
    /// `pub(crate)` and not public: `cosmic_text::FontSystem` is an
    /// implementation detail §3.3 keeps inside this crate, and handing it out
    /// would let a caller mutate the database a cached `ShapedLine` was shaped
    /// against. [`crate::raster`] needs it because `SwashCache::get_image` takes
    /// it, and `crate::raster` is inside this crate for exactly that reason.
    pub(crate) fn font_system_mut(&mut self) -> &mut FontSystem {
        &mut self.font_system
    }

    /// Shape one line of text.
    ///
    /// The `Arc` is not decoration: an unchanged line is returned as the same
    /// allocation every caller already holds, so a repeat request costs a hash
    /// lookup and a refcount bump rather than a shaping pass. See this module's
    /// doc for why that is a backstop and not the phase's fast path.
    pub fn shape_line(
        &mut self,
        text: &SharedString,
        font_size: f32,
        runs: &[FontRun],
    ) -> Result<Arc<ShapedLine>, ShapeError> {
        let covered: usize = runs.iter().map(|run| run.len).sum();
        if covered != text.len() {
            return Err(ShapeError::RunLengthMismatch {
                runs: covered,
                text: text.len(),
            });
        }

        let key = ShapeCacheKey {
            text: text.clone(),
            font_size_bits: font_size.to_bits(),
            runs: runs.iter().map(FontRunKey::from).collect(),
        };
        if let Some(cached) = self.cache.get_mut(&key) {
            cached.last_used = self.frame;
            self.stats.cache_hits += 1;
            return Ok(cached.line.clone());
        }

        let line = Arc::new(self.shape_line_uncached(text.as_str(), font_size, runs)?);
        self.cache.insert(
            key,
            CachedLine {
                line: line.clone(),
                last_used: self.frame,
            },
        );
        Ok(line)
    }

    /// Measure text through the same cached shaping path used for emission.
    pub fn measure_line(
        &mut self,
        text: &SharedString,
        font_size: f32,
        runs: &[FontRun],
    ) -> Result<TextMeasurement, ShapeError> {
        let line = self.shape_line(text, font_size, runs)?;
        Ok(TextMeasurement {
            width: line.width,
            ascent: line.ascent,
            descent: line.descent,
            byte_length: line.len,
        })
    }

    /// Shape without consulting or populating the cache.
    ///
    /// Public because measuring the cost the cache avoids requires a way to pay
    /// it deliberately.
    pub fn shape_line_uncached(
        &mut self,
        text: &str,
        font_size: f32,
        runs: &[FontRun],
    ) -> Result<ShapedLine, ShapeError> {
        self.stats.lines_shaped += 1;

        let mut attributes = AttrsList::new(&Attrs::new());
        let mut offset = 0;
        for run in runs {
            let loaded = self.loaded(run.font_id)?;
            let features = loaded.features.clone();
            let database_id = loaded.face.id();
            let stretch = self
                .font_system
                .db()
                .face(database_id)
                .map(|face| face.stretch)
                .unwrap_or(fontdb::Stretch::Normal);
            let family = self.family_name(database_id).unwrap_or_default();

            attributes.add_span(
                offset..(offset + run.len),
                // `metadata` carries our own `FontId` through shaping so the
                // glyph loop below can attribute each glyph to the run that
                // asked for it — and detect when fallback substituted a
                // different face, which is the one case where it cannot.
                &Attrs::new()
                    .metadata(run.font_id.0)
                    .family(Family::Name(&family))
                    .stretch(stretch)
                    .style(run.style.into())
                    .weight(run.weight.into())
                    .letter_spacing(run.letter_spacing)
                    .font_features(features),
            );
            offset += run.len;
        }

        let shaped = ShapeLine::new(
            &mut self.font_system,
            text,
            &attributes,
            Shaping::Advanced,
            TAB_WIDTH,
        );

        let mut layout_lines = Vec::with_capacity(1);
        shaped.layout_to_buffer(
            &mut self.scratch,
            font_size,
            // Wrapping is this crate's own job (`line_wrapper`), not
            // cosmic-text's, exactly as in the legacy system — the wrapper has
            // to agree with the truncation and boundary rules the frontend
            // exposes, which cosmic-text does not know about.
            None,
            Wrap::None,
            Ellipsize::None,
            None,
            &mut layout_lines,
            None,
            Hinting::default(),
        );

        let Some(layout) = layout_lines.first() else {
            // Empty input shapes to no visual lines at all, which is a valid
            // answer and not a failure.
            return Ok(ShapedLine {
                font_size,
                len: text.len(),
                ..ShapedLine::default()
            });
        };

        let mut runs_out: Vec<ShapedRun> = Vec::new();
        for glyph in &layout.glyphs {
            let mut font_id = FontId(glyph.metadata);
            if self
                .loaded_fonts
                .get(font_id.0)
                .is_none_or(|loaded| loaded.face.id() != glyph.font_id)
            {
                // Fallback picked a face the caller never asked for. Load it
                // now so the run carries a `FontId` that resolves, rather than
                // one that points at the requested face and lies about which
                // atlas entry the glyph belongs to.
                font_id = self.load_face(
                    glyph.font_id,
                    glyph.font_weight,
                    cosmic_text::FontFeatures::new(),
                )?;
            }
            let is_emoji = self.loaded(font_id)?.is_known_emoji_font;

            // cosmic-text reports a missing-glyph box as glyph 3 from an emoji
            // face when a codepoint has no colour form; drawing it produces a
            // visible tofu where the legacy system draws nothing.
            if glyph.glyph_id == MISSING_EMOJI_GLYPH && is_emoji {
                continue;
            }

            let shaped_glyph = ShapedGlyph {
                id: GlyphId(u32::from(glyph.glyph_id)),
                position: [glyph.x, glyph.y],
                index: glyph.start,
                is_emoji,
            };

            match runs_out.last_mut() {
                Some(last) if last.font_id == font_id => last.glyphs.push(shaped_glyph),
                _ => runs_out.push(ShapedRun {
                    font_id,
                    glyphs: vec![shaped_glyph],
                }),
            }
        }

        Ok(ShapedLine {
            font_size,
            width: layout.w,
            ascent: layout.max_ascent,
            descent: layout.max_descent,
            runs: runs_out,
            len: text.len(),
        })
    }
}

/// Tab stops every four columns, matching the legacy system's `layout_line`.
const TAB_WIDTH: u16 = 4;

/// cosmic-text's missing-glyph index.
const MISSING_EMOJI_GLYPH: u16 = 3;

#[cfg(test)]
mod tests {
    use super::*;

    /// A shaper over a database holding exactly one embedded face.
    ///
    /// Test fonts are the machine's own, which is the honest thing to do for a
    /// crate whose entire job is talking to the system font database — but it
    /// means a test must not assert glyph *ids* or exact advances, only
    /// structural facts. Every assertion below is structural for that reason,
    /// and it is a real limitation, not a shortcut: an advance-width regression
    /// in cosmic-text would not be caught here.
    fn shaper() -> TextShaper {
        TextShaper::new()
    }

    fn some_font(shaper: &mut TextShaper) -> FontId {
        shaper
            .resolve_font(&font("Segoe UI"))
            .expect("the font database must offer at least one face")
    }

    #[test]
    fn shared_string_equality_short_circuits_on_a_clone() {
        let original = SharedString::from("a fairly long row of list text");
        let cloned = original.clone();
        let rebuilt = SharedString::from("a fairly long row of list text");

        assert!(original.is_clone_of(&cloned));
        assert!(!original.is_clone_of(&rebuilt));
        // Both compare equal; only the first can do it without reading bytes.
        assert_eq!(original, cloned);
        assert_eq!(original, rebuilt);
    }

    #[test]
    fn shared_string_hashes_by_content_so_a_rebuilt_row_still_hits_the_cache() {
        use std::collections::hash_map::DefaultHasher;
        let hash = |value: &SharedString| {
            let mut hasher = DefaultHasher::new();
            value.hash(&mut hasher);
            hasher.finish()
        };
        assert_eq!(
            hash(&SharedString::from("row")),
            hash(&SharedString::from("row"))
        );
    }

    #[test]
    fn shaping_a_line_produces_one_glyph_per_ascii_character() {
        let mut shaper = shaper();
        let font_id = some_font(&mut shaper);
        let text = SharedString::from("Hello");
        let line = shaper
            .shape_line(&text, 16.0, &[FontRun::new(text.len(), font_id)])
            .expect("shaping plain ASCII must succeed");

        assert_eq!(line.len, 5);
        assert_eq!(line.font_size, 16.0);
        assert_eq!(line.glyph_count(), 5);
        assert!(line.width > 0.0, "shaped text must have an advance width");
        assert!(line.ascent > 0.0);
        assert!(
            line.runs.iter().all(|run| !run.glyphs.is_empty()),
            "an empty run must never be emitted"
        );
    }

    #[test]
    fn glyph_positions_advance_left_to_right() {
        let mut shaper = shaper();
        let font_id = some_font(&mut shaper);
        let text = SharedString::from("abcdef");
        let line = shaper
            .shape_line(&text, 16.0, &[FontRun::new(text.len(), font_id)])
            .expect("shaping must succeed");
        let xs: Vec<f32> = line
            .runs
            .iter()
            .flat_map(|run| run.glyphs.iter().map(|glyph| glyph.position[0]))
            .collect();
        assert!(
            xs.windows(2).all(|pair| pair[1] > pair[0]),
            "left-to-right text must advance monotonically: {xs:?}"
        );
    }

    #[test]
    fn an_empty_line_shapes_to_nothing_rather_than_failing() {
        let mut shaper = shaper();
        let text = SharedString::from("");
        let line = shaper
            .shape_line(&text, 16.0, &[])
            .expect("an empty line is a valid line");
        assert_eq!(line.glyph_count(), 0);
        assert_eq!(line.len, 0);
    }

    #[test]
    fn runs_that_do_not_cover_the_text_are_rejected_instead_of_shaping_garbage() {
        let mut shaper = shaper();
        let font_id = some_font(&mut shaper);
        let text = SharedString::from("Hello");
        assert_eq!(
            shaper.shape_line(&text, 16.0, &[FontRun::new(2, font_id)]),
            Err(ShapeError::RunLengthMismatch { runs: 2, text: 5 })
        );
    }

    #[test]
    fn an_unissued_font_id_is_reported_rather_than_indexed_into() {
        let mut shaper = shaper();
        let text = SharedString::from("Hi");
        assert_eq!(
            shaper.shape_line(&text, 16.0, &[FontRun::new(2, FontId(9_999))]),
            Err(ShapeError::UnknownFont(FontId(9_999)))
        );
    }

    #[test]
    fn reshaping_identical_input_hits_the_cache_and_returns_the_same_allocation() {
        let mut shaper = shaper();
        let font_id = some_font(&mut shaper);
        let text = SharedString::from("cached row");
        let runs = [FontRun::new(text.len(), font_id)];

        shaper.reset_stats();
        let first = shaper.shape_line(&text, 16.0, &runs).expect("shape");
        assert_eq!(shaper.stats().lines_shaped, 1);
        assert_eq!(shaper.stats().cache_hits, 0);

        let second = shaper.shape_line(&text, 16.0, &runs).expect("shape");
        assert_eq!(
            shaper.stats().lines_shaped,
            1,
            "an identical request must not reach cosmic-text again"
        );
        assert_eq!(shaper.stats().cache_hits, 1);
        assert!(Arc::ptr_eq(&first, &second));

        // A different size is a different line and must actually shape.
        shaper.shape_line(&text, 17.0, &runs).expect("shape");
        assert_eq!(shaper.stats().lines_shaped, 2);
    }

    #[test]
    fn a_rebuilt_but_equal_string_still_hits_the_cache() {
        let mut shaper = shaper();
        let font_id = some_font(&mut shaper);
        let runs = [FontRun::new(3, font_id)];

        shaper.reset_stats();
        shaper
            .shape_line(&SharedString::from("row"), 16.0, &runs)
            .expect("shape");
        shaper
            .shape_line(&SharedString::from("row"), 16.0, &runs)
            .expect("shape");
        assert_eq!(shaper.stats().lines_shaped, 1);
        assert_eq!(shaper.stats().cache_hits, 1);
    }

    #[test]
    fn sweeping_drops_lines_that_have_aged_out_and_keeps_the_ones_in_use() {
        let mut shaper = shaper();
        let font_id = some_font(&mut shaper);
        let stale = SharedString::from("scrolled away");
        let live = SharedString::from("still on screen");

        shaper
            .shape_line(&stale, 16.0, &[FontRun::new(stale.len(), font_id)])
            .expect("shape");
        for _ in 0..5 {
            shaper.begin_frame();
        }
        shaper
            .shape_line(&live, 16.0, &[FontRun::new(live.len(), font_id)])
            .expect("shape");

        assert_eq!(shaper.cached_line_count(), 2);
        assert_eq!(shaper.sweep(2), 1);
        assert_eq!(shaper.cached_line_count(), 1);
        // The one still in use survives, and is still a cache hit.
        shaper.reset_stats();
        shaper
            .shape_line(&live, 16.0, &[FontRun::new(live.len(), font_id)])
            .expect("shape");
        assert_eq!(shaper.stats().cache_hits, 1);
    }

    #[test]
    fn resolving_the_same_font_twice_loads_one_face() {
        let mut shaper = shaper();
        shaper.reset_stats();
        let first = shaper.resolve_font(&font("Segoe UI")).expect("resolve");
        let resolved_once = shaper.stats().fonts_resolved;
        let second = shaper.resolve_font(&font("Segoe UI")).expect("resolve");
        assert_eq!(first, second);
        assert_eq!(shaper.stats().fonts_resolved, resolved_once);
    }

    #[test]
    fn an_unavailable_family_falls_back_rather_than_failing() {
        let mut shaper = shaper();
        let resolved = shaper.resolve_font(&font("A Font That Does Not Exist At All"));
        assert!(
            resolved.is_ok(),
            "a missing family must fall back to some face, as the legacy system does"
        );
    }
}
