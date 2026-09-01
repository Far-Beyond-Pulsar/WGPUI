use std::cell::{RefCell, RefMut};
use std::rc::Rc;
use std::sync::Arc;

use wgpui_core::patch::emit::Emission;
use wgpui_core::scene::atlas::GlyphTileSource;

use crate::patch::{ConversionStats, RunPlacement, glyph_runs};
use crate::shaping::{FontRun, ShapeError, ShapedLine, SharedString, TextShaper};

/// The shared CPU-side text service used by rich text elements.
///
/// Shaping and tile conversion are kept together at the element boundary, but
/// the shaper itself can be shared with the renderer-owned rasterizer. This is
/// what lets a public element populate the same atlas that the WGPU frame loop
/// uploads.
pub struct TextEngine {
    shaper: SharedTextShaper,
    tiles: Box<dyn GlyphTileSource>,
}

impl TextEngine {
    /// Create an engine with an owned shaper and a tile source.
    pub fn new(shaper: TextShaper, tiles: Box<dyn GlyphTileSource>) -> Self {
        Self::new_with_shared_shaper(Rc::new(RefCell::new(shaper)), tiles)
    }

    /// Create an engine that shares its shaper with a rasterizer.
    pub fn new_with_shared_shaper(
        shaper: SharedTextShaper,
        tiles: Box<dyn GlyphTileSource>,
    ) -> Self {
        Self { shaper, tiles }
    }

    /// The shared shaper, for resolving fonts and reading shaping counters.
    pub fn shaper(&self) -> RefMut<'_, TextShaper> {
        self.shaper.borrow_mut()
    }

    /// The shaper handle used by this engine's rasterizer.
    pub fn shared_shaper(&self) -> SharedTextShaper {
        self.shaper.clone()
    }

    /// Access the shared atlas source for layout code that emits wrapped runs.
    pub fn tiles(&mut self) -> &mut dyn GlyphTileSource {
        self.tiles.as_mut()
    }

    /// Shape one line.
    pub fn shape_line(
        &self,
        text: &SharedString,
        font_size: f32,
        runs: &[FontRun],
    ) -> Result<Arc<ShapedLine>, ShapeError> {
        self.shaper.borrow_mut().shape_line(text, font_size, runs)
    }

    /// Convert an already-shaped line to glyph patch payloads.
    pub fn convert_line(
        &mut self,
        line: &ShapedLine,
        placement: RunPlacement,
        emission: &mut Emission,
    ) -> ConversionStats {
        let (converted, stats) = glyph_runs(line, placement, self.tiles.as_mut());
        for run in converted {
            emission.glyph_run(run);
        }
        stats
    }
}

/// A text engine several elements share on the foreground thread.
pub type SharedTextEngine = Rc<RefCell<TextEngine>>;

/// A shaper shared by the text engine and the real glyph rasterizer.
pub type SharedTextShaper = Rc<RefCell<TextShaper>>;
