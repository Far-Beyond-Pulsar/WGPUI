//! A wrapped line: a shaped line plus where it breaks, and how that becomes
//! patch payloads — today's `src/text_system/line.rs`. See
//! docs/gpu-native-architecture.md §3.3.
//!
//! # What a wrapped line is, in 2.0
//!
//! One shaped line and a boundary list, never several shaped lines. Shaping
//! happens once for the whole logical line, and wrapping is a decision *about*
//! that shaping rather than an input to it — which is what makes re-wrapping on
//! a container resize cheap: the boundaries move, the glyphs do not, and
//! [`WrappedLine::glyph_runs`] re-places the same shaped glyphs at new offsets
//! without touching [`crate::shaping::TextShaper`].
//!
//! That is a real difference from the legacy path, where a resize re-runs the
//! wrapper *and* re-lays-out each resulting fragment, and it falls out of
//! wrapping a shaped line rather than wrapping text.

use crate::line_wrapper::{WrapBoundary, wrap_boundaries};
use crate::patch::{RunPlacement, glyph_runs as convert};
use crate::shaping::{ShapedLine, ShapedRun};
use std::sync::Arc;
use wgpui_core::patch::primitive::GlyphRun;
use wgpui_core::scene::atlas::GlyphTileSource;

/// A shaped line together with where it wraps.
#[derive(Clone, Debug)]
pub struct WrappedLine {
    /// The shaped line, in full. Wrapping never re-shapes it.
    pub line: Arc<ShapedLine>,
    /// Where it breaks. Empty means it fits on one visual line.
    pub boundaries: Vec<WrapBoundary>,
}

impl WrappedLine {
    /// Wrap `line` to `wrap_width`. A non-positive width means no wrapping.
    pub fn new(line: Arc<ShapedLine>, text: &str, wrap_width: f32) -> Self {
        let boundaries = wrap_boundaries(&line, text, wrap_width);
        Self { line, boundaries }
    }

    /// A line that is not wrapped at all.
    pub fn unwrapped(line: Arc<ShapedLine>) -> Self {
        Self {
            line,
            boundaries: Vec::new(),
        }
    }

    /// How many visual lines this occupies — always at least one.
    pub fn visual_line_count(&self) -> usize {
        self.boundaries.len() + 1
    }

    /// The rectangle this occupies at `line_height`.
    ///
    /// Width is the widest visual line rather than the shaped line's full width,
    /// which is the difference between a wrapped paragraph's box and the box it
    /// would have needed unwrapped.
    pub fn size(&self, line_height: f32) -> [f32; 2] {
        let widths = self.visual_line_widths(None);
        [widths.iter().copied().fold(0.0, f32::max), line_height * widths.len() as f32]
    }

    /// The widths of the visual lines, optionally limited to a line clamp.
    pub fn visual_line_widths(&self, max_lines: Option<usize>) -> Vec<f32> {
        self.visual_ranges()
            .into_iter()
            .take(max_lines.unwrap_or(usize::MAX))
            .map(|(start, end)| self.advance_between(start, end))
            .collect()
    }

    /// The byte range of each visual line, in order.
    fn visual_ranges(&self) -> Vec<(usize, usize)> {
        let mut ranges = Vec::with_capacity(self.visual_line_count());
        let mut start = 0usize;
        for boundary in &self.boundaries {
            ranges.push((start, boundary.index));
            start = boundary.index;
        }
        ranges.push((start, self.line.len));
        ranges
    }

    fn advance_between(&self, start: usize, end: usize) -> f32 {
        crate::line_layout::x_for_index(&self.line, end)
            - crate::line_layout::x_for_index(&self.line, start)
    }

    /// Which visual line `index` falls on, and the byte it starts at.
    pub fn visual_line_for_index(&self, index: usize) -> (usize, usize) {
        let mut line_number = 0usize;
        let mut start = 0usize;
        for boundary in &self.boundaries {
            if index < boundary.index {
                break;
            }
            line_number += 1;
            start = boundary.index;
        }
        (line_number, start)
    }

    /// Where `index` sits, relative to the wrapped block's top-left.
    ///
    /// `None` past the end of the text, rather than a clamped position: a caller
    /// asking about an index the line does not contain has a bug, and a plausible
    /// wrong answer hides it.
    pub fn position_for_index(&self, index: usize, line_height: f32) -> Option<[f32; 2]> {
        if index > self.line.len {
            return None;
        }
        let (line_number, start) = self.visual_line_for_index(index);
        Some([
            self.advance_between(start, index),
            line_number as f32 * line_height,
        ])
    }

    /// Convert to patch payloads, one set per visual line.
    ///
    /// The shaped glyphs are re-placed, never re-shaped: each visual line is
    /// emitted with its own origin, offset down by `line_height` and back to the
    /// left by where the line starts. That is why re-wrapping on a resize costs
    /// no shaping.
    pub fn glyph_runs(
        &self,
        placement: RunPlacement,
        line_height: f32,
        tiles: &mut dyn GlyphTileSource,
    ) -> Vec<GlyphRun> {
        self.glyph_runs_limited(placement, line_height, tiles, None)
    }

    /// Convert only the first `max_lines` visual lines to patch payloads.
    pub fn glyph_runs_limited(
        &self,
        placement: RunPlacement,
        line_height: f32,
        tiles: &mut dyn GlyphTileSource,
        max_lines: Option<usize>,
    ) -> Vec<GlyphRun> {
        let mut runs = Vec::new();
        for (line_number, (start, end)) in self
            .visual_ranges()
            .into_iter()
            .take(max_lines.unwrap_or(usize::MAX))
            .enumerate()
        {
            let Some(fragment) = self.fragment(start, end) else {
                continue;
            };
            let leading = crate::line_layout::x_for_index(&self.line, start);
            let fragment_placement = RunPlacement {
                origin: [
                    placement.origin[0] - leading,
                    placement.origin[1] + line_number as f32 * line_height,
                ],
                ..placement
            };
            let (converted, _) = convert(&fragment, fragment_placement, tiles);
            runs.extend(converted);
        }
        runs
    }

    /// The shaped glyphs whose source bytes fall in `start..end`, as a line of
    /// their own.
    fn fragment(&self, start: usize, end: usize) -> Option<ShapedLine> {
        if start >= end {
            return None;
        }
        let runs: Vec<ShapedRun> = self
            .line
            .runs
            .iter()
            .filter_map(|run| {
                let glyphs: Vec<_> = run
                    .glyphs
                    .iter()
                    .filter(|glyph| glyph.index >= start && glyph.index < end)
                    .copied()
                    .collect();
                if glyphs.is_empty() {
                    None
                } else {
                    Some(ShapedRun {
                        font_id: run.font_id,
                        glyphs,
                    })
                }
            })
            .collect();
        if runs.is_empty() {
            return None;
        }
        Some(ShapedLine {
            font_size: self.line.font_size,
            width: self.advance_between(start, end),
            ascent: self.line.ascent,
            descent: self.line.descent,
            len: end - start,
            runs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shaping::{FontId, GlyphId, ShapedGlyph};
    use std::collections::HashMap;
    use wgpui_core::patch::primitive::AtlasTileId;
    use wgpui_core::scene::atlas::{GlyphRasterKey, GlyphTile};

    #[derive(Default)]
    struct EveryGlyphHasATile {
        issued: HashMap<GlyphRasterKey, GlyphTile>,
    }

    impl GlyphTileSource for EveryGlyphHasATile {
        fn tile_for(&mut self, key: GlyphRasterKey) -> Option<GlyphTile> {
            let next = self.issued.len();
            Some(*self.issued.entry(key).or_insert_with(|| GlyphTile {
                tile: AtlasTileId::new(0, next as u32).expect("in range"),
                atlas_origin: [0.0, 0.0],
                atlas_size: [8.0, 10.0],
                bearing: [0.0, 0.0],
            }))
        }
    }

    /// Uniform 10px glyphs, one per ASCII byte, as elsewhere in this crate's
    /// arithmetic tests.
    fn shaped(text: &str) -> Arc<ShapedLine> {
        let glyphs: Vec<ShapedGlyph> = text
            .bytes()
            .enumerate()
            .map(|(index, _)| ShapedGlyph {
                id: GlyphId(index as u32),
                position: [index as f32 * 10.0, 0.0],
                index,
                is_emoji: false,
            })
            .collect();
        Arc::new(ShapedLine {
            font_size: 16.0,
            width: glyphs.len() as f32 * 10.0,
            ascent: 12.0,
            descent: 4.0,
            len: text.len(),
            runs: vec![ShapedRun {
                font_id: FontId(0),
                glyphs,
            }],
        })
    }

    #[test]
    fn an_unwrapped_line_is_one_visual_line() {
        let wrapped = WrappedLine::unwrapped(shaped("short"));
        assert_eq!(wrapped.visual_line_count(), 1);
        assert_eq!(wrapped.size(20.0), [50.0, 20.0]);
    }

    #[test]
    fn a_wrapped_lines_box_is_the_widest_visual_line_not_the_shaped_width() {
        let text = "aaa bbb ccc";
        let wrapped = WrappedLine::new(shaped(text), text, 70.0);
        assert_eq!(wrapped.visual_line_count(), 2);
        let [width, height] = wrapped.size(20.0);
        assert!(
            width < 110.0,
            "the box must be the wrapped width, not the unwrapped one: {width}"
        );
        assert_eq!(height, 40.0);
    }

    #[test]
    fn visual_line_widths_respect_a_line_clamp() {
        let wrapped = WrappedLine::new(shaped("aa bb cc dd"), "aa bb cc dd", 30.0);
        let all = wrapped.visual_line_widths(None);
        let first_two = wrapped.visual_line_widths(Some(2));

        assert!(all.len() > 2);
        assert_eq!(first_two, all[..2]);
    }

    #[test]
    fn a_position_on_the_second_visual_line_is_offset_down_and_back_to_the_left() {
        let text = "aaa bbb ccc";
        let wrapped = WrappedLine::new(shaped(text), text, 70.0);
        let first = wrapped.position_for_index(0, 20.0).expect("start of line");
        let second = wrapped.position_for_index(8, 20.0).expect("start of wrap");
        assert_eq!(first, [0.0, 0.0]);
        assert_eq!(
            second,
            [0.0, 20.0],
            "the wrapped fragment starts at the left edge of its own line"
        );
    }

    #[test]
    fn an_index_past_the_end_is_refused_rather_than_clamped() {
        let wrapped = WrappedLine::unwrapped(shaped("abc"));
        assert_eq!(wrapped.position_for_index(3, 20.0), Some([30.0, 0.0]));
        assert_eq!(wrapped.position_for_index(4, 20.0), None);
    }

    #[test]
    fn every_glyph_survives_wrapping_exactly_once() {
        let text = "aaa bbb ccc";
        let wrapped = WrappedLine::new(shaped(text), text, 70.0);
        let mut tiles = EveryGlyphHasATile::default();
        let runs = wrapped.glyph_runs(RunPlacement::default(), 20.0, &mut tiles);
        let emitted: usize = runs.iter().map(|run| run.glyphs.len()).sum();
        assert_eq!(
            emitted,
            wrapped.line.glyph_count(),
            "wrapping must place every shaped glyph, and place none of them twice"
        );
    }

    #[test]
    fn re_wrapping_at_a_new_width_reuses_the_same_shaped_line() {
        let text = "aaa bbb ccc";
        let line = shaped(text);
        let narrow = WrappedLine::new(line.clone(), text, 40.0);
        let wide = WrappedLine::new(line.clone(), text, 200.0);
        assert!(
            Arc::ptr_eq(&narrow.line, &wide.line),
            "a resize must move the boundaries, not re-shape the glyphs"
        );
        assert!(narrow.visual_line_count() > wide.visual_line_count());
        assert_eq!(wide.visual_line_count(), 1);
    }

    #[test]
    fn the_second_visual_lines_glyphs_are_emitted_one_line_height_down() {
        let text = "aaa bbb ccc";
        let wrapped = WrappedLine::new(shaped(text), text, 70.0);
        let mut tiles = EveryGlyphHasATile::default();
        let runs = wrapped.glyph_runs(RunPlacement::default(), 20.0, &mut tiles);
        let ys: Vec<f32> = runs
            .iter()
            .flat_map(|run| run.glyphs.iter().map(|glyph| glyph.position[1]))
            .collect();
        assert!(ys.contains(&0.0));
        assert!(
            ys.contains(&20.0),
            "the wrapped fragment must move down: {ys:?}"
        );
    }
}
