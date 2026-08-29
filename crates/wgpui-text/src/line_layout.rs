//! Mapping between byte indices and x positions in a shaped line — today's
//! `src/text_system/line_layout.rs` query half. See
//! docs/gpu-native-architecture.md §3.3.
//!
//! # Moved, not rebuilt — but not copied either
//!
//! The legacy file is 763 lines, most of which is the `LineLayoutCache`: a
//! frame-indexed arena of shaped lines with `reuse_layouts`/`truncate_layouts`
//! bookkeeping, existing because the legacy renderer re-lays-out the whole
//! window every frame and needed somewhere to avoid re-shaping. 2.0 does not
//! re-lay-out the whole window every frame — that is the entire point of
//! ambient reconciliation — and the shaping it does reach is memoised by
//! [`crate::shaping::TextShaper`]'s own cache. Porting the frame-index arena
//! would be porting a workaround for a problem this architecture removed.
//!
//! What is genuinely needed and is here: the query functions. A text cursor, a
//! selection drag, and a click all have to turn an x coordinate into a byte
//! index and back, and those answers are properties of the shaped line rather
//! than of any caching strategy.
//!
//! # Why these operate on advances derived from positions
//!
//! [`crate::shaping::ShapedGlyph`] carries a pen position, not an advance,
//! because that is what the patch conversion needs and what
//! `cosmic-text` reports. An advance is the gap to the next glyph, and the last
//! glyph's advance is the gap to the line's own width — which is why every
//! function here takes the whole [`ShapedLine`] and not a glyph slice.

use crate::shaping::{FontId, ShapedGlyph, ShapedLine};

/// One glyph's placement, flattened across runs.
#[derive(Copy, Clone, Debug, PartialEq)]
struct Placement {
    index: usize,
    x: f32,
    font_id: FontId,
}

fn placements(line: &ShapedLine) -> Vec<Placement> {
    line.runs
        .iter()
        .flat_map(|run| {
            run.glyphs.iter().map(move |glyph: &ShapedGlyph| Placement {
                index: glyph.index,
                x: glyph.position[0],
                font_id: run.font_id,
            })
        })
        .collect()
}

/// The x position of the leading edge of the glyph containing `index`.
///
/// An `index` past the end of the line answers with the line's width, which is
/// what a caret at the end of a line wants; the alternative — refusing — would
/// make every caller special-case the most common position in a text field.
pub fn x_for_index(line: &ShapedLine, index: usize) -> f32 {
    for placement in placements(line) {
        if placement.index >= index {
            return placement.x;
        }
    }
    line.width
}

/// The byte index of the glyph whose horizontal extent contains `x`, or `None`
/// if `x` falls outside the line.
///
/// Outside is a real answer, not a failure: a click to the left of a
/// left-aligned line is not on the line. A caller that wants "the nearest
/// index regardless" asks [`closest_index_for_x`], which is a different
/// question and says so.
pub fn index_for_x(line: &ShapedLine, x: f32) -> Option<usize> {
    if x < 0.0 || x > line.width {
        return None;
    }
    let placements = placements(line);
    for (position, placement) in placements.iter().enumerate() {
        let next = placements
            .get(position + 1)
            .map(|next| next.x)
            .unwrap_or(line.width);
        if x < next {
            return Some(placement.index);
        }
    }
    placements.last().map(|placement| placement.index)
}

/// The byte index nearest `x`, clamped to the line.
///
/// "Nearest" is by glyph *midpoint*, not by leading edge: clicking the right
/// half of a character puts the caret after it, which is what every text editor
/// does and what a leading-edge comparison gets wrong for exactly half of all
/// clicks.
pub fn closest_index_for_x(line: &ShapedLine, x: f32) -> usize {
    let placements = placements(line);
    let mut best = line.len;
    let mut best_distance = f32::INFINITY;

    for (position, placement) in placements.iter().enumerate() {
        let next_x = placements
            .get(position + 1)
            .map(|next| next.x)
            .unwrap_or(line.width);
        let next_index = placements
            .get(position + 1)
            .map(|next| next.index)
            .unwrap_or(line.len);

        for (candidate_x, candidate_index) in
            [(placement.x, placement.index), (next_x, next_index)]
        {
            let distance = (candidate_x - x).abs();
            if distance < best_distance {
                best_distance = distance;
                best = candidate_index;
            }
        }
    }

    if placements.is_empty() { 0 } else { best }
}

/// The face the glyph containing `index` came from, or `None` past the end.
///
/// Needed because fallback means one line can span faces, and a caller
/// measuring or re-rendering a fragment has to ask which face it is in rather
/// than assume the line's nominal one.
pub fn font_id_for_index(line: &ShapedLine, index: usize) -> Option<FontId> {
    placements(line)
        .into_iter()
        .find(|placement| placement.index >= index)
        .map(|placement| placement.font_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shaping::{GlyphId, ShapedRun};

    /// A synthetic line of five 10px-wide glyphs at byte indices 0..5.
    ///
    /// Synthetic on purpose: these functions are arithmetic over positions, and
    /// testing them against real font metrics would make the assertions depend
    /// on which fonts the machine has. Real shaping is exercised in
    /// `shaping.rs` and `patch.rs`.
    fn line() -> ShapedLine {
        ShapedLine {
            font_size: 16.0,
            width: 50.0,
            ascent: 12.0,
            descent: 4.0,
            len: 5,
            runs: vec![ShapedRun {
                font_id: FontId(0),
                glyphs: (0..5)
                    .map(|index| ShapedGlyph {
                        id: GlyphId(index as u32),
                        position: [index as f32 * 10.0, 0.0],
                        index,
                        is_emoji: false,
                    })
                    .collect(),
            }],
        }
    }

    #[test]
    fn x_for_index_walks_the_leading_edges_and_ends_at_the_width() {
        let line = line();
        assert_eq!(x_for_index(&line, 0), 0.0);
        assert_eq!(x_for_index(&line, 3), 30.0);
        assert_eq!(x_for_index(&line, 5), 50.0);
        assert_eq!(
            x_for_index(&line, 99),
            50.0,
            "a caret past the end sits at the end, not nowhere"
        );
    }

    #[test]
    fn index_for_x_reports_the_glyph_whose_extent_contains_the_point() {
        let line = line();
        assert_eq!(index_for_x(&line, 0.0), Some(0));
        assert_eq!(index_for_x(&line, 9.9), Some(0));
        assert_eq!(index_for_x(&line, 10.0), Some(1));
        assert_eq!(index_for_x(&line, 49.9), Some(4));
    }

    #[test]
    fn a_point_outside_the_line_is_reported_as_outside_rather_than_clamped() {
        let line = line();
        assert_eq!(index_for_x(&line, -1.0), None);
        assert_eq!(index_for_x(&line, 51.0), None);
    }

    #[test]
    fn closest_index_snaps_at_the_glyph_midpoint_not_its_leading_edge() {
        let line = line();
        assert_eq!(closest_index_for_x(&line, 4.0), 0);
        assert_eq!(
            closest_index_for_x(&line, 6.0),
            1,
            "clicking the right half of a character puts the caret after it"
        );
        assert_eq!(closest_index_for_x(&line, -20.0), 0, "clamped, not refused");
        assert_eq!(closest_index_for_x(&line, 200.0), 5);
    }

    #[test]
    fn an_empty_line_answers_every_query_without_panicking() {
        let empty = ShapedLine {
            font_size: 16.0,
            len: 0,
            ..ShapedLine::default()
        };
        assert_eq!(x_for_index(&empty, 0), 0.0);
        assert_eq!(index_for_x(&empty, 0.0), None);
        assert_eq!(closest_index_for_x(&empty, 0.0), 0);
        assert_eq!(font_id_for_index(&empty, 0), None);
    }

    #[test]
    fn the_face_reported_for_an_index_is_the_one_that_glyph_came_from() {
        let mut line = line();
        line.runs.push(ShapedRun {
            font_id: FontId(7),
            glyphs: vec![ShapedGlyph {
                id: GlyphId(9),
                position: [50.0, 0.0],
                index: 5,
                is_emoji: true,
            }],
        });
        line.width = 60.0;
        line.len = 6;
        assert_eq!(font_id_for_index(&line, 2), Some(FontId(0)));
        assert_eq!(
            font_id_for_index(&line, 5),
            Some(FontId(7)),
            "fallback means one line can span faces"
        );
    }
}
