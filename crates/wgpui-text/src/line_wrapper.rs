//! Line wrapping — today's `src/text_system/line_wrapper.rs`. See
//! docs/gpu-native-architecture.md §3.3.
//!
//! # Moved, and simplified by what the move made available
//!
//! The legacy wrapper measures as it goes: it walks the text a character at a
//! time, asking the platform text system for each character's advance and
//! keeping a `HashMap<char, Pixels>` cache to make that affordable. It works
//! that way because it runs *before* shaping, so per-character advances are all
//! it has.
//!
//! This wraps an already-shaped line, so every advance is already known exactly
//! — including the ones a per-character cache gets wrong, which is not a small
//! set: kerning, ligatures, and any script where a cluster's width is not the
//! sum of its characters' widths. The result is both simpler and more correct,
//! and it costs nothing extra, because [`crate::shaping::TextShaper`] shapes the
//! full line anyway to produce glyph positions.
//!
//! What is preserved from the legacy wrapper is the rule, not the mechanism:
//! break at word boundaries, fall back to breaking mid-word only when a single
//! word does not fit at all.

use crate::line_layout::x_for_index;
use crate::shaping::ShapedLine;

/// Where a wrapped line breaks.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct WrapBoundary {
    /// Byte index in the source text at which the next visual line starts.
    pub index: usize,
}

/// Whether a byte can end a word, and so is a candidate break point.
///
/// Deliberately narrow: ASCII whitespace only. The legacy wrapper's
/// `is_word_char` is wider (it handles CJK, where every character is a break
/// opportunity, and a set of punctuation that may be broken after), and
/// widening this to match is a self-contained change to this one function.
/// Narrow-and-correct beats wide-and-approximate here: a missed break
/// opportunity wraps a line later than ideal, while a wrong one wraps inside a
/// word that should have stayed whole.
fn is_break_opportunity(byte: u8) -> bool {
    byte == b' ' || byte == b'\t'
}

/// Break `line` into visual lines no wider than `wrap_width`.
///
/// Returns the boundaries only — the first visual line always starts at 0 and is
/// not reported, so an empty result means "fits on one line". That is the shape
/// callers want: a boundary list is a list of *breaks*, and reporting a break at
/// the start would make every caller skip it.
pub fn wrap_boundaries(line: &ShapedLine, text: &str, wrap_width: f32) -> Vec<WrapBoundary> {
    let mut boundaries = Vec::new();
    if wrap_width <= 0.0 || line.width <= wrap_width {
        return boundaries;
    }

    let bytes = text.as_bytes();
    let mut line_start = 0usize;
    let mut last_opportunity: Option<usize> = None;
    let mut index = 0usize;

    while index < text.len() {
        if !text.is_char_boundary(index) {
            index += 1;
            continue;
        }

        let x = x_for_index(line, index) - x_for_index(line, line_start);
        // `>` rather than `>=`: a glyph whose leading edge sits exactly at the
        // wrap width still starts within the line's own extent.
        if x > wrap_width && index > line_start {
            let break_at = match last_opportunity {
                // Break after the whitespace, so the space stays on the line it
                // ended rather than starting the next one.
                Some(opportunity) if opportunity > line_start => opportunity,
                // A single word wider than the wrap width: break mid-word at
                // the last index that fit, because the alternative is a line
                // that overflows forever.
                _ => index,
            };
            boundaries.push(WrapBoundary { index: break_at });
            line_start = break_at;
            last_opportunity = None;
            continue;
        }

        if bytes.get(index).copied().is_some_and(is_break_opportunity) {
            last_opportunity = Some(index + 1);
        }
        index += 1;
    }

    boundaries
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shaping::{FontId, GlyphId, ShapedGlyph, ShapedRun};

    /// A shaped line of uniform 10px glyphs, one per ASCII byte.
    ///
    /// Synthetic for the same reason `line_layout`'s are: wrapping is arithmetic
    /// over advances, and real font metrics would make the expected break points
    /// depend on which fonts the machine has.
    fn line(text: &str) -> ShapedLine {
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
        ShapedLine {
            font_size: 16.0,
            width: glyphs.len() as f32 * 10.0,
            ascent: 12.0,
            descent: 4.0,
            len: text.len(),
            runs: vec![ShapedRun {
                font_id: FontId(0),
                glyphs,
            }],
        }
    }

    #[test]
    fn a_line_that_fits_reports_no_breaks() {
        let text = "short";
        assert_eq!(wrap_boundaries(&line(text), text, 100.0), vec![]);
    }

    #[test]
    fn a_break_lands_after_the_space_not_before_it() {
        // "aaa bbb ccc" — 11 glyphs, 110px. At 70px the first line is
        // "aaa bbb" (70px) and the next starts at 'c', index 8.
        let text = "aaa bbb ccc";
        assert_eq!(
            wrap_boundaries(&line(text), text, 70.0),
            vec![WrapBoundary { index: 8 }],
            "the space stays on the line it ended"
        );
    }

    #[test]
    fn several_breaks_are_reported_in_order_and_none_is_at_the_start() {
        let text = "aa bb cc dd ee";
        let boundaries = wrap_boundaries(&line(text), text, 50.0);
        assert!(!boundaries.is_empty());
        assert!(boundaries.iter().all(|boundary| boundary.index > 0));
        assert!(
            boundaries.windows(2).all(|pair| pair[1] > pair[0]),
            "boundaries must advance: {boundaries:?}"
        );
    }

    #[test]
    fn a_word_wider_than_the_wrap_width_breaks_mid_word_rather_than_overflowing() {
        let text = "aaaaaaaaaa";
        let boundaries = wrap_boundaries(&line(text), text, 35.0);
        assert!(
            !boundaries.is_empty(),
            "a line that cannot break at a word must still break, or it overflows forever"
        );
        assert!(boundaries.iter().all(|boundary| boundary.index > 0));
    }

    #[test]
    fn a_nonpositive_wrap_width_is_treated_as_no_wrapping_rather_than_looping() {
        let text = "aaa bbb";
        assert_eq!(wrap_boundaries(&line(text), text, 0.0), vec![]);
        assert_eq!(wrap_boundaries(&line(text), text, -5.0), vec![]);
    }

    #[test]
    fn wrapping_never_breaks_inside_a_multi_byte_character() {
        // Three two-byte characters. Any boundary must be an index the string
        // can actually be split at.
        let text = "ééé";
        let mut shaped = line("aaa");
        shaped.len = text.len();
        for (position, glyph) in shaped
            .runs
            .iter_mut()
            .flat_map(|run| run.glyphs.iter_mut())
            .enumerate()
        {
            glyph.index = position * 2;
        }
        for boundary in wrap_boundaries(&shaped, text, 15.0) {
            assert!(
                text.is_char_boundary(boundary.index),
                "index {} splits a character in {text:?}",
                boundary.index
            );
        }
    }
}
