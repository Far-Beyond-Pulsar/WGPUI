//! A reference rasterizer for [`crate::patch::primitive::Quad`], so §5.2's
//! differential harness can compare *pixels* rather than primitive lists.
//! See docs/gpu-native-architecture.md §5.2, R-N §8.5.
//!
//! # Why this exists at all
//!
//! §8's Phase 3 gate is "culled/unculled scenes match exactly." Checked as
//! "the culled list is a subsequence of the unculled one and the dropped
//! entries were covered," that gate restates the culling rule and can only
//! catch a transcription slip, never a wrong rule. Checked as "paint both lists
//! and compare the framebuffers," it catches a wrong rule — which is the class
//! of bug R-N §8.5 says is "invisible in the common case and catastrophic in
//! the rare one."
//!
//! # What it is not
//!
//! Not the renderer. There is no antialiasing, no blur, no gradient, no atlas,
//! and no gamma: one sample at each pixel centre, straight-alpha `over`
//! compositing in linear `f32`. That is a deliberate simplification and it is
//! *sound* for what the harness asks of it, because the conservative opaque
//! region ([`crate::occlusion::coverage::opaque_region`]) is a subset of the
//! area this rasterizer fills with alpha one:
//!
//! - Inset by the corner radius, so every pixel of the region is inside the
//!   rounded rectangle this paints.
//! - Inset by the border width when the border is translucent, so every pixel
//!   of the region takes the background rather than the border band.
//! - When the border is opaque the region is not inset by it — and a pixel
//!   landing in that band takes an alpha-one border colour, which overwrites
//!   just the same.
//!
//! An alpha-one `over` in `f32` is `destination * 0 + source * 1`, exactly, so
//! a culled primitive's pixels are bit-identical to what the occluder writes
//! over them. The harness therefore asserts bit equality, not a tolerance.

use std::cmp::Ordering;

use crate::geometry::Rect;
use crate::patch::primitive::Quad;

/// A straight-alpha linear RGBA framebuffer.
#[derive(Clone, Debug, PartialEq)]
pub struct Framebuffer {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Row-major pixels.
    pub pixels: Vec<[f32; 4]>,
}

/// Where two framebuffers first disagree.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PixelDifference {
    /// Column.
    pub x: u32,
    /// Row.
    pub y: u32,
    /// The left framebuffer's value.
    pub left: [f32; 4],
    /// The right framebuffer's value.
    pub right: [f32; 4],
}

impl std::fmt::Display for PixelDifference {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "pixel ({}, {}): {:?} vs {:?}",
            self.x, self.y, self.left, self.right
        )
    }
}

impl Framebuffer {
    /// A transparent framebuffer.
    pub fn new(width: u32, height: u32) -> Framebuffer {
        Framebuffer {
            width,
            height,
            pixels: vec![[0.0; 4]; (width as usize) * (height as usize)],
        }
    }

    /// The first pixel at which the two differ, scanning row-major.
    ///
    /// Bit equality, deliberately: see this module's doc for why a tolerance
    /// would be the wrong tool.
    pub fn first_difference(&self, other: &Framebuffer) -> Option<PixelDifference> {
        if self.width != other.width || self.height != other.height {
            return Some(PixelDifference {
                x: u32::MAX,
                y: u32::MAX,
                left: [self.width as f32, self.height as f32, 0.0, 0.0],
                right: [other.width as f32, other.height as f32, 0.0, 0.0],
            });
        }
        for index in 0..self.pixels.len() {
            let (Some(left), Some(right)) = (self.pixels.get(index), other.pixels.get(index))
            else {
                continue;
            };
            if left != right {
                let width = self.width.max(1) as usize;
                return Some(PixelDifference {
                    x: (index % width) as u32,
                    y: (index / width) as u32,
                    left: *left,
                    right: *right,
                });
            }
        }
        None
    }

    /// How many pixels this framebuffer actually painted — a guard against a
    /// harness that proves two blank images equal.
    pub fn painted_pixel_count(&self) -> usize {
        self.pixels.iter().filter(|pixel| pixel[3] > 0.0).count()
    }
}

/// Paint `quads` in `draw_order`, skipping any whose `keep` flag is false.
///
/// `draw_order` holds indices into `quads`. `keep`, when supplied, is indexed
/// by the *primitive* index, not by position in `draw_order`, so the two arms
/// of the differential harness differ in exactly one argument.
pub fn rasterize(
    quads: &[Quad],
    draw_order: &[u32],
    keep: Option<&[bool]>,
    width: u32,
    height: u32,
) -> Framebuffer {
    let mut framebuffer = Framebuffer::new(width, height);
    for index in draw_order {
        let position = usize::try_from(*index).unwrap_or(usize::MAX);
        if let Some(flags) = keep
            && !flags.get(position).copied().unwrap_or(true)
        {
            continue;
        }
        let Some(quad) = quads.get(position) else {
            continue;
        };
        paint(&mut framebuffer, quad);
    }
    framebuffer
}

/// Every primitive index in paint order — the identity permutation, for a
/// caller checking that a computed draw order changes nothing visible.
pub fn paint_order(count: usize) -> Vec<u32> {
    (0..count)
        .map(|index| u32::try_from(index).unwrap_or(u32::MAX))
        .collect()
}

fn paint(framebuffer: &mut Framebuffer, quad: &Quad) {
    let bounds = Rect::from_origin_size(quad.origin, quad.size);
    if bounds.is_empty() {
        return;
    }
    // A radius wider than half the box would fold the corner arcs through each
    // other; clamping is what a real rounded-rect shader does, and it only ever
    // makes the painted area *larger* than the conservative opaque region,
    // which is the direction that keeps the harness sound.
    let radius = quad
        .max_corner_radius()
        .min(bounds.width() / 2.0)
        .max(0.0)
        .min(bounds.height() / 2.0);

    let first_x = clamp_to_pixels(bounds.min_x, framebuffer.width);
    let last_x = clamp_to_pixels(bounds.max_x.ceil(), framebuffer.width);
    let first_y = clamp_to_pixels(bounds.min_y, framebuffer.height);
    let last_y = clamp_to_pixels(bounds.max_y.ceil(), framebuffer.height);

    for y in first_y..last_y {
        let sample_y = y as f32 + 0.5;
        for x in first_x..last_x {
            let sample_x = x as f32 + 0.5;
            if !inside_rounded_rect(sample_x, sample_y, bounds, radius) {
                continue;
            }
            let edge_distance = (sample_x - bounds.min_x)
                .min(bounds.max_x - sample_x)
                .min(sample_y - bounds.min_y)
                .min(bounds.max_y - sample_y);
            let source = if edge_distance < quad.max_border_width() {
                quad.border_color
            } else {
                quad.background
            };
            let index = (y as usize) * (framebuffer.width as usize) + (x as usize);
            if let Some(destination) = framebuffer.pixels.get_mut(index) {
                *destination = over(source, *destination);
            }
        }
    }
}

fn clamp_to_pixels(value: f32, limit: u32) -> u32 {
    // Written against `partial_cmp` rather than as `value <= 0.0` because the
    // two differ on NaN, and NaN has to clamp to zero: a degenerate quad must
    // rasterize to nothing rather than to an unbounded span.
    if !matches!(value.partial_cmp(&0.0), Some(Ordering::Greater)) {
        return 0;
    }
    let limit_f32 = limit as f32;
    if value >= limit_f32 {
        limit
    } else {
        value as u32
    }
}

fn inside_rounded_rect(x: f32, y: f32, bounds: Rect, radius: f32) -> bool {
    if x < bounds.min_x || x >= bounds.max_x || y < bounds.min_y || y >= bounds.max_y {
        return false;
    }
    if radius <= 0.0 {
        return true;
    }
    let centre_x = x.clamp(bounds.min_x + radius, bounds.max_x - radius);
    let centre_y = y.clamp(bounds.min_y + radius, bounds.max_y - radius);
    let dx = x - centre_x;
    let dy = y - centre_y;
    dx * dx + dy * dy <= radius * radius
}

/// Straight-alpha `over`. At `source[3] == 1.0` this is an exact copy, which is
/// the property the differential harness rests on.
fn over(source: [f32; 4], destination: [f32; 4]) -> [f32; 4] {
    let inverse = 1.0 - source[3];
    [
        source[0] * source[3] + destination[0] * destination[3] * inverse,
        source[1] * source[3] + destination[1] * destination[3] * inverse,
        source[2] * source[3] + destination[2] * destination[3] * inverse,
        source[3] + destination[3] * inverse,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad(x: f32, y: f32, width: f32, height: f32, alpha: f32) -> Quad {
        Quad {
            origin: [x, y],
            size: [width, height],
            background: [0.25, 0.5, 0.75, alpha],
            border_color: [0.0, 0.0, 0.0, 1.0],
            corner_radii: [0.0; 4],
            border_widths: [0.0; 4],
            material: crate::patch::primitive::Material::Solid,
        }
    }

    #[test]
    fn an_opaque_quad_overwrites_exactly() {
        let below = quad(0.0, 0.0, 4.0, 4.0, 1.0);
        let mut above = quad(0.0, 0.0, 4.0, 4.0, 1.0);
        above.background = [0.1, 0.2, 0.3, 1.0];

        let both = rasterize(&[below, above], &[0, 1], None, 4, 4);
        let only_above = rasterize(&[below, above], &[0, 1], Some(&[false, true]), 4, 4);
        assert_eq!(both.first_difference(&only_above), None);
        assert_eq!(both.pixels.first().copied(), Some([0.1, 0.2, 0.3, 1.0]));
    }

    #[test]
    fn a_translucent_quad_does_not_overwrite() {
        let below = quad(0.0, 0.0, 4.0, 4.0, 1.0);
        let above = quad(0.0, 0.0, 4.0, 4.0, 0.5);
        let both = rasterize(&[below, above], &[0, 1], None, 4, 4);
        let dropped = rasterize(&[below, above], &[0, 1], Some(&[false, true]), 4, 4);
        assert!(
            both.first_difference(&dropped).is_some(),
            "the rasterizer must be able to see a wrong cull, or the harness proves nothing"
        );
    }

    #[test]
    fn a_rounded_quad_leaves_its_corners_unpainted() {
        let mut rounded = quad(0.0, 0.0, 8.0, 8.0, 1.0);
        rounded.corner_radii = [4.0; 4];
        let framebuffer = rasterize(&[rounded], &[0], None, 8, 8);
        assert_eq!(framebuffer.pixels.first().copied(), Some([0.0; 4]));
        assert_eq!(
            framebuffer.pixels.get(3 * 8 + 4).copied().map(|p| p[3]),
            Some(1.0)
        );
    }

    #[test]
    fn a_border_paints_over_the_edge_band() {
        let mut bordered = quad(0.0, 0.0, 6.0, 6.0, 1.0);
        bordered.border_widths = [2.0; 4];
        bordered.border_color = [1.0, 0.0, 0.0, 1.0];
        let framebuffer = rasterize(&[bordered], &[0], None, 6, 6);
        assert_eq!(
            framebuffer.pixels.first().copied(),
            Some([1.0, 0.0, 0.0, 1.0])
        );
        assert_eq!(
            framebuffer.pixels.get(3 * 6 + 3).copied(),
            Some([0.25, 0.5, 0.75, 1.0])
        );
    }

    #[test]
    fn quads_are_clipped_to_the_framebuffer() {
        let framebuffer = rasterize(&[quad(-10.0, -10.0, 40.0, 40.0, 1.0)], &[0], None, 4, 4);
        assert_eq!(framebuffer.painted_pixel_count(), 16);
    }

    #[test]
    fn a_framebuffer_size_mismatch_is_reported_rather_than_ignored() {
        let small = Framebuffer::new(2, 2);
        let large = Framebuffer::new(4, 4);
        assert!(small.first_difference(&large).is_some());
    }

    #[test]
    fn painting_nothing_paints_nothing() {
        let framebuffer = rasterize(&[], &[], None, 4, 4);
        assert_eq!(framebuffer.painted_pixel_count(), 0);
    }
}
