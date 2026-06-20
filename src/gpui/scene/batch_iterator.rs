use std::iter::Peekable;
use std::slice;

use super::{
    BackdropBlur,
    MonochromeSprite,
    PaintSurface,
    Path,
    PolychromeSprite,
    PrimitiveBatch,
    PrimitiveKind,
    Quad,
    Scene,
    Shadow,
    Underline,
};
use crate::ScaledPixels;

pub(super) struct BatchIterator<'a> {
    pub(super) shadows: &'a [Shadow],
    pub(super) shadows_start: usize,
    pub(super) shadows_iter: Peekable<slice::Iter<'a, Shadow>>,
    pub(super) backdrop_blurs: &'a [BackdropBlur],
    pub(super) backdrop_blurs_start: usize,
    pub(super) backdrop_blurs_iter: Peekable<slice::Iter<'a, BackdropBlur>>,
    pub(super) quads: &'a [Quad],
    pub(super) quads_start: usize,
    pub(super) quads_iter: Peekable<slice::Iter<'a, Quad>>,
    pub(super) paths: &'a [Path<ScaledPixels>],
    pub(super) paths_start: usize,
    pub(super) paths_iter: Peekable<slice::Iter<'a, Path<ScaledPixels>>>,
    pub(super) underlines: &'a [Underline],
    pub(super) underlines_start: usize,
    pub(super) underlines_iter: Peekable<slice::Iter<'a, Underline>>,
    pub(super) monochrome_sprites: &'a [MonochromeSprite],
    pub(super) monochrome_sprites_start: usize,
    pub(super) monochrome_sprites_iter: Peekable<slice::Iter<'a, MonochromeSprite>>,
    pub(super) polychrome_sprites: &'a [PolychromeSprite],
    pub(super) polychrome_sprites_start: usize,
    pub(super) polychrome_sprites_iter: Peekable<slice::Iter<'a, PolychromeSprite>>,
    pub(super) surfaces: &'a [PaintSurface],
    pub(super) surfaces_start: usize,
    pub(super) surfaces_iter: Peekable<slice::Iter<'a, PaintSurface>>,
}

impl<'a> BatchIterator<'a> {
    pub(super) fn new(scene: &'a Scene) -> Self {
        Self {
            shadows: &scene.shadows,
            shadows_start: 0,
            shadows_iter: scene.shadows.iter().peekable(),
            backdrop_blurs: &scene.backdrop_blurs,
            backdrop_blurs_start: 0,
            backdrop_blurs_iter: scene.backdrop_blurs.iter().peekable(),
            quads: &scene.quads,
            quads_start: 0,
            quads_iter: scene.quads.iter().peekable(),
            paths: &scene.paths,
            paths_start: 0,
            paths_iter: scene.paths.iter().peekable(),
            underlines: &scene.underlines,
            underlines_start: 0,
            underlines_iter: scene.underlines.iter().peekable(),
            monochrome_sprites: &scene.monochrome_sprites,
            monochrome_sprites_start: 0,
            monochrome_sprites_iter: scene.monochrome_sprites.iter().peekable(),
            polychrome_sprites: &scene.polychrome_sprites,
            polychrome_sprites_start: 0,
            polychrome_sprites_iter: scene.polychrome_sprites.iter().peekable(),
            surfaces: &scene.surfaces,
            surfaces_start: 0,
            surfaces_iter: scene.surfaces.iter().peekable(),
        }
    }
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = PrimitiveBatch<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut orders_and_kinds = [
            (
                self.shadows_iter.peek().map(|s| s.order),
                PrimitiveKind::Shadow,
            ),
            (
                self.backdrop_blurs_iter.peek().map(|b| b.order),
                PrimitiveKind::BackdropBlur,
            ),
            (self.quads_iter.peek().map(|q| q.order), PrimitiveKind::Quad),
            (self.paths_iter.peek().map(|q| q.order), PrimitiveKind::Path),
            (
                self.underlines_iter.peek().map(|u| u.order),
                PrimitiveKind::Underline,
            ),
            (
                self.monochrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::MonochromeSprite,
            ),
            (
                self.polychrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::PolychromeSprite,
            ),
            (
                self.surfaces_iter.peek().map(|s| s.order),
                PrimitiveKind::Surface,
            ),
        ];
        orders_and_kinds.sort_by_key(|(order, kind)| (order.unwrap_or(u32::MAX), *kind));

        let first = orders_and_kinds[0];
        let second = orders_and_kinds[1];
        let (batch_kind, max_order_and_kind) = if first.0.is_some() {
            (first.1, (second.0.unwrap_or(u32::MAX), second.1))
        } else {
            return None;
        };

        match batch_kind {
            PrimitiveKind::Shadow => self.next_shadows(batch_kind, max_order_and_kind),
            PrimitiveKind::BackdropBlur => self.next_backdrop_blurs(batch_kind, max_order_and_kind),
            PrimitiveKind::Quad => self.next_quads(batch_kind, max_order_and_kind),
            PrimitiveKind::Path => self.next_paths(batch_kind, max_order_and_kind),
            PrimitiveKind::Underline => self.next_underlines(batch_kind, max_order_and_kind),
            PrimitiveKind::MonochromeSprite => {
                self.next_monochrome_sprites(batch_kind, max_order_and_kind)
            }
            PrimitiveKind::PolychromeSprite => {
                self.next_polychrome_sprites(batch_kind, max_order_and_kind)
            }
            PrimitiveKind::Surface => self.next_surfaces(batch_kind, max_order_and_kind),
        }
    }
}
