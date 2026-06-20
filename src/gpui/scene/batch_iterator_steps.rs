use super::batch_iterator::BatchIterator;
use super::{PrimitiveBatch, PrimitiveKind};

impl<'a> BatchIterator<'a> {
    pub(super) fn next_shadows(
        &mut self,
        batch_kind: PrimitiveKind,
        max_order_and_kind: (u32, PrimitiveKind),
    ) -> Option<PrimitiveBatch<'a>> {
        let start = self.shadows_start;
        let mut end = start + 1;
        self.shadows_iter.next();
        while self
            .shadows_iter
            .next_if(|shadow| (shadow.order, batch_kind) < max_order_and_kind)
            .is_some()
        {
            end += 1;
        }
        self.shadows_start = end;
        Some(PrimitiveBatch::Shadows(&self.shadows[start..end]))
    }

    pub(super) fn next_backdrop_blurs(
        &mut self,
        batch_kind: PrimitiveKind,
        max_order_and_kind: (u32, PrimitiveKind),
    ) -> Option<PrimitiveBatch<'a>> {
        let start = self.backdrop_blurs_start;
        let mut end = start + 1;
        self.backdrop_blurs_iter.next();
        while self
            .backdrop_blurs_iter
            .next_if(|backdrop_blur| (backdrop_blur.order, batch_kind) < max_order_and_kind)
            .is_some()
        {
            end += 1;
        }
        self.backdrop_blurs_start = end;
        Some(PrimitiveBatch::BackdropBlurs(
            &self.backdrop_blurs[start..end],
        ))
    }

    pub(super) fn next_quads(
        &mut self,
        batch_kind: PrimitiveKind,
        max_order_and_kind: (u32, PrimitiveKind),
    ) -> Option<PrimitiveBatch<'a>> {
        let start = self.quads_start;
        let mut end = start + 1;
        self.quads_iter.next();
        while self
            .quads_iter
            .next_if(|quad| (quad.order, batch_kind) < max_order_and_kind)
            .is_some()
        {
            end += 1;
        }
        self.quads_start = end;
        Some(PrimitiveBatch::Quads(&self.quads[start..end]))
    }

    pub(super) fn next_paths(
        &mut self,
        batch_kind: PrimitiveKind,
        max_order_and_kind: (u32, PrimitiveKind),
    ) -> Option<PrimitiveBatch<'a>> {
        let start = self.paths_start;
        let mut end = start + 1;
        self.paths_iter.next();
        while self
            .paths_iter
            .next_if(|path| (path.order, batch_kind) < max_order_and_kind)
            .is_some()
        {
            end += 1;
        }
        self.paths_start = end;
        Some(PrimitiveBatch::Paths(&self.paths[start..end]))
    }

    pub(super) fn next_underlines(
        &mut self,
        batch_kind: PrimitiveKind,
        max_order_and_kind: (u32, PrimitiveKind),
    ) -> Option<PrimitiveBatch<'a>> {
        let start = self.underlines_start;
        let mut end = start + 1;
        self.underlines_iter.next();
        while self
            .underlines_iter
            .next_if(|underline| (underline.order, batch_kind) < max_order_and_kind)
            .is_some()
        {
            end += 1;
        }
        self.underlines_start = end;
        Some(PrimitiveBatch::Underlines(&self.underlines[start..end]))
    }

    pub(super) fn next_monochrome_sprites(
        &mut self,
        batch_kind: PrimitiveKind,
        max_order_and_kind: (u32, PrimitiveKind),
    ) -> Option<PrimitiveBatch<'a>> {
        let first = self.monochrome_sprites_iter.peek()?;
        let texture_id = first.tile.texture_id;
        let start = self.monochrome_sprites_start;
        let mut end = start + 1;
        self.monochrome_sprites_iter.next();
        while self
            .monochrome_sprites_iter
            .next_if(|sprite| {
                (sprite.order, batch_kind) < max_order_and_kind
                    && sprite.tile.texture_id == texture_id
            })
            .is_some()
        {
            end += 1;
        }
        self.monochrome_sprites_start = end;
        Some(PrimitiveBatch::MonochromeSprites {
            texture_id,
            sprites: &self.monochrome_sprites[start..end],
        })
    }

    pub(super) fn next_polychrome_sprites(
        &mut self,
        batch_kind: PrimitiveKind,
        max_order_and_kind: (u32, PrimitiveKind),
    ) -> Option<PrimitiveBatch<'a>> {
        let first = self.polychrome_sprites_iter.peek()?;
        let texture_id = first.tile.texture_id;
        let start = self.polychrome_sprites_start;
        let mut end = self.polychrome_sprites_start + 1;
        self.polychrome_sprites_iter.next();
        while self
            .polychrome_sprites_iter
            .next_if(|sprite| {
                (sprite.order, batch_kind) < max_order_and_kind
                    && sprite.tile.texture_id == texture_id
            })
            .is_some()
        {
            end += 1;
        }
        self.polychrome_sprites_start = end;
        Some(PrimitiveBatch::PolychromeSprites {
            texture_id,
            sprites: &self.polychrome_sprites[start..end],
        })
    }

    pub(super) fn next_surfaces(
        &mut self,
        batch_kind: PrimitiveKind,
        max_order_and_kind: (u32, PrimitiveKind),
    ) -> Option<PrimitiveBatch<'a>> {
        let start = self.surfaces_start;
        let mut end = start + 1;
        self.surfaces_iter.next();
        while self
            .surfaces_iter
            .next_if(|surface| (surface.order, batch_kind) < max_order_and_kind)
            .is_some()
        {
            end += 1;
        }
        self.surfaces_start = end;
        Some(PrimitiveBatch::Surfaces(&self.surfaces[start..end]))
    }
}
