use super::{PrimitiveKind, Scene};

#[derive(Default)]
pub(super) struct BatchCounter {
    shadows_index: usize,
    backdrop_blurs_index: usize,
    quads_index: usize,
    paths_index: usize,
    underlines_index: usize,
    monochrome_sprites_index: usize,
    polychrome_sprites_index: usize,
    surfaces_index: usize,
}

impl BatchCounter {
    pub(super) fn advance(&mut self, scene: &Scene) -> bool {
        let mut orders_and_kinds = [
            (
                scene.shadows.get(self.shadows_index).map(|s| s.order),
                PrimitiveKind::Shadow,
            ),
            (
                scene
                    .backdrop_blurs
                    .get(self.backdrop_blurs_index)
                    .map(|b| b.order),
                PrimitiveKind::BackdropBlur,
            ),
            (
                scene.quads.get(self.quads_index).map(|q| q.order),
                PrimitiveKind::Quad,
            ),
            (
                scene.paths.get(self.paths_index).map(|p| p.order),
                PrimitiveKind::Path,
            ),
            (
                scene.underlines.get(self.underlines_index).map(|u| u.order),
                PrimitiveKind::Underline,
            ),
            (
                scene
                    .monochrome_sprites
                    .get(self.monochrome_sprites_index)
                    .map(|s| s.order),
                PrimitiveKind::MonochromeSprite,
            ),
            (
                scene
                    .polychrome_sprites
                    .get(self.polychrome_sprites_index)
                    .map(|s| s.order),
                PrimitiveKind::PolychromeSprite,
            ),
            (
                scene.surfaces.get(self.surfaces_index).map(|s| s.order),
                PrimitiveKind::Surface,
            ),
        ];
        orders_and_kinds.sort_by_key(|(order, kind)| (order.unwrap_or(u32::MAX), *kind));

        let first = orders_and_kinds[0];
        let second = orders_and_kinds[1];
        if first.0.is_none() {
            return false;
        }

        let batch_kind = first.1;
        let max_order_and_kind = (second.0.unwrap_or(u32::MAX), second.1);

        match batch_kind {
            PrimitiveKind::Shadow => {
                self.shadows_index += 1;
                while let Some(shadow) = scene.shadows.get(self.shadows_index)
                    && (shadow.order, batch_kind) < max_order_and_kind
                {
                    self.shadows_index += 1;
                }
            }
            PrimitiveKind::BackdropBlur => {
                self.backdrop_blurs_index += 1;
                while let Some(blur) = scene.backdrop_blurs.get(self.backdrop_blurs_index)
                    && (blur.order, batch_kind) < max_order_and_kind
                {
                    self.backdrop_blurs_index += 1;
                }
            }
            PrimitiveKind::Quad => {
                self.quads_index += 1;
                while let Some(quad) = scene.quads.get(self.quads_index)
                    && (quad.order, batch_kind) < max_order_and_kind
                {
                    self.quads_index += 1;
                }
            }
            PrimitiveKind::Path => {
                self.paths_index += 1;
                while let Some(path) = scene.paths.get(self.paths_index)
                    && (path.order, batch_kind) < max_order_and_kind
                {
                    self.paths_index += 1;
                }
            }
            PrimitiveKind::Underline => {
                self.underlines_index += 1;
                while let Some(underline) = scene.underlines.get(self.underlines_index)
                    && (underline.order, batch_kind) < max_order_and_kind
                {
                    self.underlines_index += 1;
                }
            }
            PrimitiveKind::MonochromeSprite => {
                let Some(texture_id) = scene
                    .monochrome_sprites
                    .get(self.monochrome_sprites_index)
                    .map(|sprite| sprite.tile.texture_id)
                else {
                    return false;
                };
                self.monochrome_sprites_index += 1;
                while let Some(sprite) = scene.monochrome_sprites.get(self.monochrome_sprites_index)
                    && (sprite.order, batch_kind) < max_order_and_kind
                    && sprite.tile.texture_id == texture_id
                {
                    self.monochrome_sprites_index += 1;
                }
            }
            PrimitiveKind::PolychromeSprite => {
                let Some(texture_id) = scene
                    .polychrome_sprites
                    .get(self.polychrome_sprites_index)
                    .map(|sprite| sprite.tile.texture_id)
                else {
                    return false;
                };
                self.polychrome_sprites_index += 1;
                while let Some(sprite) = scene.polychrome_sprites.get(self.polychrome_sprites_index)
                    && (sprite.order, batch_kind) < max_order_and_kind
                    && sprite.tile.texture_id == texture_id
                {
                    self.polychrome_sprites_index += 1;
                }
            }
            PrimitiveKind::Surface => {
                self.surfaces_index += 1;
                while let Some(surface) = scene.surfaces.get(self.surfaces_index)
                    && (surface.order, batch_kind) < max_order_and_kind
                {
                    self.surfaces_index += 1;
                }
            }
        }

        true
    }
}
