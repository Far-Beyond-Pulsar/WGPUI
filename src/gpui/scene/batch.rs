use super::batch_counter::BatchCounter;
use super::batch_iterator::BatchIterator;
use super::{
    BackdropBlur,
    MonochromeSprite,
    PaintSurface,
    Path,
    PolychromeSprite,
    Quad,
    Scene,
    Shadow,
    Underline,
};
use crate::{AtlasTextureId, ScaledPixels};

#[derive(Debug)]
pub(crate) enum PrimitiveBatch<'a> {
    Shadows(&'a [Shadow]),
    BackdropBlurs(&'a [BackdropBlur]),
    Quads(&'a [Quad]),
    Paths(&'a [Path<ScaledPixels>]),
    Underlines(&'a [Underline]),
    MonochromeSprites {
        texture_id: AtlasTextureId,
        sprites: &'a [MonochromeSprite],
    },
    PolychromeSprites {
        texture_id: AtlasTextureId,
        sprites: &'a [PolychromeSprite],
    },
    Surfaces(&'a [PaintSurface]),
}

impl Scene {
    pub fn total_batch_count(&self) -> u32 {
        let mut counter = BatchCounter::default();
        let mut count = 0u32;
        while counter.advance(self) {
            count = count.saturating_add(1);
        }
        count
    }

    pub(crate) fn batches(&self) -> impl Iterator<Item = PrimitiveBatch<'_>> {
        BatchIterator::new(self)
    }
}
