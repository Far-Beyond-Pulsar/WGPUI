//! The one part of the glyph path that needs a device: copying an atlas page's
//! texels into a `wgpu::Texture`. See docs/gpu-native-architecture.md §3.5, and
//! §9's "Load-bearing, disclosed by Phase 5" row.
//!
//! # What this is a port of
//!
//! `src/platform/cross/atlas.rs`'s `WgpuAtlasState::push_texture` (texture
//! creation and the `Monochrome`/`Polychrome` format mapping) and
//! `upload_texture` (the `COPY_BYTES_PER_ROW_ALIGNMENT` row padding and the
//! `queue.write_texture` that follows it, including the comment explaining why
//! it is `write_texture` and not a staging buffer). Phase 5 named both as
//! deliberately absent and mechanical; this is the mechanical part, done.
//!
//! # Why it is a separate type from `GlyphAtlas` rather than fields on it
//!
//! So that `GlyphAtlas` stays device-free and every packing and blitting
//! assertion keeps running headlessly, which is the argument
//! `render/atlas.rs`'s own doc makes and which this file would undo if it
//! folded a `wgpu::Texture` into a `Page`. The two are joined by
//! [`GlyphAtlas::drain_uploads`], which is a plain list of rectangles: the CPU
//! side does not know a device exists, and this side does not know how anything
//! was packed.
//!
//! # One `write_texture` per rectangle
//!
//! Same as the legacy, and for the reason `render/atlas.rs` gives: the bin
//! packer scatters glyphs, so a frame's writes have a bounding box very close to
//! the whole page and coalescing them would move megabytes to change kilobytes.
//! A frame that rasterises 60 new glyphs issues 60 copies of a few hundred bytes
//! each, which is what the legacy backend already does every time a new glyph
//! appears.

use crate::render::atlas::GlyphAtlas;
use std::collections::HashMap;
use wgpui_core::scene::atlas::AtlasKind;

/// The texture format each atlas kind maps onto.
///
/// The legacy `push_texture`'s match, unchanged: `R8Unorm` for a coverage mask
/// and `Rgba8Unorm` for colour, which is what makes
/// [`AtlasKind::bytes_per_pixel`] one and four.
pub const fn texture_format(kind: AtlasKind) -> wgpu::TextureFormat {
    match kind {
        AtlasKind::Monochrome => wgpu::TextureFormat::R8Unorm,
        AtlasKind::Polychrome => wgpu::TextureFormat::Rgba8Unorm,
    }
}

/// What one [`AtlasTextures::sync`] call did.
///
/// Counted rather than inferred, because "the upload happened" is otherwise
/// indistinguishable from "there was nothing to upload" — and the second is what
/// a broken drain looks like.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct UploadReport {
    /// Rectangles copied.
    pub rectangles: usize,
    /// Texel bytes copied, not counting row padding.
    pub texel_bytes: usize,
    /// Textures created for pages that did not have one.
    pub pages_created: usize,
    /// Textures dropped for pages the atlas no longer has.
    pub pages_destroyed: usize,
    /// Rectangles the atlas named but whose page it could not produce texels
    /// for.
    ///
    /// Always zero in a consistent atlas — a queued upload's page is live by
    /// construction, since `destroy_page` drops its uploads. Reported rather
    /// than asserted so a future inconsistency surfaces as a number rather than
    /// as a panic in a render loop.
    pub skipped: usize,
}

struct PageTexture {
    kind: AtlasKind,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

/// One `wgpu::Texture` per live atlas page, kept in step with a [`GlyphAtlas`].
pub struct AtlasTextures {
    page_size: u32,
    pages: HashMap<u32, PageTexture>,
}

impl AtlasTextures {
    /// Textures for an atlas whose pages are `page_size` texels on a side.
    pub fn new(page_size: u32) -> Self {
        Self {
            page_size,
            pages: HashMap::new(),
        }
    }

    /// Textures sized to match `atlas`.
    pub fn for_atlas(atlas: &GlyphAtlas) -> Self {
        Self::new(atlas.page_size())
    }

    /// The view a sprite pipeline binds for `page`.
    pub fn view(&self, page: u32) -> Option<&wgpu::TextureView> {
        self.pages.get(&page).map(|page| &page.view)
    }

    /// The texture behind `page` — for a readback, or for a copy.
    pub fn texture(&self, page: u32) -> Option<&wgpu::Texture> {
        self.pages.get(&page).map(|page| &page.texture)
    }

    /// How many pages currently have a texture.
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Create textures for new pages, drop them for destroyed ones, and copy
    /// every rectangle the atlas has queued.
    ///
    /// Draining is why `atlas` is taken mutably: an upload is reported once, to
    /// one uploader, the same contract `drain_evictions` has.
    pub fn sync(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        atlas: &mut GlyphAtlas,
    ) -> UploadReport {
        let mut report = UploadReport::default();

        let live: Vec<u32> = atlas.page_indices();
        let stale: Vec<u32> = self
            .pages
            .keys()
            .copied()
            .filter(|page| !live.contains(page))
            .collect();
        for page in stale {
            if let Some(dropped) = self.pages.remove(&page) {
                // Eagerly, as the legacy `WgpuAtlas::remove` does: an atlas page
                // is megabytes, and waiting for a drop that happens whenever the
                // map next rehashes is how a device runs out of memory holding
                // pictures nothing references.
                dropped.texture.destroy();
                report.pages_destroyed += 1;
            }
        }

        let uploads = atlas.drain_uploads();
        for upload in uploads {
            if !self.pages.contains_key(&upload.page) {
                self.pages
                    .insert(upload.page, self.create_page(device, upload.kind));
                report.pages_created += 1;
            }
            let Some(texels) = atlas.page_texels(upload.page) else {
                report.skipped += 1;
                continue;
            };
            let Some(page) = self.pages.get(&upload.page) else {
                report.skipped += 1;
                continue;
            };
            let bytes_per_pixel = page.kind.bytes_per_pixel() as usize;
            let page_stride = self.page_size as usize * bytes_per_pixel;
            let unpadded_bytes_per_row = upload.size[0] as usize * bytes_per_pixel;
            // The legacy expression, kept: `COPY_BYTES_PER_ROW_ALIGNMENT` is 256
            // and a copy's `bytes_per_row` must be a multiple of it.
            let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
            let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
            let height = upload.size[1] as usize;

            let mut staged = vec![0u8; padded_bytes_per_row * height];
            let mut copied = 0usize;
            for row in 0..height {
                let source = (upload.origin[1] as usize + row) * page_stride
                    + upload.origin[0] as usize * bytes_per_pixel;
                let destination = row * padded_bytes_per_row;
                let (Some(from), Some(into)) = (
                    texels.get(source..source + unpadded_bytes_per_row),
                    staged.get_mut(destination..destination + unpadded_bytes_per_row),
                ) else {
                    break;
                };
                into.copy_from_slice(from);
                copied += unpadded_bytes_per_row;
            }
            if copied != unpadded_bytes_per_row * height {
                report.skipped += 1;
                continue;
            }

            // `write_texture` rather than staging through a buffer, per the
            // legacy file's own comment: "Work around driver issues by using
            // queue.write_texture directly instead of staging through a buffer
            // (see helio/ship_flight repro)."
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &page.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: upload.origin[0],
                        y: upload.origin[1],
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                &staged,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row as u32),
                    rows_per_image: None,
                },
                wgpu::Extent3d {
                    width: upload.size[0],
                    height: upload.size[1],
                    depth_or_array_layers: 1,
                },
            );
            report.rectangles += 1;
            report.texel_bytes += copied;
        }

        report
    }

    fn create_page(&self, device: &wgpu::Device, kind: AtlasKind) -> PageTexture {
        let format = texture_format(kind);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("wgpui glyph atlas page"),
            size: wgpu::Extent3d {
                width: self.page_size,
                height: self.page_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            // The legacy usage set, unchanged. `COPY_SRC` is what makes the
            // upload verifiable — a texture that cannot be read back is a
            // texture whose contents are an assumption.
            usage: wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("wgpui glyph atlas page view"),
            format: Some(format),
            dimension: Some(wgpu::TextureViewDimension::D2),
            usage: Some(texture.usage()),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        });
        PageTexture {
            kind,
            texture,
            view,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_format_mapping_is_the_legacy_one_and_agrees_with_the_texel_width() {
        assert_eq!(
            texture_format(AtlasKind::Monochrome),
            wgpu::TextureFormat::R8Unorm
        );
        assert_eq!(
            texture_format(AtlasKind::Polychrome),
            wgpu::TextureFormat::Rgba8Unorm
        );
        // The two numbers have to agree or every row offset in `sync` is wrong,
        // and they are declared in two crates, so they are checked against each
        // other rather than against a comment.
        for kind in [AtlasKind::Monochrome, AtlasKind::Polychrome] {
            assert_eq!(
                texture_format(kind)
                    .block_copy_size(None)
                    .expect("an uncompressed colour format has a block size"),
                kind.bytes_per_pixel()
            );
        }
    }
}
