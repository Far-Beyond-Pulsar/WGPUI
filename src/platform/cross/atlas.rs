use std::sync::Arc;

use collections::FxHashMap;
use etagere::BucketedAtlasAllocator;
use parking_lot::Mutex;

use crate::{
    AtlasKey, AtlasTextureId, AtlasTextureKind, AtlasTile, Bounds, DevicePixels, PlatformAtlas,
    Point, Size,
    platform::{AtlasTextureList, cross::render_context::WgpuContext},
};

pub(crate) struct WgpuAtlas(Mutex<WgpuAtlasState>);

impl WgpuAtlas {
    pub(crate) fn new(context: Arc<WgpuContext>) -> Self {
        WgpuAtlas(Mutex::new(WgpuAtlasState {
            atlas_target: None,
            atlas_target_view: None,
            context,
            storage: WgpuAtlasStorage::default(),
            tiles_by_key: FxHashMap::default(),
            
            uploads: Vec::new(),
        }))
    }

    pub fn before_frame(&self, encoder: &mut wgpu::CommandEncoder) {
        self.0.lock().flush(encoder);
    }

    pub(crate) fn get_texture_info(&self, texture_id: AtlasTextureId) -> WgpuTextureInfo {
        let state = self.0.lock();
        let texture = &state.storage[texture_id];

        WgpuTextureInfo {
            raw_view: texture.raw_view.clone(),
        }
    }

    /// Sum of every live atlas texture's backing memory, monochrome and
    /// polychrome combined (Phase 3 of the profiling epic, issue #59).
    #[cfg(feature = "flamegraph")]
    pub(crate) fn memory_usage(&self) -> u64 {
        let state = self.0.lock();
        atlas_texture_list_memory_usage(&state.storage.monochrome_textures)
            + atlas_texture_list_memory_usage(&state.storage.polychrome_textures)
    }

    /// One atlas texture page's identity/metadata needed to read its pixel
    /// contents back for a triggered GPU deep capture (Phase 4b of the
    /// profiling epic, issue #72). Distinct from `get_texture_info`, which
    /// only exposes a `TextureView` -- enough to bind a sprite pipeline, but
    /// `copy_texture_to_buffer` needs the underlying `wgpu::Texture`
    /// directly, plus the pixel dimensions and texel size a caller needs to
    /// compute `wgpu::COPY_BYTES_PER_ROW_ALIGNMENT` row padding. Returns
    /// `None` if `texture_id` no longer refers to a live page (e.g. it was
    /// evicted between the draw call that touched it and this lookup --
    /// shouldn't happen within one frame's own command stream, but reported
    /// rather than panicking since there's no way to distinguish that from
    /// a genuinely stale id).
    #[cfg(feature = "flamegraph")]
    pub(crate) fn texture_snapshot(&self, texture_id: AtlasTextureId) -> Option<WgpuAtlasTextureSnapshot> {
        let state = self.0.lock();
        let texture = state.storage[texture_id.kind]
            .textures
            .get(texture_id.index as usize)?
            .as_ref()?;
        Some(WgpuAtlasTextureSnapshot {
            texture: texture.raw.clone(),
            width: texture.raw.width(),
            height: texture.raw.height(),
            // `texel_size` (not `WgpuAtlasTexture::bytes_per_pixel`, which
            // panics on any format outside the two this atlas currently
            // creates) so this stays correct rather than crashing if the
            // atlas ever grows a third texture kind.
            bytes_per_pixel: super::render_context::texel_size(texture.format) as u32,
        })
    }
}

/// See [`WgpuAtlas::texture_snapshot`].
#[cfg(feature = "flamegraph")]
pub(crate) struct WgpuAtlasTextureSnapshot {
    pub(crate) texture: wgpu::Texture,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) bytes_per_pixel: u32,
}

#[cfg(feature = "flamegraph")]
fn atlas_texture_list_memory_usage(textures: &AtlasTextureList<WgpuAtlasTexture>) -> u64 {
    textures
        .textures
        .iter()
        .flatten()
        .map(|texture| super::render_context::texture_memory_bytes(&texture.raw))
        .sum()
}

impl PlatformAtlas for WgpuAtlas {
    fn get_or_insert_with<'a>(
        &self,
        key: &AtlasKey,
        build: &mut dyn FnMut() -> anyhow::Result<
            Option<(Size<DevicePixels>, std::borrow::Cow<'a, [u8]>)>,
        >,
    ) -> anyhow::Result<Option<AtlasTile>> {
        let mut atlas = self.0.lock();

        match atlas.tiles_by_key.get(key) {
            Some(tile) => {
                #[cfg(feature = "flamegraph")]
                crate::record_atlas_cache_hit();
                Ok(Some(tile.clone()))
            }
            None => Ok({
                profiling::scope!("new tile");
                #[cfg(feature = "flamegraph")]
                crate::record_atlas_cache_miss();

                match build()? {
                    Some((size, bytes)) => {
                        let tile = atlas.allocate(size, key.texture_kind())?;
                        #[cfg(feature = "flamegraph")]
                        crate::record_atlas_tile_allocated();

                        atlas.upload_texture(tile.texture_id, tile.bounds, &bytes);
                        atlas.tiles_by_key.insert(key.clone(), tile.clone());

                        Some(tile)
                    }
                    None => None,
                }
            }),
        }
    }

    fn remove(&self, key: &AtlasKey) {
        let mut atlas = self.0.lock();

        let Some(id) = atlas.tiles_by_key.remove(key).map(|x| x.texture_id) else {
            return;
        };
        #[cfg(feature = "flamegraph")]
        crate::record_atlas_tile_evicted();

        let Some(texture_slot) = atlas.storage[id.kind].textures.get_mut(id.index as usize) else {
            return;
        };

        if let Some(mut texture) = texture_slot.take() {
            texture.decrement_ref_count();

            if texture.is_unreferenced() {
                atlas.storage[id.kind]
                    .free_list
                    .push(texture.id.index as usize);

                // Eagerly destroy to free GPU memory immediately.
                texture.destroy(&atlas.context);
            } else {
                *texture_slot = Some(texture);
            }
        }
    }
}

struct WgpuAtlasState {
    atlas_target: Option<wgpu::Texture>,
    atlas_target_view: Option<wgpu::TextureView>,
    context: Arc<WgpuContext>,
    storage: WgpuAtlasStorage,
    tiles_by_key: FxHashMap<AtlasKey, AtlasTile>,
    uploads: Vec<PendingUpload>,
}

impl WgpuAtlasState {
    fn allocate(
        &mut self,
        size: Size<DevicePixels>,
        texture_kind: AtlasTextureKind,
    ) -> anyhow::Result<AtlasTile> {
        {
            let textures = &mut self.storage[texture_kind];

            if let Some(tile) = textures
                .iter_mut()
                .rev()
                .find_map(|texture| texture.allocate(size))
            {
                return Ok(tile);
            }
        }

        let texture = self.push_texture(size, texture_kind);

        texture.allocate(size).ok_or_else(|| {
            anyhow::anyhow!(
                "newly created atlas texture of size {}x{} could not satisfy allocation of size {}x{}",
                texture.raw.width(),
                texture.raw.height(),
                size.width.0,
                size.height.0,
            )
        })
    }

    fn push_texture(
        &mut self,
        min_size: Size<DevicePixels>,
        texture_kind: AtlasTextureKind,
    ) -> &mut WgpuAtlasTexture {
        const DEFAULT_ATLAS_SIZE: Size<DevicePixels> = Size {
            width: DevicePixels(1024),
            height: DevicePixels(1024),
        };

        let size = min_size.max(&DEFAULT_ATLAS_SIZE);

        let (format, usage) = match texture_kind {
            AtlasTextureKind::Monochrome => (
                wgpu::TextureFormat::R8Unorm,
                wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::TEXTURE_BINDING,
            ),
            AtlasTextureKind::Polychrome => (
                wgpu::TextureFormat::Rgba8Unorm,
                wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::TEXTURE_BINDING,
            ),
        };

        let texture_raw = self
            .context
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("Atlas Texture"),
                size: wgpu::Extent3d {
                    width: size.width.0 as u32,
                    height: size.height.0 as u32,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            });

        let texture_raw_view = texture_raw.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Atlas Texture Raw"),
            format: Some(format),
            dimension: Some(wgpu::TextureViewDimension::D2),
            usage: Some(texture_raw.usage()),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        });

        let texture_list = &mut self.storage[texture_kind];

        let index = texture_list.free_list.pop();

        let atlas_texture = WgpuAtlasTexture {
            id: AtlasTextureId {
                kind: texture_kind,
                index: index.unwrap_or(texture_list.textures.len()) as u32,
            },
            allocator: BucketedAtlasAllocator::new(size.into()),
            raw: texture_raw,
            raw_view: texture_raw_view,
            format,
            live_atlas_keys: 0,
        };

        // If we popped a free slot from the free list, place the texture there;
        // otherwise append to the end.
        match index {
            Some(index) => {
                texture_list.textures[index] = Some(atlas_texture);
                texture_list
                    .textures
                    .get_mut(index)
                    .unwrap()
                    .as_mut()
                    .unwrap()
            }
            None => {
                texture_list.textures.push(Some(atlas_texture));
                texture_list.textures.last_mut().unwrap().as_mut().unwrap()
            }
        }
    }

    fn upload_texture(
        &mut self,
        texture_id: AtlasTextureId,
        bounds: Bounds<DevicePixels>,
        bytes: &[u8],
    ) {
        let texture = &self.storage[texture_id];
        let bytes_per_pixel = texture.bytes_per_pixel();
        let unpadded_bytes_per_row = bounds.size.width.to_bytes(bytes_per_pixel) as usize;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
        let padded_bytes_per_row = (unpadded_bytes_per_row + align - 1) / align * align;
        let height = bounds.size.height.0 as usize;

        let padded_data = if padded_bytes_per_row != unpadded_bytes_per_row {
            let mut padded = vec![0u8; padded_bytes_per_row * height];
            for row in 0..height {
                let src_start = row * unpadded_bytes_per_row;
                let dst_start = row * padded_bytes_per_row;
                padded[dst_start..dst_start + unpadded_bytes_per_row]
                    .copy_from_slice(&bytes[src_start..src_start + unpadded_bytes_per_row]);
            }
            Some(padded)
        } else {
            None
        };

        let contents = padded_data.as_deref().unwrap_or(bytes);

        // Work around driver issues by using queue.write_texture directly
        // instead of staging through a buffer (see helio/ship_flight repro).
        let texture = &self.storage[texture_id];

        self.context.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture.raw,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: bounds.origin.x.into(),
                    y: bounds.origin.y.into(),
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            contents,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row as u32),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: bounds.size.width.into(),
                height: bounds.size.height.into(),
                depth_or_array_layers: 1,
            },
        )
    }

    fn flush(&mut self, encoder: &mut wgpu::CommandEncoder) {
        for upload in self.uploads.drain(..) {
            let texture = &self.storage[upload.texture_id];

            encoder.copy_buffer_to_texture(
                wgpu::TexelCopyBufferInfo {
                    buffer: &upload.buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: upload.offset,
                        bytes_per_row: Some(upload.padded_bytes_per_row),
                        rows_per_image: None,
                    },
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &texture.raw,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: upload.bounds.origin.x.into(),
                        y: upload.bounds.origin.y.into(),
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: upload.bounds.size.width.into(),
                    height: upload.bounds.size.height.into(),
                    depth_or_array_layers: 1,
                },
            );
        }
    }
}

pub(crate) struct WgpuAtlasTexture {
    id: AtlasTextureId,
    allocator: BucketedAtlasAllocator,
    raw: wgpu::Texture,
    raw_view: wgpu::TextureView,
    format: wgpu::TextureFormat,
    live_atlas_keys: u32,
}

impl WgpuAtlasTexture {
    fn allocate(&mut self, size: Size<DevicePixels>) -> Option<AtlasTile> {
        let allocation = self.allocator.allocate(size.into())?;

        let tile = AtlasTile {
            texture_id: self.id,
            tile_id: allocation.id.into(),
            padding: 0,
            bounds: Bounds {
                origin: allocation.rectangle.min.into(),
                size,
            },
        };

        self.live_atlas_keys += 1;

        Some(tile)
    }

    fn bytes_per_pixel(&self) -> u8 {
        self.format
            .block_copy_size(None)
            .expect("atlas texture format should have a known block copy size") as u8
    }

    fn decrement_ref_count(&mut self) {
        self.live_atlas_keys = self.live_atlas_keys.saturating_sub(1);
    }

    fn is_unreferenced(&self) -> bool {
        self.live_atlas_keys == 0
    }

    fn destroy(self, _context: &WgpuContext) {
        // NOTE(mdeand): In wgpu, textures are automatically cleaned up when dropped.
        // NOTE(mdeand): If there were any additional resources to free, they would be handled here.
    }
}

impl std::ops::Index<AtlasTextureKind> for WgpuAtlasStorage {
    type Output = AtlasTextureList<WgpuAtlasTexture>;
    fn index(&self, kind: AtlasTextureKind) -> &Self::Output {
        match kind {
            crate::AtlasTextureKind::Monochrome => &self.monochrome_textures,
            crate::AtlasTextureKind::Polychrome => &self.polychrome_textures,
        }
    }
}

impl std::ops::IndexMut<AtlasTextureKind> for WgpuAtlasStorage {
    fn index_mut(&mut self, kind: AtlasTextureKind) -> &mut Self::Output {
        match kind {
            crate::AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
            crate::AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
        }
    }
}

impl std::ops::Index<AtlasTextureId> for WgpuAtlasStorage {
    type Output = WgpuAtlasTexture;
    fn index(&self, id: AtlasTextureId) -> &Self::Output {
        let textures = match id.kind {
            crate::AtlasTextureKind::Monochrome => &self.monochrome_textures,
            crate::AtlasTextureKind::Polychrome => &self.polychrome_textures,
        };

        textures[id.index as usize].as_ref().unwrap()
    }
}

#[derive(Default)]
struct WgpuAtlasStorage {
    monochrome_textures: AtlasTextureList<WgpuAtlasTexture>,
    polychrome_textures: AtlasTextureList<WgpuAtlasTexture>,
}

pub(crate) struct WgpuTextureInfo {
    pub raw_view: wgpu::TextureView,
}

struct PendingUpload {
    texture_id: AtlasTextureId,
    bounds: Bounds<DevicePixels>,
    buffer: wgpu::Buffer,
    offset: u64,
    padded_bytes_per_row: u32,
}

impl From<Size<DevicePixels>> for etagere::Size {
    fn from(size: Size<DevicePixels>) -> Self {
        etagere::Size::new(size.width.into(), size.height.into())
    }
}

impl From<etagere::Point> for Point<DevicePixels> {
    fn from(value: etagere::Point) -> Self {
        Point {
            x: DevicePixels::from(value.x),
            y: DevicePixels::from(value.y),
        }
    }
}

impl From<etagere::Size> for Size<DevicePixels> {
    fn from(size: etagere::Size) -> Self {
        Size {
            width: DevicePixels::from(size.width),
            height: DevicePixels::from(size.height),
        }
    }
}

impl From<etagere::Rectangle> for Bounds<DevicePixels> {
    fn from(rectangle: etagere::Rectangle) -> Self {
        Bounds {
            origin: rectangle.min.into(),
            size: rectangle.size().into(),
        }
    }
}
