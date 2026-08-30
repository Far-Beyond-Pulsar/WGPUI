use crate::{AssetSource, DevicePixels, IsZero, RenderImage, Result, SharedString, Size, swap_rgba_pa_to_bgra};
use image::Frame;
use smallvec::SmallVec;
use std::sync::Arc;

/// When rendering SVGs, we render them at twice the size to get a higher-quality result.
pub const SMOOTH_SVG_SCALE_FACTOR: f32 = 2.;

#[derive(Clone, PartialEq, Hash, Eq)]
pub(crate) struct RenderSvgParams {
    pub(crate) path: SharedString,
    pub(crate) size: Size<DevicePixels>,
}

#[allow(missing_docs)]
pub enum SvgSize {
    Size(Size<DevicePixels>),
    ScaleFactor(f32),
}

#[cfg(not(target_family = "wasm"))]
mod backend {
    use super::*;
    use resvg::tiny_skia::Pixmap;
    use std::sync::LazyLock;

    static FONT_DB: LazyLock<Arc<usvg::fontdb::Database>> = LazyLock::new(|| {
        let mut db = usvg::fontdb::Database::new();
        db.load_system_fonts();
        Arc::new(db)
    });

    fn make_usvg_options() -> Arc<usvg::Options<'static>> {
        let default_font_resolver = usvg::FontResolver::default_font_selector();
        let font_resolver = Box::new(
            move |font: &usvg::Font, db: &mut Arc<usvg::fontdb::Database>| {
                if db.is_empty() {
                    *db = FONT_DB.clone();
                }
                default_font_resolver(font, db)
            },
        );
        let options = usvg::Options {
            font_resolver: usvg::FontResolver {
                select_font: font_resolver,
                select_fallback: usvg::FontResolver::default_fallback_selector(),
            },
            ..Default::default()
        };
        Arc::new(options)
    }

    pub fn render_single_frame_inner(
        bytes: &[u8],
        scale_factor: f32,
        to_brga: bool,
        asset_source: &Arc<dyn AssetSource>,
    ) -> Result<Arc<RenderImage>> {
        let usvg_options = make_usvg_options();
        let tree = usvg::Tree::from_data(bytes, &usvg_options)?;
        let svg_size = tree.size();
        let scale = scale_factor * SMOOTH_SVG_SCALE_FACTOR;
        let mut pixmap = Pixmap::new(
            (svg_size.width() * scale) as u32,
            (svg_size.height() * scale) as u32,
        )
        .ok_or(usvg::Error::InvalidSize)?;
        let transform = resvg::tiny_skia::Transform::from_scale(scale, scale);
        resvg::render(&tree, transform, &mut pixmap.as_mut());

        let mut buffer =
            image::ImageBuffer::from_raw(pixmap.width(), pixmap.height(), pixmap.take())
                .unwrap();

        if to_brga {
            for pixel in buffer.chunks_exact_mut(4) {
                swap_rgba_pa_to_bgra(pixel);
            }
        }

        let mut image = RenderImage::new(SmallVec::from_const([Frame::new(buffer)]));
        image.scale_factor = SMOOTH_SVG_SCALE_FACTOR;
        Ok(Arc::new(image))
    }

    pub fn render_alpha_mask_inner(
        params: &RenderSvgParams,
        bytes: Option<&[u8]>,
        asset_source: &Arc<dyn AssetSource>,
    ) -> Result<Option<(Size<DevicePixels>, Vec<u8>)>> {
        anyhow::ensure!(!params.size.is_zero(), "can't render at a zero size");

        fn render_pixmap(bytes: &[u8], params: &RenderSvgParams) -> Result<Pixmap> {
            let usvg_options = make_usvg_options();
            let tree = usvg::Tree::from_data(bytes, &usvg_options)?;
            let tree_size = tree.size();
            let scale = params.size.width.0 as f32 / tree_size.width();
            let mut pixmap = Pixmap::new(
                (tree_size.width() * scale) as u32,
                (tree_size.height() * scale) as u32,
            )
            .ok_or(usvg::Error::InvalidSize)?;
            let transform = resvg::tiny_skia::Transform::from_scale(scale, scale);
            resvg::render(&tree, transform, &mut pixmap.as_mut());
            Ok(pixmap)
        }

        if let Some(bytes) = bytes {
            let pixmap = render_pixmap(bytes, params)?;
            let size = Size::new(
                DevicePixels(pixmap.width() as i32),
                DevicePixels(pixmap.height() as i32),
            );
            let alpha_mask = pixmap.pixels().iter().map(|p| p.alpha()).collect::<Vec<_>>();
            Ok(Some((size, alpha_mask)))
        } else if let Some(bytes) = asset_source.load(&params.path)? {
            let pixmap = render_pixmap(&bytes, params)?;
            let size = Size::new(
                DevicePixels(pixmap.width() as i32),
                DevicePixels(pixmap.height() as i32),
            );
            let alpha_mask = pixmap.pixels().iter().map(|p| p.alpha()).collect::<Vec<_>>();
            Ok(Some((size, alpha_mask)))
        } else {
            Ok(None)
        }
    }
}

#[cfg(target_family = "wasm")]
mod backend {
    use super::*;

    pub fn render_single_frame_inner(
        _bytes: &[u8],
        _scale_factor: f32,
        _to_brga: bool,
        _asset_source: &Arc<dyn AssetSource>,
    ) -> Result<Arc<RenderImage>> {
        anyhow::bail!("SVG rendering is not available on this platform");
    }

    pub fn render_alpha_mask_inner(
        _params: &RenderSvgParams,
        _bytes: Option<&[u8]>,
        _asset_source: &Arc<dyn AssetSource>,
    ) -> Result<Option<(Size<DevicePixels>, Vec<u8>)>> {
        anyhow::bail!("SVG rendering is not available on this platform");
    }
}

#[derive(Clone)]
#[allow(missing_docs)]
pub struct SvgRenderer {
    asset_source: Arc<dyn AssetSource>,
}

#[allow(missing_docs)]
impl SvgRenderer {
    pub fn new(asset_source: Arc<dyn AssetSource>) -> Self {
        Self { asset_source }
    }

    pub fn render_single_frame(
        &self,
        bytes: &[u8],
        scale_factor: f32,
        to_brga: bool,
    ) -> Result<Arc<RenderImage>> {
        backend::render_single_frame_inner(bytes, scale_factor, to_brga, &self.asset_source)
    }

    pub fn render_svg(
        &self,
        path: &SharedString,
        color: crate::Hsla,
        factor: f64,
    ) -> Result<Arc<RenderImage>> {
        let Some(bytes) = self.asset_source.load(path.as_ref()).ok().flatten() else {
            anyhow::bail!("SVG not found: {path}");
        };
        self.render_single_frame(&bytes, factor as f32, true)
    }

    pub(crate) fn render_alpha_mask(
        &self,
        params: &RenderSvgParams,
        bytes: Option<&[u8]>,
    ) -> Result<Option<(Size<DevicePixels>, Vec<u8>)>> {
        backend::render_alpha_mask_inner(params, bytes, &self.asset_source)
    }
}
