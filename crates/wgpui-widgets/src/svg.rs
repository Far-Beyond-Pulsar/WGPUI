//! `svg()` element. See docs/gpu-native-architecture.md §3.4, §8 (Phase 6.2).
//!
//! # Why this file is thin, and why that is the finding rather than a shortcut
//!
//! §8's Phase 6.2 row says SVG "piggybacks on this once the polychrome path
//! exists: `resvg`/`usvg` rasterize to a bitmap on CPU exactly as legacy does,
//! uploaded through the same polychrome tile path as any other image." That
//! turned out to be exactly true, and the honest way to show it is for this file
//! to contain the *difference* between an SVG and a bitmap and nothing else.
//!
//! The difference is one thing: an SVG has no natural pixel size, only a
//! document size and whatever scale you rasterise it at. Everything downstream
//! — the tile, the sprite, the object-fit, the reconciliation key — is
//! [`crate::img`]'s, unchanged, because once `resvg` has produced texels an SVG
//! *is* a bitmap.
//!
//! So [`load`] is the whole of the loading half, and [`Svg`] is [`crate::img::Img`]
//! with a key of its own.
//!
//! # What is deliberately not here: the tinted alpha-mask path
//!
//! The legacy backend has **two** SVG paths, and only one of them is this one.
//! `SvgRenderer::render_single_frame` produces full-colour RGBA and is what
//! `ImageAssetLoader` uses for an `img()` whose bytes turn out to be SVG — that
//! is what this file ports. `SvgRenderer::render_alpha_mask` produces a
//! single-channel coverage mask that the legacy `svg()` element tints with an
//! `Hsla`, which is how icon sets recolour one asset per theme
//! (`src/svg_renderer.rs`, `render_alpha_mask` / `render_svg`).
//!
//! The tinted path lands in the **monochrome** atlas and therefore in
//! `MonoSpritePipeline`, not in this phase's polychrome one. It is a genuinely
//! separate piece of work — a second rasterisation mode, a second atlas kind for
//! the same source, and a colour on the primitive that `PolySprite` does not
//! carry — and it is named as open in docs/phase-6.2-results.md rather than
//! quietly folded into "SVG works."

use std::any::Any;
use wgpui_core::invalidation::axes::Invalidation;
use wgpui_core::patch::emit::{Emission, EmitContext};
use wgpui_core::reconcile::description::Description;
use wgpui_core::reconcile::diff_key::ReconcileKey;
use wgpui_layout::taffy_tree::{Dimension, LayoutSize, LayoutStyle};

use crate::image_cache::{ImageDecodeError, decode_svg_at};
use crate::img::{ImageSourceId, ImageStyle, Img, SharedImageEngine};

/// Rasterise SVG bytes into `engine`'s cache and return the source that holds
/// them.
///
/// `scale_factor` is the device-pixel ratio the document is rasterised for; the
/// cache stores the result at [`crate::image_cache::SMOOTH_SVG_SCALE_FACTOR`]
/// times that, which is the legacy smoothing behaviour.
///
/// A separate call rather than something [`Svg`] does on describe, because
/// rasterising is expensive and *when* it happens is the caller's decision — the
/// same reason `wgpui_widgets::image_cache::ImageCache` does not fetch. An
/// element that rasterised inside `describe` would do it on every frame that
/// rebuilt it.
pub fn load(
    engine: &SharedImageEngine,
    bytes: &[u8],
    scale_factor: f32,
) -> Result<ImageSourceId, ImageDecodeError> {
    let rasterised = decode_svg_at(bytes, scale_factor)?;
    engine.borrow_mut().cache().hold(rasterised)
}

/// The fingerprint an [`Svg`] presents to ambient reconciliation.
///
/// §6.2's standing rule — "every first-party element type ships with `diff_key`
/// implemented" — applied to an element added after the rule was written, which
/// is what the rule is for. It is [`crate::img::ImgKey`] minus the two fields an
/// SVG cannot have: there is no animation frame in a rasterised document, and
/// there is no load state because [`load`] is synchronous and either produced a
/// source or returned an error.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SvgKey {
    /// Which rasterised document is displayed.
    pub source: ImageSourceId,
    /// The box the element asked layout for.
    pub requested_size: [f32; 2],
    /// How that box is drawn.
    pub style: ImageStyle,
}

impl ReconcileKey for SvgKey {
    fn compare(&self, previous: &dyn ReconcileKey) -> Invalidation {
        let Some(previous) = previous.as_any().downcast_ref::<SvgKey>() else {
            return Invalidation::all();
        };
        let mut axes = Invalidation::empty();
        if previous.requested_size != self.requested_size {
            axes |= Invalidation::LAYOUT;
            axes |= Invalidation::DISPLAY;
        }
        if previous.source != self.source || previous.style != self.style {
            axes |= Invalidation::DISPLAY;
        }
        axes
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A rasterised SVG document, drawn through the same sprite path as any image.
#[derive(Clone)]
pub struct Svg {
    image: Img,
}

/// An SVG showing the document `load` put at `source`.
pub fn svg(source: ImageSourceId, engine: SharedImageEngine) -> Svg {
    Svg::new(source, engine)
}

impl Svg {
    /// An SVG showing the document [`load`] put at `source`.
    pub fn new(source: ImageSourceId, engine: SharedImageEngine) -> Self {
        Self {
            image: Img::new(source, engine),
        }
    }

    /// Request a size.
    pub fn size(mut self, width: f32, height: f32) -> Self {
        self.image = self.image.size(width, height);
        self
    }

    /// Set how the document is drawn.
    pub fn style(mut self, style: ImageStyle) -> Self {
        self.image = self.image.style(style);
        self
    }

    /// The rasterised pixel size of the document.
    ///
    /// Already multiplied by [`crate::image_cache::SMOOTH_SVG_SCALE_FACTOR`],
    /// because that is the size the bitmap actually is. A caller that wants the
    /// document's own size divides by it — which is exactly what the legacy
    /// `RenderImage::render_size` does with its `scale_factor` field.
    pub fn rasterised_size(&self) -> Option<[u32; 2]> {
        self.image.natural_size()
    }

    /// This element's fingerprint.
    pub fn diff_key(&self) -> SvgKey {
        let image = self.image.diff_key();
        SvgKey {
            source: image.source,
            requested_size: image.requested_size,
            style: image.style,
        }
    }

    /// The per-frame description of this SVG.
    ///
    /// Its own element type and its own key, and [`Img`]'s layout size and
    /// emission: an SVG's *pixels* are an image's pixels once `resvg` has run,
    /// so producing them twice would be two code paths that can disagree about a
    /// sprite. Only the fingerprint and the type differ, and only because an SVG
    /// has fewer things that can change about it.
    pub fn describe(&self) -> Description {
        let [width, height] = self.image.layout_size();
        let element = self.clone();
        Description::new::<Svg>()
            .diff_key(self.diff_key())
            .style(LayoutStyle {
                size: LayoutSize {
                    width: Dimension::length(width),
                    height: Dimension::length(height),
                },
                flex_shrink: 0.0,
                ..LayoutStyle::default()
            })
            .emit(move |context: &EmitContext, emission: &mut Emission| {
                element.emit_into(context, emission);
            })
    }

    /// Write this element's sprite into `emission`, for a caller driving
    /// emission directly.
    pub fn emit_into(&self, context: &EmitContext, emission: &mut Emission) {
        self.image.emit_into(context, emission);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image_cache::{ImageCache, SMOOTH_SVG_SCALE_FACTOR};
    use crate::img::ImageEngine;
    use std::cell::RefCell;
    use std::rc::Rc;
    use wgpui_core::patch::primitive::AtlasTileId;
    use wgpui_core::scene::atlas::{ImageRasterKey, ImageTile, ImageTileSource, RasterizedImage};

    #[derive(Default)]
    struct Tiles {
        issued: std::collections::HashMap<ImageRasterKey, ImageTile>,
    }

    impl ImageTileSource for Tiles {
        fn tile_for(
            &mut self,
            key: ImageRasterKey,
            decode: &mut dyn FnMut(ImageRasterKey) -> Option<RasterizedImage>,
        ) -> Option<ImageTile> {
            if let Some(tile) = self.issued.get(&key) {
                return Some(*tile);
            }
            let raster = decode(key)?;
            let next = self.issued.len() as u32;
            let tile = ImageTile {
                tile: AtlasTileId::new(0, next).expect("in range"),
                atlas_origin: [0.0, 0.0],
                atlas_size: [raster.size[0] as f32, raster.size[1] as f32],
            };
            self.issued.insert(key, tile);
            Some(tile)
        }
    }

    // `r##`, not `r#`: the colour literal contains `"#`.
    const DOCUMENT: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" width="20" height="10">
        <rect width="20" height="10" fill="#cc3366"/>
    </svg>"##;

    fn engine() -> SharedImageEngine {
        Rc::new(RefCell::new(ImageEngine::new(
            ImageCache::new(),
            Box::new(Tiles::default()),
        )))
    }

    #[test]
    fn an_svg_loads_through_the_same_cache_an_image_does() {
        let engine = engine();
        let source = load(&engine, DOCUMENT.as_bytes(), 1.0).expect("a real document rasterises");
        let element = svg(source, Rc::clone(&engine));
        assert_eq!(
            element.rasterised_size(),
            Some([
                (20.0 * SMOOTH_SVG_SCALE_FACTOR) as u32,
                (10.0 * SMOOTH_SVG_SCALE_FACTOR) as u32
            ]),
            "the document is held at the legacy smoothing scale"
        );
    }

    #[test]
    fn a_requested_scale_multiplies_the_smoothing_one() {
        let engine = engine();
        let source = load(&engine, DOCUMENT.as_bytes(), 2.0).expect("rasterises");
        assert_eq!(
            svg(source, engine).rasterised_size(),
            Some([80, 40]),
            "20 x 2 (requested) x 2 (smoothing)"
        );
    }

    #[test]
    fn an_svg_emits_the_same_sprite_kind_an_image_does() {
        use wgpui_core::scene::layer::{BoundaryId, LayerId, LayerKey};
        let engine = engine();
        let source = load(&engine, DOCUMENT.as_bytes(), 1.0).expect("rasterises");
        let element = svg(source, Rc::clone(&engine)).size(40.0, 40.0);

        let mut emission = Emission::new();
        element.emit_into(
            &EmitContext {
                bounds: wgpui_layout::taffy_tree::LayoutRect {
                    x: 0.0,
                    y: 0.0,
                    width: 40.0,
                    height: 40.0,
                },
                layer: LayerId::from_key(LayerKey::untiled(BoundaryId::ROOT)),
                boundary: BoundaryId::ROOT,
                clip: None,
            },
            &mut emission,
        );
        let sprite = emission
            .poly_sprites()
            .first()
            .copied()
            .expect("one sprite, on the same path as any image");
        assert_eq!(sprite.atlas_size, [40.0, 20.0], "the rasterised extent");
        assert!(!sprite.atlas_tile.is_none());
        assert!(
            emission.quads().is_empty() && emission.glyph_runs().is_empty(),
            "an SVG contributes to exactly one kind"
        );
    }

    #[test]
    fn each_field_of_the_key_reports_the_axes_it_affects() {
        let engine = engine();
        let source = load(&engine, DOCUMENT.as_bytes(), 1.0).expect("rasterises");
        let base = svg(source, Rc::clone(&engine)).size(40.0, 40.0);

        assert_eq!(
            base.diff_key().compare(&base.diff_key()),
            Invalidation::empty()
        );
        assert_eq!(
            base.clone()
                .size(40.0, 41.0)
                .diff_key()
                .compare(&base.diff_key()),
            Invalidation::LAYOUT.union(Invalidation::DISPLAY),
            "a resize moves the Taffy leaf and repaints"
        );
        assert_eq!(
            base.clone()
                .style(ImageStyle {
                    grayscale: true,
                    ..ImageStyle::default()
                })
                .diff_key()
                .compare(&base.diff_key()),
            Invalidation::DISPLAY
        );

        let other = load(&engine, DOCUMENT.as_bytes(), 1.0).expect("rasterises");
        assert_eq!(
            svg(other, Rc::clone(&engine))
                .size(40.0, 40.0)
                .diff_key()
                .compare(&base.diff_key()),
            Invalidation::DISPLAY,
            "the same bytes loaded twice are two sources, per `ImageSourceId`'s \
             contract, and the element must say so"
        );
    }

    #[test]
    fn bytes_that_are_not_a_document_are_reported_rather_than_rasterised_blank() {
        assert!(load(&engine(), b"<not-svg", 1.0).is_err());
    }
}
