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
//! # What is deliberately not here: the legacy tinted alpha-mask path
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
//! carry — and it remains explicitly unsupported rather than quietly folded into
//! "SVG works." `text_color` uses the supported RGBA path and creates an
//! immutable derived image source instead.

use std::any::Any;
use wgpui_core::element::Element;
use wgpui_core::invalidation::axes::Invalidation;
use wgpui_core::patch::emit::{Emission, EmitContext};
use wgpui_core::reconcile::description::Description;
use wgpui_core::reconcile::diff_key::ReconcileKey;
use wgpui_layout::taffy_tree::{Dimension, LayoutSize, LayoutStyle};

use crate::animation::Transformation;
use crate::assets::{AssetRegistry, AssetState, Resource};
use crate::image_cache::{ImageCache, ImageDecodeError, decode_svg_at};
use crate::img::{
    ImageSourceId, ImageStyle, Img, SharedImageEngine, pending_engine, resource_source_id,
};
use wgpui_text::shaping::SharedString;

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
/// is what the rule is for. It is [`crate::img::ImgKey`] minus the animation
/// frame, while retaining load state because resource-backed SVGs resolve
/// asynchronously.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SvgKey {
    /// Which rasterised document is displayed.
    pub source: ImageSourceId,
    /// The box the element asked layout for.
    pub requested_size: [f32; 2],
    /// How that box is drawn.
    pub style: ImageStyle,
    /// Whether the document, loading placeholder, or failure fallback paints.
    pub load_state: crate::img::ImageLoadState,
    pub transformation: Transformation,
    pub text_color: Option<[f32; 4]>,
    /// Whether the rendered sprite is clipped to this element's layout box.
    pub overflow_hidden: bool,
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
        if previous.load_state != self.load_state {
            axes |= Invalidation::LAYOUT;
            axes |= Invalidation::DISPLAY;
        }
        if previous.source != self.source
            || previous.style != self.style
            || previous.transformation != self.transformation
            || previous.text_color != self.text_color
            || previous.overflow_hidden != self.overflow_hidden
        {
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
    path: Option<SharedString>,
    transformation: Transformation,
    text_color: Option<[f32; 4]>,
    overflow_hidden: bool,
}

/// Construct the authoritative source-ID/engine SVG.
pub fn svg_with_engine(source: ImageSourceId, engine: SharedImageEngine) -> Svg {
    Svg::new(source, engine)
}

/// Construct an asset-backed SVG builder.
pub fn svg() -> SvgBuilder {
    SvgBuilder {
        svg: Svg::pending(),
    }
}

pub struct SvgBuilder {
    svg: Svg,
}

impl SvgBuilder {
    pub fn path(mut self, path: impl Into<SharedString>) -> Self {
        self.svg = self.svg.path(path);
        self
    }

    pub fn size(self, size: impl crate::styled::IntoStylePixels) -> Self {
        self.size_square(size)
    }

    fn size_square(mut self, size: impl crate::styled::IntoStylePixels) -> Self {
        let size = size.into_style_pixels();
        self.svg = self.svg.size(size, size);
        self
    }

    pub fn size_8(mut self) -> Self {
        self.svg = self.svg.size_8();
        self
    }

    pub fn size_12(mut self) -> Self {
        self.svg = self.svg.size_12();
        self
    }

    pub fn size_16(mut self) -> Self {
        self.svg = self.svg.size_16();
        self
    }

    pub fn size_full(mut self) -> Self {
        self.svg = self.svg.size_full();
        self
    }

    pub fn h(mut self, height: impl crate::styled::IntoStylePixels) -> Self {
        self.svg = self.svg.h(height);
        self
    }

    pub fn w(mut self, width: impl crate::styled::IntoStylePixels) -> Self {
        self.svg = self.svg.w(width);
        self
    }

    pub fn max_w_full(mut self) -> Self {
        self.svg = self.svg.max_w_full();
        self
    }

    pub fn text_2xl(self) -> Self {
        self.size_square(24.0)
    }

    pub fn overflow_hidden(mut self) -> Self {
        self.svg = self.svg.overflow_hidden();
        self
    }

    pub fn text_color(mut self, color: impl Into<wgpui_core::color::Hsla>) -> Self {
        self.svg = self.svg.text_color(color);
        self
    }

    pub fn with_transformation(mut self, transformation: Transformation) -> Self {
        self.svg = self.svg.with_transformation(transformation);
        self
    }
}

impl Element for SvgBuilder {
    fn into_description(self) -> Description {
        self.svg.into_description()
    }

    fn into_description_in(
        mut self,
        _window: &mut wgpui_core::window::Window,
        app: &wgpui_core::App,
    ) -> Description {
        let Some(path) = self.svg.path.clone() else {
            return self.svg.into_description();
        };
        let Some(registry) = app.global::<AssetRegistry>() else {
            return self
                .svg
                .load_state(crate::img::ImageLoadState::Failed)
                .into_description();
        };
        let resource = Resource::from(path);
        let request = registry.load_async(resource.clone(), app);
        let state = registry.state(&resource);
        request.detach();
        match state {
            Some(AssetState::Ready) => match registry.cached(&resource) {
                Some(image) => {
                    let source = resource_source_id(&resource);
                    self.svg.image = self.svg.image.set_decoded(source, image);
                }
                None => {
                    self.svg.image = self
                        .svg
                        .image
                        .load_state(crate::img::ImageLoadState::Failed);
                }
            },
            Some(AssetState::Loading) => {
                self.svg.image = self
                    .svg
                    .image
                    .load_state(crate::img::ImageLoadState::Loading);
            }
            Some(AssetState::Failed(_)) | None => {
                self.svg.image = self
                    .svg
                    .image
                    .load_state(crate::img::ImageLoadState::Failed);
            }
            Some(AssetState::Cancelled) => {
                self.svg.image = self
                    .svg
                    .image
                    .load_state(crate::img::ImageLoadState::Cancelled);
            }
        }
        self.svg.into_description()
    }
}

impl Svg {
    fn pending() -> Self {
        Self::new(
            ImageSourceId::from_raw(1),
            pending_engine(ImageCache::new()),
        )
    }

    pub fn from_resource(resource: Resource) -> Self {
        let source = resource_source_id(&resource);
        Self::new(source, pending_engine(ImageCache::new()))
            .load_state(crate::img::ImageLoadState::Loading)
    }

    /// An SVG showing the document [`load`] put at `source`.
    pub fn new(source: ImageSourceId, engine: SharedImageEngine) -> Self {
        Self {
            image: Img::new(source, engine),
            path: None,
            transformation: Transformation::default(),
            text_color: None,
            overflow_hidden: false,
        }
    }

    fn load_state(mut self, load_state: crate::img::ImageLoadState) -> Self {
        self.image = self.image.load_state(load_state);
        self
    }

    /// Request a size.
    pub fn size(mut self, width: f32, height: f32) -> Self {
        self.image = self.image.size(width, height);
        self
    }

    pub fn size_8(self) -> Self {
        self.size(32.0, 32.0)
    }

    pub fn size_12(self) -> Self {
        self.size(48.0, 48.0)
    }

    pub fn size_16(self) -> Self {
        self.size(64.0, 64.0)
    }

    pub fn size_full(mut self) -> Self {
        self.image = self.image.size_full();
        self
    }

    pub fn h(mut self, height: impl crate::styled::IntoStylePixels) -> Self {
        self.image = self.image.h(height);
        self
    }

    pub fn w(mut self, width: impl crate::styled::IntoStylePixels) -> Self {
        self.image = self.image.w(width);
        self
    }

    pub fn max_w_full(mut self) -> Self {
        self.image = self.image.max_w_full();
        self
    }

    /// Decode a local asset path into this SVG's retained source.
    ///
    /// URI resources stay in `Loading` until an application-owned loader
    /// supplies an engine-backed source; no global loader is consulted here.
    pub fn path(mut self, path: impl Into<SharedString>) -> Self {
        let path = path.into();
        self.path = Some(path.clone());
        self.image = Self::from_resource(Resource::from(path)).image;
        if let Some(text_color) = self.text_color {
            self.image = self.image.tint(text_color);
        }
        self
    }

    /// Clip the SVG's rendered sprite to its layout bounds.
    pub fn overflow_hidden(mut self) -> Self {
        self.overflow_hidden = true;
        self
    }

    /// Set the requested icon colour with an immutable derived RGBA source.
    pub fn text_color(mut self, color: impl Into<wgpui_core::color::Hsla>) -> Self {
        let color: [f32; 4] = color.into().into();
        self.text_color = Some(color);
        self.image = self.image.tint(color);
        self
    }

    pub fn with_transformation(mut self, transformation: Transformation) -> Self {
        self.transformation = transformation;
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
            load_state: image.load_state,
            transformation: self.transformation,
            text_color: self.text_color,
            overflow_hidden: self.overflow_hidden,
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
        let description = Description::new::<Svg>()
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
            });
        if self.overflow_hidden {
            description.clip_children()
        } else {
            description
        }
    }

    /// Write this element's sprite into `emission`, for a caller driving
    /// emission directly.
    pub fn emit_into(&self, context: &EmitContext, emission: &mut Emission) {
        self.image
            .emit_into_with_transform(context, emission, self.transformation);
        if self.overflow_hidden {
            emission.clip_to(context.bounds);
        }
    }
}

impl Element for Svg {
    fn into_description(self) -> Description {
        self.describe()
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
        let element = svg_with_engine(source, Rc::clone(&engine));
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
            svg_with_engine(source, engine).rasterised_size(),
            Some([80, 40]),
            "20 x 2 (requested) x 2 (smoothing)"
        );
    }

    #[test]
    fn svg_text_colours_are_per_instance_and_reuse_matching_derived_sources() {
        let engine = engine();
        let source = load(&engine, DOCUMENT.as_bytes(), 1.0).expect("rasterises");
        let original = engine
            .borrow_mut()
            .cache()
            .get(source)
            .expect("the SVG source is held")
            .clone();
        let red = svg_with_engine(source, Rc::clone(&engine))
            .text_color(wgpui_core::color::rgb(0xff0000));
        let green = svg_with_engine(source, Rc::clone(&engine))
            .text_color(wgpui_core::color::rgb(0x00ff00));
        let red_again = svg_with_engine(source, Rc::clone(&engine))
            .text_color(wgpui_core::color::rgb(0xff0000));

        assert_ne!(red.diff_key().source, green.diff_key().source);
        assert_eq!(red.diff_key().source, red_again.diff_key().source);
        assert_eq!(
            red.diff_key().compare(&green.diff_key()),
            Invalidation::DISPLAY
        );
        assert_eq!(
            red.diff_key().compare(&red_again.diff_key()),
            Invalidation::empty()
        );
        assert_eq!(
            engine.borrow_mut().cache().get(source),
            Some(&original),
            "SVG recolouring never changes the shared raster"
        );
    }

    #[test]
    fn an_svg_emits_the_same_sprite_kind_an_image_does() {
        use wgpui_core::scene::layer::{BoundaryId, LayerId, LayerKey};
        let engine = engine();
        let source = load(&engine, DOCUMENT.as_bytes(), 1.0).expect("rasterises");
        let element = svg_with_engine(source, Rc::clone(&engine)).size(40.0, 40.0);

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
        let base = svg_with_engine(source, Rc::clone(&engine)).size(40.0, 40.0);

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
        assert_eq!(
            base.clone()
                .load_state(crate::img::ImageLoadState::Loading)
                .diff_key()
                .compare(&base.diff_key()),
            Invalidation::LAYOUT.union(Invalidation::DISPLAY),
            "a resource transition swaps the SVG fallback and image"
        );
        assert_eq!(
            base.clone()
                .overflow_hidden()
                .diff_key()
                .compare(&base.diff_key()),
            Invalidation::DISPLAY,
            "changing the local clip changes only the retained paint"
        );

        let other = load(&engine, DOCUMENT.as_bytes(), 1.0).expect("rasterises");
        assert_eq!(
            svg_with_engine(other, Rc::clone(&engine))
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

    #[test]
    fn compatibility_builder_lowers_path_colour_and_transform_to_the_key() {
        let description = svg()
            .path("missing/icon.svg")
            .size_16()
            .text_color(wgpui_core::color::rgb(0xff3366))
            .with_transformation(Transformation::rotate(90.0))
            .into_description();
        let key = description
            .key()
            .and_then(|key| key.as_any().downcast_ref::<SvgKey>())
            .expect("SVG builders expose their retained SVG key");

        assert_eq!(key.requested_size, [64.0, 64.0]);
        assert_eq!(key.transformation.rotation, 90.0);
        let text_color = key.text_color.expect("the tint is retained");
        for (actual, expected) in text_color.into_iter().zip([1.0, 0.2, 0.4, 1.0]) {
            assert!((actual - expected).abs() < 0.0001);
        }
    }

    #[test]
    fn compatibility_builder_sizing_aliases_reach_the_image_layout_request() {
        let size = svg().size_12().into_description();
        let key = size
            .key()
            .and_then(|key| key.as_any().downcast_ref::<SvgKey>())
            .expect("SVG sizing is retained in the SVG key");
        assert_eq!(key.requested_size, [48.0, 48.0]);

        let text_size = svg().text_2xl().into_description();
        let key = text_size
            .key()
            .and_then(|key| key.as_any().downcast_ref::<SvgKey>())
            .expect("SVG text sizing is retained in the SVG key");
        assert_eq!(key.requested_size, [24.0, 24.0]);
    }

    #[test]
    fn overflow_hidden_clips_cover_and_transformed_svg_content_at_both_boundaries() {
        use wgpui_core::scene::layer::{BoundaryId, LayerId, LayerKey};
        use wgpui_layout::taffy_tree::LayoutRect;

        let engine = engine();
        let source = load(&engine, DOCUMENT.as_bytes(), 1.0).expect("rasterises");
        let element = svg_with_engine(source, Rc::clone(&engine))
            .size(20.0, 20.0)
            .style(ImageStyle {
                object_fit: crate::img::ObjectFit::Cover,
                ..ImageStyle::default()
            })
            .with_transformation(
                Transformation::default()
                    .with_scaling(wgpui_core::geometry::Size::new(
                        wgpui_core::geometry::Pixels(2.0),
                        wgpui_core::geometry::Pixels(1.0),
                    ))
                    .with_translation(wgpui_core::geometry::Point::new(
                        wgpui_core::geometry::Pixels(5.0),
                        wgpui_core::geometry::Pixels(-5.0),
                    )),
            )
            .overflow_hidden();
        assert!(
            element.describe().clips_children(),
            "overflow_hidden must remain visible in the retained description"
        );

        let mut emission = Emission::new();
        element.emit_into(
            &EmitContext {
                bounds: LayoutRect {
                    x: 10.0,
                    y: 10.0,
                    width: 20.0,
                    height: 20.0,
                },
                layer: LayerId::from_key(LayerKey::untiled(BoundaryId::ROOT)),
                boundary: BoundaryId::ROOT,
                clip: Some(LayoutRect {
                    x: 15.0,
                    y: 15.0,
                    width: 10.0,
                    height: 10.0,
                }),
            },
            &mut emission,
        );
        let inherited_clip = LayoutRect {
            x: 15.0,
            y: 15.0,
            width: 10.0,
            height: 10.0,
        };
        emission.clip_to(inherited_clip);

        let sprite = emission
            .poly_sprites()
            .first()
            .copied()
            .expect("a ready SVG emits one sprite");
        assert_eq!(sprite.origin, [15.0, 15.0]);
        assert_eq!(sprite.size, [10.0, 10.0]);
        assert_eq!(sprite.atlas_origin, [15.0, 10.0]);
        assert_eq!(sprite.atlas_size, [5.0, 10.0]);
    }
}
