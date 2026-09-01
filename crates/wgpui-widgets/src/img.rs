//! `Img` — gets `diff_key` here (§6.2 invariant, Phase 5), closing R-N
//! Phase 7's self-documented gap. See docs/gpu-native-architecture.md §3.4,
//! §6.2.
//!
//! # The legacy blocker, and why it does not apply here
//!
//! `Img` not having a `diff_key` was not an oversight anybody forgot about —
//! the legacy element says so in a twelve-line comment above its `Element`
//! impl (`src/elements/img.rs`, just before `impl Element for Img`), and the
//! reason it gives is a real one:
//!
//! > What `paint` shows for an `Img` depends on `ImgState` (per-element,
//! > `with_optional_element_state`-keyed: `frame_index`, `started_loading`,
//! > `last_frame_time`) and `ImgLayoutState.replacement` (a fallback/loading
//! > `AnyElement` substituted in when `request_layout` finds no data yet) —
//! > neither of which is reachable from `Img::diff_key(&self, _)`.
//!
//! That is a statement about *where the state lived*, not about images. In the
//! legacy element the animation frame and the load phase are discovered during
//! `request_layout`/`paint`, which run strictly after `diff_key` is asked for
//! its answer, so a key over `source`/`style` alone would report "unchanged"
//! across a GIF advancing a frame or a pending load resolving, and paint would
//! replay stale content. Opting out unconditionally was the correct call under
//! that ordering.
//!
//! 2.0 does not have that ordering. An element contributes a
//! [`Description`] — built from a value that already holds its resolved state,
//! the same way [`crate::wgpu_surface::WgpuSurface`] already holds its
//! resolved `surface_id` — and the fingerprint is taken from that value. So
//! [`ImgKey`] carries [`ImgKey::frame_index`] and [`ImgKey::load_state`]
//! directly, and the two transitions the legacy comment names are exactly the
//! two the key reports as changed. The fix is the state being *addressable*,
//! not a cleverer comparison.
//!
//! # What the key deliberately does not hold
//!
//! Not the decoded pixels, and not anything requiring a decode to compute.
//! §6.2's whole point is that the key is cheap enough to take every frame for
//! every element in the tree; hashing an image's texels would cost more than
//! the rebuild it exists to avoid. Source *identity* plus the resolved frame
//! index is a complete answer regardless: two different pixel buffers cannot
//! share one source identity at one frame index without the image cache having
//! substituted content behind the same handle, which it does not do — a
//! reloaded source gets a new [`ImageSourceId`].

use std::any::Any;
use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::{path::PathBuf, time::Instant};
use wgpui_core::element::Element;
use wgpui_core::invalidation::axes::Invalidation;
use wgpui_core::patch::emit::{Emission, EmitContext};
use wgpui_core::patch::primitive::{AtlasTileId, PolySprite};
use wgpui_core::reconcile::description::Description;
use wgpui_core::reconcile::diff_key::ReconcileKey;
use wgpui_core::scene::atlas::{ImageRasterKey, ImageTileSource};
use wgpui_layout::taffy_tree::{Dimension, LayoutSize, LayoutStyle};

use crate::assets::{AssetRegistry, AssetState, Resource};
use crate::image_cache::ImageCache;
use crate::styled::IntoStylePixels;

/// Identity of the thing an image is loaded from — a path, a URI, an asset
/// handle, an in-memory buffer's registration.
///
/// Opaque on purpose. What reconciliation needs is that two `Img`s showing the
/// same resource compare equal and two showing different resources do not; how
/// a source is named is the image cache's business (`image_cache.rs`), not the
/// fingerprint's. A source that is reloaded — re-fetched, re-decoded, replaced
/// on disk — is issued a new id rather than mutating in place, which is what
/// makes comparing identity rather than content sound.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ImageSourceId(u64);

impl ImageSourceId {
    /// Wrap a raw source handle.
    pub const fn from_raw(raw: u64) -> Self {
        ImageSourceId(raw)
    }

    /// The raw source handle.
    pub const fn as_raw(self) -> u64 {
        self.0
    }
}

/// Which of an image's three possible renderings is actually on screen.
///
/// The legacy element expresses this as `ImgLayoutState.replacement`: an
/// `AnyElement` standing in for the image while it loads or after it fails.
/// Swapping that in or out changes both what is painted and what the layout
/// tree contains, so it is named here rather than left implicit.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum ImageLoadState {
    /// Decoded and available; the image itself is what paints.
    #[default]
    Ready,
    /// Still loading; a placeholder subtree paints instead.
    Loading,
    /// Loading failed; a fallback subtree paints instead.
    Failed,
    /// Loading was cancelled; the fallback subtree remains active.
    Cancelled,
}

/// How an image's own aspect ratio is reconciled with the box it was given.
///
/// Mirrors the legacy `ObjectFit` (`src/elements/img.rs`). It affects only
/// where inside an already-decided rectangle the content lands, so it is a
/// `DISPLAY`-axis property, never a `LAYOUT` one.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum ObjectFit {
    /// Scale to fill, ignoring aspect ratio.
    Fill,
    /// Scale to fit entirely inside, preserving aspect ratio.
    #[default]
    Contain,
    /// Scale to cover entirely, preserving aspect ratio, cropping the excess.
    Cover,
    /// Draw at natural size.
    None,
    /// `None`, unless that overflows, in which case `Contain`.
    ScaleDown,
}

impl ObjectFit {
    /// Where inside `bounds` an image of `image_size` pixels is drawn.
    ///
    /// `bounds` and the result are `[x, y, width, height]`. A transcription of
    /// the legacy `ObjectFit::get_bounds` (`src/style.rs`), expression for
    /// expression including the order of its multiplications and divisions,
    /// because "the same idea" and "the same float" are not the same thing and
    /// the second is what a differential compares.
    ///
    /// A zero-area image or box has no ratio; both answer with `bounds`, which
    /// is `Fill`'s answer and the only one that is not a division by zero. The
    /// legacy function does not guard this — it produces `NaN` bounds — and this
    /// is one of the two places 2.0 deliberately does not reproduce legacy
    /// behaviour exactly. See docs/phase-6.2-results.md.
    pub fn fit(self, bounds: [f32; 4], image_size: [u32; 2]) -> [f32; 4] {
        let [x, y, box_width, box_height] = bounds;
        let image_width = image_size[0] as f32;
        let image_height = image_size[1] as f32;
        if image_width <= 0.0 || image_height <= 0.0 || box_width <= 0.0 || box_height <= 0.0 {
            return bounds;
        }
        let image_ratio = image_width / image_height;
        let bounds_ratio = box_width / box_height;

        let centred = |width: f32, height: f32| {
            [
                x + (box_width - width) / 2.0,
                y + (box_height - height) / 2.0,
                width,
                height,
            ]
        };
        let contained = || {
            if bounds_ratio > image_ratio {
                centred(image_width * (box_height / image_height), box_height)
            } else {
                centred(box_width, image_height * (box_width / image_width))
            }
        };

        match self {
            ObjectFit::Fill => bounds,
            ObjectFit::Contain => contained(),
            ObjectFit::ScaleDown => {
                if image_width > box_width || image_height > box_height {
                    contained()
                } else {
                    centred(image_width, image_height)
                }
            }
            ObjectFit::Cover => {
                if bounds_ratio > image_ratio {
                    centred(box_width, image_height * (box_width / image_width))
                } else {
                    centred(image_width * (box_height / image_height), box_height)
                }
            }
            // The legacy arm, kept exactly: the natural size at the box's own
            // origin, *not* centred. `Contain`/`Cover`/`ScaleDown` centre and
            // `None` does not, which reads like an inconsistency and is the
            // behaviour every existing caller has.
            ObjectFit::None => [x, y, image_width, image_height],
        }
    }
}

/// The display-affecting styling of an image.
///
/// Deliberately not the legacy `ImageStyle` verbatim: that type also holds
/// `loading` and `fallback`, which are `Box<dyn Fn() -> AnyElement>` closures.
/// Closures are not comparable and, per R-N §2.4, are never compared —
/// [`ImageLoadState`] carries the part of them that is observable (which of the
/// three renderings is active), and the closures themselves are swapped in
/// unconditionally like any other listener.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ImageStyle {
    /// Draw desaturated.
    pub grayscale: bool,
    /// How the content is fitted into its box.
    pub object_fit: ObjectFit,
    /// Straight alpha the image composites at.
    pub opacity: f32,
    /// Uniform corner radius the image is clipped to.
    pub corner_radius: f32,
}

impl Default for ImageStyle {
    /// Opaque, contained, square-cornered, full colour.
    ///
    /// Written out rather than derived, because the derive would make
    /// [`Self::opacity`] zero and an image styled `..ImageStyle::default()`
    /// would be invisible — a default that is wrong exactly once and silently.
    /// Before Phase 6.2 nothing read the field, which is why the derive was
    /// harmless until it was not.
    fn default() -> Self {
        Self {
            grayscale: false,
            object_fit: ObjectFit::Contain,
            opacity: 1.0,
            corner_radius: 0.0,
        }
    }
}

/// The fingerprint an `Img` presents to ambient reconciliation.
///
/// Five fields, each of which changes what a viewer sees without any of the
/// others changing — which is the test for whether a field belongs in a key at
/// all. Everything else about an image is either derived from these (the
/// decoded texels, from the source and the frame) or invisible (the cache
/// handle it was fetched through).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ImgKey {
    /// Which resource is displayed.
    pub source: ImageSourceId,
    /// Which frame of an animated source is displayed. `0` for still images,
    /// which is why a still image's key is stable across frames for free.
    pub frame_index: u32,
    /// Whether the image, a loading placeholder, or a failure fallback is what
    /// actually paints.
    pub load_state: ImageLoadState,
    /// The box the image asked layout for.
    pub requested_size: [f32; 2],
    /// How that box is drawn.
    pub style: ImageStyle,
    pub size_full: bool,
    pub max_width_full: bool,
}

impl ReconcileKey for ImgKey {
    fn compare(&self, previous: &dyn ReconcileKey) -> Invalidation {
        let Some(previous) = previous.as_any().downcast_ref::<ImgKey>() else {
            return Invalidation::all();
        };
        let mut axes = Invalidation::empty();
        if previous.requested_size != self.requested_size
            || previous.size_full != self.size_full
            || previous.max_width_full != self.max_width_full
        {
            // Moves the Taffy leaf and repaints, exactly like a resized
            // `WgpuSurface`.
            axes |= Invalidation::LAYOUT;
            axes |= Invalidation::DISPLAY;
        }
        if previous.load_state != self.load_state {
            // A load transition swaps a whole subtree in or out (the legacy
            // `ImgLayoutState.replacement`), so it is a layout change and not
            // only a repaint — the one case where being conservative is not
            // merely defensible but required.
            axes |= Invalidation::LAYOUT;
            axes |= Invalidation::DISPLAY;
        }
        if previous.source != self.source
            || previous.frame_index != self.frame_index
            || previous.style != self.style
        {
            axes |= Invalidation::DISPLAY;
        }
        axes
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Where an [`Img`] gets pixels: decoded frames, and the atlas tiles they land
/// in.
///
/// The image counterpart of [`crate::styled_text::TextEngine`], and it exists
/// for the identical reason that type's doc gives: the two halves of the path
/// are separately owned by design (§3.4 puts the decoder here, §3.5 puts the
/// atlas in `wgpui-wgpu`) and an element needs both at once, so this is where
/// they meet. It holds the [`ImageCache`] and an
/// [`ImageTileSource`] beside it so a frame can be decoded and placed in one
/// call.
pub struct ImageEngine {
    cache: ImageCache,
    tiles: Box<dyn ImageTileSource>,
}

impl ImageEngine {
    /// An engine decoding into `cache` and allocating through `tiles`.
    pub fn new(cache: ImageCache, tiles: Box<dyn ImageTileSource>) -> Self {
        Self { cache, tiles }
    }

    /// The decoded-image cache, for loading sources and reading their sizes.
    pub fn cache(&mut self) -> &mut ImageCache {
        &mut self.cache
    }

    /// The natural pixel size of one frame, if the source is decoded.
    pub fn frame_size(&self, source: ImageSourceId, frame_index: u32) -> Option<[u32; 2]> {
        Some(self.cache.frame(source, frame_index)?.size)
    }

    /// Return the looping frame selected by the decoded delays.
    pub fn frame_index_at(
        &self,
        source: ImageSourceId,
        elapsed: std::time::Duration,
    ) -> Option<u32> {
        Some(self.cache.get(source)?.frame_index_at(elapsed))
    }

    /// The tile holding one frame, allocating and uploading it if needed.
    ///
    /// `None` means the sprite draws nothing this frame — the source is not
    /// decoded, or the atlas refused it — which is ordinary and not an error.
    fn tile_for(
        &mut self,
        source: ImageSourceId,
        frame_index: u32,
    ) -> Option<TilePlacementOfFrame> {
        let frame_index = match self.cache.get(source) {
            Some(image) => {
                let frame_count = u32::try_from(image.frame_count()).ok()?;
                frame_index % frame_count
            }
            None => return None,
        };
        let key = ImageRasterKey {
            source: source.as_raw(),
            frame_index,
            // Phase 6.2 decodes at 1×. The field exists in the key because a
            // 2× decode is a different bitmap and the atlas has to be able to
            // hold both; nothing yet asks for the second one.
            scale_factor_bits: 1.0f32.to_bits(),
        };
        // Split borrows, and the reason [`ImageTileSource`] takes its decoder
        // per call: the cache is read and the atlas is mutated, by two halves
        // that live in two crates, so the closure holds the cache and the source
        // holds the atlas and neither reaches into the other.
        let cache = &self.cache;
        let tile = self.tiles.tile_for(key, &mut |key| cache.raster(key))?;
        Some(TilePlacementOfFrame {
            tile: tile.tile,
            atlas_origin: tile.atlas_origin,
            atlas_size: tile.atlas_size,
        })
    }
}

/// Where one frame's bitmap ended up, as [`Img`]'s emission needs it.
#[derive(Copy, Clone, Debug, PartialEq)]
struct TilePlacementOfFrame {
    tile: AtlasTileId,
    atlas_origin: [f32; 2],
    atlas_size: [f32; 2],
}

/// An [`ImageEngine`] several elements share.
///
/// `Rc<RefCell<_>>` rather than a lock, for the reason
/// [`crate::styled_text::SharedTextEngine`] gives: everything that reaches it
/// runs on the frame's thread. Shared rather than owned per element because the
/// decode cache and the atlas are only useful if every avatar on screen is
/// looking at the same ones.
pub type SharedImageEngine = Rc<RefCell<ImageEngine>>;

struct PendingImageTiles;

impl ImageTileSource for PendingImageTiles {
    fn tile_for(
        &mut self,
        _key: ImageRasterKey,
        _decode: &mut dyn FnMut(
            ImageRasterKey,
        ) -> Option<wgpui_core::scene::atlas::RasterizedImage>,
    ) -> Option<wgpui_core::scene::atlas::ImageTile> {
        None
    }
}

pub(crate) fn resource_source_id(resource: &Resource) -> ImageSourceId {
    let mut hasher = DefaultHasher::new();
    resource.hash(&mut hasher);
    ImageSourceId::from_raw(hasher.finish().max(1))
}

pub(crate) fn pending_engine(cache: ImageCache) -> SharedImageEngine {
    Rc::new(RefCell::new(ImageEngine::new(
        cache,
        Box::new(PendingImageTiles),
    )))
}

/// An image element's description shape.
///
/// Like [`crate::wgpu_surface::WgpuSurface`], this is the shape an image
/// presents to reconciliation and emission, not the full element: there is no
/// `AnyElement` replacement subtree here and no asset resolution, because those
/// need `App`/`Window`, which §3.4 puts elsewhere. What is real is the
/// fingerprint, the decode behind it, and the sprite it emits.
///
/// No longer `Copy`, as of Phase 6.2: it holds a [`SharedImageEngine`], exactly
/// as [`crate::styled_text::StyledText`] holds a text engine and for the same
/// reason — an element that draws pixels has to be able to reach the thing that
/// produces them. [`ImgKey`] is still `Copy`, which is what matters, since the
/// key is what reconciliation compares every frame.
#[derive(Clone)]
pub struct Img {
    element_id: Option<wgpui_core::reconcile::description::ElementId>,
    source: ImageSourceId,
    frame_index: u32,
    load_state: ImageLoadState,
    requested_size: [f32; 2],
    style: ImageStyle,
    engine: SharedImageEngine,
    started: Instant,
    automatic_frame: bool,
    size_full: bool,
    max_width_full: bool,
}

impl std::fmt::Debug for Img {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The engine is shared mutable state and formatting it would either
        // borrow it (and panic inside a frame that already holds it) or print an
        // address. The fingerprint is what identifies an `Img` anyway.
        formatter
            .debug_struct("Img")
            .field("key", &self.diff_key())
            .finish_non_exhaustive()
    }
}

impl PartialEq for Img {
    /// Two images are equal when their fingerprints are.
    ///
    /// The engine is deliberately not compared: it is shared, so two `Img`s
    /// drawing the same source through the same window necessarily hold the
    /// same one, and comparing `Rc` identity would make an element that
    /// re-created its engine look changed when nothing a viewer can see did.
    fn eq(&self, other: &Self) -> bool {
        self.diff_key() == other.diff_key()
    }
}

/// Construct the authoritative source-ID/engine image.
pub fn img_with_engine(source: ImageSourceId, engine: SharedImageEngine) -> Img {
    Img::new(source, engine)
}

pub type ImageInputResolver =
    Box<dyn FnOnce(&mut wgpui_core::window::Window, &mut wgpui_core::App) -> Resource>;

/// An additive resource-backed image builder.
pub struct ImgBuilder {
    image: Img,
    resource: Option<Resource>,
    resolver: Option<ImageInputResolver>,
}

pub trait IntoImageInput: 'static {
    fn into_image_input(self) -> (Option<Resource>, Option<ImageInputResolver>);
}

impl IntoImageInput for Resource {
    fn into_image_input(self) -> (Option<Resource>, Option<ImageInputResolver>) {
        (Some(self), None)
    }
}

impl IntoImageInput for PathBuf {
    fn into_image_input(self) -> (Option<Resource>, Option<ImageInputResolver>) {
        Resource::from(self).into_image_input()
    }
}

impl<F> IntoImageInput for F
where
    F: FnOnce(&mut wgpui_core::window::Window, &mut wgpui_core::App) -> Resource + 'static,
{
    fn into_image_input(self) -> (Option<Resource>, Option<ImageInputResolver>) {
        (None, Some(Box::new(self)))
    }
}

impl IntoImageInput for String {
    fn into_image_input(self) -> (Option<Resource>, Option<ImageInputResolver>) {
        Resource::from(self).into_image_input()
    }
}
impl IntoImageInput for &'static str {
    fn into_image_input(self) -> (Option<Resource>, Option<ImageInputResolver>) {
        Resource::from(self).into_image_input()
    }
}

/// Construct an image from a resource using the retained image representation.
pub fn img(source: impl IntoImageInput) -> ImgBuilder {
    let (resource, resolver) = source.into_image_input();
    let initial_resource = resource
        .clone()
        .unwrap_or_else(|| Resource::Embedded("pending-image".into()));
    ImgBuilder {
        image: Img::from_resource(initial_resource),
        resource,
        resolver,
    }
}

impl ImgBuilder {
    pub fn from_decoded(
        source: ImageSourceId,
        image: std::sync::Arc<crate::assets::RenderImage>,
    ) -> Self {
        let mut cache = ImageCache::new();
        let load_state = cache.hold_at(source, (*image).clone()).is_ok();
        Self {
            image: Img::new(source, pending_engine(cache)).load_state(if load_state {
                ImageLoadState::Ready
            } else {
                ImageLoadState::Failed
            }),
            resource: None,
            resolver: None,
        }
    }
    pub fn size(mut self, size: impl IntoStylePixels) -> Self {
        let size = size.into_style_pixels();
        self.image = self.image.size(size, size);
        self
    }

    pub fn size_8(self) -> Self {
        self.size(32.0)
    }

    pub fn size_12(self) -> Self {
        self.size(48.0)
    }

    pub fn size_16(self) -> Self {
        self.size(64.0)
    }

    pub fn size_full(mut self) -> Self {
        self.image = self.image.size_full();
        self
    }

    pub fn h(mut self, height: impl IntoStylePixels) -> Self {
        self.image = self.image.h(height);
        self
    }

    pub fn w(mut self, width: impl IntoStylePixels) -> Self {
        self.image = self.image.w(width);
        self
    }

    pub fn max_w_full(mut self) -> Self {
        self.image = self.image.max_w_full();
        self
    }

    pub fn object_fit(mut self, object_fit: ObjectFit) -> Self {
        let mut style = self.image.diff_key().style;
        style.object_fit = object_fit;
        self.image = self.image.style(style);
        self
    }

    pub fn id(mut self, id: impl Into<wgpui_core::reconcile::description::ElementId>) -> Self {
        self.image = self.image.id(id);
        self
    }
}

impl Element for ImgBuilder {
    fn into_description(self) -> Description {
        self.image.into_description()
    }

    fn into_description_in(
        self,
        _window: &mut wgpui_core::window::Window,
        app: &wgpui_core::App,
    ) -> Description {
        let resource = match (self.resource, self.resolver) {
            (Some(resource), _) => resource,
            (None, Some(resolver)) => resolver(_window, &mut app.clone()),
            (None, None) => return self.image.into_description(),
        };
        let Some(registry) = app.global::<AssetRegistry>() else {
            return self
                .image
                .load_state(ImageLoadState::Failed)
                .into_description();
        };
        let request = registry.load_async(resource.clone(), app);
        let state = registry.state(&resource);
        request.detach();
        match state {
            Some(AssetState::Ready) => match registry.cached(&resource) {
                Some(image) => {
                    let source = resource_source_id(&resource);
                    let load_state = if self
                        .image
                        .engine
                        .borrow_mut()
                        .cache()
                        .hold_at(source, (*image).clone())
                        .is_ok()
                    {
                        ImageLoadState::Ready
                    } else {
                        ImageLoadState::Failed
                    };
                    self.image.load_state(load_state).into_description()
                }
                None => self
                    .image
                    .load_state(ImageLoadState::Failed)
                    .into_description(),
            },
            Some(AssetState::Loading) => self
                .image
                .load_state(ImageLoadState::Loading)
                .into_description(),
            Some(AssetState::Failed(_)) | None => self
                .image
                .load_state(ImageLoadState::Failed)
                .into_description(),
            Some(AssetState::Cancelled) => self
                .image
                .load_state(ImageLoadState::Cancelled)
                .into_description(),
        }
    }
}

impl Img {
    pub(crate) fn set_decoded(
        mut self,
        source: ImageSourceId,
        image: std::sync::Arc<crate::assets::RenderImage>,
    ) -> Self {
        let loaded = self
            .engine
            .borrow_mut()
            .cache()
            .hold_at(source, (*image).clone())
            .is_ok();
        self.source = source;
        self.load_state(if loaded {
            ImageLoadState::Ready
        } else {
            ImageLoadState::Failed
        })
    }

    /// An image showing `source`, at its first frame, ready, unsized, unstyled.
    pub fn new(source: ImageSourceId, engine: SharedImageEngine) -> Self {
        Self {
            element_id: None,
            source,
            frame_index: 0,
            load_state: ImageLoadState::Ready,
            requested_size: [0.0, 0.0],
            style: ImageStyle {
                grayscale: false,
                object_fit: ObjectFit::Contain,
                opacity: 1.0,
                corner_radius: 0.0,
            },
            engine,
            started: Instant::now(),
            automatic_frame: true,
            size_full: false,
            max_width_full: false,
        }
    }

    /// Select the frame of an animated source that is currently displayed.
    pub fn frame_index(mut self, frame_index: u32) -> Self {
        self.frame_index = frame_index;
        self.automatic_frame = false;
        self
    }

    /// Select the frame visible at `now - started` and return this image for
    /// the next ordinary description pass.
    pub fn frame_at(mut self, started: std::time::Instant, now: std::time::Instant) -> Self {
        let elapsed = now.saturating_duration_since(started);
        let frame_index = self
            .engine
            .borrow()
            .frame_index_at(self.source, elapsed)
            .unwrap_or(0);
        self.started = started;
        self.frame_index = frame_index;
        self.automatic_frame = true;
        self
    }

    /// Request a display tick while this image has another GIF/WebP frame.
    pub fn request_next_frame(
        &self,
        now: Instant,
        scheduler: &mut wgpui_core::window::animation::AnimationScheduler,
    ) {
        if !self.automatic_frame {
            return;
        }
        let elapsed = now.saturating_duration_since(self.started);
        if self
            .engine
            .borrow()
            .cache
            .get(self.source)
            .and_then(|image| image.time_until_next_frame(elapsed))
            .is_some()
        {
            scheduler.request_animation_frame();
        }
    }

    /// Record which of the three renderings is active this frame.
    pub fn load_state(mut self, load_state: ImageLoadState) -> Self {
        self.load_state = load_state;
        self
    }

    /// Request a size.
    ///
    /// Named *requested* for the same reason [`crate::wgpu_surface::WgpuSurface::bounds`]
    /// is: resolved bounds arrive from layout, after the description exists. An
    /// image resolved somewhere new without its own request changing is still
    /// handled, by `patch::emit`'s own "did this element move" rule.
    pub fn size(mut self, width: f32, height: f32) -> Self {
        self.requested_size = [width, height];
        self.size_full = false;
        self
    }

    pub fn id(
        mut self,
        element_id: impl Into<wgpui_core::reconcile::description::ElementId>,
    ) -> Self {
        self.element_id = Some(element_id.into());
        self
    }

    pub fn from_resource(resource: Resource) -> Self {
        let source = resource_source_id(&resource);
        Self::new(source, pending_engine(ImageCache::new())).load_state(ImageLoadState::Loading)
    }

    /// Request a fixed square size using the standard spacing scale.
    pub fn size_8(self) -> Self {
        self.size(32.0, 32.0)
    }

    /// Request a fixed square size using the standard spacing scale.
    pub fn size_12(self) -> Self {
        self.size(48.0, 48.0)
    }

    /// Request a fixed square size using the standard spacing scale.
    pub fn size_16(self) -> Self {
        self.size(64.0, 64.0)
    }

    /// Let the image fill its parent in both axes.
    pub fn size_full(mut self) -> Self {
        self.size_full = true;
        self
    }

    /// Set the height and derive the width from the decoded image ratio.
    pub fn h(mut self, height: impl IntoStylePixels) -> Self {
        self.requested_size = [0.0, height.into_style_pixels()];
        self.size_full = false;
        self
    }

    /// Set the width and derive the height from the decoded image ratio.
    pub fn w(mut self, width: impl IntoStylePixels) -> Self {
        self.requested_size = [width.into_style_pixels(), 0.0];
        self.size_full = false;
        self
    }

    /// Limit the image's width to its containing block.
    pub fn max_w_full(mut self) -> Self {
        self.max_width_full = true;
        self
    }

    /// Set how the image is drawn.
    pub fn style(mut self, style: ImageStyle) -> Self {
        self.style = style;
        self
    }

    pub fn tint(self, color: [f32; 4]) -> Self {
        self.engine.borrow_mut().cache().tint(self.source, color);
        self
    }

    /// This image's fingerprint.
    pub fn diff_key(&self) -> ImgKey {
        ImgKey {
            source: self.source,
            frame_index: self.frame_index,
            load_state: self.load_state,
            requested_size: self.requested_size,
            style: self.style,
            size_full: self.size_full,
            max_width_full: self.max_width_full,
        }
    }

    /// The natural pixel size of the frame this image is showing, if its source
    /// has been decoded.
    ///
    /// `None` while the source is still loading, which is the same condition
    /// [`ImageLoadState::Loading`] names — and the reason an image can occupy a
    /// box before it has pixels.
    pub fn natural_size(&self) -> Option<[u32; 2]> {
        self.engine
            .borrow()
            .frame_size(self.source, self.frame_index)
    }

    /// The box this image asks layout for.
    ///
    /// An explicit [`Self::size`] wins. Otherwise a decoded image asks for its
    /// own natural size, which is the concrete thing having real dimensions buys
    /// — before Phase 6.2 an unsized `Img` asked for a zero-sized box and
    /// disappeared. An undecoded source still asks for zero, because it has
    /// nothing else to ask for; see [`Self::natural_size`] and
    /// docs/phase-6.2-results.md on why this is *not* the same as discharging
    /// §6.2's `estimated_size` half.
    pub fn layout_size(&self) -> [f32; 2] {
        if self.size_full {
            return [0.0, 0.0];
        }
        if self.requested_size != [0.0, 0.0] {
            let natural = self.natural_size();
            return match (self.requested_size, natural) {
                ([0.0, height], Some([width, natural_height])) if natural_height > 0 => {
                    [height * width as f32 / natural_height as f32, height]
                }
                ([width, 0.0], Some([natural_width, height])) if natural_width > 0 => {
                    [width, width * height as f32 / natural_width as f32]
                }
                (requested, _) => requested,
            };
        }
        match self.natural_size() {
            Some([width, height]) => [width as f32, height as f32],
            None => [0.0, 0.0],
        }
    }

    /// The per-frame description of this image.
    pub fn describe(&self) -> Description {
        let image = if self.automatic_frame {
            self.clone().frame_at(self.started, Instant::now())
        } else {
            self.clone()
        };
        let [width, height] = image.layout_size();
        let is_active_animation = image.load_state == ImageLoadState::Ready
            && image.automatic_frame
            && image
                .engine
                .borrow()
                .cache
                .get(image.source)
                .is_some_and(|source| source.is_animated());
        let mut layout_style = LayoutStyle {
            size: LayoutSize {
                width: Dimension::length(width),
                height: Dimension::length(height),
            },
            flex_shrink: 0.0,
            ..LayoutStyle::default()
        };
        if image.size_full {
            layout_style.size = LayoutSize {
                width: Dimension::percent(1.0),
                height: Dimension::percent(1.0),
            };
        }
        if image.max_width_full {
            layout_style.max_size.width = Dimension::percent(1.0);
        }
        let description = Description::new::<Img>()
            .diff_key(image.diff_key())
            .style(layout_style)
            .emit(move |context: &EmitContext, emission: &mut Emission| {
                image.emit_into(context, emission);
            });
        let description = if let Some(element_id) = self.element_id.clone() {
            description.id(element_id)
        } else {
            description
        };
        if is_active_animation {
            description.active_animation()
        } else {
            description
        }
    }

    pub fn emit_into(&self, context: &EmitContext, emission: &mut Emission) {
        self.emit_into_with_transform(
            context,
            emission,
            crate::animation::Transformation::default(),
        );
    }

    pub(crate) fn emit_into_with_transform(
        &self,
        context: &EmitContext,
        emission: &mut Emission,
        transformation: crate::animation::Transformation,
    ) {
        if self.load_state != ImageLoadState::Ready {
            // A loading or failed image paints its replacement subtree, not
            // itself. 2.0 has no `AnyElement` replacement (§3.4 puts it with
            // `App`), so it paints nothing — and emits nothing, rather than
            // emitting a blank sprite, because a subtree that is not this
            // element's is not this element's to hold a slab slot for.
            return;
        }

        let placement = self
            .engine
            .borrow_mut()
            .tile_for(self.source, self.frame_index);
        let natural = self.natural_size();
        let bounds = [
            context.bounds.x,
            context.bounds.y,
            context.bounds.width,
            context.bounds.height,
        ];
        // Object-fit needs the image's own size to have a ratio to fit. With no
        // decoded frame there is no ratio, so the sprite takes the whole box —
        // which is what it will occupy once its pixels arrive under `Fill`, and
        // which draws nothing either way because its tile is `NONE`.
        let drawn = match natural {
            Some(size) => self.style.object_fit.fit(bounds, size),
            None => bounds,
        };

        let transformed_size = [
            drawn[2] * transformation.scale[0],
            drawn[3] * transformation.scale[1],
        ];
        let origin = [
            drawn[0] + (drawn[2] - transformed_size[0]) * 0.5 + transformation.translation[0],
            drawn[1] + (drawn[3] - transformed_size[1]) * 0.5 + transformation.translation[1],
        ];
        emission.poly_sprite(PolySprite {
            origin,
            size: transformed_size,
            atlas_origin: placement
                .map(|tile| tile.atlas_origin)
                .unwrap_or([0.0, 0.0]),
            atlas_size: placement.map(|tile| tile.atlas_size).unwrap_or([0.0, 0.0]),
            corner_radius: self.style.corner_radius,
            opacity: self.style.opacity,
            grayscale: self.style.grayscale,
            // An undecoded frame keeps its slot and draws nothing, exactly as a
            // whitespace glyph does. That is what makes the frame it finally
            // decodes in a value update rather than an insert.
            atlas_tile: placement.map(|tile| tile.tile).unwrap_or(AtlasTileId::NONE),
        });
    }
}

impl Element for Img {
    fn into_description(self) -> Description {
        self.describe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpui_core::reconcile::description::ElementId;
    use wgpui_core::reconcile::diff_key::compare_by_equality;
    use wgpui_core::reconcile::instance::InstanceKey;
    use wgpui_core::reconcile::plan::{FramePlan, NodeOutcome, PlannedNode, RebuildReason};
    use wgpui_core::reconcile::reconciler::{ReconcileError, Reconciler};
    use wgpui_layout::taffy_tree::{FlexDirection, LayoutTree};

    struct Panel;

    #[derive(PartialEq, Debug)]
    struct PanelKey(u32);

    impl ReconcileKey for PanelKey {
        fn compare(&self, previous: &dyn ReconcileKey) -> Invalidation {
            compare_by_equality(self, previous, Invalidation::DISPLAY)
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    const IMG_SLOT: [ElementId; 2] = [ElementId::Slot(0), ElementId::Slot(0)];
    const PANEL_SLOT: [ElementId; 2] = [ElementId::Slot(0), ElementId::Slot(1)];

    /// A tile source that hands out one tile per distinct key and counts what it
    /// was asked for.
    ///
    /// A substitute for the real atlas, not a mock of it, on
    /// `wgpui_text::patch`'s own principle: this module's tests measure what an
    /// `Img` emits, and the allocator's packing is tested where it lives, in
    /// `wgpui-wgpu`'s `render/atlas.rs`. The two meet for real in
    /// `wgpui-wgpu/tests/image_sprite_draw.rs`.
    #[derive(Default)]
    struct FakeTiles {
        tiles: std::collections::HashMap<ImageRasterKey, wgpui_core::scene::atlas::ImageTile>,
        /// Shared with the test rather than read back off the boxed source, so
        /// counting costs the shipped `ImageEngine` no test-only accessor.
        decodes: Rc<std::cell::Cell<usize>>,
        requests: Rc<std::cell::Cell<usize>>,
    }

    impl ImageTileSource for FakeTiles {
        fn tile_for(
            &mut self,
            key: ImageRasterKey,
            decode: &mut dyn FnMut(
                ImageRasterKey,
            )
                -> Option<wgpui_core::scene::atlas::RasterizedImage>,
        ) -> Option<wgpui_core::scene::atlas::ImageTile> {
            self.requests.set(self.requests.get() + 1);
            if let Some(tile) = self.tiles.get(&key) {
                return Some(*tile);
            }
            self.decodes.set(self.decodes.get() + 1);
            let raster = decode(key)?;
            let next = self.tiles.len() as u32;
            let tile = wgpui_core::scene::atlas::ImageTile {
                tile: AtlasTileId::new(0, next).expect("test tiles stay in range"),
                atlas_origin: [next as f32 * 64.0, 0.0],
                atlas_size: [raster.size[0] as f32, raster.size[1] as f32],
            };
            self.tiles.insert(key, tile);
            Some(tile)
        }
    }

    /// How many times the substitute tile source was asked, and how many of
    /// those it had to decode for.
    #[derive(Clone, Default)]
    struct TileCounters {
        requests: Rc<std::cell::Cell<usize>>,
        decodes: Rc<std::cell::Cell<usize>>,
    }

    /// An engine holding one decoded source of the given size.
    fn engine_with(width: u32, height: u32) -> (SharedImageEngine, ImageSourceId) {
        let (engine, source, _) = engine_counting(width, height);
        (engine, source)
    }

    /// [`engine_with`], with the tile source's counters handed back.
    fn engine_counting(
        width: u32,
        height: u32,
    ) -> (SharedImageEngine, ImageSourceId, TileCounters) {
        use crate::image_cache::{DecodedFrame, DecodedImage};

        let counters = TileCounters::default();
        let mut cache = ImageCache::new();
        let frame = DecodedFrame {
            size: [width, height],
            texels: vec![0x80; (width * height * 4) as usize],
            delay: std::time::Duration::ZERO,
        };
        // Built rather than decoded: this module's tests are about what an `Img`
        // emits, and running a PNG decoder to get a size would make every one of
        // them depend on the decoder too.
        let source = cache
            .hold(DecodedImage::from_frames(vec![frame]).expect("one frame is a valid image"))
            .expect("holding a decoded image must succeed");
        let engine = Rc::new(RefCell::new(ImageEngine::new(
            cache,
            Box::new(FakeTiles {
                requests: Rc::clone(&counters.requests),
                decodes: Rc::clone(&counters.decodes),
                ..FakeTiles::default()
            }),
        )));
        (engine, source, counters)
    }

    fn animated_engine() -> (SharedImageEngine, ImageSourceId, TileCounters) {
        use crate::image_cache::{DecodedFrame, DecodedImage};

        let counters = TileCounters::default();
        let mut cache = ImageCache::new();
        let frames = (0..2)
            .map(|frame_index| DecodedFrame {
                size: [16, 16],
                texels: vec![frame_index as u8; 16 * 16 * 4],
                delay: std::time::Duration::from_millis(10),
            })
            .collect();
        let source = cache
            .hold(DecodedImage::from_frames(frames).expect("two frames are a valid image"))
            .expect("holding an animated image must succeed");
        let engine = Rc::new(RefCell::new(ImageEngine::new(
            cache,
            Box::new(FakeTiles {
                requests: Rc::clone(&counters.requests),
                decodes: Rc::clone(&counters.decodes),
                ..FakeTiles::default()
            }),
        )));
        (engine, source, counters)
    }

    /// An engine holding nothing, for the still-loading case.
    fn empty_engine() -> SharedImageEngine {
        Rc::new(RefCell::new(ImageEngine::new(
            ImageCache::new(),
            Box::new(FakeTiles::default()),
        )))
    }

    fn base() -> Img {
        let (engine, source) = engine_with(64, 32);
        Img::new(source, engine).size(48.0, 48.0)
    }

    /// `base()`'s shape, but sharing one engine so two `Img`s can be compared
    /// without the engine being a difference between them.
    fn sized(engine: &SharedImageEngine, source: ImageSourceId) -> Img {
        Img::new(source, Rc::clone(engine)).size(48.0, 48.0)
    }

    /// The one sprite an `Img` emits at the given bounds.
    fn emitted(image: &Img, bounds: [f32; 4]) -> Option<PolySprite> {
        use wgpui_core::scene::layer::{BoundaryId, LayerId, LayerKey};
        let mut emission = Emission::new();
        image.emit_into(
            &EmitContext {
                bounds: wgpui_layout::taffy_tree::LayoutRect {
                    x: bounds[0],
                    y: bounds[1],
                    width: bounds[2],
                    height: bounds[3],
                },
                layer: LayerId::from_key(LayerKey::untiled(BoundaryId::ROOT)),
                boundary: BoundaryId::ROOT,
                clip: None,
            },
            &mut emission,
        );
        emission.poly_sprites().first().copied()
    }

    /// An avatar beside a plain reconciled sibling — SFD §3's own list-row
    /// shape, minus the text half, which `styled_text.rs` covers.
    fn tree(image: Img) -> Description {
        Description::new::<Panel>()
            .diff_key(PanelKey(0))
            .style(LayoutStyle {
                flex_direction: FlexDirection::Column,
                ..LayoutStyle::default()
            })
            .child(image.describe())
            .child(
                Description::new::<Panel>()
                    .diff_key(PanelKey(0))
                    .style(LayoutStyle {
                        size: LayoutSize {
                            width: Dimension::length(120.0),
                            height: Dimension::length(40.0),
                        },
                        flex_shrink: 0.0,
                        ..LayoutStyle::default()
                    }),
            )
    }

    fn node_at<'plan>(plan: &'plan FramePlan, path: &[ElementId]) -> Option<&'plan PlannedNode> {
        plan.node_for_instance(InstanceKey::from_path(path))
    }

    #[test]
    fn an_unchanged_image_is_reused_like_any_other_element() -> Result<(), ReconcileError> {
        let mut reconciler = Reconciler::new();
        let mut layout = LayoutTree::new();

        let first = reconciler.reconcile(tree(base()), &mut layout)?;
        let before = node_at(&first, &IMG_SLOT).copied();
        assert_eq!(
            before.map(|node| node.outcome),
            Some(NodeOutcome::Rebuilt(RebuildReason::NewInstance))
        );

        let second = reconciler.reconcile(tree(base()), &mut layout)?;
        let after = node_at(&second, &IMG_SLOT).copied();
        assert_eq!(
            after.map(|node| node.outcome),
            Some(NodeOutcome::Reused),
            "an unchanged image must not rebuild — this is the gap §6.2 names"
        );
        assert_eq!(
            after.map(|node| node.skipped_prepaint_and_paint()),
            node_at(&second, &PANEL_SLOT)
                .copied()
                .map(|node| node.skipped_prepaint_and_paint()),
            "the image and the ordinary element must get the same treatment"
        );
        assert_eq!(
            after.map(|node| node.layout_node),
            before.map(|node| node.layout_node)
        );
        assert!(second.fully_reused());
        Ok(())
    }

    /// The two transitions the legacy comment names as the reason `Img` had no
    /// key at all — a GIF advancing a frame, and a pending load resolving —
    /// are the two this key exists to report.
    #[test]
    fn the_state_the_legacy_key_could_not_reach_is_exactly_what_this_key_reports() {
        let still = base();
        let next_frame = base().frame_index(1);
        let loading = base().load_state(ImageLoadState::Loading);

        assert_eq!(
            still.diff_key().compare(&still.diff_key()),
            Invalidation::empty()
        );
        assert_eq!(
            next_frame.diff_key().compare(&still.diff_key()),
            Invalidation::DISPLAY,
            "an animated source advancing a frame must repaint"
        );
        assert_eq!(
            loading.diff_key().compare(&still.diff_key()),
            Invalidation::LAYOUT.union(Invalidation::DISPLAY),
            "a load transition swaps a replacement subtree, so it is a layout change too"
        );
    }

    #[test]
    fn each_field_reports_exactly_the_axes_it_affects() -> Result<(), ReconcileError> {
        let cases: [(&str, Img, Invalidation); 5] = [
            (
                "a different source",
                // A second source in a *fresh* engine, so the comparison is
                // about the id and not about which engine happens to hold it.
                {
                    let (engine, source) = engine_with(64, 32);
                    sized(&engine, ImageSourceId::from_raw(source.as_raw() + 1))
                },
                Invalidation::DISPLAY,
            ),
            (
                "a new animation frame",
                base().frame_index(3),
                Invalidation::DISPLAY,
            ),
            (
                "a load transition",
                base().load_state(ImageLoadState::Failed),
                Invalidation::LAYOUT.union(Invalidation::DISPLAY),
            ),
            (
                "a resize",
                base().size(48.0, 49.0),
                Invalidation::LAYOUT.union(Invalidation::DISPLAY),
            ),
            (
                "a style change",
                base().style(ImageStyle {
                    grayscale: true,
                    object_fit: ObjectFit::Contain,
                    opacity: 1.0,
                    corner_radius: 0.0,
                }),
                Invalidation::DISPLAY,
            ),
        ];

        for (what, changed, expected) in cases {
            assert_ne!(
                changed.diff_key(),
                base().diff_key(),
                "{what} must actually differ from the control"
            );

            let mut reconciler = Reconciler::new();
            let mut layout = LayoutTree::new();
            reconciler.reconcile(tree(base()), &mut layout)?;
            let plan = reconciler.reconcile(tree(changed), &mut layout)?;

            let image = node_at(&plan, &IMG_SLOT).copied();
            assert_eq!(
                image.map(|node| node.outcome),
                Some(NodeOutcome::Rebuilt(RebuildReason::KeyChanged)),
                "{what} must rebuild the image"
            );
            assert_eq!(
                image.map(|node| node.invalidation),
                Some(expected),
                "{what} must report exactly the axes it affects"
            );
            assert_eq!(
                node_at(&plan, &PANEL_SLOT)
                    .copied()
                    .map(|node| node.skipped_prepaint_and_paint()),
                Some(true),
                "{what} must not disturb the sibling"
            );
        }
        Ok(())
    }

    #[test]
    fn object_fit_is_a_display_change_and_never_a_layout_one() {
        let contained = base();
        let covered = base().style(ImageStyle {
            object_fit: ObjectFit::Cover,
            ..contained.style
        });
        assert_eq!(
            covered.diff_key().compare(&contained.diff_key()),
            Invalidation::DISPLAY,
            "object-fit decides where content sits inside a box already decided by layout"
        );
    }

    #[test]
    fn a_key_compared_against_a_different_element_type_is_a_full_invalidation() {
        assert_eq!(base().diff_key().compare(&PanelKey(0)), Invalidation::all());
    }

    // ---- Phase 6.2: the sprite an image actually emits -------------------

    #[test]
    fn a_ready_image_emits_one_sprite_carrying_its_tile() {
        let (engine, source) = engine_with(64, 32);
        let image = sized(&engine, source);
        let sprite = emitted(&image, [10.0, 20.0, 48.0, 48.0]).expect("one sprite");

        assert_eq!(sprite.atlas_size, [64.0, 32.0], "the tile's own extent");
        assert!(
            !sprite.atlas_tile.is_none(),
            "a decoded, ready image must reference a real tile"
        );
        assert_eq!(sprite.opacity, 1.0);
        assert!(!sprite.grayscale);
    }

    #[test]
    fn an_undecoded_image_emits_a_sprite_that_holds_its_slot_and_draws_nothing() {
        let image = Img::new(ImageSourceId::from_raw(99), empty_engine()).size(48.0, 48.0);
        let sprite = emitted(&image, [10.0, 20.0, 48.0, 48.0]).expect("a slot is still emitted");
        assert!(
            sprite.atlas_tile.is_none(),
            "no pixels means no tile, exactly as a whitespace glyph carries none"
        );
        assert_eq!(sprite.atlas_size, [0.0, 0.0]);
        assert_eq!(
            [sprite.origin, sprite.size],
            [[10.0, 20.0], [48.0, 48.0]],
            "an image with no ratio to fit takes the whole box"
        );
    }

    #[test]
    fn a_loading_or_failed_image_emits_nothing_at_all() {
        let (engine, source) = engine_with(64, 32);
        for state in [ImageLoadState::Loading, ImageLoadState::Failed] {
            let image = sized(&engine, source).load_state(state);
            assert_eq!(
                emitted(&image, [0.0, 0.0, 48.0, 48.0]),
                None,
                "{state:?} paints a replacement subtree, and a subtree that is \
                 not this element's is not this element's to hold a slot for"
            );
        }
    }

    #[test]
    fn object_fit_places_the_sprite_the_way_the_legacy_expression_does() {
        let (engine, source) = engine_with(100, 50);
        // A 2:1 image in a 100x100 box.
        let bounds = [0.0, 0.0, 100.0, 100.0];
        let fit = |mode: ObjectFit| {
            let image = Img::new(source, Rc::clone(&engine)).style(ImageStyle {
                object_fit: mode,
                opacity: 1.0,
                ..ImageStyle::default()
            });
            emitted(&image, bounds).map(|sprite| [sprite.origin, sprite.size])
        };

        assert_eq!(
            fit(ObjectFit::Fill),
            Some([[0.0, 0.0], [100.0, 100.0]]),
            "fill ignores the ratio"
        );
        assert_eq!(
            fit(ObjectFit::Contain),
            Some([[0.0, 25.0], [100.0, 50.0]]),
            "contain fits the width and centres vertically"
        );
        assert_eq!(
            fit(ObjectFit::Cover),
            Some([[-50.0, 0.0], [200.0, 100.0]]),
            "cover fills the height and overflows horizontally, centred"
        );
        assert_eq!(
            fit(ObjectFit::None),
            Some([[0.0, 0.0], [100.0, 50.0]]),
            "the legacy `None` arm is natural size at the box's *origin*, not centred"
        );
        assert_eq!(
            fit(ObjectFit::ScaleDown),
            Some([[0.0, 25.0], [100.0, 50.0]]),
            "a 100x50 image in a 100x100 box is not larger in either dimension, \
             so scale-down centres it at natural size — which here is contain's \
             answer, because the width already matches"
        );
    }

    #[test]
    fn scale_down_only_shrinks() {
        // Small image, big box: scale-down must *not* enlarge it.
        assert_eq!(
            ObjectFit::ScaleDown.fit([0.0, 0.0, 100.0, 100.0], [10, 10]),
            [45.0, 45.0, 10.0, 10.0]
        );
        // Big image, small box: it falls back to contain.
        assert_eq!(
            ObjectFit::ScaleDown.fit([0.0, 0.0, 50.0, 50.0], [200, 100]),
            ObjectFit::Contain.fit([0.0, 0.0, 50.0, 50.0], [200, 100])
        );
    }

    #[test]
    fn a_zero_area_image_or_box_answers_with_the_box_rather_than_a_nan() {
        for fit in [
            ObjectFit::Fill,
            ObjectFit::Contain,
            ObjectFit::Cover,
            ObjectFit::None,
            ObjectFit::ScaleDown,
        ] {
            let bounds = [1.0, 2.0, 30.0, 40.0];
            assert_eq!(fit.fit(bounds, [0, 10]), bounds, "{fit:?} with no width");
            assert_eq!(fit.fit(bounds, [10, 0]), bounds, "{fit:?} with no height");
            assert_eq!(
                fit.fit([1.0, 2.0, 0.0, 40.0], [10, 10]),
                [1.0, 2.0, 0.0, 40.0],
                "{fit:?} in an empty box"
            );
        }
    }

    /// The concrete payoff of having real dimensions: an unsized image asks
    /// layout for its own size instead of vanishing.
    #[test]
    fn an_unsized_image_asks_layout_for_its_natural_size() {
        let (engine, source) = engine_with(64, 32);
        let unsized_image = Img::new(source, Rc::clone(&engine));
        assert_eq!(unsized_image.natural_size(), Some([64, 32]));
        assert_eq!(unsized_image.layout_size(), [64.0, 32.0]);

        // An explicit request still wins.
        assert_eq!(
            Img::new(source, Rc::clone(&engine))
                .size(10.0, 10.0)
                .layout_size(),
            [10.0, 10.0]
        );

        // And a source that has not decoded has nothing to ask for.
        let pending = Img::new(ImageSourceId::from_raw(99), empty_engine());
        assert_eq!(pending.natural_size(), None);
        assert_eq!(pending.layout_size(), [0.0, 0.0]);
    }

    #[test]
    fn a_resident_frame_is_never_decoded_twice_however_many_rows_show_it() {
        let (engine, source, counters) = engine_counting(16, 16);
        for _ in 0..40 {
            let image = sized(&engine, source);
            assert!(emitted(&image, [0.0, 0.0, 48.0, 48.0]).is_some());
        }
        assert_eq!(counters.requests.get(), 40, "every row asked");
        assert_eq!(
            counters.decodes.get(),
            1,
            "and exactly one of them paid for a decode — forty avatars sharing \
             one source is the case this cache exists for, and a claim about \
             work *not* happening is only checkable by counting it when it does"
        );
    }

    #[test]
    fn automatic_animation_marks_the_description_active() {
        let (engine, source, _) = animated_engine();
        let description = Img::new(source, engine).describe();

        assert!(description.has_active_animation());
    }

    #[test]
    fn a_non_ready_animation_does_not_keep_the_frame_loop_alive() {
        let (engine, source, _) = animated_engine();
        let description = Img::new(source, engine)
            .load_state(ImageLoadState::Loading)
            .describe();

        assert!(!description.has_active_animation());
    }

    #[test]
    fn equivalent_looped_frame_indices_share_atlas_residency() {
        let (engine, source, counters) = animated_engine();
        let first = Img::new(source, Rc::clone(&engine)).frame_index(0);
        let wrapped = Img::new(source, Rc::clone(&engine)).frame_index(2);

        assert!(emitted(&first, [0.0, 0.0, 16.0, 16.0]).is_some());
        assert!(emitted(&wrapped, [0.0, 0.0, 16.0, 16.0]).is_some());
        assert_eq!(counters.requests.get(), 2);
        assert_eq!(
            counters.decodes.get(),
            1,
            "frame indices loop through the decoded image and must not duplicate atlas tiles"
        );
    }

    #[test]
    fn resource_builder_retains_stable_identity_and_sizing_metadata() {
        let first = img("missing/image.png")
            .size(wgpui_core::geometry::Pixels(24.0))
            .id("avatar")
            .into_description();
        let second = img("missing/image.png")
            .size(wgpui_core::geometry::Pixels(24.0))
            .into_description();

        assert_eq!(first.element_id(), Some(&ElementId::from("avatar")));
        let first_key = first
            .key()
            .and_then(|key| key.as_any().downcast_ref::<ImgKey>())
            .expect("resource images expose their retained image key");
        let second_key = second
            .key()
            .and_then(|key| key.as_any().downcast_ref::<ImgKey>())
            .expect("resource images expose their retained image key");
        assert_eq!(first_key.source, second_key.source);
        assert_eq!(first_key.requested_size, [24.0, 24.0]);
        assert_eq!(first_key.load_state, ImageLoadState::Loading);
    }
}
