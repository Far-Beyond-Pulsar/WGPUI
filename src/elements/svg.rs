use std::{any::Any, fs, path::Path, sync::Arc};

use crate::{
    App, Asset, Bounds, Element, GlobalElementId, Hitbox, InspectorElementId, InteractiveElement,
    Interactivity, IntoElement, Invalidation, LayoutId, Pixels, Point, Radians, ReconcileKey,
    SharedString, Size, StyleRefinement, Styled, TransformationMatrix, Window,
    elements::div::classify_style_change, geometry::Negate as _, point, px, radians, size,
};
use crate::util::ResultExt;

/// An SVG element.
pub struct Svg {
    interactivity: Interactivity,
    transformation: Option<Transformation>,
    path: Option<SharedString>,
    external_path: Option<SharedString>,
}

/// Create a new SVG element.
#[track_caller]
pub fn svg() -> Svg {
    Svg {
        interactivity: Interactivity::new(),
        transformation: None,
        path: None,
        external_path: None,
    }
}

impl Svg {
    /// Set the path to the SVG file for this element.
    pub fn path(mut self, path: impl Into<SharedString>) -> Self {
        self.path = Some(path.into());
        self
    }

    /// Set the path to the SVG file for this element.
    pub fn external_path(mut self, path: impl Into<SharedString>) -> Self {
        self.external_path = Some(path.into());
        self
    }

    /// Transform the SVG element with the given transformation.
    /// Note that this won't effect the hitbox or layout of the element, only the rendering.
    pub fn with_transformation(mut self, transformation: Transformation) -> Self {
        self.transformation = Some(transformation);
        self
    }
}

impl Element for Svg {
    type RequestLayoutState = ();
    type PrepaintState = Option<Hitbox>;

    fn id(&self) -> Option<crate::ElementId> {
        self.interactivity.element_id.clone()
    }

    fn source_location(&self) -> Option<&'static std::panic::Location<'static>> {
        self.interactivity.source_location()
    }

    fn request_layout(
        &mut self,
        global_id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let layout_id = self.interactivity.request_layout(
            global_id,
            inspector_id,
            window,
            cx,
            |style, window, cx| window.request_layout(style, None, cx),
        );
        (layout_id, ())
    }

    fn prepaint(
        &mut self,
        global_id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<Hitbox> {
        self.interactivity.prepaint(
            global_id,
            inspector_id,
            bounds,
            bounds.size,
            window,
            cx,
            |_, _, hitbox, _, _| hitbox,
        )
    }

    fn paint(
        &mut self,
        global_id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        hitbox: &mut Option<Hitbox>,
        window: &mut Window,
        cx: &mut App,
    ) where
        Self: Sized,
    {
        self.interactivity.paint(
            global_id,
            inspector_id,
            bounds,
            hitbox.as_ref(),
            window,
            cx,
            |style, window, cx| {
                if let Some((path, color)) = self.path.as_ref().zip(style.text.color) {
                    let transformation = self
                        .transformation
                        .as_ref()
                        .map(|transformation| {
                            transformation.into_matrix(bounds.center(), window.scale_factor())
                        })
                        .unwrap_or_default();

                    window
                        .paint_svg(bounds, path.clone(), None, transformation, color.to_hsla(), cx)
                        .log_err();
                } else if let Some((path, color)) =
                    self.external_path.as_ref().zip(style.text.color)
                {
                    let Some(bytes) = window
                        .use_asset::<SvgAsset>(path, cx)
                        .and_then(|asset| asset.log_err())
                    else {
                        return;
                    };

                    let transformation = self
                        .transformation
                        .as_ref()
                        .map(|transformation| {
                            transformation.into_matrix(bounds.center(), window.scale_factor())
                        })
                        .unwrap_or_default();

                    window
                        .paint_svg(
                            bounds,
                            path.clone(),
                            Some(&bytes),
                            transformation,
                            color.to_hsla(),
                            cx,
                        )
                        .log_err();
                }
            },
        )
    }

    fn diff_key(&self, _window: &Window) -> Option<Box<dyn ReconcileKey>> {
        // Same reasoning as `Div::diff_key`: resolving hover/focus/active
        // pseudo-state correctly requires reproducing
        // `Interactivity::compute_style_internal`'s merge order, which this
        // method deliberately does not attempt to duplicate.
        if self.interactivity.hover_style.is_some()
            || self.interactivity.group_hover_style.is_some()
            || self.interactivity.active_style.is_some()
            || self.interactivity.group_active_style.is_some()
            || self.interactivity.focus_style.is_some()
            || self.interactivity.in_focus_style.is_some()
            || self.interactivity.focus_visible_style.is_some()
            || !self.interactivity.drag_over_styles.is_empty()
            || !self.interactivity.group_drag_over_styles.is_empty()
        {
            return None;
        }

        // `external_path` resolves through `window.use_asset`, and `paint`
        // paints nothing at all until that resolves (see the `let Some(bytes)
        // = ... else { return; }` above). That readiness transition is
        // invisible to a `&self`-only `diff_key`, for the same reason `Img`
        // does not implement this method at all (see its module for the
        // fuller explanation) — reconciling here could freeze an
        // asset-in-flight SVG in its blank state indefinitely.
        if self.external_path.is_some() {
            return None;
        }

        Some(Box::new(SvgDiffKey {
            path: self.path.clone(),
            transformation: self.transformation,
            style: (*self.interactivity.base_style).clone(),
        }))
    }
}

/// [`Svg`]'s [`ReconcileKey`] (#92). Scoped to the `path` (not `external_path`
/// — see `diff_key`'s doc comment) case, where nothing about what gets
/// painted depends on state this key can't see.
struct SvgDiffKey {
    path: Option<SharedString>,
    transformation: Option<Transformation>,
    style: StyleRefinement,
}

impl ReconcileKey for SvgDiffKey {
    fn compare(&self, previous: &dyn ReconcileKey) -> Invalidation {
        let Some(previous) = previous.as_any().downcast_ref::<SvgDiffKey>() else {
            return Invalidation::all();
        };

        let mut axes = classify_style_change(&self.style, &previous.style);
        if self.path != previous.path || self.transformation != previous.transformation {
            axes |= Invalidation::DISPLAY;
        }
        axes
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl IntoElement for Svg {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Styled for Svg {
    fn style(&mut self) -> &mut StyleRefinement {
        &mut self.interactivity.base_style
    }
}

impl InteractiveElement for Svg {
    fn interactivity(&mut self) -> &mut Interactivity {
        &mut self.interactivity
    }
}

/// A transformation to apply to an SVG element.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transformation {
    scale: Size<f32>,
    translate: Point<Pixels>,
    rotate: Radians,
}

impl Default for Transformation {
    fn default() -> Self {
        Self {
            scale: size(1.0, 1.0),
            translate: point(px(0.0), px(0.0)),
            rotate: radians(0.0),
        }
    }
}

impl Transformation {
    /// Create a new Transformation with the specified scale along each axis.
    pub fn scale(scale: Size<f32>) -> Self {
        Self {
            scale,
            translate: point(px(0.0), px(0.0)),
            rotate: radians(0.0),
        }
    }

    /// Create a new Transformation with the specified translation.
    pub fn translate(translate: Point<Pixels>) -> Self {
        Self {
            scale: size(1.0, 1.0),
            translate,
            rotate: radians(0.0),
        }
    }

    /// Create a new Transformation with the specified rotation in radians.
    pub fn rotate(rotate: impl Into<Radians>) -> Self {
        let rotate = rotate.into();
        Self {
            scale: size(1.0, 1.0),
            translate: point(px(0.0), px(0.0)),
            rotate,
        }
    }

    /// Update the scaling factor of this transformation.
    pub fn with_scaling(mut self, scale: Size<f32>) -> Self {
        self.scale = scale;
        self
    }

    /// Update the translation value of this transformation.
    pub fn with_translation(mut self, translate: Point<Pixels>) -> Self {
        self.translate = translate;
        self
    }

    /// Update the rotation angle of this transformation.
    pub fn with_rotation(mut self, rotate: impl Into<Radians>) -> Self {
        self.rotate = rotate.into();
        self
    }

    fn into_matrix(self, center: Point<Pixels>, scale_factor: f32) -> TransformationMatrix {
        //Note: if you read this as a sequence of matrix multiplications, start from the bottom
        TransformationMatrix::unit()
            .translate(center.scale(scale_factor) + self.translate.scale(scale_factor))
            .rotate(self.rotate)
            .scale(self.scale)
            .translate(center.scale(scale_factor).negate())
    }
}

enum SvgAsset {}

impl Asset for SvgAsset {
    type Source = SharedString;
    type Output = Result<Arc<[u8]>, Arc<std::io::Error>>;

    fn load(
        source: Self::Source,
        _cx: &mut App,
    ) -> impl Future<Output = Self::Output> + Send + 'static {
        async move {
            let bytes = fs::read(Path::new(source.as_ref())).map_err(|e| Arc::new(e))?;
            let bytes = Arc::from(bytes);
            Ok(bytes)
        }
    }
}
