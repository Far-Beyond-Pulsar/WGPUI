//! Retained custom drawing for native WGPUI elements.
//!
//! A canvas callback is evaluated while the retained tree is emitted. It only
//! produces scene primitives; it never receives a window or a GPU encoder.
//! This keeps custom drawing on the same clipping, invalidation, and untiled
//! fallback path as built-in widgets.

use anyhow::Error;
use lyon::geom::Angle;
use lyon::math::{Point as LyonPoint, Transform, point as lyon_point, vector};
use lyon::path::traits::SvgPathBuilder;
use lyon::path::{ArcFlags, Polygon};
use lyon::tessellation::{
    BuffersBuilder, FillOptions, FillTessellator, FillVertex, StrokeOptions, StrokeTessellator,
    StrokeVertex, VertexBuffers,
};
use wgpui_core::color::{Background, Hsla, Rgba};
use wgpui_core::element::Element;
use wgpui_core::geometry::{Bounds, Pixels, Point};
use wgpui_core::patch::emit::{Emission, EmitContext};
use wgpui_core::patch::primitive::{BackdropFilter, Material, Path, Quad};
use wgpui_core::reconcile::description::{Description, ElementId};
use wgpui_layout::taffy_tree::LayoutRect;

use crate::div::interactivity::style::DivStyle;
use crate::styled::Styled;

/// Convert a public colour into the straight-alpha representation used by the
/// retained scene.
pub trait IntoCanvasColor {
    fn into_canvas_color(self) -> [f32; 4];
}

/// Convert a public retained material into the path payload used by a canvas.
pub trait IntoCanvasMaterial {
    fn into_canvas_material(self) -> ([f32; 4], Material);
}

/// Convert a native rectangle into canvas coordinates.
pub trait IntoCanvasBounds {
    fn into_canvas_bounds(self) -> ([f32; 2], [f32; 2]);
}

impl IntoCanvasBounds for Bounds<Pixels> {
    fn into_canvas_bounds(self) -> ([f32; 2], [f32; 2]) {
        (
            [self.origin.x.value(), self.origin.y.value()],
            [self.size.width.value(), self.size.height.value()],
        )
    }
}

impl IntoCanvasBounds for LayoutRect {
    fn into_canvas_bounds(self) -> ([f32; 2], [f32; 2]) {
        ([self.x, self.y], [self.width, self.height])
    }
}

impl IntoCanvasColor for [f32; 4] {
    fn into_canvas_color(self) -> [f32; 4] {
        self
    }
}

impl IntoCanvasColor for Rgba {
    fn into_canvas_color(self) -> [f32; 4] {
        self.into()
    }
}

impl IntoCanvasColor for Hsla {
    fn into_canvas_color(self) -> [f32; 4] {
        self.into()
    }
}

impl<T: IntoCanvasColor> IntoCanvasMaterial for T {
    fn into_canvas_material(self) -> ([f32; 4], Material) {
        (self.into_canvas_color(), Material::Solid)
    }
}

impl IntoCanvasMaterial for Material {
    fn into_canvas_material(self) -> ([f32; 4], Material) {
        ([0.0; 4], self)
    }
}

impl IntoCanvasMaterial for Background {
    fn into_canvas_material(self) -> ([f32; 4], Material) {
        match self {
            Background::Solid(color) => (color.into(), Material::Solid),
            Background::LinearGradient { angle, colors, .. } => (
                [0.0; 4],
                Material::Linear {
                    direction: [angle.to_radians().cos(), angle.to_radians().sin()],
                    colors: [colors[0].color.into(), colors[1].color.into()],
                },
            ),
            Background::RadialGradient {
                center,
                radius,
                colors,
                ..
            } => (
                [0.0; 4],
                Material::Radial {
                    center,
                    radius,
                    colors: [colors[0].color.into(), colors[1].color.into()],
                },
            ),
            Background::PatternSlash {
                color,
                width,
                interval,
            } => (
                [0.0; 4],
                Material::Slash {
                    color: color.into(),
                    width,
                    interval,
                },
            ),
        }
    }
}

/// Build a solid retained rectangle for a canvas callback.
pub fn fill(bounds: impl IntoCanvasBounds, color: impl IntoCanvasColor) -> Quad {
    let (origin, size) = bounds.into_canvas_bounds();
    let background = color.into_canvas_color();
    let mut border_color = background;
    border_color[3] = 0.0;
    Quad {
        origin,
        size,
        background,
        border_color,
        corner_radii: [0.0; 4],
        border_widths: [0.0; 4],
        material: Material::Solid,
    }
}

/// Context passed to a retained canvas callback.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CanvasContext {
    bounds: LayoutRect,
    clip: Option<LayoutRect>,
}

impl CanvasContext {
    fn new(context: &EmitContext) -> Self {
        Self {
            bounds: context.bounds,
            clip: context.clip,
        }
    }

    /// The canvas element's absolute layout rectangle.
    pub fn bounds(&self) -> LayoutRect {
        self.bounds
    }

    /// The inherited clip, before the emitter clips this callback's output.
    pub fn clip(&self) -> Option<LayoutRect> {
        self.clip
    }

    /// Make a rectangle in the canvas coordinate space.
    pub fn fill(&self, bounds: impl IntoCanvasBounds, color: impl IntoCanvasColor) -> Quad {
        fill(bounds, color)
    }

    /// Apply the canvas bounds as a conservative per-path mask.
    pub fn path(&self, path: Path, material: impl IntoCanvasMaterial) -> Path {
        let (color, material) = material.into_canvas_material();
        path.with_color(color)
            .with_material(material)
            .with_clip([self.bounds.x, self.bounds.y], [self.bounds.width, self.bounds.height])
    }

    /// Make a backdrop filter covering the canvas bounds.
    pub fn backdrop_blur(&self, radius: Pixels) -> BackdropFilter {
        BackdropFilter {
            origin: [self.bounds.x, self.bounds.y],
            size: [self.bounds.width, self.bounds.height],
            clip_origin: [self.bounds.x, self.bounds.y],
            clip_size: [self.bounds.width, self.bounds.height],
            corner_radii: [0.0; 4],
            blur_radius: radius.value().max(0.0),
            opacity: 1.0,
        }
    }
}

/// A retained element whose callback contributes paths, quads, and filters.
type CanvasPainter = dyn Fn(&CanvasContext, &mut Emission);

pub struct Canvas {
    style: DivStyle,
    element_id: Option<ElementId>,
    painter: Box<CanvasPainter>,
}

impl Canvas {
    /// Construct an empty-styled canvas.
    pub fn new(painter: impl Fn(&CanvasContext, &mut Emission) + 'static) -> Self {
        Self {
            style: DivStyle::default(),
            element_id: None,
            painter: Box::new(painter),
        }
    }

    /// Assign a stable element identity.
    pub fn id(mut self, element_id: impl Into<ElementId>) -> Self {
        self.element_id = Some(element_id.into());
        self
    }
}

/// Construct a retained custom-drawing element.
pub fn canvas(painter: impl Fn(&CanvasContext, &mut Emission) + 'static) -> Canvas {
    Canvas::new(painter)
}

impl Element for Canvas {
    fn into_description(self) -> Description {
        let Canvas {
            style,
            element_id,
            painter,
        } = self;
        let mut description = Description::new::<Canvas>().style(style.layout.clone());
        if let Some(element_id) = element_id {
            description = description.id(element_id);
        }
        description.emit(move |context: &EmitContext, emission: &mut Emission| {
            style.paint(context.bounds, emission);
            painter(&CanvasContext::new(context), emission);
        })
    }
}

impl Styled for Canvas {
    fn style(&mut self) -> &mut DivStyle {
        &mut self.style
    }
}

/// The tessellation mode used by [`PathBuilder`].
pub enum PathStyle {
    /// Tessellate the path as a stroke.
    Stroke(StrokeOptions),
    /// Tessellate the path as a fill.
    Fill(FillOptions),
}

/// A Lyon-backed builder for retained triangle-list paths.
pub struct PathBuilder {
    raw: lyon::path::builder::WithSvg<lyon::path::BuilderImpl>,
    transform: Option<Transform>,
    style: PathStyle,
    dash_array: Option<Vec<Pixels>>,
}

impl Default for PathBuilder {
    fn default() -> Self {
        Self {
            raw: lyon::path::Path::builder().with_svg(),
            transform: None,
            style: PathStyle::Fill(FillOptions::default()),
            dash_array: None,
        }
    }
}

impl From<lyon::path::Builder> for PathBuilder {
    fn from(builder: lyon::path::Builder) -> Self {
        Self {
            raw: builder.with_svg(),
            ..Self::default()
        }
    }
}

fn lyon_point_from(point: Point<Pixels>) -> LyonPoint {
    lyon_point(point.x.value(), point.y.value())
}

fn lyon_vector_from(point: Point<Pixels>) -> lyon::math::Vector {
    vector(point.x.value(), point.y.value())
}

impl PathBuilder {
    /// Start a stroke path.
    pub fn stroke(width: Pixels) -> Self {
        Self {
            style: PathStyle::Stroke(StrokeOptions::default().with_line_width(width.value())),
            ..Self::default()
        }
    }

    /// Start a fill path.
    pub fn fill() -> Self {
        Self::default()
    }

    /// Replace the tessellation options.
    pub fn with_style(mut self, style: PathStyle) -> Self {
        self.style = style;
        self
    }

    /// Set a CSS/SVG-style dash array. Invalid or empty arrays are treated as
    /// an undashed path so building can never get stuck on a zero-length dash.
    pub fn dash_array(mut self, dash_array: &[Pixels]) -> Self {
        let mut values = dash_array.to_vec();
        if values.len() % 2 == 1 {
            values.extend_from_within(..);
        }
        self.dash_array = Some(values);
        self
    }

    pub fn move_to(&mut self, to: Point<Pixels>) {
        self.raw.move_to(lyon_point_from(to));
    }

    pub fn line_to(&mut self, to: Point<Pixels>) {
        self.raw.line_to(lyon_point_from(to));
    }

    pub fn curve_to(&mut self, to: Point<Pixels>, control: Point<Pixels>) {
        self.raw
            .quadratic_bezier_to(lyon_point_from(control), lyon_point_from(to));
    }

    pub fn cubic_bezier_to(
        &mut self,
        to: Point<Pixels>,
        control_a: Point<Pixels>,
        control_b: Point<Pixels>,
    ) {
        self.raw.cubic_bezier_to(
            lyon_point_from(control_a),
            lyon_point_from(control_b),
            lyon_point_from(to),
        );
    }

    pub fn arc_to(
        &mut self,
        radii: Point<Pixels>,
        x_rotation: Pixels,
        large_arc: bool,
        sweep: bool,
        to: Point<Pixels>,
    ) {
        self.raw.arc_to(
            lyon_vector_from(radii),
            Angle::degrees(x_rotation.value()),
            ArcFlags { large_arc, sweep },
            lyon_point_from(to),
        );
    }

    pub fn relative_arc_to(
        &mut self,
        radii: Point<Pixels>,
        x_rotation: Pixels,
        large_arc: bool,
        sweep: bool,
        to: Point<Pixels>,
    ) {
        self.raw.relative_arc_to(
            lyon_vector_from(radii),
            Angle::degrees(x_rotation.value()),
            ArcFlags { large_arc, sweep },
            lyon_vector_from(to),
        );
    }

    pub fn add_polygon(&mut self, points: &[Point<Pixels>], closed: bool) {
        let points = points
            .iter()
            .copied()
            .map(lyon_point_from)
            .collect::<Vec<_>>();
        self.raw.add_polygon(Polygon {
            points: points.as_ref(),
            closed,
        });
    }

    pub fn close(&mut self) {
        self.raw.close();
    }

    pub fn transform(&mut self, transform: Transform) {
        self.transform = Some(transform);
    }

    pub fn translate(&mut self, offset: Point<Pixels>) {
        self.transform = Some(match self.transform {
            Some(transform) => transform.then_translate(lyon_vector_from(offset)),
            None => Transform::translation(offset.x.value(), offset.y.value()),
        });
    }

    pub fn scale(&mut self, scale: f32) {
        self.transform = Some(match self.transform {
            Some(transform) => transform.then_scale(scale, scale),
            None => Transform::scale(scale, scale),
        });
    }

    pub fn rotate(&mut self, angle: f32) {
        let rotation = Angle::radians(angle.to_radians());
        self.transform = Some(match self.transform {
            Some(transform) => transform.then_rotate(rotation),
            None => Transform::rotation(rotation),
        });
    }

    /// Tessellate the builder into a retained path with opaque white as its
    /// initial colour. Canvas callbacks normally replace it with `path(...)`.
    pub fn build(self) -> Result<Path, Error> {
        let path = if let Some(transform) = self.transform {
            self.raw.build().transformed(&transform)
        } else {
            self.raw.build()
        };
        match self.style {
            PathStyle::Stroke(options) => Self::tessellate_stroke(self.dash_array, &path, &options),
            PathStyle::Fill(options) => Self::tessellate_fill(&path, &options),
        }
    }

    fn tessellate_fill(path: &lyon::path::Path, options: &FillOptions) -> Result<Path, Error> {
        let mut buffers: VertexBuffers<LyonPoint, u16> = VertexBuffers::new();
        FillTessellator::new().tessellate_path(
            path,
            options,
            &mut BuffersBuilder::new(&mut buffers, |vertex: FillVertex| vertex.position()),
        )?;
        Ok(Path::from_lyon_tessellation(buffers, [1.0; 4]))
    }

    fn tessellate_stroke(
        dash_array: Option<Vec<Pixels>>,
        path: &lyon::path::Path,
        options: &StrokeOptions,
    ) -> Result<Path, Error> {
        let dashed_path = dash_array
            .filter(|values| {
                !values.is_empty()
                    && values
                        .iter()
                        .all(|value| value.value().is_finite() && value.value() > 0.0)
            })
            .and_then(|values| {
                let measurements =
                    lyon::algorithms::measure::PathMeasurements::from_path(path, 0.01);
                let mut sampler = measurements
                    .create_sampler(path, lyon::algorithms::measure::SampleType::Normalized);
                let total_length = sampler.length();
                if !total_length.is_finite() || total_length <= f32::EPSILON {
                    return None;
                }
                let mut builder = lyon::path::Path::builder();
                let mut position = 0.0;
                let mut dash_index = 0;
                while position < total_length {
                    let dash_length = values[dash_index % values.len()].value();
                    let next_position = (position + dash_length).min(total_length);
                    if dash_index % 2 == 0 {
                        sampler.split_range(
                            position / total_length..next_position / total_length,
                            &mut builder,
                        );
                    }
                    position = next_position;
                    dash_index += 1;
                }
                Some(builder.build())
            });
        let path = dashed_path.as_ref().unwrap_or(path);
        let mut buffers: VertexBuffers<LyonPoint, u16> = VertexBuffers::new();
        StrokeTessellator::new().tessellate_path(
            path,
            options,
            &mut BuffersBuilder::new(&mut buffers, |vertex: StrokeVertex| vertex.position()),
        )?;
        Ok(Path::from_lyon_tessellation(buffers, [1.0; 4]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::styled::Styled;
    use wgpui_core::color::red;
    use wgpui_core::geometry::{point, px, size};

    fn bounds() -> Bounds<Pixels> {
        Bounds::new(point(px(10.0), px(20.0)), size(px(100.0), px(60.0)))
    }

    #[test]
    fn fill_builds_a_solid_quad() {
        let quad = fill(bounds(), red());
        assert_eq!(quad.origin, [10.0, 20.0]);
        assert_eq!(quad.size, [100.0, 60.0]);
        assert_eq!(quad.background, <Hsla as Into<[f32; 4]>>::into(red()));
        assert_eq!(quad.border_color[3], 0.0);
    }

    #[test]
    fn path_builder_tessellates_fill_and_stroke() {
        let mut fill_builder = PathBuilder::fill();
        fill_builder.move_to(point(px(0.0), px(0.0)));
        fill_builder.line_to(point(px(20.0), px(0.0)));
        fill_builder.line_to(point(px(0.0), px(20.0)));
        fill_builder.close();
        assert!(
            !fill_builder
                .build()
                .expect("fill tessellation")
                .vertices
                .is_empty()
        );

        let mut stroke_builder = PathBuilder::stroke(px(2.0));
        stroke_builder.move_to(point(px(0.0), px(0.0)));
        stroke_builder.line_to(point(px(20.0), px(0.0)));
        assert!(
            !stroke_builder
                .build()
                .expect("stroke tessellation")
                .vertices
                .is_empty()
        );
    }

    #[test]
    fn empty_dash_array_does_not_panic_or_loop() {
        let mut builder = PathBuilder::stroke(px(2.0)).dash_array(&[]);
        builder.move_to(point(px(0.0), px(0.0)));
        builder.line_to(point(px(20.0), px(0.0)));
        assert!(
            !builder
                .build()
                .expect("undashed fallback")
                .vertices
                .is_empty()
        );
    }

    #[test]
    fn canvas_is_a_retained_emitting_element() {
        let element = canvas(|context, emission| {
            emission.quad(context.fill(bounds(), red()));
        })
        .w(100.0)
        .h(60.0)
        .id("drawing");
        let description = element.into_description();
        assert_eq!(
            description.element_id(),
            Some(&ElementId::Name("drawing".into()))
        );
        assert!(description.emits());
    }
}
