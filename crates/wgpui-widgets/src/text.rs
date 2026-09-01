//! Retained standalone text elements.

use std::sync::Arc;

use crate::styled_text::{Highlight, Highlights, SharedTextEngine, StyledText, TextStyle};
use wgpui_core::element::Element;
use wgpui_core::geometry::{Pixels, Size};
use wgpui_core::reconcile::description::Description;
use wgpui_text::shaping::{FontRun, SharedString};

/// A standalone text element backed by the shared shaping and atlas engine.
///
/// The engine is an argument instead of a process-global so a window can share
/// its shaper and glyph tile source with the renderer that presents it.
pub struct Text {
    value: SharedString,
    style: TextStyle,
    engine: SharedTextEngine,
    highlights: Highlights,
    size: Option<Size<Pixels>>,
}

/// Arguments accepted by [`text`]. Both argument orders are supported so the
/// engine can be placed first in code that keeps it near other render state.
pub trait TextArguments {
    fn build(self) -> Text;
}

impl<T> TextArguments for (T, SharedTextEngine)
where
    T: Into<SharedString>,
{
    fn build(self) -> Text {
        Text::new(self.0, TextStyle::default(), self.1)
    }
}

impl TextArguments for (SharedTextEngine, &'static str) {
    fn build(self) -> Text {
        Text::new(self.1, TextStyle::default(), self.0)
    }
}

impl TextArguments for (SharedTextEngine, String) {
    fn build(self) -> Text {
        Text::new(self.1, TextStyle::default(), self.0)
    }
}

impl TextArguments for (SharedTextEngine, SharedString) {
    fn build(self) -> Text {
        Text::new(self.1, TextStyle::default(), self.0)
    }
}

impl<T> TextArguments for (T, TextStyle, SharedTextEngine)
where
    T: Into<SharedString>,
{
    fn build(self) -> Text {
        Text::new(self.0, self.1, self.2)
    }
}

impl<T> TextArguments for (T, SharedTextEngine, TextStyle)
where
    T: Into<SharedString>,
{
    fn build(self) -> Text {
        Text::new(self.0, self.2, self.1)
    }
}

/// Construct a retained text element.
pub fn text<A, B>(first: A, second: B) -> Text
where
    (A, B): TextArguments,
{
    (first, second).build()
}

pub fn text_with_style(
    value: impl Into<SharedString>,
    style: TextStyle,
    engine: SharedTextEngine,
) -> Text {
    Text::new(value, style, engine)
}

impl Text {
    pub fn new(value: impl Into<SharedString>, style: TextStyle, engine: SharedTextEngine) -> Self {
        Self {
            value: value.into(),
            style,
            engine,
            highlights: Arc::from([]),
            size: None,
        }
    }

    pub fn size(
        mut self,
        width: impl crate::styled::IntoStylePixels,
        height: impl crate::styled::IntoStylePixels,
    ) -> Self {
        self.size = Some(Size::pixels(
            width.into_style_pixels(),
            height.into_style_pixels(),
        ));
        self
    }

    pub fn style(mut self, style: TextStyle) -> Self {
        self.style = style;
        self
    }

    pub fn font_size(mut self, size: f32) -> Self {
        self.style.font_size = size;
        self
    }

    pub fn line_height(mut self, height: f32) -> Self {
        self.style.line_height = height;
        self
    }

    pub fn color(mut self, color: [f32; 4]) -> Self {
        self.style.color = color;
        self
    }

    pub fn with_highlights(mut self, highlights: impl Into<Highlights>) -> Self {
        self.highlights = highlights.into();
        self
    }

    pub fn highlight(mut self, highlight: Highlight) -> Self {
        let mut highlights = self.highlights.to_vec();
        highlights.push(highlight);
        self.highlights = Arc::from(highlights);
        self
    }

    pub fn value(&self) -> &str {
        self.value.as_str()
    }

    fn measured_width(&self) -> f32 {
        if self.value.is_empty() {
            return 0.0;
        }
        let font_id = {
            let engine = self.engine.borrow();
            let mut shaper = engine.shaper();
            match shaper.resolve_font(&self.style.font) {
                Ok(font_id) => font_id,
                Err(_) => return 0.0,
            }
        };
        let run = FontRun {
            len: self.value.len(),
            font_id,
            weight: self.style.font.weight,
            style: self.style.font.style,
            letter_spacing: 0.0,
        };
        self.engine
            .borrow()
            .shape_line(&self.value, self.style.font_size, &[run])
            .map_or(0.0, |line| line.width)
    }

    pub fn describe(self) -> Description {
        let size = self
            .size
            .unwrap_or_else(|| Size::pixels(self.measured_width(), self.style.line_height));
        let styled_text = StyledText::new(self.value, self.style, self.engine)
            .with_highlights(self.highlights)
            .size(size.width.value(), size.height.value());
        wgpui_core::element::IntoElement::into_description(styled_text)
    }
}

impl Element for Text {
    fn into_description(self) -> Description {
        self.describe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;
    use wgpui_core::scene::atlas::{GlyphRasterKey, GlyphTile, GlyphTileSource};
    use wgpui_text::engine::TextEngine;
    use wgpui_text::shaping::TextShaper;

    #[derive(Default)]
    struct EmptyTiles;

    impl GlyphTileSource for EmptyTiles {
        fn tile_for(&mut self, _key: GlyphRasterKey) -> Option<GlyphTile> {
            None
        }
    }

    fn engine() -> SharedTextEngine {
        Rc::new(RefCell::new(TextEngine::new(
            TextShaper::new(),
            Box::new(EmptyTiles),
        )))
    }

    #[test]
    fn text_lowers_to_shaped_styled_text_with_intrinsic_size() {
        let description = text("native", engine()).describe();
        assert_eq!(description.type_name(), std::any::type_name::<StyledText>());
        assert_ne!(
            description.layout_style().size.width,
            wgpui_layout::taffy_tree::Dimension::length(0.0)
        );
        assert!(description.emits());
    }

    #[test]
    fn text_supports_explicit_style_and_size() {
        let description = Text::new("native", TextStyle::default(), engine())
            .font_size(18.0)
            .size(120.0, 24.0)
            .describe();
        assert_eq!(
            description.layout_style().size.width,
            wgpui_layout::taffy_tree::Dimension::length(120.0)
        );
        assert_eq!(
            description.layout_style().size.height,
            wgpui_layout::taffy_tree::Dimension::length(24.0)
        );
    }
}
