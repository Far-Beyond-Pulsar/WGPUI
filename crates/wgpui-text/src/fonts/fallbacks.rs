//! Configured font fallbacks — today's `src/text_system/font_fallbacks.rs`.
//! See docs/gpu-native-architecture.md §3.3.
//!
//! Moved, not rebuilt, minus the `serde`/`schemars` derives, for the reason
//! `features.rs` gives at greater length.

use std::sync::Arc;

/// The fallback font families configured for a font, in priority order.
#[derive(Default, Clone, Eq, PartialEq, Hash, Debug)]
pub struct FontFallbacks(pub Arc<Vec<String>>);

impl FontFallbacks {
    /// The fallback family names, in priority order.
    pub fn fallback_list(&self) -> &[String] {
        self.0.as_slice()
    }

    /// Fallbacks from a list of family names.
    pub fn from_fonts(fonts: Vec<String>) -> Self {
        FontFallbacks(Arc::new(fonts))
    }
}

impl From<Vec<String>> for FontFallbacks {
    fn from(fonts: Vec<String>) -> Self {
        Self::from_fonts(fonts)
    }
}
