//! OpenType feature settings — today's `src/text_system/font_features.rs`.
//! See docs/gpu-native-architecture.md §3.3.
//!
//! Moved, not rebuilt, with one deliberate subtraction: the legacy type carries
//! ~110 lines of hand-written `serde`/`schemars` plumbing so a feature map can
//! be written in a settings file. That is a *configuration* concern belonging
//! to whatever reads settings, not to shaping, and taking `serde`, `schemars`
//! and `serde_json` into `wgpui-text` to carry it across would make a crate
//! whose whole job is "call cosmic-text" depend on a JSON schema generator.
//! Whichever phase moves settings into the workspace re-attaches those impls
//! here; nothing about the shaping path needs them.

use std::sync::Arc;

/// The OpenType features configured for a font.
///
/// Ordered pairs rather than a map, because the order is what a font sees and
/// two orderings of the same pairs are two different `FontFeatures` as far as
/// the shaping cache is concerned — cheaper to keep them distinct than to
/// canonicalise on every comparison.
#[derive(Default, Clone, Eq, PartialEq, Hash)]
pub struct FontFeatures(pub Arc<Vec<(String, u32)>>);

impl FontFeatures {
    /// Features from a list of `(tag, value)` pairs.
    pub fn from_pairs(pairs: Vec<(String, u32)>) -> Self {
        FontFeatures(Arc::new(pairs))
    }

    /// Disables `calt`.
    pub fn disable_ligatures() -> Self {
        Self(Arc::new(vec![("calt".into(), 0)]))
    }

    /// The `(tag, value)` pairs, in order.
    pub fn tag_value_list(&self) -> &[(String, u32)] {
        self.0.as_slice()
    }

    /// Whether `calt` is enabled, or `None` if it is not mentioned.
    pub fn is_calt_enabled(&self) -> Option<bool> {
        self.0
            .iter()
            .find(|(feature, _)| feature == "calt")
            .map(|(_, value)| *value == 1)
    }
}

impl std::fmt::Debug for FontFeatures {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = formatter.debug_struct("FontFeatures");
        for (tag, value) in self.tag_value_list() {
            debug.field(tag, value);
        }
        debug.finish()
    }
}

/// A feature tag that is not exactly four ASCII bytes, and so cannot be turned
/// into an OpenType tag.
///
/// The legacy conversion `.context("Incorrect feature flag format")?`s on this;
/// carrying it as a typed error keeps `wgpui-text` free of `anyhow` for the one
/// fallible thing it does.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InvalidFeatureTag {
    /// The tag as written.
    pub tag: String,
}

impl std::fmt::Display for InvalidFeatureTag {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "OpenType feature tag {:?} is not four bytes long",
            self.tag
        )
    }
}

impl std::error::Error for InvalidFeatureTag {}

impl TryFrom<&FontFeatures> for cosmic_text::FontFeatures {
    type Error = InvalidFeatureTag;

    fn try_from(features: &FontFeatures) -> Result<Self, Self::Error> {
        let mut result = cosmic_text::FontFeatures::new();
        for (tag, value) in features.tag_value_list() {
            let bytes: [u8; 4] = tag
                .as_bytes()
                .try_into()
                .map_err(|_| InvalidFeatureTag { tag: tag.clone() })?;
            result.set(cosmic_text::FeatureTag::new(&bytes), *value);
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_four_byte_tag_converts_and_a_shorter_one_reports_rather_than_panicking() {
        let good = FontFeatures::from_pairs(vec![("liga".into(), 1)]);
        assert!(cosmic_text::FontFeatures::try_from(&good).is_ok());

        let bad = FontFeatures::from_pairs(vec![("lig".into(), 1)]);
        assert_eq!(
            cosmic_text::FontFeatures::try_from(&bad).err(),
            Some(InvalidFeatureTag { tag: "lig".into() })
        );
    }

    #[test]
    fn calt_reports_absent_enabled_and_disabled_distinctly() {
        assert_eq!(FontFeatures::default().is_calt_enabled(), None);
        assert_eq!(FontFeatures::disable_ligatures().is_calt_enabled(), Some(false));
        assert_eq!(
            FontFeatures::from_pairs(vec![("calt".into(), 1)]).is_calt_enabled(),
            Some(true)
        );
    }
}
