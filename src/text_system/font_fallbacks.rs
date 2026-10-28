use std::sync::Arc;

use cosmic_text::{Fallback, FontSystem, PlatformFallback, fontdb};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use unicode_script::{Script, UnicodeScript};

/// The fallback fonts that can be configured for a given font.
/// Fallback fonts family names are stored here.
#[derive(Default, Clone, Eq, PartialEq, Hash, Debug, Deserialize, Serialize, JsonSchema)]
pub struct FontFallbacks(pub Arc<Vec<String>>);

impl FontFallbacks {
    /// Get the fallback fonts family names
    pub fn fallback_list(&self) -> &[String] {
        self.0.as_slice()
    }

    /// Create a font fallback from a list of strings
    pub fn from_fonts(fonts: Vec<String>) -> Self {
        FontFallbacks(Arc::new(fonts))
    }
}

/// Creates a font system with installed fonts, system generic families and regional CJK fallbacks.
pub fn new_font_system() -> FontSystem {
    let locale = sys_locale::get_locale().unwrap_or_else(|| "en-US".into());
    new_font_system_with_locale(locale)
}

/// Creates a font system using the given locale for regional font fallbacks.
pub fn new_font_system_with_locale(locale: impl Into<String>) -> FontSystem {
    let locale = locale.into();
    let mut database = fontdb::Database::new();
    database.load_system_fonts();
    configure_generic_families(&mut database);
    let common = PlatformFallback
        .common_fallback()
        .iter()
        .copied()
        .chain(cjk_font_fallbacks(&locale).iter().copied())
        .collect();
    FontSystem::new_with_locale_and_db_and_fallback(locale, database, RegionalFallback { common })
}

fn configure_generic_families(database: &mut fontdb::Database) {
    let families: &[(fontdb::Family<'_>, &[&str])] = &[
        (
            fontdb::Family::SansSerif,
            &[
                "Segoe UI",
                "Helvetica Neue",
                "Arial",
                "Noto Sans",
                "DejaVu Sans",
            ],
        ),
        (
            fontdb::Family::Serif,
            &["Times New Roman", "Noto Serif", "DejaVu Serif"],
        ),
        (
            fontdb::Family::Monospace,
            &["Consolas", "Menlo", "Noto Sans Mono", "DejaVu Sans Mono"],
        ),
    ];
    for (family, candidates) in families {
        let configured = database.family_name(family);
        // fontdb reads fontconfig aliases on Linux; retain an installed configured family.
        if !cfg!(target_os = "windows") && has_family(database, configured) {
            continue;
        }
        if let Some(name) = candidates.iter().find(|name| has_family(database, name)) {
            match family {
                fontdb::Family::SansSerif => database.set_sans_serif_family(*name),
                fontdb::Family::Serif => database.set_serif_family(*name),
                fontdb::Family::Monospace => database.set_monospace_family(*name),
                _ => {}
            }
        }
    }
}

fn has_family(database: &fontdb::Database, name: &str) -> bool {
    database.faces().any(|face| {
        face.families
            .iter()
            .any(|(family, _)| family.eq_ignore_ascii_case(name))
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CjkRegion {
    Japanese,
    Korean,
    SimplifiedChinese,
    TraditionalChinese,
    HongKong,
}

impl CjkRegion {
    fn from_locale(locale: &str) -> Self {
        let locale = locale
            .split(['.', '@'])
            .next()
            .unwrap_or(locale)
            .replace('_', "-")
            .to_ascii_lowercase();
        let mut parts = locale.split('-');
        match parts.next() {
            Some("ja") => Self::Japanese,
            Some("ko") => Self::Korean,
            Some("zh") => {
                let parts: Vec<_> = parts.collect();
                if parts.contains(&"hans") {
                    Self::SimplifiedChinese
                } else if parts.contains(&"hk") || parts.contains(&"mo") {
                    Self::HongKong
                } else if parts.contains(&"hant") || parts.contains(&"tw") {
                    Self::TraditionalChinese
                } else {
                    Self::SimplifiedChinese
                }
            }
            _ => Self::SimplifiedChinese,
        }
    }

    fn families(self) -> &'static [&'static str] {
        match self {
            Self::Japanese => &[
                #[cfg(target_os = "macos")]
                "Hiragino Sans",
                "Yu Gothic UI",
                "Yu Gothic",
                "Meiryo",
                "Noto Sans CJK JP",
                "Noto Sans JP",
                "Source Han Sans JP",
                "Source Han Sans",
                "IPAexGothic",
                "IPAGothic",
                "MS Gothic",
            ],
            Self::Korean => &[
                #[cfg(target_os = "macos")]
                "Apple SD Gothic Neo",
                "Malgun Gothic",
                "Noto Sans CJK KR",
                "Noto Sans KR",
                "Source Han Sans KR",
                "Source Han Sans K",
                "NanumGothic",
            ],
            Self::SimplifiedChinese => &[
                #[cfg(target_os = "macos")]
                "PingFang SC",
                "Microsoft YaHei UI",
                "Microsoft YaHei",
                "Noto Sans CJK SC",
                "Noto Sans SC",
                "Source Han Sans SC",
                "Source Han Sans CN",
                "SimHei",
                "SimSun",
            ],
            Self::TraditionalChinese => &[
                #[cfg(target_os = "macos")]
                "PingFang TC",
                "Microsoft JhengHei UI",
                "Microsoft JhengHei",
                "Noto Sans CJK TC",
                "Noto Sans TC",
                "Source Han Sans TC",
                "Source Han Sans TW",
                "MingLiU",
            ],
            Self::HongKong => &[
                #[cfg(target_os = "macos")]
                "PingFang HK",
                "Noto Sans CJK HK",
                "Noto Sans HK",
                "Source Han Sans HC",
                "MingLiU_HKSCS",
                "Microsoft JhengHei UI",
                "Noto Sans CJK TC",
            ],
        }
    }
}

/// Returns regional CJK families for the current platform and locale.
pub fn cjk_font_fallbacks(locale: &str) -> &'static [&'static str] {
    CjkRegion::from_locale(locale).families()
}

pub(crate) fn is_cjk(character: char) -> bool {
    let extension = character.script_extension();
    matches!(
        character.script(),
        Script::Han | Script::Hiragana | Script::Katakana | Script::Hangul | Script::Bopomofo
    ) || (!extension.is_common()
        && !extension.is_inherited()
        && [
            Script::Han,
            Script::Hiragana,
            Script::Katakana,
            Script::Hangul,
            Script::Bopomofo,
        ]
        .iter()
        .any(|script| extension.contains_script(*script)))
        || matches!(character as u32, 0x3000..=0x303f | 0xff01..=0xff60 | 0xffe0..=0xffee)
}

pub(crate) fn is_font_selection_ignorable(character: char) -> bool {
    character.is_control()
        || matches!(character as u32, 0x200c..=0x200d | 0xfe00..=0xfe0f | 0xe0100..=0xe01ef)
}

struct RegionalFallback {
    common: Vec<&'static str>,
}

impl Fallback for RegionalFallback {
    fn common_fallback(&self) -> &[&'static str] {
        &self.common
    }

    fn forbidden_fallback(&self) -> &[&'static str] {
        PlatformFallback.forbidden_fallback()
    }

    fn script_fallback(&self, script: Script, locale: &str) -> &[&'static str] {
        match script {
            Script::Han
            | Script::Bopomofo
            | Script::Hiragana
            | Script::Katakana
            | Script::Hangul => cjk_font_fallbacks(locale),
            _ => PlatformFallback.script_fallback(script, locale),
        }
    }
}
