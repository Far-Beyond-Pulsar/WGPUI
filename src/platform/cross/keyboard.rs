use crate::PlatformKeyboardLayout;
use std::sync::OnceLock;

fn detect_keyboard_layout() -> &'static str {
    static LAYOUT: OnceLock<String> = OnceLock::new();
    LAYOUT.get_or_init(|| {
        std::env::var("LANG")
            .ok()
            .and_then(|lang| {
                lang.rsplit_once('.')
                    .map(|(tag, _)| tag.to_string())
                    .or(Some(lang))
            })
            .and_then(|tag| tag.rsplit_once('_').map(|(_, code)| code.to_ascii_lowercase()))
            .unwrap_or_else(|| "us".to_string())
    })
}

pub(crate) struct CrossKeyboardLayout;

impl PlatformKeyboardLayout for CrossKeyboardLayout {
    fn id(&self) -> &str {
        detect_keyboard_layout()
    }

    fn name(&self) -> &str {
        detect_keyboard_layout()
    }
}
