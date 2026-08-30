//! Emoji Display Example
//!
//! Renders mixed text and emoji sequences to help diagnose glyph rasterization
//! and atlas upload issues across different GPU/driver combinations.

#[path = "../prelude.rs"]
mod example_prelude;

use example_prelude::init_example;
use wgpui::{
    App, Application, Bounds, Context, Render, Window, WindowBounds, WindowOptions, div,
    prelude::*, px, rgb, size,
};

struct EmojiDisplay;

impl Render for EmojiDisplay {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .bg(rgb(0x111827))
            .text_color(rgb(0xf8fafc))
            .p_4()
            .flex()
            .flex_col()
            .gap_2()
            .child(div().text_xl().font_weight(wgpui::FontWeight::BOLD).child("Emoji Display"))
            .child(div().text_sm().text_color(rgb(0x94a3b8)).child(
                "Use this window to verify regular text, emoji color glyphs, and ZWJ/variation sequences.",
            ))
            .child(div().text_base().child("ASCII text: The quick brown fox jumps over the lazy dog 0123456789"))
            .child(div().text_base().child("Symbols: !@#$%^&*() [] {} <> +-=/ ~|"))
            .child(div().text_lg().child("Simple emoji: 😀 😁 😂 🤣 😎 🤖 🧠 💡 🚀 ✅"))
            .child(div().text_lg().child("People + skin tones: 👍 👍🏻 👍🏽 👍🏿 🙋‍♀️ 🙋🏽‍♂️ 🧑🏿‍💻"))
            .child(div().text_lg().child("ZWJ sequences: 👨‍👩‍👧‍👦 👩‍❤️‍💋‍👨 🧑‍🔬 🧑‍🚀 🏳️‍🌈 🏳️‍⚧️"))
            .child(div().text_lg().child("Flags: 🇺🇸 🇫🇷 🇯🇵 🇺🇦 🇧🇷 🇨🇦 🇰🇷"))
            .child(div().text_lg().child("Variation selectors: ✌︎ ✌️ ✊︎ ✊️ ☺︎ ☺️ ❤️ ❤"))
            .child(div().text_lg().child("Keycaps + misc: 0️⃣ 1️⃣ 2️⃣ #️⃣ *️⃣ ™️ ©️ ®️ ⚠️"))
            .child(
                div()
                    .mt_2()
                    .text_sm()
                    .text_color(rgb(0xfbbf24))
                    .child("If you see square blocks/noise, capture a screenshot of this window."),
            )
    }
}

fn main() {
    Application::new().run(|cx: &mut App| {
        init_example(cx, "Emoji Display");

        let bounds = Bounds::centered(None, size(px(980.0), px(640.0)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| cx.new(|_| EmojiDisplay),
        )
        .expect("failed to open emoji display window");
    });
}
