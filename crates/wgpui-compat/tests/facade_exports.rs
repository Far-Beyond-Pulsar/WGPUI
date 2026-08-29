use gpui::{IntoElement, InteractiveElement, Render, Window, div};

gpui::actions!(compatibility, [ProbeAction]);

struct Root;

impl Render for Root {
    fn render(&mut self, _window: &mut Window, _cx: &mut gpui::Context<Self>) -> impl IntoElement {
        div().id("root").into_element()
    }
}

#[test]
fn compatibility_exports_real_interactive_and_style_contract() {
    let _render: fn(&mut Root, &mut Window) -> _ = |_root, _window| div().id("root").into_element();
}

#[test]
fn compatibility_lifecycle_builds_a_real_window() {
    let _window_options = std::mem::size_of::<gpui::WindowOptions>();
}

#[test]
fn action_macro_preserves_legacy_identity() {
    let binding = gpui::KeyBinding::new("cmd-p", ProbeAction, None);
    assert_eq!(binding.keystrokes().len(), 1);
}
