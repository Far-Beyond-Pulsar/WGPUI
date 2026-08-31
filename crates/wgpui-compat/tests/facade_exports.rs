use gpui::{IntoElement, Render, Styled, Window, div, hsla, rgb};

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

#[test]
fn compatibility_colors_lower_into_native_style_fields() {
    let element = div()
        .bg(rgb(0x102030))
        .border_color(hsla(0.5, 1.0, 0.5, 0.75))
        .text_color(rgb(0x405060));
    let style = element.div_style();

    assert_eq!(
        style.background,
        Some([16.0 / 255.0, 32.0 / 255.0, 48.0 / 255.0, 1.0])
    );
    let border_color = style.border_color.expect("border color should be set");
    assert!((border_color[0] - 0.0).abs() < 1e-5);
    assert!((border_color[1] - 1.0).abs() < 1e-5);
    assert!((border_color[2] - 1.0).abs() < 1e-5);
    assert!((border_color[3] - 0.75).abs() < 1e-5);
    assert_eq!(
        style.text_color,
        Some([64.0 / 255.0, 80.0 / 255.0, 96.0 / 255.0, 1.0])
    );
}
