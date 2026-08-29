use gpui::{
    Application, Context, ElementId, InstanceKey, IntoElement, Pixels, Render, Window,
    WindowOptions, Styled, div, rgb, size,
};

gpui::actions!(compatibility, [ProbeAction]);

#[test]
fn exports_are_backed_by_2_0_types() {
    let element_id = ElementId::from("root");
    let key = InstanceKey::from_path(std::slice::from_ref(&element_id));
    let _description = div().id(element_id).describe();

    assert_ne!(key.as_raw(), 0);
    assert_eq!(Pixels::ZERO.value(), 0.0);
}

struct Root;
impl Render for Root {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        div().id("root")
    }
}

#[test]
fn lifecycle_renders_and_retains_a_description() {
    let mut descriptions = 0;
    Application::new().run(|app| {
        app.open_window(WindowOptions::default(), |_, cx| cx.new(|_| Root))
            .expect("window callback should build");
        descriptions = app.descriptions().len();
    });
    assert_eq!(descriptions, 1);
}

#[test]
fn color_and_geometry_constructors_preserve_legacy_values() {
    let color = rgb(0x123456);
    assert_eq!(
        <[f32; 4]>::from(color),
        [
            0x12 as f32 / 255.0,
            0x34 as f32 / 255.0,
            0x56 as f32 / 255.0,
            1.0
        ]
    );
    assert_eq!(size(Pixels(3.0), Pixels(4.0)).height.value(), 4.0);
}

#[test]
fn color_adapters_preserve_alpha_and_hue_conversion() {
    assert_eq!(gpui::rgb(0x123456).opacity(0.25).a, 0.25);
    let red: gpui::Hsla = gpui::rgb(0xff0000).into();
    assert!((red.h - 0.0).abs() < f32::EPSILON);
    assert!((red.s - 1.0).abs() < f32::EPSILON);
}

#[test]
fn action_macro_and_keybinding_retain_legacy_identity() {
    let binding = gpui::KeyBinding::new("cmd-p", ProbeAction, None);
    assert_eq!(binding.action, "compatibility::ProbeAction");
}

#[test]
fn legacy_geometry_adapters_reach_resolved_style_values() {
    let element = gpui::div()
        .size_16()
        .rounded(gpui::px(8.0))
        .border(gpui::px(2.0))
        .shadow(vec![gpui::BoxShadow {
            color: gpui::hsla(0.0, 0.5, 0.5, 0.25),
            offset: gpui::point(gpui::px(1.0), gpui::px(2.0)),
            blur_radius: gpui::px(4.0),
            spread_radius: gpui::px(0.0),
        }]);
    let style = element.div_style();
    assert_eq!(style.corner_radii.top_left, 8.0);
    assert_eq!(style.border_widths.top, 2.0);
    assert_eq!(style.box_shadow[0].offset, [1.0, 2.0]);
    assert_eq!(style.box_shadow[0].blur_radius, 4.0);
}
