use gpui::{
    Application, Context, ElementId, InstanceKey, IntoElement, Pixels, Render, Window,
    WindowOptions, div, rgb, size,
};

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
