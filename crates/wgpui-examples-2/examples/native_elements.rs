use wgpui::{Application, IntoElement, Render, WindowOptions, div};

struct NativeRoot;

impl Render for NativeRoot {
    fn render(&mut self) -> impl IntoElement + 'static {
        div().id("root").child("native WGPUI")
    }
}

fn main() {
    let mut root = NativeRoot;
    let _description = wgpui::render_description(&mut root);
    let application = Application::new(WindowOptions::default(), move |_window| {
        div().id("root").child("native WGPUI")
    });
    drop(application);
}
