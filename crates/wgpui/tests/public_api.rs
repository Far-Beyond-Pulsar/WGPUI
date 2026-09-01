use wgpui::{App, Component, Context, Element, IntoElement, Render, RenderOnce, Window, div};

struct View;

impl Render for View {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        div().child("view")
    }
}

struct Owned;

impl RenderOnce for Owned {
    fn render(self, _window: &mut Window, _cx: &mut App) -> impl IntoElement {
        div().child("owned")
    }
}

fn assert_render_contract<T: Render>() {}
fn assert_render_once_contract<T: RenderOnce>() {}
fn assert_element_contract<T: Element>() {}

#[test]
fn public_render_contract_uses_the_wgpu_window() {
    assert_render_contract::<View>();
    assert_render_once_contract::<Owned>();

    let component = Component::new(Owned);
    let _: &Owned = component.component();
    assert_element_contract::<Component<Owned>>();
}

#[test]
fn public_window_keeps_core_interaction_and_winit_access_at_the_boundary() {
    fn inspect_window(window: &mut wgpui::Window) {
        let interaction = window.interaction_mut();
        let _ = interaction.refresh_requested();
        let _ = window.winit_window().id();
    }

    let _ = inspect_window;
}
