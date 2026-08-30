use wgpui::{
    Component, Element, ElementId, IntoElement, Render, RenderOnce, Stateful, div,
    render_description,
};

#[derive(IntoElement)]
struct Badge {
    label: String,
}

impl RenderOnce for Badge {
    fn render(self) -> impl IntoElement + 'static {
        div().id("badge").child(self.label)
    }
}

struct Root {
    value: String,
}

impl Render for Root {
    fn render(&mut self) -> impl IntoElement + 'static {
        div()
            .id("root")
            .child(Badge {
                label: self.value.clone(),
            })
            .child("stable sibling")
    }
}

#[test]
fn render_and_derive_lower_nested_descriptions() {
    let mut root = Root {
        value: "native".to_string(),
    };
    let description = render_description(&mut root);
    assert_eq!(
        description.element_id(),
        Some(&ElementId::Name("root".into()))
    );
    assert_eq!(description.child_descriptions().len(), 2);
    assert!(
        description.child_descriptions()[0]
            .type_name()
            .ends_with("::Div")
    );
    assert_eq!(
        description.child_descriptions()[0]
            .child_descriptions()
            .len(),
        1
    );
}

#[test]
fn derived_component_is_a_real_component_element() {
    let element = Badge {
        label: "component".to_string(),
    }
    .into_element();
    assert_eq!(element.component().label, "component");
    let description = Element::into_description(element);
    assert_eq!(
        description.element_id(),
        Some(&ElementId::Name("badge".into()))
    );
}

#[test]
fn stateful_element_retains_state_in_the_native_store() {
    let stateful = Stateful::new(div()).id("counter");
    let mut store = wgpui::core::reconcile::ElementStateStore::new();
    assert_eq!(
        stateful.with_state(
            &mut store,
            1,
            || 0_u32,
            |value| {
                *value += 1;
                *value
            }
        ),
        Some(1)
    );
    assert_eq!(
        stateful.with_state(&mut store, 2, || 0_u32, |value| *value),
        Some(1)
    );
    assert_eq!(
        Element::into_description(stateful).element_id(),
        Some(&ElementId::Name("counter".into()))
    );
}

#[allow(dead_code)]
fn assert_component_type(_: Component<Badge>) {}
