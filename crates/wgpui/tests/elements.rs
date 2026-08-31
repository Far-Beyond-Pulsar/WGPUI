use wgpui::{
    Component, Element, ElementId, IntoElement, Render, RenderOnce, Stateful, Styled, TextAlign,
    TextOverflow, div, hsla, linear_color_stop, px, render_description,
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

#[derive(IntoElement)]
struct GenericBadge<T> {
    value: T,
}

impl<T: 'static> RenderOnce for GenericBadge<T> {
    fn render(self) -> impl IntoElement + 'static {
        let _value = self.value;
        div().id("generic-badge").child("generic")
    }
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
fn opaque_native_elements_can_be_nested_without_a_facade_trait() {
    fn child() -> impl IntoElement {
        div().id("child").child("content")
    }

    let description = div().id("root").child(child()).describe();
    assert_eq!(description.child_descriptions().len(), 1);
    assert_eq!(
        description.child_descriptions()[0].element_id(),
        Some(&ElementId::Name("child".into()))
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
fn derived_generic_component_owns_its_render_once_value() {
    let description = Element::into_description(
        GenericBadge {
            value: String::from("owned"),
        }
        .into_element(),
    );
    assert_eq!(
        description.element_id(),
        Some(&ElementId::Name("generic-badge".into()))
    );
}

#[test]
fn native_style_changes_report_display_invalidation_through_reconciliation() {
    let mut reconciler = wgpui::Reconciler::new();
    let mut layout = wgpui::LayoutTree::new();
    reconciler
        .reconcile(
            div()
                .id("root")
                .w(100.0)
                .bg(wgpui::rgb(0x112233))
                .describe(),
            &mut layout,
        )
        .expect("first frame should reconcile");
    let plan = reconciler
        .reconcile(
            div()
                .id("root")
                .w(100.0)
                .bg(wgpui::rgb(0x445566))
                .describe(),
            &mut layout,
        )
        .expect("second frame should reconcile");
    assert!(
        plan.nodes()[0]
            .invalidation
            .contains(wgpui::invalidation::axes::Invalidation::DISPLAY)
    );
    assert!(
        !plan.nodes()[0]
            .invalidation
            .contains(wgpui::invalidation::axes::Invalidation::LAYOUT)
    );
}

#[test]
fn native_style_aliases_lower_to_layout_and_text_properties() {
    let style = div()
        .flex_initial()
        .flex_wrap_reverse()
        .flex_shrink()
        .flex_basis(24.0)
        .self_baseline()
        .content_normal()
        .aspect_ratio(2.0)
        .scrollbar_width(px(7.0))
        .mt(px(3.0))
        .border_x(2.0)
        .border_y(3.0)
        .text_align(TextAlign::Right)
        .whitespace_normal()
        .text_overflow(TextOverflow::Truncate("...".into()))
        .text_gradient(
            45.0,
            linear_color_stop(hsla(0.0, 1.0, 0.5, 1.0), 0.0),
            linear_color_stop(hsla(0.5, 1.0, 0.5, 1.0), 1.0),
        )
        .div_style()
        .clone();

    assert_eq!(style.layout.flex_grow, 0.0);
    assert_eq!(style.layout.flex_shrink, 1.0);
    assert_eq!(style.layout.flex_basis, wgpui::Dimension::length(24.0));
    assert_eq!(style.layout.flex_wrap, wgpui::FlexWrap::WrapReverse);
    assert_eq!(style.layout.align_self, Some(wgpui::AlignItems::Baseline));
    assert_eq!(style.layout.align_content, None);
    assert_eq!(style.layout.aspect_ratio, Some(2.0));
    assert_eq!(style.layout.scrollbar_width, 7.0);
    assert_eq!(
        style.layout.margin.top,
        wgpui::LengthPercentageAuto::length(3.0)
    );
    assert_eq!(
        style.layout.border.left,
        wgpui::LengthPercentage::length(2.0)
    );
    assert_eq!(
        style.layout.border.right,
        wgpui::LengthPercentage::length(2.0)
    );
    assert_eq!(
        style.layout.border.top,
        wgpui::LengthPercentage::length(3.0)
    );
    assert_eq!(
        style.layout.border.bottom,
        wgpui::LengthPercentage::length(3.0)
    );
    assert_eq!(style.text_alignment, 2);
    assert!(!style.text_white_space_nowrap);
    assert!(style.text_ellipsis);
    assert_eq!(style.text_gradient_angle, Some(45.0));
    assert_eq!(style.text_gradient.as_ref().map(Vec::len), Some(2));
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
