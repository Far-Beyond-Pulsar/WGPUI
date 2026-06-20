use std::cell::{Cell, RefCell};
use std::rc::Rc;

use crate::{
    App,
    AvailableSpace,
    Bounds,
    Context,
    Element,
    ElementId,
    GlobalElementId,
    InspectorElementId,
    IntoElement,
    LayoutId,
    ParentElement,
    Pixels,
    Render,
    Size,
    Style,
    TestAppContext,
    Window,
    div,
    px,
    size,
};

fn draw_window(cx: &mut crate::VisualTestContext) {
    cx.update(|window, cx| window.draw(cx).clear());
}

struct MeasuredRoot {
    text_units: u32,
    style_units: u32,
    render_count: Rc<Cell<usize>>,
    measure_count: Rc<Cell<usize>>,
    measured_size: Rc<RefCell<Option<Size<Pixels>>>>,
}

impl Render for MeasuredRoot {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        self.render_count.set(self.render_count.get() + 1);
        div().child(MeasuredElement {
            text_units: self.text_units,
            style_units: self.style_units,
            measure_count: self.measure_count.clone(),
            measured_size: self.measured_size.clone(),
        })
    }
}

struct MeasuredElement {
    text_units: u32,
    style_units: u32,
    measure_count: Rc<Cell<usize>>,
    measured_size: Rc<RefCell<Option<Size<Pixels>>>>,
}

impl IntoElement for MeasuredElement {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Element for MeasuredElement {
    type RequestLayoutState = ();
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        Some(ElementId::Name("retained-measurement".into()))
    }

    fn source_location(&self) -> Option<&'static std::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        _cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let text_units = self.text_units;
        let style_units = self.style_units;
        let measure_count = self.measure_count.clone();
        let measured_size = self.measured_size.clone();
        let layout_id =
            window.request_measured_layout(Style::default(), move |_, available, _, _| {
                measure_count.set(measure_count.get() + 1);
                let width = match available.width {
                    AvailableSpace::Definite(width) => width.min(px(text_units as f32)),
                    AvailableSpace::MinContent | AvailableSpace::MaxContent => {
                        px(text_units as f32)
                    }
                };
                let measured = size(width, px(style_units as f32));
                measured_size.borrow_mut().replace(measured);
                measured
            });
        (layout_id, ())
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        _window: &mut Window,
        _cx: &mut App,
    ) {
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        _prepaint: &mut Self::PrepaintState,
        _window: &mut Window,
        _cx: &mut App,
    ) {
    }
}

#[gpui::test]
fn retained_measured_layout_reuses_clean_measurement_and_invalidates_changes(
    cx: &mut TestAppContext,
) {
    let render_count = Rc::new(Cell::new(0));
    let measure_count = Rc::new(Cell::new(0));
    let measured_size = Rc::new(RefCell::new(None));
    let render_count_for_root = render_count.clone();
    let measure_count_for_root = measure_count.clone();
    let measured_size_for_root = measured_size.clone();
    let (root, cx) = cx.add_window_view(|_window, _cx| MeasuredRoot {
        text_units: 10,
        style_units: 5,
        render_count: render_count_for_root,
        measure_count: measure_count_for_root,
        measured_size: measured_size_for_root,
    });
    let initial_measure_count = measure_count.get();
    let initial_size = *measured_size.borrow();

    assert_eq!(render_count.get(), 1);
    assert!(initial_measure_count > 0);
    assert_eq!(initial_size, Some(size(px(10.), px(5.))));

    draw_window(cx);

    assert_eq!(render_count.get(), 1);
    assert_eq!(measure_count.get(), initial_measure_count);
    assert_eq!(*measured_size.borrow(), initial_size);

    root.update(cx, |root, cx| {
        root.text_units = 20;
        cx.notify();
    });
    cx.run_until_parked();
    draw_window(cx);

    let text_change_measure_count = measure_count.get();
    assert!(text_change_measure_count > initial_measure_count);
    assert_eq!(*measured_size.borrow(), Some(size(px(20.), px(5.))));

    root.update(cx, |root, cx| {
        root.style_units = 8;
        cx.notify();
    });
    cx.run_until_parked();
    draw_window(cx);

    assert!(measure_count.get() > text_change_measure_count);
    assert_eq!(*measured_size.borrow(), Some(size(px(20.), px(8.))));
}
