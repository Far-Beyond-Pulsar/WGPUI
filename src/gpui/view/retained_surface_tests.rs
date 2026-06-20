use std::cell::Cell;
use std::rc::Rc;

use crate::{
    Context,
    InteractiveElement,
    IntoElement,
    ListAlignment,
    ListState,
    Modifiers,
    MouseButton,
    ParentElement,
    Render,
    StatefulInteractiveElement,
    Styled,
    TestAppContext,
    Window,
    canvas,
    deferred,
    div,
    list,
    point,
    px,
};

fn draw_window(cx: &mut crate::VisualTestContext) {
    cx.update(|window, cx| window.draw(cx).clear());
}

struct ElementSurfaceRoot {
    render_count: Rc<Cell<usize>>,
    canvas_prepaint_count: Rc<Cell<usize>>,
    canvas_paint_count: Rc<Cell<usize>>,
    deferred_prepaint_count: Rc<Cell<usize>>,
    deferred_paint_count: Rc<Cell<usize>>,
}

impl Render for ElementSurfaceRoot {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        self.render_count.set(self.render_count.get() + 1);
        let canvas_prepaint_count = self.canvas_prepaint_count.clone();
        let canvas_paint_count = self.canvas_paint_count.clone();
        let deferred_prepaint_count = self.deferred_prepaint_count.clone();
        let deferred_paint_count = self.deferred_paint_count.clone();

        div()
            .child("retained text")
            .child(canvas(
                move |_, _, _| canvas_prepaint_count.set(canvas_prepaint_count.get() + 1),
                move |_, _, _, _| canvas_paint_count.set(canvas_paint_count.get() + 1),
            ))
            .child(deferred(canvas(
                move |_, _, _| deferred_prepaint_count.set(deferred_prepaint_count.get() + 1),
                move |_, _, _, _| deferred_paint_count.set(deferred_paint_count.get() + 1),
            )))
    }
}

#[gpui::test]
fn retained_replay_preserves_text_canvas_and_deferred_outputs(cx: &mut TestAppContext) {
    let render_count = Rc::new(Cell::new(0));
    let canvas_prepaint_count = Rc::new(Cell::new(0));
    let canvas_paint_count = Rc::new(Cell::new(0));
    let deferred_prepaint_count = Rc::new(Cell::new(0));
    let deferred_paint_count = Rc::new(Cell::new(0));
    let (_root, cx) = cx.add_window_view(|_window, _cx| ElementSurfaceRoot {
        render_count: render_count.clone(),
        canvas_prepaint_count: canvas_prepaint_count.clone(),
        canvas_paint_count: canvas_paint_count.clone(),
        deferred_prepaint_count: deferred_prepaint_count.clone(),
        deferred_paint_count: deferred_paint_count.clone(),
    });

    assert_eq!(render_count.get(), 1);
    assert_eq!(canvas_prepaint_count.get(), 1);
    assert_eq!(canvas_paint_count.get(), 1);
    assert_eq!(deferred_prepaint_count.get(), 1);
    assert_eq!(deferred_paint_count.get(), 1);

    draw_window(cx);

    assert_eq!(render_count.get(), 1);
    assert_eq!(canvas_prepaint_count.get(), 1);
    assert_eq!(canvas_paint_count.get(), 1);
    assert_eq!(deferred_prepaint_count.get(), 1);
    assert_eq!(deferred_paint_count.get(), 1);
}

struct ListSurfaceRoot {
    render_count: Rc<Cell<usize>>,
    item_render_count: Rc<Cell<usize>>,
    state: ListState,
}

impl Render for ListSurfaceRoot {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        self.render_count.set(self.render_count.get() + 1);
        let item_render_count = self.item_render_count.clone();
        list(self.state.clone(), move |_, _, _| {
            item_render_count.set(item_render_count.get() + 1);
            div().h(px(10.)).into_any_element()
        })
        .h(px(40.))
    }
}

#[gpui::test]
fn retained_replay_preserves_list_item_cache_output(cx: &mut TestAppContext) {
    let render_count = Rc::new(Cell::new(0));
    let item_render_count = Rc::new(Cell::new(0));
    let state = ListState::new(10, ListAlignment::Top, px(10.));
    let (_root, cx) = cx.add_window_view(|_window, _cx| ListSurfaceRoot {
        render_count: render_count.clone(),
        item_render_count: item_render_count.clone(),
        state,
    });

    assert_eq!(render_count.get(), 1);
    assert!(item_render_count.get() > 0);
    let item_render_count_after_first_draw = item_render_count.get();

    draw_window(cx);

    assert_eq!(render_count.get(), 1);
    assert_eq!(item_render_count.get(), item_render_count_after_first_draw);
}

struct FocusClickRoot {
    focus_handle: crate::FocusHandle,
    click_count: Rc<Cell<usize>>,
    render_count: Rc<Cell<usize>>,
}

impl Render for FocusClickRoot {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        self.render_count.set(self.render_count.get() + 1);
        let click_count = self.click_count.clone();
        div()
            .id("retained-focus-click")
            .track_focus(&self.focus_handle)
            .on_click(move |_, _, _| click_count.set(click_count.get() + 1))
            .size(px(20.))
    }
}

#[gpui::test]
fn retained_replay_preserves_focus_hitbox_and_click_handler(cx: &mut TestAppContext) {
    let click_count = Rc::new(Cell::new(0));
    let render_count = Rc::new(Cell::new(0));
    let click_count_for_root = click_count.clone();
    let render_count_for_root = render_count.clone();
    let (root, cx) = cx.add_window_view(|window, cx| {
        let focus_handle = cx.focus_handle();
        window.focus(&focus_handle, cx);
        FocusClickRoot {
            focus_handle,
            click_count: click_count_for_root,
            render_count: render_count_for_root,
        }
    });
    let focus_handle = root.read_with(cx, |root, _cx| root.focus_handle.clone());

    assert_eq!(render_count.get(), 1);
    assert!(cx.update(|window, _cx| focus_handle.is_focused(window)));

    draw_window(cx);

    assert_eq!(render_count.get(), 1);
    assert!(cx.update(|window, _cx| focus_handle.is_focused(window)));
    cx.simulate_click(point(px(1.), px(1.)), Modifiers::default());
    let first_click_increment = click_count.get();
    assert!(first_click_increment > 0);

    draw_window(cx);

    assert!(cx.update(|window, _cx| focus_handle.is_focused(window)));
    let click_count_before_replay_click = click_count.get();
    cx.simulate_click(point(px(1.), px(1.)), Modifiers::default());
    assert_eq!(
        click_count.get() - click_count_before_replay_click,
        first_click_increment
    );

    cx.simulate_mouse_down(
        point(px(1.), px(1.)),
        MouseButton::Left,
        Modifiers::default(),
    );
    assert!(cx.update(|window, _cx| focus_handle.is_focused(window)));
}

struct SurfacePrimitiveRoot {
    render_count: Rc<Cell<usize>>,
}

impl Render for SurfacePrimitiveRoot {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        self.render_count.set(self.render_count.get() + 1);
        canvas(
            |_, _, _| {},
            |bounds, _, window, _| {
                window.paint_wgpu_surface(
                    bounds,
                    crate::platform::cross::surface_registry::SurfaceId(7),
                );
            },
        )
        .size(px(20.))
    }
}

#[gpui::test]
fn retained_replay_preserves_surface_primitives(cx: &mut TestAppContext) {
    let render_count = Rc::new(Cell::new(0));
    let render_count_for_root = render_count.clone();
    let (_root, cx) = cx.add_window_view(|_window, _cx| SurfacePrimitiveRoot {
        render_count: render_count_for_root,
    });

    assert_eq!(render_count.get(), 1);
    let first_surface_count = cx.update(|window, _cx| window.rendered_frame.scene.surfaces.len());
    assert_eq!(first_surface_count, 1);

    draw_window(cx);

    assert_eq!(render_count.get(), 1);
    let (surface_count, changed_ranges_empty) = cx.update(|window, _cx| {
        (
            window.rendered_frame.scene.surfaces.len(),
            window
                .rendered_frame
                .scene
                .changed_ranges()
                .surfaces
                .is_empty(),
        )
    });
    assert_eq!(surface_count, first_surface_count);
    assert!(changed_ranges_empty);
}
