use std::cell::Cell;
use std::rc::Rc;

use crate::{AppContext, Context, IntoElement, ParentElement, Render, TestAppContext, Window, div};

struct RenderCounter {
    count: Rc<Cell<usize>>,
}

impl Render for RenderCounter {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        self.count.set(self.count.get() + 1);
        div()
    }
}

struct Root {
    child: crate::Entity<RenderCounter>,
}

impl Render for Root {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        div().child(self.child.clone())
    }
}

fn draw_window(cx: &mut crate::VisualTestContext) {
    cx.update(|window, cx| window.draw(cx).clear());
}

#[gpui::test]
fn retained_view_does_not_rerender_clean_child_by_default(cx: &mut TestAppContext) {
    let render_count = Rc::new(Cell::new(0));
    let render_count_for_child = render_count.clone();
    let (_root, cx) = cx.add_window_view(|_window, cx| {
        let child = cx.new(|_| RenderCounter {
            count: render_count_for_child,
        });
        Root { child }
    });

    assert_eq!(render_count.get(), 1);

    draw_window(cx);
    let render_count_after_warmup = render_count.get();

    draw_window(cx);

    assert_eq!(render_count.get(), render_count_after_warmup);
}

#[gpui::test]
fn retained_view_rerenders_only_notified_child(cx: &mut TestAppContext) {
    struct TwoChildren {
        first: crate::Entity<RenderCounter>,
        second: crate::Entity<RenderCounter>,
    }

    impl Render for TwoChildren {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            div().child(self.first.clone()).child(self.second.clone())
        }
    }

    let first_count = Rc::new(Cell::new(0));
    let second_count = Rc::new(Cell::new(0));
    let first_count_for_child = first_count.clone();
    let second_count_for_child = second_count.clone();
    let (root, cx) = cx.add_window_view(|_window, cx| {
        let first = cx.new(|_| RenderCounter {
            count: first_count_for_child,
        });
        let second = cx.new(|_| RenderCounter {
            count: second_count_for_child,
        });
        TwoChildren { first, second }
    });

    assert_eq!(first_count.get(), 1);
    assert_eq!(second_count.get(), 1);

    draw_window(cx);
    let first_count_after_warmup = first_count.get();
    let second_count_after_warmup = second_count.get();

    let second = root.read_with(cx, |root, _cx| root.second.clone());
    second.update(cx, |_second, cx| cx.notify());
    cx.run_until_parked();
    draw_window(cx);

    assert_eq!(first_count.get(), first_count_after_warmup);
    assert_eq!(second_count.get(), second_count_after_warmup + 1);
}
