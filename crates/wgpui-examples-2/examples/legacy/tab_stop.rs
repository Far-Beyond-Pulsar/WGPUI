use wgpui::{
    App, Application, Bounds, Context, Div, ElementId, FocusHandle, KeyBinding, SharedString,
    Window, WindowBounds, WindowOptions, actions, div, prelude::*, px, size,
};

actions!(example, [Tab, TabPrev]);

struct Example {
    focus_handle: FocusHandle,
    items: Vec<FocusHandle>,
    message: SharedString,
}

impl Example {
    fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let items = vec![
            cx.focus_handle().tab_index(1).tab_stop(true),
            cx.focus_handle().tab_index(2).tab_stop(true),
            cx.focus_handle().tab_index(3).tab_stop(true),
            cx.focus_handle(),
            cx.focus_handle().tab_index(2).tab_stop(true),
        ];

        let focus_handle = cx.focus_handle();
        window.focus(&focus_handle, cx);

        Self {
            focus_handle,
            items,
            message: SharedString::from("Press `Tab`, `Shift-Tab` to switch focus."),
        }
    }

    fn on_tab(&mut self, _: &Tab, window: &mut Window, cx: &mut Context<Self>) {
        window.focus_next(cx);
        self.message = SharedString::from("You have pressed `Tab`.");
    }

    fn on_tab_prev(&mut self, _: &TabPrev, window: &mut Window, cx: &mut Context<Self>) {
        window.focus_prev(cx);
        self.message = SharedString::from("You have pressed `Shift-Tab`.");
    }
}

impl Render for Example {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        fn tab_stop_style<T: Styled>(this: T) -> T {
            this.border_3().border_color(wgpui::blue())
        }

        fn button(id: impl Into<ElementId>) -> Div {
            div()
                .id(id)
                .h_10()
                .flex_1()
                .flex()
                .justify_center()
                .items_center()
                .border_1()
                .border_color(wgpui::black())
                .bg(wgpui::black())
                .text_color(wgpui::white())
                .focus(tab_stop_style)
                .shadow_sm(),
            )
            .id(id)
        }

        div()
            .id("app")
            .track_focus(&self.focus_handle)
            .on_action(wgpui::public_listener(cx, Self::on_tab))
            .on_action(wgpui::public_listener(cx, Self::on_tab_prev))
            .size_full()
            .flex()
            .flex_col()
            .p_4()
            .gap_3()
            .bg(wgpui::white())
            .text_color(wgpui::black())
            .child(self.message.clone())
            .children(
                self.items
                    .clone()
                    .into_iter()
                    .enumerate()
                    .map(|(ix, item_handle)| {
                        let item = div()
                            .id(("item", ix))
                            .track_focus(&item_handle)
                            .h_10()
                            .w_full()
                            .flex()
                            .justify_center()
                            .items_center()
                            .border_1()
                            .border_color(wgpui::black())
                            .when(
                                item_handle.tab_stop
                                    && window.interaction_mut().is_focused(&item_handle),
                                tab_stop_style,
                            );
                        match item_handle.tab_stop {
                            true => item
                                .hover(|this| this.bg(wgpui::black().opacity(0.1)))
                                .child(format!("tab_index: {}", item_handle.tab_index)),
                            false => item.opacity(0.4).child("tab_stop: false"),
                        }
                    }),
            )
            .child(
                div()
                    .flex()
                    .flex_row()
                    .gap_3()
                    .items_center()
                    .child(
                        button("el1")
                            .tab_index(4)
                            .child("Button 1")
                            .on_click(wgpui::public_listener(cx, |this, _, _, cx| {
                                this.message = "You have clicked Button 1.".into();
                                cx.notify();
                            })),
                    )
                    .child(
                        button("el2")
                            .tab_index(5)
                            .child("Button 2")
                            .on_click(wgpui::public_listener(cx, |this, _, _, cx| {
                                this.message = "You have clicked Button 2.".into();
                                cx.notify();
                            })),
                    ),
            )
    }
}

fn main() {
    Application::new().run(|cx: &mut App| {
        cx.bind_keys([
            KeyBinding::new("tab", Tab, None),
            KeyBinding::new("shift-tab", TabPrev, None),
        ]);

        let bounds = Bounds::centered(None, size(px(800.), px(600.0)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |window, cx| cx.new(|cx| Example::new(window, cx)),
        )
        .unwrap();

        cx.activate(true);
    });
}
