use wgpui::{
    Action, App, AppWindowExt, Application, ClickEvent, Context, FocusHandle, Focusable,
    IntoElement, KeyBinding, Render, RenderOnce, Styled, Window, WindowBounds, WindowOptions,
    actions, div, px, size, rgb, white,
};

actions!(input_example, [AppendA, AppendB, Backspace, Clear, Quit]);

#[derive(Default)]
struct InputState {
    text: String,
    recent_actions: Vec<String>,
}

impl InputState {
    fn apply<A: Action>(&mut self, action: &A) {
        let name = action.name().to_owned();
        if action.as_any().is::<AppendA>() {
            self.text.push('a');
        } else if action.as_any().is::<AppendB>() {
            self.text.push('b');
        } else if action.as_any().is::<Backspace>() {
            self.text.pop();
        } else if action.as_any().is::<Clear>() {
            self.text.clear();
        }
        self.recent_actions.push(name);
        self.recent_actions.truncate(8);
    }
}

#[derive(IntoElement)]
struct InputSummary {
    text: String,
    recent_actions: Vec<String>,
}

impl RenderOnce for InputSummary {
    fn render(self, _window: &mut Window, _cx: &mut App) -> impl IntoElement {
        div()
            .flex()
            .flex_col()
            .gap_2()
            .child(format!("Value: {}", self.text))
            .children(self.recent_actions.into_iter().rev().map(|action| {
                div()
                    .text_sm()
                    .text_color(rgb(0x667085))
                    .child(action)
            }))
    }
}

struct InputExample {
    state: InputState,
    focus_handle: FocusHandle,
}

impl Focusable for InputExample {
    fn focus_handle(&self) -> FocusHandle {
        self.focus_handle
    }
}

impl InputExample {
    fn apply<A: Action>(&mut self, action: &A, cx: &mut Context<Self>) {
        self.state.apply(action);
        cx.notify();
    }

    fn on_append_a(
        &mut self,
        action: &AppendA,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.apply(action, cx);
    }

    fn on_append_b(
        &mut self,
        action: &AppendB,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.apply(action, cx);
    }

    fn on_backspace(
        &mut self,
        action: &Backspace,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.apply(action, cx);
    }

    fn on_clear(
        &mut self,
        action: &Clear,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.apply(action, cx);
    }

    fn on_reset(
        &mut self,
        _event: &ClickEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.state = InputState::default();
        cx.notify();
    }

    fn on_quit(
        &mut self,
        _action: &Quit,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        cx.quit();
    }
}

impl Render for InputExample {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .id("input-example")
            .track_focus(&self.focus_handle)
            .on_action(wgpui::public_listener(cx, Self::on_append_a))
            .on_action(wgpui::public_listener(cx, Self::on_append_b))
            .on_action(wgpui::public_listener(cx, Self::on_backspace))
            .on_action(wgpui::public_listener(cx, Self::on_clear))
            .on_action(wgpui::public_listener(cx, Self::on_quit))
            .size_full()
            .flex()
            .flex_col()
            .gap_4()
            .p_6()
            .bg(rgb(0xf2f4f7))
            .child(div().text_2xl().child("Current input actions"))
            .child(
                div()
                    .p_4()
                    .w_full()
                    .bg(white())
                    .border_1()
                    .border_color(rgb(0xd0d5dd))
                    .child(InputSummary {
                        text: self.state.text.clone(),
                        recent_actions: self.state.recent_actions.clone(),
                    }),
            )
            .child(
                div()
                    .w_full()
                    .p_3()
                    .bg(rgb(0x175cd3))
                    .text_color(white())
                    .child("Click here to reset")
                    .on_click(wgpui::public_listener(cx, Self::on_reset)),
            )
            .child("Press A, B, Backspace, or Ctrl-Backspace. Text entry and IME support are separate APIs.")
    }
}

fn main() {
    let result = Application::new().run(|cx: &mut App| {
        cx.bind_keys([
            KeyBinding::new("a", AppendA, None),
            KeyBinding::new("b", AppendB, None),
            KeyBinding::new("backspace", Backspace, None),
            KeyBinding::new("ctrl-backspace", Clear, None),
            KeyBinding::new("cmd-q", Quit, None),
        ]);

        let window_bounds = WindowBounds::Windowed(wgpui::Bounds::centered(
            None,
            size(px(520.0), px(360.0)),
            cx,
        ));
        if let Err(error) = cx.open_window(
            WindowOptions {
                title: "WGPUI input actions".to_owned(),
                window_bounds: Some(window_bounds),
                ..WindowOptions::default()
            },
            |window, cx| {
                let focus_handle = FocusHandle::new();
                window.focus(&focus_handle, &mut *cx);
                cx.new_entity(InputExample {
                    state: InputState::default(),
                    focus_handle,
                })
            },
        ) {
            eprintln!("failed to open input example window: {error}");
            cx.quit();
        }
        cx.activate(true);
    });

    if let Err(error) = result {
        eprintln!("input example failed: {error}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actions_update_text_without_splitting_utf8() {
        let mut state = InputState {
            text: "é".to_owned(),
            ..InputState::default()
        };

        state.apply(&Backspace);
        state.apply(&AppendA);

        assert_eq!(state.text, "a");
        assert_eq!(
            state.recent_actions,
            ["input_example::Backspace", "input_example::AppendA"]
        );
    }
}
